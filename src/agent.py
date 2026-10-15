"""Decision and risk foundation for the Candle Predictor trading agent.

This module deliberately does not place orders. It accepts caller-supplied market and
account observations, fails closed on missing/stale/invalid inputs, and returns an
auditable decision. No synthetic market/account values are generated here.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import math


class Action(str, Enum):
    BUY = "BUY"
    HOLD = "HOLD"
    REJECT = "REJECT"


@dataclass(frozen=True)
class AgentConfig:
    min_probability_up: float = 0.60
    min_expected_edge_bps: float = 25.0
    max_quote_fraction_per_trade: float = 0.05
    max_daily_loss_fraction: float = 0.02
    max_open_positions: int = 3
    max_spread_bps: float = 30.0
    max_candle_age_seconds: int = 1_200
    max_quote_age_seconds: int = 10
    max_price_deviation_bps: float = 100.0

    def __post_init__(self) -> None:
        if not 0.5 < self.min_probability_up < 1.0:
            raise ValueError("min_probability_up must be between 0.5 and 1.0")
        if self.min_expected_edge_bps < 0:
            raise ValueError("min_expected_edge_bps cannot be negative")
        if not 0 < self.max_quote_fraction_per_trade <= 1:
            raise ValueError("max_quote_fraction_per_trade must be in (0, 1]")
        if not 0 < self.max_daily_loss_fraction < 1:
            raise ValueError("max_daily_loss_fraction must be in (0, 1)")
        if self.max_open_positions < 1:
            raise ValueError("max_open_positions must be positive")


@dataclass(frozen=True)
class MarketObservation:
    symbol: str
    last_price: float
    predicted_next_close: float
    probability_up: float
    candle_timestamp: datetime
    quote_timestamp: datetime
    spread_bps: float
    fee_bps_per_side: float
    slippage_bps_per_side: float


@dataclass(frozen=True)
class AccountObservation:
    # Must come from the connected paper account; never fill these with defaults.
    quote_currency: str
    available_quote_balance: float
    equity: float
    daily_pnl: float
    open_position_count: int
    already_holds_symbol: bool


@dataclass(frozen=True)
class Decision:
    action: Action
    symbol: str
    reason: str
    expected_move_bps: float | None = None
    estimated_round_trip_cost_bps: float | None = None
    estimated_net_edge_bps: float | None = None
    suggested_quote_amount: float | None = None
    evaluated_at: str = ""


def _age_seconds(timestamp: datetime, now: datetime) -> float:
    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        raise ValueError("timestamps must be timezone-aware")
    return (now - timestamp.astimezone(timezone.utc)).total_seconds()


def evaluate_opportunity(
    market: MarketObservation,
    account: AccountObservation,
    config: AgentConfig = AgentConfig(),
    *,
    now: datetime | None = None,
) -> Decision:
    """Evaluate one spot-long opportunity without submitting an order."""
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None or now.utcoffset() is None:
        raise ValueError("now must be timezone-aware")

    def reject(reason: str) -> Decision:
        return Decision(Action.REJECT, market.symbol, reason, evaluated_at=now.isoformat())

    numeric_values = (
        market.last_price, market.predicted_next_close, market.probability_up,
        market.spread_bps, market.fee_bps_per_side, market.slippage_bps_per_side,
        account.available_quote_balance, account.equity, account.daily_pnl,
    )
    if not market.symbol.strip() or not account.quote_currency.strip():
        return reject("Missing symbol or quote currency.")
    if not all(math.isfinite(value) for value in numeric_values):
        return reject("Non-finite market or account value; refusing to decide.")
    if market.last_price <= 0 or market.predicted_next_close <= 0:
        return reject("Invalid price; refusing to decide.")
    if not 0 <= market.probability_up <= 1:
        return reject("Invalid model probability.")
    if min(market.spread_bps, market.fee_bps_per_side, market.slippage_bps_per_side) < 0:
        return reject("Invalid transaction-cost observation.")
    if account.available_quote_balance < 0 or account.equity <= 0:
        return reject("Invalid account balance/equity.")
    if account.open_position_count < 0:
        return reject("Invalid open-position count.")

    try:
        candle_age = _age_seconds(market.candle_timestamp, now)
        quote_age = _age_seconds(market.quote_timestamp, now)
    except ValueError as exc:
        return reject(str(exc))
    if candle_age < -5 or candle_age > config.max_candle_age_seconds:
        return reject(f"Candle is stale or timestamp is in the future (age={candle_age:.1f}s).")
    if quote_age < -2 or quote_age > config.max_quote_age_seconds:
        return reject(f"Market quote is stale or timestamp is in the future (age={quote_age:.1f}s).")
    if market.spread_bps > config.max_spread_bps:
        return reject(f"Spread {market.spread_bps:.1f} bps exceeds configured limit.")
    if account.daily_pnl <= -(account.equity * config.max_daily_loss_fraction):
        return reject("Daily-loss limit reached; new entries are blocked.")
    if account.open_position_count >= config.max_open_positions and not account.already_holds_symbol:
        return reject("Maximum open-position count reached.")
    if account.already_holds_symbol:
        return Decision(Action.HOLD, market.symbol, "Position already exists; duplicate entry blocked.",
                        evaluated_at=now.isoformat())

    deviation_bps = abs(market.last_price - market.predicted_next_close) / market.last_price * 10_000
    if deviation_bps > config.max_price_deviation_bps:
        return reject("Forecast is too far from current price; likely unstable or mis-scaled.")

    expected_move_bps = (market.predicted_next_close / market.last_price - 1) * 10_000
    round_trip_cost_bps = 2 * (market.fee_bps_per_side + market.slippage_bps_per_side) + market.spread_bps
    net_edge_bps = expected_move_bps - round_trip_cost_bps

    if market.probability_up < config.min_probability_up:
        return Decision(Action.HOLD, market.symbol, "Model confidence below entry threshold.",
                        expected_move_bps, round_trip_cost_bps, net_edge_bps,
                        evaluated_at=now.isoformat())
    if net_edge_bps < config.min_expected_edge_bps:
        return Decision(Action.HOLD, market.symbol, "Expected move does not clear estimated transaction costs and minimum edge.",
                        expected_move_bps, round_trip_cost_bps, net_edge_bps,
                        evaluated_at=now.isoformat())
    if account.available_quote_balance <= 0:
        return reject("No available quote-currency balance.")

    quote_amount = account.available_quote_balance * config.max_quote_fraction_per_trade
    return Decision(Action.BUY, market.symbol, "All configured data, confidence, edge, and risk gates passed.",
                    expected_move_bps, round_trip_cost_bps, net_edge_bps, quote_amount,
                    now.isoformat())
