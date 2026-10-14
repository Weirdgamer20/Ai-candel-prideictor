"""Computer-use trading policy for the BTC/INR project.

This module intentionally does not click the exchange UI or submit orders.
It gates any future CUA adapter behind explicit validation and per-order human
approval. The current build must remain research/paper-only.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class ExecutionMode(str, Enum):
    DISABLED = "DISABLED"
    PAPER = "PAPER"
    MANUAL_CONFIRMATION = "MANUAL_CONFIRMATION"
    LIVE = "LIVE"


@dataclass(frozen=True)
class TradeIntent:
    symbol: str
    side: str
    quote_amount_inr: float
    reference_price_inr: float
    signal_id: str
    expected_edge_pct: float
    estimated_round_trip_cost_pct: float


@dataclass(frozen=True)
class GateStatus:
    data_fresh: bool = False
    model_validated: bool = False
    strategy_walk_forward_passed: bool = False
    paper_trading_passed: bool = False
    execution_tests_passed: bool = False
    kill_switch_clear: bool = False
    daily_loss_inr: float = 0.0
    max_daily_loss_inr: float = 50.0
    max_total_loss_inr: float = 250.0
    total_loss_inr: float = 0.0


def validate_trade_intent(intent: TradeIntent, gates: GateStatus) -> list[str]:
    """Return every blocking reason; an empty list means eligible for manual review only."""
    blockers: list[str] = []
    if intent.symbol.upper().replace("-", "/") not in {"BTC/INR", "BTCINR"}:
        blockers.append("Only BTC/INR spot is permitted.")
    if intent.side.upper() not in {"BUY", "SELL"}:
        blockers.append("Side must be BUY or SELL.")
    if not intent.signal_id.strip():
        blockers.append("A unique signal ID is required.")
    if intent.quote_amount_inr <= 0:
        blockers.append("Order amount must be positive.")
    if intent.quote_amount_inr > 500:
        blockers.append("Order amount exceeds the initial ₹500 capital budget.")
    if intent.reference_price_inr <= 0:
        blockers.append("Reference price must be positive.")
    if intent.expected_edge_pct <= intent.estimated_round_trip_cost_pct:
        blockers.append("Expected edge does not exceed estimated round-trip costs.")
    if not gates.data_fresh:
        blockers.append("Market data is not fresh.")
    if not gates.model_validated:
        blockers.append("Model validation gate has not passed.")
    if not gates.strategy_walk_forward_passed:
        blockers.append("Walk-forward strategy validation has not passed.")
    if not gates.paper_trading_passed:
        blockers.append("Paper-trading gate has not passed.")
    if not gates.execution_tests_passed:
        blockers.append("Execution failure/recovery tests have not passed.")
    if not gates.kill_switch_clear:
        blockers.append("Kill switch is active or unverified.")
    if gates.daily_loss_inr >= gates.max_daily_loss_inr:
        blockers.append("Daily loss limit reached.")
    if gates.total_loss_inr >= min(250.0, gates.max_total_loss_inr):
        blockers.append("Total affordable loss limit of ₹250 reached.")
    return blockers


def authorize_action(
    mode: ExecutionMode,
    intent: TradeIntent,
    gates: GateStatus,
    *,
    explicit_user_confirmation: bool = False,
) -> dict[str, Any]:
    """Never executes. Produces a policy decision for a future isolated CUA adapter."""
    blockers = validate_trade_intent(intent, gates)
    if mode in {ExecutionMode.DISABLED, ExecutionMode.PAPER}:
        return {
            "allowed": False,
            "mode": mode.value,
            "decision": "PAPER_ONLY" if mode == ExecutionMode.PAPER else "DISABLED",
            "blockers": blockers or ["Live website actions are disabled in this build."],
            "requires_human_confirmation": False,
            "order_submitted": False,
        }
    if mode == ExecutionMode.MANUAL_CONFIRMATION:
        return {
            "allowed": not blockers and explicit_user_confirmation,
            "mode": mode.value,
            "decision": "MANUAL_CONFIRMATION" if not blockers else "BLOCKED",
            "blockers": blockers if blockers else ([] if explicit_user_confirmation else ["Explicit confirmation for this exact order is required."]),
            "requires_human_confirmation": True,
            "order_submitted": False,
        }
    # LIVE deliberately cannot be enabled through this module.
    return {
        "allowed": False,
        "mode": mode.value,
        "decision": "LIVE_MODE_NOT_IMPLEMENTED",
        "blockers": blockers + ["Live CUA execution is not implemented or enabled."],
        "requires_human_confirmation": True,
        "order_submitted": False,
    }
