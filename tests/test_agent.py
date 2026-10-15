from datetime import datetime, timedelta, timezone

from src.agent import (
    AccountObservation, Action, AgentConfig, MarketObservation, evaluate_opportunity,
)


def observations(now):
    market = MarketObservation(
        symbol="I-BTC_INR",
        last_price=100.0,
        predicted_next_close=102.0,
        probability_up=0.8,
        candle_timestamp=now - timedelta(minutes=1),
        quote_timestamp=now,
        spread_bps=5.0,
        fee_bps_per_side=2.0,
        slippage_bps_per_side=2.0,
    )
    account = AccountObservation(
        quote_currency="INR",
        available_quote_balance=10_000.0,
        equity=10_000.0,
        daily_pnl=0.0,
        open_position_count=0,
        already_holds_symbol=False,
    )
    return market, account


def test_buy_decision_requires_real_observations_and_passes_gates():
    now = datetime.now(timezone.utc)
    market, account = observations(now)
    result = evaluate_opportunity(market, account, now=now)
    assert result.action is Action.BUY
    assert result.suggested_quote_amount == 500.0
    assert result.estimated_net_edge_bps is not None


def test_stale_quote_fails_closed():
    now = datetime.now(timezone.utc)
    market, account = observations(now)
    market = MarketObservation(**{
        **market.__dict__,
        "quote_timestamp": now - timedelta(seconds=60),
    })
    result = evaluate_opportunity(market, account, now=now)
    assert result.action is Action.REJECT
    assert "stale" in result.reason.lower()


def test_daily_loss_blocks_new_entries():
    now = datetime.now(timezone.utc)
    market, account = observations(now)
    account = AccountObservation(**{**account.__dict__, "daily_pnl": -200.0})
    result = evaluate_opportunity(market, account, now=now)
    assert result.action is Action.REJECT
    assert "daily-loss" in result.reason.lower()


def test_costs_can_make_a_prediction_untradeable():
    now = datetime.now(timezone.utc)
    market, account = observations(now)
    market = MarketObservation(**{
        **market.__dict__,
        "predicted_next_close": 100.1,
        "probability_up": 0.9,
    })
    result = evaluate_opportunity(market, account, now=now)
    assert result.action is Action.HOLD
    assert "transaction costs" in result.reason.lower()


def test_existing_position_blocks_duplicate_entry():
    now = datetime.now(timezone.utc)
    market, account = observations(now)
    account = AccountObservation(**{**account.__dict__, "already_holds_symbol": True})
    result = evaluate_opportunity(market, account, now=now)
    assert result.action is Action.HOLD
    assert "duplicate" in result.reason.lower()
