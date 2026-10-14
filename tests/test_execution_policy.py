import pytest

from src.execution_policy import (
    ExecutionMode, GateStatus, TradeIntent, authorize_action, validate_trade_intent
)


def intent(**changes):
    values = dict(symbol="BTC/INR", side="BUY", quote_amount_inr=100,
                  reference_price_inr=8000000, signal_id="sig-001",
                  expected_edge_pct=0.8, estimated_round_trip_cost_pct=0.3)
    values.update(changes)
    return TradeIntent(**values)


def passed_gates(**changes):
    values = dict(data_fresh=True, model_validated=True,
                  strategy_walk_forward_passed=True, paper_trading_passed=True,
                  execution_tests_passed=True, kill_switch_clear=True,
                  daily_loss_inr=0, total_loss_inr=0)
    values.update(changes)
    return GateStatus(**values)


def test_paper_mode_never_submits_order():
    result = authorize_action(ExecutionMode.PAPER, intent(), passed_gates())
    assert result["order_submitted"] is False
    assert result["allowed"] is False


def test_unvalidated_model_blocks_manual_action():
    gates = passed_gates(model_validated=False)
    blockers = validate_trade_intent(intent(), gates)
    assert any("Model validation" in b for b in blockers)


def test_total_loss_limit_blocks_trade():
    gates = passed_gates(total_loss_inr=250)
    assert any("Total affordable loss" in b for b in validate_trade_intent(intent(), gates))


def test_manual_mode_requires_explicit_confirmation():
    gates = passed_gates()
    no = authorize_action(ExecutionMode.MANUAL_CONFIRMATION, intent(), gates)
    yes = authorize_action(ExecutionMode.MANUAL_CONFIRMATION, intent(), gates, explicit_user_confirmation=True)
    assert not no["allowed"]
    assert yes["allowed"]
    assert yes["order_submitted"] is False


def test_live_mode_is_not_implemented_even_if_gates_pass():
    result = authorize_action(ExecutionMode.LIVE, intent(), passed_gates(), explicit_user_confirmation=True)
    assert not result["allowed"]
    assert result["order_submitted"] is False
