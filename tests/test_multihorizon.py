import numpy as np
import pandas as pd
import pytest

from src.multihorizon import forecast_ohlc


def _bundle():
    class ConstantModel:
        def __init__(self, value):
            self.value = value
        def predict(self, x):
            return np.array([self.value] * len(x))

    models = {}
    intervals = {}
    for h in range(1, 16):
        models[h] = {
            "open": ConstantModel(0.001 * h),
            "high": ConstantModel(0.003 * h),
            "low": ConstantModel(-0.003 * h),
            "close": ConstantModel(0.002 * h),
        }
        intervals[str(h)] = {k: [-0.001, 0.001] for k in ("open", "high", "low", "close")}
    return {"version": 2, "horizons": list(range(1, 16)), "models": models,
            "residual_intervals": intervals}


def _candles(n=100):
    close = 8000000 + np.arange(n) * 10.0
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="15min", tz="UTC"),
        "open": close - 20, "high": close + 100, "low": close - 100,
        "close": close, "volume": np.ones(n),
    })


def test_forecast_has_15_valid_ordered_ohlc_candles():
    result = forecast_ohlc(_candles(), bundle=_bundle())
    assert len(result["candles"]) == 15
    for candle in result["candles"]:
        assert candle["high"] >= max(candle["open"], candle["close"], candle["low"])
        assert candle["low"] <= min(candle["open"], candle["close"], candle["high"])
        assert candle["open"] > 0 and candle["close"] > 0
        assert len(candle["uncertainty"]["close"]) == 2


def test_forecast_timestamps_are_15_minute_steps():
    result = forecast_ohlc(_candles(), bundle=_bundle())
    times = pd.to_datetime([c["time"] for c in result["candles"]])
    assert (times[1] - times[0]).total_seconds() == 900
    assert (times[-1] - times[0]).total_seconds() == 14 * 900


def test_missing_multihorizon_model_fails_closed(tmp_path):
    with pytest.raises(FileNotFoundError):
        forecast_ohlc(_candles(), model_path=str(tmp_path / "missing.joblib"))
