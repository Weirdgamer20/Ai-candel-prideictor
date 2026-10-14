"""Direct multi-horizon OHLC forecasting.

This module predicts each future horizon directly from information available at
the latest completed candle. It never feeds its own predictions back as inputs.
Predictions are research estimates, not executable trading signals.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from .features import FEATURES, make_features

HORIZONS = tuple(range(1, 16))
TARGETS = ("open_return", "high_return", "low_return", "close_return")


def _prepare(raw: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required candle columns: {sorted(missing)}")
    candles = raw.copy().sort_values("time").drop_duplicates("time").reset_index(drop=True)
    candles["time"] = pd.to_datetime(candles["time"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        candles[col] = pd.to_numeric(candles[col], errors="coerce")
    candles = candles.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    if (candles[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("OHLC prices must be positive")
    if (candles["high"] < candles[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Invalid candle: high is below open, close or low")
    if (candles["low"] > candles[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid candle: low is above open, close or high")

    feat = make_features(candles, include_unlabeled=True)
    base = candles.set_index("time")
    feat = feat.copy()
    for h in horizons:
        future = base[["open", "high", "low", "close"]].shift(-h)
        for col in ("open", "high", "low", "close"):
            feat[f"y_{h}_{col}"] = (
                future[col].reindex(pd.DatetimeIndex(feat["time"])) .to_numpy()
                / feat["close"].to_numpy() - 1.0
            )
    return feat.replace([np.inf, -np.inf], np.nan)


def train_multihorizon(raw: pd.DataFrame, output_dir: str = "models") -> dict:
    """Fit direct models and evaluate horizons on chronological holdout data."""
    data = _prepare(raw)
    target_cols = [f"y_{h}_{col}" for h in HORIZONS for col in ("open", "high", "low", "close")]
    data = data.dropna(subset=FEATURES + target_cols).reset_index(drop=True)
    if len(data) < 3000:
        raise ValueError(f"Need at least 3,000 complete feature rows; found {len(data)}.")

    n = len(data)
    train_end, val_end = int(n * .70), int(n * .85)
    train, validation, test = data.iloc[:train_end], data.iloc[train_end:val_end], data.iloc[val_end:]
    if min(len(train), len(validation), len(test)) == 0:
        raise ValueError("Insufficient rows for chronological train/validation/test splits.")

    # Select the common complexity using validation loss at all horizons.
    candidates = {
        "conservative": dict(max_iter=100, learning_rate=.04, max_leaf_nodes=7, l2_regularization=2.0),
        "standard": dict(max_iter=180, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0),
    }
    candidate_scores = {}
    for name, params in candidates.items():
        losses = []
        for h in HORIZONS:
            target = f"y_{h}_close"
            model = HistGradientBoostingRegressor(random_state=42, **params)
            model.fit(train[FEATURES], train[target])
            pred = model.predict(validation[FEATURES])
            losses.append(mean_absolute_error(validation[target], pred))
        candidate_scores[name] = float(np.mean(losses))
    selected = min(candidate_scores, key=candidate_scores.get)

    fit_rows = data.iloc[:val_end]
    params = candidates[selected]
    models = {}
    metrics = {}
    for h in HORIZONS:
        models[h] = {}
        metrics[str(h)] = {}
        for col in ("open", "high", "low", "close"):
            target = f"y_{h}_{col}"
            model = HistGradientBoostingRegressor(random_state=42, **params)
            model.fit(fit_rows[FEATURES], fit_rows[target])
            prediction = model.predict(test[FEATURES])
            actual = test[target].to_numpy(dtype=float)
            metrics[str(h)][col] = {
                "mae_return_pct_points": float(mean_absolute_error(actual, prediction) * 100),
                "rmse_return_pct_points": float(np.sqrt(mean_squared_error(actual, prediction)) * 100),
                "direction_accuracy": float(np.mean((prediction >= 0) == (actual >= 0))) if col == "close" else None,
            }
            models[h][col] = model

    # Residual quantiles from validation provide empirical, not guaranteed, intervals.
    intervals = {}
    for h in HORIZONS:
        intervals[str(h)] = {}
        for col in ("open", "high", "low", "close"):
            model = HistGradientBoostingRegressor(random_state=42, **params)
            model.fit(train[FEATURES], train[f"y_{h}_{col}"])
            residual = validation[f"y_{h}_{col}"].to_numpy() - model.predict(validation[FEATURES])
            intervals[str(h)][col] = [
                float(np.quantile(residual, .05)),
                float(np.quantile(residual, .95)),
            ]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": 1,
        "features": FEATURES,
        "horizons": list(HORIZONS),
        "models": models,
        "residual_intervals": intervals,
        "selected_complexity": selected,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_training_candle": str(data["time"].iloc[val_end - 1]),
    }
    joblib.dump(bundle, output / "multihorizon_ohlc.joblib")
    report = {
        "version": 1,
        "selected_complexity": selected,
        "validation_mean_close_mae_by_candidate": candidate_scores,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test)},
        "test_start": str(test["time"].iloc[0]),
        "test_end": str(test["time"].iloc[-1]),
        "metrics_by_horizon": metrics,
        "interval_note": "Empirical validation residual 5th-95th percentile bands; coverage is not guaranteed.",
        "warning": "Research metrics only; not proof of profitability. Test data was not used for model selection.",
    }
    (output / "multihorizon_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def forecast_ohlc(raw: pd.DataFrame, bundle: dict | None = None, model_path: str = "models/multihorizon_ohlc.joblib") -> dict:
    """Generate 15 direct OHLC forecasts from the latest valid completed candle."""
    if bundle is None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError("Multi-horizon model missing. Run python train_multihorizon.py first.")
        bundle = joblib.load(path)
    if bundle.get("version") != 1:
        raise RuntimeError("Unsupported multi-horizon model version; retrain it.")

    featured = make_features(raw, include_unlabeled=True)
    if featured.empty:
        raise ValueError("Not enough valid candles for forecasting.")
    latest = featured.iloc[[-1]]
    last_close = float(latest["close"].iloc[0])
    last_time = pd.to_datetime(latest["time"].iloc[0], utc=True)
    forecasts = []
    for h in bundle["horizons"]:
        pred = {}
        for col in ("open", "high", "low", "close"):
            relative = float(bundle["models"][h][col].predict(latest[FEATURES])[0])
            pred[col] = last_close * (1.0 + relative)
        # Keep the candle mathematically valid while preserving predicted close.
        pred["high"] = max(pred["high"], pred["open"], pred["close"])
        pred["low"] = min(pred["low"], pred["open"], pred["close"])
        ranges = bundle["residual_intervals"][str(h)]
        pred["uncertainty"] = {
            col: [
                last_close * (1.0 + float(bundle["models"][h][col].predict(latest[FEATURES])[0]) + ranges[col][0]),
                last_close * (1.0 + float(bundle["models"][h][col].predict(latest[FEATURES])[0]) + ranges[col][1]),
            ] for col in ("open", "high", "low", "close")
        }
        forecasts.append({
            "horizon_candles": int(h),
            "time": (last_time + pd.Timedelta(minutes=15 * h)).isoformat(),
            "open": pred["open"], "high": pred["high"], "low": pred["low"], "close": pred["close"],
            "uncertainty": pred["uncertainty"],
        })
    return {
        "symbol": "BTC/INR",
        "timeframe": "15m",
        "last_completed_candle": last_time.isoformat(),
        "last_close": last_close,
        "model_version": bundle["version"],
        "forecast_generated_at": datetime.now(timezone.utc).isoformat(),
        "candles": forecasts,
        "status": "RESEARCH_FORECAST",
    }
