"""Direct multi-horizon OHLC forecasting with purged chronological splits.

Each horizon is predicted directly from features known at the latest completed
candle. Forecasts are research estimates, never trade instructions.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

from .features import FEATURES, make_features

HORIZONS = tuple(range(1, 16))
OHLC = ("open", "high", "low", "close")
MODEL_VERSION = 2


def _prepare(raw: pd.DataFrame, horizons=HORIZONS) -> pd.DataFrame:
    required = {"time", "open", "high", "low", "close", "volume"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Missing required candle columns: {sorted(missing)}")
    candles = raw.copy()
    candles["time"] = pd.to_datetime(candles["time"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        candles[col] = pd.to_numeric(candles[col], errors="coerce")
    candles = candles.dropna(subset=list(required)).sort_values("time").drop_duplicates("time").reset_index(drop=True)
    if candles.empty:
        raise ValueError("No valid candles remain after normalization")
    if (candles[list(OHLC)] <= 0).any().any() or (candles["volume"] < 0).any():
        raise ValueError("Prices must be positive and volume cannot be negative")
    if (candles["high"] < candles[["open", "close", "low"]].max(axis=1)).any():
        raise ValueError("Invalid candle: high below open, close or low")
    if (candles["low"] > candles[["open", "close", "high"]].min(axis=1)).any():
        raise ValueError("Invalid candle: low above open, close or high")

    feat = make_features(candles, include_unlabeled=True).copy()
    base = candles.set_index("time")
    for h in horizons:
        future = base[list(OHLC)].shift(-h)
        aligned = future.reindex(pd.DatetimeIndex(feat["time"]))
        for col in OHLC:
            feat[f"y_{h}_{col}"] = aligned[col].to_numpy() / feat["close"].to_numpy() - 1.0
    return feat.replace([np.inf, -np.inf], np.nan)


def train_multihorizon(raw: pd.DataFrame, output_dir: str = "models") -> dict:
    data = _prepare(raw)
    target_cols = [f"y_{h}_{col}" for h in HORIZONS for col in OHLC]
    data = data.dropna(subset=FEATURES + target_cols).reset_index(drop=True)
    if len(data) < 3000:
        raise ValueError(f"Need at least 3,000 complete feature rows; found {len(data)}")

    # Purge 15 rows at each boundary: labels at one split must not reach into
    # the next split's target period.
    n = len(data)
    raw_train_end, raw_val_end = int(n * .70), int(n * .85)
    purge = max(HORIZONS)
    train = data.iloc[:raw_train_end - purge]
    validation = data.iloc[raw_train_end:raw_val_end - purge]
    test = data.iloc[raw_val_end:]
    if min(len(train), len(validation), len(test)) < 100:
        raise ValueError("Insufficient data after purging multi-horizon split boundaries")

    candidates = {
        "conservative": dict(max_iter=100, learning_rate=.04, max_leaf_nodes=7, l2_regularization=2.0),
        "standard": dict(max_iter=180, learning_rate=.05, max_leaf_nodes=15, l2_regularization=1.0),
    }
    candidate_scores = {}
    for name, params in candidates.items():
        losses = []
        for h in HORIZONS:
            model = HistGradientBoostingRegressor(random_state=42, **params)
            model.fit(train[FEATURES], train[f"y_{h}_close"])
            losses.append(mean_absolute_error(validation[f"y_{h}_close"], model.predict(validation[FEATURES])))
        candidate_scores[name] = float(np.mean(losses))
    selected = min(candidate_scores, key=candidate_scores.get)
    params = candidates[selected]

    # Final model fits on train+validation, still purged from test labels.
    fit_rows = data.iloc[:raw_val_end - purge]
    models, metrics, intervals = {}, {}, {}
    for h in HORIZONS:
        models[h], metrics[str(h)], intervals[str(h)] = {}, {}, {}
        for col in OHLC:
            target = f"y_{h}_{col}"
            model = HistGradientBoostingRegressor(random_state=42, **params)
            model.fit(fit_rows[FEATURES], fit_rows[target])
            pred = model.predict(test[FEATURES])
            actual = test[target].to_numpy(dtype=float)
            metrics[str(h)][col] = {
                "mae_return_pct_points": float(mean_absolute_error(actual, pred) * 100),
                "rmse_return_pct_points": float(np.sqrt(mean_squared_error(actual, pred)) * 100),
                "direction_accuracy": float(np.mean((pred >= 0) == (actual >= 0))) if col == "close" else None,
            }
            models[h][col] = model

            # Fit only on train to get validation residual bands; no test residuals
            # are used to construct uncertainty intervals.
            interval_model = HistGradientBoostingRegressor(random_state=42, **params)
            interval_model.fit(train[FEATURES], train[target])
            residual = validation[target].to_numpy(dtype=float) - interval_model.predict(validation[FEATURES])
            intervals[str(h)][col] = [float(np.quantile(residual, .05)), float(np.quantile(residual, .95))]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    bundle = {
        "version": MODEL_VERSION, "features": FEATURES, "horizons": list(HORIZONS),
        "models": models, "residual_intervals": intervals, "selected_complexity": selected,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "last_training_candle": str(fit_rows["time"].iloc[-1]),
    }
    joblib.dump(bundle, output / "multihorizon_ohlc.joblib")
    report = {
        "version": MODEL_VERSION, "selected_complexity": selected,
        "validation_mean_close_mae_by_candidate": candidate_scores,
        "split_rows": {"train": len(train), "validation": len(validation), "test": len(test), "purge_rows_each_boundary": purge},
        "test_start": str(test["time"].iloc[0]), "test_end": str(test["time"].iloc[-1]),
        "metrics_by_horizon": metrics,
        "interval_note": "Empirical validation residual 5th–95th percentile bands; coverage is not guaranteed.",
        "warning": "Research metrics only. This report does not demonstrate profitability or authorize live trading.",
    }
    (output / "multihorizon_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def forecast_ohlc(raw: pd.DataFrame, bundle: dict | None = None, model_path: str = "models/multihorizon_ohlc.joblib") -> dict:
    if bundle is None:
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError("Multi-horizon model missing. Run python train_multihorizon.py first.")
        bundle = joblib.load(path)
    if bundle.get("version") != MODEL_VERSION:
        raise RuntimeError("Unsupported multi-horizon model version; retrain it.")

    featured = _prepare_latest_features(raw)
    latest = featured.iloc[[-1]]
    last_close = float(latest["close"].iloc[0])
    last_time = pd.to_datetime(latest["time"].iloc[0], utc=True)
    forecasts = []
    for h in bundle["horizons"]:
        point, uncertainty = {}, {}
        for col in OHLC:
            relative = float(bundle["models"][h][col].predict(latest[FEATURES])[0])
            point[col] = last_close * (1 + relative)
            lo, hi = bundle["residual_intervals"][str(h)][col]
            uncertainty[col] = [max(0.0, last_close * (1 + relative + lo)), last_close * (1 + relative + hi)]
        point["high"] = max(point["high"], point["open"], point["close"])
        point["low"] = min(point["low"], point["open"], point["close"])
        forecasts.append({
            "horizon_candles": int(h),
            "time": (last_time + pd.Timedelta(minutes=15 * h)).isoformat(),
            "open": point["open"], "high": point["high"], "low": point["low"], "close": point["close"],
            "uncertainty": uncertainty,
        })
    return {
        "symbol": "BTC/INR", "timeframe": "15m",
        "last_completed_candle": last_time.isoformat(), "last_close": last_close,
        "model_version": bundle["version"], "forecast_generated_at": datetime.now(timezone.utc).isoformat(),
        "candles": forecasts, "status": "RESEARCH_FORECAST",
    }


def _prepare_latest_features(raw: pd.DataFrame) -> pd.DataFrame:
    candles = raw.copy()
    candles["time"] = pd.to_datetime(candles["time"], errors="coerce", utc=True)
    for col in ("open", "high", "low", "close", "volume"):
        candles[col] = pd.to_numeric(candles[col], errors="coerce")
    candles = candles.dropna(subset=["time", "open", "high", "low", "close", "volume"])
    candles = candles.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    featured = make_features(candles, include_unlabeled=True)
    if featured.empty:
        raise ValueError("Not enough valid candles to generate a forecast")
    return featured
