"""Local read-only API for the CoinDCX Chrome overlay.

Binds to loopback only. No trading routes or exchange credentials are supported.
"""
from __future__ import annotations

import os
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from .data import discover_btc_inr_pair, fetch_candles, save_candles
from .multihorizon import forecast_ohlc
from .modeling import load_models, predict_next

DATA_PATH = Path(os.getenv("BTCINR_DATA_PATH", "data/raw/btc_inr_15m.csv"))
MODEL_DIR = os.getenv("BTCINR_MODEL_DIR", "models")
MAX_STALE_MINUTES = int(os.getenv("BTCINR_MAX_STALE_MINUTES", "20"))
app = FastAPI(title="BTC/INR Read-only Prediction API", version="1.0.0")


@app.middleware("http")
async def restrict_origin(request: Request, call_next):
    origin = request.headers.get("origin")
    allowed = os.getenv("BTCINR_EXTENSION_ID", "").strip()
    if origin and origin.startswith("chrome-extension://"):
        extension_id = origin.removeprefix("chrome-extension://").rstrip("/")
        if not allowed or extension_id != allowed:
            return JSONResponse(status_code=403, content={"detail": "Extension origin not allow-listed"})
    return await call_next(request)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "btc-inr-prediction-api",
        "trading_enabled": False,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/api/v1/prediction")
def get_prediction():
    if not DATA_PATH.exists():
        raise HTTPException(status_code=503, detail="Historical/live candle CSV is missing")
    try:
        raw = pd.read_csv(DATA_PATH, parse_dates=["time"])
        required = {"time", "open", "high", "low", "close", "volume"}
        if not required.issubset(raw.columns) or raw.empty:
            raise ValueError("Candle CSV is empty or missing required OHLCV fields")
        raw["time"] = pd.to_datetime(raw["time"], errors="coerce", utc=True)
        raw = raw.dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
        latest_time = raw["time"].iloc[-1]
        age_minutes = (pd.Timestamp.now(tz="UTC") - latest_time).total_seconds() / 60
        if age_minutes < -15:
            raise ValueError("Latest candle timestamp is in the future")
        if age_minutes > MAX_STALE_MINUTES:
            raise HTTPException(status_code=503, detail=f"Market data is stale ({age_minutes:.1f} minutes old)")

        result = {
            "symbol": "BTC/INR",
            "timeframe": "15m",
            "data_status": "FRESH",
            "data_age_minutes": round(age_minutes, 2),
            "trading_enabled": False,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }
        try:
            direction_model, price_model = load_models(MODEL_DIR)
            result["one_step_diagnostic"] = predict_next(raw, direction_model, price_model)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            result["one_step_diagnostic"] = None
            result["one_step_model_status"] = f"UNAVAILABLE: {exc}"

        try:
            result["forecast"] = forecast_ohlc(raw, model_path=str(Path(MODEL_DIR) / "multihorizon_ohlc.joblib"))
            result["forecast_status"] = "READY"
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            result["forecast"] = None
            result["forecast_status"] = f"UNAVAILABLE: {exc}"

        if result["forecast"] is None:
            result["status"] = "NO_FORECAST"
        else:
            result["status"] = "RESEARCH_FORECAST"
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Prediction unavailable: {type(exc).__name__}: {exc}") from exc
