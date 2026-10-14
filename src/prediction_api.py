"""Local read-only API for the CoinDCX Chrome overlay.

Binds to loopback only. No trading routes or exchange credentials are supported.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response

from .multihorizon import forecast_ohlc
from .modeling import load_models, predict_next

DATA_PATH = Path(os.getenv("BTCINR_DATA_PATH", "data/raw/btc_inr_15m.csv"))
MODEL_DIR = os.getenv("BTCINR_MODEL_DIR", "models")
MAX_STALE_MINUTES = int(os.getenv("BTCINR_MAX_STALE_MINUTES", "20"))
app = FastAPI(title="BTC/INR Read-only Prediction API", version="1.1.0")


@app.middleware("http")
async def restrict_origin_and_enable_extension_cors(request: Request, call_next):
    origin = request.headers.get("origin")
    allowed_id = os.getenv("BTCINR_EXTENSION_ID", "").strip()
    allowed_origin = f"chrome-extension://{allowed_id}" if allowed_id else None
    if origin and origin.startswith("chrome-extension://") and origin != allowed_origin:
        return JSONResponse(status_code=403, content={"detail": "Extension origin not allow-listed"})
    if request.method == "OPTIONS":
        if origin != allowed_origin or not allowed_origin:
            return Response(status_code=403)
        response = Response(status_code=204)
    else:
        response = await call_next(request)
    if origin and origin == allowed_origin:
        response.headers["Access-Control-Allow-Origin"] = allowed_origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET, OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


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
        for column in ("open", "high", "low", "close", "volume"):
            raw[column] = pd.to_numeric(raw[column], errors="coerce")
        raw = raw.dropna(subset=list(required)).drop_duplicates("time").sort_values("time")
        now = pd.Timestamp.now(tz="UTC")
        raw = raw.loc[raw["time"] + pd.Timedelta(minutes=15) <= now].copy()
        if raw.empty:
            raise HTTPException(status_code=503, detail="No completed 15-minute candles are available")
        latest_time = raw["time"].iloc[-1]
        latest_close_time = latest_time + pd.Timedelta(minutes=15)
        age_minutes = (now - latest_close_time).total_seconds() / 60
        if age_minutes < 0:
            raise ValueError("Completed-candle timestamp validation failed")
        if age_minutes > MAX_STALE_MINUTES:
            raise HTTPException(status_code=503, detail=f"Market data is stale ({age_minutes:.1f} minutes since candle close)")

        result = {
            "symbol": "BTC/INR", "timeframe": "15m", "data_status": "FRESH",
            "data_age_minutes": round(age_minutes, 2), "last_completed_candle": latest_time.isoformat(),
            "trading_enabled": False, "generated_at": datetime.now(timezone.utc).isoformat(),
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
        result["status"] = "RESEARCH_FORECAST" if result["forecast"] is not None else "NO_FORECAST"
        return result
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Prediction unavailable: {type(exc).__name__}: {exc}") from exc
