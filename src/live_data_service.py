"""Incremental CoinDCX candle updater for the local overlay.

Refreshes a short overlap so the latest still-forming candle is replaced by the
final exchange candle after close. This service only uses public market data.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd

from .data import discover_btc_inr_pair, fetch_candles, save_candles

DATA_PATH = Path(os.getenv("BTCINR_DATA_PATH", "data/raw/btc_inr_15m.csv"))
POLL_SECONDS = max(15, int(os.getenv("BTCINR_POLL_SECONDS", "30")))


def update_once() -> int:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"{DATA_PATH} does not exist. Run download_data.py first to obtain historical candles."
        )
    pair = os.getenv("BTCINR_PAIR") or discover_btc_inr_pair()
    existing = pd.read_csv(DATA_PATH, parse_dates=["time"])
    if existing.empty:
        raise ValueError("Local candle CSV is empty")
    existing["time"] = pd.to_datetime(existing["time"], errors="coerce", utc=True)
    existing = existing.dropna(subset=["time"]).drop_duplicates("time").sort_values("time")
    overlap_start = existing["time"].max() - pd.Timedelta(minutes=60)
    refreshed = fetch_candles(pair=pair, start_date=overlap_start.isoformat())
    combined = pd.concat([existing, refreshed], ignore_index=True)
    combined["time"] = pd.to_datetime(combined["time"], errors="coerce", utc=True)
    combined = combined.dropna(subset=["time"]).drop_duplicates("time", keep="last").sort_values("time")
    temporary = DATA_PATH.with_suffix(".tmp.csv")
    save_candles(combined, str(temporary))
    temporary.replace(DATA_PATH)
    return len(combined)


def main() -> None:
    print(f"CoinDCX read-only updater: {DATA_PATH}; polling every {POLL_SECONDS}s")
    while True:
        try:
            count = update_once()
            print(f"Updated dataset: {count:,} rows")
        except KeyboardInterrupt:
            print("Updater stopped.")
            break
        except Exception as exc:
            print(f"Update failed: {type(exc).__name__}: {exc}")
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
