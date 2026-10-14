"""Incremental CoinDCX candle updater for the local overlay.

Run as a separate Windows process. It only downloads public market data and
writes a local CSV; it cannot access balances or place orders.
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
    before = pd.read_csv(DATA_PATH, usecols=["time"])
    previous = before["time"].max() if not before.empty else None
    updated = fetch_candles(pair=pair, resume_path=str(DATA_PATH))
    latest = updated["time"].max() if not updated.empty else None
    if latest is not None and (previous is None or str(latest) != str(previous)):
        temporary = DATA_PATH.with_suffix(".tmp.csv")
        save_candles(updated, str(temporary))
        temporary.replace(DATA_PATH)
    return len(updated)


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
