"""Train and evaluate direct 1-15 candle OHLC forecasts."""
from pathlib import Path
import json
import pandas as pd

from src.multihorizon import train_multihorizon


def main():
    data_path = Path("data/raw/btc_inr_15m.csv")
    if not data_path.exists():
        raise SystemExit("Missing data/raw/btc_inr_15m.csv. Run python download_data.py first.")
    raw = pd.read_csv(data_path, parse_dates=["time"])
    report = train_multihorizon(raw)
    print(json.dumps(report, indent=2))
    print("\nSaved models/multihorizon_ohlc.joblib and models/multihorizon_metrics.json")
    print("Forecasts are research estimates; evaluate per horizon before relying on them.")


if __name__ == "__main__":
    main()
