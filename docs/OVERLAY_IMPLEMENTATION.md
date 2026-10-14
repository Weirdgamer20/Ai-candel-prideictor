# CoinDCX Future-Candle Overlay: Implementation Notes

## Runtime

- Windows 11 native Python process (no WSL dependency for the overlay).
- Chrome Manifest V3 extension restricted to CoinDCX BTC/INR trade pages.
- Read-only local API bound to 127.0.0.1:8765.
- No exchange credentials, balance access, or order-placement routes.
- The extension is click-through: overlay and canvas use pointer-events:none.

## Forecasts

The existing model predicts one next close and cannot produce 10-15 full OHLC candles.
Run `python train_multihorizon.py` after verifying the local OHLCV dataset to train direct horizon-specific open/high/low/close regressors. The script writes `models/multihorizon_ohlc.joblib` and `models/multihorizon_metrics.json`. Evaluate metrics separately for each horizon before relying on any forecast.

The displayed uncertainty ranges are empirical residual bands, not guaranteed confidence intervals. Forecasts are research-only.

## Start the local API

Install the added dependencies from requirements.txt, then run:

```powershell
python -m uvicorn src.prediction_api:app --host 127.0.0.1 --port 8765
```

The API returns no forecast when the local dataset is stale or the multi-horizon model is unavailable. It does not include trading endpoints.

## Load the extension

1. Open `chrome://extensions`.
2. Enable Developer mode.
3. Select Load unpacked and choose the repository's `extension` directory.
4. Open the supported CoinDCX BTC/INR trading page.

## Important rendering limitation

CoinDCX's chart uses its own drawing surface and price/time scales. A reliable future-candle renderer must identify the visible chart bounds, price-axis mapping, time-axis mapping, and candle spacing. If these cannot be mapped unambiguously, the extension must hide the forecast rather than paint at guessed coordinates. The current renderer is intentionally fail-closed; it is an integration scaffold, not yet a completed candle projection.

OCR/OpenCV chart observation remains a separate validation step. OCR values must never silently replace verified API candles.
