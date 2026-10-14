# CoinDCX Future-Candle Overlay: Implementation Notes

## Selected decisions

- Native Windows 11 Python processes.
- Chrome Manifest V3 extension matched only to CoinDCX BTC/INR trading URLs.
- Click-through transparent canvas; no overlay buttons or mouse handlers.
- Public CoinDCX OHLCV is authoritative; OCR/OpenCV is a visual cross-check only.
- Forecast horizon is 1–15 candles, but display/use only horizons that pass chronological evaluation.
- Predicted OHLC candles include empirical residual bands and fade with horizon.
- If data is stale, the model is unavailable, or chart geometry cannot be mapped reliably, **hide the forecast**.
- No order-placement endpoints or credentials exist in this phase.

## Local setup

1. Create and activate a Python 3.12 virtual environment.
2. Install `requirements.txt`.
3. Verify and download history with `python check_coindcx.py` and `python download_data.py --pair auto --start-date 2018-10-18`.
4. Train the existing one-step models with `python train_model.py`.
5. Train direct horizon-specific OHLC models with `python train_multihorizon.py`.
6. Open `chrome://extensions`, enable Developer mode, and select Load unpacked for the `extension` folder.
7. Copy the extension ID from Chrome.
8. Run `run_overlay_windows.bat` and paste the extension ID.
9. Open the supported CoinDCX BTC/INR chart.

The local API binds only to `127.0.0.1:8765`. It checks the extension ID using `BTCINR_EXTENSION_ID`, serves read-only predictions, and has no order routes.

## Multi-horizon training

`python train_multihorizon.py` writes `models/multihorizon_ohlc.joblib` and `models/multihorizon_metrics.json`. The direct model predicts each horizon from features known at the last completed candle; it does not recursively use its own outputs. Metrics are evaluated per horizon on a chronological holdout. The 5th–95th percentile residual bands are empirical and are not guaranteed confidence intervals.

## Critical rendering status

The extension skeleton detects a large chart canvas and receives the forecast API payload. **The current content script intentionally fails closed and does not draw forecast candles yet.** CoinDCX chart internals, candle spacing, and price/time-axis mapping must be measured and tested against the actual page before drawing; painting at guessed coordinates would be misleading. The extension is therefore an integration scaffold, not a finished visual overlay.

The final renderer must draw only inside the chart bounds, never over the order book or buy/sell controls, and must remove its forecast if geometry confidence is insufficient. OCR/OpenCV must not silently replace API-verified OHLCV.
