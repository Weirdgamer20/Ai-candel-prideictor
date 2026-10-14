# CoinDCX Future-Candle Overlay and CUA Roadmap

## Current build boundaries

- Native Windows 11 Python processes; Chrome Manifest V3 extension.
- Public CoinDCX 15-minute OHLCV is authoritative; OCR/OpenCV is visual corroboration only.
- The local API binds to `127.0.0.1:8765`, allow-lists the exact extension origin, and serves read-only predictions.
- The extension is click-through and must not cover order controls.
- Trading is disabled. No API keys, private account endpoints, order routes, or live order-submission code are included.
- A future computer-use (CUA) adapter must be opt-in, default-off, restricted to BTC/INR spot, require explicit confirmation for each exact order, and verify order state after any UI action. It must never trade on a forecast alone.

## Setup on Windows

1. Create the virtual environment: `py -3.12 -m venv .venv`
2. Install dependencies: `.venv\Scripts\python.exe -m pip install -r requirements.txt`
3. Verify/download genuine CoinDCX history: `python check_coindcx.py` and `python download_data.py --pair auto --start-date 2018-10-18`
4. Train baseline models: `python train_model.py`
5. Train the multi-horizon models: `python train_multihorizon.py`
6. Load unpacked extension folder in `chrome://extensions`.
7. Copy its ID and run `run_overlay_windows.bat`; paste the ID.
8. Open the exact supported BTC/INR chart URL.

## Model evaluation

The direct horizon models predict each horizon from known features. Training purges 15 rows at split boundaries to reduce overlapping-label leakage. The holdout report must be inspected before using any horizon. Residual 5th–95th percentile bands are empirical ranges, not guaranteed confidence intervals. No model should be called profitable without a separate walk-forward strategy simulation that includes fees, spread, slippage, order rejection, and missed execution.

## Renderer status

The extension currently receives forecasts but intentionally does not draw them until chart bounds, price-axis mapping, and candle-time spacing can be identified and verified on the real CoinDCX chart. It must hide forecasts when those measurements are ambiguous. OCR/OpenCV must not silently replace API-verified candles.

## CUA status and release gates

The current `src/execution_policy.py` is a policy gate only; it never executes an order. Live CUA remains disabled until all gates pass:

1. Stable chart/page detection and no accidental clicks on unrelated controls.
2. Forecast renderer verified against known chart prices and timestamps.
3. Historical walk-forward evaluation and fee-aware strategy baseline comparison.
4. Paper trading with logged signals, hypothetical fills, and drawdown.
5. Browser failure/recovery tests: stale data, lost connection, changed DOM, rejected order, partial fill, duplicate submission, and kill switch.
6. Per-order human confirmation showing side, quantity, limit price, estimated fees, and maximum loss.
7. Explicitly separate user action to enable any live mode; never auto-enable it after restart.

Until then, this is a research and paper-trading build, not a live trading bot.
