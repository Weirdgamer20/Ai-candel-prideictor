@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the environment first: py -3.12 -m venv .venv
  echo Then run: .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
start "BTC INR Data Updater" cmd /k ""%CD%\.venv\Scripts\python.exe" -m src.live_data_service"
start "BTC INR Prediction API" cmd /k ""%CD%\.venv\Scripts\python.exe" -m uvicorn src.prediction_api:app --host 127.0.0.1 --port 8765"
echo Started local read-only services. Forecast overlay remains hidden unless valid multi-horizon models and fresh data are available.
