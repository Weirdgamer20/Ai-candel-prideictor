@echo off
setlocal
cd /d "%~dp0"
set "PROJECT_ROOT=%CD%"
if not exist ".venv\Scripts\python.exe" (
  echo Create the environment first: py -3.12 -m venv .venv
  echo Then run: .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
if not exist "data\raw\btc_inr_15m.csv" (
  echo Historical data is missing. Run:
  echo .venv\Scripts\python.exe download_data.py --pair auto --start-date 2018-10-18
  pause
  exit /b 1
)
set /p BTCINR_EXTENSION_ID=Paste the extension ID from chrome://extensions:
if "%BTCINR_EXTENSION_ID%"=="" (
  echo Extension ID is required. Load extension\ first, then rerun this launcher.
  pause
  exit /b 1
)
start "BTC INR Data Updater" powershell -NoExit -Command "$env:BTCINR_EXTENSION_ID='%BTCINR_EXTENSION_ID%'; Set-Location -LiteralPath '%PROJECT_ROOT%'; & '.\.venv\Scripts\python.exe' -m src.live_data_service"
start "BTC INR Prediction API" powershell -NoExit -Command "$env:BTCINR_EXTENSION_ID='%BTCINR_EXTENSION_ID%'; Set-Location -LiteralPath '%PROJECT_ROOT%'; & '.\.venv\Scripts\python.exe' -m uvicorn src.prediction_api:app --host 127.0.0.1 --port 8765"
echo Started read-only local services. Train models before expecting forecasts.
