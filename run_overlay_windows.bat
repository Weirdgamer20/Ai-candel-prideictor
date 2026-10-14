@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Create the environment first: py -3.12 -m venv .venv
  echo Then run: .venv\Scripts\python.exe -m pip install -r requirements.txt
  pause
  exit /b 1
)
set /p BTCINR_EXTENSION_ID=Paste the extension ID from chrome://extensions:
if "%BTCINR_EXTENSION_ID%"=="" (
  echo Extension ID is required. Load extension\ first, then rerun this launcher.
  pause
  exit /b 1
)
start "BTC INR Data Updater" cmd /k "set BTCINR_EXTENSION_ID=%BTCINR_EXTENSION_ID%&& cd /d %CD% && .venv\Scripts\python.exe -m src.live_data_service"
start "BTC INR Prediction API" cmd /k "set BTCINR_EXTENSION_ID=%BTCINR_EXTENSION_ID%&& cd /d %CD% && .venv\Scripts\python.exe -m uvicorn src.prediction_api:app --host 127.0.0.1 --port 8765"
echo Started read-only local services. Multi-horizon model must be trained before forecasts are available.
