@echo off
setlocal ENABLEDELAYEDEXPANSION

REM Determine script directory
set SCRIPT_DIR=%~dp0
cd /d %SCRIPT_DIR%

IF NOT EXIST .venv (
  echo Creating virtual environment (.venv)...
  REM Try system python, then py launcher
  where python >nul 2>nul
  if %errorlevel%==0 (
    python -m venv .venv
  ) else (
    py -3 -m venv .venv
  )
)

IF NOT EXIST .venv\Scripts\python.exe (
  echo Python executable not found in venv.
  where python >nul 2>nul
  if %errorlevel%==0 (
    echo Will attempt to use system 'python' from PATH.
  ) else (
    where py >nul 2>nul
    if %errorlevel%==0 (
      echo Will attempt to use 'py -3' launcher.
    ) else (
      echo No Python found on PATH. Please install Python 3 and try again.
      exit /b 1
    )
  )
)

REM Verify that the venv python is runnable. If it's broken (e.g., points to another user's path), recreate it.
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe -c "import sys" >nul 2>nul
  if not %errorlevel%==0 (
    echo Detected broken .venv python. Recreating .venv...
    rmdir /s /q .venv
    where python >nul 2>nul
    if %errorlevel%==0 (
      python -m venv .venv
    ) else (
      py -3 -m venv .venv
    )
  )
)

IF EXIST requirements.txt (
  echo Installing dependencies...
  if exist .venv\Scripts\python.exe (
    .venv\Scripts\python.exe -m pip install -r requirements.txt >nul
  ) else (
    where python >nul 2>nul
    if %errorlevel%==0 (
      python -m pip install -r requirements.txt >nul
    ) else (
      py -3 -m pip install -r requirements.txt >nul
    )
  )
)

echo Launching app...
if exist .venv\Scripts\python.exe (
  .venv\Scripts\python.exe app.py
) else (
  where python >nul 2>nul
  if %errorlevel%==0 (
    python app.py
  ) else (
    py -3 app.py
  )
)
