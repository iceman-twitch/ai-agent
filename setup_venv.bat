@echo off
setlocal
cd /d "%~dp0"

python -m venv .venv
if errorlevel 1 (
    echo Failed to create the virtual environment.
    exit /b 1
)

.venv\Scripts\python.exe -m pip install -r requirements.txt
if errorlevel 1 (
    echo Failed to install requirements.
    exit /b 1
)

echo Virtual environment is ready.
