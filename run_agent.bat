@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup_venv.bat first.
    exit /b 1
)

.venv\Scripts\python.exe ensure_api_key.py
if errorlevel 1 (
    exit /b 1
)

.venv\Scripts\python.exe codeagent.py %*
exit /b %errorlevel%
