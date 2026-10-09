@echo off
setlocal
set "APP_DIR=%~dp0"

if not exist "%APP_DIR%.venv\Scripts\python.exe" (
    echo Virtual environment not found. Run setup_venv.bat first.
    exit /b 1
)

"%APP_DIR%.venv\Scripts\python.exe" "%APP_DIR%ensure_api_key.py"
if errorlevel 1 (
    exit /b 1
)

"%APP_DIR%.venv\Scripts\python.exe" "%APP_DIR%codeagent.py" %*
exit /b %errorlevel%
