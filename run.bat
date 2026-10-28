@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual environment not found. Running install_requirements.bat first...
    call "%~dp0install_requirements.bat" --no-pause
    if errorlevel 1 exit /b 1
)

".venv\Scripts\python.exe" main.py %*
exit /b %ERRORLEVEL%
