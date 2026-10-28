@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual environment not found. Running install_requirements.bat first...
    call "%~dp0install_requirements.bat" --no-pause
    if errorlevel 1 exit /b 1
)

where nvidia-smi >nul 2>nul
if not errorlevel 1 (
    ".venv\Scripts\python.exe" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
    if errorlevel 1 (
        echo [INFO] NVIDIA GPU detected with CPU-only PyTorch. Upgrading to CUDA PyTorch for low latency...
        ".venv\Scripts\python.exe" -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124
    )
)

".venv\Scripts\python.exe" main.py %*
exit /b %ERRORLEVEL%
