@echo off
setlocal EnableExtensions

cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [INFO] Virtual environment not found. Running install_requirements.bat first...
    call "%~dp0install_requirements.bat" --no-pause
    if errorlevel 1 exit /b 1
)

set "VENV_PY=.venv\Scripts\python.exe"
set "CHECK_ONLY="

rem ---- --check-devices only reports; it never installs or changes packages ----
:scan_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--check-devices" set "CHECK_ONLY=1"
shift
goto :scan_args
:args_done
if defined CHECK_ONLY goto :run

where nvidia-smi >nul 2>nul
if not errorlevel 1 goto :nvidia

rem ---- no NVIDIA driver: is DirectML usable for an AMD/Intel GPU? ----
"%VENV_PY%" -c "import torch_directml; raise SystemExit(0 if torch_directml.is_available() else 1)" >nul 2>nul
if not errorlevel 1 goto :run

"%VENV_PY%" -c "import torch_directml" >nul 2>nul
if not errorlevel 1 (
    echo [HINT] torch-directml is installed but reports no usable device.
    echo        Update your AMD/Intel GPU driver, then check with:  run.bat --check-devices
) else (
    echo [HINT] No NVIDIA GPU and no DirectML PyTorch found, so inference would run on the CPU.
    echo        torch-directml is a SEPARATE package from PyTorch - install it for AMD/Intel GPUs:
    echo            .venv\Scripts\python.exe -m pip install torch-directml
    echo        or:
    echo            install_requirements.bat --directml
    echo        Diagnose at any time with:  run.bat --check-devices
)
goto :run

:nvidia
"%VENV_PY%" -c "import torch; raise SystemExit(0 if torch.cuda.is_available() else 1)" >nul 2>nul
if not errorlevel 1 goto :run
echo [INFO] NVIDIA GPU detected with CPU-only PyTorch. Upgrading to CUDA PyTorch for low latency...
"%VENV_PY%" -m pip install --upgrade --force-reinstall torch torchvision --index-url https://download.pytorch.org/whl/cu124

:run
"%VENV_PY%" main.py %*
exit /b %ERRORLEVEL%
