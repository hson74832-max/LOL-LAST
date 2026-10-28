@echo off
setlocal EnableExtensions

cd /d "%~dp0"

set "USE_SYSTEM=0"
set "NO_PAUSE=0"

:parse_args
if "%~1"=="" goto :args_done
if /I "%~1"=="--system" set "USE_SYSTEM=1"
if /I "%~1"=="--no-pause" set "NO_PAUSE=1"
shift
goto :parse_args
:args_done

echo ============================================================
echo  League of Legends Last Hit Companion - Dependency Installer
echo ============================================================

set "PYTHON_CMD="
where python >nul 2>nul
if not errorlevel 1 (
    python -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
    if not errorlevel 1 set "PYTHON_CMD=python"
)

if not defined PYTHON_CMD (
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=py -3"
    )
)

if not defined PYTHON_CMD (
    echo [ERROR] Python 3.9 or newer was not found on PATH.
    echo Please install Python from https://www.python.org/downloads/ and check "Add Python to PATH".
    goto :fail
)

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found in %CD%.
    goto :fail
)

if "%USE_SYSTEM%"=="1" (
    set "TARGET_PYTHON=%PYTHON_CMD%"
    echo [INFO] Installing packages into the system Python environment...
) else (
    if not exist ".venv\Scripts\python.exe" (
        echo [INFO] Creating virtual environment in .venv ...
        %PYTHON_CMD% -m venv .venv
        if errorlevel 1 (
            echo [ERROR] Failed to create virtual environment in .venv.
            goto :fail
        )
    ) else (
        echo [INFO] Using existing virtual environment in .venv ...
    )
    set "TARGET_PYTHON=.venv\Scripts\python.exe"
)

echo [INFO] Upgrading pip, setuptools, and wheel...
%TARGET_PYTHON% -m pip install --upgrade pip "setuptools<81.0.0" wheel
if errorlevel 1 (
    echo [ERROR] Failed to upgrade pip/setuptools/wheel.
    goto :fail
)

echo [INFO] Installing Python requirements from requirements.txt...
%TARGET_PYTHON% -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install packages from requirements.txt.
    goto :fail
)

if not exist "yolov9\models\common.py" (
    echo [INFO] Setting up YOLOv9 repository in .\yolov9 ...
    where git >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] Git is not installed or not on PATH, so .\yolov9 could not be cloned automatically.
        echo Please install Git from https://git-scm.com/ or extract YOLOv9 into .\yolov9.
        goto :fail
    )
    git clone --depth 1 https://github.com/WongKinYiu/yolov9.git yolov9
    if errorlevel 1 (
        echo [ERROR] Failed to clone https://github.com/WongKinYiu/yolov9.git into .\yolov9.
        goto :fail
    )
) else (
    echo [INFO] .\yolov9 is already present.
)

if not exist "weights" mkdir "weights"
if not exist "weights\best.pt" (
    echo [NOTE] Model weights were not found at .\weights\best.pt.
    echo        Download the trained YOLOv9 weights and place them at .\weights\best.pt before running.
)

echo.
echo [SUCCESS] All requirements are installed!
echo           To start the companion, run:
echo               run.bat
echo           or:
echo               %TARGET_PYTHON% main.py --monitor_idx 1 --visualize True --minion_color red
if "%NO_PAUSE%"=="0" pause
exit /b 0

:fail
echo.
echo [FAILED] Installation did not complete successfully.
if "%NO_PAUSE%"=="0" pause
exit /b 1
