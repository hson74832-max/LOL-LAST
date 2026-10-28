#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

USE_SYSTEM=0
for arg in "$@"; do
    case "$arg" in
        --system) USE_SYSTEM=1 ;;
    esac
done

echo "============================================================"
echo " League of Legends Last Hit Companion - Dependency Installer"
echo "============================================================"

PYTHON_CMD=""
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1; then
        if "$candidate" -c "import sys; raise SystemExit(0 if sys.version_info >= (3, 9) else 1)" >/dev/null 2>&1; then
            PYTHON_CMD="$candidate"
            break
        fi
    fi
done

if [[ -z "$PYTHON_CMD" ]]; then
    echo "[ERROR] Python 3.9 or newer was not found on PATH." >&2
    exit 1
fi

if [[ ! -f "requirements.txt" ]]; then
    echo "[ERROR] requirements.txt not found in $SCRIPT_DIR." >&2
    exit 1
fi

if [[ "$USE_SYSTEM" -eq 1 ]]; then
    TARGET_PYTHON="$PYTHON_CMD"
    echo "[INFO] Installing packages into the active Python environment..."
else
    if [[ ! -x ".venv/bin/python" ]]; then
        echo "[INFO] Creating virtual environment in .venv ..."
        "$PYTHON_CMD" -m venv .venv
    else
        echo "[INFO] Using existing virtual environment in .venv ..."
    fi
    TARGET_PYTHON=".venv/bin/python"
fi

echo "[INFO] Upgrading pip, setuptools, and wheel..."
"$TARGET_PYTHON" -m pip install --upgrade pip "setuptools<81.0.0" wheel

echo "[INFO] Installing Python requirements from requirements.txt..."
"$TARGET_PYTHON" -m pip install -r requirements.txt

if [[ ! -f "yolov9/models/common.py" ]]; then
    echo "[INFO] Setting up YOLOv9 repository in ./yolov9 ..."
    if ! command -v git >/dev/null 2>&1; then
        echo "[ERROR] git is not installed or not on PATH, so ./yolov9 could not be cloned automatically." >&2
        exit 1
    fi
    git clone --depth 1 https://github.com/WongKinYiu/yolov9.git yolov9
else
    echo "[INFO] ./yolov9 is already present."
fi

mkdir -p weights
if [[ ! -f "weights/best.pt" ]]; then
    echo "[NOTE] Model weights were not found at ./weights/best.pt."
    echo "       Download the trained YOLOv9 weights and place them at ./weights/best.pt before running."
fi

echo ""
echo "[SUCCESS] All requirements are installed!"
echo "          Run the companion with:"
echo "              $TARGET_PYTHON main.py --monitor_idx 1 --visualize True --minion_color red"
