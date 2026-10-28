# League of Legends Last Hit (Non-Injection)

This repository provides a **read-only, non-injection** visual companion for identifying and prioritizing last-hittable minions in League of Legends from desktop screen capture.


## Key Guarantees
- **Zero Injection**: Performs **no** game process/memory reading or DLL injection, and **no** synthetic mouse/keyboard input (`pynput` removed).
- **Only Last-Hittable Minions Shown**: Minions whose conservative upper-bound HP exceeds your configured effective attack damage—or whose detection/health-bar reading is ambiguous, clipped, or low-confidence—are excluded and never rendered.
- **Minion Prioritization Preserved**: Eligible last-hittable minions are ranked by distance to the player anchor (`#1` nearest priority target highlighted in yellow, `#2+` secondary last-hittable minions in green), with deterministic tie-breaking by remaining HP upper bound and bounding box.

## Installation

### Windows (Batch File)
Double-click **`install_requirements.bat`** or run it from Command Prompt / PowerShell:
```bat
install_requirements.bat
```
This script automatically:
1. Detects your Python 3 installation and creates a local `.venv` virtual environment (pass `--system` to install globally instead).
2. Upgrades `pip`, `setuptools`, and `wheel`.
3. Installs all dependencies from `requirements.txt`.
4. Clones `WongKinYiu/yolov9` into `./yolov9` if it is not already present and automatically downloads `./weights/best.pt`.

### Linux / macOS (Shell Script)
```bash
./install_requirements.sh
```

### Model Weights
`install_requirements.bat`, `install_requirements.sh`, and `main.py` all automatically download `weights/best.pt` on first run if it is missing. If you prefer to download it manually, grab the trained YOLOv9 weights from [this link](https://drive.google.com/file/d/135Bt-YX-Aqa_oaBxN220YSe9uGWiuljo/view?usp=share_link) and place `best.pt` inside `./weights/best.pt` (placing it in the project root or `~/Downloads/best.pt` is also detected automatically).

## Usage

On Windows, you can launch the companion using **`run.bat`** (which automatically uses `.venv`, upgrades CPU-only PyTorch to CUDA PyTorch if an NVIDIA GPU is detected, and installs requirements if needed):
```bat
run.bat --monitor_idx 1 --visualize True --minion_color red --H 720 --W 1280 --effective_damage 80
```

Or run `main.py` directly with Python:
```bash
python main.py --monitor_idx 1 --visualize True --minion_color red --H 720 --W 1280 --effective_damage 80
```

Inspect available monitor indices:
```bash
python main.py --list-monitors
```

### Parameters
- `--monitor_idx`: Monitor index to capture screenshots from.
- `--visualize` / `--no-visualize`: Enable or disable the read-only preview window.
- `--minion_color`: Enemy minion color to evaluate (`red` or `blue`).
- `--H`, `--W`: Target frame height and width (default `720` x `1280` for low latency).
- `--imgsz`: YOLOv9 inference resolution (default `640` for low latency; `960` for maximum range).
- `--sync`: Run capture and inference synchronously instead of the default low-latency asynchronous pipeline.
- `--effective_damage`: Conservative lower bound on post-mitigation basic-attack damage.
- `--caster_max_hp`, `--melee_max_hp`, `--siege_max_hp`: Conservative upper bounds on minion maximum HP.
- `--health_margin`: Additional fractional safety margin added to measured health bar fraction (default `0.0`).
- `--min_confidence`: Minimum detection confidence for minions and health bars (default `0.25`).
- `--player_anchor`: Normalized `(x, y)` anchor in `[0, 1]` for distance prioritization (default `0.5 0.5`).
- `--max_distance`: Optional maximum pixel distance from the player anchor.
