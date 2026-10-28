# League of Legends Last Hit (Non-Injection)

This repository provides a **read-only, non-injection** visual companion for identifying and prioritizing last-hittable minions in League of Legends from desktop screen capture.


## Key Guarantees
- **Zero Injection**: Performs **no** game process/memory reading or DLL injection, and **no** synthetic mouse/keyboard input (`pynput` removed).
- **Only Last-Hittable Minions Shown**: Minions whose conservative upper-bound HP exceeds your configured effective attack damage—or whose detection/health-bar reading is ambiguous, clipped, or low-confidence—are excluded and never rendered.
- **Minion Prioritization Preserved**: Eligible last-hittable minions are ranked by distance to the player anchor (`#1` nearest priority target highlighted in yellow, `#2+` secondary last-hittable minions in green), with deterministic tie-breaking by remaining HP upper bound and bounding box.
- **Markers On Your Game Screen**: The prioritized minions are drawn by a **click-through, always-on-top overlay** directly on the captured desktop, not just inside the companion preview window. The overlay is a separate top-most Tk window: it never reads the game process, never sends input, and never takes keyboard focus (`--no-overlay` disables it).

## Display Modes
| Mode | Command | What you see |
| --- | --- | --- |
| Overlay + preview (default) | `run.bat --monitor_idx 1` | Triangle markers over the game plus the annotated preview window |
| Overlay only | `run.bat --monitor_idx 1 --no-visualize` | Only the click-through markers over the game (quit with `Ctrl+C`) |
| Preview only | `run.bat --monitor_idx 1 --no-overlay` | Markers only inside the preview window |

Overlay markers are small triangles drawn above each minion's head (`--overlay_gap`, default 16 px) so they never cover a health bar and never corrupt the next frame's health reading. Like screen capture, the overlay requires the game to run in **Borderless** or **Windowed** mode; Exclusive Fullscreen hides any external window.

## GPU Support (NVIDIA / AMD / Intel / Apple)
The detector runs on whatever backend is available; it never silently requires CUDA:

| Hardware | Backend | Install |
| --- | --- | --- |
| NVIDIA | CUDA (`cuda:0`, fp16) | `install_requirements.bat` (auto-detects `nvidia-smi`) |
| AMD / Intel on Windows | DirectML (`--device directml`) | `install_requirements.bat --directml` |
| Apple Silicon | MPS | default on macOS |
| Nothing usable | CPU | default fallback |

- DirectML is picked **automatically** when `torch-directml` is installed; force it with `--device directml`. It is Windows-only (on Linux use ROCm PyTorch).
- DirectML runs in fp32 (no fp16 support) and is slower than CUDA, but much faster than CPU.
- If the detector starts on CPU, `main.py` prints the exact commands to enable GPU inference — CPU inference is also the most common reason markers lag behind the game.

Check the first lines of console output, they state the device explicitly, e.g.:
```
[INFO] Detector initialized on device=privateuseone:0 [DirectML] (fp16=False, imgsz=(640, 640))
```

## Marker Strictness (no markers on dead minions)
Confusing or delayed markers come from two places: **stale detections** (boxes from an older frame are projected onto the live frame) and **one-frame misreads** (a body/bar box for a minion that just died). Both are now throttled:

| Flag | Default | Effect |
| --- | --- | --- |
| `--max_detection_age` | `0.35` s | Detector boxes older than this are **discarded** instead of being drawn on the live frame. Lower (`0.15`–`0.25`) = stricter/snappier; raise it if your machine cannot infer that fast; `0` disables the limit. |
| `--confirm_frames` | `2` | A candidate must be seen in that many **consecutive detector results** before it is drawn. Higher (3) = fewer flickers and no phantom markers, but slower to appear. `1` disables confirmation. |
| `--min_bar_confidence` | `0.20` | Minimum confidence for health bars (separate from bodies because low-HP bars score lower). Raise to `0.35`–`0.5` to reject corpse flashes and spell-effect artifacts. |
| `--min_confidence` | `0.25` | Minimum confidence for minion bodies. Raise to `0.4`–`0.5` to suppress phantom boxes. |
| `--stale_frame_fallback` | off | Legacy behaviour: when the live frame yields no match, re-run association on the previous frame. Off because that is exactly what produces delayed markers on dead minions. |

A marker disappears on the **first** frame a candidate is no longer detected — there is no hold time and no memory of dead minions. When detections are older than `--max_detection_age`, the companion prints a one-time warning naming the fix (GPU install or a larger age limit) instead of showing wrong markers.

Strictest useful setup on a mid-range GPU:
```bat
run.bat --monitor_idx 1 --device directml --confirm_frames 3 --max_detection_age 0.2 --min_bar_confidence 0.35
```

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

Overlay-only mode (markers over the game, no preview window):
```bash
python main.py --monitor_idx 1 --no-visualize --minion_color red --H 720 --W 1280 --effective_damage 80
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
- `--overlay` / `--no-overlay`: Enable or disable the click-through desktop overlay that draws last-hit markers on the game screen (default enabled; automatically skipped with a warning if the Python build has no `tkinter`).
- `--overlay_gap`: Distance in frame pixels between a minion's bounding box top and its overlay marker (default `16`).
- `--effective_damage`: Conservative lower bound on post-mitigation basic-attack damage.
- `--caster_max_hp`, `--melee_max_hp`, `--siege_max_hp`: Conservative upper bounds on minion maximum HP.
- `--health_margin`: Additional fractional safety margin added to measured health bar fraction (default `0.0`).
- `--min_confidence`: Minimum detection confidence for minion bodies (default `0.25`).
- `--min_bar_confidence`: Minimum detection confidence for health bars (default `0.20`).
- `--max_detection_age`: Maximum age in seconds of detector boxes reused on a live frame (default `0.35`, `0` disables).
- `--confirm_frames`: Consecutive detector results a candidate must be seen in before it is drawn (default `2`).
- `--stale_frame_fallback`: Re-enable the legacy previous-frame fallback (off by default).
- `--player_anchor`: Normalized `(x, y)` anchor in `[0, 1]` for distance prioritization (default `0.5 0.5`).
- `--max_distance`: Optional maximum pixel distance from the player anchor.
- `--device`: `cpu`, `cuda:0` (NVIDIA), `mps` (Apple) or `directml` (AMD/Intel on Windows). Auto-detected when omitted.
