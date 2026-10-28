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

## Letting the game keep its frames (performance)
The companion never injects anything, but it can still cost the game frames if it keeps the GPU and CPU busy back-to-back. Three dials control that:

| Dial | Default | What it does |
| --- | --- | --- |
| `--infer_fps` | `12` | Maximum YOLO inferences per second. **The biggest lever.** Each inference is ~25 ms on DirectML; at unlimited rate the GPU never idles. 10–15 fps of detection is plenty for last hits because health bars are re-read on every captured frame (that part is ~1–4 ms). |
| `--max_fps` | `30` | Capture/processing loop rate. Each iteration costs ~4.5 ms of CPU at 1280×720 (capture, bar reads, drawing), so an unlimited loop pegs a whole core. `30` keeps it near ~15 % of one core. |
| `--overlay_fps` | `30` | Overlay redraw rate (each redraw is a Tk window update). |
| `--priority` | `below` | Sets **this process'** OS priority (`below` = below normal, `lowest` = idle class on Windows). It lets Windows give the game's threads the CPU first. Never touches the game process. |
| `--torch_threads` | unset | Limits PyTorch intra-op threads; useful on CPU inference so the game keeps cores (e.g. `--torch_threads 4`). |

Recommended when the game stutters:
```bat
run.bat --imgsz 512 --H 720 --W 1280 --infer_fps 10 --max_fps 24 --priority lowest --no-visualize --bar_locator True --confirm_frames 2
```
Measured effect of `--infer_fps`: detection work drops from ~40 inferences/s to 10 — roughly **a 75 % cut in GPU time** — while markers stay responsive because health bars are still re-read up to `--max_fps` times per second.

Debug output now also reports per-stage timings, so you can see where time goes:
```
[DEBUG] ... | infer=24ms box_age=31ms | capture=6.1ms evaluate=4.3ms overlay=0.9ms preview=1.8ms loop=38.0ms
```
`--debug` itself is cheap now: brightness is measured on a 4×4 sub-sample and the annotated PNG is encoded on a background thread (it used to cost 15–25 ms per second on the capture loop).

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

### Check your backend first
```bat
run.bat --check-devices
```
prints Python/torch versions, whether CUDA, MPS and DirectML are usable, **why** the missing one is missing, the exact fix command, and the backend that would be selected. It never installs anything.

Example output on an AMD machine without DirectML installed:
```
[INFO] Inference backend diagnostics
[INFO] Python              : 3.11.9 (64-bit) at C:\...\.venv\Scripts\python.exe
[INFO] torch               : 2.9.0+cpu
[INFO] CUDA (NVIDIA)       : not available
[INFO] Apple MPS           : not available
[INFO] DirectML (AMD/Intel): not available - torch-directml is not installed (ModuleNotFoundError: No module named 'torch_directml')
[INFO]                       Fix: .venv\Scripts\python.exe -m pip install torch-directml
[INFO]                       (separate package from PyTorch; downgrades torch to 2.4.1)
[INFO] Selected backend    : cpu
[INFO] Expected speed     : slow (CPU). Markers will lag behind the game.
```

### "pytorch is already installed" but DirectML is missing
`torch` and `torch-directml` are **different packages** — installing PyTorch does not install DirectML. Also note:

- Install it **into the project's venv** (the interpreter `run.bat` uses), otherwise Python cannot see it:
  ```bat
  .venv\Scripts\python.exe -m pip install torch-directml
  ```
  or simply `install_requirements.bat --directml`.
- Windows only, **64-bit Python 3.8 – 3.12** (no wheels for 3.13+). The installer verifies this before trying.
- `torch-directml` pins `torch==2.4.1` and `torchvision==0.19.1`, so pip **downgrades** torch. That is expected and supported by this project.
- If `--device directml` is requested but unavailable, the companion prints the reason and the fix, then keeps running on CPU instead of crashing. Add `--strict_device` if you prefer a hard error.
- `torch_directml.is_available()` returning `False` on a machine that has it installed usually means an outdated AMD/Intel driver or a CPU-only torch build — `run.bat --check-devices` shows which one it is.

Check the first lines of console output, they state the device explicitly, e.g.:
```
[INFO] Detector initialized on device=privateuseone:0 [DirectML] (fp16=False, imgsz=(640, 640))
```

## Missing health bars no longer hide minions
YOLOv9's `*_health_bar` class is the weakest link in a real fight: reports of 8–9 detected minion bodies with 0–2 detected bars are common, and without a bar a minion can never be associated, so **no marker can be drawn at all**.

The associator therefore reads the bar **geometrically** from the captured pixels around each detected body: it looks for the League bar signature (saturated fill on the left, dark depleted part, ~1 px border) in a window above/around the body, requires a plausible width relative to the body and consistent rows, and uses the rightmost filled column as a conservative HP upper bound — exactly like the template-based reader.

- Every reported reading is still an upper bound, so the "only last-hittable" guarantee is unchanged.
- Bodies whose YOLO bar was rejected as clipped/ambiguous stay rejected; the locator only fills in bodies that had **no** bar.
- Seen in `--debug`: `matched=8 (yolo bars=2, geo bars=6)`.
- Disable with `--no-bar_locator` to require YOLOv9 bar detections only.

## "No markers at all" — start with `--debug`
If nothing is drawn (neither on the game nor in the preview), run with diagnostics; every second it prints what each pipeline stage saw and *why* nothing was shown:

```bat
run.bat --debug --max_detection_age 1 --min_confidence 0.2 --min_bar_confidence 0.2 --confirm_frames 1
```

Example output:
```
[INFO] effective_damage=80 health_margin=0 min_confidence=0.2
[INFO] red caster        : only while the bar reads <= 27% HP
[INFO] red melee         : only while the bar reads <= 16% HP
[INFO] red siege         : only while the bar reads <= 5% HP
[DEBUG] brightness=63.4 | red bodies=7 bars=5 | other-color bodies=0 bars=0 | matched=4 hp=[62%, 11%] | eligible=1 drawn=1 | infer=95ms box_age=120ms
```
and, when empty, one line naming the fix, e.g.:
```
[DEBUG] no markers: detector boxes were too old, so markers were suppressed. infer=580ms, box_age=620ms. Speed up inference (install GPU support) or raise --max_detection_age (0 disables the limit).
[DEBUG] no markers: 3 minion bodies detected but no health bars. Health bars score lower than bodies: lower --min_bar_confidence (try 0.15) or raise --imgsz 960 for small bars.
[DEBUG] no markers: 2 bodies and 1 bar were detected but none were matched to each other ...
[DEBUG] no markers: 1 minion read, but none is last-hittable with the current damage/HP model. Raise --effective_damage ...
[DEBUG] no markers: 1 last-hittable minion found but hidden by the confirmation gate: lower --confirm_frames (1 disables it).
[DEBUG] no markers: the captured screen is black. League of Legends must run in 'Borderless' or 'Windowed' mode ...
[DEBUG] no markers: only blue minions were detected - you are looking for red ones. Try --minion_color blue.
```

`--debug` also writes an annotated `frames/debug/latest.png` (orange = detected minion bodies, magenta = detected health bars, coloured boxes = confirmed markers), so you can look at what the model actually saw. Related flags: `--debug_frames_dir`, `--debug_interval`.

**Most common cause of "suddenly no markers":** a strict `--max_detection_age` (e.g. `0.2`) on a machine that is still running CPU inference (~300–600 ms per frame). Every box is then considered stale and suppressed — remove the flag or set `--max_detection_age 0` until `run.bat --check-devices` reports a GPU backend.

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

## Command reference
Everything is a flag on `run.bat` (Windows) or `main.py`; `run.bat --help` prints the same list.

Diagnostics / setup:
| Command | Purpose |
| --- | --- |
| `run.bat --check-devices` | Which backend (CUDA / MPS / DirectML / CPU) is usable and how to fix it |
| `run.bat --list-monitors` | Monitor indices for `--monitor_idx` |
| `run.bat --debug` | Per-frame diagnostics + `frames/debug/latest.png` + speed tips |
| `install_requirements.bat --directml` | Install AMD/Intel GPU support (torch-directml) |
| `install_requirements.bat --system` | Install into the system Python instead of `.venv` |

Display:
| Flag | Purpose |
| --- | --- |
| `--visualize` / `--no-visualize` | Capture preview window (markers drawn in it) |
| `--overlay` / `--no-overlay` | Click-through markers on the game screen |
| `--overlay_fps`, `--overlay_gap` | Overlay redraw rate, marker distance above the minion |
| `--preview_W`, `--preview_H`, `--mask_preview` | Preview window size, mask it out of the capture |
| `--monitor_idx` | Which monitor to capture |

Performance:
`--infer_fps`, `--max_fps`, `--priority`, `--torch_threads`, `--imgsz`, `--H`, `--W`, `--sync`.

Detection / strictness:
`--min_confidence`, `--min_bar_confidence`, `--max_detection_age`, `--confirm_frames`, `--bar_locator` / `--no-bar_locator`, `--stale_frame_fallback`, `--device`, `--strict_device`, `--weights`.

Last-hit maths:
`--effective_damage`, `--caster_max_hp`, `--melee_max_hp`, `--siege_max_hp`, `--health_margin`, `--minion_color`, `--player_anchor`, `--max_distance`.

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
- `--bar_locator` / `--no-bar_locator`: Read health bars geometrically around each body when YOLOv9 reports none (default on).
- `--infer_fps`: Maximum YOLO inferences per second (default `12`, `0` = unlimited).
- `--max_fps`: Maximum capture/processing frames per second (default `30`, `0` = unlimited).
- `--overlay_fps`: Maximum overlay redraws per second (default `30`).
- `--priority`: This process' priority: `normal`, `below` (default) or `lowest`.
- `--torch_threads`: Limit PyTorch intra-op threads (CPU inference).
- `--stale_frame_fallback`: Re-enable the legacy previous-frame fallback (off by default).
- `--debug` / `--no-debug`: Per-frame diagnostics plus a one-line explanation of why no marker is drawn; writes `frames/debug/latest.png`.
- `--debug_frames_dir`, `--debug_interval`: Where/how often the diagnostics write frames and lines.
- `--player_anchor`: Normalized `(x, y)` anchor in `[0, 1]` for distance prioritization (default `0.5 0.5`).
- `--max_distance`: Optional maximum pixel distance from the player anchor.
- `--device`: `cpu`, `cuda:0` (NVIDIA), `mps` (Apple) or `directml` (AMD/Intel on Windows). Auto-detected when omitted.
- `--strict_device`: Fail instead of falling back to CPU when the requested `--device` is unavailable.
- `--check-devices`: Print the inference-backend diagnostics (CUDA / MPS / DirectML / CPU) and exit.
