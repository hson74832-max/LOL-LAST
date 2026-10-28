# League of Legends Last Hit Companion (Non-Injection)

This repository provides a **read-only, non-injection** visual companion for identifying and prioritizing last-hittable minions in League of Legends from desktop screen capture.

![](figures/futuristic.png)

## Key Guarantees
- **Zero Injection**: Performs **no** game process/memory reading or DLL injection, and **no** synthetic mouse/keyboard input (`pynput` removed).
- **Only Last-Hittable Minions Shown**: Minions whose conservative upper-bound HP exceeds your configured effective attack damage—or whose detection/health-bar reading is ambiguous, clipped, or low-confidence—are excluded and never rendered.
- **Minion Prioritization Preserved**: Eligible last-hittable minions are ranked by distance to the player anchor (`#1` nearest priority target highlighted in yellow, `#2+` secondary last-hittable minions in green), with deterministic tie-breaking by remaining HP upper bound and bounding box.

## Usage

Run the companion using `main.py`:
```bash
python main.py --monitor_idx 1 --visualize True --minion_color red --H 540 --W 960 --effective_damage 80
```

Inspect available monitor indices:
```bash
python main.py --list-monitors
```

### Parameters
- `--monitor_idx`: Monitor index to capture screenshots from.
- `--visualize` / `--no-visualize`: Enable or disable the read-only preview window.
- `--minion_color`: Enemy minion color to evaluate (`red` or `blue`).
- `--H`, `--W`: Target frame height and width.
- `--effective_damage`: Conservative lower bound on post-mitigation basic-attack damage.
- `--caster_max_hp`, `--melee_max_hp`, `--siege_max_hp`: Conservative upper bounds on minion maximum HP.
- `--health_margin`: Additional fractional safety margin added to measured health bar fraction (default `0.02`).
- `--min_confidence`: Minimum detection confidence for minions and health bars (default `0.6`).
- `--player_anchor`: Normalized `(x, y)` anchor in `[0, 1]` for distance prioritization (default `0.5 0.5`).
- `--max_distance`: Optional maximum pixel distance from the player anchor.
