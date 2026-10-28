"""Helpers that keep the companion from stealing frames from the game.

Running inference as fast as the loop can submit frames keeps the GPU (or CPU)
permanently busy, which is what makes the *game* stutter.  These helpers pace the
capture loop and the detector, lower our own process priority, and collect
per-stage timings so it is obvious where the remaining time goes.

Nothing here touches the game process: priorities are applied to this Python
process only.
"""

import os
import time
from math import isfinite

PRIORITY_LEVELS = ("normal", "below", "lowest")


def validate_fps(value, name: str = "fps", allow_zero: bool = True) -> float:
    """Return a validated frame-rate cap (``0`` means unlimited when allowed)."""
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number") from exc
    if not isfinite(number) or number < 0:
        raise ValueError(f"{name} must be a finite, non-negative number")
    if number == 0 and not allow_zero:
        raise ValueError(f"{name} must be greater than zero")
    return number


class FpsLimiter:
    """Sleep just enough to keep a loop at (or below) a target rate."""

    def __init__(self, fps: float = 0.0) -> None:
        self.fps = validate_fps(fps, "fps")
        self.interval = 1.0 / self.fps if self.fps else 0.0
        self._next_at = 0.0

    def sleep(self) -> float:
        """Block until the next slot. Returns the slept seconds (0 when unlimited)."""
        if not self.interval:
            return 0.0
        now = time.perf_counter()
        if not self._next_at:
            self._next_at = now + self.interval
            return 0.0
        delay = self._next_at - now
        if delay > 0:
            time.sleep(delay)
            self._next_at += self.interval
            return delay
        # We are behind schedule: do not pile up debt, restart the cadence.
        self._next_at = now + self.interval
        return 0.0


class StageTimer:
    """Rolling per-stage timings (capture, inference, drawing, ...)."""

    def __init__(self, window: int = 30) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        self.window = int(window)
        self._samples: dict[str, list[float]] = {}

    def add(self, name: str, seconds: float) -> None:
        if not isfinite(seconds) or seconds < 0:
            return
        samples = self._samples.setdefault(name, [])
        samples.append(float(seconds))
        if len(samples) > self.window:
            del samples[0 : len(samples) - self.window]

    def measure(self, name: str):
        """Context manager: ``with timer.measure("capture"): ...``."""
        timer = self

        class _Measurement:
            def __enter__(self):
                self._started = time.perf_counter()
                return self

            def __exit__(self, *_exc):
                timer.add(name, time.perf_counter() - self._started)
                return False

        return _Measurement()

    def averages(self) -> dict[str, float]:
        return {
            name: sum(values) / len(values)
            for name, values in self._samples.items()
            if values
        }

    def reset(self) -> None:
        self._samples.clear()

    def format(self, averages: dict[str, float] | None = None) -> str:
        values = self.averages() if averages is None else averages
        if not values:
            return ""
        return " ".join(f"{name}={value * 1000:.1f}ms" for name, value in values.items())


def apply_process_priority(level: str = "below", process=None) -> str | None:
    """Lower this process' scheduling priority so the game keeps its CPU time.

    Returns a human readable status, or None when psutil is unavailable. Never raises
    for environmental failures: a priority change is a bonus, not a requirement.
    """
    if level not in PRIORITY_LEVELS:
        raise ValueError(f"priority must be one of {', '.join(PRIORITY_LEVELS)}")
    try:
        import psutil  # noqa: PLC0415  (optional dependency)
    except Exception:
        return None

    try:
        target = process if process is not None else psutil.Process(os.getpid())
        if os.name == "nt":
            windows_levels = {
                "normal": psutil.NORMAL_PRIORITY_CLASS,
                "below": psutil.BELOW_NORMAL_PRIORITY_CLASS,
                "lowest": psutil.IDLE_PRIORITY_CLASS,
            }
            target.nice(windows_levels[level])
        else:
            target.nice({"normal": 0, "below": 10, "lowest": 19}[level])
    except Exception as exc:  # pragma: no cover - depends on the OS/permissions
        return f"could not change process priority ({type(exc).__name__}: {exc})"
    return f"process priority set to '{level}'"


def set_torch_threads(count: int) -> str | None:
    """Limit intra-op threads (CPU inference) so the game keeps cores free."""
    try:
        count = int(count)
    except (TypeError, ValueError) as exc:
        raise ValueError("torch thread count must be an integer") from exc
    if count < 1:
        raise ValueError("torch thread count must be at least 1")
    try:
        import torch  # noqa: PLC0415
    except Exception:
        return None
    try:
        torch.set_num_threads(count)
    except Exception as exc:  # pragma: no cover - torch build dependent
        return f"could not set torch threads ({type(exc).__name__})"
    return f"torch intra-op threads limited to {count}"


def capture_and_processing_tips(imgsz: int, frame_w: int | None, frame_h: int | None) -> list[str]:
    """Concrete suggestions for this configuration, cheapest win first."""
    tips = []
    if imgsz > 512:
        tips.append(
            f"--imgsz 512 (or 480) instead of {imgsz}: roughly 35-45% less GPU time per frame"
        )
    if frame_w and frame_h and frame_w * frame_h > 1280 * 720:
        tips.append(
            "--H 720 --W 1280 (or --H 540 --W 960): smaller frames mean cheaper captures, "
            "health-bar reads and window drawing"
        )
    tips.append("--infer_fps 10: the biggest win - detection at 10-12 fps is plenty for last hits")
    tips.append("--no-visualize if you only use the overlay: no preview window to draw")
    tips.append("--priority lowest: gives the game's threads priority over this companion")
    return tips
