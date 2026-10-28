"""Explain *why* no last-hit marker is visible.

When nothing is drawn, the useful question is: did the detector see minions at all,
did the health bars match, did the HP filter reject everything, or were the boxes
simply too old?  This module turns the per-frame pipeline numbers into one log line
plus an actionable hint, and can dump annotated debug frames to disk.

It never changes detection results: it only observes.
"""

import threading
from dataclasses import dataclass, field
from math import isfinite
from pathlib import Path
from time import monotonic

import cv2
import numpy as np

from my_utils.visualizers.annotator import AIAnnotator

BODY_BOX_COLOR = (255, 128, 0)     # orange (BGR): every detected minion body
BAR_BOX_COLOR = (255, 0, 255)      # magenta (BGR): every detected health bar
TEXT_COLOR = (255, 255, 255)

MIN_VISIBLE_BRIGHTNESS = 4.0       # below this the capture is black / protected


@dataclass
class FrameStats:
    """Numbers that explain one processed frame."""

    brightness: float = 0.0
    bodies: int = 0
    bars: int = 0
    other_color_bodies: int = 0
    other_color_bars: int = 0
    matched: int = 0
    yolo_matches: int = 0
    locator_matches: int = 0
    eligible: int = 0
    kept: int = 0
    held: int = 0
    health: list[float] = field(default_factory=list)
    stale: bool = False
    detection_age: float | None = None
    inference_seconds: float | None = None

    @property
    def candidates(self) -> int:
        return self.kept


def _count(detection_result, key):
    try:
        return len(detection_result.get(key) or [])
    except AttributeError:
        return 0


def stats_from_frame(
    image,
    detection_result,
    minion_color: str = "red",
    pipeline: dict | None = None,
    detection_age_seconds: float | None = None,
    inference_seconds: float | None = None,
) -> FrameStats:
    """Combine raw detections, pipeline counters and image brightness."""
    pipeline = pipeline or {}
    other = "blue" if minion_color == "red" else "red"
    brightness = 0.0
    if isinstance(image, np.ndarray) and image.size:
        # Sub-sampled mean: same information, ~16x cheaper on a 1080p frame.
        brightness = float(image[::4, ::4].mean())

    stats = FrameStats(
        brightness=brightness,
        bodies=_count(detection_result, f"{minion_color}_minions"),
        bars=_count(detection_result, f"{minion_color}_health_bars"),
        other_color_bodies=_count(detection_result, f"{other}_minions"),
        other_color_bars=_count(detection_result, f"{other}_health_bars"),
        matched=int(pipeline.get("matched", 0)),
        yolo_matches=int(pipeline.get("yolo_matches", 0)),
        locator_matches=int(pipeline.get("fallback_matches", 0)),
        eligible=int(pipeline.get("eligible", 0)),
        kept=int(pipeline.get("kept", 0)),
        held=int(pipeline.get("held", 0)),
        health=[float(v) for v in pipeline.get("health", []) if isfinite(float(v))],
        stale=bool(pipeline.get("stale", False)),
        detection_age=detection_age_seconds,
        inference_seconds=inference_seconds,
    )
    return stats


def _plural(count: int, noun: str, plural: str | None = None) -> str:
    if count == 1:
        return f"{count} {noun}"
    if plural is None:
        plural = f"{noun[:-1]}ies" if noun.endswith("y") else f"{noun}s"
    return f"{count} {plural}"


def _ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value * 1000:.0f}ms"


def format_stats(
    stats: FrameStats,
    minion_color: str = "red",
    timings: dict | None = None,
) -> str:
    """One compact line with everything needed to diagnose a missing marker."""
    health = ", ".join(f"{100 * h:.0f}%" for h in stats.health) or "-"
    line = (
        f"brightness={stats.brightness:.1f} | "
        f"{minion_color} bodies={stats.bodies} bars={stats.bars} | "
        f"other-color bodies={stats.other_color_bodies} bars={stats.other_color_bars} | "
        f"matched={stats.matched} (yolo bars={stats.yolo_matches}, "
        f"geo bars={stats.locator_matches}) hp=[{health}] | "
        f"eligible={stats.eligible} drawn={stats.kept}"
        + (f" (held {stats.held})" if stats.held else "")
        + " | "
        f"infer={_ms(stats.inference_seconds)} box_age={_ms(stats.detection_age)}"
    )
    if timings:
        line += " | " + " ".join(
            f"{name}={value * 1000:.1f}ms" for name, value in timings.items()
        )
    return line


def no_marker_hint(stats: FrameStats, minion_color: str = "red") -> str | None:
    """Return the single most likely reason no marker is drawn (None if markers exist)."""
    if stats.candidates > 0:
        return None

    if stats.brightness < MIN_VISIBLE_BRIGHTNESS:
        return (
            "the captured screen is black. League of Legends must run in 'Borderless' or "
            "'Windowed' mode; exclusive fullscreen cannot be captured."
        )

    if stats.stale:
        return (
            "detector boxes were too old, so markers were suppressed. "
            f"infer={_ms(stats.inference_seconds)}, box_age={_ms(stats.detection_age)}. "
            "Speed up inference (install GPU support) or raise --max_detection_age "
            "(0 disables the limit)."
        )

    if stats.bodies == 0 and stats.bars == 0:
        if stats.other_color_bodies or stats.other_color_bars:
            return (
                f"only {('blue' if minion_color == 'red' else 'red')} minions were detected - "
                f"you are looking for {minion_color} ones. Try --minion_color "
                f"{'blue' if minion_color == 'red' else 'red'}."
            )
        return (
            "no minions and no health bars were detected at all. Check that the game is "
            "visible on the captured monitor, then try --min_confidence 0.15 --imgsz 960 "
            "(and --debug to save annotated frames)."
        )

    if stats.bodies > 0 and stats.matched == 0:
        return (
            f"{_plural(stats.bodies, 'minion body', 'minion bodies')} detected but no readable "
            f"health bar was found for them ({_plural(stats.bars, 'YOLOv9 bar')}, no geometric "
            "reading either). If YOLOv9 reports no bars, lower --min_bar_confidence (0.15); for "
            "small or overlapping bars raise --imgsz 960. Check frames/debug/latest.png."
        )

    if stats.eligible == 0:
        return (
            f"{_plural(stats.matched, 'minion')} read, but none is last-hittable with the "
            "current damage/HP model. Raise --effective_damage, lower --health_margin, or "
            "lower the --caster_max_hp/--melee_max_hp/--siege_max_hp bounds."
        )

    if stats.kept == 0:
        return (
            f"{_plural(stats.eligible, 'last-hittable minion')} found but hidden by the "
            "confirmation gate: lower --confirm_frames (1 disables it)."
        )

    return None


def annotate_debug_frame(image, detection_result, candidates, minion_color: str = "red"):
    """Draw *all* raw detections plus the confirmed markers, for troubleshooting."""
    if image is None:
        raise ValueError("image is required")
    canvas = image.copy()
    height, width = canvas.shape[:2]

    def draw(boxes, color, label):
        for entry in boxes or []:
            try:
                box = entry[0] if isinstance(entry, (tuple, list)) else entry
                x1, y1, x2, y2 = (int(round(float(v))) for v in box[:4])
            except (TypeError, ValueError, IndexError):
                continue
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)
            cv2.putText(
                canvas,
                label,
                (max(0, min(x1, width - 60)), max(12, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                color,
                1,
                cv2.LINE_AA,
            )

    draw(detection_result.get(f"{minion_color}_minions"), BODY_BOX_COLOR, "body")
    draw(detection_result.get(f"{minion_color}_health_bars"), BAR_BOX_COLOR, "bar")
    if candidates:
        canvas = AIAnnotator()(candidates=candidates, img=canvas, copy=False)
    cv2.putText(
        canvas,
        f"{minion_color} minions / health bars (debug)",
        (8, height - 8),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        TEXT_COLOR,
        1,
        cv2.LINE_AA,
    )
    return canvas


def describe_thresholds(settings, minion_color: str = "red") -> list[str]:
    """Explain which HP fraction counts as last-hittable for each minion type."""
    lines = [
        f"effective_damage={settings.effective_damage:.0f} "
        f"health_margin={settings.health_margin:g} min_confidence={settings.min_confidence:g}"
    ]
    for minion_type in ("caster", "melee", "siege"):
        max_hp = settings.max_health.get(minion_type)
        if not max_hp:
            continue
        # health_upper_bound = min(1, hp + margin) * max_hp <= effective_damage
        fraction = settings.effective_damage / max_hp - settings.health_margin
        label = f"{minion_color} {minion_type}"
        if fraction <= 0:
            lines.append(f"{label:<18}: never last-hittable (damage below the margin)")
        elif fraction >= 1:
            lines.append(f"{label:<18}: any HP (whole bar would be lethal)")
        else:
            lines.append(
                f"{label:<18}: only while the bar reads <= {100 * fraction:.0f}% HP"
            )
    return lines


class DebugReporter:
    """Print pipeline statistics and (optionally) write annotated debug frames."""

    def __init__(
        self,
        minion_color: str = "red",
        frames_dir: str | Path | None = None,
        interval: float = 1.0,
        print_lines: bool = True,
        async_writes: bool = True,
    ) -> None:
        self.minion_color = minion_color
        self.frames_dir = Path(frames_dir) if frames_dir else None
        self.interval = max(0.0, float(interval))
        self.print_lines = print_lines
        # Encoding a 1280x720 PNG costs 15-25 ms; doing it on the capture loop shows up
        # as a hitch in the game, so it is handed to a writer thread.
        self.async_writes = bool(async_writes)
        self._last_print = 0.0
        self._last_save = 0.0
        self._last_hint: str | None = None
        self.last_saved_path: Path | None = None
        self._pending_write: tuple[Path, np.ndarray] | None = None
        self._write_condition = threading.Condition()
        self._writer: threading.Thread | None = None
        self._closed = False
        if self.frames_dir is not None:
            self.frames_dir.mkdir(parents=True, exist_ok=True)

    def report(
        self,
        stats: FrameStats,
        force: bool = False,
        timings: dict | None = None,
    ) -> str | None:
        """Print the stats line + hint (rate limited). Returns the hint if it changed."""
        now = monotonic()
        hint = no_marker_hint(stats, self.minion_color)
        hint_changed = hint != self._last_hint
        if self.print_lines and (force or hint_changed or now - self._last_print >= self.interval):
            self._last_print = now
            print(f"[DEBUG] {format_stats(stats, self.minion_color, timings)}")
            if hint:
                print(f"[DEBUG] no markers: {hint}")
        self._last_hint = hint
        return hint

    def save(
        self,
        image,
        detection_result,
        candidates,
        force: bool = False,
    ) -> Path | None:
        """Write an annotated debug frame (PNG, lossless) and return its path."""
        if self.frames_dir is None:
            return None
        now = monotonic()
        if not force and now - self._last_save < max(self.interval, 0.2):
            return None
        self._last_save = now
        try:
            annotated = annotate_debug_frame(
                image, detection_result, candidates, self.minion_color
            )
        except Exception:
            return None
        path = self.frames_dir / "latest.png"
        self.last_saved_path = path
        if not self.async_writes:
            try:
                return path if cv2.imwrite(str(path), annotated) else None
            except Exception:
                return None
        with self._write_condition:
            if self._closed:
                return None
            self._pending_write = (path, annotated)
            if self._writer is None:
                self._writer = threading.Thread(
                    target=self._write_loop, name="LastHitDebugWriter", daemon=True
                )
                self._writer.start()
            self._write_condition.notify()
        return path

    def _write_loop(self) -> None:
        while True:
            with self._write_condition:
                while self._pending_write is None and not self._closed:
                    self._write_condition.wait()
                if self._pending_write is None:
                    return
                path, image = self._pending_write
                self._pending_write = None
            try:
                cv2.imwrite(str(path), image)
            except Exception:
                pass

    def close(self, timeout: float = 2.0) -> None:
        """Flush a pending debug frame and stop the writer thread."""
        with self._write_condition:
            self._closed = True
            pending = self._pending_write
            self._pending_write = None
            self._write_condition.notify_all()
        if pending is not None:
            try:
                cv2.imwrite(str(pending[0]), pending[1])
            except Exception:
                pass
        writer = self._writer
        if writer is not None and writer.is_alive():
            writer.join(timeout=timeout)
