"""Click-through desktop overlay that draws last-hit markers over the game.

The overlay is an ordinary top-most, colour-key transparent Tk window placed on
the captured monitor.  It never touches the game process, never sends input, and
never takes keyboard focus, so the companion stays fully read-only.

Markers are drawn *above* each minion's bounding box (never on top of its health
bar) which keeps the health-bar reading of the next captured frame intact.

Tkinter is imported lazily: the module stays importable on machines without Tk
(servers, headless CI) and :class:`OverlayUnavailable` is raised instead.
"""

from math import isfinite
from time import monotonic

import numpy as np

# Colour key used for the fully transparent window areas.  A saturated magenta
# is used because it effectively never occurs in a League of Legends frame.
OVERLAY_KEY_COLOR = "#ff00fe"
PRIMARY_TARGET_COLOR = "#ffff00"     # #1 priority last-hit target (yellow)
SECONDARY_TARGET_COLOR = "#00c800"   # #2+ last-hittable minions (green)
HELD_PRIMARY_COLOR = "#b8b800"       # dimmed yellow: marker held, minion not detected
HELD_SECONDARY_COLOR = "#009a00"     # dimmed green
LABEL_MODES = ("hp", "hp_damage", "percent")
LABEL_COLOR = "#ffffff"
LABEL_OUTLINE = "#000000"

DEFAULT_MARKER_GAP = 16.0            # frame pixels above the bounding box
MIN_SCALE = 0.2
MAX_SCALE = 5.0


class OverlayUnavailable(RuntimeError):
    """Raised when no click-through desktop overlay can be created."""


def _import_tkinter():
    try:
        import tkinter  # noqa: PLC0415  (kept lazy so headless environments still import this module)
    except Exception:  # pragma: no cover - depends on the interpreter build
        return None
    return tkinter


def display_scale(logical_width, physical_width) -> float:
    """Ratio between Tk's logical screen units and captured physical pixels.

    Tkinter reports coordinates in DPI-virtualised logical units on Windows while
    ``mss`` captures physical pixels.  Comparing both widths keeps the overlay
    aligned on 125%/150% scaled displays without any Windows API calls.
    """
    try:
        logical = float(logical_width)
        physical = float(physical_width)
    except (TypeError, ValueError):
        return 1.0
    if not (isfinite(logical) and isfinite(physical)) or logical <= 0 or physical <= 0:
        return 1.0
    return min(MAX_SCALE, max(MIN_SCALE, logical / physical))


def marker_polygon(bbox, gap: float = DEFAULT_MARKER_GAP, size: float | None = None):
    """Return a downward-pointing triangle sitting ``gap`` pixels above ``bbox``."""
    if len(bbox) != 4:
        raise ValueError("bbox must have four coordinates")
    values = tuple(float(v) for v in bbox)
    if not all(isfinite(v) for v in values):
        raise ValueError("bbox coordinates must be finite")
    x1, y1, x2, _ = values
    gap = float(gap)
    if not isfinite(gap) or gap < 0:
        raise ValueError("marker gap must be a finite, non-negative number")
    size = float(gap * 0.65 if size is None else size)
    if not isfinite(size) or size <= 0:
        raise ValueError("marker size must be a finite, positive number")

    center_x = 0.5 * (x1 + x2)
    tip_y = y1 - gap
    base_y = tip_y - size
    return (
        (center_x, tip_y),
        (center_x - size, base_y),
        (center_x + size, base_y),
    )


class DesktopOverlay:
    """Always-on-top, click-through window that mirrors last-hit markers."""

    def __init__(
        self,
        monitor,
        primary_width: float | None = None,
        marker_gap: float = DEFAULT_MARKER_GAP,
        key_color: str = OVERLAY_KEY_COLOR,
        topmost: bool = True,
        ignore_input: bool = True,
        max_fps: float = 60.0,
        show_labels: bool = True,
        label_font: tuple = ("Segoe UI", 9, "bold"),
        label_offset: float = 6.0,
        label_mode: str = "hp",
        damage: float | None = None,
    ) -> None:
        if label_mode not in LABEL_MODES:
            raise ValueError(f"label mode must be one of {', '.join(LABEL_MODES)}")
        try:
            left = int(monitor["left"])
            top = int(monitor["top"])
            width = int(monitor["width"])
            height = int(monitor["height"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("monitor must provide integer left/top/width/height") from exc
        if width <= 0 or height <= 0:
            raise ValueError("monitor width and height must be positive")
        if marker_gap < 0 or not isfinite(float(marker_gap)):
            raise ValueError("marker gap must be a finite, non-negative number")

        tkinter = _import_tkinter()
        if tkinter is None:
            raise OverlayUnavailable(
                "tkinter is not available in this Python installation; "
                "install the Tk support (python.org builds include it) or use --no-overlay"
            )

        self.monitor_left = left
        self.monitor_top = top
        self.monitor_width = width
        self.monitor_height = height
        self.marker_gap = float(marker_gap)
        self.show_labels = bool(show_labels)
        self.label_font = label_font
        self.label_offset = float(label_offset)
        self.label_mode = label_mode
        # Shown as "hp/damage" so the player sees the remaining margin at a glance.
        self.damage = None if damage is None else float(damage)
        self._max_interval = 1.0 / max_fps if max_fps and max_fps > 0 else 0.0
        self._last_render = 0.0
        self._closed = False

        try:
            root = tkinter.Tk(className="LastHitOverlay")
        except Exception as exc:  # TclError in headless sessions, missing display, ...
            raise OverlayUnavailable(f"could not create the overlay window ({exc})") from exc

        try:
            root.overrideredirect(True)
            if topmost:
                root.attributes("-topmost", True)
            root.attributes("-transparentcolor", key_color)
            root.configure(bg=key_color)
            if ignore_input:
                # Windows-only: a disabled window never activates, so clicking a
                # marker can never steal focus away from the game.
                try:
                    root.attributes("-disabled", True)
                except tkinter.TclError:
                    pass
            canvas = tkinter.Canvas(
                root, bg=key_color, highlightthickness=0, borderwidth=0, takefocus=0
            )
            canvas.pack(fill="both", expand=True)

            self._display_scale = display_scale(root.winfo_screenwidth(), primary_width)
            geometry = "{w}x{h}+{x}+{y}".format(
                w=max(1, round(width * self._display_scale)),
                h=max(1, round(height * self._display_scale)),
                x=round(left * self._display_scale),
                y=round(top * self._display_scale),
            )
            root.geometry(geometry)
        except tkinter.TclError as exc:
            try:
                root.destroy()
            except Exception:
                pass
            raise OverlayUnavailable(
                f"colour-key transparency is unsupported on this platform ({exc}); "
                "the preview window (--visualize True) still shows the markers"
            ) from exc
        except Exception:
            try:
                root.destroy()
            except Exception:
                pass
            raise

        self._tkinter = tkinter
        self._root = root
        self._canvas = canvas
        # Pump the event loop once so the window is mapped before the first frame.
        try:
            root.update_idletasks()
        except Exception:
            pass

    @property
    def closed(self) -> bool:
        return self._closed

    def render(self, candidates, frame_shape) -> None:
        """Redraw markers for ``candidates`` expressed in ``frame_shape`` coordinates."""
        if self._closed:
            return
        if not isinstance(frame_shape, (tuple, list)) or len(frame_shape) < 2:
            raise ValueError("frame shape must provide height and width")
        height, width = int(frame_shape[0]), int(frame_shape[1])
        if width <= 0 or height <= 0:
            raise ValueError("frame width and height must be positive")

        now = monotonic()
        if self._max_interval and (now - self._last_render) < self._max_interval:
            return
        self._last_render = now

        # Frame pixels -> physical desktop pixels -> Tk logical units.
        scale_x = self.monitor_width * self._display_scale / width
        scale_y = self.monitor_height * self._display_scale / height

        try:
            self._canvas.delete("all")
            for priority_rank, candidate in enumerate(candidates, start=1):
                minion = getattr(candidate, "minion", None)
                bbox = getattr(minion, "bbox", None)
                if bbox is None or len(bbox) != 4:
                    continue
                try:
                    points = marker_polygon(bbox, gap=self.marker_gap)
                except (TypeError, ValueError):
                    continue
                points = tuple((x * scale_x, y * scale_y) for x, y in points)
                if not all(np.isfinite([coord for point in points for coord in point])):
                    continue
                held = bool(getattr(candidate, "held", False))
                if priority_rank == 1:
                    color = HELD_PRIMARY_COLOR if held else PRIMARY_TARGET_COLOR
                else:
                    color = HELD_SECONDARY_COLOR if held else SECONDARY_TARGET_COLOR
                flat = [coord for point in points for coord in point]
                self._canvas.create_polygon(
                    *flat,
                    fill=color,
                    outline="#101010",
                    width=1,
                    # Held markers are dashed so an expired reading is obvious.
                    dash=(2, 2) if held else (),
                )
                if self.show_labels:
                    label = self._label_for(candidate, priority_rank, held)
                    if label:
                        anchor_x = 0.5 * (points[1][0] + points[2][0])
                        anchor_y = points[1][1] - self.label_offset * self._display_scale
                        self._canvas.create_text(
                            anchor_x,
                            anchor_y,
                            text=label,
                            fill=LABEL_COLOR,
                            font=self.label_font,
                            anchor="s",
                            tags=("label",),
                        )
            self._root.update()
        except Exception:
            # A destroyed/invalid overlay must never take the capture loop down.
            self.close()

    def _label_for(self, candidate, priority_rank: int, held: bool) -> str:
        """Short text above a marker: estimated HP, with '?' while the reading is held."""
        try:
            health = float(getattr(candidate, "health_upper_bound", 0.0))
        except (TypeError, ValueError):
            return ""
        if not isfinite(health) or health <= 0:
            return ""
        suffix = "?" if held else ""
        mode = getattr(self, "label_mode", "hp")
        if mode == "percent":
            try:
                percent = float(getattr(candidate, "health_percent", 0.0))
            except (TypeError, ValueError):
                percent = 0.0
            return f"{percent:.0f}%{suffix}"
        if mode == "hp_damage" and self.damage is not None:
            return f"{health:.0f}/{self.damage:.0f}{suffix}"
        return f"{health:.0f} HP{suffix}"

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        root = getattr(self, "_root", None)
        if root is None:
            return
        try:
            root.destroy()
        except Exception:
            pass
