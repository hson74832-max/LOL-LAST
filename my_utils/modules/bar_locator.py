"""Locate a minion's health bar from the pixels around its detected body.

YOLOv9's dedicated ``*_health_bar`` class is the weak link in practice: in a busy
wave it frequently reports no bars (or one or two) for eight or nine detected
bodies, so nothing can be associated and no marker is ever drawn.

League's minion bars have a very stable appearance: a horizontal strip a few pixels
tall, a saturated fill on the left and a dark empty part on the right, sitting just
above the minion's head.  This module finds that strip inside a small window around
each detected body box and measures the remaining fraction the same way
``HealthBarUtils`` does (rightmost filled column => conservative upper bound).

Everything here is read-only and purely geometric: no game process access.
"""

from dataclasses import dataclass
from math import isfinite

import numpy as np

# Same colour rules as HealthBarUtils._measure_by_color_profile, so a bar that the
# template matcher accepts is also accepted here (and vice versa).
MIN_FILLED_WIDTH = 2
MIN_EMPTY_WIDTH = 2
DARK_BRIGHTNESS = 55.0
BAR_EDGE_RISE = 12.0        # brightness rise that marks the end of the bar
MAX_BAR_HEIGHT = 12         # a bar is never taller than this (rows)
# Geometry relative to the minion body box.
SEARCH_ABOVE = 1.1          # search this many body heights above the box top
SEARCH_BELOW = 0.55         # ... and this many inside the box
SEARCH_SIDE = 0.12          # expand the box sideways by this fraction
MIN_SPAN_RATIO = 0.30       # bar width / body width (lower bound)
MAX_SPAN_RATIO = 1.60       # bar width / body width (upper bound)
MAX_LEVEL_SPREAD = 0.45     # rows of one bar must agree on the fill level


@dataclass(frozen=True)
class BarReading:
    """A health bar located around a minion body."""

    box: tuple[int, int, int, int]
    level: float          # remaining health fraction, upper bound
    score: float          # bar width in pixels * sqrt(bar height)


def _filled_mask_window(window: np.ndarray, bar_color: str) -> np.ndarray:
    """Per-pixel 'this looks like the fill colour of the bar' mask, (H, W, 3) BGR float."""
    blue, green, red = window[:, :, 0], window[:, :, 1], window[:, :, 2]
    brightness = window.mean(axis=2)
    if bar_color == "red":
        return (red >= 85.0) & ((red - green) >= 18.0) & ((red - blue) >= 12.0)
    if bar_color == "blue":
        return (blue >= 75.0) & (green >= 65.0) & ((blue - red) >= -5.0) & (brightness >= 70.0)
    raise ValueError("bar_color must be 'red' or 'blue'")


def _row_reading(filled: np.ndarray, brightness: np.ndarray):
    """Return (start_x, span, level) for one image row, or None if it is not a bar."""
    if not np.any(filled):
        return None
    empty = brightness < DARK_BRIGHTNESS
    width = len(brightness)

    filled_indices = np.flatnonzero(filled)
    start = int(filled_indices[0])
    # The fill always starts at the left end of the bar.
    if start > max(4, width // 3):
        return None

    end = start
    while end < width and filled[end]:
        end += 1
    filled_width = end - start
    if filled_width < MIN_FILLED_WIDTH:
        return None

    # Walk the depleted part of the bar.  It is dark, but so is a lot of terrain, so the
    # run ends as soon as the colour rises clearly above the bar interior (the bar has a
    # darker, flatter interior than the ground behind it).  Without this test a dark
    # background would make the bar look arbitrarily wide and the HP reading far too low.
    empty_end = end
    interior_min = brightness[end] if end < width else 0.0
    while empty_end < width and empty[empty_end]:
        value = brightness[empty_end]
        interior_min = min(interior_min, value)
        if value > interior_min + BAR_EDGE_RISE and empty_end + 1 < width:
            if brightness[empty_end + 1] > interior_min + BAR_EDGE_RISE:
                break
        empty_end += 1
    empty_width = empty_end - end
    if empty_width < MIN_EMPTY_WIDTH:
        return None

    span = filled_width + empty_width
    # The bar's right border column is dark and therefore counted in the empty run;
    # credit it back so the reading is an upper bound rather than an under-estimate.
    return start, span, min(1.0, (filled_width + 1.0) / float(span))


def _group_rows(rows):
    """Merge consecutive single-row readings that describe the same bar."""
    groups = []
    for y, (start, span, level) in rows:
        if groups:
            group = groups[-1]
            last_y, last_start, last_span = group["y"], group["start"], group["span"]
            overlap = min(start + span, last_start + last_span) - max(start, last_start)
            if (
                y - last_y <= 2
                and overlap >= 0.6 * min(span, last_span)
                and abs(start - last_start) <= max(3, 0.25 * last_span)
            ):
                group["y"] = y
                group["levels"].append(level)
                group["height"] += 1
                group["start"] = min(start, last_start)
                group["span"] = max(span, last_span)
                continue
        groups.append(
            {"y": y, "start": start, "span": span, "height": 1, "levels": [level]}
        )
    return groups


class HealthBarLocator:
    """Find and measure the health bar belonging to one detected minion body."""

    def __init__(
        self,
        min_span_ratio: float = MIN_SPAN_RATIO,
        max_span_ratio: float = MAX_SPAN_RATIO,
        search_above: float = SEARCH_ABOVE,
        search_below: float = SEARCH_BELOW,
        max_level_spread: float = MAX_LEVEL_SPREAD,
    ) -> None:
        for name, value in (
            ("minimum span ratio", min_span_ratio),
            ("maximum span ratio", max_span_ratio),
            ("maximum level spread", max_level_spread),
        ):
            if not isfinite(value) or not 0 < value <= 4:
                raise ValueError(f"{name} must be a finite number in (0, 4]")
        if max_span_ratio <= min_span_ratio:
            raise ValueError("maximum span ratio must exceed the minimum span ratio")
        self.min_span_ratio = float(min_span_ratio)
        self.max_span_ratio = float(max_span_ratio)
        self.search_above = float(search_above)
        self.search_below = float(search_below)
        self.max_level_spread = float(max_level_spread)

    def locate(self, image: np.ndarray, bbox, bar_color: str = "red") -> BarReading | None:
        """Return the most likely health bar around ``bbox`` or None when unsure."""
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("image must be a 3-channel BGR array")
        if bar_color not in {"red", "blue"}:
            raise ValueError("bar_color must be 'red' or 'blue'")
        if bbox is None or len(bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(v) for v in bbox)
        except (TypeError, ValueError):
            return None
        if not all(isfinite(v) for v in (x1, y1, x2, y2)):
            return None
        body_width = x2 - x1
        body_height = y2 - y1
        if body_width < 4 or body_height < 4:
            return None

        image_height, image_width = image.shape[:2]
        left = max(0, int(round(x1 - self.search_side_px(body_width))))
        right = min(image_width, int(round(x2 + self.search_side_px(body_width))))
        top = max(0, int(round(y1 - self.search_above * body_height)))
        bottom = min(image_height, int(round(y1 + self.search_below * body_height)))
        if right - left < 6 or bottom - top < 2:
            return None

        window = image[top:bottom, left:right].astype(np.float32)
        brightness = window.mean(axis=2)
        filled = _filled_mask_window(window, bar_color)
        rows = []
        for offset in range(window.shape[0]):
            reading = _row_reading(filled[offset], brightness[offset])
            if reading is not None:
                rows.append((top + offset, reading))
        if not rows:
            return None

        min_span = max(10.0, self.min_span_ratio * body_width)
        max_span = max(min_span, self.max_span_ratio * body_width)

        best: BarReading | None = None
        for group in _group_rows(rows):
            height = group["height"]
            span = group["span"]
            if height > MAX_BAR_HEIGHT:
                continue
            if not (min_span <= span <= max_span):
                continue
            levels = group["levels"]
            if max(levels) - min(levels) > self.max_level_spread:
                continue
            level = max(levels)  # conservative upper bound (as HealthBarUtils does)
            if not 0 < level <= 1:
                continue
            # Prefer bars that sit where a minion bar belongs: just above the box top and
            # horizontally centred on the body.  This keeps neighbouring minions from
            # donating their bar to the body under evaluation.
            bar_bottom = group["y"] + 1
            bar_center_x = group["start"] + span / 2.0
            body_center_x = (x1 + x2) / 2.0 - left
            vertical = abs(bar_bottom - y1) / body_height
            horizontal = abs(bar_center_x - body_center_x) / body_width
            score = span * (height ** 0.5) / (1.0 + vertical + 0.5 * horizontal)
            if best is None or score > best.score:
                best = BarReading(
                    box=(
                        left + group["start"],
                        group["y"] - height + 1,
                        left + group["start"] + span,
                        group["y"] + 1,
                    ),
                    level=level,
                    score=float(score),
                )
        return best

    @staticmethod
    def search_side_px(body_width: float) -> float:
        return SEARCH_SIDE * body_width
