"""Desktop screenshot capture; no game process or input access."""

from dataclasses import dataclass
from math import ceil, floor
from time import monotonic

import cv2
import mss
import numpy as np


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    captured_at: float
    excluded_regions: tuple[tuple[int, int, int, int], ...] = ()


def resize_image(image, width=None, height=None):
    original_height, original_width = image.shape[:2]
    if width is None and height is None:
        return image
    if width is None:
        width = max(1, round(original_width * height / original_height))
    if height is None:
        height = max(1, round(original_height * width / original_width))
    if width <= 0 or height <= 0:
        raise ValueError("resize dimensions must be positive")
    if (width, height) == (original_width, original_height):
        return image
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def mask_window(image, monitor, window_rect, margin=24):
    """Mask a desktop-coordinate (x, y, width, height) preview rectangle.

    The rectangle and monitor may have negative coordinates on multi-monitor
    desktops.  Returning the affected region also allows detection boxes that
    straddle the mask to be rejected instead of interpreting a partial bar.
    """
    if window_rect is None:
        return image, ()
    x, y, window_width, window_height = window_rect
    if window_width <= 0 or window_height <= 0:
        raise ValueError("cannot determine the preview window's position")
    height, width = image.shape[:2]
    x1 = max(0, floor(x - monitor["left"] - margin))
    y1 = max(0, floor(y - monitor["top"] - margin))
    x2 = min(width, ceil(x + window_width - monitor["left"] + margin))
    y2 = min(height, ceil(y + window_height - monitor["top"] + margin))
    if x1 >= x2 or y1 >= y2:
        return image, ()
    masked = image.copy()
    masked[y1:y2, x1:x2] = 0
    return masked, ((x1, y1, x2, y2),)


def resize_excluded_regions(regions, old_shape, new_shape):
    if not regions or old_shape[:2] == new_shape[:2]:
        return tuple(regions)
    old_height, old_width = old_shape[:2]
    new_height, new_width = new_shape[:2]
    scale_x, scale_y = new_width / old_width, new_height / old_height
    return tuple(
        (
            max(0, floor(x1 * scale_x) - 2),
            max(0, floor(y1 * scale_y) - 2),
            min(new_width, ceil(x2 * scale_x) + 2),
            min(new_height, ceil(y2 * scale_y) + 2),
        )
        for x1, y1, x2, y2 in regions
    )


class FrameLoader:
    def __init__(self, monitor_idx=1, width=None, height=None):
        self.sct = mss.mss()
        monitors = self.sct.monitors
        if not isinstance(monitor_idx, int) or not 0 <= monitor_idx < len(monitors):
            self.sct.close()
            raise ValueError(f"monitor index {monitor_idx} is unavailable; use --list-monitors")
        self.monitor = dict(monitors[monitor_idx])
        # monitors[0] is the union of every display; monitors[1] is the primary one.
        self.primary_monitor = dict(monitors[1] if len(monitors) > 1 else monitors[0])
        self.width, self.height = width, height

    @staticmethod
    def list_monitors():
        with mss.mss() as sct:
            return [dict(monitor) for monitor in sct.monitors]

    def grab(self, window_rect=None):
        captured_at = monotonic()
        raw = np.asarray(self.sct.grab(self.monitor))
        if window_rect is None:
            resized = resize_image(raw, self.width, self.height)
            image = cv2.cvtColor(resized, cv2.COLOR_BGRA2BGR)
            return CapturedFrame(image, captured_at, ())

        image = cv2.cvtColor(raw, cv2.COLOR_BGRA2BGR)
        image, regions = mask_window(image, self.monitor, window_rect)
        old_shape = image.shape
        image = resize_image(image, self.width, self.height)
        regions = resize_excluded_regions(regions, old_shape, image.shape)
        # Resampling at the boundary must not bring the preview back into view.
        for x1, y1, x2, y2 in regions:
            image[y1:y2, x1:x2] = 0
        return CapturedFrame(image, captured_at, regions)

    def close(self):
        self.sct.close()

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()
