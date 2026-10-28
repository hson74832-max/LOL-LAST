"""Read an approximate upper bound on health from a captured health bar."""

from math import isfinite
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[3]


class HealthBarUtils:
    def __init__(
        self,
        blue_bar_template_path=None,
        red_bar_template_path=None,
        matching_threshold=0.9,
        bar_padding=0.15,
    ):
        self.matching_threshold = float(matching_threshold)
        self.bar_padding = float(bar_padding)
        if not isfinite(self.matching_threshold) or not 0 < self.matching_threshold <= 1:
            raise ValueError("health-bar matching threshold must be greater than 0 and at most 1")
        if not isfinite(self.bar_padding) or not 0 <= self.bar_padding < 1:
            raise ValueError("bar padding must be at least 0 and less than 1")

        paths = {
            "blue": Path(blue_bar_template_path or PROJECT_ROOT / "templates/health_bar_level_template.jpg"),
            "red": Path(red_bar_template_path or PROJECT_ROOT / "templates/red_health_bar_level_template.jpg"),
        }
        self.templates = {}
        for color, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"{color} health-bar template not found: {path}")
            template = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if template is None or min(template.shape[:2]) < 2:
                raise ValueError(f"invalid {color} health-bar template: {path}")
            self.templates[color] = template

    def _match_single_scale(self, bar_img: np.ndarray, template: np.ndarray, bar_color: str) -> float | None:
        height, width = bar_img.shape[:2]
        th, tw = template.shape[:2]
        if height < th or width < tw or width == 0:
            return None

        try:
            matches = cv2.matchTemplate(bar_img, template, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return None
        ys, xs = np.where(np.isfinite(matches) & (matches >= self.matching_threshold))
        edge_positions = []
        mid = tw // 2
        for y, x in zip(ys, xs):
            patch = bar_img[y:y + th, x:x + tw].astype(np.float32)
            left = patch[:, :mid].mean(axis=(0, 1))
            right = patch[:, mid:].mean(axis=(0, 1))
            # Reject flat/texture-only matches, and reject non-red edges for red bars.
            if float(left.mean() - right.mean()) < 10:
                continue
            if bar_color == "red" and float(left[2] - max(left[0], left[1])) < 10:
                continue
            edge_positions.append(int(x) + mid + 1)  # one extra pixel of uncertainty

        if not edge_positions:
            return None
        minimum_bar_width = width * (1.0 - self.bar_padding)
        return min(1.0, max(edge_positions) / minimum_bar_width)

    def get_health_level(self, health_bar_img: np.ndarray, bar_color: str = "blue") -> float | None:
        """Return an upper-bound fraction, or None when the reading is unreliable.

        The 4-pixel templates straddle the end of the colored portion of the bar.
        All credible matches are considered and the rightmost is used to avoid
        treating an earlier tick/edge as a lower health value.  Padding and a
        pixel of edge uncertainty are included in the bound.
        """
        if bar_color not in self.templates:
            raise ValueError(f"unsupported health-bar color: {bar_color}")
        if (not isinstance(health_bar_img, np.ndarray) or health_bar_img.dtype != np.uint8
                or health_bar_img.ndim != 3 or health_bar_img.shape[2] != 3):
            return None

        template = self.templates[bar_color]
        level = self._match_single_scale(health_bar_img, template, bar_color)
        if level is not None:
            return level

        height, width = health_bar_img.shape[:2]
        th, tw = template.shape[:2]
        target_h = max(th + 2, 6)
        if height > target_h * 1.3:
            scale = target_h / float(height)
            target_w = max(tw, int(round(width * scale)))
            resized = cv2.resize(health_bar_img, (target_w, target_h), interpolation=cv2.INTER_AREA)
            return self._match_single_scale(resized, template, bar_color)

        return None
