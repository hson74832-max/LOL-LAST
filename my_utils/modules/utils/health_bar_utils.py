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
        matching_threshold=0.70,
        bar_padding=0.0,
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

    def _match_single_scale(
        self, bar_img: np.ndarray, template: np.ndarray, bar_color: str
    ) -> tuple[float, float] | None:
        height, width = bar_img.shape[:2]
        th, tw = template.shape[:2]
        if height < th or width < tw or width == 0:
            return None

        try:
            matches = cv2.matchTemplate(bar_img, template, cv2.TM_CCOEFF_NORMED)
        except cv2.error:
            return None

        finite_mask = np.isfinite(matches) & (matches >= self.matching_threshold)
        if not np.any(finite_mask):
            return None

        mid = tw // 2
        ys, xs = np.where(finite_mask)
        valid_candidates: list[tuple[float, int]] = []
        for y, x in zip(ys, xs):
            patch = bar_img[y : y + th, x : x + tw].astype(np.float32)
            left = patch[:, :mid].mean(axis=(0, 1))
            right = patch[:, mid:].mean(axis=(0, 1))
            # Reject flat patches or dark empty-bar-to-border steps.
            if float(left.mean() - right.mean()) < 10 or float(left.max()) < 65:
                continue
            if bar_color == "red" and float(left[2] - max(left[0], left[1])) < 10:
                continue
            if bar_color == "blue" and float(left[0] - left[2]) < -15:
                continue
            valid_candidates.append((float(matches[y, x]), int(x) + mid))

        if not valid_candidates:
            return None

        best_score = max(score for score, _ in valid_candidates)
        near_peak_edges = [
            edge_x for score, edge_x in valid_candidates if score >= best_score - 0.05
        ]
        minimum_bar_width = width * (1.0 - self.bar_padding)
        level = min(1.0, max(1, max(near_peak_edges)) / minimum_bar_width)
        return best_score, level

    def _measure_by_color_profile(self, bar_img: np.ndarray, bar_color: str) -> float | None:
        """Fallback color-profile estimator when resolution/scaling alters the 4x4 template."""
        height, width = bar_img.shape[:2]
        if height < 3 or width < 8:
            return None

        core = bar_img[1:-1, :, :] if height >= 5 else bar_img
        col_bgr = core.astype(np.float32).mean(axis=0)  # (width, 3) in BGR
        b, g, r = col_bgr[:, 0], col_bgr[:, 1], col_bgr[:, 2]
        brightness = col_bgr.mean(axis=1)

        if bar_color == "red":
            filled = (r >= 85.0) & ((r - g) >= 18.0) & ((r - b) >= 12.0)
        else:
            filled = (b >= 75.0) & (g >= 65.0) & ((b - r) >= -5.0) & (brightness >= 70.0)

        empty = brightness < 55.0
        if not np.any(filled) or np.mean(empty) < 0.15:
            return None

        filled_indices = np.where(filled)[0]
        # The filled portion must start near the left side of the health bar.
        if int(filled_indices[0]) > max(4, width // 4):
            return None

        rightmost_filled = int(filled_indices[-1]) + 1
        # There must be dark empty bar columns to the right of the transition.
        if rightmost_filled >= width or not np.any(empty[rightmost_filled:]):
            return None

        minimum_bar_width = width * (1.0 - self.bar_padding)
        return min(1.0, rightmost_filled / minimum_bar_width)

    def get_health_level(self, health_bar_img: np.ndarray, bar_color: str = "blue") -> float | None:
        """Return an upper-bound fraction, or None when the reading is unreliable."""
        if bar_color not in self.templates:
            raise ValueError(f"unsupported health-bar color: {bar_color}")
        if (
            not isinstance(health_bar_img, np.ndarray)
            or health_bar_img.dtype != np.uint8
            or health_bar_img.ndim != 3
            or health_bar_img.shape[2] != 3
        ):
            return None

        template = self.templates[bar_color]
        height, width = health_bar_img.shape[:2]
        th, tw = template.shape[:2]
        if height < th or width < tw or width == 0:
            return None

        candidates: list[tuple[float, float]] = []
        native_match = self._match_single_scale(health_bar_img, template, bar_color)
        if native_match is not None:
            candidates.append(native_match)

        for target_h in (4, 6, 8):
            if height > target_h and abs(height - target_h) >= 2:
                scale = target_h / float(height)
                target_w = max(tw, int(round(width * scale)))
                resized = cv2.resize(
                    health_bar_img, (target_w, target_h), interpolation=cv2.INTER_AREA
                )
                scaled_match = self._match_single_scale(resized, template, bar_color)
                if scaled_match is not None:
                    candidates.append(scaled_match)

        if candidates:
            _, best_level = max(candidates, key=lambda item: item[0])
            return best_level

        return self._measure_by_color_profile(health_bar_img, bar_color)
