"""Conservative association of detected minions and their health bars."""

from dataclasses import dataclass
from math import ceil, floor, isfinite

import numpy as np
from scipy.optimize import linear_sum_assignment

from my_utils.data_structures.labels import normalize_label
from my_utils.data_structures.minion import Minion
from my_utils.modules.utils.health_bar_utils import HealthBarUtils


@dataclass(frozen=True)
class _Detection:
    bbox: tuple[float, float, float, float]
    label: str
    confidence: float


def _intersects(a, b):
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _box_iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return intersection / (area_a + area_b - intersection) if intersection else 0.0


def _deduplicate(detections, threshold=0.8):
    """Collapse duplicate boxes; discard a cluster only if its top classes are tied."""
    parents = list(range(len(detections)))

    def root(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    for i in range(len(detections)):
        for j in range(i):
            if _box_iou(detections[i].bbox, detections[j].bbox) >= threshold:
                parents[root(i)] = root(j)
    groups = {}
    for i, detection in enumerate(detections):
        groups.setdefault(root(i), []).append(detection)
    result = []
    for group in groups.values():
        ranked = sorted(group, key=lambda d: (-d.confidence, d.bbox))
        top = ranked[0]
        conflicting = [
            d for d in ranked[1:]
            if d.label != top.label and (top.confidence - d.confidence) < 0.05
        ]
        if conflicting:
            continue
        result.append(top)
    return sorted(result, key=lambda d: (d.bbox, d.label))


class ObjectAssociator:
    def __init__(
        self,
        intersection_threshold=0.5,
        min_confidence=0.25,
        ambiguity_margin=0.05,
        health_bar_utils=None,
    ):
        for name, value in (
            ("intersection threshold", intersection_threshold),
            ("minimum confidence", min_confidence),
            ("ambiguity margin", ambiguity_margin),
        ):
            if not isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if intersection_threshold == 0 or min_confidence == 0:
            raise ValueError("intersection threshold and minimum confidence must be positive")
        self.intersection_threshold = intersection_threshold
        self.min_confidence = min_confidence
        self.ambiguity_margin = ambiguity_margin
        self.health_bar_utils = health_bar_utils or HealthBarUtils()

    def __call__(self, detection_result, img, minion_type="red", excluded_regions=()):
        if minion_type not in {"red", "blue"}:
            raise ValueError("minion color must be red or blue")
        return self.match_bars_to_minions(
            detection_result[f"{minion_type}_minions"],
            detection_result[f"{minion_type}_health_bars"],
            img,
            minion_color=minion_type,
            excluded_regions=excluded_regions,
        )

    def _prepare(self, entries, img_shape, minion_color, is_bar, excluded_regions):
        height, width = img_shape[:2]
        # Low-HP health bars have very few colored pixels, so YOLOv9 reports lower confidence on them.
        threshold = min(self.min_confidence, 0.20) if is_bar else self.min_confidence
        detections = []
        for entry in entries:
            try:
                if is_bar:
                    box, confidence = entry
                    label = f"{minion_color} health bar"
                else:
                    box, label, confidence = entry
                    label = normalize_label(label)
                    if label not in {
                        f"{minion_color} {kind} minion" for kind in ("caster", "melee", "siege")
                    }:
                        continue
                box = tuple(float(value) for value in box)
                confidence = float(confidence)
                if len(box) != 4 or not all(isfinite(v) for v in box) or not isfinite(confidence):
                    continue
                x1, y1, x2, y2 = box
                # Boxes touching an edge may be clipped; partial bars are unsafe.
                if not (0 < x1 < x2 < width and 0 < y1 < y2 < height):
                    continue
                if not threshold <= confidence <= 1:
                    continue
                if any(_intersects(box, region) for region in excluded_regions):
                    continue
            except (TypeError, ValueError):
                continue
            detections.append(_Detection(box, label, confidence))
        return _deduplicate(detections)

    def match_bars_to_minions(self, minions, health_bars, img, minion_color="red", excluded_regions=()):
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("frame must be a BGR image")
        if minion_color not in {"red", "blue"}:
            raise ValueError("minion color must be red or blue")
        bodies = self._prepare(minions, img.shape, minion_color, False, excluded_regions)
        bars = self._prepare(health_bars, img.shape, minion_color, True, excluded_regions)
        if not bodies or not bars:
            return []

        bar_boxes = np.asarray([bar.bbox for bar in bars], dtype=float)
        body_boxes = np.asarray([body.bbox for body in bodies], dtype=float)
        ratios = self._get_intersected_area(bar_boxes, body_boxes)
        valid = ratios >= self.intersection_threshold

        # Prefer health bars aligned near the top of their minion's bounding box.
        bar_cy = 0.5 * (bar_boxes[:, 1] + bar_boxes[:, 3])[:, None]
        body_top = body_boxes[:, 1][None, :]
        body_h = np.maximum(1.0, body_boxes[:, 3] - body_boxes[:, 1])[None, :]
        rel_y = np.minimum(1.0, np.abs(bar_cy - body_top) / body_h)
        effective_scores = np.where(valid, ratios - 0.25 * rel_y, 0.0)

        # Prefer the largest number of valid matches, then the best top-aligned overlap.
        bonus = min(len(bars), len(bodies)) + 1
        assignment_scores = np.where(valid, bonus + effective_scores, 0.0)
        rows, columns = linear_sum_assignment(assignment_scores, maximize=True)

        matched_pairs = [
            (bar_idx, body_idx)
            for bar_idx, body_idx in zip(rows, columns)
            if valid[bar_idx, body_idx]
        ]
        assigned_bars = {bar_idx for bar_idx, _ in matched_pairs}
        assigned_bodies = {body_idx for _, body_idx in matched_pairs}
        unassigned_bars = [i for i in range(len(bars)) if i not in assigned_bars]
        unassigned_bodies = [j for j in range(len(bodies)) if j not in assigned_bodies]

        matched = []
        for bar_idx, body_idx in matched_pairs:
            score = effective_scores[bar_idx, body_idx]
            competing_bodies = [
                effective_scores[bar_idx, j]
                for j in unassigned_bodies
                if valid[bar_idx, j]
            ]
            competing_bars = [
                effective_scores[i, body_idx]
                for i in unassigned_bars
                if valid[i, body_idx]
            ]
            if any(
                alt >= score - self.ambiguity_margin
                for alt in (competing_bodies + competing_bars)
            ):
                continue

            bar, body = bars[bar_idx], bodies[body_idx]
            x1, y1, x2, y2 = bar.bbox
            crop = img[floor(y1) : ceil(y2), floor(x1) : ceil(x2)]
            health = self.health_bar_utils.get_health_level(crop, bar_color=minion_color)
            if health is None:
                continue
            try:
                health = float(health)
            except (TypeError, ValueError):
                continue
            if not isfinite(health) or not 0 < health <= 1:
                continue
            matched.append(Minion(body.label, health, body.bbox, body.confidence))
        return matched

    @staticmethod
    def _get_intersected_area(health_bar_boxes, minion_boxes):
        """Fraction of each health bar covered by each minion detection (including slight top float)."""
        body_heights = np.maximum(0.0, minion_boxes[:, 3] - minion_boxes[:, 1])
        expanded_y1 = minion_boxes[:, 1] - 0.15 * body_heights

        x1 = np.maximum(health_bar_boxes[:, 0, None], minion_boxes[:, 0])
        y1 = np.maximum(health_bar_boxes[:, 1, None], expanded_y1)
        x2 = np.minimum(health_bar_boxes[:, 2, None], minion_boxes[:, 2])
        y2 = np.minimum(health_bar_boxes[:, 3, None], minion_boxes[:, 3])
        intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)
        areas = np.maximum(health_bar_boxes[:, 2] - health_bar_boxes[:, 0], 0) * np.maximum(
            health_bar_boxes[:, 3] - health_bar_boxes[:, 1], 0
        )
        return np.divide(
            intersection,
            areas[:, None],
            out=np.zeros_like(intersection, dtype=float),
            where=areas[:, None] > 0,
        )
