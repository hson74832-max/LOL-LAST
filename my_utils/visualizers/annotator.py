"""Pure-OpenCV visualizer that displays only prioritized last-hittable minions."""

import cv2
import numpy as np

from my_utils.data_structures.minion import Minion
from my_utils.modules.last_hitter import LastHitCandidate


PRIMARY_TARGET_COLOR = (0, 255, 255)   # Yellow (BGR) for #1 priority last-hit target
SECONDARY_TARGET_COLOR = (0, 200, 0)   # Green (BGR) for #2+ last-hittable minions
HELD_PRIMARY_COLOR = (0, 184, 184)     # Dimmed yellow: marker held, minion not detected
HELD_SECONDARY_COLOR = (0, 150, 0)     # Dimmed green
TEXT_COLOR = (0, 0, 0)
HP_LABEL_COLOR = (255, 255, 255)        # white numbers for non-target minions
HP_LABEL_HELD_COLOR = (185, 185, 185)   # grey while the reading is only held


class AIAnnotator:
    """Render only minions that can currently be last-hit, ordered by priority."""

    def __init__(self, line_thickness: int = 2) -> None:
        if line_thickness <= 0:
            raise ValueError("line thickness must be positive")
        self.line_thickness = int(line_thickness)

    def __call__(
        self,
        candidates: list[LastHitCandidate],
        img: np.ndarray,
        copy: bool = True,
        labels=None,
    ) -> np.ndarray:
        return self.annotate_candidates(img, candidates, copy=copy, labels=labels)

    def annotate_candidates(
        self,
        img: np.ndarray,
        candidates: list[LastHitCandidate],
        copy: bool = True,
        labels=None,
    ) -> np.ndarray:
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("frame must be a 3-channel BGR image")

        canvas = img.copy() if copy else img
        labels = list(labels or [])
        if not candidates and not labels:
            return canvas
        height, width = canvas.shape[:2]
        labelled_boxes = []

        # Draw lower-priority candidates first so the #1 priority target is never occluded.
        indexed_candidates = list(enumerate(candidates, start=1))
        for priority_rank, candidate in reversed(indexed_candidates):
            if not isinstance(candidate, LastHitCandidate):
                raise TypeError("AIAnnotator only accepts verified LastHitCandidate instances")

            minion = candidate.minion
            if not isinstance(minion, Minion):
                raise TypeError("candidate.minion must be a Minion instance")

            x1, y1, x2, y2 = (
                max(0, min(width - 1, round(minion.bbox[0]))),
                max(0, min(height - 1, round(minion.bbox[1]))),
                max(0, min(width - 1, round(minion.bbox[2]))),
                max(0, min(height - 1, round(minion.bbox[3]))),
            )
            if x1 >= x2 or y1 >= y2:
                continue

            held = bool(getattr(candidate, "held", False))
            if priority_rank == 1:
                color = HELD_PRIMARY_COLOR if held else PRIMARY_TARGET_COLOR
            else:
                color = HELD_SECONDARY_COLOR if held else SECONDARY_TARGET_COLOR
            thickness = self.line_thickness + (1 if priority_rank == 1 and not held else 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            label = (
                f"#{priority_rank} {minion.minion_type} "
                f"<= {candidate.health_upper_bound:.0f} HP ({100 * minion.health_level:.0f}%)"
                + (" [held]" if held else "")
            )
            (text_w, text_h), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            box_h = text_h + baseline + 6
            box_w = min(width, text_w + 8)
            label_x1 = max(0, min(x1, width - box_w))
            label_y2 = y1 if y1 >= box_h else min(height, y2 + box_h)
            label_y1 = max(0, label_y2 - box_h)

            cv2.rectangle(
                canvas,
                (label_x1, label_y1),
                (min(width, label_x1 + box_w), label_y2),
                color,
                -1,
            )
            cv2.putText(
                canvas,
                label,
                (label_x1 + 4, label_y2 - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                TEXT_COLOR,
                1,
                cv2.LINE_AA,
            )
            labelled_boxes.append((x1, y1, x2, y2))

        for hp_label in labels:
            minion = getattr(hp_label, "minion", None)
            bbox = getattr(minion, "bbox", None)
            if bbox is None or len(bbox) != 4:
                continue
            box = (
                max(0, min(width - 1, round(float(bbox[0])))),
                max(0, min(height - 1, round(float(bbox[1])))),
                max(0, min(width - 1, round(float(bbox[2])))),
                max(0, min(height - 1, round(float(bbox[3])))),
            )
            if box[0] >= box[2] or box[1] >= box[3]:
                continue
            if any(_boxes_overlap(box, other) for other in labelled_boxes):
                continue  # already covered by its own marker label
            text = getattr(hp_label, "text", None) or _default_hp_text(hp_label)
            if not text:
                continue
            held = bool(getattr(hp_label, "held", False))
            color = HP_LABEL_HELD_COLOR if held else HP_LABEL_COLOR
            _put_label_above(canvas, box, text, color, (0, 0, 0))

        return canvas


def _boxes_overlap(box_a, box_b, tolerance_scale: float = 0.5) -> bool:
    """True when two boxes describe the same minion (centres nearly coincide)."""
    ax, ay = 0.5 * (box_a[0] + box_a[2]), 0.5 * (box_a[1] + box_a[3])
    bx, by = 0.5 * (box_b[0] + box_b[2]), 0.5 * (box_b[1] + box_b[3])
    size = min(box_a[2] - box_a[0], box_a[3] - box_a[1], box_b[2] - box_b[0], box_b[3] - box_b[1])
    limit = max(4.0, tolerance_scale * size)
    return (ax - bx) ** 2 + (ay - by) ** 2 <= limit * limit


def _default_hp_text(hp_label) -> str:
    """Short number for a label without a pre-rendered ``text`` field."""
    try:
        health = float(getattr(hp_label, "health_upper_bound", 0.0))
    except (TypeError, ValueError):
        return ""
    if not health > 0:
        return ""
    suffix = "?" if bool(getattr(hp_label, "held", False)) else ""
    return f"{health:.0f} HP{suffix}"


def _put_label_above(canvas, box, text, color, outline) -> None:
    """Draw ``text`` centred just above ``box``, with a dark outline for contrast."""
    height, width = canvas.shape[:2]
    (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
    origin_x = int(max(0, min(width - text_w - 2, 0.5 * (box[0] + box[2]) - text_w / 2)))
    origin_y = int(max(text_h + 2, box[1] - 6))
    for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        cv2.putText(
            canvas,
            text,
            (origin_x + dx, origin_y + dy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            outline,
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        text,
        (origin_x, origin_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        color,
        1,
        cv2.LINE_AA,
    )
