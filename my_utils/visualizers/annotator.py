"""Pure-OpenCV visualizer that displays only prioritized last-hittable minions."""

import cv2
import numpy as np

from my_utils.data_structures.minion import Minion
from my_utils.modules.last_hitter import LastHitCandidate


PRIMARY_TARGET_COLOR = (0, 255, 255)   # Yellow (BGR) for #1 priority last-hit target
SECONDARY_TARGET_COLOR = (0, 200, 0)   # Green (BGR) for #2+ last-hittable minions
TEXT_COLOR = (0, 0, 0)


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
    ) -> np.ndarray:
        return self.annotate_candidates(img, candidates)

    def annotate_candidates(
        self,
        img: np.ndarray,
        candidates: list[LastHitCandidate],
    ) -> np.ndarray:
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("frame must be a 3-channel BGR image")

        canvas = img.copy()
        height, width = canvas.shape[:2]

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

            color = PRIMARY_TARGET_COLOR if priority_rank == 1 else SECONDARY_TARGET_COLOR
            thickness = self.line_thickness + (1 if priority_rank == 1 else 0)
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

            label = (
                f"#{priority_rank} {minion.minion_type} "
                f"<= {candidate.health_upper_bound:.0f} HP ({100 * minion.health_level:.0f}%)"
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

        return canvas
