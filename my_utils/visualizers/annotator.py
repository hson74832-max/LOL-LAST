"""Pure-OpenCV visualizer that displays only prioritized last-hit candidates.

Markers are coloured by urgency, exactly like the desktop overlay:

* **orange** - the minion is getting low and is tracked before it can be killed,
* **yellow** - the #1 minion that can be last-hit right now,
* **green**  - the other minions that can be last-hit right now,
* dimmed + dashed-looking (lighter outline) while a marker is only held.

This visualizer draws no numbers: the HP value stays in the ``--debug`` output.
"""

import cv2
import numpy as np

from my_utils.data_structures.minion import Minion
from my_utils.modules.last_hitter import LastHitCandidate


PRIMARY_TARGET_COLOR = (0, 255, 255)   # Yellow (BGR): #1 last-hit target
SECONDARY_TARGET_COLOR = (0, 200, 0)   # Green (BGR): #2+ last-hit targets
EARLY_PRIMARY_COLOR = (0, 154, 255)    # Orange (BGR): #1 minion getting low
EARLY_SECONDARY_COLOR = (0, 114, 210)  # Darker orange: #2+ minions getting low
HELD_PRIMARY_COLOR = (0, 184, 184)     # Dimmed yellow: marker held, stale reading
HELD_SECONDARY_COLOR = (0, 150, 0)     # Dimmed green
HELD_EARLY_PRIMARY_COLOR = (0, 110, 184)    # Dimmed orange
HELD_EARLY_SECONDARY_COLOR = (0, 82, 150)   # Dimmed dark orange


def marker_color(priority_rank: int, killable: bool = True, held: bool = False):
    """BGR colour for one marker (shared rule with the desktop overlay)."""
    primary = priority_rank <= 1
    if not killable:
        if held:
            return HELD_EARLY_PRIMARY_COLOR if primary else HELD_EARLY_SECONDARY_COLOR
        return EARLY_PRIMARY_COLOR if primary else EARLY_SECONDARY_COLOR
    if held:
        return HELD_PRIMARY_COLOR if primary else HELD_SECONDARY_COLOR
    return PRIMARY_TARGET_COLOR if primary else SECONDARY_TARGET_COLOR


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
    ) -> np.ndarray:
        return self.annotate_candidates(img, candidates, copy=copy)

    def annotate_candidates(
        self,
        img: np.ndarray,
        candidates: list[LastHitCandidate],
        copy: bool = True,
    ) -> np.ndarray:
        if not isinstance(img, np.ndarray) or img.ndim != 3 or img.shape[2] != 3:
            raise ValueError("frame must be a 3-channel BGR image")

        canvas = img.copy() if copy else img
        if not candidates:
            return canvas
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

            held = bool(getattr(candidate, "held", False))
            killable = bool(getattr(candidate, "killable", True))
            color = marker_color(priority_rank, killable, held)
            thickness = self.line_thickness + (
                1 if priority_rank == 1 and not held and killable else 0
            )
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        return canvas
