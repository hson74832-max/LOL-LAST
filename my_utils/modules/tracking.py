"""Keep a confirmed last-hit marker alive between flickering readings.

A minion that is nearly dead is exactly the one whose 2-3 px health bar is hardest
to read, so its marker regularly blinks off for a frame or two.  Swinging the mouse
for a minion that is *still there* is a lost last hit, so confirmed markers are held
for a short, configurable grace period (``--hold_ms``) instead of vanishing on the
first unreadable frame.

Two rules keep this from resurrecting dead minions:

* only candidates that were already confirmed are held (never a fresh misread),
* the held marker keeps the **last position and last HP reading** and expires after
  ``hold_ms``; nothing is extrapolated, and a minion that stops being detected is
  gone as soon as the grace period ends.
"""

from dataclasses import replace
from math import hypot, isfinite
from time import monotonic

from my_utils.modules.last_hitter import LastHitCandidate
from my_utils.modules.stability import _center_and_size


def match_by_proximity(track_positions, measured_positions, match_ratio: float = 0.75):
    """Greedy nearest-neighbour matching, closest pairs first.

    Both arguments are sequences of ``(x, y, size)`` tuples.  Returns a mapping
    ``measured_index -> track_index``.  Matching closest pairs first keeps two
    neighbouring minions in a wave from being merged into a single track.
    """
    pairs = []
    for track_index, (track_x, track_y, track_size) in enumerate(track_positions):
        limit = max(4.0, match_ratio * track_size)
        for measured_index, (x, y, _) in enumerate(measured_positions):
            distance = hypot(x - track_x, y - track_y)
            if distance <= limit:
                pairs.append((distance, track_index, measured_index))
    pairs.sort()

    matched_measurements: dict[int, int] = {}
    for _, track_index, measured_index in pairs:
        if track_index in matched_measurements.values():
            continue
        if measured_index in matched_measurements:
            continue
        matched_measurements[measured_index] = track_index
    return matched_measurements


class MinionTracker:
    """Attach identities to confirmed candidates and hold them briefly when lost."""

    def __init__(self, hold_ms: float = 250.0, match_ratio: float = 0.75) -> None:
        try:
            hold_ms = float(hold_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError("hold time must be a number") from exc
        if not isfinite(hold_ms) or hold_ms < 0:
            raise ValueError("hold time must be a finite, non-negative number of milliseconds")
        if not isfinite(match_ratio) or not 0 < match_ratio <= 2:
            raise ValueError("match ratio must be greater than 0 and at most 2")
        self.hold_seconds = hold_ms / 1000.0
        self.match_ratio = float(match_ratio)
        self._tracks: list[dict] = []

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()

    def update(self, candidates, frame_shape=None, now: float | None = None) -> list[LastHitCandidate]:
        """Return the confirmed candidates plus any still-held ones."""
        now = monotonic() if now is None else float(now)
        candidates = list(candidates or [])

        measured = []
        for index, candidate in enumerate(candidates):
            geometry = _center_and_size(getattr(candidate, "minion", None))
            if geometry is not None:
                measured.append((index, candidate, geometry))

        matched_measurements = match_by_proximity(
            [(track["x"], track["y"], track["size"]) for track in self._tracks],
            [geometry for _, _, geometry in measured],
            self.match_ratio,
        )
        matched_tracks = {track_index: measured_index
                          for measured_index, track_index in matched_measurements.items()}

        # Refresh the tracks that are still visible, in detection order.
        visible: list[LastHitCandidate] = []
        new_tracks: list[dict] = []
        held_candidates: list[LastHitCandidate] = []
        for measured_index, (_, candidate, (x, y, size)) in enumerate(measured):
            track_index = matched_measurements.get(measured_index)
            entry = {
                "x": x,
                "y": y,
                "size": size,
                "candidate": candidate,
                "seen_at": now,
            }
            new_tracks.append(entry)
            visible.append(candidate)

        # Tracks that vanished a moment ago: re-emit the last confirmed reading once,
        # flagged as held so the visuals can show it differently.
        for track_index, track in enumerate(self._tracks):
            if track_index in matched_tracks:
                continue
            age = now - track["seen_at"]
            if self.hold_seconds and 0 <= age <= self.hold_seconds:
                held_candidates.append(replace(track["candidate"], held=True))
                new_tracks.append(track)

        self._tracks = new_tracks
        return visible + held_candidates
