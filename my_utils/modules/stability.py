"""Temporal confirmation so a single misread never becomes a visible marker.

The detector occasionally reports a minion body or a health bar for a minion that
has just died (or misreads a neighbouring bar).  Those artifacts almost never
survive the next frame, so a candidate is only shown after it has been seen
``confirm_frames`` times in a row.  A candidate that disappears is dropped
immediately: holding a marker is exactly what makes the companion point at
corpses.

Minions that are only *approaching* last-hit range ("early"/orange candidates)
are tracked the same way but do **not** go through the confirmation gate: they
cannot be hit yet, so showing them a frame earlier is safe, and gating them would
delay the orange marker that makes the yellow one appear on time.
"""

from dataclasses import dataclass
from math import hypot, isfinite

from my_utils.modules.last_hitter import LastHitCandidate


def _center_and_size(minion):
    bbox = getattr(minion, "bbox", None)
    if bbox is None or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
    except (TypeError, ValueError):
        return None
    if not all(isfinite(v) for v in (x1, y1, x2, y2)) or x2 <= x1 or y2 <= y1:
        return None
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0, max(x2 - x1, y2 - y1))


@dataclass
class _Track:
    x: float
    y: float
    size: float
    hits: int


class CandidateGate:
    """Require consecutive sightings before a last-hit candidate is rendered."""

    def __init__(self, confirm_frames: int = 2, match_ratio: float = 0.75) -> None:
        try:
            confirm_frames = int(confirm_frames)
        except (TypeError, ValueError) as exc:
            raise ValueError("confirm frames must be an integer") from exc
        if confirm_frames < 1:
            raise ValueError("confirm frames must be at least 1")
        if not isfinite(match_ratio) or not 0 < match_ratio <= 2:
            raise ValueError("match ratio must be greater than 0 and at most 2")
        self.confirm_frames = confirm_frames
        self.match_ratio = float(match_ratio)
        self._tracks: list[_Track] = []
        self._generation = None

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._generation = None

    def update(self, candidates, frame_shape=None, generation=None) -> list[LastHitCandidate]:
        """Return candidates that have been stable for ``confirm_frames`` sightings.

        Candidates with ``killable=False`` (minions getting low, not killable yet)
        are passed through unchanged: the gate only guards markers that claim a
        minion can be last-hit right now.

        ``generation`` is any object identifying the detector result behind the
        candidates (compared by identity and kept referenced).  A live frame is
        re-evaluated many times per inference, so sightings are only counted when a
        *new* detector result arrives; re-evaluating the same detection keeps a
        confirmed marker live but never confirms an unconfirmed one, which is what
        stops one-frame misreads from becoming visible markers.

        ``frame_shape`` is accepted for symmetry with the other pipeline helpers and
        is not required for matching: positions are compared in frame pixels.
        """
        new_generation = generation is not None and generation is not self._generation
        if generation is not None:
            self._generation = generation
        candidates = list(candidates or [])
        if not candidates:
            # Nothing seen this frame: every stale marker disappears immediately.
            self._tracks.clear()
            return []
        passed_through = [c for c in candidates if not bool(getattr(c, "killable", True))]
        candidates = [c for c in candidates if bool(getattr(c, "killable", True))]
        if not candidates:
            self._tracks.clear()
            return passed_through

        measured = []
        for index, candidate in enumerate(candidates):
            geometry = _center_and_size(getattr(candidate, "minion", None))
            if geometry is not None:
                measured.append((index, candidate, geometry))
        if not measured:
            self._tracks.clear()
            return passed_through

        # Greedy nearest-neighbour matching, closest pairs first, so two neighbouring
        # minions in a wave are never merged into one track.
        pairs = []
        for track_index, track in enumerate(self._tracks):
            limit = max(4.0, self.match_ratio * track.size)
            for measured_index, (_, _, (x, y, size)) in enumerate(measured):
                distance = hypot(x - track.x, y - track.y)
                if distance <= limit:
                    pairs.append((distance, track_index, measured_index))
        pairs.sort()

        matched_tracks = {}
        matched_measurements = {}
        for distance, track_index, measured_index in pairs:
            if track_index in matched_tracks or measured_index in matched_measurements:
                continue
            matched_tracks[track_index] = measured_index
            matched_measurements[measured_index] = track_index

        new_tracks: list[_Track] = []
        confirmed = []
        for measured_index, (candidate_index, candidate, (x, y, size)) in enumerate(measured):
            track_index = matched_measurements.get(measured_index)
            if track_index is None:
                hits = 1
            elif new_generation or generation is None:
                hits = self._tracks[track_index].hits + 1
            else:
                # Same detector result: a live re-read may only keep a marker alive.
                hits = self._tracks[track_index].hits
            new_tracks.append(_Track(x=x, y=y, size=size, hits=hits))
            if hits >= self.confirm_frames:
                confirmed.append((candidate_index, candidate))

        # Tracks vanish with their candidate: no memory of minions that are already dead.
        self._tracks = new_tracks
        confirmed.sort(key=lambda item: item[0])
        return [candidate for _, candidate in confirmed] + passed_through
