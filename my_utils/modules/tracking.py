"""Keep a minion's marker alive while it is approaching, and after a lost frame.

A minion that is nearly dead is exactly the one whose 2-3 px health bar is
hardest to read, so its marker regularly blinks off for a frame or two.  Swinging
the mouse for a minion that is *still there* is a lost last hit, so a marker is
kept for a short, configurable grace period (``--hold_ms``) instead of vanishing
on the first unreadable frame.

Three rules keep this from resurrecting dead minions:

* only minions that were already tracked are held (never a fresh misread),
* a held marker keeps the **last position and last reading**; nothing is
  extrapolated, and the marker expires after ``hold_ms``,
* a minion that is actually killed is removed immediately: the tracker watches
  the last health fractions it measured and drops the marker as soon as one of
  them was low enough to be a kill (see :meth:`MinionTracker.update`).
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
    """Attach identities to tracked candidates and keep them while they are low."""

    def __init__(
        self,
        hold_ms: float = 5000.0,
        match_ratio: float = 0.75,
        kill_health_percent: float = 3.0,
        history: int = 3,
    ) -> None:
        try:
            hold_ms = float(hold_ms)
        except (TypeError, ValueError) as exc:
            raise ValueError("hold time must be a number") from exc
        if not isfinite(hold_ms) or hold_ms < 0:
            raise ValueError("hold time must be a finite, non-negative number of milliseconds")
        if not isfinite(match_ratio) or not 0 < match_ratio <= 2:
            raise ValueError("match ratio must be greater than 0 and at most 2")
        if not isfinite(kill_health_percent) or not 0 <= kill_health_percent <= 100:
            raise ValueError("kill health percentage must be between 0 and 100")
        self.hold_seconds = hold_ms / 1000.0
        self.match_ratio = float(match_ratio)
        # A minion whose last reading was this low and whose body disappeared was
        # killed, so its marker goes away at once instead of lingering.
        self.kill_threshold = float(kill_health_percent) / 100.0
        self.history = max(1, int(history))
        self._tracks: list[dict] = []
        self._last_measured: list[LastHitCandidate] = []

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self._last_measured = []

    def _was_killed(self, track: dict) -> bool:
        """True when the track's recent readings say the minion died."""
        readings = track.get("readings") or []
        return self.kill_threshold > 0 and bool(readings) and min(readings) <= self.kill_threshold

    def update(self, candidates, frame_shape=None, now: float | None = None) -> list[LastHitCandidate]:
        """Return the tracked candidates plus any still-kept ones.

        ``candidates`` are this frame's measurements, including the ones that are
        only approaching (``candidate.killable`` is ``False``): those are tracked
        the same way, which is what makes an orange marker persist until the
        health bar is finally low enough for a yellow/green one.
        """
        now = monotonic() if now is None else float(now)
        candidates = list(candidates or [])
        self._last_measured = candidates

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

        visible: list[LastHitCandidate] = []
        new_tracks: list[dict] = []
        kept_candidates: list[LastHitCandidate] = []
        for measured_index, (_, candidate, (x, y, size)) in enumerate(measured):
            track_index = matched_measurements.get(measured_index)
            readings = list(self._tracks[track_index].get("readings", [])) if track_index is not None else []
            readings = (readings + [float(candidate.minion.health_level)])[-self.history:]
            new_tracks.append({
                "x": x,
                "y": y,
                "size": size,
                "candidate": candidate,
                "readings": readings,
                "seen_at": now,
            })
            visible.append(candidate)

        # Tracks that vanished: keep their last marker for a while, unless the
        # minion was clearly killed (its final reading was at death's door).
        for track_index, track in enumerate(self._tracks):
            if track_index in matched_tracks:
                continue
            if self._was_killed(track):
                continue  # the minion died: the marker is removed immediately
            age = now - track["seen_at"]
            if self.hold_seconds and 0 <= age <= self.hold_seconds:
                kept_candidates.append(replace(track["candidate"], held=True))
                new_tracks.append(track)

        self._tracks = new_tracks
        return visible + kept_candidates
