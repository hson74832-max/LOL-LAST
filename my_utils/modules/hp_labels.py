"""HP numbers above minions: shown early, kept until the minion is gone.

The overlay only marks minions that are already last-hittable, which means a
minion's number used to appear at the very last moment - too late to plan the
next two hits, and gone again as soon as the health bar flickered.

This module adds a second, independent layer: a persistent *HP label* per
minion.  A label is created as soon as a minion is detected and its bar reads at
or below ``threshold_percent`` (default 50 %), then it is kept:

* while the minion keeps being detected, even if a later bar reading flickers
  back above the threshold (the track stays "armed"), and
* for ``hold_ms`` after the minion stops being detected, drawn with a trailing
  ``?`` so a stale reading is never mistaken for a live one.

So the number stays with the minion until it dies - a dead minion disappears
from the capture, and the label follows after the short grace period.  Nothing
is invented: a label only exists for a minion that was actually seen, and the
held copy keeps the last position and the last HP reading.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import monotonic
from types import MappingProxyType
from typing import Mapping

from my_utils.data_structures.minion import Minion
from my_utils.modules.last_hitter import MINION_TYPES, _finite_number
from my_utils.modules.stability import _center_and_size
from my_utils.modules.tracking import match_by_proximity


@dataclass(frozen=True)
class HpLabel:
    """A minion's estimated remaining HP, drawn above its head."""

    minion: Minion
    health_upper_bound: float   # absolute HP upper bound (fraction x max HP)
    # True while the minion is no longer detected and the label is only held for
    # --hp_hold_ms, so the number can never be mistaken for a live reading.
    held: bool = False

    @property
    def health_percent(self) -> float:
        return 100.0 * float(self.minion.health_level)


class HpLabelTracker:
    """Create HP labels below a health threshold and keep them until death."""

    def __init__(
        self,
        max_health: Mapping[str, float] = None,
        threshold_percent: float = 50.0,
        hold_ms: float = 1000.0,
        health_margin: float = 0.0,
        min_confidence: float = 0.25,
        match_ratio: float = 0.75,
    ) -> None:
        if not isinstance(max_health, Mapping) or not max_health:
            raise ValueError("provide maximum health for at least one minion type")
        health = {}
        for minion_type, value in max_health.items():
            if minion_type not in MINION_TYPES:
                raise ValueError(f"unsupported minion type: {minion_type}")
            value = _finite_number(value, f"{minion_type} maximum health")
            if value <= 0:
                raise ValueError(f"{minion_type} maximum health must be greater than zero")
            health[minion_type] = value
        self.max_health = MappingProxyType(health)

        threshold = _finite_number(threshold_percent, "HP label threshold")
        if not 0 < threshold <= 100:
            raise ValueError("HP label threshold must be greater than 0 and at most 100")
        self.threshold_percent = threshold

        hold = _finite_number(hold_ms, "HP label hold time")
        if hold < 0:
            raise ValueError("HP label hold time must be non-negative")
        self.hold_seconds = hold / 1000.0

        margin = _finite_number(health_margin, "health margin")
        if not 0 <= margin < 1:
            raise ValueError("health margin must be at least 0 and less than 1")
        self.health_margin = margin

        confidence = _finite_number(min_confidence, "minimum confidence")
        if not 0 < confidence <= 1:
            raise ValueError("minimum confidence must be greater than 0 and at most 1")
        self.min_confidence = confidence

        ratio = _finite_number(match_ratio, "match ratio")
        if not 0 < ratio <= 2:
            raise ValueError("match ratio must be greater than 0 and at most 2")
        self.match_ratio = ratio

        self._tracks: list[dict] = []
        self.last_labels: list[HpLabel] = []

    @property
    def track_count(self) -> int:
        return len(self._tracks)

    def reset(self) -> None:
        self._tracks.clear()
        self.last_labels = []

    def label_for(self, minion) -> HpLabel | None:
        """Build a label for one detected minion, or ``None`` if it is unusable."""
        if not isinstance(minion, Minion):
            return None
        try:
            label = minion.minion_type.strip().lower().split()
            if (len(label) != 3 or label[0] not in {"red", "blue"}
                    or label[1] not in MINION_TYPES or label[2] != "minion"):
                return None
            max_health = self.max_health.get(label[1])
            if max_health is None:
                return None
            health = _finite_number(minion.health_level, "health")
            confidence = _finite_number(minion.confidence, "confidence")
        except (AttributeError, TypeError, ValueError):
            return None
        if not 0 < health <= 1 or confidence < self.min_confidence:
            return None
        if _center_and_size(minion) is None:
            return None
        health_upper_bound = min(1.0, health + self.health_margin) * max_health
        return HpLabel(minion=minion, health_upper_bound=health_upper_bound)

    def update(self, minions, now: float | None = None) -> list[HpLabel]:
        """Return the labels to draw for this frame's ``minions``."""
        now = monotonic() if now is None else float(now)
        minions = list(minions or [])

        measured = []  # (label, (x, y, size))
        for minion in minions:
            label = self.label_for(minion)
            if label is None:
                continue
            geometry = _center_and_size(minion)
            if geometry is None:
                continue
            measured.append((label, geometry))

        matched = match_by_proximity(
            [(track["x"], track["y"], track["size"]) for track in self._tracks],
            [geometry for _, geometry in measured],
            self.match_ratio,
        )

        labels: list[HpLabel] = []
        new_tracks: list[dict] = []
        for measured_index, (label, (x, y, size)) in enumerate(measured):
            track_index = matched.get(measured_index)
            previous = self._tracks[track_index] if track_index is not None else None
            # Sticky: once a minion was worth a number it keeps one, even if a
            # single bar reading flickers back above the threshold.
            armed = bool(previous and previous.get("armed"))
            armed = armed or label.health_percent <= self.threshold_percent
            new_tracks.append({
                "x": x,
                "y": y,
                "size": size,
                "armed": armed,
                "label": label,
                "seen_at": now,
            })
            if armed:
                labels.append(label)

        # Minions that vanished: keep their last number for a short grace period,
        # marked as held so a stale reading is visible as such.
        for track_index, track in enumerate(self._tracks):
            if track_index in matched.values():
                continue
            if not track.get("armed"):
                continue
            age = now - track["seen_at"]
            if self.hold_seconds and 0 <= age <= self.hold_seconds:
                labels.append(replace(track["label"], held=True))
                new_tracks.append(track)

        self._tracks = new_tracks
        self.last_labels = labels
        return labels
