"""Read-only last-hit eligibility and priority calculation.

No game state, input devices, or timing assumptions are used here.  The caller
must supply conservative estimates of damage and minion maximum health.
"""

from dataclasses import dataclass
from math import hypot, isfinite
from types import MappingProxyType
from typing import Mapping

from my_utils.data_structures.minion import Minion


MINION_TYPES = frozenset({"caster", "melee", "siege"})


def _finite_number(value, name):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not isfinite(number):
        raise ValueError(f"{name} must be a finite number")
    return number


@dataclass(frozen=True)
class LastHitSettings:
    # A LOWER bound on actual post-mitigation damage, not the champion's raw AD.
    effective_damage: float
    # UPPER bounds on maximum health, updated as the game changes.
    max_health: Mapping[str, float]
    health_margin: float = 0.0
    min_confidence: float = 0.25
    player_anchor: tuple[float, float] = (0.5, 0.5)
    max_distance: float | None = None

    def __post_init__(self):
        damage = _finite_number(self.effective_damage, "effective damage")
        if damage <= 0:
            raise ValueError("effective damage must be greater than zero")
        object.__setattr__(self, "effective_damage", damage)

        if not isinstance(self.max_health, Mapping) or not self.max_health:
            raise ValueError("provide maximum health for at least one minion type")
        health = {}
        for minion_type, value in self.max_health.items():
            if minion_type not in MINION_TYPES:
                raise ValueError(f"unsupported minion type: {minion_type}")
            value = _finite_number(value, f"{minion_type} maximum health")
            if value <= 0:
                raise ValueError(f"{minion_type} maximum health must be greater than zero")
            health[minion_type] = value
        object.__setattr__(self, "max_health", MappingProxyType(health))

        margin = _finite_number(self.health_margin, "health margin")
        confidence = _finite_number(self.min_confidence, "minimum confidence")
        if not 0 <= margin < 1:
            raise ValueError("health margin must be at least 0 and less than 1")
        if not 0 < confidence <= 1:
            raise ValueError("minimum confidence must be greater than 0 and at most 1")
        object.__setattr__(self, "health_margin", margin)
        object.__setattr__(self, "min_confidence", confidence)

        if len(self.player_anchor) != 2:
            raise ValueError("player anchor must have two coordinates")
        anchor = tuple(_finite_number(v, "player anchor") for v in self.player_anchor)
        if not all(0 <= v <= 1 for v in anchor):
            raise ValueError("player anchor coordinates must be between 0 and 1")
        object.__setattr__(self, "player_anchor", anchor)

        if self.max_distance is not None:
            max_distance = _finite_number(self.max_distance, "maximum distance")
            if max_distance <= 0:
                raise ValueError("maximum distance must be greater than zero")
            object.__setattr__(self, "max_distance", max_distance)


@dataclass(frozen=True)
class LastHitCandidate:
    minion: Minion
    health_upper_bound: float     # absolute HP upper bound (fraction x max HP)
    distance: float
    # True when the minion is no longer detected and the marker is only held for a
    # moment (--hold_ms) so a flickering health bar cannot make it blink away.
    held: bool = False

    @property
    def health_percent(self) -> float:
        return 100.0 * float(self.minion.health_level)


class LastHitAdvisor:
    """Return only health-eligible minions, nearest to the player anchor first."""

    def __init__(self, settings: LastHitSettings):
        self.settings = settings

    def rank(self, minions: list[Minion], frame_shape: tuple[int, ...]) -> list[LastHitCandidate]:
        if len(frame_shape) < 2 or frame_shape[0] <= 0 or frame_shape[1] <= 0:
            raise ValueError("frame must have a positive width and height")
        height, width = frame_shape[:2]
        player_x = width * self.settings.player_anchor[0]
        player_y = height * self.settings.player_anchor[1]
        candidates = []

        for minion in minions:
            try:
                label = minion.minion_type.strip().lower().split()
                if (len(label) != 3 or label[0] not in {"red", "blue"}
                        or label[1] not in MINION_TYPES or label[2] != "minion"):
                    continue
                max_health = self.settings.max_health.get(label[1])
                if max_health is None:
                    continue  # Never guess the stats of an unsupported/uncalibrated type.
                health = _finite_number(minion.health_level, "health")
                confidence = _finite_number(minion.confidence, "confidence")
                box = tuple(_finite_number(v, "bounding box") for v in minion.bbox)
            except (AttributeError, TypeError, ValueError):
                continue

            if not 0 < health <= 1 or not self.settings.min_confidence <= confidence <= 1:
                continue
            if len(box) != 4:
                continue
            x1, y1, x2, y2 = box
            if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
                continue

            # Both the reading and the supplied max HP are upper bounds; add an
            # extra margin for quantization/measurement error before comparing.
            health_upper_bound = min(1.0, health + self.settings.health_margin) * max_health
            if health_upper_bound > self.settings.effective_damage:
                continue
            distance = hypot((x1 + x2) / 2 - player_x, (y1 + y2) / 2 - player_y)
            if self.settings.max_distance is not None and distance > self.settings.max_distance:
                continue
            candidates.append(LastHitCandidate(minion, health_upper_bound, distance))

        # Preserve the original nearest-eligible priority, with stable tie-breaks.
        return sorted(candidates, key=lambda c: (
            c.distance, c.health_upper_bound, tuple(c.minion.bbox), c.minion.minion_type
        ))
