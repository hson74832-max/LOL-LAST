from dataclasses import dataclass


@dataclass(frozen=True)
class Minion:
    """A detected minion with a conservative upper bound on its health fraction."""

    minion_type: str
    health_level: float
    bbox: tuple[float, float, float, float]
    confidence: float
