"""Priority scoring for field operations."""

from __future__ import annotations

from .config import PRIORITY_BASE


def priority_score(condition: str, confidence: float) -> int:
    base = PRIORITY_BASE.get(condition, PRIORITY_BASE["uncertain"])
    bounded_confidence = max(0.0, min(float(confidence), 1.0))
    return int(round(base * bounded_confidence))
