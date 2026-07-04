"""OpenCV quality gate for blur, glare, and shadow detection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import cv2
import numpy as np

from .config import QUALITY_THRESHOLDS, QualityThresholds


@dataclass(frozen=True)
class QualityResult:
    passed: bool
    condition: str
    confidence: float
    issues: List[str]
    metrics: Dict[str, float]


def evaluate_quality(
    image: np.ndarray,
    thresholds: QualityThresholds = QUALITY_THRESHOLDS,
) -> QualityResult:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian_variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    mean_brightness = float(gray.mean())
    saturated_ratio = float((gray >= 245).mean())
    dark_ratio = float((gray <= 30).mean())

    metrics = {
        "laplacian_variance": round(laplacian_variance, 4),
        "mean_brightness": round(mean_brightness, 4),
        "saturated_ratio": round(saturated_ratio, 4),
        "dark_ratio": round(dark_ratio, 4),
    }

    if (
        mean_brightness >= thresholds.glare_mean_brightness
        or saturated_ratio >= thresholds.glare_saturated_ratio
    ):
        return QualityResult(False, "glare", 1.0, ["glare"], metrics)

    if (
        mean_brightness <= thresholds.shadow_mean_brightness
        or dark_ratio >= thresholds.shadow_dark_ratio
    ):
        return QualityResult(False, "shadowed", 1.0, ["shadowed"], metrics)

    if laplacian_variance <= thresholds.min_laplacian_variance:
        return QualityResult(False, "uncertain", 1.0, ["blurry"], metrics)

    return QualityResult(True, "pass", 1.0, [], metrics)
