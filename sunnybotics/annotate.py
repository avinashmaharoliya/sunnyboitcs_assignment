"""OpenCV annotations for field-review images."""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import cv2
import numpy as np

from .config import CONDITION_COLORS_BGR
from .image_io import write_jpeg


def annotate_image(image: np.ndarray, result: Dict[str, object], output_path: Path) -> bool:
    annotated = image.copy()
    height, width = annotated.shape[:2]
    condition = str(result["condition"])
    color = CONDITION_COLORS_BGR.get(condition, CONDITION_COLORS_BGR["uncertain"])
    thickness = max(4, min(width, height) // 90)

    cv2.rectangle(annotated, (0, 0), (width - 1, height - 1), color, thickness)

    lines = [
        "%s  conf %.2f" % (condition.upper(), float(result["confidence"])),
        "priority %s" % result["cleaning_priority_score"],
        "%s  %s" % (result.get("panel_id", ""), result.get("gps_status", "")),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = max(0.55, min(width, height) / 900.0)
    text_thickness = max(1, int(round(scale * 2)))
    padding = 8
    x = thickness + padding
    y = thickness + padding

    text_sizes = [cv2.getTextSize(line, font, scale, text_thickness)[0] for line in lines]
    box_width = max(size[0] for size in text_sizes) + padding * 2
    line_height = max(size[1] for size in text_sizes) + padding
    box_height = line_height * len(lines) + padding

    overlay = annotated.copy()
    cv2.rectangle(overlay, (x, y), (x + box_width, y + box_height), color, -1)
    cv2.addWeighted(overlay, 0.86, annotated, 0.14, 0, annotated)

    baseline_y = y + padding + text_sizes[0][1]
    for index, line in enumerate(lines):
        cv2.putText(
            annotated,
            line,
            (x + padding, baseline_y + index * line_height),
            font,
            scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return write_jpeg(output_path, annotated)
