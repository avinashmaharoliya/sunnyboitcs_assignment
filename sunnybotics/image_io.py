"""Image IO helpers that handle Windows paths consistently."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import cv2
import numpy as np


def read_image(path: Path) -> Optional[np.ndarray]:
    """Read an image with OpenCV, returning None for corrupt/unsupported files."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return image


def write_jpeg(path: Path, image: np.ndarray, quality: int = 92) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return False
    encoded.tofile(str(path))
    return True
