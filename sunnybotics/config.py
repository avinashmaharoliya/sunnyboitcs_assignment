"""Project-wide configuration and operational assumptions."""

from __future__ import annotations

from dataclasses import dataclass


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".heic",
}

CLASS_MERGE = {
    "clean": "clean",
    "dirty": "dirty",
    "damaged": "damaged",
    "dusty": "dirty",
    "bird-drop": "dirty",
    "electrical-damage": "damaged",
    "physical-damage": "damaged",
}

PRIORITY_BASE = {
    "damaged": 90,
    "dirty": 55,
    "glare": 40,
    "uncertain": 30,
    "shadowed": 25,
    "clean": 8,
}

CONDITION_COLORS_BGR = {
    "clean": (70, 170, 70),
    "dirty": (0, 165, 255),
    "damaged": (35, 35, 220),
    "glare": (190, 190, 190),
    "shadowed": (115, 115, 115),
    "uncertain": (130, 130, 130),
}

CONDITION_COLORS_HEX = {
    "clean": "#2ca25f",
    "dirty": "#f59e0b",
    "damaged": "#dc2626",
    "glare": "#9ca3af",
    "shadowed": "#6b7280",
    "uncertain": "#737373",
}


@dataclass(frozen=True)
class QualityThresholds:
    min_laplacian_variance: float = 35.0
    glare_mean_brightness: float = 238.0
    glare_saturated_ratio: float = 0.35
    shadow_mean_brightness: float = 35.0
    shadow_dark_ratio: float = 0.65


@dataclass(frozen=True)
class FarmConfig:
    rows: int = 5
    columns: int = 10
    panel_spacing_m: float = 2.0
    gps_noise_std_m: float = 2.0
    gps_snap_max_distance_m: float = 2.5
    seconds_per_panel: int = 30
    center_latitude: float = 37.9157
    center_longitude: float = -122.3116
    robot_id: str = "SB-ROVER-01"
    mission_id: str = "MISSION-EL-CERRITO-DEMO"
    mission_start_iso: str = "2026-07-05T09:00:00+00:00"


QUALITY_THRESHOLDS = QualityThresholds()
FARM_CONFIG = FarmConfig()
