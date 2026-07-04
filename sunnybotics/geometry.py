"""Panel registry, simulated robot metadata, and GPS snapping."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from .config import FARM_CONFIG, FarmConfig

EARTH_RADIUS_M = 6371000.0
METERS_PER_DEGREE_LAT = 111320.0


@dataclass(frozen=True)
class Panel:
    panel_id: str
    row: int
    column: int
    latitude: float
    longitude: float


def meters_to_lat_lon_delta(north_m: float, east_m: float, at_latitude: float) -> Tuple[float, float]:
    lat_delta = north_m / METERS_PER_DEGREE_LAT
    lon_scale = METERS_PER_DEGREE_LAT * math.cos(math.radians(at_latitude))
    lon_delta = east_m / lon_scale
    return lat_delta, lon_delta


def offset_coordinate(latitude: float, longitude: float, north_m: float, east_m: float) -> Tuple[float, float]:
    lat_delta, lon_delta = meters_to_lat_lon_delta(north_m, east_m, latitude)
    return latitude + lat_delta, longitude + lon_delta


def build_panel_registry(config: FarmConfig = FARM_CONFIG) -> List[Panel]:
    panels: List[Panel] = []
    row_midpoint = (config.rows - 1) / 2.0
    col_midpoint = (config.columns - 1) / 2.0

    for row in range(1, config.rows + 1):
        for column in range(1, config.columns + 1):
            north_m = (row - 1 - row_midpoint) * config.panel_spacing_m
            east_m = (column - 1 - col_midpoint) * config.panel_spacing_m
            latitude, longitude = offset_coordinate(
                config.center_latitude,
                config.center_longitude,
                north_m,
                east_m,
            )
            panels.append(
                Panel(
                    panel_id="SB-R%02d-C%02d" % (row, column),
                    row=row,
                    column=column,
                    latitude=latitude,
                    longitude=longitude,
                )
            )
    return panels


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def nearest_panel(latitude: float, longitude: float, registry: Iterable[Panel]) -> Tuple[Panel, float]:
    best_panel: Optional[Panel] = None
    best_distance = float("inf")

    for panel in registry:
        distance = haversine_m(latitude, longitude, panel.latitude, panel.longitude)
        if distance < best_distance:
            best_distance = distance
            best_panel = panel

    if best_panel is None:
        raise ValueError("Panel registry is empty")
    return best_panel, best_distance


def expected_panel_for_timestamp(
    timestamp: datetime,
    mission_start: datetime,
    registry: List[Panel],
    seconds_per_panel: int = FARM_CONFIG.seconds_per_panel,
) -> Panel:
    elapsed_seconds = max(0, int((timestamp - mission_start).total_seconds()))
    index = (elapsed_seconds // seconds_per_panel) % len(registry)
    return registry[index]


def simulate_metadata(
    sequence_index: int,
    registry: List[Panel],
    mission_start: datetime,
    rng: random.Random,
    config: FarmConfig = FARM_CONFIG,
) -> Dict[str, object]:
    expected_panel = registry[sequence_index % len(registry)]
    timestamp = mission_start + timedelta(seconds=sequence_index * config.seconds_per_panel)
    noise_north_m = rng.gauss(0.0, config.gps_noise_std_m)
    noise_east_m = rng.gauss(0.0, config.gps_noise_std_m)
    latitude, longitude = offset_coordinate(
        expected_panel.latitude,
        expected_panel.longitude,
        noise_north_m,
        noise_east_m,
    )

    return {
        "timestamp": timestamp,
        "latitude": latitude,
        "longitude": longitude,
        "expected_panel": expected_panel,
        "noise_north_m": noise_north_m,
        "noise_east_m": noise_east_m,
    }


def write_panel_registry(path: Path, registry: List[Panel]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump([asdict(panel) for panel in registry], handle, indent=2)


def load_panel_registry(path: Path) -> List[Panel]:
    with path.open("r", encoding="utf-8") as handle:
        records = json.load(handle)
    return [Panel(**record) for record in records]
