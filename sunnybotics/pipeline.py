"""End-to-end RF-01 through RF-07 processing pipeline."""

from __future__ import annotations

import csv
import json
import random
import re
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .annotate import annotate_image
from .classifier import ConditionClassifier
from .config import FARM_CONFIG, IMAGE_EXTENSIONS
from .geometry import (
    build_panel_registry,
    expected_panel_for_timestamp,
    nearest_panel,
    simulate_metadata,
    write_panel_registry,
)
from .image_io import read_image
from .quality import evaluate_quality
from .scoring import priority_score


RESULT_FIELDS = [
    "image_id",
    "timestamp",
    "latitude",
    "longitude",
    "robot_id",
    "mission_id",
    "panel_row",
    "panel_column",
    "panel_id",
    "condition",
    "confidence",
    "cleaning_priority_score",
    "annotated_image_path",
    "detected_issues",
    "original_image_path",
    "source_label",
    "model_source",
    "gps_status",
    "gps_distance_m",
    "nearest_panel_id",
    "sequence_panel_id",
    "quality_laplacian_variance",
    "quality_mean_brightness",
    "quality_saturated_ratio",
    "quality_dark_ratio",
]


def natural_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def iter_dataset_images(dataset_dir: Path) -> Iterable[Path]:
    files = [p for p in dataset_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    for image_path in sorted(files, key=lambda p: [str(part).lower() for part in p.parts[:-1]] + natural_key(p)):
        yield image_path


def unique_issues(issues: Iterable[str]) -> str:
    seen = set()
    ordered: List[str] = []
    for issue in issues:
        if issue and issue not in seen:
            seen.add(issue)
            ordered.append(issue)
    return ";".join(ordered)


def condition_issues(condition: str) -> List[str]:
    if condition == "dirty":
        return ["soiling"]
    if condition == "damaged":
        return ["structural_or_electrical_damage"]
    if condition in {"glare", "shadowed", "uncertain"}:
        return [condition]
    return []


def resolve_gps(
    latitude: float,
    longitude: float,
    timestamp: datetime,
    mission_start: datetime,
    registry,
) -> Dict[str, object]:
    nearest, distance_m = nearest_panel(latitude, longitude, registry)
    sequence_panel = expected_panel_for_timestamp(timestamp, mission_start, registry)
    issues: List[str] = []

    if distance_m > FARM_CONFIG.gps_snap_max_distance_m:
        gps_status = "uncertain_gps"
        assigned_panel = sequence_panel
        issues.append("uncertain_gps")
    elif nearest.panel_id != sequence_panel.panel_id:
        gps_status = "sequence_mismatch"
        assigned_panel = nearest
        issues.append("gps_sequence_mismatch")
    else:
        gps_status = "matched"
        assigned_panel = nearest

    return {
        "assigned_panel": assigned_panel,
        "nearest_panel": nearest,
        "sequence_panel": sequence_panel,
        "gps_distance_m": distance_m,
        "gps_status": gps_status,
        "issues": issues,
    }


def make_result_row(
    image_id: str,
    image_path: Path,
    metadata: Dict[str, object],
    gps: Dict[str, object],
    condition: str,
    confidence: float,
    model_source: str,
    annotated_path: str,
    issues: List[str],
    quality_metrics: Dict[str, float],
) -> Dict[str, object]:
    assigned_panel = gps["assigned_panel"]
    nearest = gps["nearest_panel"]
    sequence_panel = gps["sequence_panel"]
    score = priority_score(condition, confidence)
    all_issues = list(issues) + condition_issues(condition)

    return {
        "image_id": image_id,
        "timestamp": metadata["timestamp"].isoformat(),
        "latitude": round(float(metadata["latitude"]), 8),
        "longitude": round(float(metadata["longitude"]), 8),
        "robot_id": FARM_CONFIG.robot_id,
        "mission_id": FARM_CONFIG.mission_id,
        "panel_row": assigned_panel.row,
        "panel_column": assigned_panel.column,
        "panel_id": assigned_panel.panel_id,
        "condition": condition,
        "confidence": round(float(confidence), 4),
        "cleaning_priority_score": score,
        "annotated_image_path": annotated_path,
        "detected_issues": unique_issues(all_issues),
        "original_image_path": str(image_path),
        "source_label": image_path.parent.name,
        "model_source": model_source,
        "gps_status": gps["gps_status"],
        "gps_distance_m": round(float(gps["gps_distance_m"]), 3),
        "nearest_panel_id": nearest.panel_id,
        "sequence_panel_id": sequence_panel.panel_id,
        "quality_laplacian_variance": quality_metrics.get("laplacian_variance", ""),
        "quality_mean_brightness": quality_metrics.get("mean_brightness", ""),
        "quality_saturated_ratio": quality_metrics.get("saturated_ratio", ""),
        "quality_dark_ratio": quality_metrics.get("dark_ratio", ""),
    }


def write_results(output_dir: Path, rows: List[Dict[str, object]], registry) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "results.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)

    with (output_dir / "results.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    with (output_dir / "run_metadata.json").open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "farm_config": asdict(FARM_CONFIG),
                "total_results": len(rows),
                "outputs": {
                    "csv": str(output_dir / "results.csv"),
                    "json": str(output_dir / "results.json"),
                    "panel_registry": str(output_dir / "panel_registry.json"),
                },
            },
            handle,
            indent=2,
        )
    write_panel_registry(output_dir / "panel_registry.json", registry)


def process_dataset(
    dataset_dir: Path,
    output_dir: Path,
    model_path: Optional[Path] = None,
    limit: Optional[int] = None,
    allow_folder_fallback: bool = True,
    seed: int = 42,
) -> Dict[str, object]:
    dataset_dir = Path(dataset_dir)
    output_dir = Path(output_dir)
    annotated_dir = output_dir / "annotated"
    annotated_dir.mkdir(parents=True, exist_ok=True)

    registry = build_panel_registry()
    mission_start = datetime.fromisoformat(FARM_CONFIG.mission_start_iso)
    rng = random.Random(seed)
    classifier = ConditionClassifier(model_path=model_path, allow_folder_fallback=allow_folder_fallback)

    image_paths = list(iter_dataset_images(dataset_dir))
    if limit is not None:
        image_paths = image_paths[:limit]

    rows: List[Dict[str, object]] = []
    unreadable = 0
    quality_blocked = 0

    for sequence_index, image_path in enumerate(image_paths):
        image_id = str(uuid.uuid4())
        metadata = simulate_metadata(sequence_index, registry, mission_start, rng)
        gps = resolve_gps(
            float(metadata["latitude"]),
            float(metadata["longitude"]),
            metadata["timestamp"],
            mission_start,
            registry,
        )

        issues: List[str] = list(gps["issues"])
        image = read_image(image_path)
        quality_metrics: Dict[str, float] = {}
        annotated_path = ""

        if image is None:
            unreadable += 1
            condition = "uncertain"
            confidence = 1.0
            model_source = "image_ingestion"
            issues.append("unreadable_image")
        else:
            quality = evaluate_quality(image)
            quality_metrics = quality.metrics
            if not quality.passed:
                quality_blocked += 1
                condition = quality.condition
                confidence = quality.confidence
                model_source = "opencv_quality_gate"
                issues.extend(quality.issues)
            else:
                prediction = classifier.predict(image_path, image_path.parent.name)
                condition = prediction.condition
                confidence = prediction.confidence
                model_source = prediction.model_source
                issues.extend(prediction.issues)

            output_image = annotated_dir / ("%s.jpg" % image_id)
            row_preview = {
                "condition": condition,
                "confidence": confidence,
                "cleaning_priority_score": priority_score(condition, confidence),
                "panel_id": gps["assigned_panel"].panel_id,
                "gps_status": gps["gps_status"],
            }
            if annotate_image(image, row_preview, output_image):
                annotated_path = str(output_image)

        row = make_result_row(
            image_id=image_id,
            image_path=image_path,
            metadata=metadata,
            gps=gps,
            condition=condition,
            confidence=confidence,
            model_source=model_source,
            annotated_path=annotated_path,
            issues=issues,
            quality_metrics=quality_metrics,
        )
        rows.append(row)

    write_results(output_dir, rows, registry)
    return {
        "processed": len(rows),
        "unreadable": unreadable,
        "quality_blocked": quality_blocked,
        "yolo_enabled": classifier.is_yolo_enabled,
        "classifier_load_error": classifier.load_error,
        "results_csv": str(output_dir / "results.csv"),
        "results_json": str(output_dir / "results.json"),
        "annotated_dir": str(annotated_dir),
    }
