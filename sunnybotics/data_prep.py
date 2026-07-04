"""Dataset reorganization for YOLOv8 classification training."""

from __future__ import annotations

import random
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from .config import CLASS_MERGE, IMAGE_EXTENSIONS
from .image_io import read_image


@dataclass
class PrepSummary:
    train_counts: Dict[str, int]
    val_counts: Dict[str, int]
    skipped: List[str]
    output_dir: Path


def natural_key(path: Path) -> List[object]:
    parts = re.split(r"(\d+)", path.name.lower())
    return [int(part) if part.isdigit() else part for part in parts]


def iter_source_images(dataset_dir: Path) -> Iterable[Tuple[str, Path]]:
    for class_dir in sorted([p for p in dataset_dir.iterdir() if p.is_dir()], key=lambda p: p.name.lower()):
        merged_class = CLASS_MERGE.get(class_dir.name.lower())
        if not merged_class:
            continue
        files = [p for p in class_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
        for image_path in sorted(files, key=natural_key):
            yield merged_class, image_path


def safe_dest_name(image_path: Path, counter: int) -> str:
    source_class = re.sub(r"[^A-Za-z0-9]+", "-", image_path.parent.name).strip("-").lower()
    stem = re.sub(r"[^A-Za-z0-9]+", "-", image_path.stem).strip("-").lower()
    suffix = image_path.suffix.lower()
    return "%s_%05d_%s%s" % (source_class, counter, stem, suffix)


def prepare_yolo_dataset(
    dataset_dir: Path,
    output_dir: Path,
    val_ratio: float = 0.2,
    seed: int = 42,
    validate_images: bool = True,
) -> PrepSummary:
    if output_dir.exists():
        shutil.rmtree(str(output_dir))
    for split in ("train", "val"):
        for class_name in ("clean", "dirty", "damaged"):
            (output_dir / split / class_name).mkdir(parents=True, exist_ok=True)

    by_class: Dict[str, List[Path]] = {"clean": [], "dirty": [], "damaged": []}
    skipped: List[str] = []

    for merged_class, image_path in iter_source_images(dataset_dir):
        if validate_images and read_image(image_path) is None:
            skipped.append(str(image_path))
            continue
        by_class[merged_class].append(image_path)

    rng = random.Random(seed)
    train_counts = {"clean": 0, "dirty": 0, "damaged": 0}
    val_counts = {"clean": 0, "dirty": 0, "damaged": 0}
    copy_counter = 0

    for class_name, paths in by_class.items():
        paths = list(paths)
        rng.shuffle(paths)
        val_count = max(1, int(round(len(paths) * val_ratio))) if paths else 0
        val_set = set(paths[:val_count])

        for image_path in paths:
            split = "val" if image_path in val_set else "train"
            copy_counter += 1
            destination = output_dir / split / class_name / safe_dest_name(image_path, copy_counter)
            shutil.copy2(str(image_path), str(destination))
            if split == "val":
                val_counts[class_name] += 1
            else:
                train_counts[class_name] += 1

    readme = output_dir / "README.txt"
    readme.write_text(
        "YOLOv8 classification dataset generated from the original five folders.\n"
        "Class merge: Clean -> clean, Dusty/Bird-drop -> dirty, "
        "Electrical-damage/Physical-Damage -> damaged.\n"
        "Split: train/val with %.0f%% validation.\n" % (val_ratio * 100),
        encoding="utf-8",
    )

    return PrepSummary(train_counts, val_counts, skipped, output_dir)
