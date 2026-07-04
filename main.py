from __future__ import annotations

import argparse
from pathlib import Path

from sunnybotics.pipeline import process_dataset

DEFAULT_DATASET_DIR = "prepared/solar_cls"
DEFAULT_MODEL_PATH = "models/solar_yolov8n_best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Sunnybotics solar-panel inspection pipeline.")
    parser.add_argument("--dataset-dir", default=DEFAULT_DATASET_DIR, help="Prepared dataset folder to process.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for results and annotations.")
    parser.add_argument("--model", default=DEFAULT_MODEL_PATH, help="YOLOv8 classification weights path.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N images.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic GPS-noise seed.")
    parser.add_argument(
        "--require-model",
        action="store_true",
        help="Fail if YOLOv8 weights are missing or cannot be loaded.",
    )
    parser.add_argument(
        "--no-folder-fallback",
        action="store_true",
        help="Return uncertain for model-pass images if YOLOv8 weights are unavailable.",
    )
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else None
    if args.require_model and (model_path is None or not model_path.exists()):
        raise SystemExit("YOLOv8 model is required but was not found: %s" % (model_path or "<missing>"))

    summary = process_dataset(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        model_path=model_path,
        limit=args.limit,
        allow_folder_fallback=not args.no_folder_fallback,
        seed=args.seed,
    )

    if args.require_model and not summary["yolo_enabled"]:
        raise SystemExit("YOLOv8 model could not be loaded: %s" % summary["classifier_load_error"])

    print("Processed images: %s" % summary["processed"])
    print("Unreadable images: %s" % summary["unreadable"])
    print("Quality-gate flags: %s" % summary["quality_blocked"])
    print("YOLOv8 enabled: %s" % summary["yolo_enabled"])
    if summary["classifier_load_error"]:
        print("Classifier note: %s" % summary["classifier_load_error"])
    print("CSV: %s" % summary["results_csv"])
    print("JSON: %s" % summary["results_json"])
    print("Annotated images: %s" % summary["annotated_dir"])


if __name__ == "__main__":
    main()
