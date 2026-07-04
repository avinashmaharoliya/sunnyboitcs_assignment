from __future__ import annotations

import argparse
from pathlib import Path

from sunnybotics.data_prep import prepare_yolo_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the 3-class YOLOv8 training dataset.")
    parser.add_argument("--dataset-dir", default="dataset", help="Original 5-folder dataset directory.")
    parser.add_argument("--output-dir", default="prepared/solar_cls", help="YOLOv8 classification output directory.")
    parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation split ratio.")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic shuffle seed.")
    parser.add_argument("--no-validate", action="store_true", help="Copy files without OpenCV readability checks.")
    args = parser.parse_args()

    summary = prepare_yolo_dataset(
        dataset_dir=Path(args.dataset_dir),
        output_dir=Path(args.output_dir),
        val_ratio=args.val_ratio,
        seed=args.seed,
        validate_images=not args.no_validate,
    )

    print("Prepared YOLOv8 classification dataset at %s" % summary.output_dir)
    print("Train counts: %s" % summary.train_counts)
    print("Val counts: %s" % summary.val_counts)
    if summary.skipped:
        print("Skipped unreadable/unsupported images: %d" % len(summary.skipped))
        for skipped in summary.skipped[:10]:
            print("  - %s" % skipped)
        if len(summary.skipped) > 10:
            print("  ...")


if __name__ == "__main__":
    main()
