from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a YOLOv8-nano classifier for solar-panel condition labels.")
    parser.add_argument("--data", default="prepared/solar_cls", help="Prepared classification dataset root.")
    parser.add_argument("--model", default="yolov8n-cls.pt", help="Ultralytics model/checkpoint to start from.")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--imgsz", type=int, default=224)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/classify")
    parser.add_argument("--name", default="solar_yolov8n")
    args = parser.parse_args()

    data_dir = Path(args.data)
    if not (data_dir / "train").exists() or not (data_dir / "val").exists():
        raise SystemExit("Prepared dataset not found. Run: python prepare_dataset.py")

    try:
        from ultralytics import YOLO  # type: ignore
    except ImportError as exc:
        raise SystemExit(
            "Ultralytics is not installed in this Python environment.\n"
            "Install it with: python -m pip install ultralytics\n"
            "If your Python is too new for PyTorch wheels, create a Python 3.9-3.12 venv for training."
        ) from exc

    model = YOLO(args.model)
    project_dir = Path(args.project).resolve()
    model.train(
        data=str(data_dir),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=str(project_dir),
        name=args.name,
    )

    save_dir = Path(model.trainer.save_dir)
    best_weights = save_dir / "weights" / "best.pt"
    print("Training complete. Use the best weights with:")
    print("python main.py --model %s" % best_weights)


if __name__ == "__main__":
    main()
