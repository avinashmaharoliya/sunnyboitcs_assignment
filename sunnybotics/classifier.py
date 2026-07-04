"""YOLOv8 classifier wrapper with an explicit demo fallback."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from .config import CLASS_MERGE


@dataclass(frozen=True)
class Prediction:
    condition: str
    confidence: float
    model_source: str
    issues: List[str]


def source_label_to_condition(label: str) -> str:
    return CLASS_MERGE.get(label.strip().lower(), "uncertain")


def normalize_model_label(label: str) -> str:
    value = label.strip().lower().replace("_", "-")
    if value in {"clean", "dirty", "damaged"}:
        return value
    return source_label_to_condition(value)


class ConditionClassifier:
    def __init__(
        self,
        model_path: Optional[Path] = None,
        allow_folder_fallback: bool = True,
    ) -> None:
        self.model_path = Path(model_path) if model_path else None
        self.allow_folder_fallback = allow_folder_fallback
        self.model = None
        self.load_error: Optional[str] = None

        if self.model_path and self.model_path.exists():
            try:
                from ultralytics import YOLO  # type: ignore

                self.model = YOLO(str(self.model_path))
            except Exception as exc:  # pragma: no cover - depends on local ML stack
                self.load_error = str(exc)
        elif self.model_path:
            self.load_error = "model file not found: %s" % self.model_path

    @property
    def is_yolo_enabled(self) -> bool:
        return self.model is not None

    def predict(self, image_path: Path, source_label: str) -> Prediction:
        if self.model is not None:
            result = self.model(str(image_path), verbose=False)[0]
            probs = result.probs
            top_index = int(probs.top1)
            confidence = float(probs.top1conf)
            names = getattr(result, "names", None) or getattr(self.model, "names", {})
            class_name = names.get(top_index, str(top_index)) if isinstance(names, dict) else str(top_index)
            return Prediction(
                condition=normalize_model_label(class_name),
                confidence=confidence,
                model_source="yolov8",
                issues=[],
            )

        if self.allow_folder_fallback:
            return Prediction(
                condition=source_label_to_condition(source_label),
                confidence=0.75,
                model_source="folder_label_fallback",
                issues=["missing_yolov8_model"],
            )

        return Prediction(
            condition="uncertain",
            confidence=1.0,
            model_source="none",
            issues=["missing_yolov8_model"],
        )
