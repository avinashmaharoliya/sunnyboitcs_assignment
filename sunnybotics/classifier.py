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

    def _inference_prediction(self, image_path: Path) -> Prediction:
        try:
            result = self.model(str(image_path), verbose=False)[0]
            probs = getattr(result, "probs", None)
            if probs is None:
                return Prediction(
                    condition="uncertain",
                    confidence=0.0,
                    model_source="yolov8",
                    issues=["missing_probability_scores"],
                )

            top_index = int(getattr(probs, "top1", 0))
            confidence = float(getattr(probs, "top1conf", 0.0))
            names = getattr(result, "names", None) or getattr(self.model, "names", {})
            class_name = names.get(top_index, str(top_index)) if isinstance(names, dict) else str(top_index)
            condition = normalize_model_label(class_name)

            if confidence <= 0.0:
                return Prediction(
                    condition="uncertain",
                    confidence=0.0,
                    model_source="yolov8",
                    issues=["zero_confidence"],
                )

            return Prediction(
                condition=condition,
                confidence=confidence,
                model_source="yolov8",
                issues=[],
            )
        except Exception as exc:  # pragma: no cover - depends on local ML stack
            return Prediction(
                condition="uncertain",
                confidence=0.0,
                model_source="yolov8",
                issues=["inference_error:%s" % exc],
            )

    def predict(self, image_path: Path, source_label: str) -> Prediction:
        if self.model is not None:
            return self._inference_prediction(image_path)

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
