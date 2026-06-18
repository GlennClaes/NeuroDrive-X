"""Extensible object detection interface for CARLA camera frames."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

try:
    from ultralytics import YOLO
except ImportError:  # pragma: no cover - optional dependency path.
    YOLO = None  # type: ignore[assignment]


@dataclass(frozen=True)
class Detection:
    """Object detection output in image coordinates."""

    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


class ObjectDetector:
    """YOLO/OpenCV detector with a semantic-segmentation fallback.

    Ultralytics YOLO is the preferred backend for a realistic portfolio setup.
    If YOLO is unavailable or no model can be loaded, CARLA semantic segmentation
    still provides actor-level detections so the rest of the stack keeps running.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        labels: Iterable[str] | None = None,
        confidence_threshold: float = 0.45,
        backend: str = "ultralytics",
        image_size: int = 640,
        device: str = "auto",
        enabled_classes: Iterable[str] | None = None,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.labels = list(labels or [])
        self.backend = backend
        self.image_size = image_size
        self.device = None if device == "auto" else device
        self.enabled_classes = set(enabled_classes or [])
        self.yolo_model: object | None = None
        self.net: cv2.dnn_Net | None = None
        if backend == "ultralytics" and model_path and YOLO is not None:
            try:
                self.yolo_model = YOLO(str(model_path))
                LOGGER.info("Loaded Ultralytics YOLO model from %s", model_path)
                return
            except Exception:
                LOGGER.warning("Could not load Ultralytics YOLO model; semantic fallback remains active.", exc_info=True)
        elif backend == "ultralytics" and YOLO is None:
            LOGGER.info("Ultralytics is not installed; semantic object-detection fallback remains active.")

        if model_path and backend in {"opencv", "dnn"}:
            model = Path(model_path)
            if not model.exists():
                raise FileNotFoundError(f"Object detection model not found: {model}")
            cfg = str(config_path) if config_path else ""
            self.net = cv2.dnn.readNet(str(model), cfg)
            LOGGER.info("Loaded object detection model from %s", model)

    def detect(self, rgb_image: np.ndarray, semantic_image: np.ndarray | None = None) -> list[Detection]:
        """Detect dynamic actors in an RGB frame."""

        if self.yolo_model is not None:
            return self._detect_with_yolo(rgb_image)
        if self.net is None:
            return self._detect_from_semantics(semantic_image)
        if rgb_image is None or rgb_image.size == 0:
            return []

        blob = cv2.dnn.blobFromImage(rgb_image, 1.0 / 255.0, (320, 320), swapRB=False, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return self._parse_dnn_outputs(outputs, rgb_image.shape[:2])

    @classmethod
    def from_config(cls, config: dict[str, object] | None) -> "ObjectDetector":
        """Create a detector from a YAML object-detection config block."""

        config = config or {}
        return cls(
            model_path=config.get("model_path") if isinstance(config.get("model_path"), str) else None,
            confidence_threshold=float(config.get("confidence_threshold", 0.45)),
            backend=str(config.get("backend", "ultralytics")),
            image_size=int(config.get("image_size", 640)),
            device=str(config.get("device", "auto")),
            enabled_classes=config.get("enabled_classes") if isinstance(config.get("enabled_classes"), list) else None,
        )

    def _detect_with_yolo(self, rgb_image: np.ndarray) -> list[Detection]:
        if rgb_image is None or rgb_image.size == 0 or self.yolo_model is None:
            return []
        results = self.yolo_model.predict(
            source=rgb_image,
            imgsz=self.image_size,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False,
        )
        detections: list[Detection] = []
        for result in results:
            names = getattr(result, "names", {})
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            for box in boxes:
                class_id = int(box.cls[0].item())
                label = str(names.get(class_id, f"class_{class_id}"))
                if self.enabled_classes and label not in self.enabled_classes:
                    continue
                confidence = float(box.conf[0].item())
                x1, y1, x2, y2 = [int(value) for value in box.xyxy[0].tolist()]
                detections.append(Detection(label, confidence, (x1, y1, x2, y2)))
        return detections

    def _parse_dnn_outputs(self, outputs: tuple[np.ndarray, ...] | list[np.ndarray], shape: tuple[int, int]) -> list[Detection]:
        height, width = shape
        detections: list[Detection] = []
        for output in outputs:
            rows = output.reshape(-1, output.shape[-1])
            for row in rows:
                if row.shape[0] < 6:
                    continue
                confidence = float(row[4] * np.max(row[5:]))
                if confidence < self.confidence_threshold:
                    continue
                class_id = int(np.argmax(row[5:]))
                cx, cy, w, h = row[:4]
                x1 = int((cx - w / 2.0) * width)
                y1 = int((cy - h / 2.0) * height)
                x2 = int((cx + w / 2.0) * width)
                y2 = int((cy + h / 2.0) * height)
                label = self.labels[class_id] if class_id < len(self.labels) else f"class_{class_id}"
                detections.append(Detection(label, confidence, (x1, y1, x2, y2)))
        return detections

    def _detect_from_semantics(self, semantic_image: np.ndarray | None) -> list[Detection]:
        if semantic_image is None or semantic_image.size == 0:
            return []

        label_channel = semantic_image[:, :, 2] if semantic_image.ndim == 3 else semantic_image
        class_map = {
            4: "pedestrian",
            10: "vehicle",
            18: "traffic_light",
        }
        detections: list[Detection] = []
        for class_id, label in class_map.items():
            mask = (label_channel == class_id).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 80:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                confidence = min(1.0, area / 2000.0)
                detections.append(Detection(label, confidence, (x, y, x + w, y + h)))
        return detections
