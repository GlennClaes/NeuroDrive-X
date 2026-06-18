"""Extensible object detection interface for CARLA camera frames."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Detection:
    """Object detection output in image coordinates."""

    label: str
    confidence: float
    bbox_xyxy: tuple[int, int, int, int]


class ObjectDetector:
    """OpenCV DNN detector with a semantic-segmentation fallback.

    The project intentionally keeps the trained object-detection model swappable:
    pass ONNX/Darknet model paths to enable full detection, or use the semantic
    fallback during early CARLA research runs where segmentation is already
    provided by the simulator.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        config_path: str | Path | None = None,
        labels: Iterable[str] | None = None,
        confidence_threshold: float = 0.45,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.labels = list(labels or [])
        self.net: cv2.dnn_Net | None = None
        if model_path:
            model = Path(model_path)
            if not model.exists():
                raise FileNotFoundError(f"Object detection model not found: {model}")
            cfg = str(config_path) if config_path else ""
            self.net = cv2.dnn.readNet(str(model), cfg)
            LOGGER.info("Loaded object detection model from %s", model)

    def detect(self, rgb_image: np.ndarray, semantic_image: np.ndarray | None = None) -> list[Detection]:
        """Detect dynamic actors in an RGB frame."""

        if self.net is None:
            return self._detect_from_semantics(semantic_image)
        if rgb_image is None or rgb_image.size == 0:
            return []

        blob = cv2.dnn.blobFromImage(rgb_image, 1.0 / 255.0, (320, 320), swapRB=False, crop=False)
        self.net.setInput(blob)
        outputs = self.net.forward(self.net.getUnconnectedOutLayersNames())
        return self._parse_dnn_outputs(outputs, rgb_image.shape[:2])

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

