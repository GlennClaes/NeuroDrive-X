"""Camera frame preprocessing for reinforcement learning and debugging."""

from __future__ import annotations

import logging
from typing import Tuple

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


def resize_and_normalize_rgb(
    image: np.ndarray,
    output_size: Tuple[int, int] = (84, 84),
) -> np.ndarray:
    """Resize an RGB image and return a channel-first float32 tensor in [0, 1]."""

    if image is None or image.size == 0:
        LOGGER.debug("Received an empty RGB image; returning zeros.")
        width, height = output_size
        return np.zeros((3, height, width), dtype=np.float32)

    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError("Expected an RGB image with shape (H, W, 3).")

    width, height = output_size
    resized = cv2.resize(image[:, :, :3], (width, height), interpolation=cv2.INTER_AREA)
    normalized = resized.astype(np.float32) / 255.0
    return np.transpose(normalized, (2, 0, 1))


def normalize_semantic_image(image: np.ndarray) -> np.ndarray:
    """Normalize a semantic segmentation image to a single uint8 label channel."""

    if image is None or image.size == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if image.ndim == 3:
        return image[:, :, 2].astype(np.uint8)
    return image.astype(np.uint8)


def calculate_brightness(image: np.ndarray) -> float:
    """Return average image brightness normalized to [0, 1]."""

    if image is None or image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray) / 255.0)


def draw_hud_text(image: np.ndarray, lines: list[str]) -> np.ndarray:
    """Draw a compact debug HUD on a copy of an RGB image."""

    output = image.copy()
    for index, line in enumerate(lines):
        y = 24 + index * 22
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            output,
            line,
            (12, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return output

