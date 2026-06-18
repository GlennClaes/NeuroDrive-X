"""OpenCV lane detection for CARLA RGB frames."""

from __future__ import annotations

from dataclasses import dataclass
import logging

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LaneDetectionResult:
    """Lane line estimates and center offset in image space."""

    left_line: tuple[int, int, int, int] | None
    right_line: tuple[int, int, int, int] | None
    center_offset_px: float
    confidence: float
    overlay: np.ndarray


def detect_lanes(image: np.ndarray) -> LaneDetectionResult:
    """Detect lane markings using Canny edges and probabilistic Hough lines."""

    if image is None or image.size == 0:
        empty = np.zeros((1, 1, 3), dtype=np.uint8)
        return LaneDetectionResult(None, None, 0.0, 0.0, empty)

    height, width = image.shape[:2]
    gray = cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 60, 150)
    masked = _region_of_interest(edges)

    lines = cv2.HoughLinesP(
        masked,
        rho=1,
        theta=np.pi / 180,
        threshold=35,
        minLineLength=max(20, width // 12),
        maxLineGap=35,
    )

    left_segments: list[tuple[int, int, int, int]] = []
    right_segments: list[tuple[int, int, int, int]] = []
    if lines is not None:
        for line in lines[:, 0]:
            x1, y1, x2, y2 = map(int, line)
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < 0.35:
                continue
            if slope < 0:
                left_segments.append((x1, y1, x2, y2))
            else:
                right_segments.append((x1, y1, x2, y2))

    left_line = _average_line(left_segments, height)
    right_line = _average_line(right_segments, height)
    lane_center = _lane_center_at_bottom(left_line, right_line, width)
    center_offset = float(lane_center - width / 2.0)
    confidence = _confidence(left_line, right_line)
    overlay = draw_lane_overlay(image, left_line, right_line, center_offset)
    return LaneDetectionResult(left_line, right_line, center_offset, confidence, overlay)


def draw_lane_overlay(
    image: np.ndarray,
    left_line: tuple[int, int, int, int] | None,
    right_line: tuple[int, int, int, int] | None,
    center_offset_px: float,
) -> np.ndarray:
    """Return an RGB image with detected lane geometry overlaid."""

    overlay = image.copy()
    for line, color in ((left_line, (0, 255, 255)), (right_line, (0, 180, 255))):
        if line is not None:
            cv2.line(overlay, line[:2], line[2:], color, 4, cv2.LINE_AA)

    height, width = overlay.shape[:2]
    cv2.line(overlay, (width // 2, height), (width // 2, int(height * 0.65)), (255, 255, 255), 2)
    lane_x = int(width / 2 + center_offset_px)
    cv2.circle(overlay, (lane_x, int(height * 0.85)), 7, (0, 255, 0), -1)
    return overlay


def _region_of_interest(edges: np.ndarray) -> np.ndarray:
    height, width = edges.shape[:2]
    polygon = np.array(
        [
            [
                (int(width * 0.08), height),
                (int(width * 0.42), int(height * 0.56)),
                (int(width * 0.58), int(height * 0.56)),
                (int(width * 0.94), height),
            ]
        ],
        dtype=np.int32,
    )
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, polygon, 255)
    return cv2.bitwise_and(edges, mask)


def _average_line(
    segments: list[tuple[int, int, int, int]],
    image_height: int,
) -> tuple[int, int, int, int] | None:
    if not segments:
        return None
    xs: list[int] = []
    ys: list[int] = []
    for x1, y1, x2, y2 in segments:
        xs.extend([x1, x2])
        ys.extend([y1, y2])
    try:
        slope, intercept = np.polyfit(np.array(ys), np.array(xs), deg=1)
    except np.linalg.LinAlgError:
        LOGGER.debug("Could not fit lane line.", exc_info=True)
        return None

    y1 = image_height
    y2 = int(image_height * 0.58)
    x1 = int(slope * y1 + intercept)
    x2 = int(slope * y2 + intercept)
    return x1, y1, x2, y2


def _lane_center_at_bottom(
    left_line: tuple[int, int, int, int] | None,
    right_line: tuple[int, int, int, int] | None,
    image_width: int,
) -> float:
    if left_line and right_line:
        return (left_line[0] + right_line[0]) / 2.0
    if left_line:
        return left_line[0] + image_width * 0.32
    if right_line:
        return right_line[0] - image_width * 0.32
    return image_width / 2.0


def _confidence(
    left_line: tuple[int, int, int, int] | None,
    right_line: tuple[int, int, int, int] | None,
) -> float:
    if left_line and right_line:
        return 1.0
    if left_line or right_line:
        return 0.55
    return 0.0

