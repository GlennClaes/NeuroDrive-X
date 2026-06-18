"""Bird's-eye-view map rendering from LiDAR and route geometry."""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import cv2
import numpy as np


def render_bird_eye_view(
    lidar_points: np.ndarray | None,
    route_locations: Sequence[object] | None = None,
    ego_transform: object | None = None,
    size: int = 128,
    meters: float = 60.0,
    include_route: bool = True,
) -> np.ndarray:
    """Render a 3-channel BEV tensor in channel-first format.

    Channel 0 contains LiDAR occupancy, channel 1 contains nearby route points,
    and channel 2 contains the ego vehicle footprint and forward heading.
    """

    canvas = np.zeros((size, size, 3), dtype=np.float32)
    _draw_lidar(canvas, lidar_points, meters)
    if include_route and route_locations and ego_transform is not None:
        _draw_route(canvas, route_locations, ego_transform, meters)
    _draw_ego(canvas)
    return np.transpose(canvas, (2, 0, 1)).astype(np.float32)


def _draw_lidar(canvas: np.ndarray, lidar_points: np.ndarray | None, meters: float) -> None:
    if lidar_points is None:
        return
    points = np.asarray(lidar_points, dtype=np.float32)
    if points.size == 0 or points.ndim != 2 or points.shape[1] < 2:
        return
    size = canvas.shape[0]
    px, py = _points_to_pixels(points[:, 0], points[:, 1], size, meters)
    valid = (px >= 0) & (px < size) & (py >= 0) & (py < size)
    canvas[py[valid], px[valid], 0] = 1.0


def _draw_route(canvas: np.ndarray, route_locations: Sequence[object], ego_transform: object, meters: float) -> None:
    size = canvas.shape[0]
    ego_location = ego_transform.location
    yaw = math.radians(float(ego_transform.rotation.yaw))
    cos_yaw = math.cos(-yaw)
    sin_yaw = math.sin(-yaw)
    route_pixels: list[tuple[int, int]] = []

    for location in route_locations:
        dx = float(location.x - ego_location.x)
        dy = float(location.y - ego_location.y)
        local_x = dx * cos_yaw - dy * sin_yaw
        local_y = dx * sin_yaw + dy * cos_yaw
        px, py = _point_to_pixel(local_x, local_y, size, meters)
        if 0 <= px < size and 0 <= py < size:
            route_pixels.append((px, py))

    for start, end in zip(route_pixels, route_pixels[1:]):
        cv2.line(canvas, start, end, (0.0, 1.0, 0.0), 1, cv2.LINE_AA)


def _draw_ego(canvas: np.ndarray) -> None:
    size = canvas.shape[0]
    center = size // 2
    car_length = max(4, size // 14)
    car_width = max(3, size // 24)
    cv2.rectangle(
        canvas,
        (center - car_width, center - car_length // 2),
        (center + car_width, center + car_length // 2),
        (0.0, 0.0, 1.0),
        thickness=-1,
    )
    cv2.arrowedLine(canvas, (center, center), (center, center - car_length), (0.0, 0.0, 1.0), 1, tipLength=0.35)


def _points_to_pixels(x: np.ndarray, y: np.ndarray, size: int, meters: float) -> tuple[np.ndarray, np.ndarray]:
    scale = size / meters
    center = size // 2
    px = np.rint(x * scale + center).astype(np.int32)
    py = np.rint(center - y * scale).astype(np.int32)
    return px, py


def _point_to_pixel(x: float, y: float, size: int, meters: float) -> tuple[int, int]:
    px, py = _points_to_pixels(np.array([x]), np.array([y]), size, meters)
    return int(px[0]), int(py[0])

