"""LiDAR preprocessing utilities for CARLA point clouds."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class LidarSummary:
    """Compact LiDAR representation for policy observations and safety checks."""

    sector_distances: np.ndarray
    front_distance_m: float | None
    left_distance_m: float | None
    right_distance_m: float | None
    obstacle_detected: bool


def preprocess_lidar(
    points: np.ndarray | Iterable[Iterable[float]] | None,
    sectors: int = 64,
    max_range: float = 60.0,
    obstacle_threshold_m: float = 8.0,
) -> LidarSummary:
    """Convert raw XYZ LiDAR points into normalized radial sector distances."""

    if points is None:
        distances = np.ones(sectors, dtype=np.float32)
        return LidarSummary(distances, None, None, None, False)

    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        distances = np.ones(sectors, dtype=np.float32)
        return LidarSummary(distances, None, None, None, False)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("Expected LiDAR points with shape (N, 3+)").with_traceback(None)

    xyz = array[:, :3]
    planar_distance = np.linalg.norm(xyz[:, :2], axis=1)
    valid = (planar_distance > 0.25) & (planar_distance <= max_range) & (xyz[:, 2] > -2.5)
    xyz = xyz[valid]
    planar_distance = planar_distance[valid]
    if xyz.size == 0:
        distances = np.ones(sectors, dtype=np.float32)
        return LidarSummary(distances, None, None, None, False)

    angles = (np.arctan2(xyz[:, 1], xyz[:, 0]) + 2.0 * np.pi) % (2.0 * np.pi)
    sector_index = np.floor(angles / (2.0 * np.pi) * sectors).astype(np.int32)
    sector_meters = np.full(sectors, max_range, dtype=np.float32)
    np.minimum.at(sector_meters, sector_index, planar_distance)
    normalized = np.clip(sector_meters / max_range, 0.0, 1.0).astype(np.float32)

    front_distance = _sector_min(sector_meters, sectors, -15.0, 15.0)
    left_distance = _sector_min(sector_meters, sectors, 45.0, 110.0)
    right_distance = _sector_min(sector_meters, sectors, -110.0, -45.0)
    obstacle_detected = front_distance is not None and front_distance < obstacle_threshold_m
    return LidarSummary(normalized, front_distance, left_distance, right_distance, obstacle_detected)


def bird_eye_occupancy_grid(
    points: np.ndarray | None,
    size: int = 256,
    meters: float = 60.0,
) -> np.ndarray:
    """Render LiDAR points to a simple bird's-eye occupancy grid."""

    grid = np.zeros((size, size), dtype=np.uint8)
    if points is None:
        return grid
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0 or array.ndim != 2 or array.shape[1] < 2:
        return grid

    scale = size / meters
    center = size // 2
    x = np.clip((array[:, 0] * scale + center).astype(np.int32), 0, size - 1)
    y = np.clip((center - array[:, 1] * scale).astype(np.int32), 0, size - 1)
    grid[y, x] = 255
    return grid


def _sector_min(sector_meters: np.ndarray, sectors: int, start_deg: float, end_deg: float) -> float | None:
    angles = np.linspace(0.0, 360.0, sectors, endpoint=False)
    start = start_deg % 360.0
    end = end_deg % 360.0
    if start <= end:
        mask = (angles >= start) & (angles <= end)
    else:
        mask = (angles >= start) | (angles <= end)
    values = sector_meters[mask]
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.min(finite))

