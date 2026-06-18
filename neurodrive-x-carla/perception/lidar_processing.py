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


@dataclass(frozen=True)
class LidarCluster:
    """A compact obstacle cluster in ego-vehicle coordinates."""

    centroid_xyz: tuple[float, float, float]
    point_count: int
    min_distance_m: float
    width_m: float
    length_m: float


@dataclass(frozen=True)
class PointCloudFeatures:
    """Research-friendly LiDAR features beyond radial sectors."""

    filtered_points: np.ndarray
    clusters: list[LidarCluster]
    occupancy_grid: np.ndarray


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


def extract_point_cloud_features(
    points: np.ndarray | Iterable[Iterable[float]] | None,
    grid_size: int = 128,
    meters: float = 60.0,
    ground_z_threshold: float = -1.8,
    cluster_radius_m: float = 1.5,
    min_cluster_points: int = 4,
) -> PointCloudFeatures:
    """Filter LiDAR points, cluster obstacles, and render a BEV occupancy grid."""

    filtered = filter_ground_points(points, ground_z_threshold=ground_z_threshold)
    clusters = cluster_point_cloud(filtered, radius_m=cluster_radius_m, min_points=min_cluster_points)
    occupancy = bird_eye_occupancy_grid(filtered, size=grid_size, meters=meters)
    return PointCloudFeatures(filtered, clusters, occupancy)


def filter_ground_points(
    points: np.ndarray | Iterable[Iterable[float]] | None,
    ground_z_threshold: float = -1.8,
    max_height_m: float = 3.5,
) -> np.ndarray:
    """Remove ground returns and high outliers from raw CARLA LiDAR points."""

    if points is None:
        return np.empty((0, 3), dtype=np.float32)
    array = np.asarray(points, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 3:
        raise ValueError("Expected LiDAR points with shape (N, 3+)").with_traceback(None)
    xyz = array[:, :3]
    mask = (xyz[:, 2] > ground_z_threshold) & (xyz[:, 2] < max_height_m)
    return xyz[mask]


def cluster_point_cloud(
    points: np.ndarray,
    radius_m: float = 1.5,
    min_points: int = 4,
    max_clusters: int = 32,
) -> list[LidarCluster]:
    """Cluster point-cloud obstacles with a lightweight radius-growing method."""

    if points.size == 0:
        return []
    xy = points[:, :2]
    visited = np.zeros(points.shape[0], dtype=bool)
    clusters: list[LidarCluster] = []

    for start in range(points.shape[0]):
        if visited[start]:
            continue
        queue = [start]
        visited[start] = True
        members: list[int] = []
        while queue:
            index = queue.pop()
            members.append(index)
            distances = np.linalg.norm(xy - xy[index], axis=1)
            neighbors = np.where((distances <= radius_m) & (~visited))[0]
            for neighbor in neighbors.tolist():
                visited[neighbor] = True
                queue.append(neighbor)

        if len(members) < min_points:
            continue
        cluster_points = points[members]
        centroid = tuple(float(value) for value in np.mean(cluster_points, axis=0))
        min_distance = float(np.min(np.linalg.norm(cluster_points[:, :2], axis=1)))
        width = float(np.ptp(cluster_points[:, 1]))
        length = float(np.ptp(cluster_points[:, 0]))
        clusters.append(LidarCluster(centroid, len(members), min_distance, width, length))
        if len(clusters) >= max_clusters:
            break

    clusters.sort(key=lambda item: item.min_distance_m)
    return clusters


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
