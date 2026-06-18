import numpy as np

from perception.bird_eye_view import render_bird_eye_view
from perception.lidar_processing import bird_eye_occupancy_grid, cluster_point_cloud, preprocess_lidar


def test_lidar_preprocessing_detects_front_obstacle() -> None:
    points = np.array(
        [
            [4.0, 0.0, 0.2],
            [15.0, 3.0, 0.1],
            [12.0, -5.0, 0.0],
        ],
        dtype=np.float32,
    )

    summary = preprocess_lidar(points, sectors=16, max_range=40.0, obstacle_threshold_m=8.0)

    assert summary.sector_distances.shape == (16,)
    assert summary.front_distance_m is not None
    assert summary.front_distance_m < 8.0
    assert summary.obstacle_detected is True


def test_bird_eye_grid_marks_points() -> None:
    points = np.array([[1.0, 1.0, 0.0], [3.0, -2.0, 0.0]], dtype=np.float32)

    grid = bird_eye_occupancy_grid(points, size=64, meters=20.0)

    assert grid.shape == (64, 64)
    assert grid.max() == 255


def test_point_cloud_clustering_finds_obstacle_groups() -> None:
    points = np.array(
        [
            [4.0, 0.0, 0.0],
            [4.2, 0.1, 0.1],
            [3.9, -0.1, 0.2],
            [4.1, 0.2, 0.0],
            [20.0, 10.0, 0.0],
        ],
        dtype=np.float32,
    )

    clusters = cluster_point_cloud(points, radius_m=0.5, min_points=3)

    assert len(clusters) == 1
    assert clusters[0].point_count == 4


def test_render_bird_eye_view_returns_channel_first_tensor() -> None:
    points = np.array([[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]], dtype=np.float32)

    bev = render_bird_eye_view(points, size=32, meters=20.0)

    assert bev.shape == (3, 32, 32)
    assert bev.max() == 1.0
