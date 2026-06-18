from pathlib import Path
import shutil

import pytest

from analytics.metrics import EpisodeMetrics, TrainingMetricsLogger, build_leaderboard, read_metrics, summarize_metrics


def test_metrics_logger_writes_jsonl_csv_and_state() -> None:
    tmp_path = Path(__file__).resolve().parent / ".tmp_metrics_logger"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    try:
        tmp_path.mkdir()
    except PermissionError:
        pytest.skip("Current sandbox blocks Python-created temporary directories.")
    jsonl = tmp_path / "metrics.jsonl"
    csv = tmp_path / "metrics.csv"
    state = tmp_path / "dashboard_state.json"

    try:
        logger = TrainingMetricsLogger(jsonl, csv, state)
        logger.log_episode(
            EpisodeMetrics(
                episode=1,
                reward=12.5,
                steps=42,
                speed_kmh=31.0,
                collision_count=0,
                lane_invasion_count=1,
                distance_driven_m=140.0,
                success=True,
                town="Town03",
                weather="ClearNoon",
            )
        )

        metrics = read_metrics(jsonl)
        summary = summarize_metrics(metrics)

        assert csv.exists()
        assert state.exists()
        assert len(metrics) == 1
        assert summary["success_rate"] == 1.0
    finally:
        shutil.rmtree(tmp_path, ignore_errors=True)


def test_leaderboard_groups_by_model_town_weather() -> None:
    metrics = [
        EpisodeMetrics(i, 10.0 + i, 10, 20.0, 0, 0, 100.0, True, "Town01", "ClearNoon")
        for i in range(1, 4)
    ]

    leaderboard = build_leaderboard(metrics, min_episodes=3)

    assert leaderboard[0]["episodes"] == 3
    assert leaderboard[0]["success_rate"] == 1.0
