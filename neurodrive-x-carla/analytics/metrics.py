"""Training metrics logging and leaderboard support."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import csv
import json
import logging
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

LOGGER = logging.getLogger(__name__)


@dataclass
class EpisodeMetrics:
    """Serializable episode-level metrics emitted by training and evaluation."""

    episode: int
    reward: float
    steps: int
    speed_kmh: float
    collision_count: int
    lane_invasion_count: int
    distance_driven_m: float
    success: bool
    town: str
    weather: str
    detection_count: int = 0
    model_name: str = "ppo_neurodrive_x"
    route_completed_pct: float = 0.0
    average_reward_100: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Return the metric as a JSON/CSV-safe dictionary."""

        return asdict(self)


class TrainingMetricsLogger:
    """Append metrics to JSONL/CSV and keep a dashboard state file fresh."""

    def __init__(
        self,
        jsonl_path: str | Path,
        csv_path: str | Path,
        dashboard_state_path: str | Path | None = None,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.csv_path = Path(csv_path)
        self.dashboard_state_path = Path(dashboard_state_path) if dashboard_state_path else None
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if self.dashboard_state_path:
            self.dashboard_state_path.parent.mkdir(parents=True, exist_ok=True)

    def log_episode(self, metric: EpisodeMetrics) -> None:
        """Append one episode metric and update the live dashboard snapshot."""

        metric.average_reward_100 = self._average_reward_with(metric.reward, window=100)
        payload = metric.to_dict()
        with self.jsonl_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(payload.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(payload)

        if self.dashboard_state_path:
            state = {
                "latest": payload,
                "summary": summarize_metrics(read_metrics(self.jsonl_path)),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
            self.dashboard_state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
        LOGGER.info("Logged episode %s reward=%.2f success=%s", metric.episode, metric.reward, metric.success)

    def _average_reward_with(self, new_reward: float, window: int) -> float:
        recent = [item.reward for item in read_metrics(self.jsonl_path)][-(window - 1) :]
        recent.append(new_reward)
        return float(mean(recent)) if recent else float(new_reward)


def read_metrics(path: str | Path) -> list[EpisodeMetrics]:
    """Read metrics from a JSONL file."""

    metric_path = Path(path)
    if not metric_path.exists():
        return []
    rows: list[EpisodeMetrics] = []
    with metric_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            payload = json.loads(line)
            rows.append(EpisodeMetrics(**payload))
    return rows


def summarize_metrics(metrics: Iterable[EpisodeMetrics]) -> dict[str, Any]:
    """Calculate dashboard-level aggregate metrics."""

    rows = list(metrics)
    if not rows:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "average_reward": 0.0,
            "total_collisions": 0,
            "total_lane_invasions": 0,
            "total_distance_m": 0.0,
        }

    return {
        "episodes": len(rows),
        "success_rate": float(sum(row.success for row in rows) / len(rows)),
        "average_reward": float(mean(row.reward for row in rows)),
        "average_reward_100": float(mean(row.reward for row in rows[-100:])),
        "total_collisions": int(sum(row.collision_count for row in rows)),
        "total_lane_invasions": int(sum(row.lane_invasion_count for row in rows)),
        "total_distance_m": float(sum(row.distance_driven_m for row in rows)),
        "best_reward": float(max(row.reward for row in rows)),
        "latest_episode": rows[-1].episode,
    }


def build_leaderboard(metrics: Iterable[EpisodeMetrics], min_episodes: int = 3) -> list[dict[str, Any]]:
    """Aggregate runs by model/town/weather for a compact leaderboard."""

    grouped: dict[tuple[str, str, str], list[EpisodeMetrics]] = {}
    for metric in metrics:
        grouped.setdefault((metric.model_name, metric.town, metric.weather), []).append(metric)

    leaderboard: list[dict[str, Any]] = []
    for (model_name, town, weather), rows in grouped.items():
        if len(rows) < min_episodes:
            continue
        leaderboard.append(
            {
                "model_name": model_name,
                "town": town,
                "weather": weather,
                "episodes": len(rows),
                "success_rate": float(sum(row.success for row in rows) / len(rows)),
                "average_reward": float(mean(row.reward for row in rows)),
                "average_distance_m": float(mean(row.distance_driven_m for row in rows)),
                "collisions_per_episode": float(mean(row.collision_count for row in rows)),
            }
        )
    leaderboard.sort(key=lambda row: (row["success_rate"], row["average_reward"]), reverse=True)
    return leaderboard
