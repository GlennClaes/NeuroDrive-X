"""Evaluate a trained PPO driving policy in CARLA."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import sys
from typing import Any

from stable_baselines3 import PPO
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics.metrics import EpisodeMetrics, summarize_metrics
from analytics.plots import generate_all_plots
from carla_env.environment import CarlaDrivingEnv

LOGGER = logging.getLogger("neurodrive.evaluate")


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    """Run evaluation episodes and write a JSON report."""

    training_config = _load_yaml(args.training_config)
    evaluation_config = training_config.get("evaluation", {})
    model_path = Path(args.model or evaluation_config.get("model_path", "ai/models/ppo_neurodrive_x.zip"))
    model_path = model_path if model_path.is_absolute() else PROJECT_ROOT / model_path
    report_path = Path(args.report or evaluation_config.get("report_path", "analytics/training_logs/evaluation_report.json"))
    report_path = report_path if report_path.is_absolute() else PROJECT_ROOT / report_path
    report_path.parent.mkdir(parents=True, exist_ok=True)

    env = CarlaDrivingEnv(
        carla_config_path=_relative_to_project(args.carla_config),
        training_config_path=_relative_to_project(args.training_config),
        town=args.town,
        weather=args.weather,
        headless=args.headless,
    )
    model = PPO.load(model_path, env=env, device=training_config.get("device", "auto"))
    episodes = int(args.episodes or evaluation_config.get("episodes", 5))
    deterministic = bool(evaluation_config.get("deterministic", True))

    metrics: list[EpisodeMetrics] = []
    for episode_index in range(1, episodes + 1):
        observation, _ = env.reset()
        done = False
        episode_reward = 0.0
        steps = 0
        info: dict[str, Any] = {}
        while not done:
            action, _ = model.predict(observation, deterministic=deterministic)
            observation, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)
            steps += 1
            done = terminated or truncated

        metric = EpisodeMetrics(
            episode=episode_index,
            reward=episode_reward,
            steps=steps,
            speed_kmh=float(info.get("speed_kmh", 0.0)),
            collision_count=int(info.get("collision_count", 0)),
            lane_invasion_count=int(info.get("lane_invasion_count", 0)),
            distance_driven_m=float(info.get("distance_driven_m", 0.0)),
            success=bool(info.get("success", False)),
            town=str(info.get("town", args.town or "unknown")),
            weather=str(info.get("weather", args.weather or "unknown")),
            route_completed_pct=float(info.get("route_completed_pct", 0.0)),
        )
        metrics.append(metric)
        LOGGER.info("Evaluation episode %d reward=%.2f success=%s", episode_index, episode_reward, metric.success)

    env.close()
    report = {
        "model_path": str(model_path),
        "episodes": [metric.to_dict() for metric in metrics],
        "summary": summarize_metrics(metrics),
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    generate_all_plots(PROJECT_ROOT / "analytics/training_logs/metrics.csv", PROJECT_ROOT / "analytics/training_logs/plots")
    LOGGER.info("Wrote evaluation report to %s", report_path)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the evaluation CLI parser."""

    parser = argparse.ArgumentParser(description="Evaluate a NeuroDrive X PPO model in CARLA.")
    parser.add_argument("--carla-config", type=Path, default=PROJECT_ROOT / "configs/carla.yaml")
    parser.add_argument("--training-config", type=Path, default=PROJECT_ROOT / "configs/training.yaml")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    parser.add_argument("--town", type=str, default=None)
    parser.add_argument("--weather", type=str, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    evaluate(args)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _relative_to_project(path: Path) -> Path:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


if __name__ == "__main__":
    main()

