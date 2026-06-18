"""Run a trained NeuroDrive X policy live in CARLA."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

from stable_baselines3 import PPO
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from carla_env.environment import CarlaDrivingEnv

LOGGER = logging.getLogger("neurodrive.inference")


def run_inference(args: argparse.Namespace) -> None:
    """Load a PPO model and run live inference in CARLA."""

    training_config = _load_yaml(args.training_config)
    model_path = Path(args.model or training_config.get("evaluation", {}).get("model_path", "ai/models/ppo_neurodrive_x.zip"))
    model_path = model_path if model_path.is_absolute() else PROJECT_ROOT / model_path

    env = CarlaDrivingEnv(
        carla_config_path=_relative_to_project(args.carla_config),
        training_config_path=_relative_to_project(args.training_config),
        town=args.town,
        weather=args.weather,
        headless=args.headless,
    )
    model = PPO.load(model_path, env=env, device=training_config.get("device", "auto"))

    observation, _ = env.reset()
    episode_reward = 0.0
    try:
        for step in range(args.max_steps):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(int(action))
            episode_reward += float(reward)
            LOGGER.info(
                "step=%d action=%s reward=%.2f speed=%.1f km/h distance=%.1fm success=%s",
                step,
                int(action),
                reward,
                float(info.get("speed_kmh", 0.0)),
                float(info.get("distance_driven_m", 0.0)),
                bool(info.get("success", False)),
            )
            if terminated or truncated:
                LOGGER.info("Episode finished reward=%.2f info=%s", episode_reward, info)
                observation, _ = env.reset()
                episode_reward = 0.0
    finally:
        env.close()


def build_parser() -> argparse.ArgumentParser:
    """Build the live inference CLI parser."""

    parser = argparse.ArgumentParser(description="Run a trained NeuroDrive X policy in CARLA.")
    parser.add_argument("--carla-config", type=Path, default=PROJECT_ROOT / "configs/carla.yaml")
    parser.add_argument("--training-config", type=Path, default=PROJECT_ROOT / "configs/training.yaml")
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--town", type=str, default="Town05")
    parser.add_argument("--weather", type=str, default="ClearNoon")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--max-steps", type=int, default=5000)
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    run_inference(args)


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

