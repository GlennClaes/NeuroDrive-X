"""Train a PPO driving policy in CARLA."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import random
import sys
from typing import Any, Callable

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analytics.metrics import EpisodeMetrics, TrainingMetricsLogger
from analytics.plots import generate_all_plots
from carla_env.environment import CarlaDrivingEnv

LOGGER = logging.getLogger("neurodrive.train")


class EpisodeMetricsCallback(BaseCallback):
    """Collect finished episodes from SB3 and persist dashboard metrics."""

    def __init__(
        self,
        logger: TrainingMetricsLogger,
        metrics_csv: Path,
        plot_dir: Path,
        plot_frequency_episodes: int = 10,
    ) -> None:
        super().__init__()
        self.metrics_logger = logger
        self.metrics_csv = metrics_csv
        self.plot_dir = plot_dir
        self.plot_frequency_episodes = max(1, plot_frequency_episodes)
        self._episode_rewards: list[float] = [0.0]
        self._episode_steps: list[int] = [0]
        self._logged_episodes = 0

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get("rewards", [0.0]), dtype=np.float32)
        dones = np.asarray(self.locals.get("dones", [False]), dtype=bool)
        infos = self.locals.get("infos", [{}])

        self._ensure_slots(len(rewards))
        for index, reward in enumerate(rewards):
            self._episode_rewards[index] += float(reward)
            self._episode_steps[index] += 1
            if index < len(dones) and dones[index]:
                info: dict[str, Any] = infos[index] if index < len(infos) else {}
                self._logged_episodes += 1
                metric = EpisodeMetrics(
                    episode=int(info.get("episode", self._logged_episodes)),
                    reward=float(self._episode_rewards[index]),
                    steps=int(self._episode_steps[index]),
                    speed_kmh=float(info.get("speed_kmh", 0.0)),
                    collision_count=int(info.get("collision_count", 0)),
                    lane_invasion_count=int(info.get("lane_invasion_count", 0)),
                    distance_driven_m=float(info.get("distance_driven_m", 0.0)),
                    success=bool(info.get("success", False)),
                    town=str(info.get("town", "unknown")),
                    weather=str(info.get("weather", "unknown")),
                    route_completed_pct=float(info.get("route_completed_pct", 0.0)),
                )
                self.metrics_logger.log_episode(metric)
                if self._logged_episodes % self.plot_frequency_episodes == 0:
                    generate_all_plots(self.metrics_csv, self.plot_dir)
                self._episode_rewards[index] = 0.0
                self._episode_steps[index] = 0
        return True

    def _ensure_slots(self, count: int) -> None:
        while len(self._episode_rewards) < count:
            self._episode_rewards.append(0.0)
            self._episode_steps.append(0)


def make_env(
    carla_config_path: Path,
    training_config_path: Path,
    town: str | None,
    weather: str | None,
    headless: bool | None,
) -> Callable[[], Monitor]:
    """Create a callable SB3 environment factory."""

    def _factory() -> Monitor:
        env = CarlaDrivingEnv(
            carla_config_path=carla_config_path,
            training_config_path=training_config_path,
            town=town,
            weather=weather,
            headless=headless,
        )
        return Monitor(env)

    return _factory


def train(args: argparse.Namespace) -> Path:
    """Run PPO training and return the saved model path."""

    training_config = _load_yaml(args.training_config)
    seed = int(training_config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logging_config = training_config.get("logging", {})
    log_dir = _resolve(logging_config.get("log_dir", "analytics/training_logs"))
    tensorboard_dir = _resolve(logging_config.get("tensorboard_dir", "analytics/training_logs/tensorboard"))
    model_dir = _resolve(logging_config.get("model_dir", "ai/models"))
    plot_dir = log_dir / "plots"
    log_dir.mkdir(parents=True, exist_ok=True)
    tensorboard_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    metrics_jsonl = _resolve(logging_config.get("metrics_jsonl", "analytics/training_logs/metrics.jsonl"))
    metrics_csv = _resolve(logging_config.get("metrics_csv", "analytics/training_logs/metrics.csv"))
    dashboard_state = _resolve(logging_config.get("dashboard_state", "analytics/training_logs/dashboard_state.json"))
    metrics_logger = TrainingMetricsLogger(metrics_jsonl, metrics_csv, dashboard_state)

    env = VecMonitor(
        DummyVecEnv(
            [
                make_env(
                    _relative_to_project(args.carla_config),
                    _relative_to_project(args.training_config),
                    args.town,
                    args.weather,
                    args.headless,
                )
            ]
        )
    )

    algorithm_config = training_config.get("algorithm", {})
    model_path = Path(args.resume) if args.resume else None
    if model_path:
        LOGGER.info("Resuming PPO model from %s", model_path)
        model = PPO.load(model_path, env=env, device=training_config.get("device", "auto"))
    else:
        model = PPO(
            policy=algorithm_config.get("policy", "MultiInputPolicy"),
            env=env,
            learning_rate=float(algorithm_config.get("learning_rate", 3e-4)),
            n_steps=int(algorithm_config.get("n_steps", 1024)),
            batch_size=int(algorithm_config.get("batch_size", 128)),
            n_epochs=int(algorithm_config.get("n_epochs", 8)),
            gamma=float(algorithm_config.get("gamma", 0.99)),
            gae_lambda=float(algorithm_config.get("gae_lambda", 0.95)),
            clip_range=float(algorithm_config.get("clip_range", 0.2)),
            ent_coef=float(algorithm_config.get("ent_coef", 0.01)),
            vf_coef=float(algorithm_config.get("vf_coef", 0.5)),
            max_grad_norm=float(algorithm_config.get("max_grad_norm", 0.5)),
            tensorboard_log=str(tensorboard_dir),
            seed=seed,
            device=training_config.get("device", "auto"),
            verbose=1,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=int(logging_config.get("checkpoint_frequency_steps", 25000)),
        save_path=str(model_dir / "checkpoints"),
        name_prefix="ppo_neurodrive_x",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )
    metrics_callback = EpisodeMetricsCallback(
        metrics_logger,
        metrics_csv,
        plot_dir,
        plot_frequency_episodes=int(logging_config.get("plot_frequency_episodes", 10)),
    )

    total_timesteps = int(args.timesteps or training_config.get("total_timesteps", 250000))
    LOGGER.info("Starting PPO training for %d timesteps.", total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=[checkpoint_callback, metrics_callback], progress_bar=True)

    output_path = model_dir / "ppo_neurodrive_x.zip"
    model.save(output_path)
    generate_all_plots(metrics_csv, plot_dir)
    env.close()
    LOGGER.info("Saved PPO model to %s", output_path)
    return output_path


def build_parser() -> argparse.ArgumentParser:
    """Build the training CLI parser."""

    parser = argparse.ArgumentParser(description="Train NeuroDrive X PPO in CARLA.")
    parser.add_argument("--carla-config", type=Path, default=PROJECT_ROOT / "configs/carla.yaml")
    parser.add_argument("--training-config", type=Path, default=PROJECT_ROOT / "configs/training.yaml")
    parser.add_argument("--town", type=str, default=None, help="Force a CARLA town, for example Town03.")
    parser.add_argument("--weather", type=str, default=None, help="Force a weather preset, for example HardRainNoon.")
    parser.add_argument("--timesteps", type=int, default=None, help="Override total timesteps from config.")
    parser.add_argument("--headless", action="store_true", help="Enable CARLA no-rendering mode for training.")
    parser.add_argument("--resume", type=Path, default=None, help="Path to an existing PPO zip model.")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    """CLI entrypoint."""

    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train(args)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _resolve(path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def _relative_to_project(path: Path) -> Path:
    path = path.resolve()
    try:
        return path.relative_to(PROJECT_ROOT)
    except ValueError:
        return path


if __name__ == "__main__":
    main()
