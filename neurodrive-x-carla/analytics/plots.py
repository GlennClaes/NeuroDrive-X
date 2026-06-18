"""Plot generation for NeuroDrive X training analytics."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

LOGGER = logging.getLogger(__name__)


def generate_all_plots(metrics_csv: str | Path, output_dir: str | Path) -> list[Path]:
    """Generate standard training plots and return their paths."""

    csv_path = Path(metrics_csv)
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        LOGGER.info("No metrics found at %s; skipping plot generation.", csv_path)
        return []

    df = pd.read_csv(csv_path)
    if df.empty:
        return []

    outputs = [
        _plot_line(df, "episode", "reward", plot_dir / "reward_per_episode.png", "Reward per Episode"),
        _plot_line_with_average(df, plot_dir / "average_reward.png"),
        _plot_line(df, "episode", "collision_count", plot_dir / "collision_count.png", "Collisions per Episode"),
        _plot_line(df, "episode", "distance_driven_m", plot_dir / "distance_driven.png", "Distance Driven"),
        _plot_success_rate(df, plot_dir / "success_rate.png"),
    ]
    return [path for path in outputs if path is not None]


def _plot_line(df: pd.DataFrame, x_column: str, y_column: str, output_path: Path, title: str) -> Path | None:
    if x_column not in df or y_column not in df:
        return None
    plt.figure(figsize=(10, 5))
    plt.plot(df[x_column], df[y_column], color="#1f77b4", linewidth=1.8)
    plt.title(title)
    plt.xlabel(x_column.replace("_", " ").title())
    plt.ylabel(y_column.replace("_", " ").title())
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()
    return output_path


def _plot_line_with_average(df: pd.DataFrame, output_path: Path) -> Path | None:
    if "episode" not in df or "reward" not in df:
        return None
    window = min(25, max(1, len(df)))
    rolling = df["reward"].rolling(window=window, min_periods=1).mean()
    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df["reward"], color="#9ecae1", linewidth=1.0, label="Reward")
    plt.plot(df["episode"], rolling, color="#d62728", linewidth=2.2, label=f"{window}-episode average")
    plt.title("Average Reward")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()
    return output_path


def _plot_success_rate(df: pd.DataFrame, output_path: Path) -> Path | None:
    if "episode" not in df or "success" not in df:
        return None
    success = df["success"].astype(float).rolling(window=min(25, max(1, len(df))), min_periods=1).mean()
    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], success, color="#2ca02c", linewidth=2.2)
    plt.ylim(0.0, 1.05)
    plt.title("Success Rate")
    plt.xlabel("Episode")
    plt.ylabel("Rolling Success Rate")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(output_path, dpi=140)
    plt.close()
    return output_path

