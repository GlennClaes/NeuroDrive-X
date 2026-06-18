"""FastAPI routes for live training metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException

from analytics.metrics import build_leaderboard, read_metrics, summarize_metrics


def create_api_router(project_root: Path, config: dict[str, Any]) -> APIRouter:
    """Create the dashboard API router with paths resolved from config."""

    router = APIRouter(prefix="/api", tags=["metrics"])
    data_config = config.get("data", {})
    leaderboard_config = config.get("leaderboard", {})
    metrics_jsonl = _resolve(project_root, data_config.get("metrics_jsonl", "analytics/training_logs/metrics.jsonl"))
    dashboard_state = _resolve(project_root, data_config.get("dashboard_state", "analytics/training_logs/dashboard_state.json"))
    plot_dir = _resolve(project_root, data_config.get("plot_dir", "analytics/training_logs/plots"))

    @router.get("/metrics/latest")
    def latest_metrics() -> dict[str, Any]:
        if dashboard_state.exists():
            return json.loads(dashboard_state.read_text(encoding="utf-8"))
        metrics = read_metrics(metrics_jsonl)
        latest = metrics[-1].to_dict() if metrics else None
        return {"latest": latest, "summary": summarize_metrics(metrics)}

    @router.get("/metrics/history")
    def metric_history(limit: int = 200) -> dict[str, Any]:
        metrics = read_metrics(metrics_jsonl)
        rows = [metric.to_dict() for metric in metrics[-max(1, limit) :]]
        return {"items": rows, "summary": summarize_metrics(metrics)}

    @router.get("/leaderboard")
    def leaderboard() -> dict[str, Any]:
        metrics = read_metrics(metrics_jsonl)
        min_episodes = int(leaderboard_config.get("min_episodes", 3))
        return {"items": build_leaderboard(metrics, min_episodes=min_episodes)}

    @router.get("/plots")
    def plots() -> dict[str, Any]:
        if not plot_dir.exists():
            return {"items": []}
        items = [
            {
                "name": path.stem.replace("_", " ").title(),
                "url": f"/plots/{path.name}",
            }
            for path in sorted(plot_dir.glob("*.png"))
        ]
        return {"items": items}

    @router.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return router


def _resolve(project_root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else project_root / candidate

