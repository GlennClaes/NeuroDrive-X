"""FastAPI dashboard application for NeuroDrive X."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.api import create_api_router

LOGGER = logging.getLogger("neurodrive.dashboard")


def create_app(config_path: str | Path = PROJECT_ROOT / "configs/dashboard.yaml") -> FastAPI:
    """Create and configure the dashboard FastAPI app."""

    config = _load_yaml(Path(config_path))
    app = FastAPI(title="NeuroDrive X Dashboard", version="1.0.0")

    static_dir = PROJECT_ROOT / "dashboard/static"
    template_dir = PROJECT_ROOT / "dashboard/templates"
    plot_dir = PROJECT_ROOT / config.get("data", {}).get("plot_dir", "analytics/training_logs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")
    app.mount("/plots", StaticFiles(directory=str(plot_dir)), name="plots")
    app.include_router(create_api_router(PROJECT_ROOT, config))
    templates = Jinja2Templates(directory=str(template_dir))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request) -> HTMLResponse:
        return templates.TemplateResponse(
            "index.html",
            {
                "request": request,
                "refresh_seconds": int(config.get("data", {}).get("refresh_seconds", 2)),
            },
        )

    return app


app = create_app()


def build_parser() -> argparse.ArgumentParser:
    """Build the dashboard CLI parser."""

    parser = argparse.ArgumentParser(description="Run the NeuroDrive X FastAPI dashboard.")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/dashboard.yaml")
    parser.add_argument("--host", type=str, default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--reload", action="store_true")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = _load_yaml(args.config)
    server_config = config.get("server", {})
    host = args.host or server_config.get("host", "0.0.0.0")
    port = int(args.port or server_config.get("port", 8080))
    reload = bool(args.reload or server_config.get("reload", False))
    LOGGER.info("Starting dashboard at http://%s:%d", host, port)
    uvicorn.run("dashboard.app:app", host=host, port=port, reload=reload)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


if __name__ == "__main__":
    main()

