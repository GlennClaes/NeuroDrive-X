"""FastAPI dashboard backend for the NeuroDrive X React frontend."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys
from typing import Any

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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

    legacy_static_dir = PROJECT_ROOT / "dashboard/static"
    frontend_dist = PROJECT_ROOT / "dashboard/frontend/dist"
    frontend_assets = frontend_dist / "assets"
    plot_dir = PROJECT_ROOT / config.get("data", {}).get("plot_dir", "analytics/training_logs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    if legacy_static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(legacy_static_dir)), name="legacy-static")
    if frontend_assets.exists():
        app.mount("/assets", StaticFiles(directory=str(frontend_assets)), name="frontend-assets")
    app.mount("/plots", StaticFiles(directory=str(plot_dir)), name="plots")
    app.include_router(create_api_router(PROJECT_ROOT, config))

    @app.get("/{path:path}", include_in_schema=False)
    async def react_app(path: str = "") -> FileResponse | JSONResponse:
        """Serve the built React app, or explain how to start the dev UI."""

        index_file = frontend_dist / "index.html"
        requested_file = frontend_dist / path
        if path and requested_file.exists() and requested_file.is_file():
            return FileResponse(requested_file)
        if index_file.exists():
            return FileResponse(index_file)
        return JSONResponse(
            {
                "service": "NeuroDrive X Dashboard API",
                "status": "frontend_not_built",
                "api_docs": "/docs",
                "frontend_dev": "Run scripts\\frontend.ps1, then open http://localhost:5173",
                "frontend_build": "Run npm install and npm run build in dashboard/frontend",
            }
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
