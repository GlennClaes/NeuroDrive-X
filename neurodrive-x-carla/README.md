# NeuroDrive X: CARLA Autonomous Driving Lab

NeuroDrive X is a Docker-first autonomous-driving research platform using CARLA Simulator, Python 3.11, Stable-Baselines3 PPO, PyTorch, Gymnasium, Ultralytics YOLO, LiDAR point-cloud processing, FastAPI, and a React + Vite dashboard.

Everything can run through Docker Compose: CARLA, training, evaluation, live demo, backend dashboard, React frontend, and tests.

## What Runs In Docker

| Workflow | Service | Dockerfile |
| --- | --- | --- |
| CARLA simulator | `carla` | `carlasim/carla:latest` |
| PPO training / evaluation / demo / tests | `trainer`, `evaluator`, `demo`, `tests` | `docker/Dockerfile.training` |
| Production dashboard | `dashboard` | `docker/Dockerfile.dashboard` |
| React development dashboard | `frontend-dev` | `docker/Dockerfile.frontend` |
| Default dashboard image | manual `docker build -f docker/Dockerfile` | `docker/Dockerfile` |

## Features

- CARLA towns Town01, Town03, and Town05.
- Ego vehicle, Traffic Manager vehicles, pedestrians, and multi-agent traffic behavior.
- RGB camera, semantic camera, LiDAR, collision, lane invasion, GNSS, and IMU sensors.
- PPO with a custom multi-modal CNN feature extractor for RGB, bird's-eye-view maps, LiDAR, and scalar vehicle state.
- Ultralytics YOLO object detection with CARLA semantic segmentation fallback.
- LiDAR filtering, clustering, sector distances, and BEV occupancy maps.
- A* route planning over CARLA waypoint graphs.
- FastAPI API plus React + Vite + TypeScript dashboard.
- JSONL/CSV metrics, replay logs, training plots, leaderboard, emergency braking, rain mode, night mode, and headless training.

## Project Structure

```text
neurodrive-x-carla/
|-- ai/                 PPO training, evaluation, inference, rewards, policies
|-- carla_env/          Gymnasium environment and CARLA actor managers
|-- perception/         Camera, YOLO, lane, BEV, and LiDAR processing
|-- dashboard/          FastAPI backend and React frontend
|-- analytics/          Metrics logging, plots, leaderboard helpers
|-- configs/            CARLA, training, and dashboard YAML configs
|-- docker/             Dockerfiles, Compose stack, Docker entrypoints
|-- scripts/            Docker-first PowerShell/Bash scripts
`-- tests/              Offline unit tests
```

## Requirements

- Docker Desktop or Docker Engine with Compose V2.
- NVIDIA GPU and current drivers strongly recommended for CARLA.
- On Windows: WSL2-backed Docker Desktop is recommended.
- PowerShell 7+ recommended, Windows PowerShell also works for the provided scripts.

## Docker Quick Start On Windows

From PowerShell:

```powershell
cd C:\Projecten\NeuroDrive-X\neurodrive-x-carla
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\scripts\docker-build-all.ps1
.\scripts\docker-start-simulator.ps1
.\scripts\docker-start-dashboard.ps1 -Build
```

Open:

```text
http://localhost:8080
```

Run training:

```powershell
.\scripts\docker-run-training.ps1 -Timesteps 250000
```

Run evaluation:

```powershell
.\scripts\docker-run-evaluation.ps1 -Model ai/models/ppo_neurodrive_x.zip -Episodes 5
```

Run live demo:

```powershell
.\scripts\docker-run-demo.ps1 -Model ai/models/ppo_neurodrive_x.zip -Town Town05 -Weather ClearNoon
```

Run tests:

```powershell
.\scripts\docker-run-tests.ps1
```

Stop everything:

```powershell
.\scripts\docker-stop-all.ps1
```

## Docker Scripts

PowerShell scripts:

- `scripts/docker-build-all.ps1`
- `scripts/docker-start-simulator.ps1`
- `scripts/docker-start-dashboard.ps1`
- `scripts/docker-run-training.ps1`
- `scripts/docker-run-evaluation.ps1`
- `scripts/docker-run-demo.ps1`
- `scripts/docker-run-tests.ps1`
- `scripts/docker-show-logs.ps1`
- `scripts/docker-stop-all.ps1`

Bash equivalents are also available with the same names and `.sh` extensions.

## Compose Commands

The scripts wrap these Compose commands:

```powershell
docker compose -f docker/docker-compose.yml --profile simulator up -d carla
docker compose -f docker/docker-compose.yml --profile dashboard up -d --build dashboard
docker compose -f docker/docker-compose.yml --profile train up --build --abort-on-container-exit trainer
docker compose -f docker/docker-compose.yml --profile evaluate up --build --abort-on-container-exit evaluator
docker compose -f docker/docker-compose.yml --profile demo up --build --abort-on-container-exit demo
docker compose -f docker/docker-compose.yml --profile test up --build --abort-on-container-exit tests
```

To run the React dashboard in Vite dev mode entirely in Docker:

```powershell
.\scripts\docker-start-dashboard.ps1 -WithFrontendDev -Build
```

Open:

```text
http://localhost:5173
```

## Data Persistence

The Compose stack bind-mounts these host directories:

```text
analytics/training_logs -> /app/analytics/training_logs
ai/models               -> /app/ai/models
```

Training metrics, plots, replay logs, checkpoints, and final PPO models remain available on your host after containers stop.

## CARLA Docker Notes

The simulator service uses:

```text
carlasim/carla:latest
```

It exposes ports `2000-2002`, runs headless with `-RenderOffScreen`, and is reachable inside Compose as:

```text
CARLA_HOST=carla
CARLA_PORT=2000
```

For CARLA source and simulator documentation, see [carla-simulator/carla](https://github.com/carla-simulator/carla).

## Configuration

- `configs/carla.yaml`: server, towns, sensors, perception, traffic, weather, route planning, safety, replay logging.
- `configs/training.yaml`: PPO hyperparameters, custom CNN feature extractor, reward coefficients, logging, evaluation defaults.
- `configs/dashboard.yaml`: dashboard API host/port, metric paths, plot paths, refresh behavior.

## Optional Local Development

Docker is the main workflow. Local scripts remain available for quick debugging:

```powershell
.\scripts\setup.ps1 -InstallFrontend
.\scripts\dashboard.ps1
.\scripts\frontend.ps1 -Install
```

## Troubleshooting

- CARLA does not start: verify Docker Desktop GPU support and NVIDIA drivers.
- Training cannot connect to CARLA: run `.\scripts\docker-show-logs.ps1 carla` and check that port `2000` is ready.
- Docker Compose says GPU support is unavailable: install NVIDIA Container Toolkit on Linux, or enable GPU support in Docker Desktop/WSL2 on Windows.
- YOLO weights download fails: provide a local weights file or let the semantic fallback run; the pipeline remains functional.
- Dashboard is empty: run training/evaluation first, or check `analytics/training_logs`.
- Build is slow: the training image installs PyTorch, SB3, OpenCV, Ultralytics, and CARLA client dependencies.

## Future Improvements

- Vectorized CARLA training across multiple simulator containers.
- TensorRT/ONNX YOLO inference image for faster perception.
- Scenario-based route curricula and leaderboard benchmarks.
- Replay-to-video export with camera, BEV, and reward overlays.
- MLflow or Weights & Biases experiment tracking.

