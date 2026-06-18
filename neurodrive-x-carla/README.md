# NeuroDrive X: CARLA Autonomous Driving Lab

NeuroDrive X is a professional autonomous-driving research platform built around CARLA Simulator, Gymnasium, Stable-Baselines3 PPO, computer vision, analytics, Docker, and a FastAPI dashboard. It trains an ego vehicle to drive through real CARLA towns using RGB camera, semantic camera, LiDAR, collision, lane-invasion, GNSS, and IMU sensors.

## Features

- CARLA-first simulation with Town01, Town03, and Town05 support.
- Ego vehicle spawning plus Traffic Manager vehicles and pedestrian controllers.
- Weather presets including ClearNoon, WetCloudyNoon, HardRainNoon, ClearNight, WetNight, and SoftRainSunset.
- Custom Gymnasium environment with discrete driving actions: accelerate, brake, steer left, steer right, keep lane.
- PPO training with Stable-Baselines3 MultiInputPolicy.
- Dense reward shaping for progress, lane keeping, waypoint completion, safe distance, red-light behavior, smooth driving, and route success.
- Penalties for collisions, offroad driving, lane invasions, red-light violations, wrong-way driving, hard steering, and unexplained idling.
- OpenCV lane detection, camera normalization, LiDAR sector preprocessing, semantic object-detection fallback, and swappable DNN object detector.
- FastAPI dashboard with live reward, episode, speed, collisions, lane invasions, distance, success rate, leaderboard, and plots.
- JSONL/CSV metrics logging, replay step logs, automatic plot generation, headless training, emergency braking, rain mode, night mode, and bird's-eye LiDAR utilities.

## Architecture

```text
neurodrive-x-carla/
├── ai/                 PPO training, evaluation, live inference, rewards
├── carla_env/          Gymnasium environment and CARLA actor managers
├── perception/         Camera, lane, object, and LiDAR processing
├── dashboard/          FastAPI app, API routes, templates, static assets
├── analytics/          Metrics logging, plots, leaderboard
├── configs/            CARLA, training, and dashboard YAML configs
├── scripts/            Setup, CARLA start, training, evaluation, demo scripts
├── docker/             Dockerfile and Compose stack
└── tests/              Unit tests for reward, metrics, and LiDAR logic
```

## Installation

```bash
cd neurodrive-x-carla
chmod +x scripts/*.sh
./scripts/setup.sh
```

CARLA's Python API must match the CARLA server version. If you use an extracted CARLA release instead of the PyPI package, set:

```bash
export CARLA_PYTHON_EGG=/path/to/CARLA/PythonAPI/carla/dist/carla-*.egg
export PYTHONPATH="$CARLA_PYTHON_EGG:$PYTHONPATH"
```

## CARLA Setup

Install CARLA 0.9.15 or a compatible server, then start it:

```bash
export CARLA_ROOT=$HOME/CARLA_0.9.15
./scripts/start_carla.sh
```

For an already running server, configure `configs/carla.yaml`:

```yaml
server:
  host: "localhost"
  port: 2000
```

## Training

```bash
./scripts/train.sh --headless --timesteps 250000
```

Force a town or weather preset:

```bash
./scripts/train.sh --town Town03 --weather HardRainNoon
```

The trained model is saved to `ai/models/ppo_neurodrive_x.zip`. Metrics are written to `analytics/training_logs/metrics.jsonl` and `analytics/training_logs/metrics.csv`.

## Evaluation

```bash
./scripts/evaluate.sh --model ai/models/ppo_neurodrive_x.zip --episodes 5 --town Town05 --weather WetCloudyNoon
```

The evaluation report is saved to `analytics/training_logs/evaluation_report.json`.

## Demo

```bash
./scripts/run_demo.sh --model ai/models/ppo_neurodrive_x.zip --town Town05 --weather ClearNoon
```

Emergency braking is enabled in `configs/carla.yaml` and overrides unsafe policy actions near front obstacles or red lights.

## Dashboard

```bash
source .venv/bin/activate
python dashboard/app.py --host 0.0.0.0 --port 8080
```

Open `http://localhost:8080` to view live metrics, training curves, generated plots, and leaderboard entries.

## Docker Usage

Run the dashboard against a CARLA server on the host:

```bash
cd docker
docker compose up dashboard
```

Run headless training:

```bash
cd docker
docker compose --profile training up training
```

Optionally start a CARLA container:

```bash
cd docker
docker compose --profile simulator up carla
```

## Configuration

- `configs/carla.yaml`: server, towns, synchronous mode, sensors, traffic, weather, route planning, safety, replay logging.
- `configs/training.yaml`: PPO hyperparameters, reward coefficients, logging paths, evaluation defaults.
- `configs/dashboard.yaml`: dashboard host, port, metric paths, refresh rate, leaderboard rules.

## Testing

```bash
cd neurodrive-x-carla
pytest -q
```

The included tests do not require a running CARLA server.

## Troubleshooting

- `ModuleNotFoundError: carla`: install the CARLA Python API matching the server or set `PYTHONPATH` to the CARLA egg.
- `timeout while connecting to CARLA`: verify the server is running and `configs/carla.yaml` points to the correct host/port.
- `failed to spawn ego vehicle`: another actor may occupy the spawn point; retry, lower traffic count, or switch town.
- Low FPS in training: use `--headless`, `no_rendering_mode`, lower traffic count, and `Low` CARLA quality.
- Docker cannot reach CARLA: set `CARLA_HOST=host.docker.internal` on Desktop Docker, or use the host gateway IP on Linux.

## Future Improvements

- Add imitation-learning warm starts from CARLA expert routes.
- Replace the semantic object-detection fallback with an ONNX detector trained on CARLA or BDD100K.
- Add vectorized multi-server CARLA training.
- Add route curriculum learning and scenario-based evaluation.
- Export replay logs to video with overlays.
- Integrate experiment tracking with MLflow or Weights & Biases.

