#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$PROJECT_ROOT/.venv}"

echo "[NeuroDrive X] Creating virtual environment at $VENV_DIR"
"$PYTHON_BIN" -m venv "$VENV_DIR"

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$PROJECT_ROOT/requirements.txt"

if [[ -n "${CARLA_PYTHON_EGG:-}" ]]; then
  echo "[NeuroDrive X] CARLA_PYTHON_EGG is set: $CARLA_PYTHON_EGG"
  echo "export PYTHONPATH=\"$CARLA_PYTHON_EGG:\${PYTHONPATH:-}\"" > "$PROJECT_ROOT/.env.carla"
fi

mkdir -p "$PROJECT_ROOT/analytics/training_logs/plots"
mkdir -p "$PROJECT_ROOT/analytics/training_logs/replays"
mkdir -p "$PROJECT_ROOT/ai/models/checkpoints"

echo "[NeuroDrive X] Setup complete."

