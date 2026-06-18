#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${VENV_DIR:-$PROJECT_ROOT/.venv}/bin/activate"

if [[ -f "$PROJECT_ROOT/.env.carla" ]]; then
  source "$PROJECT_ROOT/.env.carla"
fi

python "$PROJECT_ROOT/ai/train.py" \
  --carla-config "$PROJECT_ROOT/configs/carla.yaml" \
  --training-config "$PROJECT_ROOT/configs/training.yaml" \
  "$@"

