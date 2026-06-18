#!/usr/bin/env bash
set -euo pipefail

CARLA_ROOT="${CARLA_ROOT:-$HOME/CARLA_0.9.15}"
RPC_PORT="${CARLA_PORT:-2000}"
QUALITY_LEVEL="${CARLA_QUALITY_LEVEL:-Low}"
HEADLESS="${HEADLESS:-true}"
CARLA_BIN="$CARLA_ROOT/CarlaUE4.sh"

if [[ ! -x "$CARLA_BIN" ]]; then
  echo "CARLA executable not found at $CARLA_BIN"
  echo "Set CARLA_ROOT to your CARLA installation directory."
  exit 1
fi

ARGS=(
  "-carla-rpc-port=$RPC_PORT"
  "-quality-level=$QUALITY_LEVEL"
)

if [[ "$HEADLESS" == "true" ]]; then
  ARGS+=("-RenderOffScreen")
fi

echo "[NeuroDrive X] Starting CARLA on RPC port $RPC_PORT"
"$CARLA_BIN" "${ARGS[@]}"

