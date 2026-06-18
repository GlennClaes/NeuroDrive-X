#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"

export CARLA_IMAGE_TAG="${CARLA_IMAGE_TAG:-latest}"
export CARLA_QUALITY_LEVEL="${CARLA_QUALITY_LEVEL:-Low}"

docker compose -f "$COMPOSE_FILE" --profile simulator up -d carla

