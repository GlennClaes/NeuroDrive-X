#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"

export DEMO_ARGS="${DEMO_ARGS:-$*}"
docker compose -f "$COMPOSE_FILE" --profile demo up --build --abort-on-container-exit demo

