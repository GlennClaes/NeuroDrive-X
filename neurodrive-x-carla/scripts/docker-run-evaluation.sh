#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"

export EVAL_ARGS="${EVAL_ARGS:-$*}"
docker compose -f "$COMPOSE_FILE" --profile evaluate up --build --abort-on-container-exit evaluator

