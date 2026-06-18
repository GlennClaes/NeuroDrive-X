#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="$PROJECT_ROOT/docker/docker-compose.yml"

if [[ "${WITH_FRONTEND_DEV:-false}" == "true" ]]; then
  docker compose -f "$COMPOSE_FILE" --profile frontend up -d --build dashboard frontend-dev
  echo "React dev UI: http://localhost:5173"
else
  docker compose -f "$COMPOSE_FILE" --profile dashboard up -d --build dashboard
  echo "Dashboard: http://localhost:8080"
fi

