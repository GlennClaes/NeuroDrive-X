"""Wait until a CARLA server accepts TCP connections."""

from __future__ import annotations

import os
import socket
import sys
import time


def main() -> int:
    host = os.getenv("CARLA_HOST", "carla")
    port = int(os.getenv("CARLA_PORT", "2000"))
    timeout_seconds = int(os.getenv("CARLA_WAIT_TIMEOUT", "180"))
    deadline = time.monotonic() + timeout_seconds
    print(f"[NeuroDrive X] Waiting for CARLA at {host}:{port} for up to {timeout_seconds}s", flush=True)

    while time.monotonic() < deadline:
        try:
            with socket.create_connection((host, port), timeout=3.0):
                print("[NeuroDrive X] CARLA TCP endpoint is reachable.", flush=True)
                return 0
        except OSError:
            time.sleep(2.0)

    print(f"[NeuroDrive X] Timed out waiting for CARLA at {host}:{port}", file=sys.stderr, flush=True)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

