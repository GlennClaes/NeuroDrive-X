"""Ego vehicle spawning, controls, and vehicle-state helpers."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
from typing import Any

try:
    import carla
except ImportError:  # pragma: no cover - exercised only without CARLA installed.
    carla = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


class CarlaDependencyError(RuntimeError):
    """Raised when the CARLA Python API is required but unavailable."""


@dataclass(frozen=True)
class VehicleControlState:
    """Current ego-control state used for reward smoothness."""

    throttle: float
    brake: float
    steer: float


class VehicleManager:
    """Manage the ego vehicle lifecycle and discrete driving actions."""

    ACTION_ACCELERATE = 0
    ACTION_BRAKE = 1
    ACTION_STEER_LEFT = 2
    ACTION_STEER_RIGHT = 3
    ACTION_KEEP_LANE = 4

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.vehicle: Any | None = None
        self.last_control = VehicleControlState(throttle=0.0, brake=0.0, steer=0.0)

    def spawn_ego_vehicle(self, world: Any, spawn_point: Any | None = None) -> Any:
        """Spawn the ego vehicle at a CARLA spawn point."""

        _require_carla()
        blueprint_library = world.get_blueprint_library()
        blueprint_filter = self.config.get("blueprint_filter", "vehicle.tesla.model3")
        candidates = blueprint_library.filter(blueprint_filter)
        if not candidates:
            raise RuntimeError(f"No CARLA vehicle blueprint matched {blueprint_filter!r}")

        blueprint = candidates[0]
        if blueprint.has_attribute("role_name"):
            blueprint.set_attribute("role_name", self.config.get("role_name", "hero"))
        if blueprint.has_attribute("color"):
            recommended = blueprint.get_attribute("color").recommended_values
            if recommended:
                blueprint.set_attribute("color", recommended[0])

        spawn_points = world.get_map().get_spawn_points()
        if spawn_point is None:
            spawn_point = spawn_points[0] if spawn_points else None
        if spawn_point is None:
            raise RuntimeError("CARLA map does not expose spawn points.")

        actor = world.try_spawn_actor(blueprint, spawn_point)
        if actor is None:
            raise RuntimeError("Failed to spawn ego vehicle. Try another town or spawn point.")
        self.vehicle = actor
        LOGGER.info("Spawned ego vehicle id=%s at %s", actor.id, spawn_point.location)
        return actor

    def apply_discrete_action(self, action: int, emergency_brake: bool = False) -> VehicleControlState:
        """Translate the project action space into a CARLA VehicleControl."""

        _require_carla()
        if self.vehicle is None:
            raise RuntimeError("Cannot apply action before the ego vehicle is spawned.")

        max_steer = float(self.config.get("max_steer", 0.7))
        throttle = float(self.config.get("throttle", 0.55))
        brake = float(self.config.get("brake", 0.85))
        keep_lane_throttle = float(self.config.get("keep_lane_throttle", 0.38))
        steer_increment = float(self.config.get("steer_increment", 0.18))

        steer = self.last_control.steer * 0.65
        next_throttle = 0.0
        next_brake = 0.0

        if emergency_brake:
            steer = self.last_control.steer * 0.4
            next_brake = 1.0
        elif action == self.ACTION_ACCELERATE:
            next_throttle = throttle
        elif action == self.ACTION_BRAKE:
            next_brake = brake
        elif action == self.ACTION_STEER_LEFT:
            next_throttle = keep_lane_throttle
            steer = self.last_control.steer - steer_increment
        elif action == self.ACTION_STEER_RIGHT:
            next_throttle = keep_lane_throttle
            steer = self.last_control.steer + steer_increment
        elif action == self.ACTION_KEEP_LANE:
            next_throttle = keep_lane_throttle
        else:
            raise ValueError(f"Unknown discrete action: {action}")

        steer = _clamp(steer, -max_steer, max_steer)
        control = carla.VehicleControl(throttle=next_throttle, brake=next_brake, steer=steer)
        self.vehicle.apply_control(control)
        self.last_control = VehicleControlState(next_throttle, next_brake, steer)
        return self.last_control

    def get_speed_kmh(self) -> float:
        """Return ego speed in km/h."""

        if self.vehicle is None:
            return 0.0
        velocity = self.vehicle.get_velocity()
        meters_per_second = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)
        return float(meters_per_second * 3.6)

    def get_speed_limit_kmh(self) -> float:
        """Return the current CARLA speed limit when available."""

        if self.vehicle is None:
            return 50.0
        try:
            return float(self.vehicle.get_speed_limit())
        except RuntimeError:
            return 50.0

    def get_location(self) -> Any | None:
        """Return the ego vehicle location."""

        if self.vehicle is None:
            return None
        return self.vehicle.get_location()

    def get_transform(self) -> Any | None:
        """Return the ego vehicle transform."""

        if self.vehicle is None:
            return None
        return self.vehicle.get_transform()

    def get_traffic_light_state(self) -> str:
        """Return Red, Yellow, Green, or None for the current traffic light."""

        if self.vehicle is None:
            return "None"
        state = self.vehicle.get_traffic_light_state()
        return getattr(state, "name", str(state).split(".")[-1])

    def is_at_red_light(self) -> bool:
        """Return whether CARLA reports the ego vehicle at a red traffic light."""

        return self.get_traffic_light_state().lower() == "red"

    def destroy(self) -> None:
        """Destroy the ego actor if it exists."""

        if self.vehicle is not None:
            try:
                self.vehicle.destroy()
                LOGGER.info("Destroyed ego vehicle id=%s", self.vehicle.id)
            except RuntimeError:
                LOGGER.debug("Ego vehicle was already destroyed.", exc_info=True)
        self.vehicle = None
        self.last_control = VehicleControlState(throttle=0.0, brake=0.0, steer=0.0)


def _require_carla() -> None:
    if carla is None:
        raise CarlaDependencyError(
            "The CARLA Python API is not installed. Install the API matching your CARLA server "
            "or set PYTHONPATH/CARLA_PYTHON_EGG before running NeuroDrive X."
        )


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)

