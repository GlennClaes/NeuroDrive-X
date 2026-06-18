"""Traffic vehicles, pedestrians, and CARLA Traffic Manager integration."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import random
from typing import Any

try:
    import carla
except ImportError:  # pragma: no cover
    carla = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass
class TrafficActors:
    """Actors owned by the traffic manager wrapper."""

    vehicles: list[Any] = field(default_factory=list)
    walkers: list[Any] = field(default_factory=list)
    walker_controllers: list[Any] = field(default_factory=list)


class CarlaTrafficManager:
    """Spawn traffic vehicles and pedestrians controlled by CARLA."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.actors = TrafficActors()
        self.traffic_manager: Any | None = None

    def configure(self, client: Any, world: Any, synchronous_mode: bool) -> Any:
        """Configure CARLA Traffic Manager and return the manager instance."""

        _require_carla()
        port = int(self.config.get("traffic_manager_port", 8000))
        manager = client.get_trafficmanager(port)
        manager.set_global_distance_to_leading_vehicle(
            float(self.config.get("global_distance_to_leading_vehicle", 2.8))
        )
        manager.global_percentage_speed_difference(float(self.config.get("speed_difference_percentage", 10.0)))
        manager.set_synchronous_mode(synchronous_mode)
        if self.config.get("hybrid_physics_mode", True):
            manager.set_hybrid_physics_mode(True)
            manager.set_hybrid_physics_radius(float(self.config.get("hybrid_physics_radius", 70.0)))
        world.set_pedestrians_cross_factor(float(self.config.get("percentage_pedestrians_crossing", 0.35)))
        self.traffic_manager = manager
        return manager

    def spawn(self, world: Any, rng: random.Random | None = None) -> TrafficActors:
        """Spawn configured vehicles and pedestrians into the current world."""

        _require_carla()
        rng = rng or random.Random()
        if not self.config.get("enabled", True):
            return self.actors
        self._spawn_vehicles(world, rng)
        self._spawn_pedestrians(world, rng)
        LOGGER.info(
            "Spawned traffic actors: vehicles=%d pedestrians=%d",
            len(self.actors.vehicles),
            len(self.actors.walkers),
        )
        return self.actors

    def destroy(self, client: Any | None = None) -> None:
        """Stop and destroy all traffic actors."""

        if carla is not None and client is not None and self.actors.walker_controllers:
            for controller in self.actors.walker_controllers:
                try:
                    controller.stop()
                except RuntimeError:
                    LOGGER.debug("Walker controller already stopped.", exc_info=True)

        for actor in [*self.actors.walker_controllers, *self.actors.walkers, *self.actors.vehicles]:
            try:
                actor.destroy()
            except RuntimeError:
                LOGGER.debug("Traffic actor already destroyed.", exc_info=True)
        self.actors = TrafficActors()

    def _spawn_vehicles(self, world: Any, rng: random.Random) -> None:
        blueprint_library = world.get_blueprint_library()
        blueprints = blueprint_library.filter("vehicle.*")
        spawn_points = world.get_map().get_spawn_points()
        rng.shuffle(spawn_points)
        max_vehicles = int(self.config.get("vehicles", 35))
        tm_port = int(self.config.get("traffic_manager_port", 8000))

        for spawn_point in spawn_points[:max_vehicles]:
            blueprint = rng.choice(blueprints)
            if blueprint.has_attribute("role_name"):
                blueprint.set_attribute("role_name", "autopilot")
            actor = world.try_spawn_actor(blueprint, spawn_point)
            if actor is None:
                continue
            actor.set_autopilot(True, tm_port)
            self.actors.vehicles.append(actor)

    def _spawn_pedestrians(self, world: Any, rng: random.Random) -> None:
        blueprint_library = world.get_blueprint_library()
        walker_blueprints = blueprint_library.filter("walker.pedestrian.*")
        controller_bp = blueprint_library.find("controller.ai.walker")
        max_walkers = int(self.config.get("pedestrians", 20))
        running_fraction = float(self.config.get("percentage_pedestrians_running", 0.15))

        spawn_locations = []
        for _ in range(max_walkers * 2):
            location = world.get_random_location_from_navigation()
            if location is not None:
                spawn_locations.append(carla.Transform(location))
            if len(spawn_locations) >= max_walkers:
                break

        for transform in spawn_locations:
            walker_bp = rng.choice(walker_blueprints)
            if walker_bp.has_attribute("is_invincible"):
                walker_bp.set_attribute("is_invincible", "false")
            walker = world.try_spawn_actor(walker_bp, transform)
            if walker is None:
                continue
            controller = world.try_spawn_actor(controller_bp, carla.Transform(), attach_to=walker)
            if controller is None:
                walker.destroy()
                continue
            self.actors.walkers.append(walker)
            self.actors.walker_controllers.append(controller)
            controller.start()
            destination = world.get_random_location_from_navigation()
            if destination is not None:
                controller.go_to_location(destination)
            controller.set_max_speed(2.8 if rng.random() < running_fraction else 1.4)


def _require_carla() -> None:
    if carla is None:
        raise RuntimeError("CARLA Python API is required for traffic management.")

