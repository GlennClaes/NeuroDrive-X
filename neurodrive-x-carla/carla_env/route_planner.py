"""Automatic route generation and route-progress tracking."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import random
from typing import Any

try:
    import carla
except ImportError:  # pragma: no cover
    carla = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class RouteStatus:
    """Current route state for observations, rewards, and termination."""

    distance_to_waypoint_m: float
    reached_waypoint: bool
    route_completed: bool
    route_completed_pct: float
    lane_center_offset_m: float
    offroad: bool
    wrong_way: bool


class CarlaRoutePlanner:
    """Create and track CARLA map routes for the ego vehicle."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.route: list[Any] = []
        self.current_index = 0
        self.target_transform: Any | None = None
        self._map: Any | None = None

    def generate_route(self, world: Any, start_transform: Any, rng: random.Random | None = None) -> list[Any]:
        """Generate a route from the ego start to a distant spawn point."""

        _require_carla()
        rng = rng or random.Random()
        self._map = world.get_map()
        spawn_points = self._map.get_spawn_points()
        if not spawn_points:
            raise RuntimeError("CARLA map has no spawn points for route generation.")

        target_distance = float(self.config.get("route_length_meters", 900.0))
        candidates = sorted(
            spawn_points,
            key=lambda transform: abs(transform.location.distance(start_transform.location) - target_distance),
        )
        self.target_transform = rng.choice(candidates[: min(10, len(candidates))])
        self.route = self._trace_route(start_transform.location, self.target_transform.location)
        self.current_index = 0
        LOGGER.info("Generated route with %d waypoints.", len(self.route))
        return self.route

    def update(self, vehicle: Any) -> RouteStatus:
        """Advance route index and return route diagnostics."""

        _require_carla()
        if self._map is None or not self.route:
            return RouteStatus(0.0, False, True, 1.0, 0.0, False, False)

        transform = vehicle.get_transform()
        location = transform.location
        target = self.route[min(self.current_index, len(self.route) - 1)]
        distance = float(location.distance(target.transform.location))
        reach_distance = float(self.config.get("target_reach_distance", 5.0))
        reached = False
        while distance < reach_distance and self.current_index < len(self.route) - 1:
            self.current_index += 1
            target = self.route[self.current_index]
            distance = float(location.distance(target.transform.location))
            reached = True

        route_completed = self.current_index >= len(self.route) - 2 and distance < reach_distance
        completed_pct = float(self.current_index / max(len(self.route) - 1, 1))

        map_waypoint = self._map.get_waypoint(location, project_to_road=True, lane_type=carla.LaneType.Driving)
        lane_center_offset = float(location.distance(map_waypoint.transform.location)) if map_waypoint else 99.0
        offroad = lane_center_offset > float(self.config.get("offroad_max_distance", 5.0))
        wrong_way = self._is_wrong_way(transform, map_waypoint)
        return RouteStatus(distance, reached, route_completed, completed_pct, lane_center_offset, offroad, wrong_way)

    def draw_debug_route(self, world: Any, life_time: float = 1.0) -> None:
        """Draw the active route in CARLA's debug renderer."""

        if not self.route:
            return
        for waypoint in self.route[self.current_index : self.current_index + 80]:
            world.debug.draw_point(
                waypoint.transform.location + carla.Location(z=0.4),
                size=0.08,
                color=carla.Color(20, 220, 120),
                life_time=life_time,
            )

    def _trace_route(self, start: Any, end: Any) -> list[Any]:
        assert self._map is not None
        sampling_resolution = float(self.config.get("waypoint_distance", 2.0))
        try:
            from agents.navigation.global_route_planner import GlobalRoutePlanner

            planner = GlobalRoutePlanner(self._map, sampling_resolution)
            traced = planner.trace_route(start, end)
            route = [item[0] for item in traced]
            if route:
                return route
        except Exception:
            LOGGER.info("Falling back to local waypoint route generation.", exc_info=True)

        route: list[Any] = []
        waypoint = self._map.get_waypoint(start, project_to_road=True, lane_type=carla.LaneType.Driving)
        route.append(waypoint)
        max_steps = int(float(self.config.get("route_length_meters", 900.0)) / sampling_resolution)
        for _ in range(max_steps):
            next_waypoints = waypoint.next(sampling_resolution)
            if not next_waypoints:
                break
            waypoint = random.choice(next_waypoints)
            route.append(waypoint)
            if waypoint.transform.location.distance(end) < 25.0:
                break
        return route

    def _is_wrong_way(self, vehicle_transform: Any, waypoint: Any | None) -> bool:
        if waypoint is None:
            return False
        vehicle_forward = vehicle_transform.get_forward_vector()
        lane_forward = waypoint.transform.get_forward_vector()
        dot = vehicle_forward.x * lane_forward.x + vehicle_forward.y * lane_forward.y + vehicle_forward.z * lane_forward.z
        dot = max(min(dot, 1.0), -1.0)
        angle = math.degrees(math.acos(dot))
        return angle > float(self.config.get("wrong_way_angle_degrees", 105.0))


def _require_carla() -> None:
    if carla is None:
        raise RuntimeError("CARLA Python API is required for route planning.")

