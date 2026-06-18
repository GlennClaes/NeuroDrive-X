"""Automatic route generation and route-progress tracking."""

from __future__ import annotations

from dataclasses import dataclass
import heapq
import itertools
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

    def get_upcoming_locations(self, limit: int = 80) -> list[Any]:
        """Return upcoming route locations for BEV/debug rendering."""

        return [waypoint.transform.location for waypoint in self.route[self.current_index : self.current_index + limit]]

    def _trace_route(self, start: Any, end: Any) -> list[Any]:
        assert self._map is not None
        sampling_resolution = float(self.config.get("waypoint_distance", 2.0))
        if str(self.config.get("algorithm", "astar")).lower() == "astar":
            route = self._trace_route_astar(start, end, sampling_resolution)
            if route:
                return route
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

    def _trace_route_astar(self, start: Any, end: Any, sampling_resolution: float) -> list[Any]:
        """Plan a route over the CARLA waypoint graph with A* search."""

        assert self._map is not None
        start_waypoint = self._map.get_waypoint(start, project_to_road=True, lane_type=carla.LaneType.Driving)
        end_waypoint = self._map.get_waypoint(end, project_to_road=True, lane_type=carla.LaneType.Driving)
        if start_waypoint is None or end_waypoint is None:
            return []

        astar_config = self.config.get("astar", {})
        max_expansions = int(astar_config.get("max_expansions", 3000))
        lane_change_penalty = float(astar_config.get("lane_change_penalty", 4.0))
        road_option_penalty = float(astar_config.get("road_option_penalty", 1.0))
        counter = itertools.count()
        start_key = self._waypoint_key(start_waypoint)
        goal_location = end_waypoint.transform.location
        frontier: list[tuple[float, int, Any]] = []
        heapq.heappush(frontier, (0.0, next(counter), start_waypoint))
        came_from: dict[tuple[int, int, int, int], tuple[int, int, int, int] | None] = {start_key: None}
        cost_so_far: dict[tuple[int, int, int, int], float] = {start_key: 0.0}
        waypoint_by_key: dict[tuple[int, int, int, int], Any] = {start_key: start_waypoint}
        best_key = start_key
        best_distance = start_waypoint.transform.location.distance(goal_location)

        for _ in range(max_expansions):
            if not frontier:
                break
            _, _, current = heapq.heappop(frontier)
            current_key = self._waypoint_key(current)
            distance_to_goal = current.transform.location.distance(goal_location)
            if distance_to_goal < best_distance:
                best_distance = distance_to_goal
                best_key = current_key
            if distance_to_goal <= max(8.0, sampling_resolution * 2.5):
                return self._reconstruct_astar_path(came_from, waypoint_by_key, current_key)

            for neighbor, transition_penalty in self._astar_neighbors(
                current,
                sampling_resolution,
                lane_change_penalty,
                road_option_penalty,
            ):
                neighbor_key = self._waypoint_key(neighbor)
                segment_cost = current.transform.location.distance(neighbor.transform.location) + transition_penalty
                new_cost = cost_so_far[current_key] + float(segment_cost)
                if neighbor_key not in cost_so_far or new_cost < cost_so_far[neighbor_key]:
                    cost_so_far[neighbor_key] = new_cost
                    waypoint_by_key[neighbor_key] = neighbor
                    priority = new_cost + neighbor.transform.location.distance(goal_location)
                    heapq.heappush(frontier, (priority, next(counter), neighbor))
                    came_from[neighbor_key] = current_key

        LOGGER.info("A* route did not reach target after %d expansions; using best partial route.", max_expansions)
        return self._reconstruct_astar_path(came_from, waypoint_by_key, best_key)

    def _astar_neighbors(
        self,
        waypoint: Any,
        sampling_resolution: float,
        lane_change_penalty: float,
        road_option_penalty: float,
    ) -> list[tuple[Any, float]]:
        neighbors: list[tuple[Any, float]] = []
        for next_waypoint in waypoint.next(sampling_resolution):
            neighbors.append((next_waypoint, road_option_penalty))

        for lane_getter in (getattr(waypoint, "get_left_lane", None), getattr(waypoint, "get_right_lane", None)):
            if lane_getter is None:
                continue
            try:
                lane_waypoint = lane_getter()
            except RuntimeError:
                lane_waypoint = None
            if lane_waypoint is None:
                continue
            if lane_waypoint.lane_type == carla.LaneType.Driving:
                neighbors.append((lane_waypoint, lane_change_penalty))
        return neighbors

    def _reconstruct_astar_path(
        self,
        came_from: dict[tuple[int, int, int, int], tuple[int, int, int, int] | None],
        waypoint_by_key: dict[tuple[int, int, int, int], Any],
        end_key: tuple[int, int, int, int],
    ) -> list[Any]:
        path_keys: list[tuple[int, int, int, int]] = []
        current: tuple[int, int, int, int] | None = end_key
        while current is not None:
            path_keys.append(current)
            current = came_from.get(current)
        path_keys.reverse()
        return [waypoint_by_key[key] for key in path_keys if key in waypoint_by_key]

    def _waypoint_key(self, waypoint: Any) -> tuple[int, int, int, int]:
        return (
            int(waypoint.road_id),
            int(waypoint.section_id),
            int(waypoint.lane_id),
            int(round(float(waypoint.s) * 10.0)),
        )

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
