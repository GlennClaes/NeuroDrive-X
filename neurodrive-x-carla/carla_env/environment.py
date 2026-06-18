"""Gymnasium environment wrapping CARLA for autonomous-driving research."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import random
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import yaml

from ai.rewards import RewardConfig, RewardInput, compute_reward
from carla_env.route_planner import CarlaRoutePlanner, RouteStatus
from carla_env.sensor_manager import SensorManager
from carla_env.traffic_manager import CarlaTrafficManager
from carla_env.vehicle_manager import VehicleManager
from carla_env.weather_manager import WeatherManager
from perception.camera_processing import resize_and_normalize_rgb
from perception.bird_eye_view import render_bird_eye_view
from perception.lidar_processing import LidarSummary, preprocess_lidar
from perception.object_detection import Detection, ObjectDetector

try:
    import carla
except ImportError:  # pragma: no cover
    carla = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepDiagnostics:
    """Per-step diagnostics used by rewards, replay logs, and dashboard info."""

    speed_kmh: float
    distance_delta_m: float
    front_obstacle_distance_m: float | None
    emergency_brake: bool
    red_light_state: str
    route: RouteStatus


class CarlaDrivingEnv(gym.Env):
    """A custom Gymnasium environment for learning to drive in CARLA.

    Observation dictionary:
    - camera_features: normalized RGB tensor, shape (3, H, W)
    - lidar_distances: normalized radial LiDAR sector distances
    - vehicle_speed: current speed in km/h
    - steering_angle: previous steering command
    - lane_invasion: one-step lane invasion flag
    - collision: one-step collision flag
    - distance_to_waypoint: meters to the active route waypoint
    - traffic_light_state: one-hot Red, Yellow, Green, None
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        carla_config_path: str | Path = "configs/carla.yaml",
        training_config_path: str | Path = "configs/training.yaml",
        town: str | None = None,
        weather: str | None = None,
        headless: bool | None = None,
        connect_on_init: bool = False,
    ) -> None:
        super().__init__()
        self.project_root = Path(__file__).resolve().parents[1]
        self.carla_config = _load_yaml(self.project_root / carla_config_path)
        self.training_config = _load_yaml(self.project_root / training_config_path)
        self.town_override = town
        self.weather_override = weather
        self.headless_override = headless
        self.rng = random.Random(int(self.training_config.get("seed", 42)))

        lidar_sectors = int(self.carla_config["sensors"]["lidar"].get("sectors", 64))
        camera = self.carla_config["sensors"]["camera"]
        camera_shape = (3, int(camera.get("output_height", 84)), int(camera.get("output_width", 84)))
        bev_config = self.carla_config.get("perception", {}).get("bird_eye_view", {})
        bev_size = int(bev_config.get("size", 128))
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict(
            {
                "camera_features": spaces.Box(0.0, 1.0, shape=camera_shape, dtype=np.float32),
                "bev_map": spaces.Box(0.0, 1.0, shape=(3, bev_size, bev_size), dtype=np.float32),
                "lidar_distances": spaces.Box(0.0, 1.0, shape=(lidar_sectors,), dtype=np.float32),
                "vehicle_speed": spaces.Box(0.0, 240.0, shape=(1,), dtype=np.float32),
                "steering_angle": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
                "lane_invasion": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "collision": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
                "distance_to_waypoint": spaces.Box(0.0, 2000.0, shape=(1,), dtype=np.float32),
                "traffic_light_state": spaces.Box(0.0, 1.0, shape=(4,), dtype=np.float32),
            }
        )

        self.client: Any | None = None
        self.world: Any | None = None
        self.vehicle_manager = VehicleManager(self.carla_config["vehicle"])
        self.sensor_manager = SensorManager(self.carla_config["sensors"])
        self.route_planner = CarlaRoutePlanner(self.carla_config["route_planning"])
        self.traffic_manager = CarlaTrafficManager(self.carla_config["traffic"])
        self.weather_manager = WeatherManager(self.carla_config["weather"])
        self.reward_config = RewardConfig.from_mapping(self.training_config.get("reward", {}))
        detection_config = self.carla_config.get("perception", {}).get("object_detection", {})
        self.object_detector = ObjectDetector.from_config(detection_config)
        self.detection_interval = max(1, int(detection_config.get("inference_interval_steps", 5)))
        self.last_detections: list[Detection] = []

        self.episode = 0
        self.steps = 0
        self.distance_driven_m = 0.0
        self.last_location: Any | None = None
        self.current_town = self.town_override or self.carla_config["simulation"].get("default_town", "Town03")
        self.current_weather = self.weather_override or self.carla_config["weather"].get("default", "ClearNoon")
        self._last_lidar_summary = preprocess_lidar(None, sectors=lidar_sectors)
        self._last_route_status = RouteStatus(0.0, False, False, 0.0, 0.0, False, False)
        self._replay_handle: Any | None = None

        if connect_on_init:
            self._connect()

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Reset CARLA actors, route, weather, and episode counters."""

        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        if self.client is None or self.world is None:
            self._connect()

        self._cleanup_actors()
        self.episode += 1
        self.steps = 0
        self.distance_driven_m = 0.0
        self.sensor_manager.reset_events()

        self.current_town = self._select_town(options)
        self.current_weather = self._select_weather(options)
        self._load_world_if_needed(self.current_town)
        self._configure_world()
        self.current_weather = self.weather_manager.apply(self.world, self.current_weather)

        spawn_point = self._choose_spawn_point()
        vehicle = self.vehicle_manager.spawn_ego_vehicle(self.world, spawn_point)
        self.traffic_manager.configure(self.client, self.world, self.carla_config["simulation"].get("synchronous_mode", True))
        self.traffic_manager.spawn(self.world, self.rng)
        self.sensor_manager.spawn_sensors(self.world, vehicle)
        self.route_planner.generate_route(self.world, vehicle.get_transform(), self.rng)

        self.last_location = vehicle.get_location()
        self._open_replay_log()
        for _ in range(8):
            self._tick_world()

        observation = self._build_observation()
        info = self._build_info(0.0, {}, False)
        LOGGER.info("Started episode=%d town=%s weather=%s", self.episode, self.current_town, self.current_weather)
        return observation, info

    def step(self, action: int) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """Apply a discrete action and advance the CARLA simulation."""

        if self.world is None or self.vehicle_manager.vehicle is None:
            raise RuntimeError("Environment must be reset before step().")

        self.steps += 1
        pre_lidar = preprocess_lidar(
            self.sensor_manager.snapshot.lidar_points,
            sectors=int(self.carla_config["sensors"]["lidar"].get("sectors", 64)),
            max_range=float(self.carla_config["sensors"]["lidar"].get("range", 60.0)),
            obstacle_threshold_m=float(self.carla_config["safety"].get("front_obstacle_distance_m", 8.0)),
        )
        emergency_brake = self._should_emergency_brake(pre_lidar)
        previous_steer = self.vehicle_manager.last_control.steer
        control = self.vehicle_manager.apply_discrete_action(int(action), emergency_brake=emergency_brake)

        action_repeat = int(self.training_config.get("environment", {}).get("action_repeat", 2))
        for _ in range(max(1, action_repeat)):
            self._tick_world()

        collision, lane_invasion = self.sensor_manager.consume_step_events()
        diagnostics = self._collect_diagnostics(pre_lidar, emergency_brake)
        reward_input = self._build_reward_input(diagnostics, collision, lane_invasion, control.steer - previous_steer)
        reward_breakdown = compute_reward(reward_input, self.reward_config)

        terminated = bool(collision or diagnostics.route.offroad or diagnostics.route.route_completed)
        max_steps = int(self.carla_config["simulation"].get("max_episode_steps", 1200))
        truncated = self.steps >= max_steps
        observation = self._build_observation(collision=collision, lane_invasion=lane_invasion)
        info = self._build_info(reward_breakdown.total, reward_breakdown.to_dict(), terminated or truncated)
        self._write_replay_step(action, diagnostics, reward_breakdown.total, terminated, truncated)

        return observation, float(reward_breakdown.total), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        """Return the latest RGB frame for human/debug rendering."""

        return self.sensor_manager.snapshot.rgb

    def close(self) -> None:
        """Close replay logs and destroy all CARLA actors owned by the environment."""

        self._cleanup_actors()
        if self._replay_handle:
            self._replay_handle.close()
            self._replay_handle = None

    def _connect(self) -> None:
        if carla is None:
            raise RuntimeError(
                "CARLA Python API is unavailable. Install a CARLA Python wheel/egg that matches your server."
            )
        server = self.carla_config["server"]
        host = os.getenv("CARLA_HOST", server.get("host", "localhost"))
        port = int(os.getenv("CARLA_PORT", str(server.get("port", 2000))))
        self.client = carla.Client(host, port)
        self.client.set_timeout(float(server.get("timeout_seconds", 20.0)))
        self.world = self.client.get_world()
        LOGGER.info("Connected to CARLA at %s:%s", host, port)

    def _load_world_if_needed(self, town: str) -> None:
        assert self.client is not None
        if self.world is None or self.world.get_map().name.split("/")[-1] != town:
            LOGGER.info("Loading CARLA town %s", town)
            self.world = self.client.load_world(town)

    def _configure_world(self) -> None:
        assert self.world is not None
        settings = self.world.get_settings()
        simulation = self.carla_config["simulation"]
        settings.synchronous_mode = bool(simulation.get("synchronous_mode", True))
        settings.fixed_delta_seconds = float(simulation.get("fixed_delta_seconds", 0.05))
        headless = self.headless_override
        if headless is None:
            headless = bool(simulation.get("headless", False) or self.training_config.get("environment", {}).get("headless", False))
        settings.no_rendering_mode = bool(headless or simulation.get("no_rendering_mode", False))
        self.world.apply_settings(settings)

    def _cleanup_actors(self) -> None:
        if self.client is not None:
            self.traffic_manager.destroy(self.client)
        self.sensor_manager.destroy()
        self.vehicle_manager.destroy()
        if self._replay_handle:
            self._replay_handle.close()
            self._replay_handle = None

    def _choose_spawn_point(self) -> Any:
        assert self.world is not None
        spawn_points = self.world.get_map().get_spawn_points()
        if not spawn_points:
            raise RuntimeError("No CARLA spawn points available.")
        retry_count = int(self.carla_config["simulation"].get("spawn_retry_count", 30))
        return self.rng.choice(spawn_points[: max(1, min(len(spawn_points), retry_count))])

    def _tick_world(self) -> None:
        assert self.world is not None
        if self.carla_config["simulation"].get("synchronous_mode", True):
            self.world.tick()
        else:
            self.world.wait_for_tick()

    def _select_town(self, options: dict[str, Any] | None) -> str:
        if options and options.get("town"):
            return str(options["town"])
        if self.town_override:
            return self.town_override
        if self.training_config.get("environment", {}).get("randomize_town", True):
            return self.rng.choice(list(self.carla_config["simulation"].get("town_sequence", ["Town03"])))
        return str(self.carla_config["simulation"].get("default_town", "Town03"))

    def _select_weather(self, options: dict[str, Any] | None) -> str:
        if options and options.get("weather"):
            return str(options["weather"])
        if self.weather_override:
            return self.weather_override
        if self.training_config.get("environment", {}).get("randomize_weather", True):
            return self.rng.choice(list(self.carla_config["weather"].get("presets", ["ClearNoon"])))
        return str(self.carla_config["weather"].get("default", "ClearNoon"))

    def _build_observation(self, collision: bool = False, lane_invasion: bool = False) -> dict[str, np.ndarray]:
        camera_config = self.carla_config["sensors"]["camera"]
        output_size = (int(camera_config.get("output_width", 84)), int(camera_config.get("output_height", 84)))
        camera_features = resize_and_normalize_rgb(self.sensor_manager.snapshot.rgb, output_size=output_size)

        self._last_lidar_summary = preprocess_lidar(
            self.sensor_manager.snapshot.lidar_points,
            sectors=int(self.carla_config["sensors"]["lidar"].get("sectors", 64)),
            max_range=float(self.carla_config["sensors"]["lidar"].get("range", 60.0)),
            obstacle_threshold_m=float(self.carla_config["safety"].get("front_obstacle_distance_m", 8.0)),
        )

        speed = self.vehicle_manager.get_speed_kmh()
        traffic_state = self._traffic_light_one_hot(self.vehicle_manager.get_traffic_light_state())
        distance = self._last_route_status.distance_to_waypoint_m
        return {
            "camera_features": camera_features.astype(np.float32),
            "bev_map": self._build_bev_map(),
            "lidar_distances": self._last_lidar_summary.sector_distances.astype(np.float32),
            "vehicle_speed": np.array([speed], dtype=np.float32),
            "steering_angle": np.array([self.vehicle_manager.last_control.steer], dtype=np.float32),
            "lane_invasion": np.array([1.0 if lane_invasion else 0.0], dtype=np.float32),
            "collision": np.array([1.0 if collision else 0.0], dtype=np.float32),
            "distance_to_waypoint": np.array([distance], dtype=np.float32),
            "traffic_light_state": traffic_state,
        }

    def _collect_diagnostics(self, lidar: LidarSummary, emergency_brake: bool) -> StepDiagnostics:
        assert self.vehicle_manager.vehicle is not None
        route_status = self.route_planner.update(self.vehicle_manager.vehicle)
        self._last_route_status = route_status

        location = self.vehicle_manager.get_location()
        distance_delta = 0.0
        if location is not None and self.last_location is not None:
            distance_delta = float(location.distance(self.last_location))
            self.distance_driven_m += max(0.0, distance_delta)
            self.last_location = location

        if self.carla_config.get("debug", {}).get("draw_route", True):
            self.route_planner.draw_debug_route(self.world, life_time=0.2)
        self._update_object_detections()

        return StepDiagnostics(
            speed_kmh=self.vehicle_manager.get_speed_kmh(),
            distance_delta_m=distance_delta,
            front_obstacle_distance_m=lidar.front_distance_m,
            emergency_brake=emergency_brake,
            red_light_state=self.vehicle_manager.get_traffic_light_state(),
            route=route_status,
        )

    def _build_reward_input(
        self,
        diagnostics: StepDiagnostics,
        collision: bool,
        lane_invasion: bool,
        steering_delta: float,
    ) -> RewardInput:
        speed = diagnostics.speed_kmh
        red = diagnostics.red_light_state.lower() == "red"
        obstacle_close = diagnostics.front_obstacle_distance_m is not None and diagnostics.front_obstacle_distance_m < 6.0
        idle = speed < float(self.carla_config["safety"].get("minimum_moving_speed_kmh", 1.0)) and not red and not obstacle_close
        return RewardInput(
            speed_kmh=speed,
            speed_limit_kmh=self.vehicle_manager.get_speed_limit_kmh(),
            lane_center_offset_m=diagnostics.route.lane_center_offset_m,
            reached_waypoint=diagnostics.route.reached_waypoint,
            front_obstacle_distance_m=diagnostics.front_obstacle_distance_m,
            stopped_for_red_light=red and speed < 1.0,
            ran_red_light=red and speed > 8.0 and not diagnostics.emergency_brake,
            collision=collision,
            offroad=diagnostics.route.offroad,
            lane_invasion=lane_invasion,
            wrong_way=diagnostics.route.wrong_way,
            steering=self.vehicle_manager.last_control.steer,
            steering_delta=steering_delta,
            idle=idle,
            route_completed=diagnostics.route.route_completed,
            distance_delta_m=diagnostics.distance_delta_m,
        )

    def _should_emergency_brake(self, lidar: LidarSummary) -> bool:
        if not self.carla_config.get("safety", {}).get("emergency_braking", True):
            return False
        if lidar.front_distance_m is not None and lidar.front_distance_m < float(
            self.carla_config["safety"].get("front_obstacle_distance_m", 8.0)
        ):
            return True
        return self.vehicle_manager.is_at_red_light()

    def _build_info(self, reward: float, reward_terms: dict[str, float], done: bool) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "step": self.steps,
            "reward": reward,
            "reward_terms": reward_terms,
            "speed_kmh": self.vehicle_manager.get_speed_kmh(),
            "collision_count": self.sensor_manager.snapshot.collision_count,
            "lane_invasion_count": self.sensor_manager.snapshot.lane_invasion_count,
            "distance_driven_m": self.distance_driven_m,
            "route_completed_pct": self._last_route_status.route_completed_pct,
            "success": bool(self._last_route_status.route_completed),
            "detections": [detection.__dict__ for detection in self.last_detections[:20]],
            "detection_count": len(self.last_detections),
            "town": self.current_town,
            "weather": self.current_weather,
            "done": done,
        }

    def _traffic_light_one_hot(self, state: str) -> np.ndarray:
        normalized = state.lower()
        values = {
            "red": [1.0, 0.0, 0.0, 0.0],
            "yellow": [0.0, 1.0, 0.0, 0.0],
            "green": [0.0, 0.0, 1.0, 0.0],
        }
        return np.array(values.get(normalized, [0.0, 0.0, 0.0, 1.0]), dtype=np.float32)

    def _open_replay_log(self) -> None:
        replay_enabled = bool(
            self.carla_config.get("debug", {}).get("replay_logging", True)
            and self.training_config.get("environment", {}).get("replay_logging", True)
        )
        if not replay_enabled:
            return
        replay_dir = self.project_root / self.carla_config.get("debug", {}).get("replay_dir", "analytics/training_logs/replays")
        replay_dir.mkdir(parents=True, exist_ok=True)
        replay_path = replay_dir / f"episode_{self.episode:06d}.jsonl"
        self._replay_handle = replay_path.open("w", encoding="utf-8")

    def _write_replay_step(
        self,
        action: int,
        diagnostics: StepDiagnostics,
        reward: float,
        terminated: bool,
        truncated: bool,
    ) -> None:
        if not self._replay_handle:
            return
        payload = {
            "episode": self.episode,
            "step": self.steps,
            "action": int(action),
            "reward": reward,
            "speed_kmh": diagnostics.speed_kmh,
            "distance_delta_m": diagnostics.distance_delta_m,
            "front_obstacle_distance_m": diagnostics.front_obstacle_distance_m,
            "red_light_state": diagnostics.red_light_state,
            "route_completed_pct": diagnostics.route.route_completed_pct,
            "emergency_brake": diagnostics.emergency_brake,
            "terminated": terminated,
            "truncated": truncated,
        }
        self._replay_handle.write(json.dumps(payload) + "\n")
        self._replay_handle.flush()

    def _build_bev_map(self) -> np.ndarray:
        perception_config = self.carla_config.get("perception", {})
        bev_config = perception_config.get("bird_eye_view", {})
        size = int(bev_config.get("size", 128))
        if not bev_config.get("enabled", True):
            return np.zeros((3, size, size), dtype=np.float32)
        ego_transform = self.vehicle_manager.get_transform()
        route_locations = self.route_planner.get_upcoming_locations(limit=100)
        return render_bird_eye_view(
            self.sensor_manager.snapshot.lidar_points,
            route_locations=route_locations,
            ego_transform=ego_transform,
            size=size,
            meters=float(bev_config.get("meters", 60.0)),
            include_route=bool(bev_config.get("include_route", True)),
        )

    def _update_object_detections(self) -> None:
        if self.steps % self.detection_interval != 0:
            return
        try:
            self.last_detections = self.object_detector.detect(
                self.sensor_manager.snapshot.rgb,
                self.sensor_manager.snapshot.semantic,
            )
        except Exception:
            LOGGER.debug("Object detection failed for this frame.", exc_info=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
