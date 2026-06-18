"""CARLA sensor spawning and frame buffering."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Any

import numpy as np

try:
    import carla
except ImportError:  # pragma: no cover
    carla = None  # type: ignore[assignment]

LOGGER = logging.getLogger(__name__)


@dataclass
class SensorSnapshot:
    """Latest synchronized sensor values collected from CARLA callbacks."""

    rgb: np.ndarray | None = None
    semantic: np.ndarray | None = None
    lidar_points: np.ndarray | None = None
    collision: bool = False
    lane_invasion: bool = False
    collision_count: int = 0
    lane_invasion_count: int = 0
    gnss: tuple[float, float, float] | None = None
    imu: dict[str, tuple[float, float, float]] = field(default_factory=dict)


class SensorManager:
    """Attach and manage the required CARLA sensors for the ego vehicle."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.sensors: list[Any] = []
        self.snapshot = SensorSnapshot()
        self._collision_this_step = False
        self._lane_invasion_this_step = False

    def spawn_sensors(self, world: Any, vehicle: Any) -> None:
        """Spawn RGB, semantic, LiDAR, collision, lane, GNSS, and IMU sensors."""

        _require_carla()
        blueprint_library = world.get_blueprint_library()

        camera_config = self.config.get("camera", {})
        rgb_bp = blueprint_library.find("sensor.camera.rgb")
        rgb_bp.set_attribute("image_size_x", str(camera_config.get("width", 320)))
        rgb_bp.set_attribute("image_size_y", str(camera_config.get("height", 180)))
        rgb_bp.set_attribute("fov", str(camera_config.get("fov", 100)))
        rgb_sensor = world.spawn_actor(rgb_bp, _transform_from_config(camera_config), attach_to=vehicle)
        rgb_sensor.listen(self._on_rgb)
        self.sensors.append(rgb_sensor)

        semantic_config = self.config.get("semantic_camera", camera_config)
        semantic_bp = blueprint_library.find("sensor.camera.semantic_segmentation")
        semantic_bp.set_attribute("image_size_x", str(semantic_config.get("width", 320)))
        semantic_bp.set_attribute("image_size_y", str(semantic_config.get("height", 180)))
        semantic_bp.set_attribute("fov", str(semantic_config.get("fov", 100)))
        semantic_sensor = world.spawn_actor(semantic_bp, _transform_from_config(semantic_config), attach_to=vehicle)
        semantic_sensor.listen(self._on_semantic)
        self.sensors.append(semantic_sensor)

        lidar_config = self.config.get("lidar", {})
        lidar_bp = blueprint_library.find("sensor.lidar.ray_cast")
        lidar_bp.set_attribute("channels", str(lidar_config.get("channels", 32)))
        lidar_bp.set_attribute("range", str(lidar_config.get("range", 60.0)))
        lidar_bp.set_attribute("points_per_second", str(lidar_config.get("points_per_second", 56000)))
        lidar_bp.set_attribute("rotation_frequency", str(lidar_config.get("rotation_frequency", 20.0)))
        lidar_bp.set_attribute("upper_fov", str(lidar_config.get("upper_fov", 10.0)))
        lidar_bp.set_attribute("lower_fov", str(lidar_config.get("lower_fov", -30.0)))
        lidar_sensor = world.spawn_actor(lidar_bp, _transform_from_config(lidar_config), attach_to=vehicle)
        lidar_sensor.listen(self._on_lidar)
        self.sensors.append(lidar_sensor)

        if self.config.get("collision", True):
            collision_sensor = world.spawn_actor(
                blueprint_library.find("sensor.other.collision"),
                carla.Transform(),
                attach_to=vehicle,
            )
            collision_sensor.listen(self._on_collision)
            self.sensors.append(collision_sensor)

        if self.config.get("lane_invasion", True):
            lane_sensor = world.spawn_actor(
                blueprint_library.find("sensor.other.lane_invasion"),
                carla.Transform(),
                attach_to=vehicle,
            )
            lane_sensor.listen(self._on_lane_invasion)
            self.sensors.append(lane_sensor)

        if self.config.get("gnss", True):
            gnss_sensor = world.spawn_actor(
                blueprint_library.find("sensor.other.gnss"),
                carla.Transform(carla.Location(x=0.0, z=2.0)),
                attach_to=vehicle,
            )
            gnss_sensor.listen(self._on_gnss)
            self.sensors.append(gnss_sensor)

        if self.config.get("imu", True):
            imu_sensor = world.spawn_actor(
                blueprint_library.find("sensor.other.imu"),
                carla.Transform(carla.Location(x=0.0, z=2.0)),
                attach_to=vehicle,
            )
            imu_sensor.listen(self._on_imu)
            self.sensors.append(imu_sensor)

        LOGGER.info("Spawned %d sensors.", len(self.sensors))

    def consume_step_events(self) -> tuple[bool, bool]:
        """Return collision/lane events since the previous call and reset flags."""

        collision = self._collision_this_step
        lane = self._lane_invasion_this_step
        self._collision_this_step = False
        self._lane_invasion_this_step = False
        self.snapshot.collision = False
        self.snapshot.lane_invasion = False
        return collision, lane

    def reset_events(self) -> None:
        """Reset counters and one-step event flags."""

        self.snapshot.collision = False
        self.snapshot.lane_invasion = False
        self.snapshot.collision_count = 0
        self.snapshot.lane_invasion_count = 0
        self._collision_this_step = False
        self._lane_invasion_this_step = False

    def destroy(self) -> None:
        """Stop and destroy all spawned sensor actors."""

        for sensor in self.sensors:
            try:
                sensor.stop()
                sensor.destroy()
            except RuntimeError:
                LOGGER.debug("Sensor was already destroyed.", exc_info=True)
        self.sensors.clear()

    def _on_rgb(self, image: Any) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self.snapshot.rgb = array[:, :, :3][:, :, ::-1].copy()

    def _on_semantic(self, image: Any) -> None:
        array = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
        self.snapshot.semantic = array[:, :, :3].copy()

    def _on_lidar(self, data: Any) -> None:
        points = np.frombuffer(data.raw_data, dtype=np.float32).reshape((-1, 4))
        self.snapshot.lidar_points = points[:, :3].copy()

    def _on_collision(self, event: Any) -> None:
        self.snapshot.collision = True
        self._collision_this_step = True
        self.snapshot.collision_count += 1
        LOGGER.warning("Collision detected with actor=%s", getattr(event.other_actor, "type_id", "unknown"))

    def _on_lane_invasion(self, event: Any) -> None:
        self.snapshot.lane_invasion = True
        self._lane_invasion_this_step = True
        self.snapshot.lane_invasion_count += 1
        markings = [str(marking.type) for marking in event.crossed_lane_markings]
        LOGGER.info("Lane invasion detected: %s", ", ".join(markings))

    def _on_gnss(self, event: Any) -> None:
        self.snapshot.gnss = (float(event.latitude), float(event.longitude), float(event.altitude))

    def _on_imu(self, event: Any) -> None:
        self.snapshot.imu = {
            "accelerometer": (float(event.accelerometer.x), float(event.accelerometer.y), float(event.accelerometer.z)),
            "gyroscope": (float(event.gyroscope.x), float(event.gyroscope.y), float(event.gyroscope.z)),
            "compass": (float(event.compass), 0.0, 0.0),
        }


def _transform_from_config(config: dict[str, Any]) -> Any:
    _require_carla()
    transform_config = config.get("transform", {})
    location = carla.Location(
        x=float(transform_config.get("x", 0.0)),
        y=float(transform_config.get("y", 0.0)),
        z=float(transform_config.get("z", 0.0)),
    )
    rotation = carla.Rotation(
        pitch=float(transform_config.get("pitch", 0.0)),
        yaw=float(transform_config.get("yaw", 0.0)),
        roll=float(transform_config.get("roll", 0.0)),
    )
    return carla.Transform(location, rotation)


def _require_carla() -> None:
    if carla is None:
        raise RuntimeError("CARLA Python API is required for sensor management.")

