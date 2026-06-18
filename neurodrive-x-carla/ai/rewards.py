"""Reward shaping for the CARLA autonomous driving task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True)
class RewardConfig:
    """Configurable coefficients for the driving reward."""

    forward_velocity_weight: float = 0.12
    lane_centering_weight: float = 0.9
    waypoint_reward: float = 12.0
    safe_distance_weight: float = 0.45
    red_light_stop_reward: float = 3.0
    smooth_driving_weight: float = 0.35
    collision_penalty: float = -100.0
    offroad_penalty: float = -35.0
    lane_invasion_penalty: float = -8.0
    red_light_penalty: float = -45.0
    wrong_way_penalty: float = -50.0
    hard_steering_penalty: float = -4.0
    idle_penalty: float = -2.5
    time_penalty: float = -0.02
    success_reward: float = 80.0

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any] | None) -> "RewardConfig":
        """Build a reward config from a dictionary, ignoring unrelated keys."""

        if not values:
            return cls()
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in values.items() if key in known})


@dataclass(frozen=True)
class RewardInput:
    """Simulator signals needed to score one environment step."""

    speed_kmh: float
    speed_limit_kmh: float
    lane_center_offset_m: float
    reached_waypoint: bool
    front_obstacle_distance_m: float | None
    stopped_for_red_light: bool
    ran_red_light: bool
    collision: bool
    offroad: bool
    lane_invasion: bool
    wrong_way: bool
    steering: float
    steering_delta: float
    idle: bool
    route_completed: bool
    distance_delta_m: float


@dataclass
class RewardBreakdown:
    """Detailed reward terms for debugging, dashboards, and tests."""

    forward: float = 0.0
    lane_centering: float = 0.0
    waypoint: float = 0.0
    safe_distance: float = 0.0
    red_light: float = 0.0
    smooth_driving: float = 0.0
    collision: float = 0.0
    offroad: float = 0.0
    lane_invasion: float = 0.0
    wrong_way: float = 0.0
    hard_steering: float = 0.0
    idle: float = 0.0
    time: float = 0.0
    success: float = 0.0
    total: float = field(init=False, default=0.0)

    def finalize(self) -> "RewardBreakdown":
        """Recalculate and return the total reward."""

        values = asdict(self)
        values.pop("total", None)
        self.total = float(sum(values.values()))
        return self

    def to_dict(self) -> dict[str, float]:
        """Return a serializable reward dictionary."""

        return asdict(self)


def compute_reward(signal: RewardInput, config: RewardConfig | None = None) -> RewardBreakdown:
    """Compute a dense autonomous-driving reward from CARLA-derived signals."""

    cfg = config or RewardConfig()
    breakdown = RewardBreakdown()

    speed_ratio = _clamp(signal.speed_kmh / max(signal.speed_limit_kmh, 1.0), 0.0, 1.4)
    forward_progress = max(signal.distance_delta_m, 0.0)
    breakdown.forward = cfg.forward_velocity_weight * (0.6 * speed_ratio + forward_progress)

    normalized_lane_error = _clamp(abs(signal.lane_center_offset_m) / 1.75, 0.0, 1.0)
    breakdown.lane_centering = cfg.lane_centering_weight * (1.0 - normalized_lane_error)

    if signal.reached_waypoint:
        breakdown.waypoint = cfg.waypoint_reward

    if signal.front_obstacle_distance_m is not None:
        distance = max(signal.front_obstacle_distance_m, 0.0)
        if distance >= 12.0:
            breakdown.safe_distance = cfg.safe_distance_weight
        elif distance >= 5.0:
            breakdown.safe_distance = cfg.safe_distance_weight * (distance - 5.0) / 7.0
        else:
            breakdown.safe_distance = -cfg.safe_distance_weight * (5.0 - distance)

    if signal.stopped_for_red_light:
        breakdown.red_light += cfg.red_light_stop_reward
    if signal.ran_red_light:
        breakdown.red_light += cfg.red_light_penalty

    steering_smoothness = 1.0 - _clamp(abs(signal.steering_delta) / 0.35, 0.0, 1.0)
    breakdown.smooth_driving = cfg.smooth_driving_weight * steering_smoothness

    if signal.collision:
        breakdown.collision = cfg.collision_penalty
    if signal.offroad:
        breakdown.offroad = cfg.offroad_penalty
    if signal.lane_invasion:
        breakdown.lane_invasion = cfg.lane_invasion_penalty
    if signal.wrong_way:
        breakdown.wrong_way = cfg.wrong_way_penalty
    if abs(signal.steering) > 0.62:
        breakdown.hard_steering = cfg.hard_steering_penalty * abs(signal.steering)
    if signal.idle:
        breakdown.idle = cfg.idle_penalty
    if signal.route_completed:
        breakdown.success = cfg.success_reward

    breakdown.time = cfg.time_penalty
    return breakdown.finalize()


def _clamp(value: float, low: float, high: float) -> float:
    return min(max(value, low), high)

