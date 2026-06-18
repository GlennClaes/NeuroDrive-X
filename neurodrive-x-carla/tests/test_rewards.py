from ai.rewards import RewardConfig, RewardInput, compute_reward


def test_collision_penalty_dominates_positive_progress() -> None:
    signal = RewardInput(
        speed_kmh=35.0,
        speed_limit_kmh=50.0,
        lane_center_offset_m=0.1,
        reached_waypoint=True,
        front_obstacle_distance_m=20.0,
        stopped_for_red_light=False,
        ran_red_light=False,
        collision=True,
        offroad=False,
        lane_invasion=False,
        wrong_way=False,
        steering=0.1,
        steering_delta=0.02,
        idle=False,
        route_completed=False,
        distance_delta_m=3.0,
    )

    reward = compute_reward(signal, RewardConfig())

    assert reward.collision < -90.0
    assert reward.total < 0.0


def test_success_reward_is_positive_without_safety_violations() -> None:
    signal = RewardInput(
        speed_kmh=28.0,
        speed_limit_kmh=50.0,
        lane_center_offset_m=0.0,
        reached_waypoint=True,
        front_obstacle_distance_m=30.0,
        stopped_for_red_light=False,
        ran_red_light=False,
        collision=False,
        offroad=False,
        lane_invasion=False,
        wrong_way=False,
        steering=0.05,
        steering_delta=0.01,
        idle=False,
        route_completed=True,
        distance_delta_m=2.4,
    )

    reward = compute_reward(signal, RewardConfig())

    assert reward.success > 0.0
    assert reward.total > 80.0

