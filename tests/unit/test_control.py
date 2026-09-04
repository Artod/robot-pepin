"""The path follower in a kinematic simulation: does it arrive, and how?"""

import math

from pepin.control import ControllerConfig, PathFollower
from pepin.odometry import Pose2D
from pepin.scanmatch import apply_motion


def simulate(
    path: list[tuple[float, float]], start: Pose2D, dt: float = 0.05, max_s: float = 60.0
) -> tuple[Pose2D, int]:
    """Integrate the commanded twist with an ideal robot; return the final pose and tick count."""
    follower = PathFollower(path)
    pose = start
    for tick in range(int(max_s / dt)):
        out = follower.step(pose)
        if out.done:
            return pose, tick
        t = out.twist
        pose = apply_motion(pose, Pose2D(t.linear * dt, 0.0, t.angular * dt))
    return pose, int(max_s / dt)


def test_reaches_a_goal_straight_ahead() -> None:
    pose, ticks = simulate([(1.0, 0.0)], Pose2D())
    assert math.hypot(pose.x - 1.0, pose.y) <= 0.12
    assert ticks < 400  # well under 20 s at 0.15 m/s


def test_turns_in_place_first_when_the_goal_is_behind() -> None:
    follower = PathFollower([(-1.0, 0.0)])
    out = follower.step(Pose2D())
    assert out.twist.linear == 0.0 and abs(out.twist.angular) > 0.5


def test_follows_an_l_shaped_path_through_the_corner() -> None:
    pose, _ = simulate([(1.0, 0.0), (1.0, 1.0)], Pose2D())
    assert math.hypot(pose.x - 1.0, pose.y - 1.0) <= 0.12


def test_stops_inside_goal_tolerance_immediately() -> None:
    out = PathFollower([(0.05, 0.0)]).step(Pose2D())
    assert out.done and out.twist.linear == 0.0


def test_speed_is_capped_by_config() -> None:
    cfg = ControllerConfig(cruise_speed_m_s=0.1, max_yaw_rate_rad_s=0.3)
    out = PathFollower([(2.0, 0.3)], cfg).step(Pose2D())
    assert out.twist.linear <= 0.1 and abs(out.twist.angular) <= 0.3
