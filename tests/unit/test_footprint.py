"""The hull sweep: forward contact, the rear swing in a turn, and the guard's order of retreat."""

import math

import numpy as np
import pytest

from pepin.footprint import Footprint, FootprintGuard, pose_after, time_to_contact
from pepin.kinematics import Twist

HULL = Footprint(front_m=0.0625, rear_m=0.30, half_width_m=0.275, margin_m=0.03)


def cluster(x: float, y: float, n: int = 4) -> np.ndarray:
    """A few lidar returns around one spot (the guard needs more than a lone noisy point)."""
    return np.array([[x + 0.01 * k, y] for k in range(n)])


def test_pose_after_is_the_exact_arc() -> None:
    x, y, theta = pose_after(Twist(0.2, math.pi / 2), 1.0)  # quarter circle, radius 0.2/(pi/2)
    r = 0.2 / (math.pi / 2)
    assert (x, y, theta) == pytest.approx((r, r, math.pi / 2))
    assert pose_after(Twist(0.15, 0.0), 2.0) == (0.3, 0.0, 0.0)


def test_something_ahead_is_hit_when_the_hull_front_reaches_it() -> None:
    ahead = cluster(0.20, 0.0)  # 20 cm ahead of the axle: 0.20 - 0.0625 - 0.03 = 0.1075 m of travel
    t = time_to_contact(ahead, Twist(0.15, 0.0), HULL, horizon_s=1.0)
    assert t is not None and 0.65 <= t <= 0.8
    assert time_to_contact(ahead, Twist(-0.15, 0.0), HULL) is None  # backing away is fine
    assert time_to_contact(ahead, Twist(0.0, 0.0), HULL) is None  # standing still touches nothing


def test_a_turn_in_place_sweeps_the_rear_corner_into_something_behind() -> None:
    behind_right = cluster(-0.36, -0.22)  # just outside the rear-right corner
    ccw = time_to_contact(behind_right, Twist(0.0, 0.6), HULL, horizon_s=1.0)
    cw = time_to_contact(behind_right, Twist(0.0, -0.6), HULL, horizon_s=1.0)
    assert (ccw is None) != (cw is None), "exactly one turning direction swings the corner into it"
    assert time_to_contact(behind_right, Twist(0.15, 0.0), HULL) is None  # driving off is fine


def test_guard_slows_then_blocks_forward_then_stops() -> None:
    guard = FootprintGuard(HULL, horizon_s=0.6)
    clear = np.array([[2.0, 0.0], [2.0, 0.5]])
    assert guard.apply(Twist(0.15, 0.1), clear) == (Twist(0.15, 0.1), "")
    near = cluster(0.16, 0.0)
    slowed, reason = guard.apply(Twist(0.15, 0.0), near)
    assert slowed.linear in (0.075, 0.0) and reason  # slowed or forward blocked, never through
    point_blank = cluster(0.10, 0.0)
    blocked, reason = guard.apply(Twist(0.15, 0.2), point_blank)
    assert blocked.linear == 0.0 and blocked.angular == 0.2 and "forward blocked" in reason
    boxed = np.vstack(
        [cluster(0.10, 0.0), cluster(-0.34, 0.0), cluster(0.0, 0.31), cluster(0.0, -0.31)]
    )
    stop, reason = guard.apply(Twist(0.15, 0.6), boxed)
    assert stop == Twist(0.0, 0.0) and "hold" in reason
