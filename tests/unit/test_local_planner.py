"""LocalPlanner: leaves a clear wish alone, steers around, backs off when too close to turn."""

import math

import numpy as np

from pepin.control import ControllerConfig
from pepin.footprint import Footprint, time_to_contact
from pepin.kinematics import STOP, Twist
from pepin.local_planner import LocalPlanner

HULL = Footprint()
CTRL = ControllerConfig()  # cruise 0.15, max yaw 0.6


def cylinder(cx: float, cy: float, radius: float = 0.15, n: int = 16) -> np.ndarray:
    angles = np.linspace(0, 2 * math.pi, n, endpoint=False)
    return np.array([(cx + radius * math.cos(a), cy + radius * math.sin(a)) for a in angles])


def far_walls() -> np.ndarray:
    ys = np.linspace(-2, 2, 40)
    return np.array([(3.0, y) for y in ys] + [(-3.0, y) for y in ys])


def test_open_floor_passes_the_wish_through_untouched() -> None:
    planner = LocalPlanner(HULL, CTRL)
    wish = Twist(0.15, 0.1)
    assert planner.steer(wish, (0.3, 0.05), far_walls()) == (wish, "")


def test_a_blocked_wish_becomes_a_contact_free_arc_toward_the_target() -> None:
    planner = LocalPlanner(HULL, CTRL)
    person = np.vstack([far_walls(), cylinder(0.38, 0.05)])  # on the line, 0.23 m off the hull
    twist, why = planner.steer(Twist(0.15, 0.0), (0.9, -0.3), person)
    assert twist != STOP and why
    assert time_to_contact(person, twist, HULL, 1.0) is None


def test_driving_into_a_blocker_never_creeps_forward() -> None:
    """Something 0.18 m ahead: every forward arc touches within the horizon; turn or reverse."""
    planner = LocalPlanner(HULL, CTRL)
    blocker = np.vstack([far_walls(), cylinder(0.33, 0.0)])  # nearest point 0.18 m ahead
    twist, why = planner.steer(Twist(0.15, 0.0), (0.1, 0.5), blocker)
    assert twist.linear <= 0.0 and why


def test_backing_off_lasts_exactly_until_the_wish_is_possible() -> None:
    """A table 0.28 m ahead: a 180-degree turn in place hits it; back a little, then turn."""
    planner = LocalPlanner(HULL, CTRL)
    turn = Twist(0.0, 0.6)
    table = lambda gap: np.vstack([far_walls(), [(gap, y) for y in np.linspace(-0.8, 0.8, 161)]])  # noqa: E731
    twist, why = planner.steer(turn, (-1.0, 0.0), table(0.20), now=0.0)
    assert twist.linear < 0 and why.startswith("back off")  # the turn touches: reverse
    twist, why = planner.steer(turn, (-1.0, 0.0), table(0.21), now=0.05)
    assert twist.linear < 0  # still touches: keep backing, no forward crawl
    twist, why = planner.steer(turn, (-1.0, 0.0), table(0.40), now=0.10)
    assert (twist, why) == (turn, "")  # the turn is clear: the wish rules again, at once
    assert not planner._backing


def test_backing_off_is_capped_and_then_says_so() -> None:
    planner = LocalPlanner(HULL, CTRL)
    table = [(0.20, y) for y in np.linspace(-0.8, 0.8, 161)]  # a table edge right ahead
    wall = [(-0.80, y) for y in np.linspace(-0.8, 0.8, 161)]  # a wall well behind
    boxed = np.vstack([far_walls(), table, wall])
    twist, why = planner.steer(Twist(0.15, 0.0), (0.5, 0.3), boxed, now=0.0)
    assert twist.linear < 0
    t = 0.0
    while twist.linear < 0 and t < 30.0:
        t += 0.5
        twist, why = planner.steer(Twist(0.15, 0.0), (0.5, 0.3), boxed, now=t)
    assert twist == STOP and why.startswith("cannot turn here")
    assert planner._backed_m <= planner.cfg.max_back_off_m + 0.05


def test_boxed_in_stops_with_a_reason() -> None:
    planner = LocalPlanner(HULL, CTRL)
    ring = cylinder(0.0, 0.0, radius=0.36, n=48)  # all around, just outside the hull
    twist, why = planner.steer(Twist(0.15, 0.0), (0.5, 0.0), ring)
    assert twist == STOP and why.startswith("boxed in")
