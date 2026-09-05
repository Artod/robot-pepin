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
    person = np.vstack([far_walls(), cylinder(0.55, 0.05)])  # on the line, 0.4 m off the hull
    twist, why = planner.steer(Twist(0.15, 0.0), (0.9, -0.3), person)
    assert twist != STOP and "steer around" in why
    assert time_to_contact(person, twist, HULL, 1.0) is None
    assert twist.linear > 0 and twist.angular < 0  # around the right, where the target is


def test_driving_into_a_blocker_never_creeps_forward() -> None:
    """Something 0.18 m ahead: every forward arc touches within the horizon; turn or reverse."""
    planner = LocalPlanner(HULL, CTRL)
    blocker = np.vstack([far_walls(), cylinder(0.33, 0.0)])  # nearest point 0.18 m ahead
    twist, why = planner.steer(Twist(0.15, 0.0), (0.1, 0.5), blocker)
    assert twist.linear <= 0.0 and why


def test_backing_off_is_kept_up_until_there_is_room_to_turn() -> None:
    """Once reversing, a clear crawl forward must not end it: back until a turn in place is free."""
    planner = LocalPlanner(HULL, CTRL)
    planner._backing = True  # the state a first back-off leaves behind
    cramped = np.vstack([far_walls(), cylinder(0.50, 0.0)])  # 0.35 m: a turn would still graze
    twist, why = planner.steer(Twist(0.15, 0.0), (0.4, 0.3), cramped)
    assert twist.linear < 0 and why.startswith("back off")
    roomy = np.vstack([far_walls(), cylinder(0.75, 0.0)])  # 0.60 m: the whole hull can swing
    assert planner.steer(Twist(0.0, 0.6), (0.1, 0.5), roomy) == (Twist(0.0, 0.6), "")
    assert not planner._backing


def test_boxed_in_stops_with_a_reason() -> None:
    planner = LocalPlanner(HULL, CTRL)
    ring = cylinder(0.0, 0.0, radius=0.36, n=48)  # all around, just outside the hull
    twist, why = planner.steer(Twist(0.15, 0.0), (0.5, 0.0), ring)
    assert twist == STOP and why.startswith("boxed in")
