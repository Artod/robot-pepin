"""The navigator in a kinematic simulation: does it route around what the map does not know?"""

import math

import numpy as np
from synthetic import raycast_room
from test_localization import room_map

from pepin.control import ControllerConfig
from pepin.navigator import Navigator, NavigatorConfig, Sense
from pepin.odometry import Pose2D
from pepin.scanmatch import apply_motion, relative_motion
from pepin.tof import TofRanges

DT = 0.1
# A brisk simulated robot keeps the whole-stack tests at a second or so.
FAST = ControllerConfig(cruise_speed_m_s=0.35, max_yaw_rate_rad_s=1.5)
PERSON = (0.0, 0.0)  # stands right on the straight line from start to goal, not in the map
PERSON_RADIUS = 0.15


def person_points(pose: Pose2D) -> np.ndarray:
    """Robot-frame lidar hits on a small cylinder at PERSON as seen from ``pose``."""
    pts = []
    for k in range(16):
        a = 2 * math.pi * k / 16
        wx, wy = PERSON[0] + PERSON_RADIUS * math.cos(a), PERSON[1] + PERSON_RADIUS * math.sin(a)
        rel = relative_motion(pose, Pose2D(wx, wy, 0.0))
        pts.append((rel.x, rel.y))
    return np.array(pts)


def scan(pose: Pose2D, with_person: bool) -> np.ndarray:
    room = raycast_room(pose, beams=180)
    return np.vstack([room, person_points(pose)]) if with_person else room


def simulate(
    nav: Navigator, start: Pose2D, *, with_person: bool, max_s: float = 60.0
) -> tuple[Pose2D, float, bool]:
    """Ideal robot: odometry is truth; returns final pose, closest approach to PERSON, done."""
    pose = start
    closest = math.inf
    for i in range(int(max_s / DT)):
        now = (i + 1) * DT
        sense = Sense(now, pose, [scan(pose, with_person)], 0.0, TofRanges(None, None, None, 0.0))
        d = nav.step(sense)
        if d.done:
            return pose, closest, True
        pose = apply_motion(pose, Pose2D(d.twist.linear * DT, 0.0, d.twist.angular * DT))
        closest = min(closest, math.hypot(pose.x - PERSON[0], pose.y - PERSON[1]))
    return pose, closest, False


def test_drives_straight_to_the_goal_in_an_empty_room() -> None:
    start, goal = Pose2D(-2.0, 0.0, 0.0), (2.0, 0.0)
    nav = Navigator(room_map(), start, goal, NavigatorConfig(controller=FAST))
    assert nav.plan is not None
    pose, _, done = simulate(nav, start, with_person=False)
    assert done
    assert math.hypot(pose.x - goal[0], pose.y - goal[1]) < 0.15


def test_routes_around_a_person_the_map_does_not_know() -> None:
    start, goal = Pose2D(-2.0, 0.0, 0.0), (2.0, 0.0)
    nav = Navigator(room_map(), start, goal, NavigatorConfig(controller=FAST))
    pose, closest, done = simulate(nav, start, with_person=True)
    assert done, f"never reached the goal; ended at {pose}"
    assert closest > PERSON_RADIUS + 0.10, f"came within {closest:.2f} m of the person"


def test_obstacle_at_the_hull_does_not_freeze_the_planner() -> None:
    """A hand 10 cm from the side used to block the robot's own cell: no plan, no motion."""
    start, goal = Pose2D(-2.0, 0.0, 0.0), (2.0, 0.0)
    nav = Navigator(room_map(), start, goal, NavigatorConfig(retry_every_s=0.0))
    hand = np.array([[0.05, 0.28]])  # robot frame: beside the left front corner
    sense = Sense(3.5, start, [np.vstack([raycast_room(start), hand])], 0.0, None)
    nav.step(sense)
    d = nav.step(Sense(4.0, start, [np.vstack([raycast_room(start), hand])], 0.0, None))
    assert nav.plan is not None, "the footprint must be exempt from inflation"
    assert not d.hold


def test_holds_still_without_lidar_and_resumes() -> None:
    start, goal = Pose2D(-2.0, 0.0, 0.0), (2.0, 0.0)
    nav = Navigator(room_map(), start, goal)
    blind = nav.step(Sense(1.0, start, [], 5.0, None))
    assert blind.hold.startswith("no lidar scan")
    assert blind.twist.linear == 0.0 and blind.twist.angular == 0.0
    seeing = nav.step(Sense(1.1, start, [raycast_room(start)], 0.0, None))
    assert not seeing.hold and seeing.twist.linear > 0.0
