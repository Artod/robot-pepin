"""The navigator in a kinematic simulation: does it route around what the map does not know?"""

import math

import numpy as np
from synthetic import raycast_room
from test_localization import room_map

from pepin.control import ControllerConfig
from pepin.footprint import time_to_contact
from pepin.kinematics import STOP, Twist
from pepin.mapping import transform_to_world
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


def test_close_tof_return_becomes_an_obstacle_where_the_sensor_looks() -> None:
    from pepin.tof import TofMount

    start = Pose2D(-2.0, 0.0, 0.0)
    cfg = NavigatorConfig(tof_mounts={"left": TofMount(0.027, 0.148, 0.0, 0.16)})
    nav = Navigator(room_map(), start, (2.0, 0.0), cfg)
    nav.step(Sense(1.0, start, [raycast_room(start)], 0.0, TofRanges(None, 0.25, None, 0.0)))
    hits = nav._hits.points(1.0)
    assert hits is not None
    # The hit is placed with the localised pose, so measure from it, not from the truth.
    expected = transform_to_world(np.array([[0.027 + 0.25, 0.148]]), nav.pose)[0]
    assert np.min(np.hypot(hits[:, 0] - expected[0], hits[:, 1] - expected[1])) < 1e-6
    far = Navigator(room_map(), start, (2.0, 0.0), cfg)
    far.step(Sense(1.0, start, [], 0.0, TofRanges(None, 0.60, None, 0.0)))
    assert far._hits.points(1.0) is None  # beyond tof_hit_max_m: the lidar's job


def test_obstacle_memory_forgets_by_time_not_by_message_count() -> None:
    from pepin.navigator import ObstacleMemory

    memory = ObstacleMemory(horizon_s=1.0)
    for k in range(50):  # a chatty sensor: 50 messages within 0.25 s
        memory.add(0.0 + k * 0.005, np.array([[1.0, float(k)]]))
    memory.add(0.6, np.array([[2.0, 2.0]]))  # a slower sensor, later
    pts = memory.points(0.7)
    assert pts is not None and len(pts) == 51  # nothing drowned out
    assert (
        memory.points(1.3) is not None and len(memory.points(1.3)) == 1
    )  # only the 0.6 s entry left
    assert memory.points(2.0) is None


def test_set_goal_replans_toward_the_new_place() -> None:
    start = Pose2D(-2.0, 0.0, 0.0)
    nav = Navigator(room_map(), start, (2.0, 0.0), NavigatorConfig(retry_every_s=0.0))
    assert nav.plan is not None and nav.plan[-1][0] > 1.5
    nav.set_goal((-1.0, 1.0))
    decision = nav.step(Sense(1.0, start, [raycast_room(start)], 0.0, None))
    assert decision.plan_changed and nav.plan is not None
    assert abs(nav.plan[-1][1] - 1.0) < 0.1 and abs(nav.plan[-1][0] + 1.0) < 0.1


def test_reflex_stop_is_swept_before_it_reaches_the_wheels() -> None:
    """A front ToF stop turns an arc into a turn in place; that turn must be judged too."""
    start = Pose2D(-2.0, 0.0, 0.0)
    nav = Navigator(room_map(), start, (2.0, 0.0), NavigatorConfig(controller=FAST))
    by_rear_left_corner = np.array([[-0.37, 0.24], [-0.365, 0.237], [-0.367, 0.242]])  # a leg
    points = np.vstack([raycast_room(start), by_rear_left_corner])
    sense = Sense(1.0, start, [points], 0.0, TofRanges(0.15, None, None, 0.0))
    nav.step(sense)
    arc = Twist(0.15, 0.6)
    assert nav.guard.apply(arc, points)[1] == ""  # the arc alone clears the cluster
    twist, veto = nav.guard_twist(arc, sense)
    assert "tof" in veto and "lidar" in veto
    assert time_to_contact(points, twist, nav.guard.footprint, nav.guard.horizon_s) is None


def test_a_new_unreachable_goal_stops_the_old_route_at_once() -> None:
    start = Pose2D(-2.0, 0.0, 0.0)
    nav = Navigator(room_map(), start, (2.0, 0.0), NavigatorConfig(controller=FAST))
    assert nav.step(Sense(1.0, start, [raycast_room(start)], 0.0, None)).twist != STOP
    nav.set_goal((3.8, 2.8))  # behind the wall
    d = nav.step(Sense(1.05, start, [raycast_room(start)], 0.0, None))
    assert d.twist == STOP and d.hold and d.plan_changed and nav.plan is None


def test_a_revolution_the_navigator_never_saw_does_not_count_as_fresh() -> None:
    start = Pose2D(-2.0, 0.0, 0.0)
    nav = Navigator(room_map(), start, (2.0, 0.0))
    assert not nav.step(Sense(1.0, start, [raycast_room(start)], 0.0, None)).hold
    stale = nav.step(Sense(3.0, start, [], 0.0, None))  # the transport says fresh, we saw nothing
    assert stale.hold.startswith("no lidar scan for")


def test_without_a_goal_the_navigator_holds() -> None:
    start = Pose2D(-2.0, 0.0, 0.0)
    nav = Navigator(room_map(), start)
    assert nav.plan is None
    assert nav.step(Sense(1.0, start, [raycast_room(start)], 0.0, None)).hold == "no goal"
