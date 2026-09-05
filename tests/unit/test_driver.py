"""The Driver against a simulated robot: goto by name and by numbers, pause, cancel, facing."""

import math

from synthetic import raycast_room
from test_localization import room_map

from pepin.base_link import BaseState
from pepin.driver import Driver, Mode
from pepin.feeds import Sense
from pepin.kinematics import STOP, Twist
from pepin.odometry import Pose2D, wrap_angle
from pepin.places import Place
from pepin.robot import Observation
from pepin.scanmatch import apply_motion
from pepin.tof import TofRanges


class SimRobot:
    """An ideal cart in the synthetic room: integrates the twist it is given, sees the walls."""

    def __init__(self, start: Pose2D, dt: float = 0.05) -> None:
        self.truth = start
        self.dt = dt
        self.commands: list[Twist] = []
        self.stopped = 0
        self.blind = False  # no lidar revolutions, stale age
        self.silent = False  # no base telemetry at all

    def observe(self, now: float) -> Observation | None:
        if self.silent:
            return None
        state = BaseState(self.truth, 0.0, 0.0, 0.0, 0.0, False, True, False, True, 5.0, now, 0.02)
        tof = TofRanges(None, None, None, 0.0)
        if self.blind:
            sense = Sense(now, self.truth, [], 5.0, tof)
        else:
            sense = Sense(now, self.truth, [raycast_room(self.truth)], 0.0, tof)
        return Observation(state, sense, [])

    def drive(self, twist: Twist) -> None:
        self.commands.append(twist)
        motion = Pose2D(twist.linear * self.dt, 0.0, twist.angular * self.dt)
        self.truth = apply_motion(self.truth, motion)

    def stop(self) -> None:
        self.stopped += 1


PLACES = {
    "far_wall": Place("far_wall", 2.0, 0.0),
    "corner": Place("corner", 1.5, 1.0, theta_deg=180.0),
}


START = Pose2D(-2.0, 0.0, 0.0)


def make(start: Pose2D = START) -> tuple[Driver, SimRobot]:
    robot = SimRobot(start)
    driver = Driver(robot, room_map(), start, places=PLACES)  # type: ignore[arg-type]
    return driver, robot


def drive_until(driver: Driver, robot: SimRobot, mode: Mode, max_ticks: int = 1500) -> int:
    for k in range(max_ticks):
        if driver.tick(now=k * robot.dt).mode is mode:
            return k
    raise AssertionError(f"never reached {mode}; last: {driver.status()}")


def test_idle_until_goto_then_arrives_at_a_named_place() -> None:
    driver, robot = make()
    assert driver.tick(0.0).mode is Mode.IDLE and robot.commands[-1] == STOP
    assert driver.goto("far_wall") == (2.0, 0.0)
    drive_until(driver, robot, Mode.ARRIVED)
    status = driver.status()
    assert status is not None and status.goal_name == "far_wall"
    assert status.distance_m is not None and status.distance_m < 0.2
    assert math.hypot(robot.truth.x - 2.0, robot.truth.y) < 0.2
    assert driver.tick(999.0).twist == STOP  # arrived: stays put
    assert "arrived" in driver.tick(999.1).summary()


def test_pause_holds_resume_continues_cancel_forgets() -> None:
    driver, robot = make()
    driver.goto((2.0, 0.0))
    for k in range(60):
        driver.tick(k * robot.dt)
    assert robot.truth.x > -1.95  # it moved
    driver.pause()
    status = driver.tick(3.0)
    assert status.mode is Mode.PAUSED and status.twist == STOP and status.reason == "paused"
    driver.resume()
    x_before = robot.truth.x
    for k in range(20):
        assert driver.tick(3.05 + k * robot.dt).mode is Mode.DRIVING
    assert robot.truth.x > x_before + 0.05  # it drives again, not merely says so
    driver.cancel()
    status = driver.tick(4.1)
    assert status.mode is Mode.IDLE and status.goal is None and robot.stopped == 1
    assert driver.navigator.paused is False
    driver.goto((2.0, 0.0))  # cancel leaves nothing behind that would hold the next goal
    moved_at = None
    for k in range(5):
        if driver.tick(4.2 + k * robot.dt).twist != STOP:
            moved_at = k
            break
    assert moved_at is not None and moved_at <= 1


def test_a_place_with_a_heading_is_faced_on_arrival() -> None:
    driver, robot = make(Pose2D(1.5, -1.0, 0.0))
    driver.goto("corner")  # 2 m north, then turn to face west
    drive_until(driver, robot, Mode.ARRIVED)
    assert abs(wrap_angle(robot.truth.theta - math.pi)) < math.radians(6.0)


def test_unknown_place_names_the_known_ones() -> None:
    driver, _ = make()
    try:
        driver.goto("balcony")
    except ValueError as exc:
        assert "corner" in str(exc) and "far_wall" in str(exc)
    else:
        raise AssertionError("an unknown place must be refused")


def test_no_base_telemetry_is_reported_not_driven_through() -> None:
    driver, robot = make()
    driver.goto((2.0, 0.0))
    robot.observe = lambda now: None  # type: ignore[method-assign]
    status = driver.tick(1.0)
    assert status.mode is Mode.NO_BASE and status.twist == STOP
    assert status.base_age_s == float("inf") and "no base" in status.summary()


def test_arrival_survives_a_base_outage_and_no_command_is_sent_meanwhile() -> None:
    driver, robot = make()
    driver.goto("far_wall")
    k = drive_until(driver, robot, Mode.ARRIVED)
    sent = len(robot.commands)
    robot.silent = True
    for j in range(1, 4):
        status = driver.tick((k + j) * robot.dt)
        assert status.mode is Mode.NO_BASE and status.twist == STOP
    assert len(robot.commands) == sent  # nothing to drive while the board is quiet
    robot.silent = False
    status = driver.tick((k + 5) * robot.dt)
    assert status.mode is Mode.ARRIVED and status.twist == STOP and status.target is None


def facing_run(driver: Driver, robot: SimRobot) -> tuple[int, Pose2D]:
    """Drive to the corner until FACING starts; returns the tick and the pose there."""
    k = drive_until(driver, robot, Mode.FACING)
    return k, robot.truth


def test_facing_resumes_without_a_lunge_after_a_pause_or_a_blind_spell() -> None:
    for interruption in ("pause", "blind", "silent"):
        driver, robot = make(Pose2D(1.5, -1.0, 0.0))
        driver.goto("corner")
        k, there = facing_run(driver, robot)
        if interruption == "pause":
            driver.pause()
        else:
            setattr(robot, interruption, True)
        for j in range(1, 41):  # 2 s of interruption: a replan falls due
            driver.tick((k + j) * robot.dt)
        if interruption == "pause":
            driver.resume()
        else:
            setattr(robot, interruption, False)
        since = len(robot.commands)
        drive_until(driver, robot, Mode.ARRIVED)
        assert all(c.linear == 0.0 for c in robot.commands[since:]), interruption
        assert math.hypot(robot.truth.x - there.x, robot.truth.y - there.y) < 1e-3, interruption
        assert abs(wrap_angle(robot.truth.theta - math.pi)) < math.radians(6.0), interruption


def test_the_hull_sweep_runs_exactly_once_per_facing_tick() -> None:
    driver, robot = make(Pose2D(1.5, -1.0, 0.0))
    driver.goto("corner")
    k, _ = facing_run(driver, robot)
    calls = 0
    original = driver.navigator.guard_twist

    def counting(twist: Twist, sense: Sense) -> tuple[Twist, str]:
        nonlocal calls
        calls += 1
        return original(twist, sense)

    driver.navigator.guard_twist = counting  # type: ignore[method-assign]
    ticks = 0
    for j in range(1, 30):
        if driver.tick((k + j) * robot.dt).mode is not Mode.FACING:
            break
        ticks += 1
    assert ticks >= 5 and calls == ticks
