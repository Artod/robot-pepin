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


class SimRobot:
    """An ideal cart in the synthetic room: integrates the twist it is given, sees the walls."""

    def __init__(self, start: Pose2D, dt: float = 0.05) -> None:
        self.truth = start
        self.dt = dt
        self.commands: list[Twist] = []
        self.stopped = 0

    def observe(self, now: float) -> Observation | None:
        state = BaseState(self.truth, 0.0, 0.0, 0.0, 0.0, False, True, False, True, 5.0, now, 0.02)
        sense = Sense(now, self.truth, [raycast_room(self.truth)], 0.0, None)
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
    assert driver.tick(3.05).mode is Mode.DRIVING
    driver.cancel()
    status = driver.tick(3.1)
    assert status.mode is Mode.IDLE and status.goal is None and robot.stopped == 1


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
