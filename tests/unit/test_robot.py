"""The Robot composition root with fake feeds: configuration and one tick of observation."""

import math

import numpy as np

from pepin.base_link import BaseState
from pepin.odometry import Pose2D
from pepin.robot import CONFIG_DIR, Robot, RobotConfig
from pepin.tof import TofRanges


class FakeLink:
    connected = True

    def __init__(self, state: BaseState | None) -> None:
        self._state = state
        self.commands: list[str] = []

    def state(self, now=None):  # type: ignore[no-untyped-def]
        return self._state

    def set_twist(self, twist) -> None:  # type: ignore[no-untyped-def]
        self.commands.append(f"twist {twist.linear:.2f}")

    def stop(self) -> None:
        self.commands.append("stop")

    def age_s(self, now=None) -> float:  # type: ignore[no-untyped-def]
        return 0.0

    def close(self) -> None:
        self.commands.append("close")


class FakeScan:
    stamp = 0.0

    def points_xy(self, mount):  # type: ignore[no-untyped-def]
        return np.array([[1.0, 0.0], [1.0, 0.1]])


class FakeLidar:
    connected = True

    def __init__(self) -> None:
        self.closed = False

    def drain(self):  # type: ignore[no-untyped-def]
        return [FakeScan()]

    def age_s(self, now=None) -> float:  # type: ignore[no-untyped-def]
        return 0.05

    def close(self) -> None:
        self.closed = True


class FakeTof:
    connected = True

    def ranges(self, now=None):  # type: ignore[no-untyped-def]
        return TofRanges(0.9, None, None, 0.0)

    def age_s(self, now=None) -> float:  # type: ignore[no-untyped-def]
        return 0.0

    def close(self) -> None:
        pass


def state(age_s: float) -> BaseState:
    return BaseState(
        Pose2D(1.0, 2.0, 0.5), 0.01, 0.01, 0.1, 0.0, True, True, False, True, 5.0, 0.0, age_s
    )


def test_config_loads_from_the_repo_and_feeds_can_be_switched_off() -> None:
    cfg = RobotConfig.load(CONFIG_DIR)
    assert cfg.enabled("lidar") and cfg.enabled("tof")
    assert cfg.port("base", 0) == 3336 and cfg.port("unknown", 7) == 7
    assert set(cfg.tof_mounts) == {"front", "left", "right"}
    quiet = cfg.without("tof")
    assert not quiet.enabled("tof") and quiet.enabled("lidar")
    assert cfg.enabled("tof")  # the original is untouched


def test_observe_composes_one_sense_from_every_feed() -> None:
    cfg = RobotConfig.load(CONFIG_DIR)
    robot = Robot(cfg, "h", FakeLink(state(0.1)), FakeLidar(), FakeTof())  # type: ignore[arg-type]
    obs = robot.observe(now=10.0)
    assert obs is not None
    assert obs.sense.odom_pose == Pose2D(1.0, 2.0, 0.5)
    assert len(obs.sense.scans) == 1 and obs.sense.scans[0].shape == (2, 2)
    assert obs.sense.scan_age_s == 0.05 and obs.sense.tof is not None and obs.sense.tof.front == 0.9
    assert len(obs.scans) == 1


def test_stale_base_telemetry_means_no_observation_and_tof_off_means_no_ranges() -> None:
    cfg = RobotConfig.load(CONFIG_DIR).without("tof")
    robot = Robot(cfg, "h", FakeLink(state(2.0)), FakeLidar(), None)  # type: ignore[arg-type]
    assert robot.observe(now=10.0) is None
    fresh = Robot(cfg, "h", FakeLink(state(0.0)), None, None)  # type: ignore[arg-type]
    obs = fresh.observe(now=10.0)
    assert obs is not None and obs.sense.tof is None and math.isinf(obs.sense.scan_age_s)


def test_close_stops_the_wheels_first_then_releases_the_feeds() -> None:
    cfg = RobotConfig.load(CONFIG_DIR)
    link, lidar = FakeLink(state(0.0)), FakeLidar()
    with Robot(cfg, "h", link, lidar, None):  # type: ignore[arg-type]
        pass
    assert link.commands[0] == "stop" and "close" in link.commands and lidar.closed
