"""The base driver against a fake bus: sign handling, clamping, encoder wrap."""

import math
from dataclasses import replace

import pytest

from pepin.base import LEFT, RIGHT, DiffDriveBase
from pepin.geometry import BaseConfig, BaseGeometry, WheelMotor
from pepin.kinematics import Twist

CFG = BaseConfig(
    geometry=BaseGeometry(wheel_diameter_m=0.125, track_width_m=0.5, ticks_per_rev=4096),
    left=WheelMotor(motor_id=7, direction=-1),
    right=WheelMotor(motor_id=8, direction=1),
    max_speed_m_s=0.3,
    max_yaw_rate_rad_s=1.0,
)


class FakeBus:
    """Records writes and serves scripted encoder positions."""

    def __init__(self) -> None:
        self.writes: list[tuple[str, dict[str, int]]] = []
        self.torque: list[tuple[str, list[str] | None]] = []
        self.positions = {LEFT: 0, RIGHT: 0}

    def sync_write(self, data_name: str, values: dict[str, int], *, normalize: bool = True) -> None:
        assert not normalize, "drivers must write raw units"
        self.writes.append((data_name, dict(values)))

    def sync_read(
        self, data_name: str, motors: list[str], *, normalize: bool = True
    ) -> dict[str, int]:
        assert data_name == "Present_Position" and not normalize
        return {m: self.positions[m] for m in motors}

    def enable_torque(self, motors: list[str] | None = None) -> None:
        self.torque.append(("on", motors))

    def disable_torque(self, motors: list[str] | None = None) -> None:
        self.torque.append(("off", motors))


@pytest.fixture
def bus() -> FakeBus:
    return FakeBus()


def test_forward_command_respects_mirrored_left_wheel(bus: FakeBus) -> None:
    DiffDriveBase(bus, CFG).set_twist(Twist(linear=0.1, angular=0.0))
    name, ticks = bus.writes[-1]
    assert name == "Goal_Velocity"
    assert ticks[LEFT] < 0 < ticks[RIGHT]
    assert ticks[LEFT] == -ticks[RIGHT]
    expected = round(0.1 / CFG.geometry.wheel_radius_m * 4096 / (2 * math.pi))
    assert ticks[RIGHT] == expected


def test_commands_are_clamped_to_configured_limits(bus: FakeBus) -> None:
    base = DiffDriveBase(bus, CFG)
    base.set_twist(Twist(linear=5.0, angular=0.0))
    fast = bus.writes[-1][1][RIGHT]
    base.set_twist(Twist(linear=CFG.max_speed_m_s, angular=0.0))
    assert fast == bus.writes[-1][1][RIGHT]


def test_stop_writes_zero_to_both_wheels(bus: FakeBus) -> None:
    DiffDriveBase(bus, CFG).stop()
    assert bus.writes[-1] == ("Goal_Velocity", {LEFT: 0, RIGHT: 0})


def test_wheel_travel_first_read_is_zero_then_signed_meters(bus: FakeBus) -> None:
    base = DiffDriveBase(bus, CFG)
    bus.positions = {LEFT: 4000, RIGHT: 100}
    assert base.read_wheel_travel() == (0.0, 0.0)
    # Left is mirrored: its encoder DEcreasing means the robot moved forward.
    bus.positions = {LEFT: 3900, RIGHT: 200}
    left, right = base.read_wheel_travel()
    assert left == pytest.approx(100 * CFG.geometry.m_per_tick)
    assert right == pytest.approx(100 * CFG.geometry.m_per_tick)


def test_wheel_travel_resolves_encoder_wrap(bus: FakeBus) -> None:
    base = DiffDriveBase(bus, CFG)
    bus.positions = {LEFT: 10, RIGHT: 4090}
    base.read_wheel_travel()
    bus.positions = {LEFT: 4086, RIGHT: 6}  # left went back by 20 ticks, right forward by 12
    left, right = base.read_wheel_travel()
    assert left == pytest.approx(20 * CFG.geometry.m_per_tick)  # mirrored sign flips it
    assert right == pytest.approx(12 * CFG.geometry.m_per_tick)


def test_context_manager_enables_then_stops_and_releases(bus: FakeBus) -> None:
    with DiffDriveBase(bus, CFG) as base:
        base.set_twist(Twist(0.1, 0.0))
    assert bus.torque == [("on", [LEFT, RIGHT]), ("off", [LEFT, RIGHT])]
    assert bus.writes[-1] == ("Goal_Velocity", {LEFT: 0, RIGHT: 0})


# -- the wheel ceiling ---------------------------------------------------------

# The real cart's numbers: axis caps 0.45 m/s and 1.0 rad/s, a 0.505 m track, ceiling 0.30.
FAST = replace(
    CFG,
    geometry=replace(CFG.geometry, track_width_m=0.505),
    max_speed_m_s=0.45,
    max_wheel_speed_m_s=0.30,
)


def _rim_speeds(bus: FakeBus, config: BaseConfig) -> tuple[float, float]:
    """The last command as (left, right) rim speeds in m/s, robot-forward positive."""
    ticks = bus.writes[-1][1]
    m_per_tick_s = config.geometry.m_per_tick
    return (
        ticks[LEFT] * config.left.direction * m_per_tick_s,
        ticks[RIGHT] * config.right.direction * m_per_tick_s,
    )


def test_a_turn_at_speed_keeps_its_arc_and_no_wheel_passes_the_ceiling(bus: FakeBus) -> None:
    """(0.30, 0.6) asks 0.45 of the outer wheel: both axes are scaled by one factor, so the arc
    (v / w = 0.5 m) is the one asked for, at 0.20 m/s instead of a cut outer wheel."""
    DiffDriveBase(bus, FAST).set_twist(Twist(linear=0.30, angular=0.6))
    left, right = _rim_speeds(bus, FAST)
    assert max(abs(left), abs(right)) == pytest.approx(0.30, abs=1e-3)
    v, w = (left + right) / 2, (right - left) / FAST.geometry.track_width_m
    assert v / w == pytest.approx(0.30 / 0.6, rel=1e-3)
    assert v == pytest.approx(0.30 * 0.30 / (0.30 + 0.6 * 0.2525), rel=1e-3)


def test_a_pivot_within_the_ceiling_is_untouched(bus: FakeBus) -> None:
    """A pivot at the yaw cap runs each rim at 1.0 x 0.2525 m/s, under 0.30: as asked."""
    DiffDriveBase(bus, FAST).set_twist(Twist(linear=0.0, angular=-1.0))
    left, right = _rim_speeds(bus, FAST)
    assert left == pytest.approx(0.2525, abs=1e-3) and right == pytest.approx(-0.2525, abs=1e-3)


def test_straight_above_the_ceiling_drives_at_the_ceiling_and_it_moves_live(bus: FakeBus) -> None:
    base = DiffDriveBase(bus, FAST)
    base.set_twist(Twist(linear=-0.40, angular=0.0))
    assert _rim_speeds(bus, FAST) == pytest.approx((-0.30, -0.30), abs=1e-3)
    base.max_wheel_speed_m_s = 0.20  # the base server's max_wheel_speed command
    base.set_twist(Twist(linear=0.40, angular=0.0))
    assert _rim_speeds(bus, FAST) == pytest.approx((0.20, 0.20), abs=1e-3)


def test_the_ceiling_scales_whole_twists_and_leaves_slow_ones_alone() -> None:
    from pepin.base import wheel_ceiling

    slow = Twist(0.2, 0.3)  # 0.2 + 0.3 x 0.25 = 0.275 m/s on the outer rim
    assert wheel_ceiling(slow, 0.25, 0.30) is slow
    fast = wheel_ceiling(Twist(-0.4, -0.8), 0.25, 0.30)  # reversing and turning: 0.6 m/s
    assert (fast.linear, fast.angular) == pytest.approx((-0.2, -0.4))


# -- BusWatchdog -------------------------------------------------------------


def test_watchdog_skips_then_stops_then_aborts() -> None:
    from pepin.base import BusWatchdog

    dog = BusWatchdog(stop_after_s=1.0, give_up_after_s=10.0)
    assert dog.failed(100.0) == "skip"  # first miss: just skip the tick
    assert dog.failed(100.5) == "skip"
    assert dog.failed(101.0) == "stop"  # one second down: command a stop, once
    assert dog.failed(101.5) == "skip"
    assert dog.failed(110.0) == "abort"
    assert dog.failures == 5


def test_watchdog_recovery_reports_the_outage_and_rearms_the_stop() -> None:
    from pepin.base import BusWatchdog

    dog = BusWatchdog()
    assert dog.recovered(0.0) is None  # nothing was wrong
    dog.failed(10.0)
    dog.failed(11.0)  # -> stop
    assert dog.recovered(11.4) == pytest.approx(1.4)
    assert dog.failed(20.0) == "skip"  # a new outage starts from scratch
    assert dog.failed(21.0) == "stop"
