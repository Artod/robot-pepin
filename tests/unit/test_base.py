"""The base driver against a fake bus: sign handling, clamping, encoder wrap."""

import math

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
