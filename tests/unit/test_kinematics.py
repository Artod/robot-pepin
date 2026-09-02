import math

import pytest

from pepin.geometry import BaseGeometry
from pepin.kinematics import DiffDriveKinematics, Twist, WheelRates

GEOM = BaseGeometry(wheel_diameter_m=0.125, track_width_m=0.505, ticks_per_rev=4096)


@pytest.fixture
def kin() -> DiffDriveKinematics:
    return DiffDriveKinematics(GEOM)


def test_straight_drive_spins_both_wheels_equally(kin: DiffDriveKinematics) -> None:
    rates = kin.twist_to_wheels(Twist(linear=0.1, angular=0.0))
    assert rates.left == pytest.approx(rates.right)
    assert rates.left == pytest.approx(0.1 / GEOM.wheel_radius_m)


def test_left_turn_in_place_runs_right_wheel_forward(kin: DiffDriveKinematics) -> None:
    rates = kin.twist_to_wheels(Twist(linear=0.0, angular=1.0))
    assert rates.right > 0 > rates.left
    assert rates.right == pytest.approx(-rates.left)
    assert rates.right == pytest.approx(GEOM.track_width_m / 2 / GEOM.wheel_radius_m)


def test_round_trip_is_identity(kin: DiffDriveKinematics) -> None:
    twist = Twist(linear=0.23, angular=-0.7)
    back = kin.wheels_to_twist(kin.twist_to_wheels(twist))
    assert back.linear == pytest.approx(twist.linear)
    assert back.angular == pytest.approx(twist.angular)


def test_one_revolution_per_second_is_ticks_per_rev(kin: DiffDriveKinematics) -> None:
    assert kin.rad_s_to_ticks_s(2 * math.pi) == GEOM.ticks_per_rev
    assert kin.rad_s_to_ticks_s(-math.pi) == -GEOM.ticks_per_rev // 2


def test_wheels_to_twist_uses_both_wheels(kin: DiffDriveKinematics) -> None:
    twist = kin.wheels_to_twist(WheelRates(left=1.0, right=1.0))
    assert twist.angular == pytest.approx(0.0)
    assert twist.linear == pytest.approx(GEOM.wheel_radius_m)
