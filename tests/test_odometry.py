import math

import pytest

from pepin.geometry import BaseGeometry
from pepin.odometry import DiffDriveOdometry, EncoderUnwrapper, Pose2D, wrap_angle

GEOM = BaseGeometry(wheel_diameter_m=0.125, track_width_m=0.5, ticks_per_rev=4096)


def test_wrap_angle_maps_into_half_open_interval() -> None:
    assert wrap_angle(3 * math.pi) == pytest.approx(math.pi)
    assert wrap_angle(-3 * math.pi / 2) == pytest.approx(math.pi / 2)
    assert wrap_angle(0.3) == pytest.approx(0.3)


def test_straight_line_integrates_along_heading() -> None:
    odom = DiffDriveOdometry(GEOM, Pose2D(theta=math.pi / 2))
    pose = odom.update(1.0, 1.0)
    assert pose.x == pytest.approx(0.0, abs=1e-12)
    assert pose.y == pytest.approx(1.0)
    assert pose.theta == pytest.approx(math.pi / 2)


def test_quarter_circle_arc_lands_on_the_geometric_endpoint() -> None:
    # Arc of radius 1 m through 90 degrees: the outer wheel travels more.
    odom = DiffDriveOdometry(GEOM)
    dtheta = math.pi / 2
    ds = 1.0 * dtheta
    half = GEOM.track_width_m / 2
    pose = odom.update(ds - half * dtheta, ds + half * dtheta)
    assert pose.x == pytest.approx(1.0)
    assert pose.y == pytest.approx(1.0)
    assert pose.theta == pytest.approx(math.pi / 2)


def test_many_small_steps_match_one_big_step() -> None:
    big = DiffDriveOdometry(GEOM)
    small = DiffDriveOdometry(GEOM)
    big.update(0.6, 0.8)
    for _ in range(1000):
        small.update(0.0006, 0.0008)
    assert small.pose.x == pytest.approx(big.pose.x, abs=1e-6)
    assert small.pose.y == pytest.approx(big.pose.y, abs=1e-6)
    assert small.pose.theta == pytest.approx(big.pose.theta, abs=1e-9)


def test_full_spin_in_place_returns_to_heading_zero() -> None:
    odom = DiffDriveOdometry(GEOM)
    arc = math.pi * GEOM.track_width_m  # circumference of the turning circle, radius L/2
    for _ in range(4):
        odom.update(-arc / 4, arc / 4)
    assert odom.pose.x == pytest.approx(0.0, abs=1e-9)
    assert odom.pose.theta == pytest.approx(0.0, abs=1e-9)


def test_unwrapper_first_reading_is_zero_delta() -> None:
    assert EncoderUnwrapper(4096).delta(1234) == 0


def test_unwrapper_handles_forward_and_backward_wraps() -> None:
    unw = EncoderUnwrapper(4096)
    unw.delta(4090)
    assert unw.delta(6) == 12  # 4090 -> 4095, 0 -> 6
    assert unw.delta(4094) == -8


def test_unwrapper_plain_deltas() -> None:
    unw = EncoderUnwrapper(4096)
    unw.delta(100)
    assert unw.delta(150) == 50
    assert unw.delta(120) == -30
