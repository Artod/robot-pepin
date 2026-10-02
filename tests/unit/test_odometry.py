import math

import pytest

from pepin.geometry import BaseGeometry
from pepin.kinematics import Twist
from pepin.odometry import (
    DiffDriveOdometry,
    EncoderUnwrapper,
    Pose2D,
    TwistFromPose,
    wrap_angle,
)

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


def test_twist_from_pose_primes_then_measures() -> None:
    """The first sample has nothing to difference; the second is the measured body twist."""
    estimator = TwistFromPose()
    assert estimator.update(Pose2D(0.0, 0.0, 0.0), 10.0) == Twist(0.0, 0.0)
    twist = estimator.update(Pose2D(0.1, 0.0, 0.2), 10.5)
    assert twist.linear == pytest.approx(0.2)
    assert twist.angular == pytest.approx(0.4)


def test_twist_from_pose_signs_a_reverse_step() -> None:
    """Driving backwards is a negative forward speed, not a positive distance."""
    estimator = TwistFromPose()
    estimator.update(Pose2D(0.0, 0.0, math.pi / 2.0), 0.0)
    twist = estimator.update(Pose2D(0.0, -0.05, math.pi / 2.0), 0.5)
    assert twist.linear == pytest.approx(-0.1)
    assert twist.angular == pytest.approx(0.0)


def test_twist_from_pose_wraps_the_heading_step() -> None:
    """A pivot across +-pi is a small turn, not a full circle at 12 rad/s."""
    estimator = TwistFromPose()
    estimator.update(Pose2D(0.0, 0.0, math.pi - 0.05), 0.0)
    twist = estimator.update(Pose2D(0.0, 0.0, -math.pi + 0.05), 0.1)
    assert twist.angular == pytest.approx(1.0)


def test_twist_from_pose_re_primes_after_a_gap() -> None:
    """A link that went quiet for seconds must not be divided by its own silence."""
    estimator = TwistFromPose(max_gap_s=0.5)
    estimator.update(Pose2D(0.0, 0.0, 0.0), 0.0)
    assert estimator.update(Pose2D(1.0, 0.0, 0.0), 3.0) == Twist(0.0, 0.0)
    twist = estimator.update(Pose2D(1.1, 0.0, 0.0), 3.1)
    assert twist.linear == pytest.approx(1.0)


def test_twist_from_pose_contract_the_cpp_bridge_mirrors() -> None:
    """The table the C++ port must reproduce line for line.

    The bridge that actually runs on the board is the C++ one
    (ros/pepin_base_cpp/include/pepin_base_cpp/twist_from_pose.hpp); that package has no test
    target, so this is the contract both sides implement: the prime, a forward step, a pivot in
    place, a signed backward step, a board that restarted its monotonic clock, a silence longer
    than ``max_gap_s`` and the sample right after it. Change the maths here first.
    """
    estimator = TwistFromPose(max_gap_s=1.0)
    samples = [
        # (pose, board stamp, expected forward m/s, expected yaw rate rad/s)
        (Pose2D(0.0, 0.0, 0.0), 100.00, 0.0, 0.0),  # primes: nothing to difference yet
        (Pose2D(0.01, 0.0, 0.0), 100.05, 0.2, 0.0),  # 1 cm forward in 50 ms
        (Pose2D(0.01, 0.0, 0.05), 100.10, 0.0, 1.0),  # pivot in place: 0.05 rad in 50 ms
        (Pose2D(0.0, 0.0, 0.05), 100.15, -0.19975, 0.0),  # pushed back: forward is signed
        (Pose2D(0.0, 0.0, 0.05), 99.00, 0.0, 0.0),  # the board's clock restarted: no twist
        (Pose2D(0.0, 0.0, 0.05), 101.00, 0.0, 0.0),  # 2 s > max_gap_s: re-primes instead
        (Pose2D(0.01, 0.0, 0.05), 101.05, 0.19975, 0.0),  # measuring again on the next line
    ]
    for pose, stamp, forward, yaw_rate in samples:
        twist = estimator.update(pose, stamp)
        assert twist.linear == pytest.approx(forward, abs=1e-6), f"forward at t={stamp}"
        assert twist.angular == pytest.approx(yaw_rate, abs=1e-6), f"yaw rate at t={stamp}"

    across_pi = TwistFromPose()
    assert across_pi.update(Pose2D(0.0, 0.0, math.pi - 0.05), 200.00) == Twist(0.0, 0.0)
    twist = across_pi.update(Pose2D(0.0, 0.0, -math.pi + 0.05), 200.05)
    assert twist.angular == pytest.approx(2.0)  # 0.1 rad across +-pi, not a full circle
