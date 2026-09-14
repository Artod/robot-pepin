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


def test_a_runaway_odometry_frame_is_refused_while_its_twist_says_the_cart_stands_still() -> None:
    """2026-09-14: the EKF flew to 43 km at 60 m/s after a bad /vo input while the wheels
    reported next to nothing. Two samples of the same topic are enough to say so."""
    from pepin.odometry import Pose2D, RunawayWatch

    watch = RunawayWatch()
    assert not watch.judge(Pose2D(0.0, 0.0, 0.0), 100.0, 0.0, 0.0), "the first sample: no reference"
    watch.adopt(Pose2D(0.0, 0.0, 0.0), 100.0)
    assert watch.judge(Pose2D(3493.7, -395.6, 0.0), 100.05, 0.01, 0.0), "43 km in 50 ms"
    assert watch.streak == 1
    assert watch.judge(Pose2D(3493.8, -395.6, 0.0), 100.10, 0.01, 0.0), "still away: still refused"
    assert watch.streak == 2, "one episode, said once"
    assert not watch.judge(Pose2D(0.01, 0.0, 0.0), 100.15, 0.01, 0.0), "back where it was left"


def test_a_twist_that_claims_more_than_the_cart_can_do_cannot_justify_a_jump() -> None:
    """The EKF integrates its own velocity, so during the flight the twist reported the flight:
    60 m/s is not a measurement of a cart whose top speed is 0.3."""
    from pepin.odometry import Pose2D, RunawayWatch

    watch = RunawayWatch()
    watch.adopt(Pose2D(), 0.0)
    assert watch.judge(Pose2D(3.0, 0.0, 0.0), 0.05, 60.0, 0.0)


def test_real_driving_and_the_jitter_of_that_tape_are_never_refused() -> None:
    """The guard must not touch a drive: 0.3 m/s is this cart flat out, and the worst single
    step of the runaway tape itself (0.045 m over 55 ms, 0.825 m/s) is under the threshold."""
    from pepin.odometry import Pose2D, RunawayWatch

    watch = RunawayWatch()
    watch.adopt(Pose2D(), 0.0)
    assert not watch.judge(Pose2D(0.015, 0.0, 0.0), 0.05, 0.30, 0.0), "flat out, 15 mm per sample"
    watch.adopt(Pose2D(0.015, 0.0, 0.0), 0.05)
    assert not watch.judge(Pose2D(0.060, 0.0, 0.0), 0.105, 0.009, 0.041), "the tape's worst step"
    watch.adopt(Pose2D(0.060, 0.0, 0.0), 0.105)
    assert not watch.judge(Pose2D(0.30, 0.0, 0.0), 0.105, 0.0, 0.0), "no time passed: not judged"
    assert watch.streak == 0


def test_a_push_by_hand_faster_than_the_cart_drives_is_not_a_runaway() -> None:
    """A metre per second of human arm moves the world too; the frame is right and the guard
    stays out of it."""
    from pepin.odometry import Pose2D, RunawayWatch

    watch = RunawayWatch()
    watch.adopt(Pose2D(), 0.0)
    assert not watch.judge(Pose2D(0.05, 0.0, 0.0), 0.05, 0.2, 0.1), "1.0 m/s, under the limit"


def test_the_history_is_fed_everything_but_the_runaway_and_the_switch_puts_it_back() -> None:
    """What the tracker's node actually calls: the sample reaches the odometry history, or it
    does not and the carry keeps the last pose that made sense."""
    from pepin.odometry import Pose2D, RunawayWatch

    class FakeHistory:
        def __init__(self) -> None:
            self.trail: list[tuple[float, Pose2D]] = []

        def add(self, stamp: float, pose: Pose2D) -> None:
            self.trail.append((stamp, pose))

    history, watch = FakeHistory(), RunawayWatch()
    assert watch.feed(history, Pose2D(), 0.0, 0.0, 0.0, True)
    assert not watch.feed(history, Pose2D(3493.7, -395.6, 0.0), 0.05, 0.01, 0.0, True)
    assert watch.feed(history, Pose2D(0.01, 0.0, 0.0), 0.10, 0.01, 0.0, True)
    assert [round(s, 2) for s, _ in history.trail] == [0.0, 0.10], "the runaway never got in"

    loose, guarded = FakeHistory(), RunawayWatch()
    assert guarded.feed(loose, Pose2D(), 0.0, 0.0, 0.0, False)
    assert guarded.feed(loose, Pose2D(3493.7, -395.6, 0.0), 0.05, 0.01, 0.0, False)
    assert len(loose.trail) == 2, "the guard off: every sample carried, as before"
    assert not guarded.feed(loose, Pose2D(0.0, 0.0, 0.0), 0.10, 0.01, 0.0, True), (
        "and the reference followed the frame, so the switch can be thrown back live"
    )
