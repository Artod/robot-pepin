"""What may reach the EKF from the camera's own odometry, and what the drift at rest means."""

import math

import pytest

from pepin.visual_odometry import (
    LOST_VARIANCE,
    RestWatch,
    VoGate,
    VoPose,
    is_lost,
    planar_covariance,
)


def _covariance(variance: float = 0.001) -> list[float]:
    """A 6x6 row-major covariance with ``variance`` on the diagonal, as rtabmap publishes."""
    matrix = [0.0] * 36
    for i in range(6):
        matrix[i * 6 + i] = variance
    return matrix


def test_a_lost_frame_is_the_one_rtabmap_marked() -> None:
    """rtabmap publishes a message per frame either way and writes 9999 on the diagonal of the
    ones it did not measure; a covariance too short to read is not a measurement either."""
    assert not is_lost(_covariance())
    assert is_lost(_covariance(LOST_VARIANCE))
    assert is_lost([0.001] * 12), "a truncated covariance is not a pose"
    half_lost = _covariance()
    half_lost[35] = LOST_VARIANCE  # yaw alone
    assert is_lost(half_lost)


def test_the_constant_covariance_says_nothing_about_the_axes_nobody_measured() -> None:
    """x, y and yaw carry the documented sigmas; z, roll and pitch carry a variance so large
    that robot_localization can only ignore them."""
    matrix = planar_covariance(0.02, 5.0)
    assert len(matrix) == 36
    assert matrix[0] == pytest.approx(0.0004) and matrix[7] == pytest.approx(0.0004)
    assert matrix[35] == pytest.approx(math.radians(5.0) ** 2)
    for unfused in (14, 21, 28):  # z, roll, pitch
        assert matrix[unfused] >= 1e6
    assert not any(v for i, v in enumerate(matrix) if i % 7), "only the diagonal is filled"
    with pytest.raises(ValueError, match="a sigma is positive"):
        planar_covariance(0.0, 5.0)


def test_the_gate_drops_a_lost_frame_and_lets_a_walking_pace_through() -> None:
    gate = VoGate()
    assert gate.admit(VoPose(0.0, 0.0, 0.0, 0.0), lost=True) == "rtabmap lost the frame"
    assert gate.admit(VoPose(1.0, 0.0, 0.0, 0.0), lost=False) is None, "the first pose is a start"
    # 2.5 cm in an eighth of a second: 0.2 m/s, the cart's own pace.
    assert gate.admit(VoPose(1.125, 0.025, 0.0, 0.0), lost=False) is None


def test_a_tracking_restart_costs_one_sample_and_never_the_session() -> None:
    """rtabmap re-initialising puts the pose back at its origin. The jump is refused — and it
    becomes the anchor, or every pose after it would be a jump too and the gate would go deaf."""
    gate = VoGate()
    gate.admit(VoPose(0.0, 0.0, 0.0, 0.0), lost=False)
    gate.admit(VoPose(1.0, 0.2, 0.0, 0.0), lost=False)
    refused = gate.admit(VoPose(1.1, 0.0, 0.0, 0.0), lost=False)  # back to the origin
    assert refused is not None and "jump" in refused
    assert gate.admit(VoPose(1.2, 0.01, 0.0, 0.0), lost=False) is None, "deaf after one restart"


def test_the_gate_refuses_a_turn_no_cart_could_make_and_a_pose_out_of_order() -> None:
    gate = VoGate(max_turn_deg_s=180.0)
    gate.admit(VoPose(0.0, 0.0, 0.0, 0.0), lost=False)
    spun = gate.admit(VoPose(0.1, 0.0, 0.0, math.radians(90.0)), lost=False)
    assert spun is not None and "turn" in spun
    gate = VoGate()
    gate.admit(VoPose(2.0, 0.0, 0.0, 0.0), lost=False)
    assert "out of order" in str(gate.admit(VoPose(1.9, 0.0, 0.0, 0.0), lost=False))
    # ...and the anchor did not move backwards with it: the next honest pose still passes.
    assert gate.admit(VoPose(2.1, 0.01, 0.0, 0.0), lost=False) is None


def test_drift_is_measured_only_while_the_wheels_report_a_hard_zero() -> None:
    """The cart on its charger is the only truth available: what the camera walks there is its
    own. A drive, and the stretch is thrown away — the motion was real."""
    watch = RestWatch(settle_s=1.0)
    watch.wheels(0.0, 0.0, 0.0)
    watch.pose(VoPose(0.0, 0.0, 0.0, 0.0), now=0.0)
    watch.pose(VoPose(60.0, 0.004, 0.003, math.radians(0.1)), now=60.0)
    drift = watch.drift
    assert drift is not None
    assert drift.metres == pytest.approx(0.005)
    assert drift.degrees == pytest.approx(0.1, abs=1e-6)
    assert drift.seconds == pytest.approx(60.0)
    assert "0.5 cm" in str(drift)
    watch.wheels(61.0, 0.2, 0.0)  # a drive starts
    watch.pose(VoPose(61.5, 0.3, 0.0, 0.0), now=61.5)
    assert watch.drift is None, "what moved while the wheels turned is not drift"
    assert watch.worst is not None and watch.worst.metres == pytest.approx(0.005)


def test_without_a_single_wheel_message_there_is_no_such_thing_as_drift_at_rest() -> None:
    """The number that decides whether this source may be fused is "what it walked while the
    wheels stood still". With no /odom — a dead bridge route, a QoS that never matched — nothing
    says the wheels stood still, and a drift printed then would be a drive's, not a drift's."""
    watch = RestWatch(settle_s=0.0)
    watch.pose(VoPose(0.0, 0.0, 0.0, 0.0), now=0.0)
    watch.pose(VoPose(1.0, 0.5, 0.0, 0.0), now=1.0)  # half a metre: the cart was driving
    assert watch.drift is None and watch.worst is None
    assert "no /odom" in watch.report()
    watch.wheels(1.5, 0.0, 0.0)  # the route comes up: only now is stillness known
    watch.pose(VoPose(2.0, 0.5, 0.0, 0.0), now=2.0)
    watch.pose(VoPose(3.0, 0.502, 0.0, 0.0), now=3.0)
    assert watch.drift is not None and watch.drift.metres == pytest.approx(0.002)


def test_the_settling_second_is_counted_on_the_receiving_clock() -> None:
    """A cart that has just stopped is still rocking; that motion is the body's, not the
    camera's. The window is measured on the node's own clock because the wheels are stamped by
    the board and the poses by the laptop — here two clocks a minute apart, the way they were on
    2026-09-04 — and a window compared across them would count the rocking as drift."""
    watch = RestWatch(settle_s=1.0)
    watch.wheels(10.0, 0.3, 0.0)  # the wheels turn; the board's stamps say 70.0
    watch.pose(VoPose(70.5, 0.05, 0.0, 0.0), now=10.5)
    assert watch.drift is None
    watch.pose(VoPose(71.5, 0.10, 0.0, 0.0), now=11.5)  # past the settling time: the anchor
    watch.pose(VoPose(72.5, 0.101, 0.0, 0.0), now=12.5)
    assert watch.drift is not None and watch.drift.metres == pytest.approx(0.001)
    assert watch.drift.seconds == pytest.approx(1.0), "the duration is the visual stamps'"


def test_a_restart_starts_the_rest_measurement_again() -> None:
    """The origin moved under the watch: a drift measured across it would be the restart's."""
    watch = RestWatch(settle_s=0.0)
    watch.wheels(0.0, 0.0, 0.0)
    watch.pose(VoPose(0.0, 0.0, 0.0, 0.0), now=0.0)
    watch.pose(VoPose(1.0, 0.002, 0.0, 0.0), now=1.0)
    assert watch.drift is not None
    watch.restart()
    assert watch.drift is None
    watch.pose(VoPose(2.0, 5.0, 0.0, 0.0), now=2.0)  # the new origin, metres from the old one
    watch.pose(VoPose(3.0, 5.001, 0.0, 0.0), now=3.0)
    assert watch.drift is not None and watch.drift.metres == pytest.approx(0.001)
    assert "at rest" in watch.report()


def test_a_moving_cart_reports_no_drift_at_all() -> None:
    watch = RestWatch()
    watch.wheels(0.0, 0.0, 0.5)  # turning in place: the wheels say so
    watch.pose(VoPose(0.1, 0.0, 0.0, 0.0), now=0.1)
    watch.pose(VoPose(0.2, 0.05, 0.0, 0.0), now=0.2)
    assert watch.drift is None
    assert "moving" in watch.report()
