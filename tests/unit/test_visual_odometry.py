"""What may reach the EKF from the camera's own odometry, and what the drift at rest means."""

import math
from itertools import pairwise

import pytest

from pepin.visual_odometry import (
    LOST_VARIANCE,
    PublishCap,
    RestWatch,
    VoGate,
    VoPose,
    VoTrack,
    is_lost,
    planar_covariance,
)


def _gated(gate: VoGate, track: VoTrack, poses: list[VoPose]) -> list[VoPose]:
    """Run poses through a gate and its track exactly as pepin_bringup.visual_odometry does;
    returns what would have been published."""
    published = []
    for pose in poses:
        if gate.admit(pose, lost=False) is not None:
            track.anchor(gate.anchor)
            continue
        published.append(track.advance(pose))
    return published


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
    """rtabmap re-initialising puts the pose back at its origin. It is refused — by the origin
    check, which names it, or by the speed ceiling when it lands far enough out — and it becomes
    the anchor, or every pose after it would be a jump too and the gate would go deaf."""
    gate = VoGate()
    gate.admit(VoPose(0.0, 0.0, 0.0, 0.0), lost=False)
    gate.admit(VoPose(1.0, 0.2, 0.0, 0.0), lost=False)
    refused = gate.admit(VoPose(1.1, 0.0, 0.0, 0.0), lost=False)  # back to the origin
    assert refused is not None and "origin" in refused
    assert gate.admit(VoPose(1.2, 0.01, 0.0, 0.0), lost=False) is None, "deaf after one restart"


def test_a_jump_across_a_gap_is_refused_even_when_it_looks_slow() -> None:
    """The speed and turn ceilings are ratios: divided by seconds, a metre-scale jump is a
    walking pace. That is exactly what a respawned rgbd_odometry produces — the pose back at its
    origin after a gap of seconds — so the gap is refused on its own evidence and re-anchors."""
    gate = VoGate(max_speed_m_s=1.0, max_gap_s=1.0)
    gate.admit(VoPose(0.0, 0.0, 0.0, 0.0), lost=False)
    gate.admit(VoPose(0.1, 0.02, 0.0, 0.0), lost=False)
    # 0.5 m in 4 s is 0.125 m/s: under the speed ceiling, and pure fiction.
    refused = gate.admit(VoPose(4.1, 0.52, 0.0, 0.0), lost=False)
    assert refused is not None and "gap" in refused
    assert gate.admit(VoPose(4.2, 0.53, 0.0, 0.0), lost=False) is None, "the gap re-anchored"


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


def test_a_refused_jump_never_reaches_the_filter_inside_the_next_pose() -> None:
    """The bug of 2026-09-14: the gate re-anchors on a tracking restart, the EKF does not, and
    the next pose that passes carries the whole discontinuity. The published track walks the
    admitted centimetres and stands still across the jump."""
    track = VoTrack()
    poses = [
        VoPose(stamp=0.0, x=3.00, y=1.00, yaw=0.0),
        VoPose(stamp=0.1, x=3.02, y=1.00, yaw=0.0),
        VoPose(stamp=0.2, x=0.00, y=0.00, yaw=0.0),  # rtabmap re-initialised at its origin
        VoPose(stamp=0.3, x=0.02, y=0.00, yaw=0.0),
        VoPose(stamp=0.4, x=0.04, y=0.00, yaw=0.0),
    ]
    published = _gated(VoGate(), track, poses)
    assert len(published) == 4, "only the reset itself is dropped"
    steps = [math.hypot(b.x - a.x, b.y - a.y) for a, b in pairwise(published)]
    assert max(steps) == pytest.approx(0.02, abs=1e-9), (
        "the largest step the filter can difference is a real one, not the 3.16 m of the reset"
    )
    assert published[-1].x == pytest.approx(0.06), "the two centimetres before the reset count"


def test_the_track_stands_still_across_a_gap_and_keeps_walking_after() -> None:
    """A stalled camera re-anchors both the gate and the track: the seconds nobody measured are
    not motion, and the poses after them are."""
    track = VoTrack()
    published = _gated(
        VoGate(max_gap_s=1.0),
        track,
        [
            VoPose(stamp=0.0, x=0.0, y=0.0, yaw=0.0),
            VoPose(stamp=0.1, x=0.1, y=0.0, yaw=0.0),
            VoPose(stamp=5.0, x=4.0, y=0.0, yaw=0.0),  # a 4.9 s gap: dropped, re-anchors
            VoPose(stamp=5.1, x=4.1, y=0.0, yaw=0.0),
        ],
    )
    assert [round(p.x, 6) for p in published] == [0.0, 0.1, 0.2]
    assert track.pose[0] == pytest.approx(0.2)


def test_a_reset_to_rtabmap_origin_is_refused_even_when_it_is_close() -> None:
    """Odom/ResetCountdown puts a lost tracking back at the origin; within 11 cm of it the speed
    ceiling calls that a walking pace, so the origin itself is the evidence."""
    gate = VoGate(reset_radius_m=0.05)
    assert gate.admit(VoPose(stamp=0.0, x=0.09, y=0.0, yaw=0.0), lost=False) is None
    refused = gate.admit(VoPose(stamp=0.1, x=0.0, y=0.0, yaw=0.0), lost=False)
    assert refused is not None and "origin" in refused
    assert gate.anchor is not None and gate.anchor.x == 0.0, "and it re-anchors there"
    assert gate.admit(VoPose(stamp=0.2, x=0.01, y=0.0, yaw=0.0), lost=False) is None
    off = VoGate(reset_radius_m=0.0)
    assert off.admit(VoPose(stamp=0.0, x=0.09, y=0.0, yaw=0.0), lost=False) is None
    assert off.admit(VoPose(stamp=0.1, x=0.0, y=0.0, yaw=0.0), lost=False) is None


def test_a_pose_that_starts_at_the_origin_is_not_a_reset() -> None:
    """Every session's first poses sit on rtabmap's origin; only leaving it and coming back is a
    re-initialisation."""
    gate = VoGate()
    assert gate.admit(VoPose(stamp=0.0, x=0.0, y=0.0, yaw=0.0), lost=False) is None
    assert gate.admit(VoPose(stamp=0.1, x=0.01, y=0.0, yaw=0.0), lost=False) is None


def test_the_publish_cap_holds_the_rate_and_refuses_a_stamp_that_goes_backwards() -> None:
    """Three hertz is three hertz ON AVERAGE, a bunch of two after a pause is not refused, and two
    messages the filter cannot order are a division by a gap of zero."""
    cap = PublishCap(hz=3.0, burst=2.0)
    assert cap.refuse(100.0) is None
    assert cap.refuse(100.1) is None, "the second of a bunch: the budget is an average"
    refused = cap.refuse(100.2)
    assert refused is not None and "budget" in refused, "a third back to back is over it"
    assert cap.refuse(100.5) is None, "a third of a second refills one pose at 3 Hz"
    repeated = cap.refuse(100.5)
    assert repeated is not None and "behind" in repeated
    behind = cap.refuse(100.2)
    assert behind is not None and "behind" in behind
    out = sum(cap.refuse(101.0 + i * 0.05) is None for i in range(200))
    assert out <= 3.0 * 10.0 + 2, "ten seconds at 20 poses a second: no more than 3 Hz and a burst"


def test_the_cap_at_zero_publishes_every_pose_but_still_orders_them() -> None:
    """0 Hz is the old behaviour; the stamp rule is not a rate and never switches off."""
    cap = PublishCap(hz=0.0)
    assert cap.refuse(10.0) is None
    assert cap.refuse(10.001) is None
    assert cap.refuse(10.001) is not None


def test_the_dynamic_covariance_adds_the_scale_s_share_of_the_step() -> None:
    """A frame the cart barely moved in is worth the registration's own sigma; a long step is
    worth what the network's depth scale is worth (SCALE_ERROR of the step), the two added in
    quadrature; a registration claiming millimetres is floored."""
    from pepin.visual_odometry import SCALE_ERROR, SIGMA_FLOOR_M, scaled_covariance

    still = scaled_covariance(0.0066**2, 0.0, yaw_sigma_deg=5.0)
    assert math.isclose(math.sqrt(still[0]), 0.0066, rel_tol=1e-6)
    stepped = scaled_covariance(0.0066**2, 0.20, yaw_sigma_deg=5.0)
    assert math.isclose(math.sqrt(stepped[0]), math.hypot(0.0066, SCALE_ERROR * 0.20), rel_tol=1e-6)
    assert math.sqrt(stepped[0]) > 3 * math.sqrt(still[0]), "the step dominates a long frame"
    floored = scaled_covariance(1e-12, 0.0, yaw_sigma_deg=5.0)
    assert math.isclose(math.sqrt(floored[0]), SIGMA_FLOOR_M, rel_tol=1e-6)
    assert floored[35] > 0.0 and floored[14] > 1e3, "yaw measured, z and roll left to others"
