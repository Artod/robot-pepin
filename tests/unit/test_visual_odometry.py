"""What may reach the EKF from the camera's own odometry, and what the drift at rest means."""

import math
from itertools import pairwise
from typing import Any

import numpy as np
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


def _body_velocity(a: VoPose, b: VoPose) -> tuple[float, float]:
    """What robot_localization's differential mode fuses from two published poses: the step
    ``a^-1 * b`` in a's body frame, divided by the gap (ros_filter.cpp, prev.inverseTimes(cur))."""
    dx, dy = b.x - a.x, b.y - a.y
    c, s = math.cos(a.yaw), math.sin(a.yaw)
    dt = b.stamp - a.stamp
    return (c * dx + s * dy) / dt, (-s * dx + c * dy) / dt


def _straight(t0: float, x0: float, y0: float, yaw: float, n: int) -> list[VoPose]:
    """``n`` source poses 0.1 s apart driving 2 cm a step (0.2 m/s) along ``yaw``."""
    return [
        VoPose(t0 + 0.1 * i, x0 + 0.02 * i * math.cos(yaw), y0 + 0.02 * i * math.sin(yaw), yaw)
        for i in range(n)
    ]


def test_a_source_frame_that_starts_turned_still_reads_as_driving_forward() -> None:
    """A VIO's gravity frame has whatever yaw the chip and the pan gave it at init: a cart driving
    straight ahead with the source at yaw 90 deg is forward motion to the EKF, not sideways
    (the track summed in the source's axes said vy +0.200 m/s, 2026-10-02)."""
    published = _gated(VoGate(), VoTrack(), _straight(0.0, 1.0, -2.0, math.pi / 2, 6))
    for a, b in pairwise(published):
        vx, vy = _body_velocity(a, b)
        assert vx == pytest.approx(0.200, abs=1e-9)
        assert vy == pytest.approx(0.000, abs=1e-9)


def test_a_turn_inside_a_gated_interval_leaves_no_sideways_velocity_behind() -> None:
    """A 30 deg body turn while the gaze gate withheld the poses: the track stands still across
    it, and after it the cart drives along its new heading — the EKF must read forward motion,
    not vx 0.173 / vy 0.100 for the rest of the session. The withheld turn reaches nobody, its
    yaw included: the track's heading stays where it was (the gyro owns the turn) and the steps
    after it are body steps, so the differenced velocity does not care."""
    gate, track = VoGate(max_gap_s=1.0), VoTrack()
    published = _gated(gate, track, _straight(0.0, 0.0, 0.0, 0.0, 4))
    turn = [VoPose(0.4 + 0.1 * i, 0.06, 0.0, math.radians(10 * (i + 1))) for i in range(3)]
    leg = _straight(0.7, 0.06, 0.0, math.radians(30.0), 5)
    for pose in [*turn, leg[0]]:  # withheld: each only re-anchors, as visual_odometry._gated does
        gate.reanchor(pose)
        track.anchor(gate.anchor)
    after = _gated(gate, track, leg[1:])
    for a, b in pairwise(after):
        vx, vy = _body_velocity(a, b)
        assert vx == pytest.approx(0.200, abs=1e-9)
        assert vy == pytest.approx(0.000, abs=1e-9)
    assert track.pose[2] == pytest.approx(0.0, abs=1e-12), "the gated turn is not differenced"
    assert published[-1].x == pytest.approx(0.06), "the leg before the turn is untouched"


def test_the_track_replays_body_steps_from_its_own_heading() -> None:
    """A source turned 90 deg against the track: a step to the source's +y (its forward) lands on
    the track's +x; a turn then rotates every later step with the track."""
    track = VoTrack()
    track.advance(VoPose(0.0, 5.0, 5.0, math.pi / 2))
    first = track.advance(VoPose(0.1, 5.0, 5.1, math.pi / 2))
    assert (first.x, first.y) == pytest.approx((0.1, 0.0), abs=1e-12)
    track.advance(VoPose(0.2, 5.0, 5.1, math.pi))  # a 90 deg left turn in place
    second = track.advance(VoPose(0.3, 4.9, 5.1, math.pi))  # forward again along the source's -x
    assert (second.x, second.y) == pytest.approx((0.1, 0.1), abs=1e-12)
    assert second.yaw == pytest.approx(math.pi / 2)


# ---- the visual-inertial source --------------------------------------------------------------
def _rot_z(angle: float) -> Any:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def _rot_y(angle: float) -> Any:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _base_to_imu(pan: float, tilt: float) -> Any:
    """base_link <- head_imu through the neck as neck.hpp chains it: the pan pivot over the front
    axle, the tilt axis 1.134 m up, the lens 0.025 m ahead and 0.086 m above it, the mount pitch
    23.8 deg down, and the IMU 4 cm beside the left eye, its axes permuted (a chip glued on its
    side)."""
    from pepin.visual_odometry import homogeneous

    pan_t = homogeneous(_rot_z(pan), (0.0, 0.0, 0.0))
    tilt_axis = homogeneous(np.eye(3), (0.0, 0.0, 1.134))
    tilt_t = homogeneous(_rot_y(tilt + math.radians(23.8)), (0.0, 0.0, 0.0))
    lens = homogeneous(np.eye(3), (0.025, 0.0, 0.086))
    chip = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    imu = homogeneous(chip, (0.0, 0.0305, 0.04))
    return pan_t @ tilt_axis @ tilt_t @ lens @ imu


def test_a_parked_cart_with_its_head_panning_stands_still_in_the_composed_pose() -> None:
    """T_G_I swings with the pan (the IMU on its lever arms), T_I_B swings back through the
    neck's TF chain: the composed base pose stands still to well under a millimetre."""
    from pepin.visual_odometry import compose_base_pose, homogeneous

    t_g_b = homogeneous(_rot_z(math.radians(37.0)), (1.5, -0.4, 0.0))  # G's arbitrary yaw
    poses = []
    for i in range(19):
        pan, tilt = math.radians(-45.0 + 5.0 * i), math.radians(10.0 * math.sin(i))
        t_b_i = _base_to_imu(pan, tilt)
        t_g_i = t_g_b @ t_b_i
        poses.append(compose_base_pose(t_g_i, np.linalg.inv(t_b_i), stamp=0.1 * i))
    xs = [p.x for p in poses]
    ys = [p.y for p in poses]
    assert max(xs) - min(xs) < 1e-3 and max(ys) - min(ys) < 1e-3
    assert poses[0].x == pytest.approx(1.5) and poses[0].y == pytest.approx(-0.4)
    assert all(p.yaw == pytest.approx(math.radians(37.0)) for p in poses)


def test_a_gravity_frame_yawed_90_deg_still_reads_as_forward_motion_through_the_track() -> None:
    """OpenVINS's G has whatever yaw init gave it: a cart driving straight ahead reads in G as
    motion along G's y. Composed and summed in SE(2), the EKF is told vx 0.2, vy 0."""
    from pepin.visual_odometry import compose_base_pose, homogeneous

    t_b_i = _base_to_imu(math.radians(30.0), 0.0)  # the head looking 30 deg left
    gate, track = VoGate(), VoTrack()
    published = []
    for i in range(6):
        t_g_b = homogeneous(_rot_z(math.pi / 2), (0.3, 0.02 * i, 0.0))
        pose = compose_base_pose(t_g_b @ t_b_i, np.linalg.inv(t_b_i), stamp=0.1 * i)
        assert gate.admit(pose, lost=False) is None
        published.append(track.advance(pose))
    for a, b in pairwise(published):
        vx, vy = _body_velocity(a, b)
        assert vx == pytest.approx(0.2, abs=1e-9) and vy == pytest.approx(0.0, abs=1e-9)


def test_the_vio_health_counts_a_step_of_its_covariance_as_a_reinit() -> None:
    """The marginal covariance only grows; a drop (or a jump) by 10x is a re-initialisation."""
    from pepin.visual_odometry import VioHealth

    health = VioHealth()
    growing = [_covariance(v) for v in (1e-4, 1.2e-4, 1.5e-4, 2e-4)]
    assert not any(health.observe(c) for c in growing)
    assert health.observe(_covariance(1e-6)), "re-initialised: back to the init covariance"
    assert health.reinits == 1
    assert not health.observe([0.0] * 12), "a truncated covariance says nothing"


def test_the_three_lost_rules() -> None:
    """(a) 0.25 m/s against wheels at 0.05 for a second, the board not at rest; (b) creeping at
    0.05 m/s while /zupt says rest; (c) 12 features for a second while moving. Each episode is
    counted once."""
    from pepin.visual_odometry import VioLost

    lost = VioLost(lost_speed_m_s=0.1, lost_s=1.0, min_features=20)
    lost.wheels(0.0, 0.05)
    assert lost.check(0.0, 0.25) is None, "a disagreement has to last"
    lost.wheels(0.5, 0.05)
    assert lost.check(0.5, 0.25) is None
    lost.wheels(1.0, 0.05)
    reason = lost.check(1.0, 0.25)
    assert reason is not None and "wheels" in reason
    lost.wheels(1.1, 0.05)
    assert lost.check(1.1, 0.25) is not None, "the same episode"
    lost.wheels(1.2, 0.20)
    assert lost.check(1.2, 0.22) is None, "agreeing again"
    lost.zupt(2.0)
    lost.wheels(2.0, 0.0)
    rest = lost.check(2.0, 0.05)
    assert rest is not None and "/zupt" in rest
    assert lost.check(2.1, 0.01) is None, "still at rest, and the VIO agrees"
    lost.features(3.0, 12)
    assert lost.check(3.0, 0.0) is None, "few features have to last, as a disagreement does"
    lost.features(4.0, 12)
    few = lost.check(4.0, 0.0)
    assert few is not None and "12 features" in few and "while moving" in few
    assert lost.counts == {"wheels": 1, "rest": 1, "features": 1}
    assert "lost 3 (wheels 1, rest 1, features 1)" in lost.report()
    silent = VioLost()
    assert silent.check(10.0, 0.5) is None, "no witness heard: no verdict"


def test_body_velocity_is_the_step_in_the_heading_it_starts_from() -> None:
    from pepin.visual_odometry import body_velocity

    a = VoPose(stamp=0.0, x=1.0, y=1.0, yaw=math.pi / 2)
    b = VoPose(stamp=0.5, x=1.0, y=1.1, yaw=math.pi / 2)  # 10 cm along G's y = straight ahead
    forward, left = body_velocity(a, b) or (math.nan, math.nan)
    assert forward == pytest.approx(0.2) and left == pytest.approx(0.0, abs=1e-12)
    assert body_velocity(b, a) is None, "a stamp that does not advance says nothing"


def test_the_guard_refuses_what_no_cart_makes_and_what_the_wheels_deny() -> None:
    """Over 1 m/s is refused whatever the wheels say; 0.6 m/s sideways against still wheels is
    refused by the wheel rule; 0.3 m/s forward with the wheels at 0.3 passes and ends the run."""
    from pepin.visual_odometry import VioGuard

    guard = VioGuard(max_speed_m_s=1.0, wheel_diff_m_s=0.5)
    assert guard.check(0.0, None) is None and guard.consecutive == 0, "the first: not judged"
    fast = guard.check(0.0, (3.0, 0.0))
    assert fast is not None and "3.00 m/s" in fast, "no wheels heard: the speed rule alone"
    assert guard.check(0.0, (0.0, 0.6)) is None, "no wheels heard: nothing to disagree with"
    guard.wheels(1.0, 0.0, 0.0)
    sideways = guard.check(1.0, (0.0, 0.6))
    assert sideways is not None and "wheels" in sideways
    assert guard.consecutive == 1, "the pass between them ended the first run"
    guard.wheels(1.1, 0.3, 0.0)
    assert guard.check(1.1, (0.3, 0.02)) is None and guard.consecutive == 0
    assert guard.counts == {"speed": 1, "wheels": 1}
    guard.wheels(5.0, 0.3, 0.0)
    assert guard.check(7.0, (0.0, 0.6)) is None, "wheels heard 2 s ago say nothing"
    assert "guard rejected 2 (speed 1, wheels 1)" in guard.report()


def test_the_guard_restarts_only_after_a_run_of_rejections_at_rest_and_not_in_a_loop() -> None:
    """20 implausible samples in a row restart the VIO once the wheels have said rest for 2 s;
    moving wheels, a run cut short or a restart less than 10 s ago do not."""
    from pepin.visual_odometry import VioGuard

    guard = VioGuard(restart_rejects=20, rest_s=2.0, restart_gap_s=10.0)
    t = 0.0
    for _ in range(25):  # diverged while the cart drives: refused, never restarted
        guard.wheels(t, 0.2, 0.0)
        assert guard.check(t, (4.0, 0.0)) is not None
        assert guard.restart_due(t) is None
        t += 0.1
    for _ in range(19):  # stopped, still diverged; rest under 2 s
        guard.wheels(t, 0.0, 0.0)
        guard.check(t, (4.0, 0.0))
        assert guard.restart_due(t) is None
        t += 0.1
    while guard.at_rest_s(t) < 2.0:
        guard.wheels(t, 0.0, 0.0)
        guard.check(t, (4.0, 0.0))
        assert guard.restart_due(t) is None
        t += 0.1
    guard.wheels(t, 0.0, 0.0)
    guard.check(t, (4.0, 0.0))
    due = guard.restart_due(t)
    assert due is not None and "in a row" in due and guard.restarts == 1
    assert guard.consecutive == 0
    restarted = t
    for _ in range(30):  # diverged again at once: the gap holds it
        t += 0.1
        guard.wheels(t, 0.0, 0.0)
        guard.check(t, (4.0, 0.0))
        assert guard.restart_due(t) is None or t - restarted >= 10.0
    assert guard.restarts == 1
    off = VioGuard(restart_rejects=0)
    off.wheels(0.0, 0.0, 0.0)
    for i in range(100):
        off.wheels(0.1 * i, 0.0, 0.0)
        off.check(0.1 * i, (4.0, 0.0))
    assert off.restart_due(10.0) is None, "0 never restarts"


def test_no_features_at_rest_is_not_lost_and_the_shortage_counts_from_the_first_motion() -> None:
    """2026-10-04 live: the head still, OpenVINS's MSCKF update had 0 features (no track ends at
    rest) and rule (c) withheld 300 of 300 poses. Under /zupt a shortage is never lost; when the
    base starts moving the second it must last starts then, not when the count fell."""
    from pepin.visual_odometry import VioLost

    lost = VioLost(lost_s=1.0, min_features=20)
    for i in range(50):  # 5 s at rest, the count 0 throughout
        t = 0.1 * i
        lost.zupt(t)
        lost.features(t, 0)
        assert lost.check(t, 0.0) is None
    assert lost.counts["features"] == 0
    for i in range(1, 10):  # the base moves (/zupt silent), still 0: under a second of motion
        t = 4.9 + 0.1 * i
        lost.features(t, 0)
        assert lost.check(t, 0.0) is None, f"{t:.1f}: the shortage under motion is younger than 1 s"
    lost.features(6.5, 0)
    assert lost.check(6.5, 0.0) is not None and lost.counts["features"] == 1
    lost.features(6.6, 45)
    assert lost.check(6.6, 0.0) is None, "features back"


def test_the_guard_wheel_rule_switches_off_and_the_speed_rule_does_not() -> None:
    """vio_guard off: 0.3 m/s against still wheels (a slip the VIO sees through) passes; 1.5 m/s
    is still a divergence; on again, the same 0.3 m/s is refused."""
    from pepin.visual_odometry import VioGuard

    guard = VioGuard(max_speed_m_s=1.0, wheel_diff_m_s=0.2, wheel_rule=False)
    guard.wheels(0.0, 0.0, 0.0)
    assert guard.check(0.0, (0.3, 0.0)) is None
    fast = guard.check(0.1, (1.5, 0.0))
    assert fast is not None and "1.50 m/s" in fast
    assert "the wheel rule off" in guard.report()
    guard.wheel_rule = True
    assert guard.check(0.2, (0.3, 0.0)) is not None
    assert guard.counts == {"speed": 1, "wheels": 1}


def test_the_track_remembers_the_step_it_admitted_until_an_anchor() -> None:
    track = VoTrack()
    a = VoPose(stamp=0.0, x=0.0, y=0.0, yaw=0.0)
    b = VoPose(stamp=0.1, x=0.02, y=0.0, yaw=0.0)
    track.advance(a)
    assert track.last_step is None, "the first pose only anchors"
    track.advance(b)
    assert track.last_step == (a, b)
    track.anchor(b)
    assert track.last_step is None, "nothing is differenced across an anchor"


def test_se2_twist_reads_an_arc_as_its_own_speed_with_no_sideways_leak() -> None:
    """0.2 m/s forward and 0.5 rad/s for 0.1 s from (1, 2) heading 40 deg: the twist is exactly
    (0.2, 0, 0.5); the chord in the starting heading (body_velocity) reads 0.19992 forward and
    5.0 mm/s sideways, v * w * dt / 2."""
    from pepin.visual_odometry import body_velocity, se2_twist

    v, w, dt, yaw0 = 0.2, 0.5, 0.1, math.radians(40.0)
    turn = w * dt
    forward, left = v / w * math.sin(turn), v / w * (1.0 - math.cos(turn))
    c, s = math.cos(yaw0), math.sin(yaw0)
    a = VoPose(stamp=10.0, x=1.0, y=2.0, yaw=yaw0)
    b = VoPose(
        stamp=10.0 + dt,
        x=1.0 + c * forward - s * left,
        y=2.0 + s * forward + c * left,
        yaw=yaw0 + turn,
    )
    twist = se2_twist(a, b)
    assert twist is not None
    assert twist == pytest.approx((0.2, 0.0, 0.5), abs=1e-9)
    chord = body_velocity(a, b)
    assert chord is not None
    assert chord[0] == pytest.approx(0.19992, abs=1e-5)
    assert chord[1] == pytest.approx(0.0049990, abs=1e-6)
    straight = se2_twist(a, VoPose(stamp=10.5, x=1.0 + 0.15 * c, y=2.0 + 0.15 * s, yaw=yaw0))
    assert straight == pytest.approx((0.3, 0.0, 0.0), abs=1e-9)
    spin = se2_twist(a, VoPose(stamp=10.2, x=1.0, y=2.0, yaw=yaw0 - 0.2))
    assert spin == pytest.approx((0.0, 0.0, -1.0), abs=1e-9)
    assert se2_twist(b, a) is None, "a stamp that does not advance says nothing"


def _imu_covariance(linear: tuple[float, ...], angular: tuple[float, ...]) -> list[float]:
    """odomimu's twist covariance: diagonal, linear then angular, in the IMU's own axes."""
    matrix = [0.0] * 36
    for i, value in enumerate((*linear, *angular)):
        matrix[i * 6 + i] = value
    return matrix


def test_openvins_velocity_covariance_is_turned_into_base_axes() -> None:
    """A chip on its side (base x = IMU z, base y = -IMU x, base z = -IMU y): base vx takes the
    IMU's z variance 9e-4, vy its x variance 1e-4, vyaw its y-rate 4e-6; every other axis 1e6.
    A 30 deg yaw on top mixes x and y: cos^2 * 4e-4 + sin^2 * 1e-4 = 3.25e-4 on vx and
    (4e-4 - 1e-4) * sin * cos = 1.299e-4 between them."""
    from pepin.visual_odometry import WEIGHTLESS_VARIANCE, base_twist_covariance

    chip = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    source = _imu_covariance((1e-4, 4e-4, 9e-4), (1e-6, 4e-6, 9e-6))
    base = base_twist_covariance(source, chip)
    assert base is not None
    assert (base[0], base[7], base[35]) == pytest.approx((9e-4, 1e-4, 4e-6))
    assert base[1] == pytest.approx(0.0) and base[5] == pytest.approx(0.0)
    assert [base[i * 6 + i] for i in (2, 3, 4)] == [WEIGHTLESS_VARIANCE] * 3
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    yawed = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    mixed = base_twist_covariance(_imu_covariance((4e-4, 1e-4, 9e-4), (1e-6,) * 3), yawed)
    assert mixed is not None
    assert mixed[0] == pytest.approx(3.25e-4) and mixed[7] == pytest.approx(1.75e-4)
    assert mixed[1] == pytest.approx(1.299e-4, rel=1e-3) and mixed[6] == mixed[1]


def test_a_covariance_that_is_not_one_weighs_nothing() -> None:
    from pepin.visual_odometry import base_twist_covariance

    eye = np.eye(3)
    assert base_twist_covariance([0.0] * 36, eye) is None, "OpenVINS before an update"
    nan = _imu_covariance((math.nan, 1e-4, 1e-4), (1e-6,) * 3)
    assert base_twist_covariance(nan, eye) is None
    assert base_twist_covariance(_imu_covariance((1e-4,) * 3, (0.0,) * 3), eye) is None
    assert base_twist_covariance([1e-4] * 12, eye) is None, "too short"


def test_twist_covariances_match_a_pose_by_stamp_and_forget_the_past() -> None:
    """odomimu at 200 Hz: a pose at 100.0123 takes the 100.010 sample; nothing within 20 ms is
    nothing; older than 2 s is gone; a stamp going back (a restarted OpenVINS) starts afresh."""
    from pepin.visual_odometry import TwistCovariances

    buffer = TwistCovariances(match_s=0.02, horizon_s=2.0)
    for i in range(600):
        buffer.add(98.0 + 0.005 * i, [float(i)] * 36)
    found = buffer.nearest(100.0123)
    assert found is not None and found[0] == 402.0  # 98.0 + 0.005 * 402 = 100.010
    assert buffer.nearest(100.995 + 0.03) is None, "past the newest by more than 20 ms"
    assert buffer.nearest(98.5) is None, "beyond the 2 s horizon"
    buffer.add(50.0, [7.0] * 36)
    assert buffer.nearest(100.0) is None and buffer.nearest(50.0) == tuple([7.0] * 36)
    assert buffer.received == 601
