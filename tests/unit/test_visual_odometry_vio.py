"""The relay's visual-inertial input under the ROS stubs: OpenVINS's IMU pose composed through TF
into base_link, gated and tracked like rtabmap's, published with the per-step `vio` covariance."""

from __future__ import annotations

import math
from itertools import pairwise
from typing import Any

import numpy as np
import pytest
import ros_stubs

ros_stubs.install()

import pepin_bringup.visual_odometry as relay  # noqa: E402
from pepin_bringup.msgs import stamp_from_seconds, yaw_of  # noqa: E402
from pepin_bringup.visual_odometry import (  # noqa: E402
    VIO_ODOM_TOPIC,
    VIO_POINTS_TOPIC,
    VIO_POSE_TOPIC,
    VIO_RESTART_SERVICE,
    VIO_SLAM_POINTS_TOPIC,
    VO_TOPIC,
    VO_TWIST_TOPIC,
    WHEELS_TOPIC,
    ZUPT_TOPIC,
    VisualOdometry,
)

from pepin.depth import quaternion_from_matrix  # noqa: E402

T0 = 1_759_500_000.0
# base_link <- head_imu: the IMU 1.2 m up and 3 cm ahead, its axes turned (a chip on its side)
CHIP = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
T_B_I = np.eye(4)
T_B_I[:3, :3] = CHIP
T_B_I[:3, 3] = (0.03, 0.03, 1.2)


def _yaw(angle: float) -> Any:
    c, s = math.cos(angle), math.sin(angle)
    m = np.eye(4)
    m[:2, :2] = [[c, -s], [s, c]]
    return m


def _transform(matrix: Any) -> Any:
    """``head_imu <- base_link`` as tf2 would hand it back."""
    qx, qy, qz, qw = quaternion_from_matrix(matrix[:3, :3])
    msg = ros_stubs.TransformStamped()
    msg.header.frame_id, msg.child_frame_id = "head_imu", "base_link"
    t = msg.transform.translation
    t.x, t.y, t.z = (float(v) for v in matrix[:3, 3])
    r = msg.transform.rotation
    r.x, r.y, r.z, r.w = qx, qy, qz, qw
    return msg


def _poseimu(t: float, t_g_b: Any, trace: float = 1e-4) -> Any:
    """OpenVINS's poseimu for a base pose in G: the IMU's pose T_G_I = T_G_B * T_B_I."""
    t_g_i = t_g_b @ T_B_I
    msg = ros_stubs.PoseWithCovarianceStamped()
    msg.header.stamp = stamp_from_seconds(T0 + t)
    msg.header.frame_id = "global"
    p = msg.pose.pose.position
    p.x, p.y, p.z = (float(v) for v in t_g_i[:3, 3])
    o = msg.pose.pose.orientation
    o.x, o.y, o.z, o.w = quaternion_from_matrix(t_g_i[:3, :3])
    msg.pose.covariance = [trace / 6 if i % 7 == 0 else 0.0 for i in range(36)]
    return msg


def _node(**params: Any) -> VisualOdometry:
    with ros_stubs.parameters(vo_input="vio", **params):
        node = VisualOdometry()
    assert node._tf is not None
    node._tf.buffer.transforms[("head_imu", "base_link")] = _transform(np.linalg.inv(T_B_I))
    return node


def test_the_vio_input_subscribes_openvins_and_pins_the_per_step_covariance() -> None:
    node = _node()
    assert {VIO_POSE_TOPIC, VIO_POINTS_TOPIC, ZUPT_TOPIC} <= set(node.subs)
    assert "/vo/raw" not in node.subs, "rtabmap's output is not read"
    assert VIO_RESTART_SERVICE in node.service_clients
    assert node._covariance_mode() == "vio", "OpenVINS's marginal is no weight"
    assert node.set_parameters([ros_stubs.Parameter("vo_covariance", value="rtabmap")])[
        0
    ].successful
    assert node._covariance_mode() == "vio", "the registration's modes read as vio"
    assert node.set_parameters([ros_stubs.Parameter("vo_covariance", value="constant")])[
        0
    ].successful
    assert node._covariance_mode() == "constant"
    refused = node.set_parameters([ros_stubs.Parameter("vo_input", value="stereo")])
    assert not refused[0].successful, "launched with vio: no rtabmap odometry to go back to"
    node.close()


def test_a_gravity_frame_turned_90_deg_reaches_the_ekf_as_forward_motion() -> None:
    """G's yaw is whatever init gave it; the cart drives straight at 0.2 m/s along G's y. The
    published /vo, differenced the way robot_localization does, is vx 0.2, vy 0; its sigma is
    the floor (vo_sigma_m 0.07) at a 2 cm step."""
    node = _node()
    for i in range(6):
        t_g_b = _yaw(math.pi / 2)
        t_g_b[:3, 3] = (0.5, 0.02 * i, 0.0)
        node.subs[VIO_POSE_TOPIC][1](_poseimu(0.1 * i, t_g_b))
    sent = node.pubs[VO_TOPIC].sent
    assert len(sent) == 6
    assert sent[0].header.frame_id == "odom" and sent[0].child_frame_id == "base_link"
    poses = [
        (
            float(m.header.stamp.sec) + m.header.stamp.nanosec * 1e-9,
            m.pose.pose.position.x,
            m.pose.pose.position.y,
            yaw_of(m.pose.pose.orientation),
        )
        for m in sent
    ]
    for (ta, xa, ya, yawa), (tb, xb, yb, _) in pairwise(poses):
        dx, dy, dt = xb - xa, yb - ya, tb - ta
        vx = (math.cos(yawa) * dx + math.sin(yawa) * dy) / dt
        vy = (-math.sin(yawa) * dx + math.cos(yawa) * dy) / dt
        assert vx == pytest.approx(0.2, abs=1e-6) and vy == pytest.approx(0.0, abs=1e-6)
    assert math.sqrt(sent[-1].pose.covariance[0]) == pytest.approx(math.hypot(0.07, 0.03 * 0.02))
    node.close()


def test_the_vio_covariance_is_the_floor_for_a_short_step_and_grows_with_a_long_one() -> None:
    node = _node()
    assert math.sqrt(node._vio_covariance(0.02)[0]) == pytest.approx(0.07, rel=1e-3)
    assert math.sqrt(node._vio_covariance(1.0)[0]) == pytest.approx(math.hypot(0.07, 0.03))
    node.close()


def test_a_missing_neck_chain_skips_a_pose_and_a_reinit_reanchors() -> None:
    node = _node()
    still = np.eye(4)
    node.subs[VIO_POSE_TOPIC][1](_poseimu(0.0, still))
    node._tf.buffer.error = RuntimeError("Extrapolation into the future")  # type: ignore[union-attr]
    node.subs[VIO_POSE_TOPIC][1](_poseimu(0.1, still))
    node._tf.buffer.error = None  # type: ignore[union-attr]
    moved = np.eye(4)
    moved[:3, 3] = (3.0, 0.0, 0.0)  # a re-initialised G: the pose lands somewhere else
    node.subs[VIO_POSE_TOPIC][1](_poseimu(0.2, moved, trace=1e-7))
    node.subs[VIO_POSE_TOPIC][1](_poseimu(0.3, moved @ _shift(0.02), trace=1.1e-7))
    sent = node.pubs[VO_TOPIC].sent
    xs = [m.pose.pose.position.x for m in sent]
    assert max(xs) < 0.05, "the re-init's 3 m never reach the EKF"
    counts = node._tally.take().counts
    assert counts["tf_miss"] == 1 and counts["vio_reinit"] == 1
    node.close()


def test_a_drift_while_the_board_says_rest_is_withheld_as_lost() -> None:
    node = _node()
    node.subs[ZUPT_TOPIC][1](ros_stubs.Odometry())
    for i in range(4):  # creeping 1 cm a frame: 0.1 m/s while the board says rest
        node.subs[VIO_POSE_TOPIC][1](_poseimu(0.1 * i, _shift(0.01 * i)))
    assert len(node.pubs[VO_TOPIC].sent) == 1, "only the first, before any speed was known"
    w = node._tally.take()
    assert w.counts["vio_lost"] == 3
    node._report()
    line = node.get_logger().texts("info")[-1]
    assert (
        "reinit 0 (0 since the start)" in line and "lost 1 (wheels 0, rest 1, features 0)" in line
    )
    node.close()


def _shift(x: float) -> Any:
    m = np.eye(4)
    m[0, 3] = x
    return m


class _Clock:
    """The relay's ``time`` module with a monotonic clock the test owns."""

    def __init__(self) -> None:
        self.now = 100.0

    def monotonic(self) -> float:
        return self.now

    def time(self) -> float:
        return T0 + self.now


def _rot_z(angle: float) -> Any:
    return _yaw(angle)


# base_link <- head_imu through a panning neck: the pan axis 0.78 m up, the IMU 4 cm ahead of it
# and 3 cm to the side, its axes those of the chip on its side (CHIP)
NECK = np.eye(4)
NECK[:3, 3] = (-0.05, 0.0, 0.78)
IMU_ON_NECK = np.eye(4)
IMU_ON_NECK[:3, :3] = CHIP
IMU_ON_NECK[:3, 3] = (0.04, 0.03, 0.42)


def _base_to_imu(pan: float) -> Any:
    return NECK @ _rot_z(pan) @ IMU_ON_NECK


def _feed(node: VisualOdometry, t: float, t_g_b: Any, pan: float) -> None:
    """One OpenVINS sample of a base pose in G with the head panned: the neck chain into the TF
    buffer, the IMU's pose T_G_I = T_G_B * T_B_I out of OpenVINS."""
    t_b_i = _base_to_imu(pan)
    node._tf.buffer.transforms[("head_imu", "base_link")] = _transform(  # type: ignore[union-attr]
        np.linalg.inv(t_b_i)
    )
    msg = _poseimu(t, t_g_b @ t_b_i @ np.linalg.inv(T_B_I))
    node.subs[VIO_POSE_TOPIC][1](msg)


def _wheels(node: VisualOdometry, forward: float) -> None:
    msg = ros_stubs.Odometry()
    msg.twist.twist.linear.x = forward
    node.subs[WHEELS_TOPIC][1](msg)


def _published(node: VisualOdometry) -> list[tuple[float, float, float]]:
    return [
        (m.pose.pose.position.x, m.pose.pose.position.y, yaw_of(m.pose.pose.orientation))
        for m in node.pubs[VO_TOPIC].sent
    ]


def test_a_head_turned_90_deg_on_a_parked_cart_sends_zero_base_motion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cart stands at an arbitrary pose in G, the wheels and /zupt say rest, the head pans
    0 -> 90 deg in 10 samples (the IMU swings 4-5 cm on its arm): every sample passes the guard
    and the lost rules, reaches the EKF, and the published track does not move."""
    clock = _Clock()
    monkeypatch.setattr(relay, "time", clock)
    node = _node()
    t_g_b = _rot_z(math.radians(37.0))
    t_g_b[:3, 3] = (1.5, -0.4, 0.0)
    for i in range(11):
        clock.now += 0.1
        _wheels(node, 0.0)
        node.subs[ZUPT_TOPIC][1](ros_stubs.Odometry())
        _feed(node, 0.1 * i, t_g_b, math.radians(9.0 * i))
    sent = _published(node)
    assert len(sent) == 11, "nothing refused"
    assert all(abs(x) < 1e-9 and abs(y) < 1e-9 and abs(yaw) < 1e-9 for x, y, yaw in sent)
    counts = node._tally.take().counts
    assert counts["rejected"] == 0 and counts["vio_lost"] == 0
    node.close()


def test_a_base_translation_comes_through_unchanged_with_the_head_turned(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The cart drives 1 m straight at 0.2 m/s along a G heading of 120 deg with the head held
    at 30 deg: the published track moves 1 m straight ahead, vx 0.2, vy 0."""
    clock = _Clock()
    monkeypatch.setattr(relay, "time", clock)
    node = _node()
    heading = math.radians(120.0)
    for i in range(51):
        clock.now += 0.1
        _wheels(node, 0.2)
        t_g_b = _rot_z(heading)
        t_g_b[:3, 3] = (0.3 + 0.02 * i * math.cos(heading), 0.02 * i * math.sin(heading), 0.0)
        _feed(node, 0.1 * i, t_g_b, math.radians(30.0))
    sent = _published(node)
    assert len(sent) == 51
    x, y, yaw = sent[-1]
    assert x == pytest.approx(1.0, abs=1e-9) and y == pytest.approx(0.0, abs=1e-9)
    assert yaw == pytest.approx(0.0, abs=1e-9)
    node.close()


def test_a_divergence_is_refused_and_restarts_openvins_once_at_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OpenVINS runs away at 5 m/s while the wheels say rest: nothing of it reaches the EKF,
    the guard counts every sample, and after 20 in a row (the wheels still for 2 s) the relay
    calls /vio/restart once and says so once."""
    clock = _Clock()
    monkeypatch.setattr(relay, "time", clock)
    node = _node()
    node._restart.ready = True
    for _ in range(25):  # the wheels at rest for 2.5 s before anything happens
        clock.now += 0.1
        _wheels(node, 0.0)
    for i in range(30):
        clock.now += 0.1
        _wheels(node, 0.0)
        t_g_b = np.eye(4)
        t_g_b[:3, 3] = (0.5 * i, 0.0, 0.0)  # 5 m/s
        _feed(node, 0.1 * i, t_g_b, 0.0)
    sent = _published(node)
    assert len(sent) == 1, "the first sample only, before any velocity was known"
    assert node._restart.calls and len(node._restart.calls) == 1
    restarting = [t for t in node.get_logger().texts("warning") if "restarting OpenVINS" in t]
    assert len(restarting) == 1 and "20 implausible samples in a row" in restarting[0]
    # 28 refused by the guard; the one after the restart is not judged (the VIO's state starts
    # again) and the gate refuses it as the jump it is
    node._report()
    line = node.get_logger().texts("info")[-1]
    assert "28 rejected by the guard" in line and "restarts 1" in line
    assert "1 dropped (last: a jump of 50 cm" in line
    node.close()


def test_a_divergence_while_the_cart_drives_is_refused_but_not_restarted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    monkeypatch.setattr(relay, "time", clock)
    node = _node()
    node._restart.ready = True
    for i in range(40):
        clock.now += 0.1
        _wheels(node, 0.2)
        t_g_b = np.eye(4)
        t_g_b[:3, 3] = (0.5 * i, 0.0, 0.0)
        _feed(node, 0.1 * i, t_g_b, 0.0)
    assert len(_published(node)) == 1
    assert not node._restart.calls, "OpenVINS initialises from stillness: never mid-drive"
    node.close()


def test_the_input_switches_live_between_stereo_and_vio_without_a_jump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Launched on stereo: vo_input vio picks OpenVINS up live, rtabmap's poses are then ignored,
    the published track carries on from where stereo left it; depth is refused (no
    rgbd_odometry runs); stereo again reads /vo/raw."""
    clock = _Clock()
    monkeypatch.setattr(relay, "time", clock)
    node = VisualOdometry()
    assert node._tf is None and VIO_POSE_TOPIC not in node.subs
    for i in range(3):  # stereo drives 4 cm along its own x
        clock.now += 0.1
        raw = ros_stubs.Odometry()
        raw.header.stamp = stamp_from_seconds(T0 + 0.1 * i)
        raw.pose.pose.position.x = 0.02 * i
        raw.pose.pose.orientation.w = 1.0
        raw.pose.covariance = [1e-4 if k % 7 == 0 else 0.0 for k in range(36)]
        node.subs["/vo/raw"][1](raw)
    assert _published(node)[-1][0] == pytest.approx(0.04)
    assert node.set_parameters([ros_stubs.Parameter("vo_input", value="vio")])[0].successful
    assert {VIO_POSE_TOPIC, VIO_POINTS_TOPIC, ZUPT_TOPIC} <= set(node.subs)
    before = len(node.pubs[VO_TOPIC].sent)
    node.subs["/vo/raw"][1](raw)
    assert len(node.pubs[VO_TOPIC].sent) == before, "rtabmap is not read under vio"
    for i in range(3):  # OpenVINS's G is elsewhere entirely; the cart drives 4 cm more
        clock.now += 0.1
        t_g_b = _rot_z(1.0)
        t_g_b[:3, 3] = (7.0 + 0.02 * i * math.cos(1.0), -3.0 + 0.02 * i * math.sin(1.0), 0.0)
        _feed(node, 1.0 + 0.1 * i, t_g_b, 0.0)
    xs = [x for x, _, _ in _published(node)]
    assert xs[before:] == pytest.approx([0.04, 0.06, 0.08]), "no jump across the switch"
    assert not node.set_parameters([ros_stubs.Parameter("vo_input", value="depth")])[0].successful
    assert node.set_parameters([ros_stubs.Parameter("vo_input", value="stereo")])[0].successful
    assert node._input == "stereo" and node._covariance_mode() == "dynamic"
    node.close()


def test_the_report_says_latency_whether_the_ekf_reads_and_where_its_odom_stands() -> None:
    node = _node()
    node.subscriber_nodes[VO_TOPIC] = ["ekf_filter_node", "bag_recorder"]
    odom = np.eye(4)
    odom[:3, 3] = (0.012, -0.003, 0.0)
    msg = _transform(odom)
    node._tf.buffer.transforms[("odom", "base_link")] = msg  # type: ignore[union-attr]
    node.subs[VIO_POSE_TOPIC][1](_poseimu(0.0, np.eye(4)))
    node._report()
    line = node.get_logger().texts("info")[-1]
    assert "(1 in, 1 out)" in line and "stamp to receipt p50 " in line
    assert "/vo read by ekf_filter_node: yes (2 subscriber nodes)" in line
    assert "EKF odom -> base_link (0.012, -0.003, 0.0 deg)" in line
    assert "0 rejected by the guard" in line and "covariance vio" in line
    node.close()


def _cloud(n: int) -> Any:
    msg = ros_stubs.PointCloud2()
    msg.width, msg.height = n, 1
    return msg


def test_a_still_cart_with_no_msckf_features_still_reaches_the_ekf(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The live static test of 2026-10-04: head still, /ov_msckf/points_msckf empty every update,
    300 of 300 poses withheld as lost. Under /zupt nothing is lost for seeing no features; the
    report sums the two clouds."""
    clock = _Clock()
    monkeypatch.setattr(relay, "time", clock)
    node = _node()
    assert VIO_SLAM_POINTS_TOPIC in node.subs
    for i in range(30):  # 3 s at rest, 0 MSCKF features, a few SLAM ones
        clock.now += 0.1
        _wheels(node, 0.0)
        node.subs[ZUPT_TOPIC][1](ros_stubs.Odometry())
        node.subs[VIO_POINTS_TOPIC][1](_cloud(0))
        node.subs[VIO_SLAM_POINTS_TOPIC][1](_cloud(7))
        _feed(node, 0.1 * i, np.eye(4), 0.0)
    assert len(node.pubs[VO_TOPIC].sent) == 30, "nothing withheld at rest"
    node._report()
    line = node.get_logger().texts("info")[-1]
    assert "features 7 (msckf 0 + slam 7)" in line and "features 0)" in line
    node.close()


# OpenVINS's twist covariance in the IMU's own axes (odomimu): linear then angular, diagonal.
IMU_LINEAR = (1e-4, 4e-4, 9e-4)
IMU_ANGULAR = (1e-6, 4e-6, 9e-6)


def _odomimu(t: float, linear: Any = IMU_LINEAR, angular: Any = IMU_ANGULAR) -> Any:
    msg = ros_stubs.Odometry()
    msg.header.stamp = stamp_from_seconds(T0 + t)
    msg.header.frame_id, msg.child_frame_id = "global", "imu"
    msg.twist.covariance = [0.0] * 36
    for i, value in enumerate((*linear, *angular)):
        msg.twist.covariance[i * 6 + i] = value
    return msg


def _drive_with_a_panning_head(node: VisualOdometry, pans: list[float]) -> None:
    """The base drives straight along G's y (G's yaw 90 deg) at 0.2 m/s while the head pans;
    OpenVINS's odomimu at every IMU sample (5 ms) and poseimu at every 0.1 s."""
    for i, pan in enumerate(pans):
        for k in range(-3, 17):
            node.subs[VIO_ODOM_TOPIC][1](_odomimu(0.1 * i + 0.005 * k))
        t_g_b = _yaw(math.pi / 2)
        t_g_b[:3, 3] = (0.5, 0.02 * i, 0.0)
        _feed(node, 0.1 * i, t_g_b, pan)


def test_the_twist_output_sends_the_base_velocity_with_openvins_own_covariance() -> None:
    """vo_output twist with the head panning 0 -> 0.5 rad while the cart drives straight at
    0.2 m/s: /vo_twist reads vx 0.2, vy 0, vyaw 0 (the neck's own motion subtracted), weighed by
    the IMU's velocity covariance turned into base_link through the neck at that stamp — at the
    last pan, vx cos^2(0.5) * 9e-4 + sin^2(0.5) * 1e-4 = 7.161e-4, vy 2.839e-4, vyaw the IMU's
    y-rate 4e-6, no floor. /vo carries the track weightless; the first pose has no step."""
    node = _node()
    assert node.set_parameters([ros_stubs.Parameter("vo_output", value="twist")])[0].successful
    _drive_with_a_panning_head(node, [0.1 * i for i in range(6)])
    twists = node.pubs[VO_TWIST_TOPIC].sent
    assert len(twists) == 5
    for msg in twists:
        assert msg.header.frame_id == "base_link"
        v = msg.twist.twist
        assert (v.linear.x, v.linear.y, v.angular.z) == pytest.approx((0.2, 0.0, 0.0), abs=1e-6)
    last = twists[-1].twist.covariance
    c2, s2 = math.cos(0.5) ** 2, math.sin(0.5) ** 2
    assert last[0] == pytest.approx(c2 * 9e-4 + s2 * 1e-4)
    assert last[7] == pytest.approx(s2 * 9e-4 + c2 * 1e-4)
    assert last[35] == pytest.approx(4e-6)
    assert [last[i * 6 + i] for i in (2, 3, 4)] == [1e6] * 3
    poses = node.pubs[VO_TOPIC].sent
    assert len(poses) == 5
    assert [poses[-1].pose.covariance[i * 6 + i] for i in range(6)] == [1e6] * 6
    w = node._tally.take()
    assert w.counts["no_step"] == 1 and w.counts["twist_out"] == 5
    node.close()


def test_the_pose_output_sends_no_twist_and_reports_the_sigmas() -> None:
    """vo_output pose (the default): /vo as before (the 7 cm floor), /vo_twist silent, and the
    report line carries OpenVINS's base-axis sigmas so the switch is measured before it is made:
    sqrt(9e-4) = 3.00 cm/s, sqrt(1e-4) = 1.00 cm/s, sqrt(4e-6) rad/s = 0.11 deg/s."""
    node = _node()
    _drive_with_a_panning_head(node, [0.0] * 6)
    assert not node.pubs[VO_TWIST_TOPIC].sent
    poses = node.pubs[VO_TOPIC].sent
    assert len(poses) == 6
    assert math.sqrt(poses[-1].pose.covariance[0]) == pytest.approx(0.07, rel=1e-3)
    node._report()
    line = node.get_logger().texts("info")[-1]
    assert "output pose: sigma p50 vx 3.00 vy 1.00 cm/s vyaw 0.11 deg/s (6)" in line
    assert "0 twists sent" in line and "odomimu 120 in" in line
    node.close()


def test_a_twist_without_its_covariance_is_withheld_and_counted() -> None:
    """No odomimu near the stamp (before OpenVINS's first second, or not subscribed): cov
    missing; a zero matrix (no update yet): cov bad. Neither reaches the EKF."""
    node = _node()
    assert node.set_parameters([ros_stubs.Parameter("vo_output", value="twist")])[0].successful
    for i in range(3):
        t_g_b = np.eye(4)
        t_g_b[:3, 3] = (0.02 * i, 0.0, 0.0)
        node.subs[VIO_POSE_TOPIC][1](_poseimu(0.1 * i, t_g_b))
    for i in range(3, 6):
        node.subs[VIO_ODOM_TOPIC][1](_odomimu(0.1 * i, (0.0,) * 3, (0.0,) * 3))
        t_g_b = np.eye(4)
        t_g_b[:3, 3] = (0.02 * i, 0.0, 0.0)
        node.subs[VIO_POSE_TOPIC][1](_poseimu(0.1 * i, t_g_b))
    assert not node.pubs[VO_TWIST_TOPIC].sent and not node.pubs[VO_TOPIC].sent
    node._report()
    line = node.get_logger().texts("info")[-1]
    assert "withheld: cov missing 2, cov bad 3, no step 1" in line
    node.close()


def test_the_output_and_the_guard_switch_live_and_rtabmap_stays_pose() -> None:
    from pepin.deployment import VO_TWIST_TOPIC as DEPLOYED

    assert f"/{DEPLOYED}" == VO_TWIST_TOPIC, "the flags table needs a literal; the EKF reads this"
    node = _node()
    assert node._guard.wheel_rule
    assert node.set_parameters([ros_stubs.Parameter("vio_guard", value=False)])[0].successful
    assert not node._guard.wheel_rule, "the wheel rule off, live"
    assert not node.set_parameters([ros_stubs.Parameter("vo_output", value="velocity")])[
        0
    ].successful
    node.close()
    with ros_stubs.parameters(vo_input="stereo", vo_output="twist"):
        stereo = VisualOdometry()
    assert stereo._output_mode() == "pose", "rtabmap's inputs have no velocity covariance"
    stereo.close()


def test_the_camera_edges_come_from_the_config_not_from_a_late_tf_static() -> None:
    """Under vio the relay's buffer holds camera_link -> camera_optical -> head_imu from
    config/camera.json before any /tf_static arrives, built exactly as camera_stream broadcasts
    them (the same reader, the same builder): Kalibr's T_cam_imu translation to the micrometre."""
    from pepin_bringup.msgs import camera_edges, pose_from_transform

    from pepin.mounts import load_camera_mounts

    node = _node()
    buffer = node._tf.buffer  # type: ignore[union-attr]
    expected = camera_edges(load_camera_mounts(), stamp_from_seconds(T0))
    assert len(expected) == 2, "the active rig carries a head_imu block"
    for edge in expected:
        seeded = buffer.transforms[(edge.header.frame_id, edge.child_frame_id)]
        a, b = pose_from_transform(seeded), pose_from_transform(edge)
        assert np.allclose(a.rotation, b.rotation) and np.allclose(a.translation, b.translation)
    node._report()
    line = node.get_logger().texts("info")[-1]
    assert (
        "static edges from config/camera.json: camera_link -> camera_optical,"
        " camera_optical -> head_imu" in line
    )
    node.close()
