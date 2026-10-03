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

from pepin_bringup.msgs import stamp_from_seconds, yaw_of  # noqa: E402
from pepin_bringup.visual_odometry import (  # noqa: E402
    VIO_POINTS_TOPIC,
    VIO_POSE_TOPIC,
    VO_TOPIC,
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
    assert node._switches["vo_covariance"] == "vio", "OpenVINS's marginal is no weight"
    refused = node.set_parameters([ros_stubs.Parameter("vo_covariance", value="dynamic")])
    assert not refused[0].successful
    assert node.set_parameters([ros_stubs.Parameter("vo_covariance", value="constant")])[
        0
    ].successful
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
    assert "vio: reinit 0" in line and "lost 1 (wheels 0, rest 1, features 0)" in line
    node.close()


def _shift(x: float) -> Any:
    m = np.eye(4)
    m[0, 3] = x
    return m
