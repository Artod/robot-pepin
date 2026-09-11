"""One owner of the pose of a frame: the carry through the odometry, the camera in the map,
and a history that cannot say."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pepin.depth import rotation_matrix
from pepin.frame_pose import FramePoser, PoseHistory
from pepin.tsdf import RigidPose


def _yaw(theta: float, x: float = 0.0, y: float = 0.0) -> RigidPose:
    """A planar pose: a yaw about the vertical axis and a position."""
    return RigidPose(
        rotation_matrix(0.0, 0.0, math.sin(theta / 2), math.cos(theta / 2)),
        np.array([x, y, 0.0]),
    )


class FakeHistory:
    """The cart turning left at 0.1 rad/s in the odometry, offset in the map, its camera on a
    mount; nothing before t = 0."""

    def __init__(self) -> None:
        self.asked: list[tuple[float, str, str]] = []

    def pose_at(self, stamp: float, frame: str, fixed: str) -> RigidPose | None:
        self.asked.append((stamp, frame, fixed))
        if stamp < 0.0:
            return None
        base = _yaw(0.1 * stamp, 0.5 * stamp, 0.0)
        if fixed == "map":
            base = _yaw(0.1 * stamp + 1.0, 0.5 * stamp + 3.0, 2.0)
        if frame == "base_link":
            return base
        mount = RigidPose(np.eye(3), np.array([0.0, 0.0, 1.23]))  # the camera on the neck
        return RigidPose(
            base.rotation @ mount.rotation, base.rotation @ mount.translation + base.translation
        )


def test_a_scan_is_carried_through_the_odometry_to_the_frame_s_moment() -> None:
    """A point 2 m dead ahead at t 0; by t 1 the cart turned 0.1 rad left and moved 0.5 m
    forward, so the point sits 0.1 rad to the right and nearer, and the odometry frame is the
    one asked (the map's corrections must not tear a scan)."""
    history = FakeHistory()
    poser = FramePoser(history)
    ahead = np.array([[2.0, 0.0, 0.2]])
    moved = poser.carry(ahead, 0.0, 1.0)
    assert moved is not None
    world = np.array([2.0, 0.0, 0.2])  # the point in odom: the cart stood at the origin at t 0
    pose = _yaw(0.1, 0.5, 0.0)
    expected = pose.rotation.T @ (world - pose.translation)
    assert moved[0] == pytest.approx(expected, abs=1e-12)
    assert moved[0, 1] < 0.0 and moved[0, 0] < 2.0 and moved[0, 2] == pytest.approx(0.2)
    assert {fixed for _, _, fixed in history.asked} == {"odom"}
    assert poser.carry(ahead, -1.0, 1.0) is None  # before the history begins
    assert poser.motion(1.0, 1.0) is not None
    same = poser.carry(ahead, 1.0, 1.0)
    assert same is not None and same == pytest.approx(ahead, abs=1e-12)


def test_the_camera_and_the_cart_are_placed_in_the_map_at_the_stamp() -> None:
    poser = FramePoser(FakeHistory())
    camera = poser.camera_in_map(2.0)
    base = poser.base_in_map(2.0)
    assert camera is not None and base is not None
    assert base.translation == pytest.approx([4.0, 2.0, 0.0])
    assert camera.translation == pytest.approx([4.0, 2.0, 1.23])
    assert camera.rotation == pytest.approx(base.rotation)
    on_map = poser.to_map(np.array([[1.0, 0.0, 0.0]]), 2.0)
    assert on_map is not None
    assert on_map[0] == pytest.approx([4.0 + math.cos(1.2), 2.0 + math.sin(1.2), 0.0])
    assert poser.camera_in_map(-5.0) is None and poser.to_map(np.zeros((1, 3)), -5.0) is None


def test_the_frames_asked_for_are_the_poser_s_own_names() -> None:
    history = FakeHistory()
    poser = FramePoser(history, base="base", camera="cam", map_frame="world", odom_frame="odo")
    poser.camera_in_map(1.0)
    poser.carry(np.zeros((1, 3)), 0.0, 1.0)
    assert history.asked == [(1.0, "cam", "world"), (0.0, "base", "odo"), (1.0, "base", "odo")]
    fake: PoseHistory = history  # the fake satisfies the protocol
    assert fake.pose_at(0.0, "base", "odo") is not None
