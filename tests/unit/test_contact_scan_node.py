"""The contact node under the ROS stubs: where it takes the camera's pose from.

The contact line is the ray through the floor's last pixel, lifted onto the plane — so it is
exactly as right as the camera's pose. Until 2026-09-30 that pose was config/camera.json's mount
(the neck's link at its reference pose, straight ahead); here it is TF's ``base_link <- the depth's
frame`` at the frame's own stamp, and a panned head's scan turns with it. rclpy is faked
(``ros_stubs``); the room is test_contact's floor and box.
"""

from __future__ import annotations

import math
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import ros_stubs

ros_stubs.install()

from pepin_bringup.contact_scan import ContactScan  # noqa: E402
from pepin_bringup.msgs import image_from_array  # noqa: E402
from test_contact import CAM, INTR, _scene  # noqa: E402

from pepin.camera import OPTICAL_RPY  # noqa: E402
from pepin.depth import quaternion_from_matrix  # noqa: E402
from pepin.mounts import rotation_from_rpy  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
STAMP = ros_stubs.Time(sec=1_000, nanosec=250_000_000)


def _edge(pan_deg: float) -> Any:
    """``base_link <- camera_optical`` of test_contact's camera (1.23 m up, 26 deg down) with
    the neck turned ``pan_deg`` to the left."""
    link = rotation_from_rpy(0.0, CAM.pitch, math.radians(pan_deg))
    qx, qy, qz, qw = quaternion_from_matrix(link @ rotation_from_rpy(*OPTICAL_RPY))
    return ros_stubs.TransformStamped(
        header=ros_stubs.Header(stamp=STAMP),
        transform=ros_stubs.Transform(
            translation=ros_stubs.Vector3(x=CAM.x, y=CAM.y, z=CAM.z),
            rotation=ros_stubs.Quaternion(x=qx, y=qy, z=qz, w=qw),
        ),
    )


def _node() -> ContactScan:
    with ros_stubs.parameters(config=str(REPO / "config" / "camera.json")):
        node = ContactScan()
    node._on_info(
        SimpleNamespace(
            k=[INTR.fx, 0.0, INTR.cx, 0.0, INTR.fy, INTR.cy, 0.0, 0.0, 1.0],
            width=INTR.width,
            height=INTR.height,
        )
    )
    return node


def _frame(node: ContactScan) -> None:
    """A box 1.5 m ahead of the LENS: whichever way the neck turns, the picture is the same."""
    node._process(
        image_from_array(_scene(1.5).astype(np.float32), "32FC1", STAMP, "camera_optical")
    )


@pytest.mark.parametrize("pan_deg", [0.0, 30.0, -60.0])
def test_the_contact_scan_turns_with_the_neck(pan_deg: float) -> None:
    """The same picture of a box on the optical axis, the neck at 0, +30 and -60 deg: the box's
    contact marks the bearing the head looks along, in base_link, and the window's angle_min is
    that bearing minus 40 deg. With the config's mount it marked 0 deg at every pan."""
    node = _node()
    node._tf.buffer.transforms[("base_link", "camera_optical")] = _edge(pan_deg)
    _frame(node)
    scan = node.pubs["/contact_scan"].sent[-1]
    assert scan.header.frame_id == "base_link"
    assert math.degrees(scan.angle_min) == pytest.approx(pan_deg - 40.0, abs=1e-6)
    r = np.asarray(scan.ranges, dtype=float)
    angles = np.degrees(scan.angle_min + scan.angle_increment * np.arange(r.size))
    marked = angles[np.isfinite(r)]
    assert marked.size, "the box's foot is marked"
    assert float(np.median(marked)) == pytest.approx(pan_deg, abs=1.0)


def test_a_frame_tf_cannot_place_is_dropped_not_placed_straight_ahead() -> None:
    """No base_link <- camera_optical in TF: nothing is published and the report says why —
    the config mount's straight-ahead pose is never a stand-in for the head's."""
    node = _node()
    _frame(node)
    assert not node.pubs["/contact_scan"].sent
    node._report()
    assert "1 TF could not place" in node.logger.texts("info")[-1]


def test_a_still_head_keeps_its_floor_and_a_turned_one_rebuilds_it() -> None:
    """The floor's per-pixel geometry is rebuilt only when the camera's pose moves: two frames
    at one pan cost one build, a third after the neck turned costs another."""
    node = _node()
    node._tf.buffer.transforms[("base_link", "camera_optical")] = _edge(10.0)
    _frame(node)
    _frame(node)
    node._tf.buffer.transforms[("base_link", "camera_optical")] = _edge(25.0)
    _frame(node)
    assert node._tally.take().counts["planes"] == 2
