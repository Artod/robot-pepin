"""The fusion node under the ROS stubs: what it is willing to paint the world with.

The volume is written in the MAP frame and a TSDF cannot be un-integrated, so an observation
placed by a wrong pose does not add noise — it deletes the room. Both paint paths ask
:class:`pepin.watch.PaintTrust` first, and this file drives the lidar one revolution at a time:
a good fit heard just now, on a fresh ``map -> odom`` edge, is integrated; a low fit, a fit that
stopped arriving, a sigma over ``paint_sigma_m`` and a stale edge each leave the volume exactly
as it was.

rclpy and message_filters are faked (``ros_stubs``); the room is a synthetic box, as in
test_worldmap.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import ros_stubs

ros_stubs.install()

from pepin_bringup.depth_fusion import DepthFusion  # noqa: E402
from ros_stubs import Header, LaserScan, Parameter, String, TransformStamped  # noqa: E402
from ros_stubs import Time as TimeMsg  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SCAN_S = 100.0  # the stamp every revolution of this file carries
BEAMS = 360


def stamp(t: float) -> Any:
    whole = math.floor(t)
    return TimeMsg(sec=whole, nanosec=round((t - whole) * 1e9))


def edge(parent: str, child: str, t: float, z: float = 0.0) -> Any:
    """A TF edge stamped ``t``: what the poser's lookups and the pose gate's freshness read."""
    tf = TransformStamped(header=Header(stamp=stamp(t), frame_id=parent))
    tf.child_frame_id = child
    tf.transform.translation.z = z
    tf.transform.rotation.w = 1.0
    return tf


def scan_msg(t: float = SCAN_S) -> Any:
    """One revolution of a lidar standing in the middle of a 4 m box."""
    angles = np.linspace(-math.pi, math.pi, BEAMS, endpoint=False)
    with np.errstate(divide="ignore"):
        tx = np.where(np.cos(angles) > 0, 2.0 / np.cos(angles), -2.0 / np.cos(angles))
        ty = np.where(np.sin(angles) > 0, 2.0 / np.sin(angles), -2.0 / np.sin(angles))
    return LaserScan(
        header=Header(stamp=stamp(t), frame_id="laser"),
        angle_min=float(angles[0]),
        angle_increment=float(angles[1] - angles[0]),
        range_max=12.0,
        ranges=np.minimum(np.abs(tx), np.abs(ty)).tolist(),
    )


def small_config(tmp_path: Path) -> Path:
    """config/fusion.json with a 6 x 6 m box around the origin instead of the flat's: the same
    voxel, laws and bands, on a grid the synthetic room of this file sits inside."""
    config = json.loads((REPO / "config" / "fusion.json").read_text())
    config["origin_m"], config["shape"] = [-3.0, -3.0, -0.15], [120, 120, 34]
    path = tmp_path / "fusion.json"
    path.write_text(json.dumps(config))
    return path


@pytest.fixture
def node(tmp_path: Path) -> DepthFusion:
    """A fusion node on this checkout's configs, with the laser mount and a fresh map -> odom
    edge in TF, its pose gate on and a tracker that has just reported a good fit."""
    with ros_stubs.parameters(
        config=str(small_config(tmp_path)),
        lidar_config=str(REPO / "config" / "lidar.json"),
        world_path=str(tmp_path / "world.npz"),
        resume_volume=False,
        snapshot_s=0.0,
        imu_lean=False,
    ):
        node = DepthFusion()
    buffer = node._tf.buffer
    buffer.transforms[("base_link", "laser")] = edge("base_link", "laser", SCAN_S, z=0.383)
    buffer.transforms[("map", "odom")] = edge("map", "odom", SCAN_S)
    buffer.transforms[("map", "base_link")] = edge("map", "base_link", SCAN_S)
    node._on_fit(ros_stubs.Float32(data=0.9))
    return node


def painted(node: DepthFusion) -> float:
    """How much the lidar has written into the volume: the weight it owns, in total."""
    return float(node._world.lidar_weight.sum())


def test_a_trusted_pose_paints_the_room(node: DepthFusion) -> None:
    """The control: a fit of 0.9 heard just now, a map -> odom edge stamped with the scan, no
    sigma on the wire. The revolution goes in and nothing is withheld."""
    assert painted(node) == 0.0
    node._on_scan_work(scan_msg())
    assert painted(node) > 0.0, "the beams wrote the box"
    counts = node._tally.take().counts
    assert counts["revolutions"] == 1 and counts["untrusted"] == 0


@pytest.mark.parametrize(
    ("what", "reason"),
    [
        ("low_fit", "fit 0.31 under 0.50"),
        ("stale_fit", "the fit stopped"),
        ("wide_sigma", "sigma 0.42 m over 0.25 m"),
        ("stale_edge", "the map -> odom edge is"),
        ("no_edge", "no map -> odom edge"),
    ],
)
def test_a_pose_nobody_trusts_paints_nothing(node: DepthFusion, what: str, reason: str) -> None:
    """Each way the pose can be untrustworthy, one per case: the volume is untouched, the
    revolution is counted as withheld, and the report line carries the reason."""
    if what == "low_fit":
        node._on_fit(ros_stubs.Float32(data=0.31))
    elif what == "stale_fit":
        node._fit_at -= 60.0  # the topic stopped arriving a minute ago; the number is still 0.9
    elif what == "wide_sigma":
        node._on_sigma(String(data=json.dumps({"sigma_xy": 0.42, "sigma_yaw": 2.0})))
    elif what == "stale_edge":
        node._tf.buffer.transforms[("map", "odom")] = edge("map", "odom", SCAN_S - 5.0)
    elif what == "no_edge":
        del node._tf.buffer.transforms[("map", "odom")]

    node._on_scan_work(scan_msg())
    assert painted(node) == 0.0, "nothing of a pose nobody trusts reaches the volume"
    window = node._tally.take()
    assert window.counts["untrusted"] == 1 and window.counts["revolutions"] == 0
    assert reason in window.notes["untrusted"]


def test_the_gate_off_paints_at_whatever_pose_tf_gives(node: DepthFusion) -> None:
    """The old behaviour stays reachable (CLAUDE.md rule 19), which is also how SLAM mode runs:
    no tracker speaks there, so the launch brings the gate up off and every revolution is
    integrated at the pose TF has."""
    node._on_fit(ros_stubs.Float32(data=0.0))
    assert node.set_parameters([Parameter("lidar_fit_gate", value=False)])[0].successful
    node._on_scan_work(scan_msg())
    assert painted(node) > 0.0
    assert node._tally.take().counts["untrusted"] == 0


def test_a_sigma_nobody_publishes_is_not_a_refusal(node: DepthFusion) -> None:
    """The topic may not exist yet: an absent sigma leaves the fit gate as the whole test, and a
    malformed message is counted and ignored rather than stopping the room being painted."""
    assert node._sigma_xy_m is None
    node._on_sigma(String(data="not json at all"))
    node._on_scan_work(scan_msg())
    assert painted(node) > 0.0
    counts = node._tally.take().counts
    assert counts["revolutions"] == 1 and counts["bad_sigma"] == 1


def test_the_report_line_says_what_was_withheld_and_whether_a_sigma_speaks(
    node: DepthFusion,
) -> None:
    """A withheld revolution must be visible without a debugger: the count, the last reason and
    the sigma (or that nobody publishes one) are in the world line of every report."""
    node._on_fit(ros_stubs.Float32(data=0.10))
    node._on_scan_work(scan_msg())
    line = node._world_line(node._tally.take())
    assert (
        "lidar revolutions withheld: 1 (pose not trusted: fit 0.10 under 0.50 and no sigma"
        " to vouch for it)"
    ) in line
    assert "no /localization/sigma" in line
