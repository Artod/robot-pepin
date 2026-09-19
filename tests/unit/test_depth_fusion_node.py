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


# ---- one map, one file: which room the volume is, what it resumes and what it saves ----------
def fusion_node(tmp_path: Path, **overrides: Any) -> DepthFusion:
    """A fusion node with the laser mount and a fresh map -> odom edge in TF, its pose gate on
    and a tracker that has just reported a good fit — the fixture's node with the parameters a
    test wants to move (room, world_path, resume_volume, snapshot_s)."""
    params: dict[str, Any] = {
        "config": str(small_config(tmp_path)),
        "lidar_config": str(REPO / "config" / "lidar.json"),
        "resume_volume": False,
        "snapshot_s": 0.0,
        "imu_lean": False,
    }
    params.update(overrides)
    with ros_stubs.parameters(**params):
        node = DepthFusion()
    buffer = node._tf.buffer
    buffer.transforms[("base_link", "laser")] = edge("base_link", "laser", SCAN_S, z=0.383)
    buffer.transforms[("map", "odom")] = edge("map", "odom", SCAN_S)
    buffer.transforms[("map", "base_link")] = edge("map", "base_link", SCAN_S)
    node._on_fit(ros_stubs.Float32(data=0.9))
    return node


def in_room(tmp_path: Path, **overrides: Any) -> DepthFusion:
    """A node in a named room whose volume lives in ``tmp_path``. The path is passed whole and
    the tests below compare it against a LITERAL string: deriving the expected name with the
    code's own expression is how a path bug passed two tests and reached the robot (2026-09-18)."""
    return fusion_node(
        tmp_path, room="flat_test", world_path=f"{tmp_path}/flat_test.world.npz", **overrides
    )


def test_a_room_names_the_only_file_the_volume_has(tmp_path: Path) -> None:
    """ONE MAP, ONE FILE, AND NO PICTURE OF IT. The room names the volume's snapshot and there is
    nothing else on disk — no pgm is read as a seed and none is written as a cache."""
    node = fusion_node(tmp_path, room="flat_test")
    assert str(node._world_path) == "/maps/flat_test.world.npz"
    assert not hasattr(node, "_export_path"), "nothing in the loop exports a picture"
    assert node._world.identity.provenance == "room:flat_test"
    assert not node._world.lidar_weight.any(), "a room's first volume is empty, not a pgm"


def test_an_unrecognised_place_is_born_empty_under_the_cart(tmp_path: Path) -> None:
    """Nothing named the room, so the box is centred on the start (a box laid out for another
    flat would not even contain the cart) and the map is born with an identity of its own."""
    node = fusion_node(tmp_path)
    nx, ny, _nz = node._spec.shape
    assert node._spec.origin[0] == pytest.approx(-nx * node._spec.voxel_m / 2)
    assert node._spec.origin[1] == pytest.approx(-ny * node._spec.voxel_m / 2)
    assert node._world.identity.provenance == "fresh"
    assert node._world.identity.token and not node._world.lidar_weight.any()
    # ...and the painted revolutions alone make it a map: nothing here needs a pgm. Two of them,
    # from two PLACES — a revolution is worth LidarLaw.hit_weight 1.0 per cell against
    # map_min_weight 2.0, and a second one from the same place is the same view and is not
    # integrated at all (view_gate). That is the regime the wake-up runs in: the room grows as the
    # cart moves, and a cell joins the matcher's slice the moment two places have seen it.
    node._on_scan_work(scan_msg())
    assert node._world.lidar_slice(node._map_law()).counts()["occupied"] == 0
    node._on_scan_work(scan_msg(SCAN_S + 0.1))
    assert node._tally.take().counts["same_view"] == 1, "the same view again is not evidence"
    node._tf.buffer.transforms[("map", "base_link")].transform.translation.x = 0.4
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    assert node._world.lidar_slice(node._map_law()).counts()["occupied"] > 0


def test_a_recognition_names_the_room_and_a_late_one_is_refused(tmp_path: Path) -> None:
    """KNOWN OR FRESH IS DETECTED. A place recogniser names the room and that room's volume is
    resumed — but only while this one is still empty: after the first revolution a swap would
    throw away the room the session has measured."""
    first = in_room(tmp_path, snapshot_s=1.0)
    first._on_scan_work(scan_msg())
    token, voxels = first._world.identity.token, first._world.maturity()["voxels"]
    assert Path(f"{tmp_path}/flat_test.world.npz").exists()

    node = fusion_node(tmp_path, resume_volume=True)
    assert node._room == "" and node._world.identity.provenance == "fresh"
    node._on_room(String(data=json.dumps({"room": f"{tmp_path}/flat_test"})))
    assert node._room.endswith("flat_test")
    assert node._world.identity.token == token, "the recognised room, not a new one"
    assert node._world.maturity()["voxels"] == voxels

    node._on_room(String(data=json.dumps({"room": "somewhere_else"})))
    window = node._tally.take()
    assert window.counts["room_refused"] == 1 and node._room.endswith("flat_test")
    assert "somewhere_else" in window.notes["room_refused"]
    node._on_room(String(data="not json"))
    assert node._tally.take().counts["bad_room"] == 1


def test_a_resumed_room_keeps_its_identity_grid_and_age(tmp_path: Path) -> None:
    """A room that exists owns its own lattice and its own name: the snapshot's grid comes back
    with it, the id its snapshot carries is kept, and the report line says how stale it was."""
    first = in_room(tmp_path, snapshot_s=1.0)
    first._on_scan_work(scan_msg())
    token, spec = first._world.identity.token, first._world.spec

    again = in_room(tmp_path, resume_volume=True)
    assert again._world.identity.token == token and again._world.spec == spec
    assert again._resumed_age_s < 60.0
    assert "resumed 0.0 h old" in again._map_line(again._tally.take())
    assert "flat_test" in again._map_line(again._tally.take())


def test_a_snapshot_taken_after_trust_was_lost_never_replaces_the_last_good_one(
    tmp_path: Path,
) -> None:
    """One false word, painted and saved, is permanent damage (ros/maps/
    world_live.npz.mess-20260917). So the volume is written only while the painting is trusted,
    and a run that has lost the tracker leaves the file it woke up on exactly where it is."""
    node = in_room(tmp_path, snapshot_s=1.0)
    node._on_scan_work(scan_msg())
    good = Path(f"{tmp_path}/flat_test.world.npz").read_bytes()
    assert node._tally.take().counts["snapshots"] == 1

    node._on_fit(ros_stubs.Float32(data=0.10))  # the tracker is lost from here on
    node._on_scan_work(scan_msg())
    node._trust.last_trusted_s -= 60.0  # ...and a minute of it has gone by
    node._snapshot()
    window = node._tally.take()
    assert window.counts["snapshots"] == 0 and window.counts["snapshot_refused"] == 1
    assert Path(f"{tmp_path}/flat_test.world.npz").read_bytes() == good
    assert "fit 0.10" in window.notes["snapshot_refused"]
    assert "NOT SAVED 1x" in node._map_line(window)

    node.close()  # ...and shutdown is not a licence either
    assert Path(f"{tmp_path}/flat_test.world.npz").read_bytes() == good


def test_a_reset_empties_the_room_without_touching_its_file(tmp_path: Path) -> None:
    """With no pgm to fall back to, a reset means empty — and that is safe because a grid with no
    known cell is refused by MapChoice before any tracker rebuilds on it. The file on disk is not
    part of a reset at all."""
    node = in_room(tmp_path, snapshot_s=1.0)
    node._on_scan_work(scan_msg())
    saved = Path(f"{tmp_path}/flat_test.world.npz").read_bytes()
    token = node._world.identity.token
    wiped = node._fresh_world()
    assert not wiped.lidar_weight.any() and not wiped.frames
    assert wiped.identity.token == token, "the same room, wiped"
    assert Path(f"{tmp_path}/flat_test.world.npz").read_bytes() == saved


def test_the_map_says_which_room_it_is_on_a_topic_of_its_own(tmp_path: Path) -> None:
    """A nav_msgs/OccupancyGrid carries nothing that says which room it is, so every consumer
    derives size@origin off the message. The minted token goes out beside it, with the legacy id
    of every grid this node publishes, so the consumers can be moved onto it one at a time."""
    node = in_room(tmp_path)
    node._on_scan_work(scan_msg())
    node._publish_map()
    sent = node.pubs["/map_identity"].sent
    assert len(sent) == 1, "latched and on change: an unchanged identity is an unchanged room"
    said = json.loads(sent[-1].data)
    assert said["id"] == node._world.identity.token
    assert said["from"] == "room:flat_test" and said["room"] == "flat_test"
    fields = node._world.lidar_slice(node._map_law()).message_fields()
    assert said["legacy"]["/map_lidar"] == fields.legacy_id()
    assert said["world_path"] == f"{tmp_path}/flat_test.world.npz"
    node._publish_map()
    assert len(node.pubs["/map_identity"].sent) == 1, "the same room is not said twice"


def test_the_matchers_read_the_frozen_reference_and_the_planner_reads_everything(
    tmp_path: Path,
) -> None:
    """The cure for the closed loop, in the node: /map_lidar and /map_camera carry the volume AS
    RESUMED wherever it knows the cell and this session's paint only where it does not, while /map
    and the snapshot stay complete. And with the reference frozen, /map_lidar's cells stop changing
    in a room the cart knows — so the board stops re-adopting, which costs it its matcher, its mask
    and its tracker every time."""
    # THE EARLIER SESSION: the room driven once, a voxel at a time so the cells accumulate the
    # map's own maturity (map_min_weight 2.0) from more than one place.
    first = in_room(tmp_path, snapshot_s=1.0)
    for i in range(8):
        first._tf.buffer.transforms[("map", "base_link")].transform.translation.x = 0.06 * i
        first._on_scan_work(scan_msg(SCAN_S + 0.1 * i))
    assert first._world.lidar_slice(first._map_law()).counts()["known"] > 0
    first._snapshot()  # the whole session on disk: the clock would have saved only its first scan

    node = in_room(tmp_path, resume_volume=True)
    assert node._reference.known > 0, "the resumed room is this session's gauge"
    assert node._resumed_age_s < 60.0
    node._tf.buffer.transforms[("map", "base_link")].transform.translation.x = 1.0
    node._on_scan_work(scan_msg(SCAN_S + 1.0))
    node._publish_map()
    line = node._world_line(node._tally.take())
    assert f"reference {node._reference.known} cells from a snapshot" in line
    assert "session paint" in line and "cells in unknown space" in line

    # the same room again: the cells a matcher reads have not moved, so nothing is republished
    node._tf.buffer.transforms[("map", "base_link")].transform.translation.x = 1.06
    node._on_scan_work(scan_msg(SCAN_S + 2.0))
    node._publish_map()
    before = len(node.pubs["/map_lidar"].sent)
    node._publish_map()
    assert len(node.pubs["/map_lidar"].sent) == before
    assert node._tally.take().counts["lidar_map_unchanged"] >= 1

    # ...and with the flag off a matcher is handed this session's paint again: the 2026-09-18 drift
    assert node.set_parameters([Parameter("frozen_reference", value=False)])[0].successful
    assert "reference: OFF" in node._world_line(node._tally.take())
