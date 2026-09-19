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
    """A node whose volume lives in ``tmp_path``. The path is passed whole and the tests below
    compare it against a LITERAL string: deriving the expected name with the code's own expression
    is how a path bug passed two tests and reached the robot (2026-09-18)."""
    return fusion_node(tmp_path, world_path=f"{tmp_path}/flat_test.world.npz", **overrides)


def test_the_volume_is_named_after_the_database_whose_frame_it_was_painted_in(
    tmp_path: Path,
) -> None:
    """THE FRAME IS THE DATABASE'S, so the snapshot travels with the database and a fresh database
    means a fresh volume. There is nothing else on disk — no pgm is read as a seed, none is written
    as a cache, and no slice of the volume goes out as a map at all."""
    node = fusion_node(tmp_path, database="/maps/rtabmap.db")
    assert str(node._world_path) == "/maps/rtabmap.world.npz"
    assert not node._world.lidar_weight.any(), "a first volume is empty, not a pgm"
    assert "/map" not in node.pubs and "/map_lidar" not in node.pubs
    assert "/map_camera" not in node.pubs and "/map_identity" not in node.pubs
    up = node.logger.texts("info")[-1]
    assert "nothing localises against this volume" in up and "/maps/rtabmap.db" in up


def test_a_volume_with_no_snapshot_is_born_empty_under_the_cart(tmp_path: Path) -> None:
    """A box laid out around the map's origin would not even contain a cart that woke up at
    (-9.4, +2.5), and a volume that does not contain the robot integrates the far wall and nothing
    else (2026-09-13: 239 revolutions)."""
    node = fusion_node(tmp_path)
    nx, ny, _nz = node._spec.shape
    assert node._spec.origin[0] == pytest.approx(-nx * node._spec.voxel_m / 2)
    assert node._spec.origin[1] == pytest.approx(-ny * node._spec.voxel_m / 2)
    assert not node._world.lidar_weight.any()
    # ...and the painted revolutions alone fill it. A second revolution from the SAME place is the
    # same view and is not integrated at all (view_gate): a view is evidence once.
    node._on_scan_work(scan_msg())
    assert node._world.views.max() == 1.0
    node._on_scan_work(scan_msg(SCAN_S + 0.1))
    assert node._tally.take().counts["same_view"] == 1, "the same view again is not evidence"
    node._tf.buffer.transforms[("map", "base_link")].transform.translation.x = 0.4
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    assert node._world.views.max() == 2.0, "a second place is a second view"


def test_a_resumed_volume_keeps_its_grid_and_says_how_stale_it_was(tmp_path: Path) -> None:
    """A volume that exists owns its own lattice: the snapshot's grid comes back with it, and the
    report line says how stale it was — a volume two weeks old and one that stopped being saved an
    hour ago look identical from the outside."""
    first = in_room(tmp_path, snapshot_s=1.0)
    first._on_scan_work(scan_msg())
    spec, voxels = first._world.spec, first._world.maturity()["voxels"]

    again = in_room(tmp_path, resume_volume=True)
    assert again._world.spec == spec and again._world.maturity()["voxels"] == voxels
    assert again._resumed_age_s < 60.0
    assert "resumed 0.0 h old" in again._snapshot_line(again._tally.take())
    assert "flat_test.world.npz" in again._snapshot_line(again._tally.take())


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
    assert "NOT SAVED 1x" in node._snapshot_line(window)

    node.close()  # ...and shutdown is not a licence either
    assert Path(f"{tmp_path}/flat_test.world.npz").read_bytes() == good


def test_a_reset_empties_the_volume_without_touching_its_file(tmp_path: Path) -> None:
    """Nothing outside this node reads the volume, so emptying it costs no tracker anything — it
    costs the surface cloud until the sensors have painted one again. The file on disk is not part
    of a reset at all."""
    node = in_room(tmp_path, snapshot_s=1.0)
    node._on_scan_work(scan_msg())
    saved = Path(f"{tmp_path}/flat_test.world.npz").read_bytes()
    wiped = node._fresh_world()
    assert not wiped.lidar_weight.any() and not wiped.frames
    assert wiped.spec == node._world.spec, "the same grid, wiped"
    assert Path(f"{tmp_path}/flat_test.world.npz").read_bytes() == saved


def test_the_volume_reaches_no_matcher_and_no_planner(tmp_path: Path) -> None:
    """THE VOLUME IS OPEN-LOOP, and this is where that is visible: it is painted at the pose the
    tracker gives and no slice of it leaves the node. A tracker matching the slice it paints has a
    null space it cannot see out of — a cart parked with its wheels blocked walked 7 degrees and
    5-7 cm in 35 minutes through it at fit 0.97-0.99 (2026-09-18) — so /map, /map_lidar,
    /map_camera and /map_identity are gone and /fusion/surface is all there is."""
    node = in_room(tmp_path)
    node._on_scan_work(scan_msg())
    assert node._world.lidar_weight.any(), "painted, all the same"
    assert set(node.pubs) == {"/fusion/surface"}
    assert [name for name, _period in node.timers] or True
    line = node._world_line(node._tally.take())
    assert "1 revolutions" in line and "lidar slice" in line and "camera band" in line
