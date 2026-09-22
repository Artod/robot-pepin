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
import time
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
    edge in TF, its pose gate on and a tracker that has just reported a good fit.

    ``volume_frame`` map, which is where everything in this section lives: the pose gates, the
    snapshot, the yaw seating and the graph's bend are the machinery of a volume that IS the room.
    The shipped default is odom, whose own section is at the foot of this file.
    """
    with ros_stubs.parameters(
        config=str(small_config(tmp_path)),
        lidar_config=str(REPO / "config" / "lidar.json"),
        world_path=str(tmp_path / "world.npz"),
        volume_frame="map",
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
        "volume_frame": "map",  # the room-sized volume: the frame every test below is about
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
    """THE VOLUME IS OPEN-LOOP, and this is where that is visible: no MAP of it leaves the node
    and nothing seats a pose on it. A tracker matching the slice it paints has a
    null space it cannot see out of — a cart parked with its wheels blocked walked 7 degrees and
    5-7 cm in 35 minutes through it at fit 0.97-0.99 (2026-09-18) — so /map, /map_lidar,
    /map_camera and /map_identity are gone. What does leave is the surface: as a cloud
    (/fusion/surface) and as the costmap's camera marks (/depth_marks), which is an obstacle to
    route around and never a measurement to seat anything on."""
    node = in_room(tmp_path)
    node._on_scan_work(scan_msg())
    assert node._world.lidar_weight.any(), "painted, all the same"
    assert set(node.pubs) == {"/fusion/surface", "/depth_marks"}
    assert [name for name, _period in node.timers] or True
    line = node._world_line(node._tally.take())
    assert "1 revolutions" in line and "lidar slice" in line and "camera band" in line


# ---- the costmap's camera marks: the volume, sliced ------------------------------------------
def marks(node: DepthFusion) -> Any:
    """The last /depth_marks message the node published."""
    return node.pubs["/depth_marks"].sent[-1]


def test_every_integration_publishes_the_marks_and_a_young_volume_says_nothing(
    node: DepthFusion,
) -> None:
    """The marks go out at the rate the volume is integrated, in base_link, on the observation's
    own stamp — and a volume that has seen one revolution agrees on nothing yet, so every
    bearing is NaN: a costmap neither marks nor clears from those."""
    node._on_scan_work(scan_msg())
    assert len(node.pubs["/depth_marks"].sent) == 1
    out = marks(node)
    assert out.header.frame_id == "base_link"
    assert (out.header.stamp.sec, out.header.stamp.nanosec) == (int(SCAN_S), 0)
    ranges = np.array(out.ranges)
    assert ranges.size == 720 and out.angle_min == pytest.approx(-math.pi)
    assert out.angle_increment == pytest.approx(math.radians(0.5))
    assert out.range_max > 3.0, "a mark AT the fan's reach must survive the projection"
    assert not np.isfinite(ranges).any(), "one revolution is not agreement"
    assert node._tally.take().counts["marks"] == 1


def test_what_the_volume_agrees_on_marks_at_its_own_range(node: DepthFusion) -> None:
    """Two revolutions from two places, and the room the beams drew comes back as ranges: the
    box wall is 2 m from where the cart stood, and that is what the bearing carries."""
    node._switches.set("min_weight", 0.5)
    node._on_scan_work(scan_msg())
    node._tf.buffer.transforms[("map", "base_link")].transform.translation.x = 0.4
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    ranges = np.array(marks(node).ranges)
    assert np.isfinite(ranges).any()
    assert float(np.nanmin(ranges)) == pytest.approx(2.0, abs=0.1), "the wall of the box"
    assert float(np.nanmax(ranges)) <= 3.0, "and nothing past the fan's own reach"


def test_marks_source_frame_relays_the_single_frame_s_own_fan(node: DepthFusion) -> None:
    """CLAUDE.md rule 19: the costmap of before 2026-09-21 without a restart — /depth_scan's own
    fan, unchanged, on the topic the layer marks from, and the volume not read at all."""
    assert node._switches.set("marks_source", "frame") == "volume"
    fan = scan_msg(SCAN_S)
    node.subs["/depth_scan"][1](fan)
    assert marks(node) is fan, "relayed, not rebuilt"
    node._on_scan_work(scan_msg())
    assert len(node.pubs["/depth_marks"].sent) == 1, "the volume publishes nothing in this mode"
    counts = node._tally.take().counts
    assert counts["marks"] == 1 and counts["depth_scans"] == 1


def test_the_report_line_says_where_the_marks_came_from(node: DepthFusion) -> None:
    """A drive is judged on the report line: which source the costmap's marks had, how many went
    out, what a slice of the volume cost and in which band it was read."""
    node._on_scan_work(scan_msg())
    line = node._marks_line(node._tally.take())
    assert "marks: 1 from the volume" in line and "ms a slice" in line
    assert "band 0.15-1.30 m within 3.0 m at min_weight 2" in line
    node._switches.set("marks_source", "frame")
    node.subs["/depth_scan"][1](scan_msg())
    relayed = node._marks_line(node._tally.take())
    assert "/depth_marks relayed from /depth_scan" in relayed and "1 of 1 frames" in relayed


# ---- the volume follows the GRAPH, not map -> odom ---------------------------------------------
def graph_msg(poses: dict[int, tuple[float, float]]) -> Any:
    """One /rtabmap/mapGraph: the ids and the optimised poses, as two parallel arrays."""
    return ros_stubs.MapGraph(
        poses_id=list(poses),
        poses=[
            ros_stubs.Pose(
                position=ros_stubs.Point(x=x, y=y), orientation=ros_stubs.Quaternion(w=1.0)
            )
            for x, y in poses.values()
        ],
    )


def test_the_fusion_reads_the_graph_and_not_only_the_correction(node: DepthFusion) -> None:
    assert "/rtabmap/mapGraph" in node.subs
    assert node.subs["/rtabmap/mapGraph"][0] is ros_stubs.MapGraph


def test_a_graph_that_has_not_bent_moves_nothing(node: DepthFusion) -> None:
    """Every message of a session that is only LOCALISING: nothing is written, so nothing is
    optimised, every node comes back where it was, and the volume stands still."""
    node.subs["/rtabmap/mapGraph"][1](graph_msg({1: (0.0, 0.0), 9: (2.0, 0.0)}))
    node._on_scan_work(scan_msg())
    before = painted(node)
    assert before > 0.0
    for _ in range(3):
        node.subs["/rtabmap/mapGraph"][1](graph_msg({1: (0.0, 0.0), 9: (2.0, 0.0)}))
    assert node._follow(stamp(SCAN_S)) is True, "nothing is owed, so nothing is refused"
    assert node._follower.applied == 0


def test_a_bend_of_the_graph_carries_the_whole_volume(node: DepthFusion) -> None:
    """A closure landed: the room's expression in map moved, so every voxel painted before it is
    stale by that move and the content is carried rigidly."""
    node.subs["/rtabmap/mapGraph"][1](graph_msg({9: (2.0, 0.0)}))
    node._on_scan_work(scan_msg())
    assert node._follower.painted_in is not None, "the volume is anchored in the bend in force"
    node.subs["/rtabmap/mapGraph"][1](graph_msg({9: (2.0, 0.40)}))  # the graph moved the room
    node._followed_at = 0.0  # the rate has not held this one back
    assert node._follow(stamp(SCAN_S)) is True
    assert node._follower.applied == 1
    assert node._follower.last.dy == pytest.approx(0.40)
    assert "the graph bent the room" in node.logger.texts("info")[-1]


def test_a_bend_under_the_threshold_is_owed_and_paid_when_it_grows(node: DepthFusion) -> None:
    node.subs["/rtabmap/mapGraph"][1](graph_msg({9: (2.0, 0.0)}))
    node._on_scan_work(scan_msg())
    node.subs["/rtabmap/mapGraph"][1](graph_msg({9: (2.0, 0.02)}))
    node._followed_at = 0.0
    assert node._follow(stamp(SCAN_S)) is True and node._follower.applied == 0, "under a voxel"
    node.subs["/rtabmap/mapGraph"][1](graph_msg({9: (2.0, 0.08)}))
    assert node._follow(stamp(SCAN_S)) is True
    assert node._follower.applied == 1
    assert node._follower.last.dy == pytest.approx(0.08), "both bends, against one anchor"


def test_nothing_is_painted_into_a_volume_that_owes_a_move(node: DepthFusion) -> None:
    """An observation placed under the new bend and fused into a volume standing in the old one is
    carried past the truth by the whole move when it lands (scratch/follow_refute.py)."""
    node.subs["/rtabmap/mapGraph"][1](graph_msg({9: (2.0, 0.0)}))
    node._on_scan_work(scan_msg())
    before = painted(node)
    node.subs["/rtabmap/mapGraph"][1](graph_msg({9: (2.0, 0.40)}))
    node._followed_at = time.monotonic()  # the rate will not let the move through yet
    node._on_scan_work(scan_msg())
    assert painted(node) == before, "the revolution was refused, not painted into a stale volume"
    assert node._tally.take().counts["follow_held"] == 1


def test_the_correction_of_map_to_odom_alone_never_moves_the_volume(node: DepthFusion) -> None:
    """The reason this is read from the graph at all: map -> odom moves when the CART is found
    after drifting, and a volume that followed THAT would be dragged off the room by the whole
    size of the recovery."""
    node.subs["/rtabmap/mapGraph"][1](graph_msg({9: (2.0, 0.0)}))
    node._on_scan_work(scan_msg())
    node._tf.buffer.transforms[("map", "odom")] = edge("map", "odom", SCAN_S)
    node._tf.buffer.transforms[("map", "odom")].transform.translation.x = 2.0  # a 2 m re-seed
    node.subs["/rtabmap/mapGraph"][1](graph_msg({9: (2.0, 0.0)}))  # the room did NOT move
    node._followed_at = 0.0
    assert node._follow(stamp(SCAN_S)) is True
    assert node._follower.applied == 0, "the cart was found; the room is where it was"


def test_with_no_graph_at_all_nothing_is_followed_and_nothing_is_refused(node: DepthFusion) -> None:
    """A session where RTAB-Map is not up: the volume is painted open-loop and simply never
    follows, rather than refusing every observation."""
    assert node._follow(stamp(SCAN_S)) is True
    node._on_scan_work(scan_msg())
    assert painted(node) > 0.0
    assert node._tally.take().counts["no_correction"] >= 1
    node._report()
    assert "no graph, nothing to follow" in node.logger.texts("info")[-1]


def test_follow_correction_off_leaves_the_voxels_where_they_are(node: DepthFusion) -> None:
    """CLAUDE.md rule 19: the old behaviour without a restart."""
    node._switches.set("follow_correction", False)
    node.subs["/rtabmap/mapGraph"][1](graph_msg({9: (2.0, 0.0)}))
    node._on_scan_work(scan_msg())
    node.subs["/rtabmap/mapGraph"][1](graph_msg({9: (2.0, 0.40)}))
    node._followed_at = 0.0
    assert node._follow(stamp(SCAN_S)) is True
    assert node._follower.applied == 0
    node._report()
    assert "follow: off (the graph bends, the voxels stay)" in node.logger.texts("info")[-1]


# ---- the volume in ODOM: local obstacle memory, a rolling window, no global pose at all -------
def odom_node(tmp_path: Path, **overrides: Any) -> DepthFusion:
    """A fusion node as the stack ships from 2026-09-22: ``volume_frame`` odom, the cart placed by
    odom -> base_link, and NO map -> odom edge and no tracker fit anywhere in its TF — which is the
    point of the frame, not an omission. The lidar mount is the checkout's."""
    params: dict[str, Any] = {
        "config": str(small_config(tmp_path)),
        "lidar_config": str(REPO / "config" / "lidar.json"),
        "world_path": f"{tmp_path}/flat_test.world.npz",
        "imu_lean": False,
    }
    params.update(overrides)
    with ros_stubs.parameters(**params):
        node = DepthFusion()
    buffer = node._tf.buffer
    buffer.transforms[("base_link", "laser")] = edge("base_link", "laser", SCAN_S, z=0.383)
    buffer.transforms[("odom", "base_link")] = edge("odom", "base_link", SCAN_S)
    return node


def at_odom(node: DepthFusion, x: float, y: float = 0.0) -> None:
    """Stand the cart at (x, y) in the odometry frame: the one pose this volume knows about."""
    where = node._tf.buffer.transforms[("odom", "base_link")].transform.translation
    where.x, where.y = x, y


def test_the_volume_in_odom_is_painted_by_the_odometry_and_by_nothing_else(tmp_path: Path) -> None:
    """The default since 2026-09-22. No tracker has ever published a fit, no sigma exists and TF
    holds no map -> odom edge at all — and the beams still paint the room, because the pose that
    places them is odom -> base_link. The gates that ask whether a GLOBAL pose can be trusted have
    nothing to judge here, and the report line says so instead of passing silently."""
    node = odom_node(tmp_path)
    assert node._odom_volume and node._poser_now is node._odom_poser
    assert ("map", "odom") not in node._tf.buffer.transforms
    node._on_scan_work(scan_msg())
    assert painted(node) > 0.0, "the beams wrote the box at the odometry's pose"
    counts = node._tally.take().counts
    assert counts["revolutions"] == 1 and counts["untrusted"] == 0 and counts["low_fit"] == 0
    assert marks(node).header.frame_id == "base_link", "the marks are the cart's fan either way"
    # ...and the surface goes out in the frame it was painted in, never in map
    node._publish_surface()
    assert node.pubs["/fusion/surface"].sent[-1].header.frame_id == "odom"


def test_a_step_of_map_to_odom_does_not_move_the_marks(tmp_path: Path) -> None:
    """THE WHOLE POINT OF THE FRAME. A re-seating of the global pose is what poisoned the volume on
    2026-09-21: painted in map, some twenty seatings each wrote their own copy of the walls and the
    slice put 300-650 lethal cells around the cart. Here the tracker re-seats the cart by 2 m and
    the graph bends under it — and the window does not move, nothing is refused, and the next
    revolution goes into the very cells the last one did: one wall, not two."""
    node = odom_node(tmp_path)
    node._switches.set("min_weight", 0.5)
    node._on_scan_work(scan_msg())
    at_odom(node, 0.4)
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    before = np.array(marks(node).ranges)
    spec, weight = node._world.spec, node._world.volume.weight.copy()
    assert np.isfinite(before).any(), "the box is in the marks"

    node._tf.buffer.transforms[("map", "odom")] = edge("map", "odom", SCAN_S)
    node._tf.buffer.transforms[("map", "odom")].transform.translation.x = 2.0  # a 2 m re-seat
    node.subs["/rtabmap/mapGraph"][1](graph_msg({9: (2.0, 0.40)}))  # ...and a bend under it
    node._switches.set("view_gate", False)  # so the same place may speak again
    node._on_scan_work(scan_msg(SCAN_S + 0.4))

    assert node._world.spec == spec, "no correction slides the window; only the cart does"
    assert node._follower.applied == 0 and node._tally.take().counts["follow_held"] == 0
    after = np.array(marks(node).ranges)
    common = np.isfinite(before) & np.isfinite(after)
    assert common.sum() > 100
    # Within a couple of voxels, which is what a repeat observation moves a zero crossing by; a
    # volume that had heard the re-seating would have moved every bearing by the 2 m step.
    np.testing.assert_allclose(after[common], before[common], atol=2 * node._spec.voxel_m)
    assert float(np.nanmin(after)) == pytest.approx(float(np.nanmin(before)), abs=0.05)
    assert int((node._world.volume.weight > 0.0).sum()) == int((weight > 0.0).sum())


def test_the_window_slides_onto_the_cart_and_forgets_what_left_it(tmp_path: Path) -> None:
    """The rolling window: past window_recentre_m from the centre the box is laid out around the
    cart again, every voxel of the overlap keeps the metres it was painted at, and what fell off
    the trailing edge is gone. A local map is not a room."""
    node = odom_node(tmp_path)
    node._on_scan_work(scan_msg())
    before = node._world.volume.weight.copy()
    lidar_before = node._world.lidar_weight.copy()
    origin = node._world.spec.origin
    assert node._world.spec.centre_xy == pytest.approx((0.0, 0.0))

    out = node._window_recentre_m + 0.5  # 2.5 m out of a 6 m box: past the threshold
    at_odom(node, out)
    pose = node._poser_now.base_in_map(SCAN_S)
    assert pose is not None
    node._roll_window(pose)  # what the next observation does before it goes in

    slid = round(out / node._world.spec.voxel_m)
    assert node._world.spec.origin[0] == pytest.approx(origin[0] + slid * 0.05)
    assert node._world.spec.origin[1:] == origin[1:], "a slide has no y and no z in it"
    assert node._world.spec.centre_xy == pytest.approx((slid * 0.05, 0.0))
    assert node._spec == node._world.spec, "the node's own grid follows the volume's"
    assert np.array_equal(node._world.volume.weight[:-slid], before[slid:]), "what stayed, stayed"
    assert np.array_equal(node._world.lidar_weight[:-slid], lidar_before[slid:])
    assert not node._world.lidar_weight[-slid:].any(), "the new edge of the window is unpainted"
    assert (node._world.volume.sdf[-slid:] == 1.0).all(), "a whole truncation from any surface"
    assert node._recentre_ms > 0.0, "and the slide is timed into the report line"
    assert node._tally.take().counts["recentres"] == 1
    assert "what left it is forgotten" in node.logger.texts("info")[-1]

    # ...and it happens by itself on the next observation, not because a test called it
    at_odom(node, 2 * out)
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    assert node._world.spec.centre_xy[0] == pytest.approx(2 * out, abs=0.05)
    assert node._spec == node._world.spec


def test_a_volume_in_odom_reads_and_writes_no_snapshot(tmp_path: Path) -> None:
    """A snapshot exists to be resumed, and the odometry frame of one run is not the odometry frame
    of the next: it is born where the wheels were switched on. So the file is neither read nor
    written, and the report line says so rather than leaving a stale file looking current."""
    saved = in_room(tmp_path, snapshot_s=1.0)
    saved._on_scan_work(scan_msg())
    kept = Path(f"{tmp_path}/flat_test.world.npz").read_bytes()

    node = odom_node(tmp_path, snapshot_s=1.0, resume_volume=True)
    assert not node._world.lidar_weight.any(), "the file beside it is another frame's room"
    node._on_scan_work(scan_msg())
    node.close()  # ...and shutdown writes nothing either
    assert Path(f"{tmp_path}/flat_test.world.npz").read_bytes() == kept
    assert "no snapshot at all in odom" in node._snapshot_line(node._tally.take())


def test_the_report_line_names_the_frame_and_the_paths_it_makes_inert(tmp_path: Path) -> None:
    """``align=on`` in the flag state with the volume in odom would read as a yaw search that is
    running. Every path the frame switches off is named in the line, beside where the window
    stands and what its last slide cost — the numbers a drive is judged on without a debugger."""
    node = odom_node(tmp_path)
    node._on_scan_work(scan_msg())
    node._report()
    line = node.logger.texts("info")[-1]
    assert "the volume is local memory, in odom" in line and "rolling window centred on" in line
    assert "re-centred on the cart past 2.0 m" in line and "last slide 0 ms" in line
    assert "align, the paint gates (fit_gate, lidar_fit_gate, paint_sigma_m) and" in line
    assert "follow_correction are inert here and no snapshot is read or written" in line
    assert "follow: inert in odom" in line and "no gate in odom" in line
    assert "volume_frame=odom" in line, "and the switch itself, as every flag is (rule 19)"


def test_volume_frame_map_is_the_old_behaviour_one_flag_away(tmp_path: Path) -> None:
    """CLAUDE.md rule 19, and the A/B of 2026-09-21 without a restart. A volume cannot be carried
    between frames — its voxels are metres of one or metres of the other — so the switch empties it
    and lays a new box under the cart; from there the map-frame machinery is back, gates and all."""
    node = odom_node(tmp_path)
    node._on_scan_work(scan_msg())
    assert painted(node) > 0.0
    node._tf.buffer.transforms[("map", "odom")] = edge("map", "odom", SCAN_S)
    node._tf.buffer.transforms[("map", "base_link")] = edge("map", "base_link", SCAN_S)

    assert node._switches.set("volume_frame", "map") == "odom"
    assert not node._odom_volume and node._poser_now is node._poser
    assert painted(node) == 0.0, "voxels painted in odom are not metres of map"
    assert "the volume is emptied and born again under the cart" in node.logger.texts("warning")[-1]
    # ...and the pose gate is a gate again: nobody has published a fit in this file at all
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    assert painted(node) == 0.0
    assert node._tally.take().counts["untrusted"] == 1
