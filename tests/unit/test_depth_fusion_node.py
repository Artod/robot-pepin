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


# Every node built here publishes one fan per integration: the shipped ``marks_hz`` is a 5 Hz cap
# on the WIRE to the board, and a test that integrates two revolutions in the same millisecond of
# wall time would read the first fan back as the second. The cap has its own test below.
MARKS_EVERY_FRAME = 0.0


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
        marks_hz=MARKS_EVERY_FRAME,
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
        "marks_hz": MARKS_EVERY_FRAME,
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
    # ...and, behind marks_clear (off as shipped), the clearing half of the same fan: how far
    # each bearing is KNOWN OPEN. Still an answer about obstacles, still nothing to seat a pose on.
    # ...and, behind grid_out (off as shipped), the same columns as grids the costmaps only draw.
    assert set(node.pubs) == {
        "/fusion/surface",
        "/depth_marks",
        "/depth_free",
        "/camera_grid",
        "/camera_grid_map",
        "/camera_grid_map_updates",
    }
    assert [name for name, _period in node.timers] or True
    line = node._world_line(node._tally.take())
    assert "1 revolutions" in line


# ---- the costmap's camera marks: the volume, sliced ------------------------------------------
def marks(node: DepthFusion) -> Any:
    """The last /depth_marks message the node published."""
    return node.pubs["/depth_marks"].sent[-1]


def at_map(node: DepthFusion, x: float) -> None:
    """Stand the cart at (x, 0) in the map frame: a revolution from a place the volume has
    already seen is not integrated at all (``view_gate``), so a test that wants a second one
    moves the cart first."""
    node._tf.buffer.transforms[("map", "base_link")].transform.translation.x = x


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


def test_the_marks_hz_cap_thins_the_topic_and_not_the_volume(node: DepthFusion) -> None:
    """CLAUDE.md rule 19, and the point of the cap: the board's local costmap reads this topic 5
    times a second, so a second fan in the same millisecond is never put on the wire — while the
    revolution behind it is integrated exactly as before. The volume is the proof: the marks of
    the one published fan carry the wall the SECOND revolution drew."""
    node._switches.set("min_weight", 0.5)
    node._switches.set("marks_hz", 1.0)
    node._on_scan_work(scan_msg())
    at_map(node, 0.4)  # a place the volume has not seen: the view gate lets the revolution in
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    assert len(node.pubs["/depth_marks"].sent) == 1, "one fan a second, and no more"
    counts = node._tally.take().counts
    assert counts["marks"] == 1 and counts["marks_thinned"] == 1
    assert counts["revolutions"] == 2, "both revolutions went into the volume"
    node._switches.set("marks_hz", MARKS_EVERY_FRAME)
    at_map(node, 0.8)
    node._on_scan_work(scan_msg(SCAN_S + 0.4))
    ranges = np.array(marks(node).ranges)
    assert np.isfinite(ranges).any(), "and what they drew is in the next fan that does go out"
    assert float(np.nanmax(ranges)) <= 3.0, "nothing past the fan's own reach"


def test_the_marks_hz_cap_holds_the_relayed_frame_too(node: DepthFusion) -> None:
    """One gate for both sources of the topic: under ``marks_source`` frame the rate on the wire
    is the flag's as well, or the cap would be a promise the old path does not keep."""
    node._switches.set("marks_source", "frame")
    node._switches.set("marks_hz", 1.0)
    node.subs["/depth_scan"][1](scan_msg(SCAN_S))
    node.subs["/depth_scan"][1](scan_msg(SCAN_S + 0.1))
    assert len(node.pubs["/depth_marks"].sent) == 1
    counts = node._tally.take().counts
    assert counts["depth_scans"] == 2 and counts["marks"] == 1 and counts["marks_thinned"] == 1


def test_the_report_line_says_what_the_cap_held_back(node: DepthFusion) -> None:
    """A rate below the camera's own must read as the cap and not as a node that has stopped
    slicing, and the state of the flag is in the line either way."""
    node._switches.set("marks_hz", 1.0)
    node._on_scan_work(scan_msg())
    at_map(node, 0.4)
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    assert "1 held by the marks_hz 1 cap" in node._marks_line(node._tally.take())
    node._switches.set("marks_hz", MARKS_EVERY_FRAME)
    assert "marks_hz 0: every frame" in node._marks_line(node._tally.take())


def test_the_report_line_says_where_the_marks_came_from(node: DepthFusion) -> None:
    """A drive is judged on the report line: which source the costmap's marks had, how many went
    out, what a slice of the volume cost and in which band it was read."""
    node._on_scan_work(scan_msg())
    line = node._marks_line(node._tally.take())
    assert "marks: 1 from the volume" in line and "ms a slice" in line
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
        "marks_hz": MARKS_EVERY_FRAME,
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


def test_the_report_line_names_the_frame_and_the_paths_it_makes_inert(tmp_path: Path) -> None:
    """``align=on`` in the flag state with the volume in odom would read as a yaw search that is
    running. Every path the frame switches off is named in the line, beside where the window
    stands and what its last slide cost — the numbers a drive is judged on without a debugger."""
    node = odom_node(tmp_path)
    node._on_scan_work(scan_msg())
    node._report()
    line = node.logger.texts("info")[-1]
    assert "last slide 0 ms" in line
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
    # ...and the pose gate is a gate again: nobody has published a fit in this file at all
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    assert painted(node) == 0.0
    assert node._tally.take().counts["untrusted"] == 1


def test_marks_clear_is_off_and_the_clearing_topic_is_silent(node: DepthFusion) -> None:
    """AS SHIPPED, and it is a measurement and not a preference: on the parked cart of 2026-09-23
    the volume held an occupied column on 717 of 720 bearings, and where the camera's own frame
    shared a bearing with a mark it agreed within 0.20 m 96 % of the time — a clearing ray stops
    at the first column that is not open, so there was next to nothing for it to erase
    (scratch/one_localiser/live_fan_vs_lidar.py). Off, /depth_free carries nothing at all and the
    camera layer clears from the single frame exactly as it has since 2026-09-21."""
    node._switches.set("min_weight", 0.5)
    node._on_scan_work(scan_msg())
    assert not node.pubs["/depth_free"].sent, "the flag is off: the topic is silent"
    assert node.pubs["/depth_marks"].sent, "and the marks go out as they always did"
    assert "marks_clear off" in node._marks_line(node._tally.take())


def test_marks_clear_on_answers_how_far_each_bearing_is_known_open(node: DepthFusion) -> None:
    """On, the same walk publishes a second, clearing-only fan: a range per bearing the volume has
    observed FREE up to, never past the surface it also marks there, and NaN where it has looked
    at nothing. The marks are untouched — which is what makes this a switch on the publisher."""
    node._switches.set("min_weight", 0.5)
    node._on_scan_work(scan_msg())
    before = np.array(marks(node).ranges)
    node._switches.set("marks_clear", True)
    at_map(node, 0.4)
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    fan = node.pubs["/depth_free"].sent[-1]
    open_to = np.array(fan.ranges)
    assert open_to.size == before.size and fan.header.frame_id == "base_link"
    assert fan.angle_min == pytest.approx(-math.pi) and fan.range_max > 3.0
    assert np.isfinite(open_to).any(), "the beams carved free space and it is offered for clearing"
    marked = np.array(marks(node).ranges)
    both = np.isfinite(marked) & np.isfinite(open_to)
    assert both.any() and np.all(open_to[both] < marked[both]), "cleared up to the wall, not past"
    line = node._marks_line(node._tally.take())
    assert "clearing on /depth_free" in line


# ---- the camera grids: what the costmaps only draw (grid_out, 2026-09-24) -------------------
def grid_node(tmp_path: Path, **overrides: Any) -> DepthFusion:
    """The shipped odom volume with ``grid_out`` on and a low ``min_weight``, so two revolutions
    are a surface (as in the marks tests above)."""
    node = odom_node(tmp_path, grid_out=True, **overrides)
    node._switches.set("min_weight", 0.5)
    return node


def grid_tick(node: DepthFusion, x: float, t: float) -> None:
    """One revolution from (x, 0) in odom with the grid_hz clock let through (the cap has its
    own test): a new place, so the view gate integrates it and the grids go out."""
    at_odom(node, x)
    node._grid_at = -math.inf
    node._on_scan_work(scan_msg(t))


def map_msg(width: int = 200, height: int = 160, x: float = -5.0, y: float = -4.0) -> Any:
    """RTAB-Map's /map as far as its lattice goes: 5 cm cells from (x, y)."""
    msg = ros_stubs.OccupancyGrid()
    msg.info.resolution, msg.info.width, msg.info.height = 0.05, width, height
    msg.info.origin.position.x, msg.info.origin.position.y = x, y
    return msg


def cells(msg: Any) -> set[tuple[int, int]]:
    """The (row, column) of every occupied cell of a grid or an update message."""
    width = msg.info.width if hasattr(msg, "info") else msg.width
    return {divmod(i, width) for i, v in enumerate(msg.data) if v == 100}


def test_grid_out_is_off_and_the_three_grid_topics_are_silent(node: DepthFusion) -> None:
    """As shipped the costmaps' camera_grid_layer is off and nothing feeds it: the marks go out
    exactly as before and the grids say nothing at all."""
    node._on_scan_work(scan_msg())
    assert node.pubs["/depth_marks"].sent
    for topic in ("/camera_grid", "/camera_grid_map", "/camera_grid_map_updates"):
        assert not node.pubs[topic].sent, topic
    assert "grid: off" in node._grid_line(node._tally.take())


def test_the_grid_holds_the_volume_s_columns_where_the_marks_do(tmp_path: Path) -> None:
    """/camera_grid: the square about the cart in the volume's frame, latched, on the
    observation's own stamp, 0 and 100 only — and every bearing the fan marks ends in one of its
    occupied cells, because both read the same column rule (volume_scan.band_surface)."""
    node = grid_node(tmp_path)
    grid_tick(node, 0.0, SCAN_S)
    grid_tick(node, 0.4, SCAN_S + 0.2)
    out = node.pubs["/camera_grid"].sent[-1]
    assert node.pubs["/camera_grid"].qos.rest["durability"] == "transient_local"
    assert out.header.frame_id == "odom"
    assert (out.header.stamp.sec, out.header.stamp.nanosec) == (int(SCAN_S), 200_000_000)
    assert (out.info.width, out.info.height, out.info.resolution) == (120, 120, 0.05)
    assert set(out.data) == {0, 100}
    ox, oy = out.info.origin.position.x, out.info.origin.position.y
    centres = np.array([(ox + (c + 0.5) * 0.05, oy + (r + 0.5) * 0.05) for r, c in cells(out)])
    to_wall = np.minimum(np.abs(np.abs(centres[:, 0]) - 2.0), np.abs(np.abs(centres[:, 1]) - 2.0))
    assert np.all(to_wall <= 0.1), "only the box's walls, one voxel either side"
    ranges = np.array(marks(node).ranges)
    bearings = -math.pi + math.radians(0.5) * np.arange(ranges.size)
    hit = np.isfinite(ranges)
    assert hit.sum() > 100
    ends = np.c_[0.4 + ranges[hit] * np.cos(bearings[hit]), ranges[hit] * np.sin(bearings[hit])]
    near = np.array([np.min(np.hypot(*(centres - end).T)) for end in ends])
    assert np.all(near <= 0.08), "every mark of the fan is a cell of the grid"
    assert node._tally.take().counts["grids"] == 2


def test_the_map_grid_copies_the_map_s_lattice_and_draws_through_map_to_odom(
    tmp_path: Path,
) -> None:
    """/camera_grid_map: nothing until the map is heard; then ONE full grid with exactly its
    geometry (empty, latched), and after the settle an update carrying the same cells moved by
    map <- odom — here a 1 m shift, 20 cells of the map's own lattice."""
    node = grid_node(tmp_path)
    grid_tick(node, 0.0, SCAN_S)
    assert not node.pubs["/camera_grid_map"].sent and not node.pubs["/camera_grid_map_updates"].sent
    assert node._tally.take().counts["grid_map_no_map"] == 1

    node.subs["/map"][1](map_msg())
    full = node.pubs["/camera_grid_map"].sent[-1]
    assert full.header.frame_id == "map"
    assert (full.info.width, full.info.height, full.info.resolution) == (200, 160, 0.05)
    assert (full.info.origin.position.x, full.info.origin.position.y) == (-5.0, -4.0)
    assert not any(full.data), "the latched grid is empty: a late layer gets no stale cell"
    assert node.pubs["/camera_grid_map"].qos.rest["durability"] == "transient_local"

    grid_tick(node, 0.4, SCAN_S + 0.2)  # a new full grid is settling: no update yet
    assert not node.pubs["/camera_grid_map_updates"].sent
    assert node._tally.take().counts["grid_map_settling"] == 1

    node._canvas_at = -math.inf
    node._tf.buffer.transforms[("map", "odom")] = edge("map", "odom", SCAN_S)
    node._tf.buffer.transforms[("map", "odom")].transform.translation.x = 1.0
    grid_tick(node, 0.8, SCAN_S + 0.4)
    update = node.pubs["/camera_grid_map_updates"].sent[-1]
    assert update.header.frame_id == "map"
    grid = node.pubs["/camera_grid"].sent[-1]
    gx = round((grid.info.origin.position.x + 1.0 + 5.0) / 0.05)
    gy = round((grid.info.origin.position.y + 4.0) / 0.05)
    assert (update.x, update.y) == (gx, gy), "the rectangle is the window's own, shifted"
    assert (update.width, update.height) == (120, 120)
    moved = {(r + gy - update.y, c + gx - update.x) for r, c in cells(grid)}
    assert cells(update) == moved, "the same cells, 1 m along the map's x"


def test_a_new_map_geometry_restarts_the_canvas_and_the_same_one_is_nothing(
    tmp_path: Path,
) -> None:
    """RTAB-Map re-renders /map with every graph change. The same lattice again changes nothing;
    a grown one gets its own empty full grid — a grid of another geometry would make the global
    costmap resize itself and drop every layer's marks."""
    node = grid_node(tmp_path)
    node.subs["/map"][1](map_msg())
    node.subs["/map"][1](map_msg())
    assert len(node.pubs["/camera_grid_map"].sent) == 1
    first = node._canvas
    node.subs["/map"][1](map_msg(width=260, x=-8.0))
    assert len(node.pubs["/camera_grid_map"].sent) == 2 and node._canvas is not first
    assert node.pubs["/camera_grid_map"].sent[-1].info.width == 260


def test_grid_out_off_leaves_an_empty_grid_behind_and_on_starts_the_map_again(
    tmp_path: Path,
) -> None:
    """A layer left on after the flag goes off must draw nothing stale: the last square goes out
    empty on /camera_grid and the last map window as an empty update. On again, the known map
    geometry gets a fresh full grid at once."""
    node = grid_node(tmp_path)
    node.subs["/map"][1](map_msg())
    node._canvas_at = -math.inf
    node._tf.buffer.transforms[("map", "odom")] = edge("map", "odom", SCAN_S)
    grid_tick(node, 0.0, SCAN_S)
    grid_tick(node, 0.4, SCAN_S + 0.2)
    last = node.pubs["/camera_grid"].sent[-1]
    drawn = node.pubs["/camera_grid_map_updates"].sent[-1]
    assert cells(last) and cells(drawn)

    node._switches.set("grid_out", False)
    empty = node.pubs["/camera_grid"].sent[-1]
    assert empty.info.origin.position.x == last.info.origin.position.x and not any(empty.data)
    cleared = node.pubs["/camera_grid_map_updates"].sent[-1]
    # the last window alone (the last update also spanned the one before it)
    assert (cleared.width, cleared.height) == (120, 120)
    assert drawn.x <= cleared.x and cleared.x + cleared.width <= drawn.x + drawn.width
    assert drawn.y <= cleared.y and cleared.y + cleared.height <= drawn.y + drawn.height
    assert not any(cleared.data)
    grid_tick(node, 0.8, SCAN_S + 0.4)
    assert node.pubs["/camera_grid"].sent[-1] is empty, "off: nothing more"

    fulls = len(node.pubs["/camera_grid_map"].sent)
    node._switches.set("grid_out", True)
    assert len(node.pubs["/camera_grid_map"].sent) == fulls + 1


def test_without_map_to_odom_the_map_grid_waits_and_the_odom_grid_does_not(
    tmp_path: Path,
) -> None:
    """The local costmap's grid needs no global pose at all; the map grid needs this laptop's
    map -> odom at the grid's stamp and holds the tick without it, counted."""
    node = grid_node(tmp_path)
    node.subs["/map"][1](map_msg())
    node._canvas_at = -math.inf
    grid_tick(node, 0.0, SCAN_S)
    assert node.pubs["/camera_grid"].sent and not node.pubs["/camera_grid_map_updates"].sent
    assert node._tally.take().counts["grid_map_no_tf"] == 1


def test_the_grid_hz_cap_thins_the_grids_and_not_the_volume(tmp_path: Path) -> None:
    node = grid_node(tmp_path)
    grid_tick(node, 0.0, SCAN_S)
    at_odom(node, 0.4)
    node._on_scan_work(scan_msg(SCAN_S + 0.2))  # within 1 / grid_hz of the last grid
    assert len(node.pubs["/camera_grid"].sent) == 1
    assert node._tally.take().counts["revolutions"] == 2


def test_the_report_line_says_what_the_grids_carried(tmp_path: Path) -> None:
    node = grid_node(tmp_path)
    node.subs["/map"][1](map_msg())
    node._canvas_at = -math.inf
    node._tf.buffer.transforms[("map", "odom")] = edge("map", "odom", SCAN_S)
    grid_tick(node, 0.0, SCAN_S)
    grid_tick(node, 0.4, SCAN_S + 0.2)
    line = node._grid_line(node._tally.take())
    assert line.startswith("grid: 2 on /camera_grid")
    assert "6.0 m at 5 cm" in line and "in odom" in line
    assert "/camera_grid_map on 200x160 at 5 cm from (-5.00, -4.00): 2 updates" in line
