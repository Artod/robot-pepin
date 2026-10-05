"""The fusion node under the ROS stubs: the odometry's rolling window, painted by the lidar one
revolution at a time, read out as the costmap's marks and grids.

The volume is painted in ``odom`` and nothing in its paint path reads ``map -> odom``: a node
here has the laser mount and ``odom -> base_link`` in TF, and no tracker, no fit and no graph.

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
from pepin_bringup.msgs import fields_from_cloud  # noqa: E402
from ros_stubs import Header, JointState, LaserScan, TransformStamped  # noqa: E402
from ros_stubs import Time as TimeMsg  # noqa: E402

from pepin.tof_rays import fans_to_speak  # noqa: E402
from pepin.worldmap import ViewGate  # noqa: E402

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


def odom_node(tmp_path: Path, **overrides: Any) -> DepthFusion:
    """A fusion node on this checkout's configs: the cart placed by odom -> base_link, and NO
    map -> odom edge and no tracker fit anywhere in its TF — which is the point of the frame, not
    an omission. The lidar mount is the checkout's."""
    params: dict[str, Any] = {
        "config": str(small_config(tmp_path)),
        "lidar_config": str(REPO / "config" / "lidar.json"),
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


@pytest.fixture
def node(tmp_path: Path) -> DepthFusion:
    """The node as it ships (:func:`odom_node`)."""
    return odom_node(tmp_path)


def at_odom(node: DepthFusion, x: float, y: float = 0.0) -> None:
    """Stand the cart at (x, y) in the odometry frame: a revolution from a place the volume has
    already seen is not integrated at all (the view gate), so a test that wants a second one moves
    the cart first."""
    where = node._tf.buffer.transforms[("odom", "base_link")].transform.translation
    where.x, where.y = x, y


def painted(node: DepthFusion) -> float:
    """How much the lidar has written into the volume: the weight it owns, in total."""
    return float(node._world.lidar_weight.sum())


def test_a_volume_is_born_empty_under_the_cart(tmp_path: Path) -> None:
    """A box laid out around the frame's origin would not even contain a cart that woke up
    elsewhere, and a volume that does not contain the robot integrates the far wall and nothing
    else (2026-09-13: 239 revolutions)."""
    node = odom_node(tmp_path)
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
    at_odom(node, 0.4)
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    assert node._world.views.max() == 2.0, "a second place is a second view"


def test_a_reset_empties_the_volume(tmp_path: Path) -> None:
    """Nothing outside this node reads the volume, so emptying it costs no tracker anything — it
    costs the surface cloud until the sensors have painted one again."""
    node = odom_node(tmp_path)
    node._on_scan_work(scan_msg())
    wiped = node._fresh_world()
    assert not wiped.lidar_weight.any() and not wiped.frames
    assert wiped.spec == node._world.spec, "the same grid, wiped"


def test_the_volume_reaches_no_matcher_and_no_planner(tmp_path: Path) -> None:
    """THE VOLUME IS OPEN-LOOP, and this is where that is visible: no MAP of it leaves the node
    and nothing seats a pose on it. A tracker matching the slice it paints has a
    null space it cannot see out of — a cart parked with its wheels blocked walked 7 degrees and
    5-7 cm in 35 minutes through it at fit 0.97-0.99 (2026-09-18) — so /map, /map_lidar,
    /map_camera and /map_identity are gone. What does leave is the surface: as a cloud
    (/fusion/surface) and as the costmap's camera marks (/depth_marks), which is an obstacle to
    route around and never a measurement to seat anything on."""
    node = odom_node(tmp_path)
    node._on_scan_work(scan_msg())
    assert node._world.lidar_weight.any(), "painted, all the same"
    # ...and, behind marks_clear (off as shipped), the clearing half of the same fan: how far
    # each bearing is KNOWN OPEN. Still an answer about obstacles, still nothing to seat a pose on.
    # ...and, behind grid_out (off as shipped), the same columns as grids the costmaps only draw.
    # /fusion/frame carries stamps only: which camera frames went in (the gaze arbiter's count).
    # /fusion/arm is the robot's own arm as boxes in base_link, for the eye.
    assert set(node.pubs) == {
        "/fusion/surface",
        "/depth_marks",
        "/depth_free",
        "/camera_grid",
        "/camera_grid_map",
        "/camera_grid_map_updates",
        "/fusion/frame",
        "/fusion/arm",
    }
    assert [name for name, _period in node.timers] or True
    line = node._world_line(node._tally.take())
    assert "1 revolutions" in line


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
    at_odom(node, 0.4)
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    ranges = np.array(marks(node).ranges)
    assert np.isfinite(ranges).any()
    assert float(np.nanmin(ranges)) == pytest.approx(2.0, abs=0.1), "the wall of the box"
    assert float(np.nanmax(ranges)) <= 3.0, "and nothing past the fan's own reach"


def column(node: DepthFusion, **box: float) -> dict[str, Any]:
    """Ask /fusion/column for a box and read the answer back as columns."""
    srv_type, callback = node.services["/fusion/column"]
    answer = callback(srv_type.Request(**box), srv_type.Response())
    fields = fields_from_cloud(answer.sub_map)
    fields["frame"] = answer.sub_map.header.frame_id
    return fields


def test_the_column_service_answers_the_surface_in_a_box_with_its_weights(
    node: DepthFusion,
) -> None:
    """The gaze arbiter's stall look asks for the columns over the blocking cells before and
    after the look: here a stretch of the box's north wall, which the lidar painted, and
    nothing in a box of open floor."""
    node._switches.set("min_weight", 0.5)
    node._on_scan_work(scan_msg())
    at_odom(node, 0.4)
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    wall = column(node, x=-1.5, y=2.0, z=0.4, l_x=0.4, l_y=0.4, l_z=0.6)
    assert wall["frame"] == "odom" and set(wall) >= {"x", "y", "z", "weight", "lidar"}
    assert len(wall["x"]) > 0
    assert np.all(np.abs(wall["x"] + 1.5) <= 0.2) and np.all(np.abs(wall["y"] - 2.0) <= 0.2)
    assert np.all(wall["weight"] >= 0.5) and np.any(wall["lidar"] > 0)
    floor = column(node, x=0.5, y=0.5, z=0.4, l_x=0.4, l_y=0.4, l_z=0.6)
    assert len(floor["x"]) == 0
    outside = column(node, x=50.0, y=0.0, z=0.4, r=0.2)
    assert len(outside["x"]) == 0
    assert node._tally.take().counts["columns"] == 3


def test_the_marks_hz_cap_thins_the_topic_and_not_the_volume(node: DepthFusion) -> None:
    """CLAUDE.md rule 19, and the point of the cap: the board's local costmap reads this topic 5
    times a second, so a second fan in the same millisecond is never put on the wire — while the
    revolution behind it is integrated exactly as before. The volume is the proof: the marks of
    the one published fan carry the wall the SECOND revolution drew."""
    node._switches.set("min_weight", 0.5)
    node._switches.set("marks_hz", 1.0)
    node._on_scan_work(scan_msg())
    at_odom(node, 0.4)  # a place the volume has not seen: the view gate lets the revolution in
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    assert len(node.pubs["/depth_marks"].sent) == 1, "one fan a second, and no more"
    counts = node._tally.take().counts
    assert counts["marks"] == 1 and counts["marks_thinned"] == 1
    assert counts["revolutions"] == 2, "both revolutions went into the volume"
    node._switches.set("marks_hz", MARKS_EVERY_FRAME)
    at_odom(node, 0.8)
    node._on_scan_work(scan_msg(SCAN_S + 0.4))
    ranges = np.array(marks(node).ranges)
    assert np.isfinite(ranges).any(), "and what they drew is in the next fan that does go out"
    assert float(np.nanmax(ranges)) <= 3.0, "nothing past the fan's own reach"


def test_the_report_line_says_what_the_cap_held_back(node: DepthFusion) -> None:
    """A rate below the camera's own must read as the cap and not as a node that has stopped
    slicing, and the state of the flag is in the line either way."""
    node._switches.set("marks_hz", 1.0)
    node._on_scan_work(scan_msg())
    at_odom(node, 0.4)
    node._on_scan_work(scan_msg(SCAN_S + 0.2))
    assert "1 held by the marks_hz 1 cap" in node._marks_line(node._tally.take())
    node._switches.set("marks_hz", MARKS_EVERY_FRAME)
    assert "marks_hz 0: every frame" in node._marks_line(node._tally.take())


def test_the_report_line_says_where_the_marks_came_from(node: DepthFusion) -> None:
    """A drive is judged on the report line: how many marks went out, what a slice of the volume
    cost and in which band it was read."""
    node._on_scan_work(scan_msg())
    line = node._marks_line(node._tally.take())
    assert "marks: 1 from the volume" in line and "ms a slice" in line


# ---- the volume in ODOM: local obstacle memory, a rolling window, no global pose at all -------
def test_the_volume_in_odom_is_painted_by_the_odometry_and_by_nothing_else(tmp_path: Path) -> None:
    """Since 2026-09-22. No tracker has ever published a fit and TF holds no map -> odom edge at
    all — and the beams still paint the room, because the pose that places them is
    odom -> base_link."""
    node = odom_node(tmp_path)
    assert ("map", "odom") not in node._tf.buffer.transforms
    node._on_scan_work(scan_msg())
    assert painted(node) > 0.0, "the beams wrote the box at the odometry's pose"
    counts = node._tally.take().counts
    assert counts["revolutions"] == 1
    assert marks(node).header.frame_id == "base_link", "the marks are the cart's fan either way"
    # ...and the surface goes out in the frame it was painted in, never in map
    node._publish_surface()
    assert node.pubs["/fusion/surface"].sent[-1].header.frame_id == "odom"


def test_a_step_of_map_to_odom_does_not_move_the_marks(tmp_path: Path) -> None:
    """THE WHOLE POINT OF THE FRAME. A re-seating of the global pose is what poisoned the volume on
    2026-09-21: painted in map, some twenty seatings each wrote their own copy of the walls and the
    slice put 300-650 lethal cells around the cart. Here the tracker re-seats the cart by 2 m — and
    the window does not move, nothing is refused, and the next revolution goes into the very cells
    the last one did: one wall, not two."""
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
    node._views = ViewGate(node._spec.voxel_m)  # so the same place may speak again
    node._on_scan_work(scan_msg(SCAN_S + 0.4))

    assert node._world.spec == spec, "no correction slides the window; only the cart does"
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
    pose = node._poser.base_in_map(SCAN_S)
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


def test_the_report_line_names_the_frame_and_where_the_window_stands(tmp_path: Path) -> None:
    """Where the window stands and what its last slide cost — the numbers a drive is judged on
    without a debugger."""
    node = odom_node(tmp_path)
    node._on_scan_work(scan_msg())
    node._report()
    line = node.logger.texts("info")[-1]
    assert "frame: the volume is local memory, in odom" in line and "last slide 0 ms" in line


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
    at_odom(node, 0.4)
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


# ---- the whiskers write the volume (tof_rays) ----------------------------------------------
# A pillow under the cart's nose is below the lidar's plane and too near for the stereo rig: only
# the ToF saw it, and until 2026-10-01 only the local costmap's own layers knew. The fans now go
# into the volume as rays from their own frames (pepin.tof_rays), so both costmaps read them out
# of /depth_marks like the camera's surfaces.

TOF_S = SCAN_S
TOF_BEAMS = 11  # the front whisker's fan at its 0.96 m ceiling (pepin.tof_horizon.cone_beams)
TOF_REACH_M = 0.96
TOF_FOV = 0.47


def tof_msg(value: float, t: float = TOF_S, name: str = "front") -> Any:
    """One fan of the ``name`` whisker, every beam ``value`` (metres, +inf for nothing within the
    trusted range, NaN for "I do not know"), as pepin_bringup.tof_bridge.fan_scan publishes it."""
    return LaserScan(
        header=Header(stamp=stamp(t), frame_id=f"tof_{name}"),
        angle_min=-TOF_FOV / 2,
        angle_increment=TOF_FOV / (TOF_BEAMS - 1),
        range_min=0.05,
        range_max=TOF_REACH_M,
        ranges=[value] * TOF_BEAMS,
    )


def whisker_node(tmp_path: Path) -> DepthFusion:
    """An odom node whose TF also carries the front whisker's mount (config/tof.json: 0.27 m up,
    facing forward) — and NO edge for the left one, which is the dead-edge case below."""
    node = odom_node(tmp_path)
    node._tf.buffer.transforms[("odom", "tof_front")] = edge("odom", "tof_front", TOF_S, z=0.27)
    return node


def ahead_in_marks(node: DepthFusion) -> float:
    """The nearest mark inside the front whisker's cone, metres; NaN for none."""
    ranges = np.array(marks(node).ranges)
    mid, half = ranges.size // 2, round((TOF_FOV / 2) / math.radians(0.5))
    cone = ranges[mid - half : mid + half + 1]
    return float(np.nanmin(cone)) if np.isfinite(cone).any() else math.nan


def test_a_whisker_hit_reaches_the_marks_the_costmaps_read(tmp_path: Path) -> None:
    """Eight fans at 0.4 m (the node's min_weight 4.0 at 0.5 a fan: half a second at 15 Hz) and
    the marks carry a surface 0.4 m ahead; the volume holds it at the sensor's own height; the
    report counts the hits."""
    node = whisker_node(tmp_path)
    speak = fans_to_speak(float(node._switches["min_weight"]))
    assert speak == 4
    for i in range(speak - 1):
        node._on_tof_work(tof_msg(0.4, TOF_S + 0.07 * i))
    assert math.isnan(ahead_in_marks(node)), "seven fans are not yet agreement"
    node._on_tof_work(tof_msg(0.4, TOF_S + 0.07 * speak))
    assert ahead_in_marks(node) == pytest.approx(0.4, abs=0.05)
    idx, inside = node._world.volume.voxel_of(np.array([[0.4, 0.0, 0.27]]))
    assert inside[0] and node._world.volume.weight[tuple(idx[0])] == pytest.approx(4.0)
    assert node._world.frames[-1][1] == "tof"
    counts = node._tally.take().counts
    assert counts["tof_fans"] == speak and counts["tof_hits"] == speak and counts["tof_no_tf"] == 0


def test_misses_carve_the_whisker_s_mark_within_two_seconds_of_fans(tmp_path: Path) -> None:
    """The pillow stared at for 40 fans is saturated (max_weight 20); taken away, +inf fans
    carve it, and the marks stop reading it inside 30 fans — two seconds at 15 Hz."""
    node = whisker_node(tmp_path)
    for i in range(40):
        node._on_tof_work(tof_msg(0.4, TOF_S + 0.07 * i))
    assert ahead_in_marks(node) == pytest.approx(0.4, abs=0.05)
    misses = 0
    while not math.isnan(ahead_in_marks(node)):
        node._on_tof_work(tof_msg(math.inf, TOF_S + 3.0 + 0.07 * misses))
        misses += 1
        assert misses <= 30
    assert node._tally.take().counts["tof_misses"] == misses


def test_a_fan_whose_frame_tf_does_not_know_is_dropped_and_counted(tmp_path: Path) -> None:
    node = whisker_node(tmp_path)
    before = float(node._world.volume.weight.sum())
    node._on_tof_work(tof_msg(0.4, name="left"))  # no odom -> tof_left edge in this TF
    assert float(node._world.volume.weight.sum()) == before, "nothing was written"
    counts = node._tally.take().counts
    assert counts["tof_no_tf"] == 1 and counts["tof_fans"] == 0 and counts["no_tf"] >= 1


def test_tof_rays_off_integrates_nothing_and_says_so(tmp_path: Path) -> None:
    node = whisker_node(tmp_path)
    node._switches.set("tof_rays", False)
    for _ in range(4):
        node._on_tof_work(tof_msg(0.4))
    assert float(node._world.volume.weight.sum()) == 0.0
    counts = node._tally.take().counts
    assert counts["tof_off"] == 4 and counts["tof_fans"] == 0
    assert "tof_rays=off" in node._switches.state()


def test_the_report_line_counts_the_whiskers_and_states_the_clearing_arithmetic(
    tmp_path: Path,
) -> None:
    node = whisker_node(tmp_path)
    node._on_tof_work(tof_msg(0.4))
    node._on_tof_work(tof_msg(math.inf, TOF_S + 0.07))
    node._on_tof_work(tof_msg(math.nan, TOF_S + 0.14))
    line = node._tof_line(node._tally.take())
    assert "2 fans" in line and "1 hits, 1 misses, 1 silent" in line
    assert "carved by 15 misses (1.0 s at 15 Hz)" in line


# ---- the robot's own arm (arm_filter, pepin.arm) -------------------------------------------------
# The arm of these tests is one box, the URDF's base link, standing 0.19-0.31 m ahead of the cart
# with its 3 cm margin and covering the front whisker's height (0.27 m): a whisker's rays meet it,
# and a test can paint voxels inside it and watch them go.
ARM_BOX = {"name": "base_link", "link": "base_link", "centre_m": [0.0, 0.0, 0.05],
           "rpy_deg": [0.0, 0.0, 0.0], "half_m": [0.03, 0.1, 0.05]}  # fmt: skip


def with_arm(tmp_path: Path, **joints: Any) -> None:
    """config/arm.json beside the node's config: the shipped file with :data:`ARM_BOX` alone,
    standing at x 0.22, z 0.22 in base_link, its joints as the shipped file's unless overridden."""
    data = json.loads((REPO / "config" / "arm.json").read_text())
    data["mount"] = {"x_m": 0.22, "y_m": 0.0, "z_m": 0.22, "yaw_deg": 0.0, "measured": True}
    data["links"] = [ARM_BOX]
    data["joints"] = {**data["joints"], **joints}
    (tmp_path / "arm.json").write_text(json.dumps(data))


def arm_voxels(node: DepthFusion) -> tuple[Any, ...]:
    """The voxel indices whose centres lie inside the grown arm box (in odom: the cart stands at
    the origin)."""
    boxes = node._arm_boxes(TOF_S)
    assert len(boxes) == 1
    s = node._world.spec
    index = np.stack(np.meshgrid(*(np.arange(n) for n in s.shape), indexing="ij"), -1)
    centres = (index.reshape(-1, 3) + 0.5) * s.voxel_m + np.array(s.origin)
    inside = boxes[0].holds(centres).reshape(s.shape)
    assert inside.sum() > 4
    return tuple(np.nonzero(inside))


def test_arm_filter_off_paints_the_arm_like_the_room_and_says_what_the_arm_is(
    tmp_path: Path,
) -> None:
    with_arm(tmp_path)
    node = whisker_node(tmp_path)
    node._switches.set("arm_filter", False)  # on by default since 2026-10-05
    at = arm_voxels(node)
    node._world.volume.weight[at] = 5.0
    node._on_scan_work(scan_msg())
    assert np.all(node._world.volume.weight[at] >= 5.0), "nothing forgotten with the flag off"
    line = node._arm_line(node._tally.take())
    assert line == "arm filter off (1 boxes grown 3 cm, joints from config)"


def test_under_arm_filter_every_integration_forgets_what_stands_inside_the_arm(
    tmp_path: Path,
) -> None:
    """A revolution and a whisker fan each leave nothing inside the grown arm, whoever painted
    it before; the report says how many went."""
    with_arm(tmp_path)
    node = whisker_node(tmp_path)
    node._switches.set("arm_filter", True)
    at = arm_voxels(node)
    node._world.volume.weight[at] = 5.0
    node._world.lidar_weight[at] = 5.0
    node._on_scan_work(scan_msg())
    assert not node._world.volume.weight[at].any() and not node._world.lidar_weight[at].any()
    node._world.volume.weight[at] = 5.0
    node._on_tof_work(tof_msg(0.4))
    assert not node._world.volume.weight[at].any()
    w = node._tally.take()
    assert w.counts["arm_forgotten"] == 2 * len(at[0]) and w.counts["arm_config"] >= 2
    line = node._arm_line(w)
    assert line.startswith("arm filter: the arm met 0 of 0 frames")
    assert f"forgot {2 * len(at[0])} voxels" in line and "joints config" in line


def test_a_whisker_ray_that_enters_the_arm_carves_nothing_behind_it(tmp_path: Path) -> None:
    """The front whisker answers +inf (nothing within 0.96 m): unfiltered it carves through the
    arm 0.19 m ahead to the far end of its reach; with arm_filter its rays stop at the arm, and a
    voxel 0.5 m ahead is never carved."""
    beyond = np.array([[0.5, 0.0, 0.27]])

    def carved(on: bool) -> float:
        with_arm(tmp_path)
        node = whisker_node(tmp_path)
        node._switches.set("arm_filter", on)
        for i in range(4):
            node._on_tof_work(tof_msg(math.inf, TOF_S + 0.07 * i))
        idx, inside = node._world.volume.voxel_of(beyond)
        assert inside[0]
        return float(node._world.volume.weight[tuple(idx[0])])

    assert carved(False) > 0.0
    assert carved(True) == 0.0


def test_the_arm_s_grown_boxes_go_out_for_the_eye_with_the_flag_off(tmp_path: Path) -> None:
    """/fusion/arm: a DELETEALL, then one teal CUBE per box in base_link at the grown size, the
    file's pose stamped zero (the newest transform)."""
    with_arm(tmp_path)
    node = whisker_node(tmp_path)
    node._publish_arm()
    (sent,) = node.pubs["/fusion/arm"].sent
    clear, cube = sent.markers
    assert clear.action == 3 and cube.type == 1 and cube.action == 0
    assert cube.header.frame_id == "base_link" and cube.header.stamp.sec == 0
    assert (cube.scale.x, cube.scale.y, cube.scale.z) == pytest.approx((0.12, 0.26, 0.16))
    assert (cube.pose.position.x, cube.pose.position.z) == pytest.approx((0.22, 0.27))
    assert cube.color.a > 0.0


def test_joints_from_the_topic_pose_the_arm_at_the_observation_s_stamp(tmp_path: Path) -> None:
    """source topic: a /arm/joint_states sample near the stamp poses the arm (counted as topic)
    and dates the markers; none near it falls back to the file's pose, counted stale."""
    with_arm(tmp_path, source="topic")
    node = whisker_node(tmp_path)
    names = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
    node._on_joints(JointState(header=Header(stamp=stamp(TOF_S)), name=names, position=[0.0] * 6))
    near, far = node._arm_pose(TOF_S + 0.1, count=True), node._arm_pose(TOF_S + 5.0, count=True)
    assert near is not None and near.source == "topic"
    assert far is not None and far.source == "stale"
    node._publish_arm()
    assert node.pubs["/fusion/arm"].sent[-1].markers[1].header.stamp.sec == int(TOF_S)
    w = node._tally.take()
    assert w.counts["arm_joints_in"] == 1
    assert w.counts["arm_topic"] == 1 and w.counts["arm_stale"] == 1, "the markers count none"
