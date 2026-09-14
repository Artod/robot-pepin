"""pepin.worldmap: the volume as the map — the lidar's layer, the slices, the snapshot.

A synthetic room (a box the beams are cast against analytically) is enough to check every
rule: the walls land where they are, the inside is carved free, the outside stays unknown, the
camera cannot repaint the lidar's layer, and the slice comes back out as the three things the
stack reads it as (a ROS message, the tracker's log-odds grid, a map_server pair).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from pepin.depth import Intrinsics
from pepin.lean import Lean
from pepin.mapping import grid_from_pgm
from pepin.tsdf import GridSpec, RigidPose
from pepin.worldmap import (
    FREE,
    OCCUPIED,
    UNKNOWN,
    LidarLaw,
    OccupancySlice,
    PlanarMount,
    SliceLaw,
    SnapshotClock,
    WorldMap,
    bearings_in_base,
    trinary_from_log_odds,
)

ROOM_M = 2.0  # the box: walls at x, y = +-2 m
PLANE_M = 0.383  # the lidar's height, as config/lidar.json has it


def spec() -> GridSpec:
    """A 6 x 6 x 1.7 m grid at 5 cm: the synthetic room with a metre of margin."""
    return GridSpec(origin=(-3.0, -3.0, -0.15), shape=(120, 120, 34), camera_band_m=(0.15, 1.30))


def mount() -> PlanarMount:
    return PlanarMount(z_m=PLANE_M, min_range_m=0.05, max_range_m=12.0)


def box_scan(x: float = 0.0, y: float = 0.0, beams: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """Bearings and ranges of a lidar standing at (x, y) inside the box: the exact distance
    from the sensor to the wall along every beam."""
    angles = np.linspace(-math.pi, math.pi, beams, endpoint=False)
    cx, cy = np.cos(angles), np.sin(angles)
    with np.errstate(divide="ignore"):
        tx = np.where(cx > 0, (ROOM_M - x) / cx, (-ROOM_M - x) / cx)
        ty = np.where(cy > 0, (ROOM_M - y) / cy, (-ROOM_M - y) / cy)
    return angles, np.minimum(np.abs(tx), np.abs(ty))


def at(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> RigidPose:
    """The cart standing at (x, y) with heading ``yaw``, on a floor at zero."""
    c, s = math.cos(yaw), math.sin(yaw)
    return RigidPose(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([x, y, 0.0]))


def room(scans: int = 8) -> WorldMap:
    """A world map with the synthetic box scanned a few times from the middle: enough scans
    for the free space to reach the default ``min_weight`` (a crossing weighs less than a
    return, so open floor takes about a second of scanning to be believed)."""
    world = WorldMap(spec(), mount())
    for i in range(scans):
        angles, ranges = box_scan()
        world.integrate_scan(angles, ranges, at(), stamp=100.0 + i)
    return world


def cell_of(view: OccupancySlice, x: float, y: float) -> int:
    """The slice's value at a map point."""
    return int(view.values[rc(view, x, y)])


def rc(view: OccupancySlice, x: float, y: float) -> tuple[int, int]:
    """The (row, column) of a map point in the slice."""
    col = int((x - view.origin[0]) / view.resolution_m)
    row = int((y - view.origin[1]) / view.resolution_m)
    return row, col


def wall_at(view: OccupancySlice, x: float, y: float) -> bool:
    """Whether the slice marks a surface within one voxel of a map point. A wall lands in the
    cell that holds the zero crossing, which is one side or the other of a cell boundary: "the
    wall is here, within a voxel" is the claim the map actually makes."""
    row, col = rc(view, x, y)
    return bool((view.values[row - 1 : row + 2, col - 1 : col + 2] == OCCUPIED).any())


# ---- ray carving ---------------------------------------------------------------------------
def test_the_lidar_slice_reproduces_the_box_within_one_voxel() -> None:
    view = room().lidar_slice()
    rows, cols = np.nonzero(view.values == OCCUPIED)
    xs = view.origin[0] + (cols + 0.5) * view.resolution_m
    ys = view.origin[1] + (rows + 0.5) * view.resolution_m
    assert xs.size > 200, "the walls of a 4 m box at 5 cm are hundreds of cells"
    off_wall = np.minimum(np.abs(np.abs(xs) - ROOM_M), np.abs(np.abs(ys) - ROOM_M))
    assert off_wall.max() <= 0.05, "every occupied cell sits on a wall, within one voxel"


def test_the_inside_is_free_and_the_outside_unknown() -> None:
    view = room().lidar_slice()
    for x, y in ((0.0, 0.0), (1.5, 0.0), (0.0, -1.5), (-1.0, 1.0)):
        assert cell_of(view, x, y) == FREE, f"({x}, {y}) is open floor"
    for x, y in ((2.5, 0.0), (0.0, -2.6), (2.8, 2.8)):
        assert cell_of(view, x, y) == UNKNOWN, f"({x}, {y}) is behind a wall: never observed"


def test_a_wall_on_a_cell_boundary_survives_every_viewpoint() -> None:
    """The walls of this box fall exactly on cell boundaries, which is the case ray sampling
    alone loses: no rung of the ladder need land within half a voxel of the wall, both
    neighbouring cells then read as "no surface here", and every further viewpoint averages
    the wall away harder (18 % of it left after nine, before the return was sampled).
    """
    world = WorldMap(spec(), mount())
    views = [(0.0, 0.0), (1.0, 1.0), (-1.0, 0.5), (0.5, -1.2)]
    for i, (x, y) in enumerate(views * 4):
        world.integrate_scan(*box_scan(x, y), at(x, y), stamp=100.0 + i)
    view = world.lidar_slice()
    strip = [wall_at(view, ROOM_M, y) for y in np.arange(-1.5, 1.51, 0.05)]
    assert np.mean(strip) > 0.95, "the wall on the boundary is in the map, cell by cell"
    rows, cols = np.nonzero(view.values == OCCUPIED)
    xs = view.origin[0] + (cols + 0.5) * view.resolution_m
    ys = view.origin[1] + (rows + 0.5) * view.resolution_m
    off_wall = np.minimum(np.abs(np.abs(xs) - ROOM_M), np.abs(np.abs(ys) - ROOM_M))
    assert off_wall.max() <= 0.05, "and it did not grow inward while doing it"


def test_a_beam_past_the_sensor_reach_carves_free_space_and_marks_nothing() -> None:
    """A range beyond the mount's reach: the beam says "nothing out to here", not "a wall
    here", because the reach is how far this sensor may be believed."""
    world = WorldMap(spec(), PlanarMount(z_m=PLANE_M, max_range_m=1.0))
    angles, ranges = box_scan()
    for _ in range(8):
        world.integrate_scan(angles, ranges, at())
    view = world.lidar_slice()
    assert cell_of(view, 0.5, 0.0) == FREE
    assert cell_of(view, 1.99, 0.0) == UNKNOWN, "the wall is out of reach: not marked"
    assert not np.any(view.values == OCCUPIED)


def test_a_beam_with_no_return_writes_nothing_until_the_flag_opens_the_door() -> None:
    """What the lidar really delivers for an open door is NaN, not a long range: everything
    past ``range_max`` comes back as no return at all. Off by default it writes nothing (a
    mirror and a black chair leg say NaN too); on, it carves free space to the reach."""
    angles, ranges = box_scan()
    doorway = np.abs(angles) < 0.2  # a wedge of beams that came back with nothing
    open_door = np.where(doorway, np.nan, ranges)

    shut = WorldMap(spec(), mount())
    for _ in range(20):
        shut.integrate_scan(angles, open_door, at())
    view = shut.lidar_slice()
    assert cell_of(view, 1.0, 0.0) == UNKNOWN, "no return, no claim: nothing is written"
    assert cell_of(view, 0.0, 1.0) == FREE, "the beams that did return still carve"

    carving = WorldMap(spec(), mount(), LidarLaw(no_return_free=True))
    for _ in range(20):
        carving.integrate_scan(angles, open_door, at())
    through = carving.lidar_slice()
    assert cell_of(through, 1.0, 0.0) == FREE, "the door is open all the way out"
    assert not wall_at(through, ROOM_M, 0.0), "and nothing is marked at the end of the beam"
    assert wall_at(through, 0.0, ROOM_M), "the walls the other beams found are still there"


def test_a_scan_that_misses_the_volume_changes_nothing() -> None:
    world = WorldMap(spec(), mount())
    angles, ranges = box_scan()
    assert world.integrate_scan(angles, ranges, at(x=100.0, y=100.0)) == 0
    assert world.integrate_scan(angles, np.full_like(ranges, np.nan), at()) == 0
    assert world.maturity()["voxels"] == 0.0


def test_the_layer_is_at_the_lidars_plane_and_one_voxel_thick() -> None:
    world = room()
    written = np.flatnonzero(world.lidar_weight.any(axis=(0, 1)))
    lo = world.spec.origin[2] + written[0] * world.spec.voxel_m
    hi = world.spec.origin[2] + (written[-1] + 1) * world.spec.voxel_m
    assert lo <= PLANE_M <= hi
    assert hi - lo <= 3 * world.spec.voxel_m, "the plane plus or minus one voxel, no more"
    assert world.slice(1.0, 1.2).counts()["known"] == 0, "the lidar wrote nothing above itself"


# ---- the body's lean -------------------------------------------------------------------------
TABLE_Z_M = 0.7  # the tabletop the sensor's plane never sees while the cart stands level
TABLE_X_M = (2.0, 4.0)  # how far it reaches in front of the cart
TABLE_HALF_Y_M = 0.5
WALL_X_M = 5.0  # the wall behind it
TIP_DEG = 6.0
# The three channels after LEVEL_POSES scanned the box eight times, as the implementation that
# wrote flat planes left them (scratch/lidar_level_is_bit_identical.py runs the two
# implementations side by side out of git: 0 voxels differ). A level run's map is this, to the
# bit, for ever — the digest only ever moves when the level arithmetic itself does, and then
# that probe says so. It moved once, at ce22d98, where the return started being sampled where
# it came back instead of only on the ray's ladder.
LEVEL_DIGEST = "7e40351d9305a2820aebba7604fe95429e175c005bf38666fe30b974ead5cbe0"
LEVEL_POSES = ((0.0, 0.0, 0.0), (0.5, -0.3, 0.7), (-0.8, 0.9, -2.1), (0.2, 0.2, math.pi))


def tipped(pitch_deg: float, x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> RigidPose:
    """The cart at (x, y, yaw) with its nose down by ``pitch_deg`` (negative: nose up), the lean
    composed onto the planar pose exactly as :class:`pepin.frame_pose.FramePoser` composes it."""
    planar = at(x, y, yaw)
    lean = Lean(0.0, math.radians(pitch_deg), 0.0)
    return RigidPose(planar.rotation @ lean.rotation(), planar.translation)


def tilt_spec() -> GridSpec:
    """A grid reaching 7 m ahead: the wall of the tipping room is at 5 m, out of the box's."""
    return GridSpec(origin=(-1.0, -3.0, -0.15), shape=(160, 120, 34), camera_band_m=(0.15, 1.30))


def fan_scan(
    pose: RigidPose, sensor: PlanarMount, beams: int = 121, half_fov_deg: float = 30.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What a lidar on ``pose`` really measures in a room with a tabletop at 0.7 m (2 to 4 m
    ahead, half a metre either side) and a wall at 5 m behind it: bearings, ranges, and the
    true (n, 3) map points the returns came from.

    The beams are cast as the 3D rays they are, so a level body's fan stays at the sensor's
    plane and sees only the wall, while a tipped body's fan climbs into the tabletop.
    """
    b = np.radians(np.linspace(-half_fov_deg, half_fov_deg, beams))
    m = np.asarray(pose.rotation, dtype=float)
    origin = np.asarray(pose.translation, dtype=float) + m @ np.array(
        [sensor.x_m, sensor.y_m, sensor.z_m]
    )
    d = m @ np.stack([np.cos(b), np.sin(b), np.zeros_like(b)])
    with np.errstate(divide="ignore", invalid="ignore"):
        to_wall = np.where(d[0] > 0.0, (WALL_X_M - origin[0]) / d[0], np.inf)
        to_table = (TABLE_Z_M - origin[2]) / d[2]
    hit = origin[:, None] + d * to_table
    on_table = (
        (to_table > 0.0)
        & (TABLE_X_M[0] <= hit[0])
        & (hit[0] <= TABLE_X_M[1])
        & (np.abs(hit[1]) <= TABLE_HALF_Y_M)
    )
    ranges = np.minimum(to_wall, np.where(on_table, to_table, np.inf))
    return b, ranges, (origin[:, None] + d * ranges).T


def surface_at(world: WorldMap, point: np.ndarray) -> bool:
    """Whether the lidar marked a surface within one voxel of a map point in the volume."""
    s = world.spec
    at_ = [int((float(point[i]) - s.origin[i]) / s.voxel_m) for i in range(3)]
    box = tuple(slice(max(i - 1, 0), i + 2) for i in at_)
    near = np.abs(world.volume.sdf[box]) <= 0.5 * s.voxel_m / s.truncation_m + 1e-5
    return bool((near & (world.lidar_weight[box] > 0.0)).any())


def channels(world: WorldMap) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The three channels a scan writes, for comparing two maps element for element."""
    return world.volume.sdf, world.volume.weight, world.lidar_weight


def scanned_box(pose_of: Callable[[float, float, float], RigidPose]) -> WorldMap:
    """The box scanned eight times from LEVEL_POSES, each pose built by ``pose_of``."""
    world = WorldMap(spec(), mount())
    for i in range(8):
        x, y, yaw = LEVEL_POSES[i % len(LEVEL_POSES)]
        angles, ranges = box_scan(x, y)
        world.integrate_scan(angles, ranges, pose_of(x, y, yaw), stamp=100.0 + i)
    return world


def test_a_level_body_writes_the_voxels_it_has_always_written() -> None:
    """CLAUDE.md rule 19's old behaviour, to the bit. A pure-yaw pose is what the planar EKF
    gives and what ``imu_lean`` off leaves, and the map it writes must be the map that was
    written before the beams became rays — not "within a voxel": a night's map has to be
    comparable with last night's, and a difference nobody can name is a bug nobody can find."""
    digest = hashlib.sha256()
    for channel in channels(scanned_box(at)):
        digest.update(np.ascontiguousarray(channel).tobytes())
    assert digest.hexdigest() == LEVEL_DIGEST


def test_a_lean_of_zero_is_the_level_map_bit_for_bit() -> None:
    """And the switch's on state with nothing to correct is the same map again: a Lean of zero
    composed onto a planar pose leaves a pure yaw, which is the plane the beams always swept."""
    for level, leaned in zip(
        channels(scanned_box(at)),
        channels(scanned_box(lambda x, y, yaw: tipped(0.0, x, y, yaw))),
        strict=True,
    ):
        np.testing.assert_array_equal(level, leaned)


def test_a_tipped_body_writes_its_beams_where_they_really_went() -> None:
    """A rear wheel climbs a threshold, the nose lifts 6 degrees for a second, and the whole fan
    climbs with it — by tan(6 deg), 10.5 cm per metre out, whatever the bearing. At 3 m that is
    the underside of a tabletop. Placed level, those returns are a wall 3 m ahead in the one
    layer the cart drives by, where the room is open floor; placed along the rays they really
    are, they are a tabletop at its own height and the plane says nothing at all."""
    pose = tipped(-TIP_DEG)
    angles, ranges, points = fan_scan(pose, mount())
    leaning, level = WorldMap(tilt_spec(), mount()), WorldMap(tilt_spec(), mount())
    for i in range(8):
        leaning.integrate_scan(angles, ranges, pose, stamp=100.0 + i)
        level.integrate_scan(angles, ranges, at(), stamp=100.0 + i)
    ahead = points[points.shape[0] // 2]
    assert abs(ahead[2] - TABLE_Z_M) < 1e-9, "the beam straight ahead ends on the tabletop"
    assert 2.9 < float(ranges[ranges.size // 2]) < 3.2, "3 m out, where the tabletop starts"
    assert surface_at(leaning, ahead), "the tabletop is written at the height it was seen at"
    on_table = np.abs(points[:, 2] - TABLE_Z_M) < 1e-9
    assert on_table.sum() >= 15, "a score of beams climb into the tabletop"
    assert all(surface_at(leaning, p) for p in points), "every return, at its own height"
    # the wall behind it: the beams that pass beside the tabletop still reach it, and it is
    # still a wall — at the height 6 degrees of nose put the beam, 45 cm above the plane
    wall = points[~on_table][-1]
    assert wall[0] == pytest.approx(WALL_X_M, abs=1e-9) and wall[2] > PLANE_M + 0.4
    assert surface_at(leaning, wall)
    # and what the level assumption makes of the same returns: a wall across the open floor
    flat = np.array([float(ranges[ranges.size // 2]), 0.0, PLANE_M])
    assert surface_at(level, flat), "placed level, the tabletop is a false wall 3 m ahead"
    assert not surface_at(leaning, flat), "placed along the ray, that floor stays open"
    assert not surface_at(leaning, np.array([WALL_X_M, 0.0, PLANE_M])), "nor is the wall there"


def test_a_tip_does_not_widen_the_band_the_camera_hands_back() -> None:
    """The beams of a tipped body climb into the camera's band and are written there — but the
    layer the camera hands back stays the plane's own rows, whatever the body did.

    Ownership never expires (``lidar_weight`` does not decay), so a band that grew with the rays
    would let one second over a slipper take half the camera's band away for the rest of the
    run: at the scan gate's 3 degrees an 8 m ray is 40 cm off the plane, and every voxel it
    crossed on the way would be camera-proof for good."""
    pose = tipped(-TIP_DEG)
    angles, ranges, points = fan_scan(pose, mount())
    world, level = WorldMap(tilt_spec(), mount()), WorldMap(tilt_spec(), mount())
    world.integrate_scan(angles, ranges, pose, stamp=100.0)
    level.integrate_scan(angles, ranges, at(), stamp=100.0)
    written = np.flatnonzero(world.lidar_weight.any(axis=(0, 1)))
    top = world.spec.origin[2] + (written[-1] + 1) * world.spec.voxel_m
    assert top >= points[:, 2].max(), "the climbing beams are written at their own height"
    rows = level.protected_rows
    assert rows is not None
    assert world.protected_rows == rows, "the handback band is the plane's layer, not the rays'"
    assert written[-1] + 1 > rows[1], "and the rays did reach well above it"


def test_a_beam_that_climbs_out_of_the_volume_is_carved_as_far_as_it_reaches() -> None:
    """A steep tip aims the fan over the volume's ceiling: what is inside is still carved, and
    the part above it is simply not written — no wrapping, no clipping to the top row."""
    world = WorldMap(tilt_spec(), mount())
    pose = tipped(-30.0)
    angles = np.array([0.0])
    ranges = np.array([6.0])  # 3 m up at its end, over the 1.55 m ceiling
    assert world.integrate_scan(angles, ranges, pose, stamp=100.0) > 0
    s = world.spec
    tilt = math.radians(30.0)
    ox, oz = -math.sin(tilt) * PLANE_M, math.cos(tilt) * PLANE_M  # the mount, tipped with it
    ceiling = s.origin[2] + s.shape[2] * s.voxel_m
    leaves_at = ox + math.cos(tilt) * (ceiling - oz) / math.sin(tilt)
    columns = np.flatnonzero(world.lidar_weight.any(axis=(1, 2)))
    far = s.origin[0] + (columns[-1] + 1) * s.voxel_m
    assert far == pytest.approx(leaves_at, abs=0.15), "carved to where the ray leaves the ceiling"
    assert far < ox + math.cos(tilt) * 3.0, "far short of the beam's own end at 6 m"


def test_a_wall_that_moved_away_is_cleared_by_the_beams_that_cross_it() -> None:
    """Maturity is the weight in a cell, not a flag: the lidar's cap is what lets the map
    change its mind in seconds instead of never."""
    world = WorldMap(spec(), mount(), LidarLaw(max_weight=6.0))
    angles, ranges = box_scan()
    near = np.minimum(ranges, 1.0)  # a screen a metre out, in front of every wall
    for _ in range(8):
        world.integrate_scan(angles, near, at())
    assert wall_at(world.lidar_slice(), 1.0, 0.0)
    for _ in range(20):
        world.integrate_scan(angles, ranges, at())  # the screen is gone
    view = world.lidar_slice()
    assert cell_of(view, 1.0, 0.0) == FREE
    assert not wall_at(view, 1.0, 0.0), "the screen left no ghost behind"
    assert wall_at(view, 2.0, 0.0), "the wall behind it is there instead"


# ---- the two sensors in one volume ---------------------------------------------------------
def flat_depth(intr: Intrinsics, metres: float) -> np.ndarray:
    return np.full((intr.height, intr.width), metres, dtype=np.float64)


def looking_ahead(z: float) -> tuple[Intrinsics, RigidPose]:
    """A camera at height ``z`` on the cart, looking along +x (optical z forward, y down)."""
    intr = Intrinsics(fx=60.0, fy=60.0, cx=32.0, cy=24.0, width=64, height=48)
    rotation = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
    return intr, RigidPose(rotation, np.array([0.0, 0.0, z]))


def test_the_camera_does_not_repaint_the_lidars_layer() -> None:
    """The network's depth is scale-uncertain: a frame that puts the wall a metre too near
    must not move the layer the cart drives by."""
    world = room()
    before = world.lidar_slice()
    intr, pose = looking_ahead(PLANE_M)
    world.integrate_depth(flat_depth(intr, 1.0), None, intr, pose, stamp=200.0)
    after = world.lidar_slice()
    assert np.array_equal(before.values, after.values)
    assert cell_of(after, 1.0, 0.0) == FREE, "the lie the camera told is not in the layer"


def test_the_camera_writes_its_own_band_and_the_flag_can_let_it_in() -> None:
    world = room()
    intr, pose = looking_ahead(0.8)
    assert world.integrate_depth(flat_depth(intr, 1.0), None, intr, pose, stamp=200.0) > 0
    assert wall_at(world.camera_band_slice(), 1.0, 0.0), "a tabletop the lidar's plane misses"

    open_world = room()
    open_world.protect_lidar_layer = False
    intr, pose = looking_ahead(PLANE_M)
    for _ in range(8):
        open_world.integrate_depth(flat_depth(intr, 1.0), None, intr, pose)
    assert wall_at(open_world.lidar_slice(), 1.0, 0.0), "unprotected: the camera wins"


def test_the_matchers_band_holds_only_cells_many_frames_agree_on() -> None:
    """The camera's reference is cut with its own, far higher threshold: a tabletop two frames
    painted is a picture, not evidence about the pose those two frames were placed at. Painting
    is untouched — the cell is in the volume from the first frame, and simply does not appear in
    the slice until it is heavy."""
    world = room()
    intr, pose = looking_ahead(0.8)
    for _ in range(2):  # two frames at 1 m: weight 4 each, the cap
        world.integrate_depth(flat_depth(intr, 1.0), None, intr, pose, stamp=200.0)
    assert wall_at(world.camera_band_slice(SliceLaw(min_weight=2.0)), 1.0, 0.0)
    assert not wall_at(world.camera_band_slice(SliceLaw(min_weight=20.0)), 1.0, 0.0)
    for _ in range(4):  # six frames in all: 24, over the threshold
        world.integrate_depth(flat_depth(intr, 1.0), None, intr, pose, stamp=200.0)
    assert wall_at(world.camera_band_slice(SliceLaw(min_weight=20.0)), 1.0, 0.0)


def test_the_band_says_how_hard_it_is_and_the_report_line_carries_it() -> None:
    """The share is what tells an operator whether the matcher has a reference at all: at the
    threshold the map is cut with it is 1, and above everything in the volume it is 0."""
    world = room()
    intr, pose = looking_ahead(0.8)
    for _ in range(3):
        world.integrate_depth(flat_depth(intr, 1.0), None, intr, pose, stamp=200.0)
    soft = SliceLaw()
    assert world.hardness(soft)["share"] == 1.0, "the cut and the floor are the same cut"
    empty = world.hardness(SliceLaw(min_weight=1e6))
    assert empty["occupied"] == 0.0 and empty["share"] == 0.0
    assert empty["occupied_floor"] > 0.0 and empty["min_weight"] == 1e6
    line = world.report(camera_law=SliceLaw(min_weight=20.0))
    assert "hard" in line and "above weight 20" in line


def test_the_camera_may_fill_the_layer_where_the_lidar_never_spoke() -> None:
    """Protection is per cell, not per height: the layer is the lidar's only where it looked."""
    world = room()  # nothing behind the wall was ever seen by the lidar
    intr, pose = looking_ahead(PLANE_M)
    far = RigidPose(pose.rotation, np.array([2.4, 0.0, PLANE_M]))
    world.integrate_depth(flat_depth(intr, 0.4), None, intr, far)
    assert wall_at(world.lidar_slice(), 2.8, 0.0)


def test_what_a_tipped_beam_claimed_above_the_plane_is_the_cameras_again() -> None:
    """A tip writes returns in the camera's own band — a tabletop at 0.7 m — and those voxels
    must stay the camera's to correct: ownership never expires, so a band that took them would
    keep the tabletop of one bad second until the next reset. The plane's layer is defended as
    it always was, and the band above it is not."""
    pose = tipped(-TIP_DEG)
    angles, ranges, points = fan_scan(pose, mount())
    world = WorldMap(tilt_spec(), mount())
    world.integrate_scan(angles, ranges, pose, stamp=100.0)
    ahead = points[points.shape[0] // 2]  # the beam straight ahead, ending on the tabletop
    assert abs(ahead[2] - TABLE_Z_M) < 1e-9
    s = world.spec
    ix, iy, iz = (int((float(ahead[i]) - s.origin[i]) / s.voxel_m) for i in range(3))
    assert world.lidar_weight[ix, iy, iz] > 0.0, "the climbing beam marked the tabletop here"
    assert abs(float(world.volume.sdf[ix, iy, iz])) < 0.5, "as a surface, not as free space"
    rows = world.protected_rows
    assert rows is not None
    owned = world.lidar_weight[:, :, rows[0] : rows[1]] > 0.0
    layer = world.volume.sdf[:, :, rows[0] : rows[1]][owned].copy()
    intr, camera = looking_ahead(TABLE_Z_M)
    for i in range(8):
        world.integrate_depth(flat_depth(intr, 3.5), None, intr, camera, stamp=200.0 + i)
    # eight frames of open air at 3 m outweigh the one claim (0.44 each against the lidar's 1.0),
    # so the voxel walks from the surface it was to the free space the camera sees
    assert float(world.volume.sdf[ix, iy, iz]) > 0.7, "the camera carved its own band free again"
    np.testing.assert_array_equal(
        world.volume.sdf[:, :, rows[0] : rows[1]][owned], layer, "the plane's layer is untouched"
    )


# ---- the slice -----------------------------------------------------------------------------
def test_the_slice_thresholds_on_weight_and_on_distance() -> None:
    world = WorldMap(spec(), mount())
    world.integrate_scan(*box_scan(), at())  # one scan: at most one observation per voxel
    assert world.lidar_slice().counts()["known"] == 0, "min_weight 2: one scan is not yet a map"
    lenient = world.lidar_slice(SliceLaw(min_weight=0.2))
    assert lenient.counts()["occupied"] > 0
    strict = world.lidar_slice(SliceLaw(min_weight=0.2, free_above=1.01))
    assert strict.counts()["free"] == 0, "nothing is free enough for that threshold"
    assert strict.counts()["occupied"] == lenient.counts()["occupied"]


def test_the_slice_geometry_matches_the_volumes_footprint() -> None:
    view = room().lidar_slice()
    assert view.shape == (120, 120)
    assert view.origin == (-3.0, -3.0)
    assert view.resolution_m == 0.05
    assert view.band_m[0] <= PLANE_M <= view.band_m[1]


# ---- what the stack reads the slice as -----------------------------------------------------
def test_the_occupancy_message_fields_are_what_ros_expects() -> None:
    view = room().lidar_slice()
    fields = view.message_fields()
    assert (fields.width, fields.height) == (120, 120)
    assert (fields.origin_x, fields.origin_y, fields.resolution) == (-3.0, -3.0, 0.05)
    assert fields.data.dtype == np.int8
    assert fields.data.size == fields.width * fields.height
    assert set(np.unique(fields.data)) <= {FREE, OCCUPIED, UNKNOWN}
    # data[i + j * width] is the cell i along x, j along y from the origin corner
    rows, cols = np.nonzero(view.values == OCCUPIED)
    for row, col in zip(rows[::37], cols[::37], strict=True):
        assert fields.data[col + row * fields.width] == OCCUPIED
        assert fields.as_list()[col + row * fields.width] == OCCUPIED
    row, col = rc(view, 0.0, 0.0)
    assert fields.data[col + row * fields.width] == FREE


def test_the_slice_is_the_tracker_map_the_relocalizer_builds_from_a_message() -> None:
    """``grid_from_msg`` in the node turns the three values into log-odds; the slice hands the
    tracker the same grid without a round trip through ROS."""
    view = room().lidar_slice()
    grid = view.to_log_odds()
    assert grid.spec.resolution_m == 0.05
    assert (grid.spec.x_min_m, grid.spec.y_min_m) == (-3.0, -3.0)
    assert grid.log_odds.shape == view.values.shape
    occupied = grid.occupied_xy()
    assert occupied.shape[0] == view.counts()["occupied"]
    assert np.abs(np.abs(occupied[:, 0]) - ROOM_M).min() < 0.05


def test_the_exported_pair_is_read_back_by_the_existing_map_loader(tmp_path: Path) -> None:
    world = room()
    view = world.lidar_slice()
    yaml_path = world.export_pgm_yaml(tmp_path / "world_test.npz", view)
    assert yaml_path.name == "world_test.yaml"
    assert (tmp_path / "world_test.pgm").exists()
    grid = grid_from_pgm(yaml_path)
    assert grid.spec.shape == view.shape
    assert grid.spec.resolution_m == pytest.approx(0.05)
    assert grid.spec.x_min_m == pytest.approx(-3.0)
    back = np.where(grid.log_odds > 0, OCCUPIED, np.where(grid.log_odds < 0, FREE, UNKNOWN))
    assert np.array_equal(back, view.values)


# ---- the snapshot --------------------------------------------------------------------------
def test_a_snapshot_round_trip_is_the_same_map(tmp_path: Path) -> None:
    world = room()
    intr, pose = looking_ahead(0.8)
    rgb = np.zeros((intr.height, intr.width, 3), dtype=np.uint8)
    rgb[:] = (10, 20, 30)
    world.integrate_depth(flat_depth(intr, 1.0), rgb, intr, pose, stamp=300.0)
    path = world.save(tmp_path / "world_room.npz")
    back = WorldMap.load(path, mount())
    assert back.spec == world.spec
    assert np.array_equal(back.volume.sdf, world.volume.sdf)
    assert np.array_equal(back.volume.weight, world.volume.weight)
    assert np.array_equal(back.lidar_weight, world.lidar_weight)
    assert np.array_equal(back.volume.rgb, world.volume.rgb)
    assert back.stamp == 300.0
    assert back.lidar_plane_m == pytest.approx(PLANE_M)
    assert np.array_equal(back.lidar_slice().values, world.lidar_slice().values)
    # the frame index a loop closure would replay: stamp, sensor and the pose each frame went in at
    assert [f[1] for f in back.frames] == ["lidar"] * 8 + ["camera"]
    assert [f[0] for f in back.frames] == [*range(100, 108), 300.0]
    assert np.allclose(back.frames[-1][2][:, 3], pose.translation)
    # the loaded volume keeps defending the lidar's layer
    intr, flat = looking_ahead(PLANE_M)
    before = back.lidar_slice().values.copy()
    back.integrate_depth(flat_depth(intr, 1.0), None, intr, flat)
    assert np.array_equal(back.lidar_slice().values, before)
    assert back.protected_rows == world.protected_rows, "and defends that layer, not more"


def test_a_snapshot_does_not_bring_back_a_tips_claim_on_the_cameras_band(tmp_path: Path) -> None:
    """The band is rebuilt from the saved plane, not from every row a beam ever reached: a
    restart after a tip must not hand the lidar the camera's band for the next run either."""
    pose = tipped(-TIP_DEG)
    angles, ranges, _points = fan_scan(pose, mount())
    world = WorldMap(tilt_spec(), mount())
    world.integrate_scan(angles, ranges, pose, stamp=100.0)
    back = WorldMap.load(world.save(tmp_path / "tipped.npz"), mount())
    written = np.flatnonzero(back.lidar_weight.any(axis=(0, 1)))
    rows = world.protected_rows
    assert rows is not None
    assert back.protected_rows == rows, "the plane's layer, as before the snapshot"
    assert written[-1] + 1 > rows[1], "though the tipped beams wrote well above it"


def test_a_snapshot_of_another_version_is_refused(tmp_path: Path) -> None:
    world = WorldMap(spec(), mount())
    path = world.save(tmp_path / "old.npz")
    data = dict(np.load(path))
    data["version"] = np.array(99)
    np.savez_compressed(path, **data)
    with pytest.raises(ValueError, match="version 99"):
        WorldMap.load(path)


def test_a_saved_map_seeds_the_layer_as_the_starting_state() -> None:
    """A known room is a loaded map written into the volume — after that nothing in the stack
    can tell it from a room the cart discovered itself."""
    world = room()
    fresh = WorldMap(spec(), mount())
    view = world.lidar_slice()
    seeded = fresh.seed_from_grid(view.values, view.resolution_m, view.origin)
    assert seeded == view.counts()["known"]
    assert np.array_equal(fresh.lidar_slice().values, view.values)
    # and it keeps growing: a scan through a seeded wall clears it like any other cell
    angles, ranges = box_scan()
    for _ in range(30):
        fresh.integrate_scan(angles, np.full_like(ranges, 2.5), at())
    assert cell_of(fresh.lidar_slice(), 2.0, 0.0) == FREE
    assert not wall_at(fresh.lidar_slice(), 2.0, 0.0)


def test_a_saved_pgm_seeds_the_volume_through_the_existing_loader(tmp_path: Path) -> None:
    """The path a known room takes today: map_server's pair -> log-odds -> the lidar's layer."""
    world = room()
    yaml_path = world.export_pgm_yaml(tmp_path / "flat.npz")
    fresh = WorldMap(spec(), mount())
    assert fresh.seed_from_grid(*trinary_from_log_odds(grid_from_pgm(yaml_path))) > 0
    assert np.array_equal(fresh.lidar_slice().values, world.lidar_slice().values)


def test_seeding_a_map_that_misses_the_volume_writes_nothing() -> None:
    world = WorldMap(spec(), mount())
    values = np.full((10, 10), OCCUPIED, dtype=np.int8)
    assert world.seed_from_grid(values, 0.05, (50.0, 50.0)) == 0


# ---- odds and ends -------------------------------------------------------------------------
def test_a_seeded_map_becomes_the_lidars_own_layer_at_once() -> None:
    """The pgm a known room starts from is the lidar's word: the camera hands those rows back
    from the first frame. Without the claim the depth repainted the seeded walls in the seconds
    before the first revolution arrived, and the slice a tracker matches on started out worse
    than the file it was seeded from."""
    world = room()
    view = world.lidar_slice()
    fresh = WorldMap(spec(), mount())
    assert fresh.protected_rows is None
    assert fresh.seed_from_grid(view.values, view.resolution_m, view.origin) > 0
    assert fresh.protected_rows is not None, "the seeded rows are the lidar's layer"
    intr, pose = looking_ahead(PLANE_M)
    for _ in range(8):  # the camera insisting the wall is at 1 m, into the seeded layer
        fresh.integrate_depth(flat_depth(intr, 1.0), None, intr, pose)
    assert np.array_equal(fresh.lidar_slice().values, view.values), "the seed stands"


def test_a_seeded_volume_hands_a_matcher_cut_above_the_seed_s_weight_nothing() -> None:
    """What a seeded start costs the camera, stated once so nobody discovers it on the robot:
    seeding writes ONE weight per cell (4.0), so the map's own cut carries every seeded wall
    while a matcher's cut in the tens carries none of them, and /map_camera goes out
    all-unknown until the camera has painted its own frames on a cell. The band then hardens at
    the lidar's poses, which is the point — but the camera says nothing while it does."""
    world = room()
    view = world.lidar_slice()
    fresh = WorldMap(spec(), mount())
    seeded = fresh.seed_from_grid(view.values, view.resolution_m, view.origin)
    assert seeded > 0
    walls = view.counts()["occupied"]
    assert fresh.camera_band_slice(SliceLaw(min_weight=2.0)).counts()["occupied"] == walls
    hard = fresh.hardness(SliceLaw(min_weight=20.0), SliceLaw(min_weight=2.0))
    assert hard == {
        "occupied": 0.0,
        "occupied_floor": float(walls),
        "share": 0.0,
        "min_weight": 20.0,
    }, "a seeded wall is invisible to any cut above the weight seeding wrote"


def test_maturity_and_the_report_line_say_what_is_in_the_volume() -> None:
    world = room()
    stats = world.maturity()
    assert stats["voxels"] > 0
    assert stats["lidar_voxels"] == stats["voxels"], "only the lidar has spoken"
    assert 2.0 <= stats["mean_weight"] <= 8.0, "eight scans, each worth at most one"
    assert stats["frames"] == 8.0
    assert stats["plane_m"] == pytest.approx(PLANE_M)
    text = world.report()
    assert "lidar slice" in text and "camera band" in text
    assert WorldMap(spec(), mount()).maturity()["mean_weight"] == 0.0


def test_bearings_in_base_undoes_an_upside_down_mount() -> None:
    angles = np.array([0.0, 0.5])
    assert np.allclose(bearings_in_base(angles, 0.25, False), [0.25, 0.75])
    assert np.allclose(bearings_in_base(angles, 0.25, True), [0.25, -0.25])


def test_the_snapshot_clock_fires_on_the_period() -> None:
    clock = SnapshotClock(every_s=60.0)
    assert clock.due(10.0) and clock.age_s(10.0) == math.inf
    clock.done(10.0)
    assert not clock.due(30.0)
    assert clock.age_s(30.0) == 20.0
    assert clock.due(70.0)


def test_the_message_packs_the_slice_the_way_map_server_decodes_it() -> None:
    """The row order of /map, /map_lidar and /map_camera, pinned.

    A tracker pointed at /map_lidar scored fit 0.00 at the true pose on 2026-09-14 and re-seated
    four metres away at 0.99, and a flipped row order was the first suspect. It is not the fault
    (scratch/map_lidar_vs_pgm.py: the live slice agrees with the served file on 69.7 % of its
    walls as packed and on 15-21 % under every flip — the volume is simply not the file yet), so
    this fixes the convention in a test instead of in a memory: data row 0 is the ORIGIN row,
    the lowest y, and the pgm's first row is the top one, the highest y.
    """
    values = np.full((3, 4), UNKNOWN, dtype=np.int8)
    values[0, 1] = OCCUPIED  # the second cell of the origin row
    values[2, 3] = FREE  # the far corner: highest y, highest x
    slice_ = OccupancySlice(
        values=values,
        sdf=np.zeros((3, 4), dtype=np.float32),
        weight=np.zeros((3, 4), dtype=np.float32),
        resolution_m=0.5,
        origin=(-1.0, -2.0),
        band_m=(0.3, 0.4),
    )
    fields = slice_.message_fields()
    assert (fields.width, fields.height) == (4, 3)
    assert (fields.origin_x, fields.origin_y) == (-1.0, -2.0)

    data = fields.as_list()
    assert len(data) == 12
    # map_server's own decoding: index = row * width + col, cell centre at
    # (origin_x + (col + 0.5) * res, origin_y + (row + 0.5) * res).
    occupied = [i for i, v in enumerate(data) if v == OCCUPIED]
    assert occupied == [1]
    row, col = divmod(occupied[0], fields.width)
    x = fields.origin_x + (col + 0.5) * fields.resolution
    y = fields.origin_y + (row + 0.5) * fields.resolution
    assert (x, y) == (-0.25, -1.75), "the occupied cell is where the slice put it"
    assert data[2 * fields.width + 3] == FREE

    # ...and the pgm the same volume writes is that grid with its rows the other way up, which
    # is what map_server reads back (pepin.mapping.grid_from_pgm flips it again).
    body = slice_.to_pgm().split(b"\n", 3)[3]
    pixels = np.frombuffer(body, dtype=np.uint8).reshape(3, 4)
    assert pixels[2, 1] < 64, "the occupied cell sits in the pgm's LAST row: row 0 is the top"
