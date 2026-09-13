"""pepin.worldmap: the volume as the map — the lidar's layer, the slices, the snapshot.

A synthetic room (a box the beams are cast against analytically) is enough to check every
rule: the walls land where they are, the inside is carved free, the outside stays unknown, the
camera cannot repaint the lidar's layer, and the slice comes back out as the three things the
stack reads it as (a ROS message, the tracker's log-odds grid, a map_server pair).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pepin.depth import Intrinsics
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


def test_a_beam_past_the_sensor_reach_carves_free_space_and_marks_nothing() -> None:
    """An open door: the beam says "nothing out to here", not "a wall here"."""
    world = WorldMap(spec(), PlanarMount(z_m=PLANE_M, max_range_m=1.0))
    angles, ranges = box_scan()
    for _ in range(8):
        world.integrate_scan(angles, ranges, at())
    view = world.lidar_slice()
    assert cell_of(view, 0.5, 0.0) == FREE
    assert cell_of(view, 1.99, 0.0) == UNKNOWN, "the wall is out of reach: not marked"
    assert not np.any(view.values == OCCUPIED)


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


def test_the_camera_may_fill_the_layer_where_the_lidar_never_spoke() -> None:
    """Protection is per cell, not per height: the layer is the lidar's only where it looked."""
    world = room()  # nothing behind the wall was ever seen by the lidar
    intr, pose = looking_ahead(PLANE_M)
    far = RigidPose(pose.rotation, np.array([2.4, 0.0, PLANE_M]))
    world.integrate_depth(flat_depth(intr, 0.4), None, intr, far)
    assert wall_at(world.lidar_slice(), 2.8, 0.0)


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


def test_seeding_a_map_that_misses_the_volume_writes_nothing() -> None:
    world = WorldMap(spec(), mount())
    values = np.full((10, 10), OCCUPIED, dtype=np.int8)
    assert world.seed_from_grid(values, 0.05, (50.0, 50.0)) == 0


# ---- odds and ends -------------------------------------------------------------------------
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
