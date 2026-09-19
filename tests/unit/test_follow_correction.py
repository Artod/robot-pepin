"""The volume follows the graph's correction: a loop closure moves the map, not only the pose.

RTAB-Map optimises its graph and ``map -> odom`` jumps. Everything already painted was placed
through the old edge, so the room it drew is stale by exactly that difference. Here a synthetic
room is painted, a correction of 10 cm / 3 degrees is applied, and the slice a tracker matches
on must have moved by that and by nothing else — with the lidar's own layer, the seeded cells
and the snapshot travelling as one body with it.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pepin.tsdf import GridSpec, PlanarShift, RigidPose
from pepin.worldmap import (
    OCCUPIED,
    UNKNOWN,
    CorrectionFollower,
    OccupancySlice,
    PlanarMount,
    WorldMap,
)

ROOM_M = 2.0  # the box: walls at x, y = +-2 m
PLANE_M = 0.383  # the lidar's height, as config/lidar.json has it


def spec() -> GridSpec:
    """A 6 x 6 x 1.7 m grid at 5 cm: the synthetic room with a metre of margin."""
    return GridSpec(origin=(-3.0, -3.0, -0.15), shape=(120, 120, 34), camera_band_m=(0.15, 1.30))


def mount() -> PlanarMount:
    return PlanarMount(z_m=PLANE_M, min_range_m=0.05, max_range_m=12.0)


def box_scan(x: float = 0.0, y: float = 0.0, beams: int = 720) -> tuple[np.ndarray, np.ndarray]:
    """Bearings and ranges of a lidar standing at (x, y) inside the box."""
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
    """The synthetic box scanned a few times from the middle: walls, free floor and a frame
    index, all written by the real integration path."""
    world = WorldMap(spec(), mount())
    for i in range(scans):
        world.integrate_scan(*box_scan(), at(), stamp=100.0 + i)
    return world


def walls(view: OccupancySlice) -> np.ndarray:
    """The map coordinates of every occupied cell of a slice, as (n, 2) metres."""
    rows, cols = np.nonzero(view.values == OCCUPIED)
    return np.stack(
        [
            view.origin[0] + (cols + 0.5) * view.resolution_m,
            view.origin[1] + (rows + 0.5) * view.resolution_m,
        ],
        axis=1,
    )


def off_the_moved_wall(view: OccupancySlice, shift: PlanarShift) -> np.ndarray:
    """How far every occupied cell of a moved slice is from the box's wall as the correction
    puts it: the cell is carried back through the move and measured against the room it came
    from, so the claim is about each cell and not about an average that hides a smear."""
    points = walls(view)
    sx, sy = shift.source_of(points[:, 0], points[:, 1])
    return np.minimum(np.abs(np.abs(sx) - ROOM_M), np.abs(np.abs(sy) - ROOM_M))


# ---- the move itself -------------------------------------------------------------------------
def test_a_correction_carries_the_wall_by_exactly_what_it_says() -> None:
    """10 cm and 3 degrees of graph correction: every wall cell in the slice a tracker matches on
    stands where the correction puts it, within the half voxel the resample quantises to.

    THE MAXIMUM IS DERIVED AND UNCHANGED; THE MEDIAN MOVED WITH THE PAINTING LAW (2026-09-18). One
    voxel is the sum of two halves and always was: a cell is called occupied while the crossing is
    within half a voxel of it (``SliceLaw.occupied_t``), and ``nearest`` — the default resample
    since the footprint law reversed the two (see the law comparison below) — takes the source
    column within half a voxel. The old median bound of 0.02 was not derived: it described a volume
    painted before ``LidarLaw.beam_footprint``, whose box came out as 718 occupied cells clustered
    on the surface. The footprint law keeps only the 430 a return resolved, and a thin wall's cells
    are spread across the voxel instead of piled at its centre, so the honest median is half of the
    maximum. Measured: 5.00 cm and 2.44 cm against the 5.00 and 2.50 those two sentences allow.
    """
    world = room()
    before = world.lidar_slice().counts()["occupied"]
    shift = PlanarShift(0.10, -0.04, math.radians(3.0))
    world.shift(shift)  # the default law: nearest
    view = world.lidar_slice()
    off = off_the_moved_wall(view, shift)
    voxel = view.resolution_m
    assert off.max() <= voxel, "half a voxel of cell plus half a voxel of resample, and no more"
    assert np.median(off) <= 0.5 * voxel, "spread across that voxel, not piled at its edge"
    assert view.counts()["occupied"] >= 0.8 * before, "it is still a room, not a smear"
    assert view.origin == (-3.0, -3.0) and view.resolution_m == 0.05, "the grid never moves"


def test_a_whole_voxel_of_translation_is_the_map_shifted_cell_for_cell() -> None:
    """A move that is a whole number of voxels needs no interpolation at all, so the check can
    be exact: every occupied cell is where it was, two columns along."""
    world = room()
    before = world.lidar_slice().values.copy()
    world.shift(PlanarShift(0.10, 0.0, 0.0))
    after = world.lidar_slice().values
    assert np.array_equal(after[:, 2:], before[:, :-2]), "the map moved two cells in x, as given"


def test_a_correction_below_a_micrometre_is_not_a_move() -> None:
    world = room()
    before = world.lidar_slice().values.copy()
    world.shift(PlanarShift(1e-9, 0.0, 0.0))
    assert np.array_equal(world.lidar_slice().values, before)


def test_a_move_that_carries_the_room_off_the_grid_leaves_an_unknown_room() -> None:
    """Nothing is invented at the edges: a column whose source is off the grid comes back
    unknown, which is what a map that just grew a new edge should say."""
    world = room()
    world.shift(PlanarShift(100.0, 0.0, 0.0))
    counts = world.lidar_slice().counts()
    assert counts["known"] == 0 and counts["unknown"] == 120 * 120
    assert not world.lidar_weight.any(), "and the lidar's layer left with it"


# ---- what must survive the move ---------------------------------------------------------------
def test_the_lidars_layer_travels_with_the_room_and_still_defends_it() -> None:
    """The weight channel that makes the lidar's layer defensible moves through the very same
    column map as the field: after the correction the camera still may not repaint the wall,
    and it may not repaint it at the wall's NEW place."""
    world = room()
    rows = world.protected_rows
    shift = PlanarShift(0.10, -0.04, math.radians(3.0))
    world.shift(shift)
    assert world.protected_rows == rows, "a planar move has no z in it: the layer's rows stand"
    assert world.lidar_plane_m == pytest.approx(PLANE_M)
    view = world.lidar_slice()
    lo, hi = rows if rows is not None else (0, 0)
    owned = world.lidar_weight[:, :, lo:hi].max(axis=2) > 0.0
    occupied = view.values.T == OCCUPIED
    assert occupied.sum() > 200
    assert (owned & occupied).sum() / occupied.sum() > 0.99, "every moved wall is the lidar's"


def test_a_seeded_map_moves_as_one_body_and_stays_the_lidars() -> None:
    """A volume seeded from a saved map is not a second entity: the correction carries the
    seeded walls exactly as it carries painted ones, and they are still the lidar's own cells
    afterwards (which is what a known map is). In SLAM there is no seed and in known-map mode
    the volume does not follow at all — this is the guarantee, not the working case."""
    view = room().lidar_slice()
    fresh = WorldMap(spec(), mount())
    assert fresh.seed_from_grid(view.values, view.resolution_m, view.origin) > 0
    before = fresh.lidar_slice().counts()["occupied"]
    shift = PlanarShift(0.10, -0.04, math.radians(3.0))
    fresh.shift(shift)
    moved = fresh.lidar_slice()
    assert off_the_moved_wall(moved, shift).max() <= 0.05, "the seeded wall moved with the room"
    assert moved.counts()["occupied"] >= 0.8 * before
    assert fresh.lidar_weight.any(), "and it is still the lidar's word, at its new place"
    rows = fresh.protected_rows
    assert rows is not None
    owned = fresh.lidar_weight[:, :, rows[0] : rows[1]].max(axis=2) > 0.0
    assert (owned & (moved.values.T == OCCUPIED)).sum() / moved.counts()["occupied"] > 0.99


def test_the_frame_index_travels_with_the_room_it_painted() -> None:
    """The poses a re-fusion would replay by are map poses like any other: they move too, or
    the index would point at the old room."""
    world = room(scans=2)
    pose = world.frames[0][2].copy()
    shift = PlanarShift(0.10, -0.04, math.radians(3.0))
    world.shift(shift)
    moved = shift.applied_to(RigidPose(pose[:, :3], pose[:, 3]))
    assert np.allclose(world.frames[0][2][:, 3], moved.translation)
    assert np.allclose(world.frames[0][2][:, :3], moved.rotation)
    assert world.frames[0][0] == 100.0, "the stamps are untouched"


def test_a_snapshot_taken_after_the_move_reloads_as_the_moved_map(tmp_path: Path) -> None:
    """The snapshot is the map as it now stands, in the map frame as the graph now draws it:
    what is on disk and what is published can never disagree about where the wall is."""
    world = room()
    world.shift(PlanarShift(0.10, -0.04, math.radians(3.0)))
    path = world.save(tmp_path / "world.npz")
    back = WorldMap.load(path, mount())
    assert np.array_equal(back.lidar_slice().values, world.lidar_slice().values)
    assert back.protected_rows == world.protected_rows
    assert np.allclose(back.lidar_weight, world.lidar_weight)


def test_the_moved_volume_is_still_a_model_the_camera_can_be_seated_on() -> None:
    """The field keeps its law through the move: it is carried as distance times weight and
    divided back out, so a wall stays a surface with the weight it had rather than a smear
    pulled toward the unknown next door."""
    world = room()
    before = float(world.volume.weight.max())
    world.shift(PlanarShift(0.10, -0.04, math.radians(3.0)))
    assert world.volume.weight.max() == pytest.approx(before, rel=0.2)
    assert np.all(np.abs(world.volume.sdf) <= 1.0 + 1e-6), "still truncation units"
    assert np.all(world.volume.sdf[world.volume.weight == 0.0] == 1.0), "unknown reads empty"


# ---- when the volume owes the graph a move ----------------------------------------------------
def correction(x: float = 0.0, y: float = 0.0, yaw: float = 0.0) -> RigidPose:
    """A map -> odom correction: the edge the graph moves when it optimises."""
    return at(x, y, yaw)


def test_nothing_is_owed_until_a_correction_has_anchored_the_volume() -> None:
    follower = CorrectionFollower()
    assert follower.pending(correction(1.0), 0.05, 1.0) is None, "nothing painted yet"
    follower.anchor(correction(1.0))
    assert follower.pending(correction(1.0), 0.05, 1.0) is None, "and it stands where it was"


def test_small_corrections_accumulate_and_move_the_volume_together() -> None:
    """Three corrections of 2 cm are not three resamples of a 2.4 million voxel volume: each
    is measured against what is PAINTED, so the third one is 6 cm and moves it once."""
    follower = CorrectionFollower()
    follower.anchor(correction())
    assert follower.pending(correction(0.02), 0.05, 1.0) is None
    assert follower.pending(correction(0.04), 0.05, 1.0) is None
    shift = follower.pending(correction(0.06), 0.05, 1.0)
    assert shift is not None and shift.translation_m == pytest.approx(0.06)


def test_a_move_re_anchors_the_volume_on_the_correction_it_was_moved_to() -> None:
    follower = CorrectionFollower()
    follower.anchor(correction())
    shift = follower.pending(correction(0.30), 0.05, 1.0)
    assert shift is not None
    follower.moved(correction(0.30), shift)
    assert follower.applied == 1 and follower.last.translation_m == pytest.approx(0.30)
    assert follower.pending(correction(0.30), 0.05, 1.0) is None, "it stands there now"
    assert follower.pending(correction(0.32), 0.05, 1.0) is None, "2 cm more is still small"


def test_a_turn_alone_is_worth_a_move() -> None:
    """A loop closure that only rotates the graph moves every wall metres away at the far end
    of the flat: the threshold on the turn is not the same question as the one on the shift."""
    follower = CorrectionFollower()
    follower.anchor(correction())
    assert follower.pending(correction(yaw=math.radians(0.5)), 0.05, 1.0) is None
    shift = follower.pending(correction(yaw=math.radians(3.0)), 0.05, 1.0)
    assert shift is not None and shift.yaw_deg == pytest.approx(3.0)


def test_the_move_is_the_difference_between_two_corrections_not_one_of_them() -> None:
    """A cart that has driven off the origin: what the volume owes is new after old undone,
    which for a turn is a turn about the map's origin plus the translation that comes with it."""
    old, new = correction(1.0, 0.0, 0.0), correction(1.0, 0.0, math.radians(10.0))
    shift = PlanarShift.between(old, new)
    assert shift.yaw_deg == pytest.approx(10.0)
    point = np.array([2.0, 0.0])  # painted under `old`, two metres out
    want = new.rotation[:2, :2] @ (old.inverse().rotation[:2, :2] @ (point - old.translation[:2]))
    want = want + new.translation[:2]
    assert np.allclose(np.array(shift.moved(point[0], point[1])), want)


def test_an_unknown_room_is_where_the_correction_puts_it() -> None:
    """End to end on the map itself: paint, correct, and the cell that held the wall is empty
    while the cell the correction points at holds it."""
    world = room()
    view = world.lidar_slice()
    row, col = 60, int((ROOM_M - view.origin[0]) / view.resolution_m)
    assert view.values[row, col - 1 : col + 1].max() == OCCUPIED
    world.shift(PlanarShift(0.20, 0.0, 0.0))
    moved = world.lidar_slice()
    assert moved.values[row, col + 3 : col + 5].max() == OCCUPIED, "four cells along"
    assert moved.values[row, col - 1 : col + 1].max() != OCCUPIED, "and gone from where it was"
    assert moved.values[row, 0] == UNKNOWN


def test_the_lidars_claim_does_not_grow_by_a_ring_at_every_move() -> None:
    """The claim is a fact about cells the lidar swept, and a move carries it — it does not
    spread it. A bilinear blend of the weight channel alone would hand the lidar every cell with
    one owned source column among its four, +18 % of the layer over ten corrections
    (scratch/follow_refute.py), and lock the camera out of room nobody ever saw."""
    world = room()
    rows = world.protected_rows
    assert rows is not None
    lo, hi = rows

    def claimed() -> int:
        return int(np.count_nonzero(world.lidar_weight[:, :, lo:hi].max(axis=2) > 0.0))

    before = claimed()
    world.shift(PlanarShift(0.10, -0.04, math.radians(3.0)))
    assert claimed() <= before + 5, "a move carries the claim, it does not spread it"
    assert claimed() >= 0.97 * before, "and it does not eat it either"


def test_the_nearest_law_moves_the_same_room_without_thinning_it() -> None:
    """The two resample laws held against each other, and the order they come in reversed on
    2026-09-18: ``nearest`` keeps the room, ``blend`` widens it.

    ``blend`` used to be the default because a weighted average THINS a wall — a surface averaged
    with the free space in front of it — and the sensors repaint a thin wall while they never
    repaint a bias. ``LidarLaw.beam_footprint`` took the premise away: a far crossing now weighs
    only the share of its own disc that the voxel covers, so the free space in front of a wall is
    weak while the return is full weight, and the average is pulled INTO the wall. Measured here:
    blend takes 430 occupied cells to 516 — a wall two cells thick — with its worst cell 5.85 cm
    from where the correction points, past the voxel; nearest takes 430 to 431 with its worst at
    exactly half a voxel. So this test now pins the reversal rather than the old ordering, and the
    node's ``follow_correction_law`` default moved with it.
    """
    world, sharp = room(), room()
    shift = PlanarShift(0.10, -0.04, math.radians(3.0))
    before = world.lidar_slice().counts()["occupied"]
    lo, hi = world.protected_rows or (0, 0)
    claim = int(np.count_nonzero(world.lidar_weight[:, :, lo:hi].max(axis=2) > 0.0))
    world.shift(shift, "blend")
    sharp.shift(shift, "nearest")
    keen, blended = sharp.lidar_slice(), world.lidar_slice()
    assert keen.counts()["occupied"] >= before * 0.95, "nearest keeps every cell the map had"
    assert keen.counts()["occupied"] <= before * 1.05, "...and invents none: it is a resample"
    assert blended.counts()["occupied"] > keen.counts()["occupied"] * 1.1, (
        "blend is the one that widens the wall now, by a fifth of the cells"
    )
    assert off_the_moved_wall(keen, shift).max() <= keen.resolution_m, "a voxel, as derived above"
    assert off_the_moved_wall(blended, shift).max() > blended.resolution_m, "and blend is past it"
    assert off_the_moved_wall(keen, shift).max() <= 0.05, "and each is on the moved wall"
    moved_claim = int(np.count_nonzero(sharp.lidar_weight[:, :, lo:hi].max(axis=2) > 0.0))
    assert moved_claim <= claim + 5, "a nearest move cannot spread the lidar's claim either"
    assert np.all(sharp.volume.sdf[sharp.volume.weight == 0.0] == 1.0), "unknown reads empty"
