"""The volume's occupied columns as the grids the costmaps only draw (pepin.camera_grid).

The wall is the plane x = 2 m painted by a camera at the origin, as in test_volume_scan: the
window about the cart must hold it where the fan holds it, the map canvas must carry it through
``map <- odom`` onto the map's own lattice, and each redraw must erase the last window whole — a
StaticLayer draws what it is sent and remembers nothing else, so a cell left behind here is a
phantom there.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from pepin.camera_grid import (
    FREE,
    OCCUPIED,
    CellRect,
    GridWindow,
    MapCanvas,
    MapGeometry,
    grid_volume,
    to_map_xy,
    window_box,
)
from pepin.depth import Intrinsics
from pepin.tsdf import GridSpec, RigidPose, Tsdf
from pepin.volume_scan import MarksLaw, band_surface, marks_ranges

INTR = Intrinsics(fx=200.0, fy=200.0, cx=80.0, cy=45.0, width=160, height=90)
WALL_X = 2.0


def spec() -> GridSpec:
    """A 6 x 6 m room 1.7 m tall on the robot's 5 cm lattice, the cart at its centre."""
    return GridSpec(origin=(-3.0, -3.0, -0.15), shape=(120, 120, 34))


def pose(x: float = 0.0, y: float = 0.0, yaw: float = 0.0, z: float = 0.0) -> RigidPose:
    """A planar pose: frame <- body, turned by ``yaw`` about z."""
    c, s = math.cos(yaw), math.sin(yaw)
    return RigidPose(np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]), np.array([x, y, z]))


def painted(frames: int = 3) -> Tsdf:
    """The wall x = WALL_X integrated ``frames`` times by a level camera at 0.6 m."""
    volume = Tsdf(spec())
    optical = RigidPose(
        np.stack([[0.0, -1.0, 0.0], [0.0, 0.0, -1.0], [1.0, 0.0, 0.0]], axis=1),
        np.array([0.0, 0.0, 0.6]),
    )
    for _ in range(frames):
        volume.integrate(np.full((INTR.height, INTR.width), WALL_X), None, INTR, optical)
    return volume


def occupied_xy(window: GridWindow, values: np.ndarray) -> np.ndarray:
    """The centres of a window's occupied cells, (n, 2) metres."""
    rows, cols = np.nonzero(values == OCCUPIED)
    r = window.resolution_m
    return np.c_[window.origin[0] + (cols + 0.5) * r, window.origin[1] + (rows + 0.5) * r]


# ---- the window about the cart -----------------------------------------------------------------
def test_the_window_is_centred_on_the_cart_and_moves_in_half_metre_steps() -> None:
    """The corner snaps to the half-metre lattice anchored at the volume's origin: the cart stays
    within a quarter metre of the centre, and a drive of less than that moves nothing (the
    rolling local layer re-sizes itself — a line in Nav2's log — only on a real step)."""
    lattice = (-3.0, -3.0)
    window = GridWindow.around((0.1, -0.2), lattice=lattice)
    assert window.cells == 120 and window.size_m == pytest.approx(6.0)
    assert window.origin == pytest.approx((-3.0, -3.0))
    assert window.centre == pytest.approx((0.0, 0.0))
    assert GridWindow.around((0.24, 0.0), lattice=lattice) == GridWindow.around(
        (0.0, 0.0), lattice=lattice
    )
    moved = GridWindow.around((0.26, 0.0), lattice=lattice)
    assert moved.origin == pytest.approx((-2.5, -3.0))
    # the corner stays on the volume's own voxel lattice, so a cell IS a voxel column
    odd = GridWindow.around((5.0, 5.0), lattice=(-3.02, 0.01))
    assert ((odd.origin[0] + 3.02) / 0.05) == pytest.approx(round((odd.origin[0] + 3.02) / 0.05))
    assert ((odd.origin[1] - 0.01) / 0.05) == pytest.approx(round((odd.origin[1] - 0.01) / 0.05))


def test_a_point_marks_the_one_cell_it_falls_in_rows_are_y() -> None:
    """OccupancyGrid's layout: row-major from the origin corner, rows along y. A point outside
    the square is dropped, not clamped onto the edge."""
    window = GridWindow((0.0, 0.0), 0.5, 4)
    values = window.draw(np.array([[0.1, 1.6, 0.3], [1.9, 0.1, 0.3], [2.5, 0.1, 0.3]]))
    assert values.dtype == np.int8 and values.shape == (4, 4)
    assert values[3, 0] == OCCUPIED and values[0, 3] == OCCUPIED
    assert int(np.count_nonzero(values)) == 2
    fields = window.fields(values)
    assert (fields.width, fields.height, fields.resolution) == (4, 4, 0.5)
    assert fields.data[3 * 4 + 0] == OCCUPIED and fields.data[0 * 4 + 3] == OCCUPIED
    assert set(fields.as_list()) == {FREE, OCCUPIED}, "0 and 100 only: no unknown is sent"


def test_the_grid_holds_the_wall_where_the_fan_does() -> None:
    """THE SAME COLUMN RULE AS /depth_marks: every bearing the fan marks ends in an occupied cell
    of the window (within one cell), and the occupied cells stand on the wall and nowhere else."""
    volume, cart, law = painted(), pose(), MarksLaw()
    window = GridWindow.around((0.0, 0.0), lattice=spec().origin[:2])
    twin = grid_volume(volume, window, cart, law)
    assert twin is not None
    values = window.draw(band_surface(twin, cart, law))
    cells = occupied_xy(window, values)
    assert cells.shape[0] > 10, "the wall is in the grid"
    assert np.all(np.abs(cells[:, 0] - WALL_X) <= 0.1), "on the wall, one voxel either side"
    ranges = marks_ranges(volume, cart, law)
    bearings = -math.pi + law.step * np.arange(ranges.size)
    hit = np.isfinite(ranges)
    ends = np.c_[ranges[hit] * np.cos(bearings[hit]), ranges[hit] * np.sin(bearings[hit])]
    for end in ends:
        assert np.min(np.hypot(*(cells - end).T)) <= 0.08, f"the fan's mark at {end} is a cell"


def test_a_window_off_the_volume_reads_nothing() -> None:
    """A cart outside its own volume (a window laid far away) copies nothing out."""
    window = GridWindow.around((50.0, 50.0))
    assert window_box(spec(), window, pose(50.0, 50.0), MarksLaw()) is None
    assert grid_volume(Tsdf(spec()), window, pose(50.0, 50.0), MarksLaw()) is None


# ---- the map's lattice -------------------------------------------------------------------------
def test_two_geometries_are_the_same_map_within_nav2_s_own_epsilon() -> None:
    """StaticLayer resizes the whole global costmap for a map it does not call the same: the
    canvas copies the map's geometry exactly, and this is the test it is compared by."""
    g = MapGeometry(0.05, 200, 160, -5.0, -4.0)
    assert g.same_as(MapGeometry(0.05, 200, 160, -5.0 + 5e-6, -4.0))
    assert not g.same_as(MapGeometry(0.05, 201, 160, -5.0, -4.0))
    assert not g.same_as(MapGeometry(0.05, 200, 160, -5.0 + 1e-4, -4.0))
    assert g.text() == "200x160 at 5 cm from (-5.00, -4.00)"


def test_a_rectangle_is_clipped_to_the_map_and_none_off_it() -> None:
    g = MapGeometry(0.1, 50, 40, 0.0, 0.0)
    assert g.rect_of(np.array([[1.0, 1.0], [2.05, 1.5]])) == CellRect(10, 10, 11, 5)
    assert g.rect_of(np.array([[-1.0, -1.0], [0.5, 0.5]])) == CellRect(0, 0, 5, 5)
    assert g.rect_of(np.array([[-3.0, -3.0], [-1.0, -1.0]])) is None
    assert g.rect_of(np.zeros((0, 2))) is None
    assert CellRect(0, 0, 2, 2).union(CellRect(5, 1, 1, 4)) == CellRect(0, 0, 6, 5)
    assert CellRect(3, 3, 1, 1).union(None) == CellRect(3, 3, 1, 1)


def test_each_redraw_erases_the_last_window_and_the_update_covers_both() -> None:
    """Nothing accumulates: the second draw clears every cell of the first window, and the
    rectangle it returns spans the old window and the new, so ONE update tells the layer both."""
    canvas = MapCanvas(MapGeometry(0.1, 100, 100, 0.0, 0.0))
    first = np.array([[1.0, 1.0], [3.0, 3.0]])
    rect = canvas.draw(np.array([[2.05, 2.05]]), first)
    assert rect == CellRect(10, 10, 20, 20)
    assert canvas.values[20, 20] == OCCUPIED
    second = np.array([[5.0, 5.0], [7.0, 7.0]])
    rect = canvas.draw(np.array([[6.05, 6.05]]), second)
    assert rect == CellRect(10, 10, 60, 60)
    assert canvas.values[20, 20] == FREE, "the old window is gone"
    assert canvas.values[60, 60] == OCCUPIED
    update = canvas.update(rect)
    assert (update.x, update.y, update.width, update.height) == (10, 10, 60, 60)
    assert update.data.size == 3600 and update.data[50 * 60 + 50] == OCCUPIED
    assert canvas.forget() == CellRect(50, 50, 20, 20)
    assert not canvas.values.any() and canvas.forget() is None
    full = canvas.full()
    assert (full.width, full.height, full.origin_x) == (100, 100, 0.0)


def test_a_window_off_the_map_still_erases_the_last_one() -> None:
    canvas = MapCanvas(MapGeometry(0.1, 20, 20, 0.0, 0.0))
    canvas.draw(np.array([[0.55, 0.55]]), np.array([[0.0, 0.0], [1.0, 1.0]]))
    rect = canvas.draw(np.array([[50.0, 50.0]]), np.array([[40.0, 40.0], [60.0, 60.0]]))
    assert rect == CellRect(0, 0, 10, 10) and not canvas.values.any()
    assert canvas.draw(np.zeros((0, 2)), np.array([[40.0, 40.0], [60.0, 60.0]])) is None


def test_points_are_carried_into_the_map_by_map_from_odom() -> None:
    """``map <- odom`` turned 90 degrees and shifted: a point 1 m ahead in odom lands 1 m to the
    left of the shift in map."""
    moved = to_map_xy(np.array([[1.0, 0.0, 0.4]]), pose(2.0, 3.0, math.pi / 2))
    np.testing.assert_allclose(moved, [[2.0, 4.0]], atol=1e-12)
    np.testing.assert_allclose(to_map_xy(np.array([[1.0, 0.0]]), pose(1.0)), [[2.0, 0.0]])
    assert to_map_xy(np.zeros((0, 3)), pose()).shape == (0, 2)
