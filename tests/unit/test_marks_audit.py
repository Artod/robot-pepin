"""pepin.marks_audit: who painted a lethal cell, on a grid built by hand.

Pure numpy — no ROS, no robot. The scenario is the one of run 0431 in miniature: a 1 m square of
local costmap at 5 cm, one cell the lidar returns from, one cell only the camera's fan covers, one
cell nothing explains, and one lethal cell outside the radius that must not be counted at all.
"""

from __future__ import annotations

import math

import numpy as np

from pepin.marks_audit import (
    INSCRIBED,
    LETHAL,
    audit_marks,
    cell_centres,
    nearest_distance,
    scan_points,
    transform_xy,
)

RES = 0.05
ORIGIN = (-1.0, -1.0)  # a 40x40 cell window from (-1, -1) to (+1, +1), the cart at the middle
SIZE = 40


def grid_with(*cells: tuple[int, int, int]) -> np.ndarray:
    """An empty grid with ``(row, col, value)`` written into it."""
    g = np.zeros((SIZE, SIZE), dtype=int)
    for row, col, value in cells:
        g[row, col] = value
    return g


def centre_of(row: int, col: int) -> tuple[float, float]:
    """The grid-frame centre of one cell, the way the module places them."""
    return ORIGIN[0] + (col + 0.5) * RES, ORIGIN[1] + (row + 0.5) * RES


def test_a_cell_is_placed_at_its_own_centre() -> None:
    mask = np.zeros((SIZE, SIZE), dtype=bool)
    mask[3, 7] = True
    got = cell_centres(mask, ORIGIN, RES)
    assert got.shape == (1, 2)
    assert np.allclose(got[0], centre_of(3, 7))


def test_nearest_distance_answers_infinity_when_there_is_nothing_to_measure_to() -> None:
    targets = np.array([[0.0, 0.0], [1.0, 0.0]])
    assert np.isinf(nearest_distance(targets, np.zeros((0, 2)))).all()
    assert nearest_distance(np.zeros((0, 2)), targets).shape == (0,)
    near = nearest_distance(targets, np.array([[0.0, 0.3], [1.0, 1.0]]))
    assert np.allclose(near, [0.3, 1.0])


def test_a_scan_becomes_points_and_a_bearing_without_a_return_is_gone() -> None:
    angles = np.array([0.0, math.pi / 2, math.pi])
    ranges = np.array([2.0, np.nan, 1.0])
    pts = scan_points(angles, ranges)
    assert pts.shape == (2, 2)
    assert np.allclose(pts[0], [2.0, 0.0])
    assert np.allclose(pts[1], [-1.0, 0.0], atol=1e-9)


def test_points_are_placed_by_the_full_rotation_not_by_a_yaw() -> None:
    """An upside-down lidar: a roll of pi mirrors y, and the height it gains is dropped."""
    roll = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])
    placed = transform_xy(np.array([[1.0, 0.5]]), roll, np.array([0.2, 0.0, 0.18]))
    assert np.allclose(placed[0], [1.2, -0.5])
    assert transform_xy(np.zeros((0, 2)), roll, np.zeros(3)).shape == (0, 2)


def _scenario() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The grid, the lidar's returns and the camera's fan of the scenario in the docstring."""
    lidar_cell, camera_cell, orphan_cell, far_cell = (20, 30), (24, 30), (16, 30), (20, 39)
    grid = grid_with(*[(r, c, LETHAL) for r, c in (lidar_cell, camera_cell, orphan_cell, far_cell)])
    lidar = np.array([centre_of(*lidar_cell)])
    camera = np.array([centre_of(*camera_cell)])
    return grid, lidar, camera


def test_each_lethal_cell_goes_to_the_sensor_that_can_account_for_it() -> None:
    grid, lidar, camera = _scenario()
    v = audit_marks(grid, ORIGIN, RES, (0.0, 0.0), lidar, camera, radius_m=0.8)
    assert (v.lethal, v.lidar_backed, v.camera_only, v.unexplained) == (3, 1, 1, 1)
    # the far cell sits 0.975 m out, past the 0.8 m radius: it is not judged at all
    assert v.lethal == 3
    assert v.camera_only_xy.shape == (1, 2)
    assert np.allclose(v.camera_only_xy[0], centre_of(24, 30))
    assert v.nearest_camera_only_m == np.hypot(*centre_of(24, 30))
    assert "camera-only 1" in v.report() and "nearest camera-only" in v.report()


def test_the_lidar_answers_first_when_both_sensors_cover_a_cell() -> None:
    """A cell both sensors see is the lidar's: only what the lidar cannot account for is the
    camera's, which is the whole point of the question."""
    grid, lidar, _ = _scenario()
    v = audit_marks(grid, ORIGIN, RES, (0.0, 0.0), lidar, lidar, radius_m=0.8)
    assert (v.lidar_backed, v.camera_only, v.unexplained) == (1, 0, 2)


def test_a_beam_further_than_match_cells_explains_nothing() -> None:
    """1.5 cells at 5 cm is 7.5 cm: a return 10 cm from the cell leaves it unexplained, and
    opening the tolerance to 3 cells takes it back."""
    grid = grid_with((20, 30, LETHAL))
    x, y = centre_of(20, 30)
    lidar = np.array([[x + 0.10, y]])
    assert audit_marks(grid, ORIGIN, RES, (0.0, 0.0), lidar, np.zeros((0, 2))).unexplained == 1
    wide = audit_marks(grid, ORIGIN, RES, (0.0, 0.0), lidar, np.zeros((0, 2)), match_cells=3.0)
    assert wide.lidar_backed == 1


def test_the_inflation_band_is_judged_only_when_it_is_asked_for() -> None:
    grid = grid_with((20, 30, LETHAL), (20, 31, INSCRIBED), (20, 32, 80))
    plain = audit_marks(grid, ORIGIN, RES, (0.0, 0.0), np.zeros((0, 2)), np.zeros((0, 2)))
    assert plain.lethal == 1
    wide = audit_marks(
        grid, ORIGIN, RES, (0.0, 0.0), np.zeros((0, 2)), np.zeros((0, 2)), inscribed_counts=True
    )
    assert wide.lethal == 2, "99 counts, the 80 of the inflation ramp never does"


def test_an_empty_grid_and_a_silent_camera_say_so_without_dividing_by_anything() -> None:
    empty = audit_marks(
        np.zeros((SIZE, SIZE), dtype=int),
        ORIGIN,
        RES,
        (0.0, 0.0),
        np.zeros((0, 2)),
        np.zeros((0, 2)),
    )
    assert (empty.lethal, empty.lidar_backed, empty.camera_only, empty.unexplained) == (0, 0, 0, 0)
    assert math.isinf(empty.nearest_camera_only_m)
    assert "nearest camera-only none" in empty.report()
    grid, lidar, _ = _scenario()
    blind = audit_marks(grid, ORIGIN, RES, (0.0, 0.0), lidar, np.zeros((0, 2)), radius_m=0.8)
    assert (blind.camera_only, blind.unexplained) == (0, 2)


def test_unknown_cells_are_never_a_mark() -> None:
    """-1 is unknown, and it must not be dragged in by a `>=` on a signed grid."""
    grid = grid_with((20, 30, -1), (21, 30, LETHAL))
    v = audit_marks(grid, ORIGIN, RES, (0.0, 0.0), np.zeros((0, 2)), np.zeros((0, 2)))
    assert v.lethal == 1
