"""The stall look's arithmetic (pepin.stall_look): the plan ahead, the swept hull, the blockers,
the volume's columns and the verdict."""

from __future__ import annotations

import numpy as np
import pytest

from pepin.footprint import HULL
from pepin.stall_look import (
    Evidence,
    centroid,
    column_box,
    evidence,
    find_blockers,
    in_columns,
    path_ahead,
    swept_cells,
    verdict,
)

RES = 0.05
SIZE = 80  # a 4 x 4 m grid from (-1, -2)
ORIGIN = (-1.0, -2.0)
STRAIGHT = np.column_stack((np.arange(0.0, 3.0, 0.05), np.zeros(60)))


def cell(x: float, y: float) -> tuple[int, int]:
    """The (row, col) of the cell holding a point."""
    return int((y - ORIGIN[1]) / RES), int((x - ORIGIN[0]) / RES)


def centre(x: float, y: float) -> tuple[float, float]:
    row, col = cell(x, y)
    return ORIGIN[0] + (col + 0.5) * RES, ORIGIN[1] + (row + 0.5) * RES


def grid(*points: tuple[float, float]) -> np.ndarray:
    values = np.zeros((SIZE, SIZE), dtype=np.int16)
    for x, y in points:
        values[cell(x, y)] = 100
    return values


def test_the_plan_ahead_starts_at_the_vertex_nearest_the_cart() -> None:
    poses, s = path_ahead(STRAIGHT, (0.52, 0.03), ahead_m=1.0, step_m=0.25)
    assert s.tolist() == pytest.approx([0.0, 0.25, 0.5, 0.75, 1.0])
    assert poses[0, :2].tolist() == pytest.approx([0.5, 0.0])
    assert poses[-1, :2].tolist() == pytest.approx([1.5, 0.0])
    assert np.allclose(poses[:, 2], 0.0)


def test_the_plan_ahead_follows_a_turn_and_stops_at_its_end() -> None:
    corner = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.3]])
    poses, s = path_ahead(corner, (0.0, 0.0), ahead_m=2.0, step_m=0.1)
    assert s[-1] == pytest.approx(0.8)
    assert poses[-1, :2].tolist() == pytest.approx([0.5, 0.3])
    assert poses[-1, 2] == pytest.approx(np.pi / 2)
    assert path_ahead(np.array([[1.0, 1.0]]), (0.0, 0.0))[0].shape == (0, 3)


def test_the_swept_hull_covers_its_width_and_the_front_of_its_last_pose() -> None:
    poses = np.array([[0.0, 0.0, 0.0], [0.5, 0.0, 0.0]])
    rows, cols, s = swept_cells((SIZE, SIZE), ORIGIN, RES, poses, np.array([0.0, 0.5]), HULL, 0.0)
    xs = ORIGIN[0] + (cols + 0.5) * RES
    ys = ORIGIN[1] + (rows + 0.5) * RES
    assert xs.max() <= 0.5 + HULL.front_m and xs.min() >= -HULL.rear_m
    assert np.abs(ys).max() <= HULL.half_width_m
    ahead = (xs > HULL.front_m + 0.01) & (xs < 0.5)
    assert np.all(s[ahead] == 0.5) and np.all(s[~ahead] <= 0.5)


def test_blockers_are_the_lethal_cells_under_the_swept_hull_split_by_who_saw_them() -> None:
    values = grid((0.6, -0.1), (0.8, 0.1), (1.0, 0.0), (0.8, 1.0), (2.2, 0.0))
    lidar = np.array([centre(1.0, 0.0)])
    camera = np.array([centre(0.8, 0.1)])
    found = find_blockers(values, ORIGIN, RES, STRAIGHT, (0.5, 0.0), lidar, camera, ahead_m=1.0)
    assert found.count == 3  # beside the path and past the first metre are not blockers
    assert int(found.lidar.sum()) == 1 and int(found.camera.sum()) == 1
    assert int(found.unexplained.sum()) == 1
    candidates = found.xy[found.candidates(0.25)]
    assert sorted(map(tuple, np.round(candidates, 3))) == sorted(
        [tuple(np.round(centre(0.6, -0.1), 3)), tuple(np.round(centre(0.8, 0.1), 3))]
    )
    assert "3 lethal cells" in found.describe(1.0) and "camera-only 1" in found.describe(1.0)


def test_only_the_first_blocker_is_a_candidate() -> None:
    values = grid((0.7, 0.0), (1.4, 0.0))
    found = find_blockers(
        values, ORIGIN, RES, STRAIGHT, (0.5, 0.0), np.zeros((0, 2)), np.zeros((0, 2))
    )
    assert found.count == 2
    near = found.xy[found.candidates(0.25)]
    assert near.shape == (1, 2) and np.allclose(near[0], centre(0.7, 0.0))


def test_no_plan_or_only_the_lidars_cells_leave_nothing_to_look_at() -> None:
    values = grid((0.8, 0.0))
    none = find_blockers(
        values, ORIGIN, RES, np.zeros((0, 2)), (0.5, 0.0), np.zeros((0, 2)), np.zeros((0, 2))
    )
    assert none.count == 0 and "no lethal cell" in none.describe(1.0)
    lidar = np.array([centre(0.8, 0.0)])
    backed = find_blockers(values, ORIGIN, RES, STRAIGHT, (0.5, 0.0), lidar, np.zeros((0, 2)))
    assert backed.count == 1 and not np.any(backed.candidates())


def test_the_column_box_holds_the_cells_and_the_heights() -> None:
    box = column_box(np.array([[0.6, -0.1], [0.8, 0.1]]), RES, (0.0, 1.3))
    assert box.centre == pytest.approx((0.7, 0.0, 0.65))
    assert box.size == pytest.approx((0.3, 0.3, 1.3))


def test_points_over_the_cells_their_centroid_and_the_evidence() -> None:
    cells = np.array([[0.6, 0.0], [0.8, 0.0]])
    points = np.array([[0.61, 0.01, 0.2], [0.79, -0.02, 0.4], [1.2, 0.0, 0.3]])
    weights = np.array([10.0, 30.0, 5.0])
    over = in_columns(points, cells, RES)
    assert over.tolist() == [True, True, False]
    assert centroid(points[over], weights[over]) == pytest.approx((0.745, -0.0125, 0.35))
    assert centroid(np.zeros((0, 3)), np.zeros(0)) is None
    seen = evidence(points, weights, cells, RES)
    assert seen == Evidence(cells=2, occupied=2, weight=40.0) and seen.text() == (
        "2/2 cells, weight 40"
    )


@pytest.mark.parametrize(
    ("before", "after", "word"),
    [
        (Evidence(2, 0, 0.0), Evidence(2, 0, 0.0), "empty"),
        (Evidence(2, 2, 40.0), Evidence(2, 0, 0.0), "carved"),
        (Evidence(2, 2, 40.0), Evidence(2, 1, 12.0), "partly carved"),
        (Evidence(2, 2, 40.0), Evidence(2, 2, 55.0), "confirmed"),
    ],
)
def test_the_verdict(before: Evidence, after: Evidence, word: str) -> None:
    assert verdict(before, after) == word
