import math

import numpy as np
import pytest

from pepin.mapping import GridSpec, OccupancyGrid, transform_to_world
from pepin.odometry import Pose2D


def test_transform_rotates_then_translates() -> None:
    pts = np.array([[1.0, 0.0]])
    out = transform_to_world(pts, Pose2D(x=2.0, y=3.0, theta=math.pi / 2))
    assert out[0] == pytest.approx([2.0, 4.0])


def test_hit_cell_becomes_occupied_and_beam_cells_free() -> None:
    grid = OccupancyGrid(GridSpec(resolution_m=0.1, x_min_m=-1, y_min_m=-1, width_m=4, height_m=2))
    for _ in range(3):
        grid.integrate(Pose2D(), np.array([[1.0, 0.0]]))
    p = grid.probability()
    hit = grid.world_to_cell(np.array([[1.0, 0.0]]))[0]
    mid = grid.world_to_cell(np.array([[0.5, 0.0]]))[0]
    untouched = grid.world_to_cell(np.array([[0.0, 0.5]]))[0]
    assert p[hit[0], hit[1]] > 0.9
    assert p[mid[0], mid[1]] < 0.3
    assert p[untouched[0], untouched[1]] == pytest.approx(0.5)


def test_points_outside_the_grid_are_ignored() -> None:
    grid = OccupancyGrid(GridSpec(resolution_m=0.5, x_min_m=0, y_min_m=0, width_m=1, height_m=1))
    grid.integrate(Pose2D(), np.array([[5.0, 5.0], [-3.0, 0.0]]))
    assert grid.log_odds.shape == (2, 2)


def test_log_odds_are_clamped() -> None:
    grid = OccupancyGrid(GridSpec(resolution_m=0.1, x_min_m=-1, y_min_m=-1, width_m=2, height_m=2))
    for _ in range(50):
        grid.integrate(Pose2D(), np.array([[0.5, 0.0]]))
    assert grid.log_odds.max() <= 5.0 and grid.log_odds.min() >= -5.0
