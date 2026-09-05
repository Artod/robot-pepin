import itertools
import math

import numpy as np
import pytest

from pepin.mapping import GridSpec, OccupancyGrid
from pepin.planning import GridPlanner, PlannerConfig, inflate, path_length

FREE = -5.0
OCCUPIED = 5.0
UNKNOWN = 0.0

# A 6 x 4 m room at 10 cm resolution: 40 rows by 60 cols, walls all round and a
# dividing wall at col 30 (x = 3.05 m) with a gap over rows 26..33 (y = 2.6..3.4).
WALL_COL = 30
GAP_ROWS = slice(26, 34)
WALL_X = 3.05
ROBOT_RADIUS_M = 0.20  # exactly two cells of inflation on the 10 cm grid
GAP_LOW_Y = 2.8  # first free row centre in the gap after inflation, minus half a cell
GAP_HIGH_Y = 3.2


def _room(gap_log_odds: float = FREE) -> OccupancyGrid:
    """Walled room with one dividing wall pierced by a gap of the given occupancy."""
    grid = OccupancyGrid(
        GridSpec(resolution_m=0.1, x_min_m=0.0, y_min_m=0.0, width_m=6.0, height_m=4.0)
    )
    grid.log_odds[:] = FREE
    grid.log_odds[0, :] = grid.log_odds[-1, :] = OCCUPIED
    grid.log_odds[:, 0] = grid.log_odds[:, -1] = OCCUPIED
    grid.log_odds[:, WALL_COL] = OCCUPIED
    grid.log_odds[GAP_ROWS, WALL_COL] = gap_log_odds
    return grid


def _crossing_ys(path: list[tuple[float, float]], x_line: float) -> list[float]:
    """Heights at which the path polyline crosses the vertical line ``x_line``."""
    ys = []
    for (x1, y1), (x2, y2) in itertools.pairwise(path):
        if min(x1, x2) <= x_line <= max(x1, x2):
            t = 0.0 if x1 == x2 else (x_line - x1) / (x2 - x1)
            ys.append(y1 + t * (y2 - y1))
    return ys


def test_inflate_grows_a_cell_into_a_disc() -> None:
    mask = np.zeros((11, 11), dtype=bool)
    mask[5, 5] = True
    grown = inflate(mask, 2)
    assert grown.sum() == 13  # cells with dr^2 + dc^2 <= 4
    assert grown[5, 7] and grown[3, 5] and not grown[3, 4]
    assert mask.sum() == 1  # the input is left alone


def test_inflate_is_identity_at_zero_radius_and_does_not_wrap() -> None:
    mask = np.zeros((5, 5), dtype=bool)
    mask[0, 0] = True
    assert np.array_equal(inflate(mask, 0), mask)
    grown = inflate(mask, 1)
    assert grown[0, 1] and grown[1, 0] and not grown[4, 0] and not grown[0, 4]


def test_path_goes_through_the_gap_only() -> None:
    planner = GridPlanner(_room(), PlannerConfig(robot_radius_m=ROBOT_RADIUS_M))
    path = planner.plan((0.5, 1.0), (5.5, 1.0))
    assert path is not None
    crossings = _crossing_ys(path, WALL_X)
    assert crossings, "the path never reaches the dividing wall"
    assert all(GAP_LOW_Y <= y <= GAP_HIGH_Y for y in crossings)


def test_path_length_is_close_to_the_geometric_detour() -> None:
    planner = GridPlanner(_room(), PlannerConfig(robot_radius_m=ROBOT_RADIUS_M))
    start, goal = (0.5, 1.0), (5.5, 1.0)
    path = planner.plan(start, goal)
    assert path is not None
    # Shortest possible route: straight to the lower corner of the free gap, then straight on.
    corner = (WALL_X, GAP_LOW_Y + 0.05)
    reference = math.dist(path[0], corner) + math.dist(corner, path[-1])
    assert reference <= path_length(path) <= 1.10 * reference


def test_blocked_or_outside_endpoints_give_no_path() -> None:
    planner = GridPlanner(_room(), PlannerConfig(robot_radius_m=ROBOT_RADIUS_M))
    assert planner.plan((0.5, 1.0), (WALL_X, 1.0)) is None  # goal inside the dividing wall
    assert planner.plan((0.05, 0.05), (5.5, 1.0)) is None  # start walled into a corner
    assert planner.plan((0.5, 1.0), (9.0, 1.0)) is None  # goal off the grid
    assert planner.plan((-1.0, 1.0), (5.5, 1.0)) is None  # start off the grid


def test_start_inside_the_inflation_zone_plans_from_its_own_cell() -> None:
    """The footprint is exempt from inflation: the path starts where the robot is, not nudged."""
    planner = GridPlanner(_room(), PlannerConfig(robot_radius_m=ROBOT_RADIUS_M))
    path = planner.plan((0.25, 2.0), (1.5, 2.0))  # start two cells from the left wall
    assert path is not None
    assert path[0] == pytest.approx((0.25, 2.05))
    assert path[-1] == pytest.approx((1.55, 2.05))


def test_unknown_cells_block_unless_declared_free() -> None:
    grid = _room(gap_log_odds=UNKNOWN)
    blocked = GridPlanner(grid, PlannerConfig(robot_radius_m=ROBOT_RADIUS_M))
    assert blocked.plan((0.5, 1.0), (5.5, 1.0)) is None
    optimistic = GridPlanner(
        grid, PlannerConfig(robot_radius_m=ROBOT_RADIUS_M, unknown_is_free=True)
    )
    path = optimistic.plan((0.5, 1.0), (5.5, 1.0))
    assert path is not None
    assert all(GAP_LOW_Y <= y <= GAP_HIGH_Y for y in _crossing_ys(path, WALL_X))


def test_straight_corridor_collapses_to_two_waypoints() -> None:
    grid = OccupancyGrid(
        GridSpec(resolution_m=0.1, x_min_m=0.0, y_min_m=0.0, width_m=3.0, height_m=1.0)
    )
    grid.log_odds[:] = FREE
    planner = GridPlanner(grid, PlannerConfig(robot_radius_m=0.0))
    path = planner.plan((0.25, 0.5), (2.75, 0.5))
    assert path == [pytest.approx((0.25, 0.55)), pytest.approx((2.75, 0.55))]
    assert path_length(path) == pytest.approx(2.5)


def test_path_length_sums_the_segments() -> None:
    assert path_length([(0.0, 0.0), (3.0, 4.0)]) == pytest.approx(5.0)
    assert path_length([(1.0, 1.0)]) == 0.0


def test_live_obstacle_forces_a_detour_and_can_block_completely() -> None:
    import numpy as np

    from pepin.planning import GridPlanner, PlannerConfig, path_length

    planner = GridPlanner(_room(), PlannerConfig(robot_radius_m=ROBOT_RADIUS_M))
    direct = planner.plan((0.5, 3.0), (2.5, 3.0))  # straight run in the open left half
    assert direct is not None
    person = np.array([[1.5, 3.0], [1.5, 3.1], [1.5, 2.9]])  # standing on that line
    detour = planner.plan((0.5, 3.0), (2.5, 3.0), obstacles_xy=person)
    assert detour is not None and path_length(detour) > path_length(direct)
    assert all(abs(x - 1.5) > 0.2 or abs(y - 3.0) > 0.3 for x, y in detour)
    # People filling the only gap in the dividing wall: no way to the far side.
    crowd = np.array([[WALL_X, y] for y in np.arange(2.4, 3.61, 0.1)])
    assert planner.plan((0.5, 3.0), (5.5, 3.0), obstacles_xy=crowd) is None
    assert planner.plan((0.5, 3.0), (5.5, 3.0)) is not None  # the live layer does not stick


def test_start_next_to_a_live_obstacle_still_gets_a_plan() -> None:
    """Inflation must not swallow the robot's own cell: a hand at the hull is not a wall."""
    from test_localization import room_map

    planner = GridPlanner(room_map(), PlannerConfig(robot_radius_m=0.30))
    start, goal = (-1.0, 0.0), (1.0, 0.0)
    hand = np.array([[-0.95, 0.20]])  # 20 cm to the left of the start, inside the inflation disc
    path = planner.plan(start, goal, obstacles_xy=hand)
    assert path is not None
    assert path[-1] == pytest.approx(goal, abs=0.05)
    # The path leaves the footprint without crossing the hand itself.
    for (x0, y0), (x1, y1) in itertools.pairwise(path):
        for t in np.linspace(0.0, 1.0, 50):
            px, py = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            assert math.hypot(px - hand[0, 0], py - hand[0, 1]) > 0.05


def test_open_floor_path_is_one_straight_leg_not_a_staircase() -> None:
    """Start and goal offset diagonally by an odd ratio: A* zig-zags, the shortcut does not."""
    grid = OccupancyGrid(
        GridSpec(resolution_m=0.1, x_min_m=0.0, y_min_m=0.0, width_m=4.0, height_m=3.0)
    )
    grid.log_odds[:] = FREE
    planner = GridPlanner(grid, PlannerConfig(robot_radius_m=0.0))
    path = planner.plan((0.25, 0.25), (3.75, 1.45))
    assert path is not None
    assert len(path) == 2, path


def test_live_hits_on_mapped_walls_do_not_change_the_plan() -> None:
    """Wall points seen by the lidar (with localiser error) are the map, not new obstacles."""
    planner = GridPlanner(_room(), PlannerConfig(robot_radius_m=ROBOT_RADIUS_M))
    start, goal = (0.5, 1.0), (5.5, 1.0)
    plain = planner.plan(start, goal)
    wall_seen_again = np.array([[WALL_X + 0.05, y] for y in np.arange(0.2, 2.5, 0.1)])
    assert planner.plan(start, goal, obstacles_xy=wall_seen_again) == plain
    person = np.array([[3.05, 3.0]])  # inside the gap: genuinely new
    assert planner.plan(start, goal, obstacles_xy=person) is None  # the gap is the only way


def test_goal_covered_by_a_live_hit_is_served_from_the_nearest_free_cell() -> None:
    grid = OccupancyGrid(
        GridSpec(resolution_m=0.1, x_min_m=0.0, y_min_m=0.0, width_m=4.0, height_m=3.0)
    )
    grid.log_odds[:] = FREE
    planner = GridPlanner(grid, PlannerConfig(robot_radius_m=0.2))
    goal = (3.0, 1.5)
    someone_there = np.array([[3.05, 1.55]])
    path = planner.plan((0.5, 1.5), goal, obstacles_xy=someone_there)
    assert path is not None
    assert 0.1 <= math.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]) <= 0.5


def test_goal_next_to_a_wall_is_served_from_the_nearest_reachable_cell() -> None:
    """A goal 25 cm from a wall with a 35 cm radius: stop as close as the footprint allows."""
    planner = GridPlanner(_room(), PlannerConfig(robot_radius_m=0.35))
    goal = (1.0, 0.25)  # 25 cm above the bottom wall
    path = planner.plan((1.0, 2.0), goal)
    assert path is not None
    assert path[-1][1] > goal[1]  # pulled away from the wall, not into it
    assert math.hypot(path[-1][0] - goal[0], path[-1][1] - goal[1]) <= 0.5
    assert planner.plan((1.0, 2.0), (1.0, 0.05)) is None  # inside the wall itself: refused


def test_shortcut_does_not_squeeze_between_cells_that_touch_at_a_corner() -> None:
    grid = OccupancyGrid(
        GridSpec(resolution_m=0.1, x_min_m=0.0, y_min_m=0.0, width_m=2.0, height_m=2.0)
    )
    grid.log_odds[:] = FREE
    grid.log_odds[9, 10] = OCCUPIED  # two blocked cells sharing only a corner
    grid.log_odds[10, 9] = OCCUPIED
    planner = GridPlanner(grid, PlannerConfig(robot_radius_m=0.0))
    assert not planner._line_free((9, 9), (10, 10))
    path = planner.plan((0.95, 0.95), (1.85, 1.85))
    assert path is not None
    for (x1, y1), (x2, y2) in itertools.pairwise(path):
        assert not (x1 < 1.0 < x2 and y1 < 1.0 < y2 and abs((x2 - x1) - (y2 - y1)) < 1e-9)
