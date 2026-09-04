"""Path planning on an occupancy grid: obstacle inflation plus A*.

The robot is a disc, so instead of planning for a shape we grow every
obstacle by the robot radius and plan for a point. What comes out is a
short list of world waypoints (cell centres) with the straight runs
collapsed — only the turns survive.
"""

from __future__ import annotations

import heapq
import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pepin.mapping import OccupancyGrid

UNKNOWN_TOLERANCE = 0.05
"""How close to p = 0.5 a cell must be to count as never observed."""

START_RESCUE_CELLS = 3
"""How far to look for free ground when the robot starts inside the inflation zone."""

_SQRT2 = math.sqrt(2.0)
_STEPS: tuple[tuple[int, int, float], ...] = (
    (-1, 0, 1.0),
    (1, 0, 1.0),
    (0, -1, 1.0),
    (0, 1, 1.0),
    (-1, -1, _SQRT2),
    (-1, 1, _SQRT2),
    (1, -1, _SQRT2),
    (1, 1, _SQRT2),
)

Cell = tuple[int, int]


def inflate(occupied: NDArray[np.bool_], radius_cells: float) -> NDArray[np.bool_]:
    """Binary dilation: grow every occupied cell into a disc of ``radius_cells`` cells.

    The radius is not rounded up to whole cells: 0.32 m on a 5 cm grid is a
    6.4-cell disc, not a 7-cell (0.35 m) one — that extra cell is what closes
    doorways and goals near walls.
    """
    grown: NDArray[np.bool_] = np.array(occupied, dtype=bool, copy=True)
    if radius_cells <= 0:
        return grown
    rows, cols = occupied.shape
    reach = math.ceil(radius_cells)
    for dr in range(-reach, reach + 1):
        for dc in range(-reach, reach + 1):
            if dr * dr + dc * dc > radius_cells * radius_cells:
                continue
            dst = (slice(max(0, dr), rows + min(0, dr)), slice(max(0, dc), cols + min(0, dc)))
            src = (slice(max(0, -dr), rows + min(0, -dr)), slice(max(0, -dc), cols + min(0, -dc)))
            grown[dst] |= occupied[src]
    return grown


@dataclass(frozen=True)
class PlannerConfig:
    """What the planner treats as impassable and how wide the robot is, in meters."""

    occupied_threshold: float = 0.65
    # Hull plus margin. Must stay larger than SafetyBox.body_half_width_m (the hull):
    # a path the planner accepts would otherwise be vetoed by the lidar guard next to
    # every obstacle, and the robot would stand beside it twitching instead of passing.
    robot_radius_m: float = 0.30
    unknown_is_free: bool = False


class GridPlanner:
    """A* over an occupancy grid inflated by the robot radius.

    ``blocked`` is the boolean mask the search runs on: obstacles (and, by
    default, never-observed cells) grown by the robot radius in cells.
    """

    def __init__(self, grid: OccupancyGrid, config: PlannerConfig | None = None) -> None:
        self.grid = grid
        """Builds the inflated blocked mask once; the grid is not kept in sync afterwards."""
        self.spec = grid.spec
        self.config = config or PlannerConfig()
        probability = grid.probability()
        obstacles: NDArray[np.bool_] = probability >= self.config.occupied_threshold
        if not self.config.unknown_is_free:
            obstacles |= np.abs(probability - 0.5) <= UNKNOWN_TOLERANCE
        radius_cells = self.config.robot_radius_m / self.spec.resolution_m
        self._radius_cells = radius_cells
        self._raw: NDArray[np.bool_] = obstacles  # before inflation: what is really there
        self._static: NDArray[np.bool_] = inflate(obstacles, radius_cells)
        self.blocked: NDArray[np.bool_] = self._static  # static plus the last plan's live obstacles

    def plan(
        self,
        start_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        obstacles_xy: NDArray[np.float64] | None = None,
    ) -> list[tuple[float, float]] | None:
        """World waypoints from start to goal inclusive, or ``None`` if no free route exists.

        ``obstacles_xy`` are live sensor hits (map frame, meters) that are not
        in the map — a person, a moved chair: they are inflated like walls
        for this plan only. Straight runs are collapsed, so consecutive
        waypoints are turns.

        The robot's own footprint is exempt from inflation: something 10 cm
        from the hull would otherwise block the very cell the robot stands on
        and no plan could exist. Inside one robot radius of the start only the
        raw obstacle cells count, so the path leaves in any direction that does
        not cross the thing itself; full inflation applies from there on.
        """
        raw = self._raw
        self.blocked = self._static
        if obstacles_xy is not None and len(obstacles_xy):
            live = np.zeros_like(self._static)
            hits = self.grid.world_to_cell(obstacles_xy)
            rows, cols = live.shape
            ok = (hits[:, 0] >= 0) & (hits[:, 0] < rows) & (hits[:, 1] >= 0) & (hits[:, 1] < cols)
            live[hits[ok, 0], hits[ok, 1]] = True
            raw = raw | live
            self.blocked = self._static | inflate(live, self._radius_cells)
        start, goal = self._cell(start_xy), self._cell(goal_xy)
        if start is None or goal is None or self.blocked[goal]:
            return None
        if self.blocked[start]:
            self.blocked = self._clear_footprint(self.blocked, raw, start)
        if self.blocked[start]:
            rescued = self._nearest_free(start)
            if rescued is None:
                return None
            start = rescued
        cells = self._astar(start, goal)
        if cells is None:
            return None
        return [self._world(cell) for cell in _drop_collinear(cells)]

    def _cell(self, xy: tuple[float, float]) -> Cell | None:
        """World (x, y) to a (row, col) cell, or ``None`` when it falls off the grid."""
        rows, cols = self.blocked.shape
        col = math.floor((xy[0] - self.spec.x_min_m) / self.spec.resolution_m)
        row = math.floor((xy[1] - self.spec.y_min_m) / self.spec.resolution_m)
        if not (0 <= row < rows and 0 <= col < cols):
            return None
        return (row, col)

    def _world(self, cell: Cell) -> tuple[float, float]:
        """Centre of a (row, col) cell in world meters."""
        row, col = cell
        return (
            self.spec.x_min_m + (col + 0.5) * self.spec.resolution_m,
            self.spec.y_min_m + (row + 0.5) * self.spec.resolution_m,
        )

    def _clear_footprint(
        self, blocked: NDArray[np.bool_], raw: NDArray[np.bool_], start: Cell
    ) -> NDArray[np.bool_]:
        """Copy of ``blocked``; within one robot radius of ``start`` only raw obstacles block."""
        rows, cols = blocked.shape
        r = self._radius_cells
        reach = math.ceil(r)
        r0, r1 = max(0, start[0] - reach), min(rows, start[0] + reach + 1)
        c0, c1 = max(0, start[1] - reach), min(cols, start[1] + reach + 1)
        rr, cc = np.ogrid[r0:r1, c0:c1]
        disc = (rr - start[0]) ** 2 + (cc - start[1]) ** 2 <= r * r
        cleared = blocked.copy()
        window = cleared[r0:r1, c0:c1]
        window[disc] = raw[r0:r1, c0:c1][disc]
        return cleared

    def _nearest_free(self, cell: Cell) -> Cell | None:
        """Closest free cell within ``START_RESCUE_CELLS``, or ``None`` if the robot is boxed in."""
        rows, cols = self.blocked.shape
        reach = START_RESCUE_CELLS
        offsets = [
            (dr, dc)
            for dr in range(-reach, reach + 1)
            for dc in range(-reach, reach + 1)
            if dr * dr + dc * dc <= reach * reach
        ]
        offsets.sort(key=lambda o: o[0] * o[0] + o[1] * o[1])
        for dr, dc in offsets:
            row, col = cell[0] + dr, cell[1] + dc
            if 0 <= row < rows and 0 <= col < cols and not self.blocked[row, col]:
                return (row, col)
        return None

    def _astar(self, start: Cell, goal: Cell) -> list[Cell] | None:
        """8-connected A* with an octile heuristic; diagonals may not cut blocked corners."""
        rows, cols = self.blocked.shape
        queue: list[tuple[float, float, Cell]] = [(_octile(start, goal), 0.0, start)]
        came_from: dict[Cell, Cell] = {}
        best: dict[Cell, float] = {start: 0.0}
        closed: set[Cell] = set()
        while queue:
            _, cost, cell = heapq.heappop(queue)
            if cell == goal:
                return _trace(came_from, goal)
            if cell in closed:
                continue
            closed.add(cell)
            row, col = cell
            for dr, dc, step in _STEPS:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or self.blocked[nr, nc]:
                    continue
                if dr and dc and (self.blocked[row + dr, col] or self.blocked[row, col + dc]):
                    continue
                candidate = cost + step
                if candidate < best.get((nr, nc), math.inf):
                    best[(nr, nc)] = candidate
                    came_from[(nr, nc)] = cell
                    heapq.heappush(
                        queue, (candidate + _octile((nr, nc), goal), candidate, (nr, nc))
                    )
        return None


def _octile(cell: Cell, goal: Cell) -> float:
    """Cost of the cheapest obstacle-free 8-connected route between two cells."""
    dr, dc = abs(cell[0] - goal[0]), abs(cell[1] - goal[1])
    return float(max(dr, dc) + (_SQRT2 - 1.0) * min(dr, dc))


def _trace(came_from: dict[Cell, Cell], goal: Cell) -> list[Cell]:
    """Walk the parent links back from ``goal`` and return the cells start-first."""
    path = [goal]
    while path[-1] in came_from:
        path.append(came_from[path[-1]])
    path.reverse()
    return path


def _drop_collinear(cells: Sequence[Cell]) -> list[Cell]:
    """Keep the endpoints and the cells where the step direction changes."""
    if len(cells) <= 2:
        return list(cells)
    kept = [cells[0]]
    for previous, cell, following in zip(cells, cells[1:], cells[2:], strict=False):
        before = (cell[0] - previous[0], cell[1] - previous[1])
        after = (following[0] - cell[0], following[1] - cell[1])
        if before != after:
            kept.append(cell)
    kept.append(cells[-1])
    return kept


def path_length(path: Sequence[tuple[float, float]]) -> float:
    """Total travelled distance along the waypoints, in meters."""
    return sum(math.dist(a, b) for a, b in itertools.pairwise(path))
