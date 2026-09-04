"""Occupancy-grid mapping from recorded poses and lidar scans.

A log-odds grid: every lidar return raises the odds of its cell being
occupied and lowers the odds of every cell the beam crossed. Fed with raw
odometry poses this produces the "before" map — the honest picture of how
far dead reckoning drifts — and later the same grid takes corrected poses.
"""

from __future__ import annotations

import math
from dataclasses import astuple, dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pepin.odometry import Pose2D

LOG_ODDS_HIT = 0.85
LOG_ODDS_MISS = -0.4
LOG_ODDS_CLAMP = 5.0


@dataclass(frozen=True)
class GridSpec:
    """Grid geometry: cell size and the world extent it covers, in meters."""

    resolution_m: float = 0.05
    x_min_m: float = -10.0
    y_min_m: float = -10.0
    width_m: float = 20.0
    height_m: float = 20.0

    @property
    def shape(self) -> tuple[int, int]:
        """Grid size in cells as (rows, cols) = (height, width), rounded up."""
        return (
            math.ceil(self.height_m / self.resolution_m),
            math.ceil(self.width_m / self.resolution_m),
        )


def transform_to_world(points_xy: NDArray[np.float64], pose: Pose2D) -> NDArray[np.float64]:
    """Robot-frame points (N, 2) into the world frame given the robot pose."""
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    rotation = np.array([[c, -s], [s, c]])
    world: NDArray[np.float64] = points_xy @ rotation.T + np.array([pose.x, pose.y])
    return world


class OccupancyGrid:
    """Log-odds occupancy grid; rows are y, columns are x."""

    def __init__(self, spec: GridSpec) -> None:
        """Allocates the log-odds array for ``spec``; zero everywhere means "unknown"."""
        self.spec = spec
        self.log_odds: NDArray[np.float64] = np.zeros(spec.shape, dtype=np.float64)

    def world_to_cell(self, xy: NDArray[np.float64]) -> NDArray[np.int64]:
        """(N, 2) world points to (N, 2) integer [row, col]; may fall outside the grid."""
        cols = np.floor((xy[:, 0] - self.spec.x_min_m) / self.spec.resolution_m)
        rows = np.floor((xy[:, 1] - self.spec.y_min_m) / self.spec.resolution_m)
        return np.column_stack((rows, cols)).astype(np.int64)

    def _inside(self, cells: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Boolean mask of the (N, 2) cells that actually land on the grid."""
        rows, cols = self.spec.shape
        return (cells[:, 0] >= 0) & (cells[:, 0] < rows) & (cells[:, 1] >= 0) & (cells[:, 1] < cols)

    def integrate(self, pose: Pose2D, points_robot: NDArray[np.float64]) -> None:
        """Add one scan taken at ``pose``: free space along each beam, a hit at its end.

        ``points_robot`` is (N, 2) meters in the robot frame; cells gain
        +0.85 log-odds per hit, -0.4 per crossing, clamped to +-5.
        """
        if len(points_robot) == 0:
            return
        hits = transform_to_world(points_robot, pose)
        origin = np.array([pose.x, pose.y])
        # Sample each beam at one cell spacing, stopping short of the hit cell.
        lengths = np.linalg.norm(hits - origin, axis=1)
        steps = max(1, int(np.ceil(lengths.max() / self.spec.resolution_m)))
        fractions = np.linspace(0.0, 1.0, steps, endpoint=False)[1:]
        samples = origin + (hits - origin)[:, None, :] * fractions[None, :, None]
        keep = fractions[None, :] * lengths[:, None] < lengths[:, None] - self.spec.resolution_m
        free = self.world_to_cell(samples[keep].reshape(-1, 2))
        free = free[self._inside(free)]
        np.add.at(self.log_odds, (free[:, 0], free[:, 1]), LOG_ODDS_MISS)
        hit_cells = self.world_to_cell(hits)
        hit_cells = hit_cells[self._inside(hit_cells)]
        np.add.at(self.log_odds, (hit_cells[:, 0], hit_cells[:, 1]), LOG_ODDS_HIT)
        np.clip(self.log_odds, -LOG_ODDS_CLAMP, LOG_ODDS_CLAMP, out=self.log_odds)

    def save(self, path: str | Path) -> None:
        """Write the grid (log-odds and geometry) to a compressed ``.npz`` file."""
        np.savez_compressed(path, log_odds=self.log_odds, spec=np.array(astuple(self.spec)))

    @classmethod
    def load(cls, path: str | Path) -> OccupancyGrid:
        """Read a grid saved with :meth:`save`."""
        data = np.load(path)
        res, x_min, y_min, width, height = (float(v) for v in data["spec"])
        grid = cls(GridSpec(res, x_min, y_min, width, height))
        grid.log_odds = data["log_odds"].astype(np.float64)
        return grid

    def occupied_xy(self, threshold: float = 0.7) -> NDArray[np.float64]:
        """World (x, y) centres of cells whose occupancy probability exceeds ``threshold``."""
        rows, cols = np.nonzero(self.probability() > threshold)
        xs = self.spec.x_min_m + (cols + 0.5) * self.spec.resolution_m
        ys = self.spec.y_min_m + (rows + 0.5) * self.spec.resolution_m
        return np.column_stack((xs, ys))

    def probability(self) -> NDArray[np.float64]:
        """Occupancy probability per cell, 0.5 where nothing was observed."""
        probability: NDArray[np.float64] = 1.0 / (1.0 + np.exp(-self.log_odds))
        return probability
