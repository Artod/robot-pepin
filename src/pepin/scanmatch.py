"""Correlative scan-to-map matching: let the lidar correct the odometry pose.

Odometry says roughly where the robot is; the map built so far says what
the room looks like. For every new scan we try candidate poses around the
odometry guess and keep the one under which the scan's points land on the
most occupied (and least known-free) cells. Brute-force search over a small
window, no gradients, no local minima surprises — the simplest matcher that
is robust enough to be worth understanding first.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pepin.mapping import OccupancyGrid
from pepin.odometry import Pose2D, wrap_angle


def relative_motion(before: Pose2D, after: Pose2D) -> Pose2D:
    """Motion from ``before`` to ``after`` expressed in the ``before`` frame."""
    dx, dy = after.x - before.x, after.y - before.y
    c, s = math.cos(before.theta), math.sin(before.theta)
    return Pose2D(c * dx + s * dy, -s * dx + c * dy, wrap_angle(after.theta - before.theta))


def apply_motion(pose: Pose2D, motion: Pose2D) -> Pose2D:
    """Pose reached by performing ``motion`` (in the robot frame) from ``pose``."""
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    return Pose2D(
        pose.x + c * motion.x - s * motion.y,
        pose.y + s * motion.x + c * motion.y,
        wrap_angle(pose.theta + motion.theta),
    )


@dataclass(frozen=True)
class SearchWindow:
    """Candidate poses around the guess: +-xy_m by xy_step_m, +-theta_deg by theta_step_deg."""

    xy_m: float = 0.08
    xy_step_m: float = 0.02
    theta_deg: float = 6.0
    theta_step_deg: float = 0.25

    def widened_for(self, motion: Pose2D, factor: float = 1.5) -> SearchWindow:
        """The same window, grown so that a large odometry step (a bus gap) still fits."""
        return SearchWindow(
            xy_m=max(self.xy_m, factor * math.hypot(motion.x, motion.y)),
            xy_step_m=self.xy_step_m,
            theta_deg=max(self.theta_deg, factor * abs(math.degrees(motion.theta))),
            theta_step_deg=self.theta_step_deg,
        )


def should_keyframe(
    motion: Pose2D, min_distance_m: float = 0.03, min_turn_deg: float = 2.0
) -> bool:
    """Match/integrate only after enough motion: sub-step drift is invisible to the search."""
    return (
        math.hypot(motion.x, motion.y) >= min_distance_m
        or abs(math.degrees(motion.theta)) >= math.radians(min_turn_deg) * 180 / math.pi
    )


@dataclass(frozen=True)
class MatchResult:
    pose: Pose2D
    score: float
    guess_score: float

    @property
    def improved(self) -> bool:
        return self.score > self.guess_score


class CorrelativeMatcher:
    """Finds the pose in a window around the guess that best explains a scan.

    Scores are read from a smoothed copy of the map (occupied cells spread
    over their 3x3 neighbourhood, free cells kept sharp) so that a
    sub-cell pose change still moves the score instead of stepping.
    """

    def __init__(
        self, grid: OccupancyGrid, window: SearchWindow | None = None, max_points: int = 200
    ) -> None:
        self._grid = grid
        self._window = window or SearchWindow()
        self._max_points = max_points
        self._field: NDArray[np.float64] | None = None

    def invalidate(self) -> None:
        """Call after the map changed; the score field is rebuilt lazily."""
        self._field = None

    def _score_field(self) -> NDArray[np.float64]:
        if self._field is None:
            lo = self._grid.log_odds
            occupied = np.maximum(lo, 0.0)
            spread = occupied.copy()
            for dr, dc, w in (
                (1, 0, 0.6),
                (-1, 0, 0.6),
                (0, 1, 0.6),
                (0, -1, 0.6),
                (1, 1, 0.4),
                (1, -1, 0.4),
                (-1, 1, 0.4),
                (-1, -1, 0.4),
            ):
                spread = np.maximum(spread, w * np.roll(np.roll(occupied, dr, axis=0), dc, axis=1))
            self._field = spread + np.minimum(lo, 0.0)
        return self._field

    def _subsample(self, points: NDArray[np.float64]) -> NDArray[np.float64]:
        if len(points) <= self._max_points:
            return points
        step = math.ceil(len(points) / self._max_points)
        return points[::step]

    def score(self, pose: Pose2D, points: NDArray[np.float64]) -> float:
        """Smoothed-map score of the scan placed at ``pose`` (0 off the map)."""
        return float(self._scores(pose.theta, points, np.array([[pose.x, pose.y]]))[0])

    def _scores(
        self, theta: float, points: NDArray[np.float64], positions: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Scores for one heading and many (x, y) positions; shape (len(positions),)."""
        field = self._score_field()
        c, s = math.cos(theta), math.sin(theta)
        rotated = points @ np.array([[c, s], [-s, c]])  # (P, 2) in world orientation
        world = positions[:, None, :] + rotated[None, :, :]  # (N, P, 2)
        cells = self._grid.world_to_cell(world.reshape(-1, 2))
        rows, cols = self._grid.spec.shape
        inside = (
            (cells[:, 0] >= 0) & (cells[:, 0] < rows) & (cells[:, 1] >= 0) & (cells[:, 1] < cols)
        )
        values = np.zeros(len(cells))
        values[inside] = field[cells[inside, 0], cells[inside, 1]]
        result: NDArray[np.float64] = values.reshape(len(positions), -1).sum(axis=1)
        return result

    def match(
        self, guess: Pose2D, points: NDArray[np.float64], window: SearchWindow | None = None
    ) -> MatchResult:
        window = window or self._window
        pts = self._subsample(points)
        n = round(window.xy_m / window.xy_step_m)
        offsets = np.arange(-n, n + 1) * window.xy_step_m
        xy_offsets = np.array([(dx, dy) for dx in offsets for dy in offsets])
        m = round(window.theta_deg / window.theta_step_deg)
        theta_offsets = np.radians(np.arange(-m, m + 1) * window.theta_step_deg)
        # Slight preference for staying near the guess breaks ties on flat score surfaces.
        xy_penalty = 1e-3 * np.abs(xy_offsets).sum(axis=1) / window.xy_step_m
        theta_penalty = 1e-3 * np.abs(theta_offsets) / math.radians(window.theta_step_deg)

        positions = np.array([[guess.x, guess.y]]) + xy_offsets
        best_score, best_pose = -math.inf, guess
        for k, dtheta in enumerate(theta_offsets):
            scores = (
                self._scores(guess.theta + dtheta, pts, positions) - xy_penalty - theta_penalty[k]
            )
            i = int(np.argmax(scores))
            if scores[i] > best_score:
                best_score = float(scores[i])
                best_pose = Pose2D(
                    positions[i, 0], positions[i, 1], wrap_angle(guess.theta + dtheta)
                )
        return MatchResult(pose=best_pose, score=best_score, guess_score=self.score(guess, pts))
