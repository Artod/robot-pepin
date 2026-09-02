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
    """Motion from ``before`` to ``after`` expressed in the ``before`` frame.

    The result reads as "x meters forward, y meters left, theta radians CCW",
    which is what the search window and the keyframe test are sized against.
    """
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
        """The same window grown to fit a large odometry step, keeping the candidate count.

        Ranges grow to ``factor`` times the step; the search steps grow by the
        same ratio, so a 90-degree jump costs the same as a 6-degree one.
        A fine pass around the coarse winner restores the resolution.
        """
        xy = max(self.xy_m, factor * math.hypot(motion.x, motion.y))
        theta = max(self.theta_deg, factor * abs(math.degrees(motion.theta)))
        return SearchWindow(
            xy_m=xy,
            xy_step_m=self.xy_step_m * xy / self.xy_m,
            theta_deg=theta,
            theta_step_deg=self.theta_step_deg * theta / self.theta_deg,
        )


def should_keyframe(
    motion: Pose2D, min_distance_m: float = 0.03, min_turn_deg: float = 2.0
) -> bool:
    """True once the robot moved far enough to be worth matching and integrating.

    Below ``min_distance_m`` of travel and ``min_turn_deg`` of turn the pose
    change is finer than the search step, so matching would only thicken walls.
    """
    return (
        math.hypot(motion.x, motion.y) >= min_distance_m
        or abs(math.degrees(motion.theta)) >= math.radians(min_turn_deg) * 180 / math.pi
    )


@dataclass(frozen=True)
class MatchResult:
    """Winning pose of a search, its score, and the score of the odometry guess."""

    pose: Pose2D
    score: float
    guess_score: float

    @property
    def improved(self) -> bool:
        """True when the corrected pose explains the scan better than raw odometry did."""
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
        """``window`` bounds the search around every guess; scans are thinned to at most
        ``max_points`` beams, which is what caps the cost of one match."""
        self._grid = grid
        self._window = window or SearchWindow()
        self._max_points = max_points
        self._field: NDArray[np.float64] | None = None

    def invalidate(self) -> None:
        """Call after the map changed; the score field is rebuilt lazily."""
        self._field = None

    def _score_field(self) -> NDArray[np.float64]:
        """The map blurred for scoring, cached until :meth:`invalidate`.

        Occupied cells bleed into their 3x3 neighbourhood (0.6 orthogonal, 0.4
        diagonal), free cells stay sharp and negative, so a wall attracts from ~1 cell away.
        """
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
        """Every k-th beam, down to at most ``max_points``: the score surface barely
        changes with beam count, the cost of the search scales with it linearly."""
        if len(points) <= self._max_points:
            return points
        step = math.ceil(len(points) / self._max_points)
        return points[::step]

    def score(self, pose: Pose2D, points: NDArray[np.float64]) -> float:
        """How well robot-frame ``points`` (N, 2) in meters fit the map when placed at ``pose``.

        Sums the smoothed map value under every point: positive on occupied
        cells, negative on known-free ones, zero outside the grid. Higher is better.
        """
        return float(self._scores(pose.theta, points, np.array([[pose.x, pose.y]]))[0])

    def _scores(
        self, theta: float, points: NDArray[np.float64], positions: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Vectorised :meth:`score` for one heading and many candidate origins.

        ``points`` is (P, 2) robot-frame meters, ``positions`` (N, 2) world meters;
        the scan is rotated once by ``theta`` and translated to each position.
        """
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

    def match_around(
        self, guess: Pose2D, points: NDArray[np.float64], motion: Pose2D, window: SearchWindow
    ) -> MatchResult:
        """Match with a window sized to the odometry ``motion``: coarse pass, then fine.

        Small steps use ``window`` directly. A large step (a bus gap, a
        turn-in-place) widens the window with coarser steps first, and the
        fine pass around that winner brings back the base resolution.
        """
        wide = window.widened_for(motion)
        if wide == window:
            return self.match(guess, points, window)
        coarse = self.match(guess, points, wide)
        return self.match(coarse.pose, points, window)

    def inlier_fraction(
        self, pose: Pose2D, points: NDArray[np.float64], min_field: float = 1.0
    ) -> float:
        """Share of scan points that land on confidently occupied cells when placed at ``pose``.

        A match acceptance test: a correct pose puts most points on walls, a
        wrong one scatters them into free or unknown space.
        """
        if len(points) == 0:
            return 0.0
        field = self._score_field()
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        world = points @ np.array([[c, s], [-s, c]]) + np.array([pose.x, pose.y])
        cells = self._grid.world_to_cell(world)
        rows, cols = self._grid.spec.shape
        inside = (
            (cells[:, 0] >= 0) & (cells[:, 0] < rows) & (cells[:, 1] >= 0) & (cells[:, 1] < cols)
        )
        values = np.full(len(cells), -np.inf)
        values[inside] = field[cells[inside, 0], cells[inside, 1]]
        return float((values >= min_field).mean())

    def match(
        self, guess: Pose2D, points: NDArray[np.float64], window: SearchWindow | None = None
    ) -> MatchResult:
        """Best pose for this scan on the grid of candidates around the odometry ``guess``.

        ``points`` are the scan's robot-frame (P, 2) meters. Every candidate is
        scored exhaustively; near-equal scores break toward ``guess``.
        """
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
