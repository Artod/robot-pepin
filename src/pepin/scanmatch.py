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
        self._field_version = -1

    def invalidate(self) -> None:
        """Force a rebuild of the score field; only needed after editing ``log_odds`` by hand.

        Scans added through ``OccupancyGrid.integrate`` bump the grid's version
        and the field follows on its own (a map built while driving works).
        """
        self._field = None

    def _score_field(self) -> NDArray[np.float64]:
        """The map blurred for scoring, rebuilt whenever the grid's version changes.

        Occupied cells bleed into their 3x3 neighbourhood (0.6 orthogonal, 0.4
        diagonal), free cells stay sharp and negative, so a wall attracts from ~1 cell away.
        """
        if self._field is None or self._field_version != self._grid.version:
            self._field_version = self._grid.version
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
        self,
        pose: Pose2D,
        points: NDArray[np.float64],
        min_field: float = 1.0,
        min_known: float = 0.25,
    ) -> float:
        """Share of judgeable scan points that land on confidently occupied cells at ``pose``.

        Only points falling on cells the map has observed (occupied or free)
        are judged — unknown space cannot vote against a match. Returns 0.0
        when fewer than ``min_known`` of the points are judgeable at all, so a
        handful of lucky hits on a tiny local map cannot pass as a closure.
        """
        if len(points) == 0:
            return 0.0
        values = self._values_at(pose, points)
        known = values != 0.0
        if known.mean() < min_known:
            return 0.0
        return float((values[known] >= min_field).mean())

    def field_score(self, pose: Pose2D, points: NDArray[np.float64]) -> float:
        """How exactly the scan sits on walls at ``pose``: the mean positive field value under
        the points, 1.0 when every point lands on an occupied cell. Sharper than the inlier
        fraction, which counts a point one cell off a wall the same as one on it."""
        if len(points) == 0:
            return 0.0
        top = float(np.maximum(self._score_field(), 0.0).max())
        if top <= 0.0:
            return 0.0
        return float(np.maximum(self._values_at(pose, points), 0.0).mean() / top)

    def contradiction_fraction(
        self, pose: Pose2D, points: NDArray[np.float64], max_field: float = -1.0
    ) -> float:
        """Share of scan points that land on cells the map knows to be free at ``pose``.

        The inlier fraction says how much of the scan a pose explains; this says
        how much of it the map denies. Twins (a symmetric room) explain the same
        share, but the wrong twin puts the one asymmetric chair on open floor.
        """
        if len(points) == 0:
            return 0.0
        values = self._values_at(pose, points)
        return float((values <= max_field).mean())

    def _values_at(self, pose: Pose2D, points: NDArray[np.float64]) -> NDArray[np.float64]:
        """Score-field value under every point of the scan placed at ``pose``; 0 off the grid."""
        field = self._score_field()
        c, s = math.cos(pose.theta), math.sin(pose.theta)
        world = points @ np.array([[c, s], [-s, c]]) + np.array([pose.x, pose.y])
        cells = self._grid.world_to_cell(world)
        rows, cols = self._grid.spec.shape
        inside = (
            (cells[:, 0] >= 0) & (cells[:, 0] < rows) & (cells[:, 1] >= 0) & (cells[:, 1] < cols)
        )
        values = np.zeros(len(cells))
        values[inside] = field[cells[inside, 0], cells[inside, 1]]
        return values

    def match(
        self, guess: Pose2D, points: NDArray[np.float64], window: SearchWindow | None = None
    ) -> MatchResult:
        """Best pose for this scan on the grid of candidates around the odometry ``guess``.

        ``points`` are the scan's robot-frame (P, 2) meters. Every candidate is
        scored exhaustively; near-equal scores break toward ``guess``.
        """
        window = window or self._window
        pts = self._subsample(points)
        pose, score = self._peak(*self._lattice(guess, pts, window))
        return MatchResult(pose=pose, score=score, guess_score=self.score(guess, pts))

    def match_two(
        self,
        guess: Pose2D,
        points: NDArray[np.float64],
        window: SearchWindow | None = None,
        apart_steps: int = 3,
    ) -> tuple[MatchResult, MatchResult]:
        """The best candidate, and the best one at least ``apart_steps`` lattice steps from it.

        Two peaks that score alike mean the scan fits two places (a symmetric
        room, look-alike rooms); the caller decides whether to trust the winner.
        """
        window = window or self._window
        pts = self._subsample(points)
        scores, positions, headings = self._lattice(guess, pts, window)
        best_pose, best_score = self._peak(scores, positions, headings)
        far_xy = (
            np.abs(positions - [best_pose.x, best_pose.y]).max(axis=1)
            >= apart_steps * window.xy_step_m - 1e-9
        )
        turned = headings - best_pose.theta
        far_theta = (
            np.abs(np.arctan2(np.sin(turned), np.cos(turned)))
            >= math.radians(apart_steps * window.theta_step_deg) - 1e-9
        )
        rival_pose, rival_score = self._peak(
            scores, positions, headings, far_theta[:, None] | far_xy[None, :]
        )
        guess_score = self.score(guess, pts)
        return (
            MatchResult(pose=best_pose, score=best_score, guess_score=guess_score),
            MatchResult(pose=rival_pose, score=rival_score, guess_score=guess_score),
        )

    def match_top(
        self,
        guess: Pose2D,
        points: NDArray[np.float64],
        k: int,
        window: SearchWindow | None = None,
        apart_steps: int = 3,
    ) -> list[MatchResult]:
        """The ``k`` best candidates, each at least ``apart_steps`` lattice steps from the
        ones before it (non-maximum suppression), best first.

        A coarse lattice on a max-pooled grid over-scores clutter: a speckle field
        looks like a wall to it. Refining several peaks instead of one lets the
        fine grid, which sees the speckles for what they are, pick the real place.
        """
        window = window or self._window
        pts = self._subsample(points)
        scores, positions, headings = self._lattice(guess, pts, window)
        guess_score = self.score(guess, pts)
        mask = np.ones(scores.shape, dtype=bool)
        peaks: list[MatchResult] = []
        for _ in range(k):
            if not mask.any():
                break
            pose, score = self._peak(scores, positions, headings, mask)
            peaks.append(MatchResult(pose=pose, score=score, guess_score=guess_score))
            near_xy = (
                np.abs(positions - [pose.x, pose.y]).max(axis=1)
                < apart_steps * window.xy_step_m - 1e-9
            )
            turned = headings - pose.theta
            near_theta = (
                np.abs(np.arctan2(np.sin(turned), np.cos(turned)))
                < math.radians(apart_steps * window.theta_step_deg) - 1e-9
            )
            mask &= ~(near_theta[:, None] & near_xy[None, :])
        return peaks

    def match_everywhere(
        self,
        points: NDArray[np.float64],
        theta_step_deg: float = 5.0,
        top_k: int = 8,
        pool: int = 1,
        max_range_m: float = 6.0,
    ) -> list[MatchResult]:
        """Every position on the whole grid at every heading, exhaustively, best first.

        For one heading the score of all positions at once is a cross-correlation
        of the score field with the scan's cell offsets, done with FFTs, so the
        cost is a handful of transforms per heading instead of a pose lattice:
        exact over the map where a coarse-to-fine lottery could miss the truth
        (Olson's correlative matching, computed as Cartographer does it in one
        go). Walls attract, known-free floor repels, unknown is silent. The
        winners are ``top_k`` local maxima at least 3 cells or one heading apart;
        their ``score`` is the mean field value under the scan (comparable between
        headings, not to ``match``'s score). ``pool`` mean-pools the field first
        (``pool`` = 2 quarters the transforms; the refine that follows recovers the
        lost resolution; a mean keeps walls and shrinks lone speckle cells, where a
        max would turn a speckle field into a wall), ``max_range_m`` drops far
        returns that only enlarge the transforms.
        """
        # Walls attract, everything else is silent here: the free-floor penalty of the fine
        # field is judged afterwards as "denial", where it cannot drown a true pose whose room
        # has gained a chair since the map was made.
        field = np.maximum(self._score_field(), 0.0)
        if pool > 1:  # mean-pool: a wall keeps half its weight, a lone speckle cell a quarter
            rows0, cols0 = field.shape
            r0, c0 = rows0 // pool, cols0 // pool
            field = field[: r0 * pool, : c0 * pool].reshape(r0, pool, c0, pool).mean(axis=(1, 3))
        rows, cols = field.shape
        res = self._grid.spec.resolution_m * pool
        pts = points[np.isfinite(points).all(axis=1)]
        pts = pts[
            np.hypot(pts[:, 0], pts[:, 1]) <= max_range_m
        ]  # far returns: little discrimination, big transforms
        if len(pts) == 0:
            return []
        reach = int(np.ceil(np.abs(pts).max() / res)) + 1
        shape = (rows + 2 * reach, cols + 2 * reach)
        padded = np.zeros(shape)
        padded[reach : reach + rows, reach : reach + cols] = field
        field_f = np.fft.rfft2(padded)
        headings = np.arange(-180.0, 180.0, theta_step_deg)
        candidates: list[tuple[float, int, int, float]] = []
        for deg in headings:
            theta = math.radians(float(deg))
            c, s = math.cos(theta), math.sin(theta)
            world = pts @ np.array([[c, s], [-s, c]])
            d_col = np.floor(world[:, 0] / res).astype(int) % shape[1]
            d_row = np.floor(world[:, 1] / res).astype(int) % shape[0]
            image = np.zeros(shape)
            np.add.at(image, (d_row, d_col), 1.0)
            corr = np.fft.irfft2(field_f * np.conj(np.fft.rfft2(image)), s=shape)
            scores = corr[reach : reach + rows, reach : reach + cols] / len(pts)
            for _ in range(3):  # a few local maxima per heading, suppressed within 3 cells
                flat = int(np.argmax(scores))
                r, cidx = int(flat // scores.shape[1]), int(flat % scores.shape[1])
                value = float(scores[r, cidx])
                if value <= 0.0:
                    break
                candidates.append((value, r, cidx, theta))
                scores[max(0, r - 3) : r + 4, max(0, cidx - 3) : cidx + 4] = -np.inf
        candidates.sort(key=lambda x: -x[0])
        spec = self._grid.spec
        results: list[MatchResult] = []
        for value, r, cidx, theta in candidates:
            pose = Pose2D(
                spec.x_min_m + (cidx + 0.5) * res, spec.y_min_m + (r + 0.5) * res, wrap_angle(theta)
            )
            if any(
                math.hypot(pose.x - q.pose.x, pose.y - q.pose.y) < 3 * res
                and abs(wrap_angle(pose.theta - q.pose.theta)) < math.radians(theta_step_deg) + 1e-9
                for q in results
            ):
                continue
            # At most two hypotheses per place (a spot and its turned twin): a clutter field
            # scores at every heading and would otherwise fill the list on its own.
            if sum(math.hypot(pose.x - q.pose.x, pose.y - q.pose.y) < 1.0 for q in results) >= 2:
                continue
            results.append(MatchResult(pose=pose, score=value, guess_score=0.0))
            if len(results) >= top_k:
                break
        return results

    def _lattice(
        self, guess: Pose2D, pts: NDArray[np.float64], window: SearchWindow
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Score every candidate of ``window`` around ``guess``.

        Returns (T, P) scores over headings x positions, the (P, 2) positions and
        the (T,) headings. A slight penalty on distance from the guess breaks
        ties on flat score surfaces toward it.
        """
        n = round(window.xy_m / window.xy_step_m)
        offsets = np.arange(-n, n + 1) * window.xy_step_m
        xy_offsets = np.array([(dx, dy) for dx in offsets for dy in offsets])
        m = round(window.theta_deg / window.theta_step_deg)
        theta_offsets = np.radians(np.arange(-m, m + 1) * window.theta_step_deg)
        xy_penalty = 1e-3 * np.abs(xy_offsets).sum(axis=1) / window.xy_step_m
        theta_penalty = 1e-3 * np.abs(theta_offsets) / math.radians(window.theta_step_deg)
        positions = np.array([[guess.x, guess.y]]) + xy_offsets
        scores = np.empty((len(theta_offsets), len(positions)))
        for k, dtheta in enumerate(theta_offsets):
            scores[k] = (
                self._scores(guess.theta + dtheta, pts, positions) - xy_penalty - theta_penalty[k]
            )
        return scores, positions, guess.theta + theta_offsets

    @staticmethod
    def _peak(
        scores: NDArray[np.float64],
        positions: NDArray[np.float64],
        headings: NDArray[np.float64],
        mask: NDArray[np.bool_] | None = None,
    ) -> tuple[Pose2D, float]:
        """The highest-scoring candidate (among those where ``mask`` is True) and its score."""
        field = scores if mask is None else np.where(mask, scores, -np.inf)
        k, i = np.unravel_index(int(np.argmax(field)), field.shape)
        pose = Pose2D(
            float(positions[i, 0]), float(positions[i, 1]), wrap_angle(float(headings[k]))
        )
        return pose, float(field[k, i])
