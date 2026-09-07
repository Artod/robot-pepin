"""Localisation on a saved map: odometry predicts, the lidar corrects.

The map is frozen, so every correction is absolute — errors do not compound
the way they do while mapping. When a scan fits the map poorly (an open
door, furniture that moved, a bad match) the odometry prediction is kept and
the lidar is asked again on the next scan.
"""

from __future__ import annotations

import logging
import math

import numpy as np
from numpy.typing import NDArray

from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import (
    CorrelativeMatcher,
    MatchResult,
    SearchWindow,
    apply_motion,
    relative_motion,
)

logger = logging.getLogger(__name__)

# A rival global fix is a real twin only if the map denies it no more than this
# share of the scan beyond the winner (about four beams of a 180-beam scan).
TWIN_DENIAL_MARGIN = 0.02
TWIN_MARGIN = 0.10  # a runner-up within 10% of the winner's field score explains the scan alike
DENIAL_WEIGHT = 0.5  # of the denied share, charged against the field score of a global candidate
# Lost for this many scans in a row (about two seconds) and the local recovery has not
# helped: search the whole map again, and again every so many scans after that.
GLOBAL_RETRY_EVERY = 20
# The whole-map search runs on the grid pooled this many times coarser (0.05 m -> 0.2 m),
# with headings this far apart: at 3 m a 5-degree error moves a point one coarse cell.
GLOBAL_POOL_FACTOR = 4
GLOBAL_THETA_STEP_DEG = 5.0
GLOBAL_PEAKS = 6  # exhaustive peaks refined on the fine grid before the winner is chosen
GLOBAL_FFT_THETA_STEP_DEG = 9.0  # exhaustive pass heading step; refine's +-6 deg window closes it
GLOBAL_FFT_POOL = (
    2  # exhaustive pass on a 0.1 m grid; pool 3 let a speckle field crowd the truth out
)
# A global fix must place most of the scan on cells the map knows: a pose outside the
# walls, with a sliver of the scan on them and the rest in the unknown, is no fix at all.
GLOBAL_MIN_KNOWN = 0.6


def pooled(grid: OccupancyGrid, factor: int) -> OccupancyGrid:
    """The grid ``factor`` times coarser, every cell the max log-odds of its block.

    Max-pooling makes a coarse score an upper bound of the fine one (Olson's
    multi-resolution correlative matching): a pose whose true basin falls between
    two coarse lattice points still scores high instead of vanishing between them.
    """
    spec = grid.spec
    rows, cols = spec.shape
    r, c = rows // factor, cols // factor
    coarse = OccupancyGrid(
        GridSpec(
            spec.resolution_m * factor,
            spec.x_min_m,
            spec.y_min_m,
            c * spec.resolution_m * factor,
            r * spec.resolution_m * factor,
        )
    )
    blocks = grid.log_odds[: r * factor, : c * factor].reshape(r, factor, c, factor)
    coarse.log_odds[:] = blocks.max(axis=(1, 3))
    return coarse


class Localizer:
    """Tracks the robot pose on a fixed occupancy grid from odometry and lidar scans.

    Tracking takes the best match inside a small window around the odometry
    prediction unconditionally — exactly what kept the SLAM front end straight.
    The inlier fraction is only a health signal: when it stays very low for
    several scans the robot is declared lost and a wide search is allowed to
    relocate it, but only if that far hypothesis explains the scan clearly
    better than the local one (flats are full of look-alike rooms).
    """

    def __init__(
        self,
        grid: OccupancyGrid,
        initial: Pose2D,
        window: SearchWindow | None = None,
        lost_below: float = 0.25,
        lost_after: int = 5,
        recovery_min_inliers: float = 0.6,
        relocalise_min_inliers: float = 0.5,  # mid-run: this flat's true pose scores 0.5-0.65
        recovery_margin: float = 0.08,
        recovery: SearchWindow | None = None,
        min_points: int = 50,
        max_points: int = 200,  # beams per match; what caps the cost of one scan
        global_retry: bool = True,  # False: a lost tracker searches locally only
    ) -> None:
        self._grid = grid
        self._matcher = CorrelativeMatcher(grid, max_points=max_points)
        self._global_retry = global_retry
        self._coarse_matcher: CorrelativeMatcher | None = None
        self._coarse_version = -1
        self._window = window or SearchWindow()
        self._recovery = recovery or SearchWindow(0.2, 0.02, 10.0, 0.5)
        self._lost_below = lost_below
        self._lost_after = lost_after
        self._recovery_min_inliers = recovery_min_inliers  # for the first fix, before moving
        self._relocalise_min_inliers = relocalise_min_inliers  # for a fix while lost mid-run
        self._recovery_margin = recovery_margin
        self._min_points = min_points  # a degenerate scan must not move the pose
        self.pose = initial
        self.confidence = 0.0  # inlier fraction of the current pose
        self.weak_scans = 0  # consecutive scans with confidence below lost_below
        self._drift = Pose2D()  # |motion| accumulated while weak
        self._last_odom: Pose2D | None = None

    @property
    def lost(self) -> bool:
        """True once several scans in a row fit poorly: the wide recovery search is active."""
        return self.weak_scans >= self._lost_after

    def _recovery_window(self) -> SearchWindow:
        """Recovery window sized to the uncertainty: grows with motion since the last good fit.

        Odometry over-counts turns by tens of percent on carpet, so after a
        weak stretch the true pose can be far outside the tracking window.
        Ranges scale with the accumulated |motion| (capped), steps scale with
        the ranges so the candidate count stays constant; a fine pass follows.
        """
        base = self._recovery
        xy = min(1.5, base.xy_m + 1.5 * math.hypot(self._drift.x, self._drift.y))
        theta = min(90.0, base.theta_deg + 1.5 * math.degrees(self._drift.theta))
        return SearchWindow(
            xy_m=xy,
            xy_step_m=base.xy_step_m * xy / base.xy_m,
            theta_deg=theta,
            theta_step_deg=base.theta_step_deg * theta / base.theta_deg,
        )

    def initialize(
        self, points: NDArray[np.float64], window: SearchWindow, global_fallback: bool = True
    ) -> float:
        """Search ``window`` around the start pose once, before moving, and adopt the best fit.

        A robot placed on its mark by hand is off by decimetres and degrees,
        beyond the tracking window; without this the whole run is offset. If
        even the wide window fits poorly and ``global_fallback`` is on, the
        whole map is searched (the robot may have been put down anywhere).
        Returns the inlier fraction of the adopted pose. A poor or ambiguous fit
        (below ``recovery_min_inliers``) keeps the given start and marks the
        localiser lost at once, so the navigator holds instead of driving off
        an unconfirmed guess.
        """
        coarse = self._matcher.match(self.pose, points, window)
        fine = self._matcher.match(coarse.pose, points, self._window)
        confidence = self._matcher.inlier_fraction(fine.pose, points)
        if confidence < self._recovery_min_inliers and global_fallback:
            logger.info("start fits poorly (inliers %.2f); searching the whole map", confidence)
            fine, confidence = self.global_search(points, prior=self.pose)
        if confidence >= self._recovery_min_inliers:
            logger.info("initial fix %s, inliers %.2f", fine.pose, confidence)
            self.pose = fine.pose
        else:
            logger.warning("initial fix rejected (inliers %.2f); keeping the start", confidence)
            self.weak_scans = self._lost_after
        self.confidence = confidence
        return confidence

    def global_search(
        self,
        points: NDArray[np.float64],
        theta_step_deg: float = GLOBAL_THETA_STEP_DEG,
        thin_to: int = 120,
        prior: Pose2D | None = None,
    ) -> tuple[MatchResult, float]:
        """Coarse-to-fine search over the whole grid: any position, any heading, once.

        ``prior`` is where the robot was believed to be; when the scan fits two
        places alike, the twin nearer to it wins instead
        of refusing (2026-09-06: the base itself was refused for a 0.53 look-alike
        six metres away while the belief sat on the base).

        The coarse pass uses a 0.2 m / 15 degree lattice on a thinned scan
        (well under a second); its best peaks (``GLOBAL_PEAKS``, well apart)
        are each refined with a medium and then the tracking window, and the
        fine results are ranked: a max-pooled grid over-scores speckle fields,
        so the coarse winner is not trusted before the fine grid has spoken.
        When the runner-up explains the
        scan nearly as well (within ``recovery_margin``) *and* the map denies
        it no more than the winner (no extra points on known-free floor), the
        scan fits two places — a symmetric room, look-alike rooms — and the fix
        is refused with confidence 0: "tell me where I am" beats driving off
        from the wrong twin.
        """
        spec = self._grid.spec
        centre = Pose2D(spec.x_min_m + spec.width_m / 2, spec.y_min_m + spec.height_m / 2, 0.0)
        thinned = points[:: max(1, len(points) // thin_to)]
        whole_map = SearchWindow(
            xy_m=max(spec.width_m, spec.height_m) / 2,
            xy_step_m=spec.resolution_m * GLOBAL_POOL_FACTOR,
            theta_deg=180.0,
            theta_step_deg=theta_step_deg,
        )
        # Exhaustive over the whole grid at every heading (FFT correlation on the fine grid);
        # the pooled lattice it replaces once ranked the truth 5th and missed it on real maps.
        peaks = self._matcher.match_everywhere(
            points,
            theta_step_deg=GLOBAL_FFT_THETA_STEP_DEG,
            top_k=GLOBAL_PEAKS,
            pool=GLOBAL_FFT_POOL,
        )
        if not peaks:
            peaks = self._coarse().match_top(centre, thinned, GLOBAL_PEAKS, whole_map)
        # Ranked by the field score (how exactly the scan sits on the walls): the inlier
        # fraction saturates one cell off a wall and let a look-alike 5 m away tie with the
        # truth at the cluttered base (0.56 vs 0.58) where the field score said 1.69 vs 2.16.
        refined = sorted(
            (self.refine(peak.pose, points) for peak in peaks),
            key=lambda r: -self._rank(r[0].pose, points),
        )
        distinct: list[tuple[MatchResult, float]] = []  # several peaks refine into one basin
        for candidate in refined:
            if not any(
                math.hypot(
                    candidate[0].pose.x - kept[0].pose.x, candidate[0].pose.y - kept[0].pose.y
                )
                < 0.3
                and abs(wrap_angle(candidate[0].pose.theta - kept[0].pose.theta))
                < math.radians(15.0)
                for kept in distinct
            ):
                distinct.append(candidate)
        best, best_confidence = distinct[0]
        second, second_confidence = distinct[1] if len(distinct) > 1 else distinct[0]
        apart = math.hypot(best.pose.x - second.pose.x, best.pose.y - second.pose.y) > 0.5 or abs(
            wrap_angle(best.pose.theta - second.pose.theta)
        ) > math.radians(30.0)
        best_sharp = self._rank(best.pose, points)
        second_sharp = self._rank(second.pose, points)
        explains_alike = second_sharp >= best_sharp * (1.0 - TWIN_MARGIN)
        denied = self._matcher.contradiction_fraction(second.pose, points)
        denied_best = self._matcher.contradiction_fraction(best.pose, points)
        if apart and explains_alike and denied <= denied_best + TWIN_DENIAL_MARGIN:
            if prior is not None:
                near_best = math.hypot(best.pose.x - prior.x, best.pose.y - prior.y)
                near_second = math.hypot(second.pose.x - prior.x, second.pose.y - prior.y)
                # The twin nearer the previous belief is the better bet; a wrong pick shows up as
                # a poor fit within seconds and is searched again, a refusal helps nobody.
                if True:
                    chosen, chosen_confidence = (
                        (best, best_confidence)
                        if near_best <= near_second
                        else (second, second_confidence)
                    )
                    logger.info(
                        "twins (%.2f vs %.2f); kept %s, %.1f m from the prior",
                        best_confidence,
                        second_confidence,
                        chosen.pose,
                        min(near_best, near_second),
                    )
                    return chosen, chosen_confidence
            logger.warning(
                "the scan fits two places alike: %s (inliers %.2f, denied %.2f) and %s "
                "(%.2f, %.2f); no fix without a start pose",
                best.pose, best_confidence, denied_best, second.pose, second_confidence, denied,
            )  # fmt: skip
            return best, 0.0
        logger.info(
            "global fix %s (inliers %.2f, denied %.2f); runner-up %s (%.2f, %.2f)",
            best.pose, best_confidence, denied_best, second.pose, second_confidence, denied,
        )  # fmt: skip
        return best, best_confidence

    def refine(self, pose: Pose2D, points: NDArray[np.float64]) -> tuple[MatchResult, float]:
        """A coarse candidate sharpened with a medium and then the tracking window."""
        medium = SearchWindow(xy_m=0.2, xy_step_m=0.05, theta_deg=6.0, theta_step_deg=1.5)
        refined = self._matcher.match(pose, points, medium)
        fine = self._matcher.match(refined.pose, points, self._window)
        confidence = self._matcher.inlier_fraction(fine.pose, points, min_known=GLOBAL_MIN_KNOWN)
        return fine, confidence

    def _rank(self, pose: Pose2D, points: NDArray[np.float64]) -> float:
        """How a global candidate is ranked: how exactly the scan sits on walls, minus a mild
        charge for points the map puts on open floor. Mild on purpose: at a cluttered spot the
        true pose denies 40% of the scan (furniture moved since the map), a look-alike 30%, and
        a heavy charge would crown the look-alike; the sharpness term must stay decisive."""
        return self._matcher.field_score(pose, points) - DENIAL_WEIGHT * (
            self._matcher.contradiction_fraction(pose, points)
        )

    def _coarse(self) -> CorrelativeMatcher:
        """Matcher on the pooled grid for the whole-map search; rebuilt when the map grows."""
        if self._coarse_matcher is None or self._coarse_version != self._grid.version:
            self._coarse_matcher = CorrelativeMatcher(pooled(self._grid, GLOBAL_POOL_FACTOR))
            self._coarse_version = self._grid.version
        return self._coarse_matcher

    def predict(self, odom: Pose2D) -> Pose2D:
        """Advance the pose by odometry alone (between scans); the next scan corrects it."""
        motion = Pose2D() if self._last_odom is None else relative_motion(self._last_odom, odom)
        self._last_odom = odom
        self._drift = Pose2D(
            self._drift.x + abs(motion.x),
            self._drift.y + abs(motion.y),
            self._drift.theta + abs(motion.theta),
        )
        self.pose = apply_motion(self.pose, motion)
        return self.pose

    def update(self, odom: Pose2D, points: NDArray[np.float64]) -> Pose2D:
        """Advance by the odometry step since the last call, then correct with the scan."""
        motion = Pose2D() if self._last_odom is None else relative_motion(self._last_odom, odom)
        self._last_odom = odom
        self._drift = Pose2D(  # motion since the last good fit; reset below when the scan fits
            self._drift.x + abs(motion.x),
            self._drift.y + abs(motion.y),
            self._drift.theta + abs(motion.theta),
        )
        prediction = apply_motion(self.pose, motion)
        if len(points) < self._min_points:
            self.pose = prediction
            self.confidence = 0.0
            self.weak_scans += 1
            return self.pose
        local = self._matcher.match_around(prediction, points, motion, self._window)
        pose, confidence = local.pose, self._matcher.inlier_fraction(local.pose, points)

        if self.lost:
            coarse = self._matcher.match(prediction, points, self._recovery_window())
            far = self._matcher.match(coarse.pose, points, self._window)
            far_confidence = self._matcher.inlier_fraction(far.pose, points)
            if (
                far_confidence >= self._relocalise_min_inliers
                and far_confidence >= confidence + self._recovery_margin
            ):
                logger.info("relocalised: inliers %.2f vs %.2f locally", far_confidence, confidence)
                pose, confidence = far.pose, far_confidence
            elif (
                self._global_retry
                and (self.weak_scans - self._lost_after) % GLOBAL_RETRY_EVERY == 0
            ):
                # A slipped wheel, a push by hand: odometry lied by more than any window
                # sized to it. The robot stands still while lost, so the whole-map search
                # (a few hundred ms) is affordable here; a twin refuses itself (0.0).
                anywhere, anywhere_confidence = self.global_search(points)
                if (
                    anywhere_confidence >= self._relocalise_min_inliers
                    and anywhere_confidence >= confidence + self._recovery_margin
                ):
                    logger.info(
                        "relocalised globally at %s: inliers %.2f vs %.2f locally",
                        anywhere.pose, anywhere_confidence, confidence,
                    )  # fmt: skip
                    pose, confidence = anywhere.pose, anywhere_confidence

        self.pose = pose
        self.confidence = confidence
        if confidence < self._lost_below:
            self.weak_scans += 1
        else:
            self.weak_scans = 0
            self._drift = Pose2D()
        return self.pose
