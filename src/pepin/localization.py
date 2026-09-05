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

from pepin.mapping import OccupancyGrid
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
        recovery_margin: float = 0.15,
        recovery: SearchWindow | None = None,
        min_points: int = 50,
    ) -> None:
        self._grid = grid
        self._matcher = CorrelativeMatcher(grid)
        self._window = window or SearchWindow()
        self._recovery = recovery or SearchWindow(0.2, 0.02, 10.0, 0.5)
        self._lost_below = lost_below
        self._lost_after = lost_after
        self._recovery_min_inliers = recovery_min_inliers
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
            fine, confidence = self._global_search(points)
        if confidence >= self._recovery_min_inliers:
            logger.info("initial fix %s, inliers %.2f", fine.pose, confidence)
            self.pose = fine.pose
        else:
            logger.warning("initial fix rejected (inliers %.2f); keeping the start", confidence)
            self.weak_scans = self._lost_after
        self.confidence = confidence
        return confidence

    def _global_search(self, points: NDArray[np.float64]) -> tuple[MatchResult, float]:
        """Coarse-to-fine search over the whole grid: any position, any heading, once.

        The coarse pass uses a 0.2 m / 15 degree lattice on a thinned scan
        (well under a second); its two best peaks are each refined with a
        medium and then the tracking window. When the runner-up explains the
        scan nearly as well (within ``recovery_margin``) *and* the map denies
        it no more than the winner (no extra points on known-free floor), the
        scan fits two places — a symmetric room, look-alike rooms — and the fix
        is refused with confidence 0: "tell me where I am" beats driving off
        from the wrong twin.
        """
        spec = self._grid.spec
        centre = Pose2D(spec.x_min_m + spec.width_m / 2, spec.y_min_m + spec.height_m / 2, 0.0)
        thinned = points[:: max(1, len(points) // 120)]
        whole_map = SearchWindow(
            xy_m=max(spec.width_m, spec.height_m) / 2,
            xy_step_m=0.2,
            theta_deg=180.0,
            theta_step_deg=15.0,
        )
        coarse, rival = self._matcher.match_two(centre, thinned, whole_map)
        best, best_confidence = self._refine(coarse.pose, points)
        second, second_confidence = self._refine(rival.pose, points)
        apart = math.hypot(best.pose.x - second.pose.x, best.pose.y - second.pose.y) > 0.5 or abs(
            wrap_angle(best.pose.theta - second.pose.theta)
        ) > math.radians(30.0)
        explains_alike = second_confidence >= best_confidence - self._recovery_margin
        denied = self._matcher.contradiction_fraction(second.pose, points)
        denied_best = self._matcher.contradiction_fraction(best.pose, points)
        if apart and explains_alike and denied <= denied_best + TWIN_DENIAL_MARGIN:
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

    def _refine(self, pose: Pose2D, points: NDArray[np.float64]) -> tuple[MatchResult, float]:
        """A coarse candidate sharpened with a medium and then the tracking window."""
        medium = SearchWindow(xy_m=0.3, xy_step_m=0.05, theta_deg=15.0, theta_step_deg=2.0)
        refined = self._matcher.match(pose, points, medium)
        fine = self._matcher.match(refined.pose, points, self._window)
        return fine, self._matcher.inlier_fraction(fine.pose, points)

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
                far_confidence >= self._recovery_min_inliers
                and far_confidence >= confidence + self._recovery_margin
            ):
                logger.info("relocalised: inliers %.2f vs %.2f locally", far_confidence, confidence)
                pose, confidence = far.pose, far_confidence

        self.pose = pose
        self.confidence = confidence
        if confidence < self._lost_below:
            self.weak_scans += 1
        else:
            self.weak_scans = 0
            self._drift = Pose2D()
        return self.pose
