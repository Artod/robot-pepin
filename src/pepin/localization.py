"""Localisation on a saved map: odometry predicts, the sensors correct.

The map is frozen, so every correction is absolute — errors do not compound
the way they do while mapping. When a scan fits the map poorly (an open
door, furniture that moved, a bad match) the odometry prediction is kept and
the sensor is asked again on the next scan.

Any scan-shaped evidence corrects: the lidar's revolution, the camera's virtual scan, the
floor-contact scan (:mod:`pepin.sources`). Each enabled source is matched separately against
the same map with the same machinery, each match carries a covariance read off its score
surface, and the matches are fused by their information (:mod:`pepin.fusion`) before the pose
is corrected — so lidar-only, camera-only and both go through one code path.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pepin.dynamic import StaticMask, voting_mask
from pepin.fusion import PoseMeasurement, at_edge, covariance_from_score_surface, fuse
from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import (
    CorrelativeMatcher,
    MatchResult,
    SearchWindow,
    apply_motion,
    relative_motion,
)
from pepin.sources import LIDAR, ScanObservation, ScanSource, SourceRegistry

__all__ = ["Localizer", "Running", "ScanObservation", "TrackStats", "pooled"]

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


@dataclass
class Running:
    """Count, mean and extremes of a stream of numbers, for one report line."""

    n: int = 0
    total: float = 0.0
    lo: float = math.inf
    hi: float = -math.inf

    def add(self, value: float) -> None:
        """Count ``value`` in."""
        self.n += 1
        self.total += value
        self.lo, self.hi = min(self.lo, value), max(self.hi, value)

    @property
    def mean(self) -> float:
        """The mean so far (NaN when nothing was counted)."""
        return self.total / self.n if self.n else float("nan")

    def text(self, scale: float = 1.0, digits: int = 2) -> str:
        """``mean/min/max`` scaled, or ``-`` when empty."""
        if not self.n:
            return "-"
        mean, lo, hi = (v * scale for v in (self.mean, self.lo, self.hi))
        return f"{mean:.{digits}f}/{lo:.{digits}f}/{hi:.{digits}f}"


@dataclass
class TrackStats:
    """What the tracker did since the last report: every match counted by what happened to it.

    ``released - rested - skipped`` at the node must equal ``matched + thin`` here, so a scan that
    went nowhere shows up as a hole in this arithmetic instead of vanishing.
    """

    matched: int = 0  # scans matched (whatever the gain)
    thin: int = 0  # scans with too few returns to match: odometry only
    rest_locked: int = 0  # matches blended under the rest lock
    carries: int = 0  # rest matches taken whole: two in a row agreed on a new pose (a carry)
    weak: int = 0  # matches whose fit was below lost_below
    lost: int = 0  # matches made while lost (the recovery search ran)
    silenced_scans: int = 0  # matches with a vote mask in force
    silenced_points: int = 0  # returns the mask kept out of the score, in total
    rest_dt_s: Running = field(default_factory=Running)  # seconds between rest matches
    rest_gain: Running = field(default_factory=Running)  # the gain the lock used per match
    fit: Running = field(default_factory=Running)  # confidence at the matched pose
    published_fit: Running = field(default_factory=Running)  # fit at the pose actually published
    step_xy_m: float = 0.0  # the largest position correction one match applied (map->odom's step)
    step_deg: float = 0.0  # the largest heading correction one match applied
    source_fit: dict[str, Running] = field(default_factory=dict)  # fit per source, at its match
    fused: int = 0  # updates whose correction was fused from more than one source
    rejected: int = 0  # source measurements a fusion left out for disagreeing with the surest
    bound: int = 0  # updates the anchor's match sat on the window's edge and corrected alone

    def summary(self) -> str:
        """One log line, mean/min/max where a distribution matters; the per-source fits and
        the fusion count only once a second source has spoken."""
        line = (
            f"matched {self.matched} (thin {self.thin}), rest-locked {self.rest_locked} "
            f"(dt {self.rest_dt_s.text(digits=1)} s, gain {self.rest_gain.text()}), "
            f"carries {self.carries}, weak {self.weak}, lost {self.lost}, "
            f"silenced {self.silenced_points} returns over {self.silenced_scans} scans, "
            f"fit {self.fit.text()} at the match / {self.published_fit.text()} published, "
            f"max step {self.step_xy_m * 100:.1f} cm / {self.step_deg:.2f} deg"
        )
        if set(self.source_fit) - {LIDAR}:
            per_source = ", ".join(
                f"{name} {stat.text()} ({stat.n})" for name, stat in self.source_fit.items()
            )
            line += (
                f"; sources {per_source}; fused {self.fused}, rejected {self.rejected}, "
                f"anchor bound {self.bound}"
            )
        return line


CARRY_MIN_GAIN = 0.08  # a carry moves the WHOLE scan onto the map: the field score at the matched
# pose beats the held pose by this much (real carries on the tapes: >= +0.115; a person standing by
# the lidar or a chair pushed against the cart: <= +0.043 — they agree twice too, but the map fits
# them no better at the new pose; a review probe, 2026-09-11)


class Localizer:
    """Tracks the robot pose on a fixed occupancy grid from odometry and scans.

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
        correction_gain: float = 1.0,  # share of each match's residual applied (see ``_blend``)
        jump_m: float = 0.15,  # a residual past this is not noise: taken whole
        jump_deg: float = 6.0,
        interpolate: bool = True,  # answer between the candidates, not only on them
        rest_lock: bool = True,  # hold the pose while the cart stands still (see ``update``)
        rest_gain: float = 0.05,  # share per match when the caller times nothing (see ``_blend``)
        rest_tau_s: float = 6.0,  # time constant of the rest lock when the caller passes ``dt_s``
        explained_vote: bool = True,  # returns the static map cannot explain do not score
        carry_m: float = 0.06,  # a rest residual past this, twice in a row, is a carry, not noise
        carry_deg: float = 4.0,
        sources: SourceRegistry | None = None,  # which sensors correct; the lidar alone by default
        fusion: bool = True,  # off: the widest enabled source corrects alone, the rest only report
    ) -> None:
        self._grid = grid
        # The tracker wants a continuous correction: quantised to the search step it corrects the
        # heading in 1.5 degree jumps and the robot weaves. A pose graph selects, and does not.
        self._matcher = CorrelativeMatcher(grid, max_points=max_points, interpolate=interpolate)
        self._global_retry = global_retry
        self._correction_gain = correction_gain
        self._jump_m = jump_m
        self._jump_deg = jump_deg
        # The live switches: the node's parameter callback writes them between two scans.
        self.rest_lock = rest_lock
        self.rest_gain = rest_gain
        self.rest_tau_s = rest_tau_s
        self.explained_vote = explained_vote
        self.sources = sources if sources is not None else SourceRegistry()
        self.fusion = fusion
        # The last update, source by source: every source's word, the one they were fused into
        # (the anchor's own when it stood alone or was a bound), who anchored, and the pose it
        # all corrected from — what /localization/sources shows (``sources_report``).
        self.measurements: list[PoseMeasurement] = []
        self.fused: PoseMeasurement | None = None
        self.anchor: str | None = None
        self.prediction = initial
        # The carry thresholds are below the jump ones on purpose: the tracking window caps a
        # residual at its own size (9 cm on the board), so a 15 cm jump can never be seen at rest,
        # while two matches of a standing cart agreeing beyond 6 cm / 4 deg never happened on the
        # rest-locked tapes (0182, 0183, 0193: p99 of the rest residual 5 cm / 2.8 deg).
        self._carry_m = carry_m
        self._carry_deg = carry_deg
        self._rest_hint: Pose2D | None = None  # a rest match beyond the carry thresholds, once
        self.stats = TrackStats()
        self.published_fit = 0.0  # inlier fraction at the pose ``update`` returned
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

    def settings(self) -> str:
        """The live switches and constants as one phrase, for the report line."""
        return (
            f"rest_lock {'on' if self.rest_lock else 'off'}, "
            f"explained_vote {'on' if self.explained_vote else 'off'}, "
            f"rest_tau_s {self.rest_tau_s:.1f}, gain {self._correction_gain:.2f}, "
            f"carry {self._carry_m * 100:.0f} cm / {self._carry_deg:.0f} deg, "
            f"sources {','.join(self.sources.enabled) or 'none'}, "
            f"fusion {'on' if self.fusion else 'off'}"
        )

    def report(self) -> TrackStats:
        """The counters since the previous report, which are reset."""
        stats, self.stats = self.stats, TrackStats()
        return stats

    def switch(self, name: str, value: Any) -> None:
        """A live switch by its flag's name, between two updates: ``sources`` (the names of
        the sensors that correct) goes to the roster, every other name is the attribute it
        names (``rest_lock``, ``fusion``, ...); ``ValueError`` for a name that is neither."""
        if name == "sources":
            self.sources.enable(value)
        elif hasattr(self, name) and not name.startswith("_"):
            setattr(self, name, value)
        else:
            raise ValueError(f"{name}: not a switch of the tracker")

    def sources_report(self, now: float) -> dict[str, Any]:
        """Every source's word on the last update, ready for JSON: the anchor, the sources
        fused, the ones a fusion rejected, and per source on the roster its health at ``now``
        (``off`` when the flag has it off) with, when it measured, its fit, the correction it
        proposed from the prediction (``delta``: cm, cm, deg), the roots of its covariance
        diagonal (``sigma``: cm, cm, deg) and whether its match was a bound (``edge``)."""
        fused = self.fused
        report: dict[str, Any] = {
            "anchor": self.anchor,
            "fused": None if fused is None else fused.source,
            "rejected": [] if fused is None else list(fused.rejected),
            "fit": round(self.confidence, 3),
            "sources": {},
        }
        by_name = {m.source: m for m in self.measurements}
        for name in self.sources.names:
            health = self.sources.health(name).text(now) if self.sources.is_enabled(name) else "off"
            entry: dict[str, Any] = {"health": health}
            measurement = by_name.get(name)
            if measurement is not None:
                p = self.prediction
                sx, sy, st = measurement.sigmas
                entry.update(
                    fit=round(measurement.fit, 3),
                    delta=[
                        round((measurement.x - p.x) * 100.0, 2),
                        round((measurement.y - p.y) * 100.0, 2),
                        round(math.degrees(wrap_angle(measurement.yaw - p.theta)), 2),
                    ],
                    sigma=[round(sx * 100.0, 2), round(sy * 100.0, 2), round(math.degrees(st), 2)],
                    edge=measurement.edge,
                )
            report["sources"][name] = entry
        return report

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
        refuse_twins: bool = True,
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
            if not refuse_twins:  # a lost robot prefers the better-ranked guess to no guess at all
                logger.info(
                    "twins (%.2f vs %.2f); taking the better-ranked %s, the tracker will tell",
                    best_confidence, second_confidence, best.pose,
                )  # fmt: skip
                return best, best_confidence
            if prior is not None:
                near_best = math.hypot(best.pose.x - prior.x, best.pose.y - prior.y)
                near_second = math.hypot(second.pose.x - prior.x, second.pose.y - prior.y)
                # The twin nearer the previous belief is the better bet; a wrong pick shows up as
                # a poor fit within seconds and is searched again, a refusal helps nobody.
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

    def adopt(self, pose: Pose2D, confidence: float) -> None:
        """Take ``pose`` as the truth: a re-seed from a whole-map search, or the operator's word.

        The one door for writing the belief from outside. It clears everything the old belief
        implied — the lost counter and the drift that widens the recovery window — because a
        re-seed that left the drift behind kept searching as if the robot were still lost.
        """
        self.pose = pose
        self.confidence = self.published_fit = confidence
        self.weak_scans = 0
        self._drift = Pose2D()
        self._rest_hint = None
        self._last_odom = None  # the next update measures its step from the next reading

    def _voting(
        self, points: NDArray[np.float64], vote: NDArray[np.bool_] | None, min_points: int
    ) -> NDArray[np.float64]:
        """The returns the matcher is allowed to score: ``points[vote]``, or all of them.

        A mask that would leave the match with fewer returns than a pose can be fixed from
        (``min_points``) is ignored — a thin scan is worse than a scan with some furniture in it.
        """
        if vote is None:
            return points
        kept: NDArray[np.float64] = points[vote]
        return kept if len(kept) >= min_points else points

    def _min_points_for(self, source: ScanSource) -> int:
        """Fewer returns than this fix no pose: the constructor's floor for the lidar (the
        callers that size it keep their word), the roster's for every other source."""
        return self._min_points if source.name == LIDAR else source.min_points

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

    def _within(self, a: Pose2D, b: Pose2D, xy_m: float, deg: float) -> bool:
        """True when the two poses differ by at most ``xy_m`` in position and ``deg`` in heading."""
        return math.hypot(a.x - b.x, a.y - b.y) <= xy_m and abs(
            wrap_angle(a.theta - b.theta)
        ) <= math.radians(deg)

    def _carried(self, matched: Pose2D, prediction: Pose2D, gain: float) -> bool:
        """Was the standing cart carried? Two rest matches in a row agreeing on a pose beyond
        ``carry_m`` / ``carry_deg`` from the held one, AND the whole scan fitting the map better
        at that pose by CARRY_MIN_GAIN, say yes. A single one is match noise and only becomes the
        hint the next match is held against; an intruder — a person by the lidar, a chair pushed
        against the cart — agrees with itself twice but gains the map nothing at the new pose."""
        hint = self._rest_hint
        agrees = hint is not None and self._within(
            matched, hint, self._carry_m / 2, self._carry_deg / 2
        )
        beyond = not self._within(matched, prediction, self._carry_m, self._carry_deg)
        self._rest_hint = matched if beyond and not agrees else None
        return agrees and gain >= CARRY_MIN_GAIN

    def _rest_gain_for(self, dt_s: float | None) -> float:
        """The lock's share of a residual: per second when timed, per match when not; never
        above the driving gain, so one match after a long silence cannot be taken whole."""
        gain = (
            self.rest_gain
            if dt_s is None
            else 1.0 - math.exp(-max(dt_s, 0.0) / max(self.rest_tau_s, 1e-3))
        )
        return min(gain, self._correction_gain)

    def _blend(
        self,
        prediction: Pose2D,
        matched: Pose2D,
        at_rest: bool = False,
        dt_s: float | None = None,
        lost: bool = False,
        carry_gain: float = 0.0,
    ) -> tuple[Pose2D, float]:
        """Move from the prediction toward the match by ``correction_gain`` of the way; returns
        the pose and the gain used.

        One match is a noisy measurement (about 1 degree and 1 cm of noise with 120 beams on a
        5 cm map), and the pose it corrects is published as a transform the controller steers by:
        at full gain that noise reaches the wheels ten times a second and the robot weaves. The
        residual being corrected is odometry error, which grows slowly, so a fraction of it per
        scan converges in a few tenths of a second and filters the noise. Only the residual is
        damped — the motion itself is already in the prediction, so nothing lags behind the robot.
        A residual too large to be noise (a push, a carry, a re-seed) is taken whole.

        ``at_rest`` swaps that gain for a slow average: a standing cart has no odometry error to
        correct, so every scan's residual there is match noise, and taking half of each one is
        what moves the published heading inside a several-degree band while nothing moves.
        The average is in SECONDS, not in matches: ``dt_s`` (the time since the previous match)
        gives ``gain = 1 - exp(-dt_s / rest_tau_s)``, so the same residual dies with the same
        time constant (``rest_tau_s``, 6 s by default) whether the caller matches at 10 Hz (an
        offline replay) or about once a second (the node, which rests between matches —
        ``timeline.MotionFilter``). A fixed per-match gain cannot: 0.05 per match is 2 s at
        10 Hz and 20 s at 1 Hz, and at 20 s a 6 cm nudge inside the jump window outlives a
        parking manoeuvre. ``dt_s=None`` keeps the old per-match ``rest_gain`` for callers that
        time nothing. Either way the gain is capped at ``correction_gain``: after a long gap
        (a whole-map search, an expired scan) the first match back is one noisy match, not the
        truth, and ``1 - exp(-15 / 6)`` would have taken it nine tenths whole.

        The jump rule is deliberately NOT an escape hatch from the rest lock: at rest the wheels
        and the gyro have both said for half a second that nothing turned, so a residual of six
        degrees is a bad match, not a rotation. A carry is: the cart lifted straight, or skidded
        sideways, ticks no wheel and turns no gyro, and the lock would absorb the new pose with
        its time constant while the stop reflex ran on the old one. So a residual beyond the
        carry thresholds that the NEXT rest match agrees with is taken whole (``_carried``); a
        single one is blended like any rest match. A tracker that is ``lost`` drops the lock and
        takes what the recovery search found, at whatever gain it would use while driving.

        Note what the caller gets back: ``confidence`` is measured at the MATCHED pose, not at
        the blended pose returned here (deliberate: the fit must answer "does the scan fit the
        map here", not "how far has the average crawled"); ``published_fit`` is the same measure
        at the blended pose, so the two diverging is what a carry looks like in the report.
        """
        dx, dy = matched.x - prediction.x, matched.y - prediction.y
        dtheta = wrap_angle(matched.theta - prediction.theta)
        jump = math.hypot(dx, dy) > self._jump_m or abs(dtheta) > math.radians(self._jump_deg)
        stats = self.stats
        if at_rest and self.rest_lock and not lost:
            if self._carried(matched, prediction, carry_gain):
                gain = 1.0
                stats.carries += 1
            else:
                gain = self._rest_gain_for(dt_s)
                stats.rest_locked += 1
                stats.rest_gain.add(gain)
                if dt_s is not None:
                    stats.rest_dt_s.add(dt_s)
        else:
            self._rest_hint = None  # the wheels explain the next residual; a carry needs rest
            gain = 1.0 if jump else self._correction_gain
        stats.step_xy_m = max(stats.step_xy_m, gain * math.hypot(dx, dy))
        stats.step_deg = max(stats.step_deg, gain * abs(math.degrees(dtheta)))
        return Pose2D(
            prediction.x + gain * dx,
            prediction.y + gain * dy,
            wrap_angle(prediction.theta + gain * dtheta),
        ), gain

    def update(
        self,
        odom: Pose2D,
        points: NDArray[np.float64],
        trust_odometry: bool = True,
        at_rest: bool = False,
        vote: NDArray[np.bool_] | None = None,
        dt_s: float | None = None,
        mask: StaticMask | None = None,
    ) -> Pose2D:
        """Advance by the odometry step since the last call, then correct with the lidar scan.

        The single-lidar entry: ``points`` is the lidar's (N, 2) base-frame scan, and the call
        is :meth:`update_from` with that one observation — every argument means what it means
        there. ``vote``, an explicit (N,) mask over ``points``, replaces the static map's.
        """
        return self.update_from(
            odom,
            [ScanObservation(LIDAR, points, vote=vote)],
            trust_odometry=trust_odometry,
            at_rest=at_rest,
            dt_s=dt_s,
            mask=mask,
        )

    def _measure(
        self,
        observation: ScanObservation,
        source: ScanSource,
        prediction: Pose2D,
        motion: Pose2D,
        mask: StaticMask | None,
    ) -> PoseMeasurement:
        """One source's scan matched around the prediction: its pose, its fit on the whole scan,
        the covariance read off the score surface scaled by the source's trust, and whether
        the match sat on the window's edge (``edge``: a bound, the truth lies beyond)."""
        points, vote = observation.points, observation.vote
        if vote is None and mask is not None and self.explained_vote:
            vote = voting_mask(points, prediction, mask, min_points=source.vote_min_points)
        voting = self._voting(points, vote, self._min_points_for(source))
        if vote is not None:
            self.stats.silenced_scans += 1
            self.stats.silenced_points += len(points) - len(voting)
        local, surface = self._matcher.match_around_surface(
            prediction, voting, motion, self._window
        )
        fit = self._matcher.inlier_fraction(local.pose, points)
        covariance = covariance_from_score_surface(surface, fit, trust=source.trust)
        return PoseMeasurement(
            local.pose.x,
            local.pose.y,
            local.pose.theta,
            covariance,
            observation.source,
            observation.stamp,
            fit,
            edge=at_edge(surface),
        )

    def update_from(
        self,
        odom: Pose2D,
        scans: Sequence[ScanObservation],
        *,
        trust_odometry: bool = True,
        at_rest: bool = False,
        dt_s: float | None = None,
        mask: StaticMask | None = None,
    ) -> Pose2D:
        """Advance by the odometry step since the last call, then correct with whatever scans
        arrived: each enabled source's scan is matched separately around the same prediction,
        the matches are fused by their information (or, with ``fusion`` off, the widest
        source's is taken alone) and the fused pose corrects the belief under the same rules
        as ever — the rest lock, the carry, the driving gain, the lost recovery.

        ``trust_odometry=False`` discards the wheel step (slipping wheels): the pose is
        corrected from where it was, and the step is still consumed so it is never re-applied.

        ``at_rest`` says the cart is standing (wheels and gyro agree): with ``rest_lock`` on
        the residual is then averaged in slowly instead of being taken every scan — see
        :meth:`_blend`. The caller passes what the sensors say and nothing else; the switch is
        this object's.

        ``dt_s`` is the time since the PREVIOUS call, in seconds; it makes that average a time
        constant (``rest_tau_s``) instead of a per-match share, so a caller matching once a
        second and one matching ten times a second settle at rest at the same speed. ``None``
        (the default) keeps the old per-match ``rest_gain``; it changes nothing while moving.

        ``mask`` is the static map's own explanation of the scans (``dynamic.StaticMask``): with
        ``explained_vote`` on it is read at the pose this call predicts from, and only the
        returns it explains score a match, which is how returns the map has no wall for (a
        moved chair, a blanket) are kept from pulling the heading. Confidence is always measured
        on the WHOLE scan of the anchor — the widest enabled source, the lidar when it is there
        — at the fused pose, so the fit this reports, the lost counter and the occlusion
        verdict built on them mean exactly what they meant before; ``published_fit`` is the
        same measure at the pose returned. The anchor's bound is taken alone: when its match
        sits on the window's edge the truth lies beyond what was searched, and a fan that sees
        a quarter of the room has look-alikes inside the window where the revolution has none,
        so it may not overrule that (``bound``); the next update searches from the new pose
        and the fan has its say once the anchor is back inside — with the anchor bound, the
        fused tracker steps exactly as the lidar-only one. Scans from sources the flag has
        off, and scans thinner than their source's floor, are ignored; with nothing left to
        match the prediction stands (``thin``). Every source's measurement is kept in
        ``measurements``.
        """
        motion = (
            Pose2D()
            if self._last_odom is None or not trust_odometry
            else relative_motion(self._last_odom, odom)
        )
        self._last_odom = odom
        self._drift = Pose2D(  # motion since the last good fit; reset below when the scan fits
            self._drift.x + abs(motion.x),
            self._drift.y + abs(motion.y),
            self._drift.theta + abs(motion.theta),
        )
        prediction = self.prediction = apply_motion(self.pose, motion)
        stats = self.stats
        usable = [
            (observation, self.sources.source(observation.source))
            for observation in scans
            if self.sources.is_enabled(observation.source)
            and len(observation.points)
            >= self._min_points_for(self.sources.source(observation.source))
        ]
        if not usable:
            self.pose = prediction
            self.confidence = self.published_fit = 0.0
            self.weak_scans += 1
            stats.thin += 1
            self.measurements = []
            self.fused = self.anchor = None
            return self.pose
        # The anchor: the widest fan on offer (the lidar when it is enabled and thick enough).
        # Its whole scan measures the confidence, the recovery searches with it, the carry and
        # the published fit are judged on it — a +-40 degree fan alone cannot say "lost".
        anchor, _ = max(usable, key=lambda pair: pair[1].fov_deg)
        points = anchor.points
        self.measurements = [
            self._measure(observation, source, prediction, motion, mask)
            for observation, source in usable
        ]
        for measurement in self.measurements:
            stats.source_fit.setdefault(measurement.source, Running()).add(measurement.fit)
        anchored = next(m for m in self.measurements if m.source == anchor.source)
        # The anchor's bound is taken alone (see the docstring): a fan blind along a wall is a
        # plateau whose winner is the guess, and un-widened it out-voted the edge-bound lidar
        # and held a 12 cm slip for seconds under the rest lock (a review probe, 2026-09-11).
        # The covariance now widens such a plateau too (fusion.bound_directions); this rule is
        # what makes the fused tracker provably no slower than the lidar alone on a bound.
        fused = fuse(self.measurements) if self.fusion and not anchored.edge else anchored
        assert fused is not None  # usable is not empty
        self.fused, self.anchor = fused, anchor.source
        if len(self.measurements) > 1 and self.fusion:
            if anchored.edge:
                stats.bound += 1
            else:
                stats.fused += 1
                stats.rejected += len(fused.rejected)
        matched = fused.pose
        pose = matched
        confidence = (
            fused.fit
            if fused.source == anchor.source
            else self._matcher.inlier_fraction(matched, points)
        )

        was_lost = self.lost
        if was_lost:
            stats.lost += 1
            # The whole scan, never the mask: a mask is read at a pose, and a lost tracker's pose
            # is the thing in doubt. Silencing what it cannot explain would silence the evidence.
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

        # The counter first, the blend after: the scan that makes the tracker lost must not be
        # averaged in under the rest lock, and the scan that finds it again (the fit is back,
        # the counter clears) must still be taken at the driving gain, not crawled into.
        self.confidence = confidence
        if confidence < self._lost_below:
            self.weak_scans += 1
            stats.weak += 1
        else:
            self.weak_scans = 0
            self._drift = Pose2D()
        carry_gain = (
            self._matcher.field_score(matched, points)
            - self._matcher.field_score(prediction, points)
            if at_rest and self.rest_lock
            else 0.0
        )
        self.pose, gain = self._blend(
            prediction,
            pose,
            at_rest=at_rest,
            dt_s=dt_s,
            lost=was_lost or self.lost,
            carry_gain=carry_gain,
        )
        self.published_fit = (
            confidence if gain >= 1.0 else self._matcher.inlier_fraction(self.pose, points)
        )
        stats.matched += 1
        stats.fit.add(confidence)
        stats.published_fit.add(self.published_fit)
        return self.pose
