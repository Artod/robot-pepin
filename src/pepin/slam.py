"""Graph-based SLAM front end: keyframes, scan-matched chain edges, loop closure.

Every keyframe becomes a node; the scan matcher supplies the edge to the
previous node. When the path returns close to an old keyframe, the current
scan is matched against a small map built around that keyframe; if most of
its points land on walls, that match becomes a loop-closure edge, the pose
graph is optimised and the map is rebuilt from the corrected poses. The
matcher keeps the map locally straight; the graph keeps it globally closed.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pepin.mapping import GridSpec, OccupancyGrid
from pepin.odometry import Pose2D
from pepin.posegraph import Edge, PoseGraph
from pepin.scanmatch import (
    CorrelativeMatcher,
    SearchWindow,
    apply_motion,
    relative_motion,
    should_keyframe,
)

logger = logging.getLogger(__name__)

CHAIN_INFO = (2500.0, 2500.0, 1000.0)  # matcher edge: ~2 cm, ~1.8 deg
LOOP_INFO = (2500.0, 2500.0, 3283.0)  # accepted loop edge: ~2 cm, ~1 deg


@dataclass(frozen=True)
class Keyframe:
    """A scan worth keeping: where odometry thought it was, where SLAM put it, and its points."""

    index: int
    odom: Pose2D
    pose: Pose2D
    points: NDArray[np.float64]


@dataclass(frozen=True)
class LoopClosureConfig:
    """When to look for a revisit and when to believe one."""

    min_index_gap: int = 30  # a revisit must be that many keyframes back, not the recent past
    search_radius_m: float = 0.5  # candidate old keyframes within this distance of the estimate
    local_map_halfwidth: int = 5  # keyframes on each side of the candidate that form its map
    coarse: SearchWindow = field(
        default_factory=lambda: SearchWindow(0.3, 0.06, 15.0, 1.0)
    )  # wide and cheap: 11x11 positions x 31 headings
    fine: SearchWindow = field(
        default_factory=lambda: SearchWindow(0.06, 0.02, 1.5, 0.25)
    )  # tight around the coarse winner
    min_inlier_fraction: float = 0.6
    cooldown_keyframes: int = 10  # after a closure, do not look for another one for a while


@dataclass(frozen=True)
class LoopClosure:
    """An accepted revisit: the edge added to the graph and how well the scan fit."""

    edge: Edge
    inlier_fraction: float


class GraphSlam:
    """Incremental map + pose graph from a stream of (odometry pose, scan points)."""

    def __init__(
        self,
        spec: GridSpec,
        window: SearchWindow | None = None,
        loop: LoopClosureConfig | None = None,
    ) -> None:
        self._spec = spec
        self.grid = OccupancyGrid(spec)
        self._matcher = CorrelativeMatcher(self.grid)
        self._window = window or SearchWindow()
        self._loop = loop or LoopClosureConfig()
        self.graph = PoseGraph()
        self.keyframes: list[Keyframe] = []
        self.closures: list[LoopClosure] = []
        self._last_closure_index = -(10**9)

    @property
    def pose(self) -> Pose2D:
        """Current best estimate: the last keyframe's pose (origin before the first one)."""
        return self.keyframes[-1].pose if self.keyframes else Pose2D()

    def process(self, odom: Pose2D, points: NDArray[np.float64]) -> Keyframe | None:
        """Feed one scan; returns the new keyframe, or None if the robot has not moved enough."""
        if not self.keyframes:
            return self._add_keyframe(odom, odom, points, motion=None)
        last = self.keyframes[-1]
        motion = relative_motion(last.odom, odom)
        if not should_keyframe(motion):
            return None
        guess = apply_motion(last.pose, motion)
        result = self._matcher.match_around(guess, points, motion, self._window)
        return self._add_keyframe(odom, result.pose, points, motion)

    def _add_keyframe(
        self, odom: Pose2D, pose: Pose2D, points: NDArray[np.float64], motion: Pose2D | None
    ) -> Keyframe:
        kf = Keyframe(len(self.keyframes), odom, pose, points)
        self.keyframes.append(kf)
        node = self.graph.add_node(pose)
        if node > 0:
            prev = self.keyframes[-2]
            self.graph.add_edge(Edge(node - 1, node, relative_motion(prev.pose, pose), CHAIN_INFO))
        closure = self.detect_loop(kf)
        if closure is not None:
            self.graph.add_edge(closure.edge)
            self.closures.append(closure)
            self._last_closure_index = kf.index
            self.graph.optimize()
            self._adopt_graph_poses()
            self.rebuild_map()
            logger.info(
                "loop closed: keyframe %d -> %d, inliers %.0f%%, graph error %.1f",
                kf.index,
                closure.edge.j,
                100 * closure.inlier_fraction,
                self.graph.total_error(),
            )
        else:
            self.grid.integrate(pose, points)
            self._matcher.invalidate()
        return self.keyframes[-1]

    def detect_loop(self, kf: Keyframe) -> LoopClosure | None:
        """Try to match ``kf`` against the map around the nearest old keyframe; None if no fit."""
        cfg = self._loop
        if kf.index - self._last_closure_index < cfg.cooldown_keyframes:
            return None
        old = self.keyframes[: max(0, kf.index - cfg.min_index_gap)]
        if not old:
            return None
        nearest = min(old, key=lambda k: math.hypot(k.pose.x - kf.pose.x, k.pose.y - kf.pose.y))
        if math.hypot(nearest.pose.x - kf.pose.x, nearest.pose.y - kf.pose.y) > cfg.search_radius_m:
            return None
        local = OccupancyGrid(self._spec)
        lo, hi = nearest.index - cfg.local_map_halfwidth, nearest.index + cfg.local_map_halfwidth
        for k in self.keyframes[max(0, lo) : hi + 1]:
            local.integrate(k.pose, k.points)
        matcher = CorrelativeMatcher(local, max_points=300)
        coarse = matcher.match(kf.pose, kf.points, cfg.coarse)
        result = matcher.match(coarse.pose, kf.points, cfg.fine)
        inliers = matcher.inlier_fraction(result.pose, kf.points)
        if inliers < cfg.min_inlier_fraction:
            return None
        measured = relative_motion(kf.pose, result.pose)  # correction, in kf's frame
        edge = Edge(kf.index, nearest.index, relative_motion(result.pose, nearest.pose), LOOP_INFO)
        logger.debug(
            "loop candidate %d->%d: correction %s, inliers %.2f",
            kf.index,
            nearest.index,
            measured,
            inliers,
        )
        return LoopClosure(edge=edge, inlier_fraction=inliers)

    def _adopt_graph_poses(self) -> None:
        self.keyframes = [
            Keyframe(k.index, k.odom, node, k.points)
            for k, node in zip(self.keyframes, self.graph.nodes, strict=True)
        ]

    def rebuild_map(self) -> None:
        """Re-integrate every keyframe at its current graph pose (after optimisation)."""
        self.grid = OccupancyGrid(self._spec)
        for k in self.keyframes:
            self.grid.integrate(k.pose, k.points)
        self._matcher = CorrelativeMatcher(self.grid)
