"""Go-to-goal on a frozen map: localise, plan around what the sensors see, follow, stay safe.

One object owns the whole decision chain, so every entry point (the navigate
script, a replay, the simulator in the tests) drives the robot through the same
rules in the same order:

1. localise: scan matching against the map, odometry alone between scans;
2. remember what the lidar saw within reach as live obstacles for about a second;
3. hold still when paused, blind (no scan) or lost;
4. (re)plan when there is no plan, on the routine cadence, or right after a
   guard vetoed the forward motion — the plan then routes around the obstacle;
5. follow the plan, then let the guards trim the command: lidar box ahead,
   ToF reflex with hysteresis.

:meth:`Navigator.step` never touches hardware: one tick of sensor data in, one
:class:`Decision` out. The caller owns the wheels, the recorder and the screen.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pepin.control import ControllerConfig, PathFollower
from pepin.kinematics import Twist
from pepin.localization import Localizer
from pepin.mapping import OccupancyGrid, transform_to_world
from pepin.odometry import Pose2D
from pepin.planning import GridPlanner, PlannerConfig
from pepin.safety import Reflex, SafetyBox, guard_forward
from pepin.tof import ReflexConfig, TofRanges

logger = logging.getLogger(__name__)

STOP = Twist(0.0, 0.0)


@dataclass(frozen=True)
class NavigatorConfig:
    """Timing of the decision loop plus the configs of the parts it composes."""

    scan_timeout_s: float = 1.0  # no lidar scan for this long: hold still
    replan_every_s: float = 3.0  # routine replan cadence while driving
    retry_every_s: float = 0.5  # replan cadence while blocked or boxed in
    obstacle_memory_scans: int = 10  # ~1 s of lidar: a person who moved on frees the path
    obstacle_range_m: float = 1.5  # lidar points nearer than this join the live obstacle layer
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    controller: ControllerConfig = field(default_factory=ControllerConfig)
    # Autonomous mode: stale ToF data holds the robot instead of being ignored.
    reflex: ReflexConfig = field(default_factory=lambda: ReflexConfig(blocked_when_stale=True))
    safety_box: SafetyBox = field(default_factory=SafetyBox)


@dataclass(frozen=True)
class Sense:
    """One tick of sensor input, already in the robot's own units and frame."""

    now: float  # time.monotonic() of this tick
    odom_pose: Pose2D  # wheel odometry, integrated by the caller
    scans: Sequence[NDArray[np.float64]]  # robot-frame (N, 2) point sets since the last tick
    scan_age_s: float  # seconds since the newest scan ever received; inf before the first
    tof: TofRanges | None  # None when running without the ToF sensors


@dataclass(frozen=True)
class Decision:
    """What the robot should do this tick and why."""

    twist: Twist
    pose: Pose2D  # localised pose used for this decision
    confidence: float  # localiser inlier fraction, a health signal
    hold: str = ""  # non-empty: the reason the robot stays still
    veto: str = ""  # non-empty: what a guard trimmed off the follower's command
    target: tuple[float, float] | None = None  # waypoint being chased
    plan_changed: bool = False  # the caller should redraw ``Navigator.plan``
    done: bool = False  # the goal is reached; the caller stops the loop


class Navigator:
    """Localiser, obstacle memory, planner, follower and guards behind one ``step``."""

    def __init__(
        self,
        grid: OccupancyGrid,
        start: Pose2D,
        goal: tuple[float, float],
        config: NavigatorConfig | None = None,
    ) -> None:
        """Plans the first path immediately; ``plan`` is None if the goal is unreachable."""
        self.cfg = config or NavigatorConfig()
        self.goal = goal
        self.localizer = Localizer(grid, start)
        self.planner = GridPlanner(grid, self.cfg.planner)
        self.reflex = Reflex(self.cfg.reflex)
        self.paused = False
        self.plan: list[tuple[float, float]] | None = None
        self._follower: PathFollower | None = None
        self._hits: deque[NDArray[np.float64]] = deque(maxlen=self.cfg.obstacle_memory_scans)
        self._latest_points: NDArray[np.float64] | None = None
        self._last_plan_at = -float("inf")
        self._vetoed_forward = False
        self._replan(start, now=0.0)

    @property
    def pose(self) -> Pose2D:
        """Current localised pose."""
        return self.localizer.pose

    def step(self, sense: Sense) -> Decision:
        """Consume one tick of sensor data and decide the body twist for it."""
        pose = self._localise(sense)
        confidence = self.localizer.confidence
        hold = self._hold_reason(sense)
        changed = False
        if not hold and self._replan_due(sense.now):
            changed = self._replan(pose, sense.now)
        if hold or self._follower is None:
            reason = hold or "no path around the obstacles; waiting"
            return Decision(STOP, pose, confidence, hold=reason, plan_changed=changed)
        out = self._follower.step(pose)
        if out.done:
            return Decision(STOP, pose, confidence, plan_changed=changed, done=True)
        twist, veto = self._guard(out.twist, sense)
        # A vetoed forward wish means the plan runs into something the map lacks:
        # the next tick replans with the live obstacles instead of pushing.
        self._vetoed_forward = bool(veto) and out.twist.linear > 0.0
        return Decision(twist, pose, confidence, veto=veto, target=out.target, plan_changed=changed)

    # -- the five stages ----------------------------------------------------

    def _localise(self, sense: Sense) -> Pose2D:
        """Fold odometry and new scans into the pose; nearby points join the obstacle memory."""
        if not sense.scans:
            self.localizer.predict(sense.odom_pose)
            return self.localizer.pose
        for points in sense.scans:
            pose = self.localizer.update(sense.odom_pose, points)
            self._latest_points = points
            if len(points):
                near = points[np.hypot(points[:, 0], points[:, 1]) < self.cfg.obstacle_range_m]
                self._hits.append(transform_to_world(near, pose))
        return self.localizer.pose

    def _hold_reason(self, sense: Sense) -> str:
        """Why the robot must not move this tick, or "" when it may."""
        if self.paused:
            return "paused"
        if sense.scan_age_s > self.cfg.scan_timeout_s:
            if sense.scan_age_s == float("inf"):
                return "no lidar scan yet"
            return f"no lidar scan for {sense.scan_age_s:.1f} s"
        if self.localizer.lost:
            return f"localiser lost (confidence {self.localizer.confidence:.2f})"
        return ""

    def _replan_due(self, now: float) -> bool:
        """Routine cadence while driving; a quicker retry when boxed in or just vetoed."""
        since = now - self._last_plan_at
        if self._follower is None or self._vetoed_forward:
            return since >= self.cfg.retry_every_s
        return since >= self.cfg.replan_every_s

    def _replan(self, pose: Pose2D, now: float) -> bool:
        """Plan from ``pose`` around the remembered obstacles; True when the plan changed."""
        obstacles = np.vstack(self._hits) if self._hits else None
        fresh = self.planner.plan((pose.x, pose.y), self.goal, obstacles_xy=obstacles)
        self._last_plan_at = now
        self._vetoed_forward = False
        if fresh == self.plan:
            return False
        if fresh is None:
            logger.warning("no path from %s to %s around the current obstacles", pose, self.goal)
        self.plan = fresh
        self._follower = PathFollower(fresh, self.cfg.controller) if fresh else None
        return True

    def _guard(self, twist: Twist, sense: Sense) -> tuple[Twist, str]:
        """Lidar box ahead, then the ToF reflex; returns the trimmed twist and what was trimmed."""
        vetoes = []
        if self._latest_points is not None:
            twist, blocker = guard_forward(twist, self._latest_points, self.cfg.safety_box)
            if blocker is not None:
                vetoes.append(f"lidar: obstacle {blocker:.2f} m ahead")
        if sense.tof is not None:
            decision = self.reflex.step(twist, sense.tof)
            if decision.blocked:
                vetoes.append(f"tof: {decision.reason}")
                twist = decision.twist
        return twist, "; ".join(vetoes)
