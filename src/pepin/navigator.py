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
from collections.abc import Mapping, Sequence
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
from pepin.scanmatch import SearchWindow
from pepin.tof import ReflexConfig, TofMount, TofRanges

logger = logging.getLogger(__name__)

STOP = Twist(0.0, 0.0)


@dataclass(frozen=True)
class NavigatorConfig:
    """Timing of the decision loop plus the configs of the parts it composes."""

    scan_timeout_s: float = 1.0  # no lidar scan for this long: hold still
    replan_every_s: float = 3.0  # routine replan cadence while driving
    retry_every_s: float = 0.5  # replan cadence while blocked or boxed in
    no_path_patience_s: float = 2.0  # keep the previous plan this long before holding
    obstacle_memory_s: float = 1.0  # live obstacles are forgotten after this: a person moves on
    obstacle_range_m: float = 1.5  # lidar points nearer than this join the live obstacle layer
    # ToF returns nearer than this join the layer too (farther ones are the lidar's job);
    # only sensors with a measured mount can place a hit.
    tof_hit_max_m: float = 0.35
    tof_mounts: Mapping[str, TofMount] = field(default_factory=dict)
    # Before the first move, search this far around the given start pose: a robot placed
    # on its mark by hand is off by decimetres and degrees, beyond the tracking window.
    initial_search: SearchWindow | None = field(
        default_factory=lambda: SearchWindow(
            xy_m=0.6, xy_step_m=0.06, theta_deg=40.0, theta_step_deg=4.0
        )
    )
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


class ObstacleMemory:
    """What any sensor saw recently, as map-frame points, forgotten after ``horizon_s``.

    Time-based on purpose: a memory of "the last N scans" fills up with whichever
    sensor reports fastest (ToF at 20 Hz drowned the lidar at 10 Hz) and forgets
    a person only when enough messages, not seconds, have passed.
    """

    def __init__(self, horizon_s: float) -> None:
        self._horizon_s = horizon_s
        self._entries: deque[tuple[float, NDArray[np.float64]]] = deque()

    def add(self, now: float, points_world: NDArray[np.float64]) -> None:
        """Remember ``points_world`` (N, 2) seen at ``now``; empty sets are ignored."""
        if len(points_world):
            self._entries.append((now, points_world))
        self._expire(now)

    def points(self, now: float) -> NDArray[np.float64] | None:
        """Every point still inside the horizon at ``now``, stacked; None when nothing is left."""
        self._expire(now)
        if not self._entries:
            return None
        return np.vstack([pts for _, pts in self._entries])

    def _expire(self, now: float) -> None:
        while self._entries and now - self._entries[0][0] > self._horizon_s:
            self._entries.popleft()


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
        self._hits = ObstacleMemory(self.cfg.obstacle_memory_s)
        self._latest_points: NDArray[np.float64] | None = None
        self._last_plan_at = -float("inf")
        self._vetoed_forward = False
        self._no_path_since: float | None = None
        self._initialised = False
        self._replan(start, now=0.0)

    @property
    def pose(self) -> Pose2D:
        """Current localised pose."""
        return self.localizer.pose

    def step(self, sense: Sense) -> Decision:
        """Consume one tick of sensor data and decide the body twist for it."""
        pose = self._localise(sense)
        self._remember_tof(sense, pose)
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
            if not self._initialised:
                self._initialised = True
                if self.cfg.initial_search is not None and len(points) >= 50:
                    self.localizer.initialize(points, self.cfg.initial_search)
            pose = self.localizer.update(sense.odom_pose, points)
            self._latest_points = points
            # A lost localiser would smear real walls into phantom obstacles: remember nothing.
            if len(points) and not self.localizer.lost:
                near = points[np.hypot(points[:, 0], points[:, 1]) < self.cfg.obstacle_range_m]
                self._hits.add(sense.now, transform_to_world(near, pose))
        return self.localizer.pose

    def _remember_tof(self, sense: Sense, pose: Pose2D) -> None:
        """Close ToF returns become obstacle-layer points, so low things get routed around too."""
        if sense.tof is None or sense.tof.age_s > self.cfg.reflex.max_age_s:
            return
        hits = [
            self.cfg.tof_mounts[name].hit_xy(r)
            for name, r in sense.tof.by_name().items()
            if r is not None
            and self.cfg.reflex.min_valid_m <= r < self.cfg.tof_hit_max_m
            and name in self.cfg.tof_mounts
        ]
        if hits:
            self._hits.add(sense.now, transform_to_world(np.array(hits), pose))

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
        """Plan from ``pose`` around the remembered obstacles; True when the plan changed.

        A momentary "no path" (a person crossing, a mislocalised scan) keeps the
        previous plan for ``no_path_patience_s``; only a persistent one drops it,
        which makes the caller hold still.
        """
        obstacles = self._hits.points(now)
        fresh = self.planner.plan((pose.x, pose.y), self.goal, obstacles_xy=obstacles)
        self._last_plan_at = now
        self._vetoed_forward = False
        if fresh is None:
            if self._no_path_since is None:
                self._no_path_since = now
            patient = now - self._no_path_since < self.cfg.no_path_patience_s
            if patient and self._follower is not None:
                logger.info("no path right now; keeping the previous plan")
                return False
            logger.warning("no path from %s to %s around the current obstacles", pose, self.goal)
            changed = self.plan is not None
            self.plan, self._follower = None, None
            return changed
        self._no_path_since = None
        if fresh == self.plan:
            return False
        self.plan = fresh
        self._follower = PathFollower(fresh, self.cfg.controller)
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
