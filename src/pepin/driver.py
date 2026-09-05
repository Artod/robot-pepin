"""The object you talk to: it drives the robot to a place and says how that is going.

A keyboard, a script, a menu-bar button or a language model all want the
same verbs — ``goto("kitchen")``, ``pause()``, ``resume()``, ``cancel()`` —
and one honest report per tick, never scans, twists or deadmen. The
:class:`Driver` owns a :class:`pepin.robot.Robot` (the hardware), a
:class:`pepin.navigator.Navigator` (the decisions) and, optionally, the
session recorder, and turns them into those verbs plus a :class:`Status`.
Nothing here blocks on the network: :meth:`Driver.tick` is the unit of work
and the caller owns the clock.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum

from pepin.feeds import Sense
from pepin.kinematics import STOP, Twist
from pepin.mapping import OccupancyGrid
from pepin.navigator import Decision, Navigator, NavigatorConfig
from pepin.odometry import Pose2D, wrap_angle
from pepin.places import Place
from pepin.recording import SessionRecorder
from pepin.robot import Observation, Robot

logger = logging.getLogger(__name__)

SLOW_TICK_S = 0.25  # a tick slower than this is logged with where the time went
FACE_TOLERANCE_RAD = math.radians(5.0)
FACE_YAW_GAIN = 1.2  # rad/s per rad of heading error while turning to a place's heading
FACE_MAX_YAW_RAD_S = 0.5


class Mode(StrEnum):
    """What the driver is doing, for anyone watching."""

    IDLE = "idle"  # no goal; the wheels rest on the board's deadman
    DRIVING = "driving"
    HOLDING = "holding"  # a goal, but standing still for a stated reason (blind, lost, boxed in)
    PAUSED = "paused"
    FACING = "facing"  # at the place, turning to its preferred heading
    ARRIVED = "arrived"
    NO_BASE = "no base"  # the board has gone quiet; its deadman stopped the wheels


@dataclass(frozen=True)
class Status:
    """One tick's report: mode, where we think we are, what we aim for, and why we do not move."""

    mode: Mode
    pose: Pose2D
    confidence: float  # localiser inlier fraction
    goal: tuple[float, float] | None
    goal_name: str | None
    distance_m: float | None  # straight-line distance to the goal
    target: tuple[float, float] | None  # waypoint being chased
    reason: str  # hold or veto text; "" when driving freely
    twist: Twist  # what went to the wheels
    base_age_s: float
    deadman: bool
    bus_ok: bool

    def summary(self) -> str:
        """One terminal line."""
        where = f"x={self.pose.x:+.2f} y={self.pose.y:+.2f} th={math.degrees(self.pose.theta):+.0f}"
        aim = (
            f"-> {self.goal_name or self.goal} {self.distance_m:.2f} m"
            if self.goal is not None and self.distance_m is not None
            else "no goal"
        )
        why = f" [{self.reason}]" if self.reason else ""
        base = f"base {self.base_age_s * 1000:.0f} ms" + (" DEADMAN" if self.deadman else "")
        return (
            f"{self.mode:<8} {where} conf={self.confidence:.2f} {aim}{why} | "
            f"v={self.twist.linear:+.2f} w={self.twist.angular:+.2f} | {base}"
        )


class Driver:
    """Robot, Navigator and recorder behind goto/pause/resume/cancel and one tick per period."""

    def __init__(
        self,
        robot: Robot,
        grid: OccupancyGrid,
        start: Pose2D,
        *,
        config: NavigatorConfig | None = None,
        places: Mapping[str, Place] | None = None,
        recorder: SessionRecorder | None = None,
        on_plan: Callable[[list[tuple[float, float]] | None], None] | None = None,
    ) -> None:
        """``places`` lets ``goto`` take names; ``on_plan`` hears every plan change (a viewer)."""
        self.robot = robot
        self.navigator = Navigator(grid, start, config=config)
        self.places = dict(places or {})
        self.recorder = recorder
        self._on_plan = on_plan
        self._goal: tuple[float, float] | None = None
        self._goal_name: str | None = None
        self._face: float | None = None
        self._mode = Mode.IDLE
        self._last: Status | None = None

    # -- verbs ----------------------------------------------------------------

    def goto(self, goal: str | tuple[float, float]) -> tuple[float, float]:
        """Head for a place by name or for map coordinates; returns the coordinates.

        Raises ``ValueError`` naming the known places for an unknown name.
        """
        if isinstance(goal, str):
            place = self.places.get(goal)
            if place is None:
                known = ", ".join(sorted(self.places)) or "none"
                raise ValueError(f"unknown place {goal!r}; known: {known}")
            xy, self._goal_name, self._face = place.xy, place.name, place.theta
        else:
            xy, self._goal_name, self._face = (float(goal[0]), float(goal[1])), None, None
        self._goal = xy
        self.navigator.set_goal(xy)
        self.navigator.paused = False
        self._mode = Mode.DRIVING
        logger.info("goto %s %s", self._goal_name or "", xy)
        return xy

    def pause(self) -> None:
        """Stand still, keep the goal."""
        if self._goal is not None:
            self.navigator.paused = True
            self._mode = Mode.PAUSED

    def resume(self) -> None:
        """Carry on toward the goal after a pause."""
        if self._goal is not None:
            self.navigator.paused = False
            self._mode = Mode.DRIVING

    def cancel(self) -> None:
        """Forget the goal and stop."""
        self._goal, self._goal_name, self._face = None, None, None
        self.navigator.set_goal(None)
        self._mode = Mode.IDLE
        self.robot.stop()

    @property
    def mode(self) -> Mode:
        """What the driver is doing right now."""
        return self._mode

    def status(self) -> Status | None:
        """The newest tick's report (None before the first tick)."""
        return self._last

    # -- the loop -------------------------------------------------------------

    def tick(self, now: float) -> Status:
        """One control period: observe, decide, record, act; returns what to show."""
        t0 = time.perf_counter()
        obs = self.robot.observe(now)
        t1 = time.perf_counter()
        if obs is None:
            status, t2 = self._no_base(), t1
        else:
            decision = self.navigator.step(obs.sense)
            t2 = time.perf_counter()
            status = self._act(obs, decision)
        t3 = time.perf_counter()
        if t3 - t0 > SLOW_TICK_S:
            logger.warning(
                "slow tick %.2f s: observe %.0f ms, navigator %.0f ms, act %.0f ms",
                t3 - t0, (t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3,
            )  # fmt: skip
        self._last = status
        return status

    # -- internals ------------------------------------------------------------

    def _no_base(self) -> Status:
        """The board is quiet: nothing to command (its deadman already stopped the wheels)."""
        if self._mode not in (Mode.IDLE, Mode.NO_BASE):
            logger.warning("no word from the base server; its deadman has the wheels")
        self._mode = Mode.NO_BASE if self._goal is not None else Mode.IDLE
        pose = self.navigator.pose
        return Status(
            self._mode, pose, self.navigator.localizer.confidence, self._goal, self._goal_name,
            self._distance(pose), None, "no telemetry from the base server", STOP,
            float("inf"), True, False,
        )  # fmt: skip

    def _act(self, obs: Observation, decision: Decision) -> Status:
        twist, reason = self._command(decision, obs.sense)
        if self.recorder is not None:
            self._record(obs, decision, twist)
        if decision.plan_changed and self._on_plan is not None:
            self._on_plan(self.navigator.plan)
        self.robot.drive(twist)  # also the board's deadman heartbeat
        state = obs.state
        return Status(
            self._mode, decision.pose, decision.confidence, self._goal, self._goal_name,
            self._distance(decision.pose), decision.target, reason, twist,
            state.age_s, state.deadman, state.bus_ok,
        )  # fmt: skip

    def _command(self, decision: Decision, sense: Sense) -> tuple[Twist, str]:
        """Mode bookkeeping around the navigator's decision: the twist that goes out and why."""
        if self._goal is None:
            self._mode = Mode.IDLE
            return STOP, ""
        if self._mode is Mode.ARRIVED:
            return STOP, ""
        if self.navigator.paused:
            self._mode = Mode.PAUSED
            return STOP, "paused"
        if decision.hold:
            self._mode = Mode.HOLDING
            return STOP, decision.hold
        if decision.done or self._mode is Mode.FACING:
            return self._arrive(decision.pose, sense)
        self._mode = Mode.DRIVING
        return decision.twist, decision.veto

    def _arrive(self, pose: Pose2D, sense: Sense) -> tuple[Twist, str]:
        """At the place: turn to its preferred heading if it has one, then report arrival."""
        if self._face is not None:
            error = wrap_angle(self._face - pose.theta)
            if abs(error) > FACE_TOLERANCE_RAD:
                self._mode = Mode.FACING
                yaw = max(-FACE_MAX_YAW_RAD_S, min(FACE_MAX_YAW_RAD_S, FACE_YAW_GAIN * error))
                twist, veto = self.navigator.guard_twist(Twist(0.0, yaw), sense)
                why = f"facing {math.degrees(self._face):.0f} deg"
                return twist, f"{why} ({veto})" if veto else why
        if self._mode is not Mode.ARRIVED:
            logger.info("arrived at %s %s", self._goal_name or "", self._goal)
        self._mode = Mode.ARRIVED
        self.navigator.paused = True
        return STOP, ""

    def _record(self, obs: Observation, decision: Decision, twist: Twist) -> None:
        """Session topics for offline replay: odometry, scans, localised pose, ToF, the command."""
        rec = self.recorder
        assert rec is not None
        rec.pose(obs.state.pose, (obs.state.d_left_m, obs.state.d_right_m))
        for scan in obs.scans:
            rec.scan(scan)
        pose = decision.pose
        rec.write(
            "loc",
            {"x": pose.x, "y": pose.y, "theta": pose.theta, "confidence": decision.confidence},
        )
        if obs.sense.tof is not None:
            r = obs.sense.tof
            rec.write("tof", {"front": r.front, "left": r.left, "right": r.right, "age": r.age_s})
        rec.command(twist)

    def _distance(self, pose: Pose2D) -> float | None:
        if self._goal is None:
            return None
        return math.hypot(self._goal[0] - pose.x, self._goal[1] - pose.y)
