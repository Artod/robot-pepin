"""The wheels' expressive motions: small turns in place and the rock, on a thread of their own.

The speaker gaze turns the body toward a person past its forward arc (at most ``max_turn_deg``
per request), and the gesture engine's body atoms rock the cart (at most ``max_rock_m``) or sway
it. :class:`WheelsExecutor` is both doors (the gaze's :class:`pepin.expression.executors.Wheels`,
the gestures' ``body`` :class:`pepin.expression.executors.Executor`) and the one owner of the
wheels in the voice cascade. NOTHING HERE EVER WAITS ON THE CALLER'S THREAD: every request is
queued for the wheels' thread and answered through a callback, and :meth:`WheelsExecutor.halt`
is a flag and one non-blocking line to the base, so the voice (ears, brain, voice) never waits
on the wheels.

THE PATH is the measured motions' method (``ros/tools/move.py``, ``ros/goto.sh round/move``)
run from this Mac, not their process: ``twist`` lines to the board's base server on :3336
through :class:`pepin.base_link.BaseClient` (connected at the first motion), judged by the
base server's own wheel odometry in its 50 Hz state lines (the yaw source with the least
latency: the same connection, ~20-40 ms; the EKF lives in ROS and reaches this process only
through a request), a time cap per leg, the base server's 0.5 s deadman behind it. Never Nav2.
A turn is one continuous motion with a trapezoidal profile (``yaw_rate_deg_s``,
``yaw_acc_deg_s2``, both times ``speed_scale``), done within ``settle_deg``; a straight leg
likewise in metres. THE BRAKING IS PLANNED ON THE CART, NOT ON THE COMMAND: the wheel servos
ramp at their own ``yaw_brake_deg_s2`` whatever the twists ask, and a twist moves the cart
``lead_s`` after it is sent (the WiFi hop, the board's tick, the servos' response: a transport
delay, so what the cart does over the next ``lead_s`` is what was sent over the last), so every
tick brakes from where the cart WILL BE when this tick's twist takes effect (its position plus
the twists in flight) on the lesser of the ramp asked and the cart's own, and sends the zero
once the cart follows and that place is within half of ``settle_deg``: the cart's own last
brake from the profile's speed there runs it the other half at most, so it lands inside
``settle_deg`` whether it brakes at its ramp or stops at once. Braking on the command's own
speed (2026-10-13: the ramp 180, a lead of 0.06 s) landed five turns at 45 deg/s +6.4..+8.1
deg past the ask and the 2 cm back-off steps at 3 cm.

EVERY MOTION IS CHECKED before it starts and at every control tick (25 Hz): the base server's state
is fresh, the cart stands (no twist commanded, the wheels still), the servo bus answers, no stop of
someone else's within ``stop_hold_s`` (the arm is not the wheels' business: they move together, each
within its own caps); the goal server beside Nav2 (:class:`GoalServerView`, asked ten times a second
while a motion runs) says nothing navigates and depth_fusion's newest obstacle marks are fresh, and
under ``clearance_check`` (off since 2026-10-14, Artem: the marks 0.3 m ahead, the dock and his own
legs, refused 19 of 21 turns) the swept footprint keeps ``clearance_m`` from every mark
(:mod:`pepin.expression.clearance`: the
voxel volume's slice, lidar, camera depth and ToF fused, remembered round the cart). Without the
marks nothing moves: a silent depth_fusion is a refusal with its reason, never the raw scan instead.
A refusal says why; nothing is ever clamped.
IT NEVER FIGHTS ANOTHER DRIVER. Every state line is checked against the twists this executor
sent within the last ``ECHO_S``: a line carrying any other twist is someone else's command (the
red button's stop, the teleop, Nav2) and the motion ends on that tick without another twist;
a foreign STOP is answered with a stop of our own (the same word, so the wheels are still at
once instead of after the deadman) and holds every motion off for ``stop_hold_s``. A deadman in
the state line (our twists stopped arriving: the WiFi) ends the motion too.

THE ONE RECOVERY (:class:`pepin.expression.plan.BoundedTurn`): a turn the wheels could not
finish because they stalled (less than a third of the commanded motion over ``stall_s``) or the
arc is blocked steps ``backoff_m`` straight away from the nearest obstacle mark (forward unless
that mark lies ahead of the footprint's middle; checked like any leg), or, when that step cannot
go, turns ``backoff_deg`` back the other way; then it tries the rest of the turn once more and
gives up if that ends short too. A halt, a foreign command, a drive or a failed precondition is
final.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np

from pepin.base_link import DEADMAN_S, BaseState, StateListener
from pepin.expression.animation import Clip, Layer, PartName
from pepin.expression.clearance import (
    Footprint,
    GoalServerView,
    Points,
    Surroundings,
    View,
    away_from,
    drive_blocker,
    turn_blocker,
)
from pepin.expression.executors import (
    Outcome,
    WheelMove,
    WheelReport,
    WheelsDone,
    WheelsMoved,
)
from pepin.expression.mix import track_value
from pepin.expression.parts import move_speed, needed_acc
from pepin.expression.plan import BodyLimits, BoundedTurn, Step
from pepin.kinematics import Twist
from pepin.odometry import Pose2D
from pepin.overlay import read_json

logger = logging.getLogger(__name__)

CONFIG_FILE = "expression.json"
TICK_S = 0.04  # the control loop: a twist every 40 ms, well inside the board's 0.5 s deadman
ECHO_S = 0.3  # a twist of ours shows in the state lines within this (WiFi out, a tick, WiFi in)
HELD_S = DEADMAN_S + ECHO_S  # ...and may still show this long: the board holds it to its deadman
RATE_WINDOW_S = 0.08  # the cart's rate is read over at least this much of the base's lines
STATE_STALE_S = 0.25  # the base's newest state line older than this: no word from the base
FIRST_STATE_S = 1.5  # the first motion connects to the base server and waits this for a line
VIEW_STALE_S = 0.5  # the goal server's newest answer older than this: no word from it
FIRST_VIEW_S = 0.6  # a motion waits this long for the goal server's first fresh answer
SETTLE_S = 0.4  # after a leg's last twist and the lead, how long the wheels get to stand still
STILL_M = 0.0002  # a state line's wheel travel under this (1 cm/s at 50 Hz): standing
STALL_SHARE = 1.0 / 3.0  # a leg that makes less than this of its commanded motion is stalled
STALL_MIN_DEG, STALL_MIN_M = 2.0, 0.005  # ...once at least this much was commanded
TURN_FLOOR_DEG_S, DRIVE_FLOOR_M_S = 2.0, 0.01  # the least speed a leg commands (stiction)
SAME = 1e-6  # two twists this close are the same
BODY_DOFS = ("yaw_deg", "x_m")
Source = Literal["gaze", "gesture", "door"]
Report = Callable[[str], None]


def say(line: str) -> None:
    """One line of the wheels in the cascade's output."""
    print(f"  (wheels: {line})", flush=True)


# ---- the knobs ---------------------------------------------------------------------------------
@dataclass(frozen=True)
class WheelsSettings:
    """config/expression.json's ``wheels`` block: the request limits (:class:`BodyLimits`), the
    motion's caps at full speed and ``speed_scale``, the tolerances and the checks."""

    limits: BodyLimits = field(default_factory=BodyLimits)
    speed_scale: float = 1.0
    yaw_rate_deg_s: float = 45.0
    yaw_acc_deg_s2: float = 180.0
    yaw_brake_deg_s2: float = 100.0
    lead_s: float = 0.17
    speed_m_s: float = 0.10
    acc_m_s2: float = 0.5
    settle_deg: float = 2.0
    settle_m: float = 0.005
    clearance_check: bool = True  # the swept footprint against the obstacle marks
    clearance_m: float = 0.10
    marks_age_s: float = 1.5
    stall_s: float = 0.5
    stop_hold_s: float = 30.0
    latency_s: float = 0.15

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> WheelsSettings:
        """From config/expression.json's object; ``ValueError`` naming a value no motion can
        use."""
        block = data.get("wheels", {})
        limits = BodyLimits(
            **{
                name: type(default)(block[name])
                for name, default in vars(BodyLimits()).items()
                if name in block
            }
        )
        d = cls()
        values = {
            name: type(default)(block[name])
            for name, default in vars(d).items()
            if name != "limits" and name in block
        }
        settings = replace(d, limits=limits, **values)
        if not 0.0 < settings.speed_scale <= 1.0:
            raise ValueError(
                f"wheels.speed_scale is above 0 and at most 1, not {settings.speed_scale}"
            )
        positive = ("yaw_rate_deg_s", "yaw_acc_deg_s2", "yaw_brake_deg_s2", "speed_m_s",
                    "acc_m_s2", "settle_deg", "settle_m", "marks_age_s", "stall_s")  # fmt: skip
        for name in positive:
            if getattr(settings, name) <= 0.0:
                raise ValueError(f"wheels.{name} must be above 0")
        if settings.lead_s < 0.0:
            raise ValueError("wheels.lead_s must be 0 or above")
        if limits.step_deg <= 0.0:
            raise ValueError("wheels.step_deg must be above 0")
        if not 0.0 < limits.backoff_m <= limits.max_rock_m:
            raise ValueError(f"wheels.backoff_m is above 0 and at most max_rock_m"
                             f" ({limits.max_rock_m}), not {limits.backoff_m}")  # fmt: skip
        if not 0.0 < limits.backoff_deg <= limits.max_turn_deg:
            raise ValueError(f"wheels.backoff_deg is above 0 and at most max_turn_deg"
                             f" ({limits.max_turn_deg}), not {limits.backoff_deg}")  # fmt: skip
        return settings

    @classmethod
    def load(cls, path: str | Path | None = None) -> WheelsSettings:
        """Read config/expression.json (or ``path``)."""
        if path is None:
            from pepin.deployment import config_file

            path = config_file(CONFIG_FILE)
        data = read_json(path)
        if not isinstance(data, dict):
            raise ValueError(f"{path} holds no JSON object")
        return cls.from_config(data)

    def yaw_caps(self) -> tuple[float, float]:
        """The turn's top rate and ramp at this speed scale (rad/s, rad/s^2)."""
        s = self.speed_scale
        return math.radians(self.yaw_rate_deg_s) * s, math.radians(self.yaw_acc_deg_s2) * s * s

    def drive_caps(self) -> tuple[float, float]:
        """A straight leg's top speed and ramp at this speed scale (m/s, m/s^2)."""
        s = self.speed_scale
        return self.speed_m_s * s, self.acc_m_s2 * s * s

    def yaw_brake(self) -> float:
        """The ramp a turn is braked on (rad/s^2): the lesser of the ramp asked at this speed
        scale and the cart's own (the wheel servos' ``yaw_brake_deg_s2``, whatever is asked)."""
        return min(self.yaw_caps()[1], math.radians(self.yaw_brake_deg_s2))


def footprint_of(path: str | Path | None = None) -> Footprint:
    """config/base.json's footprint."""
    if path is None:
        from pepin.deployment import config_file

        path = config_file("base.json")
    return Footprint.from_config(read_json(path))


# ---- what the wheels read ----------------------------------------------------------------------
class BaseLink(Protocol):
    """The base server as the wheels drive it (:class:`pepin.base_link.BaseClient`)."""

    def set_twist(self, twist: Twist) -> None:
        """One non-blocking twist line."""
        ...

    def stop(self) -> None:
        """One non-blocking stop line."""
        ...

    def state(self, now: float | None = None) -> BaseState | None:
        """The newest state line, aged at ``now``; None before the first."""
        ...


# ---- a body clip as legs -----------------------------------------------------------------------
@dataclass(frozen=True)
class Leg:
    """One straight piece of a body clip: from ``at_s`` to ``until_s`` (seconds from the clip's
    start) the cart goes to ``target`` (degrees of yaw, + left, or metres, + forward, from where
    the clip began) on ``axis``."""

    at_s: float
    until_s: float
    axis: Literal["yaw", "x"]
    target: float


def body_keyframes(clip: Clip) -> list[tuple[float, float, float]]:
    """A body clip as ``(seconds, yaw_deg, x_m)`` at every key time of its tracks, each the sum
    of its tracks there (a merged clip's steps are offsets: they add up)."""
    tracks = [t for t in clip.tracks if t.dof in BODY_DOFS]
    times = sorted({t for track in tracks for t in track.times})

    def axis(dof: str, t: float) -> float:
        return sum(track_value(track, t) for track in tracks if track.dof == dof)

    return [(t, axis("yaw_deg", t), axis("x_m", t)) for t in times]


def legs_of(clip: Clip, settings: WheelsSettings) -> tuple[Leg, ...] | str:
    """A body clip as its legs, or why it cannot be played: past the limits, turning and
    driving in one leg, or faster than the caps at the speed scale (never clamped: the gesture
    engine stretches the body's clips by 1 / ``speed_scale`` as it does the arm's)."""
    if clip.part != "body":
        return f"an {clip.part} clip is not the body's"
    frames = body_keyframes(clip)
    if not frames:
        return "the clip moves no wheel"
    lim = settings.limits
    far_yaw = max(abs(f[1]) for f in frames)
    far_x = max(abs(f[2]) for f in frames)
    if far_yaw > lim.max_turn_deg + 1e-9:
        return f"it turns {far_yaw:.0f} deg, past wheels.max_turn_deg {lim.max_turn_deg:.0f}"
    if far_x > lim.max_rock_m + 1e-9:
        return f"it drives {far_x * 100:.1f} cm, past wheels.max_rock_m {lim.max_rock_m * 100:.0f}"
    yaw_rate, yaw_acc = settings.yaw_caps()
    speed, acc = settings.drive_caps()
    legs: list[Leg] = []
    t0, yaw0, x0 = frames[0]
    for t, yaw, x in frames[1:]:
        turns, drives = abs(yaw - yaw0) > SAME, abs(x - x0) > SAME
        if turns and drives:
            return f"it turns and drives at once at {t:.2f} s"
        if turns or drives:
            seconds = t - t0
            travel = math.radians(yaw - yaw0) if turns else x - x0
            top, ramp = (yaw_rate, yaw_acc) if turns else (speed, acc)
            if needed_acc(travel, seconds) > ramp * (1.0 + 1e-6) or (
                move_speed(travel, seconds, ramp) > top * (1.0 + 1e-6)
            ):
                unit = "deg" if turns else "cm"
                amount = abs(yaw - yaw0) if turns else abs(x - x0) * 100.0
                return (
                    f"{amount:.1f} {unit} in {seconds:.2f} s at {t0:.2f} s is faster than the"
                    f" wheels' caps at speed {settings.speed_scale:g}"
                )
            legs.append(Leg(t0, t, "yaw" if turns else "x", yaw if turns else x))
        t0, yaw0, x0 = t, yaw, x
    return tuple(legs) if legs else "the clip moves no wheel"


# ---- the executor ------------------------------------------------------------------------------
@dataclass
class Job:
    """One request on the wheels' queue: a gaze turn or a door's motion (``move``), or a
    gesture's legs, who asked, the callbacks, when it was asked, its cancel flag and why."""

    source: Source
    move: WheelMove | None = None
    legs: tuple[Leg, ...] = ()
    done: WheelsDone | None = None
    moved: WheelsMoved | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    why: str = ""
    cut: bool = False

    def describe(self) -> str:
        """``turn +30 deg``, ``rock 2.0 cm x2``, ``clip of 4 legs``."""
        m = self.move
        if m is None:
            return f"clip of {len(self.legs)} legs"
        if m.rocks or m.rock_m:
            return f"rock {m.rock_m * 100:+.1f} cm x{m.rocks}"
        return f"turn {m.turn_deg:+.0f} deg"


@dataclass(frozen=True)
class LegEnd:
    """How one leg ended: where the cart got to on the leg's axis (from its origin), and why it
    stopped short (:class:`pepin.expression.plan.Step`'s ``blocked`` and ``stopped``)."""

    at: float
    blocked: str = ""
    stopped: str = ""


Pause = Callable[[threading.Event, float], None]


def _real_pause(event: threading.Event, seconds: float) -> None:
    event.wait(seconds)


class WheelsExecutor:
    """The wheels' one owner in the voice cascade: the speaker gaze's
    :class:`pepin.expression.executors.Wheels` and the gesture engine's ``body`` part. One
    motion at a time on one daemon thread; a gaze turn or a door's motion replaces anything
    running, a gesture never replaces a gaze turn (refused while one runs) and replaces an older
    gesture."""

    part: PartName = "body"

    def __init__(
        self,
        link: Callable[[StateListener], BaseLink],
        surroundings: Surroundings,
        *,
        load: Callable[[], WheelsSettings] = WheelsSettings.load,
        footprint: Footprint | None = None,
        clock: Callable[[], float] = time.monotonic,
        pause: Pause = _real_pause,
        report: Report = say,
    ) -> None:
        """``link(listener)`` opens the base link with ``listener`` hearing every state line
        (called on the wheels' thread at the first motion); ``load`` reads the knobs (at once,
        and again at every motion); ``pause(cancel, s)`` waits ``s`` or until ``cancel``."""
        self._link_factory = link
        self.around = surroundings
        self._load = load
        self.settings = load()
        self.footprint = footprint if footprint is not None else Footprint()
        self.clock, self.pause, self.report = clock, pause, report
        self.base: BaseLink | None = None
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._pending: Job | None = None
        self._current: Job | None = None
        self._closed = False
        self._thread: threading.Thread | None = None
        # The twists this executor sent (laptop time, v, w), and the twist before its first:
        # a state line showing anything else is someone else's command.
        self._sent: deque[tuple[float, float, float]] = deque()
        self._before = (0.0, 0.0)
        self._commanding = False  # the last twist sent was not zero
        self._first_at: float | None = None  # when this job's first twist went
        self._foreign_stop_at: float | None = None
        self._settling = threading.Event()  # never set: a settle waits its ticks out
        self.counts: dict[str, int] = {}

    # ---- the gaze's door (Wheels) --------------------------------------------------------------
    def start(self, request: WheelMove, done: WheelsDone, moved: WheelsMoved | None = None) -> None:
        """A turn (or a rock) for the speaker gaze: queued, answered through ``done``."""
        self._queue(Job("gaze", move=request, done=done, moved=moved))

    def turn(
        self, deg: float, done: WheelsDone | None = None, moved: WheelsMoved | None = None
    ) -> None:
        """Turn in place by ``deg`` (+ left), as the command line asks."""
        self._queue(Job("door", move=WheelMove(turn_deg=deg), done=done, moved=moved))

    def rock(self, metres: float, times: int = 1, done: WheelsDone | None = None) -> None:
        """Forward ``metres`` and back (backward first when negative), ``times`` times."""
        self._queue(Job("door", move=WheelMove(rock_m=metres, rocks=times), done=done))

    def halt(self, why: str) -> None:
        """Stop the motion running at once (one stop line when ours was the last twist) and
        drop what waits. Never blocks."""
        with self._lock:
            dropped, self._pending = self._pending, None
            if self._current is not None:
                self._halt_locked(self._current, why)
        self._answer(dropped, WheelReport(False, why=f"dropped: {why}", cut=True))

    def hush(self, why: str) -> None:
        """Stop a gesture's motion at once and drop a gesture's waiting one (the person
        speaks); a turn toward the speaker goes on. Never blocks."""
        self._halt_source("gesture", why)

    def preview(self, request: WheelMove) -> str | None:
        """Why ``request`` would be refused now, checked without moving (the knobs, the base's
        state, the goal server's word, the swept footprint); None when it would go. Blocks the
        caller up to ~2 s: the command line's dry run only, never the cascade."""
        job = Job("door", move=request)
        refusal = self._invalid(job) or self._ready_to_move(job)
        try:
            if refusal is not None:
                return refusal
            state, view = self._base_state(), self.around.latest()
            if state is None or view is None:
                return "the base server or the goal server fell silent"
            if not self.settings.clearance_check:
                return None
            margin = self.settings.clearance_m
            if request.rocks or request.rock_m:
                hit = drive_blocker(view.points, self.footprint, 0.0, request.rock_m, margin)
                return f"the way is blocked: {hit.text('cm')}" if hit is not None else None
            turn = math.radians(request.turn_deg)
            hit = turn_blocker(view.points, self.footprint, 0.0, turn, margin)
            return f"the arc is blocked: {hit.text('deg')}" if hit is not None else None
        finally:
            self.around.watch(False)

    # ---- the gestures' door (Executor) ---------------------------------------------------------
    @property
    def latency_s(self) -> float:
        """From :meth:`play` to the first motion: the goal server's first answer, the WiFi."""
        return self.settings.latency_s

    def ready(self) -> str | None:
        """Why a gesture cannot move the wheels now (a gaze turn runs, someone else's stop
        holds them); None otherwise. Reads only what is in memory."""
        with self._lock:
            current = self._current
            held = self._held_locked(self.clock())
        if current is not None and current.source != "gesture":
            return "the wheels turn toward the speaker"
        return held

    def gesture_moving(self) -> bool:
        """Whether a gesture's motion (a body clip: a dance's rock) runs or waits its turn; a
        gaze turn or a door's motion is none. Reads only what is in memory: the head part asks
        it as a clip leaves, so the arbiter lets the clip ride on this rock
        (:class:`pepin.expression.parts.HeadPart`)."""
        with self._lock:
            jobs = (self._current, self._pending)
        return any(job is not None and job.source == "gesture" for job in jobs)

    def play(self, clip: Clip) -> Outcome:
        """A gesture's body clip: its legs queued (refused at once past the limits or caps)."""
        legs = legs_of(clip, self.settings)
        if isinstance(legs, str):
            return Outcome(False, legs)
        refusal = self.ready()
        if refusal is not None:
            return Outcome(False, refusal)
        self._queue(Job("gesture", legs=legs))
        return Outcome(True, f"{len(legs)} legs", clip.duration_s)

    def fade(self, layer: Layer, seconds: float) -> None:
        """A gesture let go: its motion stops where it is (the wheels have no base layer)."""
        self._halt_source("gesture", f"the {layer} layer faded")

    def rest(self) -> None:
        """Every gesture let go: its motion stops (a gaze turn goes on)."""
        self._halt_source("gesture", "the gestures rest")

    # ---- the base's word -----------------------------------------------------------------------
    def heard(self, state: BaseState, at: float) -> None:
        """One state line (the base link's reader thread, ``at`` its arrival): the motion ends
        at once when it carries a twist this executor did not send."""
        with self._lock:
            job = self._current
            if job is None or self._first_at is None or job.cancel.is_set():
                return
            if state.deadman and at - self._first_at >= ECHO_S:  # earlier: a teleop's old one
                self._end_locked(job, "the board's deadman stopped the wheels (our twists"
                                      " stopped arriving: the WiFi?)")  # fmt: skip
                return
            if self._ours(state.v, state.w, at):
                return
            if abs(state.v) < SAME and abs(state.w) < SAME:
                self._foreign_stop_at = at
                self._end_locked(job, "someone else stopped the wheels (the red button, a"
                                      " teleop's stop)", agree=True)  # fmt: skip
            else:
                self._end_locked(job, f"someone else drives the wheels (v {state.v:+.2f} m/s,"
                                      f" w {state.w:+.2f} rad/s)")  # fmt: skip

    # ---- the thread ------------------------------------------------------------------------------
    def close(self) -> None:
        """Halt, and stop the thread (the base server's deadman and our leaving stop the rest)."""
        self.halt("closing")
        with self._lock:
            self._closed = True
            self._wake.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.around.watch(False)
        closer = getattr(self.base, "close", None)
        if callable(closer):
            closer()

    def _queue(self, job: Job) -> None:
        refusal = self._invalid(job)
        if refusal is not None:
            self._answer(job, WheelReport(False, why=f"refused: {refusal}"))
            return
        with self._lock:
            current = self._current
            if job.source == "gesture" and current is not None and current.source != "gesture":
                busy: str | None = "the wheels turn toward the speaker"
                superseded = None
            else:
                busy = None
                superseded, self._pending = self._pending, job
                if current is not None:
                    self._halt_locked(current, f"a newer motion ({job.describe()})")
                if self._thread is None:
                    self._thread = threading.Thread(target=self._run, name="wheels", daemon=True)
                    self._thread.start()
                self._wake.notify_all()
        if busy is not None:
            self._answer(job, WheelReport(False, why=f"refused: {busy}"))
        self._answer(superseded, WheelReport(False, why="superseded by a newer motion", cut=True))

    def _invalid(self, job: Job) -> str | None:
        """What the knobs refuse outright (no I/O)."""
        m, lim = job.move, self.settings.limits
        if m is None:
            return None
        if (m.rocks or m.rock_m) and m.turn_deg:
            return "a turn and a rock in one request"
        if m.rocks or m.rock_m:
            if m.rocks < 1 or abs(m.rock_m) < SAME:
                return "a rock is at least one, of more than nothing"
            if abs(m.rock_m) > lim.max_rock_m + 1e-9:
                return f"a rock of {m.rock_m * 100:.1f} cm is past {lim.max_rock_m * 100:.0f}"
            return None
        if abs(m.turn_deg) > lim.max_turn_deg + 1e-9:
            return f"a turn of {m.turn_deg:+.0f} deg is past {lim.max_turn_deg:.0f}"
        return None

    def _run(self) -> None:
        while True:
            with self._lock:
                while self._pending is None and not self._closed:
                    self._wake.wait()
                if self._closed:
                    return
                job, self._pending = self._pending, None
                assert job is not None
                self._current = job
                self._sent.clear()
                self._commanding, self._first_at = False, None
            began = self.clock()
            try:
                report = self._perform(job)
            except Exception as error:  # a broken motion stops the wheels and never the voice
                logger.exception("wheels: the motion broke")
                report = WheelReport(False, why=f"broke: {type(error).__name__}: {error}")
            with self._lock:
                if self._commanding and self.base is not None:
                    self.base.stop()
                self._commanding = False
                self._current = None
            self.around.watch(False)
            self.counts[job.source] = self.counts.get(job.source, 0) + 1
            took = self.clock() - began
            state = "done" if report.done else report.why
            self.report(f"{job.describe()} ({job.source}): {state}; turned"
                        f" {report.turned_deg:+.1f} deg, backed {report.backed_m * 100:.0f} cm,"
                        f" {report.attempts} attempt(s), {took:.1f} s")  # fmt: skip
            self._answer(job, report)

    def _perform(self, job: Job) -> WheelReport:
        try:
            self.settings = self._load()
        except (OSError, ValueError, TypeError) as error:
            logger.warning(
                "wheels: config/%s unreadable (%s): the last knobs stand", CONFIG_FILE, error
            )
        refusal = self._ready_to_move(job)
        if refusal is not None:
            return WheelReport(False, why=f"refused: {refusal}", cut=job.cut)
        m = job.move
        if m is not None and not (m.rocks or m.rock_m):
            return BoundedTurn(_Steps(self, job), self.settings.limits).move(m)
        if m is not None:
            legs = []
            for _ in range(m.rocks):
                legs += [Leg(0.0, 0.0, "x", m.rock_m), Leg(0.0, 0.0, "x", 0.0)]
            return self._play_legs(job, tuple(legs), timed=False)
        return self._play_legs(job, job.legs, timed=True)

    # ---- the checks ----------------------------------------------------------------------------
    def _ready_to_move(self, job: Job) -> str | None:
        """Every precondition of a motion that starts now (may wait on the goal server's first
        answer and the base's first line: the wheels' thread only)."""
        if job.cancel.is_set():
            return job.why
        with self._lock:
            held = self._held_locked(self.clock())
        if held is not None:
            return held
        if self.base is None:
            self.base = self._link_factory(self.heard)
        self.around.watch(True)
        asked = self.clock()
        state = self._wait_for(job, lambda: self._base_state(), FIRST_STATE_S)
        if state is None:
            return "no word from the base server (is pepin-base up?)"
        why = standing(state)
        if why is not None:
            return why
        view = self._wait_for(job, lambda: self._fresh_view(asked), FIRST_VIEW_S)
        if job.cancel.is_set():
            return job.why
        if view is None:
            latest = self.around.latest()
            if latest is not None and latest.why:
                return latest.why
            return f"the goal server did not answer within {FIRST_VIEW_S:.1f} s (ros/laptop.sh nav)"
        return self._view_refusal(view, self.clock())

    def _wait_for(self, job: Job, get: Callable[[], Any], seconds: float) -> Any:
        deadline = self.clock() + seconds
        while True:
            value = get()
            if value is not None or job.cancel.is_set() or self.clock() >= deadline:
                return value
            self.pause(job.cancel, 0.02)

    def _base_state(self) -> BaseState | None:
        state = self.base.state(self.clock()) if self.base is not None else None
        if state is None or state.age_s > STATE_STALE_S:
            return None
        return state

    def _fresh_view(self, since: float) -> View | None:
        view = self.around.latest()
        return view if view is not None and view.at >= since else None

    def _view_refusal(self, view: View, now: float) -> str | None:
        """What the goal server's word refuses: no word, a recent cancel (the red button), a
        drive, old or no obstacle marks."""
        if now - view.at > VIEW_STALE_S:
            return f"the goal server's newest word is {now - view.at:.1f} s old"
        if view.why:
            return view.why
        hold = self.settings.stop_hold_s
        if view.cancel_age_s is not None and view.cancel_age_s + (now - view.at) < hold:
            ago = view.cancel_age_s + (now - view.at)
            return (f"a cancel reached the goal server {ago:.0f} s ago (ros/stop.sh, goto.sh"
                    f" cancel): no motion for {hold:.0f} s (wheels.stop_hold_s)")  # fmt: skip
        if view.navigating is None:
            return "the goal server did not say whether Nav2 navigates"
        if view.navigating:
            return "Nav2 navigates (`ros/goto.sh where`)"
        if not self.settings.clearance_check:
            return None
        if view.marks_age_s is None:
            return "no obstacle marks have reached the goal server"
        age = view.marks_age_s + (now - view.at)
        if age > self.settings.marks_age_s:
            return f"the newest obstacle marks are {age:.1f} s old (wheels.marks_age_s)"
        return None

    def _held_locked(self, now: float) -> str | None:
        at = self._foreign_stop_at
        hold = self.settings.stop_hold_s
        if at is not None and now - at < hold:
            return (f"someone else stopped the wheels {now - at:.0f} s ago: no motion for"
                    f" {hold:.0f} s (wheels.stop_hold_s)")  # fmt: skip
        return None

    # ---- commanding --------------------------------------------------------------------------
    def _command(self, job: Job, twist: Twist) -> bool:
        """Send ``twist`` unless the job was halted (checked and sent under one lock, so no
        twist of ours follows a halt); whether it went."""
        with self._lock:
            if job.cancel.is_set() or self.base is None:
                return False
            if self._first_at is None:
                state = self.base.state(self.clock())
                self._before = (state.v, state.w) if state is not None else (0.0, 0.0)
                self._first_at = self.clock()
            self.base.set_twist(twist)
            self._sent.append((self.clock(), twist.linear, twist.angular))
            self._commanding = abs(twist.linear) > SAME or abs(twist.angular) > SAME
            return True

    def _ours(self, v: float, w: float, at: float) -> bool:
        """Whether a state line received at ``at`` shows a twist this executor sent: one sent
        within ``HELD_S`` (the board holds a twist until its deadman, and the line takes the
        echo to come back), the last before that, or the twist before its first while that may
        still stand."""
        while len(self._sent) > 1 and self._sent[1][0] < at - HELD_S:
            self._sent.popleft()
        older = [s for s in self._sent if s[0] < at - HELD_S]
        recent = [s for s in self._sent if s[0] >= at - HELD_S]
        known = [(sv, sw) for _, sv, sw in recent]
        if older:
            known.append((older[-1][1], older[-1][2]))
        elif self._first_at is None or at - self._first_at < ECHO_S:
            known.append(self._before)
        return any(abs(v - kv) < SAME and abs(w - kw) < SAME for kv, kw in known)

    def _halt_locked(self, job: Job, why: str) -> None:
        if job.cancel.is_set():
            return
        job.why, job.cut = f"halted: {why}", True
        job.cancel.set()
        if self._commanding and self.base is not None:
            self.base.stop()
            self._sent.append((self.clock(), 0.0, 0.0))
        self._commanding = False

    def _end_locked(self, job: Job, why: str, *, agree: bool = False) -> None:
        """Someone else's word ends the motion: no twist of ours after it (a stop of our own
        when it was a stop: the same word)."""
        job.why = why
        job.cancel.set()
        if agree and self._commanding and self.base is not None:
            self.base.stop()
            self._sent.append((self.clock(), 0.0, 0.0))
        self._commanding = False

    def _halt_source(self, source: Source, why: str) -> None:
        with self._lock:
            dropped = None
            if self._pending is not None and self._pending.source == source:
                dropped, self._pending = self._pending, None
            if self._current is not None and self._current.source == source:
                self._halt_locked(self._current, why)
        self._answer(dropped, WheelReport(False, why=f"dropped: {why}", cut=True))

    def _answer(self, job: Job | None, report: WheelReport) -> None:
        if job is None or job.done is None:
            return
        try:
            job.done(report)
        except Exception:  # a listener never stops the wheels
            logger.exception("wheels: the listener of a motion's end broke")

    # ---- the legs ----------------------------------------------------------------------------
    def turn_leg(self, job: Job, deg: float, before_deg: float = 0.0) -> Step:
        """One continuous turn by ``deg`` from where the cart stands (the retry's attempt;
        ``before_deg`` turned by the attempts before it, for the progress told)."""
        state = self._base_state()
        if state is None:
            return Step(0.0, stopped=job.why or "no word from the base server")
        origin = state.pose
        rate, acc = self.settings.yaw_caps()
        end = self._leg(job, "yaw", origin, math.radians(deg), rate, acc,
                        self._progress(job, before_deg))  # fmt: skip
        return Step(math.degrees(end.at), end.blocked, end.stopped, job.cut)

    def step_away(self, job: Job, metres: float) -> Step:
        """Straight by ``metres`` away from the nearest obstacle mark in the newest view
        (:func:`pepin.expression.clearance.away_from`), checked like any leg."""
        state = self._base_state()
        if state is None:
            return Step(0.0, stopped=job.why or "no word from the base server")
        view = self.around.latest()
        sign = away_from(view.points, self.footprint) if view is not None else 1.0
        speed, acc = self.settings.drive_caps()
        end = self._leg(job, "x", state.pose, sign * abs(metres), speed, acc, None)
        way = "forward" if sign > 0 else "back"
        blocked = f"the step {way} is blocked: {end.blocked}" if end.blocked else ""
        return Step(abs(end.at), blocked, end.stopped, job.cut)

    def _play_legs(self, job: Job, legs: tuple[Leg, ...], *, timed: bool) -> WheelReport:
        """A rock or a clip: each leg from the motion's origin, at its time when ``timed``."""
        state = self._base_state()
        if state is None:
            return WheelReport(False, why=job.why or "no word from the base server")
        origin, began = state.pose, self.clock()
        yaw = 0.0
        for leg in legs:
            if timed and (wait := began + leg.at_s - self.clock()) > 0:
                self.pause(job.cancel, wait)
            if job.cancel.is_set():
                return WheelReport(False, math.degrees(yaw), why=job.why, cut=job.cut)
            yaw_rate, yaw_acc = self.settings.yaw_caps()
            speed, acc = self.settings.drive_caps()
            top, ramp = (yaw_rate, yaw_acc) if leg.axis == "yaw" else (speed, acc)
            target = math.radians(leg.target) if leg.axis == "yaw" else leg.target
            if timed:  # the trapezoid that ends on the leg's time, within the caps
                now = self._base_state()
                here = offset(now.pose, origin, leg.axis) if now is not None else 0.0
                top = min(top, move_speed(target - here, leg.until_s - leg.at_s, ramp))
            end = self._leg(job, leg.axis, origin, target, top, ramp, None)
            if leg.axis == "yaw":
                yaw = end.at
            if end.blocked or end.stopped:
                return WheelReport(False, math.degrees(yaw), why=end.stopped or end.blocked,
                                   cut=job.cut)  # fmt: skip
        return WheelReport(True, math.degrees(yaw), attempts=1)

    def _progress(self, job: Job, before_deg: float) -> Callable[[float], None] | None:
        """The job's progress listener, fed radians of this attempt, told degrees in all."""
        if job.moved is None:
            return None
        listener = job.moved

        def tell(rad: float) -> None:
            try:
                listener(before_deg + math.degrees(rad))
            except Exception:
                logger.exception("wheels: the listener of a turn's progress broke")

        return tell

    def _leg(
        self,
        job: Job,
        axis: Literal["yaw", "x"],
        origin: Pose2D,
        target: float,
        top: float,
        ramp: float,
        progress: Callable[[float], None] | None,
    ) -> LegEnd:
        """Drive ``axis`` to ``target`` (radians or metres from ``origin``): checked every tick,
        braked on what is left from where the cart will be when the twist takes effect, the
        zero sent within half the tolerance of it, stopped and settled at the end."""
        s = self.settings
        tol = math.radians(s.settle_deg) if axis == "yaw" else s.settle_m
        floor = min(top, math.radians(TURN_FLOOR_DEG_S) if axis == "yaw" else DRIVE_FLOOR_M_S)
        stall_min = math.radians(STALL_MIN_DEG) if axis == "yaw" else STALL_MIN_M
        step = math.radians(s.limits.step_deg)
        # The brake: the cart's own ramp when it is under the one asked (the wheel servos ramp
        # at theirs whatever the twists say; a straight leg's ramp is about theirs already).
        brake = min(ramp, s.yaw_brake()) if axis == "yaw" else ramp
        state = self._base_state()
        if state is None:
            return LegEnd(0.0, stopped=job.why or "no word from the base server")
        start = offset(state.pose, origin, axis)
        tol = min(tol, abs(target - start) / 3.0) if abs(target - start) > SAME else tol
        direction = 1.0 if target >= start else -1.0
        began = self.clock()
        cap = 2.0 * (abs(target - start) / max(top, SAME) + top / ramp) + 1.0
        last, told, integral, then = 0.0, start, 0.0, began
        history: deque[tuple[float, float, float]] = deque([(began, 0.0, start)])
        reads: deque[tuple[float, float]] = deque()  # (board stamp, here): the measured rate
        here = start
        while True:
            now = self.clock()
            if job.cancel.is_set():
                return self._stop_leg(job, axis, origin, here, stopped=job.why)
            state = self._base_state()
            if state is None:
                silent = "no word from the base server for a while"
                return self._stop_leg(job, axis, origin, here, stopped=silent)
            here = offset(state.pose, origin, axis)
            if not reads or state.stamp_s > reads[-1][0]:
                reads.append((state.stamp_s, here))
            rate = measured_rate(reads, RATE_WINDOW_S)
            left = target - here
            if abs(left) <= tol or left * direction <= 0.0:
                break
            stopped, blocked = self._check(axis, origin, state.pose, here, target, last, now)
            if stopped or blocked:
                return self._stop_leg(job, axis, origin, here, stopped=stopped, blocked=blocked)
            integral += last * (now - then)
            then = now
            history.append((now, integral, here))
            while len(history) > 1 and now - history[1][0] >= s.stall_s + s.lead_s:
                history.popleft()
            stall = stalled(history, now, s.stall_s, direction, stall_min, axis, s.lead_s)
            if stall:
                return self._stop_leg(job, axis, origin, here, blocked=stall)
            if now - began > cap:
                return self._stop_leg(job, axis, origin, here,
                                      stopped=f"the time cap ({cap:.1f} s) ran out")  # fmt: skip
            # Where the cart will be when this tick's twist takes effect: the twists in flight
            # (sent since the line's stamp less the lead) are what it does meanwhile, once it
            # follows them at all (a cart standing still is the stall judge's or the cap's,
            # never done by a prediction).
            ahead = here + in_flight(history, integral, now, s.lead_s + state.age_s)
            left = target - ahead
            if rate != 0.0 and (abs(left) <= tol / 2.0 or left * direction <= 0.0):
                break
            room = max(left * direction, 0.0)
            speed = min(top, math.sqrt(2.0 * brake * room), abs(last) + ramp * TICK_S)
            speed = max(speed, floor)
            last = direction * speed
            twist = Twist(0.0, last) if axis == "yaw" else Twist(last, 0.0)
            if not self._command(job, twist):
                return self._stop_leg(job, axis, origin, here, stopped=job.why)
            if progress is not None and abs(here - told) >= step:
                told = here
                progress(here - start)
            self.pause(job.cancel, TICK_S)
        return self._stop_leg(job, axis, origin, here)

    def _stop_leg(
        self,
        job: Job,
        axis: Literal["yaw", "x"],
        origin: Pose2D,
        here: float,
        *,
        stopped: str = "",
        blocked: str = "",
    ) -> LegEnd:
        """The leg's last twist (zero, unless halted: the halt sent the stop), the wheels let
        settle (the lead, then ``SETTLE_S``), and where they stood."""
        self._command(job, Twist(0.0, 0.0))
        deadline = self.clock() + self.settings.lead_s + SETTLE_S
        while self.clock() < deadline:
            state = self._base_state()
            if state is None:
                break
            here = offset(state.pose, origin, axis)
            if (
                abs(state.d_left_m) < STILL_M
                and abs(state.d_right_m) < STILL_M
                and not state.moving
            ):
                break
            self.pause(self._settling, TICK_S)  # never set: a halted leg still settles
        if job.cancel.is_set() and not stopped:
            stopped = job.why
        return LegEnd(here, blocked, stopped)

    def _check(
        self,
        axis: Literal["yaw", "x"],
        origin: Pose2D,
        pose: Pose2D,
        here: float,
        target: float,
        speed: float,
        now: float,
    ) -> tuple[str, str]:
        """(stopped, blocked) from the goal server's newest word: a drive, old or no marks, the
        footprint swept from here to the target (and what the cart moved since the marks' stamp)
        within ``clearance_m`` of a mark."""
        view = self.around.latest()
        if view is None:
            return "the goal server has said nothing", ""
        refusal = self._view_refusal(view, now)
        if refusal is not None:
            return refusal, ""
        if not self.settings.clearance_check:
            return "", ""
        points = in_origin_frame(view.points, pose, origin)
        # What the cart may have moved since the marks' stamp, past the target; none on the way
        # back toward where the motion began (the cart stood there).
        homeward = min(0.0, here) <= target <= max(0.0, here)
        age = (view.marks_age_s or 0.0) + now - view.at
        lag = 0.0 if homeward else abs(speed) * (age + self.settings.lead_s)
        end = target + math.copysign(lag, target - here)
        margin = self.settings.clearance_m
        if axis == "yaw":
            hit = turn_blocker(points, self.footprint, here, end, margin)
            return ("", f"the arc is blocked: {hit.text('deg')}") if hit is not None else ("", "")
        hit = drive_blocker(points, self.footprint, here, end, margin)
        return ("", f"the way is blocked: {hit.text('cm')}") if hit is not None else ("", "")


class _Steps:
    """:class:`pepin.expression.plan.TurnSteps` for one job of a :class:`WheelsExecutor`."""

    def __init__(self, wheels: WheelsExecutor, job: Job) -> None:
        """The executor that moves, the job it moves for."""
        self.wheels, self.job = wheels, job
        self.turned_deg = 0.0  # by the attempts so far: the progress told is the whole turn's

    def turn(self, deg: float) -> Step:
        """One attempt."""
        step = self.wheels.turn_leg(self.job, deg, self.turned_deg)
        self.turned_deg += step.amount
        return step

    def step_away(self, metres: float) -> Step:
        """Straight away from the nearest mark."""
        return self.wheels.step_away(self.job, metres)


def stalled(
    history: deque[tuple[float, float, float]],
    now: float,
    window_s: float,
    direction: float,
    least: float,
    axis: Literal["yaw", "x"],
    lead_s: float = 0.0,
) -> str:
    """Why a leg is stalled, "" while it is not: over the last ``window_s`` the cart made less
    than a third of what the twists in force then asked, at least ``least`` (``history`` of
    (time, commanded travel so far, measured position), oldest first). The twists in force over
    the window are those sent ``lead_s`` earlier (the transport delay): the last ``lead_s`` of
    commands has moved nothing yet and is not held against the cart (2026-10-13: a turn's first
    half second asked 18.6 deg at the ramp 180 of a cart that answers 0.19 s later at its own
    101 and had made 6.2, a false stall)."""
    if now - history[0][0] < window_s + lead_s:
        return ""

    def newest_before(t: float) -> tuple[float, float, float]:
        entry = history[0]
        for item in history:
            if item[0] > t:
                break
            entry = item
        return entry

    _, _, at_old = newest_before(now - window_s)
    _, commanded_now, _ = newest_before(now - lead_s)
    _, commanded_old, _ = newest_before(now - window_s - lead_s)
    at_now = history[-1][2]
    commanded = (commanded_now - commanded_old) * direction
    made = (at_now - at_old) * direction
    if commanded < least or made >= STALL_SHARE * commanded:
        return ""
    if axis == "yaw":
        return (f"stalled: {math.degrees(made):+.1f} of {math.degrees(commanded):.1f} deg in"
                f" {window_s:.1f} s")  # fmt: skip
    return f"stalled: {made * 100:+.1f} of {commanded * 100:.1f} cm in {window_s:.1f} s"


def in_flight(
    history: deque[tuple[float, float, float]], integral: float, now: float, lead_s: float
) -> float:
    """What the twists sent over the last ``lead_s`` move the cart on a leg's axis (signed):
    ``integral`` (the commanded travel so far, at ``now``) less the commanded travel at
    ``now - lead_s`` (``history``: (time, commanded travel, position), oldest first; before its
    first entry nothing had been commanded)."""
    then = now - lead_s
    before = 0.0
    for t, commanded, _ in history:
        if t > then:
            break
        before = commanded
    return integral - before


def measured_rate(reads: deque[tuple[float, float]], window_s: float) -> float:
    """The cart's rate on a leg's axis from its reads of the base's lines ((board stamp,
    position), newest last): over the oldest read at least ``window_s`` back, else over all of
    them; 0 before two reads (the cart stands when a leg begins). Reads older than twice the
    window are dropped."""
    if len(reads) < 2:
        return 0.0
    newest = reads[-1][0]
    while len(reads) > 2 and newest - reads[1][0] >= 2.0 * window_s:
        reads.popleft()
    oldest = next((r for r in reversed(reads) if newest - r[0] >= window_s), reads[0])
    dt = newest - oldest[0]
    return (reads[-1][1] - oldest[1]) / dt if dt > 0.0 else 0.0


def offset(pose: Pose2D, origin: Pose2D, axis: Literal["yaw", "x"]) -> float:
    """Where ``pose`` is from ``origin`` on ``axis``: the heading turned (radians, + left) or
    the travel along the origin's heading (metres, + forward)."""
    if axis == "yaw":
        return math.remainder(pose.theta - origin.theta, math.tau)
    return (pose.x - origin.x) * math.cos(origin.theta) + (pose.y - origin.y) * math.sin(
        origin.theta
    )


def in_origin_frame(points: Points, pose: Pose2D, origin: Pose2D) -> Points:
    """Points seen from ``pose`` (base_link there) as the cart at ``origin`` sees them."""
    if len(points) == 0:
        return points
    yaw = math.remainder(pose.theta - origin.theta, math.tau)
    c, s = math.cos(yaw), math.sin(yaw)
    dx, dy = pose.x - origin.x, pose.y - origin.y
    co, so = math.cos(-origin.theta), math.sin(-origin.theta)
    shift = np.array([co * dx - so * dy, so * dx + co * dy])
    rotated = np.column_stack(
        (c * points[:, 0] - s * points[:, 1], s * points[:, 0] + c * points[:, 1])
    )
    out: Points = rotated + shift
    return out


def standing(state: BaseState) -> str | None:
    """Why the cart is not standing for an expressive motion (a twist commanded: a drive or a
    teleop owns the wheels; the wheels rolling; the bus silent); None when it is."""
    if state.moving or abs(state.v) > SAME or abs(state.w) > SAME:
        twist = f"v {state.v:+.2f} m/s, w {state.w:+.2f} rad/s"
        return f"the wheels are commanded ({twist}): a drive or a teleop"
    if abs(state.d_left_m) > STILL_M or abs(state.d_right_m) > STILL_M:
        return "the cart is rolling"
    if not state.bus_ok:
        return "the servo bus did not answer"
    return None


class WheelsGuard:
    """The cascade's moments that stop the wheels at once (a :class:`pepin.expression.
    conversation.ConversationObserver`): a drive and the close stop every motion; the person
    starting to speak stops a gesture's (the body's turn toward the person speaking goes on: it
    is the speaker gaze's base layer, as the head's tracking is). Each is one call on the
    cascade's own loop: a flag and at most one non-blocking line."""

    def __init__(self, wheels: WheelsExecutor) -> None:
        """The executor it halts."""
        self.wheels = wheels

    def opened(self, wake: Any) -> None:
        """Nothing."""

    def speech_started(self, t: float) -> None:
        """The person speaks: a gesture's motion stops where it is."""
        self.wheels.hush("the person speaks")

    def speech_ended(self, heard: Any) -> None:
        """Nothing."""

    def part_queued(self, turn: int, index: int, part: Any) -> None:
        """Nothing."""

    def voice_started(self, turn: int) -> None:
        """Nothing."""

    def part_started(self, turn: int, index: int, stream_s: float) -> None:
        """Nothing."""

    def text_ahead(self, turn: int, sentence: int, text: str) -> None:
        """Nothing."""

    def part_spoken(self, turn: int, index: int) -> None:
        """Nothing."""

    def reply_cut(self, turn: int) -> None:
        """Nothing."""

    def reply_done(self, turn: int) -> None:
        """Nothing."""

    def moment(self, name: str, text: str = "") -> None:
        """Nothing."""

    def drive(self, active: bool, ran: bool = True) -> None:
        """A drive starts: the wheels are the drive's (its end, run or not, changes nothing)."""
        if active:
            self.wheels.halt("a drive starts")

    def closed(self, reason: str) -> None:
        """The conversation closed: nothing turns toward anyone."""
        self.wheels.halt(f"the conversation closed ({reason})")


def robot_wheels(
    host: str,
    *,
    load: Callable[[], WheelsSettings] = WheelsSettings.load,
    report: Report = say,
) -> WheelsExecutor:
    """The wheels on the robot: the base server at ``host`` (connected at the first motion),
    the goal server on this Mac, config/expression.json's ``wheels`` and config/base.json's
    footprint."""

    def link(listener: StateListener) -> BaseLink:
        from pepin.base_link import BaseClient

        client = BaseClient(host)
        client.on_state = listener
        return client.start()

    return WheelsExecutor(
        link, GoalServerView(), load=load, footprint=footprint_of(), report=report
    )
