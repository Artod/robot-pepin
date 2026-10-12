"""Fakes of every seam of the expression layer, for tests and dry runs: each one records what it
was asked in ``calls`` and answers what its fields script, with no thread, socket or clock of its
own."""

from __future__ import annotations

import math
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from pepin.audio_link import DoaReading
from pepin.base_link import BaseState, StateListener
from pepin.expression.animation import Clip, Layer, PartName
from pepin.expression.clearance import View
from pepin.expression.conversation import Heard
from pepin.expression.executors import Outcome, WheelMove, WheelReport, WheelsDone, WheelsMoved
from pepin.expression.head import ClipLook
from pepin.expression.markers import Part
from pepin.expression.speaker import (
    BodySighting,
    Ease,
    FaceFrame,
    FaceSighting,
    GazeAnswer,
    HeadView,
)
from pepin.expression.trajectory import ArmAnswer, ArmStatus, Trajectory
from pepin.gaze import Aim, Speed
from pepin.kinematics import Twist
from pepin.odometry import Pose2D

Call = tuple[str, tuple[Any, ...]]


@dataclass
class FakePlayout:
    """The speaker's clock: ``now`` stream seconds played, ``marks`` by (turn, part)."""

    now: float = 0.0
    marks: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)

    def played(self) -> float:
        """Stream seconds played."""
        return self.now

    def mark(self, turn: int, index: int) -> tuple[float, float] | None:
        """The part's span, if fed."""
        return self.marks.get((turn, index))


@dataclass
class RecordingObserver:
    """A conversation observer that only records the moments."""

    calls: list[Call] = field(default_factory=list)

    def opened(self, wake: Heard) -> None:
        """Recorded."""
        self.calls.append(("opened", (wake,)))

    def speech_started(self, t: float) -> None:
        """Recorded."""
        self.calls.append(("speech_started", (t,)))

    def speech_ended(self, heard: Heard) -> None:
        """Recorded."""
        self.calls.append(("speech_ended", (heard,)))

    def part_queued(self, turn: int, index: int, part: Part) -> None:
        """Recorded."""
        self.calls.append(("part_queued", (turn, index, part)))

    def voice_started(self, turn: int) -> None:
        """Recorded."""
        self.calls.append(("voice_started", (turn,)))

    def part_spoken(self, turn: int, index: int) -> None:
        """Recorded."""
        self.calls.append(("part_spoken", (turn, index)))

    def reply_cut(self, turn: int) -> None:
        """Recorded."""
        self.calls.append(("reply_cut", (turn,)))

    def reply_done(self, turn: int) -> None:
        """Recorded."""
        self.calls.append(("reply_done", (turn,)))

    def moment(self, name: str) -> None:
        """Recorded."""
        self.calls.append(("moment", (name,)))

    def drive(self, active: bool) -> None:
        """Recorded."""
        self.calls.append(("drive", (active,)))

    def closed(self, reason: str) -> None:
        """Recorded."""
        self.calls.append(("closed", (reason,)))


@dataclass
class FakeFlag:
    """A live switch named ``flag``; ``refusal`` set: every switch is refused with it."""

    flag: str
    on: bool = False
    refusal: str | None = None

    def switch(self, on: bool) -> str | None:
        """Set, unless scripted to refuse."""
        if self.refusal is None:
            self.on = on
        return self.refusal


@dataclass
class FakeExecutor:
    """A body part: ``why_not`` scripts :meth:`ready`; every clip played is kept."""

    part: PartName
    latency_s: float = 0.0
    why_not: str | None = None
    calls: list[Call] = field(default_factory=list)

    def ready(self) -> str | None:
        """As scripted."""
        return self.why_not

    def play(self, clip: Clip) -> Outcome:
        """Recorded; refused when not ready."""
        self.calls.append(("play", (clip,)))
        if self.why_not is not None:
            return Outcome(False, self.why_not)
        return Outcome(True, seconds=clip.duration_s)

    def fade(self, layer: Layer, seconds: float) -> None:
        """Recorded."""
        self.calls.append(("fade", (layer, seconds)))

    def rest(self) -> None:
        """Recorded."""
        self.calls.append(("rest", ()))


@dataclass
class FakeFace:
    """The face: every expression shown is kept."""

    calls: list[Call] = field(default_factory=list)

    def express(
        self,
        name: str,
        *,
        intensity: float = 1.0,
        hold_s: float | None = None,
        transition_ms: float | None = None,
    ) -> None:
        """Recorded."""
        self.calls.append(("express", (name, intensity, hold_s, transition_ms)))


@dataclass
class FakeArmLink:
    """The board's arm: ``state`` and ``answers`` script what it says back."""

    state: ArmStatus | None = None
    answers: dict[str, ArmAnswer] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)

    def send(self, trajectory: Trajectory) -> None:
        """Recorded."""
        self.calls.append(("send", (trajectory,)))

    def rest(self) -> None:
        """Recorded."""
        self.calls.append(("rest", ()))

    def halt(self) -> None:
        """Recorded."""
        self.calls.append(("halt", ()))

    def status(self) -> ArmStatus | None:
        """As scripted."""
        return self.state

    def answer(self, trajectory_id: str) -> ArmAnswer | None:
        """As scripted."""
        return self.answers.get(trajectory_id)


@dataclass
class FakeGazeDoor:
    """The arbiter's door for one source: aims are answered ``done`` at the aim asked (or with
    the next of ``refusals``); ``view`` scripts the state; ``live`` what a renewal finds."""

    view: HeadView | None = None
    live: int = 1
    refusals: deque[GazeAnswer] = field(default_factory=deque)
    calls: list[Call] = field(default_factory=list)

    def aim(self, aim: Aim, *, speed: Speed, ttl_s: float, ease: Ease | None = None) -> GazeAnswer:
        """Recorded; done, unless a refusal is scripted."""
        self.calls.append(("aim", (aim, speed, ttl_s, ease)))
        if self.refusals:
            return self.refusals.popleft()
        return GazeAnswer("done", head=aim)

    def renew(self) -> int:
        """Recorded; as scripted."""
        self.calls.append(("renew", ()))
        return self.live

    def release(self) -> None:
        """Recorded."""
        self.calls.append(("release", ()))

    def state(self) -> HeadView | None:
        """As scripted."""
        return self.view


@dataclass
class FakeHeadDoor:
    """The arbiter's door for head overlays: every overlay plays, unless the next of
    ``refusals`` (a reason, or None for one that plays) is scripted."""

    refusals: deque[str | None] = field(default_factory=deque)
    calls: list[Call] = field(default_factory=list)

    def look(self, request: ClipLook) -> str | None:
        """Recorded; plays, unless a refusal is scripted."""
        self.calls.append(("look", (request,)))
        return self.refusals.popleft() if self.refusals else None

    def withdraw(self, source: str, fade_s: float) -> None:
        """Recorded."""
        self.calls.append(("withdraw", (source, fade_s)))

    def looks(self) -> list[ClipLook]:
        """The overlays asked for, in order."""
        return [args[0] for name, args in self.calls if name == "look"]


@dataclass
class FakeFaceFinder:
    """Frames in the order scripted; None once they run out."""

    frames: deque[FaceFrame | None] = field(default_factory=deque)
    looks: int = 0
    asked: list[float | None] = field(default_factory=list)  # each look's not_before

    def look(self, not_before: float | None = None) -> FaceFrame | None:
        """The next scripted frame."""
        self.looks += 1
        self.asked.append(not_before)
        return self.frames.popleft() if self.frames else None


@dataclass
class FakePerson:
    """Someone in the room for :class:`FakeScene`, in the cart's frame (degrees): their face's
    bearing and tilt (+ down), its width (0: the face is not seen, turned away), whether their
    body is seen, where a body detector puts their head (default: the face's place) and who."""

    pan_deg: float
    tilt_deg: float = 3.0
    face_px: float = 60.0
    body: bool = False
    head_pan_deg: float | None = None
    head_tilt_deg: float | None = None
    who: str | None = None


@dataclass
class FakeScene:
    """The head camera over :class:`FakePerson` s: a look sees each face and body within
    ``half_pan_deg`` and ``half_tilt_deg`` of where ``door``'s state says the head points (move
    the people between looks to make them walk)."""

    door: Any
    people: list[FakePerson] = field(default_factory=list)
    half_pan_deg: float = 30.0
    half_tilt_deg: float = 25.0
    looks: int = 0
    asked: list[float | None] = field(default_factory=list)  # each look's not_before

    def look(self, not_before: float | None = None) -> FaceFrame | None:
        """The faces and bodies in view now."""
        self.looks += 1
        self.asked.append(not_before)
        view = self.door.state()
        if view is None or view.head is None:
            return None
        head_pan, head_tilt = math.degrees(view.head.pan_rad), math.degrees(view.head.tilt_rad)
        faces, bodies = [], []
        for p in self.people:
            pan = (p.pan_deg - head_pan + 180.0) % 360.0 - 180.0
            tilt = p.tilt_deg - head_tilt
            if abs(pan) > self.half_pan_deg or abs(tilt) > self.half_tilt_deg:
                continue
            if p.face_px > 0.0:
                faces.append(FaceSighting(pan, tilt, p.face_px, 0.8, p.who))
            if p.body:
                hpan = (p.pan_deg if p.head_pan_deg is None else p.head_pan_deg) - head_pan
                htilt = (p.tilt_deg if p.head_tilt_deg is None else p.head_tilt_deg) - head_tilt
                bodies.append(BodySighting(pan, 12.0, hpan, htilt, 40.0, 0.7))
        return FaceFrame(0.0, tuple(faces), bodies=tuple(bodies))


@dataclass
class FakeDoa:
    """Direction readings as scripted (board clock)."""

    readings: list[DoaReading] = field(default_factory=list)

    def latest_doa(self) -> DoaReading | None:
        """The newest scripted reading."""
        return self.readings[-1] if self.readings else None

    def doa_between(self, t0: float, t1: float) -> list[DoaReading]:
        """The scripted readings within the window."""
        return [r for r in self.readings if t0 <= r.stamp_s <= t1]


@dataclass
class FakeWheels:
    """Wheel motions recorded and left running: a test moves them on (:meth:`move_on`) and ends
    them (:meth:`finish`) when it wants, so a caller is seen to return before any of it."""

    calls: list[Call] = field(default_factory=list)
    running: list[tuple[WheelMove, WheelsDone, WheelsMoved | None]] = field(default_factory=list)

    def start(self, request: WheelMove, done: WheelsDone, moved: WheelsMoved | None = None) -> None:
        """Recorded; it runs until :meth:`finish`."""
        self.calls.append(("start", (request,)))
        self.running.append((request, done, moved))

    def halt(self, why: str) -> None:
        """Recorded (the running motion ends when the test finishes it)."""
        self.calls.append(("halt", (why,)))

    def move_on(self, deg: float) -> None:
        """The newest motion has turned ``deg`` so far."""
        _, _, moved = self.running[-1]
        if moved is not None:
            moved(deg)

    def finish(self, report: WheelReport | None = None) -> None:
        """The newest motion ends (in full unless ``report`` says otherwise)."""
        request, done, _ = self.running.pop()
        done(report if report is not None else WheelReport(True, request.turn_deg, attempts=1))


class FakeClock:
    """A clock that moves only when told."""

    def __init__(self, now: float = 100.0) -> None:
        """Starts at ``now``."""
        self.now = now

    def __call__(self) -> float:
        """The time."""
        return self.now


class FakeBase:
    """The board's base server as the wheels drive it (:class:`pepin.expression.wheels.BaseLink`),
    on a :class:`FakeClock`: a command lands ``lag_s`` after it is sent, the wheels' odometry
    integrates the twist applied (none of a stalled axis), a state line goes out every 20 ms to
    the listener, and the board's 0.5 s deadman stops a twist nobody renews. Tests script things
    at times (:meth:`at`): someone else's stop, a stall, the arm starting."""

    TRACK_M = 0.544

    def __init__(self, clock: FakeClock, *, lag_s: float = 0.04, line_s: float = 0.02) -> None:
        """At the origin, standing, armed off, its first line already out."""
        self.clock, self.lag_s, self.line_s = clock, lag_s, line_s
        self.pose = Pose2D(0.0, 0.0, 0.0)
        self.applied = Twist(0.0, 0.0)
        self.arriving: deque[tuple[float, Twist, str]] = deque()
        self.listener: StateListener | None = None
        self.sent: list[tuple[float, str, Twist]] = []  # (time, twist|stop, twist) as sent
        self.stalled_turns = False
        self.stalled_drives = False
        self.drop_ours = False  # our lines lost on the way (the WiFi)
        self.deadman = False
        self.bus_ok = True
        self.arm_phase: str | None = None
        self.travel = (0.0, 0.0)  # the wheels' travel in the last line
        self.silent = False  # the base server says nothing (no line at all)
        self.events: list[tuple[float, Callable[[FakeBase], None]]] = []
        self.lines: list[tuple[float, BaseState]] = []  # every line, with its time
        self._foreign: tuple[Twist, bool] | None = None
        self._last_cmd = clock()
        self._line: BaseState | None = None
        self._line_at = clock()
        self.line()

    def attach(self, listener: StateListener) -> FakeBase:
        """The link the wheels open: ``listener`` hears every line."""
        self.listener = listener
        return self

    def at(self, t: float, action: Callable[[FakeBase], None]) -> None:
        """Do ``action`` when the clock reaches ``t``."""
        self.events.append((t, action))

    def set_twist(self, twist: Twist) -> None:
        """One twist line, landing ``lag_s`` later."""
        self.sent.append((self.clock(), "twist", twist))
        if not self.drop_ours:
            self.arriving.append((self.clock() + self.lag_s, twist, "ours"))

    def stop(self) -> None:
        """One stop line."""
        self.sent.append((self.clock(), "stop", Twist(0.0, 0.0)))
        if not self.drop_ours:
            self.arriving.append((self.clock() + self.lag_s, Twist(0.0, 0.0), "ours"))

    def foreign(self, twist: Twist, *, keep: bool = False) -> None:
        """Someone else's command, applied now and after this step's own arrivals, so a line
        shows it (the red button's one stop: a zero twist); ``keep``: a driver that goes on
        commanding (Nav2, the teleop) every step."""
        self.applied, self._last_cmd, self.deadman = twist, self.clock(), False
        self._foreign = (twist, keep)

    def state(self, now: float | None = None) -> BaseState | None:
        """The newest line, aged."""
        if self._line is None:
            return None
        now = self.clock() if now is None else now
        return replace(self._line, age_s=now - self._line_at)

    def advance(self, seconds: float) -> None:
        """Let ``seconds`` pass: commands land, scripted events fire, the cart moves."""
        end = self.clock.now + seconds
        while self.clock.now < end - 1e-12:
            step = min(self.line_s, end - self.clock.now)
            self.clock.now += step
            now = self.clock.now
            for t, action in [e for e in self.events if e[0] <= now]:
                self.events.remove((t, action))
                action(self)
            while self.arriving and self.arriving[0][0] <= now:
                _, twist, _ = self.arriving.popleft()
                self.applied, self._last_cmd, self.deadman = twist, now, False
            if self._foreign is not None:
                twist, keep = self._foreign
                self.applied, self._last_cmd, self.deadman = twist, now, False
                self._foreign = self._foreign if keep else None
            moving = abs(self.applied.linear) > 0 or abs(self.applied.angular) > 0
            if moving and now - self._last_cmd > 0.5:
                self.applied, self.deadman = Twist(0.0, 0.0), True
            v = 0.0 if self.stalled_drives else self.applied.linear
            w = 0.0 if self.stalled_turns else self.applied.angular
            theta = self.pose.theta + w * step
            self.pose = Pose2D(
                self.pose.x + v * math.cos(theta) * step,
                self.pose.y + v * math.sin(theta) * step,
                theta,
            )
            half = w * self.TRACK_M / 2.0
            self.travel = ((v - half) * step, (v + half) * step)
            self.line()

    def line(self) -> None:
        """One state line now, the fields as they stand (none while ``silent``)."""
        if self.silent:
            self._line = None
            return
        a = self.applied
        moving = abs(a.linear) > 0 or abs(a.angular) > 0
        self._line = BaseState(self.pose, self.travel[0], self.travel[1], a.linear, a.angular,
                               moving, True, self.deadman, self.bus_ok, 1.0, self.clock(), 0.0,
                               arm_phase=self.arm_phase)  # fmt: skip
        self._line_at = self.clock()
        self.lines.append((self._line_at, self._line))
        if self.listener is not None:
            self.listener(self._line, self.clock())

    def pause(self, event: threading.Event, seconds: float) -> None:
        """:class:`pepin.expression.wheels.WheelsExecutor`'s pause on this clock: time passes
        unless the event is already set."""
        if not event.is_set():
            self.advance(seconds)

    def twists(self) -> list[Twist]:
        """Every twist sent (stops as zero twists)."""
        return [twist for _, _, twist in self.sent]


@dataclass
class FakeView:
    """The goal server's word (:class:`pepin.expression.clearance.Surroundings`), always fresh
    on the base's clock: whether Nav2 navigates, the scan's age, and fixed points of the room
    (world frame) seen from where the fake base's cart stands; ``why`` for a goal server away;
    ``cancelled_at`` the base clock's time of its last cancel (ros/stop.sh)."""

    base: FakeBase
    navigating: bool | None = False
    scan_age_s: float | None = 0.05
    world: list[tuple[float, float]] = field(default_factory=list)
    why: str = ""
    silent: bool = False  # never answers
    cancelled_at: float | None = None
    watched: list[bool] = field(default_factory=list)

    def watch(self, on: bool) -> None:
        """Recorded."""
        self.watched.append(on)

    def latest(self) -> View | None:
        """A fresh view, the room's points in the cart's frame now."""
        if self.silent:
            return None
        p = self.base.pose
        c, s = math.cos(-p.theta), math.sin(-p.theta)
        points = [
            (c * (x - p.x) - s * (y - p.y), s * (x - p.x) + c * (y - p.y)) for x, y in self.world
        ]
        array = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        now = self.base.clock()
        cancel = None if self.cancelled_at is None else now - self.cancelled_at
        return View(now, self.navigating, self.scan_age_s, array, self.why, cancel)


if TYPE_CHECKING:  # every fake, and the real clients that already exist, satisfy their seams

    def _conforms() -> None:
        from pepin.audio_link import AudioClient
        from pepin.base_link import BaseClient
        from pepin.expression.clearance import GoalServerView, Surroundings
        from pepin.expression.conversation import ConversationObserver, LiveFlag
        from pepin.expression.door import OverlayDoor
        from pepin.expression.executors import Executor, FaceChannel, NoBody, NoWheels, Wheels
        from pepin.expression.markers import Playout
        from pepin.expression.parts import HeadDoor
        from pepin.expression.speaker import DoaSource, FaceFinder, GazeDoor
        from pepin.expression.trajectory import ArmLink
        from pepin.expression.wheels import BaseLink, WheelsExecutor, WheelsGuard
        from pepin.head_link import HeadClient

        playout: Playout = FakePlayout()
        executor = WheelsExecutor(lambda listener: FakeBase(FakeClock()), GoalServerView())
        observers: list[ConversationObserver] = [RecordingObserver(), WheelsGuard(executor)]
        flag: LiveFlag = FakeFlag("gestures")
        parts: list[Executor] = [FakeExecutor("arm"), NoBody(), executor]
        faces: list[FaceChannel] = [FakeFace(), HeadClient("board", source="voice")]
        arm: ArmLink = FakeArmLink()
        door: GazeDoor = FakeGazeDoor()
        finder: list[FaceFinder] = [FakeFaceFinder(), FakeScene(FakeGazeDoor())]
        doa: list[DoaSource] = [FakeDoa(), AudioClient("board")]
        wheels: list[Wheels] = [FakeWheels(), NoWheels(), executor]
        heads: list[HeadDoor] = [FakeHeadDoor(), OverlayDoor()]
        bases: list[BaseLink] = [FakeBase(FakeClock()), BaseClient("board")]
        views: list[Surroundings] = [FakeView(FakeBase(FakeClock())), GoalServerView()]
        del playout, observers, flag, parts, faces, arm, door, finder, doa, wheels, heads
        del bases, views
