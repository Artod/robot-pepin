"""The gaze arbiter: the one owner of every head decision.

Consumers do not move the neck; they ASK for a look (:class:`Look`: which views, how many still
depth frames at each, how long to hold, which band, how long the request may live) and the
:class:`Arbiter` alone writes the neck, through a :class:`HeadDriver`. The rules:

* lower band wins (:data:`BANDS`); within a band the holder keeps the head unless the newcomer
  sets ``preempt``, or the holder has been answered and only dwells; a request from a source that
  already holds one replaces it (a consumer changing its mind);
* a request past its TTL is dropped, and the head falls to the next live request, finally home:
  home is the standing request nobody has to make, so "look, then return" needs no code in a
  tool;
* a look is answered once the head has settled on the last view and ``frames`` depth frames
  stamped after settling (and after one frame period of blur) were fused; it then holds the head
  ``dwell_s`` more before letting go;
* a GLANCE (``hold_s`` > 0) is atomic once it holds the head: it is answered at ``frames`` frames
  or ``hold_s`` after settling, whichever comes first, and until then neither its TTL, nor a
  release, nor a newcomer of its own band takes the head from it; a better band does, and so
  does its own source's :meth:`Arbiter.withdraw`;
* a look whose frames the depth gate has been dropping as too dark (``dark_frame``, depth_stream's
  ``/depth/dark``) for ``dark_patience_s`` without one fused frame between gives up: ended "dark,
  no frames", status expired, instead of waiting out its TTL in a room the auto exposure cannot
  light; a window's transition (1-2 s of dark frames that then turn light) still completes;
* every look that held the head is booked when it lets go (:class:`HeldLook`): how long, how it
  ended, and the frames fused while it held the head still (late frames are counted for
  :data:`LATE_S`), for the node's report;
* the head is BLIND from the write that starts a move until it has settled plus one frame period:
  :meth:`Arbiter.state` says so, with the interval's two ends, for every frame consumer to gate on.

Pure: time is passed in, the encoders' readings and the fused frames' stamps are fed in, and the
ROS node (``pepin_bringup.gaze``) wires the topics, the base server and the doors. Two drivers
speak to the base server: :class:`BaseServerHead` (``neck_goto``/``neck_home``: one move at a
time, refused while the wheels turn) and :class:`NeckTargetHead` (``neck_target`` with a lease,
written while driving).
"""

from __future__ import annotations

import json
import math
import threading
import uuid
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, cast

from pepin.neck import (
    RAD_PER_TICK,
    NeckAngles,
    NeckConfig,
    angle_limits,
    camera_pose,
    joint_angles,
    ticks_for,
)

Kind = Literal["point", "direction", "angles", "scan", "track", "home"]
Speed = Literal["saccade", "slow", "return"]  # "return": the slow way home after a drive
Status = Literal["queued", "granted", "done", "denied", "preempted", "expired"]
Phase = Literal["home", "still", "saccade", "track", "returning"]

KINDS: tuple[str, ...] = ("point", "direction", "angles", "scan", "track", "home")
SPEEDS: tuple[str, ...] = ("saccade", "slow")

# The bands, lowest first: who asks, and what they win over. HOME is the arbiter's own.
OPERATOR, NAVIGATION, PERSON, SENSOR, DRIVING, IDLE, HOME = range(7)
BANDS = {
    OPERATOR: "operator",
    NAVIGATION: "navigation",
    PERSON: "person",
    SENSOR: "sensor",
    DRIVING: "driving",
    IDLE: "idle",
}
STILL_RAD = 2 * RAD_PER_TICK  # two encoder readings this close are one pose (the encoder's jitter)
RETRY_S = 0.2  # a write the base server refused for now is tried again after this
FRAMES_KEPT = 1024  # the fused frames' stamps kept: a look held for a minute and more at 10 fps
DARK_ENDED = "dark, no frames"  # how a look that gave up on dark frames is booked and answered
LATE_S = 1.5  # a look's frames are counted this long after it let go: the fusion's latency
# What a person's look is told during a drive by a base server that moves the neck only at rest.
DRIVE_REFUSAL = (
    "the head does not move during a drive (the base server moves the neck only at rest):"
    " wait for the drive to end, or cancel it"
)


@dataclass(frozen=True)
class Aim:
    """Where the head points: pan left of the nose (+), tilt (pitch) below level (+), radians."""

    pan_rad: float
    tilt_rad: float

    def off(self, other: Aim) -> float:
        """The larger of the two joint differences, radians."""
        return max(abs(self.pan_rad - other.pan_rad), abs(self.tilt_rad - other.tilt_rad))

    def as_dict(self) -> dict[str, float]:
        """Radians and degrees, rounded, for a JSON answer."""
        return {
            "pan_rad": round(self.pan_rad, 4),
            "tilt_rad": round(self.tilt_rad, 4),
            "pan_deg": round(math.degrees(self.pan_rad), 1),
            "tilt_deg": round(math.degrees(self.tilt_rad), 1),
        }


@dataclass(frozen=True)
class HeadReading:
    """One encoder reading of the neck as angles, stamped with the board's clock."""

    pan_rad: float
    tilt_rad: float
    stamp: float

    @property
    def aim(self) -> Aim:
        """The reading as an :class:`Aim`."""
        return Aim(self.pan_rad, self.tilt_rad)


@dataclass(frozen=True)
class Refusal:
    """A write the base server refused: why, and whether trying again shortly may succeed."""

    why: str
    transient: bool


@dataclass(frozen=True)
class GazeSettings:
    """The arbiter's numbers (config/knobs.json's ``gaze`` block, live); ``ttl_s`` is the
    default TTL of the operator, navigation, person, sensor, driving and idle bands;
    ``dark_patience_s`` how long a look waits on frames the depth gate drops as dark (0: until
    its TTL)."""

    frames: int = 3
    settle_tol_deg: float = 1.0
    move_timeout_s: float = 3.0
    frame_period_s: float = 0.105
    ttl_s: tuple[float, float, float, float, float, float] = (0.5, 3.0, 10.0, 2.0, 0.5, 20.0)
    dark_patience_s: float = 1.5

    def ttl_for(self, band: int) -> float:
        """The default TTL of a band's request."""
        return self.ttl_s[min(max(band, 0), len(self.ttl_s) - 1)]


def new_id() -> str:
    """A short request id."""
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True)
class Look:
    """One request: its views (empty: home), the still frames wanted at each, the hold after the
    last, the band, the TTL, the speed, whether it takes the head from its own band, and
    ``hold_s``: above 0, a glance (answered at ``frames`` or ``hold_s`` after settling, atomic)."""

    source: str
    views: tuple[Aim, ...]
    band: int
    frames: int
    dwell_s: float
    ttl_s: float
    speed: Speed = "saccade"
    preempt: bool = False
    kind: Kind = "angles"
    id: str = field(default_factory=new_id)
    hold_s: float = 0.0


@dataclass
class Outcome:
    """What a request is answered with: its status, why, where the head got to and when it
    settled (board clock), the frames seen after settling and how long it took."""

    id: str
    source: str
    status: Status
    reason: str = ""
    reached: bool = False
    pan_rad: float | None = None
    tilt_rad: float | None = None
    settled_stamp: float | None = None
    frames_seen: int = 0
    took_ms: float = 0.0

    def as_dict(self) -> dict[str, Any]:
        """The result as the doors send it."""
        return {
            "request_id": self.id,
            "source": self.source,
            "status": self.status,
            "reason": self.reason,
            "reached": self.reached,
            "pan_rad": None if self.pan_rad is None else round(self.pan_rad, 4),
            "tilt_rad": None if self.tilt_rad is None else round(self.tilt_rad, 4),
            "settled_stamp": self.settled_stamp,
            "frames_seen": self.frames_seen,
            "took_ms": round(self.took_ms, 1),
        }


class HeadDriver(Protocol):
    """The neck as the arbiter writes it: one target at a time, answered asynchronously."""

    @property
    def moves_while_driving(self) -> bool:
        """Whether a write is accepted while the wheels turn."""
        ...

    def blocked(self, now: float) -> Refusal | None:
        """Why a write cannot go out now, or ``None`` when it can."""
        ...

    def write(self, aim: Aim | None, *, speed: Speed, hold: bool, now: float) -> None:
        """Send the head to ``aim`` (``None``: home); ``hold`` keeps the servos energised."""
        ...

    def take_refusal(self) -> Refusal | None:
        """The newest write's refusal, once."""
        ...

    def take_arrival(self) -> HeadReading | None:
        """The newest write's confirmed arrival, once (a driver without one answers None)."""
        ...

    def keep(self, now: float) -> None:
        """Called every step: renew whatever the driver must renew (a lease)."""
        ...


@dataclass
class _Held:
    """A request as the arbiter holds it."""

    look: Look
    created: float
    on_done: Callable[[Outcome], None] | None
    view: int = 0
    answered: bool = False
    answered_at: float = 0.0
    waiting_since: float | None = None
    settled: HeadReading | None = None
    frames_seen: int = 0
    took_head: float | None = None  # when it first held the head
    settled_at: float | None = None  # when the arbiter first saw the head settled for it
    still_from: float | None = None  # its frames are fused after this (board clock)
    let_go: bool = False  # released while glancing: dropped once answered


@dataclass
class HeldLook:
    """One look that held the head, booked when it let go: its aim, when it took and left the
    head, how it ended, and the frames fused while it held the head still."""

    id: str
    source: str
    band: int
    aim: Aim
    since: float
    until: float
    ended: str
    still_from: float | None  # None: the head never settled for it, so it wrote nothing
    frames: int = 0

    def text(self) -> str:
        """``nav.path +38/24 deg 0.31 s preempted``."""
        pan, tilt = math.degrees(self.aim.pan_rad), math.degrees(self.aim.tilt_rad)
        return (
            f"{self.source} {pan:+.0f}/{tilt:.0f} deg {self.until - self.since:.2f} s {self.ended}"
        )


@dataclass
class _Write:
    """The arbiter's last accepted write and what became of it."""

    aim: Aim
    home: bool
    at: float  # when it went out: the blind interval starts here
    owner: str  # the request that wrote it ("" for the arbiter's own home)
    settled: HeadReading | None = None  # the reading it settled on; None while the head moves
    timed_out: bool = False


@dataclass(frozen=True)
class GazeState:
    """The one gate (``/gaze/state``): the phase, the head, the request holding it, and the
    blind interval (``blind_from`` the write that started the move, ``blind_until`` settled plus
    one frame period, ``None`` while the head still moves)."""

    phase: Phase
    head: Aim | None
    target: Aim | None
    since: float
    request_id: str
    source: str
    band: int
    blind: bool
    blind_from: float | None
    blind_until: float | None

    def as_dict(self) -> dict[str, Any]:
        """The JSON ``/gaze/state`` carries."""
        return {
            "phase": self.phase,
            "pan_rad": None if self.head is None else round(self.head.pan_rad, 4),
            "tilt_rad": None if self.head is None else round(self.head.tilt_rad, 4),
            "target": None if self.target is None else self.target.as_dict(),
            "since": round(self.since, 3),
            "request_id": self.request_id,
            "source": self.source,
            "band": self.band,
            "blind": self.blind,
            "blind_from": None if self.blind_from is None else round(self.blind_from, 3),
            "blind_until": None if self.blind_until is None else round(self.blind_until, 3),
        }

    def to_json(self) -> str:
        """:meth:`as_dict` as one line."""
        return json.dumps(self.as_dict())


class Arbiter:
    """Bands, TTLs, preemption, home and the blind interval over one :class:`HeadDriver`.

    Thread-safe: requests arrive from the doors' threads, readings and frames from the
    subscriptions, :meth:`step` from a timer; answers are delivered outside the lock.
    """

    def __init__(self, driver: HeadDriver, home: Aim, settings: GazeSettings | None = None) -> None:
        self._driver = driver
        self.home = home
        self.settings = settings if settings is not None else GazeSettings()
        self._lock = threading.RLock()
        self._held: list[_Held] = []
        self._active: _Held | None = None
        self._head: _Write | None = None  # None: nothing written since the start
        self._before: _Write | None = None  # the belief before the newest write (a refusal)
        self._retry_at = 0.0
        self._readings: deque[HeadReading] = deque(maxlen=2)
        self._frames: deque[float] = deque(maxlen=FRAMES_KEPT)
        self._dark: deque[float] = deque(maxlen=FRAMES_KEPT)  # frames the depth gate found dark
        self._closing: list[HeldLook] = []  # let go less than LATE_S ago: still counting
        self._closed: list[HeldLook] = []
        self._phase: Phase = "home"
        self._since = 0.0
        self._outbox: list[tuple[Callable[[Outcome], None], Outcome]] = []
        self.counts: Counter[str] = Counter()

    @property
    def driver(self) -> HeadDriver:
        """The driver this arbiter writes through."""
        return self._driver

    # ---- inputs ------------------------------------------------------------------------------
    def observe(self, reading: HeadReading) -> None:
        """One encoder reading (``/neck/state``)."""
        with self._lock:
            self._readings.append(reading)

    def frame(self, stamp: float) -> None:
        """One depth frame fused into the volume, at its own stamp (``/fusion/frame``)."""
        with self._lock:
            self._frames.append(stamp)
            for booked in self._closing:
                if _claims(booked, stamp):
                    booked.frames += 1

    def dark_frame(self, stamp: float) -> None:
        """One frame the depth stream's gate dropped as too dark, at its own stamp
        (``/depth/dark``): it will never be fused, and a look waiting on it learns why."""
        with self._lock:
            self._dark.append(stamp)

    def submit(
        self, look: Look, now: float, on_done: Callable[[Outcome], None] | None = None
    ) -> Outcome:
        """Take a request; the immediate answer is ``granted``, ``queued`` or ``denied``, and
        ``on_done`` gets the final one."""
        with self._lock:
            answer = self._take(look, now, on_done)
        self._deliver()
        return answer

    def release(self, now: float, by: str, *, keep: Callable[[Look], bool]) -> int:
        """Drop every request ``keep`` refuses, as preempted ``by``; how many went. A glance under
        way is not cut: it goes once answered."""
        with self._lock:
            refused = [h for h in self._held if not keep(h.look)]
            for held in refused:
                if self._glancing(held):
                    held.let_go = True
                else:
                    self._finish(held, "preempted", f"by {by}", now)
        self._deliver()
        return sum(1 for h in refused if not h.let_go)

    def withdraw(self, source: str, now: float, by: str) -> int:
        """Take back every request of ``source``, as preempted ``by``, a glance under way too:
        its own asker's word, which :meth:`release` is not (the reverse look let go at the first
        forward command); how many went."""
        with self._lock:
            mine = [h for h in self._held if h.look.source == source]
            for held in mine:
                self._finish(held, "preempted", f"by {by}", now)
        self._deliver()
        return len(mine)

    def holds(self, source: str) -> bool:
        """Whether a request of ``source`` holds the head."""
        with self._lock:
            return self._active is not None and self._active.look.source == source

    def glancing(self, source: str) -> bool:
        """Whether ``source`` holds the head with a glance not yet answered."""
        with self._lock:
            active = self._active
            return active is not None and active.look.source == source and self._glancing(active)

    def take_looks(self, now: float) -> list[HeldLook]:
        """The looks that let go of the head more than :data:`LATE_S` ago, once each."""
        with self._lock:
            self._close_books(now)
            taken, self._closed = self._closed, []
        return taken

    def adopt(self, now: float, tol_rad: float) -> bool:
        """Take the head as the encoders say it is when that is more than ``tol_rad`` off what
        the arbiter last wrote (a jog, a hand, ``ros/neck.sh``): with nothing holding it, the
        next step then brings it home. Whether it was adopted."""
        with self._lock:
            if not self._readings:
                return False
            reading = self._readings[-1]
            believed = self._head.aim if self._head is not None else self.home
            if reading.aim.off(believed) <= tol_rad:
                return False
            self._head = _Write(reading.aim, False, now, "", settled=reading)
            self.counts["adopted"] += 1
            return True

    def looked_at(self) -> Aim | None:
        """Where the newest write sent the head for a look; ``None`` when it sent it home, the
        encoders found it elsewhere (:meth:`adopt`) or nothing was written."""
        with self._lock:
            head = self._head
            return head.aim if head is not None and head.owner and not head.home else None

    def renew(self, source: str, now: float) -> int:
        """Restart the TTL of ``source``'s requests, and the dwell of one already answered
        (``see`` keeps a look; path gaze keeps its aim); how many."""
        with self._lock:
            mine = [h for h in self._held if h.look.source == source]
            for held in mine:
                held.created = now
                if held.answered:
                    held.answered_at = now
        return len(mine)

    # ---- the decision ------------------------------------------------------------------------
    def step(self, now: float) -> None:
        """Expire, choose, and drive the head toward the chosen request's view or home."""
        with self._lock:
            self._close_books(now)
            self._expire(now)
            self._choose(now)
            self._take_refusal(now)
            self._settle(now)
            active = self._active
            if active is None:
                self._go_home(now)
            elif not active.answered:
                self._serve(active, now)
            phase = self._current_phase()
            if phase != self._phase:
                self._phase, self._since = phase, self._phase_stamp(phase, now)
            self._driver.keep(now)
        self._deliver()

    def state(self, now: float) -> GazeState:
        """The published state."""
        with self._lock:
            active, head = self._active, self._head
            moving = head is not None and head.settled is None
            until = None
            if head is not None and head.settled is not None:
                until = head.settled.stamp + self.settings.frame_period_s
            return GazeState(
                phase=self._current_phase(),
                head=self._readings[-1].aim if self._readings else None,
                target=None if head is None else head.aim,
                since=self._since,
                request_id=active.look.id if active is not None else "",
                source=active.look.source if active is not None else "",
                band=active.look.band if active is not None else HOME,
                blind=moving or (until is not None and now < until),
                blind_from=None if head is None else head.at,
                blind_until=until,
            )

    def pending(self) -> list[Look]:
        """The live requests, best first."""
        with self._lock:
            return [h.look for h in sorted(self._held, key=lambda h: (h.look.band, h.created))]

    # ---- requests ----------------------------------------------------------------------------
    def _take(self, look: Look, now: float, on_done: Callable[[Outcome], None] | None) -> Outcome:
        if look.kind == "track":
            return self._deny(look, "track is not built yet", on_done)
        if not 0 <= look.band < HOME:
            return self._deny(look, f"band {look.band} is not one of 0..{HOME - 1}", on_done)
        if look.kind != "home" and not look.views:
            return self._deny(look, "no view to look at", on_done)
        for older in [h for h in self._held if h.look.source == look.source]:
            if self._glancing(older):
                older.let_go = True  # a glance under way ends whole, then the newer one runs
            else:
                self._finish(older, "preempted", f"by a newer request of {look.source}", now)
        held = _Held(look, now, on_done)
        self._held.append(held)
        self.counts["requests"] += 1
        status: Status = "granted" if self._best() is held else "queued"
        return Outcome(look.id, look.source, status)

    def _glancing(self, held: _Held) -> bool:
        """A glance that holds the head and is not answered yet: nothing of its band takes it."""
        return held.look.hold_s > 0.0 and held is self._active and not held.answered

    def _best(self) -> _Held | None:
        """The request that should hold the head: the lowest band; within it the holder, unless
        a newcomer preempts (never a glance under way) or the holder only dwells after its
        answer."""
        if not self._held:
            return None
        band = min(h.look.band for h in self._held)
        mine = [h for h in self._held if h.look.band == band]
        active = self._active
        if active is None or active not in mine:
            return min(mine, key=lambda h: h.created)
        if self._glancing(active):
            return active
        newcomers = [h for h in mine if h is not active]
        for held in newcomers:
            if held.look.preempt:
                return held
        if active.answered and newcomers:
            return min(newcomers, key=lambda h: h.created)
        return active

    def _expire(self, now: float) -> None:
        for held in list(self._held):
            age = now - held.created
            if held.answered:
                if now - held.answered_at >= held.look.dwell_s or age > held.look.ttl_s:
                    self._drop(held, now, "done")
            elif age > held.look.ttl_s and not self._glancing(held):
                self._finish(held, "expired", f"not done within its {held.look.ttl_s:.1f} s", now)

    def _choose(self, now: float) -> None:
        best = self._best()
        active = self._active
        if best is active:
            return
        if active is not None and active in self._held and best is not None:
            self._finish(active, "preempted", f"by {best.look.source}", now)
        self._active = best
        if best is not None:
            best.view = 0
            if best.took_head is None:
                best.took_head = now

    # ---- the head ----------------------------------------------------------------------------
    def _current_phase(self) -> Phase:
        head = self._head
        if head is None:
            return "home"
        if head.settled is None:
            return "returning" if head.home else "saccade"
        return "home" if head.home else "still"

    def _phase_stamp(self, phase: Phase, now: float) -> float:
        """When a phase began, as the gate reads it (``since``): a move at the write that
        started it, a settled head at the reading it settled on, anything else ``now``."""
        head = self._head
        if head is None:
            return now
        if phase in ("saccade", "returning"):
            return head.at
        return head.settled.stamp if head.settled is not None else now

    def _serve(self, held: _Held, now: float) -> None:
        """Move toward the request's current view, then count its frames."""
        look = held.look
        home = not look.views
        aim = self.home if home else look.views[held.view]
        head = self._head
        if head is not None and head.timed_out and head.owner == look.id:
            self._finish(
                held,
                "done",
                f"the head did not settle within {self.settings.move_timeout_s:.1f} s",
                now,
            )
            return
        if head is None or head.aim.off(aim) > 1e-6:
            self._write(held, aim, home=home, now=now)
            return
        if head.settled is None:
            return
        held.waiting_since = None
        held.settled = head.settled
        floor = head.settled.stamp + self.settings.frame_period_s
        if held.settled_at is None:
            held.settled_at = now
            took = held.took_head if held.took_head is not None else floor
            held.still_from = max(floor, took)
        held.frames_seen = sum(1 for stamp in self._frames if stamp > floor)
        glanced = look.hold_s > 0.0 and now - held.settled_at >= look.hold_s
        if not home and held.frames_seen < look.frames and not glanced:
            dark_s = self._dark_spell(floor)
            patience = self.settings.dark_patience_s
            if patience > 0.0 and dark_s >= patience:
                self.counts["dark"] += 1
                self._finish(
                    held,
                    "expired",
                    f"{DARK_ENDED}: every frame of the last {dark_s:.1f} s was too dark for the"
                    f" depth (the gate's dark floor), {held.frames_seen} of {look.frames} fused",
                    now,
                    ended=DARK_ENDED,
                )
            return
        if held.view + 1 < len(look.views):
            held.view += 1
            return
        self._answer(held, now)

    def _dark_spell(self, floor: float) -> float:
        """How long the frames after ``floor`` (board s) have been dark without a break: from
        the first dark frame after the last fused one to the newest dark frame; 0 when the
        newest frame heard of was fused, or none was dark."""
        fused = max((t for t in self._frames if t > floor), default=floor)
        spell = [t for t in self._dark if t > fused]
        return spell[-1] - spell[0] if spell else 0.0

    def _write(self, held: _Held, aim: Aim, *, home: bool, now: float) -> None:
        """One write toward ``aim``, or a wait while the driver cannot take it."""
        if now < self._retry_at:
            return
        refusal = self._driver.blocked(now)
        if refusal is not None:
            self._wait(held, refusal, now)
            return
        self._send(aim, home=home, owner=held.look.id, speed=held.look.speed, now=now)

    def _wait(self, held: _Held, refusal: Refusal, now: float) -> None:
        if held.waiting_since is None:
            held.waiting_since = now
        if not refusal.transient or now - held.waiting_since > self.settings.move_timeout_s:
            self._finish(held, "denied", refusal.why, now)

    def _go_home(self, now: float) -> None:
        """No request: the head goes home once, after whatever last moved it."""
        head = self._head
        if head is None or head.aim == self.home or now < self._retry_at:
            return
        if self._driver.blocked(now) is not None:
            return
        self._send(self.home, home=True, owner="", speed="saccade", now=now)

    def _send(self, aim: Aim, *, home: bool, owner: str, speed: Speed, now: float) -> None:
        self._driver.write(None if home else aim, speed=speed, hold=not home, now=now)
        self._before, self._head = self._head, _Write(aim, home, now, owner)
        self.counts["writes"] += 1

    def _take_refusal(self, now: float) -> None:
        """A refused write never moved the head: the belief goes back to before it, and the
        request that wrote it waits (a transient refusal) or is denied."""
        refusal = self._driver.take_refusal()
        if refusal is None or self._head is None:
            return
        owner = self._head.owner
        self._head, self._before = self._before, None
        self._retry_at = now + RETRY_S
        self.counts["refused"] += 1
        active = self._active
        if active is not None and active.look.id == owner and not active.answered:
            self._wait(active, refusal, now)

    def _settle(self, now: float) -> None:
        """Whether the head has settled on the last write: the driver confirmed it, or two
        readings after the write agree with each other and with the target; past the move's
        timeout it is taken as wherever the encoders last said."""
        head = self._head
        if head is None or head.settled is not None:
            return
        tol = math.radians(self.settings.settle_tol_deg)
        arrival = self._driver.take_arrival()
        after = [r for r in self._readings if r.stamp > head.at]
        if arrival is not None and arrival.aim.off(head.aim) <= tol:
            head.settled = arrival
        elif (
            len(after) == 2
            and after[0].aim.off(after[1].aim) <= STILL_RAD
            and after[1].aim.off(head.aim) <= tol
        ):
            head.settled = after[1]
        elif now - head.at > self.settings.move_timeout_s:
            newest = self._readings[-1] if self._readings else None
            head.settled = newest or HeadReading(head.aim.pan_rad, head.aim.tilt_rad, now)
            head.timed_out = True
            if newest is not None:
                head.aim = newest.aim  # where it really is: the next request writes again
            self.counts["timeouts"] += 1
            return
        else:
            return
        self.counts["settled"] += 1

    # ---- answers -----------------------------------------------------------------------------
    def _answer(self, held: _Held, now: float) -> None:
        settled = held.settled
        outcome = Outcome(
            held.look.id,
            held.look.source,
            "done",
            reached=True,
            pan_rad=settled.pan_rad if settled else None,
            tilt_rad=settled.tilt_rad if settled else None,
            settled_stamp=settled.stamp if settled else None,
            frames_seen=held.frames_seen,
            took_ms=(now - held.created) * 1000.0,
        )
        held.answered, held.answered_at = True, now
        self.counts["done"] += 1
        if held.on_done is not None:
            self._outbox.append((held.on_done, outcome))
        if held.look.dwell_s <= 0.0 or held.let_go:
            self._drop(held, now, "done")

    def _finish(
        self, held: _Held, status: Status, reason: str, now: float, *, ended: str | None = None
    ) -> None:
        """End a request that was not answered as done: status, reason, where the head is;
        ``ended`` books the look under another word than its status."""
        self._drop(held, now, ended or status)
        if held.answered:
            return
        head = self._readings[-1] if self._readings else None
        outcome = Outcome(
            held.look.id,
            held.look.source,
            status,
            reason=reason,
            reached=False,
            pan_rad=head.pan_rad if head else None,
            tilt_rad=head.tilt_rad if head else None,
            settled_stamp=held.settled.stamp if held.settled else None,
            frames_seen=held.frames_seen,
            took_ms=(now - held.created) * 1000.0,
        )
        self.counts[status] += 1
        if held.on_done is not None:
            self._outbox.append((held.on_done, outcome))

    def _drop(self, held: _Held, now: float, ended: str) -> None:
        if held in self._held:
            self._held.remove(held)
            self._book(held, now, ended)
        if self._active is held:
            self._active = None

    def _book(self, held: _Held, now: float, ended: str) -> None:
        """A look that held the head, with the frames already fused while it held it still."""
        look = held.look
        if held.took_head is None or not look.views:
            return
        booked = HeldLook(
            look.id,
            look.source,
            look.band,
            look.views[held.view],
            held.took_head,
            now,
            ended,
            held.still_from,
        )
        booked.frames = sum(1 for stamp in self._frames if _claims(booked, stamp))
        self._closing.append(booked)

    def _close_books(self, now: float) -> None:
        """Looks let go more than LATE_S ago stop counting frames."""
        done = [b for b in self._closing if now - b.until > LATE_S]
        if done:
            self._closing = [b for b in self._closing if now - b.until <= LATE_S]
            self._closed.extend(done)

    def _deny(self, look: Look, why: str, on_done: Callable[[Outcome], None] | None) -> Outcome:
        outcome = Outcome(look.id, look.source, "denied", reason=why)
        self.counts["denied"] += 1
        if on_done is not None:
            self._outbox.append((on_done, outcome))
        return outcome

    def _deliver(self) -> None:
        with self._lock:
            outbox, self._outbox = self._outbox, []
        for callback, outcome in outbox:
            callback(outcome)


def _claims(booked: HeldLook, stamp: float) -> bool:
    """Whether a fused frame at ``stamp`` was taken while ``booked`` held the head still."""
    return booked.still_from is not None and booked.still_from < stamp <= booked.until


class LookTally:
    """The booked looks of a report window or a drive, by source, for one log line."""

    def __init__(self) -> None:
        self.looks: list[HeldLook] = []

    def add(self, booked: HeldLook) -> None:
        """One more look."""
        self.looks.append(booked)

    def text(self, shown: int = 3) -> str:
        """``looks nav.path 12 (31 frames), nav.reverse 2 (6 frames); 1 WROTE NO FRAMES: ...``."""
        if not self.looks:
            return "looks none"
        by: dict[str, list[HeldLook]] = {}
        for booked in self.looks:
            by.setdefault(booked.source, []).append(booked)
        parts = ", ".join(
            f"{source} {len(group)} ({sum(b.frames for b in group)} frames)"
            for source, group in sorted(by.items())
        )
        empty = [b for b in self.looks if b.frames == 0]
        if not empty:
            return f"looks {parts}; every one wrote frames"
        listed = "; ".join(b.text() for b in empty[:shown])
        more = f" (+{len(empty) - shown} more)" if len(empty) > shown else ""
        return f"looks {parts}; {len(empty)} WROTE NO FRAMES: {listed}{more}"


# ---- geometry: where to point ----------------------------------------------------------------
@dataclass(frozen=True)
class Reach:
    """The neck's joint range, radians."""

    pan: tuple[float, float]
    tilt: tuple[float, float]

    @classmethod
    def of(cls, cfg: NeckConfig) -> Reach:
        """config/neck.json's tick limits through the angle model."""
        pan, tilt = angle_limits(cfg)
        return cls(pan, tilt)

    def refusal(self, aim: Aim) -> str | None:
        """Why the neck cannot point there, in words; None when it can."""
        lo, hi = self.pan
        if not lo <= aim.pan_rad <= hi:
            return (
                f"pan {math.degrees(aim.pan_rad):+.0f} deg is past the neck's reach,"
                f" {math.degrees(hi):.0f} deg left to {-math.degrees(lo):.0f} deg right;"
                " to look further round, the robot itself has to turn"
            )
        lo, hi = self.tilt
        if not lo <= aim.tilt_rad <= hi:
            return (
                f"tilt {math.degrees(aim.tilt_rad):+.0f} deg is past the neck's reach,"
                f" {-math.degrees(lo):.0f} deg up to {math.degrees(hi):.0f} deg down"
            )
        return None

    def clamp(self, aim: Aim) -> Aim:
        """The nearest aim inside the reach."""
        return Aim(
            min(max(aim.pan_rad, self.pan[0]), self.pan[1]),
            min(max(aim.tilt_rad, self.tilt[0]), self.tilt[1]),
        )


def lens_at(cfg: NeckConfig, aim: Aim) -> tuple[float, float, float]:
    """The camera's position in base_link with the head at ``aim``, metres."""
    x, y, z, _roll, _pitch, _yaw = camera_pose(cfg, NeckAngles(aim.pan_rad, aim.tilt_rad))
    return x, y, z


def aim_at_point(cfg: NeckConfig, point: tuple[float, float, float], iterations: int = 6) -> Aim:
    """The joint angles that put a base_link point on the camera's optical axis (the lever arms
    of config/neck.json included, by fixed-point iteration from the pivot's own bearing)."""
    px, py, pz = point
    aim = Aim(math.atan2(py, px), math.radians(cfg.reference.pitch_deg))
    for _ in range(iterations):
        cx, cy, cz = lens_at(cfg, aim)
        aim = Aim(math.atan2(py - cy, px - cx), math.atan2(cz - pz, math.hypot(px - cx, py - cy)))
    return aim


def depression_deg(cfg: NeckConfig, aim: Aim, point: tuple[float, float, float]) -> float:
    """How far below level the point is as seen from the lens with the head at ``aim``."""
    cx, cy, cz = lens_at(cfg, aim)
    return math.degrees(math.atan2(cz - point[2], math.hypot(point[0] - cx, point[1] - cy)))


def home_aim(cfg: NeckConfig) -> Aim:
    """Home: pan 0, the reference pitch (the mount was measured there)."""
    return Aim(0.0, math.radians(cfg.reference.pitch_deg))


def reading_from_ticks(
    cfg: NeckConfig, pan_ticks: int, tilt_ticks: int, stamp: float
) -> HeadReading:
    """Encoder ticks as a :class:`HeadReading`."""
    angles = joint_angles(cfg, pan_ticks, tilt_ticks)
    return HeadReading(angles.pan_rad, angles.pitch_rad, stamp)


# ---- the requests' JSON ----------------------------------------------------------------------
Resolver = Callable[[str, dict[str, float]], tuple[float, float, float] | str]


def look_from_json(
    data: dict[str, Any], settings: GazeSettings, cfg: NeckConfig, to_base: Resolver
) -> Look | str:
    """A request as the doors receive it, resolved to views; the refusal in words otherwise.

    ``kind`` point (``target`` {frame, x, y, z}), direction ({bearing_rad, pitch_rad} in
    base_link), angles ({pan_rad, tilt_rad}), scan ({views: [{pan_rad, tilt_rad}, ...]}) or
    home; ``hold`` true keeps the head on the last view for the rest of the TTL (``dwell_s`` =
    ``ttl_s``); ``to_base(frame, xyz)`` puts a point of another frame in base_link (or says why
    not).
    """
    kind = str(data.get("kind", "angles"))
    if kind not in KINDS:
        return f"kind {kind!r} is not one of {', '.join(KINDS)}"
    source = str(data.get("source", "")).strip()
    if not source:
        return "a request names its source"
    try:
        band = int(data.get("band", PERSON))
        frames = int(data.get("frames", settings.frames))
        ttl_s = float(data.get("ttl_s", settings.ttl_for(band)))
        dwell_s = ttl_s if data.get("hold") else float(data.get("dwell_s", 0.0))
    except (TypeError, ValueError) as exc:
        return f"bad number: {exc}"
    speed = str(data.get("speed", "saccade"))
    if speed not in SPEEDS:
        return f"speed {speed!r} is not one of {', '.join(SPEEDS)}"
    target = data.get("target") or {}
    if not isinstance(target, dict):
        return "target is an object"
    views = _views(kind, target, cfg, to_base)
    if isinstance(views, str):
        return views
    reach = Reach.of(cfg)
    for view in views:
        why = reach.refusal(view)
        if why is not None:
            return why
    return Look(
        source=source,
        views=views,
        band=band,
        frames=max(frames, 0),
        dwell_s=max(dwell_s, 0.0),
        ttl_s=max(ttl_s, 0.0),
        speed=cast(Speed, speed),
        preempt=bool(data.get("preempt", False)),
        kind=cast(Kind, kind),
        id=str(data.get("id") or new_id()),
    )


def _views(
    kind: str, target: dict[str, Any], cfg: NeckConfig, to_base: Resolver
) -> tuple[Aim, ...] | str:
    try:
        if kind in ("home", "track"):
            return ()
        if kind == "angles":
            return (Aim(float(target["pan_rad"]), float(target["tilt_rad"])),)
        if kind == "scan":
            return tuple(Aim(float(v["pan_rad"]), float(v["tilt_rad"])) for v in target["views"])
        if kind == "direction":
            frame = str(target.get("frame", "base_link"))
            if frame != "base_link":
                return f"a direction is in base_link, not {frame}"
            return (Aim(float(target["bearing_rad"]), float(target["pitch_rad"])),)
        point = to_base(
            str(target.get("frame", "base_link")), {k: float(target[k]) for k in ("x", "y", "z")}
        )
    except (KeyError, TypeError, ValueError) as exc:
        return f"target incomplete for a {kind}: {exc}"
    if isinstance(point, str):
        return point
    return (Reach.of(cfg).clamp(aim_at_point(cfg, point)),)


# ---- the drivers -----------------------------------------------------------------------------
STATE_FRESH_S = 1.0  # a state line older than this says nothing about the wheels
REPLY_PATIENCE_S = 5.0  # a move whose answer never came is given up after this (board: 3 s)
TRANSIENT = ("wheels are moving", "already under way", "jog is under way")


class _BaseLink:
    """What both drivers read from the base server's state lines: the wheels and their twist."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._state_at: float | None = None
        self.wheels_moving = False
        self.twist = (0.0, 0.0)

    def _on_state(self, message: dict[str, Any], now: float) -> None:
        self.wheels_moving = bool(message.get("moving", False))
        self.twist = (float(message.get("v", 0.0)), float(message.get("w", 0.0)))
        self._state_at = now

    def _quiet(self, now: float) -> Refusal | None:
        if self._state_at is None or now - self._state_at > STATE_FRESH_S:
            return Refusal("no state line from the base server", True)
        return None


class BaseServerHead(_BaseLink):
    """:class:`HeadDriver` over the base server's ``neck_goto``/``neck_home``: one move at a time,
    answered when the head arrives, refused while the wheels turn."""

    moves_while_driving = False

    def __init__(self, cfg: NeckConfig, send: Callable[[bytes], bool]) -> None:
        super().__init__()
        self._cfg = cfg
        self._send = send
        self._in_flight_since: float | None = None
        self._refusal: Refusal | None = None
        self._arrival: HeadReading | None = None

    def on_line(self, message: dict[str, Any], now: float) -> None:
        """One line from the base server (its reader thread): state lines and move answers."""
        kind = message.get("type")
        with self._lock:
            if kind == "state":
                self._on_state(message, now)
                return
            if kind != "neck_goto" or self._in_flight_since is None:
                return
            self._in_flight_since = None
            pan, tilt = message.get("pan_ticks"), message.get("tilt_ticks")
            if pan is None or tilt is None:
                why = str(message.get("error") or "the base server refused the move")
                self._refusal = Refusal(why, any(t in why for t in TRANSIENT))
            elif message.get("reached"):
                self._arrival = reading_from_ticks(self._cfg, int(pan), int(tilt), now)

    def blocked(self, now: float) -> Refusal | None:
        """A move in flight, wheels that turn, or a base server that has gone quiet."""
        with self._lock:
            if self._in_flight_since is not None:
                if now - self._in_flight_since < REPLY_PATIENCE_S:
                    return Refusal("a neck move is under way", True)
                self._in_flight_since = None
            quiet = self._quiet(now)
            if quiet is not None:
                return quiet
            if self.wheels_moving:
                return Refusal(
                    "the wheels are moving (the base server moves the neck only at rest)", True
                )
        return None

    def write(self, aim: Aim | None, *, speed: Speed, hold: bool, now: float) -> None:
        """``neck_home`` or ``neck_goto`` in ticks; the profile speed is the board's own."""
        if aim is None:
            message: dict[str, Any] = {"cmd": "neck_home", "hold": hold}
        else:
            pan, tilt = ticks_for(self._cfg, NeckAngles(aim.pan_rad, aim.tilt_rad))
            message = {"cmd": "neck_goto", "pan_ticks": pan, "tilt_ticks": tilt, "hold": hold}
        with self._lock:
            self._refusal, self._arrival = None, None
            self._in_flight_since = now
        if not self._send((json.dumps(message) + "\n").encode()):
            with self._lock:
                self._in_flight_since = None
                self._refusal = Refusal("the link to the base server is down", True)

    def take_refusal(self) -> Refusal | None:
        """The newest write's refusal, once."""
        with self._lock:
            refusal, self._refusal = self._refusal, None
        return refusal

    def take_arrival(self) -> HeadReading | None:
        """The newest write's arrival (a ``reached`` answer), once."""
        with self._lock:
            arrival, self._arrival = self._arrival, None
        return arrival

    def keep(self, now: float) -> None:
        """Nothing to renew: a move ends by itself."""


class NeckTargetHead(_BaseLink):
    """:class:`HeadDriver` over ``neck_target`` (joint angles, written while driving): every
    target renews the board's lease, whose lapse sends the head home and lets it go, so a held
    target is re-sent every ``renew_s()`` and home is one target that is not renewed."""

    moves_while_driving = True

    def __init__(
        self,
        cfg: NeckConfig,
        send: Callable[[bytes], bool],
        *,
        slow_deg_s: Callable[[], float],
        renew_s: Callable[[], float],
        return_deg_s: Callable[[], float] | None = None,
    ) -> None:
        super().__init__()
        self._cfg = cfg
        self._send = send
        self._slow_deg_s = slow_deg_s
        self._renew_s = renew_s
        self._return_deg_s = return_deg_s
        self._held: dict[str, Any] | None = None
        self._sent_at = 0.0
        self._refusal: Refusal | None = None

    def on_line(self, message: dict[str, Any], now: float) -> None:
        """State lines (the wheels) and a refused target's answer."""
        kind = message.get("type")
        with self._lock:
            if kind == "state":
                self._on_state(message, now)
            elif kind == "neck_target" and message.get("error"):
                why = str(message["error"])
                self._refusal = Refusal(why, "jog" in why)

    def blocked(self, now: float) -> Refusal | None:
        """Only a base server that has gone quiet."""
        with self._lock:
            return self._quiet(now)

    def write(self, aim: Aim | None, *, speed: Speed, hold: bool, now: float) -> None:
        """One ``neck_target``: the board's own top speed for a saccade, ``slow_deg_s`` for a slow
        look, ``return_deg_s`` for the way home after a drive."""
        where = aim if aim is not None else home_aim(self._cfg)
        message: dict[str, Any] = {
            "cmd": "neck_target",
            "pan_rad": where.pan_rad,
            "tilt_rad": where.tilt_rad,
        }
        if speed == "slow":
            message["speed_deg_s"] = self._slow_deg_s()
        elif speed == "return" and self._return_deg_s is not None:
            message["speed_deg_s"] = self._return_deg_s()
        with self._lock:
            self._held = message if hold else None
            self._sent_at = now
        if not self._send((json.dumps(message) + "\n").encode()):
            with self._lock:
                self._refusal = Refusal("the link to the base server is down", True)

    def take_refusal(self) -> Refusal | None:
        """A refused target, once."""
        with self._lock:
            refusal, self._refusal = self._refusal, None
        return refusal

    def take_arrival(self) -> HeadReading | None:
        """No answer on arrival: the encoders in the state line settle it."""
        return None

    def keep(self, now: float) -> None:
        """Re-send a held target every ``renew_s()``, so a live laptop never lets it lapse."""
        with self._lock:
            held = self._held
            due = held is not None and now - self._sent_at >= self._renew_s()
            if due:
                self._sent_at = now
        if due and held is not None:
            self._send((json.dumps(held) + "\n").encode())


def speaks_target(state: dict[str, Any]) -> bool:
    """Whether a base server's state line carries the neck's encoders: the server that reads
    them in the wheels' tick is the one that takes ``neck_target`` while driving."""
    return "pan_ticks" in state and "tilt_ticks" in state


class EitherHead:
    """:class:`HeadDriver` that speaks ``neck_target`` to a base server whose state lines carry
    the neck (:func:`speaks_target`, once seen) and ``neck_goto`` to any other."""

    def __init__(self, goto: BaseServerHead, target: NeckTargetHead) -> None:
        self.goto, self.target = goto, target
        self.speaks_target = False

    @property
    def chosen(self) -> BaseServerHead | NeckTargetHead:
        """The driver in use."""
        return self.target if self.speaks_target else self.goto

    @property
    def moves_while_driving(self) -> bool:
        """The chosen driver's."""
        return self.chosen.moves_while_driving

    @property
    def wheels_moving(self) -> bool:
        """Whether the newest state line said the wheels turn."""
        return self.chosen.wheels_moving

    @property
    def twist(self) -> tuple[float, float]:
        """The newest state line's commanded (v, w)."""
        return self.chosen.twist

    def on_line(self, message: dict[str, Any], now: float) -> None:
        """Both drivers read every line; a state line with the neck in it picks ``neck_target``."""
        if message.get("type") == "state" and speaks_target(message):
            self.speaks_target = True
        self.goto.on_line(message, now)
        self.target.on_line(message, now)

    def blocked(self, now: float) -> Refusal | None:
        """The chosen driver's."""
        return self.chosen.blocked(now)

    def write(self, aim: Aim | None, *, speed: Speed, hold: bool, now: float) -> None:
        """The chosen driver's."""
        self.chosen.write(aim, speed=speed, hold=hold, now=now)

    def take_refusal(self) -> Refusal | None:
        """The chosen driver's."""
        return self.chosen.take_refusal()

    def take_arrival(self) -> HeadReading | None:
        """The chosen driver's."""
        return self.chosen.take_arrival()

    def keep(self, now: float) -> None:
        """The chosen driver's."""
        self.chosen.keep(now)
