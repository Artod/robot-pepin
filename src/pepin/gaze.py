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
  :meth:`Arbiter.state` says so, with the interval's two ends, for every frame consumer to gate on
  (the interval's end against the newest encoder reading's stamp: both on the board's clock);
* a POINT look that FOLLOWS (``follow``, :meth:`Arbiter.retarget`) keeps the head on its point
  while the cart turns or moves under it: its caller re-resolves the point from the pose of now
  (the node: TF) every step, and a view that moved past ``settle_tol_deg`` is written again, an
  answered look's too (the speaker gaze's person in the map, 2026-10-15);
* an OVERLAY (kind ``clip``, :class:`Overlay`) is keyed offsets ADDED to the aim of whatever holds
  the head (home when nothing does), each key written at its time at its own speed ceiling: it
  takes nothing from the holder, which keeps its status, its TTL and its aim, and the head goes
  back to the holder's aim when it ends. One plays at a time: a newer one replaces it, except
  that an ``idle`` overlay (breathing, a fidget) gives way to any other and never replaces one.
  It plays only over a holder of its band or a worse one, needs a driver that takes targets
  while moving (``neck_target``), and is refused whole (never clamped) past the reach, past
  ``clip_max_deg``, with a tilt that reverses faster than ``clip_max_hz`` or with a key asking
  the tilt for a ramp of its own (``tilt_acc_deg_s2``: a quick nod) above
  ``clip_acc_max_deg_s2``. A gesture's overlay (not ``idle``) rides the holder's aim its first
  key was written on: the holder's moves while it plays (a newer aim of its source, a follow's
  re-resolved point) wait for its end, where the way back goes to the holder's aim of then
  (2026-10-15: the speaker gaze's follow wrote the head seven times in 0.22 s under a shake and
  its middle key was never written); an idle one follows the holder; a ``loop`` repeats
  its keys until withdrawn or its TTL lapses (renewed like any request). An overlay whose
  offsets all stay within ``clip_still_deg`` (a breath) writes without starting a move: the head
  is not blind for it.

Pure: time is passed in, the encoders' readings and the fused frames' stamps are fed in, and the
ROS node (``pepin_bringup.gaze``) wires the topics, the base server and the doors. Two drivers
speak to the base server: :class:`BaseServerHead` (``neck_goto``/``neck_home``: one move at a
time, refused while the wheels turn) and :class:`NeckTargetHead` (``neck_target`` with a lease,
written while driving).
"""

from __future__ import annotations

import itertools
import json
import math
import threading
import uuid
from collections import Counter, deque
from collections.abc import Callable
from dataclasses import dataclass, field, replace
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

Kind = Literal["point", "direction", "angles", "scan", "track", "home", "clip"]
Speed = Literal["saccade", "slow", "return"]  # "return": the slow way home after a drive
Status = Literal["queued", "granted", "done", "denied", "preempted", "expired"]
Phase = Literal["home", "still", "saccade", "track", "returning"]

KINDS: tuple[str, ...] = ("point", "direction", "angles", "scan", "track", "home", "clip")
CLIP_KEYS_MAX = 64  # the keys one overlay may carry
# What an overlay is told on a driver that moves the neck one move at a time, at rest.
OVERLAY_DRIVER = (
    "a head overlay needs a base server that takes neck_target (keys while the head moves);"
    " this one moves the neck one move at a time"
)
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

    def plus(self, other: Aim) -> Aim:
        """This aim with ``other``'s angles added (an overlay's offset)."""
        return Aim(self.pan_rad + other.pan_rad, self.tilt_rad + other.tilt_rad)

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


ZERO = Aim(0.0, 0.0)  # no offset

# The blind tail in frame periods when the gaze node's frame_period_s knob is 0 (its default):
# 1.05 periods of config/camera.json's rate, the 0.105 s it was at 10 fps (one 9.5 fps frame).
BLIND_TAIL_PER_PERIOD = 1.05


@dataclass(frozen=True)
class GazeSettings:
    """The arbiter's numbers (config/knobs.json's ``gaze`` block, live); ``ttl_s`` is the
    default TTL of the operator, navigation, person, sensor, driving and idle bands;
    ``dark_patience_s`` how long a look waits on frames the depth gate drops as dark (0: until
    its TTL); ``clip_*`` bound an overlay (offset, tilt reversal rate, the still overlay, the
    most a key may ask the tilt's ramp for)."""

    frames: int = 3
    settle_tol_deg: float = 1.0
    move_timeout_s: float = 3.0
    frame_period_s: float = 0.105
    ttl_s: tuple[float, float, float, float, float, float] = (0.5, 3.0, 10.0, 2.0, 0.5, 20.0)
    dark_patience_s: float = 1.5
    clip_max_deg: float = 30.0
    clip_max_hz: float = 2.5
    clip_still_deg: float = 1.0
    clip_acc_max_deg_s2: float = 600.0

    def ttl_for(self, band: int) -> float:
        """The default TTL of a band's request."""
        return self.ttl_s[min(max(band, 0), len(self.ttl_s) - 1)]


def new_id() -> str:
    """A short request id."""
    return uuid.uuid4().hex[:8]


@dataclass(frozen=True)
class OverlayKey:
    """One key of an overlay: seconds from its start, the pan (+ left) and tilt (+ down) offsets
    in radians added to the holder's aim, the speed ceiling of the move to it and its ramp's
    (None: the neck's own; the idle layer's keys glide on a gentle one), and the tilt's own ramp
    (None: the neck's tilt ceiling; a quick nod asks more, at most ``clip_acc_max_deg_s2``)."""

    at_s: float
    pan_rad: float
    tilt_rad: float
    speed_deg_s: float
    acc_deg_s2: float | None = None
    tilt_acc_deg_s2: float | None = None

    @property
    def offset(self) -> Aim:
        """The key's offset as an :class:`Aim`."""
        return Aim(self.pan_rad, self.tilt_rad)


@dataclass(frozen=True)
class Overlay:
    """A clip's keys in time order; ``loop``: they repeat every :attr:`period_s` until the look
    is withdrawn or lapses; ``idle``: the idle layer's, giving way to any other overlay."""

    keys: tuple[OverlayKey, ...]
    loop: bool = False
    idle: bool = False

    @property
    def period_s(self) -> float:
        """The last key's time: one pass (a loop's period)."""
        return self.keys[-1].at_s if self.keys else 0.0

    @property
    def reach_rad(self) -> float:
        """The largest offset of any key on either axis."""
        return max((max(abs(k.pan_rad), abs(k.tilt_rad)) for k in self.keys), default=0.0)

    def end_s(self) -> float:
        """When a pass is over: the last key's time plus its move from the key before it at its
        own ceiling (the travel the head still owes after the last write)."""
        if not self.keys:
            return 0.0
        last = self.keys[-1]
        before = self.keys[-2].offset if len(self.keys) > 1 else Aim(0.0, 0.0)
        travel_deg = math.degrees(last.offset.off(before))
        return last.at_s + travel_deg / max(last.speed_deg_s, 1e-6)

    def key_at(self, elapsed_s: float) -> tuple[int, int]:
        """The key due ``elapsed_s`` after the start (-1 before the first) and the pass it is in
        (always 0 without ``loop``)."""
        if not self.keys:
            return -1, 0
        cycle = 0
        if self.loop and self.period_s > 0.0 and elapsed_s >= self.period_s:
            cycle = int(elapsed_s // self.period_s)
            elapsed_s -= cycle * self.period_s
        due = [i for i, key in enumerate(self.keys) if key.at_s <= elapsed_s]
        if due:
            return due[-1], cycle
        if cycle > 0:  # a loop's next pass before its first key: the previous pass's last key
            return len(self.keys) - 1, cycle - 1
        return -1, 0

    def offset_at(self, index: int) -> Aim:
        """Key ``index``'s offset; none (0, 0) before the first."""
        return self.keys[index].offset if 0 <= index < len(self.keys) else Aim(0.0, 0.0)

    def refusal(self, settings: GazeSettings) -> str | None:
        """Why the overlay is refused by the knobs alone, in words; None when it plays."""
        limit = settings.clip_max_deg
        if math.degrees(self.reach_rad) > limit + 1e-9:
            return (
                f"an overlay key {math.degrees(self.reach_rad):.1f} deg off the holder's aim is"
                f" past clip_max_deg {limit:.0f}"
            )
        if self.loop and self.period_s <= 0.0:
            return "a looped overlay needs its last key after 0 s (its period)"
        asked = max((k.tilt_acc_deg_s2 or 0.0 for k in self.keys), default=0.0)
        if asked > settings.clip_acc_max_deg_s2 + 1e-9:
            return (
                f"an overlay key asks the tilt for {asked:.0f} deg/s^2, past clip_acc_max_deg_s2"
                f" {settings.clip_acc_max_deg_s2:.0f} (the mast rings after a sharp tilt)"
            )
        fastest = _tilt_half_period_s(self)
        hz = settings.clip_max_hz
        if hz > 0.0 and fastest is not None and fastest < 0.5 / hz - 1e-9:
            return (
                f"the tilt reverses {fastest:.2f} s after its last turn: faster than"
                f" clip_max_hz {hz:.1f} (the mast rings at 5.3 Hz)"
            )
        return None


def _tilt_half_period_s(overlay: Overlay) -> float | None:
    """The shortest time between two turns of the tilt's direction (the start of its motion
    counts as a turn; a loop wraps round: its closing key is the next pass's first, never
    written itself), None when it never turns."""
    keys = overlay.keys
    if overlay.loop and len(keys) > 1:
        keys = keys[:-1]
    points = [(0.0, 0.0)] + [(k.at_s, k.tilt_rad) for k in keys]
    if overlay.loop and keys:
        period = overlay.period_s
        points += [(period + k.at_s, k.tilt_rad) for k in keys]
        points.append((2 * period + keys[0].at_s, keys[0].tilt_rad))
    turns: list[float] = []
    direction = 0
    previous = points[0][1]
    for at_s, tilt in points[1:]:
        step = tilt - previous
        previous = tilt
        if abs(step) < 1e-9:
            continue
        sign = 1 if step > 0 else -1
        if sign != direction:
            turns.append(at_s)
            direction = sign
    gaps = [b - a for a, b in itertools.pairwise(turns)]
    return min(gaps) if gaps else None


@dataclass(frozen=True)
class Look:
    """One request: its views (empty: home), the still frames wanted at each, the hold after the
    last, the band, the TTL, the speed, whether it takes the head from its own band, and
    ``hold_s``: above 0, a glance (answered at ``frames`` or ``hold_s`` after settling, atomic).
    A ``clip`` carries its :class:`Overlay` instead of views. ``speed_deg_s`` and ``acc_deg_s2``:
    the move's own ceilings over ``speed``'s (``neck_target``'s; None: the speed's own), so a
    look can ease in and out on the servo's trapezoid. A ``point`` look keeps its point, and
    with ``follow`` its view is re-resolved from it as the cart moves (:meth:`Arbiter.retarget`)."""

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
    overlay: Overlay | None = None
    speed_deg_s: float | None = None
    acc_deg_s2: float | None = None
    point: tuple[str, float, float, float] | None = None  # a point look's (frame, x, y, z)
    follow: bool = False  # a point look kept on its point as the cart moves (:meth:`retarget`)


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

    def write(
        self,
        aim: Aim | None,
        *,
        speed: Speed,
        hold: bool,
        now: float,
        speed_deg_s: float | None = None,
        acc_deg_s2: float | None = None,
        tilt_acc_deg_s2: float | None = None,
    ) -> None:
        """Send the head to ``aim`` (``None``: home); ``hold`` keeps the servos energised;
        ``speed_deg_s`` and ``acc_deg_s2`` ceilings of the move's own (an overlay's key, an
        eased look) over ``speed``'s; ``tilt_acc_deg_s2`` the tilt's own ramp (an overlay's
        quick nod)."""
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
    """The arbiter's last accepted write and what became of it: ``aim`` is the holder's (the
    base), ``offset`` an overlay's on top of it, and the head was sent to their sum."""

    aim: Aim
    home: bool
    at: float  # when it went out: the blind interval starts here
    owner: str  # the request that wrote it ("" for the arbiter's own home)
    settled: HeadReading | None = None  # the reading it settled on; None while the head moves
    timed_out: bool = False
    offset: Aim = field(default_factory=lambda: Aim(0.0, 0.0))

    @property
    def sent(self) -> Aim:
        """Where the head was sent: the base plus the overlay's offset."""
        return self.aim.plus(self.offset)


@dataclass
class _Playing:
    """The overlay playing: its request, its TTL clock, when it started, and the key, pass and
    base it was last written at."""

    look: Look
    overlay: Overlay
    created: float
    started: float
    on_done: Callable[[Outcome], None] | None
    key: int = -1
    cycle: int = 0
    base: Aim | None = None
    still: bool = False  # every offset within clip_still_deg: written without starting a move


@dataclass(frozen=True)
class GazeState:
    """The one gate (``/gaze/state``): the phase, the head, the request holding it, and the
    blind interval (``blind_from`` the write that started the move, ``blind_until`` settled plus
    one frame period, ``None`` while the head still moves). ``head`` is the newest encoder
    reading and ``head_stamp`` that reading's own stamp (``/neck/state``'s header: the moment of
    the encoder read on the board's clock, the clock the camera's frames are stamped on), so a
    reader places a frame with the pose at the frame's stamp without its own clock."""

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
    head_stamp: float | None = None

    def as_dict(self) -> dict[str, Any]:
        """The JSON ``/gaze/state`` carries."""
        return {
            "phase": self.phase,
            "pan_rad": None if self.head is None else round(self.head.pan_rad, 4),
            "tilt_rad": None if self.head is None else round(self.head.tilt_rad, 4),
            "head_stamp": None if self.head_stamp is None else round(self.head_stamp, 4),
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

    def __init__(
        self,
        driver: HeadDriver,
        home: Aim,
        settings: GazeSettings | None = None,
        reach: Reach | None = None,
    ) -> None:
        self._driver = driver
        self.home = home
        self.settings = settings if settings is not None else GazeSettings()
        self.reach = reach  # an overlay's keys over the holder's aim are refused past it
        self._back_deg_s = 0.0  # the speed the head goes back to its holder at after an overlay
        self._back_acc: float | None = None  # and its ramp (the overlay's last key's own, if any)
        self._back_tilt_acc: float | None = None  # and the tilt's own (a quick nod's last key's)
        self._kept_back = False  # a holder's write kept back under a gesture: the way back owes it
        self._back_still = False
        self._lock = threading.RLock()
        self._held: list[_Held] = []
        self._active: _Held | None = None
        self._playing: _Playing | None = None  # the overlay on top of the holder
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
        way is not cut: it goes once answered. An overlay ``keep`` refuses ends too."""
        with self._lock:
            refused = [h for h in self._held if not keep(h.look)]
            for held in refused:
                if self._glancing(held):
                    held.let_go = True
                else:
                    self._finish(held, "preempted", f"by {by}", now)
            playing = self._playing
            ended = playing is not None and not keep(playing.look)
            if playing is not None and ended:
                self._end_overlay(playing, "preempted", f"by {by}", now)
        self._deliver()
        return sum(1 for h in refused if not h.let_go) + int(ended)

    def withdraw(self, source: str, now: float, by: str, fade_s: float = 0.0) -> int:
        """Take back every request of ``source``, as preempted ``by``, a glance under way too:
        its own asker's word, which :meth:`release` is not (the reverse look let go at the first
        forward command); its overlay goes too, the head back on the holder's aim over
        ``fade_s`` (0: at the overlay's last key's speed); how many went."""
        with self._lock:
            mine = [h for h in self._held if h.look.source == source]
            for held in mine:
                self._finish(held, "preempted", f"by {by}", now)
            playing = self._playing
            overlay = playing is not None and playing.look.source == source
            if playing is not None and overlay:
                self._end_overlay(playing, "preempted", f"by {by}", now, fade_s=fade_s)
        self._deliver()
        return len(mine) + int(overlay)

    def holds(self, source: str) -> bool:
        """Whether a request of ``source`` holds the head."""
        with self._lock:
            return self._active is not None and self._active.look.source == source

    def overlay(self) -> Look | None:
        """The overlay playing on top of the holder, None when none plays."""
        with self._lock:
            return self._playing.look if self._playing is not None else None

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
            believed = self._head.sent if self._head is not None else self.home
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

    def following(self) -> Look | None:
        """The look holding the head when it follows its point (``follow``): its caller
        re-resolves the point from the pose of now and hands the view to :meth:`retarget`; None
        when the holder is no such look."""
        with self._lock:
            active = self._active
            if active is None or not active.look.follow or active.look.point is None:
                return None
            return active.look

    def retarget(self, look_id: str, aim: Aim, now: float) -> bool:
        """The following look ``look_id``'s point re-resolved to ``aim`` (the cart turned or
        moved under it): past ``settle_tol_deg`` of its view, ``aim`` is its view and the head is
        written there, an answered look's head too (it holds the head on its point; one not yet
        answered is written by the next step). Whether it was; the same view, or a look that no
        longer holds the head, changes nothing."""
        with self._lock:
            active = self._active
            if active is None or active.look.id != look_id or not active.look.follow:
                return False
            views = active.look.views
            if views and aim.off(views[-1]) <= math.radians(self.settings.settle_tol_deg):
                return False
            active.look = replace(active.look, views=(aim,))
            active.view = 0
            self.counts["follows"] += 1
            if active.answered:
                self._write(active, aim, home=False, now=now)
        self._deliver()
        return True

    def renew(self, source: str, now: float) -> int:
        """Restart the TTL of ``source``'s requests, and the dwell of one already answered
        (``see`` keeps a look; path gaze keeps its aim), and of its overlay; how many."""
        with self._lock:
            mine = [h for h in self._held if h.look.source == source]
            for held in mine:
                held.created = now
                if held.answered:
                    held.answered_at = now
            playing = self._playing
            overlay = playing is not None and playing.look.source == source
            if playing is not None and overlay:
                playing.created = now
        return len(mine) + int(overlay)

    # ---- the decision ------------------------------------------------------------------------
    def step(self, now: float) -> None:
        """Expire, choose, and drive the head toward the chosen request's view or home."""
        with self._lock:
            self._close_books(now)
            self._expire(now)
            self._choose(now)
            playing, active = self._playing, self._active
            if playing is not None and active is not None and active.look.band < playing.look.band:
                # before the better look writes: its write carries no offset
                self._end_overlay(playing, "preempted", f"by {active.look.source}", now)
            self._take_refusal(now)
            self._settle(now)
            active = self._active
            if active is None:
                self._go_home(now)
            elif not active.answered:
                self._serve(active, now)
            self._play(now)
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
            reading = self._readings[-1] if self._readings else None
            # the blind tail ends on the board's clock: the newest encoder reading's stamp is its
            # "now" (never this Mac's clock against a board stamp; none read since: still blind)
            board = None if reading is None else reading.stamp
            return GazeState(
                phase=self._current_phase(),
                head=None if reading is None else reading.aim,
                target=None if head is None else head.aim,
                since=self._since,
                request_id=active.look.id if active is not None else "",
                source=active.look.source if active is not None else "",
                band=active.look.band if active is not None else HOME,
                blind=moving or (until is not None and (board is None or board < until)),
                blind_from=None if head is None else head.at,
                blind_until=until,
                head_stamp=None if reading is None else reading.stamp,
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
        if look.kind == "clip":
            return self._take_overlay(look, now, on_done)
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
        """One write toward ``aim``, or a wait while the driver cannot take it or a gesture's
        overlay plays on the holder's aim of its start (its end writes the head back to the
        holder's aim of then)."""
        if self._riding() is not None:
            self._kept_back = True  # written by the gesture's way back
            return
        if now < self._retry_at:
            return
        refusal = self._driver.blocked(now)
        if refusal is not None:
            self._wait(held, refusal, now)
            return
        look = held.look
        self._send(
            aim,
            home=home,
            owner=look.id,
            speed=look.speed,
            now=now,
            speed_deg_s=look.speed_deg_s,
            acc_deg_s2=look.acc_deg_s2,
        )

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

    def _send(
        self,
        aim: Aim,
        *,
        home: bool,
        owner: str,
        speed: Speed,
        now: float,
        offset: Aim | None = None,
        speed_deg_s: float | None = None,
        acc_deg_s2: float | None = None,
        tilt_acc_deg_s2: float | None = None,
    ) -> None:
        """One write of the base ``aim`` with an overlay's ``offset`` on top (the playing
        overlay's current one when None): a lifted home is held like any aim."""
        lift = self._overlay_offset() if offset is None else offset
        lifted = lift != ZERO
        self._out(
            None if home and not lifted else aim.plus(lift),
            speed,
            not home or lifted,
            now,
            speed_deg_s,
            acc_deg_s2,
            tilt_acc_deg_s2,
        )
        written = _Write(aim, home and not lifted, now, owner, offset=lift)
        self._before, self._head = self._head, written

    def _out(
        self,
        aim: Aim | None,
        speed: Speed,
        hold: bool,
        now: float,
        speed_deg_s: float | None,
        acc_deg_s2: float | None = None,
        tilt_acc_deg_s2: float | None = None,
    ) -> None:
        """One write to the driver; a ceiling of its own is passed only when there is one."""
        ceilings: dict[str, float] = {}
        if speed_deg_s is not None:
            ceilings["speed_deg_s"] = speed_deg_s
        if acc_deg_s2 is not None:
            ceilings["acc_deg_s2"] = acc_deg_s2
        if tilt_acc_deg_s2 is not None:
            ceilings["tilt_acc_deg_s2"] = tilt_acc_deg_s2
        self._driver.write(aim, speed=speed, hold=hold, now=now, **ceilings)
        self.counts["writes"] += 1

    # ---- the overlay -------------------------------------------------------------------------
    def _take_overlay(
        self, look: Look, now: float, on_done: Callable[[Outcome], None] | None
    ) -> Outcome:
        """An overlay is played over the holder (or home), refused whole, or replaces the one
        playing; an idle one gives way to any other."""
        overlay = look.overlay
        if overlay is None or not overlay.keys:
            return self._deny(look, "a clip carries its keys", on_done)
        if not self._driver.moves_while_driving:
            return self._deny(look, OVERLAY_DRIVER, on_done)
        holder = self._best()
        if holder is not None and holder.look.band < look.band:
            return self._deny(
                look,
                f"the head is held by {holder.look.source} ({BANDS[holder.look.band]} band),"
                f" which an overlay of the {BANDS[look.band]} band never plays over",
                on_done,
            )
        why = overlay.refusal(self.settings)
        base = self._holder_aim(holder)
        for key in overlay.keys:
            if why is None:
                why = self._reach_refusal(base.plus(key.offset))
        if why is not None:
            return self._deny(look, why, on_done)
        playing = self._playing
        if playing is not None:
            mine = playing.look.source == look.source
            if not mine and overlay.idle and not playing.overlay.idle:
                return self._deny(
                    look, f"{playing.look.source}'s overlay plays: an idle one gives way", on_done
                )
            by = f"a newer request of {look.source}" if mine else look.source
            self._end_overlay(playing, "preempted", f"by {by}", now)
        still = math.degrees(overlay.reach_rad) <= self.settings.clip_still_deg + 1e-9
        self._playing = _Playing(look, overlay, now, now, on_done, still=still)
        self.counts["requests"] += 1
        self.counts["overlays"] += 1
        return Outcome(look.id, look.source, "granted")

    def _holder_aim(self, held: _Held | None = None) -> Aim:
        """The aim of ``held`` (the request holding the head when None): home for none."""
        held = self._active if held is None else held
        if held is None or not held.look.views:
            return self.home
        return held.look.views[min(held.view, len(held.look.views) - 1)]

    def _reach_refusal(self, aim: Aim) -> str | None:
        return self.reach.refusal(aim) if self.reach is not None else None

    def _riding(self) -> Aim | None:
        """The holder's aim a gesture's overlay rides, once its first key is written (None: no
        overlay, an idle one that follows the holder, or none written yet)."""
        playing = self._playing
        if playing is None or playing.overlay.idle:
            return None
        return playing.base

    def _overlay_offset(self) -> Aim:
        """The playing overlay's offset as last written; none before its first key."""
        playing = self._playing
        if playing is None:
            return ZERO
        return playing.overlay.offset_at(playing.key)

    def _play(self, now: float) -> None:
        """The overlay's key due now written over the holder's aim; its end, its TTL, a better
        band, the reach; with none playing, the head owed back to the holder's aim."""
        playing = self._playing
        if playing is None:
            self._write_back(now)
            return
        look, overlay = playing.look, playing.overlay
        if now - playing.created > look.ttl_s:
            self._end_overlay(playing, "expired", f"not ended within its {look.ttl_s:.1f} s", now)
            self._write_back(now)
            return
        elapsed = now - playing.started
        if not overlay.loop and elapsed >= overlay.end_s():
            self._end_overlay(playing, "done", "", now)
            self._write_back(now)
            return
        index, cycle = overlay.key_at(elapsed)
        ridden = self._riding()
        base = ridden if ridden is not None else self._holder_aim()
        if index < 0 or ((index, cycle) == (playing.key, playing.cycle) and base == playing.base):
            return
        key = overlay.keys[index]
        why = self._reach_refusal(base.plus(key.offset))
        if why is not None:
            self._end_overlay(playing, "denied", why, now)
            self._write_back(now)
            return
        head = self._head
        if head is not None and head.aim == base and head.offset == key.offset:
            playing.key, playing.cycle, playing.base = index, cycle, base  # already written
            return
        if now < self._retry_at or self._driver.blocked(now) is not None:
            return
        if head is not None and head.settled is None:
            # the holder's move (or the last key's) is still in flight: a key written now would
            # hand the servo this key's speed and ramp mid-flight, and a glide key's tiny ramp
            # takes its brakes away (2026-10-13: the head through its limits and its zero);
            # the key waits for the head to settle, the next due key is the one written
            return
        playing.key, playing.cycle, playing.base = index, cycle, base
        self._lift(base, key.offset, playing.still, look.id, key.speed_deg_s, now,
                   key.acc_deg_s2, key.tilt_acc_deg_s2)  # fmt: skip

    def _lift(
        self,
        base: Aim,
        offset: Aim,
        still: bool,
        owner: str,
        speed_deg_s: float,
        now: float,
        acc_deg_s2: float | None = None,
        tilt_acc_deg_s2: float | None = None,
    ) -> None:
        """The head sent to ``base`` plus ``offset`` at the key's ceilings: a still overlay over a
        settled head moves it without starting a move (no blind interval), anything else is a
        write like any."""
        head = self._head
        active = self._active
        home = active is None or not active.look.views
        if still and head is not None and head.settled is not None and head.aim == base:
            lifted = offset != ZERO
            target = None if home and not lifted else base.plus(offset)
            self._out(target, "saccade", not home or lifted, now, speed_deg_s, acc_deg_s2,
                      tilt_acc_deg_s2)  # fmt: skip
            head.offset, head.home = offset, home and not lifted
            return
        holder = active.look.id if active is not None and offset == ZERO else owner
        self._send(
            base,
            home=home,
            owner=holder,
            speed="saccade",
            now=now,
            offset=offset,
            speed_deg_s=speed_deg_s,
            acc_deg_s2=acc_deg_s2,
            tilt_acc_deg_s2=tilt_acc_deg_s2,
        )

    def _write_back(self, now: float) -> None:
        """No overlay plays and the head still carries an offset, or stands where an answered
        holder no longer looks (its move kept back under a gesture): back to the holder's aim,
        at the ended overlay's return speed."""
        head = self._head
        if head is None:
            return
        active = self._active
        behind = self._kept_back and active is not None and head.aim != self._holder_aim()
        if head.offset == ZERO and not behind:
            self._kept_back = False
            return
        if now < self._retry_at or self._driver.blocked(now) is not None:
            return
        if head.settled is None:
            return  # the last key's move still flies: the way back waits for it (its ramp is small)
        self._kept_back = False
        self._lift(self._holder_aim(), ZERO, self._back_still, "", self._back_deg_s, now,
                   self._back_acc, self._back_tilt_acc)  # fmt: skip

    def _end_overlay(
        self, playing: _Playing, status: Status, reason: str, now: float, *, fade_s: float = 0.0
    ) -> None:
        """The overlay ends: answered, and the speed the head goes back at is kept (its offset
        over ``fade_s``, else the last key's ceiling)."""
        if self._playing is playing:
            self._playing = None
        head = self._head
        offset_deg = math.degrees(head.offset.off(ZERO)) if head is not None else 0.0
        keys = playing.overlay.keys
        last = keys[-1].speed_deg_s if keys else 0.0
        self._back_deg_s = offset_deg / fade_s if fade_s > 0.0 and offset_deg > 0.0 else last
        self._back_acc = keys[-1].acc_deg_s2 if keys and not fade_s > 0.0 else None
        self._back_tilt_acc = keys[-1].tilt_acc_deg_s2 if keys and not fade_s > 0.0 else None
        self._back_still = playing.still
        reading = self._readings[-1] if self._readings else None
        outcome = Outcome(
            playing.look.id,
            playing.look.source,
            status,
            reason=reason,
            reached=status == "done",
            pan_rad=reading.pan_rad if reading else None,
            tilt_rad=reading.tilt_rad if reading else None,
            took_ms=(now - playing.started) * 1000.0,
        )
        self.counts[status] += 1
        if playing.on_done is not None:
            self._outbox.append((playing.on_done, outcome))

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
        playing = self._playing
        if playing is not None and playing.look.id == owner:
            self._end_overlay(playing, "denied", refusal.why, now)

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
        sent = head.sent
        if arrival is not None and arrival.aim.off(sent) <= tol:
            head.settled = arrival
        elif (
            len(after) == 2
            and after[0].aim.off(after[1].aim) <= STILL_RAD
            and after[1].aim.off(sent) <= tol
        ):
            head.settled = after[1]
        elif now - head.at > self.settings.move_timeout_s:
            newest = self._readings[-1] if self._readings else None
            head.settled = newest or HeadReading(sent.pan_rad, sent.tilt_rad, now)
            head.timed_out = True
            if newest is not None:  # where it really is: the next request writes again
                head.aim = Aim(
                    newest.pan_rad - head.offset.pan_rad, newest.tilt_rad - head.offset.tilt_rad
                )
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

    def clamp(self, aim: Aim, from_pan: float | None = None) -> Aim:
        """The aim inside the reach: the tilt and a pan past the reach at the nearer end.

        The two ends of a wide pan reach are a few degrees apart through the dead zone behind
        and a whole sweep apart through the front, and a bearing behind, kept in (-pi, pi],
        changes sign there: +184 deg reads -176 and the nearest end BY NUMBER is the far one.
        With ``from_pan`` (the head's own pan, radians) the end nearer to THAT wins, so a target
        behind never takes the head across the front for a few degrees of nearness (the whip of
        2026-10-13: +153 -> -167, 320 deg at full speed, for the same person)."""
        lo, hi = self.pan
        pan = min(max(aim.pan_rad, lo), hi)
        if from_pan is not None and not lo <= aim.pan_rad <= hi:
            pan = lo if abs(from_pan - lo) < abs(from_pan - hi) else hi
        return Aim(pan, min(max(aim.tilt_rad, self.tilt[0]), self.tilt[1]))


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
    data: dict[str, Any],
    settings: GazeSettings,
    cfg: NeckConfig,
    to_base: Resolver,
    from_pan: float | None = None,
) -> Look | str:
    """A request as the doors receive it, resolved to views; the refusal in words otherwise.

    ``kind`` point (``target`` {frame, x, y, z}; past the reach at the end nearer ``from_pan``,
    the head's own pan, :meth:`Reach.clamp`; ``follow`` true keeps the head on the point as the
    cart moves, :meth:`Arbiter.retarget`), direction ({bearing_rad, pitch_rad} in
    base_link), angles ({pan_rad, tilt_rad}; ``speed_deg_s`` and ``acc_deg_s2``, optional, are the
    move's own ceilings: an eased look), scan ({views: [{pan_rad, tilt_rad}, ...]}), home,
    or clip (an overlay, band idle by default: {keys: [{at_s, pan_rad, tilt_rad, speed_deg_s,
    acc_deg_s2 (optional: the key's own ramp), tilt_acc_deg_s2 (optional: the tilt's own, at
    most clip_acc_max_deg_s2)}, ...], loop, idle}; a loop's last key closes its
    period and equals its first); ``hold`` true
    keeps the head on the last view for the rest of the TTL (``dwell_s`` = ``ttl_s``);
    ``to_base(frame, xyz)`` puts a point of another frame in base_link (or says why not).
    """
    kind = str(data.get("kind", "angles"))
    if kind not in KINDS:
        return f"kind {kind!r} is not one of {', '.join(KINDS)}"
    source = str(data.get("source", "")).strip()
    if not source:
        return "a request names its source"
    try:
        band = int(data.get("band", IDLE if kind == "clip" else PERSON))
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
    follow = bool(data.get("follow", False))
    if follow and kind != "point":
        return f"follow is a point look's, not a {kind}'s: the other kinds hold joint angles"
    if kind == "clip":
        overlay = overlay_from_json(target, settings)
        if isinstance(overlay, str):
            return overlay
        return Look(
            source=source,
            views=(),
            band=band,
            frames=0,
            dwell_s=0.0,
            ttl_s=max(ttl_s, 0.0),
            preempt=bool(data.get("preempt", False)),
            kind="clip",
            id=str(data.get("id") or new_id()),
            overlay=overlay,
        )
    ceilings = _ceilings(data)
    if isinstance(ceilings, str):
        return ceilings
    views = _views(kind, target, cfg, to_base, from_pan)
    if isinstance(views, str):
        return views
    point = None
    if kind == "point":  # parsed by _views already: every coordinate is a number
        frame = str(target.get("frame", "base_link"))
        point = (frame, float(target["x"]), float(target["y"]), float(target["z"]))
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
        speed_deg_s=ceilings[0],
        acc_deg_s2=ceilings[1],
        point=point,
        follow=follow,
    )


def _ceilings(data: dict[str, Any]) -> tuple[float | None, float | None] | str:
    """A look's own ``speed_deg_s`` and ``acc_deg_s2`` (absent: None), or why refused."""
    found: list[float | None] = []
    for key in ("speed_deg_s", "acc_deg_s2"):
        value = data.get(key)
        if value is None:
            found.append(None)
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return f"{key} is a number, not {value!r}"
        if not math.isfinite(value) or value <= 0.0:
            return f"{key} is a finite number above 0, not {value!r}"
        found.append(float(value))
    return found[0], found[1]


def overlay_from_json(target: dict[str, Any], settings: GazeSettings) -> Overlay | str:
    """A clip's ``target`` as an :class:`Overlay`, or why it is refused (the knobs' bounds
    included: the arbiter never clamps)."""
    raw = target.get("keys")
    if not isinstance(raw, list) or not raw:
        return "a clip carries its keys: [{at_s, pan_rad, tilt_rad, speed_deg_s}, ...]"
    if len(raw) > CLIP_KEYS_MAX:
        return f"a clip carries at most {CLIP_KEYS_MAX} keys, not {len(raw)}"
    keys: list[OverlayKey] = []
    try:
        for item in raw:
            acc, tilt_acc = item.get("acc_deg_s2"), item.get("tilt_acc_deg_s2")
            key = OverlayKey(
                float(item["at_s"]),
                float(item.get("pan_rad", 0.0)),
                float(item.get("tilt_rad", 0.0)),
                float(item["speed_deg_s"]),
                None if acc is None else float(acc),
                None if tilt_acc is None else float(tilt_acc),
            )
            values = (key.at_s, key.pan_rad, key.tilt_rad, key.speed_deg_s, key.acc_deg_s2 or 1.0,
                      key.tilt_acc_deg_s2 or 1.0)  # fmt: skip
            if not all(math.isfinite(v) for v in values):
                return "a clip's key is a finite number on every field"
            if key.at_s < 0.0 or (keys and key.at_s < keys[-1].at_s):
                return "a clip's keys are in time order, from 0 s"
            if key.speed_deg_s <= 0.0:
                return "a clip's key moves at a speed above 0 deg/s"
            if any(r is not None and r <= 0.0 for r in (key.acc_deg_s2, key.tilt_acc_deg_s2)):
                return "a clip's key ramps at more than 0 deg/s^2"
            keys.append(key)
    except (KeyError, TypeError, ValueError, AttributeError) as exc:
        return f"a clip's key is incomplete: {exc!r}"
    overlay = Overlay(tuple(keys), loop=bool(target.get("loop")), idle=bool(target.get("idle")))
    why = overlay.refusal(settings)
    return why if why is not None else overlay


def _views(
    kind: str,
    target: dict[str, Any],
    cfg: NeckConfig,
    to_base: Resolver,
    from_pan: float | None = None,
) -> tuple[Aim, ...] | str:
    try:
        if kind in ("home", "track"):
            return ()
        if kind == "angles":
            return (_tilt_within(cfg, Aim(float(target["pan_rad"]), float(target["tilt_rad"]))),)
        if kind == "scan":
            return tuple(
                _tilt_within(cfg, Aim(float(v["pan_rad"]), float(v["tilt_rad"])))
                for v in target["views"]
            )
        if kind == "direction":
            frame = str(target.get("frame", "base_link"))
            if frame != "base_link":
                return f"a direction is in base_link, not {frame}"
            return (
                _tilt_within(cfg, Aim(float(target["bearing_rad"]), float(target["pitch_rad"]))),
            )
        point = to_base(
            str(target.get("frame", "base_link")), {k: float(target[k]) for k in ("x", "y", "z")}
        )
    except (KeyError, TypeError, ValueError) as exc:
        return f"target incomplete for a {kind}: {exc}"
    if isinstance(point, str):
        return point
    return (Reach.of(cfg).clamp(aim_at_point(cfg, point), from_pan),)


def _tilt_within(cfg: NeckConfig, aim: Aim) -> Aim:
    """``aim`` with its tilt inside the neck's reach: the head looks as far up or down as it
    reaches (2026-10-14: a person standing over the robot asked for 21.9 deg up, 2 deg past the
    stop, and the base server refused the 1791-tick target outright, so the head never moved).
    The pan is left to the holder's own travel-aware clamp (:meth:`Reach.clamp`)."""
    lo, hi = Reach.of(cfg).tilt
    return Aim(aim.pan_rad, min(max(aim.tilt_rad, lo), hi))


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

    def write(
        self,
        aim: Aim | None,
        *,
        speed: Speed,
        hold: bool,
        now: float,
        speed_deg_s: float | None = None,
        acc_deg_s2: float | None = None,
        tilt_acc_deg_s2: float | None = None,
    ) -> None:
        """``neck_home`` or ``neck_goto`` in ticks; the profile speed is the board's own (no
        ceiling of a move's own: overlays are refused on this driver, an eased look moves at
        the board's pace)."""
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

    def write(
        self,
        aim: Aim | None,
        *,
        speed: Speed,
        hold: bool,
        now: float,
        speed_deg_s: float | None = None,
        acc_deg_s2: float | None = None,
        tilt_acc_deg_s2: float | None = None,
    ) -> None:
        """One ``neck_target``: the board's own top speed for a saccade, ``slow_deg_s`` for a slow
        look, ``return_deg_s`` for the way home after a drive, ``speed_deg_s`` when given (an
        overlay's key, an eased look); ``acc_deg_s2`` the ramp ceiling when given (the board
        keeps the tilt's own cap under it), ``tilt_acc_deg_s2`` the tilt's own ramp when given
        (an overlay's quick nod: the board honours it up to its top ramp)."""
        where = aim if aim is not None else home_aim(self._cfg)
        message: dict[str, Any] = {
            "cmd": "neck_target",
            "pan_rad": where.pan_rad,
            "tilt_rad": where.tilt_rad,
        }
        if speed_deg_s is not None and speed_deg_s > 0.0:
            message["speed_deg_s"] = speed_deg_s
        elif speed == "slow":
            message["speed_deg_s"] = self._slow_deg_s()
        elif speed == "return" and self._return_deg_s is not None:
            message["speed_deg_s"] = self._return_deg_s()
        if acc_deg_s2 is not None and acc_deg_s2 > 0.0:
            message["acc_deg_s2"] = acc_deg_s2
        if tilt_acc_deg_s2 is not None and tilt_acc_deg_s2 > 0.0:
            message["tilt_acc_deg_s2"] = tilt_acc_deg_s2
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

    def write(
        self,
        aim: Aim | None,
        *,
        speed: Speed,
        hold: bool,
        now: float,
        speed_deg_s: float | None = None,
        acc_deg_s2: float | None = None,
        tilt_acc_deg_s2: float | None = None,
    ) -> None:
        """The chosen driver's."""
        self.chosen.write(
            aim,
            speed=speed,
            hold=hold,
            now=now,
            speed_deg_s=speed_deg_s,
            acc_deg_s2=acc_deg_s2,
            tilt_acc_deg_s2=tilt_acc_deg_s2,
        )

    def take_refusal(self) -> Refusal | None:
        """The chosen driver's."""
        return self.chosen.take_refusal()

    def take_arrival(self) -> HeadReading | None:
        """The chosen driver's."""
        return self.chosen.take_arrival()

    def keep(self, now: float) -> None:
        """The chosen driver's."""
        self.chosen.keep(now)
