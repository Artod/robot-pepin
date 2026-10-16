"""The parts that play clips, and the animator that owns their layers.

One :class:`Executor` per body part turns a rendered clip into requests to that part's one owner:

* ``arm``: a dense trajectory over the clip's tracks on top of the arm's base pose (rest), sent
  to the board's base server (``arm_trajectory``, :mod:`pepin.expression.trajectory`), which plays
  it on its own tick; fading the gesture layer is ``arm_rest``;
* ``head``: the clip's offsets as a relative ``clip`` request to the gaze arbiter
  (:mod:`pepin.expression.head`), atomic like a glance and played on top of whatever holds the
  head; fading it withdraws the request and the head falls back to its holder;
* ``face``: the clip's events as the head server's timed expressions (:class:`FaceChannel`);
* ``eyes``: the clip's look as the head server's timed eyes overrides, its blinks, winks and
  effects at their times (:class:`EyesChannel`); fading it lets the eyes go;
* ``body``: small motions of the wheels while standing, through :class:`Wheels`
  (:class:`pepin.expression.wheels.WheelsExecutor`; :class:`NoBody` and :class:`NoWheels` where
  there is no robot).

THE WHEELS never drive for expression, and never through Nav2: a turn in place of at most
``max_turn_deg`` per request (30) or a rock of at most ``max_rock_m`` (0.02), by the measured
motions' method (``ros/tools/move.py``'s: velocity commands judged by the odometry's swept
heading and travel, depth_fusion's obstacle marks, a time cap per leg, the base server's deadman
behind it), only while the goal server says ``"navigating": false``. When the wheels stall or the
marks say the arc is blocked, the cart steps ``backoff_m`` away from the nearest mark (or turns
``backoff_deg`` back when it cannot), tries the rest once more, and then gives up: the speaker
gaze then looks as far as the head reaches. Every call to the wheels returns at once: a motion
runs on the wheels' own thread and answers through a callback, so nothing the voice waits on
ever waits on the wheels.

The :class:`Animator` is the one place a gesture is asked for: the cascade's cues, the
``gesture`` tool and the laptop's command line all go through it.
"""

from __future__ import annotations

import bisect
import math
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pepin.expression.animation import Clip, Layer, PartName, Performance

NO_BODY = "the wheels make no expressive motion here: no robot"


@dataclass(frozen=True)
class Outcome:
    """How a request to play went: started (or refused, and why), for how long it plays, and
    when its stroke lands from now as the part planned it (None: where its clip says; the arm
    adds the board's approach and its stretch to the caps)."""

    ok: bool
    why: str = ""
    seconds: float = 0.0
    stroke_s: float | None = None


class Executor(Protocol):
    """One body part as the animator plays it."""

    @property
    def part(self) -> PartName:
        """Which part it plays."""
        ...

    @property
    def latency_s(self) -> float:
        """From :meth:`play` to the part's first motion (a knob): a clip is sent this early."""
        ...

    def ready(self) -> str | None:
        """Why the part cannot play now (a drive, an owner away, a better band holding the
        head), None when it can."""
        ...

    def play(self, clip: Clip) -> Outcome:
        """Start ``clip`` on its layer now, taking the part from the clip playing there."""
        ...

    def fade(self, layer: Layer, seconds: float) -> None:
        """Let ``layer`` go over ``seconds``: the part settles on the layer under it."""
        ...

    def rest(self) -> None:
        """Every layer let go: the part at its base (a drive starts, a conversation ends)."""
        ...


@runtime_checkable
class Renewing(Protocol):
    """A part whose looping clip lapses at its owner unless renewed (the head: the arbiter's
    TTL of one period and a second); the engine renews it once a period while it plays."""

    def renew(self, layer: Layer) -> None:
        """Keep ``layer``'s looping clip playing one more period (never blocks)."""
        ...


class FaceChannel(Protocol):
    """The face as :class:`pepin.head_link.HeadClient` sends to it (non-blocking lines): its
    expression and its drawing style."""

    def express(
        self,
        name: str,
        *,
        intensity: float = 1.0,
        hold_s: float | None = None,
        transition_ms: float | None = None,
    ) -> None:
        """Show expression ``name`` (held until replaced when ``hold_s`` is None)."""
        ...

    def style(self, name: str | None, ms: float | None = None) -> None:
        """Draw the mouth in style ``name``: for ``ms``, or standing when None; None lets this
        source's styles go."""
        ...


class EyesChannel(Protocol):
    """The eyes as :class:`pepin.head_link.HeadClient` sends to them (non-blocking lines, one
    source's)."""

    def eyes(self, *, ms: float | None = None, side: str = "both", **fields: Any) -> None:
        """Override the eyes' look (:class:`pepin.head_link.Eyes`' fields, config/face.json's
        units) on ``side``: until replaced when ``ms`` is None, else for ``ms``."""
        ...

    def eyes_release(self) -> None:
        """This source overrides the eyes no more (its effect ends too)."""
        ...

    def blink(self, side: str = "both") -> None:
        """One blink: both eyes, or a wink of ``side``."""
        ...

    def eyes_effect(self, name: str | None, ms: float | None = None) -> None:
        """The inner rings' effect ``name`` for ``ms`` (None: until replaced); None ends it."""
        ...


class Animator(Protocol):
    """The gesture engine: renders intents, owns every part's gesture and idle layers. In a
    conversation it is also an observer of the cascade, and places each marker of a queued part
    on that part's audio itself; :meth:`perform` is the door of the tool and the command line."""

    def perform(self, intent: str, *, source: str) -> Outcome:
        """Perform ``intent`` now (a composition's name, or ``name:arg`` as a marker writes it:
        ``point:printer``); refused when the gestures are off, a drive runs, or the book has no
        such intent."""
        ...

    def hush(self, why: str) -> None:
        """Fade the gesture layer of every part (the person speaks); the base layer stays."""
        ...

    def last(self) -> Performance | None:
        """The latest performance started, for the logs and the tool's answer."""
        ...


@dataclass(frozen=True)
class WheelMove:
    """One small motion of the wheels: a turn in place (degrees, + left) or ``rocks`` rocks of
    ``rock_m`` forward and back; never both."""

    turn_deg: float = 0.0
    rock_m: float = 0.0
    rocks: int = 0


@dataclass(frozen=True)
class WheelReport:
    """How a wheel motion ended: whether it did all it was asked, the heading it turned, the
    distance it stepped away (forward or back), the attempts it took (two at most), why it
    stopped short, and whether a halt cut it (a drive began, a newer motion: worth finishing
    later)."""

    done: bool
    turned_deg: float = 0.0
    backed_m: float = 0.0
    attempts: int = 0
    why: str = ""
    cut: bool = False


WheelsDone = Callable[[WheelReport], None]  # a motion's end, told on any thread: never block it
WheelsMoved = Callable[[float], None]  # a turn's heading so far (degrees), every step_deg
HEADINGS_KEPT_S = 30.0  # the odometry's headings kept: an utterance's readings and then some


class Headings:
    """The cart's heading as the base server's wheel odometry tells it in every state line, on
    the board's clock (the clock the microphone array stamps its directions with): the last
    ``kept_s`` seconds of (stamp, heading), the heading unwrapped (radians, + left). Read at any
    stamp, interpolated between two lines, the nearest end outside them (before the first line
    the cart had not moved: the link opens before a motion's first twist). Thread-safe: the base
    link's reader adds, the speaker gaze reads."""

    def __init__(self, kept_s: float = HEADINGS_KEPT_S) -> None:
        """Keeps ``kept_s`` seconds."""
        self.kept_s = kept_s
        self._lock = threading.Lock()
        self._stamps: list[float] = []
        self._headings: list[float] = []

    def add(self, stamp_s: float, theta: float) -> None:
        """One state line's heading (wrapped or not) at its board stamp; a stamp not newer than
        the newest is ignored."""
        with self._lock:
            if self._stamps:
                if stamp_s <= self._stamps[-1]:
                    return
                last = self._headings[-1]
                theta = last + math.remainder(theta - last, math.tau)
            self._stamps.append(stamp_s)
            self._headings.append(theta)
            old = bisect.bisect_left(self._stamps, stamp_s - self.kept_s)
            if old > len(self._stamps) // 2:  # trimmed in halves: amortised, one kept before
                del self._stamps[: old - 1], self._headings[: old - 1]

    def latest(self) -> tuple[float, float] | None:
        """The newest (stamp, heading); None before any line."""
        with self._lock:
            return (self._stamps[-1], self._headings[-1]) if self._stamps else None

    def at(self, stamp_s: float) -> float | None:
        """The heading at board time ``stamp_s``; None before any line."""
        with self._lock:
            stamps, headings = self._stamps, self._headings
            if not stamps:
                return None
            if stamp_s <= stamps[0]:
                return headings[0]
            if stamp_s >= stamps[-1]:
                return headings[-1]
            k = bisect.bisect_right(stamps, stamp_s)
            s0, s1, h0, h1 = stamps[k - 1], stamps[k], headings[k - 1], headings[k]
            return h0 + (h1 - h0) * (stamp_s - s0) / (s1 - s0)


class Wheels(Protocol):
    """The wheels' expressive motions with the bounded retry above. Every call returns at once:
    the motion runs on the wheels' own thread; ``done`` hears its end (possibly before
    :meth:`start` returns, for a request refused outright) and ``moved`` its progress, both on a
    thread that is not the caller's, so each only hands the word to its owner's thread."""

    def start(self, request: WheelMove, done: WheelsDone, moved: WheelsMoved | None = None) -> None:
        """Run ``request``: it replaces the motion running (that one answers ``cut``)."""
        ...

    def halt(self, why: str) -> None:
        """Stop the motion running now and drop what waits (never blocks: the stop is one
        non-blocking line to the base)."""
        ...

    def headings(self) -> Headings:
        """The cart's heading over the last seconds (empty until the base has said anything):
        whoever turned the cart, the speaker gaze's frame follows it."""
        ...


class NoWheels:
    """No wheels to move for expression (no robot): every motion refused."""

    def __init__(self) -> None:
        """No heading ever."""
        self._headings = Headings()

    def start(self, request: WheelMove, done: WheelsDone, moved: WheelsMoved | None = None) -> None:
        """Refused, answered at once."""
        done(WheelReport(False, why=NO_BODY))

    def halt(self, why: str) -> None:
        """Nothing moves."""

    def headings(self) -> Headings:
        """Always empty: the cart never turns here."""
        return self._headings


class NoBody:
    """The body where there are no wheels to move (no robot): never ready, every clip refused,
    so a gesture plays without its body steps."""

    part: PartName = "body"
    latency_s: float = 0.0

    def ready(self) -> str | None:
        """Always the reason it cannot play."""
        return NO_BODY

    def play(self, clip: Clip) -> Outcome:
        """Refused."""
        return Outcome(False, NO_BODY)

    def fade(self, layer: Layer, seconds: float) -> None:
        """Nothing plays: nothing to let go."""

    def rest(self) -> None:
        """Nothing plays: nothing to let go."""
