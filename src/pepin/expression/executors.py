"""The parts that play clips, and the animator that owns their layers.

One :class:`Executor` per body part turns a rendered clip into requests to that part's one owner:

* ``arm``: a dense trajectory over the clip's tracks on top of the arm's base pose (rest), sent
  to the board's base server (``arm_trajectory``, :mod:`pepin.expression.trajectory`), which plays
  it on its own tick; fading the gesture layer is ``arm_rest``;
* ``head``: the clip's offsets as a relative ``clip`` request to the gaze arbiter
  (:mod:`pepin.expression.head`), atomic like a glance and played on top of whatever holds the
  head; fading it withdraws the request and the head falls back to its holder;
* ``face``: the clip's events as the head server's timed expressions (:class:`FaceChannel`);
* ``body``: small motions of the wheels while standing, through :class:`Wheels`
  (:class:`NoBody` and :class:`NoWheels` today).

THE WHEELS never drive for expression, and never through Nav2: a turn in place of at most
``max_turn_deg`` per request (30) or a rock of at most ``max_rock_m`` (0.02), on the board's
measured-motion path (``ros/tools/move.py``'s: velocity commands judged by the odometry's swept
heading and travel, the lidar's cone, a time cap per leg, the base server's deadman behind it),
and only while the goal server's ``where`` says ``"navigating": false``. A turn goes in steps
of ``step_deg``; when the wheels stall or the lidar or the ToF says the arc is blocked, the cart
backs off ``backoff_m`` and turns again, at most ``attempts`` times and ``max_backoff_m`` of
backing in all, and then gives up: the speaker gaze then looks as far as the head reaches.

The :class:`Animator` is the one place a gesture is asked for: the cascade's cues, the
``gesture`` tool and the laptop's command line all go through it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pepin.expression.animation import Clip, Layer, PartName, Performance

NO_BODY = "the wheels make no expressive motion: their executor is not built"


@dataclass(frozen=True)
class Outcome:
    """How a request to play went: started (or refused, and why), and for how long it plays."""

    ok: bool
    why: str = ""
    seconds: float = 0.0


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


class FaceChannel(Protocol):
    """The face as :class:`pepin.head_link.HeadClient` sends to it (a non-blocking line)."""

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


class Animator(Protocol):
    """The gesture engine: renders intents, owns every part's gesture and idle layers. In a
    conversation it is also an observer of the cascade, and places each marker of a queued part
    on that part's audio itself; :meth:`perform` is the door of the tool and the command line."""

    def perform(self, intent: str, *, source: str) -> Outcome:
        """Perform ``intent`` now; refused when the gestures are off, a drive runs, or the book
        has no such intent."""
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
    distance it backed off, the attempts it took, and why it stopped short."""

    done: bool
    turned_deg: float = 0.0
    backed_m: float = 0.0
    attempts: int = 0
    why: str = ""


class Wheels(Protocol):
    """The measured-motion path for expression (the board's), with the bounded retry above."""

    def move(self, request: WheelMove) -> WheelReport:
        """Run ``request`` to its end or give up (blocks: a worker thread calls it); refused
        outright while a drive runs or past the knobs' ranges."""
        ...


class NoWheels:
    """The wheels while their expression path is not built: every motion refused."""

    def move(self, request: WheelMove) -> WheelReport:
        """Refused."""
        return WheelReport(False, why=NO_BODY)


class NoBody:
    """The wheels' executor while no wheel motion for expression exists: never ready, every
    clip refused, so a gesture plays without its body steps and the speaker gaze turns only the
    head."""

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
