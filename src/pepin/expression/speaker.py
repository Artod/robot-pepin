"""The speaker gaze's contract: where the voice came from, the faces and the people in a frame,
the arbiter's door as the speaker gaze uses it (with a move's own ease), and the seams for
choosing whom to look at.

Angles follow the neck (:class:`pepin.gaze.Aim`): pan positive to the left of the cart's nose,
tilt positive below level. The microphone array sits on the cart, not on the head, so its
direction of arrival is a bearing in the cart's frame once :class:`DoaMap` (measured on the robot)
has turned it: with the array's readings of a voice left of, behind and right of the cart, the
bearing is interpolated between the four quarter turns (an array off the cart's centre reads
them unevenly); with the front alone, ``bearing = sign * (doa_deg - front_deg)``, wrapped to
-180..180.
A face's bearing comes from the head camera's left eye through its lens
(:meth:`pepin.face_sensor.geometry.Lens.bearing_deg`), relative to where the head pointed when
the frame was taken, so a face only counts in a frame taken while the head stood still. A
person's box (:class:`BodySighting`) comes through the same lens with an estimate of where the
head is in it, so a person whose face is not seen (turned away, lying, far) still gives an aim.

Whom to look at is :mod:`pepin.expression.policy`'s over the people tracked
(:mod:`pepin.expression.tracks`), and how to turn there is :class:`GazePlan`'s split between the
head and the wheels: the head alone within the forward arc, the body turned toward a person past
it when the wheels may move, and the head as far as it reaches when they may not.

What the gesture engine may know of attention is :class:`Attention` (read through
:class:`AttentionView`, never written), and the one thing it is asked is :class:`ArmRelease`;
:func:`head_refusal` and :func:`arm_refusal` are the rules it keeps (the contract's; owned by
none of the gesticulation slices).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Protocol

from pepin.audio_link import DoaReading
from pepin.gaze import Aim, Speed

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

    from pepin.face_sensor.models import Detection


@dataclass(frozen=True)
class DoaMap:
    """The array's angle that points along the cart's x axis, whether the array's angle grows
    toward the cart's left (+1) or right (-1), and, when measured, its angles of a voice left
    of, behind and right of the cart (``quarters``: the map then interpolates between them)."""

    front_deg: float
    sign: int
    quarters: tuple[float, float, float] | None = None


class DoaSource(Protocol):
    """The array's direction of arrival as the audio link keeps it (:class:`pepin.audio_link.
    AudioClient`): readings stamped with the board's clock, a few seconds of them."""

    def latest_doa(self) -> DoaReading | None:
        """The newest reading, None before any."""
        ...

    def doa_between(self, t0: float, t1: float) -> list[DoaReading]:
        """The readings stamped within ``t0..t1`` (board clock)."""
        ...


@dataclass(frozen=True)
class FaceSighting:
    """One face in a frame: its bearing from the optical axis in degrees (pan + left, tilt +
    down), its box width in pixels, the detector's score, who it is when the finder knows, how
    far the head is turned from the camera (degrees, from the detector's five points; 0 faces
    the robot) when the finder reads it, and the detection itself (box and five points: what
    person recognition aligns the face on) when the finder keeps it; a finder that keeps no
    pixels (the perception service's) gives the crop's sharpness and brightness instead, the
    measures person recognition's gate reads."""

    pan_deg: float
    tilt_deg: float
    width_px: float
    score: float
    who: str | None = None
    yaw_deg: float | None = None
    detection: Detection | None = field(default=None, compare=False, repr=False)
    sharpness: float | None = field(default=None, compare=False, repr=False)
    brightness: float | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class BodySighting:
    """One person's box in a frame (a body detector's, :class:`pepin.expression.finder.
    BodyDetector`), in degrees from the optical axis (pan + left, tilt + down): the box's centre
    pan and half width, the head estimated from it (near the box's top: standing, sitting,
    slouching and lying all give one), its height, the detector's score, and whether the frame's
    top edge cuts it (the head is then above the picture: the estimate is the edge)."""

    pan_deg: float
    half_width_deg: float
    head_pan_deg: float
    head_tilt_deg: float
    height_deg: float
    score: float
    cut: bool = False


@dataclass(frozen=True)
class FaceFrame:
    """The faces of one frame, its capture time (the camera's own stamp), when the finder keeps
    it the picture (BGR) the faces were found in, the people's boxes when it has a body
    detector, and the frame's number in the perception service's ring when it came from there
    (what person recognition names a face by, in place of the pixels)."""

    stamp: float
    faces: tuple[FaceSighting, ...]
    image: NDArray[np.uint8] | None = field(default=None, compare=False, repr=False)
    bodies: tuple[BodySighting, ...] = ()
    seq: int | None = field(default=None, compare=False)


class FaceFinder(Protocol):
    """Faces on demand: one frame from the head camera, searched now (never a stream)."""

    def look(self, not_before: float | None = None) -> FaceFrame | None:
        """The faces in a fresh frame; None when no frame came. ``not_before``: the board-clock
        moment from which a frame is still (the arbiter's ``blind_until``): a finder that reads
        frames already taken waits for one exposed after it; one that takes its picture now may
        ignore it."""
        ...


@dataclass(frozen=True)
class GazeAnswer:
    """The arbiter's answer to an aim: ``done``, ``queued``, ``denied``, ``preempted`` or
    ``expired``, why, and where the head got to."""

    status: str
    reason: str = ""
    head: Aim | None = None


@dataclass(frozen=True)
class HeadView:
    """The arbiter's state as the speaker gaze reads it: the head's angles, the source and band
    holding it, whether a frame now would be blurred (the blind interval, the board's clock),
    whether a drive runs, and the source of the overlay playing on top ("" for none)."""

    head: Aim | None
    holder: str
    band: int
    blind: bool
    blind_until: float | None
    driving: bool
    overlay: str = ""


@dataclass(frozen=True)
class Ease:
    """A move's own speed and ramp ceilings (``neck_target``'s): the servo's trapezoid at them
    accelerates evenly to the middle and decelerates evenly to the end (an ease in and out: a
    triangle of velocity for a move shorter than ``speed^2 / acc``, a plateau at ``speed`` for a
    longer one). ``name`` says which of the gaze's profiles it is, for the log."""

    speed_deg_s: float
    acc_deg_s2: float
    name: str = ""

    def seconds(self, deg: float) -> float:
        """How long a move of ``deg`` takes at this profile (the servo's own lag aside)."""
        d, v, a = abs(deg), self.speed_deg_s, self.acc_deg_s2
        if d <= 0.0 or v <= 0.0 or a <= 0.0:
            return 0.0
        if d <= v * v / a:
            return 2.0 * math.sqrt(d / a)
        return d / v + v / a


class GazeDoor(Protocol):
    """The arbiter's door (``POST /look``, ``/renew``, ``/release``, ``GET /state``) for one
    source; every call may wait on the network, so only a worker thread makes them."""

    def aim(self, aim: Aim, *, speed: Speed, ttl_s: float, ease: Ease | None = None) -> GazeAnswer:
        """Hold the head at ``aim`` (replacing this source's last aim), moving at ``speed`` or,
        given, on ``ease``; answered when the head has settled there, or refused."""
        ...

    def renew(self) -> int:
        """Restart this source's TTL; how many of its requests are live (0: it lost the head)."""
        ...

    def release(self) -> None:
        """Let the head go now (it goes home unless another request holds it)."""
        ...

    def state(self) -> HeadView | None:
        """The arbiter's state; None when its door does not answer."""
        ...


@dataclass(frozen=True)
class GazePlan:
    """How to face an aim: the wheels' turn in place first (radians, + left; 0 when the head
    suffices or the wheels may not move), then the head's aim relative to the turned cart."""

    body_rad: float
    head: Aim


# What the attention engine is doing, as the gesture engine reads it: no conversation (or the
# switch off), a drive, looking for the person engaged (a seek's stops, or a check of a new
# voice), the body turning toward them, the head on their face, or on their voice (nobody seen)
AttentionPhase = Literal["idle", "paused", "seek", "check", "turning", "track", "voice"]
HEAD_SEEN_S = 1.0  # a gesture overlay plays only on a person seen within this (DESIGN.md §8.4)


@dataclass(frozen=True)
class Attention:
    """A read-only snapshot of the attention engine (the speaker gaze), laptop monotonic clock:
    its phase and since when, whether it has eyes (a face finder: frames to protect), when the
    person engaged was last seen by a face or a body (None: never), their bearing and tilt
    (degrees, the cart's frame, + left, + down) and rough distance (None: unknown), whether they
    stand past the forward arc (a body turn is wanted) and when their first clear face of the
    conversation was found."""

    phase: AttentionPhase = "idle"
    since_s: float = 0.0
    eyes: bool = False
    seen_s: float | None = None
    bearing_deg: float | None = None
    tilt_deg: float | None = None
    distance_m: float | None = None
    past_arc: bool = False
    first_face_s: float | None = None


NO_ATTENTION = Attention()  # the speaker gaze off, or no conversation: nothing to keep clear


class AttentionView(Protocol):
    """The attention engine as the gesture engine reads it (never writes): the latest snapshot,
    from any thread."""

    def attention(self) -> Attention:
        """The snapshot now (:data:`NO_ATTENTION` while the gaze is off)."""
        ...


class ArmRelease(Protocol):
    """The one arbitration between the arm and the wheels: the gaze's body turn asks the gesture
    engine for the arm first (the board refuses the wheels while the arm moves or holds)."""

    def release_arm(self, why: str) -> float:
        """Send the arm to rest now (no new arm motion until the turn is over); the seconds until
        it is there (0: it is already). Never blocks on the arm."""
        ...


def head_refusal(view: Attention, now: float) -> str | None:
    """Why a gesture overlay may not take the head now, None when it may: while attention looks
    for the person (a seek, a check), turns the body, or has not seen them within
    :data:`HEAD_SEEN_S`, a gesture's skipped frames would blind it (the idle layer's still
    overlays never do). No rule without eyes, a conversation, or during a drive."""
    if not view.eyes or view.phase in ("idle", "paused"):
        return None
    if view.phase != "track":
        return f"attention: {view.phase}"
    if view.seen_s is None or now - view.seen_s > HEAD_SEEN_S:
        unseen = "never" if view.seen_s is None else f"{now - view.seen_s:.1f} s"
        return f"attention: the person unseen for {unseen}"
    return None


def arm_refusal(view: Attention) -> str | None:
    """Why no arm motion may start now, None when it may: the person engaged stands past the
    forward arc or the body turns toward them (the board would refuse the wheels or the arm)."""
    if view.phase == "turning":
        return "attention: the body turns toward the person"
    if view.past_arc:
        return "attention: the person stands past the forward arc"
    return None
