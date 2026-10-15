"""The speaker gaze's contract: where the voice came from, the faces and the people in a frame,
the arbiter's door as the speaker gaze uses it (with a move's own ease), and the seams for
choosing whom to look at.

PEOPLE ARE POINTS IN THE MAP (since 2026-10-15): every sighting and every voice is placed as a
:class:`Point` in the ``map`` frame by the poses of its own stamp (TF's, through the goal server's
``pose_at``: :mod:`pepin.expression.poses`), and the head is asked to look at a point
(:meth:`GazeDoor.aim`), which the arbiter turns into pan and tilt from TF's pose of now and keeps
as the cart turns under it. Angles follow the neck (:class:`pepin.gaze.Aim`): pan positive to the
left of the cart's nose, tilt positive below level. The microphone array sits on the cart, not on
the head, so its
direction of arrival is a bearing in the cart's frame once :class:`DoaMap` (measured on the robot)
has turned it: with the array's readings of a voice left of, behind and right of the cart, the
bearing is interpolated between the four quarter turns (an array off the cart's centre reads
them unevenly); with the front alone, ``bearing = sign * (doa_deg - front_deg)``, wrapped to
-180..180.
A face's angles come from the head camera's left eye through its lens
(:meth:`pepin.face_sensor.geometry.Lens.bearing_deg`), relative to the eye's optical axis when
the frame was exposed: the ray they name is carried into the map by the camera's pose at the
frame's stamp. A person's box (:class:`BodySighting`) comes through the same lens with an
estimate of where the head is in it, so a person whose face is not seen (turned away, lying,
far) still gives a point.

Whom to look at is :mod:`pepin.expression.policy`'s over the people in the register
(:mod:`pepin.expression.people_map`), and how to turn there is :class:`GazePlan`'s split between the
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
class Point:
    """A point in the ``map`` frame, metres (``z`` above the floor): where a person's head is,
    or where the head is asked to look."""

    x: float
    y: float
    z: float

    def plus(self, dx: float, dy: float, dz: float) -> Point:
        """This point moved by ``(dx, dy, dz)`` metres."""
        return Point(self.x + dx, self.y + dy, self.z + dz)

    def distance(self, other: Point) -> float:
        """Metres to ``other``."""
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))

    def rounded(self, digits: int = 2) -> list[float]:
        """``[x, y, z]`` rounded, for an event."""
        return [round(self.x, digits), round(self.y, digits), round(self.z, digits)]


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
    measures person recognition's gate reads; ``depth_m`` the stereo depth at the face when the
    finder measures one (None: unmeasured, the face's size gives the range)."""

    pan_deg: float
    tilt_deg: float
    width_px: float
    score: float
    who: str | None = None
    yaw_deg: float | None = None
    detection: Detection | None = field(default=None, compare=False, repr=False)
    sharpness: float | None = field(default=None, compare=False, repr=False)
    brightness: float | None = field(default=None, compare=False, repr=False)
    depth_m: float | None = None


@dataclass(frozen=True)
class BodySighting:
    """One person's box in a frame (a body detector's, :class:`pepin.expression.finder.
    BodyDetector`), in degrees from the optical axis (pan + left, tilt + down): the box's centre
    pan and half width, the head estimated from it (near the box's top: standing, sitting,
    slouching and lying all give one), its height, the detector's score, whether the frame's
    top edge cuts it (the head is then above the picture: the estimate is the edge), and the
    stereo depth at the head estimate when the finder measures one (None: unmeasured)."""

    pan_deg: float
    half_width_deg: float
    head_pan_deg: float
    head_tilt_deg: float
    height_deg: float
    score: float
    cut: bool = False
    depth_m: float | None = None


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
    whether a drive runs, the source of the overlay playing on top ("" for none), and the
    board's stamp of the encoder reading ``head`` is (None from an arbiter that sends none)."""

    head: Aim | None
    holder: str
    band: int
    blind: bool
    blind_until: float | None
    driving: bool
    overlay: str = ""
    head_stamp: float | None = None


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

    def aim(
        self, point: Point, *, speed: Speed, ttl_s: float, ease: Ease | None = None
    ) -> GazeAnswer:
        """Hold the head on ``point`` in the map (replacing this source's last aim): the
        arbiter resolves it to pan and tilt from the pose of now and keeps it there as the cart
        moves; moving at ``speed`` or, given, on ``ease``; answered when the head has settled
        there, or refused."""
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
# A gesture overlay plays only on a person seen within this (DESIGN.md §8.4; the engine's knob
# layers.head_seen_s): frames come at 1 Hz while the robot speaks, so a second and a margin
HEAD_SEEN_S = 1.2


@dataclass(frozen=True)
class Attention:
    """A read-only snapshot of the attention engine (the speaker gaze), laptop monotonic clock:
    its phase and since when, whether it has eyes (a face finder: frames to protect), when the
    person engaged was last seen by a face or a body (None: never), their bearing and tilt
    (degrees, the head's pan and tilt that look at them from the cart's newest pose, + left,
    + down) and rough distance (None: unknown), whether they stand past the forward arc (a body
    turn is wanted), when their first clear face of the conversation was found, and their point
    in the map."""

    phase: AttentionPhase = "idle"
    since_s: float = 0.0
    eyes: bool = False
    seen_s: float | None = None
    bearing_deg: float | None = None
    tilt_deg: float | None = None
    distance_m: float | None = None
    past_arc: bool = False
    first_face_s: float | None = None
    point: Point | None = None


NO_ATTENTION = Attention()  # the speaker gaze off, or no conversation: nothing to keep clear


class AttentionView(Protocol):
    """The attention engine as the gesture engine reads it (never writes): the latest snapshot,
    from any thread."""

    def attention(self) -> Attention:
        """The snapshot now (:data:`NO_ATTENTION` while the gaze is off)."""
        ...


class ArmRelease(Protocol):
    """The gaze's body turn asks the gesture engine for the arm first: attention's rule (no arm
    motion while the body turns toward a person, gesticulation DESIGN.md §6.5), the one place
    the arm waits for the wheels."""

    def release_arm(self, why: str) -> float:
        """Send the arm to rest now (no new arm motion until the turn is over); the seconds until
        it is there (0: it is already). Never blocks on the arm."""
        ...


def head_refusal(view: Attention, now: float, seen_s: float = HEAD_SEEN_S) -> str | None:
    """Why a gesture overlay may not take the head now, None when it may: while attention looks
    for the person (a seek, a check), turns the body, or has not seen them within ``seen_s``,
    a gesture's skipped frames would blind it (the idle layer's still overlays never do). No
    rule without eyes, a conversation, or during a drive."""
    if not view.eyes or view.phase in ("idle", "paused"):
        return None
    if view.phase != "track":
        return f"attention: {view.phase}"
    if view.seen_s is None or now - view.seen_s > seen_s:
        unseen = "never" if view.seen_s is None else f"{now - view.seen_s:.1f} s"
        return f"attention: the person unseen for {unseen}"
    return None


def arm_refusal(view: Attention, past_arc_rule: bool = True) -> str | None:
    """Why no arm motion may start now, None when it may: the body turns toward the person
    (attention's turn goes first), or, under ``past_arc_rule`` (layers.arm_past_arc; off since
    2026-10-14, Artem: the rule held the arm for a person a room away), the person engaged
    stands past the forward arc."""
    if view.phase == "turning":
        return "attention: the body turns toward the person"
    if past_arc_rule and view.past_arc:
        return "attention: the person stands past the forward arc"
    return None
