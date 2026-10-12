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
head and the wheels: the head alone within its comfortable pan, the wheels for the rest in small
steps when they may move, and the head as far as it reaches when they may not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

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
    person recognition aligns the face on) when the finder keeps it."""

    pan_deg: float
    tilt_deg: float
    width_px: float
    score: float
    who: str | None = None
    yaw_deg: float | None = None
    detection: Detection | None = field(default=None, compare=False, repr=False)


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
    it the picture (BGR) the faces were found in, and the people's boxes when it has a body
    detector."""

    stamp: float
    faces: tuple[FaceSighting, ...]
    image: NDArray[np.uint8] | None = field(default=None, compare=False, repr=False)
    bodies: tuple[BodySighting, ...] = ()


class FaceFinder(Protocol):
    """Faces on demand: one frame from the head camera, searched now (never a stream)."""

    def look(self) -> FaceFrame | None:
        """The faces in a fresh frame; None when no frame came."""
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
