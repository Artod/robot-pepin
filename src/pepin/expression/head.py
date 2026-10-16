"""The head's contract with the gaze arbiter: who asks, in which band, and the ``clip`` a head
gesture is.

The arbiter (:mod:`pepin.gaze`, ``pepin_bringup.gaze``) stays the head's one owner, and
expression asks it through its JSON door (:mod:`pepin.gaze_link`) in the IDLE band, below every
other band (operator, navigation, the person's word, sensor checks, driving): a ``look`` the
person asks for takes the head from the conversation.

* The speaker gaze, source :data:`SPEAKER`, HOLDS the head on the person (the base layer): an
  ``angles`` look with ``hold``, a short TTL renewed while the conversation lasts, replaced by
  its own next aim. A request that took the head from it ends it (the arbiter's rule), and the
  speaker gaze asks again when its renewal finds nothing to renew.
* A head gesture, source :data:`GESTURE` (or :data:`IDLE_LOOK` for the idle layer), is a
  :class:`HeadClip` OVERLAY (``kind`` ``"clip"``, :class:`pepin.gaze.Overlay`): keyed offsets added
  to the aim of whatever holds the head (home when nothing does), each key written at its time at
  its own speed (and its own ramp when it carries one: the idle layer's keys glide). An overlay
  takes nothing from the holder, which keeps its status, its TTL and its aim; when the overlay ends
  the head is written back to the holder's aim, so a thinking glance pulls the head off the person
  and brings it back without the speaker gaze losing them. It plays only over a holder of its band
  or none (never over the person's look, a sensor check or a drive), one at a time: a newer overlay
  replaces the one playing, except that an ``idle`` one (breathing, a fidget) gives way to any other
  and never replaces one. A ``loop`` repeats its keys (its last key closes the period and equals its
  first) until released or its TTL lapses, renewed like any request: the idle layer's continuous
  breath is one looped clip. The head is blind while an overlay's moves run, as for any write,
  except for an overlay whose offsets all stay within the arbiter's ``clip_still_deg`` (a breath):
  that one moves the head without blinding it. Refused whole, never clamped: a key past the reach or
  ``clip_max_deg``, a tilt reversing faster than ``clip_max_hz`` (2.5 Hz: the mast rings at 5.3 Hz;
  the board's tilt ramp caps every move at 600 deg/s^2 by itself), a key asking the tilt for a
  ramp of its own above ``clip_acc_max_deg_s2`` (a paced nod's, as config/knobs.json ships the
  knob: the engine sizes the nod to it); the gesture then plays without its head.
  ``POST /release`` with ``fade_s`` (the head's of config/expression.json's ``layers.fade_s``)
  takes it back over that long.

On the door a clip is a :class:`ClipLook`, answered when it ends or is refused (a loop at once):
:func:`pepin.expression.parts.clip_look` makes it from a rendered clip (each key written one move
before its keyframe, at the speed that lands it on time), :class:`pepin.expression.parts.HeadPart`
plays it, and :class:`pepin.expression.door.OverlayDoor` sends it with each source's
:class:`pepin.gaze_link.GazeClient`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, NotRequired, TypedDict

from pepin.gaze import IDLE

BAND = IDLE  # the band of every expression request
SPEAKER = "voice.speaker"  # the speaker gaze's source (the head's base layer)
GESTURE = "voice.gesture"  # a gesture's head overlay
IDLE_LOOK = "voice.idle"  # the idle layer's micro-motions
CLIP = "clip"  # the arbiter's look kind for a timed overlay of relative keys


@dataclass(frozen=True)
class HeadKey:
    """One key of a head clip: seconds from the clip's start, the pan (+ left) and tilt (+ down)
    offsets in radians, the speed ceiling of the move to it and its ramp's (None: the neck's),
    and the tilt's own ramp (None: the neck's tilt ramp)."""

    at_s: float
    pan_rad: float
    tilt_rad: float
    speed_deg_s: float
    acc_deg_s2: float | None = None
    tilt_acc_deg_s2: float | None = None


@dataclass(frozen=True)
class HeadClip:
    """A head gesture: its keys in time order, offsets from the holder's aim; the last key is
    normally zero, and the head returns to the holder's aim after it in any case."""

    keys: tuple[HeadKey, ...]


class ClipKey(TypedDict):
    """A :class:`HeadKey` on the door (``acc_deg_s2`` only when the key has a ramp of its own,
    ``tilt_acc_deg_s2`` only when its tilt asks for more than the neck's tilt ramp: a paced
    nod's, at most the arbiter's ``clip_acc_max_deg_s2``)."""

    at_s: float
    pan_rad: float
    tilt_rad: float
    speed_deg_s: float
    acc_deg_s2: NotRequired[float]
    tilt_acc_deg_s2: NotRequired[float]


class ClipTarget(TypedDict):
    """The ``target`` of a clip look: its keys, whether they repeat, whether it is the idle
    layer's (it gives way to any other overlay)."""

    keys: list[ClipKey]
    loop: NotRequired[bool]
    idle: NotRequired[bool]


class ClipLook(TypedDict):
    """``POST /look`` of a head overlay, answered when it ends or is refused."""

    source: str
    kind: Literal["clip"]
    band: int
    ttl_s: float
    target: ClipTarget
