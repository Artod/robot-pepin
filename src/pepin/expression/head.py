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
  :class:`HeadClip` OVERLAY (``kind`` ``"clip"``): keyed offsets added to the aim of whatever
  holds the head (home when nothing does), each key written at its time at its own speed. An
  overlay takes nothing from the holder, which keeps its status, its TTL and its aim; when the
  overlay ends the head is written back to the holder's aim. It plays only over a holder of its
  band or none (never over the person's look, a sensor check or a drive), one at a time (a newer
  overlay replaces the one playing), and the head is blind while it plays. A key past the reach
  refuses the whole overlay (the arbiter never clamps): the gesture then plays without its head.

On the door a clip is a :class:`ClipLook`, answered when it ends or is refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypedDict

from pepin.gaze import IDLE

BAND = IDLE  # the band of every expression request
SPEAKER = "voice.speaker"  # the speaker gaze's source (the head's base layer)
GESTURE = "voice.gesture"  # a gesture's head overlay
IDLE_LOOK = "voice.idle"  # the idle layer's micro-motions
CLIP = "clip"  # the arbiter's look kind for a timed overlay of relative keys


@dataclass(frozen=True)
class HeadKey:
    """One key of a head clip: seconds from the clip's start, the pan (+ left) and tilt (+ down)
    offsets in radians, and the speed ceiling of the move to it."""

    at_s: float
    pan_rad: float
    tilt_rad: float
    speed_deg_s: float


@dataclass(frozen=True)
class HeadClip:
    """A head gesture: its keys in time order, offsets from the holder's aim; the last key is
    normally zero, and the head returns to the holder's aim after it in any case."""

    keys: tuple[HeadKey, ...]


class ClipKey(TypedDict):
    """A :class:`HeadKey` on the door."""

    at_s: float
    pan_rad: float
    tilt_rad: float
    speed_deg_s: float


class ClipTarget(TypedDict):
    """The ``target`` of a clip look: its keys."""

    keys: list[ClipKey]


class ClipLook(TypedDict):
    """``POST /look`` of a head overlay, answered when it ends or is refused."""

    source: str
    kind: Literal["clip"]
    band: int
    ttl_s: float
    target: ClipTarget
