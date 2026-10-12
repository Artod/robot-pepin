"""Whom to look at: the voice that woke the robot, then whoever the array hears instead.

The array's direction of arrival names the loudest voice of an utterance, so a new speaker is an
utterance whose bearing is ``doa_jump_deg`` or more from the person the head holds. In a frame,
the speaker's face is the one within ``match_deg`` of the voice's bearing, nearest to it; with no
such face (or no bearing: the map unmeasured, the robot's own voice), the face nearest the
frame's centre, the widest on a tie.

The seams for later: :class:`Addressee` names whom the conversation addresses ("look at NAME", two
interlocutors), and a face the finder knows (:attr:`FaceSighting.who`) with that name is taken
before any other.
"""

from __future__ import annotations

import math
from typing import Protocol

from pepin.expression.speaker import FaceFrame, FaceSighting
from pepin.gaze import Aim


class Addressee(Protocol):
    """Whom the conversation looks at by name (LATER: the brain's "look at NAME", the second of
    two interlocutors); None for whoever speaks."""

    def wanted(self) -> str | None:
        """The name to look at, None for the speaker."""
        ...


class SpeakerPolicy:
    """:class:`pepin.expression.speaker.TargetPolicy` for one speaker at a time."""

    def __init__(
        self, *, match_deg: float, doa_jump_deg: float, addressee: Addressee | None = None
    ) -> None:
        """The face-to-voice match and the jump that makes a new speaker, in degrees."""
        self.match_rad = math.radians(match_deg)
        self.jump_rad = math.radians(doa_jump_deg)
        self.addressee = addressee

    def choose(self, voice: float | None, frame: FaceFrame | None, head: Aim) -> Aim | None:
        """The aim that centres the chosen face of ``frame`` (taken with the head at ``head``),
        None when it has none."""
        face = self.pick(voice, frame, head)
        if face is None:
            return None
        return face_aim(head, face)

    def pick(self, voice: float | None, frame: FaceFrame | None, head: Aim) -> FaceSighting | None:
        """The speaker's face: the addressee's by name, else the one nearest the voice within
        ``match_deg``, else the one nearest the centre (widest on a tie)."""
        if frame is None or not frame.faces:
            return None
        name = self.addressee.wanted() if self.addressee is not None else None
        named = [f for f in frame.faces if name is not None and f.who == name]
        if named:
            return max(named, key=lambda f: f.width_px)
        if voice is not None:
            heard = voice

            def miss(face: FaceSighting) -> float:
                return abs(_wrap(face_aim(head, face).pan_rad - heard))

            near = [f for f in frame.faces if miss(f) <= self.match_rad]
            if near:
                return min(near, key=lambda f: (miss(f), -f.width_px))
        return min(frame.faces, key=lambda f: (abs(f.pan_deg), -f.width_px))

    def new_speaker(self, held: float | None, heard: float | None) -> bool:
        """Whether an utterance heard at bearing ``heard`` comes from someone other than the
        person held at ``held`` (radians, + left)."""
        if heard is None:
            return False
        if held is None:
            return True
        return abs(_wrap(heard - held)) > self.jump_rad


def face_aim(head: Aim, face: FaceSighting) -> Aim:
    """The aim that puts ``face``, seen with the head at ``head``, on the optical axis."""
    pan, tilt = math.radians(face.pan_deg), math.radians(face.tilt_deg)
    return Aim(head.pan_rad + pan, head.tilt_rad + tilt)


def _wrap(rad: float) -> float:
    return math.atan2(math.sin(rad), math.cos(rad))
