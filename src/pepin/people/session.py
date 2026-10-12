"""One conversation's speakers, and how a name binds to one of them (collect, then bind).

Every utterance's measurements (:class:`Observation`: its voice embedding when long enough,
the speaker's face when one passed the gate, the other faces in view) go to a :class:`Track`:
the track of the person the verdict names, or for an utterance nobody is named for, the
stranger heard earlier in this conversation it resembles (face at ``track_face``, voice at
``track_voice``), else a new stranger. Tracks keep their last ``track_keep`` samples per
modality. Nothing of a stranger reaches the store unless a name binds them; the conversation's
end forgets them.

``remember_person`` binds :meth:`Session.to_bind`: the stranger who spoke last (they said their
own name), else (a known person introduced someone: "это Саша") the one stranger's face in view
at the latest utterance. :func:`learnable` says which samples a known person learns from an
utterance named ``sure``: a modality's sample is learned when that modality agrees, or when the
person has no sample of it yet (a voice learned from a sure face, a face from a sure voice).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from pepin.face_sensor.identity import Vector, centroid
from pepin.people.config import SessionKnobs
from pepin.people.fusion import Verdict
from pepin.people.matcher import Reading


@dataclass(frozen=True)
class Observation:
    """What one utterance gave: the voice embedding (None: too short), the speaker's face
    embedding (None: no face passed), its width, and the other faces in view that passed."""

    end_s: float
    voice: Vector | None = None
    speech_s: float = 0.0
    face: Vector | None = None
    face_px: float = 0.0
    others: tuple[Vector, ...] = ()

    @property
    def evidence(self) -> bool:
        """Whether there is anything to tell a person by."""
        return self.voice is not None or self.face is not None


@dataclass
class Track:
    """One speaker of this conversation: the person when named, their recent samples."""

    id: int
    person: str | None
    voice: list[Vector] = field(default_factory=list)
    face: list[Vector] = field(default_factory=list)
    utterances: int = 0
    last_s: float = 0.0

    def add(self, obs: Observation, keep: int) -> None:
        """One utterance's samples, the newest ``keep`` of each modality kept."""
        if obs.voice is not None:
            self.voice = [*self.voice, obs.voice][-keep:]
        if obs.face is not None:
            self.face = [*self.face, obs.face][-keep:]
        self.utterances += 1
        self.last_s = max(self.last_s, obs.end_s)

    def resembles(self, obs: Observation, knobs: SessionKnobs) -> float:
        """How well ``obs`` fits this track: the best of the face and the voice cosine to the
        track's centroids, each counted only over its threshold (0: no fit)."""
        best = 0.0
        if obs.face is not None and self.face:
            cos = float(centroid(np.stack(self.face)) @ obs.face)
            if cos >= knobs.track_face:
                best = max(best, cos)
        if obs.voice is not None and self.voice:
            cos = float(centroid(np.stack(self.voice)) @ obs.voice)
            if cos >= knobs.track_voice:
                best = max(best, cos)
        return best


class Session:
    """The speakers of one conversation, in the order they were heard."""

    def __init__(self, knobs: SessionKnobs) -> None:
        """No one heard yet."""
        self.knobs = knobs
        self.tracks: list[Track] = []
        self.latest: Track | None = None
        self.latest_obs: Observation | None = None

    def attribute(self, obs: Observation, verdict: Verdict) -> Track | None:
        """Put the utterance on its speaker's track (a new one when needed); None when it
        names nobody and carries nothing to tell a stranger by."""
        self.latest_obs = obs
        track: Track | None
        if verdict.person is not None:
            track = next((t for t in self.tracks if t.person == verdict.person), None)
            if track is None:
                track = self._new(verdict.person)
        else:
            if not obs.evidence:
                self.latest = None
                return None
            strangers = [t for t in self.tracks if t.person is None]
            scored = [(t.resembles(obs, self.knobs), t) for t in strangers]
            fit = max(scored, key=lambda s: s[0], default=(0.0, None))
            track = fit[1] if fit[0] > 0 else self._new(None)
        assert track is not None
        track.add(obs, self.knobs.track_keep)
        self.latest = track
        return track

    def to_bind(self) -> Track | None:
        """The stranger who spoke last, if the last utterance was a stranger's."""
        latest = self.latest
        if latest is not None and latest.person is None and (latest.voice or latest.face):
            return latest
        return None

    def strangers_in_view(self, known: Callable[[Vector], bool]) -> list[Vector]:
        """The faces in view at the latest utterance that ``known`` does not recognise (the
        speaker's face among them when it was not recognised)."""
        obs = self.latest_obs
        if obs is None:
            return []
        faces = [*obs.others, *([obs.face] if obs.face is not None else [])]
        return [f for f in faces if not known(f)]

    def bound(self, track: Track, person: str) -> None:
        """``track`` now is ``person`` (a name was given)."""
        track.person = person

    def forget(self, person: str) -> None:
        """Drop every track of ``person`` (they asked to be forgotten)."""
        self.tracks = [t for t in self.tracks if t.person != person]
        if self.latest is not None and self.latest.person == person:
            self.latest = None

    def _new(self, person: str | None) -> Track:
        track = Track(len(self.tracks) + 1, person)
        self.tracks.append(track)
        return track


def learnable(reading: Reading | None, person: str, has_samples: bool) -> bool:
    """Whether an utterance named ``person`` sure teaches them this modality's sample: the
    modality agrees (its best is ``person`` and it is no stranger's), or ``person`` has no
    sample of it yet."""
    if reading is None:
        return False
    if not has_samples:
        return True
    return reading.person == person and reading.label != "stranger"
