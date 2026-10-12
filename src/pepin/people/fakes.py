"""Fakes of person recognition's models and camera, for tests: synthetic embeddings at chosen
cosines, voices told apart by loudness, faces by where their box starts."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from pepin.expression.speaker import FaceFrame, FaceSighting
from pepin.face_sensor.identity import Vector, unit
from pepin.face_sensor.models import Detection
from pepin.people.voice import RATE

VOICE_STEP = 1000  # a fake voice's speaker number is its peak amplitude / VOICE_STEP


def random_unit(seed: int, dim: int = 64) -> Vector:
    """A random unit vector (the same for the same seed)."""
    return unit(np.random.default_rng(seed).standard_normal(dim))


def near(v: Vector, cos: float, seed: int) -> Vector:
    """A unit vector at cosine ``cos`` to ``v``."""
    noise = random_unit(seed, len(v))
    noise = unit(noise - float(noise @ v) * v)
    return unit(cos * v + float(np.sqrt(max(0.0, 1.0 - cos * cos))) * noise)


def speech(speaker: int, seconds: float, lead_s: float = 0.3) -> bytes:
    """PCM a :class:`FakeVoiceNet` reads as ``speaker``: quiet lead-in, then ``seconds`` of a
    loud tone whose peak is ``speaker * VOICE_STEP``, then quiet."""
    quiet = np.zeros(int(lead_s * RATE), dtype=np.int16)
    t = np.arange(int(seconds * RATE)) / RATE
    tone = (speaker * VOICE_STEP * np.sin(2 * np.pi * 220 * t)).astype(np.int16)
    tone[len(tone) // 2] = speaker * VOICE_STEP  # the exact peak
    return np.concatenate([quiet, tone, quiet]).astype("<i2").tobytes()


@dataclass
class FakeVoiceNet:
    """:class:`pepin.people.voice.VoiceEmbedder` whose speaker is the PCM's peak / step:
    ``voices[k]``, or ``scripted`` vectors in order when given."""

    voices: dict[int, Vector] = field(default_factory=dict)
    scripted: list[Vector] = field(default_factory=list)
    name: str = "fake-voice"
    calls: int = 0

    def embed(self, samples: NDArray[np.int16]) -> Vector:
        """The speaker's vector."""
        self.calls += 1
        if self.scripted:
            return self.scripted.pop(0)
        speaker = round(int(np.abs(samples.astype(np.int32)).max()) / VOICE_STEP)
        return self.voices[speaker]


@dataclass
class FakeIdentity:
    """:class:`pepin.face_sensor.models.Embedder` whose face is keyed by its box's left edge."""

    faces: dict[float, Vector] = field(default_factory=dict)
    name: str = "fake-face"
    calls: int = 0

    def embed(self, image: NDArray[np.uint8], face: Detection) -> Vector:
        """The face's vector."""
        self.calls += 1
        return self.faces[face.box[0]]


def textured(height: int = 600, width: int = 800, seed: int = 0) -> NDArray[np.uint8]:
    """A picture sharp and mid-grey enough for every face crop in it to pass the gate."""
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, size=(height, width, 3), dtype=np.uint8)


def detection(x0: float, width: float, score: float = 0.9, yaw: float = 0.0) -> Detection:
    """A face box at ``x0`` (its key in :class:`FakeIdentity`), square, with five points; the
    nose shifted for ``yaw`` degrees."""
    y0 = 100.0
    lx, rx = x0 + 0.3 * width, x0 + 0.7 * width
    nose = (lx + rx) / 2 + float(np.sin(np.radians(yaw))) * (rx - lx) / 2
    kps = ((lx, y0 + 0.4 * width), (rx, y0 + 0.4 * width), (nose, y0 + 0.6 * width),
           (lx, y0 + 0.8 * width), (rx, y0 + 0.8 * width))  # fmt: skip
    return Detection((x0, y0, x0 + width, y0 + width), score, kps)


def frame(*faces: Detection, image: NDArray[np.uint8] | None = None) -> FaceFrame:
    """A finder's frame with these faces (their detections and the picture kept)."""
    sightings = tuple(
        FaceSighting(0.0, 0.0, d.width, d.score, yaw_deg=0.0, detection=d) for d in faces
    )
    return FaceFrame(0.0, sightings, image=textured() if image is None else image)


@dataclass
class FakeFinder:
    """:class:`pepin.expression.speaker.FaceFinder` answering ``frames`` in order (the last
    one again when they run out; None with none)."""

    frames: list[FaceFrame] = field(default_factory=list)
    looks: int = 0

    def look(self) -> FaceFrame | None:
        """The next scripted frame."""
        self.looks += 1
        if not self.frames:
            return None
        return self.frames.pop(0) if len(self.frames) > 1 else self.frames[0]

    def show(self, frames: Sequence[FaceFrame]) -> None:
        """Script the next frames."""
        self.frames = list(frames)
