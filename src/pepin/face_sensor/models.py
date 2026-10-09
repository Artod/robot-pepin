"""The four models of the face sensor as small protocols, and what each one answers.

The real ones (SCRFD + ArcFace from InsightFace, EmotiEffLib's expression net, MediaPipe's Face
Landmarker) are in :mod:`pepin.face_sensor.backends` and import their libraries only there; the
pipeline and the tests see these protocols.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Image = NDArray[np.uint8]
Vector = NDArray[np.float32]

# EmotiEffLib's eight AffectNet classes, in its output order (enet_b0_8_*).
EMOTIONS = (
    "anger",
    "contempt",
    "disgust",
    "fear",
    "happiness",
    "neutral",
    "sadness",
    "surprise",
)


@dataclass(frozen=True)
class Detection:
    """One face box: corners in pixels, the detector's score, and the five landmarks
    (eyes, nose, mouth corners) the identity crop is aligned on."""

    box: tuple[float, float, float, float]
    score: float
    kps: tuple[tuple[float, float], ...] = ()

    @property
    def width(self) -> float:
        """The box width in pixels: the size every gate is keyed on."""
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        """The box height in pixels."""
        return self.box[3] - self.box[1]

    @property
    def centre(self) -> tuple[float, float]:
        """The box centre in pixels."""
        return (self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0


@dataclass(frozen=True)
class Expression:
    """One crop's expression: a probability per :data:`EMOTIONS` entry, valence and arousal
    (roughly -1..1)."""

    probs: tuple[float, ...]
    valence: float
    arousal: float

    def top(self) -> tuple[str, float]:
        """The most probable label and its probability."""
        i = int(np.argmax(self.probs))
        return EMOTIONS[i], self.probs[i]


@dataclass(frozen=True)
class Mesh:
    """What the landmarker read on a crop: head pose in degrees (0, 0 = facing the camera) and
    the blendshapes the sensor uses (0..1)."""

    yaw: float
    pitch: float
    roll: float
    jaw_open: float
    smile: float = 0.0
    blink: float = 0.0


class Detector(Protocol):
    """Finds faces in a BGR image."""

    def detect(self, image: Image) -> list[Detection]:
        """Every face above the detector's threshold."""
        ...


class Embedder(Protocol):
    """The identity vector of one detected face."""

    name: str

    def embed(self, image: Image, face: Detection) -> Vector:
        """The face's embedding, aligned on its landmarks, scaled to unit length."""
        ...


class ExpressionNet(Protocol):
    """Expression probabilities and valence/arousal of a face crop."""

    def predict(self, crop: Image) -> Expression:
        """The crop is BGR, the tight detector box."""
        ...


class Landmarker(Protocol):
    """Head pose and blendshapes of a face crop."""

    def read(self, crop: Image) -> Mesh | None:
        """The crop is BGR, the box grown and squared; ``None`` when no face is found in it."""
        ...
