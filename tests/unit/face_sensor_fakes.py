"""Stand-in models and pictures for the face sensor's unit tests."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

from pepin.face_sensor.identity import unit
from pepin.face_sensor.models import EMOTIONS, Detection, Expression, Mesh

Image = NDArray[np.uint8]


def noise(height: int = 600, width: int = 800, seed: int = 0) -> Image:
    """A textured mid-grey picture: sharp and neither dark nor bright to the quality gate."""
    rng = np.random.default_rng(seed)
    return rng.integers(60, 200, size=(height, width, 3), dtype=np.uint8)


def flat(height: int = 600, width: int = 800, level: int = 128) -> Image:
    """A featureless picture: zero sharpness."""
    return np.full((height, width, 3), level, dtype=np.uint8)


def box(cx: float, cy: float, width: float) -> Detection:
    """A square face box of ``width`` px centred at ``(cx, cy)`` with plausible landmarks."""
    h = width / 2
    kps = (
        (cx - h / 2, cy - h / 3),
        (cx + h / 2, cy - h / 3),
        (cx, cy),
        (cx - h / 3, cy + h / 2),
        (cx + h / 3, cy + h / 2),
    )
    return Detection((cx - h, cy - h, cx + h, cy + h), 0.9, kps)


def direction(seed: int, dim: int = 512) -> NDArray[np.float32]:
    """A random unit vector: one person's identity."""
    return unit(np.random.default_rng(seed).normal(size=dim))


def near(vector: NDArray[np.float32], cos: float, seed: int = 99) -> NDArray[np.float32]:
    """A unit vector at cosine ``cos`` to ``vector``."""
    rng = np.random.default_rng(seed)
    other = rng.normal(size=vector.shape[0]).astype(np.float32)
    other -= float(other @ vector) * vector
    other = unit(other)
    return unit(cos * vector + np.sqrt(max(0.0, 1 - cos * cos)) * other)


class FakeDetector:
    """Answers the same boxes for every image (or what ``by_image`` says), counting calls."""

    def __init__(
        self,
        faces: list[Detection] | None = None,
        by_image: Callable[[Image], list[Detection]] | None = None,
    ) -> None:
        self.faces = faces or []
        self.by_image = by_image
        self.calls = 0

    def detect(self, image: Image) -> list[Detection]:
        self.calls += 1
        return self.by_image(image) if self.by_image else list(self.faces)


class FakeEmbedder:
    """Answers a fixed embedding (or the next of a list), counting calls."""

    name = "fake_pack"

    def __init__(self, vectors: list[NDArray[np.float32]]) -> None:
        self.vectors = vectors
        self.calls = 0

    def embed(self, image: Image, face: Detection) -> NDArray[np.float32]:
        v = self.vectors[min(self.calls, len(self.vectors) - 1)]
        self.calls += 1
        return v


def expression(
    top: str = "neutral", p: float = 0.6, valence: float = 0.1, arousal: float = -0.2
) -> Expression:
    """An expression with ``p`` on ``top`` and the rest spread evenly."""
    rest = (1 - p) / (len(EMOTIONS) - 1)
    return Expression(tuple(p if e == top else rest for e in EMOTIONS), valence, arousal)


class FakeExpression:
    """Answers a fixed expression, counting calls."""

    def __init__(self, answer: Expression | None = None) -> None:
        self.answer = answer or expression()
        self.calls = 0

    def predict(self, crop: Image) -> Expression:
        self.calls += 1
        return self.answer


class FakeLandmarker:
    """Answers meshes in turn (the last one repeats), counting calls and crop shapes."""

    def __init__(self, meshes: list[Mesh | None] | None = None) -> None:
        self.meshes = meshes or [Mesh(yaw=2.0, pitch=-3.0, roll=0.0, jaw_open=0.05)]
        self.calls = 0
        self.shapes: list[tuple[int, ...]] = []

    def read(self, crop: Image) -> Mesh | None:
        self.shapes.append(crop.shape)
        mesh = self.meshes[min(self.calls, len(self.meshes) - 1)]
        self.calls += 1
        return mesh
