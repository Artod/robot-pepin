"""Enrolment: the embeddings of one person's face from still images or a short clip.

A frame gives a sample only when its face is unambiguous (one face, or the widest at least
:data:`DOMINANT` times as wide as the next), at least ``enrol_min_px`` wide, and passes the
quality gate (not dark, bright or blurry). The samples are added to ``data/faces/<name>.npz``
(:func:`pepin.face_sensor.identity.save_person`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from pepin.face_sensor.config import FaceSensorConfig
from pepin.face_sensor.frames import Frame
from pepin.face_sensor.geometry import crop_box
from pepin.face_sensor.identity import Vector
from pepin.face_sensor.models import Detector, Embedder
from pepin.face_sensor.pipeline import best_face
from pepin.face_sensor.quality import gate, measure

DOMINANT = 1.5


@dataclass(frozen=True)
class Sample:
    """One frame's verdict: the embedding when it was taken, or why it was not."""

    index: int
    face_px: float
    embedding: Vector | None
    reason: str


def collect(
    frames: Iterable[Frame], detector: Detector, embedder: Embedder, cfg: FaceSensorConfig
) -> list[Sample]:
    """A :class:`Sample` per frame of ``frames``."""
    samples = []
    for frame in frames:
        image = frame.image()
        faces = detector.detect(image)
        best = best_face(faces)
        if best is None:
            samples.append(Sample(frame.index, 0.0, None, "no face"))
            continue
        others = sorted((f.width for f in faces if f is not best), reverse=True)
        if others and best.width < DOMINANT * others[0]:
            samples.append(Sample(frame.index, best.width, None, f"{len(faces)} faces"))
            continue
        if best.width < cfg.identity.enrol_min_px:
            samples.append(Sample(frame.index, best.width, None, "small"))
            continue
        g = gate(best.width, measure(crop_box(image, best.box)), cfg.gate)
        if not g.identity:
            samples.append(Sample(frame.index, best.width, None, g.reason))
            continue
        samples.append(Sample(frame.index, best.width, embedder.embed(image, best), "ok"))
    return samples
