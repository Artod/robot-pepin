"""One modality's verdict: an embedding against everyone's centroid.

A person's centroid is the normalised mean of their samples of that modality and model
(:func:`pepin.face_sensor.identity.centroid`); an embedding is scored against every centroid
by cosine (:class:`pepin.face_sensor.identity.Gallery`). The verdict is ``match`` when the best
cosine reaches ``accept`` and beats the runner-up person by ``margin`` (two people alike, or a
sibling, abstain rather than guess), ``stranger`` when even the best is under ``stranger``,
``unsure`` in between. ``sure`` marks a match strong enough to stand alone (fusion,
:mod:`pepin.people.fusion`).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from pepin.face_sensor.identity import Gallery, Vector, centroid, unit
from pepin.people.store import Modality

Label = Literal["match", "stranger", "unsure"]


@dataclass(frozen=True)
class Thresholds:
    """Cosines of one modality: a match (``accept`` and ``margin`` over the runner-up), a
    stranger (under ``stranger``), a match that stands alone (``sure``)."""

    accept: float
    margin: float
    stranger: float
    sure: float

    def check(self, what: str) -> None:
        """``ValueError`` unless -1 <= stranger <= accept <= sure and margin >= 0."""
        if not -1.0 <= self.stranger <= self.accept <= self.sure or self.margin < 0:
            raise ValueError(f"{what}: needs -1 <= stranger <= accept <= sure, margin >= 0")


@dataclass(frozen=True)
class Reading:
    """One modality's verdict on one utterance: the best person (an id; None with nobody to
    compare with), the cosine, the runner-up's, the label, whether it stands alone, and whether
    the observation was a strong one (a face close and facing; a voice long enough)."""

    modality: Modality
    person: str | None
    cos: float
    second: float
    label: Label
    sure: bool = False
    strong: bool = False


def galleries(samples: Mapping[str, NDArray[np.float32]]) -> Gallery:
    """Everyone's centroid from their samples (id to an ``(n, d)`` matrix)."""
    return Gallery({pid: centroid(rows) for pid, rows in samples.items() if len(rows)})


def judge(
    gallery: Gallery,
    embedding: Vector,
    t: Thresholds,
    modality: Modality,
    *,
    strong: bool = False,
) -> Reading:
    """``embedding`` against ``gallery``: the best person, the label by ``t``."""
    m = gallery.match(unit(embedding))
    if m.name is not None and m.cos >= t.accept and m.cos - m.second >= t.margin:
        return Reading(modality, m.name, m.cos, m.second, "match", m.cos >= t.sure, strong)
    label: Label = "stranger" if m.cos < t.stranger else "unsure"
    return Reading(modality, m.name, m.cos, m.second, label, False, strong)


def spread(rows: NDArray[np.float32]) -> float:
    """The mean cosine of samples to their own centroid (1.0 for one sample)."""
    if len(rows) == 0:
        return 0.0
    return float(np.mean(rows @ centroid(rows)))
