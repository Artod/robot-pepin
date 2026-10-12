"""Face and voice into one answer: who said this utterance, how sure, and on what grounds.

======================  ==================  ===============================================
face                    voice               verdict
======================  ==================  ===============================================
match A                 match A             A, sure
match A                 match B             A tentative when the face is strong (close,
                                            facing, sharp), else B tentative
match A                 none / unsure       A, sure when the face's cosine reaches its
                                            ``sure``, else tentative
none / unsure           match A             A, sure when the voice's reaches its ``sure``,
                                            else tentative
match A                 stranger            A, tentative (someone else may have spoken)
stranger                match A             A, tentative (A may speak from beside a guest)
stranger                stranger            new
stranger                none / unsure       new
none / unsure           stranger            new
none / unsure           none / unsure       unknown
======================  ==================  ===============================================

``new`` means evidence of a person nobody has met; ``unknown`` means nothing to go on (a short
"да" with no face in view), which names nobody and claims no stranger either.

A modality that has never met the person the other one names (Sasha was named by his face and
has not spoken yet) reads him as a stranger only for want of samples: :func:`comparable` makes
that reading ``unsure`` before the table is applied, so his face alone can be sure and his
voice is learned from it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal

from pepin.people.matcher import Reading

Confidence = Literal["sure", "tentative", "new", "unknown"]


@dataclass(frozen=True)
class Verdict:
    """Who spoke (a person's id, None for new and unknown), how sure, and why in words."""

    person: str | None
    confidence: Confidence
    how: str
    face: Reading | None = None
    voice: Reading | None = None


def _label(reading: Reading | None) -> str:
    return "none" if reading is None else reading.label


def fuse(face: Reading | None, voice: Reading | None) -> Verdict:
    """One verdict from the two modalities' (either may be None: no face, too short)."""
    f, v = _label(face), _label(voice)
    if f == "match" and v == "match":
        assert face is not None and voice is not None
        if face.person == voice.person:
            return Verdict(face.person, "sure", "face+voice", face, voice)
        if face.strong:
            return Verdict(face.person, "tentative", "face (the voice disagrees)", face, voice)
        return Verdict(voice.person, "tentative", "voice (the face disagrees)", face, voice)
    if f == "match":
        assert face is not None
        if v == "stranger":
            return Verdict(
                face.person, "tentative", "face (the voice is a stranger's)", face, voice
            )
        return Verdict(face.person, "sure" if face.sure else "tentative", "face", face, voice)
    if v == "match":
        assert voice is not None
        if f == "stranger":
            return Verdict(voice.person, "tentative", "voice (a stranger's face)", face, voice)
        return Verdict(voice.person, "sure" if voice.sure else "tentative", "voice", face, voice)
    if "stranger" in (f, v):
        how = "+".join(m for m, x in (("face", f), ("voice", v)) if x == "stranger")
        return Verdict(None, "new", how, face, voice)
    return Verdict(None, "unknown", "nothing to go on", face, voice)


def comparable(
    face: Reading | None, voice: Reading | None, known: Callable[[Reading, str], bool]
) -> tuple[Reading | None, Reading | None]:
    """The two readings, a stranger verdict turned ``unsure`` when its modality has no sample
    of the person the other modality matched (``known(reading, person)``: whether
    ``reading``'s modality has samples of ``person``)."""

    def fair(reading: Reading | None, other: Reading | None) -> Reading | None:
        if reading is None or other is None or reading.label != "stranger":
            return reading
        if other.label == "match" and other.person is not None and not known(reading, other.person):
            return replace(reading, label="unsure")
        return reading

    return fair(face, voice), fair(voice, face)
