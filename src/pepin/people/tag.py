"""The speaker in the tag line the brain reads in front of the person's words.

``[говорит: Саша]`` (sure), ``[говорит: похоже, Саша]`` (tentative: the brain may ask "Саша, это
ты?"), ``[говорит: кто-то новый]`` (a person nobody has met yet: the brain may ask the name), and
nothing for an utterance that names nobody (a short "да" with no face in view). The prosody
words, measured against that person's own baseline, follow after a semicolon:
``[говорит: Саша; тише обычного]``.
"""

from __future__ import annotations

from pepin.people.fusion import Verdict

SPEAKS = "говорит"
LIKELY = "похоже"
NEW = "кто-то новый"


def head(verdict: Verdict, name: str | None) -> str:
    """The speaker part of the tag line (no brackets), "" for nobody."""
    if verdict.confidence == "new":
        return f"{SPEAKS}: {NEW}"
    if name is None or verdict.confidence == "unknown":
        return ""
    if verdict.confidence == "tentative":
        return f"{SPEAKS}: {LIKELY}, {name}"
    return f"{SPEAKS}: {name}"
