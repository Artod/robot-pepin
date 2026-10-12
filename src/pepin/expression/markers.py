"""The marker grammar: how the brain's text names an intent, a face or a voice style anywhere in
a sentence, and what a sentence is once its markers are taken out.

A marker is one pair of braces with no brace inside::

    {laugh}               an intent (a composition of config/animation.json) by its bare name
    {gesture:laugh}       the same, the channel spelled out
    {face:smile}          one expression of config/face.json, on its own
    {style:whispering}    the voice's manner from here on, in free words as the voice takes them
    {}                    the plain voice again
    {whispering}          a bare name that is no intent is a style (the grammar before markers)

A marker is never spoken: the voice gets the text without it. Its place is the character offset
in its part's spoken text where it stood, so it belongs to the word that FOLLOWS it: that word's
start is the marker's sync point, where the intent's stroke lands. A style marker changes the
voice only at a clause boundary (the sentence's start, or after ``, ; : —``), where the sentence
is cut into two voice calls (:class:`Part`); anywhere else it takes effect at the next boundary.
A marker that names nothing the :class:`Vocabulary` knows is dropped (kept in
:attr:`ParsedSentence.dropped` for the log) and never reaches the voice.

The parser needs every marker whole: the sentence splitter must not end or cut a sentence inside
an open brace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol

Channel = Literal["gesture", "face", "style"]
CHANNELS: tuple[Channel, ...] = ("gesture", "face", "style")

MARKER = re.compile(r"\{([^{}]*)\}")  # one marker, no nesting
SPELLED = re.compile(r"^\s*(gesture|face|style)\s*:\s*(.*?)\s*$", re.IGNORECASE)
NAME = re.compile(r"^[a-z][a-z0-9_]{0,31}$")  # an intent's or a face's name
# The text before a style marker ends here for the marker to cut the sentence: nothing, or a
# clause's punctuation.
CLAUSE_END = re.compile(r"(?:^|[,;:\u2013\u2014-])\s*$")
# Seconds a spoken character takes while a part's audio is still being made (the same estimate
# as pepin.voice_cascade.player.Player.heard): a part's length before its last chunk is fed.
CHAR_S = 0.07


@dataclass(frozen=True)
class Vocabulary:
    """The names a marker may carry: the intents (config/animation.json's compositions) and the
    face's expressions (config/face.json). A bare name is an intent when it is one, else a
    style."""

    intents: frozenset[str]
    faces: frozenset[str]


@dataclass(frozen=True)
class Marker:
    """One marker: its channel, its name (a style's words; "" with ``style``: the plain voice),
    the character offset in its part's spoken text it stands before, and the text as written."""

    channel: Channel
    name: str
    at: int
    raw: str = ""


@dataclass(frozen=True)
class Part:
    """What one voice call says: the spoken text (markers removed), the style it is said in
    (None: the style in force goes on) and the markers inside it, in order. A sentence is one
    part unless a style marker at a clause boundary cuts it."""

    text: str
    style: str | None = None
    markers: tuple[Marker, ...] = ()


@dataclass(frozen=True)
class ParsedSentence:
    """A sentence of the brain's stream as the voice and the body take it: its parts in order,
    and the markers that named nothing (as written)."""

    parts: tuple[Part, ...]
    dropped: tuple[str, ...] = ()


@dataclass(frozen=True)
class Cue:
    """A marker placed on a reply's audio: in part ``index`` of reply ``turn`` (the player's mark
    key), ``share`` of the way through that part's audio (the marker's offset over the part's
    spoken length)."""

    marker: Marker
    turn: int
    index: int
    share: float


class Playout(Protocol):
    """The speaker's clock as a cue needs it (:class:`pepin.voice_cascade.player.Player`): the
    stream seconds played by now, and where a part's audio sits in that stream."""

    def played(self) -> float:
        """Stream seconds played by now."""
        ...

    def mark(self, turn: int, index: int) -> tuple[float, float] | None:
        """The first and the last stream second of part ``index`` of reply ``turn`` fed so far;
        None before its first chunk (or after the turn was forgotten)."""
        ...
