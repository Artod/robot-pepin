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
# The next clause boundary after a style marker that stood elsewhere: its punctuation and the
# space after it (a hyphen inside a word is none).
CLAUSE_BOUNDARY = re.compile(r"[,;:\u2013\u2014-]\s+")
# Seconds a spoken character takes while a part's audio is still being made (the same estimate
# as pepin.voice_cascade.player.Player.heard): a part's length before its last chunk is fed.
CHAR_S = 0.07
_SPACES = re.compile(r"\s+")
_PUNCTUATION = frozenset(".,!?;:…»)]")  # the end of a sentence or a clause


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


EMPTY = Vocabulary(frozenset(), frozenset())  # no intents, no faces: every bare name a style


def classify(inner: str, vocabulary: Vocabulary) -> tuple[Channel, str] | None:
    """What a marker's inside names: ``(channel, name)``, a style's words with their spaces
    collapsed (``""``: the plain voice); None for a gesture or a face the vocabulary lacks."""
    text = inner.strip()
    if not text:
        return "style", ""
    spelled = SPELLED.match(text)
    if spelled is not None:
        channel, name = spelled.group(1).lower(), spelled.group(2)
        if channel == "gesture":
            return ("gesture", name.lower()) if name.lower() in vocabulary.intents else None
        if channel == "face":
            return ("face", name.lower()) if name.lower() in vocabulary.faces else None
        return "style", _SPACES.sub(" ", name).strip()
    if text.lower() in vocabulary.intents:
        return "gesture", text.lower()
    return "style", _SPACES.sub(" ", text)


def parse(sentence: str, vocabulary: Vocabulary = EMPTY) -> ParsedSentence:
    """A sentence of the brain's stream as parts for the voice, its markers placed on them.

    The markers are taken out (the text around each joined with one space) and each is placed
    at the offset of the word after it. A style marker at a clause boundary (the sentence's
    start, or after a comma, a semicolon, a colon or a dash) starts a new part in that style;
    one anywhere else waits for the next boundary after it, or for the next sentence: then the
    sentence ends with an empty part that only carries the style. A part's ``style`` None keeps
    the one in force.
    """
    text, pos = "", 0
    placed: list[Marker] = []
    dropped: list[str] = []
    for match in MARKER.finditer(sentence):
        text = _joined(text, sentence[pos : match.start()])
        pos = match.end()
        named = classify(match.group(1), vocabulary)
        if named is None:
            dropped.append(match.group(0))
        else:
            placed.append(Marker(named[0], named[1], len(text), match.group(0)))
    text = _joined(text, sentence[pos:])
    cuts: dict[int, str] = {}  # where a part starts -> the style it is said in
    for marker in placed:
        if marker.channel != "style":
            continue
        if CLAUSE_END.search(text[: marker.at]):
            cuts[marker.at] = marker.name
            continue
        boundary = CLAUSE_BOUNDARY.search(text, marker.at)
        cuts[boundary.end() if boundary is not None else len(text)] = marker.name
    # a style for the next sentence: an empty part after the last one (a marker at the very
    # end belongs to the last spoken part, its share 1)
    after = cuts.pop(len(text)) if text and len(text) in cuts else None
    starts = sorted({0, *cuts})
    parts: list[Part] = []
    for begin, end in zip(starts, [*starts[1:], len(text) + 1], strict=True):
        body = text[begin:end]
        lead = len(body) - len(body.lstrip())
        spoken = body.strip()
        inside = tuple(
            Marker(m.channel, m.name, min(max(0, m.at - begin - lead), len(spoken)), m.raw)
            for m in placed
            if m.channel != "style" and begin <= m.at < end
        )
        if spoken or inside or begin in cuts:
            parts.append(Part(spoken, cuts.get(begin), inside))
    if after is not None:
        parts.append(Part("", after))
    return ParsedSentence(tuple(parts), tuple(dropped))


def _joined(text: str, segment: str) -> str:
    """``text`` and the ``segment`` after it, one space where a marker stood between them and
    none before the punctuation that followed it (``Слово {x}.`` is ``Слово.``)."""
    if not text or text[-1].isspace():
        rest = segment.lstrip()
        if rest[:1] in _PUNCTUATION:
            return text.rstrip() + rest
        return text + rest
    return text + segment


def part_length_s(mark: tuple[float, float], chars: int, spoken: bool, char_s: float) -> float:
    """How long a part lasts on the player's clock: its mark's length once all its audio is
    fed (``spoken``), else at least ``char_s`` a character of its text."""
    length = mark[1] - mark[0]
    return length if spoken else max(length, char_s * chars)


def sync_s(cue: Cue, mark: tuple[float, float], chars: int, spoken: bool, char_s: float) -> float:
    """The stream second a cue's word starts: its part's start plus its share of the part's
    length (:func:`part_length_s`), the timing law of Player.heard."""
    return mark[0] + cue.share * part_length_s(mark, chars, spoken, char_s)


def cues(turn: int, index: int, part: Part) -> list[Cue]:
    """The part's gesture and face markers as cues on its audio, in order."""
    size = max(1, len(part.text))
    return [Cue(m, turn, index, min(1.0, m.at / size)) for m in part.markers]
