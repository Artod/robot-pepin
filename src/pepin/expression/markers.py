"""The marker grammar: how the brain's text names an intent, a face or a voice style anywhere in
a sentence, and what a sentence is once its markers are taken out.

A marker is one pair of braces with no brace inside::

    {laugh}               an intent (a composition of config/animation.json) by its bare name
    {size:big}            an intent with its argument: one the composition takes (its ``args``),
                          or any short name where it takes ``*`` ({point:printer}, {point:plan
                          b}: 1-24 characters in any script, Cyrillic too, read lower-case with
                          single spaces and the Cyrillic yo read as ye)
    {!}                   another spelling of an intent: the book's ``aliases`` map it (and
                          ``{beat}``, ``{point:i}``, ``{arc:rise}``) to the canonical name and
                          argument (``accent``, ``point:me``, ``arc:up``)
    {snap:3,fast}         a list after the colon (2026-10-14): at most one argument that chooses
                          (as above), words and numbers that set the intent's parameters (the
                          book's ``settings``: ``fast``, a number; an atom's auto-wrapper: one
                          number is its repeats, several its parameters in their order) and
                          ``key=value`` for a parameter by name ({head_nod:amp_deg=6};
                          ``s=`` is a step's ``hold_s``, the seconds a look holds: {glow:red,s=2})
    {gesture:laugh}       the same, the channel spelled out
    {face:smile}          one expression of config/face.json, on its own
    {style:rose}          the mouth drawn in a style of config/face.json: the book's ``style``
                          intent when it takes the argument ({style:rose,s=0}, {style:rose,hold}
                          from now on; :func:`pepin.expression.book.with_styles`); {style} alone
                          the standing style again (a ``bare`` intent: its first argument)
    {style:whispering}    any other: the voice's manner from here on, in free words as the voice
                          takes them
    {}                    the plain voice again
    {whispering}          a bare name that is no intent is a style when it is made of the
                          persona's style words (:data:`STYLE_WORDS`: ``{happy, excited}``)

Everything else is dropped, never a voice style: a bare name of no intent (``{nod}``), an
``{x:y}`` whose ``x`` names no intent, an intent that needs an argument written without one (or
with none it takes). An argument the intent does not take is left out and kept in
:attr:`Marker.ignored` for the log; the rest of the mark plays. A marker is never spoken: the
voice gets the text without it.
Its place is the character offset in its part's spoken text where it stood, so it belongs to the
word that FOLLOWS it: that word's start is the marker's sync point, where the intent's stroke
lands. A style marker changes the voice only at a clause boundary (the sentence's start, or after
``, ; : —``), where the sentence is cut into two voice calls (:class:`Part`); anywhere else it
takes effect at the next boundary. A dropped marker is kept in :attr:`ParsedSentence.dropped` as
written, for the log; a kept one carries its canonical name and argument (:class:`Marker`) and
its spelling as written (``raw``).

The parser needs every marker whole: the sentence splitter must not end or cut a sentence inside
an open brace. When a cue's word is heard is the word clock's (:mod:`pepin.expression.forecast`,
its one estimate of a character's seconds). Owners after slice 0 (gesticulation DESIGN.md §11):
C owns the grammar here (``{name:arg}``, aliases, ``{!}``); :class:`Marker`, :class:`Cue` and
:class:`Playout` are the contract.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
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
# The persona's words for the voice's manner: a bare marker of these alone (in any number,
# commas or spaces between) is a style; config/voice_cascade_persona.md lists the same words.
STYLE_WORDS = frozenset(
    ("whispering", "quiet", "sad", "tired", "happy", "excited", "calm", "slow", "warm")
)
_STYLE_SPLIT = re.compile(r"[\s,]+")
# A parameter's short spelling in a mark's key=value: s, the seconds a step holds (its hold_s,
# pepin.expression.animation.HOLD)
KEYS: Mapping[str, str] = {"s": "hold_s"}
ANY = "*"  # a composition's argument that takes any short name (a place, a referent)
ARG = re.compile(r"^[^\W_][\w .'\-]{0,23}$")  # such a name: a letter or a digit first, 1-24 long
_VOWELS = re.compile("[аеёиоуыэюяaeiouy]", re.IGNORECASE)  # noqa: RUF001 -- Cyrillic vowels
_WORD = re.compile(r"\w+")
_SPACES = re.compile(r"\s+")
_PUNCTUATION = frozenset(".,!?;:…»)]")  # the end of a sentence or a clause


@dataclass(frozen=True)
class Takes:
    """What a mark of an intent may carry beside the argument that chooses (``Vocabulary.args``):
    words that set its parameters (``fast``), how many numbers it reads in order (an
    auto-wrapper: its atom's parameters, one number alone being its repeats; else 1 when the
    book gives it a ``number``) and the parameters ``key=value`` may set."""

    words: frozenset[str] = frozenset()
    numbers: int = 0
    keys: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Vocabulary:
    """The names a marker may carry: the intents (config/animation.json's compositions), the
    face's expressions (config/face.json), the arguments each intent takes that choose
    (``args``: an intent absent from it takes none; ``"*"`` any short name), those that set its
    parameters (``takes``), the intents' other spellings (``aliases``: ``"beat"`` ->
    ``"accent"``, ``"point:i"`` -> ``"point:me"``) and those a bare marker may name though
    they take arguments that choose (``bare``: the renderer plays their first). A bare name is
    an intent when it is one, else a style when it is made of :data:`STYLE_WORDS`."""

    intents: frozenset[str]
    faces: frozenset[str]
    args: Mapping[str, tuple[str, ...]] = field(default_factory=dict, hash=False)
    aliases: Mapping[str, str] = field(default_factory=dict, hash=False)
    takes: Mapping[str, Takes] = field(default_factory=dict, hash=False)
    bare: frozenset[str] = frozenset()


@dataclass(frozen=True)
class Marker:
    """One marker: its channel, its name (a style's words; "" with ``style``: the plain voice),
    the character offset in its part's spoken text it stands before, the text as written, its
    argument that chooses (``{point:you}``: ``"you"``; None: none), the arguments that set
    parameters as read (``{snap:3,fast}``: ``("3", "fast")``; ``key=value`` as written) and
    those it carried that the intent does not take (left out, for the log)."""

    channel: Channel
    name: str
    at: int
    raw: str = ""
    arg: str | None = None
    args: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()


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
    spoken length); ``word`` is the anchor word's character span in the part's spoken text
    (start, end) and ``vowel`` the offset of its first vowel (its end when it has none)."""

    marker: Marker
    turn: int
    index: int
    share: float
    word: tuple[int, int] = (0, 0)
    vowel: int = 0


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
Named = tuple[Channel, str, str | None]  # a marker's channel, canonical name and argument


@dataclass(frozen=True)
class Mark:
    """A marker's inside as read: :data:`Named`'s three, then the arguments that set
    parameters and the ones left out (:class:`Marker`)."""

    channel: Channel
    name: str
    arg: str | None = None
    args: tuple[str, ...] = ()
    ignored: tuple[str, ...] = ()

    @property
    def named(self) -> Named:
        """The channel, the name and the argument that chooses."""
        return self.channel, self.name, self.arg


def read(inner: str, vocabulary: Vocabulary) -> Named | None:
    """What a marker's inside names: ``(channel, name, arg)``, a gesture's canonical name and
    argument through the aliases, a style's words with their spaces collapsed (``""``: the plain
    voice); None for what names nothing (the module's notes): never a style for an ``x:y``.
    The arguments that set parameters: :func:`read_mark`."""
    found = read_mark(inner, vocabulary)
    return None if found is None else found.named


def read_mark(inner: str, vocabulary: Vocabulary) -> Mark | None:
    """:func:`read` with the arguments that set parameters and those left out."""
    text = inner.strip()
    if not text:
        return Mark("style", "")
    spelled = SPELLED.match(text)
    if spelled is not None:
        channel, name = spelled.group(1).lower(), spelled.group(2)
        if channel == "gesture":
            return mark(name, vocabulary)
        if channel == "face":
            return Mark("face", name.lower()) if name.lower() in vocabulary.faces else None
        drawn = mark(text, vocabulary)  # {style:rose}: the face's style, when the book takes it
        if drawn is not None:
            return drawn
        return Mark("style", _SPACES.sub(" ", name).strip())
    found = mark(text, vocabulary)
    if found is not None or ":" in text:
        return found
    if set(_STYLE_SPLIT.split(text.lower())) - {""} <= STYLE_WORDS:
        return Mark("style", _SPACES.sub(" ", text))
    return None


def mark(text: str, vocabulary: Vocabulary) -> Mark | None:
    """``name``, ``name:arg`` or ``name:a,b,key=value`` as an intent and its arguments: one
    argument whole as :func:`gesture` reads it; else each of the list: an argument that
    chooses (through the aliases, the intent's own, any short name for ``*``; the first such),
    a word or a number that sets parameters (:class:`Takes`; an argument the intent lists
    chooses before a number does: ``{count:2}``), ``key=value`` with a number for a parameter
    it takes; anything else left out. None when no intent is named or one that needs an
    argument that chooses got none: a ``bare`` one plays its first when only numbers and
    ``key=value`` were left out (``{nod:5}`` is ``{nod}``, the 5 left out), never past a word
    (``{style:whispering}`` is the voice's manner)."""
    whole = gesture(text, vocabulary)
    if whole is not None:
        return Mark(*whole)
    name, colon, rest = text.partition(":")
    if not colon:
        return None
    name = name.strip().lower()
    given: str | None = None
    if name in vocabulary.aliases:
        name, _, alias_arg = vocabulary.aliases[name].partition(":")
        given = alias_arg or None
    if name not in vocabulary.intents:
        return None
    choose = vocabulary.args.get(name, ())
    takes = vocabulary.takes.get(name, Takes())
    chosen, args, ignored = given, [], []
    numbers = 0
    for raw in rest.split(","):
        item = argument(raw)
        if item is None:
            if raw.strip():
                ignored.append(raw.strip())
            continue
        key, equals, value = (x.strip() for x in item.partition("="))
        if equals:
            key = KEYS.get(key, key)
            if key in takes.keys and number_of(value) is not None:
                args.append(f"{key}={value}")
            else:
                ignored.append(item)
            continue
        alias = vocabulary.aliases.get(f"{name}:{item}", "")
        target, _, alias_arg = alias.partition(":")
        if chosen is None and target == name and alias_arg:
            chosen = alias_arg
        elif chosen is None and item in choose and item != ANY:
            chosen = item
        elif item in takes.words:
            args.append(item)
        elif number_of(item) is not None and numbers < takes.numbers:
            args.append(item)
            numbers += 1
        elif chosen is None and ANY in choose and ARG.match(item) is not None:
            chosen = item
        else:
            ignored.append(item)
    if chosen is None and choose:
        words = [i for i in ignored if "=" not in i and number_of(i) is None]
        if name not in vocabulary.bare or words:
            return None
    return Mark("gesture", name, chosen, tuple(args), tuple(ignored))


def gesture(text: str, vocabulary: Vocabulary) -> Named | None:
    """``name`` or ``name:arg`` as an intent and its argument: the whole spelling through the
    aliases first, then the name alone (its argument kept); None unless the intent takes that
    argument, or, written bare, needs none (a list of arguments: :func:`mark`)."""
    name, colon, rest = text.partition(":")
    name = name.strip().lower()
    arg = argument(rest) if colon else None
    if colon and arg is None:
        return None
    aliases = vocabulary.aliases
    whole = aliases.get(f"{name}:{arg}") if arg is not None else None
    if whole is not None:
        name, _, given = whole.partition(":")
        arg = given or None
    elif name in aliases:
        canonical, _, given = aliases[name].partition(":")
        if given and arg is not None:
            return None  # an alias that brings its own argument takes no other
        name, arg = canonical, arg if arg is not None else (given or None)
    if name not in vocabulary.intents:
        return None
    takes = vocabulary.args.get(name, ())
    if arg is None:
        return None if takes and name not in vocabulary.bare else ("gesture", name, None)
    if arg in takes or (ANY in takes and ARG.match(arg) is not None):
        return "gesture", name, arg
    return None


def number_of(text: str) -> float | None:
    """A mark's argument as a number (``3``, ``0.4``, ``-5``), None when it is none or not
    finite."""
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def argument(raw: str) -> str | None:
    """A marker's argument as the book and the gesture space read it: lower case, single
    spaces, the Cyrillic yo as ye; None when empty or holding a colon or a brace."""
    arg = _SPACES.sub(" ", raw.strip().lower()).replace("ё", "е")  # noqa: RUF001
    return arg if arg and not set(arg) & set(":{}") else None


def classify(inner: str, vocabulary: Vocabulary) -> tuple[Channel, str] | None:
    """What a marker's inside names, its argument aside (:func:`read`)."""
    named = read(inner, vocabulary)
    return None if named is None else (named[0], named[1])


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
        found = read_mark(match.group(1), vocabulary)
        if found is None:
            dropped.append(match.group(0))
        else:
            placed.append(Marker(found.channel, found.name, len(text), match.group(0),
                                 found.arg, found.args, found.ignored))  # fmt: skip
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
            replace(m, at=min(max(0, m.at - begin - lead), len(spoken)))
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


def cues(turn: int, index: int, part: Part) -> list[Cue]:
    """The part's gesture and face markers as cues on its audio, in order, each with its anchor
    word's span and first vowel."""
    size = max(1, len(part.text))
    return [
        Cue(m, turn, index, min(1.0, m.at / size), *anchor(part.text, m.at)) for m in part.markers
    ]


def anchor(text: str, at: int) -> tuple[tuple[int, int], int]:
    """The word a marker at ``at`` belongs to (the first at or after it): its span in ``text``
    and the offset of its first vowel; ``((len, len), len)`` when no word follows."""
    word = _WORD.search(text, at)
    if word is None:
        return (len(text), len(text)), len(text)
    vowel = _VOWELS.search(text, word.start(), word.end())
    return (word.start(), word.end()), vowel.start() if vowel is not None else word.end()
