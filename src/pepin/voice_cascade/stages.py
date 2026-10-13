"""The cascade's stages as small protocols, and what flows between them.

Ears cut the mic into utterances (:class:`pepin.voice_cascade.ears.Ears`); a
:class:`Transcriber` reads one; a :class:`Tagger` describes how it was said; a
:class:`SpeakerId` says who said it; a :class:`Brain`
streams the reply as sentences and tool calls; a :class:`Voice` turns each sentence into PCM as
it is produced; the player (:class:`pepin.voice_cascade.player.Player`) paces it to the speaker.
Each has a fake in :mod:`pepin.voice_cascade.fake`, so the whole loop runs in a test.
"""

from __future__ import annotations

import math
from collections.abc import Awaitable, Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from pepin.tools.registry import Result


class Transcriber(Protocol):
    """Speech to text on the Mac (:class:`pepin.voice_live.wake.MlxWhisper` is one)."""

    def transcribe(self, pcm: bytes) -> str:
        """The words in 16 kHz s16le mono ``pcm``."""
        ...


@dataclass(frozen=True)
class Unit:
    """What a speech-to-text model emitted inside a word, a letter (a character model) or a
    token (a BPE model), with its span in seconds from the start of the PCM."""

    text: str
    start: float
    end: float

    @property
    def seconds(self) -> float:
        """How long it lasted."""
        return self.end - self.start


@dataclass(frozen=True)
class Vowel:
    """A vowel heard inside a word: the span of its voiced, loud sound in seconds, and which of
    the word's units is its letter (None: not known)."""

    start: float
    end: float
    unit: int | None = None

    @property
    def seconds(self) -> float:
        """How long it lasted."""
        return self.end - self.start


@dataclass(frozen=True)
class TimedWord:
    """One word as a model heard it: its letters only (no punctuation), its span, its units
    (``letters``: one per letter), its log-probability (None: the model gives none), and its
    vowels as the sound has them (:mod:`pepin.voice_cascade.timing`)."""

    text: str
    start: float
    end: float
    units: tuple[Unit, ...]
    letters: bool
    logprob: float | None = None
    vowels: tuple[Vowel, ...] = ()


@dataclass(frozen=True)
class Transcript:
    """An utterance's words: ``text`` as the transcriber writes it, ``words`` with their timings
    (empty when the engine gives none or timings are off; they may be another model's reading,
    :class:`pepin.voice_cascade.stt.gigaam.GigaAM`)."""

    text: str
    words: tuple[TimedWord, ...] = ()


@runtime_checkable
class TimedTranscriber(Protocol):
    """A transcriber that can also say when each word, and each letter or token, was said."""

    def transcribe_timed(self, pcm: bytes) -> Transcript:
        """The words in ``pcm`` (the text :meth:`Transcriber.transcribe` gives) with timings."""
        ...


@dataclass(frozen=True)
class Prosody:
    """How one utterance sounded: plain acoustic numbers over its voiced span (None where the
    pitch tracker found nothing), a neural model's reading when one is configured, and the
    words' timings when the transcriber gives them (joined after both are read)."""

    active_s: float
    loudness_db: float
    f0_median_st: float | None = None  # semitones re 100 Hz
    f0_range_st: float | None = None  # 5-95 % range, semitones
    pause_ratio: float = 0.0
    arousal: float | None = None  # a dimensional model's 0..1
    label: str | None = None  # a categorical model's top class, with its score
    words: tuple[TimedWord, ...] = ()


class Tagger(Protocol):
    """The tag line in front of the user's words (prosody.md section 7): measured while Whisper
    reads (:meth:`measure`), written once the words are known (:meth:`line`)."""

    def measure(self, pcm: bytes) -> Prosody:
        """The numbers of one utterance (16 kHz s16le mono)."""
        ...

    def line(self, prosody: Prosody, text: str, keep: bool = True) -> str:
        """``[Артём: тише и медленнее обычного]`` against the running baseline, or "" when
        there is nothing to say yet; the utterance joins the baseline unless ``keep`` is False
        (a line written before the utterance closed)."""
        ...

    def describe(self, prosody: Prosody, text: str, key: str, keep: bool = True) -> str:
        """The words alone (``тише и медленнее обычного``) against ``key``'s own baseline (a
        recognised person's), "" while there is nothing to say; the utterance joins it unless
        ``keep`` is False."""
        ...


@dataclass(frozen=True)
class Speaker:
    """Who said an utterance, as person recognition names them: ``head`` is the tag line's
    speaker part (``говорит: Саша``, "" for nobody), ``key`` keys their prosody baseline
    (None: nobody known, so no prosody words), ``log`` what the conversation log keeps."""

    head: str
    key: str | None = None
    log: dict[str, Any] | None = None

    def line(self, body: str) -> str:
        """The tag line: ``[говорит: Саша; тише обычного]``, a part left out when empty, ""
        when both are."""
        parts = [part for part in (self.head, body) if part]
        return f"[{'; '.join(parts)}]" if parts else ""


class SpeakerId(Protocol):
    """Person recognition (:mod:`pepin.people`): measured while Whisper reads, named once the
    words are known, behind its live switch."""

    @property
    def on(self) -> bool:
        """Whether recognition runs (its switch)."""
        ...

    def measure(self, pcm: bytes, end_s: float) -> Callable[[str], Speaker]:
        """The voice and the face of one utterance (on a worker thread; ``end_s`` the laptop's
        monotonic end of the speech); returns the naming step, called on the event loop with
        the words, never for Whisper's noise line."""
        ...


class Voice(Protocol):
    """Text to speech, one sentence at a time, PCM as it is produced."""

    rate: int  # Hz of the PCM it yields

    def load(self) -> None:
        """Load and warm up (a second call does nothing)."""
        ...

    def stream(self, text: str, style: str = "") -> Iterator[bytes]:
        """``text`` spoken as s16le mono chunks at :attr:`rate`, each yielded when ready;
        ``style`` ("шёпотом", "грустно") says how, and a voice that cannot do it ignores it."""
        ...


@dataclass(frozen=True)
class VoiceBill:
    """One sentence of a voice that pays per call: who said it, how soon, at what price."""

    engine: str  # the model that spoke (the fallback's name when it stood in)
    first_audio_ms: int | None = None  # the call's start to its first audio
    usd: float = 0.0
    text_tokens: int = 0
    audio_tokens: int = 0
    calls: int = 0  # cloud calls made (0: served from the cache or by the fallback alone)
    fallback: str | None = None  # why the local voice spoke instead (None: it did not)


@runtime_checkable
class Billed(Protocol):
    """A voice that pays per sentence (a cloud voice): no clips are made with it at load (each
    would be a call), and each sentence leaves a bill."""

    def bill(self) -> VoiceBill | None:
        """The bill of the sentence last streamed to its end (or cut), taken once."""
        ...


@dataclass(frozen=True)
class FirstToken:
    """The brain's first text of the turn arrived."""


@dataclass(frozen=True)
class Sentence:
    """One sentence to say, from round ``round`` of the turn (a round is one model call)."""

    text: str
    round: int


@dataclass(frozen=True)
class Ahead:
    """A sentence of round ``round`` before it is whole, from the token stream (a marker just
    closed, the reply's first word): its text so far, markers as written; the sentence itself
    still comes as a :class:`Sentence`."""

    text: str
    round: int


@dataclass(frozen=True)
class ToolUse:
    """The brain calls a tool."""

    id: str
    name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class ToolDone:
    """A tool's result went back to the brain."""

    id: str
    name: str
    result: str
    ms: int


@dataclass(frozen=True)
class Usage:
    """One model call's tokens and its price estimate."""

    input: int = 0
    cache_read: int = 0
    cache_write: int = 0
    output: int = 0
    usd: float = 0.0
    stop: str | None = None
    ms: int = 0  # the call's wall time

    def plus(self, other: Usage) -> Usage:
        """Two calls summed (the later stop reason kept)."""
        return Usage(
            self.input + other.input,
            self.cache_read + other.cache_read,
            self.cache_write + other.cache_write,
            self.output + other.output,
            self.usd + other.usd,
            other.stop or self.stop,
            self.ms + other.ms,
        )


CUT = "cancelled"  # the stop of a call cut mid-stream: its usage is an estimate
CHARS_PER_TOKEN = 3.0  # streamed Russian text per output token, for a call cut mid-stream


def text_tokens(chars: int) -> int:
    """The output tokens ``chars`` characters of streamed text cost, estimated: a call cut
    mid-stream never reports its own (its thinking is not seen, so not counted)."""
    return math.ceil(chars / CHARS_PER_TOKEN)


BrainEvent = FirstToken | Sentence | Ahead | ToolUse | ToolDone | Usage
FOLLOW_UP = "follow_up"  # a tool result's key: False ends the turn without another model call
ToolExecutor = Callable[[str, dict[str, Any]], Awaitable[Result]]
Emit = Callable[[BrainEvent], None]


@dataclass
class Played:
    """What of a reply the person heard before cutting in: whole sentences, then the start of
    the one being said."""

    sentences: int = 0
    partial: str = ""


class Brain(Protocol):
    """The talker: a conversation's history and one turn at a time.

    A turn ends in exactly one of :meth:`commit` (it ran to its end), :meth:`interrupt` (the
    person cut in: the history keeps what was heard) or :meth:`abandon` (dropped before it did
    or said anything: the person was still talking).
    """

    def begin(self, places: list[str] | None) -> None:
        """A new conversation; ``places``: the names go_to takes (None: unknown)."""
        ...

    async def turn(self, text: str, tools: ToolExecutor, emit: Emit) -> None:
        """Answer ``text``: sentences, tool calls and usage reported through ``emit`` as they
        happen; tools run through ``tools``. Cancellation leaves the turn open for
        :meth:`interrupt` or :meth:`abandon`; a call it cuts mid-stream still reports its
        usage (stop :data:`CUT`, an estimate), so the ledger sees what was spent."""
        ...

    def commit(self) -> None:
        """The turn ran to its end: keep it."""
        ...

    def interrupt(self, played: Played) -> None:
        """The person cut in: keep what they heard of the turn, and the tools that ran."""
        ...

    def abandon(self) -> None:
        """Drop the turn (it ran no tool and said nothing)."""
        ...

    @property
    def acted(self) -> bool:
        """The open turn has run a tool."""
        ...

    async def prewarm(self) -> Usage | None:
        """Write the prompt cache ahead of the first turn (None when off or unsupported)."""
        ...
