"""The word clock: when a character of a reply is heard, from the moment its part's text reaches
the engine to the moment its own audio exists (gesticulation DESIGN.md §7.4). One estimator for
every reader: the gesture scheduler, and what a barge-in says was heard
(:meth:`WordClock.heard_share`, read by :meth:`pepin.voice_cascade.player.Player.heard`; its
fallback :func:`part_length_s` and :data:`CHAR_S`).

A part of a reply (one voice call: a sentence, or a piece of one a style marker cut) passes three
moments, and the clock answers from the best one it has, never waiting for one:

* TEXT: the part went to the voice. Where it starts is predicted: a reply's first part
  ``tts_first_s`` after it went (the voice's first chunk at its p10, so a stroke errs early, the
  free side) and after the audio a queued filler still holds the reply back
  (``min(the filler's rest, what was sent ahead + its fade)``); a later part where the one before
  it ends.
* START: its first chunk is with the player, so its stream start is exact.
* AUDIO: the prosody listener's nuclei cover the word: the target is the most prominent nucleus
  in the word's span, its stress as the voice said it.

Inside a part a character sits after the voice's leading silence (``lead_silence_s``) at its
share of the rest of the part's length (MEASURED on Pepin's voice: 0.08 s typical error, 0.26 s
at p90 inside a part). Before the AUDIO moment the target is the word's first vowel, at or before
its stressed one. A part's length is its fed audio once all of it is fed, else at least
``sigma`` seconds a character: :data:`CHAR_S` until this conversation's spoken parts give a
running median. A stream second is a time on the engine's clock while the audio flows,
``now + (second - played())``; the speaker sounds ``audio_lag_s`` later still.

Each answer carries its ``spread``: how much EARLIER the word may turn out than said (a late
stroke is the costly side), from the measured errors of each moment.
"""

from __future__ import annotations

import statistics
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from pepin.expression.markers import Playout

# Seconds a spoken character takes before a conversation's own parts say otherwise (MEASURED
# 2026-10-10..12: 0.096 a character per reply, p10 0.071, p90 0.137; 0.089 on 2026-10-12)
CHAR_S = 0.09
SIGMA_PARTS = 20  # the running median's window: this many spoken parts
SIGMA_MIN_PARTS = 3  # fewer spoken parts than this: CHAR_S
SIGMA_MIN_CHARS = 8  # a shorter part is mostly its silences: not counted
# How much earlier a word may be heard than forecast (MEASURED, research_timing.md §4): inside a
# part from its leading silence's early side (p50 0.16 - p10 0.12) growing with the seconds into
# it up to the char-share law's p90; across a part whose length is not yet known, its p50
SPREAD_AUDIO_S = 0.03
SPREAD_PART_S = 0.04
SPREAD_PER_S = 0.2
SPREAD_INSIDE_S = 0.26
SPREAD_ACROSS_S = 0.3
SPAN_PAD_S = 0.1  # a word's span in the stream, widened by this for its nuclei
SNAP_S = 0.25  # a vowel with no nucleus in its word's span snaps to one this near

Stage = Literal["text", "start", "audio"]
STAGES: tuple[Stage, ...] = ("text", "start", "audio")


@dataclass(frozen=True)
class ClockKnobs:
    """config/expression.json's ``timing`` numbers the clock reads."""

    char_s: float = CHAR_S
    tts_first_s: float = 0.10
    lead_silence_s: float = 0.16
    audio_lag_s: float = 0.15


@dataclass
class Spoken:
    """A part of a reply as the clock has it: its spoken text, whether all its audio is with the
    player, the stream second its first chunk began at (None before it) and when its text went
    to the voice on the engine's clock (None until the clock is next told the time)."""

    text: str
    spoken: bool = False
    start_s: float | None = None
    queued_s: float | None = None


@dataclass(frozen=True)
class Forecast:
    """When a character is heard: on the engine's clock (``heard``, the speaker's sound), how much
    earlier it may turn out (``spread``), from which moment, and the player's stream second of it
    (None while the part has no place in the stream yet)."""

    heard: float
    spread: float
    stage: Stage
    stream: float | None = None


@dataclass(frozen=True)
class QueuedAudio:
    """What the player holds ahead of the speaker: seconds fed and not yet played, of them the
    seconds already sent to the speaker, and the fade a queued filler is taken over with (0: none,
    or one the reply waits for whole)."""

    fed_s: float
    sent_s: float
    fade_s: float


@runtime_checkable
class Queue(Protocol):
    """A player that tells what it holds ahead of the speaker (the pacer's queue)."""

    def queued_audio(self) -> QueuedAudio:
        """Now."""
        ...


class Nuclei(Protocol):
    """The prosody listener's product (:class:`pepin.expression.schedule.Nuclei`), as read here."""

    def nuclei(self, turn: int, index: int) -> Sequence[Nucleus]:
        """The part's nuclei found so far, in order."""
        ...

    def complete(self, turn: int, index: int) -> bool:
        """Whether the part's whole audio has been analysed."""
        ...


class Nucleus(Protocol):
    """A syllable's nucleus: its stream second and prominence."""

    @property
    def second(self) -> float:
        """The stream second."""
        ...

    @property
    def z(self) -> float:
        """The prominence (a z-sum over the voice's own statistics)."""
        ...


def part_length_s(
    mark: tuple[float, float], chars: int, spoken: bool, sigma: float = CHAR_S
) -> float:
    """How long a part lasts on the player's clock: its mark's length once all its audio is fed
    (``spoken``), else at least ``sigma`` a character of its text."""
    length = mark[1] - mark[0]
    return length if spoken else max(length, sigma * chars)


def filler_wait_s(queued: QueuedAudio) -> float:
    """How long a reply's first chunk waits behind what is queued: a filler taken over with a
    fade only for what was sent ahead and the ramp, anything else whole."""
    if queued.fade_s <= 0.0:
        return queued.fed_s
    return min(queued.fed_s, queued.sent_s + queued.fade_s)


class WordClock:
    """The clock over a conversation's parts (the module's notes); not thread-safe: the gesture
    engine's thread owns it."""

    def __init__(
        self, playout: Playout, knobs: ClockKnobs | None = None, nuclei: Nuclei | None = None
    ) -> None:
        """``playout``: the player (:class:`Queue` too when it tells its queue); ``nuclei``: the
        prosody listener (None: no AUDIO moment)."""
        self.playout = playout
        self.knobs = knobs or ClockKnobs()
        self.nuclei = nuclei
        self.parts: dict[tuple[int, int], Spoken] = {}
        self._rates: deque[float] = deque(maxlen=SIGMA_PARTS)
        self._median: float | None = None  # the rates' running median, once enough of them

    # -- what the conversation says ---------------------------------------------------------------

    def queued(self, turn: int, index: int, text: str, now: float | None = None) -> None:
        """Part ``index`` of reply ``turn`` went to the voice saying ``text`` (at ``now``; None:
        the next :meth:`stamp`)."""
        self.parts[(turn, index)] = Spoken(text, queued_s=now)

    def stamp(self, now: float) -> None:
        """Every part queued without a time went at ``now`` (the engine's clock reaches the
        scheduler with its next call, in the same step as the part)."""
        for part in self.parts.values():
            if part.queued_s is None:
                part.queued_s = now

    def started(self, turn: int, index: int, stream_s: float) -> None:
        """The part's first chunk is with the player at ``stream_s``."""
        part = self.parts.get((turn, index))
        if part is not None:
            part.start_s = stream_s

    def spoken(self, turn: int, index: int) -> None:
        """All the part's audio is with the player: its length is final, and its seconds a
        character join the running median."""
        part = self.parts.get((turn, index))
        if part is None:
            return
        part.spoken = True
        mark = self.playout.mark(turn, index)
        chars = len(part.text)
        if mark is not None and chars >= SIGMA_MIN_CHARS and mark[1] > mark[0]:
            self._rates.append((mark[1] - mark[0]) / chars)
            if len(self._rates) >= SIGMA_MIN_PARTS:
                self._median = statistics.median(self._rates)

    def forget(self, turn: int, *, only: bool = False) -> None:
        """The parts of replies up to ``turn`` (``only``: that one) are no longer placed (a cut
        reply's stream seconds are reused after the player's flush)."""
        for key in [k for k in self.parts if (k[0] == turn if only else k[0] <= turn)]:
            del self.parts[key]

    # -- the estimate ---------------------------------------------------------------------------

    def sigma(self) -> float:
        """Seconds a character takes: the running median of the spoken parts, ``char_s`` before
        there are enough of them (a number kept as each part is spoken: any thread may read it)."""
        median = self._median
        return self.knobs.char_s if median is None else median

    def length(self, turn: int, index: int) -> float:
        """The part's length on the player's clock, as known now (:func:`part_length_s`)."""
        part = self.parts.get((turn, index))
        if part is None:
            return 0.0
        mark = self.playout.mark(turn, index)
        if mark is None:
            return self.sigma() * len(part.text)
        return part_length_s(mark, len(part.text), part.spoken, self.sigma())

    def heard_share(self, turn: int, index: int, played: float) -> float | None:
        """The share of part ``index``'s characters heard once the stream has played up to
        ``played`` (what a barge-in says the person heard): the leading silence, then the rest
        at the part's length (:meth:`length`); None for a part the clock does not hold or one
        with no place in the stream. Any thread: it reads the player's mark and the part's
        record, never a collection the engine's thread changes while it iterates."""
        part = self.parts.get((turn, index))
        mark = self.playout.mark(turn, index)
        if part is None or mark is None:
            return None
        length = part_length_s(mark, len(part.text), part.spoken, self.sigma())
        lead = min(self.knobs.lead_silence_s, length)
        if length - lead <= 0.0:
            return 1.0 if played >= mark[0] + length else 0.0
        return min(1.0, max(0.0, (played - mark[0] - lead) / (length - lead)))

    def wall(self, stream_s: float, now: float) -> float:
        """A stream second on the engine's clock (while the audio flows)."""
        return now + (stream_s - self.playout.played())

    def start(self, turn: int, index: int, now: float) -> tuple[float, float | None, bool]:
        """Where the part starts: on the engine's clock, in the stream (None: not known yet), and
        whether that is exact (its own first chunk, or every part before it of the reply spoken
        and placed)."""
        mark = self.playout.mark(turn, index)
        part = self.parts.get((turn, index))
        stream = mark[0] if mark is not None else part.start_s if part is not None else None
        if stream is not None:
            return self.wall(stream, now), stream, True
        before = self.parts.get((turn, index - 1)) if index > 0 else None
        if before is not None:
            wall, stream, exact = self.start(turn, index - 1, now)
            length = self.length(turn, index - 1)
            exact = exact and before.spoken
            if stream is not None:
                return wall + length, stream + length, exact
            return max(wall + length, now), None, False
        queued = part.queued_s if part is not None and part.queued_s is not None else now
        wait = filler_wait_s(self.playout.queued_audio()) if isinstance(self.playout, Queue) else 0
        return max(queued + self.knobs.tts_first_s + wait, now), None, False

    def at(
        self,
        turn: int,
        index: int,
        char: int,
        now: float,
        word: tuple[int, int] | None = None,
    ) -> Forecast | None:
        """When character ``char`` of the part is heard (its vowel, at TEXT and START; with
        ``word``, its span, the stressed nucleus inside it at AUDIO); None for a part the clock
        does not hold."""
        part = self.parts.get((turn, index))
        if part is None:
            return None
        wall, stream, exact = self.start(turn, index, now)
        mark = self.playout.mark(turn, index)
        stage: Stage = "start" if mark is not None or part.start_s is not None else "text"
        into = self._into(part, turn, index, char)
        spread = min(SPREAD_INSIDE_S, SPREAD_PART_S + SPREAD_PER_S * into)
        if stage == "text" and index > 0 and not exact and self._before_unknown(turn, index):
            spread = SPREAD_ACROSS_S
        lag = self.knobs.audio_lag_s
        if stream is not None and word is not None and stage == "start":
            snapped = self._snap(part, turn, index, stream + into, word)
            if snapped is not None:
                return Forecast(self.wall(snapped, now) + lag, SPREAD_AUDIO_S, "audio", snapped)
        return Forecast(wall + into + lag, spread, stage, None if stream is None else stream + into)

    def end(self, turn: int, now: float) -> float | None:
        """When the reply's speech queued so far ends on the engine's clock (its last part's
        end; the speaker's sound), None for a reply with no part."""
        indices = [i for t, i in self.parts if t == turn]
        if not indices:
            return None
        last = max(indices)
        wall, _, _ = self.start(turn, last, now)
        return wall + self.length(turn, last) + self.knobs.audio_lag_s

    def _into(self, part: Spoken, turn: int, index: int, char: int) -> float:
        """Seconds from the part's start to character ``char``: the leading silence, then its
        share of the rest."""
        chars = max(1, len(part.text))
        lead = self.knobs.lead_silence_s
        share = min(1.0, max(0, char) / chars)
        return lead + share * max(0.0, self.length(turn, index) - lead)

    def _before_unknown(self, turn: int, index: int) -> bool:
        """Some part before ``index`` of the reply has no place in the stream yet, or is still
        being made: its length is a forecast."""
        for i in range(index):
            before = self.parts.get((turn, i))
            if before is not None and not before.spoken:
                return True
        return False

    def _snap(
        self, part: Spoken, turn: int, index: int, vowel_s: float, word: tuple[int, int]
    ) -> float | None:
        """The AUDIO moment: once the nuclei cover the word's span, its most prominent nucleus
        (the stress as said); with none inside, the nearest within :data:`SNAP_S` of the vowel;
        None before the word's audio is analysed."""
        if self.nuclei is None:
            return None
        found = self.nuclei.nuclei(turn, index)
        if not found:
            return None
        first = self._stream_of(turn, index, word[0]) - SPAN_PAD_S
        last = self._stream_of(turn, index, word[1]) + SPAN_PAD_S
        if not (self.nuclei.complete(turn, index) or found[-1].second >= last):
            return None
        inside = [n for n in found if first <= n.second <= last]
        if inside:
            return max(inside, key=lambda n: n.z).second
        near = min(found, key=lambda n: abs(n.second - vowel_s))
        return near.second if abs(near.second - vowel_s) <= SNAP_S else None

    def _stream_of(self, turn: int, index: int, char: int) -> float:
        """Character ``char`` of a part with a place in the stream, as a stream second (the
        clock's time does not enter a stream second)."""
        _, stream, _ = self.start(turn, index, 0.0)
        part = self.parts[(turn, index)]
        return (stream or 0.0) + self._into(part, turn, index, char)
