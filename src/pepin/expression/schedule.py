"""The gesture engine's scheduler: requests from both trigger sources (the brain's markers; the
engine's own listeners and moments; a person's ask), each anchored on a word, a stream second or
the clock, placed on the body's parts by one priority table (gesticulation DESIGN.md §6.5).

A :class:`Request` says what (an intent of the book, or clips already made: the melody), where
(an :data:`Anchor`: a character of a reply's part, a second of the player's stream, a time on
the engine's clock, or now), how much it matters (:class:`Priority`, and a rank inside it), who
asked, how tightly it binds to its word and the family a marker of the same kind absorbs it
into. The engine (:class:`pepin.expression.animator.GestureEngine`) renders it and checks the
caps; :class:`Scheduler` keeps each clip until its time and says which go now;
:class:`Reservations` says which request holds each part.

The core here keeps the engine's timing as it was: a clip anchored on a word waits for the
part's audio and goes when the stream has played up to

    the word's start + the clip's offset from it + audio_lag_s - the part's latency_s

(:func:`pepin.expression.markers.sync_s`; a stream second in place of the word likewise), at once
when that is past by at most ``late_s``, dropped later; a clip anchored on the clock goes at its
time less its part's latency; one asked for now goes with the performance's earliest clip at
once. Per part one reservation at a time: state, attention and a person's ask take a part
whenever they come; any other replaces the one holding it only from a higher or equal place in
the table and only before that one's stroke; a lower one waits for a free part (it is refused
while one holds); the idle layer gives way to everything and takes only a free part.

Owners (gesticulation slices, DESIGN.md §11): B owns this file's internals (the word clock in
place of ``sync_s``, render at TEXT and send at the last moment, the ladder, arm units, free
windows around strokes); A posts requests (:meth:`pepin.expression.animator.GestureEngine.
request`) and reads the holds (:meth:`Reservations.last`); the names and signatures the engine
calls are the contract and stay.
"""

from __future__ import annotations

import enum
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from pepin.expression.animation import Bind, Clip, PartName, Performance
from pepin.expression.markers import Channel, Cue, Marker, Playout, anchor, sync_s

Record = Callable[[str, dict[str, Any]], None]
NOT_YET = math.inf  # the end of a reservation that lasts until let go (a looping idle clip)


class Priority(enum.IntEnum):
    """Who wins a part (DESIGN.md §6.5), the first above all."""

    STATE = 0  # a drive, the person starting to speak, the close, a barge-in
    ATTENTION = 1  # the speaker gaze's turn, seek and check (never the engine's own requests)
    ASK = 2  # a person's ask: the gesture tool, scripts/gesture.py
    MARK = 3  # the brain's markers
    MOMENT = 4  # the robot's moments (pepin.expression.conversation.MOMENTS orders them)
    PULSE = 5  # the voice's accents
    IDLE = 6  # the melody, the breath, the drift


@dataclass(frozen=True)
class AtWord:
    """A character of a reply's part: the word that follows it (a marker's place)."""

    turn: int
    part: int
    char: int


@dataclass(frozen=True)
class AtStream:
    """A second of the player's stream (a syllable's nucleus), of reply ``turn`` when known: it
    is dropped with that reply (a flush reuses the stream's seconds)."""

    second: float
    turn: int | None = None


@dataclass(frozen=True)
class AtTime:
    """A time on the engine's clock (laptop monotonic): a moment's sync point."""

    t: float


Anchor = AtWord | AtStream | AtTime


@dataclass(frozen=True)
class Request:
    """Something to perform: ``intent`` (a composition of the book, or the name its ready
    ``clips`` are logged under), its ``anchor`` (None: now), its place in the table
    (``priority``, then ``rank``: lower first), who asked (``source``: "marker", "moment",
    "tool", "pulse", "idle", ...), how its stroke binds to its word and the family a marker of
    the same kind absorbs it into."""

    intent: str
    anchor: Anchor | None = None
    priority: Priority = Priority.ASK
    source: str = "tool"
    bind: Bind = "utterance"
    family: str | None = None
    rank: int = 0
    clips: tuple[Clip, ...] = ()


@dataclass
class Reservation:
    """What holds a part: the performance and its clip, sent at ``started`` (engine clock), its
    stroke and its end on the same clock, and the request's place in the table."""

    performance: Performance
    clip: Clip
    started: float
    stroke: float
    ends: float
    priority: Priority = Priority.ASK
    rank: int = 0
    source: str = "tool"


class Reservations:
    """Per part one reservation at a time, by the priority table (the module's notes)."""

    def __init__(self) -> None:
        """Every part free."""
        self.held: dict[PartName, Reservation] = {}
        self._last: dict[PartName, Reservation] = {}

    def refusal(self, part: PartName, priority: Priority, rank: int, now: float) -> str | None:
        """Why a request at ``priority`` (and ``rank``) may not take ``part`` now; None: it may."""
        held = self.held.get(part)
        if held is None or now >= held.ends or priority <= Priority.ASK:
            return None
        if held.priority == Priority.IDLE:
            return None
        if priority == Priority.IDLE:
            return f"the {part} plays {held.performance.intent} ({held.source})"
        if (priority, rank) > (held.priority, held.rank):
            return f"the {part} plays {held.performance.intent}, which goes first ({held.source})"
        if now >= held.stroke:
            return f"the {part} is past the stroke of {held.performance.intent} ({held.source})"
        return None

    def reserve(self, part: PartName, reservation: Reservation, now: float) -> str | None:
        """Take ``part`` for ``reservation``: None when taken, else why not."""
        refusal = self.refusal(part, reservation.priority, reservation.rank, now)
        if refusal is None:
            self.held[part] = self._last[part] = reservation
        return refusal

    def release(self, part: PartName) -> Reservation | None:
        """``part`` let go (a fade, a rest): what held it."""
        return self.held.pop(part, None)

    def holder(self, part: PartName, now: float) -> Reservation | None:
        """What holds ``part`` now (None: it is free)."""
        held = self.held.get(part)
        return held if held is not None and now < held.ends else None

    def last(self, part: PartName) -> Reservation | None:
        """The latest reservation of ``part``, playing or played (its stroke: a listener's gap)."""
        return self._last.get(part)

    def clear(self) -> None:
        """Every part free (everything rests)."""
        self.held.clear()


@dataclass
class Waiting:
    """A clip waiting for its time: its request, the cue of a word's anchor, the time on the
    engine's clock of one anchored on the clock."""

    performance: Performance
    clip: Clip
    request: Request
    cue: Cue | None = None
    due: float = 0.0

    @property
    def source(self) -> str:
        """Who asked."""
        return self.request.source

    @property
    def turn(self) -> int | None:
        """The reply it belongs to (dropped with it), None for none."""
        if self.cue is not None:
            return self.cue.turn
        anchor_ = self.request.anchor
        return anchor_.turn if isinstance(anchor_, AtStream) else None


@dataclass(frozen=True)
class Due:
    """A clip whose time has come, and the stream seconds it was planned for and went at (None
    for one anchored on the clock)."""

    waiting: Waiting
    timing: tuple[float, float] | None = None


@dataclass
class Spoken:
    """A part of a reply as the voice has it: its spoken text, whether all its audio is with the
    player, and the stream second its first chunk began at (None before it)."""

    text: str
    spoken: bool = False
    start_s: float | None = None


class Scheduler:
    """Each clip of a request kept until its time (the module's notes), the parts' reservations,
    and what it needs of the conversation to place a word."""

    def __init__(
        self, playout: Playout, latency: Callable[[PartName], float], record: Record
    ) -> None:
        """``latency(part)``: the part's own delay, its clips go this much earlier."""
        self.playout = playout
        self.latency = latency
        self.record = record
        self.reservations = Reservations()
        self.waiting: list[Waiting] = []
        self.spoken: dict[tuple[int, int], Spoken] = {}
        self.audio_lag_s, self.late_s, self.poll_s = 0.15, 0.5, 0.02

    def configure(self, *, audio_lag_s: float, late_s: float, poll_s: float) -> None:
        """The knobs (config/expression.json ``timing``)."""
        self.audio_lag_s, self.late_s, self.poll_s = audio_lag_s, late_s, poll_s

    # -- the conversation ------------------------------------------------------------------------

    def part_queued(self, turn: int, index: int, text: str) -> None:
        """Part ``index`` of reply ``turn`` went to the voice saying ``text``."""
        self.spoken[(turn, index)] = Spoken(text)

    def part_started(self, turn: int, index: int, stream_s: float) -> None:
        """The part's first chunk is with the player at ``stream_s``."""
        spoken = self.spoken.get((turn, index))
        if spoken is not None:
            spoken.start_s = stream_s

    def part_spoken(self, turn: int, index: int) -> None:
        """All of the part's audio is with the player: its length is known."""
        spoken = self.spoken.get((turn, index))
        if spoken is not None:
            spoken.spoken = True

    def text_ahead(self, turn: int, sentence: int, text: str) -> None:
        """A sentence of reply ``turn`` before it is whole (nothing placed on it here)."""

    def forget(self, turn: int, *, only: bool = False) -> None:
        """The parts of replies up to ``turn`` (``only``: that one) are no longer placed."""
        for key in [k for k in self.spoken if (k[0] == turn if only else k[0] <= turn)]:
            del self.spoken[key]

    def cue(self, at: AtWord, name: str, channel: Channel = "gesture") -> Cue | None:
        """A word's anchor as a cue on its part (None before the part went to the voice)."""
        spoken = self.spoken.get((at.turn, at.part))
        if spoken is None:
            return None
        char = min(max(0, at.char), len(spoken.text))
        share = min(1.0, char / max(1, len(spoken.text)))
        return Cue(Marker(channel, name, char), at.turn, at.part, share, *anchor(spoken.text, char))

    # -- the clips -------------------------------------------------------------------------------

    def place(
        self, performance: Performance, request: Request, cue: Cue | None, now: float
    ) -> list[Waiting]:
        """Keep each clip of ``performance`` until its time; the ones due at once, in order."""
        anchor_ = request.anchor
        clips = sorted(performance.clips, key=lambda c: c.offset_s)
        first = min((c.offset_s for c in clips), default=0.0)
        at_once: list[Waiting] = []
        for clip in clips:
            waiting = Waiting(performance, clip, request, cue)
            if cue is None and not isinstance(anchor_, AtStream):
                if isinstance(anchor_, AtTime):
                    waiting.due = anchor_.t + clip.offset_s - self.latency(clip.part)
                else:
                    waiting.due = now + clip.offset_s - first
                if waiting.due <= now:
                    at_once.append(waiting)
                    continue
            self.waiting.append(waiting)
        return at_once

    def due(self, now: float) -> tuple[list[Due], float | None]:
        """The clips whose time has come (a late one past ``late_s`` dropped and logged), and the
        seconds until the next check (None: nothing waits)."""
        later: list[Waiting] = []
        going: list[Due] = []
        wait: float | None = None

        def check(seconds: float) -> None:
            nonlocal wait
            wait = seconds if wait is None else min(wait, seconds)

        for waiting in self.waiting:
            if waiting.cue is None and not isinstance(waiting.request.anchor, AtStream):
                if waiting.due <= now:
                    going.append(Due(waiting))
                else:
                    later.append(waiting)
                    check(waiting.due - now)
                continue
            target = self.target(waiting)
            if target is None:
                spoken = self._spoken(waiting)
                if spoken is not None and spoken.spoken:
                    going.append(Due(waiting))  # a part with no audio: its cues at once
                    continue
                later.append(waiting)
                check(self.poll_s)
                continue
            played = self.playout.played()
            if played < target:  # checked again soon: the part's length grows while it is made
                later.append(waiting)
                check(max(0.001, min(self.poll_s, target - played)))
                continue
            if played - target > self.late_s:
                self.record("late", {"intent": waiting.performance.intent,
                                     "part": waiting.clip.part, "planned": round(target, 3),
                                     "played": round(played, 3)})  # fmt: skip
                continue
            going.append(Due(waiting, (target, played)))
        self.waiting = later
        return going, wait

    def target(self, waiting: Waiting) -> float | None:
        """The stream second a clip anchored on a word or a stream second goes at; None before
        its part's first audio."""
        latency = self.latency(waiting.clip.part)
        anchor_ = waiting.request.anchor
        if waiting.cue is None and isinstance(anchor_, AtStream):
            return anchor_.second + waiting.clip.offset_s + self.audio_lag_s - latency
        cue = waiting.cue
        assert cue is not None
        mark = self.playout.mark(cue.turn, cue.index)
        spoken = self._spoken(waiting)
        if mark is None or spoken is None:
            return None
        word = sync_s(cue, mark, len(spoken.text), spoken.spoken)
        return word + waiting.clip.offset_s + self.audio_lag_s - latency

    def _spoken(self, waiting: Waiting) -> Spoken | None:
        cue = waiting.cue
        return None if cue is None else self.spoken.get((cue.turn, cue.index))

    def drop(self, turn: int, why: str, *, only: bool = False) -> list[Waiting]:
        """What waits for replies up to ``turn`` (``only``: that one) dropped and logged."""

        def gone(t: int | None) -> bool:
            return t is not None and (t == turn if only else t <= turn)

        dropped = [w for w in self.waiting if gone(w.turn)]
        for waiting in dropped:
            self.record("dropped", {"intent": waiting.performance.intent, "part": waiting.clip.part,
                                    "turn": turn, "why": why})  # fmt: skip
        self.waiting = [w for w in self.waiting if w not in dropped]
        self.forget(turn, only=only)
        return dropped

    def clear(self) -> None:
        """Nothing waits and every part is free (a drive, the close, the switch off)."""
        self.waiting.clear()
        self.reservations.clear()


@dataclass(frozen=True)
class Nucleus:
    """A syllable's nucleus heard in the voice's own audio: of part ``index`` of reply ``turn``,
    at stream second ``second``, its prominence (a z-sum over the voice's own statistics) and
    its pitch over the voice's median (semitones; None unvoiced)."""

    turn: int
    index: int
    second: float
    z: float
    pitch_st: float | None = None


class Nuclei(Protocol):
    """The prosody listener's product as the word clock reads it (A writes it, B reads it)."""

    def nuclei(self, turn: int, index: int) -> Sequence[Nucleus]:
        """The part's nuclei found so far, in order."""
        ...

    def complete(self, turn: int, index: int) -> bool:
        """Whether the part's whole audio has been analysed."""
        ...


class Places(Protocol):
    """The robot's places (list_places) as directions from where it stands (C)."""

    def bearing_deg(self, name: str) -> float | None:
        """Place ``name``'s bearing in the cart's frame (degrees, + left), from the robot's map
        pose; None for no such place or no pose."""
        ...
