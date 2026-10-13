"""The gesture engine's scheduler: requests from both trigger sources (the brain's markers; the
engine's own listeners and moments; a person's ask), each anchored on a word, a stream second or
the clock, placed on the body's parts by one priority table (gesticulation DESIGN.md §6.5, §7).

A :class:`Request` says what (an intent of the book, or clips already made: the melody), where
(an :data:`Anchor`: a character of a reply's part, a second of the player's stream, a time on
the engine's clock, or now), how much it matters (:class:`Priority`, and a rank inside it), who
asked, how tightly it binds to its word and the family a marker of the same kind absorbs it
into. The engine (:class:`pepin.expression.animator.GestureEngine`) renders it and checks the
caps; :class:`Scheduler` fits it to the time there is, keeps each clip until its time and says
which go now; :class:`Reservations` says which request holds each part, and which waiting clip
each part is promised to.

ON THE WORD (a request anchored on a character of a part):

* Decided at TEXT, when the part goes to the voice: the word's time comes from the word clock
  (:class:`pepin.expression.forecast.WordClock`), the stroke aimed ``stroke_bias_s`` before the
  word's vowel is heard. THE LADDER (§7.6) picks the first version whose computed preparation fits
  the lead less the forecast's spread (at most 0.3 s): as drawn, the arm from where it will be
  (rest, or the pose a unit holds); its preparation compressed to 0.9 of the caps; for a PHRASE mark
  the same on one of the next two words (no pause between, at most ``phrase_slide_s`` later);
  another, smaller variant; the floor (the book's head-and-face variant, else the drawn one without
  the arm and the wheels); else dropped "no time". The arm's preparation is its plan
  (:meth:`pepin.expression.parts.ArmPart.plan`: the board's approach and the stretched stroke); for
  a WORD or PHRASE mark a plan stretched past the caps is a step down, never a silent late stroke. A
  head or face clip that cannot make it is left out, the rest keeps its time (one composition, many
  parts: each anchored on the word). The arm and the wheels play together (one body, each part
  within its own caps); the wheels never bind to a word. A version whose arm or wheels run longer
  than ``short_s`` plays only when the speech after its stroke covers the motion after it, up to 0.5
  s past (the fed audio as it is, 0.7 of the forecast of the rest: the stroke's own time is that
  same forecast, so the share is taken of what follows it, not of the reply from its start), a
  ``long`` one (hello, bye) up to 1 s past it.
* Never in a pause: a WORD or PHRASE mark after its part's last word is dropped (a sentence's
  end is 0.40-1.01 s of silence, MEASURED: no slide crosses it); an UTTERANCE one there takes
  the part's last word; a mark before punctuation already belongs to the word after it.
* Sent at the last moment: each poll the clock's best estimate (TEXT, START, AUDIO) moves the
  send time; a clip goes when its stroke can still land on time. A WORD or PHRASE stroke is
  never later than ``stroke_late_s``, an UTTERANCE one never later than ``late_s`` nor past the
  reply's speech: past that the clip is dropped and the rest of its composition plays.
* Units (§7.10): an arm clip with another arm stroke waiting within ``units.hold_s`` after its
  end goes ``then: hold`` until shortly after that stroke's start, which plans from the held pose
  (at most ``units.strokes`` a unit); otherwise ``then: rest``. The listeners' POSTURE (the arm
  held at the book's ``ready`` while a long reply goes on, :data:`POSTURE`) opens a unit with no
  stroke of its own, and each posture sent starts the unit's count of strokes again: while the
  listeners want the arm up for a reply (:attr:`Scheduler.stay_up`: its posture due or up), its
  last stroke ends ``then: hold`` instead of rest, for the rest of the posture's hold when that
  outlasts it, else ``units.hold_s`` (the posture takes the arm from that pose).
* Ahead of TEXT (§7.11): a marker the brain's token stream shows before its sentence is whole
  (``text_ahead``) may send the arm to its home pose, held, when it could not reach its stroke
  from rest otherwise: a preparation only, never a stroke.

Reservations are made when a request is placed (render time): a clip is refused a part promised
to a request that goes first (a higher place in the table, or an equal one placed earlier),
and displaces the waiting clips of the ones it goes before; the idle layer never books and
gives way to everything. Per part one request plays at a time (:meth:`Reservations.refusal`).

A clip anchored on a stream second goes when the stream has played up to that second + its
offset + ``audio_lag_s`` - its part's latency, at once when that is past by at most ``late_s``
(``stroke_late_s`` for a word- or phrase-bound one), dropped later; on the clock, at its time
less its part's latency; now, with the performance's earliest clip at once. A TAIL (bound
``after``: the laugh's rock) starts when its stream second is heard, its offset aside, and goes
as long as its stroke still lands within ``after_s`` of that second and inside the reply's
speech; one whose stroke alone is longer is dropped when placed.

What waits is visible to the engine's listeners (:meth:`Scheduler.pending`: each clip's
intent, part, place in the table and window), and a requester may withdraw its own clips
(:meth:`Scheduler.withdraw`).

Logged per clip of a word: ``ladder`` at render (binding, variant, step and why, the lead, the
forecast's moment and spread), ``send`` at its send (the heard time at TEXT, START and AUDIO,
the send, the planned stroke and its error against the aim, all seconds after the part's TEXT),
``late``, ``dropped``, ``refused``, ``displaced`` and ``prepared`` when they happen.

Owners (gesticulation slices, DESIGN.md §11): B owns this file's internals; A posts requests
(:meth:`pepin.expression.animator.GestureEngine.request`) and reads the holds
(:meth:`Reservations.last`) and what waits (:meth:`pepin.expression.animator.GestureEngine.
pending`); the names and signatures the engine calls are the contract.
"""

from __future__ import annotations

import enum
import math
import re
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal, Protocol

from pepin.expression.animation import (
    Bind,
    Clip,
    Composition,
    PartName,
    Performance,
    Scene,
    Track,
)
from pepin.expression.book import vocabulary
from pepin.expression.executors import Executor
from pepin.expression.forecast import (
    SPREAD_INSIDE_S,
    SPREAD_PART_S,
    SPREAD_PER_S,
    ClockKnobs,
    Forecast,
    Spoken,
    Stage,
    WordClock,
)
from pepin.expression.markers import Channel, Cue, Marker, Playout, anchor, parse
from pepin.expression.mix import PREP_CAPS, prep_min_s
from pepin.expression.parts import ArmPart, Planned
from pepin.expression.trajectory import JOINTS

__all__ = ["Spoken"]  # the clock's part record, as the engine clears it (``spoken``)

Record = Callable[[str, dict[str, Any]], None]
SceneHook = Callable[[Scene, "Request"], Scene]  # the engine's: what a rendering may use, refined
# How a request's stroke binds: a word's, a phrase's, an utterance's (the book's), or a tail's
# (``after``: within ``after_s`` of its stream second; the listeners' laugh rock)
RequestBind = Bind | Literal["after"]
NOT_YET = math.inf  # the end of a reservation that lasts until let go (a looping idle clip)
MARGIN_S = 0.3  # the most a forecast's spread holds a preparation back (DESIGN.md §7.6)
EARLY_S = 0.3  # a stroke this much before its vowel is still on it (early is free: Nirme 2024)
POST_STROKE_S = 0.15  # a unit's next stroke leaves the last one no sooner after its stroke
UNIT_SLACK_S = 0.3  # a unit's hold reaches this far past the next stroke's start
LONG_PAST_S = 1.0  # a long composition (hello, bye) may run this far past the speech
SHORT_PAST_S = 0.5  # any other this far (DESIGN.md §7.9)
SURE_SHARE = 0.7  # of the speech still a forecast, the share a long gesture may count on
PHRASE_WORDS = 2  # a PHRASE mark may slide to one of this many next words
HOME = "ready"  # the book's pose an arm goes to ahead of a mark it cannot reach from rest
POSTURE = "arm_up"  # the listeners' posture: the arm held at HOME while a long reply goes on
AHEAD_S = 0.18  # a marker's sentence reaches the voice this long after it streams in (p50)
PREPARE = "prepare:"  # a preparation's name: the arm sent home ahead of a marker (no stroke)
SAME_POSE_DEG = 0.5  # a held pose within this of a book's pose on every joint is that pose
_WORD = re.compile(r"\w+")
_VOWEL = re.compile("[аеёиоуыэюяaeiouy]", re.IGNORECASE)  # noqa: RUF001 -- Cyrillic vowels
_PAUSE = re.compile("[.,!?;:\u2013\u2014\u2026]")  # a pause between two words


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
    "tool", "pulse", "idle", ...), how its stroke binds to its word (or ``after`` it: a tail)
    and the family a marker of the same kind absorbs it into. A request is itself:
    :meth:`Scheduler.withdraw` and :meth:`pepin.expression.animator.GestureEngine.release` find
    its clips by identity."""

    intent: str
    anchor: Anchor | None = None
    priority: Priority = Priority.ASK
    source: str = "tool"
    bind: RequestBind = "utterance"
    family: str | None = None
    rank: int = 0
    clips: tuple[Clip, ...] = ()


@dataclass(frozen=True)
class Timing:
    """config/expression.json's ``timing`` and ``units`` blocks as the scheduler reads them."""

    audio_lag_s: float = 0.15
    late_s: float = 0.5
    poll_s: float = 0.02
    char_s: float = 0.09
    tts_first_s: float = 0.10
    lead_silence_s: float = 0.16
    stroke_bias_s: float = -0.05
    stroke_late_s: float = 0.15
    short_s: float = 1.0
    phrase_slide_s: float = 0.5
    after_s: float = 2.5
    unit_hold_s: float = 2.5
    unit_strokes: int = 3

    def clock(self) -> ClockKnobs:
        """The word clock's share of them."""
        return ClockKnobs(self.char_s, self.tts_first_s, self.lead_silence_s, self.audio_lag_s)


def timing_from(data: Mapping[str, Any]) -> Timing:
    """config/expression.json's object as the scheduler's knobs (defaults for what it lacks)."""
    timing, units, d = data.get("timing", {}), data.get("units", {}), Timing()
    numbers = {
        name: float(timing.get(name, getattr(d, name)))
        for name in (
            "audio_lag_s", "late_s", "poll_s", "char_s", "tts_first_s", "lead_silence_s",
            "stroke_bias_s", "stroke_late_s", "short_s", "phrase_slide_s", "after_s",
        )
    }  # fmt: skip
    return Timing(
        **numbers,
        unit_hold_s=float(units.get("hold_s", d.unit_hold_s)),
        unit_strokes=int(units.get("strokes", d.unit_strokes)),
    )


@dataclass
class Reservation:
    """What holds a part: the performance and its clip, sent at ``started`` (engine clock), its
    stroke and its end on the same clock, the request's place in the table, and the request
    itself (None: an idle clip's)."""

    performance: Performance
    clip: Clip
    started: float
    stroke: float
    ends: float
    priority: Priority = Priority.ASK
    rank: int = 0
    source: str = "tool"
    request: Request | None = field(default=None, compare=False, repr=False)


@dataclass(frozen=True)
class Pending:
    """A clip waiting for its time, as a listener sees it: its intent, part, place in the table,
    who asked, the reply's part it belongs to (None: none), and its window on the engine's clock
    (start of motion, stroke, end) as its word's forecast says now."""

    intent: str
    part: PartName
    priority: Priority
    rank: int
    source: str
    turn: int | None
    index: int | None
    start: float
    stroke: float
    end: float


@dataclass
class Booking:
    """A part promised to a waiting clip: when it starts moving, its stroke and its end (engine
    clock, refined as the word's forecast is), the request's place, and whose it is."""

    part: PartName
    start: float
    stroke: float
    end: float
    priority: Priority
    rank: int
    intent: str
    owner: int

    def first(self, priority: Priority, rank: int) -> bool:
        """Whether this booking goes before a request at ``priority`` and ``rank`` (a person's
        ask and above take any part)."""
        return priority > Priority.ASK and (self.priority, self.rank) <= (priority, rank)


class Held(dict[PartName, Reservation]):
    """What holds each part (the engine writes a clip's reservation here as it sends it), the
    latest of each part remembered after it is let go."""

    def __init__(self) -> None:
        """Every part free."""
        super().__init__()
        self.latest: dict[PartName, Reservation] = {}

    def __setitem__(self, part: PartName, reservation: Reservation) -> None:
        """``part`` held by ``reservation``."""
        super().__setitem__(part, reservation)
        self.latest[part] = reservation


class Reservations:
    """Per part one reservation at a time, by the priority table, and the bookings of the clips
    that wait (the module's notes)."""

    def __init__(self) -> None:
        """Every part free."""
        self.held: Held = Held()
        self.booked: dict[PartName, list[Booking]] = {}

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
        if now >= held.stroke and held.clip.then != "hold":  # a unit's hold waits for the next
            return f"the {part} is past the stroke of {held.performance.intent} ({held.source})"
        return None

    def reserve(self, part: PartName, reservation: Reservation, now: float) -> str | None:
        """Take ``part`` for ``reservation``: None when taken, else why not."""
        refusal = self.refusal(part, reservation.priority, reservation.rank, now)
        if refusal is None:
            self.held[part] = reservation
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
        return self.held.latest.get(part)

    def book(self, booking: Booking) -> tuple[Booking | None, list[Booking]]:
        """Promise ``booking``'s part to it: the booking it clashes with that goes first (then
        nothing is booked), else the ones it goes before (the caller lets them go)."""
        clashing = [
            b
            for b in self.booked.get(booking.part, [])
            if b.owner != booking.owner and b.start < booking.end and booking.start < b.end
        ]
        for other in clashing:
            if other.first(booking.priority, booking.rank):
                return other, []
        self.booked.setdefault(booking.part, []).append(booking)
        return None, clashing

    def unbook(self, owner: int) -> None:
        """Every booking of ``owner`` let go."""
        for part, bookings in list(self.booked.items()):
            self.booked[part] = [b for b in bookings if b.owner != owner]

    def bookings(self, part: PartName) -> list[Booking]:
        """The part's bookings, in time order."""
        return sorted(self.booked.get(part, []), key=lambda b: b.start)

    def clear(self) -> None:
        """Every part free (everything rests)."""
        self.held.clear()
        self.booked.clear()


@dataclass
class Fit:
    """How a clip on a word was fitted: its binding, the ladder's step and why, the lead it had
    at render (to its aimed stroke, its part's latency off), when its part's text went to the
    voice, its preparation (send to stroke, latency apart; the arm's plan), the first heard
    time of each moment of the word clock, where the arm starts it from, the waiting clip a unit
    chains it to, its predicted window on the engine's clock, the pose it leaves the arm in, and
    whether it plays only from a pose a unit holds (a variant that needs the arm up)."""

    bind: Bind
    step: str
    why: str
    lead_s: float
    text_s: float
    prep_s: float
    heard: dict[Stage, float] = field(default_factory=dict)
    base: str = "-"
    chained: int | None = None
    chain: int = 0
    start: float = 0.0
    stroke: float = 0.0
    end: float = 0.0
    final: dict[str, float] | None = None
    held_only: bool = False


@dataclass(eq=False)
class Waiting:
    """A clip waiting for its time: its request, the cue of a word's anchor, the time on the
    engine's clock of one anchored on the clock, and how it was fitted to its word."""

    performance: Performance
    clip: Clip
    request: Request
    cue: Cue | None = None
    due: float = 0.0
    fit: Fit | None = None

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
    for one anchored on the clock, or sent before its part had a place in the stream)."""

    waiting: Waiting
    timing: tuple[float, float] | None = None


@dataclass(frozen=True)
class ArmFrom:
    """Where an arm clip may start from: ``rest``, a ``unit``'s held pose (the clip rebased on
    it, the board starting from it; the waiting clip it chains to and the strokes before it) or
    ``returning`` to rest (the board's approach from the last pose), and the window its start
    must fall in (engine clock)."""

    kind: str
    pose: dict[str, float] | None
    knot: tuple[float, ...] | None
    after: float
    until: float
    chained: int | None = None
    chain: int = 0


@dataclass(frozen=True)
class Candidate:
    """One rung of the ladder: its name, the performance, the cue its movers aim at, and whether
    the arm's preparation is compressed."""

    step: str
    performance: Performance
    cue: Cue
    compress: bool = False


class Rendering(Protocol):
    """What the scheduler asks of the engine's renderer (:class:`pepin.expression.render.
    BookRenderer`): its book, and another variant of a composition."""

    @property
    def book(self) -> Any:
        """The book (its compositions and poses)."""
        ...

    def render(
        self,
        intent: str,
        scene: Scene,
        seed: int,
        cue: Cue | None = None,
        *,
        variant: int | None = None,
    ) -> Performance:
        """The performance of ``intent`` by ``variant``."""
        ...


class Scheduler:
    """Each request fitted to its time and each clip kept until it (the module's notes), the
    parts' reservations, and the word clock over the conversation's parts."""

    def __init__(
        self, playout: Playout, latency: Callable[[PartName], float], record: Record
    ) -> None:
        """``latency(part)``: the part's own delay, its clips go this much earlier."""
        self.playout = playout
        self.latency = latency
        self.record = record
        self.timing = Timing()
        self.words = WordClock(playout, self.timing.clock())
        self.reservations = Reservations()
        self.waiting: list[Waiting] = []
        self.parts: Mapping[PartName, Executor] = {}
        self._renderer: Callable[[], Rendering] | None = None
        self._hooks: Sequence[SceneHook] = ()
        self._now = 0.0  # the engine's clock as last told (a call that carries none uses it)
        self._seen: set[tuple[int, int, int, str]] = set()  # token-stream marks looked at
        self._arm_last: tuple[dict[str, float], float] | None = None  # the last motion's end
        self._unit = 0  # strokes of the unit the arm holds
        # whether the listeners want the arm up after reply ``turn``'s last stroke (its posture)
        self.stay_up: Callable[[int], bool] = _never

    @property
    def spoken(self) -> dict[tuple[int, int], Spoken]:
        """The parts the clock holds (the engine clears them at the conversation's close)."""
        return self.words.parts

    @property
    def audio_lag_s(self) -> float:
        """How much later the speaker sounds than the player's clock says."""
        return self.timing.audio_lag_s

    @property
    def late_s(self) -> float:
        """How late an UTTERANCE-bound stroke may still land."""
        return self.timing.late_s

    @property
    def poll_s(self) -> float:
        """How often a waiting clip is looked at."""
        return self.timing.poll_s

    def configure(
        self, *, audio_lag_s: float, late_s: float, poll_s: float, timing: Timing | None = None
    ) -> None:
        """The knobs (config/expression.json ``timing`` and ``units``): the three the engine
        reads itself over ``timing``'s (:func:`timing_from`; its defaults when not given)."""
        base = timing if timing is not None else self.timing
        self.timing = replace(base, audio_lag_s=audio_lag_s, late_s=late_s, poll_s=poll_s)
        self.words.knobs = self.timing.clock()

    def attach(
        self,
        parts: Mapping[PartName, Executor],
        renderer: Callable[[], Rendering] | None = None,
        nuclei: Nuclei | None = None,
        hooks: Sequence[SceneHook] = (),
    ) -> None:
        """What the ladder computes with: the engine's parts (the arm's planner, who is ready),
        its renderer (the book's other variants and floors; a callable: the engine makes a new
        one when its book is read again), the prosody listener's nuclei (the AUDIO moment) and
        the engine's scene hooks (the list itself: every rendering of another variant passes
        them, as the engine's own does: the quiet rule, the laugh without the arm). Without
        them a word's clips are fitted as drawn, the floor being the drawn version without the
        arm and the wheels."""
        self.parts = parts
        self._renderer = renderer
        self.words.nuclei = nuclei
        self._hooks = hooks

    # -- the conversation ------------------------------------------------------------------------

    def part_queued(self, turn: int, index: int, text: str) -> None:
        """Part ``index`` of reply ``turn`` went to the voice saying ``text`` (TEXT)."""
        self.words.queued(turn, index, text)

    def part_started(self, turn: int, index: int, stream_s: float) -> None:
        """The part's first chunk is with the player at ``stream_s`` (START)."""
        self.words.started(turn, index, stream_s)

    def part_spoken(self, turn: int, index: int) -> None:
        """All of the part's audio is with the player: its length is known."""
        self.words.spoken(turn, index)

    def text_ahead(self, turn: int, sentence: int, text: str) -> None:
        """A sentence of reply ``turn`` before it is whole: for each new marker in it, the arm
        sent home ahead of it when it could not reach the stroke from rest (a preparation only,
        never a stroke)."""
        rendering = self._rendering()
        if rendering is None:
            return
        words = vocabulary(rendering.book, frozenset())  # the arguments and other spellings too
        first = sum(1 for t, _ in self.words.parts if t == turn)
        for k, part in enumerate(parse(text, words).parts):
            for marker in part.markers:
                key = (turn, first + k, marker.at, marker.name)
                if marker.channel != "gesture" or key in self._seen:
                    continue
                self._seen.add(key)
                self._ahead(rendering, turn, first + k, part.text, marker)

    def forget(self, turn: int, *, only: bool = False) -> None:
        """The parts of replies up to ``turn`` (``only``: that one) are no longer placed."""

        def gone(t: int) -> bool:
            return t == turn if only else t <= turn

        self.words.forget(turn, only=only)
        self._seen = {s for s in self._seen if not gone(s[0])}

    def cue(self, at: AtWord, name: str, channel: Channel = "gesture") -> Cue | None:
        """A word's anchor as a cue on its part (None before the part went to the voice)."""
        spoken = self.words.parts.get((at.turn, at.part))
        if spoken is None:
            return None
        char = min(max(0, at.char), len(spoken.text))
        share = min(1.0, char / max(1, len(spoken.text)))
        return Cue(Marker(channel, name, char), at.turn, at.part, share, *anchor(spoken.text, char))

    def arm_at(self) -> str | None:
        """Where a rendering finds the arm (:attr:`pepin.expression.animation.Scene.arm_at`):
        the pose the latest waiting arm clip will leave held for a next stroke, else the one a
        unit holds now (by the book's name when it is one of its poses, ``held`` otherwise);
        ``rest`` when neither; None when no arm planner was lent (unknown)."""
        arm = self._arm_part()
        if arm is None:
            return None
        strokes = self.timing.unit_strokes
        waiting = [
            w.fit
            for w in self.waiting
            if w.clip.part == "arm" and w.fit is not None and w.fit.final is not None
        ]
        latest = max(waiting, key=lambda fit: fit.stroke, default=None)  # fmt: skip
        if latest is not None:
            pose = latest.final if latest.chain + 1 < strokes else None
        else:
            pose = arm.holding() if self._unit < strokes else None
        if pose is None:
            return "rest"
        rendering = self._rendering()
        poses: Mapping[str, Mapping[str, float]] = rendering.book.poses if rendering else {}
        return next(
            (name for name, known in poses.items()
             if all(abs(known.get(j, math.inf) - pose[j]) < SAME_POSE_DEG for j in JOINTS)),
            "held",
        )  # fmt: skip

    # -- placing ---------------------------------------------------------------------------------

    def place(
        self, performance: Performance, request: Request, cue: Cue | None, now: float
    ) -> list[Waiting]:
        """Fit ``performance`` to its anchor and keep each clip until its time; the ones due at
        once, in order."""
        self._tell(now)
        anchor_ = request.anchor
        at_once: list[Waiting] = []
        if cue is not None:
            for waiting in self._fit(performance, request, cue, now):
                if not self._book(waiting, now):
                    continue
                self.waiting.append(waiting)
                checked = self._check(waiting, now)
                if isinstance(checked, Due) or checked is None:
                    self.waiting.remove(waiting)
                if isinstance(checked, Due):
                    at_once.append(waiting)
            return at_once
        clips = sorted(performance.clips, key=lambda c: c.offset_s)
        first = min((c.offset_s for c in clips), default=0.0)
        if request.intent == POSTURE and any(c.part == "arm" and c.then == "hold" for c in clips):
            self._unit, self._arm_last = 0, None  # a unit opened anew
        for clip in clips:
            waiting = Waiting(performance, clip, request)
            if _tail(waiting) and clip.stroke_s > self.timing.after_s:
                why = f"its stroke {clip.stroke_s:.2f} s after its start, past after_s"
                self.record("dropped", {"intent": performance.intent, "part": clip.part,
                                        "why": why})  # fmt: skip
                continue
            if isinstance(anchor_, AtTime):
                waiting.due = anchor_.t + clip.offset_s - self.latency(clip.part)
            elif not isinstance(anchor_, AtStream):
                waiting.due = now + clip.offset_s - first
            if not self._book(waiting, now):
                continue
            if not isinstance(anchor_, AtStream) and waiting.due <= now:
                self.reservations.unbook(id(waiting))
                at_once.append(waiting)
                continue
            self.waiting.append(waiting)
        return at_once

    def due(self, now: float) -> tuple[list[Due], float | None]:
        """The clips whose time has come (a late one dropped and logged), and the seconds until
        the next check (None: nothing waits)."""
        self._tell(now)
        going: list[Due] = []
        later: list[Waiting] = []
        wait: float | None = None
        for waiting in list(self.waiting):
            if waiting not in self.waiting:
                continue  # displaced by one sent before it in this poll
            checked = self._check(waiting, now)
            if isinstance(checked, Due):
                going.append(checked)
            elif checked is None:
                self.reservations.unbook(id(waiting))
            else:
                later.append(waiting)
                wait = checked if wait is None else min(wait, checked)
        self.waiting = later
        return going, wait

    def drop(self, turn: int, why: str, *, only: bool = False) -> list[Waiting]:
        """What waits for replies up to ``turn`` (``only``: that one) dropped and logged, with
        the clock's forecasts of their parts."""

        def gone(t: int | None) -> bool:
            return t is not None and (t == turn if only else t <= turn)

        dropped = [w for w in self.waiting if gone(w.turn)]
        for waiting in dropped:
            self.reservations.unbook(id(waiting))
            self.record("dropped", {"intent": waiting.performance.intent, "part": waiting.clip.part,
                                    "turn": turn, "why": why})  # fmt: skip
        self.waiting = [w for w in self.waiting if w not in dropped]
        self.forget(turn, only=only)
        return dropped

    def clear(self) -> None:
        """Nothing waits and every part is free (a drive, the close, the switch off)."""
        self.waiting.clear()
        self.reservations.clear()
        self._arm_last = None
        self._unit = 0

    def withdraw(self, request: Request) -> list[Waiting]:
        """The clips of ``request`` (this very request) still waiting, let go: its requester
        ended it."""
        gone = [w for w in self.waiting if w.request is request]
        for waiting in gone:
            self.reservations.unbook(id(waiting))
        self.waiting = [w for w in self.waiting if w.request is not request]
        return gone

    def pending(
        self, part: PartName | None = None, turn: int | None = None, now: float | None = None
    ) -> list[Pending]:
        """What waits for its time (of ``part``, of reply ``turn`` when given), each with its
        window as the forecasts say at ``now`` (the engine's clock as last told when None), in
        the order of their strokes."""
        at = self._now if now is None else now
        out: list[Pending] = []
        for waiting in self.waiting:
            clip, request, cue = waiting.clip, waiting.request, waiting.cue
            if (part is not None and clip.part != part) or (
                turn is not None and waiting.turn != turn
            ):
                continue
            start, stroke, end = self._window(waiting, at)
            out.append(Pending(waiting.performance.intent, clip.part, request.priority,
                               request.rank, request.source, waiting.turn,
                               cue.index if cue is not None else None, start, stroke,
                               end))  # fmt: skip
        return sorted(out, key=lambda p: p.stroke)

    def target(self, waiting: Waiting) -> float | None:
        """The stream second a clip goes at: one on a stream second, or one on a word whose part
        has a place in the stream (None: not yet, or anchored on the clock)."""
        anchor_ = waiting.request.anchor
        latency = self.latency(waiting.clip.part)
        lag = self.timing.audio_lag_s
        if waiting.cue is None and isinstance(anchor_, AtStream):
            return anchor_.second + _offset(waiting) + lag - latency
        if waiting.cue is None or waiting.fit is None:
            return None
        forecast = self._forecast(waiting.cue, self._now)
        if forecast is None or forecast.stream is None:
            return None
        return forecast.stream + lag + self.timing.stroke_bias_s - latency - waiting.fit.prep_s

    def _tell(self, now: float) -> None:
        """The engine's clock, as a call brought it: parts queued since take it as their TEXT,
        and a preparation made on an older time starts no sooner than now."""
        self._now = now
        self.words.stamp(now)
        for waiting in self.waiting:
            fit = waiting.fit
            if fit is not None and fit.step == "prepare" and waiting.due < now:
                shift = now - waiting.due
                waiting.due = now
                fit.start, fit.stroke, fit.end = (
                    t + shift for t in (fit.start, fit.stroke, fit.end)
                )

    # -- the ladder (at render) ------------------------------------------------------------------

    def _fit(
        self, performance: Performance, request: Request, cue: Cue, now: float
    ) -> list[Waiting]:
        """``performance`` fitted to its word by the ladder: its clips, waiting (none: dropped
        and logged)."""
        spoken = self.words.parts.get((cue.turn, cue.index))
        text_s = spoken.queued_s if spoken is not None and spoken.queued_s is not None else now
        bind = _word_bind(request.bind)
        if spoken is None or not spoken.text:  # a part with no audio: its cues at its end
            return [
                Waiting(performance, c, request, cue,
                        fit=Fit(bind, "silent part", "", 0.0, text_s, 0.0))
                for c in performance.clips
            ]  # fmt: skip
        target = self._in_speech(cue, bind)
        if target is None:
            why = "in a pause: no word after it"
            self.record("dropped", {"intent": performance.intent, "turn": cue.turn,
                                    "part": cue.index, "why": why})  # fmt: skip
            return []
        composition = self._composition(performance.intent)
        reasons: list[str] = []
        for candidate in self._candidates(performance, request, target, composition, now):
            fitted = self._try(candidate, request, target, composition, text_s, now, reasons)
            if fitted:
                self._log_ladder(performance, request, candidate, fitted, reasons, now)
                return fitted
        self.record("dropped", {"intent": performance.intent, "turn": cue.turn,
                                "part": cue.index, "why": "no time",
                                "needs": "; ".join(reasons)})  # fmt: skip
        return []

    def _candidates(
        self,
        drawn: Performance,
        request: Request,
        cue: Cue,
        composition: Composition | None,
        now: float,
    ) -> Iterator[Candidate]:
        """The ladder's rungs in order (DESIGN.md §7.6; a PHRASE mark slides before it shrinks)."""
        arm = any(c.part == "arm" for c in drawn.clips)
        yield Candidate("authored", drawn, cue)
        if arm:
            yield Candidate("compressed", drawn, cue, compress=True)
        if request.bind == "phrase":
            for slid in self._slides(cue, now):
                yield Candidate("slid", drawn, slid)
                if arm:
                    yield Candidate("slid, compressed", drawn, slid, compress=True)
        rendering = self._rendering()
        scene = self._scene(request)
        floors: list[Performance] = []
        if composition is not None and rendering is not None:
            for k, variant in enumerate(composition.variants):
                if k == drawn.variant:
                    continue
                other = rendering.render(drawn.intent, scene, drawn.seed, cue, variant=k)
                if variant.floor:
                    floors.append(other)
                    continue
                if not arm and any(c.part == "arm" for c in other.clips):
                    continue  # the engine's own version had no arm (its cap, its scene): none
                yield Candidate("smaller", other, cue)
                if any(c.part == "arm" for c in other.clips):
                    yield Candidate("smaller, compressed", other, cue, compress=True)
        if arm and any(c.part == "body" for c in drawn.clips):  # the wheels' step back in
            yield Candidate("without the arm", replace(drawn, clips=_without(drawn, "arm")), cue)
        for floor in floors:
            yield Candidate("floor", floor, cue)
        rest = _without(drawn, "arm", "body")
        if rest and len(rest) < len(drawn.clips):
            yield Candidate("floor", replace(drawn, clips=rest), cue)

    def _try(
        self,
        candidate: Candidate,
        request: Request,
        mark: Cue,
        composition: Composition | None,
        text_s: float,
        now: float,
        reasons: list[str],
    ) -> list[Waiting]:
        """The candidate's clips fitted to the time there is, or none when its arm or wheels
        cannot make it (a head or face clip that cannot is left out alone)."""
        bind = _word_bind(request.bind)
        clips = list(candidate.performance.clips)
        if bind != "utterance":
            clips = [c for c in clips if c.part != "body"]  # never on a word
        allowance = self.timing.late_s if bind == "utterance" else 0.0
        held_only = self._held_only(candidate.performance)
        fitted: list[Waiting] = []
        for clip in clips:
            cue = mark if clip.part == "face" else candidate.cue  # the face leads, on the mark
            forecast = self._forecast(cue, now)
            if forecast is None:
                continue
            aim = forecast.heard + self.timing.stroke_bias_s
            latency = self.latency(clip.part)
            lead = aim - now - latency
            room = lead + allowance - min(forecast.spread, MARGIN_S)
            fit = Fit(bind, candidate.step, "", lead, text_s, clip.stroke_s)
            fit.heard[forecast.stage] = forecast.heard
            if clip.part == "arm":
                deadline = now + latency + room
                fit.held_only = held_only
                planned = self._arm(clip, fit, aim, deadline, candidate.compress, now)
                if isinstance(planned, str):
                    reasons.append(f"{candidate.step}: {planned}")
                    return []
                clip = planned
            elif fit.prep_s > room:
                if clip.part == "body":
                    reasons.append(f"{candidate.step}: the wheels need {fit.prep_s:.2f} s")
                    return []
                continue  # this part alone cannot make it: the rest keeps its time
            else:  # an utterance-bound one may land late: it starts as soon as it can
                fit.start = max(aim - fit.prep_s, now + latency)
                fit.stroke, fit.end = fit.start + fit.prep_s, fit.start + clip.duration_s
            fitted.append(Waiting(candidate.performance, clip, request, cue, fit=fit))
        if not fitted:
            reasons.append(f"{candidate.step}: no part in time")
            return []
        why = self._too_long(fitted, mark, composition, now)
        if why is not None:
            reasons.append(f"{candidate.step}: {why}")
            return []
        return fitted

    def _arm(
        self, clip: Clip, fit: Fit, aim: float, deadline: float, compress: bool, now: float
    ) -> Clip | str:
        """The arm's clip planned from where the arm is when the board takes it (a unit's held
        pose, returning to rest, or at rest: the first whose window lets it start in time; a
        unit's alone for ``fit.held_only``), its preparation compressed when asked, its stroke
        at ``aim`` or as soon after it as it can, never after ``deadline``; ``fit`` filled in.
        Why it cannot, as a string."""
        arm = self._arm_part()
        earliest = now + self.latency("arm")
        if arm is None:  # no planner lent: the clip's own time from rest
            if fit.held_only:
                return "the arm: no unit to play it from"
            if earliest + clip.stroke_s > deadline:
                return f"the arm needs {clip.stroke_s:.2f} s, {deadline - earliest:.2f} s ahead"
            fit.start = max(aim - clip.stroke_s, earliest)
            fit.stroke, fit.end = fit.start + clip.stroke_s, fit.start + clip.duration_s
            return clip
        whys: list[str] = []
        froms = self._arm_froms(arm, now)
        if fit.held_only:
            froms = [base for base in froms if base.kind == "unit"]
            if not froms:
                return "the arm: no unit to play it from"
        for base in froms:
            first = max(base.after, earliest)
            limit = deadline - first  # the most preparation this start leaves
            planned = arm.plan(clip, base=base.pose, start=base.knot)
            if isinstance(planned, str):
                whys.append(f"from {base.kind}: {planned}")
                continue
            fitted = clip
            if compress and planned.stroke_s > limit > 0.0:
                squeezed = arm.fit(clip, limit, base=base.pose, start=base.knot)
                if squeezed is None:
                    whys.append(f"from {base.kind}: {planned.stroke_s:.2f} s even compressed")
                    continue
                fitted = squeezed
                planned = arm.plan(fitted, base=base.pose, start=base.knot)
                if isinstance(planned, str):
                    whys.append(f"from {base.kind}: {planned}")
                    continue
            if fit.bind != "utterance" and planned.stretch > 1.0 + 1e-6:
                whys.append(f"from {base.kind}: {planned.stretch:.2f}x the caps")
                continue
            if planned.stroke_s > limit:
                whys.append(f"from {base.kind}: {planned.stroke_s:.2f} s, {limit:.2f} ahead")
                continue
            start = max(aim - planned.stroke_s, first)
            if start > base.until:
                whys.append(f"from {base.kind}: the arm is busy then")
                continue
            fit.prep_s = planned.stroke_s
            fit.base, fit.chained, fit.chain = base.kind, base.chained, base.chain
            fit.start, fit.stroke = start, start + planned.stroke_s
            fit.end = start + planned.seconds
            fit.final = arm.final(planned)
            return fitted
        return "the arm " + ", ".join(whys)

    def _arm_froms(self, arm: ArmPart, now: float) -> list[ArmFrom]:
        """Where an arm clip may start from, each with the window its start must fall in: the
        pose a unit holds (the latest arm motion's: a waiting clip's it will be held after,
        else a sent clip's, after its stroke; the arm passes through no earlier one), then
        returning to rest after the last motion, then at rest once every motion known has ended
        and gone back."""
        hold_s, strokes = self.timing.unit_hold_s, self.timing.unit_strokes
        froms: list[ArmFrom] = []
        playing = self.reservations.held.get("arm")
        moving = playing is not None and now < playing.ends
        free = playing.ends if playing is not None and moving else now
        waiting = [w for w in self.waiting if w.clip.part == "arm" and w.fit is not None
                   and w.fit.final is not None]  # fmt: skip
        latest = max(waiting, key=lambda w: w.fit.stroke if w.fit else 0.0, default=None)
        held = arm.holding()
        if held is not None:
            after = now
            if playing is not None and moving:
                post = 0.0 if playing.performance.intent.startswith(PREPARE) else POST_STROKE_S
                after = playing.stroke + post
            if self._unit < strokes and latest is None:
                froms.append(ArmFrom("unit", held, _knot(held), after, arm.held_until))
            free = max(free, arm.held_until + self._return_s(arm, held))
        for other in waiting:
            fit = other.fit
            if fit is None or fit.final is None:
                continue
            post = 0.0 if fit.step == "prepare" else POST_STROKE_S
            if other is latest and fit.chain + 1 < strokes:
                froms.append(ArmFrom("unit", fit.final, _knot(fit.final), fit.stroke + post,
                                     fit.end + hold_s, id(other), fit.chain + 1))  # fmt: skip
            free = max(free, fit.end + self._return_s(arm, fit.final))
        if self._arm_last is not None:
            pose, ended = self._arm_last
            back = ended + self._return_s(arm, pose)
            froms.append(ArmFrom("returning", None, _knot(pose), max(free, ended), back))
            free = max(free, back)
        froms.append(ArmFrom("rest", None, None, free, math.inf))
        return froms

    @staticmethod
    def _return_s(arm: ArmPart, pose: Mapping[str, float]) -> float:
        """The board's way back to rest from ``pose``."""
        return arm.approach_s(_knot(arm.base), _knot(pose))

    def _too_long(
        self, fitted: list[Waiting], mark: Cue, composition: Composition | None, now: float
    ) -> str | None:
        """Why the arm or the wheels would run past what the speech after their stroke covers
        (DESIGN.md §7.9: a motion over ``short_s``, the sure speech after the stroke against the
        motion after it); None when they would not."""
        movers = [w.fit for w in fitted if w.clip.part in ("arm", "body") and w.fit is not None]
        if not movers:
            return None
        first, last = min(f.start for f in movers), max(f.end for f in movers)
        if last - first <= self.timing.short_s:
            return None
        stroke = min(f.stroke for f in movers)
        ahead = self._speech_after(mark.turn, stroke, now)
        if ahead is None:
            return None
        long = composition is not None and composition.long
        if ahead + (LONG_PAST_S if long else SHORT_PAST_S) >= last - stroke:
            return None
        return f"{last - stroke:.1f} s of motion after its stroke, {ahead:.1f} s of speech ahead"

    def _speech_after(self, turn: int, at: float, now: float) -> float | None:
        """The reply's speech after ``at`` (engine clock, the speaker's sound) as far as it is
        sure: a part fed whole as it is, a forecast at :data:`SURE_SHARE` (the share applies to
        what follows ``at`` alone: a stroke's own time comes from that same forecast); None: no
        part placed."""
        indices = sorted(i for t, i in self.words.parts if t == turn)
        if not indices:
            return None
        wall, _, _ = self.words.start(turn, indices[0], now)
        wall += self.timing.audio_lag_s
        ahead = 0.0
        for index in indices:
            length = self.words.length(turn, index)
            after = max(0.0, wall + length - max(wall, at))
            ahead += after if self.words.parts[(turn, index)].spoken else SURE_SHARE * after
            wall += length
        return ahead

    def _speech_end(self, turn: int, now: float) -> float | None:
        """The reply's speech ahead as far as it is sure: the parts fed whole, then
        :data:`SURE_SHARE` of the forecast of the rest (engine clock, the speaker's sound)."""
        indices = sorted(i for t, i in self.words.parts if t == turn)
        if not indices:
            return None
        wall, _, _ = self.words.start(turn, indices[0], now)
        for index in indices:
            spoken = self.words.parts[(turn, index)]
            length = self.words.length(turn, index)
            wall += length if spoken.spoken else SURE_SHARE * length
        return wall + self.timing.audio_lag_s

    def _in_speech(self, cue: Cue, bind: Bind) -> Cue | None:
        """The cue a stroke may aim at inside speech: the mark's own word; with no word after it
        in its part, an UTTERANCE mark's last word of the part (a little early: free); None (a
        pause: a sentence's end is 0.40-1.01 s of silence, MEASURED, no PHRASE slides across it)
        otherwise."""
        text = self.words.parts[(cue.turn, cue.index)].text
        if cue.word[0] < len(text):
            return cue
        words = list(_WORD.finditer(text))
        if bind != "utterance" or not words:
            return None
        return self._on_word(cue, cue.index, text, words[-1].start())

    def _slides(self, cue: Cue, now: float) -> Iterator[Cue]:
        """A PHRASE mark's next words in its part: no pause between, within ``phrase_slide_s``."""
        text = self.words.parts[(cue.turn, cue.index)].text
        here = self._forecast(cue, now)
        if here is None:
            return
        at = cue.word[1]
        for _ in range(PHRASE_WORDS):
            word = _WORD.search(text, at)
            if word is None or _PAUSE.search(text, at, word.start()):
                return
            moved = self._on_word(cue, cue.index, text, word.start())
            there = self._forecast(moved, now)
            if there is None or there.heard - here.heard > self.timing.phrase_slide_s:
                return
            yield moved
            at = word.end()

    @staticmethod
    def _on_word(cue: Cue, index: int, text: str, at: int) -> Cue:
        """``cue`` moved onto the word at ``at`` of part ``index``."""
        span, vowel = anchor(text, at)
        share = min(1.0, at / max(1, len(text)))
        return Cue(replace(cue.marker, at=at), cue.turn, index, share, span, vowel)

    def _forecast(self, cue: Cue, now: float) -> Forecast | None:
        """When the cue's vowel is heard (its stressed nucleus once the audio says)."""
        return self.words.at(cue.turn, cue.index, cue.vowel, now, cue.word)

    def _log_ladder(
        self,
        drawn: Performance,
        request: Request,
        candidate: Candidate,
        fitted: list[Waiting],
        reasons: list[str],
        now: float,
    ) -> None:
        """The ``ladder`` row of a fitted performance."""
        mover = next((w for w in fitted if w.clip.part == "arm"), fitted[0])
        fit, cue = mover.fit, mover.cue
        assert fit is not None and cue is not None
        forecast = self._forecast(cue, now)
        self.record("ladder", {
            "intent": drawn.intent, "source": request.source, "turn": cue.turn,
            "part": cue.index, "char": cue.marker.at, "bind": request.bind,
            "drawn": drawn.variant, "variant": candidate.performance.variant,
            "step": candidate.step, "why": "; ".join(reasons),
            "lead_s": round(fit.lead_s, 3), "base": fit.base,
            "stage": forecast.stage if forecast else None,
            "spread_s": round(forecast.spread, 3) if forecast else None,
            "parts": [w.clip.part for w in fitted],
        })  # fmt: skip

    # -- sending (each poll) ---------------------------------------------------------------------

    def _check(self, waiting: Waiting, now: float) -> Due | float | None:
        """A waiting clip now: its :class:`Due`, the seconds until it is looked at again, or None
        when it was dropped."""
        anchor_ = waiting.request.anchor
        if waiting.cue is None and not isinstance(anchor_, AtStream):
            if waiting.due <= now:
                self.reservations.unbook(id(waiting))
                return Due(waiting)
            return waiting.due - now
        if waiting.cue is None:
            return self._check_stream(waiting)
        fit, cue = waiting.fit, waiting.cue
        spoken = self.words.parts.get((cue.turn, cue.index))
        if fit is None or spoken is None:
            return None
        if not spoken.text:
            if not spoken.spoken:
                return self.timing.poll_s
            self.reservations.unbook(id(waiting))
            return Due(waiting)
        forecast = self._forecast(cue, now)
        if forecast is None:
            return None
        fit.heard.setdefault(forecast.stage, forecast.heard)
        aim = forecast.heard + self.timing.stroke_bias_s
        latency = self.latency(waiting.clip.part)
        send = aim - latency - fit.prep_s
        self._rebook(waiting, aim - fit.prep_s, aim)
        if now < send:
            return max(0.001, min(self.timing.poll_s, send - now))
        planned: Planned | None = None
        if waiting.clip.part == "arm" and self._arm_part() is not None:
            exact = self._plan_now(waiting)
            if isinstance(exact, str):
                self._late(waiting, f"the arm: {exact}")
                return None
            planned = exact
            if planned.stroke_s < fit.prep_s - self.timing.poll_s:  # it starts nearer: later
                fit.prep_s = planned.stroke_s
                return self.timing.poll_s
        prep = planned.stroke_s if planned is not None else fit.prep_s
        stroke = now + latency + prep
        allowed = self.timing.late_s if fit.bind == "utterance" else self.timing.stroke_late_s
        if stroke - aim > allowed + 1e-9:
            self._late(waiting, f"{stroke - aim:.2f} s late")
            return None
        if fit.bind == "utterance" and stroke > aim:
            end = self._speech_end(cue.turn, now)
            if end is not None and stroke > end:
                self._late(waiting, "past the speech")
                return None
        if planned is not None:
            arm = self._arm_part()  # where the arm really starts (a posture may have come since)
            if arm is not None and arm.holding() is not None:
                fit.base = "unit"
            elif fit.base == "unit":
                fit.base = "rest"
            waiting.clip = self._unit_end(waiting, now, planned)
        self.reservations.unbook(id(waiting))
        self._log_send(waiting, forecast, now, stroke, aim)
        timing = None
        if forecast.stream is not None:  # the player's clock runs: the send as its stream second
            played = self.playout.played()
            timing = (send - now + played, played)
        return Due(waiting, timing)

    def _check_stream(self, waiting: Waiting) -> Due | float | None:
        """A clip on a stream second (the pulse's): at its second, late by at most its bind's
        allowance."""
        target = self.target(waiting)
        assert target is not None
        played = self.playout.played()
        if played < target:
            return max(0.001, min(self.timing.poll_s, target - played))
        bind = waiting.request.bind
        if bind == "after":  # a tail: its stroke within after_s of its second
            late = self.timing.after_s - waiting.clip.stroke_s
        else:
            late = self.timing.late_s if bind == "utterance" else self.timing.stroke_late_s
        self.reservations.unbook(id(waiting))
        why = f"{played - target:.2f} s late" if played - target > late else None
        turn = waiting.turn
        if why is None and bind == "after" and turn is not None:
            end = self._speech_end(turn, self._now)
            stroke = self._now + self.latency(waiting.clip.part) + waiting.clip.stroke_s
            if end is not None and stroke > end:
                why = "past the speech"
        if why is not None:
            self.record("late", {"intent": waiting.performance.intent,
                                 "part": waiting.clip.part, "planned": round(target, 3),
                                 "played": round(played, 3), "why": why})  # fmt: skip
            return None
        return Due(waiting, (target, played))

    def _late(self, waiting: Waiting, why: str) -> None:
        """A word's clip that can no longer land in its window: dropped and logged, the rest of
        its composition keeping its time (for the arm: the floor)."""
        self.reservations.unbook(id(waiting))
        fit = waiting.fit
        self.record("late", {"intent": waiting.performance.intent, "part": waiting.clip.part,
                             "bind": fit.bind if fit else None, "why": why})  # fmt: skip

    def _plan_now(self, waiting: Waiting) -> Planned | str:
        """The arm's clip planned as the arm will play it now (from the pose a unit holds, else
        from rest by the newest state line); a stretch past the caps refused for a word."""
        arm = self._arm_part()
        fit = waiting.fit
        assert arm is not None and fit is not None
        held = arm.holding()
        if held is None and fit.held_only:
            return "no unit holds it any more"
        planned = arm.plan(waiting.clip, base=held, start=None if held is None else _knot(held))
        if isinstance(planned, str):
            return planned
        if fit.bind != "utterance" and planned.stretch > 1.0 + 1e-6:
            return f"{planned.stretch:.2f}x the caps"
        return planned

    def _unit_end(self, waiting: Waiting, now: float, planned: Planned) -> Clip:
        """The arm's clip as sent: held for the next stroke of its unit waiting to chain to it,
        or for the rest of a posture's hold that outlasts it, else back to rest; the unit's
        strokes and the arm's last pose kept."""
        arm = self._arm_part()
        assert arm is not None
        held = arm.holding()
        self._unit = self._unit + 1 if held is not None else 1
        ends = now + self.latency("arm") + planned.seconds
        nexts = [w.fit.start for w in self.waiting
                 if w.fit is not None and w.fit.chained == id(waiting)]  # fmt: skip
        if nexts and self._unit < self.timing.unit_strokes:
            hold = min(
                self.timing.unit_hold_s, max(POST_STROKE_S, min(nexts) - ends + UNIT_SLACK_S)
            )
            self._arm_last = None
            return replace(waiting.clip, then="hold", hold_s=round(hold, 3))
        turn = waiting.cue.turn if waiting.cue is not None else None
        if turn is not None and self._stays_up(turn):
            keep = arm.held_until - ends if held is not None else 0.0
            hold = keep if keep > POST_STROKE_S else self.timing.unit_hold_s
            speech = self._speech_end(turn, now)  # never far past the reply's sure speech
            if speech is not None:
                hold = min(hold, max(POST_STROKE_S, speech + SHORT_PAST_S - ends))
            self._arm_last = None
            return replace(waiting.clip, then="hold", hold_s=round(hold, 3))
        self._arm_last = (arm.final(planned), ends)
        return replace(waiting.clip, then="rest", hold_s=0.0)

    def _log_send(
        self, waiting: Waiting, forecast: Forecast, now: float, stroke: float, aim: float
    ) -> None:
        """The ``send`` row: the heard time by moment, the send and the planned stroke, seconds
        after the part's TEXT, and the stroke's error against its aim."""
        fit, cue = waiting.fit, waiting.cue
        assert fit is not None and cue is not None
        base = fit.text_s
        self.record("send", {
            "intent": waiting.performance.intent, "part": waiting.clip.part, "turn": cue.turn,
            "index": cue.index, "bind": fit.bind, "step": fit.step, "stage": forecast.stage,
            **{f"w_{stage}": round(t - base, 3) for stage, t in fit.heard.items()},
            "send": round(now - base, 3), "stroke": round(stroke - base, 3),
            "error_s": round(stroke - aim, 3), "base": fit.base,
            "then": waiting.clip.then, "hold_s": waiting.clip.hold_s,
        })  # fmt: skip

    # -- bookings --------------------------------------------------------------------------------

    def _book(self, waiting: Waiting, now: float) -> bool:
        """Promise the clip's part to it at render (the idle layer never books): False, logged,
        when the part is promised to one that goes first; the waiting clips it goes before are
        let go."""
        request, clip, fit = waiting.request, waiting.clip, waiting.fit
        if request.priority == Priority.IDLE:
            return True
        start, stroke, end = self._window(waiting, now)
        booking = Booking(clip.part, start, stroke, end, request.priority, request.rank,
                          waiting.performance.intent, id(waiting))  # fmt: skip
        if fit is not None and fit.chained is not None:  # a unit's next stroke follows its last
            self.reservations.booked.setdefault(clip.part, []).append(booking)
            return True
        first, displaced = self.reservations.book(booking)
        if first is not None:
            why = f"the {clip.part} is promised to {first.intent}"
            self.record("refused", {"intent": waiting.performance.intent, "part": clip.part,
                                    "why": why})  # fmt: skip
            return False
        for other in displaced:
            self.reservations.unbook(other.owner)
            for lost in [w for w in self.waiting if id(w) == other.owner]:
                self.waiting.remove(lost)
                self.record("displaced", {"intent": lost.performance.intent, "part": clip.part,
                                          "by": waiting.performance.intent})  # fmt: skip
        return True

    def _window(self, waiting: Waiting, now: float) -> tuple[float, float, float]:
        """When a waiting clip's part starts moving, its stroke and its end (engine clock): a
        word's from its fit, a stream second's on the stream, else from its due time."""
        fit, clip, anchor_ = waiting.fit, waiting.clip, waiting.request.anchor
        if fit is not None:
            return fit.start, fit.stroke, fit.end
        if isinstance(anchor_, AtStream):
            lag = self.timing.audio_lag_s
            start = self.words.wall(anchor_.second + _offset(waiting) + lag, now)
        else:
            start = max(now, waiting.due) + self.latency(clip.part)
        return start, start + clip.stroke_s, start + clip.duration_s

    def _rebook(self, waiting: Waiting, start: float, aim: float) -> None:
        """A waiting clip's window and booking moved with its word's forecast."""
        fit = waiting.fit
        if fit is None:
            return
        fit.end += start - fit.start
        fit.start, fit.stroke = start, aim
        for booking in self.reservations.booked.get(waiting.clip.part, []):
            if booking.owner == id(waiting):
                booking.start, booking.stroke, booking.end = fit.start, fit.stroke, fit.end

    # -- ahead of the text -----------------------------------------------------------------------

    def _ahead(
        self, rendering: Rendering, turn: int, index: int, text: str, marker: Marker
    ) -> None:
        """A marker of the token stream whose every version for its argument moves the arm for
        certain: when the arm is free and could not reach its stroke from rest once the part is
        queued, the arm sent to its home pose and held (the version drawn at TEXT then starts
        from there; a marker whose arm is left to chance, or whose place does not resolve,
        prepares nothing: a raise for nothing is noise). Each version is rendered for the
        marker's argument (a version written for another renders nothing and is passed)."""
        composition = rendering.book.compositions.get(marker.name)
        arm, home, now = self._arm_part(), rendering.book.poses.get(HOME), self._now
        if not isinstance(composition, Composition) or arm is None or home is None:
            return
        busy = self.reservations.holder("arm", now) is not None or arm.holding() is not None
        if busy or arm.ready() is not None or any(w.clip.part == "arm" for w in self.waiting):
            return
        share = min(1.0, marker.at / max(1, len(text)))
        cue = Cue(marker, turn, index, share, *anchor(text, marker.at))
        ahead = Request(marker.name, None, Priority.MARK, "marker", composition.bind)
        need = math.inf
        for k, variant in enumerate(composition.variants):
            if variant.floor:
                continue
            drawn = rendering.render(marker.name, self._scene(ahead), 0, cue, variant=k)
            if not drawn.clips:
                continue  # written for another argument
            if not any(rendering.book.atoms[s.atom].part == "arm" and s.p >= 1.0
                       for s in variant.steps):  # fmt: skip
                return
            clip = next((c for c in drawn.clips if c.part == "arm"), None)
            planned = arm.plan(clip, start=_knot(arm.base)) if clip is not None else "no arm"
            if clip is None or isinstance(planned, str):
                return
            least = prep_min_s(clip, arm.base, JOINTS, *arm.caps(PREP_CAPS))
            need = min(need, planned.stroke_s, least)
        if not math.isfinite(need):
            return
        lead, spread = self._ahead_lead(turn, index, text, marker.at, now)
        room = lead - AHEAD_S - self.latency("arm") - min(spread, MARGIN_S)
        if room >= need:  # the arm makes it from rest once the sentence is whole
            return
        home_clip = self._home_clip(arm, home)
        intent = f"{PREPARE}{marker.name}"
        request = Request(intent, None, Priority.MARK, "marker", clips=(home_clip,))
        waiting = Waiting(Performance(intent, 0, 0, (home_clip,)), home_clip, request, due=now)
        seconds = home_clip.duration_s
        start = now + self.latency("arm")
        waiting.fit = Fit("utterance", "prepare", "", lead, now, seconds, base="rest",
                          start=start, stroke=start + seconds, end=start + seconds,
                          final={**arm.base, **home})  # fmt: skip
        self.waiting.append(waiting)
        self._unit = 0
        self.record("prepared", {"intent": marker.name, "turn": turn, "part": index,
                                 "char": marker.at, "lead_s": round(lead, 3),
                                 "needs_s": round(need, 3),
                                 "pose": HOME, "hold_s": home_clip.hold_s})  # fmt: skip

    def _ahead_lead(
        self, turn: int, index: int, text: str, char: int, now: float
    ) -> tuple[float, float]:
        """Seconds until the aimed stroke of a character of a part not queued yet, were it
        queued now, and the forecast's spread (the word clock's TEXT law)."""
        knobs = self.words.knobs
        vowel = _VOWEL.search(text, char)
        into = knobs.lead_silence_s + (vowel.start() if vowel else char) * self.words.sigma()
        before = self.words.parts.get((turn, index - 1)) if index > 0 else None
        if before is None:
            start = now + knobs.tts_first_s
        else:
            wall, _, _ = self.words.start(turn, index - 1, now)
            start = max(now, wall + self.words.length(turn, index - 1))
        heard = start + into + knobs.audio_lag_s
        spread = min(SPREAD_INSIDE_S, SPREAD_PART_S + SPREAD_PER_S * into)
        return heard + self.timing.stroke_bias_s - now, spread

    def _home_clip(self, arm: ArmPart, home: Mapping[str, float]) -> Clip:
        """Rest to the home pose at :data:`PREP_CAPS` of the caps, held for a unit's hold."""
        return pose_clip(arm, arm.base, home, self.timing.unit_hold_s)

    # -- what the engine lent ----------------------------------------------------------------------

    def _rendering(self) -> Rendering | None:
        return self._renderer() if self._renderer is not None else None

    def _stays_up(self, turn: int) -> bool:
        try:
            return self.stay_up(turn)
        except Exception:  # a listener's hook never breaks a send
            return False

    def _composition(self, intent: str) -> Composition | None:
        rendering = self._rendering()
        found = rendering.book.compositions.get(intent) if rendering is not None else None
        return found if isinstance(found, Composition) else None

    def _held_only(self, performance: Performance) -> bool:
        """Whether the performance's variant needs the arm up (the accent's flick): its arm
        plays only from the pose a unit holds."""
        composition = self._composition(performance.intent)
        if composition is None or not 0 <= performance.variant < len(composition.variants):
            return False
        return composition.variants[performance.variant].needs.get("arm") == "up"

    def _arm_part(self) -> ArmPart | None:
        arm = self.parts.get("arm")
        return arm if isinstance(arm, ArmPart) else None

    def _scene(self, request: Request) -> Scene:
        """What a rendering of another variant may use: the parts ready now, refined by the
        engine's scene hooks for ``request``."""
        ready = frozenset(p for p, part in self.parts.items() if part.ready() is None)
        scene = Scene(ready, speaking=True, conversation=True)
        for hook in self._hooks:
            scene = hook(scene, request)
        return scene


def pose_clip(
    arm: ArmPart,
    start: Mapping[str, float],
    pose: Mapping[str, float],
    hold_s: float,
    share: float = PREP_CAPS,
) -> Clip:
    """The arm from ``start`` to ``pose`` (absolute tracks of the joints ``pose`` names) at
    ``share`` of its caps (the book's cubic: 1.5 d / T, 6 d / T^2; 0.05 s at least), then
    held ``hold_s``: a preparation's way home, the listeners' posture and its renewal."""
    speed, acc = arm.caps(share)
    joints = [j for j in JOINTS if j in pose]
    begin = {**arm.base, **start}
    travel = max(abs(pose[j] - begin[j]) for j in joints)
    seconds = max(1.5 * travel / speed, math.sqrt(6.0 * travel / acc), 0.05)
    tracks = tuple(Track(j, "absolute", (0.0, seconds), (begin[j], pose[j])) for j in joints)
    return Clip("arm", 0.0, seconds, seconds, tracks, then="hold", hold_s=hold_s)


def _never(turn: int) -> bool:
    """No posture: the arm rests after a reply's last stroke."""
    return False


def _tail(waiting: Waiting) -> bool:
    """Whether a clip is a tail: bound ``after`` its stream second."""
    return waiting.request.bind == "after" and isinstance(waiting.request.anchor, AtStream)


def _offset(waiting: Waiting) -> float:
    """Where a clip on a stream second starts from it: its offset, a tail's at the second."""
    return 0.0 if _tail(waiting) else waiting.clip.offset_s


def _word_bind(bind: RequestBind) -> Bind:
    """A word's binding (a tail on a word is fitted as an utterance's)."""
    return "utterance" if bind == "after" else bind


def _without(performance: Performance, *parts: PartName) -> tuple[Clip, ...]:
    """The performance's clips but those of ``parts``."""
    return tuple(c for c in performance.clips if c.part not in parts)


def _knot(pose: Mapping[str, float]) -> tuple[float, ...]:
    """A pose as one angle per joint."""
    return tuple(pose[j] for j in JOINTS)


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
