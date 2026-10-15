"""The gesture engine: the body's gesture and idle layers, driven by the conversation.

It listens to the cascade (:class:`pepin.expression.conversation.ConversationObserver`) and is
the ``gestures`` switch (:class:`pepin.expression.conversation.LiveFlag`). Everything that moves
the body expressively is a :class:`pepin.expression.schedule.Request`: a marker of a part queued
for the voice (anchored on its word), a moment of the robot, a person's ask (now), the engine's
listeners' (:meth:`GestureEngine.request`, from any thread). Each is rendered at once (before
the voice has made a sound of a marker's word, so a preparation has time) and handed to the
scheduler (:class:`pepin.expression.schedule.Scheduler`), which keeps each clip until its time
on the player's clock:

    the word's start + the clip's offset from it + audio_lag_s - the part's latency_s,

or at once when that is past by at most ``late_s`` (later still, it is dropped and logged). An
arm clip is stretched by ``1 / speed_scale`` first, so its stroke still lands on the word; the
arm's plan says when the stroke really lands (the board's approach and the caps' stretch). A
tool or the command line performs now (:meth:`perform`).

Layers, per part: one request holds a part at a time, by the priority table of the scheduler's
reservations (state, attention and a person's ask first; then the markers, the moments, the
pulse, the idle layer); the person starting to speak fades the gesture layer of every part
(``fade_s`` per part) unless a greeting or a farewell already past its stroke ends within
``finish_s``; the conversation's close does the same and rests the rest; a drive sends every
part to rest and drops what waits; a cut reply drops its clips. An arm UNIT the board holds
(``then: hold``: a mark's next stroke due, the listeners' posture) goes to rest when the person
starts to speak, when its reply is cut or done, and when the body's turn asks for the arm,
unless a clip still moves the arm (its own end decides). The idle layer plays an atom of
the book's ``idle`` entries every ``every_s`` on a part no gesture holds, unless a state the
entry excludes holds (by default either speaker: idle only while nobody speaks), from the
entry's ``after_speech_s`` (else ``idle_after_speech_s``) after those states ended and
``idle_resume_s`` after a gesture. A looping idle clip (the breath) holds its part until a
state it excludes or another clip takes it, renewed at its owner once a period (the arbiter
lets an overlay lapse a second after its period); the other idle entries of that part wait.

Attention comes first (:class:`pepin.expression.speaker.AttentionView`, the speaker gaze's):
a gesture overlay takes the head only while attention tracks a person seen within
``head_seen_s`` (1.2 s: frames come at 1 Hz while the robot speaks), and is withdrawn when
attention starts a seek, a check or a body turn; no arm motion starts while the person stands
past the forward arc or the body turns, and the gaze's turn asks for the arm first
(:meth:`release_arm`).

The gesture space (:attr:`GestureEngine.resolver`, one :class:`pepin.expression.render.Resolver`
for the engine's life, lent to every renderer it makes, the one a switch on makes too): the
person's bearing from the same attention view, the room's places from its ``room`` (the voice
loop gives it the goal server's; none, or one never read: a place's name gets its floor), the
catchments forgotten at the conversation's close; its ``catchment`` and ``repeat`` rows go to
the engine's record and to the conversation's events.jsonl (:attr:`GestureEngine.journal`).

Caps (config/expression.json ``caps``) hold the markers and the moments back, never a tool or
the command line, which a person asked for: per part (a sentence or a clause of one), per
reply, per minute, arm gestures per minute, an intent's own per minute (``laugh``: 1, waived
while the person's own laugh is fresh), and each composition's ``cooldown_s`` per mark and
argument. The accent (``{!}``) is outside those counts, one a part; a series (``count``) counts
once a reply, its later items passing every count. Arm gestures are counted on the final
version (the ladder's: a mark that fell to its head-and-face floor used no arm).

A requester ends its own move with :meth:`release` (what that very request holds fades, what
it still waits for is dropped), and sees what waits with :meth:`pending` (each clip's part,
place in the table and window: the listeners keep their nods off a mark's stroke).

One thread does all of it, waking only for what waits: every ``poll_s`` while a clip waits for
its word, at a performance's next clip, at the next idle motion or renewal, every
``ATTENTION_POLL_S`` while a gesture holds the head, and at least every ``MAX_WAIT_S``; the
cascade's calls only post to it and never wait. Every performance is logged with its seed and
variant (a replay is ``scripts/gesture.py play NAME --seed N --variant V``), every clip sent
with the stream second it was planned for and the one it went at.

Owners (gesticulation DESIGN.md §11): the contract's file. A plugs in through
:meth:`GestureEngine.request`, :meth:`GestureEngine.release`, :meth:`GestureEngine.pending`,
:attr:`GestureEngine.taps`, :attr:`GestureEngine.scene_hooks` and the book; B through
:mod:`pepin.expression.schedule`, :mod:`pepin.expression.render`, :mod:`pepin.expression.parts`
and :attr:`GestureEngine.scene_hooks`.
"""

from __future__ import annotations

import concurrent.futures
import logging
import math
import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pepin.expression.animation import (
    PARTS,
    Book,
    Clip,
    Composition,
    Event,
    Idle,
    IdleState,
    PartName,
    Performance,
    Scene,
)
from pepin.expression.book import vocabulary
from pepin.expression.breath import ArmBreath
from pepin.expression.conversation import (
    GESTURES,
    AudioTap,
    EventSink,
    Heard,
    VoiceChunk,
    moment_rank,
)
from pepin.expression.executors import Executor, Outcome, Renewing
from pepin.expression.markers import Cue, Part, Playout, cues, gesture
from pepin.expression.mix import stretched
from pepin.expression.parts import GENTLE_SHARE, ArmPart, HeadPart
from pepin.expression.render import BookRenderer, Resolver, WeightedChooser
from pepin.expression.schedule import (
    NOT_YET,
    AtWord,
    Pending,
    Priority,
    Request,
    Reservation,
    Scheduler,
    Timing,
    Waiting,
    timing_from,
)
from pepin.expression.speaker import (
    HEAD_SEEN_S,
    NO_ATTENTION,
    Attention,
    AttentionView,
    arm_refusal,
    head_refusal,
)
from pepin.expression.trajectory import JOINTS

logger = logging.getLogger(__name__)

Record = Callable[[str, dict[str, Any]], None]
SceneHook = Callable[[Scene, Request], Scene]  # what a rendering may use, refined (lead, quiet)
TOOL_WAIT_S = 2.0  # a tool's perform waits this long for the engine's thread
ATTENTION_POLL_S = 0.1  # attention is read this often while a gesture holds the head
ARM_REST_S = 0.9  # the board's return to rest from the highest pose (raised: 0.87 at the caps)
BUSY = ("seek", "check", "turning")  # attention's phases that take the head from a gesture
MAX_WAIT_S = 1.0  # the thread wakes at least this often (a longer wait, or an endless one, is cut)
RENEW_MIN_S = 0.1  # a looping clip is renewed at most this often


@dataclass(frozen=True)
class Caps:
    """How often the markers and the moments may move the body (a tool's ask is never capped):
    the accents (``{!}``) outside the counts, at most ``accent_per_part`` a part; a series'
    later items (``count``) in a reply counted once, as its first."""

    per_part: int = 1
    per_reply: int = 3
    per_minute: int = 6
    arm_per_minute: int = 3
    intent_per_minute: Mapping[str, int] = field(default_factory=lambda: {"laugh": 1})
    laughter: str = r"(?:х[аеи][- ]?){2,}|ахах|хих|haha"  # noqa: RUF001
    laughter_waives: tuple[str, ...] = ("laugh",)
    laughter_s: float = 30.0
    accent_per_part: int = 1
    accents: tuple[str, ...] = ("accent",)
    series: tuple[str, ...] = ("count",)


Tally = Literal["counted", "accent", "series"]  # how a capped request counts (Caps)


@dataclass(frozen=True)
class EngineConfig:
    """config/expression.json's knobs as the engine reads them (switching ``gestures`` on reads
    them again)."""

    audio_lag_s: float = 0.15
    late_s: float = 0.5
    poll_s: float = 0.02
    fade_s: Mapping[str, float] = field(
        default_factory=lambda: {"arm": 0.35, "head": 0.35, "face": 0.1, "body": 0.35}
    )
    idle_resume_s: float = 2.0
    idle_after_speech_s: float = 5.0
    finish_s: float = 1.0
    finish: tuple[str, ...] = ("greeting", "farewell")
    seed: int | None = None
    face_hold_s: float = 2.0
    arm_speed_scale: float = 0.2
    body_speed_scale: float = 0.4
    head_seen_s: float = HEAD_SEEN_S
    arm_past_arc: bool = True  # the arm waits while the person stands past the forward arc
    arm_caps_share: float = GENTLE_SHARE  # the posture's raise and its gentle way down
    caps: Caps = field(default_factory=Caps)
    timing: Timing = field(default_factory=Timing)  # the scheduler's (``timing``, ``units``)


def engine_config(data: Mapping[str, Any]) -> EngineConfig:
    """config/expression.json's object as the engine's knobs (defaults for what it lacks)."""
    timing = data.get("timing", {})
    layers = data.get("layers", {})
    caps = data.get("caps", {})
    arm = data.get("parts", {}).get("arm", {})
    d, c = EngineConfig(), Caps()
    fades = layers.get("fade_s", d.fade_s)
    if not isinstance(fades, dict):
        fades = dict.fromkeys(PARTS, float(fades))
    seed = layers.get("seed")
    return EngineConfig(
        audio_lag_s=float(timing.get("audio_lag_s", d.audio_lag_s)),
        late_s=float(timing.get("late_s", d.late_s)),
        poll_s=float(timing.get("poll_s", d.poll_s)),
        fade_s={str(k): float(v) for k, v in fades.items() if k != "note"},
        idle_resume_s=float(layers.get("idle_resume_s", d.idle_resume_s)),
        idle_after_speech_s=float(layers.get("idle_after_speech_s", d.idle_after_speech_s)),
        finish_s=float(layers.get("finish_s", d.finish_s)),
        finish=tuple(str(x) for x in layers.get("finish", d.finish)),
        seed=None if seed is None else int(seed),
        face_hold_s=float(layers.get("face_hold_s", d.face_hold_s)),
        arm_speed_scale=float(arm.get("speed_scale", d.arm_speed_scale)),
        body_speed_scale=float(data.get("wheels", {}).get("speed_scale", d.body_speed_scale)),
        head_seen_s=float(layers.get("head_seen_s", d.head_seen_s)),
        arm_past_arc=bool(layers.get("arm_past_arc", d.arm_past_arc)),
        arm_caps_share=float(data.get("pulse", {}).get("arm_up_caps_share", d.arm_caps_share)),
        caps=Caps(
            per_part=int(caps.get("per_part", c.per_part)),
            per_reply=int(caps.get("per_reply", c.per_reply)),
            per_minute=int(caps.get("per_minute", c.per_minute)),
            arm_per_minute=int(caps.get("arm_per_minute", c.arm_per_minute)),
            intent_per_minute={
                str(k): int(v)
                for k, v in caps.get("intent_per_minute", c.intent_per_minute).items()
            },
            laughter=str(caps.get("laughter", c.laughter)),
            laughter_waives=tuple(caps.get("laughter_waives", c.laughter_waives)),
            laughter_s=float(caps.get("laughter_s", c.laughter_s)),
            accent_per_part=int(caps.get("accent_per_part", c.accent_per_part)),
            accents=tuple(str(x) for x in caps.get("accents", c.accents)),
            series=tuple(str(x) for x in caps.get("series", c.series)),
        ),
        timing=timing_from(data),
    )


class GestureEngine:
    """The animator (:class:`pepin.expression.executors.Animator`), a conversation observer, the
    voice's audio tap and the ``gestures`` live flag in one."""

    def __init__(
        self,
        parts: Mapping[PartName, Executor],
        playout: Playout,
        load: Callable[[], tuple[Book, EngineConfig]],
        *,
        on: bool = False,
        clock: Callable[[], float] = time.monotonic,
        record: Record | None = None,
        seeds: random.Random | None = None,
        attention: AttentionView | None = None,
    ) -> None:
        """``load`` reads the book (with the arm's rest pose) and the knobs, now and at every
        switch on; ``record(kind, fields)`` hears every performance, cue and refusal;
        ``attention``: the speaker gaze's view (None: no attention to keep clear), the
        resolver's person too."""
        self.parts = dict(parts)
        self.playout = playout
        self.load = load
        self.clock = clock
        self.record = record or (lambda kind, fields: logger.info("%s %s", kind, fields))
        self.seeds = seeds or random.Random()
        self.journal: EventSink | None = None  # the conversation's events.jsonl, once one runs
        self.resolver = Resolver(attention=attention, record=self._meaning)
        self._attention = attention
        self.taps: list[AudioTap] = []  # the voice's chunks, on the voice's thread (prosody)
        # each rendering's scene refined, in order: the arm's place first (the scheduler's)
        self.scene_hooks: list[SceneHook] = [self._arm_scene]
        self.scheduler = Scheduler(
            playout, self._latency, lambda kind, fields: self.record(kind, fields)
        )
        self.book, self.config = load()
        self.renderer = BookRenderer(self.book, WeightedChooser(), self.resolver)
        self._apply_knobs()
        self._on = on
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._inbox: deque[Callable[[], None]] = deque()
        self._thread: threading.Thread | None = None
        self._stop = False
        self._per_part: dict[tuple[int, int], int] = {}
        self._per_reply: dict[int, int] = {}
        self._accents: dict[tuple[int, int], int] = {}  # accents placed in each part
        self._series: dict[int, set[str]] = {}  # each reply's series counted (their first)
        # (when, intent, with the arm, counted under per_minute and arm_per_minute)
        self._started: deque[tuple[float, str, bool, bool]] = deque()
        self._used: dict[str, float] = {}  # a mark (intent, or intent:arg) -> when it last played
        self._laughed = float("-inf")
        self._last: Performance | None = None
        self._conversation = False
        self._driving = False
        self._listening = False  # the person is speaking
        self._replying: set[int] = set()  # turns whose parts are with the voice
        self._opened_s = float("-inf")
        self._ended: dict[IdleState, float] = {"speaking": float("-inf"),
                                               "listening": float("-inf")}  # fmt: skip
        self._idle_next: dict[int, float] = {}  # each idle entry's next motion (by its place)
        self._idle_of: dict[PartName, Idle] = {}  # the idle entry a part's idle clip came from
        self._arm_free_s = float("-inf")  # the arm was released to the body's turn until then
        self._held_turn: int | None = None  # the reply of the clip the arm holds (None: any)
        self._renew_at: dict[PartName, float] = {}  # a looping clip's next renewal

    @property
    def _playing(self) -> dict[PartName, Reservation]:
        """What holds each part (the scheduler's reservations)."""
        return self.scheduler.reservations.held

    @property
    def attention(self) -> AttentionView | None:
        """The speaker gaze's view (None: no attention to keep clear, the person ahead)."""
        return self._attention

    @attention.setter
    def attention(self, view: AttentionView | None) -> None:
        """The engine's and its resolver's view at once."""
        self._attention = view
        self.resolver.attention = view

    # -- the live flag ---------------------------------------------------------------------------

    @property
    def flag(self) -> str:
        """``gestures``."""
        return GESTURES

    @property
    def on(self) -> bool:
        """Whether gestures play."""
        return self._on

    def switch(self, on: bool) -> str | None:
        """On: the book and the knobs read again (refused, staying off, when they do not
        load); off: every part to rest, nothing waits. None when applied."""
        if on:
            try:
                book, config = self.load()
            except (OSError, ValueError, KeyError) as error:
                return f"the gesture book did not load: {error}"
            with self._lock:  # the engine's thread holds it for a whole step
                self.book, self.config = book, config
                self.renderer = BookRenderer(book, self.renderer.chooser, self.resolver)
                self._apply_knobs()
                self._on = True
                self._idle_next.clear()
                self._wake.notify()
            return None
        with self._lock:
            self._on = False
        self._post(self._rest_all)
        return None

    # -- the animator ----------------------------------------------------------------------------

    def perform(self, intent: str, *, source: str) -> Outcome:
        """Perform ``intent`` now (a tool, the command line): a composition's name, or a mark
        as the brain writes one, its argument after a colon and the book's other spellings
        (``point:printer``, ``count:second``, ``!``); refused while off, during a drive, for
        what the book does not hold, or when no part can play."""
        if self._thread is None or threading.current_thread() is self._thread:
            return self._perform_now(intent, source)
        future: concurrent.futures.Future[Outcome] = concurrent.futures.Future()

        def run() -> None:
            future.set_result(self._perform_now(intent, source))

        self._post(run)
        try:
            return future.result(TOOL_WAIT_S)
        except concurrent.futures.TimeoutError:
            return Outcome(False, "the gesture engine did not answer in time")

    def request(self, request: Request) -> None:
        """Perform ``request`` at its anchor (any thread; the engine's listeners): rendered and
        capped on the engine's thread, a marker's and a moment's under the caps, a person's ask
        and the pulse's and idle layer's under none here."""

        def run() -> None:
            self._request(request)

        self._post(run)

    def release(self, request: Request, why: str) -> None:
        """Let go of what ``request`` (this very one, as posted) holds and still waits for (any
        thread): its requester ends its own held move (a thinking glance when the reply
        comes); the parts it holds fade back over ``fade_s``."""
        self._post(lambda: self._release(request, why))

    def pending(self, part: PartName | None = None, turn: int | None = None) -> list[Pending]:
        """What waits for its time (any thread; a snapshot), of ``part`` and of reply ``turn``
        when given, each with its window on the engine's clock."""
        with self._lock:
            return self.scheduler.pending(part, turn, self.clock())

    def hush(self, why: str) -> None:
        """The gesture layer of every part let go (the person speaks)."""
        self._post(lambda: self._hush(why))

    def last(self) -> Performance | None:
        """The latest performance started."""
        return self._last

    def release_arm(self, why: str) -> float:
        """The body's motion asks for the arm (any thread: the speaker gaze's turn, the laugh's
        rock): a gesture's arm, moving or held in a unit, goes to rest now and none starts for
        as long as it takes; the seconds until it is there."""
        with self._lock:
            now = self.clock()
            part = self.parts.get("arm")
            held = self._playing.get("arm")
            moving = held is not None and now < held.ends
            if part is None or not (moving or _holds(part)):
                return 0.0
            rest_s = getattr(part, "rest_s", None)
            seconds = (rest_s() if callable(rest_s) else ARM_REST_S) + part.latency_s
            part.fade("gesture", self.config.fade_s.get("arm", 0.0))
            self.scheduler.reservations.release("arm")
            self._held_turn = None
            self._arm_free_s = now + seconds
            latest = held if moving else self.scheduler.reservations.last("arm")
            intent = latest.performance.intent if latest is not None else "unit"
            self.record("faded", {"intent": intent, "part": "arm", "why": why,
                                  "rest_s": round(seconds, 2)})  # fmt: skip
            return seconds

    def audio(self, chunk: VoiceChunk) -> None:
        """A chunk of the reply's audio (the voice's thread): to every tap, nothing else here."""
        for tap in self.taps:
            try:
                tap.audio(chunk)
            except Exception:
                logger.exception("gesture engine: an audio tap broke")

    # -- the conversation ------------------------------------------------------------------------

    def opened(self, wake: Heard) -> None:
        """A conversation: the idle layer may start once everyone is quiet."""
        self._post(self._opened)

    def speech_started(self, t: float) -> None:
        """The person speaks: the gesture layer fades, the idle layer stops."""
        self._post(self._speech_started)

    def speech_ended(self, heard: Heard) -> None:
        """The person stopped; a laugh of theirs waives the laugh's cap for a while."""
        self._post(lambda: self._speech_ended(heard))

    def part_queued(self, turn: int, index: int, part: Part) -> None:
        """A part went to the voice: its markers rendered and waiting for their words."""
        self._post(lambda: self._part_queued(turn, index, part))

    def voice_started(self, turn: int) -> None:
        """Nothing: the parts carry their own marks."""

    def part_started(self, turn: int, index: int, stream_s: float) -> None:
        """A part's first audio is with the player: to the scheduler."""
        self._post(lambda: self.scheduler.part_started(turn, index, stream_s))

    def part_spoken(self, turn: int, index: int) -> None:
        """All of that part's audio is with the player: its length is known."""
        self._post(lambda: self.scheduler.part_spoken(turn, index))

    def text_ahead(self, turn: int, sentence: int, text: str) -> None:
        """A sentence before it is whole: to the scheduler."""
        self._post(lambda: self.scheduler.text_ahead(turn, sentence, text))

    def reply_cut(self, turn: int) -> None:
        """The reply stopped short: what waits for its words is dropped, the arm's unit rests."""
        self._post(lambda: self._drop_turns(turn, "cut"))

    def reply_done(self, turn: int) -> None:
        """The reply played out: whatever still waited for it is dropped, its unit rests."""
        self._post(lambda: self._drop_turns(turn, "done", only=True))

    def moment(self, name: str, text: str = "") -> None:
        """A moment of the robot the book maps to an intent (``slow_tool`` -> thinking); the
        person's words at a pause are the listeners'."""
        self._post(lambda: self._moment(name))

    def drive(self, active: bool, ran: bool = True) -> None:
        """A drive: every part to rest and nothing new until it ends (run or not)."""
        self._post(lambda: self._drive(active))

    def closed(self, reason: str) -> None:
        """The conversation is over: every part to rest but a greeting or a farewell finishing,
        nothing waits."""
        self._post(self._closed)

    # -- the thread ------------------------------------------------------------------------------

    def start(self) -> GestureEngine:
        """Run the engine's thread; returns self so it chains."""
        self._thread = threading.Thread(target=self._run, name="gestures", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop the thread (the parts are left as they are: the board rests the arm itself)."""
        with self._lock:
            self._stop = True
            self._wake.notify()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def step(self) -> float | None:
        """One turn of the engine: the calls posted since the last one, the clips due, the idle
        layer, attention. The seconds until it has something to do again (None: until a call)."""
        with self._lock:
            while self._inbox:
                call = self._inbox.popleft()
                try:
                    call()
                except Exception:
                    logger.exception("gesture engine: a call broke")
            waits = [self._fire_due(), self._idle(), self._attend(), self._renew()]
        known = [w for w in waits if w is not None]
        return min(known) if known else None

    def _run(self) -> None:
        while True:
            wait = self.step()
            with self._lock:
                if self._stop:
                    return
                if not self._inbox:
                    self._wake.wait(bounded(wait))

    def _post(self, call: Callable[[], None]) -> None:
        with self._lock:
            self._inbox.append(call)
            self._wake.notify()

    # -- what the calls do (on the engine's thread) ----------------------------------------------

    def _opened(self) -> None:
        self._conversation = True
        self._opened_s = self.clock()
        self._idle_next.clear()

    def _speech_started(self) -> None:
        self._listening = True
        self._hush("the person speaks")
        for part, playing in list(self._playing.items()):
            entry = self._idle_of.get(part)
            if playing.clip.layer == "idle" and (entry is None or "listening" in entry.unless):
                self.parts[part].fade("idle", self.config.fade_s.get(part, 0.0))
                self.scheduler.reservations.release(part)

    def _speech_ended(self, heard: Heard) -> None:
        self._listening = False
        self._ended["listening"] = self.clock()
        if heard.text and re.search(self.config.caps.laughter, heard.text.lower()):
            self._laughed = self.clock()

    def _part_queued(self, turn: int, index: int, part: Part) -> None:
        self._replying.add(turn)
        self.scheduler.part_queued(turn, index, part.text)
        if not self._on:
            return
        for cue in cues(turn, index, part):
            marker = cue.marker
            at = AtWord(turn, index, marker.at)
            if marker.channel == "face":
                hold = self.config.face_hold_s
                face = Clip("face", 0.0, 0.0, hold, events=(Event(0.0, marker.name, hold),))
                request = Request(f"face:{marker.name}", at, Priority.MARK, "marker", clips=(face,))
                self._request(request, cue)
                continue
            request = Request(marker.name, at, Priority.MARK, "marker")
            composition = self.book.compositions.get(marker.name)
            if composition is not None:
                request = replace(request, bind=composition.bind, family=composition.family)
                arg = styled_arg(composition, marker.arg, part.style)
                if arg != marker.arg:  # the part's style chose the bare mark's argument
                    cue = replace(cue, marker=replace(marker, arg=arg))
            self._request(request, cue)

    def _drop_turns(self, turn: int, why: str, only: bool = False) -> None:
        self.scheduler.drop(turn, why, only=only)
        self._rest_unit(f"the reply is {why}", turn, only=only, gently=True)
        self._replying = {t for t in self._replying if not (t == turn if only else t <= turn)}
        self._ended["speaking"] = self.clock()

    def _moment(self, name: str) -> None:
        intent = self.book.moments.get(name)
        if intent is None or not self._on:
            return
        request = Request(intent, None, Priority.MOMENT, "moment", rank=moment_rank(name))
        composition = self.book.compositions.get(intent)
        if composition is not None:
            request = replace(request, bind=composition.bind, family=composition.family)
        self._request(request, moment=name)

    def _drive(self, active: bool) -> None:
        self._driving = active
        if active and (self._on or self._playing):
            self._rest_all()

    def _closed(self) -> None:
        self._conversation = False
        self._listening = False
        self._replying.clear()
        self.scheduler.spoken.clear()
        self.resolver.forget()
        if self._on or self._playing:  # off and idle: no line to any owner
            self._rest_all(finishing=True)

    def _rest_all(self, finishing: bool = False) -> None:
        """Every part to rest and nothing waits; ``finishing``: a greeting or a farewell past its
        stroke and ending within ``finish_s`` plays out on its parts."""
        now = self.clock()
        kept = {p: r for p, r in self._playing.items() if finishing and self._finishing(r, now)}
        self.scheduler.clear()
        self._held_turn = None
        for name, held in kept.items():
            self.scheduler.reservations.reserve(name, held, now)
        for name, part in self.parts.items():
            if name in kept:
                continue
            try:
                part.rest()
            except Exception:
                logger.exception("gesture engine: %s would not rest", part.part)

    def _finishing(self, playing: Reservation, now: float) -> bool:
        """A greeting or a farewell past its stroke, ending within ``finish_s``: it plays out."""
        return (
            playing.performance.intent in self.config.finish
            and now >= playing.stroke
            and playing.ends - now <= self.config.finish_s
        )

    def _hush(self, why: str) -> None:
        now = self.clock()
        for part, playing in list(self._playing.items()):
            if playing.clip.layer != "gesture" or now >= playing.ends:
                continue
            if self._finishing(playing, now):
                continue
            self.parts[part].fade("gesture", self.config.fade_s.get(part, 0.0))
            self.scheduler.reservations.release(part)
            self.record("faded", {"intent": playing.performance.intent, "part": part, "why": why})
        self._rest_unit(why)

    def _rest_unit(
        self, why: str, turn: int | None = None, *, only: bool = False, gently: bool = False
    ) -> bool:
        """The arm's held unit (of reply ``turn`` and those before it, ``only`` that one, when
        given) to rest, unless a clip still moves the arm; True when it went. ``gently``: the
        reply's end, where nothing hurries: the arm's own slow way down
        (:meth:`pepin.expression.parts.ArmPart.lower`), quiet like the raise; the person's start
        and the body's turn keep the board's quick way (the arm clears out)."""
        part = self.parts.get("arm")
        if part is None or not _holds(part):
            return False
        held_turn = self._held_turn
        if turn is not None and held_turn is not None:
            ours = held_turn == turn if only else held_turn <= turn
            if not ours:
                return False
        now = self.clock()
        playing = self._playing.get("arm")
        if playing is not None and now < playing.ends:
            return False
        lowered = part.lower() if gently and isinstance(part, ArmPart) else None
        if lowered is None:
            part.fade("gesture", self.config.fade_s.get("arm", 0.0))
        self.scheduler.reservations.release("arm")
        self._held_turn = None
        latest = self.scheduler.reservations.last("arm")
        self.record("faded", {"intent": latest.performance.intent if latest else "unit",
                              "part": "arm", "why": why,
                              **({"down_s": round(lowered, 2)} if lowered else {})})  # fmt: skip
        return True

    def _release(self, request: Request, why: str) -> None:
        for waiting in self.scheduler.withdraw(request):
            self.record("dropped", {"intent": waiting.performance.intent,
                                    "part": waiting.clip.part, "why": why})  # fmt: skip
        now = self.clock()
        for part, held in list(self._playing.items()):
            if held.request is not request or now >= held.ends:
                continue
            self.parts[part].fade(held.clip.layer, self.config.fade_s.get(part, 0.0))
            self.scheduler.reservations.release(part)
            self.record("faded", {"intent": held.performance.intent, "part": part, "why": why})

    # -- performing ------------------------------------------------------------------------------

    def _perform_now(self, intent: str, source: str) -> Outcome:
        with self._lock:
            if not self._on:
                return Outcome(
                    False, "gestures are off (scripts/voice_cascade.py switch --flag gestures=on)"
                )
            if self._driving:
                return Outcome(False, "a drive is under way: the body is the drive's")
            asked = self._asked(intent)
            if asked is None:
                return Outcome(
                    False,
                    f"no gesture {intent!r}; the gestures are {', '.join(self.book.compositions)}"
                    " (an argument after a colon: point:printer)",
                )
            name, arg = asked
            request = Request(name, None, Priority.ASK, source)
            performance = self._render(name, source, None, request, arg=arg)
            if performance is None:
                return Outcome(False, "no part can play it now")
            outcomes = self._place(request, performance, None, rendered=True)
        played = [o for o in outcomes if o.ok]
        if not played:
            return Outcome(False, "; ".join(o.why for o in outcomes) or "nothing played")
        seconds = max(c.offset_s + c.duration_s for c in performance.clips) - min(
            c.offset_s for c in performance.clips
        )
        return Outcome(True, f"seed {performance.seed}, variant {performance.variant}", seconds)

    def _asked(self, intent: str) -> tuple[str, str | None] | None:
        """A person's ask as a composition and its argument: a composition's name as it is
        (its first argument when it takes one), else read as a marker's inside is."""
        if intent in self.book.compositions:
            return intent, None
        named = gesture(intent, vocabulary(self.book, frozenset()))
        return None if named is None else (named[1], named[2])

    def _request(
        self, request: Request, cue: Cue | None = None, *, moment: str | None = None
    ) -> list[Outcome]:
        """``request`` capped, rendered and placed (on the engine's thread); a word's anchor
        becomes a cue on its part. Markers and moments are capped (and logged ``capped``); a
        tool's, the pulse's, the idle layer's and the engine's own compositions (the book's
        callable-false ensembles, the listeners' by name) are not, and count under none."""
        if not self._on:
            return []
        if cue is None and isinstance(request.anchor, AtWord):
            cue = self.scheduler.cue(request.anchor, request.intent)
            if cue is None:
                self.record("unplaced", {"intent": request.intent, "source": request.source,
                                         "why": "its part is not with the voice"})  # fmt: skip
                return []
        turn, index = (cue.turn, cue.index) if cue is not None else (None, None)
        composition = self.book.compositions.get(request.intent)
        own = composition is not None and not composition.callable
        capped = (
            request.priority in (Priority.MARK, Priority.MOMENT) and not request.clips and not own
        )
        tally = self._tally(request.intent, turn) if capped else "counted"
        if capped:
            refusal = self._capped(request.intent, cue, tally)
            if refusal is not None:
                where = {"moment": moment} if moment is not None else {"turn": turn, "part": index}
                self.record("capped", {"intent": request.intent, **where, "why": refusal})
                return []
        if request.clips:
            performance = Performance(request.intent, 0, 0, request.clips, cue)
        else:
            rendered = self._render(request.intent, request.source, cue, request,
                                    counted=capped and tally == "counted")  # fmt: skip
            if rendered is None:
                return []
            performance = rendered
        if capped and turn is not None and index is not None:
            if tally == "accent":
                self._accents[(turn, index)] = self._accents.get((turn, index), 0) + 1
            elif tally == "counted":
                self._per_part[(turn, index)] = self._per_part.get((turn, index), 0) + 1
                self._per_reply[turn] = self._per_reply.get(turn, 0) + 1
                if request.intent in self.config.caps.series:
                    self._series.setdefault(turn, set()).add(request.intent)
        return self._place(request, performance, cue, rendered=not request.clips)

    def _tally(self, intent: str, turn: int | None) -> Tally:
        """How a marker's or a moment's ``intent`` counts under the caps: an accent outside
        them, a series' later item in its reply as none, anything else as itself."""
        caps = self.config.caps
        if intent in caps.accents:
            return "accent"
        if turn is not None and intent in self._series.get(turn, ()):
            return "series"
        return "counted"

    def _place(
        self, request: Request, performance: Performance, cue: Cue | None, *, rendered: bool
    ) -> list[Outcome]:
        """The performance's clips to the scheduler (a word's fitted by its ladder); the ones
        due now sent at once. A ``rendered`` one counts under ``arm_per_minute`` as its final
        version (the ladder's, the bookings'): with the arm only when an arm clip was placed;
        it is the latest performance (:meth:`last`) as placed."""
        now = self.clock()
        at_once = self.scheduler.place(performance, request, cue, now)
        placed = at_once + [w for w in self.scheduler.waiting if w.request is request]
        if rendered:
            arm = any(w.clip.part == "arm" for w in placed)
            started = self._started[-1] if self._started else None
            if started is not None and started[1] == performance.intent and started[2] != arm:
                self._started[-1] = (started[0], started[1], arm, started[3])  # rendered just now
            if placed:
                self._last = replace(placed[0].performance, clips=tuple(w.clip for w in placed))
        outcomes = [self._send(waiting, now, None) for waiting in at_once]
        return outcomes + [Outcome(True, "waiting")] * (len(placed) - len(at_once))

    def _render(
        self,
        intent: str,
        source: str,
        cue: Cue | None,
        request: Request,
        *,
        counted: bool = True,
        arg: str | None = None,
    ) -> Performance | None:
        """``intent`` rendered for the parts ready now that the request may use (``arg``: an ask's
        argument), the arm's and the body's clips stretched to their speed scales (the arm left out
        past ``arm_per_minute``, unless the performance is not ``counted``: an accent, a series'
        later item); logged with its seed. None when it cannot be played."""
        if self._driving or intent not in self.book.compositions:
            return None
        ready = frozenset(p for p, part in self.parts.items() if part.ready() is None)
        if request.parts is not None:
            ready &= request.parts
        scene = Scene(ready, speaking=bool(self._replying), conversation=self._conversation,
                      distance_m=self._view().distance_m)  # fmt: skip
        for hook in self.scene_hooks:
            scene = hook(scene, request)
        seed = self.config.seed if self.config.seed is not None else self.seeds.randrange(2**31)
        performance = self.renderer.render(intent, scene, seed, cue, arg=arg)
        scales = {"arm": self.config.arm_speed_scale, "body": self.config.body_speed_scale}
        clips = tuple(
            stretched(c, 1.0 / scales[c.part]) if c.part in scales else c for c in performance.clips
        )
        capped = source in ("marker", "moment") and any(c.part == "arm" for c in clips)
        if capped and counted and self._count(lambda arm: arm) >= self.config.caps.arm_per_minute:
            clips = tuple(c for c in clips if c.part != "arm")
            self.record("capped", {"intent": intent, "part": "arm", "why": "arm_per_minute"})
        if not clips:
            self.record("unplayable", {"intent": intent, "ready": sorted(ready)})
            return None
        performance = replace(performance, clips=clips)
        now = self.clock()
        arm = any(c.part == "arm" for c in clips)
        self._started.append((now, intent, arm, counted))  # the arm's: see _place
        self._used[_mark(intent, cue, arg)] = now
        self._last = performance
        self.record("performance", {
            "intent": intent, "arg": cue.marker.arg if cue else arg,
            "source": source, "seed": performance.seed,
            "variant": performance.variant, "parts": [c.part for c in clips],
            "turn": cue.turn if cue else None, "part": cue.index if cue else None,
            "share": round(cue.share, 3) if cue else None, "arm_at": scene.arm_at,
        })  # fmt: skip
        return performance

    def _capped(self, intent: str, cue: Cue | None, tally: Tally) -> str | None:
        """Why a marker's or a moment's ``intent`` (on ``cue``'s part, when a marker's) may not
        play now, None when it may: an accent only by ``accent_per_part``, a series' later item
        by none of the counts, a cooldown per mark and argument."""
        caps, now = self.config.caps, self.clock()
        if not self._on:
            return "gestures are off"
        if self._driving:
            return "a drive is under way"
        composition = self.book.compositions.get(intent)
        if composition is None:
            return "not in the book"
        if cue is not None:
            where = (cue.turn, cue.index)
            if tally == "accent" and self._accents.get(where, 0) >= caps.accent_per_part:
                return "accent_per_part"
            if tally == "counted" and self._per_part.get(where, 0) >= caps.per_part:
                return "per_part"
            if tally == "counted" and self._per_reply.get(cue.turn, 0) >= caps.per_reply:
                return "per_reply"
        if tally == "counted" and self._count(lambda arm: True) >= caps.per_minute:
            return "per_minute"
        waived = intent in caps.laughter_waives and now - self._laughed <= caps.laughter_s
        limit = caps.intent_per_minute.get(intent)
        if limit is not None and not waived and self._count(lambda arm: True, intent) >= limit:
            return f"{intent} per minute"
        last = self._used.get(_mark(intent, cue))
        if last is not None and now - last < composition.cooldown_s and not waived:
            return f"cooldown {composition.cooldown_s:.0f} s"
        return None

    def _count(self, which: Callable[[bool], bool], intent: str | None = None) -> int:
        """Performances started in the last minute that ``which`` (with the arm?) accepts: of
        ``intent`` when given, else those counted (no accent, no series' later item)."""
        now = self.clock()
        while self._started and now - self._started[0][0] > 60.0:
            self._started.popleft()
        return sum(
            1
            for _, name, arm, counted in self._started
            if which(arm) and (name == intent if intent is not None else counted)
        )

    def _fire_due(self) -> float | None:
        """Send the clips whose time has come; the seconds until the next check."""
        now = self.clock()
        going, wait = self.scheduler.due(now)
        for due in going:
            self._send(due.waiting, now, due.timing)
        return wait

    def _send(self, waiting: Waiting, now: float, timing: tuple[float, float] | None) -> Outcome:
        """One clip to its part: it takes the part by the priority table, attention first."""
        clip, performance, request = waiting.clip, waiting.performance, waiting.request
        part = self.parts.get(clip.part)
        if part is None:
            return Outcome(False, f"no {clip.part}")
        if self._driving:
            outcome = Outcome(False, "a drive is under way")
        elif (why := self._withheld(clip, request, now)) is not None:
            outcome = Outcome(False, why)
        else:
            try:
                outcome = part.play(clip)
            except Exception as error:
                logger.exception("gesture engine: %s broke on a clip", clip.part)
                outcome = Outcome(False, f"{type(error).__name__}: {error}")
        fields: dict[str, Any] = {
            "intent": performance.intent, "part": clip.part, "layer": clip.layer,
            "source": waiting.source, "ok": outcome.ok, "why": outcome.why,
            "seconds": round(clip.duration_s, 2),
        }  # fmt: skip
        if timing is not None:
            fields.update(planned=round(timing[0], 3), played=round(timing[1], 3),
                          late_s=round(timing[1] - timing[0], 3))  # fmt: skip
        stroke_s = outcome.stroke_s if outcome.stroke_s is not None else clip.stroke_s
        if outcome.ok:
            fields["stroke_s"] = round(stroke_s, 3)
        self.record("cue", fields)
        if outcome.ok:
            if clip.part == "arm":  # the reply a held unit belongs to (closed with it)
                self._held_turn = waiting.turn if clip.then == "hold" else None
            seconds = outcome.seconds if outcome.seconds > 0 else clip.duration_s
            ends = NOT_YET if clip.loop else now + seconds
            held = Reservation(performance, clip, now, now + stroke_s, ends, request.priority,
                               request.rank, waiting.source, request)  # fmt: skip
            refusal = self.scheduler.reservations.reserve(clip.part, held, now)
            if refusal is not None:  # the table was asked just before: never, unless it changed
                logger.warning("gesture engine: %s played but not held: %s", clip.part, refusal)
            if clip.loop:
                self._renew_at[clip.part] = now + max(RENEW_MIN_S, clip.duration_s)
            else:
                self._renew_at.pop(clip.part, None)
            self._idle_of.pop(clip.part, None)  # an idle entry's own send names it again
        return outcome

    def _withheld(self, clip: Clip, request: Request, now: float) -> str | None:
        """Why ``clip`` may not take its part now: attention's rules, then the priority table;
        None when it may."""
        if request.priority > Priority.STATE:
            view = self._view()
            if clip.part == "head" and clip.layer == "gesture":
                refusal = head_refusal(view, now, self.config.head_seen_s)
                if refusal is not None:
                    return refusal
            if clip.part == "arm":
                if now < self._arm_free_s:
                    return "the arm went to rest for the body's turn"
                refusal = arm_refusal(view, self.config.arm_past_arc)
                if refusal is not None:
                    return refusal
        return self.scheduler.reservations.refusal(clip.part, request.priority, request.rank, now)

    def _view(self) -> Attention:
        """Attention now (:data:`pepin.expression.speaker.NO_ATTENTION` without a gaze)."""
        if self.attention is None:
            return NO_ATTENTION
        try:
            return self.attention.attention()
        except Exception:
            logger.exception("gesture engine: the attention view broke")
            return NO_ATTENTION

    def _attend(self) -> float | None:
        """A gesture holding the head is withdrawn once attention seeks, checks or turns the
        body; the seconds until attention is read again (None: nothing to keep clear)."""
        if self.attention is None:
            return None
        now = self.clock()
        playing = self._playing.get("head")
        if playing is None or playing.clip.layer != "gesture" or now >= playing.ends:
            return None
        phase = self._view().phase
        if phase not in BUSY:
            return ATTENTION_POLL_S
        self.parts["head"].fade("gesture", self.config.fade_s.get("head", 0.0))
        self.scheduler.reservations.release("head")
        self.record("faded", {"intent": playing.performance.intent, "part": "head",
                              "why": f"attention: {phase}"})  # fmt: skip
        return None

    def _latency(self, part: PartName) -> float:
        executor = self.parts.get(part)
        return executor.latency_s if executor is not None else 0.0

    def _arm_scene(self, scene: Scene, request: Request) -> Scene:
        """The scene hook that says where the arm is for a rendering (``arm_at``: a pose a unit
        holds or a waiting arm clip will hold, rest, or unknown without the arm's planner)."""
        return replace(scene, arm_at=self.scheduler.arm_at())

    def _meaning(self, kind: str, fields: dict[str, Any]) -> None:
        """A row of the gesture space's (a catchment, a repeat): the engine's record and the
        conversation's events.jsonl."""
        self.record(kind, fields)
        journal = self.journal
        if journal is not None:
            try:
                journal(kind, self.clock(), **fields)
            except Exception:
                logger.exception("gesture engine: the journal broke")

    # -- the idle layer --------------------------------------------------------------------------

    def _idle(self) -> float | None:
        """An idle motion on each free part whose time has come; the seconds to the next. An
        entry waits while a state it excludes holds; its looping clip fades then. While a loop
        holds a part (until a state it excludes, or another clip, takes it) no other loop is
        planned there; a finite entry (the drift) plays over it as over a free part, and the
        loop's entry sends it again after that clip, as after any other."""
        if not self._on or self._driving or not self.book.idle:
            return None
        now = self.clock()
        states: dict[IdleState, bool] = {"speaking": bool(self._replying),
                                          "listening": self._listening}  # fmt: skip
        wait: float | None = None
        for key, idle in enumerate(self.book.idle):
            if idle.when == "conversation" and not self._conversation:
                continue
            playing = self._playing.get(idle.part)
            if any(states[s] for s in idle.unless):
                self._idle_next.pop(key, None)
                mine = playing is not None and self._idle_of.get(idle.part) is idle
                if mine and playing is not None and playing.clip.loop:
                    self.parts[idle.part].fade("idle", self.config.fade_s.get(idle.part, 0.0))
                    self.scheduler.reservations.release(idle.part)
                continue
            if playing is not None and not math.isfinite(playing.ends):
                if any(self.book.atoms[name].loop for name in idle.atoms):
                    self._idle_next.pop(key, None)
                    continue  # a loop holds the part until it is let go (renewed meanwhile)
                playing = None  # a finite entry over a loop: the loop yields, comes back after
            busy_until = playing.ends if playing is not None else float("-inf")
            if playing is not None and playing.clip.layer == "gesture":
                busy_until += self.config.idle_resume_s
            quiet_since = max([self._opened_s, *(self._ended[s] for s in idle.unless)])
            after = idle.after_speech_s
            after = self.config.idle_after_speech_s if after is None else after
            earliest = max(busy_until, quiet_since + after)
            planned = self._idle_next.get(key)
            if planned is None or planned < earliest:
                planned = earliest + self.seeds.uniform(idle.every_s.lo, idle.every_s.hi)
                self._idle_next[key] = planned
            if now < planned:
                wait = planned - now if wait is None else min(wait, planned - now)
                continue
            part = self.parts.get(idle.part)
            del self._idle_next[key]
            if part is None or part.ready() is not None:
                continue
            seed = self.seeds.randrange(2**31)
            name = random.Random(seed).choice(idle.atoms)
            clip = self.renderer.render_atom(name, seed)
            performance = Performance(f"idle:{name}", seed, 0, (clip,))
            request = Request(performance.intent, None, Priority.IDLE, "idle")
            if self._send(Waiting(performance, clip, request), now, None).ok:
                self._idle_of[idle.part] = idle
        return wait

    def _renew(self) -> float | None:
        """Each looping clip still holding its part renewed at its owner once a period (the
        arbiter lets an overlay lapse a second after its period); the seconds to the next."""
        now = self.clock()
        wait: float | None = None
        for part, due in list(self._renew_at.items()):
            held = self._playing.get(part)
            if held is None or not held.clip.loop:
                del self._renew_at[part]
                continue
            if now >= due:
                executor = self.parts.get(part)
                if isinstance(executor, Renewing):
                    try:
                        executor.renew(held.clip.layer)
                    except Exception:
                        logger.exception("gesture engine: %s would not renew", part)
                due = now + max(RENEW_MIN_S, held.clip.duration_s)
                self._renew_at[part] = due
            wait = due - now if wait is None else min(wait, due - now)
        return wait

    def _apply_knobs(self) -> None:
        """The arm's part checks against the speed scale its clips are stretched by, carries the
        book's breath for the arm (none when the book has none) and the gentle share; the
        head's rest eases back over the head's fade; the scheduler's timing."""
        arm = self.parts.get("arm")
        if isinstance(arm, ArmPart):
            arm.speed_scale = self.config.arm_speed_scale
            arm.gentle_share = self.config.arm_caps_share
            breath = self.book.breath.get("arm")
            arm.breath = None if breath is None else ArmBreath(breath, JOINTS, self.clock)
        head = self.parts.get("head")
        if isinstance(head, HeadPart):
            head.fade_s = self.config.fade_s.get("head", head.fade_s)
        c = self.config
        self.scheduler.configure(
            audio_lag_s=c.audio_lag_s, late_s=c.late_s, poll_s=c.poll_s, timing=c.timing
        )


def _holds(part: object) -> bool:
    """Whether ``part`` holds a unit's pose (an arm that keeps one: the board's ``then:
    hold``)."""
    holding = getattr(part, "holding", None)
    return callable(holding) and holding() is not None


def _mark(intent: str, cue: Cue | None, arg: str | None = None) -> str:
    """A mark as its cooldown knows it: the intent, with the argument its marker (or an ask)
    gave (``point:you`` and ``point:printer`` are two)."""
    if cue is not None:
        arg = cue.marker.arg if cue.marker.name == intent else None
    return intent if arg is None else f"{intent}:{arg}"


def styled_arg(composition: Composition, arg: str | None, style: str | None) -> str | None:
    """A bare marker's argument chosen by its part's style (the composition's ``style_args``:
    ``{laugh}`` in a ``{sinister}`` part is ``{laugh:evil}``); a marker's own argument, or a
    part without a style, as given."""
    if arg is not None or style is None or not composition.style_args:
        return arg
    words = set(re.split(r"[\s,;]+", style.lower()))
    for candidate, cues_ in composition.style_args.items():
        if words & set(cues_):
            return candidate
    return arg


def bounded(wait: float | None) -> float | None:
    """A wait the engine's thread may sleep: None (until a call), else 0..:data:`MAX_WAIT_S`
    (an endless or a longer one cut there: the thread never sleeps past it)."""
    if wait is None:
        return None
    if not math.isfinite(wait):
        return MAX_WAIT_S
    return min(max(0.0, wait), MAX_WAIT_S)
