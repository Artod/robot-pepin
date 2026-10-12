"""The gesture engine: the body's gesture and idle layers, driven by the conversation.

It listens to the cascade (:class:`pepin.expression.conversation.ConversationObserver`) and is
the ``gestures`` switch (:class:`pepin.expression.conversation.LiveFlag`). A part queued for
the voice brings its markers: each intent is rendered at once (before the voice has made a
sound of it, so a preparation has time) and each clip waits for its word on the player's clock
(:func:`pepin.expression.markers.sync_s`): it is sent when the stream has played up to

    the word's start + the clip's offset from it + audio_lag_s - the part's latency_s,

or at once when that is past by at most ``late_s`` (later still, it is dropped and logged). An
arm clip is stretched by ``1 / speed_scale`` first (the first live test runs at 0.2), so its
stroke still lands on the word. A tool or the command line performs now (:meth:`perform`).

Layers, per part: a new performance takes the part (the newest wins); the person starting to
speak fades the gesture layer of every part (``fade_s`` per part) unless a greeting or a
farewell already past its stroke ends within ``finish_s``; a drive and the conversation's close
send every part to rest and drop what waits; a cut reply drops its clips. The idle layer plays
an atom of the book's ``idle`` entries every ``every_s`` on a part no gesture holds, while
nobody speaks (``idle_after_speech_s`` after either stopped, ``idle_resume_s`` after a gesture).

Caps (config/expression.json ``caps``) hold the markers and the moments back, never a tool or
the command line, which a person asked for: per part (a sentence or a clause of one), per
reply, per minute, arm gestures per minute, an intent's own per minute (``laugh``: 1, waived
while the person's own laugh is fresh), and each composition's ``cooldown_s``.

One thread does all of it, waking only for what waits: every ``poll_s`` while a clip waits for
its word, at a performance's next clip, at the next idle motion; the cascade's calls only post
to it and never wait. Every performance is logged with its seed and variant (a replay is
``scripts/gesture.py play NAME --seed N --variant V``), every clip sent with the stream second
it was planned for and the one it went at.
"""

from __future__ import annotations

import concurrent.futures
import logging
import random
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field, replace
from typing import Any

from pepin.expression.animation import (
    PARTS,
    Book,
    Clip,
    Event,
    PartName,
    Performance,
    Scene,
)
from pepin.expression.conversation import GESTURES, Heard
from pepin.expression.executors import Executor, Outcome
from pepin.expression.markers import Cue, Part, Playout, cues, sync_s
from pepin.expression.mix import stretched
from pepin.expression.parts import ArmPart
from pepin.expression.render import BookRenderer, WeightedChooser

logger = logging.getLogger(__name__)

Record = Callable[[str, dict[str, Any]], None]
TOOL_WAIT_S = 2.0  # a tool's perform waits this long for the engine's thread


@dataclass(frozen=True)
class Caps:
    """How often the markers and the moments may move the body (a tool's ask is never capped)."""

    per_part: int = 1
    per_reply: int = 2
    per_minute: int = 6
    arm_per_minute: int = 3
    intent_per_minute: Mapping[str, int] = field(default_factory=lambda: {"laugh": 1})
    laughter: str = r"(?:х[аеи][- ]?){2,}|ахах|хих|haha"  # noqa: RUF001
    laughter_waives: tuple[str, ...] = ("laugh",)
    laughter_s: float = 30.0


@dataclass(frozen=True)
class EngineConfig:
    """config/expression.json's knobs as the engine reads them (switching ``gestures`` on reads
    them again)."""

    audio_lag_s: float = 0.15
    char_s: float = 0.07
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
    caps: Caps = field(default_factory=Caps)


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
        char_s=float(timing.get("char_s", d.char_s)),
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
        ),
    )


@dataclass
class Waiting:
    """A clip waiting to be sent: for its word (``cue``) or for a time on the engine's clock."""

    performance: Performance
    clip: Clip
    source: str
    cue: Cue | None = None
    due: float = 0.0


@dataclass
class Playing:
    """What holds a part now: the performance and its clip, sent at ``started`` (engine
    clock), its stroke and its end on the same clock."""

    performance: Performance
    clip: Clip
    started: float
    stroke: float
    ends: float


@dataclass
class Spoken:
    """A part of a reply as the voice has it: its text's length, and whether all its audio is
    with the player."""

    chars: int
    spoken: bool = False


class GestureEngine:
    """The animator (:class:`pepin.expression.executors.Animator`), a conversation observer and
    the ``gestures`` live flag in one."""

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
    ) -> None:
        """``load`` reads the book (with the arm's rest pose) and the knobs, now and at every
        switch on; ``record(kind, fields)`` hears every performance, cue and refusal."""
        self.parts = dict(parts)
        self.playout = playout
        self.load = load
        self.clock = clock
        self.record = record or (lambda kind, fields: logger.info("%s %s", kind, fields))
        self.seeds = seeds or random.Random()
        self.book, self.config = load()
        self.renderer = BookRenderer(self.book, WeightedChooser())
        self._apply_speed()
        self._on = on
        self._lock = threading.RLock()
        self._wake = threading.Condition(self._lock)
        self._inbox: deque[Callable[[], None]] = deque()
        self._thread: threading.Thread | None = None
        self._stop = False
        self._waiting: list[Waiting] = []
        self._playing: dict[PartName, Playing] = {}
        self._spoken: dict[tuple[int, int], Spoken] = {}
        self._per_part: dict[tuple[int, int], int] = {}
        self._per_reply: dict[int, int] = {}
        self._started: deque[tuple[float, str, bool]] = deque()  # (when, intent, with the arm)
        self._used: dict[str, float] = {}  # intent -> when it last played
        self._laughed = float("-inf")
        self._last: Performance | None = None
        self._conversation = False
        self._driving = False
        self._listening = False  # the person is speaking
        self._replying: set[int] = set()  # turns whose parts are with the voice
        self._quiet_since = float("-inf")
        self._idle_next: dict[PartName, float] = {}

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
                self.renderer = BookRenderer(book, self.renderer.chooser)
                self._apply_speed()
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
        """Perform ``intent`` now (a tool, the command line): refused while off, during a
        drive, for an intent the book lacks, or when no part can play."""
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

    def hush(self, why: str) -> None:
        """The gesture layer of every part let go (the person speaks)."""
        self._post(lambda: self._hush(why))

    def last(self) -> Performance | None:
        """The latest performance started."""
        return self._last

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

    def part_spoken(self, turn: int, index: int) -> None:
        """All of that part's audio is with the player: its length is known."""
        self._post(lambda: self._part_spoken(turn, index))

    def reply_cut(self, turn: int) -> None:
        """The reply stopped short: what waits for its words is dropped."""
        self._post(lambda: self._drop_turns(turn, "cut"))

    def reply_done(self, turn: int) -> None:
        """The reply played out: whatever still waited for it is dropped."""
        self._post(lambda: self._drop_turns(turn, "done", only=True))

    def moment(self, name: str) -> None:
        """A moment of the robot the book maps to an intent (``slow_tool`` -> thinking)."""
        self._post(lambda: self._moment(name))

    def drive(self, active: bool) -> None:
        """A drive: every part to rest and nothing new until it ends."""
        self._post(lambda: self._drive(active))

    def closed(self, reason: str) -> None:
        """The conversation is over: every part to rest, nothing waits."""
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
        layer. The seconds until it has something to do again (None: until a call)."""
        with self._lock:
            while self._inbox:
                call = self._inbox.popleft()
                try:
                    call()
                except Exception:
                    logger.exception("gesture engine: a call broke")
            waits = [self._fire_due(), self._idle()]
        known = [w for w in waits if w is not None]
        return min(known) if known else None

    def _run(self) -> None:
        while True:
            wait = self.step()
            with self._lock:
                if self._stop:
                    return
                if not self._inbox:
                    self._wake.wait(wait)

    def _post(self, call: Callable[[], None]) -> None:
        with self._lock:
            self._inbox.append(call)
            self._wake.notify()

    # -- what the calls do (on the engine's thread) ----------------------------------------------

    def _opened(self) -> None:
        self._conversation = True
        self._quiet_since = self.clock()
        self._idle_next.clear()

    def _speech_started(self) -> None:
        self._listening = True
        self._hush("the person speaks")
        for part, playing in list(self._playing.items()):
            if playing.clip.layer == "idle":
                self.parts[part].fade("idle", self.config.fade_s.get(part, 0.0))
                del self._playing[part]

    def _speech_ended(self, heard: Heard) -> None:
        self._listening = False
        self._quiet_since = self.clock()
        if heard.text and re.search(self.config.caps.laughter, heard.text.lower()):
            self._laughed = self.clock()

    def _part_queued(self, turn: int, index: int, part: Part) -> None:
        self._replying.add(turn)
        self._spoken[(turn, index)] = Spoken(len(part.text))
        if not self._on:
            return
        for cue in cues(turn, index, part):
            if cue.marker.channel == "face":
                hold = self.config.face_hold_s
                face = Clip("face", 0.0, 0.0, hold, events=(Event(0.0, cue.marker.name, hold),))
                shown = Performance(f"face:{cue.marker.name}", 0, 0, (face,), cue)
                self._waiting.append(Waiting(shown, face, "marker", cue))
                continue
            refusal = self._capped(cue.marker.name, turn, index)
            if refusal is not None:
                self.record("capped", {"intent": cue.marker.name, "turn": turn, "part": index,
                                       "why": refusal})  # fmt: skip
                continue
            performance = self._render(cue.marker.name, "marker", cue)
            if performance is None:
                continue
            self._per_part[(turn, index)] = self._per_part.get((turn, index), 0) + 1
            self._per_reply[turn] = self._per_reply.get(turn, 0) + 1
            for clip in performance.clips:
                self._waiting.append(Waiting(performance, clip, "marker", cue))

    def _part_spoken(self, turn: int, index: int) -> None:
        spoken = self._spoken.get((turn, index))
        if spoken is not None:
            spoken.spoken = True

    def _drop_turns(self, turn: int, why: str, only: bool = False) -> None:
        def gone(t: int) -> bool:
            return t == turn if only else t <= turn

        dropped = [w for w in self._waiting if w.cue is not None and gone(w.cue.turn)]
        for waiting in dropped:
            self.record("dropped", {"intent": waiting.performance.intent, "part": waiting.clip.part,
                                    "turn": turn, "why": why})  # fmt: skip
        self._waiting = [w for w in self._waiting if w not in dropped]
        for key in [k for k in self._spoken if gone(k[0])]:
            del self._spoken[key]
        self._replying = {t for t in self._replying if not gone(t)}
        self._quiet_since = self.clock()

    def _moment(self, name: str) -> None:
        intent = self.book.moments.get(name)
        if intent is None or not self._on:
            return
        refusal = self._capped(intent, None, None)
        if refusal is not None:
            self.record("capped", {"intent": intent, "moment": name, "why": refusal})
            return
        performance = self._render(intent, "moment", None)
        if performance is not None:
            self._now(performance, "moment")

    def _drive(self, active: bool) -> None:
        self._driving = active
        if active:
            self._rest_all()

    def _closed(self) -> None:
        self._conversation = False
        self._listening = False
        self._replying.clear()
        self._spoken.clear()
        self._rest_all()

    def _rest_all(self) -> None:
        self._waiting.clear()
        self._playing.clear()
        for part in self.parts.values():
            try:
                part.rest()
            except Exception:
                logger.exception("gesture engine: %s would not rest", part.part)

    def _hush(self, why: str) -> None:
        now = self.clock()
        for part, playing in list(self._playing.items()):
            if playing.clip.layer != "gesture" or now >= playing.ends:
                continue
            finishing = (
                playing.performance.intent in self.config.finish
                and now >= playing.stroke
                and playing.ends - now <= self.config.finish_s
            )
            if finishing:
                continue
            self.parts[part].fade("gesture", self.config.fade_s.get(part, 0.0))
            del self._playing[part]
            self.record("faded", {"intent": playing.performance.intent, "part": part, "why": why})

    # -- performing ------------------------------------------------------------------------------

    def _perform_now(self, intent: str, source: str) -> Outcome:
        with self._lock:
            if not self._on:
                return Outcome(
                    False, "gestures are off (scripts/voice_cascade.py switch --flag gestures=on)"
                )
            if self._driving:
                return Outcome(False, "a drive is under way: the body is the drive's")
            if intent not in self.book.compositions:
                return Outcome(
                    False,
                    f"no gesture {intent!r}; the gestures are {', '.join(self.book.compositions)}",
                )
            performance = self._render(intent, source, None)
            if performance is None:
                return Outcome(False, "no part can play it now")
            outcomes = self._now(performance, source)
        played = [o for o in outcomes if o.ok]
        if not played:
            return Outcome(False, "; ".join(o.why for o in outcomes) or "nothing played")
        seconds = max(c.offset_s + c.duration_s for c in performance.clips) - min(
            c.offset_s for c in performance.clips
        )
        return Outcome(True, f"seed {performance.seed}, variant {performance.variant}", seconds)

    def _render(self, intent: str, source: str, cue: Cue | None) -> Performance | None:
        """``intent`` rendered for the parts ready now, the arm's clip stretched to its speed
        scale; logged with its seed. None when it cannot be played."""
        if self._driving or intent not in self.book.compositions:
            return None
        ready = frozenset(p for p, part in self.parts.items() if part.ready() is None)
        scene = Scene(ready, speaking=bool(self._replying), conversation=self._conversation)
        seed = self.config.seed if self.config.seed is not None else self.seeds.randrange(2**31)
        performance = self.renderer.render(intent, scene, seed, cue)
        clips = tuple(
            stretched(c, 1.0 / self.config.arm_speed_scale) if c.part == "arm" else c
            for c in performance.clips
        )
        capped = source in ("marker", "moment") and any(c.part == "arm" for c in clips)
        if capped and self._count(lambda arm: arm) >= self.config.caps.arm_per_minute:
            clips = tuple(c for c in clips if c.part != "arm")
            self.record("capped", {"intent": intent, "part": "arm", "why": "arm_per_minute"})
        if not clips:
            self.record("unplayable", {"intent": intent, "ready": sorted(ready)})
            return None
        performance = replace(performance, clips=clips)
        now = self.clock()
        self._started.append((now, intent, any(c.part == "arm" for c in clips)))
        self._used[intent] = now
        self._last = performance
        self.record("performance", {
            "intent": intent, "source": source, "seed": performance.seed,
            "variant": performance.variant, "parts": [c.part for c in clips],
            "turn": cue.turn if cue else None, "part": cue.index if cue else None,
            "share": round(cue.share, 3) if cue else None,
        })  # fmt: skip
        return performance

    def _capped(self, intent: str, turn: int | None, index: int | None) -> str | None:
        """Why a marker's or a moment's ``intent`` may not play now, None when it may."""
        caps, now = self.config.caps, self.clock()
        if not self._on:
            return "gestures are off"
        if self._driving:
            return "a drive is under way"
        composition = self.book.compositions.get(intent)
        if composition is None:
            return "not in the book"
        if turn is not None and index is not None:
            if self._per_part.get((turn, index), 0) >= caps.per_part:
                return "per_part"
            if self._per_reply.get(turn, 0) >= caps.per_reply:
                return "per_reply"
        if self._count(lambda arm: True) >= caps.per_minute:
            return "per_minute"
        waived = intent in caps.laughter_waives and now - self._laughed <= caps.laughter_s
        limit = caps.intent_per_minute.get(intent)
        if limit is not None and not waived and self._count(lambda arm: True, intent) >= limit:
            return f"{intent} per minute"
        last = self._used.get(intent)
        if last is not None and now - last < composition.cooldown_s and not waived:
            return f"cooldown {composition.cooldown_s:.0f} s"
        return None

    def _count(self, which: Callable[[bool], bool], intent: str | None = None) -> int:
        """Performances started in the last minute (of ``intent`` when given) that ``which``
        (with the arm?) accepts."""
        now = self.clock()
        while self._started and now - self._started[0][0] > 60.0:
            self._started.popleft()
        return sum(
            1 for _, name, arm in self._started if which(arm) and (intent is None or name == intent)
        )

    def _now(self, performance: Performance, source: str) -> list[Outcome]:
        """Each clip of a performance at its own time from now (the earliest at once)."""
        now = self.clock()
        first = min(c.offset_s for c in performance.clips)
        outcomes = []
        for clip in sorted(performance.clips, key=lambda c: c.offset_s):
            due = now + clip.offset_s - first
            if due <= now:
                outcomes.append(
                    self._send(Waiting(performance, clip, source, None, due), now, None)
                )
            else:
                self._waiting.append(Waiting(performance, clip, source, None, due))
                outcomes.append(Outcome(True, "waiting", clip.duration_s))
        return outcomes

    def _fire_due(self) -> float | None:
        """Send the clips whose time has come; the seconds until the next check."""
        now = self.clock()
        later: list[Waiting] = []
        wait: float | None = None
        for waiting in self._waiting:
            if waiting.cue is None:
                if waiting.due <= now:
                    self._send(waiting, now, None)
                else:
                    later.append(waiting)
                    wait = waiting.due - now if wait is None else min(wait, waiting.due - now)
                continue
            target = self._target(waiting)
            if target is None:
                spoken = self._spoken.get((waiting.cue.turn, waiting.cue.index))
                if spoken is not None and spoken.spoken:
                    self._send(waiting, now, None)  # a part with no audio: its cues at once
                    continue
                later.append(waiting)
                wait = self.config.poll_s if wait is None else min(wait, self.config.poll_s)
                continue
            played = self.playout.played()
            if played < target:  # checked again soon: the part's length grows while it is made
                later.append(waiting)
                soon = max(0.001, min(self.config.poll_s, target - played))
                wait = soon if wait is None else min(wait, soon)
                continue
            if played - target > self.config.late_s:
                self.record("late", {"intent": waiting.performance.intent,
                                     "part": waiting.clip.part, "planned": round(target, 3),
                                     "played": round(played, 3)})  # fmt: skip
                continue
            self._send(waiting, now, (target, played))
        self._waiting = later
        return wait

    def _target(self, waiting: Waiting) -> float | None:
        """The stream second a cued clip goes at; None before its part's first audio."""
        cue = waiting.cue
        assert cue is not None
        mark = self.playout.mark(cue.turn, cue.index)
        spoken = self._spoken.get((cue.turn, cue.index))
        if mark is None or spoken is None:
            return None
        word = sync_s(cue, mark, spoken.chars, spoken.spoken, self.config.char_s)
        latency = self.parts[waiting.clip.part].latency_s
        return word + waiting.clip.offset_s + self.config.audio_lag_s - latency

    def _send(self, waiting: Waiting, now: float, timing: tuple[float, float] | None) -> Outcome:
        """One clip to its part: it takes the part from what played there."""
        clip, performance = waiting.clip, waiting.performance
        part = self.parts.get(clip.part)
        if part is None:
            return Outcome(False, f"no {clip.part}")
        if self._driving:
            outcome = Outcome(False, "a drive is under way")
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
        self.record("cue", fields)
        if outcome.ok:
            stroke = now + clip.stroke_s
            self._playing[clip.part] = Playing(
                performance, clip, now, stroke, now + clip.duration_s
            )
        return outcome

    # -- the idle layer --------------------------------------------------------------------------

    def _idle(self) -> float | None:
        """An idle motion on each free part whose time has come; the seconds to the next."""
        if not self._on or self._driving or not self.book.idle:
            return None
        now = self.clock()
        quiet = not self._listening and not self._replying
        wait: float | None = None
        for idle in self.book.idle:
            if idle.when == "conversation" and not self._conversation:
                continue
            if not quiet:
                self._idle_next.pop(idle.part, None)
                continue
            playing = self._playing.get(idle.part)
            busy_until = playing.ends if playing is not None else float("-inf")
            if playing is not None and playing.clip.layer == "gesture":
                busy_until += self.config.idle_resume_s
            earliest = max(busy_until, self._quiet_since + self.config.idle_after_speech_s)
            planned = self._idle_next.get(idle.part)
            if planned is None or planned < earliest:
                planned = earliest + self.seeds.uniform(idle.every_s.lo, idle.every_s.hi)
                self._idle_next[idle.part] = planned
            if now < planned:
                wait = planned - now if wait is None else min(wait, planned - now)
                continue
            part = self.parts.get(idle.part)
            del self._idle_next[idle.part]
            if part is None or part.ready() is not None:
                continue
            seed = self.seeds.randrange(2**31)
            name = random.Random(seed).choice(idle.atoms)
            clip = self.renderer.render_atom(name, seed)
            performance = Performance(f"idle:{name}", seed, 0, (clip,))
            self._send(Waiting(performance, clip, "idle"), now, None)
        return wait

    def _apply_speed(self) -> None:
        """The arm's part checks against the speed scale its clips are stretched by."""
        arm = self.parts.get("arm")
        if isinstance(arm, ArmPart):
            arm.speed_scale = self.config.arm_speed_scale
