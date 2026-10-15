"""One reply end to end through the gesture engine's own thread, the alive listeners on the real
prosody listener, and the word scheduler lent the engine's parts, renderer, hooks and nuclei:
the person's pause with their words, the first face, TEXT, START and AUDIO of two sentences
with a {greeting}, the reply's end and the close."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import itertools
import json
import logging
import random
import threading
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pepin.arm_motion import ArmMotion, load_arm_motion
from pepin.expression.animation import PartName
from pepin.expression.animator import GestureEngine, engine_config
from pepin.expression.book import load_book, vocabulary, with_rest
from pepin.expression.conversation import LISTENER_PAUSE, Heard, VoiceChunk
from pepin.expression.executors import Executor
from pepin.expression.fakes import (
    FakeArmLink,
    FakeAttention,
    FakeClock,
    FakeExecutor,
    FakePlaces,
    FakePlayout,
)
from pepin.expression.head import IDLE_LOOK
from pepin.expression.listeners import (
    ARM_BREATH,
    ARM_SWAY,
    LAUGHING,
    PEAK,
    STRONG,
    WEAK,
    Alive,
    alive_knobs,
)
from pepin.expression.markers import Marker, Part, parse
from pepin.expression.parts import ArmPart, clip_look
from pepin.expression.prosody import ProsodyListener
from pepin.expression.speaker import Attention
from pepin.expression.trajectory import JOINTS, ArmStatus, Trajectory

CONFIG = Path(__file__).resolve().parents[2] / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
BOOK = with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg)
DATA = json.loads((CONFIG / "expression.json").read_text())

RATE = 24000
STEP_S = 0.02


def vowels(spec: list[tuple[float, float, float]], total_s: float) -> bytes:
    """Synthetic speech: a voiced vowel (seven harmonics of its f0) for each (centre, f0, its
    half length): a plain one 220 Hz and 0.075 s, a stressed one higher and longer."""
    t = np.arange(round(total_s * RATE)) / RATE
    x = 1e-4 * np.random.default_rng(0).standard_normal(len(t))
    for centre, f0, half in spec:
        tone = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 8))
        envelope = np.sin(np.pi / 2 * np.clip(1 - np.abs(t - centre) / half, 0, 1)) ** 2
        x = x + 0.3 * envelope * tone
    return (np.clip(x, -1, 1) * 32767).astype("<i2").tobytes()


@dataclass
class TimedLink(FakeArmLink):
    """The board's arm, each line to it kept with the clock's time."""

    clock: FakeClock = field(default_factory=FakeClock)
    times: list[tuple[str, float, Any]] = field(default_factory=list)

    def send(self, trajectory: Trajectory) -> None:
        """Recorded with its time."""
        super().send(trajectory)
        self.times.append(("send", self.clock.now, trajectory))

    def rest(self) -> None:
        """Recorded with its time."""
        super().rest()
        self.times.append(("rest", self.clock.now, None))


class Conversation:
    """The engine on its own thread over the arm's real planner on a fake board and fake head,
    face and wheels; the alive listeners (stepped here) on the real prosody listener; the word
    scheduler lent the engine's parts, renderer, hooks and the voice's nuclei; one fake clock and
    player, moved by the test, the engine's thread brought level with them at every step."""

    def __init__(self, motion: ArmMotion = MOTION, *, breath: bool = False, **pulse: Any) -> None:
        """``motion``: the arm's (config/arm.json's by default); ``breath``: the book's breath
        layer under the arm's clips (off: the book without it, the held arm's older wrist sway,
        the knots exactly the poses'); ``pulse``: knobs of the ``pulse`` block changed by name."""
        self.clock = FakeClock(1000.0)
        self.playout = FakePlayout(now=0.0)
        self.playing = False
        self.link = TimedLink(state=ArmStatus(0.0, MOTION.rest(), False, "off"), clock=self.clock)
        arm = ArmPart(self.link, motion, latency_s=0.06, clock=self.clock)
        book = BOOK if breath else replace(BOOK, breath={})
        self.parts: dict[PartName, Executor] = {
            "arm": arm, "head": FakeExecutor("head", 0.1), "face": FakeExecutor("face", 0.05),
            "body": FakeExecutor("body", 0.15),
        }  # fmt: skip
        self.records: list[tuple[str, dict[str, Any]]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.view = FakeAttention(Attention("track", 0.0, eyes=True, seen_s=self.clock.now))
        config = replace(engine_config(DATA), seed=3)
        self.engine = GestureEngine(
            self.parts, self.playout, lambda: (book, config), on=True, clock=self.clock,
            record=lambda kind, fields: self.records.append((kind, fields)),
            seeds=random.Random(3), attention=self.view,
        )  # fmt: skip
        knobs = alive_knobs(DATA)
        moments = replace(knobs.moments, listener_nod_p=1.0, ack_nod_p=0.0, think_p=0.0)
        knobs = replace(knobs, moments=moments, pulse=replace(knobs.pulse, **pulse))
        self.prosody = ProsodyListener()
        self.alive = Alive(
            self.engine, self.prosody, knobs=lambda: knobs, clock=self.clock, hour=lambda: 14,
            seed=1, journal=lambda kind, now, **f: self.events.append((kind, f)),
        )  # fmt: skip
        self.engine.taps.append(self.prosody)
        self.engine.scene_hooks.append(self.alive.scene)
        self.engine.scheduler.stay_up = self.alive.stays_up
        self.engine.scheduler.attach(
            self.engine.parts, lambda: self.engine.renderer, nuclei=self.prosody,
            hooks=self.engine.scene_hooks,
        )  # fmt: skip

    def both(self, name: str, *args: Any) -> None:
        """A moment of the conversation to both observers."""
        getattr(self.engine, name)(*args)
        getattr(self.alive, name)(*args)

    def run(self, seconds: float, at: dict[float, Any] | None = None) -> None:
        """``seconds`` of the conversation in STEP_S steps (``at``: what happens when)."""
        start, todo = self.clock.now, sorted((at or {}).items())
        for _ in range(round(seconds / STEP_S)):
            self.clock.now += STEP_S
            if self.playing:
                self.playout.now += STEP_S
            self.view.be(seen_s=self.clock.now)
            while todo and todo[0][0] <= self.clock.now - start + 1e-9:
                todo.pop(0)[1]()
            assert "no gesture" in self.engine.perform("-", source="test").why  # one step on it
            self.engine.pending()  # and that step finished
            self.alive.step()

    def feed(self, turn: int, index: int, start: float, pcm: bytes) -> None:
        """The part's audio as the voice's 0.5 s chunks, all at once (the voice runs ahead)."""
        n = RATE  # bytes in 0.5 s of 16-bit mono
        for a in range(0, len(pcm), n):
            end = start + (a + len(pcm[a : a + n])) / 2 / RATE
            self.engine.audio(VoiceChunk(turn, index, start, end, pcm[a : a + n], RATE, 0.0))

    def accents(self, part: str) -> list[dict[str, Any]]:
        """The cue rows that went of the accents' ensembles on ``part``."""
        return [f for intent in ACCENTS for f in self.sent(intent, part)]

    def sent(self, intent: str, part: str | None = None) -> list[dict[str, Any]]:
        rows = [f for k, f in self.records if k == "cue" and f["intent"] == intent and f["ok"]]
        return [f for f in rows if part is None or f["part"] == part]


def test_one_reply_end_to_end_through_the_engine_s_thread_the_listeners_and_the_word_clock(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The person speaks with a pause (their words read in time), the robot's first clear face
    comes, then the reply: TEXT for two sentences, the first with {greeting} before «привет»;
    START of each; their audio with three vowels each, analysed into nuclei (the AUDIO moment);
    the reply plays out and the conversation closes. Nothing raises on the engine's thread, the
    face smiles once, the nods keep their caps, every part of the greeting the ladder placed
    lands, and the thread stops at the close."""
    broken: list[threading.ExceptHookArgs] = []
    hook = threading.excepthook
    threading.excepthook = broken.append
    caplog.set_level(logging.ERROR, logger="pepin.expression")
    c = Conversation()
    first = "Ну что ж, привет тебе, дорогой друг!"
    second = "Рад тебя видеть снова, честно."
    lengths = (0.09 * len(first), 0.09 * len(second))  # the word clock's own law: no surprise

    def vowel_s(text: str, word: str, length: float) -> float:  # where the clock hears it
        char = text.index(word) + next(i for i, ch in enumerate(word) if ch in "аеёиоуыэюя")
        return 0.16 + char / len(text) * (length - 0.16)

    def text() -> None:
        greeting = Marker("gesture", "greeting", first.index("привет"))
        c.both("part_queued", 1, 0, Part(first, None, (greeting,)))
        c.both("part_queued", 1, 1, Part(second))

    def start(index: int) -> None:
        begin = 0.0 if index == 0 else lengths[0]
        c.playout.marks[(1, index)] = (begin, begin + lengths[index])
        c.both("part_started", 1, index, begin)
        words = (("что", "привет", "дорогой"), ("Рад", "видеть", "честно"))[index]
        texts = (first, second)[index]
        spec = [(vowel_s(texts, w, lengths[index]), 220.0, 0.075) for w in words]
        stressed = 2 - index  # «дорогой», «видеть»: higher and longer, their z over beat_z
        spec[stressed] = (spec[stressed][0], 400.0, 0.18)
        c.feed(1, index, begin, vowels(spec, lengths[index]))
        c.both("part_spoken", 1, index)
        if index == 0:
            c.playing = True

    try:
        c.engine.start()
        c.both("opened", Heard(c.clock.now - 1.0, c.clock.now, "Пепин"))
        c.run(4.2, {
            0.3: lambda: c.view.be(first_face_s=c.clock.now),
            0.5: lambda: c.both("speech_started", c.clock.now),
            1.2: lambda: c.view.be(first_face_s=c.clock.now),  # a later face: no second smile
            2.8: lambda: c.both("moment", LISTENER_PAUSE, "Я тебе сейчас расскажу"),
            3.6: lambda: c.both("speech_ended", Heard(c.clock.now - 3.1, c.clock.now,
                                                      "Я тебе сейчас расскажу одну историю.")),
            4.0: text,
            4.15: lambda: start(0),
        })  # fmt: skip
        c.run(lengths[0] - 0.4, {lengths[0] - 0.6: lambda: start(1)})
        c.run(lengths[1] + 0.8)
        c.both("reply_done", 1)
        c.run(0.4)
        c.both("closed", "goodbye")
        c.run(0.2)
    finally:
        c.engine.close()
        threading.excepthook = hook
    assert c.engine._thread is not None and not c.engine._thread.is_alive()
    assert not broken and not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert len(c.sent("smile", "face")) == 1
    pulses = {(f["turn"], f["index"]): f for k, f in c.events if k == "pulse"}
    assert set(pulses) == {(1, 0), (1, 1)} and all(f["nuclei"] == 3 for f in pulses.values())
    assert pulses[(1, 0)]["accent"] is None and pulses[(1, 0)]["skipped"] == ["a marker's part"]
    assert pulses[(1, 1)]["accent"]["kind"] in ("weak", "strong", "peak")
    assert "head" in pulses[(1, 1)]["accent"]["parts"]
    assert len([f for i in ACCENTS for f in c.sent(i, "head")]) == 1
    assert len(c.sent("listen_nod", "head")) == 1
    (ladder,) = [f for k, f in c.records if k == "ladder" and f["intent"] == "greeting"]
    landed = {f["part"] for f in c.sent("greeting")}
    assert set(ladder["parts"]) <= landed, (ladder, landed)
    if "arm" in ladder["parts"]:
        assert [name for name, _ in c.link.calls].count("send") >= 1
    sends = [f for k, f in c.records if k == "send" and f["intent"] == "greeting"]
    assert sends and all(-0.3 <= f["error_s"] <= 0.5 for f in sends)


def played(c: Conversation, part: PartName, intent: str) -> list[Any]:
    """What ``part`` played of ``intent``, in order (its ok cue rows are its plays; the arm's
    gentle way down at a reply's end is no cue's)."""
    if part == "arm":
        sent = [args[0] for name, args in c.link.calls if name == "send"]
        plays = [tr for tr in sent if tr.name != "down"]
    else:
        executor = c.parts[part]
        assert isinstance(executor, FakeExecutor)
        plays = [args[0] for name, args in executor.calls if name == "play"]
    rows = [f for k, f in c.records if k == "cue" and f["part"] == part and f["ok"]]
    return [x for x, f in zip(plays, rows, strict=True) if f["intent"] == intent]


def test_a_reply_pointing_at_a_place_and_counting_two_items_through_the_engine_s_resolver(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The persona's examples as one reply (the printer pointed at, then two counted items),
    parsed as the loop parses the brain's sentences, on the engine's own
    thread with its resolver lent the room (the printer at +30 deg) and the attention view (the
    person at -10): the shut hand turns toward the printer (the arm's pan inward of straight
    ahead, as a place on the left is), the head glances its way by at most 25 deg though it
    stands 40 deg from the person, each item takes its spot (two catchment rows, new and new,
    in the record and the conversation's journal), every mark's ladder row names its variant,
    nothing raises on the thread, the spots are forgotten at the close and the thread stops."""
    broken: list[threading.ExceptHookArgs] = []
    hook = threading.excepthook
    threading.excepthook = broken.append
    caplog.set_level(logging.ERROR, logger="pepin.expression")
    c = Conversation()
    c.view.be(bearing_deg=-10.0, distance_m=1.5)
    c.engine.resolver.room = FakePlaces({"printer": 30.0})
    c.engine.journal = lambda kind, now, **f: c.events.append((kind, f))
    words = vocabulary(BOOK, frozenset())
    written = ("Принтер стоит {point:printer} вон там.", "Во-первых, {count:1} батарея.",
               "Во-вторых, {count:2} колёса.")  # fmt: skip
    parts = [parse(sentence, words).parts[0] for sentence in written]
    lengths = [0.09 * len(p.text) for p in parts]
    begins = [sum(lengths[:k]) for k in range(len(parts))]
    stressed = (("Принтер", "вон", "там"), ("первых", "батарея"), ("вторых", "колёса"))

    def text() -> None:
        for k, part in enumerate(parts):
            c.both("part_queued", 1, k, part)

    def start(k: int) -> None:
        text, length = parts[k].text, lengths[k]
        c.playout.marks[(1, k)] = (begins[k], begins[k] + length)
        c.both("part_started", 1, k, begins[k])
        spec = []
        for word in stressed[k]:
            char = text.index(word) + next(i for i, ch in enumerate(word) if ch in "аеёиоуыэюя")
            spec.append((0.16 + char / len(text) * (length - 0.16), 300.0, 0.12))
        c.feed(1, k, begins[k], vowels(spec, length))
        c.both("part_spoken", 1, k)
        c.playing = True

    try:
        c.engine.start()
        c.both("opened", Heard(c.clock.now - 1.0, c.clock.now, "Пепин"))
        at: dict[float, Any] = {
            0.3: lambda: c.both("speech_started", c.clock.now),
            1.5: lambda: c.both("speech_ended", Heard(c.clock.now - 1.2, c.clock.now,
                                                      "Где принтер?")),
            1.8: text,
            1.95: lambda: start(0),
        }  # fmt: skip
        for k in (1, 2):
            at[round(1.95 + begins[k] - 0.6, 2)] = lambda k=k: start(k)
        c.run(1.95 + sum(lengths) + 0.8, at)
        c.both("reply_done", 1)
        c.run(0.4)
        c.both("closed", "goodbye")
        c.run(0.2)
    finally:
        c.engine.close()
        threading.excepthook = hook
    assert c.engine._thread is not None and not c.engine._thread.is_alive()
    assert not broken and not [r for r in caplog.records if r.levelno >= logging.ERROR]
    (glance,) = played(c, "head", "point")
    pans = next(t.values for t in glance.tracks if t.dof == "pan_deg")
    assert 0.0 < max(pans) <= 25.0 and min(pans) >= 0.0  # toward the printer, on the left
    (pointing,) = played(c, "arm", "point")
    pan = JOINTS.index("shoulder_pan")
    ahead = BOOK.poses[BOOK.space.shut_hand or ""]["shoulder_pan"]  # the shut hand straight ahead
    turned = min(k.deg[pan] for k in pointing.knots)
    assert 20.0 <= ahead - turned <= 40.0  # inward: the printer is 30 deg left of the nose
    loci = BOOK.space.loci
    rows = [f for k, f in c.records if k == "catchment"]
    assert rows == [{"referent": "#1", "cell": loci[0], "how": "new"},
                    {"referent": "#2", "cell": loci[1], "how": "new"}]  # fmt: skip
    assert [f for k, f in c.events if k == "catchment"] == rows
    ladder = {(f["intent"], f["part"]): f for k, f in c.records if k == "ladder"}
    assert set(ladder) == {("point", 0), ("count", 1), ("count", 2)}
    variants = BOOK.compositions["point"].variants
    place = next(k for k, v in enumerate(variants) if v.needs == {"arg": "place"} and not v.floor)
    assert ladder[("point", 0)]["variant"] == place and "arm" in ladder[("point", 0)]["parts"]
    assert all(ladder[("count", k)]["variant"] == 0 for k in (1, 2))  # the arm's, not the floor
    assert not c.engine.resolver.loci  # forgotten at the close


# -- the body alive while the robot speaks: the laugh's rock, the accent's rock, the arm up ------

READY = BOOK.poses["ready"]
ACCENTS = (WEAK, STRONG, PEAK)  # the pulse's ensembles of an accent, by strength


class Spoken:
    """A reply as the voice says it: each part's text at the voice (TEXT), then its audio from
    where the part before it ends (START, all of it at once: the voice runs ahead), its length
    by the word clock's own law (0.09 s a character), a plain vowel on each word of ``plain``
    and a stressed one (higher and longer: its z over beat_z) on each word of ``stressed``; the
    texts read as the loop reads the brain's sentences (their marks taken out)."""

    def __init__(
        self,
        c: Conversation,
        turn: int,
        texts: list[str],
        stressed: list[list[str]],
        plain: list[list[str]] | None = None,
    ) -> None:
        self.c, self.turn = c, turn
        self.parts = [parse(text, vocabulary(BOOK, frozenset())).parts[0] for text in texts]
        self.texts = [part.text for part in self.parts]
        self.stressed, self.plain = stressed, plain or [[] for _ in texts]
        self.lengths = [0.09 * len(t) for t in self.texts]
        self.begins: list[float] = []
        self.started_s: float | None = None  # the engine's clock at the first part's START

    def text(self, k: int) -> None:
        self.c.both("part_queued", self.turn, k, self.parts[k])

    def start(self, k: int) -> None:
        c, text, length = self.c, self.texts[k], self.lengths[k]
        begin = self.begins[-1] + self.lengths[k - 1] if k else c.playout.now
        self.begins.append(begin)
        if k == 0:
            self.started_s = c.clock.now
        c.playout.marks[(self.turn, k)] = (begin, begin + length)
        c.both("part_started", self.turn, k, begin)
        spec = [(vowel_s(text, w, length), 400.0, 0.18) for w in self.stressed[k]]
        spec += [(vowel_s(text, w, length), 220.0, 0.075) for w in self.plain[k]]
        c.feed(self.turn, k, begin, vowels(spec, length))
        c.both("part_spoken", self.turn, k)
        c.playing = True

    def heard_end(self) -> float:
        """When the reply's last sound is heard, on the engine's clock."""
        assert self.started_s is not None
        stream = self.begins[-1] + self.lengths[-1] - self.begins[0]
        return self.started_s + stream + DATA["timing"]["audio_lag_s"]


def vowel_s(text: str, word: str, length: float) -> float:
    """Where the word clock hears the word's first vowel in a part of ``length`` seconds."""
    char = text.index(word) + next(i for i, ch in enumerate(word) if ch.lower() in "аеёиоуыэюя")
    return 0.16 + char / len(text) * (length - 0.16)


def moments(c: Conversation, name: str) -> list[dict[str, Any]]:
    return [f for k, f in c.events if k == "moment" and f["name"] == name]


def test_a_two_sentence_reply_raises_the_arm_at_its_second_sentence_and_beats_the_wrist(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Two sentences: TEXT of the first alone moves no arm; TEXT of the second raises the arm to
    the book's ready within half a second of speech and holds it (then: hold) while the reply
    goes on, sent again before its hold lapses; the second sentence's accent beats the wrist
    from the held pose (15-25 deg: the book's beats) and keeps the hold, and the cart moves
    under the raised arm (one body); every hold sent ends within 1 s of the reply's last sound,
    and the reply's end takes the arm down its own slow way then (a trajectory to rest)."""
    caplog.set_level(logging.ERROR, logger="pepin.expression")
    c = Conversation(wrist_beat_p=1.0)
    texts = ["Сегодня утром мы поехали на дачу.", "Там было очень тихо и красиво, правда."]
    reply = Spoken(c, 1, texts, [["утром"], ["красиво"]], [["мы", "дачу"], ["Там", "тихо"]])
    marks: dict[str, float] = {}

    def second_text() -> None:
        marks["text"] = c.clock.now
        reply.text(1)

    try:
        c.engine.start()
        c.both("opened", Heard(c.clock.now - 1.0, c.clock.now, "Пепин"))
        c.run(1.0, {0.2: lambda: reply.text(0), 0.3: lambda: reply.start(0)})
        assert not c.link.times, "one sentence at the voice: the arm stays down"
        c.run(reply.lengths[0] - 1.2, {0.1: second_text})
        c.run(reply.lengths[1] + 0.8, {0.1: lambda: reply.start(1)})
        end = reply.heard_end()
        c.run(max(0.0, end + 0.1 - c.clock.now))
        c.both("reply_done", 1)
        c.run(1.0)
    finally:
        c.engine.close()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    downs = [(t, tr) for name, t, tr in c.link.times if name == "send" and tr.name == "down"]
    sends = [(t, tr) for name, t, tr in c.link.times if name == "send" and tr.name != "down"]
    (raised, up), *rest = sends
    assert raised - marks["text"] <= 0.5  # within half a second of speech
    assert up.then == "hold" and up.knots[0].deg == pytest.approx(MOTION.rest(), abs=0.01)
    assert up.knots[-1].deg == pytest.approx([READY[j] for j in JOINTS], abs=0.01)
    (moment,) = moments(c, "arm_up")
    assert moment["played"] and moment["sentences"] == 2 and moment["base"] == "rest"
    beats = [tr for intent in ACCENTS for tr in played(c, "arm", intent)]
    assert beats, "the accent's wrist beat"
    flex, roll = JOINTS.index("wrist_flex"), JOINTS.index("wrist_roll")
    for beat in beats:
        assert beat.knots[0].deg == pytest.approx(up.knots[-1].deg, abs=0.01)  # the held pose
        lift = max(k.deg[flex] for k in beat.knots) - READY["wrist_flex"]  # the flex beat
        turn = max(abs(k.deg[roll] - READY["wrist_roll"]) for k in beat.knots)  # or the roll
        assert 12.0 - 0.01 <= lift <= 24.01 or 8.0 - 0.01 <= turn <= 20.01, (lift, turn)
        assert beat.then == "hold"
    assert all(tr.then == "hold" for _, tr in sends)
    for t, tr in sends:  # every hold ends within 1 s of the last sound (the board rests it then)
        assert t + 0.06 + tr.knots[-1].t_s + tr.hold_s <= end + 1.0
    assert [r for _, r in rest if r.knots[-1].t_s < 0.1], "the posture sent again, still"
    ((went, down),) = downs  # the reply's end: the slow way down, quiet like the raise
    assert went <= end + 1.0 and down.then == "rest"
    assert down.knots[-1].deg == pytest.approx(MOTION.rest(), abs=0.01)
    assert not [t for name, t, _ in c.link.times if name == "rest"]
    assert c.accents("body"), "the cart rocks under the raised arm"


def test_a_one_sentence_reply_never_raises_the_arm_and_its_strongest_accent_rocks_the_cart(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """One sentence: no arm at all (no posture, no beat without a unit); its accent (the top
    of its two stressed words, the first of them) is the reply's strongest: the nod and the
    cart's 1.2-1.5 cm rock, the rock's stroke on the accent."""
    caplog.set_level(logging.ERROR, logger="pepin.expression")
    c = Conversation(wrist_beat_p=1.0)
    text = "Знаешь, я вчера видел огромного жука прямо на нашей кухне."
    reply = Spoken(c, 1, [text], [["огромного", "кухне"]], [["Знаешь", "вчера", "жука"]])
    try:
        c.engine.start()
        c.both("opened", Heard(c.clock.now - 1.0, c.clock.now, "Пепин"))
        c.run(0.5, {0.2: lambda: reply.text(0), 0.3: lambda: reply.start(0)})
        c.run(reply.lengths[0] + 0.5)
        c.both("reply_done", 1)
        c.run(0.5)
    finally:
        c.engine.close()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert not c.link.times and not moments(c, "arm_up")
    (pulse,) = [f for k, f in c.events if k == "pulse"]
    accent = pulse["accent"]
    assert accent["kind"] == "peak" and accent["parts"] == ["body", "head"], pulse
    first_vowel = reply.begins[0] + vowel_s(text, "огромного", reply.lengths[0])
    assert accent["s"] == pytest.approx(first_vowel, abs=0.06)
    (clip,) = played(c, "body", PEAK)
    assert 0.012 <= max(clip.tracks[0].values) <= 0.015
    (sent,) = c.sent(PEAK, "body")
    lag, latency = DATA["timing"]["audio_lag_s"], 0.15
    assert sent["planned"] == pytest.approx(accent["s"] - 0.05 - clip.stroke_s + lag - latency,
                                            abs=0.03)  # fmt: skip


def sends(c: Conversation) -> list[tuple[float, Trajectory]]:
    return [(t, tr) for name, t, tr in c.link.times if name == "send" and tr is not None]


def at_ready(trajectory: Trajectory) -> bool:
    return trajectory.knots[-1].deg == pytest.approx([READY[j] for j in JOINTS], abs=0.01)


def test_the_arm_stays_up_through_a_long_reply_within_the_board_s_three_second_hold() -> None:
    """A board that rests a held arm after 3 s (hold_max_s 3): the posture is sent again within
    it while the reply goes on, so the arm is up from its rise to the reply's last sound, and
    never longer than 1 s past it."""
    c = Conversation(motion=replace(MOTION, hold_max_s=3.0), wrist_beat_p=0.0)
    texts = ["Вчера мы долго гуляли по парку у реки.", "Там было много уток и даже лебедь.",
             "Потом пошёл дождь, и мы побежали домой."]  # fmt: skip
    reply = Spoken(c, 1, texts, [[], [], []], [["долго", "реки"], ["уток", "лебедь"],
                                               ["дождь", "домой"]])  # fmt: skip
    up: list[bool] = []
    try:
        c.engine.start()
        c.both("opened", Heard(c.clock.now - 1.0, c.clock.now, "Пепин"))
        c.run(0.5, {0.1: lambda: reply.text(0), 0.15: lambda: reply.text(1),
                    0.2: lambda: reply.start(0), 0.3: lambda: reply.text(2)})  # fmt: skip
        for k in (1, 2):
            c.run(reply.lengths[k - 1] - 0.6)
            c.run(0.1, {0.05: lambda k=k: reply.start(k)})
        end = reply.heard_end()
        while c.clock.now < end - 0.1:
            c.run(0.2)
            up.append(c.parts["arm"].holding() is not None)  # type: ignore[attr-defined]
        c.run(1.2)
    finally:
        c.engine.close()
    sent = sends(c)
    assert all(tr.hold_s <= 3.0 for _, tr in sent) and len(sent) >= 3
    assert all(at_ready(tr) for _, tr in sent)
    first_up = next(i for i, x in enumerate(up) if x)
    assert all(up[first_up:]), up  # up from its rise to the end
    last_t, last = sent[-1]
    assert last_t + 0.06 + last.knots[-1].t_s + last.hold_s <= end + 1.0  # the board rests it
    assert c.parts["arm"].holding() is None  # type: ignore[attr-defined]


def test_a_mark_during_the_posture_plans_from_ready_and_the_arm_comes_back_to_ready() -> None:
    """{offer} in the second sentence of a long reply: the arm is up at ready when the mark's
    stroke comes; the offer starts from ready (the unit's base), ends held, and the posture
    takes the arm back to ready after it."""
    c = Conversation(wrist_beat_p=0.0)
    texts = [
        "Я нашёл в шкафу старую добрую книгу.",
        "Вот, возьми, пожалуйста, эту {offer} книгу себе.",
        "Она очень интересная.",
    ]
    reply = Spoken(c, 1, texts, [[], [], []], [["нашёл", "книгу"], ["возьми", "книгу"],
                                               ["очень"]])  # fmt: skip
    try:
        c.engine.start()
        c.both("opened", Heard(c.clock.now - 1.0, c.clock.now, "Пепин"))
        c.run(0.5, {0.1: lambda: reply.text(0), 0.15: lambda: reply.text(1),
                    0.2: lambda: reply.start(0), 0.3: lambda: reply.text(2)})  # fmt: skip
        for k in (1, 2):
            c.run(reply.lengths[k - 1] - 0.6)
            c.run(0.1, {0.05: lambda k=k: reply.start(k)})
        c.run(reply.lengths[2] + 0.5)
        c.both("reply_done", 1)
        c.run(0.5)
    finally:
        c.engine.close()
    sent = [tr for _, tr in sends(c)]
    offer = next(k for k, tr in enumerate(sent) if not at_ready(tr))
    assert offer > 0 and at_ready(sent[offer - 1])  # up at ready before it
    assert sent[offer].knots[0].deg == pytest.approx(sent[offer - 1].knots[-1].deg, abs=0.01)
    assert sent[offer].then == "hold"
    assert [r["base"] for k, r in c.records if k == "send" and r["intent"] == "offer"
            and r["part"] == "arm"] == ["unit"]  # fmt: skip
    back = sent[offer + 1]
    assert at_ready(back) and back.knots[0].deg == pytest.approx(sent[offer].knots[-1].deg)


def test_the_person_speaking_rests_the_raised_arm_at_once_and_it_stays_down() -> None:
    c = Conversation(wrist_beat_p=0.0)
    texts = ["Вчера мы долго гуляли по парку у реки.", "Там было много уток и даже лебедь."]
    reply = Spoken(c, 1, texts, [[], []], [["долго"], ["уток"]])
    try:
        c.engine.start()
        c.both("opened", Heard(c.clock.now - 1.0, c.clock.now, "Пепин"))
        c.run(1.5, {0.1: lambda: reply.text(0), 0.15: lambda: reply.text(1),
                    0.2: lambda: reply.start(0)})  # fmt: skip
        assert len(sends(c)) == 1
        spoke = c.clock.now
        c.both("speech_started", spoke)
        c.run(0.1)
        resting = [t for name, t, _ in c.link.times if name == "rest"]
        assert resting and resting[0] - spoke <= 0.05
        c.run(1.5, {0.25: lambda: c.both("reply_cut", 1)})
    finally:
        c.engine.close()
    assert len(sends(c)) == 1  # no second rise in that reply


# -- one body: every part in the pulse --------------------------------------------------------


def test_a_three_sentence_reply_nods_beats_the_raised_arm_and_rocks_the_cart(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Three sentences with an accent each, on the real prosody listener: the head nods, the
    arm (raised at the second sentence) beats the wrist from the held pose on every accent after
    it, and the cart rocks, all in the same reply; each accent after the first moves two parts
    at least (the head where its gap allows: the long reply's look away holds it once; the cart
    once its last rock has settled)."""
    caplog.set_level(logging.ERROR, logger="pepin.expression")
    c = Conversation()
    texts = ["Сегодня утром мы поехали далеко на дачу.",
             "Там было очень тихо и удивительно красиво.",
             "А вечером мы долго смотрели на яркие звёзды."]  # fmt: skip
    reply = Spoken(c, 1, texts, [["утром"], ["удивительно"], ["вечером"]],
                   [["мы", "дачу"], ["Там", "красиво"], ["долго", "звёзды"]])  # fmt: skip
    try:
        c.engine.start()
        c.both("opened", Heard(c.clock.now - 1.0, c.clock.now, "Пепин"))
        c.run(0.5, {0.1: lambda: reply.text(0), 0.15: lambda: reply.text(1),
                    0.2: lambda: reply.start(0), 0.3: lambda: reply.text(2)})  # fmt: skip
        for k in (1, 2):
            c.run(reply.lengths[k - 1] - 0.6)
            c.run(0.1, {0.05: lambda k=k: reply.start(k)})
        end = reply.heard_end()
        c.run(max(0.0, end + 0.1 - c.clock.now))
        c.both("reply_done", 1)
        c.run(1.0)
    finally:
        c.engine.close()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    pulses = [f for k, f in c.events if k == "pulse" and f["turn"] == 1]
    accents = [f["accent"] for f in pulses if f["accent"]]
    assert len(accents) == 3, pulses
    assert len(c.accents("head")) >= 2, "the accents nod (where the head's gap allows)"
    beats = [tr for intent in ACCENTS for tr in played(c, "arm", intent)]
    assert len(beats) >= 2 and all(tr.then == "hold" for tr in beats), "the raised arm beats"
    assert c.accents("body"), "the cart rocks in the same reply"
    assert all(len(a["parts"]) >= 2 for a in accents[1:]), accents  # one body: parts together
    skipped = [s for f in pulses for s in f["skipped"]]
    assert all("arm" not in s.split(":")[0] for s in skipped), skipped  # the raised arm beat all
    assert moments(c, "arm_up")[0]["played"]


def test_the_voice_laughing_rocks_the_cart_on_its_accents_as_the_wheels_can_follow(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A sentence that opens with ten laughing syllables 0.35 s apart (higher and longer than
    the voice's own, as the voice laughs; no {laugh}, no text read): the real prosody listener
    finds them, each is an accent of the laugh, the cart rocks on one as soon as its lead
    covers the rock's way and on another once that rock has settled (0.4 of the wheels' speed:
    a laugh's syllables are quicker than the cart); the happy face comes with each rock."""
    caplog.set_level(logging.ERROR, logger="pepin.expression")
    c = Conversation()
    text = "Ха-ха-ха-ха-ха-ха-ха-ха-ха-ха, вот это ты меня насмешил!"
    length = 5.2
    laugh = [0.4 + 0.35 * k for k in range(10)]

    def start() -> None:
        c.playout.marks[(1, 0)] = (0.0, length)
        c.both("part_started", 1, 0, 0.0)
        spec = [(s, 400.0, 0.12) for s in laugh] + [(4.2, 220.0, 0.075), (4.7, 220.0, 0.075)]
        c.feed(1, 0, 0.0, vowels(spec, length))
        c.both("part_spoken", 1, 0)
        c.playing = True

    try:
        c.engine.start()
        c.both("opened", Heard(c.clock.now - 1.0, c.clock.now, "Пепин"))
        c.run(0.4, {0.1: lambda: c.both("part_queued", 1, 0, Part(text)), 0.2: start})
        c.run(length + 1.0)
        c.both("reply_done", 1)
        c.run(0.3)
    finally:
        c.engine.close()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    sounds = moments(c, "laugh_sound")
    assert len(sounds) >= 8, sounds
    rocked = [m for m in sounds if "body" in m["parts"]]
    assert len(rocked) >= 2, sounds
    assert rocked[1]["s"] - rocked[0]["s"] >= 1.5, "the next once the first has settled"
    rocks = c.sent(LAUGHING, "body")
    assert len(rocks) == len(rocked) == len(c.sent(LAUGHING, "face"))
    lag, latency = DATA["timing"]["audio_lag_s"], 0.15
    for moment, sent in zip(rocked, rocks, strict=True):  # each rock's stroke on its accent
        clip = played(c, "body", LAUGHING)[rocks.index(sent)]
        assert sent["planned"] == pytest.approx(
            moment["s"] - 0.05 - clip.stroke_s + lag - latency, abs=0.03
        )
    assert not [f for k, f in c.records if k == "performance" and f["intent"] == "laugh"]


def test_the_raised_arm_breathes_a_slow_wrist_sway_while_nothing_else_moves_it(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A long reply with no accent: the arm up at ready sways its wrist slowly (the book's
    arm_sway, 6-10 deg of roll over 2.5-3 s) whenever nothing moves it and its hold outlasts the
    sway, from the held pose and back to it, the hold kept; never past the reply's end."""
    caplog.set_level(logging.ERROR, logger="pepin.expression")
    c = Conversation(wrist_beat_p=0.0)
    texts = ["Вчера мы долго гуляли по большому старому парку у самой реки.",
             "Там было много уток, лебедей и даже одна очень важная цапля.",
             "Потом пошёл сильный дождь, и мы быстро побежали домой."]  # fmt: skip
    reply = Spoken(c, 1, texts, [[], [], []], [["долго", "реки"], ["уток", "цапля"],
                                               ["дождь", "домой"]])  # fmt: skip
    try:
        c.engine.start()
        c.both("opened", Heard(c.clock.now - 1.0, c.clock.now, "Пепин"))
        c.run(0.5, {0.1: lambda: reply.text(0), 0.15: lambda: reply.text(1),
                    0.2: lambda: reply.start(0), 0.3: lambda: reply.text(2)})  # fmt: skip
        for k in (1, 2):
            c.run(reply.lengths[k - 1] - 0.6)
            c.run(0.1, {0.05: lambda k=k: reply.start(k)})
        end = reply.heard_end()
        c.run(max(0.0, end + 0.1 - c.clock.now))
        c.both("reply_done", 1)
        c.run(1.0)
    finally:
        c.engine.close()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    sways = played(c, "arm", ARM_SWAY)
    assert sways, "the held arm breathes"
    roll = JOINTS.index("wrist_roll")
    ready = [READY[j] for j in JOINTS]
    for sway in sways:
        assert sway.knots[0].deg == pytest.approx(ready, abs=0.01)  # from the held pose
        assert sway.knots[-1].deg == pytest.approx(ready, abs=0.01) and sway.then == "hold"
        swing = max(abs(k.deg[roll] - READY["wrist_roll"]) for k in sway.knots)
        assert 6.0 - 0.01 <= swing <= 10.0 + 0.01
        assert 2.5 - 0.01 <= sway.knots[-1].t_s <= 3.0 + 0.01
    times = [t for name, t, tr in c.link.times if name == "send" and tr in sways]
    for t, sway in zip(times, sways, strict=True):
        assert t + 0.06 + sway.knots[-1].t_s + sway.hold_s <= end + 1.0


def test_the_raised_arm_breathes_in_every_joint_under_the_book_s_layer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The same long reply with the book's breath layer: the arm up at ready carries flat
    clips of clip_s back to back (arm_breath, the idle layer's), each joint moving by the
    layer's amplitude and phase, continuous from one clip's last knot to the next's first; the
    raise from rest carries none (never on the base); the accents' beats ride a breathing arm
    and swell it; the hold kept, never past the reply's end."""
    caplog.set_level(logging.ERROR, logger="pepin.expression")
    c = Conversation(breath=True, wrist_beat_p=1.0)
    texts = ["Вчера мы долго гуляли по большому старому парку у самой реки.",
             "Там было много уток, лебедей и даже одна очень важная цапля.",
             "Потом пошёл сильный дождь, и мы быстро побежали домой."]  # fmt: skip
    reply = Spoken(c, 1, texts, [[], ["цапля"], []], [["долго", "реки"], ["уток"],
                                                     ["дождь", "домой"]])  # fmt: skip
    try:
        c.engine.start()
        c.both("opened", Heard(c.clock.now - 1.0, c.clock.now, "Пепин"))
        c.run(0.5, {0.1: lambda: reply.text(0), 0.15: lambda: reply.text(1),
                    0.2: lambda: reply.start(0), 0.3: lambda: reply.text(2)})  # fmt: skip
        for k in (1, 2):
            c.run(reply.lengths[k - 1] - 0.6)
            c.run(0.1, {0.05: lambda k=k: reply.start(k)})
        end = reply.heard_end()
        c.run(max(0.0, end + 0.1 - c.clock.now))
        c.both("reply_done", 1)
        c.run(1.0)
    finally:
        c.engine.close()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    layer = BOOK.breath["arm"]
    sends = [tr for name, _, tr in c.link.times if name == "send"]
    up = sends[0]  # the raise from rest: no breath on it
    assert up.knots[0].deg == pytest.approx(MOTION.rest(), abs=0.01)
    assert up.knots[-1].deg == pytest.approx([READY[j] for j in JOINTS], abs=0.01)
    breaths = played(c, "arm", ARM_BREATH)
    assert len(breaths) >= 2 and not played(c, "arm", ARM_SWAY)
    swell = 1.0 + sum(layer.accent_gain.values())  # the most the accents may add
    for breath in breaths:
        assert breath.then == "hold" and breath.knots[-1].t_s == pytest.approx(layer.clip_s)
        for j, joint in enumerate(JOINTS):
            travel = max(abs(k.deg[j] - READY[joint]) for k in breath.knots)
            assert travel <= layer.joints[joint].amp_deg * swell + 0.05, joint
    moved = {j for b in breaths for j, joint in enumerate(JOINTS)
             if max(abs(k.deg[j] - READY[joint]) for k in b.knots) > 0.5}  # fmt: skip
    assert moved == set(range(len(JOINTS))) - {JOINTS.index("gripper")}, "every joint breathes"
    # ... but the jaw (2026-10-14 evening: a shut jaw stands on its limit, where a breath
    # would cost the clip its breath)
    for earlier, later in itertools.pairwise(sends):
        if later.name == "down":
            continue
        assert later.knots[0].deg == pytest.approx(earlier.knots[-1].deg, abs=0.05)  # seamless
    beats = [tr for intent in ACCENTS for tr in played(c, "arm", intent)]
    assert beats, "the accent's beat rides the breathing arm"
    assert c.alive._counts[ARM_BREATH] == len(breaths)
    times = [t for name, t, tr in c.link.times if name == "send" and tr in breaths]
    for t, breath in zip(times, breaths, strict=True):
        assert t + 0.06 + breath.knots[-1].t_s + breath.hold_s <= end + 1.0


def test_the_melody_s_keys_come_a_second_apart_and_glide(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The voice's pitch as the head's still tilt, on the real prosody listener: its overlays'
    keys at least melody_key_gap_s (1.0 s since 2026-10-13 evening) apart, within
    melody_max_deg (0.9), each written with a speed cap and its own ramp (the glide), within
    the arbiter's still bound (never blinds)."""
    caplog.set_level(logging.ERROR, logger="pepin.expression")
    c = Conversation()
    text = "Мы долго шли по берегу большой и очень тихой реки, и было хорошо."
    words = ["долго", "берегу", "большой", "очень", "тихой", "реки", "хорошо"]
    reply = Spoken(c, 1, [text], [words[1::3]], [[w for w in words if w not in words[1::3]]])
    try:
        c.engine.start()
        c.both("opened", Heard(c.clock.now - 1.0, c.clock.now, "Пепин"))
        c.run(0.5, {0.2: lambda: reply.text(0), 0.3: lambda: reply.start(0)})
        c.run(reply.lengths[0] + 0.5)
        c.both("reply_done", 1)
        c.run(0.3)
    finally:
        c.engine.close()
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    melody = played(c, "head", "melody")
    assert melody, "the melody plays"
    for clip in melody:
        (track,) = clip.tracks
        assert min(np.diff(track.times)) >= 1.0 - 1e-9
        assert max(abs(v) for v in track.values) <= 0.9 + 1e-9
        look = clip_look(clip, IDLE_LOOK)
        assert look is not None and look["target"].get("idle") is True
        for key in look["target"]["keys"]:
            assert key["speed_deg_s"] > 0.0 and key["acc_deg_s2"] > 0.0  # a cap and a glide
