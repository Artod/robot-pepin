"""One reply end to end through the gesture engine's own thread, the alive listeners on the real
prosody listener, and the word scheduler lent the engine's parts, renderer, hooks and nuclei:
the person's pause with their words, the first face, TEXT, START and AUDIO of two sentences
with a {greeting}, the reply's end and the close."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import json
import logging
import random
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import PartName
from pepin.expression.animator import GestureEngine, engine_config
from pepin.expression.book import load_book, with_rest
from pepin.expression.conversation import LISTENER_PAUSE, Heard, VoiceChunk
from pepin.expression.executors import Executor
from pepin.expression.fakes import FakeArmLink, FakeAttention, FakeClock, FakeExecutor, FakePlayout
from pepin.expression.listeners import Alive, alive_knobs
from pepin.expression.markers import Marker, Part
from pepin.expression.parts import ArmPart
from pepin.expression.prosody import ProsodyListener
from pepin.expression.speaker import Attention
from pepin.expression.trajectory import ArmStatus

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


class Conversation:
    """The engine on its own thread over the arm's real planner on a fake board and fake head,
    face and wheels; the alive listeners (stepped here) on the real prosody listener; the word
    scheduler lent the engine's parts, renderer, hooks and the voice's nuclei; one fake clock and
    player, moved by the test, the engine's thread brought level with them at every step."""

    def __init__(self) -> None:
        self.clock = FakeClock(1000.0)
        self.playout = FakePlayout(now=0.0)
        self.playing = False
        self.link = FakeArmLink(state=ArmStatus(0.0, MOTION.rest(), False, "off"))
        arm = ArmPart(self.link, MOTION, latency_s=0.06, clock=self.clock)
        self.parts: dict[PartName, Executor] = {
            "arm": arm, "head": FakeExecutor("head", 0.1), "face": FakeExecutor("face", 0.05),
            "body": FakeExecutor("body", 0.15),
        }  # fmt: skip
        self.records: list[tuple[str, dict[str, Any]]] = []
        self.events: list[tuple[str, dict[str, Any]]] = []
        self.view = FakeAttention(Attention("track", 0.0, eyes=True, seen_s=self.clock.now))
        config = replace(engine_config(DATA), seed=3)
        self.engine = GestureEngine(
            self.parts, self.playout, lambda: (BOOK, config), on=True, clock=self.clock,
            record=lambda kind, fields: self.records.append((kind, fields)),
            seeds=random.Random(3), attention=self.view,
        )  # fmt: skip
        knobs = alive_knobs(DATA)
        moments = replace(knobs.moments, listener_nod_p=1.0, ack_nod_p=0.0, think_p=0.0)
        knobs = replace(knobs, moments=moments)
        self.prosody = ProsodyListener()
        self.alive = Alive(
            self.engine, self.prosody, knobs=lambda: knobs, clock=self.clock, hour=lambda: 14,
            seed=1, journal=lambda kind, now, **f: self.events.append((kind, f)),
        )  # fmt: skip
        self.engine.taps.append(self.prosody)
        self.engine.scene_hooks.append(self.alive.scene)
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
    assert pulses[(1, 1)]["accent"]["kind"] == "nod" and len(c.sent("accent", "head")) == 1
    assert len(c.sent("listen_nod", "head")) == 1
    (ladder,) = [f for k, f in c.records if k == "ladder" and f["intent"] == "greeting"]
    landed = {f["part"] for f in c.sent("greeting")}
    assert set(ladder["parts"]) <= landed, (ladder, landed)
    if "arm" in ladder["parts"]:
        assert [name for name, _ in c.link.calls].count("send") >= 1
    sends = [f for k, f in c.records if k == "send" and f["intent"] == "greeting"]
    assert sends and all(-0.3 <= f["error_s"] <= 0.5 for f in sends)
