"""The gesture engine's core: a looping idle clip held and renewed on a thread that never waits
forever, an idle entry's own quiet after speech, the table every hold goes through (and
remembers), a requester ending its own move, what waits seen by the listeners, the tail bound
after its second, the arm counted on the ladder's final version, the knobs the scheduler gets,
and the person seen within head_seen_s."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import Book, Clip, Idle, PartName, Range, Track
from pepin.expression.animator import (
    MAX_WAIT_S,
    Caps,
    EngineConfig,
    GestureEngine,
    bounded,
    engine_config,
)
from pepin.expression.book import book_from, load_book, with_rest
from pepin.expression.conversation import Heard
from pepin.expression.executors import Executor
from pepin.expression.fakes import FakeArmLink, FakeAttention, FakeExecutor, FakePlayout
from pepin.expression.markers import Marker, Part
from pepin.expression.parts import ArmPart
from pepin.expression.schedule import AtStream, AtTime, Priority, Request, timing_from
from pepin.expression.speaker import HEAD_SEEN_S
from pepin.expression.trajectory import ArmStatus

CONFIG = Path(__file__).resolve().parents[2] / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
BOOK = with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg)
DATA = json.loads((CONFIG / "expression.json").read_text())
AWAY = Track("tilt_deg", "additive", (0.0, 0.35, 2.6, 3.0), (0.0, -10.0, -10.0, 0.0))
GLANCE = Clip("head", 0.0, 0.35, 3.0, (AWAY,))
ROCK = Clip("body", 0.0, 1.2, 2.4, (Track("x_m", "additive", (0.0, 1.2, 2.4), (0.0, 0.01, 0.0)),))
NOD = Clip("head", 0.0, 0.2, 0.4, (Track("tilt_deg", "additive", (0.0, 0.2, 0.4), (0, 3, 0)),))


class Rig:
    """The engine over fake parts (head 0.1 s, arm 0.06, face 0.05, the wheels at once), a
    fake player and a fake clock; the thread runs only when a test starts it."""

    def __init__(self, book: Book = BOOK, **knobs: Any) -> None:
        self.now = 1000.0
        self.playout = FakePlayout(now=1.0)
        latency = {"arm": 0.06, "head": 0.1, "face": 0.05, "body": 0.0}
        self.parts: dict[PartName, FakeExecutor] = {
            p: FakeExecutor(p, latency[p])  # type: ignore[arg-type]
            for p in ("arm", "head", "face", "body")
        }
        self.records: list[tuple[str, dict[str, Any]]] = []
        self.view = FakeAttention()
        config = EngineConfig(arm_speed_scale=1.0, body_speed_scale=1.0, audio_lag_s=0.15,
                              **knobs)  # fmt: skip
        parts: dict[PartName, Executor] = dict(self.parts)
        self.engine = GestureEngine(
            parts, self.playout, lambda: (book, config), on=True, clock=lambda: self.now,
            record=lambda kind, fields: self.records.append((kind, fields)),
            seeds=random.Random(1), attention=self.view,
        )  # fmt: skip

    def calls(self, part: PartName, name: str) -> list[tuple[Any, ...]]:
        return [args for n, args in self.parts[part].calls if n == name]

    def played(self, part: PartName) -> list[Clip]:
        return [args[0] for args in self.calls(part, "play")]

    def rows(self, kind: str, **match: Any) -> list[dict[str, Any]]:
        rows = [f for k, f in self.records if k == kind]
        return [r for r in rows if all(r.get(k) == v for k, v in match.items())]

    def tick(self, seconds: float, step_s: float = 0.01) -> None:
        for _ in range(round(seconds / step_s)):
            self.now += step_s
            self.playout.now += step_s
            self.engine.step()


def breathing(after: float | None = 0.0) -> Book:
    """The book with the breath as a looping idle entry of the head (no breath while the robot
    speaks) beside the drift."""
    breath = Idle("head", ("breath",), Range(0.5, 0.5), "conversation", ("speaking",), after)
    return Book(BOOK.atoms, BOOK.compositions, BOOK.poses, (breath, *BOOK.idle), BOOK.moments)


def test_a_looping_idle_clip_holds_the_head_renewed_once_a_period_until_the_robot_speaks() -> None:
    """The breath loops at the arbiter, whose TTL is one period and a second: the engine sends it
    once, renews it every period (its three breaths, 12 s) while it holds the head (the drift
    waits meanwhile, 20-60 s), never plans an endless wait, and fades it when the robot
    speaks."""
    rig = Rig(breathing())
    rig.engine.opened(Heard(0.0, 1.0))
    rig.tick(1.0)
    (loop,) = rig.played("head")
    assert loop.loop and loop.layer == "idle"
    wait = rig.engine.step()
    assert wait is not None and math.isfinite(wait) and wait <= loop.duration_s
    rig.tick(2 * loop.duration_s + 0.1, step_s=0.05)
    assert rig.calls("head", "renew") == [("idle",), ("idle",)]
    rig.tick(60.0, step_s=0.05)
    assert len(rig.played("head")) == 1, "a loop is sent once; the drift waits while it holds"
    rig.engine.part_queued(1, 0, Part("Да."))
    rig.tick(0.05)
    assert ("idle", 0.35) in rig.calls("head", "fade")
    renewed = len(rig.calls("head", "renew"))
    rig.tick(30.0, step_s=0.05)
    assert len(rig.calls("head", "renew")) == renewed, "let go: no more renewals"
    assert bounded(math.inf) == MAX_WAIT_S and bounded(None) is None and bounded(-1.0) == 0.0


def test_the_engine_s_thread_lives_through_a_looping_idle_clip() -> None:
    """A.1's crash: the loop's endless hold became an endless wait and Condition.wait raised
    OverflowError on the engine's thread. Each call below is answered from the thread."""
    rig = Rig(breathing())
    rig.engine.start()
    try:
        rig.engine.opened(Heard(0.0, 1.0))
        assert "no gesture" in rig.engine.perform("-", source="test").why  # opened on the thread
        rig.now += 1.0  # past the breath's draw: the next step sends it
        for _ in range(3):
            assert "no gesture" in rig.engine.perform("-", source="test").why
        assert [c.loop for c in rig.played("head")] == [True]
    finally:
        rig.engine.close()
    assert rig.engine._thread is not None and not rig.engine._thread.is_alive()


def test_an_idle_entry_s_own_after_speech_s_lets_a_breath_follow_a_reply_at_once() -> None:
    def after_reply(rig: Rig) -> list[Clip]:
        rig.engine.opened(Heard(0.0, 1.0))
        rig.engine.part_queued(1, 0, Part("Да."))
        rig.tick(0.5)
        rig.engine.reply_done(1)
        rig.tick(1.0)
        return [c for c in rig.played("head") if c.layer == "idle"]

    assert after_reply(Rig(breathing(after=0.0), idle_after_speech_s=5.0))
    assert not after_reply(Rig(breathing(after=None), idle_after_speech_s=5.0))
    data = json.loads((CONFIG / "animation.json").read_text())
    data["idle"] = [{"part": "head", "atoms": ["breath"], "every_s": 1, "after_speech_s": 0}]
    assert book_from(data).idle[0].after_speech_s == 0.0
    data["idle"][0]["after_speech_s"] = -1
    with pytest.raises(ValueError, match=r"idle\[0\]\.after_speech_s"):
        book_from(data)


def test_every_hold_goes_through_the_table_and_is_remembered_after_it_lets_go() -> None:
    rig = Rig()
    assert rig.engine.perform("agreement", source="tool").ok
    reservations = rig.engine.scheduler.reservations
    head = reservations.last("head")
    assert head is not None and head.performance.intent == "agreement"
    assert head.request is not None and head.request.source == "tool"
    rig.engine.hush("the person speaks")
    rig.engine.step()
    assert "head" not in reservations.held and reservations.last("head") is head


def test_a_requester_ends_its_own_held_move_and_what_it_still_waits_for() -> None:
    rig = Rig()
    glance = Request("think_glance", None, Priority.MOMENT, "moment", clips=(GLANCE,))
    later = Request("think_glance", AtTime(rig.now + 5.0), Priority.MOMENT, "moment",
                    clips=(GLANCE,))  # fmt: skip
    rig.engine.request(glance)
    rig.engine.request(later)
    rig.engine.step()
    assert rig.played("head") == [GLANCE]
    twin = Request("think_glance", None, Priority.MOMENT, "moment", clips=(GLANCE,))
    assert twin == glance
    rig.engine.release(twin, "not this one")  # an equal request is another request
    rig.engine.step()
    assert not rig.calls("head", "fade")
    rig.engine.release(glance, "the reply's voice")
    rig.engine.step()
    assert rig.calls("head", "fade") == [("gesture", 0.35)]
    assert rig.rows("faded", intent="think_glance", why="the reply's voice")
    assert [p.intent for p in rig.engine.pending("head")] == ["think_glance"]  # the later one
    rig.engine.release(later, "not wanted")
    rig.engine.step()
    assert rig.engine.pending() == [] and rig.rows("dropped", why="not wanted")


def test_the_marks_waiting_for_their_words_are_seen_by_part_and_reply() -> None:
    rig = Rig()
    text = "Принтер вон там, у стены."
    rig.engine.part_queued(1, 0, Part(text, None, (Marker("gesture", "agreement", 8),)))
    rig.engine.step()
    (head,) = rig.engine.pending("head", turn=1)
    assert head.intent == "agreement" and head.priority == Priority.MARK and head.source == "marker"
    assert (head.turn, head.index) == (1, 0) and head.start <= head.stroke <= head.end
    waiting = next(w for w in rig.engine.scheduler.waiting if w.clip.part == "head")
    assert waiting.fit is not None and head.stroke == pytest.approx(waiting.fit.stroke)
    assert rig.engine.pending("head", turn=2) == []
    assert {p.part for p in rig.engine.pending(turn=1)} >= {"head"}


def test_a_tail_starts_at_its_second_and_goes_while_its_stroke_lands_within_after_s() -> None:
    """The laugh's rock bound after its sentence's start: it starts when that second is heard,
    goes late only while its stroke still lands within after_s (2.5 s) of it inside the reply's
    speech; one whose stroke alone is longer is dropped when placed."""

    def rig_with(text: str, length: float) -> Rig:
        rig = Rig()
        rig.engine.part_queued(1, 0, Part(text))
        rig.playout.marks[(1, 0)] = (2.0, 2.0 + length)
        rig.engine.part_started(1, 0, 2.0)
        rig.engine.part_spoken(1, 0)
        return rig

    long = "Ха-ха, вот это да, смешно получилось у нас с тобой!"
    rig = rig_with(long, 4.0)
    rig.engine.request(Request("laugh_rock", AtStream(2.0, 1), Priority.MOMENT, "moment",
                               bind="after", clips=(ROCK,)))  # fmt: skip
    rig.tick(1.2)
    (cue,) = rig.rows("cue", part="body")
    assert cue["ok"] and cue["planned"] == pytest.approx(2.15)  # its second heard, offset aside
    late = rig_with(long, 4.0)
    late.playout.now = 2.15 + (2.5 - 1.2) + 0.05  # its stroke would land 2.55 s after its start
    late.engine.request(Request("laugh_rock", AtStream(2.0, 1), Priority.MOMENT, "moment",
                                bind="after", clips=(ROCK,)))  # fmt: skip
    late.tick(0.05)
    assert not late.played("body") and late.rows("late", intent="laugh_rock")
    short = rig_with("Ха.", 0.4)
    short.engine.request(Request("laugh_rock", AtStream(2.0, 1), Priority.MOMENT, "moment",
                                 bind="after", clips=(ROCK,)))  # fmt: skip
    short.tick(1.2)
    assert not short.played("body") and short.rows("late", why="past the speech")
    slow = replace(ROCK, stroke_s=3.0, duration_s=6.0)
    rig.engine.request(Request("laugh_rock", AtStream(5.0, 1), Priority.MOMENT, "moment",
                               bind="after", clips=(slow,)))  # fmt: skip
    rig.engine.step()
    assert "past after_s" in rig.rows("dropped", intent="laugh_rock")[0]["why"]


def test_the_arm_is_counted_on_the_ladder_s_final_version() -> None:
    """With one arm gesture a minute: a hello drawn with the raised hand on the reply's first word
    has no time to raise it and the ladder plays its smaller version (face and nod), which uses
    no arm, so the next mark keeps its arm."""
    rig = Rig(caps=Caps(per_part=1, per_reply=3, arm_per_minute=1), seed=4)
    arm = ArmPart(FakeArmLink(state=ArmStatus(0.0, MOTION.rest(), False, "off")), MOTION,
                  latency_s=0.06, clock=lambda: rig.now)  # fmt: skip
    rig.engine.parts["arm"] = arm
    rig.engine.scheduler.attach(rig.engine.parts, lambda: rig.engine.renderer,
                                hooks=rig.engine.scene_hooks)  # fmt: skip
    rig.engine.part_queued(1, 0, Part("Привет.", None, (Marker("gesture", "greeting", 0),)))
    rig.engine.step()
    (drawn,) = rig.rows("performance", intent="greeting")
    (first,) = rig.rows("ladder")
    assert "arm" in drawn["parts"] and "arm" not in first["parts"], (drawn, first)
    text = "Ну и вот, всё у нас получилось, как мы хотели, правда здорово."
    rig.engine.part_queued(2, 0, Part(text, None, (Marker("gesture", "agreement", 10),)))
    rig.engine.step()
    assert not rig.rows("capped", part="arm"), "the floor used no arm"
    assert "arm" in rig.rows("performance", intent="agreement")[0]["parts"]


def test_the_scheduler_gets_the_file_s_timing_and_units_and_the_head_seen_knob() -> None:
    config = engine_config(DATA)
    assert config.timing == timing_from(DATA) and config.timing.after_s == 2.5
    assert config.head_seen_s == HEAD_SEEN_S == 1.2
    assert engine_config({}) == EngineConfig()
    engine = GestureEngine({}, FakePlayout(), lambda: (BOOK, config))
    assert engine.scheduler.timing == config.timing


def test_a_gesture_takes_the_head_of_a_person_seen_within_head_seen_s() -> None:
    def nod(rig: Rig) -> dict[str, Any]:
        rig.view.be(phase="track", eyes=True, seen_s=rig.now - 1.1)  # one 1 Hz frame and more
        rig.engine.request(Request("nod", None, Priority.ASK, "tool", clips=(NOD,)))
        rig.engine.step()
        return rig.rows("cue", part="head")[-1]

    assert nod(Rig())["ok"]
    assert "unseen for 1.1 s" in nod(Rig(head_seen_s=1.0))["why"]
