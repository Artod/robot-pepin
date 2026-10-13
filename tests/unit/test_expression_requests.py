"""The gesture engine's contract for co-speech gesticulation: requests on every anchor, one
priority table per part, the idle layer's states, attention first, the arm's planned stroke,
the close that lets a farewell finish, and the book's new fields read with defaults."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import json
from pathlib import Path
from typing import Any

import pytest

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import Book, Clip, Idle, Performance, Range, Track
from pepin.expression.animator import ARM_REST_S, EngineConfig, GestureEngine
from pepin.expression.book import book_from, load_book, with_rest
from pepin.expression.conversation import ACKNOWLEDGED, GREETING_DUE, Heard, moment_rank
from pepin.expression.fakes import FakeArmLink, FakeAttention, FakeExecutor, FakePlayout
from pepin.expression.markers import Marker, Part, cues
from pepin.expression.parts import Absent, ArmPart
from pepin.expression.schedule import (
    AtStream,
    AtTime,
    AtWord,
    Priority,
    Request,
    Reservation,
    Reservations,
)
from pepin.expression.speaker import Attention
from pepin.expression.trajectory import ArmStatus

CONFIG = Path(__file__).resolve().parents[2] / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
BOOK = with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg)
NOD = Clip("head", -0.2, 0.2, 0.4, (Track("tilt_deg", "additive", (0.0, 0.2, 0.4), (0, 3, 0)),))


class Rig:
    """An engine over fake parts, a fake player, a fake clock and a fake attention."""

    def __init__(self, book: Book = BOOK, **knobs: Any) -> None:
        self.now = 1000.0
        self.playout = FakePlayout()
        self.parts = {p: FakeExecutor(p) for p in ("arm", "head", "face")}  # type: ignore[arg-type]
        self.parts["arm"].latency_s, self.parts["head"].latency_s = 0.06, 0.1
        self.records: list[tuple[str, dict[str, Any]]] = []
        self.view = FakeAttention()
        config = EngineConfig(arm_speed_scale=1.0, audio_lag_s=0.15, **knobs)
        self.engine = GestureEngine(
            {**self.parts, "body": Absent("body", "no wheels")},  # type: ignore[dict-item]
            self.playout, lambda: (book, config), on=True, clock=lambda: self.now,
            record=lambda kind, fields: self.records.append((kind, fields)), attention=self.view,
        )  # fmt: skip

    def played(self, part: str) -> list[Clip]:
        return [args[0] for name, args in self.parts[part].calls if name == "play"]

    def cues(self, part: str, ok: bool | None = None) -> list[dict[str, Any]]:
        rows = [f for k, f in self.records if k == "cue" and f["part"] == part]
        return [r for r in rows if ok is None or r["ok"] == ok]

    def tick(self, seconds: float) -> None:
        for _ in range(round(seconds / 0.01)):
            self.now += 0.01
            self.playout.now += 0.01
            self.engine.step()


def with_arm(intent: str) -> Rig:
    """A rig that has just performed ``intent`` by a variant that moves the arm."""
    for seed in range(50):
        rig = Rig(seed=seed)
        assert rig.engine.perform(intent, source="tool").ok
        last = rig.engine.last()
        if last is not None and any(c.part == "arm" for c in last.clips):
            return rig
    raise AssertionError(f"no variant of {intent} moves the arm")


def test_today_s_book_loads_with_the_new_fields_at_their_defaults_and_reads_them() -> None:
    greeting = BOOK.compositions["greeting"]
    assert (greeting.bind, greeting.args, greeting.family, greeting.long) == (
        "utterance", (), None, False,
    )  # fmt: skip
    variants = [v for c in BOOK.compositions.values() for v in c.variants]
    assert all(v.needs == {} and not v.floor for v in variants)
    assert BOOK.idle[0].unless == ("speaking", "listening")
    data = json.loads((CONFIG / "animation.json").read_text())
    data["compositions"]["point"] = {
        "bind": "word", "args": ["you", "*"], "family": "point", "long": False,
        "variants": [{"needs": {"arm": "any"}, "steps": [{"atom": "arm_beat"}]},
                     {"floor": True, "steps": [{"atom": "head_nod"}]}],
    }  # fmt: skip
    data["idle"].append({"part": "head", "atoms": ["breath"], "every_s": 1, "unless": "speaking"})
    book = book_from(data)
    point = book.compositions["point"]
    assert point.bind == "word" and point.args == ("you", "*") and point.family == "point"
    assert point.variants[0].needs == {"arm": "any"} and point.variants[1].floor
    assert book.idle[-1].unless == ("speaking",)
    data["compositions"]["point"]["bind"] = "sentence"
    with pytest.raises(ValueError, match=r"point\.bind"):
        book_from(data)


def test_a_cue_knows_its_word_s_span_and_first_vowel() -> None:
    (cue,) = cues(1, 0, Part("Это было огромное здание.", None, (Marker("gesture", "x", 9),)))
    assert cue.word == (9, 17) and cue.vowel == 9  # the word after the marker, and its vowel
    (end,) = cues(1, 0, Part("Вот и всё", None, (Marker("gesture", "x", 9),)))
    assert end.word == (9, 9) and end.vowel == 9  # no word after it


def test_the_priority_table_one_request_holds_a_part_at_a_time() -> None:
    book, now = Reservations(), 10.0

    def hold(priority: Priority, rank: int = 0) -> Reservation:
        clip = Clip("head", 0.0, 0.5, 1.0)
        performance = Performance("x", 0, 0, ())
        return Reservation(performance, clip, now, now + 0.5, now + 1.0, priority, rank)

    assert book.reserve("head", hold(Priority.MOMENT), now) is None
    assert book.refusal("head", Priority.PULSE, 0, now) is not None  # lower: only a free part
    assert book.refusal("head", Priority.MARK, 0, now + 0.2) is None  # higher, before the stroke
    assert "past the stroke" in str(book.refusal("head", Priority.MARK, 0, now + 0.6))
    assert book.refusal("head", Priority.ASK, 0, now + 0.6) is None  # a person's ask: now
    assert book.refusal("head", Priority.PULSE, 0, now + 1.0) is None  # free again
    assert book.release("head") is not None and book.holder("head", now) is None
    assert book.reserve("head", hold(Priority.IDLE), now) is None
    assert book.refusal("head", Priority.PULSE, 0, now) is None  # the idle layer gives way
    assert book.refusal("head", Priority.IDLE, 0, now) is None  # a newer idle clip replaces it
    assert book.reserve("head", hold(Priority.MOMENT, moment_rank(ACKNOWLEDGED)), now) is None
    assert book.refusal("head", Priority.MOMENT, moment_rank(GREETING_DUE), now) is None
    assert book.reserve("head", hold(Priority.MOMENT, moment_rank(GREETING_DUE)), now) is None
    assert book.refusal("head", Priority.MOMENT, moment_rank(ACKNOWLEDGED), now) is not None
    last = book.last("head")
    assert last is not None and last.rank == moment_rank(GREETING_DUE)


def test_requests_go_at_their_word_their_stream_second_and_their_time() -> None:
    rig = Rig(seed=3)
    rig.engine.part_queued(1, 0, Part("Ну что ж, привет всем!"))
    melody = Request("melody", AtStream(2.0, turn=1), Priority.IDLE, "pulse", clips=(NOD,))
    rig.engine.request(melody)
    rig.engine.request(Request("agreement", AtWord(1, 0, 10), Priority.MARK, "marker"))
    rig.engine.request(Request("nod", AtTime(rig.now + 1.0), Priority.PULSE, "pulse", clips=(NOD,)))
    rig.engine.request(Request("early", AtWord(9, 0, 0), Priority.MARK, "marker"))
    rig.engine.step()
    unplaced = [f["why"] for k, f in rig.records if k == "unplaced"]
    assert unplaced == ["its part is not with the voice"]
    rig.tick(0.68)  # the nod's stroke at +1.0: it starts 0.2 before, sent 0.1 earlier still
    assert not rig.cues("head")
    rig.tick(0.03)
    (timed,) = rig.cues("head")
    assert timed["intent"] == "nod" and timed["ok"]
    rig.playout.marks[(1, 0)] = (1.0, 2.5)
    rig.engine.part_spoken(1, 0)
    rig.tick(2.0)
    played = next(c for c in rig.cues("head") if c["intent"] == "melody")
    assert played["planned"] == pytest.approx(2.0 - 0.2 + 0.15 - 0.1)  # its second, as a word's
    assert any(c["intent"] == "agreement" for c in rig.cues("head") + rig.cues("face"))
    rig.engine.request(Request("nod", AtStream(9.0, turn=1), Priority.PULSE, "pulse", clips=(NOD,)))
    rig.engine.reply_cut(1)
    rig.tick(0.1)
    assert [f["intent"] for k, f in rig.records if k == "dropped"] == ["nod"]


def test_a_pulse_never_takes_the_head_from_a_marker_and_a_tool_always_does() -> None:
    rig = Rig(seed=1)
    assert rig.engine.perform("agreement", source="tool").ok
    rig.engine.request(Request("nod", None, Priority.PULSE, "pulse", clips=(NOD,)))
    rig.engine.step()
    (refused,) = rig.cues("head", ok=False)
    assert refused["intent"] == "nod" and "goes first" in refused["why"]
    assert rig.engine.perform("emphasis", source="tool").ok


def test_an_idle_entry_plays_through_the_states_it_allows_and_fades_at_those_it_excludes() -> None:
    breath = Idle("head", ("breath",), Range(0.5, 0.5), "conversation", ("speaking",))
    book = Book(BOOK.atoms, BOOK.compositions, BOOK.poses, (breath,), BOOK.moments)
    rig = Rig(book, idle_after_speech_s=0.0)
    rig.engine.opened(Heard(0.0, 1.0))
    rig.engine.speech_started(rig.now)
    rig.tick(1.0)
    (first,) = rig.played("head")
    assert first.layer == "idle" and first.loop, "the breath while the person speaks"
    rig.tick(3.0)
    assert len(rig.played("head")) == 1, "a loop is sent once and plays until withdrawn"
    rig.engine.part_queued(1, 0, Part("Да."))
    rig.tick(0.05)
    assert ("fade", ("idle", 0.35)) in rig.parts["head"].calls, "the robot speaks: it fades"


def test_attention_keeps_the_head_and_the_arm_clear() -> None:
    rig = Rig(seed=2)
    rig.view.be(phase="seek", eyes=True)
    rig.engine.perform("agreement", source="tool")
    assert rig.cues("head", ok=False)[-1]["why"] == "attention: seek"
    rig.view.be(phase="track", seen_s=rig.now - 1.5)
    rig.engine.perform("agreement", source="tool")
    assert "unseen for 1.5 s" in rig.cues("head", ok=False)[-1]["why"]
    rig.view.be(seen_s=rig.now)
    assert rig.engine.perform("agreement", source="tool").ok and rig.cues("head", ok=True)
    rig.tick(0.05)
    rig.view.be(phase="check")  # attention checks a new voice: the gesture leaves the head
    rig.tick(0.15)
    assert ("fade", ("gesture", 0.35)) in rig.parts["head"].calls
    assert next(f for k, f in rig.records if k == "faded")["why"] == "attention: check"
    rig.view.view = Attention("voice")  # no eyes: nothing to keep clear for the head
    assert rig.engine.perform("agreement", source="tool").ok and rig.cues("head", ok=True)


def test_no_arm_motion_starts_while_the_person_stands_past_the_arc_or_the_body_turns() -> None:
    for view in (Attention("track", past_arc=True), Attention("turning")):
        for seed in range(50):
            rig = Rig(seed=seed)
            rig.view.view = view
            rig.engine.perform("greeting", source="tool")
            rig.tick(2.0)
            if rig.cues("arm"):
                break
        assert not rig.played("arm"), view
        why = rig.cues("arm", ok=False)[-1]["why"]
        assert "past the forward arc" in why or "body turns" in why


def test_a_farewell_past_its_stroke_plays_out_when_the_goodbye_closes_the_conversation() -> None:
    """A goodbye answered in 0.5 s: «Пока!» plays and the conversation closes while the wave is
    in the air; past its stroke and ending within finish_s it finishes, the rest goes to rest."""
    rig = with_arm("farewell")
    rig.tick(3.0)
    playing = rig.engine._playing["arm"]
    rig.now = playing.ends - 0.5
    rig.engine.closed("goodbye")
    rig.engine.step()
    assert ("rest", ()) not in rig.parts["arm"].calls and ("rest", ()) in rig.parts["head"].calls
    early = with_arm("farewell")
    early.engine.closed("goodbye")
    early.engine.step()
    assert ("rest", ()) in early.parts["arm"].calls, "before its stroke it rests with the rest"


def test_the_body_s_turn_takes_the_arm_and_no_arm_starts_until_it_is_at_rest() -> None:
    assert Rig().engine.release_arm("the body turns") == 0.0, "nothing holds the arm"
    rig = with_arm("greeting")
    rig.tick(0.1)
    assert rig.played("arm")
    seconds = rig.engine.release_arm("the body turns")
    assert seconds == pytest.approx(ARM_REST_S + 0.06)
    assert ("fade", ("gesture", 0.35)) in rig.parts["arm"].calls
    sent = len(rig.played("arm"))
    for _ in range(5):
        rig.engine.perform("greeting", source="tool")
    rig.tick(0.5)
    assert len(rig.played("arm")) == sent
    assert "rest for the body's turn" in rig.cues("arm", ok=False)[-1]["why"]


def test_the_arm_s_plan_says_when_its_stroke_lands_and_a_unit_plans_from_the_held_pose() -> None:
    here = MOTION.rest()
    link = FakeArmLink(state=ArmStatus(1.0, here, True, "holding"))
    arm = ArmPart(link, MOTION)
    flick = Track("wrist_flex", "additive", (0.0, 0.1, 0.2), (0.0, -30.0, 0.0))
    fast = arm.plan(Clip("arm", 0.0, 0.1, 0.2, (flick,)))
    assert not isinstance(fast, str) and fast.approach_s == 0.0 and fast.stretch > 1.0
    assert fast.stroke_s == pytest.approx(0.1 * fast.stretch)
    link.state = ArmStatus(1.0, (here[0] + 20.0, *here[1:]), True, "holding")
    slow = Track("wrist_flex", "additive", (0.0, 0.5, 1.0), (0.0, -10.0, 0.0))
    away = arm.plan(Clip("arm", 0.0, 0.5, 1.0, (slow,)))
    assert not isinstance(away, str) and away.approach_s > 0.1 and away.stretch == 1.0
    assert away.stroke_s == pytest.approx(away.approach_s + 0.5)
    link.state = ArmStatus(1.0, here, True, "off")
    raise_ = Track("shoulder_lift", "additive", (0.0, 0.5, 1.0), (0.0, 20.0, 20.0))
    opened = arm.play(Clip("arm", 0.0, 0.5, 1.0, (raise_,), then="hold", hold_s=1.5))
    (_, (sent,)) = link.calls[-1]
    assert opened.ok and opened.stroke_s == pytest.approx(0.5)
    assert sent.then == "hold" and sent.hold_s == 1.5
    lifted = MOTION.rest_deg["shoulder_lift"] + 20.0
    assert arm.held is not None and arm.held["shoulder_lift"] == pytest.approx(lifted)
    assert arm.play(Clip("arm", 0.0, 0.5, 1.0, (slow,))).ok  # the unit's last stroke
    (_, (last,)) = link.calls[-1]
    assert last.knots[0].deg[1] == pytest.approx(lifted) and last.then == "rest"
    assert arm.held is None
