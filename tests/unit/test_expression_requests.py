"""The gesture engine's contract for co-speech gesticulation: requests on every anchor, one
priority table per part, the idle layer's states, attention first, the arm's planned stroke,
the close that lets a farewell finish, and the book's new fields; then ON THE WORD: the ladder
by the computed lead, the last-moment send, never a late word stroke, never in a pause, units,
long compositions only with the speech to cover them, barge-in, the token stream's lead, the
bookings, and the feelings' arm lengths."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import PARTS, Book, Clip, Idle, Performance, Range, Scene, Track
from pepin.expression.animator import ARM_REST_S, EngineConfig, GestureEngine
from pepin.expression.book import book_from, load_book, with_rest
from pepin.expression.conversation import ACKNOWLEDGED, GREETING_DUE, Heard, moment_rank
from pepin.expression.fakes import (
    FakeArmLink,
    FakeAttention,
    FakeExecutor,
    FakePlaces,
    FakePlayout,
)
from pepin.expression.markers import Marker, Part, cues
from pepin.expression.mix import PREP_CAPS, prep_min_s
from pepin.expression.parts import Absent, ArmPart
from pepin.expression.render import BookRenderer
from pepin.expression.schedule import (
    AtStream,
    AtTime,
    AtWord,
    Priority,
    Request,
    Reservation,
    Reservations,
    Timing,
    timing_from,
)
from pepin.expression.speaker import Attention
from pepin.expression.trajectory import JOINTS, ArmStatus
from pepin.tools.gesture import GESTURES as TWELVE

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


def test_the_book_tags_the_twelve_intents_and_reads_the_new_fields() -> None:
    greeting = BOOK.compositions["greeting"]
    assert (greeting.bind, greeting.args, greeting.family, greeting.long) == (
        "utterance", (), "greeting", True,
    )  # fmt: skip
    assert BOOK.compositions["emphasis"].bind == "word"
    for composition in BOOK.compositions.values():
        if composition.name not in TWELVE:
            continue  # the brain's primitives: a floor for each argument (test_expression_space)
        assert [v.floor for v in composition.variants].count(True) == 1, composition.name
        assert all(v.needs == {} for v in composition.variants)
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
    rig.engine.request(Request("nod", AtTime(rig.now + 0.8), Priority.PULSE, "pulse", clips=(NOD,)))
    rig.engine.request(Request("early", AtWord(9, 0, 0), Priority.MARK, "marker"))
    rig.engine.step()
    unplaced = [f["why"] for k, f in rig.records if k == "unplaced"]
    assert unplaced == ["its part is not with the voice"]
    rig.tick(0.48)  # the nod's stroke at +0.8: it starts 0.2 before, sent 0.1 earlier still
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


# -- on the word: the ladder, the last moment, units, the token stream (slice 2) ------------------


def gesture_book() -> Book:
    """Today's book and test intents of known timing: the arm to a pose with a co-stroke nod
    (``point`` 0.6 s from rest, ``slow`` 1.4 s, ``near`` to a pose 15 deg of the wrist from
    ``ready``, ``offer_it``), one over the caps with a smaller wrist version (``fast``), and a
    long one (``long``: 3 s of arm), each with its head-and-face floor."""
    data = json.loads((CONFIG / "animation.json").read_text())
    data["aliases"] = {}  # the book's aliases spell its point's arguments; this point takes none
    ready = data["poses"]["ready"]
    data["poses"]["near"] = {**ready, "wrist_flex": ready["wrist_flex"] - 15.0}
    nod = {"atom": "head_nod", "params": {"amp_deg": 3.0, "period_s": 0.4}}
    face = {"atom": "face", "params": {"name": "happy", "hold_s": 2.0}}
    floor = {"floor": True, "steps": [nod]}

    def pose(name: str, period: float, hold: float = 0.3) -> dict[str, Any]:
        return {"atom": "arm_pose", "params": {"pose": name, "period_s": period, "hold_s": hold}}

    data["compositions"].update({
        "point": {"bind": "word", "variants": [{"steps": [pose("ready", 0.6), nod]}, floor]},
        "point_hold": {"bind": "word", "variants": [
            {"steps": [pose("ready", 0.6, 0.8), nod]}, floor]},
        "slow": {"bind": "word", "variants": [{"steps": [pose("ready", 1.4), nod]}, floor]},
        "near": {"bind": "word", "variants": [{"steps": [pose("near", 0.6), nod]}, floor]},
        "offer_it": {"bind": "word", "variants": [{"steps": [pose("offer", 0.6), nod]}, floor]},
        "fast": {"bind": "word", "variants": [
            {"steps": [pose("ready", 0.33), nod]},
            {"weight": 1e-6, "steps": [
                {"atom": "arm_beat", "params": {"amp_deg": 10.0, "period_s": 0.6}}, nod]},
            floor,
        ]},
        "long": {"bind": "utterance", "variants": [
            {"steps": [pose("raised", 1.0, 2.0), face]},
            {"floor": True, "steps": [face, nod]},
        ]},
    })  # fmt: skip
    return with_rest(book_from(data), MOTION.rest_deg)


GESTURES = gesture_book()


class Ladder:
    """An engine over the arm's real planner on a fake board, fake head and face, a fake player
    and a fake clock; the scheduler is lent the engine's parts and renderer (the seam)."""

    def __init__(self, book: Book | None = None) -> None:
        self.now = 1000.0
        self.playout = FakePlayout(now=1.0)
        self.link = FakeArmLink(state=ArmStatus(0.0, MOTION.rest(), False, "off"))
        self.arm = ArmPart(self.link, MOTION, latency_s=0.06, clock=lambda: self.now)
        self.head, self.face = FakeExecutor("head", 0.1), FakeExecutor("face", 0.05)
        self.records: list[tuple[str, dict[str, Any]]] = []
        config = EngineConfig(arm_speed_scale=1.0, audio_lag_s=0.15, seed=7)
        parts = {"arm": self.arm, "head": self.head, "face": self.face,
                 "body": Absent("body", "no wheels")}  # fmt: skip
        played = book if book is not None else GESTURES
        self.engine = GestureEngine(
            parts, self.playout, lambda: (played, config), on=True,  # type: ignore[arg-type]
            clock=lambda: self.now, record=lambda kind, fields: self.records.append((kind, fields)),
        )  # fmt: skip
        self.engine.scheduler.attach(
            self.engine.parts, lambda: self.engine.renderer, hooks=self.engine.scene_hooks
        )

    def rows(self, kind: str, **match: Any) -> list[dict[str, Any]]:
        rows = [f for k, f in self.records if k == kind]
        return [r for r in rows if all(r.get(k) == v for k, v in match.items())]

    def sent(self) -> list[Any]:
        return [args[0] for name, args in self.link.calls if name == "send"]

    def tick(self, seconds: float) -> None:
        for _ in range(round(seconds / 0.01)):
            self.now += 0.01
            self.playout.now += 0.01
            self.engine.step()

    def say(self, turn: int, index: int, text: str, marker: str | None = None, at: int = 0) -> None:
        markers = (Marker("gesture", marker, at),) if marker else ()
        self.engine.part_queued(turn, index, Part(text, None, markers))
        self.engine.step()

    def start(self, turn: int, index: int, fed: float, *, spoken: bool = True) -> None:
        """The part's first chunk is fed now (all of it, ``fed`` seconds, when ``spoken``), in
        the stream where the part before it ends (or at the second playing)."""
        before = self.playout.marks.get((turn, index - 1))
        at = before[1] if before is not None else self.playout.now
        self.playout.marks[(turn, index)] = (at, at + fed)
        self.engine.part_started(turn, index, at)
        if spoken:
            self.engine.part_spoken(turn, index)

    def heard(self, turn: int, index: int, char: int) -> float:
        """When character ``char`` is heard by the clock's law, on the engine's clock."""
        forecast = self.engine.scheduler.words.at(turn, index, char, self.now)
        assert forecast is not None
        return forecast.heard


def first_vowel(text: str, word: str) -> int:
    at = text.index(word)
    return at + next(i for i, c in enumerate(word) if c.lower() in "аеёиоуыэюя")


def test_a_first_word_mark_gets_what_the_computed_ladder_gives() -> None:
    """«{point} Принтер вон там.»: at TEXT the first vowel is heard 0.10 (the voice's first chunk,
    p10) + 0.16 (its leading silence) + its share + 0.15 (the speaker) after it, 0.4 s of lead
    to the aimed stroke: the arm's least preparation from rest (0.41 s at 0.9 of the caps) does
    not fit, the head nod does: the floor, on its word."""
    rig = Ladder()
    text = "Принтер вон там."
    rig.say(1, 0, text, "point", 0)
    (row,) = rig.rows("ladder")
    aim = rig.heard(1, 0, first_vowel(text, "Принтер")) - 0.05
    forecast = rig.engine.scheduler.words.at(1, 0, first_vowel(text, "Принтер"), rig.now)
    assert forecast is not None
    room = aim - rig.now - 0.06 - min(forecast.spread, 0.3)
    drawn = rig.engine.renderer.render("point", Scene(frozenset(PARTS)), 7, variant=0)
    clip = next(c for c in drawn.clips if c.part == "arm")
    least = prep_min_s(clip, rig.arm.base, JOINTS, *rig.arm.caps(PREP_CAPS))
    assert (row["step"] == "floor") == (least > room)
    assert row["step"] == "floor" and row["parts"] == ["head"]
    rig.tick(0.1)
    rig.start(1, 0, 0.09 * len(text))
    rig.tick(1.0)
    (head,) = rig.rows("send", part="head")
    assert -0.01 <= head["error_s"] <= 0.01 and not rig.sent()


def test_the_same_mark_mid_sentence_with_0_9_s_takes_the_arm_from_rest() -> None:
    rig = Ladder()
    text = "Принтер вон там, у стены."
    rig.say(1, 0, text, "point", text.index("вон"))
    (row,) = rig.rows("ladder")
    assert 0.8 <= row["lead_s"] <= 1.2 and row["step"] == "authored" and row["base"] == "rest"
    rig.tick(0.1)
    rig.start(1, 0, 0.09 * len(text))
    rig.tick(2.0)
    (arm,) = rig.rows("send", part="arm")
    (trajectory,) = rig.sent()
    assert -0.3 <= arm["error_s"] <= 0.15 and arm["stage"] == "start"
    assert trajectory.knots[0].deg == pytest.approx(MOTION.rest(), abs=0.01)
    assert trajectory.then == "rest"
    (head,) = rig.rows("send", part="head")
    assert abs(head["stroke"] - arm["stroke"]) < 0.02  # the co-stroke


def test_a_preparation_too_slow_for_its_lead_is_compressed_and_still_lands() -> None:
    rig = Ladder()
    text = "Принтер вон там, у стены."
    rig.say(1, 0, text, "slow", text.index("вон"))
    (row,) = rig.rows("ladder")
    assert row["step"] == "compressed" and "slow" not in row["why"]
    rig.tick(0.1)
    rig.start(1, 0, 0.09 * len(text))
    rig.tick(2.0)
    (arm,) = rig.rows("send", part="arm")
    assert -0.3 <= arm["error_s"] <= 0.15
    held = rig.engine.scheduler.reservations.last("arm")
    assert held is not None and held.stroke - held.started < 1.0  # authored 1.4 s, now faster


def test_two_arm_marks_1_5_s_apart_are_one_unit_the_second_from_the_held_pose() -> None:
    rig = Ladder()
    first, second = "Есть план и ещё.", "Второй."
    rig.say(1, 0, first, "point", first.index("план"))
    rig.say(1, 1, second, "offer_it", 0)
    rig.playout.marks[(1, 0)] = (2.0, 4.0)
    rig.playout.marks[(1, 1)] = (4.0, 5.0)
    for index in (0, 1):
        rig.engine.part_started(1, index, rig.playout.marks[(1, index)][0])
        rig.engine.part_spoken(1, index)
    rig.tick(0.02)
    gap = rig.heard(1, 1, first_vowel(second, "Второй")) - rig.heard(
        1, 0, first_vowel(first, "план")
    )
    assert 1.3 <= gap <= 1.7
    rig.tick(5.0)
    opened, closed = rig.sent()
    assert opened.then == "hold" and 0.15 <= opened.hold_s <= 2.5
    assert closed.knots[0].deg == pytest.approx(opened.knots[-1].deg, abs=0.01)  # the held pose
    assert closed.then == "rest"
    assert [r["base"] for r in rig.rows("send", part="arm")] == ["rest", "unit"]


def test_three_arm_marks_in_a_row_are_one_unit_each_from_the_one_before() -> None:
    """The third stroke chains to the second (the latest motion before it), never to the first:
    the arm passes through the second's pose on its way."""
    rig = Ladder()
    texts = ("Есть план и ещё.", "Второй вот.", "Третий вот.")
    rig.say(1, 0, texts[0], "point", texts[0].index("план"))
    rig.say(1, 1, texts[1], "offer_it", 0)
    rig.say(1, 2, texts[2], "near", 0)
    marks = [(2.0, 4.0), (4.0, 5.0), (5.0, 6.0)]
    for index, mark in enumerate(marks):
        rig.playout.marks[(1, index)] = mark
        rig.engine.part_started(1, index, mark[0])
        rig.engine.part_spoken(1, index)
    rig.tick(7.0)
    first, second, third = rig.sent()
    assert [t.then for t in (first, second, third)] == ["hold", "hold", "rest"]
    assert second.knots[0].deg == pytest.approx(first.knots[-1].deg, abs=0.01)
    assert third.knots[0].deg == pytest.approx(second.knots[-1].deg, abs=0.01)
    assert [r["base"] for r in rig.rows("send", part="arm")] == ["rest", "unit", "unit"]


def test_an_accent_inside_an_open_unit_is_the_wrist_flick_from_the_held_pose() -> None:
    """{count:1} before «батарея», then «Это {!} очень важно.»: the count opens a unit; the
    accent, rendered after it, finds the arm up (the scene's ``arm_at``: the pose the count's
    clip will hold), so it is the wrist's flick, planned from that pose and sent while it
    holds. Alone (the arm at rest) the accent is the nod and the arm stays still."""
    rig = Ladder()
    first, second = "Во-первых, батарея.", "Это очень важно."
    count = Marker("gesture", "count", first.index("батарея"), "", "1")
    rig.engine.part_queued(1, 0, Part(first, None, (count,)))
    rig.engine.part_queued(1, 1, Part(second, None, (Marker("gesture", "accent", 4),)))
    rig.engine.step()
    shown = {r["intent"]: r for r in rig.rows("performance")}
    assert shown["count"]["arm_at"] == "rest" and shown["accent"]["arm_at"] != "rest"
    assert shown["accent"]["variant"] == 0  # the flick: the variant that needs the arm up
    (row,) = rig.rows("ladder", intent="accent")
    assert row["base"] == "unit" and "arm" in row["parts"]
    rig.tick(0.1)
    rig.start(1, 0, 0.09 * len(first))
    rig.tick(0.09 * len(first) - 0.2)
    rig.start(1, 1, 0.09 * len(second))
    rig.tick(3.0)
    opened, flick = rig.sent()
    assert opened.then == "hold" and flick.then == "rest"
    assert flick.knots[0].deg == pytest.approx(opened.knots[-1].deg, abs=0.01)  # the held pose
    assert [r["base"] for r in rig.rows("send", part="arm")] == ["rest", "unit"]
    alone = Ladder()
    alone.engine.part_queued(1, 0, Part(second, None, (Marker("gesture", "accent", 4),)))
    alone.engine.step()
    (nod,) = alone.rows("performance")
    assert nod["arm_at"] == "rest" and nod["variant"] == 1 and nod["parts"] == ["head"]


def test_a_unit_s_next_stroke_takes_the_arm_during_the_last_one_s_hold() -> None:
    """Marks 1.0 s apart (at TEXT, part 1 forecast after part 0's characters): the second
    stroke's preparation (from the held ``ready`` to ``near``) is compressed to start inside the
    first clip's own 0.8 s hold, and the board takes it there (the hold waits for it)."""
    rig = Ladder()
    first, second = "Есть план и ещё.", "Второй."
    rig.say(1, 0, first, "point_hold", first.index("план"))
    rig.say(1, 1, second, "near", 0)
    rig.tick(0.1)
    rig.start(1, 0, 0.09 * len(first))
    rig.tick(0.5)
    rig.start(1, 1, 0.09 * len(second))
    rig.tick(3.0)
    opened, closed = rig.sent()
    assert opened.then == "hold" and closed.knots[0].deg == pytest.approx(opened.knots[-1].deg)
    first_send, second_send = rig.rows("send", part="arm")
    assert second_send["step"] == "compressed" and second_send["base"] == "unit"
    assert second_send["send"] - first_send["send"] < opened.knots[-1].t_s  # inside the hold
    assert all(c["ok"] for c in rig.rows("cue", part="arm"))


def test_w_moving_later_never_makes_an_arm_stroke_late_and_the_head_still_lands() -> None:
    """Part 1's word is forecast where part 0's characters end; the arm commits on that (TEXT);
    part 0 then runs 0.5 s longer: the arm's stroke comes early (free), never late, and the
    head, sent at its last moment on part 1's own audio, lands on the word."""
    rig = Ladder()
    first, second = "Есть план.", "Второй план дешевле."
    rig.say(1, 0, first)
    rig.say(1, 1, second, "slow", 0)
    rig.tick(0.1)
    rig.start(1, 0, 0.3, spoken=False)  # part 0 starts as forecast, its length still unknown
    rig.tick(0.3)
    (arm,) = rig.rows("send", part="arm")
    assert arm["stage"] == "text"  # committed before part 1 had any audio
    rig.playout.marks[(1, 0)] = (rig.playout.marks[(1, 0)][0], rig.playout.marks[(1, 0)][0] + 1.4)
    rig.engine.part_spoken(1, 0)  # 1.4 s instead of the 0.9 forecast: the word 0.5 s later
    rig.tick(0.6)
    rig.start(1, 1, 0.09 * len(second))
    aim = rig.heard(1, 1, first_vowel(second, "Второй")) - 0.05 - 1000.0  # after TEXT
    rig.tick(2.0)
    assert arm["stroke"] <= aim + 0.15 and aim - arm["stroke"] > 0.3  # early, never late
    (head,) = rig.rows("send", part="head")
    assert abs(head["error_s"]) <= 0.01 and abs(head["stroke"] - aim) <= 0.01
    early = Ladder()  # the other way: part 0 shorter before the arm went: the arm is left out
    early.say(1, 0, first)
    early.say(1, 1, second, "slow", 0)
    early.tick(0.1)
    early.start(1, 0, 0.5)  # all of part 0 at once, 0.4 s shorter than its characters
    early.tick(2.5)
    early.start(1, 1, 0.09 * len(second))
    early.tick(2.0)
    assert not early.sent() and early.rows("late", part="arm")
    (late_head,) = early.rows("send", part="head")
    assert abs(late_head["error_s"]) <= 0.01


def test_a_mark_before_a_pause_takes_the_next_word_or_is_dropped() -> None:
    rig = Ladder()
    rig.say(1, 0, "Это очень важно.", "emphasis", len("Это очень важно"))
    assert rig.rows("dropped", why="in a pause: no word after it")
    feeling = Ladder()
    text = "Это очень важно."
    feeling.say(1, 0, text, "agreement", len("Это очень важно"))
    (waiting,) = [w for w in feeling.engine.scheduler.waiting if w.clip.part == "head"]
    assert waiting.cue is not None and waiting.cue.word == (text.index("важно"), len(text) - 1)
    comma = Ladder()
    text = "Цены выросли вдвое, но потом упали."
    comma.say(1, 0, text, "point", text.index(","))
    head = next(w for w in comma.engine.scheduler.waiting if w.clip.part == "head")
    assert head.cue is not None and head.cue.word[0] == text.index("но")  # after the pause


def test_a_long_composition_is_refused_with_short_speech_ahead() -> None:
    short, long = Ladder(), Ladder()
    short.say(1, 0, "Ну ладно, давай.", "long", 3)
    long.say(1, 0, "Ну ладно, давай поговорим об этом подробнее, у нас много времени.", "long", 3)
    (refused,) = short.rows("ladder")
    assert refused["step"] == "floor" and "of speech ahead" in refused["why"]
    (played,) = long.rows("ladder")
    assert played["step"] in ("authored", "compressed") and "arm" in played["parts"]


def test_a_motion_late_in_its_sentence_needs_the_speech_after_its_stroke_only() -> None:
    """«Принтер стоит {point_hold} вон там.» alone: 1.4 s of arm (0.6 to the pose, 0.8 held), its
    stroke 0.66 s before the sentence's forecast end: the 0.8 s after the stroke fit 0.7 of
    that speech and the 0.5 s past it, though the whole motion is longer than 0.7 of the
    sentence from its start would allow."""
    rig = Ladder()
    text = "Принтер стоит вон там."
    rig.say(1, 0, text, "point_hold", text.index("вон"))
    (row,) = rig.rows("ladder")
    assert row["step"] == "authored" and "arm" in row["parts"], row


def test_a_clip_over_the_caps_steps_down_and_its_stroke_is_never_later_than_planned() -> None:
    rig = Ladder()
    text = "Принтер вон там, у стены, слева."
    rig.say(1, 0, text, "fast", text.index("стены"))
    (row,) = rig.rows("ladder")
    drawn = rig.engine.renderer.render("fast", Scene(frozenset(PARTS)), 7, variant=0)
    planned = rig.arm.plan(next(c for c in drawn.clips if c.part == "arm"))
    assert not isinstance(planned, str) and 1.05 <= planned.stretch <= 1.15  # 1.1x the caps
    assert row["step"] == "smaller" and f"{planned.stretch:.2f}x the caps" in row["why"]
    rig.tick(0.1)
    rig.start(1, 0, 0.09 * len(text))
    rig.tick(3.0)
    (arm,) = rig.rows("send", part="arm")
    held = rig.engine.scheduler.reservations.last("arm")
    assert held is not None
    # the stroke the scheduler sent it for is the arm's plan (the engine's, from the send; the
    # arm's 0.06 s latency on top): never later than planned, inside the word's window
    assert held.stroke + 0.06 == pytest.approx(held.started + arm["stroke"] - arm["send"])
    assert -0.3 <= arm["error_s"] <= 0.15
    assert all(t.knots[-1].t_s <= 1.5 for t in rig.sent())


def test_barge_in_drops_the_reply_s_forecasts_and_a_flushed_stream_does_not_revive_them() -> None:
    rig = Ladder()
    rig.say(1, 0, "Принтер вон там, у стены.", "point", 8)
    assert rig.engine.scheduler.waiting
    rig.engine.reply_cut(1)
    rig.engine.step()
    scheduler = rig.engine.scheduler
    assert not scheduler.waiting and not [k for k in scheduler.spoken if k[0] == 1]
    assert scheduler.words.at(1, 0, 3, rig.now) is None and rig.rows("dropped", why="cut")
    rig.start(1, 0, 1.0)  # the flushed stream's seconds come back
    rig.tick(2.0)
    assert not rig.sent() and not rig.head.calls


def test_a_token_stream_mark_gains_its_lead() -> None:
    """«{near} Держи.»: at TEXT the first word's stroke is 0.38 s off, too soon for the arm from
    rest (0.35 s at 0.9 of the caps, with the latency and the spread); from the token stream
    0.35 s earlier (the measured p50 is 0.18, p90 0.70) the arm goes home (``ready``, 0.41 s)
    and holds, and the stroke at TEXT goes from there (0.17 s)."""
    without = Ladder()
    without.say(1, 0, "Держи.", "near", 0)
    assert without.rows("ladder")[0]["step"] == "floor"
    rig = Ladder()
    rig.engine.text_ahead(1, 0, "{near} Держи")
    rig.engine.step()
    (prepared,) = rig.rows("prepared")
    (home,) = rig.sent()
    assert home.then == "hold" and prepared["pose"] == "ready"
    ready = tuple(GESTURES.poses["ready"][j] for j in JOINTS)
    assert home.knots[-1].deg == pytest.approx(ready, abs=0.01)
    rig.tick(0.35)
    rig.say(1, 0, "Держи.", "near", 0)
    (row,) = rig.rows("ladder")
    assert row["step"] != "floor" and row["base"] == "unit"
    rig.tick(0.1)
    rig.start(1, 0, 0.54)
    rig.tick(1.0)
    stroke = rig.sent()[-1]
    assert stroke.knots[0].deg == pytest.approx(ready, abs=0.01)  # from the held home pose
    (arm,) = rig.rows("send", part="arm")
    assert -0.3 <= arm["error_s"] <= 0.15


def test_a_token_stream_mark_with_an_argument_or_another_spelling_prepares_the_arm() -> None:
    """The token stream reads marks as the parser does (the book's arguments and aliases):
    «{count:second} Колёса» is ``count:2`` on the reply's first word, rendered for its argument
    (the outer spot): too soon for the arm from rest, so it goes home first; so does a pointing
    at a place the arm reaches, and one at a place past its pans prepares nothing."""
    for text, prepares in (("{count:second} Колёса", "count"), ("{point:printer} Там", "point"),
                           ("{point:door} Дверь", None)):  # fmt: skip
        rig = Ladder(BOOK)
        rig.engine.resolver.room = FakePlaces({"printer": 10.0, "door": 80.0})
        rig.engine.text_ahead(1, 0, text)
        rig.engine.step()
        if prepares is None:
            assert not rig.rows("prepared") and not rig.sent(), text
            continue
        (prepared,) = rig.rows("prepared")
        assert prepared["intent"] == prepares and prepared["pose"] == "ready"
        assert [t.then for t in rig.sent()] == ["hold"]


def test_a_lower_request_is_refused_a_part_promised_to_a_mark_and_a_tool_takes_it() -> None:
    rig = Ladder()
    text = "Принтер вон там, у стены."
    rig.say(1, 0, text, "point", text.index("вон"))
    head = next(w for w in rig.engine.scheduler.waiting if w.clip.part == "head")
    assert head.fit is not None
    when = head.fit.stroke
    rig.engine.request(Request("nod", AtTime(when), Priority.PULSE, "pulse", clips=(NOD,)))
    rig.engine.step()
    assert rig.rows("refused", intent="nod", part="head")
    rig.tick(round(head.fit.start - rig.now - 0.2, 2))  # the person asks just before it
    assert rig.engine.perform("agreement", source="tool").ok
    assert rig.rows("displaced", part="head", by="agreement")


def test_every_feeling_holds_the_arm_at_most_2_5_s_before_the_board_s_return() -> None:
    """The utterance-bound compositions' arm clips (the board's return to rest comes after), at
    the corners of their draws; evil_click keeps its live-tuned 3-4 s (Artem's tuning)."""
    renderer = BookRenderer(BOOK)
    arm = ArmPart(FakeArmLink(), MOTION)
    scene = Scene(frozenset(PARTS))
    for name, composition in BOOK.compositions.items():
        if composition.bind != "utterance" or name == "evil_click":
            continue
        longest: Clip | None = None
        for k in range(len(composition.variants)):
            for seed in range(60):
                performance = renderer.render(name, scene, seed, variant=k)
                for clip in performance.clips:
                    if clip.part == "arm" and (
                        longest is None or clip.duration_s > longest.duration_s
                    ):
                        longest = clip
        if longest is None:
            continue
        planned = arm.plan(longest)
        assert not isinstance(planned, str), name
        assert planned.trajectory.knots[-1].t_s <= 2.5, name


def test_the_ladder_s_other_versions_pass_the_engine_s_scene_hooks() -> None:
    """A hook that keeps a part out (the quiet rule, the laugh's arm) holds for every rung:
    here the head is kept out, the drawn arm cannot make the reply's first word, and the floor
    (a nod) must not bring the head back: nothing plays."""

    def no_head(scene: Scene, request: Request) -> Scene:
        return replace(scene, ready=scene.ready - {"head"})

    rig = Ladder()
    rig.engine.scene_hooks.append(no_head)
    rig.say(1, 0, "Принтер вон там.", "point", 0)
    assert not rig.rows("ladder") and rig.rows("dropped", why="no time")
    bare = Ladder()
    bare.say(1, 0, "Принтер вон там.", "point", 0)
    assert bare.rows("ladder")[0]["parts"] == ["head"]  # without the hook: the floor's nod


def test_the_scheduler_s_knobs_are_config_expression_json_s() -> None:
    data = json.loads((CONFIG / "expression.json").read_text())
    assert timing_from(data) == Timing()
    assert timing_from({}) == Timing()
