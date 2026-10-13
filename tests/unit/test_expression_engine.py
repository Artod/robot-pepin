"""The gesture engine over fakes: the book, rendering by seed, the motion math, the parts, and
the animator's cues on a fake player's clock, its layers, caps and switch."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import copy
import itertools
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from pepin.arm_motion import ArmMotion, ServoRange, arm_motion, load_arm_motion
from pepin.expression.animation import Book, Clip, Event, Range, Scene, Track
from pepin.expression.animator import Caps, EngineConfig, GestureEngine, engine_config
from pepin.expression.book import book_from, face_names, load_book, vocabulary, with_rest
from pepin.expression.conversation import Heard
from pepin.expression.fakes import FakeArmLink, FakeExecutor, FakeFace, FakePlayout
from pepin.expression.markers import Marker, Part
from pepin.expression.mix import (
    approach,
    check_knots,
    min_jerk_time,
    pose_at,
    stretched,
    track_value,
)
from pepin.expression.parts import Absent, ArmPart, FacePart, Planned
from pepin.expression.render import BookRenderer, WeightedChooser
from pepin.expression.trajectory import JOINTS, ArmStatus

CONFIG = Path(__file__).resolve().parents[2] / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
BOOK = with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg)
EVERYTHING = Scene(frozenset({"arm", "head", "face", "body"}))
RESEARCH_NAMES = {
    "greeting", "farewell", "laugh", "doubt", "thinking", "emphasis", "surprise", "agreement",
    "disagreement", "shrug", "evil_click", "sympathy",
}  # fmt: skip
# the brain's co-speech marks (gesticulation DESIGN.md §4.4; test_expression_space.py)
PRIMITIVES = {"accent", "point", "arc", "size", "weigh", "count", "draw", "stop", "offer"}
RAISED = {"shoulder_pan": -10.0, "shoulder_lift": -40.0, "elbow_flex": 40.0, "wrist_flex": 30.0,
          "wrist_roll": -54.1, "gripper": -0.4}  # fmt: skip


# -- the book -----------------------------------------------------------------------------------


def test_the_book_holds_the_research_s_twelve_intents_and_its_atoms() -> None:
    assert set(BOOK.compositions) == RESEARCH_NAMES | PRIMITIVES
    assert len(BOOK.atoms) >= 19
    assert not BOOK.atoms["breath"].callable and not BOOK.atoms["head_drift"].callable
    assert BOOK.moments["slow_tool"] == "thinking"  # the listeners' moments name compositions too
    assert (BOOK.moments["greeting_due"], BOOK.moments["farewell_due"]) == ("greeting", "farewell")
    assert set(BOOK.moments.values()) <= set(BOOK.compositions)
    words = vocabulary(BOOK, face_names(CONFIG / "face.json"))
    assert words.intents == frozenset(RESEARCH_NAMES | PRIMITIVES) and "smile" in words.faces
    faces = {n for c in BOOK.compositions.values() for v in c.variants for s in v.steps
             for n in s.choices.get("name", ())}  # fmt: skip
    assert faces <= words.faces  # every face a composition names exists


def test_every_pose_keeps_the_arm_s_limits_and_every_pose_named_exists() -> None:
    limits = dict(zip(JOINTS, MOTION.limits(), strict=True))
    for name, pose in BOOK.poses.items():
        assert set(pose) == set(JOINTS), name
        for joint, deg in pose.items():
            low, high = limits[joint]
            assert low <= deg <= high, f"poses.{name}.{joint} {deg} outside {low}..{high}"
    named = {pose for composition in BOOK.compositions.values() for v in composition.variants
             for s in v.steps for pose in s.choices.get("pose", ()) if pose[0] != "@"}  # fmt: skip
    keys = [k for a in BOOK.atoms.values() for k in a.keys]
    keyed = {k.pose for k in keys if k.pose and k.pose[0] not in "$@"}  # @: the space's places
    assert named and named | keyed <= set(BOOK.poses)
    assert {"raised", "ready", "offer", "emphasis", "thinking", "tucked"} <= set(BOOK.poses)


def _data() -> dict[str, Any]:
    return json.loads((CONFIG / "animation.json").read_text())  # type: ignore[no-any-return]


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda d: d["atoms"]["head_nod"].update(part="tail"), "atoms.head_nod.part"),
        (lambda d: d["atoms"]["head_nod"]["params"].pop("period_s"), "period_s"),
        (lambda d: d["atoms"]["head_nod"]["keys"][1].update(tilt_deg="$nope"), "keys[1].tilt_deg"),
        (lambda d: d["atoms"]["head_nod"]["keys"][1].update(gripper=1.0), "not a degree"),
        (lambda d: d["compositions"]["laugh"]["variants"][0]["steps"][0].update(atom="x"),
         "no atom"),
        (lambda d: d["compositions"]["laugh"]["variants"][0]["steps"][0]["params"].pop("name"),
         "needs one"),
        (lambda d: d["compositions"]["laugh"]["variants"][0]["steps"].append({"atom": "breath"}),
         "idle layer"),
        (lambda d: d["moments"].update(slow_tool="dance"), "moments.slow_tool"),
        (lambda d: d["poses"].update(raised={"gripper": 1.0}), "poses.raised"),
    ],
)  # fmt: skip
def test_a_mistake_in_the_book_is_named_at_load(change: Callable[[Any], Any], error: str) -> None:
    data = _data()
    change(data)
    with pytest.raises(ValueError, match=error.replace("[", r"\[").replace("]", r"\]")):
        book_from(data)


# -- rendering ----------------------------------------------------------------------------------


def test_the_same_seed_renders_the_same_performance_and_another_seed_another() -> None:
    first = BookRenderer(BOOK).render("laugh", EVERYTHING, 1234, variant=0)
    again = BookRenderer(BOOK).render("laugh", EVERYTHING, 1234, variant=0)
    other = BookRenderer(BOOK).render("laugh", EVERYTHING, 4321, variant=0)
    assert first == again and first != other


def test_parameters_are_drawn_within_their_ranges() -> None:
    renderer = BookRenderer(BOOK)
    for seed in range(50):
        performance = renderer.render("agreement", EVERYTHING, seed)
        head = next(c for c in performance.clips if c.part == "head")
        (tilt,) = head.tracks
        assert 4.0 <= max(tilt.values) <= 6.0  # the step's amp_deg [4, 6] over the atom's
        assert tilt.mode == "additive" and tilt.values[0] == tilt.values[-1] == 0.0


def test_a_part_not_ready_and_a_pose_not_taught_are_left_out_the_rest_plays() -> None:
    untaught = Book(BOOK.atoms, BOOK.compositions, {"rest": BOOK.poses["rest"]})
    renderer = BookRenderer(untaught)
    for seed in range(30):
        clips = renderer.render("greeting", Scene(frozenset({"face", "arm"})), seed).clips
        assert {c.part for c in clips} <= {"face", "arm"} and "face" in {c.part for c in clips}
        for clip in clips:  # no pose is taught: the arm never leaves rest absolutely
            assert all(t.mode == "additive" for t in clip.tracks)
    taught = Book(BOOK.atoms, BOOK.compositions, {**untaught.poses, "raised": RAISED})
    clips = BookRenderer(taught).render("greeting", EVERYTHING, 3, variant=0).clips
    arm = next(c for c in clips if c.part == "arm")
    lift = next(t for t in arm.tracks if t.dof == "shoulder_lift")
    assert lift.mode == "absolute" and -40.0 in lift.values and lift.values[0] == -102.6


def test_optional_steps_follow_their_odds_and_one_always_plays() -> None:
    renderer = BookRenderer(BOOK)
    with_face = sum(
        any(c.part == "face" for c in renderer.render("agreement", EVERYTHING, s).clips)
        for s in range(400)
    )
    assert 160 < with_face < 240  # p 0.5
    lonely = Scene(frozenset({"face"}))  # only the optional face can play: it plays every time
    assert all(renderer.render("agreement", lonely, s).clips for s in range(30))


def test_the_chooser_never_draws_the_same_variant_twice_running() -> None:
    renderer = BookRenderer(BOOK, WeightedChooser())
    variants = [renderer.render("greeting", EVERYTHING, s).variant for s in range(40)]
    assert all(a != b for a, b in itertools.pairwise(variants))


def test_a_stroke_lands_at_its_step_s_time_from_the_sync_point() -> None:
    performance = BookRenderer(BOOK).render("emphasis", EVERYTHING, 5, variant=0)
    (head,) = performance.clips
    assert -0.15 <= head.offset_s + head.stroke_s <= 0.0  # at_s [-0.15, 0]
    (tilt,) = head.tracks
    peak = tilt.times[tilt.values.index(max(tilt.values))]
    assert peak == pytest.approx(head.stroke_s)


def test_mirror_images_turn_both_ways() -> None:
    signs = set()
    for seed in range(20):
        clips = BookRenderer(BOOK).render("thinking", Scene(frozenset({"head"})), seed).clips
        pan = next(t for t in clips[0].tracks if t.dof == "pan_deg")
        signs.add(max(pan.values, key=abs) > 0)
    assert signs == {True, False}


# -- the motion math ----------------------------------------------------------------------------


def test_a_track_passes_its_keys_and_never_overshoots_a_held_value() -> None:
    track = Track("x", "absolute", (0.0, 1.0, 2.0, 3.0), (0.0, 10.0, 10.0, 0.0))
    assert [track_value(track, t) for t in (0.0, 1.0, 2.0, 3.0)] == [0.0, 10.0, 10.0, 0.0]
    assert max(track_value(track, i / 100) for i in range(301)) <= 10.0 + 1e-9
    wave = Track("x", "additive", (0.0, 0.3, 0.7, 1.0), (0.0, 8.0, -8.0, 0.0))
    assert track_value(wave, 0.5) == pytest.approx(0.0, abs=1e-9)  # through zero, at speed
    assert track_value(wave, 0.49) > track_value(wave, 0.51)


def test_absolute_tracks_replace_the_base_and_additive_ones_add_to_it() -> None:
    clip = Clip("arm", 0.0, 0.0, 2.0, (
        Track("wrist_flex", "absolute", (0.5, 1.5), (30.0, 30.0)),
        Track("wrist_flex", "additive", (0.0, 2.0), (5.0, 5.0)),
    ))  # fmt: skip
    base = {"wrist_flex": 76.6, "gripper": 0.0}
    assert pose_at(clip, 0.2, base) == {"wrist_flex": 81.6, "gripper": 0.0}
    assert pose_at(clip, 1.0, base)["wrist_flex"] == pytest.approx(35.0)
    assert pose_at(clip, 1.9, base)["wrist_flex"] == pytest.approx(35.0)  # held to the end


def test_the_knot_check_refuses_limits_speed_acceleration_and_time_never_clamps() -> None:
    limits = [(-10.0, 10.0)]
    assert check_knots([(0.0, [0.0]), (1.0, [5.0])], limits, 90.0, 360.0) is None
    assert "outside" in str(check_knots([(0.0, [0.0]), (1.0, [11.0])], limits, 90.0, 360.0))
    assert "deg/s," in str(check_knots([(0.0, [0.0]), (0.01, [5.0])], limits, 90.0, 1e9))
    knots = [(0.0, [0.0]), (0.1, [0.0]), (0.2, [5.0])]
    assert "deg/s^2" in str(check_knots(knots, limits, 90.0, 360.0))
    assert "does not follow" in str(check_knots([(0.0, [0.0]), (0.0, [1.0])], limits, 90, 360))


def test_an_approach_is_minimum_jerk_within_both_caps() -> None:
    knots = approach([0.0, 10.0], [60.0, 10.0], 90.0, 360.0, 50.0)
    assert knots[0][1] == (0.0, 10.0) and knots[-1][1] == pytest.approx((60.0, 10.0))
    assert knots[-1][0] == pytest.approx(min_jerk_time(60.0, 90.0, 360.0))
    assert check_knots(knots, [(-90.0, 90.0)] * 2, 90.0, 360.0) is None


def test_a_stretched_clip_keeps_its_stroke_on_the_sync_point() -> None:
    clip = Clip("arm", -0.2, 0.3, 1.0, (Track("gripper", "additive", (0.0, 1.0), (0.0, 0.0)),))
    slow = stretched(clip, 5.0)
    assert slow.offset_s + slow.stroke_s == pytest.approx(clip.offset_s + clip.stroke_s)
    assert slow.duration_s == 5.0 and slow.tracks[0].times == (0.0, 5.0)


def _every_step(book: Book, end: str | None = None) -> Book:
    """``book`` with every step playing; ``end`` (lo or hi) pins every range of a step there."""
    compositions = {}
    for name, composition in book.compositions.items():
        variants = []
        for variant in composition.variants:
            steps = []
            for step in variant.steps:
                if end is None:
                    steps.append(replace(step, p=1.0))
                    continue
                ranges = {**book.atoms[step.atom].params, **step.params}
                pinned = {k: Range(getattr(r, end), getattr(r, end)) for k, r in ranges.items()}
                at = getattr(step.at_s, end)
                steps.append(replace(step, p=1.0, at_s=Range(at, at), params=pinned))
            variants.append(replace(variant, steps=tuple(steps)))
        compositions[name] = replace(composition, variants=tuple(variants))
    return replace(book, compositions=compositions)


def test_every_intent_s_arm_clip_keeps_the_caps_at_full_speed_and_at_the_first_test_s() -> None:
    """The book is sized to config/arm.json's caps and its poses: nothing it renders is
    refused, every step playing, its ranges at either end or drawn, mirrored or not."""
    fast, slow = ArmPart(FakeArmLink(), MOTION), ArmPart(FakeArmLink(), MOTION, speed_scale=0.2)
    books = (_every_step(BOOK, "lo"), _every_step(BOOK, "hi"), _every_step(BOOK))
    for intent, composition in BOOK.compositions.items():
        for variant, book, seed in itertools.product(range(len(composition.variants)), books,
                                                     range(2)):  # fmt: skip
            performance = BookRenderer(book).render(intent, EVERYTHING, seed, variant=variant)
            for clip in performance.clips:
                if clip.part == "arm":
                    where = (intent, variant, seed)
                    assert isinstance(fast.plan(clip), Planned), (where, fast.plan(clip))
                    assert isinstance(slow.plan(stretched(clip, 5.0)), Planned), where


def test_every_arm_clip_fits_the_board_s_longest_trajectory_at_the_first_test_s_speed() -> None:
    """The arm steps of a variant at their widest (the earliest start, the latest end their
    ranges allow) last at most config/arm.json's max_duration_s once stretched to speed 0.2."""
    for name, composition in BOOK.compositions.items():
        for v, variant in enumerate(composition.variants):
            starts, ends = [], []
            for step in variant.steps:
                atom = BOOK.atoms[step.atom]
                if atom.part != "arm":
                    continue
                ranges = {**atom.params, **step.params}
                period, hold = ranges["period_s"], ranges.get("hold_s", Range(0.0, 0.0))
                repeats = round(ranges.get("repeats", Range(1.0, 1.0)).hi)
                starts.append(step.at_s.lo - atom.stroke * period.hi)
                ends.append(step.at_s.hi + (repeats - atom.stroke) * period.hi + hold.hi)
            if starts:
                assert (max(ends) - min(starts)) / 0.2 <= MOTION.max_duration_s, (name, v)


# -- the arm's numbers --------------------------------------------------------------------------


def test_the_arm_s_angles_are_lerobot_s_from_the_eeprom_range() -> None:
    pan = ServoRange(1515, 3492)
    assert pan.degrees(2503) == pytest.approx(-0.04, abs=0.05)
    assert pan.span_deg() == pytest.approx((-86.9, 86.9), abs=0.05)
    assert pan.ticks(pan.degrees(2100)) == 2100
    gripper = ServoRange(1837, 3126, gripper=True)
    assert gripper.degrees(1837) == -10.0
    assert gripper.span_deg()[1] == pytest.approx(103.3, abs=0.05)


def test_the_motion_block_is_checked_whole() -> None:
    data = json.loads((CONFIG / "arm.json").read_text())
    assert MOTION.rest() == tuple(MOTION.rest_deg[j] for j in JOINTS)
    broken = copy.deepcopy(data)
    broken["motion"]["rest_deg"]["elbow_flex"] = 120.0
    with pytest.raises(ValueError, match=r"rest_deg\.elbow_flex"):
        arm_motion(broken)
    broken = copy.deepcopy(data)
    broken["motion"]["torque_limit"] = 2.0
    with pytest.raises(ValueError, match="torque_limit"):
        arm_motion(broken)
    assert MOTION.hold_max_s == 20.0  # the voice's arm held up through a long reply
    broken = copy.deepcopy(data)
    broken["motion"]["hold_max_s"] = 31.0
    with pytest.raises(ValueError, match="hold_max_s"):
        arm_motion(broken)


# -- the parts ----------------------------------------------------------------------------------


def wave_clip(amp: float = 8.0, period: float = 1.6) -> Clip:
    track = Track("wrist_flex", "additive", (0.0, 0.3 * period, 0.7 * period, period),
                  (0.0, amp, -amp, 0.0))  # fmt: skip
    return Clip("arm", -0.2, 0.2, period, (track,))


def test_the_arm_part_sends_one_trajectory_over_the_rest_pose() -> None:
    link = FakeArmLink(state=ArmStatus(1.0, MOTION.rest(), False, "off"))
    arm = ArmPart(link, MOTION, speed_scale=0.5)
    assert arm.ready() is None
    outcome = arm.play(stretched(wave_clip(), 2.0))
    assert outcome.ok and outcome.seconds == pytest.approx(3.2)
    ((name, (trajectory,)),) = link.calls
    assert name == "send" and trajectory.then == "rest" and trajectory.speed_scale == 0.5
    assert trajectory.knots[0].deg == MOTION.rest() and trajectory.knots[-1].t_s == 3.2
    wrist = [k.deg[JOINTS.index("wrist_flex")] for k in trajectory.knots]
    assert max(wrist) == pytest.approx(76.6 + 8.0, abs=0.3)
    arm.fade("gesture", 0.35)
    assert link.calls[-1] == ("rest", ())


def test_the_arm_part_refuses_with_the_reason_never_clamps() -> None:
    link = FakeArmLink(state=ArmStatus(1.0, MOTION.rest(), False, "off"))
    arm = ArmPart(link, MOTION)
    assert "outside" in arm.play(wave_clip(amp=40.0, period=10.0)).why  # 76.6 + 40 > 100.9
    assert not link.calls
    # a stroke too fast for the caps is stretched in time, its angles untouched
    fast = arm.play(wave_clip(amp=10.0, period=0.4))
    ((name, (trajectory,)),) = link.calls
    assert fast.ok and name == "send" and fast.seconds == trajectory.knots[-1].t_s > 0.4
    wrist = JOINTS.index("wrist_flex")
    swing = [k.deg[wrist] - MOTION.rest()[wrist] for k in trajectory.knots]
    assert max(swing) == pytest.approx(10.0, abs=0.05) and min(swing) == pytest.approx(
        -10.0, abs=0.05
    )
    assert (
        check_knots(
            [(k.t_s, k.deg) for k in trajectory.knots],
            MOTION.limits(),
            MOTION.max_speed_deg_s,
            MOTION.max_acc_deg_s2,
        )
        is None
    )
    # stretched past the board's longest trajectory: refused, nothing sent (the animator hands
    # the part its clip already 67x slower: 26.7 s, which the caps stretch past 30)
    link.calls.clear()
    assert (
        "longer"
        in ArmPart(link, MOTION, speed_scale=0.015)
        .play(stretched(wave_clip(amp=10.0, period=0.4), 1 / 0.015))
        .why
    )
    assert not link.calls
    assert "no word" in str(ArmPart(FakeArmLink(), MOTION).ready())
    silent = FakeArmLink(state=ArmStatus(1.0, None, False, "silent", error="no reply"))
    assert "silent" in str(ArmPart(silent, MOTION).ready())


def test_the_face_part_shows_each_event_at_its_time_and_a_fade_withdraws_it() -> None:
    face, later = FakeFace(), []
    part = FacePart(face, after=lambda s, fn: (later.append((s, fn)), lambda: later.clear())[1])
    clip = Clip("face", 0.0, 0.0, 3.0, events=(Event(0.0, "happy", 2.0), Event(1.0, "grin")))
    assert part.play(clip).ok
    assert face.calls == [("express", ("happy", 1.0, 2.0, None))]
    ((seconds, show),) = later
    show()
    assert seconds == 1.0 and face.calls[-1] == ("express", ("grin", 1.0, 2.0, None))
    part.fade("gesture", 0.1)
    assert later == []


# -- the animator -------------------------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class Rig:
    """An engine over fake parts, a fake player and a fake clock; the thread never runs."""

    def __init__(self, on: bool = True, book: Book = BOOK, **knobs: Any) -> None:
        self.clock = Clock()
        self.playout = FakePlayout()
        self.parts = {p: FakeExecutor(p) for p in ("arm", "head", "face")}  # type: ignore[arg-type]
        self.parts["arm"].latency_s = 0.06
        self.records: list[tuple[str, dict[str, Any]]] = []
        config = EngineConfig(arm_speed_scale=1.0, **knobs)
        self.engine = GestureEngine(
            {**self.parts, "body": Absent("body", "no wheels")},  # type: ignore[dict-item]
            self.playout,
            lambda: (book, config),
            on=on,
            clock=self.clock,
            record=lambda kind, fields: self.records.append((kind, fields)),
        )

    def played(self, part: str) -> list[Clip]:
        return [args[0] for name, args in self.parts[part].calls if name == "play"]

    def kinds(self, kind: str) -> list[dict[str, Any]]:
        return [fields for k, fields in self.records if k == kind]

    def tick(self, seconds: float) -> None:
        """``seconds`` of audio play while the engine steps every 10 ms."""
        for _ in range(round(seconds / 0.01)):
            self.clock.now += 0.01
            self.playout.now += 0.01
            self.engine.step()


def marked(text: str, name: str, at: int) -> Part:
    return Part(text, None, (Marker("gesture", name, at),))


def test_a_marker_s_clips_go_when_their_stroke_lands_on_its_word_s_vowel() -> None:
    """The word's first vowel is heard after the part's leading silence at its share of the
    rest, the speaker's lag later; each clip goes its preparation (and its part's latency)
    before the stroke, aimed stroke_bias_s before the vowel (the word clock)."""
    rig = Rig(seed=7)
    text = "Ну что ж, привет всем!"
    rig.engine.part_queued(1, 0, marked(text, "agreement", 10))
    rig.engine.step()
    (performance,) = rig.kinds("performance")
    assert performance["seed"] == 7
    assert performance["share"] == pytest.approx(10 / len(text), abs=1e-3)
    rig.tick(0.5)
    assert rig.played("head") == []  # no audio of the part yet: the vowel is still >1 s away
    length = 0.07 * len(text)
    rig.playout.marks[(1, 0)] = (1.0, 1.0 + length)  # its first chunk is fed
    rig.engine.part_spoken(1, 0)
    rig.tick(3.0)
    (cue,) = [c for c in rig.kinds("cue") if c["part"] == "head"]
    clip = rig.played("head")[0]
    vowel = 1.0 + 0.16 + text.index("и") / len(text) * (length - 0.16)
    assert cue["planned"] == pytest.approx(vowel + 0.15 - 0.05 - clip.stroke_s, abs=1e-3)
    assert 0.0 <= cue["late_s"] <= 0.011


def test_a_feeling_just_past_goes_at_once_a_word_s_never_late_one_long_past_dropped() -> None:
    """The word's vowel was heard at 4.6 + 0.16 + 4/9 x (1.4 - 0.16) + the 0.15 lag = 5.46 on
    the player's clock, aimed 0.05 earlier: at 5.6 an UTTERANCE feeling is 0.2 s past it,
    inside late_s, and goes at once; a WORD mark is never late, so it is dropped at render
    ("no time"), as is anything long past."""
    for intent, now, played in (("agreement", 5.6, True), ("emphasis", 5.6, False),
                                ("agreement", 9.0, False)):  # fmt: skip
        rig = Rig(seed=3)
        rig.playout.now = now
        rig.playout.marks[(2, 0)] = (4.6, 6.0)
        rig.engine.part_queued(2, 0, marked("Да да да.", intent, 4))
        rig.engine.part_spoken(2, 0)
        rig.engine.step()
        moved = bool(rig.played("head") or rig.played("face") or rig.played("arm"))
        assert moved == played, (intent, now)
        if not played:
            (dropped,) = rig.kinds("dropped")
            assert dropped["why"] == "no time"


def test_a_part_with_no_audio_plays_its_cues_once_the_voice_is_past_it() -> None:
    rig = Rig(seed=4)
    rig.engine.part_queued(1, 0, Part("", None, (Marker("gesture", "agreement", 0),)))
    rig.tick(0.1)
    assert not rig.played("head")
    rig.engine.part_spoken(1, 0)
    rig.tick(0.05)
    assert rig.played("head")


def test_a_cut_reply_drops_its_waiting_clips() -> None:
    rig = Rig(seed=4)
    rig.engine.part_queued(3, 0, marked("Сейчас расскажу.", "agreement", 0))
    rig.engine.reply_cut(3)
    rig.playout.marks[(3, 0)] = (0.0, 1.0)
    rig.tick(1.0)
    assert not rig.played("head") and rig.kinds("dropped")


def test_the_person_speaking_fades_the_gesture_layer_but_a_greeting_past_its_stroke_ends() -> None:
    rig = Rig(seed=1)
    assert rig.engine.perform("agreement", source="tool").ok
    rig.engine.speech_started(rig.clock.now)
    rig.engine.step()
    assert ("fade", ("gesture", 0.35)) in rig.parts["head"].calls
    greeting = Rig(seed=1)
    assert greeting.engine.perform("greeting", source="tool").ok
    greeting.tick(2.5)  # every clip has gone (the arm starts 2 s before the word)
    playing = greeting.engine._playing["face"]
    greeting.clock.now = playing.ends - 0.5  # past its stroke, ending within finish_s
    greeting.engine.speech_started(greeting.clock.now)
    greeting.engine.step()
    assert not [c for c in greeting.parts["face"].calls if c[0] == "fade"]


def test_a_drive_rests_every_part_and_refuses_until_it_ends() -> None:
    rig = Rig()
    rig.engine.drive(True)
    rig.engine.step()
    assert all(("rest", ()) in p.calls for p in rig.parts.values())
    refused = rig.engine.perform("laugh", source="tool")
    assert not refused.ok and "drive" in refused.why
    rig.engine.part_queued(1, 0, marked("Ха-ха.", "laugh", 0))
    rig.engine.step()
    assert rig.kinds("capped")[0]["why"] == "a drive is under way"
    rig.engine.drive(False)
    rig.engine.step()
    assert rig.engine.perform("laugh", source="tool").ok


def test_the_caps_hold_markers_back_but_never_a_person_s_ask() -> None:
    rig = Rig()
    two = (Marker("gesture", "agreement", 0), Marker("gesture", "emphasis", 4))
    rig.engine.part_queued(1, 0, Part("Да, да.", None, two))
    rig.engine.part_queued(1, 1, marked("Ещё.", "doubt", 0))
    rig.engine.part_queued(1, 2, marked("И ещё.", "shrug", 0))
    rig.engine.part_queued(1, 3, marked("И всё.", "surprise", 0))
    rig.engine.step()
    assert [r["why"] for r in rig.kinds("capped")] == ["per_part", "per_reply"]  # three a reply
    for _ in range(3):
        assert rig.engine.perform("emphasis", source="tool").ok  # cooldown or not


def test_the_accent_and_a_count_series_stand_outside_the_counts() -> None:
    """«Две причины: {count:1} батарея и {count:2} колёса, это {!} очень {!} важно.»: the series
    counts once (as its first: per_part, per_reply and the minute's), the accent counts nowhere
    and plays once a part; two more marks fill the reply's three, a third is capped."""
    rig = Rig()
    text = "Две причины: батарея и колёса, это очень важно."

    def at(word: str, name: str, arg: str | None = None) -> Marker:
        return Marker("gesture", name, text.index(word), "", arg)

    marks = (at("батарея", "count", "1"), at("колёса", "count", "2"), at("очень", "accent"),
             at("важно", "accent"))  # fmt: skip
    rig.engine.part_queued(1, 0, Part(text, None, marks))
    rig.engine.part_queued(1, 1, marked("Ещё одно.", "doubt", 0))
    rig.engine.part_queued(1, 2, marked("И третье.", "shrug", 0))
    rig.engine.part_queued(1, 3, marked("И всё.", "surprise", 0))
    rig.engine.step()
    assert [r["why"] for r in rig.kinds("capped")] == ["accent_per_part", "per_reply"]
    played = [r["intent"] for r in rig.kinds("performance")]
    assert played == ["count", "count", "accent", "doubt", "shrug"]
    assert rig.engine._count(lambda arm: True) == 3  # the series once, the accent never


def test_a_cooldown_holds_the_same_mark_and_argument_another_argument_plays() -> None:
    point = replace(BOOK.compositions["point"], cooldown_s=10.0)
    rig = Rig(book=replace(BOOK, compositions={**BOOK.compositions, "point": point}))
    text = "Это ведь ты сам сказал."

    def point_at(turn: int, arg: str) -> None:
        mark = Marker("gesture", "point", text.index("сам"), "", arg)
        rig.engine.part_queued(turn, 0, Part(text, None, (mark,)))
        rig.engine.step()
        rig.clock.now += 1.0

    point_at(1, "you")
    point_at(2, "me")  # another argument: no cooldown
    point_at(3, "you")  # the same within 10 s
    assert [r["why"] for r in rig.kinds("capped")] == ["cooldown 10 s"]
    rig.clock.now += 10.0
    point_at(4, "you")
    assert [r["intent"] for r in rig.kinds("performance")] == ["point"] * 3


def test_a_laugh_once_a_minute_unless_the_person_laughed() -> None:
    rig = Rig()
    rig.engine.part_queued(1, 0, marked("Ха.", "laugh", 0))
    rig.clock.now += 25.0  # past the laugh's cooldown, within the minute
    rig.engine.part_queued(2, 0, marked("Ха.", "laugh", 0))
    rig.engine.step()
    assert rig.kinds("capped")[-1]["why"] == "laugh per minute"
    rig.engine.speech_ended(Heard(0.0, 1.0, "Ха-ха-ха, смешно"))
    rig.engine.part_queued(3, 0, marked("Ха.", "laugh", 0))
    rig.engine.step()
    assert len(rig.kinds("performance")) == 2


def test_a_laugh_heard_in_the_robot_s_words_shares_the_laugh_s_cap_with_the_brain_s_mark(
) -> None:  # fmt: skip
    """The listeners' laugh heard counts as the laugh under intent_per_minute: a {laugh} after it
    within the minute is capped, and so is a second laugh heard; the person's laugh waives it;
    it never counts against per_minute (it moves no gesture of the brain's)."""
    rig = Rig(caps=Caps(per_minute=1))
    assert rig.engine.claim("laugh") is None
    rig.clock.now += 25.0
    rig.engine.part_queued(1, 0, marked("Ха.", "laugh", 0))
    rig.engine.step()
    assert rig.kinds("capped")[-1]["why"] == "laugh per minute"
    assert rig.engine.claim("laugh") == "laugh per minute"
    rig.engine.speech_ended(Heard(0.0, 1.0, "Ха-ха-ха, смешно"))
    rig.engine.step()
    assert rig.engine.claim("laugh") is None
    rig.engine.part_queued(2, 0, marked("Привет.", "greeting", 0))
    rig.engine.step()
    assert [r["intent"] for r in rig.kinds("performance")] == ["greeting"]  # per_minute 1 kept
    rig.clock.now += 61.0
    assert rig.engine.claim("laugh") is None
    rig.engine.switch(False)
    assert rig.engine.claim("laugh") == "gestures are off"


def test_off_nothing_moves_and_switching_on_reads_the_book_again() -> None:
    rig = Rig(on=False)
    rig.engine.part_queued(1, 0, marked("Привет.", "greeting", 0))
    rig.engine.step()
    assert not rig.kinds("performance") and not rig.engine.perform("laugh", source="tool").ok
    loads = []
    original = rig.engine.load
    rig.engine.load = lambda: (loads.append(1), original())[1]  # type: ignore[method-assign]
    assert rig.engine.switch(True) is None and rig.engine.on and loads == [1]
    assert rig.engine.perform("laugh", source="tool").ok
    assert rig.engine.switch(False) is None and not rig.engine.on
    rig.engine.step()
    assert ("rest", ()) in rig.parts["arm"].calls

    def broken() -> tuple[Book, EngineConfig]:
        raise ValueError("atoms.x.part")

    rig.engine.load = broken  # type: ignore[method-assign]
    assert "did not load" in str(rig.engine.switch(True)) and not rig.engine.on


def test_a_slow_tool_s_moment_performs_thinking_now() -> None:
    rig = Rig()
    rig.engine.moment("slow_tool")
    rig.engine.step()
    assert rig.kinds("performance")[0]["intent"] == "thinking" and rig.played("head")


def test_the_idle_layer_fidgets_only_in_a_quiet_conversation() -> None:
    rig = Rig(idle_after_speech_s=5.0)
    rig.tick(70.0)
    assert not rig.played("head")  # no conversation
    rig.engine.opened(Heard(0.0, 1.0, "Пепин"))
    rig.tick(70.0)  # 5 s of quiet, then 20-60 s
    idle = [c for c in rig.played("head") if c.layer == "idle"]
    assert idle and not rig.played("arm")
    rig.engine.speech_started(rig.clock.now)
    count = len(rig.played("head"))
    rig.tick(70.0)
    assert len(rig.played("head")) == count  # nothing while the person speaks


def test_a_tool_s_perform_is_answered_from_the_engine_s_thread() -> None:
    rig = Rig()
    rig.engine.start()
    try:
        outcome = rig.engine.perform("agreement", source="tool")
        assert outcome.ok and "seed" in outcome.why
        missing = rig.engine.perform("dance", source="tool")
        assert not missing.ok and "greeting" in missing.why
        assert rig.engine.last() is not None
    finally:
        rig.engine.close()


def test_the_config_file_s_knobs_are_read() -> None:
    config = engine_config(json.loads((CONFIG / "expression.json").read_text()))
    assert config.fade_s == {"arm": 0.35, "head": 0.35, "face": 0.1, "body": 0.35}
    assert config.arm_speed_scale == 1.0 and config.caps.intent_per_minute == {"laugh": 1}
    assert (config.caps.per_part, config.caps.per_reply, config.caps.accent_per_part) == (1, 3, 1)
    assert config.caps.accents == ("accent",) and config.caps.series == ("count",)
    assert config.finish == ("greeting", "farewell") and config.seed is None


def test_the_arm_part_takes_the_engine_s_speed_scale() -> None:
    arm = ArmPart(FakeArmLink(), MOTION)
    GestureEngine({"arm": arm}, FakePlayout(), lambda: (BOOK, EngineConfig(arm_speed_scale=0.3)))
    assert arm.speed_scale == 0.3


def test_unused_arm_motion_fields_round_trip() -> None:
    assert isinstance(MOTION, ArmMotion)
    assert MOTION.limits()[JOINTS.index("gripper")] == (-10.0, 103.3)


def test_a_clip_faster_than_the_caps_is_stretched_in_time_not_refused() -> None:
    from pepin.expression.mix import check_knots, stretch_to_caps

    # 100 deg in 1 s asks 100 deg/s; at a 20 deg/s cap the same path must take 5 s
    knots = [(0.0, [0.0, 0.0]), (1.0, [100.0, 0.0]), (2.0, [100.0, 0.0])]
    slow = stretch_to_caps(knots, max_speed=20.0, max_acc=1e9)
    assert [t for t, _ in slow] == pytest.approx([0.0, 5.0, 10.0])
    assert [d for _, d in slow] == [[0.0, 0.0], [100.0, 0.0], [100.0, 0.0]]
    assert check_knots(slow, [(-180, 180)] * 2, 20.0, 1e9) is None
    # within the caps nothing changes
    assert stretch_to_caps(knots, max_speed=200.0, max_acc=1e9) == [(t, list(d)) for t, d in knots]


def test_a_compressed_preparation_keeps_the_stroke_on_its_word_and_the_rest_untouched() -> None:
    from pepin.expression.mix import PREP_CAPS, fit_preparation, prep_min_s, track_value

    lift = Track("shoulder_lift", "absolute", (0.0, 1.0, 1.5), (-100.0, -60.0, -60.0))
    wave = Track("wrist_flex", "additive", (0.8, 1.2, 1.6), (0.0, 8.0, 0.0))
    clip = Clip("arm", -1.0, 1.0, 1.6, (lift, wave), (Event(0.5, "smile"),), then="hold")
    fast = fit_preparation(clip, 0.4)
    assert fast.offset_s + fast.stroke_s == pytest.approx(clip.offset_s + clip.stroke_s)
    assert fast.stroke_s == pytest.approx(0.4) and fast.duration_s == pytest.approx(1.0)
    assert fast.then == "hold" and fast.events[0].t_s == pytest.approx(0.2)
    for t in (1.3, 1.45, 1.6):  # between keys after the stroke: the same motion, 0.6 s sooner
        assert track_value(fast.tracks[1], t - 0.6) == pytest.approx(track_value(wave, t))
    assert fit_preparation(clip, 2.0) is clip  # never slowed
    rest = dict.fromkeys(JOINTS, 0.0) | {"shoulder_lift": -100.0}
    least = prep_min_s(clip, rest, JOINTS, 240.0 * PREP_CAPS, 2232.0 * PREP_CAPS)
    assert 0.0 < least < clip.stroke_s
    assert prep_min_s(clip, rest, JOINTS, 1.0, 1.0) == pytest.approx(clip.stroke_s)


def test_a_unit_s_clip_starts_where_the_arm_is_held_not_at_rest() -> None:
    from pepin.expression.mix import rebased

    rest = dict(MOTION.rest_deg)
    held = rest | {"shoulder_lift": -60.0}
    pose = Track("shoulder_lift", "absolute", (0.0, 1.0), (rest["shoulder_lift"], -40.0))
    trace = Track("elbow_flex", "absolute", (0.0, 1.0), (50.0, 40.0))  # starts off rest: kept
    moved = rebased(Clip("arm", 0.0, 1.0, 1.0, (pose, trace)), held, rest)
    assert moved.tracks[0].values == (-60.0, -40.0) and moved.tracks[1] == trace
    arm = ArmPart(FakeArmLink(state=ArmStatus(0.0, MOTION.rest(), True, "holding")), MOTION)
    planned = arm.plan(Clip("arm", 0.0, 1.0, 1.0, (pose,)), base=held)
    assert not isinstance(planned, str)
    assert planned.trajectory.knots[0].deg[JOINTS.index("shoulder_lift")] == pytest.approx(-60.0)
    # the approach is from where the board was told to hold, not from the sagging encoders
    knot = tuple(held[j] for j in JOINTS)
    from_knot = arm.plan(Clip("arm", 0.0, 1.0, 1.0, (pose,)), base=held, start=knot)
    assert not isinstance(from_knot, str) and from_knot.approach_s == 0.0
    assert planned.approach_s > 0.0  # the encoders at rest, 40 deg below the held pose


def test_a_hold_lapses_on_the_clock_and_the_next_clip_plans_from_rest() -> None:
    now = [10.0]
    link = FakeArmLink(state=ArmStatus(0.0, MOTION.rest(), False, "off"))
    arm = ArmPart(link, MOTION, clock=lambda: now[0])
    raise_ = Track("shoulder_lift", "additive", (0.0, 0.5, 1.0), (0.0, 20.0, 20.0))
    assert arm.play(Clip("arm", 0.0, 0.5, 1.0, (raise_,), then="hold", hold_s=1.0)).ok
    assert arm.holding() is not None and arm.held_until == pytest.approx(10.0 + 0.06 + 1.0 + 1.0)
    now[0] = 12.2
    assert arm.holding() is None  # the board has rested the arm by itself
    assert arm.play(Clip("arm", 0.0, 0.5, 1.0, (raise_,))).ok
    (_, (last,)) = link.calls[-1]
    assert last.knots[0].deg == pytest.approx(MOTION.rest(), abs=0.01)
