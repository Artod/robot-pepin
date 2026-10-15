"""The gesture book's structure (2026-10-14): a composition nested in a composition, the steps'
repeats, hold and loop, the atoms' auto-wrappers and the brain's vocabulary made from the book,
a mark's arguments, the snap that really shuts, the caps of a single move, and the scene's face
and attention."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import itertools
import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from test_expression_engine import Rig, marked

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import Book, Clip, Event, Performance, Range, Scene, Variant
from pepin.expression.animator import EngineConfig, engine_config
from pepin.expression.book import (
    book_from,
    load_book,
    mark_lines,
    span,
    takes,
    vocabulary,
    with_rest,
)
from pepin.expression.fakes import FakeArmLink
from pepin.expression.markers import Mark, Marker, Part, Takes, parse, read, read_mark
from pepin.expression.mix import rebased
from pepin.expression.parts import ArmPart, Planned
from pepin.expression.render import SHUT, BookRenderer, allowed, sub_seed
from pepin.expression.schedule import Priority, Request, Waiting
from pepin.expression.speaker import Attention
from pepin.expression.trajectory import JOINTS

CONFIG = Path(__file__).resolve().parents[2] / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
BOOK = with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg)
EVERYTHING = Scene(frozenset({"arm", "head", "face", "body"}))
WORDS = vocabulary(BOOK, frozenset({"grin", "smile"}))
ARM = ArmPart(FakeArmLink(), MOTION)


def _data() -> dict[str, Any]:
    return json.loads((CONFIG / "animation.json").read_text())  # type: ignore[no-any-return]


def _book(**compositions: dict[str, Any]) -> Book:
    data = _data()
    data["compositions"].update(compositions)
    return with_rest(book_from(data), MOTION.rest_deg)


def _clip(clips: tuple[Clip, ...], part: str) -> Clip:
    (clip,) = [c for c in clips if c.part == part]
    return clip


def _knots(clip: Clip, base: dict[str, float] | None = None) -> list[tuple[float, ...]]:
    planned = ARM.plan(clip, base=base)
    assert isinstance(planned, Planned), planned
    assert planned.stretch <= 1.0 + 1e-6, planned.stretch  # within the caps, never stretched
    return [k.deg for k in planned.trajectory.knots]


# -- nesting --------------------------------------------------------------------------------------


def test_a_nested_composition_draws_from_its_own_seed_and_lands_at_its_step_s_time() -> None:
    """The child draws its variant and parameters from ``sub_seed(parent's, step index)``:
    rendered alone from that seed it is the same motion, its clips moved by the parent step's
    at_s; the parent's own step merges with it on the face; the row lists the child."""
    book = _book(
        wonder={"variants": [{"steps": [
            {"atom": "face", "params": {"name": "smile", "hold_s": 0.5}},
            {"composition": "surprise", "at_s": 0.4},
        ]}]},
    )  # fmt: skip
    renderer = BookRenderer(book)
    for seed in range(20):
        performance = renderer.render("wonder", EVERYTHING, seed)
        assert performance == BookRenderer(book).render("wonder", EVERYTHING, seed)
        ((name, k),) = performance.nested
        assert name == "surprise" and k in (0, 1)  # never its floor while the others fit
        alone = BookRenderer(book).render("surprise", EVERYTHING, sub_seed(seed, 1))
        assert alone.variant == k  # the same draw: the scene's variants, by weight, no memory
        head, mine = _clip(alone.clips, "head"), _clip(performance.clips, "head")
        assert mine.tracks == head.tracks and mine.offset_s == pytest.approx(head.offset_s + 0.4)
        face = _clip(performance.clips, "face")
        assert [e.name for e in face.events] == ["smile", "surprised"]
    assert sub_seed(7, 1) == sub_seed(7, 1) != sub_seed(7, 2)


def test_a_nested_composition_s_variant_follows_the_scene_and_falls_to_its_floor() -> None:
    picky = {"variants": [{"needs": {"attention": "track"}, "steps": [{"atom": "head_back"}]},
                          {"floor": True, "steps": [{"atom": "head_nod"}]}]}  # fmt: skip
    book = _book(picky=picky, outer={"variants": [{"steps": [{"composition": "picky"}]}]})
    renderer = BookRenderer(book)
    tracking = replace(EVERYTHING, attention="track")
    assert renderer.render("outer", tracking, 3).nested == (("picky", 0),)
    assert renderer.render("outer", EVERYTHING, 3).nested == (("picky", 1),)  # nothing fits
    head = _clip(renderer.render("outer", EVERYTHING, 3).clips, "head")
    assert max(head.tracks[0].values) > 0.0  # the nod (+ down), not the head thrown back


def test_the_span_reaches_into_a_nested_composition() -> None:
    book = _book(outer={"variants": [{"steps": [
        {"composition": "surprise", "at_s": [0.4, 0.6]}]}]})  # fmt: skip
    before, after = span(book, "surprise", "head")
    assert span(book, "outer", "head") == pytest.approx((max(0.0, before - 0.4), after + 0.6))
    assert span(book, "outer", "arm") != (0.0, 0.0)  # surprise's raise, from its step


@pytest.mark.parametrize(
    ("compositions", "error"),
    [
        ({"a": {"variants": [{"steps": [{"composition": "nope"}]}]}}, "no composition 'nope'"),
        ({"a": {"variants": [{"steps": [{"composition": "b"}]}]},
          "b": {"variants": [{"steps": [{"composition": "a"}]}]}}, "a cycle: a > b > a"),
        ({"a": {"variants": [{"steps": [{"composition": "b"}]}]},
          "b": {"variants": [{"steps": [{"composition": "c"}]}]},
          "c": {"variants": [{"steps": [{"composition": "d"}]}]},
          "d": {"variants": [{"steps": [{"atom": "head_nod"}]}]}},
         "nested deeper than 3: a > b > c > d"),
        ({"a": {"variants": [{"steps": [{"composition": "laugh", "params": {"repeats": 2}}]}]}},
         "takes at_s and p, not params"),
        ({"a": {"variants": [{"steps": [{"composition": "breath"}]}]}},
         "a > breath plays breath, the idle layer's"),
    ],
)  # fmt: skip
def test_nesting_is_checked_at_load(compositions: dict[str, Any], error: str) -> None:
    with pytest.raises(ValueError, match=error):
        _book(**compositions)


def test_a_nested_composition_counts_as_its_parent_under_the_caps() -> None:
    """Its cooldown and its own per-minute cap are its parent's business: the laugh nested
    twice within its 20 s and its once a minute plays both times, one gesture each."""
    book = _book(giggle={"cooldown_s": 0.0, "variants": [{"steps": [
        {"composition": "laugh"}]}]})  # fmt: skip
    rig = Rig(book=book)
    rig.engine.part_queued(1, 0, marked("Смешно.", "giggle", 0))
    rig.engine.part_queued(2, 0, marked("Опять.", "giggle", 0))
    rig.engine.step()
    rows = rig.kinds("performance")
    assert [r["intent"] for r in rows] == ["giggle", "giggle"] and not rig.kinds("capped")
    assert all(r["nested"][0][0] == "laugh" for r in rows)
    assert rig.engine._count(lambda arm: True) == 2


# -- the steps' repeats, hold and loop ----------------------------------------------------------


@pytest.mark.parametrize(
    ("change", "error"),
    [
        (lambda d: d["atoms"]["head_nod"]["params"].update(repeats=2), "head_nod.params.repeats"),
        (lambda d: d["atoms"]["arm_pose"]["params"].update(hold_s=1), "arm_pose.params.hold_s"),
        (lambda d: d["atoms"]["head_nod"].update(loop=True), "head_nod.loop: a step's"),
        (lambda d: d["compositions"]["laugh"]["variants"][0]["steps"][1].update(loop=True),
         "only in the engine's own"),
    ],
)  # fmt: skip
def test_repeats_hold_and_loop_are_a_step_s_never_an_atom_s(
    change: Callable[[Any], Any], error: str
) -> None:
    data = _data()
    change(data)
    with pytest.raises(ValueError, match=error):
        book_from(data)


def test_a_step_s_repeats_and_hold_shape_its_atom_and_the_idle_breath_loops() -> None:
    renderer = BookRenderer(BOOK)
    once = renderer.atom_clip("head_nod", {"amp_deg": 4.0, "period_s": 0.4})
    thrice = renderer.atom_clip("head_nod", {"amp_deg": 4.0, "period_s": 0.4, "repeats": 3.0,
                                             "hold_s": 0.5})  # fmt: skip
    assert once.duration_s == pytest.approx(0.4) and thrice.duration_s == pytest.approx(1.7)
    assert thrice.tracks[0].values.count(4.0) == 3
    (breath,) = renderer.render("breath", Scene(frozenset({"head"})), 1, layer="idle").clips
    assert breath.loop and breath.layer == "idle" and breath.duration_s == pytest.approx(12.0)
    (drift,) = renderer.render("head_drift", Scene(frozenset({"head"})), 1, layer="idle").clips
    assert not drift.loop and drift.layer == "idle"


# -- the auto-wrappers and the brain's vocabulary ------------------------------------------------


def test_every_atom_that_plays_bare_has_an_auto_wrapper_and_a_written_one_wins() -> None:
    for name, composition in BOOK.compositions.items():
        if composition.wraps is None:
            continue
        atom = BOOK.atoms[name]
        (variant,) = composition.variants
        (step,) = variant.steps
        assert (step.atom, step.params, step.p, variant.floor) == (name, {}, 1.0, False)
        assert composition.callable == atom.callable and composition.cooldown_s == 0.0
        moved = {d for k in atom.keys for d in k.values}
        up = atom.part == "arm" and (bool(moved - {"gripper"}) or atom.mode == "absolute")
        assert variant.needs == ({"arm": "up"} if up else {}), name
    assert BOOK.compositions["gripper_snap"].variants[0].needs == {"arm": "up"}
    assert BOOK.compositions["gripper_click"].variants[0].needs == {}
    for name in ("arm_pose", "arm_arc", "arm_trace", "face"):  # a pose, a path, a name to choose
        assert name not in BOOK.compositions
    nod = BookRenderer(BOOK).render("head_nod", EVERYTHING, 3)
    (head,) = nod.clips
    assert head.duration_s == pytest.approx(head.tracks[0].times[-1])  # one cycle
    assert head.duration_s <= BOOK.atoms["head_nod"].params["period_s"].hi
    written = _book(head_nod={"variants": [{"steps": [{"atom": "head_back"}]}]})
    assert written.compositions["head_nod"].wraps is None
    assert takes(BOOK, BOOK.compositions["head_nod"]) == Takes(
        frozenset(), 2, frozenset({"amp_deg", "period_s", "repeats", "hold_s"})
    )


def test_the_brain_s_vocabulary_is_one_line_a_mark_made_from_the_book() -> None:
    lines = mark_lines(BOOK)
    names = [line.partition(" — ")[0].partition("[")[0].partition(":")[0] for line in lines]
    assert len(names) == len(set(names)) and all(n in WORDS.intents for n in names)
    for line in lines:
        assert 1 <= len(line.partition(" — ")[2].split()) <= 6, line
    assert {"emphasis", "breath", "head_drift", "pulse_weak"}.isdisjoint(names)
    # hello and bye are the brain's own since 2026-10-14 (Artem: no scripted start)
    assert lines[:2] == (
        "greeting — wave hello to someone you see",
        "farewell — wave goodbye to someone you see",
    )
    assert "point:you|me|up|far|down|back|right|NAME — point at a person or direction" in lines
    assert "snap[:N,fast|slow] — the gripper snapped shut" in lines
    assert "head_glance[:N|side_deg,up_deg,period_s] — head: pan, tilt" in lines
    assert "gripper_click[:N|amp_deg,period_s] — arm: gripper" in lines


# -- a mark's arguments --------------------------------------------------------------------------


def test_a_mark_s_list_reads_the_argument_that_chooses_those_that_set_and_those_left_out() -> None:
    def mark(inner: str) -> Mark | None:
        return read_mark(inner, WORDS)

    assert mark("snap:3,fast") == Mark("gesture", "snap", None, ("3", "fast"))
    assert mark("head_nod:6, 0.4") == Mark("gesture", "head_nod", None, ("6", "0.4"))
    assert mark("head_nod:amp_deg=6,Repeats = 2") == Mark(
        "gesture", "head_nod", None, ("amp_deg=6", "repeats=2")
    )
    assert mark("head_nod:3,zz,amp_deg=x,nope=2,1,2") == Mark(
        "gesture", "head_nod", None, ("3", "1"), ("zz", "amp_deg=x", "nope=2", "2")
    )  # two numbers for its two parameters, the third left out
    assert mark("point:you,fast") == Mark("gesture", "point", "you", (), ("fast",))
    assert mark("size:huge,2") == Mark("gesture", "size", "big", (), ("2",))  # through an alias
    assert mark("count:2") == Mark("gesture", "count", "2")  # an argument it lists chooses
    assert mark("beat:3") == Mark("gesture", "accent", None, (), ("3",))
    assert mark("laugh:loud") == Mark("gesture", "laugh", None, (), ("loud",))
    assert mark("arc:sideways,2") is None  # it needs an argument that chooses, got none
    assert mark("nod:3") is None  # no intent
    assert read("point:printer", WORDS) == ("gesture", "point", "printer")  # one: as before
    assert read("point:plan b", WORDS) == ("gesture", "point", "plan b")
    (part,) = parse("Вот так {snap:2,fast,loud} щёлк.", WORDS).parts
    (marker,) = part.markers
    assert (marker.name, marker.args, marker.ignored) == ("snap", ("2", "fast"), ("loud",))


def test_a_mark_s_arguments_set_its_steps_parameters() -> None:
    renderer = BookRenderer(BOOK)

    def head(args: tuple[str, ...]) -> Clip:
        return _clip(renderer.render("head_nod", EVERYTHING, 5, args=args).clips, "head")

    assert head(("3",)).tracks[0].values.count(max(head(("3",)).tracks[0].values)) == 3
    sized = head(("6", "0.4"))
    assert max(sized.tracks[0].values) == 6.0 and sized.duration_s == pytest.approx(0.4)
    assert max(head(("amp_deg=5",)).tracks[0].values) == 5.0
    snap = BOOK.compositions["snap"]
    settings = renderer.settings(snap, ("2", "slow", "zz"))
    assert {(s.param, s.atom, s.value) for s in settings} == {
        ("period_s", "gripper_snap", Range(1.0, 1.1)), ("repeats", "gripper_snap", Range(2, 2)),
    }  # fmt: skip
    assert renderer.settings(snap, ("repeats=3",))[0].value == Range(3, 3)
    scene = replace(EVERYTHING, arm_at="rest")
    arm = _clip(renderer.render("snap", scene, 2, variant=1, args=("2", "slow")).clips, "arm")
    (gripper,) = [t for t in arm.tracks if t.dof == "gripper" and SHUT["gripper"] in t.values]
    assert gripper.values.count(SHUT["gripper"]) == 4  # shut and held, twice
    shut = gripper.times[gripper.values.index(SHUT["gripper"])]
    assert 0.35 * 1.0 <= shut - gripper.times[0] <= 0.35 * 1.1 + 1e-9  # slow's period


def test_the_engine_logs_a_mark_s_arguments_and_those_left_out() -> None:
    rig = Rig()
    nod = Marker("gesture", "head_nod", 4, "", None, ("3",), ("loud",))
    rig.engine.part_queued(1, 0, Part("Да, да, да.", None, (nod,)))
    rig.engine.step()
    assert rig.kinds("ignored") == [{"intent": "head_nod", "args": ["loud"], "turn": 1, "part": 0}]
    (row,) = rig.kinds("performance")
    assert row["args"] == ["3"] and "nested" not in row


# -- the snap ------------------------------------------------------------------------------------


def test_the_snap_shuts_the_gripper_and_opens_it_again_to_the_held_pose_s_opening() -> None:
    """On a held pose (a unit's: the planner moves the clip's @from keys onto it) the jaw goes
    to the book's shut (the jaw's limit, -10: the evening of 2026-10-14) and back to that
    pose's opening, the arm still; from rest the arm rises to ready first and the snap opens to
    ready's; the floor is a glance. Every version at either tempo and repeats keeps the caps
    unstretched, from ready and from raised."""
    snap = BOOK.compositions["snap"]
    assert snap.bind == "phrase" and [v.needs for v in snap.variants] == [
        {"arm": "up"}, {"arm": "rest"}, {},
    ] and snap.variants[2].floor  # fmt: skip
    shut_deg = SHUT["gripper"]
    assert BOOK.atoms["gripper_snap"].params["shut_deg"] == Range(shut_deg, shut_deg)
    assert shut_deg == MOTION.limits()[JOINTS.index("gripper")][0] == -10.0  # the jaw's end
    g = JOINTS.index("gripper")
    renderer = BookRenderer(BOOK)
    for args in ((), ("fast",), ("slow",), ("2", "fast")):
        for seed in range(6):
            for pose in ("ready", "raised"):
                held = dict(BOOK.poses[pose])
                up = replace(EVERYTHING, arm_at=pose)
                clip = _clip(renderer.render("snap", up, seed, variant=0, args=args).clips, "arm")
                assert (
                    rebased(clip, held, BOOK.poses["rest"]).tracks[0].values[0]
                    != clip.tracks[0].values[0]
                )  # rendered from rest's @from; the planner moves it onto the pose
                knots = _knots(clip, held)
                assert min(k[g] for k in knots) == pytest.approx(SHUT["gripper"])
                assert knots[0][g] == knots[-1][g] == pytest.approx(held["gripper"])
                for j, joint in enumerate(JOINTS):
                    if joint != "gripper":
                        assert {round(k[j], 6) for k in knots} == {held[joint]}, joint
            rest = replace(EVERYTHING, arm_at="rest")
            clip = _clip(renderer.render("snap", rest, seed, variant=1, args=args).clips, "arm")
            knots = _knots(clip)
            ready = BOOK.poses["ready"]["gripper"]
            opened = next(i for i, k in enumerate(knots) if k[g] >= ready - 0.01)
            assert min(k[g] for k in knots[opened:]) == pytest.approx(SHUT["gripper"])  # snapped
            assert knots[-1] == pytest.approx(tuple(BOOK.poses["ready"][j] for j in JOINTS))
    floor = renderer.render("snap", EVERYTHING, 1, variant=2)
    assert [c.part for c in floor.clips] == ["head"]


def test_every_snap_of_the_book_sits_on_a_held_pose_or_after_its_variant_s_raise() -> None:
    """gripper_snap shuts from where the arm stands and opens again there; at rest the jaw
    stands shut already, so a step that snaps either needs the arm up (a unit's pose under it,
    no arm step of its own before it) or starts, at its earliest, after its variant's last
    absolute arm step has arrived, at its latest, on a pose with the jaw open (an earlier start
    sets the arm on that pose before it gets there). The steps as written: a mark's tempo word
    may move them (the snap's slow from rest starts 0.05 s early, 0.46 deg short of ready)."""
    snapping = set()
    for name, composition in BOOK.compositions.items():
        for k, variant in enumerate(composition.variants):
            arrived: float | None = None
            opening = 0.0
            for step in variant.steps:
                atom = BOOK.atoms.get(step.atom)
                if atom is None or atom.part != "arm":
                    continue
                ranges = {**atom.params, **step.params}
                period, repeats = ranges["period_s"], ranges.get("repeats", Range(1.0, 1.0))
                where = (name, k)
                if step.atom == "gripper_snap":
                    snapping.add(where)
                    if arrived is None:
                        assert variant.needs.get("arm") == "up", where
                        continue
                    assert step.at_s.lo - atom.stroke * period.hi >= arrived - 1e-9, where
                    assert opening >= 10.0, where  # a jaw open enough to be seen shutting
                elif atom.mode == "absolute":
                    arrived = step.at_s.hi + (repeats.hi - atom.stroke) * period.hi
                    named = [p.removeprefix("@pose:") for p in step.choices.get("pose", ())]
                    opening = min((BOOK.poses.get(n, SHUT)["gripper"] for n in named),
                                  default=SHUT["gripper"])  # fmt: skip
    assert {("laugh", 0), ("laugh", 1), ("laugh", 2), ("greeting", 1), ("evil_click", 0),
            ("snap", 0), ("snap", 1), ("gripper_snap", 0)} == snapping  # fmt: skip
    for composition in BOOK.compositions.values():
        if composition.wraps is None:
            assert all(s.atom != "gripper_click" for v in composition.variants for s in v.steps)


def test_the_laugh_the_greeting_and_evil_click_shut_the_gripper_on_their_raise() -> None:
    """The click that shuts (2026-10-14): each variant that snaps raises the hand, then the jaw
    goes from the pose's opening to the book's shut and back, within the caps unstretched; the
    laugh from rest and from a unit's ready."""
    g = JOINTS.index("gripper")
    renderer = BookRenderer(BOOK)
    for name, k, pose, arg in (("laugh", 0, "ready", None), ("laugh", 1, "ready", None),
                               ("laugh", 2, "ready", "evil"), ("greeting", 1, "raised", None),
                               ("evil_click", 0, "raised", None)):  # fmt: skip
        bases = (None, dict(BOOK.poses["ready"])) if name == "laugh" else (None,)
        for seed, base in itertools.product(range(8), bases):
            performance = renderer.render(name, EVERYTHING, seed, variant=k, arg=arg)
            knots = _knots(_clip(performance.clips, "arm"), base)
            opening = BOOK.poses[pose]["gripper"]
            opened = next(i for i, kn in enumerate(knots) if kn[g] >= opening - 0.01)
            shut = min(kn[g] for kn in knots[opened:])
            assert shut == pytest.approx(SHUT["gripper"]), (name, k, seed)
            assert knots[-1][g] == pytest.approx(opening), (name, k, seed)


def test_no_pose_of_the_book_stands_on_rest_s_value_of_a_joint() -> None:
    """mix.rebased moves every key of a track that starts on rest's value and stands there
    again (the snap's @from): a pose holding rest's angle on a joint would be moved too."""
    rest = BOOK.poses["rest"]
    for name, pose in BOOK.poses.items():
        if name != "rest":
            assert all(abs(pose[j] - rest[j]) > 1e-6 for j in JOINTS), name


# -- the caps of a single move -----------------------------------------------------------------


def test_a_single_move_counts_under_per_part_alone_unless_atoms_counted() -> None:
    def reply(rig: Rig) -> list[str]:
        for k, (text, name) in enumerate((("Да.", "agreement"), ("Кивок.", "head_nod"),
                                          ("Хм.", "doubt"), ("Ну.", "shrug"))):  # fmt: skip
            rig.engine.part_queued(1, k, marked(text, name, 0))
        both = (Marker("gesture", "head_shake", 0), Marker("gesture", "gripper_click", 4))
        rig.engine.part_queued(2, 0, Part("Нет, нет.", None, both))
        rig.engine.step()
        return [r["why"] for r in rig.kinds("capped")]

    rig = Rig()
    assert reply(rig) == ["per_part"]  # the second single move of one sentence
    assert rig.engine._count(lambda arm: True) == 3  # agreement, doubt, shrug
    counted = Rig(caps=replace(EngineConfig().caps, atoms_counted=True))
    assert reply(counted) == ["per_reply", "per_part"]  # the nod took the reply's third
    assert not engine_config(
        json.loads((CONFIG / "expression.json").read_text())
    ).caps.atoms_counted
    assert engine_config({"caps": {"atoms_counted": True}}).caps.atoms_counted


# -- the scene's face and attention --------------------------------------------------------------


class View:
    def __init__(self, phase: str) -> None:
        self.phase = phase

    def attention(self) -> Attention:
        return Attention(phase=self.phase)  # type: ignore[arg-type]


def test_the_scene_carries_the_face_s_expression_and_attention_s_phase() -> None:
    variant = Variant((), needs={"face": "grin", "attention": "track"})
    assert allowed(variant, replace(EVERYTHING, face="grin", attention="track"))
    assert not allowed(variant, replace(EVERYTHING, face="smile", attention="track"))
    assert not allowed(variant, replace(EVERYTHING, face="grin"))
    assert allowed(Variant((), needs={"face": "any"}), EVERYTHING)  # the part ready, as before
    rig = Rig()
    request = Request("laugh", None, Priority.MARK, "marker")
    assert rig.engine._known_scene(EVERYTHING, request) == EVERYTHING  # no face, no view
    rig.engine.attention = View("track")
    clip = Clip("face", 0.0, 0.0, 2.0, events=(Event(0.0, "grin", 1.0), Event(1.0, "smile")))
    face = Request("face:grin", None, Priority.MARK, "marker", clips=(clip,))
    assert rig.engine._send(Waiting(Performance("face:grin", 0, 0, (clip,)), clip, face),
                            rig.clock.now, None).ok  # fmt: skip
    rig.clock.now += 0.5
    scene = rig.engine._known_scene(EVERYTHING, request)
    assert (scene.face, scene.attention) == ("grin", "track")
    rig.clock.now += 1.0  # the grin's hold lapsed, the smile shows
    assert rig.engine._known_scene(EVERYTHING, request).face == "smile"
    rig.clock.now += 1.0  # the clip is over
    assert rig.engine._known_scene(EVERYTHING, request).face is None
    with pytest.raises(ValueError, match=r"needs\.attention"):
        _book(
            x={"variants": [{"needs": {"attention": "napping"}, "steps": [{"atom": "head_nod"}]}]}
        )
