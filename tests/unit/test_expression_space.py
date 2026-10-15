"""The gesture space (gesticulation DESIGN.md §4.3-§4.6, §11.3): the cells as poses, the nine
primitives within the arm's caps from rest and from every cell, the resolver's places (home on
the way to the middle, the person's bearing, a place's bearing and the head's share of it, the
paths), catchments, and the room's places from the goal server."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import itertools
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from pepin.arm import ArmModel
from pepin.arm_motion import load_arm_motion
from pepin.body import BodyModel, OrientedBox
from pepin.expression.animation import Book, Clip, Performance, Range, Scene
from pepin.expression.animator import EngineConfig, GestureEngine
from pepin.expression.book import book_from, load_book, with_rest
from pepin.expression.fakes import (
    FakeArmLink,
    FakeAttention,
    FakeExecutor,
    FakePlaces,
    FakePlayout,
)
from pepin.expression.markers import Cue, Marker, Part
from pepin.expression.mix import track_value
from pepin.expression.parts import ArmPart, NeckLimits, clip_look
from pepin.expression.places import GoalPlaces, bearing, latin
from pepin.expression.render import CAPS_DEG_S, CAPS_DEG_S2, SHUT, BookRenderer, Resolver
from pepin.expression.trajectory import JOINTS
from pepin.gaze import GazeSettings, overlay_from_json
from pepin.neck import NeckConfig

CONFIG = Path(__file__).resolve().parents[2] / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
BOOK = with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg)
EVERYTHING = Scene(frozenset({"arm", "head", "face", "body"}))
CELLS = {n: p for n, p in BOOK.poses.items() if n.startswith("cell_")}
HOLDS = {**CELLS, "ready": BOOK.poses["ready"]}  # where a unit may hold the arm
PRIMITIVES = ("accent", "point", "arc", "size", "weigh", "count", "draw", "stop", "offer")
ARM_S = 2.0  # the most a primitive holds the arm (DESIGN.md §7.9; critic_feasibility #3)
# ... but for the way home from the floor (point:down, 2026-10-14 evening: 0.7 s back to home,
# the board's straight way to rest from down would slide the hand down the cart's front)
LONG_WAY_HOME: dict[tuple[str, str | None], float] = {("point", "down"): 2.4}
HAND = ("wrist_link", "gripper_link", "moving_jaw_so101_v1_link")


def scripted(**bearings: float) -> FakePlaces:
    """Places at bearings (degrees, + left) from where the robot stands, 3 m away."""
    return FakePlaces(dict(bearings), distance_m=3.0)


def cue(name: str, arg: str | None, turn: int = 1, part: int = 0) -> Cue:
    return Cue(Marker("gesture", name, 0, "", arg), turn, part, 0.0)


def arm_of(performance: Performance) -> Clip | None:
    return next((c for c in performance.clips if c.part == "arm"), None)


def head_of(performance: Performance) -> Clip | None:
    return next((c for c in performance.clips if c.part == "head"), None)


def track(clip: Clip, dof: str) -> tuple[tuple[float, ...], tuple[float, ...]]:
    found = next(t for t in clip.tracks if t.dof == dof)
    return found.times, found.values


def at_stroke(clip: Clip, dof: str) -> float:
    """Where a joint of the arm stands at the clip's stroke."""
    return round(track_value(next(t for t in clip.tracks if t.dof == dof), clip.stroke_s), 3)


def pan_of(performance: Performance) -> float | None:
    """Where the arm's shoulder pan stands at its stroke (None: no arm)."""
    clip = arm_of(performance)
    return None if clip is None else at_stroke(clip, "shoulder_pan")


# -- the cells -----------------------------------------------------------------------------------


def _surface(box: OrientedBox) -> npt.NDArray[np.float64]:
    """Points on an oriented box's faces, about 5 mm apart."""
    pts = []
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        na = max(2, int(2 * box.half[a] / 0.005))
        nb = max(2, int(2 * box.half[b] / 0.005))
        ua, ub = np.meshgrid(np.linspace(-box.half[a], box.half[a], na),
                             np.linspace(-box.half[b], box.half[b], nb))  # fmt: skip
        for sign in (-1.0, 1.0):
            p = np.zeros((ua.size, 3))
            p[:, a], p[:, b], p[:, axis] = ua.ravel(), ub.ravel(), sign * box.half[axis]
            pts.append(p @ box.rotation.T + box.centre)
    return np.concatenate(pts)


def test_the_cells_stand_in_their_columns_rows_and_depths_clear_of_the_cart() -> None:
    """Each cell's gripper tip (pepin.arm's FK) in its column (middle, right, outer), row and
    depth, 15 cm across the columns; the hand 4 cm from config/body.json's boxes (the speaker
    box and the head's whole reach: scratch/gesticulation_space/cells.py)."""
    arm = ArmModel.load(CONFIG / "arm.json")
    body = BodyModel.load(CONFIG / "body.json")
    tips = {}
    for name, pose in {**CELLS, "cell_r_low_far": BOOK.poses["ready"]}.items():
        angles = {j: math.radians(pose[j]) for j in JOINTS}
        tips[name] = arm.link_poses(angles)["gripper_frame_link"].translation
        placed = zip(arm.links, arm.boxes_at(angles), strict=True)
        hand = [box.grown(-arm.margin_m) for item, box in placed if item.link in HAND]
        points = np.concatenate([_surface(box) for box in hand])
        for box in body.boxes:
            near = box.grown(0.04)
            inside = np.all((points >= near.lo) & (points <= near.hi), axis=1)
            assert not inside.any(), (name, box.name)
    columns = {"c": -0.06, "r": -0.13, "rr": -0.22}
    rows = {"low": 0.97, "mid": 1.03, "high": 1.12}
    for name, tip in tips.items():
        _, column, row, depth = name.split("_")
        assert abs(tip[1] - columns[column]) <= 0.02, name
        assert abs(tip[2] - rows[row]) <= 0.03, name
        assert abs(tip[0] - (0.13 if depth == "near" else 0.18)) <= 0.02, name
    assert tips["cell_c_mid_far"][1] - tips["cell_rr_mid_far"][1] >= 0.15


def test_every_pose_of_the_space_keeps_the_limits_with_room_to_spare() -> None:
    space = BOOK.space
    names = [*CELLS, "stop", space.open_hand, space.shut_hand, *space.loci]
    limits = dict(zip(JOINTS, MOTION.limits(), strict=True))
    for name in names:
        assert name is not None
        for joint, deg in BOOK.poses[name].items():
            low, high = limits[joint]
            if joint == "gripper" and deg == SHUT["gripper"]:
                continue  # a shut jaw stands on its limit by design (2026-10-14 evening)
            assert low + 8.0 <= deg <= high - 8.0, (name, joint)
    lo, hi = space.pan_deg
    assert limits["shoulder_pan"][0] < lo < hi < limits["shoulder_pan"][1]
    assert (MOTION.max_speed_deg_s, MOTION.max_acc_deg_s2) == (CAPS_DEG_S, CAPS_DEG_S2)


# -- the nine primitives within the caps ------------------------------------------------------


def _pinned(book: Book, end: str) -> Book:
    """``book`` with every step playing and every range at its ``end``."""
    comps = {}
    for name, comp in book.compositions.items():
        variants = []
        for v in comp.variants:
            steps = []
            for s in v.steps:
                ranges = {**book.atoms[s.atom].params, **s.params}
                params = {k: Range(getattr(r, end), getattr(r, end)) for k, r in ranges.items()}
                at = getattr(s.at_s, end)
                steps.append(replace(s, p=1.0, at_s=Range(at, at), params=params))
            variants.append(replace(v, steps=tuple(steps)))
        comps[name] = replace(comp, variants=tuple(variants))
    return replace(book, compositions=comps)


ROOM = scripted(east=-40.0, west=30.0, ahead=0.0)


def _arguments(name: str) -> list[str | None]:
    comp = BOOK.compositions[name]
    if not comp.args:
        return [None]
    args: list[str | None] = [a for a in comp.args if a != "*"]
    return [*args, *ROOM.bearings, "план а"] if "*" in comp.args else args


def _arm_clips(end: str) -> list[tuple[str, str | None, int, Clip]]:
    """Every primitive's arm clip, each variant written for each argument."""
    out = []
    book = _pinned(BOOK, end)
    for name in PRIMITIVES:
        for arg in _arguments(name):
            renderer = BookRenderer(book, resolver=Resolver(room=ROOM))
            renderer.render(name, EVERYTHING, 1, cue(name, arg))  # the first: places a referent
            for v, variant in enumerate(book.compositions[name].variants):
                clip = arm_of(renderer.render(name, EVERYTHING, 1, cue(name, arg), variant=v))
                if clip is not None and not variant.floor:
                    out.append((name, arg, v, clip))
    return out


def test_every_primitive_plans_within_the_caps_from_rest_and_from_every_cell() -> None:
    """Every step playing, its ranges at either end, every argument (a place at the pans' two
    ends and the middle, a referent): from rest and from each cell a unit may hold, the arm part
    plans it with no stretch (a WORD or PHRASE stroke is never planned past the caps), and from
    rest it holds the arm at most 2 s."""
    arm = ArmPart(FakeArmLink(), MOTION, speed_scale=1.0)
    seen = set()
    for end in ("lo", "hi"):
        for name, arg, v, clip in _arm_clips(end):
            seen.add(name)
            for base in (None, *HOLDS.values()):
                start = None if base is None else tuple(base[j] for j in JOINTS)
                planned = arm.plan(clip, base=base, start=start)
                where = (name, arg, v, end, base and base["shoulder_pan"])
                assert not isinstance(planned, str), (where, planned)
                assert planned.stretch <= 1.0 + 1e-6, (where, planned.stretch)
                if base is None:
                    budget = LONG_WAY_HOME.get((name, arg), ARM_S)
                    assert planned.trajectory.knots[-1].t_s <= budget + 1e-6, where
    assert seen == set(PRIMITIVES)


def test_a_way_to_the_middle_stop_and_me_goes_through_home_the_others_straight() -> None:
    """Straight from rest the hand would pass the speaker box at 1.1 cm on its way to the
    middle; the ways between stop, tucked and the sweep's top pass the head."""
    home = BOOK.poses["ready"]["shoulder_lift"]

    def lifts(name: str, arg: str | None, variant: int = 0, book: Book = BOOK) -> list[float]:
        performance = BookRenderer(book).render(name, EVERYTHING, 3, cue(name, arg),
                                                variant=variant)  # fmt: skip
        clip = arm_of(performance)
        assert clip is not None
        return list(track(clip, "shoulder_lift")[1])

    assert home in lifts("count", "1")  # the middle: through home
    assert home not in lifts("count", "2")  # the outer side: straight
    assert home not in lifts("stop", None, 0)  # the spread jaw on up: straight from rest
    assert home in lifts("stop", None, 1) and home in lifts("point", "me", 2)
    assert home in lifts("sympathy", None, 0, _pinned(BOOK, "hi"))  # the feeling's tucked too
    assert home not in lifts("offer", None)  # a plain pose: straight
    # after its hold the middle goes back home: the board's own way to rest from there would
    # pass the speaker box; the outer side stays for the board (or the next stroke)
    assert lifts("count", "1")[-1] == home and lifts("count", "2")[-1] != home
    greeting = arm_of(BookRenderer(_pinned(BOOK, "hi")).render("greeting", EVERYTHING, 3,
                                                               variant=0))  # fmt: skip
    assert greeting is not None and len(track(greeting, "shoulder_lift")[0]) == 3  # as before


# -- the person and the room's places ------------------------------------------------------------


def test_toward_turns_the_open_hand_at_the_person_and_an_edge_falls_to_the_floor() -> None:
    attention = FakeAttention()
    renderer = BookRenderer(BOOK, resolver=Resolver(attention=attention))
    arm = ArmPart(FakeArmLink(), MOTION, speed_scale=1.0)
    pans: dict[float, float] = {}
    for bearing_deg in (-40.0, -20.0, 0.0, 10.0, 25.0):
        attention.be(bearing_deg=bearing_deg, distance_m=1.2)
        performance = renderer.render("point", EVERYTHING, 5, cue("point", "you"))
        clip = arm_of(performance)
        assert clip is not None and at_stroke(clip, "wrist_roll") == 0.0  # the palm up
        pans[bearing_deg] = pan_of(performance) or 0.0
        for base in HOLDS.values():  # from every cell a unit may hold
            planned = arm.plan(clip, base=base, start=tuple(base[j] for j in JOINTS))
            assert not isinstance(planned, str) and planned.stretch <= 1.0 + 1e-6
    turned = [pans[b] for b in sorted(pans)]
    assert turned == sorted(turned, reverse=True)  # left of the nose: the pan inward
    assert pans[0.0] == pytest.approx(-37.0, abs=1.0)  # ahead at 1.2 m: in from the arm's side
    attention.be(bearing_deg=-70.0)  # past the pans: the hand at the range's edge, the nod
    reached: list[dict[str, Any]] = []
    renderer.resolver.record = lambda kind, fields: reached.append({kind: fields})
    performance = renderer.render("point", EVERYTHING, 5, cue("point", "you"))
    assert arm_of(performance) is not None and head_of(performance) is not None
    assert pan_of(performance) == pytest.approx(BOOK.space.pan_deg[1])
    (row,) = reached
    assert row["reach"]["hand"] == "point_open" and 10.0 < row["reach"]["short_deg"] < 20.0
    attention.be(bearing_deg=-150.0)  # behind: no arm says "there"; the floor, a nod
    performance = renderer.render("point", EVERYTHING, 5, cue("point", "you"))
    assert arm_of(performance) is None and head_of(performance) is not None
    renderer.resolver.attention = None  # no attention view: the person ahead
    assert pan_of(renderer.render("point", EVERYTHING, 5, cue("point", "you"))) == pans[0.0]


def test_a_place_gets_the_shut_hand_and_a_glance_past_the_pans_the_edge_behind_none() -> None:
    """A place the arm reaches gets the shut hand on its bearing and the head's glance; one past
    the clear pans (the printer left of them, 2026-10-13) the hand at the range's edge, as far
    as the arm reaches (logged how far short), the glance carrying the rest; one behind
    nothing (no arm says "there", no glance behind the person's back)."""
    attention = FakeAttention().be(bearing_deg=10.0, distance_m=1.5)
    room = scripted(printer=-30.0, door=70.0, shelf=170.0)
    reached: list[dict[str, Any]] = []
    resolver = Resolver(attention=attention, room=room, record=lambda k, f: reached.append(f))
    renderer = BookRenderer(BOOK, resolver=resolver)
    performance = renderer.render("point", EVERYTHING, 5, cue("point", "принтер"))
    arm, head = arm_of(performance), head_of(performance)
    assert arm is not None and head is not None
    assert at_stroke(arm, "gripper") == SHUT["gripper"]  # the finger, the jaw shut to its limit
    assert pan_of(performance) == pytest.approx(-27.0 + 30.0, abs=3.0)  # 3 m: little parallax
    assert min(track(head, "pan_deg")[1]) == pytest.approx(-25.0)  # 40 deg off: the most
    assert not reached
    performance = renderer.render("point", EVERYTHING, 5, cue("point", "door"))
    arm, head = arm_of(performance), head_of(performance)
    assert arm is not None and head is not None  # past the pans: the edge, the glance
    assert pan_of(performance) == pytest.approx(BOOK.space.pan_deg[0])  # the inner edge (left)
    assert max(track(head, "pan_deg")[1]) == pytest.approx(25.0)
    (row,) = reached
    assert row["hand"] == "point_shut" and 20.0 < row["short_deg"] < 40.0
    assert not renderer.render("point", EVERYTHING, 5, cue("point", "shelf")).clips  # behind


def test_every_head_clip_of_the_space_plays_whole_through_the_arbiter() -> None:
    knobs = json.loads((CONFIG / "knobs.json").read_text())["gaze"]
    bounds = GazeSettings(clip_max_deg=knobs["clip_max_deg"]["default"],
                          clip_max_hz=knobs["clip_max_hz"]["default"],
                          clip_still_deg=knobs["clip_still_deg"]["default"])  # fmt: skip
    limits = NeckLimits.of(NeckConfig.from_json(CONFIG / "neck.json").motion)
    room = scripted(**{f"p{b}": float(b) for b in range(-90, 91, 15)})
    for person, name in itertools.product((-30.0, 0.0, 30.0), PRIMITIVES):
        resolver = Resolver(attention=FakeAttention().be(bearing_deg=person), room=room)
        renderer = BookRenderer(BOOK, resolver=resolver)
        variants = range(len(BOOK.compositions[name].variants))
        for arg, v in itertools.product([*_arguments(name), *room.bearings], variants):
            head = head_of(renderer.render(name, EVERYTHING, 2, cue(name, arg), variant=v))
            if head is None:
                continue
            look = clip_look(head, "voice.gesture", limits)
            assert look is not None
            refusal = overlay_from_json(dict(look["target"]), bounds)
            assert not isinstance(refusal, str), (name, arg, v, refusal)


def test_the_paths_go_their_way_and_the_head_with_them() -> None:
    renderer = BookRenderer(BOOK)
    for arg, joint, way in (("up", "shoulder_lift", +1), ("down", "shoulder_lift", -1),
                            ("side", "shoulder_pan", +1), ("in", "elbow_flex", -1),
                            ("out", "elbow_flex", +1)):  # fmt: skip
        performance = renderer.render("arc", EVERYTHING, 4, cue("arc", arg), variant=0)
        clip = arm_of(performance)
        assert clip is not None and head_of(performance) is not None
        times, values = track(clip, joint)
        stroke = clip.stroke_s
        at_stroke = values[min(range(len(times)), key=lambda i: abs(times[i] - stroke))]
        assert (values[-1] - at_stroke) * way > 5.0, (arg, joint)
    up = head_of(renderer.render("arc", EVERYTHING, 4, cue("arc", "up"), variant=0))
    assert up is not None and min(track(up, "tilt_deg")[1]) == -3.0  # the head up with it
    for shape in ("round", "line", "wave"):
        clip = arm_of(renderer.render("draw", EVERYTHING, 4, cue("draw", shape), variant=0))
        assert clip is not None
        times = track(clip, "shoulder_pan")[0]
        assert list(times) == sorted(times) and times[-1] - clip.stroke_s > 0.8  # traced on


# -- catchments ----------------------------------------------------------------------------------


def test_a_referent_takes_a_spot_on_first_mention_keeps_it_and_count_takes_over() -> None:
    rows: list[tuple[str, dict[str, Any]]] = []

    def record(kind: str, fields: dict[str, Any]) -> None:
        rows.append((kind, fields))

    resolver = Resolver(room=scripted(printer=0.0), record=record)
    renderer = BookRenderer(BOOK, resolver=resolver)
    loci = BOOK.space.loci
    spot = {name: BOOK.poses[name]["shoulder_pan"] for name in loci}

    def point(name: str, turn: int, part: int = 0, variant: int | None = None) -> float | None:
        return pan_of(renderer.render("point", EVERYTHING, 6, cue("point", name, turn, part),
                                      variant=variant))  # fmt: skip

    assert point("план а", 1) == spot[loci[0]]  # the middle first
    assert point("план б", 1, 1) == spot[loci[1]]  # then the outer side: the widest pair
    assert point("план а", 2) == spot[loci[0]]  # again, the same spot
    assert ("catchment", {"referent": "план а", "cell": loci[0], "how": "again"}) in rows
    assert point("план в", 3) == spot[loci[2]]
    assert point("план г", 4) == spot[loci[1]]  # all taken: the least recently pointed one
    assert ("catchment", {"referent": "план б", "cell": loci[1], "how": "evicted"}) in rows
    renderer.render("count", EVERYTHING, 6, cue("count", "1", 5))
    assert ("catchment", {"referent": "план а", "cell": loci[0], "how": "overwritten"}) in rows
    assert ("catchment", {"referent": "#1", "cell": loci[0], "how": "new"}) in rows
    assert point("план а", 6) == spot[loci[2]]  # placed anew (logged), never two at once
    thing = next(k for k, v in enumerate(BOOK.compositions["point"].variants)
                 if v.needs == {"arg": "thing"})  # fmt: skip
    assert point("план д", 7, variant=thing) is None  # the ladder's renderings place nothing
    assert point("printer", 7) is not None  # a place: no referent
    assert all(locus.referent != "printer" for locus in resolver.loci.values())
    resolver.forget()  # the conversation closed
    assert not resolver.loci and point("план б", 8) == spot[loci[0]]


def test_the_engine_lends_its_one_resolver_to_every_renderer_and_forgets_at_the_close() -> None:
    """The engine's resolver: the renderer's, the one a switch on makes too; the attention view
    set on the engine is its person; its rows go to the record and the conversation's journal;
    the close forgets every spot."""
    rows: list[tuple[str, dict[str, Any]]] = []
    journal: list[tuple[str, dict[str, Any]]] = []
    parts = {p: FakeExecutor(p) for p in ("arm", "head", "face")}
    engine = GestureEngine(
        parts,  # type: ignore[arg-type]
        FakePlayout(), lambda: (BOOK, EngineConfig(arm_speed_scale=1.0)), on=True,
        clock=lambda: 10.0, record=lambda kind, fields: rows.append((kind, fields)),
    )  # fmt: skip
    resolver = engine.resolver
    assert engine.renderer.resolver is resolver and resolver.room is None
    view = FakeAttention()
    engine.attention = view
    assert resolver.attention is view
    resolver.room = scripted(printer=20.0)
    engine.journal = lambda kind, now, **fields: journal.append((kind, fields))
    text = "Есть один дешёвый план."
    mark = Marker("gesture", "point", text.index("дешёвый"), "", "план а")
    engine.part_queued(1, 0, Part(text, None, (mark,)))
    engine.step()
    new = ("catchment", {"referent": "план а", "cell": BOOK.space.loci[0], "how": "new"})
    assert new in rows and new in journal and resolver.loci
    assert engine.switch(True) is None and engine.renderer.resolver is resolver
    engine.closed("goodbye")
    engine.step()
    assert not resolver.loci


def test_a_name_with_no_places_to_ask_gets_a_nod_never_a_spot() -> None:
    unread = FakePlaces({"printer": 0.0}, read=False)  # the goal server never answered
    for resolver in (Resolver(), Resolver(room=unread)):
        renderer = BookRenderer(BOOK, resolver=resolver)
        performance = renderer.render("point", EVERYTHING, 6, cue("point", "printer"))
        assert arm_of(performance) is None and head_of(performance) is not None
        assert not renderer.resolver.loci


# -- the room's places ---------------------------------------------------------------------------


class Goals:
    """The goal server's two answers, scripted."""

    def __init__(self) -> None:
        self.book = {"printer": {"x": 2.0, "y": 0.0}, "diner_table": {"x": 0.0, "y": 3.0}}
        self.here: dict[str, Any] = {"pose": "tf", "x": 0.0, "y": 0.0, "yaw_deg": 90.0}
        self.down = False

    def where(self) -> dict[str, Any]:
        if self.down:
            raise OSError("refused")
        return {**self.here, "age_s": 0.1} if self.here.get("pose") == "tf" else self.here

    def places(self) -> dict[str, dict[str, float]]:
        if self.down:
            raise OSError("refused")
        return self.book


def test_the_room_s_places_are_the_goal_server_s_seen_from_where_the_robot_stands() -> None:
    now = [100.0]
    goals = Goals()
    room = GoalPlaces(goals, stale_s=5.0, clock=lambda: now[0])
    assert room.where("printer") is None and not room.known()  # nothing read yet
    goals.down = True
    room.refresh()
    assert not room.known()  # a server down from the start: still no book
    goals.down = False
    room.refresh()
    assert room.known()
    assert room.find("Printer") == "printer" and room.find("Diner Table") == "diner_table"
    assert room.find("принтер") == "printer" and latin("Принтер") == "printer"
    assert room.find("кухня") is None
    assert room.where("printer") == pytest.approx((-90.0, 2.0))
    assert room.where("diner_table") == pytest.approx((0.0, 3.0))
    goals.down = True
    room.refresh()  # the server down: the snapshot stays, and ages out
    assert room.where("printer") is not None
    now[0] += 6.0
    assert room.where("printer") is None
    goals.down, goals.here = False, {"pose": "none"}
    room.refresh()
    assert room.where("printer") is None and room.find("printer") == "printer"
    assert bearing((1.0, 1.0, 0.0), (1.0, 2.0)) == pytest.approx((90.0, 1.0))


def test_a_space_or_an_alias_that_names_nothing_is_named_at_load() -> None:
    data = json.loads((CONFIG / "animation.json").read_text())
    data["space"]["loci"].append("cell_nowhere")
    with pytest.raises(ValueError, match=r"space\.loci"):
        book_from(data)
    data = json.loads((CONFIG / "animation.json").read_text())
    data["aliases"]["size:enormous"] = "size:gigantic"
    with pytest.raises(ValueError, match=r"aliases\.size:enormous"):
        book_from(data)
