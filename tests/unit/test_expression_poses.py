"""The arm poses Artem shaped by hand on the evening of 2026-10-14 (scratch/arm_poses ids 2-13:
the salute, the heart, the arm far out, the finger up, down, back and to the right) and what the
book builds on them: the jaw shut to its limit, each pose's place and its room to breathe, every new
mark planned within the caps from rest and from every pose a unit may hold, their ways clear of
the body and of the head (the box model of scratch/arm_poses_1012/geometry.py, ported here:
config/body.json's boxes, the speaker box, the pan servo, an estimated head box swept over the
neck's reach), where a hold ends, and the brain's new marks."""

import itertools
import json
import math
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from pepin.arm import ArmModel
from pepin.arm_motion import GRIPPER_ZERO_DEG, load_arm_motion
from pepin.expression.animation import Book, Clip, Range, Scene
from pepin.expression.book import load_book, mark_lines, vocabulary, with_rest
from pepin.expression.fakes import FakeArmLink, FakeAttention
from pepin.expression.markers import Cue, Marker, read
from pepin.expression.parts import ArmPart, Planned
from pepin.expression.render import SHUT, BookRenderer, Resolver
from pepin.expression.trajectory import JOINTS
from pepin.neck import NeckConfig, angle_limits, pan_pivot

CONFIG = Path(__file__).resolve().parents[2] / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
BOOK = with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg)
EVERYTHING = Scene(frozenset({"arm", "head", "face", "body"}))
ARM = ArmModel.load(CONFIG / "arm.json")
Array = npt.NDArray[np.float64]

NEW = ("salute", "heart", "far", "up_low", "up", "up_open", "down_near", "down", "back_near",
       "back", "right_out", "right_up")  # fmt: skip
SHUT_POSES = ("emphasis", "tucked", "stop", "point_shut", "salute", "up_low", "up", "down_near",
              "down", "back_near", "back", "right_out", "right_up")  # fmt: skip
# The marks built on the new poses: a composition and the argument that chooses its variant
MARKS: tuple[tuple[str, str | None], ...] = (
    ("salute", None), ("crazy", None), ("point", "me"), ("point", "up"), ("point", "far"),
    ("point", "down"), ("point", "back"), ("point", "right"), ("stop", None),
)  # fmt: skip
# The poses a clip never ends on (its last arm step goes home, or on to another pose, after its
# hold): a unit never holds them, so no mark starts from them
NEVER_HELD = frozenset({"heart", "salute", "tucked", "down", "cell_c_low_far", "cell_c_mid_far",
                        "cell_c_high_far"})  # fmt: skip
# The stretch the planner may put on a clip chained from a pose far from its first leg's start
# (the via home's share is timed at render for the leg from rest: from far the elbow has
# 144 deg to home where rest has 57): measured 2.13 at most (scratch/arm_poses_1014/
# stretch_from_held.py); the engine schedules by the stretched stroke, a word-bound mark too
# slow for its word falls to its floor
CHAINED_STRETCH = 2.2
# The breath's amplitude per joint (config/animation.json breath.arm): a pose this far inside
# its limits breathes; one at a stop would lose its breath on every clip
BREATH_DEG = {"shoulder_pan": 1.5, "shoulder_lift": 2.0, "elbow_flex": 2.5, "wrist_flex": 2.0,
              "wrist_roll": 4.0, "gripper": 0.0}  # fmt: skip
NEAR_REST_DEG = 15.0  # the parked arm lies on the rim in the box model: judged once it has left
BODY_CM = 1.5  # the hand's least distance from the body on a way (the space's bar)
# ... and the upper arm's and forearm's: the right-side limit Artem's hand found (right_out,
# nothing touching the mast) has the upper arm 1.5 cm from the mast in the model, 1.497 on the
# way in: the model's margin, not a collision
ARM_CM = 1.4
HEAD_CM = 1.5
# The marks whose hand passes the head's pan sweep by nature, and the held poses any way out of
# which does: judged against the head facing ahead (pan 0, any pitch) instead of its whole reach
# (poses.back_near); the salute's hand stands at the head's right eye by Artem's design, so its
# head clearance is not the model's to judge (poses.salute: the model has 0.7 cm at pan 0 at
# the worst pitch, within its own uncertainty)
HEAD_AHEAD = {("point", "back")}
HEAD_SWEEP_POSES = frozenset({"back_near", "back"})
HEAD_FREE = {("salute", None), ("crazy", None)}
# The poses a unit may hold when a new mark starts, for the ways' clearance (a share of the
# book's: home, the raise, the stop, three cells and every new pose a clip ends on)
HOLDS = ("ready", "raised", "stop", "cell_rr_mid_far", "cell_rr_high_far", "cell_r_mid_near",
         "far", "up_low", "up", "up_open", "down_near", "back_near", "back", "right_out",
         "right_up")  # fmt: skip


# -- the box model (scratch/arm_poses_1012/geometry.py, the parts a test needs) ------------------

_BODY = {b["name"]: b for b in json.loads((CONFIG / "body.json").read_text())["boxes"]}
BOXES: dict[str, tuple[Array, Array]] = {
    "cart": (np.array(_BODY["cart"]["min_m"]), np.array(_BODY["cart"]["max_m"])),
    "speaker": (np.array([-0.03, -0.07, 0.78]), np.array([0.06, 0.13, 0.87])),
    "top_load": (
        np.array([_BODY["top_load"]["min_m"][0], -0.07, _BODY["top_load"]["min_m"][2]]),
        np.array(_BODY["top_load"]["max_m"]),
    ),
    "mast": (np.array(_BODY["mast"]["min_m"]), np.array(_BODY["mast"]["max_m"])),
    "pan_servo": (np.array([-0.078, -0.025, 1.0]), np.array([-0.038, 0.025, 1.11])),
}
_NECK = NeckConfig.from_json(CONFIG / "neck.json")
PIVOT = np.array(pan_pivot(_NECK))
_, (PITCH_LO, PITCH_HI) = angle_limits(_NECK)
HEAD_LO = np.array([-0.03, -0.05, -0.03])  # the head box in the tilt frame: an ESTIMATE
HEAD_HI = np.array([0.045, 0.05, 0.115])
PITCHES = np.radians(np.arange(-20.0, 66.0, 5.0))
HAND = ("wrist_link", "gripper_link", "moving_jaw_so101_v1_link")
ARM_LINKS = ("upper_arm_link", "lower_arm_link")


def _face_grid(half: Array, step: float) -> Array:
    """Points on a box's six faces in its own frame, about ``step`` apart."""
    pts = []
    for axis in range(3):
        a, b = [i for i in range(3) if i != axis]
        na = max(2, math.ceil(2 * half[a] / step) + 1)
        nb = max(2, math.ceil(2 * half[b] / step) + 1)
        ua, ub = np.meshgrid(np.linspace(-half[a], half[a], na), np.linspace(-half[b], half[b], nb))
        for sign in (-1.0, 1.0):
            p = np.zeros((ua.size, 3))
            p[:, a], p[:, b], p[:, axis] = ua.ravel(), ub.ravel(), sign * half[axis]
            pts.append(p)
    return np.concatenate(pts)


_GRIDS = {
    item.box.name: _face_grid(item.box.half, 0.008 if item.link in HAND else 0.015)
    for item in ARM.links
}


def _head_profile() -> tuple[Array, Array]:
    """The head's swept solid about the pan axis: per height, the largest radius any point of
    the head box reaches over the pitch range."""
    zs = np.arange(0.95, 1.30, 0.001)
    rmax = np.full(zs.size, -1.0)
    grid = np.meshgrid(*[np.linspace(HEAD_LO[i], HEAD_HI[i], 12) for i in range(3)])
    pts = np.array(grid).reshape(3, -1).T
    for pitch in np.arange(PITCH_LO, PITCH_HI + 0.005, math.radians(0.5)):
        x = pts[:, 0] * math.cos(pitch) + pts[:, 2] * math.sin(pitch)
        z = -pts[:, 0] * math.sin(pitch) + pts[:, 2] * math.cos(pitch) + PIVOT[2]
        idx = np.clip(np.round((z - zs[0]) / 0.001).astype(int), 0, zs.size - 1)
        np.maximum.at(rmax, idx, np.hypot(x, pts[:, 1]))
    keep = rmax >= 0
    return np.asarray(zs[keep], dtype=np.float64), np.asarray(rmax[keep], dtype=np.float64)


HEAD_Z, HEAD_R = _head_profile()
_OUTLINE = np.concatenate([
    np.stack([HEAD_R, HEAD_Z], axis=1),
    np.stack([np.linspace(0, HEAD_R[0], 30), np.full(30, HEAD_Z[0])], axis=1),
    np.stack([np.linspace(0, HEAD_R[-1], 30), np.full(30, HEAD_Z[-1])], axis=1),
])  # fmt: skip


def _head_boxes() -> list[tuple[Array, Array, Array]]:
    """The head box (centre, rotation, half) at pan 0 and every sampled pitch."""
    out = []
    for pitch in PITCHES:
        c, s = math.cos(pitch), math.sin(pitch)
        rot = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
        out.append((rot @ ((HEAD_LO + HEAD_HI) / 2.0) + PIVOT, rot, (HEAD_HI - HEAD_LO) / 2.0))
    return out


HEAD_AT_0 = _head_boxes()


def _aabb_distance(points: Array, lo: Array, hi: Array) -> float:
    d = np.maximum(np.maximum(lo - points, points - hi), 0.0)
    return float(np.sqrt((d * d).sum(axis=1)).min())


def _obb_distance(points: Array, centre: Array, rotation: Array, half: Array) -> float:
    with np.errstate(all="ignore"):  # macOS Accelerate raises spurious matmul warnings (numpy 2)
        local = np.abs((points - centre) @ rotation) - half
    d = np.maximum(local, 0.0)
    return float(np.sqrt((d * d).sum(axis=1)).min())


def _sweep_distance(points: Array) -> float:
    """The least distance from the points to the head's swept solid."""
    rho = np.hypot(points[:, 0] - PIVOT[0], points[:, 1] - PIVOT[1])
    z = points[:, 2]
    q = np.stack([rho, z], axis=1)
    diff = q[:, None, :] - _OUTLINE[None, :, :]
    d = np.sqrt((diff * diff).sum(axis=2)).min(axis=1)
    inside = (z >= HEAD_Z[0]) & (z <= HEAD_Z[-1]) & (rho <= np.interp(z, HEAD_Z, HEAD_R))
    d[inside] = 0.0
    return float(d.min())


def _surface(pose: Mapping[str, float]) -> tuple[Array, Array]:
    """The hand's and the arm links' sampled surface points (cart base_link, metres) at a pose
    (degrees, config/arm.json's convention), the link boxes ungrown."""
    links = ARM.link_poses({j: math.radians(pose[j]) for j in JOINTS})
    hand: list[Array] = []
    arm: list[Array] = []
    for item in ARM.links:
        if item.link not in HAND and item.link not in ARM_LINKS:
            continue
        placed = item.box.placed(links[item.link])
        with np.errstate(all="ignore"):  # the spurious matmul warnings again
            pts = _GRIDS[item.box.name] @ placed.rotation.T + placed.centre
        (hand if item.link in HAND else arm).append(pts)
    return np.concatenate(hand), np.concatenate(arm)


def clearance(pose: Mapping[str, float]) -> dict[str, float]:
    """A pose's least distances in cm: the hand and the arm links to the body boxes, each to
    the head's whole reach, and each to the head facing ahead at any pitch."""
    hand, arm = _surface(pose)
    return {
        "hand_body": min(_aabb_distance(hand, lo, hi) for lo, hi in BOXES.values()) * 100,
        "arm_body": min(_aabb_distance(arm, lo, hi) for lo, hi in BOXES.values()) * 100,
        "hand_head": _sweep_distance(hand) * 100,
        "arm_head": _sweep_distance(arm) * 100,
        "hand_head0": min(_obb_distance(hand, *box) for box in HEAD_AT_0) * 100,
        "arm_head0": min(_obb_distance(arm, *box) for box in HEAD_AT_0) * 100,
    }


def tip(pose: Mapping[str, float]) -> Array:
    """The gripper frame's origin at a pose (cart base_link, metres)."""
    links = ARM.link_poses({j: math.radians(pose[j]) for j in JOINTS})
    out: Array = links["gripper_frame_link"].translation
    return out


def away(pose: Mapping[str, float]) -> bool:
    return bool(max(abs(pose[j] - BOOK.poses["rest"][j]) for j in JOINTS) > NEAR_REST_DEG)


def line(a: Mapping[str, float], b: Mapping[str, float], n: int) -> list[dict[str, float]]:
    return [{j: a[j] + (b[j] - a[j]) * k / (n - 1) for j in JOINTS} for k in range(n)]


# -- the renderings and the plans ---------------------------------------------------------------


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


PINNED = {end: _pinned(BOOK, end) for end in ("lo", "hi")}


def cue(name: str, arg: str | None) -> Cue:
    return Cue(Marker("gesture", name, 0, "", arg), 1, 0, 0.0)


def arm_clips(name: str, arg: str | None, end: str) -> list[tuple[int, Clip]]:
    """Every arm clip of the mark's variants written for its argument (the floors aside)."""
    book = PINNED[end]
    renderer = BookRenderer(book)
    out = []
    for v, variant in enumerate(book.compositions[name].variants):
        if variant.floor or (arg is not None and variant.needs.get("arg") != arg):
            continue
        performance = renderer.render(name, EVERYTHING, 1, cue(name, arg), variant=v)
        clip = next((c for c in performance.clips if c.part == "arm"), None)
        if clip is not None:
            out.append((v, clip))
    return out


def plan(clip: Clip, base: dict[str, float] | None) -> Planned:
    arm = ArmPart(FakeArmLink(), MOTION, speed_scale=1.0)
    start = None if base is None else tuple(base[j] for j in JOINTS)
    planned = arm.plan(clip, base=base, start=start)
    assert isinstance(planned, Planned), planned
    return planned


# -- the shut jaw ----------------------------------------------------------------------------------


def test_the_jaw_s_shut_is_its_limit_for_the_snap_and_every_shut_pose() -> None:
    """The jaw pressed shut by hand read -12.0 (scratch/arm_poses id1, 2026-10-14 evening):
    the shut is config/arm.json's gripper limit, -10.0 (the servo's calibrated end; a snap to 0
    left the tip a centimetre open), one number for the resolver's finger, gripper_snap's
    shut_deg and every pose with a shut jaw; the breath leaves the jaw alone (a shut jaw stands
    on its limit, where a breath would cost the whole clip its breath)."""
    shut = MOTION.limits()[JOINTS.index("gripper")][0]
    assert shut == GRIPPER_ZERO_DEG == SHUT["gripper"] == -10.0
    assert BOOK.atoms["gripper_snap"].params["shut_deg"] == Range(shut, shut)
    for name in SHUT_POSES:
        assert BOOK.poses[name]["gripper"] == shut, name
    assert BOOK.breath["arm"].joints["gripper"].amp_deg == 0.0
    g = JOINTS.index("gripper")
    renderer = BookRenderer(BOOK)
    for pose in ("ready", "raised"):
        clip = next(c for c in renderer.render("snap", replace(EVERYTHING, arm_at=pose), 3,
                                               variant=0).clips if c.part == "arm")  # fmt: skip
        knots = plan(clip, dict(BOOK.poses[pose])).trajectory.knots
        assert min(k.deg[g] for k in knots) == pytest.approx(shut)  # to the limit, accepted


def test_the_board_takes_the_shut_on_its_limit_and_aims_at_its_calibrated_end() -> None:
    """The board's own check (pepin.arm_driver: config/arm.json's limits narrowed by the servos'
    EEPROM range, inclusive, on the knots as the wire carries them) takes the snap planned from
    a held raise, its jaw on -10.0 exactly; the gripper's goal there is its Min_Position_Limit to
    the tick (the calibrated closed end; the hand pressed the jaw 23 ticks further), never a
    tick past it."""
    from test_arm_driver import CAL, make_core, run, send

    raised = replace(EVERYTHING, arm_at="raised")
    performance = BookRenderer(BOOK).render("snap", raised, 3, variant=0)
    clip = next(c for c in performance.clips if c.part == "arm")
    trajectory = plan(clip, dict(BOOK.poses["raised"])).trajectory
    g = JOINTS.index("gripper")
    assert min(k.deg[g] for k in trajectory.knots) == -10.0
    core, bus = make_core()
    assert send(core, trajectory, 0.0)["status"] == "accepted"
    run(core, 0.0, trajectory.knots[-1].t_s + 1.5)
    goals = [v["servo6"] for name, v in bus.writes if name == "Goal_Position"]
    goals += [v["servo6"][2] for name, v in bus.writes if name == "block" and "servo6" in v]
    assert min(goals) == CAL["servo6"][0] == 1837


def test_the_laptop_refuses_nothing_the_board_takes_for_the_wire_s_rounding() -> None:
    """The wire rounds a knot's time to the millisecond: a 39.5 ms step sent as 39 ms runs 1.3 %
    faster than the curve, which the board forgives (WIRE_SLACK) and the laptop's check of its
    own curve against the exact caps never sees; the limits are judged on the wire's angles
    (a jaw on its limit) and with no slack."""
    from pepin.arm_driver import WIRE_SLACK
    from pepin.expression.mix import check_knots
    from pepin.expression.parts import refused, wire

    speed, acc, limits = MOTION.max_speed_deg_s, MOTION.max_acc_deg_s2, MOTION.limits()
    w = JOINTS.index("wrist_flex")
    rest = [MOTION.rest_deg[j] for j in JOINTS]
    curve = [(k * 0.0395, [d - (0.995 * speed * k * 0.0395 if i == w else 0.0)
                           for i, d in enumerate(rest)]) for k in range(8)]  # fmt: skip
    assert check_knots(curve, limits, speed, acc, JOINTS) is None
    assert "deg/s, over" in str(check_knots(wire(curve), limits, speed, acc, JOINTS))
    assert check_knots(wire(curve), limits, speed, acc, JOINTS, slack=WIRE_SLACK) is None
    assert refused(curve, limits, speed, acc) is None
    g = JOINTS.index("gripper")
    shut = [(t, [-10.0000001 if i == g else d for i, d in enumerate(deg)]) for t, deg in curve]
    assert refused(shut, limits, speed, acc) is None  # sent as -10.0
    past = [(t, [-10.006 if i == g else d for i, d in enumerate(deg)]) for t, deg in curve]
    assert "gripper -10.0 deg outside" in str(refused(past, limits, speed, acc))  # sent as -10.01
    fast = [(t, [d * 1.1 if i == w else d for i, d in enumerate(deg)]) for t, deg in curve]
    assert "deg/s, over" in str(refused(fast, limits, speed, acc))


# -- the poses -------------------------------------------------------------------------------------


def test_the_new_poses_stand_where_artem_shaped_them_and_breathe() -> None:
    """Each pose's tip where its note says (the FK of config/arm.json's mount): far ahead and
    up, the finger straight up beside the head, down ahead of the cart, back behind
    the robot, right out at the side, the heart on the chest, the salute at the head; every
    joint inside its limits with the breath's amplitude to spare (the captures past a limit
    held 3 deg inside it, back's lift and wrist pulled off their stops)."""
    limits = dict(zip(JOINTS, MOTION.limits(), strict=True))
    for name in NEW:
        pose = BOOK.poses[name]
        for joint, (lo, hi) in limits.items():
            margin = BREATH_DEG[joint]
            assert lo + margin <= pose[joint] <= hi - margin, (name, joint)
    at = {name: tip(BOOK.poses[name]) for name in NEW}
    assert at["far"][0] > 0.3 and at["far"][2] > 1.1
    assert at["up"][2] > at["up_low"][2] > 1.09 and at["up"][1] < -0.1 and at["up"][0] < 0.1
    assert at["up_open"][2] == at["up"][2] and BOOK.poses["up_open"]["gripper"] >= 60.0
    assert at["down"][2] < at["down_near"][2] < 0.8 and at["down"][0] > BOXES["cart"][1][0] + 0.04
    assert at["back"][0] < at["back_near"][0] < -0.14 and at["back"][2] > 1.05
    assert at["right_up"][1] < at["right_out"][1] < -0.28 and abs(at["right_up"][0]) < 0.03
    assert abs(at["heart"][1]) < 0.05 and at["heart"][0] > 0.05 and at["heart"][2] < 1.0
    assert at["salute"][2] > 1.15 and -0.1 < at["salute"][1] < -0.05
    rest = BOOK.poses["rest"]
    for name in NEW:  # mix.rebased moves keys that stand on rest's value: none may
        assert all(abs(BOOK.poses[name][j] - rest[j]) > 1e-6 for j in JOINTS), name


def test_the_new_poses_keep_their_distance_from_the_body_and_the_head() -> None:
    """At each pose: the hand and the arm links clear of the body boxes, the hand clear of the
    head's whole reach but for the salute (at the head's right eye by design) and the back
    poses (inside the pan sweep, 4 cm from the head facing ahead); the right-side limit's
    upper arm nearest the mast of all, by less than 2 cm."""
    for name in NEW:
        c = clearance(BOOK.poses[name])
        assert c["hand_body"] >= 2.0, (name, c)
        assert c["arm_body"] >= ARM_CM, (name, c)
        if name == "salute":
            continue
        assert c["hand_head0"] >= 4.0, (name, c)
        if name not in ("back_near", "back"):
            assert c["hand_head"] >= 2.5, (name, c)
    assert clearance(BOOK.poses["right_out"])["arm_body"] < 2.0
    assert clearance(BOOK.poses["up_open"])["hand_head"] == pytest.approx(
        clearance(BOOK.poses["up"])["hand_head"], abs=0.05
    )  # the jaw spreads away from the head


# -- every new mark from rest and from every held pose --------------------------------------------


def test_every_new_mark_plans_within_the_caps_from_rest_and_from_every_pose() -> None:
    """Every step playing, its ranges at either end: from rest and from home the arm part
    plans each new mark with no stretch (a stroke never planned past the caps); from every
    other pose of the book (a unit's held pose, the clip started there by the planner) it is
    never refused and stretched at most :data:`CHAINED_STRETCH` (point:far, straight, never);
    from rest a pointing mark holds the arm at most 2 s, a salute or the twirl at most 5 s (the
    twirl: the raise, two rocks of the wrist and the way home)."""
    seen = set()
    for end, (name, arg) in itertools.product(("lo", "hi"), MARKS):
        for v, clip in arm_clips(name, arg, end):
            seen.add((name, arg))
            for base_name, base in (("rest", None), *BOOK.poses.items()):
                if base_name == "rest" and base is not None:
                    continue
                planned = plan(clip, None if base is None else dict(base))
                where = (name, arg, v, end, base_name)
                tight = base is None or base_name == "ready" or (name, arg) == ("point", "far")
                bar = 1.0 + 1e-6 if tight else CHAINED_STRETCH
                assert planned.stretch <= bar, (where, planned.stretch)
                if base is None:
                    budget = 2.0 if name in ("point", "stop") else 5.0
                    if (name, arg) == ("point", "down"):
                        budget = 2.4  # the way home from the floor, 0.7 s
                    assert planned.trajectory.knots[-1].t_s <= budget + 1e-6, where
    assert seen == set(MARKS)


@pytest.mark.slow
@pytest.mark.parametrize(("name", "arg"), MARKS)
def test_no_way_of_a_new_mark_passes_the_body_or_the_head(name: str, arg: str | None) -> None:
    """Along every planned trajectory of the mark, from rest and from every pose a unit may
    hold (the ranges at one end or the other), and along the board's straight way back to
    rest from its last knot: the hand at least 1.5 cm and the arm links 1.4 from the body boxes
    (the cart's top, the speaker box, the top load, the mast, the pan servo) and from the head's
    whole reach; pointing back, and any way out of a held back pose, judged against the head
    facing ahead (they pass the pan sweep by nature); the salute's head clearance Artem's to
    judge, not the model's."""
    holds = [("rest", None)] + [(n, BOOK.poses[n]) for n in HOLDS]
    for k, (base_name, base) in enumerate(holds):
        end = ("lo", "hi")[k % 2]
        ahead = (name, arg) in HEAD_AHEAD or base_name in HEAD_SWEEP_POSES
        for v, clip in arm_clips(name, arg, end):
            planned = plan(clip, None if base is None else dict(base))
            knots = [dict(zip(JOINTS, kn.deg, strict=True))
                     for kn in planned.trajectory.knots[::3]]  # fmt: skip
            knots += line(knots[-1], BOOK.poses["rest"], 9)
            for pose in knots:
                if not away(pose):
                    continue
                c = clearance(pose)
                where = (name, arg, v, end, base_name, c)
                assert c["hand_body"] >= BODY_CM and c["arm_body"] >= ARM_CM, where
                if (name, arg) in HEAD_FREE:
                    continue
                head = (
                    min(c["hand_head0"], c["arm_head0"])
                    if ahead
                    else min(c["hand_head"], c["arm_head"])
                )
                assert head >= HEAD_CM, where


# -- the ways: home, on to another pose, and straight ----------------------------------------------


def test_the_way_home_is_the_last_arm_step_s_and_down_ends_on_down_near() -> None:
    """The way home after a hold is the last arm step's alone: point:me goes home after the
    heart (inward of via_pan_deg, as before); point:down goes home too (space.home_after:
    straight to rest from down, or from down_near, the hand would slide down the cart's front);
    point:far goes straight to far from rest (the tightest way there is) and the others
    straight to their first pose, which hands its pose to the second as it is (no home
    between)."""
    book = PINNED["hi"]
    home = book.poses["ready"]

    def last_knots(name: str, arg: str | None) -> list[tuple[float, ...]]:
        ((_, clip),) = arm_clips(name, arg, "hi")
        return [k.deg for k in plan(clip, None).trajectory.knots]

    def visits(knots: list[tuple[float, ...]], pose: Mapping[str, float]) -> bool:
        """Whether a knot stands within a degree of the pose (a key between two 40 ms knots)."""
        return bool(
            any(all(abs(k[i] - pose[j]) < 1.0 for i, j in enumerate(JOINTS)) for k in knots)
        )

    assert last_knots("point", "me")[-1] == pytest.approx(tuple(home[j] for j in JOINTS), abs=0.05)
    down = last_knots("point", "down")
    assert visits(down, home) and visits(down, book.poses["down"])
    assert down[-1] == pytest.approx(tuple(home[j] for j in JOINTS), abs=0.05)
    assert BOOK.space.home_after == {"down": "ready"} and "down_near" in BOOK.space.via
    far = last_knots("point", "far")
    assert not visits(far, home) and not visits(far, book.poses["up_low"])
    for arg, first in (("up", "up_low"), ("back", "back_near"), ("right", "right_out")):
        knots = last_knots("point", arg)
        assert not visits(knots, home) and visits(knots, book.poses[first]), arg


def test_a_salute_or_a_heart_from_a_held_pose_goes_through_home() -> None:
    """Planned from a pose a unit holds (far, the hand far out), the salute's and the
    heart's clips pass home on the way (the via, inward of via_pan_deg): never straight across
    the speaker box."""
    home = tuple(PINNED["lo"].poses["ready"][j] for j in JOINTS)
    for name, arg in (("salute", None), ("point", "me")):
        ((_, clip),) = arm_clips(name, arg, "lo")
        knots = [k.deg for k in plan(clip, dict(BOOK.poses["far"])).trajectory.knots]
        assert knots[0] == pytest.approx(tuple(BOOK.poses["far"][j] for j in JOINTS), abs=0.05)
        assert any(all(abs(a - b) < 0.05 for a, b in zip(k, home, strict=True)) for k in knots)


def test_pointing_back_waits_for_the_head_to_face_ahead_or_left() -> None:
    """The ways to and from the back poses pass the head's pan sweep (the hand 1.6 cm from the
    head at pan -10, 0.7 at -15: scratch/arm_poses_1014/back_head_pans.py): while the head looks
    at a person further right than space.head_pan_min_deg allows, {point:back} plays its floor
    (the head's glance alone); ahead, to the left, or with no attention view, the arm points."""
    assert BOOK.space.head_pan_min_deg == {"back_near": -10.0, "back": -10.0}
    for bearing, points in ((None, True), (0.0, True), (-9.0, True), (60.0, True),
                            (-12.0, False), (-45.0, False)):  # fmt: skip
        view = None if bearing is None else FakeAttention().be(phase="track", bearing_deg=bearing)
        renderer = BookRenderer(BOOK, resolver=Resolver(attention=view))
        performance = renderer.render("point", EVERYTHING, 1, cue("point", "back"))
        parts = {c.part for c in performance.clips}
        assert ("arm" in parts) == points and "head" in parts, bearing


# -- the brain's marks -----------------------------------------------------------------------------


def test_the_new_marks_are_in_the_brain_s_vocabulary_and_the_salute_waits() -> None:
    """point takes up, far, down, back and right beside you, me and a name; the stop's spread
    jaw is the brain's; the salute and the crazy twist are the engine's own
    (callable false) until Artem has seen the pose clear the head on the robot, their lines
    ready (and no alias may name them yet: the grammar reads an alias as a callable mark);
    the other spellings reach their arguments."""
    lines = mark_lines(BOOK)
    assert "point:you|me|up|far|down|back|right|NAME — point at a person or direction" in lines
    assert "stop — wait" in lines
    assert not any(line.startswith(("salute", "crazy")) for line in lines)
    for name in ("salute", "crazy"):
        composition = BOOK.compositions[name]
        assert not composition.callable and composition.long
        assert BOOK.poses["salute"]["shoulder_pan"] == -84.0  # 3 deg inside the pan's limit
    words = vocabulary(BOOK, frozenset())
    for spelling, canonical in (("point:myself", "me"), ("point:sky", "up"),
                                ("point:upward", "far"), ("point:floor", "down"),
                                ("point:behind", "back"), ("point:side", "right")):  # fmt: skip
        assert read(spelling, words) == ("gesture", "point", canonical), spelling
    me = next(v for v in BOOK.compositions["point"].variants if v.needs == {"arg": "me"})
    assert me.steps[0].choices["pose"] == ("@pose:heart",)
    stop = BOOK.compositions["stop"]
    assert [v.steps[1].choices.get("pose") for v in stop.variants if not v.floor] == [
        ("@pose:up_open",),
        ("@pose:stop",),
    ]
