"""The breath layer under the arm's clips (pepin.expression.breath, config/animation.json's
``breath`` block): the oscillations and the accents' swell, the blend from a frozen offset, the
arm part carrying it over a held pose and never from rest, the clean held pose, the gentle way
down, the book's checks, and the style that picks a bare mark's argument."""

import math
from pathlib import Path
from typing import Any

import pytest

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import Breath, Clip, Composition, Oscillation, Track, Variant
from pepin.expression.animator import styled_arg
from pepin.expression.book import book_from, load_book, with_rest
from pepin.expression.breath import ArmBreath, smoothstep
from pepin.expression.fakes import FakeArmLink
from pepin.expression.parts import ArmPart
from pepin.expression.trajectory import JOINTS, ArmStatus

CONFIG = Path(__file__).resolve().parents[2] / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
BOOK = with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg)
READY = BOOK.poses["ready"]
LAYER = Breath(
    {"shoulder_lift": Oscillation(2.0, 4.0, 0.0), "wrist_roll": Oscillation(4.0, 2.0, 0.25)},
    blend_s=0.5,
    accent_gain={"pulse_weak": 0.5, "pulse_peak": 1.5},
    accent_decay_s=1.0,
    clip_s=3.0,
)


def arm_with(breath: Breath | None, now: list[float]) -> ArmPart:
    link = FakeArmLink(state=ArmStatus(0.0, MOTION.rest(), False, "off"))
    layer = None if breath is None else ArmBreath(breath, JOINTS, lambda: now[0])
    return ArmPart(link, MOTION, latency_s=0.0, clock=lambda: now[0], breath=layer)


def flat(seconds: float = 2.0) -> Clip:
    """The listeners' breath carrier: a clip that moves nothing by itself, then held."""
    return Clip("arm", 0.0, 0.0, seconds, (), then="hold", hold_s=1.0)


def test_each_joint_breathes_at_its_own_amplitude_period_and_phase() -> None:
    layer = ArmBreath(LAYER, JOINTS, lambda: 0.0)
    assert layer.offset(0.0) == pytest.approx({"shoulder_lift": 0.0, "wrist_roll": 4.0})
    assert layer.offset(1.0)["shoulder_lift"] == pytest.approx(2.0)  # a quarter of 4 s: up
    assert layer.offset(3.0)["shoulder_lift"] == pytest.approx(-2.0)
    assert layer.offset(1.0)["wrist_roll"] == pytest.approx(-4.0)  # a quarter turn ahead
    assert set(layer.offset(0.3)) == {"shoulder_lift", "wrist_roll"}  # the others none
    assert smoothstep(0.0) == 0.0 and smoothstep(1.0) == 1.0 and smoothstep(0.5) == 0.5
    assert smoothstep(0.25) < 0.25 and smoothstep(2.0) == 1.0


def test_an_accent_swells_the_breath_by_its_gain_and_the_swell_decays() -> None:
    layer = ArmBreath(LAYER, JOINTS, lambda: 10.0)
    assert layer.envelope(10.0) == 1.0
    layer.excite(LAYER.accent_gain["pulse_peak"], at=10.0)
    assert layer.envelope(9.9) == 1.0  # nothing before it
    assert layer.envelope(10.0) == pytest.approx(2.5)
    assert layer.envelope(11.0) == pytest.approx(1.0 + 1.5 * math.exp(-1.0))
    layer.excite(0.0)  # no gain: nothing
    layer.excite(0.5, at=10.5)
    assert layer.envelope(10.5) == pytest.approx(1.0 + 1.5 * math.exp(-0.5) + 0.5)
    assert layer.offset(10.0)["wrist_roll"] == pytest.approx(2.5 * 4.0 * math.sin(math.pi / 2))
    layer.excite(1.0, at=100.0)  # the old swells forgotten after six decays
    assert layer.envelope(100.0) == pytest.approx(2.0)


def test_the_blend_starts_from_the_frozen_offset_and_a_rest_clip_fades_it_out() -> None:
    layer = ArmBreath(LAYER, JOINTS, lambda: 0.0)
    knots = [(0.1 * k, [0.0] * len(JOINTS)) for k in range(21)]  # 2 s of a still pose
    lift, roll = JOINTS.index("shoulder_lift"), JOINTS.index("wrist_roll")
    frozen = [0.0] * len(JOINTS)
    frozen[lift], frozen[roll] = 1.0, -3.0
    live = layer.breathe(knots, 0.0, frozen)
    assert live[0][1][lift] == 1.0 and live[0][1][roll] == -3.0  # where the hold left it
    at_blend = next(deg for t, deg in live if t >= LAYER.blend_s)
    assert at_blend[lift] == pytest.approx(layer.offset(LAYER.blend_s)["shoulder_lift"])
    assert at_blend[roll] == pytest.approx(layer.offset(LAYER.blend_s)["wrist_roll"])
    assert all(deg[JOINTS.index("elbow_flex")] == 0.0 for _, deg in live)  # not in the layer
    out = layer.breathe(knots, 0.0, frozen, live=False)
    assert out[0][1][lift] == 1.0 and out[-1][1][lift] == 0.0 and out[-1][1][roll] == 0.0
    assert layer.final_offset(knots, live) == pytest.approx(
        tuple(live[-1][1][j] for j in range(len(JOINTS)))
    )


def test_the_arm_part_breathes_over_a_held_pose_never_from_rest_and_holds_the_clean_pose() -> None:
    now = [100.0]
    arm = arm_with(LAYER, now)
    raise_ = Track("shoulder_lift", "absolute", (0.0, 1.0), (MOTION.rest_deg["shoulder_lift"],
                                                             READY["shoulder_lift"]))  # fmt: skip
    up = arm.play(Clip("arm", 0.0, 1.0, 1.0, (raise_,), then="hold", hold_s=5.0))
    assert up.ok and arm.held_offset is None  # from rest: no breath on the raise
    sent = arm.link.calls[-1][1][0]  # type: ignore[attr-defined]
    roll = JOINTS.index("wrist_roll")
    assert all(k.deg[roll] == pytest.approx(MOTION.rest_deg["wrist_roll"]) for k in sent.knots)
    held = arm.holding()
    assert held is not None and held["shoulder_lift"] == pytest.approx(READY["shoulder_lift"])
    now[0] = 101.5
    breathed = arm.play(flat())
    assert breathed.ok
    sent = arm.link.calls[-1][1][0]  # type: ignore[attr-defined]
    rolls = [k.deg[roll] - held["wrist_roll"] for k in sent.knots]
    assert sent.knots[0].deg == pytest.approx(tuple(held[j] for j in JOINTS), abs=0.01)
    assert max(rolls) > 3.0 and min(rolls) < -3.0  # 4 deg of roll over the held pose
    lift = JOINTS.index("shoulder_lift")
    assert max(abs(k.deg[lift] - held["shoulder_lift"]) for k in sent.knots) > 1.0
    assert arm.holding() == pytest.approx(held)  # the clean pose stays the unit's
    offset = arm.held_offset
    assert offset is not None and offset[roll] == pytest.approx(rolls[-1], abs=0.01)
    knot = arm.held_knot()
    assert knot is not None and knot[roll] == pytest.approx(sent.knots[-1].deg[roll], abs=0.01)
    now[0] = 103.6  # the next clip starts where the last knot froze the breath, then blends
    arm.play(flat())
    again = arm.link.calls[-1][1][0]  # type: ignore[attr-defined]
    assert again.knots[0].deg == pytest.approx(sent.knots[-1].deg, abs=0.02)
    assert again.knots[0].t_s == 0.0 and again.knots[-1].t_s == pytest.approx(2.0)
    plain = arm_with(None, now)
    plain.play(Clip("arm", 0.0, 1.0, 1.0, (raise_,), then="hold", hold_s=5.0))
    plain.play(flat())
    still = plain.link.calls[-1][1][0]  # type: ignore[attr-defined]
    assert all(k.deg == pytest.approx(still.knots[0].deg) for k in still.knots)  # no layer


def test_a_knot_the_breath_would_push_past_a_limit_plays_without_it() -> None:
    now = [50.0]
    arm = arm_with(LAYER, now)
    low = MOTION.limits()[JOINTS.index("shoulder_lift")][0]
    edge = Track("shoulder_lift", "absolute", (0.0, 1.0), (MOTION.rest_deg["shoulder_lift"], low))
    assert arm.play(Clip("arm", 0.0, 1.0, 1.0, (edge,), then="hold", hold_s=5.0)).ok
    now[0] = 51.5
    outcome = arm.play(flat())
    assert outcome.ok, outcome.why  # the clip plays, its breath left out
    sent = arm.link.calls[-1][1][0]  # type: ignore[attr-defined]
    lift = JOINTS.index("shoulder_lift")
    assert all(k.deg[lift] == pytest.approx(low) for k in sent.knots)
    assert arm.held_offset is None


def test_the_gentle_way_down_runs_at_the_share_blends_the_breath_out_and_ends_on_rest() -> None:
    now = [200.0]
    arm = arm_with(LAYER, now)
    pose = Track("shoulder_lift", "absolute", (0.0, 1.0), (MOTION.rest_deg["shoulder_lift"],
                                                           READY["shoulder_lift"]))  # fmt: skip
    assert arm.lower() is None  # nothing held: nothing to lower
    arm.play(Clip("arm", 0.0, 1.0, 1.0, (pose,), then="hold", hold_s=5.0))
    now[0] = 201.2
    arm.play(flat())
    frozen = arm.held_knot()
    assert frozen is not None
    now[0] = 203.3
    arm.gentle_share = 0.3
    seconds = arm.lower()
    assert seconds is not None and seconds > 0.0
    down = arm.link.calls[-1][1][0]  # type: ignore[attr-defined]
    assert down.name == "down" and down.then == "rest"
    assert down.knots[0].deg == pytest.approx(frozen, abs=0.01)  # from where it really stands
    assert down.knots[-1].deg == pytest.approx(MOTION.rest(), abs=0.01)
    top = max(abs(b - a) / (k1.t_s - k0.t_s) for k0, k1 in zip(down.knots, down.knots[1:],
                                                                strict=False)
              for a, b in zip(k0.deg, k1.deg, strict=True))  # fmt: skip
    assert top <= 0.3 * MOTION.max_speed_deg_s + 1e-6
    assert arm.holding() is None and arm.held_offset is None
    assert seconds == pytest.approx(max(1.5 * 18.6 / 72.0, LAYER.blend_s), abs=0.02)
    arm.gentle_share = 1.0
    arm.play(Clip("arm", 0.0, 1.0, 1.0, (pose,), then="hold", hold_s=5.0))
    quick = arm.lower()
    assert quick is not None and quick == pytest.approx(LAYER.blend_s)  # the blend's floor


def test_the_book_s_breath_block_is_read_and_checked() -> None:
    assert set(BOOK.breath) == {"arm"}
    layer = BOOK.breath["arm"]
    assert set(layer.joints) == set(JOINTS) and layer.when == "up"
    assert all(o.amp_deg > 0.0 and o.period_s > 0.0 for o in layer.joints.values())
    assert set(layer.accent_gain) <= set(BOOK.compositions)
    assert layer.accent_gain["pulse_weak"] < layer.accent_gain["pulse_peak"]
    data: dict[str, Any] = {
        "atoms": {"nod": {"part": "head", "params": {"period_s": 1.0},
                          "keys": [{"at": 0.0, "tilt_deg": 0.0}]}},
        "compositions": {"accent": {"variants": [{"steps": [{"atom": "nod"}]}]}},
    }  # fmt: skip
    good = {"arm": {"joints": {"wrist_roll": {"amp_deg": 2.0, "period_s": 3.0, "phase": 0.5}},
                    "accent_gain": {"accent": 1.0}}}  # fmt: skip
    book = book_from({**data, "breath": good})
    assert book.breath["arm"].joints["wrist_roll"] == Oscillation(2.0, 3.0, 0.5)
    assert book.breath["arm"].blend_s == 0.6 and book.breath["arm"].clip_s == 3.0
    assert not book_from(data).breath  # an older book: no layer
    bad: list[tuple[dict[str, Any], str]] = [
        ({"arm": {"joints": {"knee": {"amp_deg": 1.0, "period_s": 1.0}}}}, "knee"),
        ({"arm": {"joints": {"wrist_roll": {"amp_deg": -1.0, "period_s": 1.0}}}}, "amp_deg"),
        ({"arm": {"joints": {"wrist_roll": {"amp_deg": 1.0, "period_s": 1.0, "phase": 1.0}}}},
         "phase"),
        ({"arm": {"joints": {}}}, "at least one"),
        ({"arm": {"joints": {"wrist_roll": {"amp_deg": 1.0, "period_s": 1.0}},
                  "accent_gain": {"nope": 1.0}}}, "accent_gain"),
        ({"arm": {"joints": {"wrist_roll": {"amp_deg": 1.0, "period_s": 1.0}}, "when": "rest"}},
         "when"),
        ({"tail": {"joints": {}}}, "one of"),
    ]  # fmt: skip
    for block, error in bad:
        with pytest.raises(ValueError, match=error):
            book_from({**data, "breath": block})


def test_a_part_s_style_picks_a_bare_mark_s_argument_never_a_given_one() -> None:
    laugh = BOOK.compositions["laugh"]
    assert laugh.style_args == {"evil": laugh.style_args["evil"]} and not laugh.args
    assert styled_arg(laugh, None, "sinister, evil laugh") == "evil"
    assert styled_arg(laugh, None, "Зловещий шёпот") == "evil"
    assert styled_arg(laugh, None, "happy") is None and styled_arg(laugh, None, None) is None
    assert styled_arg(laugh, "evil", "happy") == "evil"  # a marker's own argument stands
    plain = Composition("nod", (Variant((), 1.0),))
    assert styled_arg(plain, None, "sinister") is None
    with pytest.raises(ValueError, match="style_args"):
        book_from({
            "atoms": {"nod": {"part": "head", "params": {"period_s": 1.0},
                              "keys": [{"at": 0.0, "tilt_deg": 0.0}]}},
            "compositions": {"laugh": {"variants": [{"steps": [{"atom": "nod"}]}],
                                       "style_args": {"evil": ["sinister"]}}},
        })  # fmt: skip
