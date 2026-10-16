"""The eyes as a part of the gesture engine (2026-10-15): the book's eye atoms rendered, the eyes
part's calls to the head server, the eyes' marks and their lines in the brain's vocabulary, and
the engine never capping a mark of the eyes alone."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_expression_engine import Rig

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import Clip, Event, Scene, Track
from pepin.expression.book import book_from, load_book, mark_lines, vocabulary, with_rest
from pepin.expression.fakes import FakeExecutor, FakeEyesChannel
from pepin.expression.markers import Mark, Marker, Part, read_mark
from pepin.expression.parts import EYE_OVERLAP_S, EyesPart, eye_calls
from pepin.expression.render import BookRenderer

CONFIG = Path(__file__).resolve().parents[2] / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
BOOK = with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg)
EYES = Scene(frozenset({"eyes", "face", "head"}))
WORDS = vocabulary(BOOK, frozenset({"smile"}))


def _eyes(performance_clips: tuple[Clip, ...]) -> Clip:
    (clip,) = [c for c in performance_clips if c.part == "eyes"]
    return clip


def _played(clip: Clip) -> list[tuple[float, str, tuple[Any, ...]]]:
    """The clip's calls with their times, as a fake channel heard them."""
    heard = FakeEyesChannel()
    calls = eye_calls(clip, heard)
    for _, call in calls:
        call()
    return [(round(t, 3), what, said) for (t, _), (what, said) in
            zip(calls, heard.calls, strict=True)]  # fmt: skip


# -- the atoms rendered ----------------------------------------------------------------------------


def test_a_frown_is_a_look_held_for_its_seconds_and_lapses_with_the_clip() -> None:
    """The mark's number is its seconds (``{frown:2}``, here the eyes mark's): the brows to the
    nose and down for 2 s, one timed override sent a tenth of a second before the word."""
    performance = BookRenderer(BOOK).render("eyes", EYES, 3, arg="frown", args=("2",))
    clip = _eyes(performance.clips)
    assert clip.offset_s == pytest.approx(-0.1) and clip.duration_s == pytest.approx(2.1)
    ((t, what, (ms, side, fields)),) = _played(clip)
    assert (t, what, side, ms) == (0.0, "eyes", "both", 2100)
    assert -0.9 <= fields["brow_pos"] <= -0.7 and -0.5 <= fields["brow_raise"] <= -0.3
    assert set(fields) == {"brow_pos", "brow_raise", "lid_top"}


def test_a_wink_is_a_moment_of_the_robot_s_own_eye_never_held() -> None:
    renderer = BookRenderer(BOOK)
    left = _eyes(renderer.render("wink", EYES, 1, args=("left",)).clips)
    assert _played(left) == [(0.0, "blink", ("left",))]
    right = _eyes(renderer.render("wink", EYES, 1, args=("right", "hold_s=2")).clips)
    assert _played(right) == [(0.0, "blink", ("right",))]  # a hold adds no second wink
    sides = {_played(_eyes(renderer.render("wink", EYES, s).clips))[0][2] for s in range(12)}
    assert sides == {("left",), ("right",)}  # bare: either eye


def test_a_glow_is_the_named_colour_and_brighter_and_an_effect_runs_for_its_seconds() -> None:
    renderer = BookRenderer(BOOK)
    glow = _eyes(renderer.render("glow", EYES, 2, arg="red", args=("hold_s=3",)).clips)
    (iris,) = glow.events
    assert (iris.dof, iris.name, iris.hold_s) == ("iris", "#FF1E1E", pytest.approx(3.1))
    ((_, what, (ms, _, fields)),) = _played(glow)
    assert what == "eyes" and ms == 3100 and fields["iris"] == "#FF1E1E"
    assert 0.8 <= fields["glow"] <= 0.9
    spin = _eyes(renderer.render("eyes", EYES, 2, arg="spin", args=("2",)).clips)
    assert _played(spin) == [(0.0, "eyes_effect", ("spin", 2100))]  # its cycle and 2 s
    sparkle = _eyes(renderer.render("eyes", EYES, 2, arg="sparkle").clips)
    assert _played(sparkle) == [(0.0, "eyes_effect", ("sparkle", 1600))]  # about its own length


def test_a_merged_clip_changes_the_look_at_each_step_and_lets_go_between() -> None:
    """Two looks of one clip: the first timed to the second's start and a little more (so the
    second replaces it first), the second to the clip's end; a gap between two looks lets the
    first lapse exactly (nothing to replace it), and the moments go at their times."""
    clip = Clip(
        "eyes", 0.0, 0.0, 4.0,
        tracks=(
            Track("brow_raise", "absolute", (0.0, 1.0), (0.8, 0.8)),
            Track("lid_top", "absolute", (1.0, 2.0), (0.5, 0.5)),
            Track("glow", "absolute", (3.0, 4.0), (0.9, 0.9)),
            Track("wink", "absolute", (2.5,), (-1.0,)),
        ),
        events=(Event(0.5, "flash", 0.3, "effect"),),
    )  # fmt: skip
    overlap = round(EYE_OVERLAP_S * 1000)
    assert _played(clip) == [
        (0.0, "eyes", (1000 + overlap, "both", {"brow_raise": 0.8})),
        (0.5, "eyes_effect", ("flash", 300)),
        (1.0, "eyes", (1000, "both", {"lid_top": 0.5})),  # nothing after it until 3.0
        (2.5, "blink", ("right",)),
        (3.0, "eyes", (1000, "both", {"glow": 0.9})),
    ]


def test_the_eyes_part_plays_each_call_at_its_time_and_a_fade_lets_the_eyes_go() -> None:
    channel, later = FakeEyesChannel(), []
    part = EyesPart(channel, after=lambda s, fn: (later.append((s, fn)), lambda: later.clear())[1])
    clip = Clip("eyes", 0.0, 0.0, 2.0, tracks=(Track("wink", "absolute", (0.0, 1.0), (1.0, 1.0)),))
    assert part.play(clip).ok and part.ready() is None
    assert channel.calls == [("blink", ("left",))]
    ((seconds, wink),) = later
    wink()
    assert seconds == 1.0 and channel.calls[-1] == ("blink", ("left",))
    part.fade("gesture", 0.1)
    assert later == [] and channel.calls[-1] == ("eyes_release", ())


def test_the_book_refuses_an_eye_colour_or_effect_it_cannot_show() -> None:
    data = json.loads((CONFIG / "animation.json").read_text())
    bad = {"variants": [{"steps": [{"atom": "glow", "params": {"name": "orange"}}]}]}
    with pytest.raises(ValueError, match="#RRGGBB, not 'orange'"):
        book_from({**data, "compositions": {**data["compositions"], "tint": bad}})
    bad = {"variants": [{"steps": [{"atom": "eye_effect", "params": {"name": "twirl"}}]}]}
    with pytest.raises(ValueError, match="pulse, flicker, doze, tear, not 'twirl'"):
        book_from({**data, "compositions": {**data["compositions"], "twirl": bad}})


def test_the_mischief_winks() -> None:
    """The eye beat of the book's feelings (2026-10-15): the wink as the evil click's hand
    arrives; every variant, the floor too."""
    for variant in BOOK.compositions["evil_click"].variants:
        assert any(s.atom == "wink" for s in variant.steps)


# -- the marks -----------------------------------------------------------------------------------


def test_the_eye_marks_read_one_argument_that_chooses_and_the_rest_by_name() -> None:
    """One positional argument at most, the rest key=value (``s`` is the hold's seconds); the
    effects' and the brows' names are marks of their own through the aliases."""
    cases = {
        "wink": Mark("gesture", "wink"),
        "wink:left": Mark("gesture", "wink", None, ("left",)),
        "frown:1.5": Mark("gesture", "eyes", "frown", ("1.5",)),
        "frown:s=2": Mark("gesture", "eyes", "frown", ("hold_s=2",)),
        "glow:red": Mark("gesture", "glow", "red"),
        "glow:red,s=2": Mark("gesture", "glow", "red", ("hold_s=2",)),
        "glow:amber": Mark("gesture", "glow", "orange"),
        "spin": Mark("gesture", "eyes", "spin"),
        "tear": Mark("gesture", "eyes", "tear"),  # the kawaii sad's drop
        "spin:2": Mark("gesture", "eyes", "spin", ("2",)),
        "spin:s=2": Mark("gesture", "eyes", "spin", ("hold_s=2",)),
        "eyes:brows_up,s=1": Mark("gesture", "eyes", "brows_up", ("hold_s=1",)),
    }
    for written, expected in cases.items():
        assert read_mark(written, WORDS) == expected, written
    for nothing in ("glow", "blink", "eyes_release", "eyes"):  # a colour needed; the engine's
        assert read_mark(nothing, WORDS) is None, nothing


def test_the_brain_reads_three_lines_for_the_eyes() -> None:
    lines = mark_lines(BOOK)
    eyes = [line for line in lines if line.startswith(("wink", "glow", "eyes"))]
    assert eyes == [
        "wink[:left|right] — a wink of one eye",
        "glow:red|orange|green|blue|pink|white[,s=N] — the eyes' colour for N seconds",
        "eyes:frown|brows_up|squint|sparkle|flash|spin|pulse|flicker|doze|tear[,s=N] — the eyes"
        " for N seconds",
    ]
    assert not any(line.startswith(("frown", "brows_up", "squint", "blink")) for line in lines)


def test_a_mark_of_the_eyes_alone_is_never_capped_as_the_face_s_are_not() -> None:
    """One mark a part (caps.per_part 1): a gesture and the eyes' frown both play."""
    rig = Rig(seed=5)
    rig.parts["eyes"] = rig.engine.parts["eyes"] = FakeExecutor("eyes")  # type: ignore[index]
    marks = (Marker("gesture", "agreement", 0), Marker("gesture", "eyes", 3, arg="frown"))
    rig.engine.part_queued(1, 0, Part("Да, конечно.", None, marks))
    rig.engine.step()
    assert not rig.kinds("capped")
    assert [p["intent"] for p in rig.kinds("performance")] == ["agreement", "eyes"]
    rig.playout.marks[(1, 0)] = (0.0, 1.0)
    rig.engine.part_spoken(1, 0)
    rig.tick(2.0)
    assert rig.played("eyes")
