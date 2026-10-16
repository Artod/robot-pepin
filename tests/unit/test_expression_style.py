"""The mouth's drawing style as the brain chooses it (2026-10-15): the book's style atom and
mark rendered, the face part's calls to the head server, the mark's grammar beside the voice's
manner, its lines in the brain's vocabulary, the engine never capping it, and the idle layer
never switching a style."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from test_expression_engine import Rig

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import Clip, Event, Scene
from pepin.expression.book import (
    book_from,
    load_book,
    mark_lines,
    style_names,
    vocabulary,
    with_rest,
    with_styles,
)
from pepin.expression.fakes import FakeExecutor, FakeFace
from pepin.expression.markers import Mark, Marker, Part, read_mark
from pepin.expression.parts import FacePart
from pepin.expression.render import BookRenderer
from pepin.face import load_face_table, style_lines

CONFIG = Path(__file__).resolve().parents[2] / "config"
MOTION = load_arm_motion(CONFIG / "arm.json")
STYLES = style_names(CONFIG / "face.json")
BOOK = with_styles(with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg), STYLES)
FACE = Scene(frozenset({"face", "eyes", "head"}))
WORDS = vocabulary(BOOK, frozenset({"grin"}))
DATA: dict[str, Any] = json.loads((CONFIG / "animation.json").read_text())


def _face(clips: tuple[Clip, ...]) -> Clip:
    (clip,) = [c for c in clips if c.part == "face"]
    return clip


# -- the mark rendered -----------------------------------------------------------------------------


def test_the_mark_is_a_gag_of_a_few_seconds_or_a_standing_choice_and_bare_lets_go() -> None:
    """``{style:rose}``: the style for the book's 4 s, held by the head server, so the clip
    lasts its period alone (the face stays free for the expressions), a tenth of a second
    before the word; ``s=N`` its seconds; ``s=0`` or ``hold`` stands; bare: ``default``."""
    renderer = BookRenderer(BOOK)
    gag = _face(renderer.render("style", FACE, 1, arg="rose").clips)
    assert gag.events == (Event(0.0, "rose", 4.0, "style"),)
    assert gag.offset_s == pytest.approx(-0.1) and gag.duration_s == pytest.approx(0.05)
    timed = _face(renderer.render("style", FACE, 1, arg="manga", args=("hold_s=2",)).clips)
    assert timed.events == (Event(0.0, "manga", 2.0, "style"),)
    for args in (("hold_s=0",), ("hold",), ("0",)):
        standing = _face(renderer.render("style", FACE, 1, arg="anime", args=args).clips)
        assert standing.events == (Event(0.0, "anime", None, "style"),), args
    for arg in (None, "default"):
        back = _face(renderer.render("style", FACE, 1, arg=arg).clips)
        assert back.events == (Event(0.0, "default", None, "style"),), arg


def test_the_face_part_draws_the_style_for_its_ms_and_a_fade_leaves_it() -> None:
    face, later = FakeFace(), []
    part = FacePart(face, after=lambda s, fn: (later.append((s, fn)), lambda: later.clear())[1])
    clip = Clip("face", 0.0, 0.0, 2.0, events=(
        Event(0.0, "rose", 4.0, "style"),
        Event(0.0, "grin", None),
        Event(1.0, "manga", None, "style"),
    ))  # fmt: skip
    assert part.play(clip).ok
    assert face.calls == [("style", ("rose", 4000)), ("express", ("grin", 1.0, 2.0, None))]
    ((seconds, standing),) = later
    standing()
    assert seconds == 1.0 and face.calls[-1] == ("style", ("manga", None))
    part.play(Clip("face", 0.0, 0.0, 0.05, events=(Event(0.0, "default", None, "style"),)))
    assert face.calls[-1] == ("style", (None, None))  # this source's styles let go
    part.fade("gesture", 0.1)
    part.rest()
    assert face.calls[-1] == ("style", (None, None))  # the fades leave the style alone


# -- the grammar -----------------------------------------------------------------------------------


def test_the_style_mark_takes_the_face_s_styles_and_leaves_the_voice_its_manner() -> None:
    """The styles are config/face.json's (none written in the book); ``{style:X}`` with any
    other X is the voice's manner as before; ``{style}`` alone lets the style go."""
    assert BOOK.compositions["style"].args == ("default", *STYLES)
    first, other = STYLES[0], STYLES[-1]
    cases = {
        "style": Mark("gesture", "style"),
        f"style:{other}": Mark("gesture", "style", other),
        f"style:{other.upper()}, 3": Mark("gesture", "style", other, ("3",)),
        f"style:{other},s=0": Mark("gesture", "style", other, ("hold_s=0",)),
        f"style:{first},hold": Mark("gesture", "style", first, ("hold",)),
        "style:default": Mark("gesture", "style", "default"),
        "style:whispering": Mark("style", "whispering"),
        "style:happy, excited": Mark("style", "happy, excited"),
        "": Mark("style", ""),
    }
    for written, expected in cases.items():
        assert read_mark(written, WORDS) == expected, written
    plain = vocabulary(with_rest(load_book(CONFIG / "animation.json"), MOTION.rest_deg),
                       frozenset())  # fmt: skip
    assert read_mark(f"style:{other}", plain) == Mark("style", other)  # no styles at load: a voice
    assert read_mark("glow", WORDS) is None  # bare stays the style mark's alone


def test_the_brain_reads_one_line_a_style_from_the_face_table() -> None:
    lines = mark_lines(BOOK, style_lines(load_face_table(CONFIG / "face.json")))
    styled = [line for line in lines if line.startswith("style")]
    assert [line.split(" — ")[0] for line in styled] == [f"style:{s}" for s in STYLES]
    eyes = lines.index(next(line for line in lines if line.startswith("eyes:")))
    assert lines[eyes + 1] == styled[0]  # in the book's order
    (own,) = [line for line in mark_lines(BOOK) if line.startswith("style")]
    assert own.startswith(f"style[:default|{'|'.join(STYLES)}[,s=N,hold]] — ")


def test_the_book_refuses_a_bare_mark_or_an_arg_name_with_no_argument_and_a_styled_idle() -> None:
    compositions = DATA["compositions"]
    bare = {"bare": True, "variants": [{"steps": [{"atom": "face", "params": {"name": "grin"}}]}]}
    with pytest.raises(ValueError, match=r"compositions\.grinning\.bare: only beside arguments"):
        book_from({**DATA, "compositions": {**compositions, "grinning": bare}})
    arg = {"variants": [{"steps": [{"atom": "style", "params": {"name": "$arg"}}]}]}
    with pytest.raises(ValueError, match=r"name \$arg needs arguments that choose"):
        book_from({**DATA, "compositions": {**compositions, "restyle": arg}})
    idle = [*DATA["idle"], {"part": "face", "compositions": ["style"], "every_s": [5.0, 9.0]}]
    with pytest.raises(ValueError, match=r"idle\[2\]\.compositions: style switches the face's"):
        book_from({**DATA, "idle": idle})


# -- the engine ------------------------------------------------------------------------------------


def test_a_style_mark_is_never_capped_and_never_waits_for_the_face() -> None:
    """One mark a part (caps.per_part 1): a gesture that holds the face and the style mark both
    play; the style neither books nor holds the face, so the face's own mark on its word plays
    too (the head server arbitrates the style beside the expression)."""
    for marks, faces in (
        ((Marker("gesture", "agreement", 0), Marker("gesture", "style", 3, arg=STYLES[-1])),
         ["agreement", "style"]),
        ((Marker("gesture", "style", 3, arg=STYLES[1]), Marker("face", "grin", 3)),
         ["style", "face:grin"]),
    ):  # fmt: skip
        rig = Rig(seed=5, book=BOOK)
        rig.parts["eyes"] = rig.engine.parts["eyes"] = FakeExecutor("eyes")  # type: ignore[index]
        rig.engine.part_queued(1, 0, Part("Да, конечно.", None, marks))
        rig.engine.step()
        assert not rig.kinds("capped") and not rig.kinds("refused")
        assert [p["intent"] for p in rig.kinds("ladder")] == faces
        rig.playout.marks[(1, 0)] = (0.0, 1.0)
        rig.engine.part_spoken(1, 0)
        rig.tick(2.0)
        played = [e for clip in rig.played("face") for e in clip.events]
        style = next(m for m in marks if m.name == "style")
        assert Event(0.0, str(style.arg), 4.0, "style") in played
        assert all(c["ok"] for c in rig.kinds("cue") if c["part"] == "face")
        if "face:grin" in faces:
            assert Event(0.0, "grin", 2.0) in played
