"""The kawaii set against its style guide (firmware/head_esp32/KAWAII_STYLE.md, measured by
pepin.face_style): one 3.5 px near-black line, the laughs and the quiet
mouths rose inside, a plain rose tongue with no highlight and no crease, the tiny sizes on the
set's own centre, shown at the style's scale."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from pepin.face import load_face_table
from pepin.face_shapes import KEYS, centre_of
from pepin.face_style import ANIME, KAWAII, faults, measure, measure_set

REPO = Path(__file__).resolve().parents[2]
FACE = REPO / "config" / "face.json"
SVG_HEAD = '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="170" viewBox="0 0 320 170">'
LINE = 'fill="none" stroke="#C04860" stroke-width="3"'


def svg(body: str) -> str:
    return f"{SVG_HEAD}<title>a test</title>{body}</svg>"


@pytest.mark.slow  # 1.4 s: a hundred drawings flattened finely
def test_the_whole_set_obeys_the_guide() -> None:
    table = load_face_table(FACE)
    assert "kawaii" in table.shape_styles
    measures = measure_set(table, KAWAII)
    assert len(measures) == 3 * len(table.expressions) + 1  # the eps drawing too
    assert {m.expression for m in measures} == set(KAWAII.class_of)
    broken = [f"{m.expression} {m.key}: {f}" for m in measures for f in m.faults]
    assert broken == [], "\n".join(broken)


def test_the_two_guides_share_the_ids_and_the_kawaii_set_sits_high() -> None:
    """The same ids; the kawaii centre is the set's own (pepin.face_shapes.centre_of: the
    fixed point of its scale, high on the panel), the thinking offset the anime guide's."""
    assert set(KAWAII.class_of) == set(ANIME.class_of)
    assert KAWAII.centre == centre_of("kawaii") == (160.0, 70.0) and ANIME.centre == (160.0, 85.0)
    assert KAWAII.aside["thinking"] == ANIME.aside["thinking"] == 34.0
    assert KAWAII.aside["struggling"] < ANIME.aside["struggling"]  # set slightly to the right


def test_the_set_is_measured_as_the_screen_shows_it() -> None:
    """The measures are the scaled drawing's: the smile's 36 px round u is 90 px wide at the
    2.5 scale, its line 10 px, centred on (160, 70); the guide's lengths scale with it."""
    table = load_face_table(FACE)
    m = measure(REPO / "config" / "kawaii" / "smile_rest.svg", "smile", KEYS[0], table, KAWAII)
    assert m.scale == 2.5 and m.widths == (3.5,) and m.widths_px == (10.0,)
    assert m.width == pytest.approx(90.0, abs=1.0)
    assert m.centre == pytest.approx((160.0, 70.0), abs=4.0)
    assert faults(m, KAWAII) == []
    narrow = replace(m, box=(142.0, 60.0, 178.0, 80.0))  # the drawing's own 36 px, unscaled
    assert faults(narrow, KAWAII) == ["width 36 px, the line class's 85..170"]


def test_an_outlier_is_named(tmp_path: Path) -> None:
    table = load_face_table(FACE)
    # the anime tongue (a rose line, a highlight and a crease): none of that in this set
    body = (
        f'<path class="line" d="M 104 85 Q 160 95 216 85" {LINE}/>'
        '<path class="tongue" d="M 170 84 L 200 84 A 15 15 0 0 1 185 114 A 15 15 0 0 1 170 84 Z"'
        ' fill="#F09AA8"/>'
        '<ellipse class="shine" cx="182" cy="92" rx="6" ry="3" fill="#F8C8D0"/>'
        '<path class="line" d="M 170 84 L 200 84 A 15 15 0 0 1 185 114 A 15 15 0 0 1 170 84 Z"'
        f' {LINE} data-follow="tongue" data-part="tongue"/>'
        '<path class="line" d="M 185 88 L 185 104" fill="none" stroke="#C04860" stroke-width="2"'
        ' data-part="crease"/>'
    )
    path = tmp_path / "tease_rest.svg"
    path.write_text(svg(body))
    found = faults(measure(path, "tease", KEYS[0], table, KAWAII), KAWAII)
    assert any("a highlight (the guide has none)" in f for f in found)
    assert any("a crease (the guide has none)" in f for f in found)
    assert any("off the palette: line #C04860" in f for f in found)  # the anime rose
    # a rose O in the set's line, drawn with the anime line's width (on the set's own centre)
    path = tmp_path / "o_mouth_rest.svg"
    path.write_text(
        svg(
            '<ellipse class="fill" cx="160" cy="70" rx="10" ry="10" fill="#DD7487"/>'
            '<ellipse class="line" cx="160" cy="70" rx="10" ry="10" fill="none" stroke="#504743"'
            ' stroke-width="3"/>'
        )
    )
    m = measure(path, "o_mouth", KEYS[0], table, KAWAII)
    assert faults(m, KAWAII) == ["stroke widths [3.0] (the guide's: 3.5 and 2.5)"]
