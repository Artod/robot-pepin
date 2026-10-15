"""The comics set against its style guide (firmware/head_esp32/COMICS_STYLE.md, measured by
pepin.face_style): one 5 px ink line and 3 px details, the ink palette, black mouths, white
teeth, one red, corners no tighter than 3 px but on the parts meant sharp, one band, the
big widths on one centre, no highlights anywhere, no inner line past the outline."""

from __future__ import annotations

from pathlib import Path

import pytest

from pepin.face import load_face_table
from pepin.face_shapes import KEYS
from pepin.face_style import ANIME, COMICS, faults, measure, measure_set

REPO = Path(__file__).resolve().parents[2]
FACE = REPO / "config" / "face.json"
SVG_HEAD = '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="170" viewBox="0 0 320 170">'
INK = 'fill="none" stroke="#1D1D1B" stroke-width="5"'
DETAIL = 'fill="none" stroke="#1D1D1B" stroke-width="3"'


def svg(body: str) -> str:
    return f"{SVG_HEAD}<title>a test</title>{body}</svg>"


@pytest.mark.slow  # 1.4 s: a hundred drawings flattened finely
def test_the_whole_set_obeys_the_guide() -> None:
    table = load_face_table(FACE)
    measures = measure_set(table, COMICS)
    assert len(measures) == 3 * len(table.expressions) + 1  # the eps drawing too
    assert {m.expression for m in measures} == set(COMICS.class_of)
    broken = [f"{m.expression} {m.key}: {f}" for m in measures for f in m.faults]
    assert broken == [], "\n".join(broken)


def test_the_two_guides_share_the_ids_and_the_centre() -> None:
    assert set(COMICS.class_of) == set(ANIME.class_of)
    assert COMICS.centre == ANIME.centre and COMICS.aside["thinking"] == ANIME.aside["thinking"]
    assert COMICS.line_px > ANIME.line_px  # the ink line is the bolder one


def test_an_outlier_is_named(tmp_path: Path) -> None:
    table = load_face_table(FACE)
    # the D laugh drawn with the anime pen: a 3 px rose line, a pink fill, a highlight, a
    # band too shallow for the comics, and too narrow for the big class
    body = (
        '<path class="fill" d="M 100 60 Q 160 70 220 60 Q 220 120 160 120 Q 100 120 100 60 Z"'
        ' fill="#EC7488"/>'
        '<path class="teeth" d="M 100 60 Q 160 70 220 60 L 220 70 Q 160 78 100 70 Z"'
        ' fill="#FFFFFF"/>'
        '<path class="shine" d="M 150 100 Q 160 95 170 100 Q 160 108 150 100 Z" fill="#C1D5FA"/>'
        '<path class="line" d="M 100 60 Q 160 70 220 60 Q 220 120 160 120 Q 100 120 100 60 Z"'
        ' fill="none" stroke="#C04860" stroke-width="2.5"/>'
    )
    path = tmp_path / "happy_rest.svg"
    path.write_text(svg(body))
    m = measure(path, "happy", KEYS[0], table, COMICS)
    found = faults(m, COMICS)
    for want in (
        "stroke widths [2.5]",
        "off the palette: fill #EC7488, line #C04860",
        "a highlight (the guide has none)",
        "band",
        "width 120 px, the big class's 146..196",
    ):
        assert any(want in f for f in found), (want, found)
    # the zigzag: its corners are meant sharp, so neither a corner nor a turn is a fault;
    # the same line untagged is
    zig = "M 108 85 L 121 78 L 134 92 L 147 78 L 160 92 L 173 78 L 186 92 L 199 78 L 212 85"
    path = tmp_path / "struggling_rest.svg"
    path.write_text(svg(f'<path class="line" d="{zig}" {INK} data-part="zigzag"/>'))
    m = measure(path, "struggling", KEYS[0], table, COMICS)
    assert faults(m, COMICS) == []
    path.write_text(svg(f'<path class="line" d="{zig}" {INK}/>'))
    m = measure(path, "struggling", KEYS[0], table, COMICS)
    assert any("line: a corner of radius" in f for f in faults(m, COMICS))
    # the gritted grid: a white pill of teeth, the bite line and the gaps as 3 px ink over
    # it, no fill; the box is the teeth, the gaps stay inside, there is no band to measure
    pill = "M 92 65 L 228 65 A 20 20 0 0 1 228 105 L 92 105 A 20 20 0 0 1 92 65 Z"
    body = (
        f'<path class="teeth" d="{pill}" fill="#FFFFFF"/>'
        f'<path class="line" d="{pill}" {INK}/>'
        f'<path class="line" d="M 72 85 Q 160 88 248 85" {DETAIL} data-part="bite"/>'
        f'<path class="line" d="M 140 66 L 140 104" {DETAIL} data-part="gap"/>'
        f'<path class="line" d="M 180 66 L 180 104" {DETAIL} data-part="gap"/>'
    )
    path = tmp_path / "clenched_rest.svg"
    path.write_text(svg(body))
    m = measure(path, "clenched", KEYS[0], table, COMICS)
    assert m.width == pytest.approx(176.0) and m.band is None and m.overrun <= 0.5
    assert faults(m, COMICS) == []
    # the kiss: red lips with no fill and no mouth line are the mouth
    lips = "M 130 85 Q 145 65 160 80 Q 175 65 190 85 Q 160 110 130 85 Z"
    path = tmp_path / "kiss_rest.svg"
    path.write_text(
        svg(
            f'<path class="tongue" d="{lips}" fill="#DE5B55"/>'
            f'<path class="line" d="M 140 86 Q 160 90 180 86" {DETAIL} data-part="seam"/>'
        )
    )
    m = measure(path, "kiss", KEYS[0], table, COMICS)
    assert m.width == pytest.approx(60.0) and not m.shine
    assert not any("no mouth" in f or "highlight" in f for f in faults(m, COMICS))
