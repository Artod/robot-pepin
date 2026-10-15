"""The anime set against its style guide (firmware/head_esp32/ANIME_STYLE.md, measured by
pepin.face_style): one line width, one palette, round corners, one band, three sizes, one
centre, highlights on the tongues out and nowhere else, no inner line past the outline."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from pepin.face import load_face_table
from pepin.face_shapes import KEYS
from pepin.face_style import (
    ANIME,
    faults,
    max_turn,
    measure,
    measure_set,
    min_radius,
    span_at,
)

REPO = Path(__file__).resolve().parents[2]
FACE = REPO / "config" / "face.json"
SVG_HEAD = '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="170" viewBox="0 0 320 170">'
LINE = 'fill="none" stroke="#C04860" stroke-width="3"'


def svg(body: str) -> str:
    return f"{SVG_HEAD}<title>a test</title>{body}</svg>"


@pytest.mark.slow  # 1.4 s: a hundred drawings flattened finely
def test_the_whole_set_obeys_the_guide() -> None:
    table = load_face_table(FACE)
    measures = measure_set(table, ANIME)
    assert len(measures) == 3 * len(table.expressions) + 1  # the eps drawing too
    assert {m.expression for m in measures} == set(ANIME.class_of)
    broken = [f"{m.expression} {m.key}: {f}" for m in measures for f in m.faults]
    assert broken == [], "\n".join(broken)


def test_the_geometry_reads_corners_turns_and_spans() -> None:
    circle = np.array(
        [
            (10 * math.cos(a), 10 * math.sin(a))
            for a in np.linspace(0, 2 * math.pi, 64, endpoint=False)
        ]
    )
    assert min_radius(circle, True, 3) == pytest.approx(10.0, abs=0.1)
    square = np.array([(float(x), float(y)) for x, y in [(0, 0), (10, 0), (10, 10), (0, 10)]])
    pts = np.concatenate(
        [np.linspace(square[i], square[(i + 1) % 4], 10, endpoint=False) for i in range(4)]
    )
    assert min_radius(pts, True, 3) < 2.5  # a right angle
    ring = ((0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0))
    assert max_turn(ring) == pytest.approx(90.0)
    there_and_back = ((0.0, 0.0), (5.0, 1.0), (10.0, 0.0), (5.0, 1.0))  # an open stroke's ring
    assert max_turn(there_and_back) < 30.0  # the turn-backs at its ends do not count
    assert span_at(square, 5.0) == pytest.approx(10.0)


def test_an_outlier_is_named(tmp_path: Path) -> None:
    table = load_face_table(FACE)
    # a square mouth (sharp corners), its band edge past the outline, a highlight with no
    # tongue out, a 2.8 px line, an off-palette fill, off centre and too wide for its class
    body = (
        '<path class="fill" d="M 90 60 L 230 60 L 230 120 L 90 120 Z" fill="#FF0000"/>'
        '<path class="teeth" d="M 92 60 L 228 60 L 228 74 L 92 74 Z" fill="#FFFFFF"/>'
        '<path class="shine" d="M 150 100 L 170 100 L 170 110 L 150 110 Z" fill="#F8C8D0"/>'
        f'<path class="line" d="M 90 60 L 230 60 L 230 120 L 90 120 Z" {LINE}/>'
        '<path class="line" d="M 80 74 L 240 74" fill="none" stroke="#C04860" stroke-width="2.8"'
        ' data-part="band"/>'
    )
    path = tmp_path / "listening_rest.svg"
    path.write_text(svg(body))
    m = measure(path, "listening", KEYS[0], table, ANIME)
    found = faults(m, ANIME)
    for want in (
        "stroke widths [2.8]",
        "off the palette: fill #FF0000",
        "fill: a corner of radius",
        "line: a corner of radius",
        "fill: a resampled turn of 90 deg",
        "past the outline",
        "a highlight without a tongue out",
        "centred at (160, 90)",
        "width 140 px, the small class's 46..66",
    ):
        assert any(want in f for f in found), (want, found)
    assert m.band == pytest.approx(14.0) and m.overrun == pytest.approx(10.0)
    # a tongue hanging out of a line, with its highlight and its crease: no fault of that kind
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
    m = measure(path, "tease", KEYS[0], table, ANIME)
    assert m.tongue_out and m.shine and m.crease and m.widths == (2.0, 3.0)
    assert not any("tongue out" in f or "highlight" in f or "crease" in f for f in faults(m, ANIME))
    # a talk drawing whose upper lip moved
    rest = m
    path = tmp_path / "tease_half.svg"
    path.write_text(svg(f'<path class="line" d="M 104 100 Q 160 110 216 100" {LINE}/>'))
    half = measure(path, "tease", KEYS[1], table, ANIME)
    assert any("the lip moved" in f for f in faults(half, ANIME, rest))
