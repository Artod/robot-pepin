"""The shape-keyframe engine's generator (pepin.face_shapes): the SVG subset it reads, the
resampling the firmware morphs on, the keyframes of config/anime and the tables it writes."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from pepin.face import load_face_table, parse_face_table
from pepin.face_shapes import (
    CENTRE,
    INDEX_HEADER,
    INDEX_SCRIPT,
    KEYS,
    LAYOUT_HEADER,
    SLOT_INDEX,
    SLOTS,
    ShapeStyle,
    centre_of,
    generated_shapes,
    keyframe_of,
    load_shapes,
    path_subpaths,
    polygon_ring,
    read_svg,
    render_style_header,
    render_style_script,
    resample,
    set_pen,
    shape_style,
    shape_styles,
    signed_area,
    stroke_ring,
    stroke_widths,
    style_header,
    style_script,
)

REPO = Path(__file__).resolve().parents[2]
FACE = REPO / "config" / "face.json"
SHAPES = REPO / "config" / "anime"
SVG_HEAD = '<svg xmlns="http://www.w3.org/2000/svg" width="320" height="170" viewBox="0 0 320 170">'
STYLE = ShapeStyle("test", "test", 2.6, 60.0, 3.5)  # a 2.6 px line, no scale


def svg(body: str, title: str = "a test") -> str:
    return f"{SVG_HEAD}<title>{title}</title>{body}</svg>"


def test_the_path_subset_flattens_absolute_and_relative_commands() -> None:
    subs = path_subpaths("M 10 20 L 30 20 l 0 10 H 10 V 20 Z M 50 50 C 60 40 70 40 80 50")
    assert len(subs) == 2
    square, curve = subs
    assert square[1] is True and curve[1] is False
    assert square[0] == [(10.0, 20.0), (30.0, 20.0), (30.0, 30.0), (10.0, 30.0), (10.0, 20.0)]
    assert curve[0][0] == (50.0, 50.0) and curve[0][-1] == (80.0, 50.0)
    assert all(40.0 < y <= 50.0 for _, y in curve[0][1:-1])  # the cubic bows up (y down)
    quad = path_subpaths("M0 0 Q 10 10 20 0 T 40 0")[0][0]
    assert quad[-1] == pytest.approx((40.0, 0.0)) and max(y for _, y in quad) > 4.0
    smooth = path_subpaths("M0 0 C 0 10 10 10 10 0 S 20 -10 20 0")[0][0]
    assert smooth[-1] == pytest.approx((20.0, 0.0)) and min(y for _, y in smooth) < -4.0
    arc = path_subpaths("M 0 0 A 10 10 0 0 1 20 0")[0][0]
    assert arc[-1] == pytest.approx((20.0, 0.0)) and len(arc) > 8
    assert min(y for _, y in arc) == pytest.approx(-10.0, abs=0.3)  # a half circle, sweeping up
    with pytest.raises(ValueError, match="a command before"):
        path_subpaths("10 20")


def test_elements_classes_transforms_and_widths_are_read(tmp_path: Path) -> None:
    body = (
        '<g transform="translate(10 5) scale(2)">'
        '<rect class="fill" x="0" y="0" width="10" height="5"/>'
        '<circle class="shine" cx="5" cy="5" r="2" style="fill:#fff"/>'
        "</g>"
        '<path class="line" d="M 10 10 L 50 10" stroke-width="3px"/>'
        '<path class="line" d="M 10 30 L 50 30" data-follow="tongue"/>'
        '<ellipse class="tongue" cx="100" cy="100" rx="10" ry="5" transform="rotate(90 100 100)"/>'
        '<polygon class="teeth" points="1,1 5,1 5,3"/>'
        '<polyline class="line" points="60 60 70 60"/>'
        '<line class="line" x1="0" y1="0" x2="5" y2="5"/>'
        '<path d="M 0 0 L 1 1" class="decoration"/>'
    )
    path = tmp_path / "t.svg"
    path.write_text(svg(body, "the drawing"))
    title, shapes, size = read_svg(path)
    assert title == "the drawing" and size == (320.0, 170.0)
    assert [s.slot for s in shapes] == [
        "fill",
        "shine",
        "line",
        "line",
        "tongue",
        "teeth",
        "line",
        "line",
    ]
    rect = shapes[0].subpaths[0][0]
    assert rect[0] == (10.0, 5.0) and rect[2] == (30.0, 15.0)  # translated and scaled
    assert shapes[1].subpaths[0][0][0] == pytest.approx((24.0, 15.0))  # the circle's rightmost
    assert shapes[2].width == 3.0 and shapes[3].width == 0.0 and shapes[3].follow == "tongue"
    rotated = shapes[4].subpaths[0][0]
    xs = [x for x, _ in rotated]
    ys = [y for _, y in rotated]
    assert max(xs) - min(xs) == pytest.approx(10.0, abs=0.1)  # the ellipse stands on end
    assert max(ys) - min(ys) == pytest.approx(20.0, abs=0.1)
    assert shapes[5].subpaths[0][1] is True and shapes[6].subpaths[0][1] is False
    path.write_text(svg('<path class="line tongue" d="M 0 0 L 1 1"/>'))
    with pytest.raises(ValueError, match="two slots"):
        read_svg(path)
    path.write_text(svg('<path class="line" d="M 0 0 L 1 1" data-follow="nose"/>'))
    with pytest.raises(ValueError, match="data-follow 'nose'"):
        read_svg(path)
    path.write_text(svg('<g transform="skewX(10)"><path class="line" d="M 0 0 L 1 1"/></g>'))
    with pytest.raises(ValueError, match="skewX"):
        read_svg(path)


def test_resampling_keeps_the_count_the_start_and_the_spacing() -> None:
    square = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
    ring = resample(square, 8, True)
    assert len(ring) == 8 and ring[0] == (0.0, 0.0) and ring[2] == (10.0, 0.0)
    steps = [math.dist(ring[i], ring[(i + 1) % 8]) for i in range(8)]
    assert all(step == pytest.approx(5.0) for step in steps)
    line = resample([(0.0, 0.0), (10.0, 0.0)], 5, False)
    assert line == [(0.0, 0.0), (2.5, 0.0), (5.0, 0.0), (7.5, 0.0), (10.0, 0.0)]
    assert resample([(3.0, 4.0)], 3, True) == [(3.0, 4.0)] * 3


def test_a_polygon_ring_is_screen_clockwise_from_its_leftmost_point() -> None:
    ccw = [(10.0, 10.0), (10.0, 20.0), (30.0, 20.0), (30.0, 10.0)]  # counter-clockwise (y down)
    assert signed_area(ccw) < 0
    ring = polygon_ring(ccw, 12)
    assert len(ring) == 12 and signed_area(ring) > 0 and ring[0] == (10.0, 10.0)
    assert ring[1][1] == 10.0 and ring[1][0] > 10.0  # along the top first
    hole = polygon_ring(ccw, 12, hole=True)
    assert signed_area(hole) < 0 and hole[0] == (10.0, 10.0)
    assert polygon_ring([(1.0, 1.0), (2.0, 2.0)], 4) == [(1.0, 1.0)] * 4  # degenerate


def test_a_stroke_ring_doubles_an_open_line_back_so_it_opens_into_a_bowl() -> None:
    line = [(100.0, 50.0), (50.0, 60.0), (0.0, 50.0)]  # drawn right to left
    ring = stroke_ring(line, False, 8)
    assert len(ring) == 8 and ring[0] == (0.0, 50.0)  # from the left end
    assert ring[4] == (100.0, 50.0)  # the right end half way round
    assert ring[1] == pytest.approx(ring[7]) and ring[3] == pytest.approx(ring[5])  # back on itself
    assert signed_area(ring) == pytest.approx(0.0, abs=1e-6)
    closed = stroke_ring([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], True, 8)
    assert closed == polygon_ring([(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], 8)
    vertical = stroke_ring([(5.0, 20.0), (5.0, 0.0)], False, 4)
    assert vertical[0] == (5.0, 0.0)  # a vertical line from its top


def test_a_keyframe_takes_each_slots_polygons_in_order_and_refuses_too_many(tmp_path: Path) -> None:
    table = load_face_table(FACE)
    body = (
        '<path class="fill" d="M 100 80 L 220 80 L 220 120 L 100 120 Z"/>'
        '<path class="line" d="M 100 80 L 220 80" stroke-width="3"/>'
        '<path class="line" d="M 100 100 L 220 100"/>'
        '<path class="line" d="M 100 110 L 220 110" stroke-width="1.5"/>'
        '<circle class="drool" cx="160" cy="125" r="4"/>'
    )
    path = tmp_path / "x_rest.svg"
    path.write_text(svg(body, "drawing A"))
    frame = keyframe_of(path, "x", "rest", table, STYLE)
    assert frame.title == "drawing A" and frame.expression == "x" and frame.key == "rest"
    fills, lines, drools = frame.of("fill"), frame.of("line"), frame.of("drool")
    assert len(fills) == 1 and len(fills[0].points) == SLOTS[SLOT_INDEX["fill"]].n
    # the generator decides the widths: the file's pen (its widest stroke, 3) and a stroke
    # without a width draw line_px wide, a thinner one keeps its ratio to the pen
    assert [p.width for p in lines] == [2.6, 2.6, 1.3] and [p.index for p in lines] == [0, 1, 2]
    assert drools[0].points == ((160.0, 125.0),) and drools[0].width == pytest.approx(8.0)
    path.write_text(svg('<path class="fill" d="M 100 80 L 220 80 L 220 120 Z"/>'))
    with pytest.raises(ValueError, match="no line"):
        keyframe_of(path, "x", "rest", table, STYLE)
    too_many = "".join(
        f'<path class="tongue" d="M {x} 80 L {x + 5} 80 L {x} 90 Z"/>' for x in (100, 120)
    )
    path.write_text(svg(too_many + '<path class="line" d="M 100 80 L 220 80"/>'))
    with pytest.raises(ValueError, match="more than 1 polygons in slot tongue"):
        keyframe_of(path, "x", "rest", table, STYLE)
    path.write_text(svg('<path class="line" d="M 2 80 L 220 80"/>'))
    with pytest.raises(ValueError, match="past the screen's margin"):
        keyframe_of(path, "x", "rest", table, STYLE)
    path.write_text(SVG_HEAD.replace("320", "300") + "</svg>")
    with pytest.raises(ValueError, match="a 320x170 canvas"):
        keyframe_of(path, "x", "rest", table, STYLE)


def test_a_style_scales_its_drawings_about_the_common_centre(tmp_path: Path) -> None:
    """A style's ``scale`` enlarges every shape about the set's centre before resampling (the
    drool's drop with them), the strokes staying at the style's line_px; a drawing that fits
    the screen at 1 and not at the scale is refused by the margin; the kawaii centre is its
    own, the others the screen's middle."""
    table = load_face_table(FACE)
    body = (
        '<path class="fill" d="M 140 75 L 180 75 L 180 95 L 140 95 Z"/>'
        '<path class="line" d="M 140 75 L 180 75 L 180 95 L 140 95 Z" stroke-width="3.5"/>'
        '<path class="line" d="M 150 85 L 170 85" stroke-width="2"/>'
        '<circle class="drool" cx="170" cy="95" r="2"/>'
    )
    path = tmp_path / "x_rest.svg"
    path.write_text(svg(body))
    one = keyframe_of(path, "x", "rest", table, ShapeStyle("t", "t", 14.0, 60.0, 3.5, 1.0))
    two = keyframe_of(path, "x", "rest", table, ShapeStyle("t", "t", 14.0, 60.0, 3.5, 2.0))
    xs = [x for x, _ in two.of("fill")[0].points]
    ys = [y for _, y in two.of("fill")[0].points]
    assert (min(xs), max(xs), min(ys), max(ys)) == (120.0, 200.0, 65.0, 105.0)  # about (160, 85)
    assert [p.width for p in one.of("line")] == [p.width for p in two.of("line")] == [14.0, 8.0]
    assert two.of("drool")[0].points == ((180.0, 105.0),)
    assert two.of("drool")[0].width == pytest.approx(8.0)
    assert one.of("drool")[0].width == pytest.approx(4.0)
    high = ShapeStyle("t", "t", 14.0, 60.0, 3.5, 2.0, (160.0, 70.0))
    assert (
        max(y for _, y in keyframe_of(path, "x", "rest", table, high).of("fill")[0].points) == 120.0
    )
    with pytest.raises(ValueError, match="past the screen's margin"):
        keyframe_of(path, "x", "rest", table, ShapeStyle("t", "t", 14.0, 60.0, 3.5, 8.0))
    assert centre_of("anime") == centre_of("comics") == CENTRE == (160.0, 85.0)
    assert centre_of("kawaii") == (160.0, 70.0)
    kawaii = shape_style(table, "kawaii")
    assert kawaii is not None and kawaii.scale == 2.5 and kawaii.line_px == 10.0
    assert kawaii.centre == (160.0, 70.0)
    for name in ("anime", "comics"):
        spec = shape_style(table, name)
        assert spec is not None and spec.scale == 1.0 and spec.centre == CENTRE
    _, shapes, _ = read_svg(path)
    assert stroke_widths(shapes, ShapeStyle("t", "t", 7.0, 60.0, 3.5)) == [7.0, 7.0, 4.0, 7.0]
    # a set's pen: a file of nothing but a 3 px detail (the comics kiss's seam) in a 5 px set
    seam = ShapeStyle("t", "t", 10.0, 60.0, 3.5, pen_px=5.0)
    assert stroke_widths(shapes[2:3], seam) == [4.0] and stroke_widths(shapes[1:2], seam) == [7.0]
    assert set_pen(REPO / "config" / "comics", ("kiss", "smile")) == 5.0
    assert set_pen(REPO / "config" / "kawaii", ("smile",)) == 3.5
    bad = copy.deepcopy(json.loads(FACE.read_text()))
    bad["styles"]["kawaii"]["scale"] = 0
    with pytest.raises(ValueError, match="style kawaii: scale is positive"):
        shape_styles(parse_face_table(bad))


def test_every_expression_has_its_three_keyframes_each_with_an_outline() -> None:
    """config/anime: the 32 expressions' rest, half and open drawings, the extra eps keyframe
    the idle gesture morphs toward and the two overlays, every mouth with an outline stroke, a
    title describing the drawing, and the talk drawings open as bowls (a
    fill)."""
    table = load_face_table(FACE)
    shapes = load_shapes(table)
    assert shapes is not None and shapes.style.name == "anime"
    assert len(shapes.keyframes) == len(table.expressions) == 32
    for trio, e in zip(shapes.keyframes, table.expressions, strict=True):
        assert [k.key for k in trio] == list(KEYS) and all(k.expression == e.name for k in trio)
        for frame in trio:
            assert frame.of("line") and frame.of("line")[0].index == 0, (e.name, frame.key)
            assert frame.title, (e.name, frame.key)
        assert trio[1].of("fill") or trio[1].of("teeth"), e.name  # half open: an inside shows
        assert trio[2].of("fill") or trio[2].of("teeth"), e.name
    assert [f.expression for f in shapes.extras] == ["cheeks", "eps", "stripes"]
    assert shapes.key_index("eps") == 3 * 32 + 1 and shapes.key_index("smile") == 3
    assert shapes.overlays == (shapes.key_index("cheeks"), shapes.key_index("stripes"))
    by_name = {e.name: i for i, e in enumerate(table.expressions)}
    assert shapes.keyframes[by_name["kiss_heart"]][0].of("fill")  # the heart lips
    assert shapes.keyframes[by_name["sleepy"]][0].of("fill")  # the small bean
    assert shapes.keyframes[by_name["tired"]][0].of("drool")  # the drop
    assert shapes.keyframes[by_name["tease"]][0].of("tongue")  # the plump tongue hanging out
    assert shapes.keyframes[by_name["sinister"]][0].of("fang")
    assert shapes.keyframes[by_name["vampire"]][0].of("fang")
    assert shapes.keyframes[by_name["moustache"]][0].of("moustache")
    # the cheeks are the overlay's: no drawn blush on blush, pout or cat_blush
    for name in ("blush", "pout", "cat_blush"):
        for frame in shapes.keyframes[by_name[name]]:
            assert not frame.of("blush") and not frame.of("cheek"), (name, frame.key)
    cheeks, stripes = shapes.extras[0], shapes.extras[2]
    assert cheeks.of("blush") and cheeks.of("cheek") and not cheeks.of("line")
    assert len(stripes.of("line")) == 3 and not stripes.of("blush")
    assert shapes.keyframes[by_name["happy"]][0].of("teeth") and shapes.keyframes[by_name["happy"]][
        0
    ].of("tongue")
    tongue = [p for p in shapes.keyframes[by_name["tease"]][0].polys if p.follow == "tongue"]
    assert tongue  # the tongue's outline follows its wag
    gestures = dict(zip([g.name for g in table.idle.gestures], shapes.gestures, strict=True))
    assert gestures["eps"].toward == "eps" and gestures["grin"].toward == "smile"
    assert gestures["twitch"].curl > 0 and gestures["breath"].sy > 1.0
    assert gestures["tilt"].toward == "" and gestures["tilt"].tilt > 0


def test_the_tables_are_current_and_carry_every_keyframe() -> None:
    """One table per shapes style for the firmware and the simulator, the layout they share
    and the two indexes, every one current; the anime table's content."""
    table = load_face_table(FACE)
    files = generated_shapes(table)
    styles = [s.name for s in shape_styles(table)]
    assert styles[0] == "anime"
    assert set(files) == {
        LAYOUT_HEADER,
        INDEX_HEADER,
        INDEX_SCRIPT,
        *(style_header(s) for s in styles),
        *(style_script(s) for s in styles),
    }
    for path, text in files.items():
        assert path.read_text() == text, f"stale: {path.name} (uv run python -m pepin.face --write)"
    layout = files[LAYOUT_HEADER]
    assert "struct ShapeTable {" in layout and f"kAnimeSlotCount = {len(SLOTS)};" in layout
    index = files[INDEX_HEADER]
    assert f"constexpr int kShapeTableCount = {len(styles)};" in index
    tables = ", ".join(f"&{s}::kTable" for s in styles)
    assert f"kShapeTables[kShapeTableCount] = {{{tables}}};" in index
    assert all(f'#include "{s}_table.h"' in index for s in styles)
    header = files[style_header("anime")]
    assert "namespace anime {" in header and "constexpr int kExtraCount = 3;" in header
    assert "kKeyframes[kFrameCount]" in header and "kGestures[kIdleGestureCount]" in header
    assert "    kStyleAnime, kExtraCount, kFrameCount, kPolyCount, kVertexCount," in header
    assert header.count("// neutral rest:") == 1 and "// eps rest:" in header
    assert "constexpr int16_t kVertices[2 * kVertexCount]" in header
    script = files[style_script("anime")]
    data = json.loads(script.split("const ANIME_TABLE = ", 1)[1].split(";\nif", 1)[0])
    assert data["style"] == 3 and data["keys"] == list(KEYS)
    assert data["extras"] == ["cheeks", "eps", "stripes"]
    assert len(data["frames"]) == 3 * 32 + 3 and data["frames"][0]["name"] == "neutral"
    assert data["overlays"] == [{"name": "cheeks", "frame": 96}, {"name": "stripes", "frame": 98}]
    assert "constexpr int kAnimeOverlayCount = 2;" in layout
    assert "kAnimeOverlayParam[kAnimeOverlayCount] = {kCheeks, kStripes};" in layout
    assert "    {96, 98},  // cheeks, stripes" in header
    assert [s["name"] for s in data["slots"]] == [s.name for s in SLOTS]
    assert data["gestures"][-2]["toward"] == 97  # eps, after the 96 keyframes and cheeks
    n_vertices = sum(len(p["points"]) // 2 for fr in data["frames"] for p in fr["polys"])
    assert f"constexpr int kVertexCount = {n_vertices};" in header
    index_script = files[INDEX_SCRIPT]
    assert "[" + ", ".join(f'require("./{s}_table.js")' for s in styles) + "]" in index_script
    assert "[" + ", ".join(f"{s.upper()}_TABLE" for s in styles) + "]" in index_script


def test_a_table_without_a_shapes_style_writes_no_shape_tables() -> None:
    data: dict[str, Any] = json.loads(FACE.read_text())
    plain = copy.deepcopy(data)
    gone = list(load_face_table(FACE).shape_styles)  # kawaii and comics through their like
    for name in gone:
        del plain["styles"][name]
    for g in plain["idle"]["gestures"]:  # the gestures of those styles alone
        if "styles" in g:
            g["styles"] = [s for s in g["styles"] if s not in gone] or ["manga"]
    plain["style"] = "pepin"  # its own default: the file's (kawaii) is one of the gone
    table = parse_face_table(plain)
    assert shape_styles(table) == () and shape_style(table) is None
    assert load_shapes(table) is None and generated_shapes(table) == {}
    assert shape_style(load_face_table(FACE), "nowhere") is None
    renamed = copy.deepcopy(data)
    renamed["styles"]["anime"]["shapes"] = "nowhere"
    with pytest.raises(ValueError, match=r"no keyframe neutral_rest\.svg"):
        load_shapes(parse_face_table(renamed))
    bad = copy.deepcopy(data)
    bad["idle"]["gestures"][0]["anime"] = {"toward": "nobody"}
    with pytest.raises(ValueError, match="toward 'nobody' is no expression"):
        load_shapes(parse_face_table(bad))
    bad["idle"]["gestures"][0]["anime"] = {"spin": 1}
    with pytest.raises(ValueError, match="idle gesture grin: anime: unknown spin"):
        load_shapes(parse_face_table(bad))


def test_the_renderers_take_a_hand_made_set(tmp_path: Path) -> None:
    """A directory with one square mouth for every keyframe renders to both tables."""
    table = load_face_table(FACE)
    root = tmp_path / "anime"
    root.mkdir()
    body = (
        '<path class="fill" d="M 120 70 L 200 70 L 200 100 L 120 100 Z"/>'
        '<path class="line" d="M 120 70 L 200 70 L 200 100 L 120 100 Z"/>'
    )
    for e in table.expressions:
        for key in KEYS:
            (root / f"{e.name}_{key}.svg").write_text(svg(body, f"{e.name} {key}"))
    (root / "eps_rest.svg").write_text(svg('<path class="line" d="M 150 80 L 170 80"/>', "eps"))
    shapes = load_shapes(table, tmp_path, "anime")
    assert shapes is not None and len(shapes.extras) == 1 and shapes.overlays == (-1, -1)
    header = render_style_header(table, shapes)
    assert (
        "// cat_blush open: cat_blush open" in header and "{-1, -1},  // cheeks, stripes" in header
    )
    (root / "cheeks_rest.svg").write_text(
        svg('<ellipse class="blush" cx="60" cy="60" rx="20" ry="8"/>')
    )
    shapes = load_shapes(table, tmp_path, "anime")
    assert shapes is not None and shapes.overlays == (shapes.key_index("cheeks"), -1)
    assert not shapes.extras[0].of("line")  # an overlay needs no outline
    script = render_style_script(table, shapes)
    assert '"name":"eps","key":"rest"' in script
