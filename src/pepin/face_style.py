"""The shape mouth sets' style guides as measurements (firmware/head_esp32/ANIME_STYLE.md,
KAWAII_STYLE.md, COMICS_STYLE.md).

A set of drawings is one thing when every drawing obeys the same pen: one line width, one
palette, the corners it allows, one band of teeth, a few sizes, one centre, the highlights
only where the guide puts them, no inner line reaching past the outline. Each guide is a
:class:`Guide` of numbers (:data:`ANIME`, :data:`KAWAII`, :data:`COMICS`) in the drawings' own
pixels; this module measures them on every keyframe of the style's directory as the screen
gets it (the flattened SVG paths scaled by the style's ``scale`` about its centre, the
resampled rings the firmware gets, the strokes at the style's ``line_px``), the guide's
lengths scaled the same way, and names the outliers; ``tests/unit/test_anime_style.py``,
``test_kawaii_style.py`` and ``test_comics_style.py`` fail on any.

    uv run python -m pepin.face_style                  # every keyframe's numbers and faults
    uv run python -m pepin.face_style --style kawaii   # one set (default: all of them)
    uv run python -m pepin.face_style --faults         # the faults alone
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from pepin.face import REPO, FaceTable, load_face_table
from pepin.face_shapes import (
    BEND_PX,
    CENTRE,
    KEYS,
    OVERLAYS,
    SLOT_INDEX,
    SLOTS,
    Point,
    RawShape,
    ShapeStyle,
    centre_of,
    keyframe_of,
    polygon_ring,
    read_svg,
    resample,
    scaled,
    shape_style,
    stroke_ring,
    stroke_widths,
)

Array = NDArray[np.float64]

RADIUS_WINDOW_PX = 2.0  # the circumradius is taken over this much path either side of a point
FLATTEN = 96  # points per cubic for the measures (the table's 24 leave kinks in the window)
TONGUE_OUT = "tongue_out"  # a highlight or a crease on every tongue out and nowhere else
NEVER = "never"  # none anywhere
FREE = "free"  # not measured


@dataclass(frozen=True)
class Guide:
    """One style guide's numbers, in the drawings' own pixels (a style's ``scale`` multiplies
    every length when the set is measured): the style (its directory under config/), the two
    stroke widths the files declare, which ``data-part`` tags are detail lines, lines inside
    a mouth, lines on a tongue and parts drawn sharp on purpose, the smallest corner radius
    and the sharpest resampled turn per slot, the band's depth, the centre (the fixed point
    of the scale) and the thinking lines' offset, the talk drawings' tolerances, the colours
    each slot may take, the width class of every expression and the symbols' sizes, and
    where highlights and creases belong."""

    style: str
    line_px: float
    detail_px: float
    detail_parts: frozenset[str]
    inner_parts: frozenset[str]
    tongue_parts: frozenset[str]
    sharp_parts: frozenset[str]
    min_radius: dict[str, float]
    max_turn_deg: dict[str, float]
    band_px: tuple[float, float]
    centre: Point
    centre_tol: float
    aside: dict[str, float]  # the mouths set aside: how far right of the centre
    top_tol: float
    width_tol: float
    overrun_px: float
    trace_px: float
    palette: dict[str, tuple[str, ...]]
    sizes: dict[str, tuple[float, float]]
    symbols: dict[str, tuple[str, float]]
    symbol_tol: float
    class_of: dict[str, str]
    shine: str
    crease: str


ANIME = Guide(
    style="anime",
    line_px=3.0,
    detail_px=2.0,
    detail_parts=frozenset(("crease", "fold", "bracket")),  # beside a mouth, not the mouth
    inner_parts=frozenset(("band", "dome", "bite", "gap")),
    tongue_parts=frozenset(("tongue", "crease")),
    sharp_parts=frozenset(),
    min_radius={
        "fill": 5.0,
        "line": 5.0,
        "tongue": 5.0,
        "teeth": 2.5,
        "shine": 2.5,
        "blush": 2.5,
        "moustache": 2.5,
        "cheek": 0.9,
    },
    max_turn_deg={"fill": 35.0, "tongue": 35.0, "line": 50.0},
    band_px=(11.0, 17.0),
    centre=CENTRE,
    centre_tol=4.0,
    aside={"thinking": 34.0, "thinking_flat": 34.0, "struggling": 34.0},
    top_tol=5.0,
    width_tol=0.2,
    overrun_px=0.5,
    trace_px=1.0,
    palette={
        "fill": ("#EC7488",),
        "line": ("#C04860",),
        "teeth": ("#FFFFFF",),
        "fang": ("#FFFFFF",),
        "tongue": ("#F09AA8",),
        "shine": ("#F8C8D0",),
        "drool": ("#F8C8D0",),
        "blush": ("#F4A6B0",),
        "cheek": ("#F4A6B0",),
        "moustache": ("#3C2A30",),
    },
    sizes={
        "line": (100.0, 124.0),
        "short": (64.0, 80.0),
        "small": (46.0, 66.0),
        "big": (134.0, 166.0),
    },
    symbols={"kiss": ("height", 48.0), "kiss_heart": ("width", 96.0), "eps": ("height", 34.0)},
    symbol_tol=0.12,
    class_of={
        "neutral": "line",
        "smile": "line",
        "happy": "big",
        "grin": "line",
        "clenched": "big",
        "sad": "line",
        "worried": "line",
        "surprised": "small",
        "thinking": "short",
        "struggling": "short",
        "flat": "line",
        "sleepy": "small",
        "focused": "line",
        "listening": "small",
        "sinister": "big",
        "kiss": "symbol",
        "kiss_heart": "symbol",
        "pout": "short",
        "thinking_flat": "short",
        "moustache": "line",
        "o_mouth": "small",
        "vampire": "big",
        "terminator": "big",
        "blush": "small",
        "tease": "line",
        "tired": "line",
        "cat": "line",
        "cat_blush": "line",
        "smile_wide": "line",
        "happy_plain": "big",
        "angry": "short",
        "shocked": "small",
        "eps": "symbol",
    },
    shine=TONGUE_OUT,
    crease=TONGUE_OUT,
)

# The kawaii set (firmware/head_esp32/KAWAII_STYLE.md): tiny mouths in one near-black line,
# the laughs and the quiet mouths rose inside, the slack mouths dark, white teeth with a thin
# outline, a plain rose tongue, the light-blue tear; drawn small (the files' 3.5 px pen, 2.5
# for the teeth's outlines and dashes) and shown enlarged by the style's scale about its centre,
# high on the panel
LINE = "#504743"  # the line, and the slack mouths' inside
ROSE = "#DD7487"
PINK = "#EAB1B7"
KAWAII = Guide(
    style="kawaii",
    line_px=3.5,
    detail_px=2.5,
    detail_parts=ANIME.detail_parts,
    inner_parts=ANIME.inner_parts | frozenset(("tooth", "fang")),  # the thin outlines inside
    tongue_parts=ANIME.tongue_parts,
    sharp_parts=frozenset(("fang",)),  # a fang's outline traces the sharp tooth
    min_radius={**ANIME.min_radius, "fill": 3.0, "line": 3.0, "tongue": 3.0},
    max_turn_deg=ANIME.max_turn_deg,
    band_px=(7.0, 13.0),
    centre=centre_of("kawaii"),
    centre_tol=ANIME.centre_tol,
    aside={"thinking": 34.0, "thinking_flat": 34.0, "struggling": 12.0},
    top_tol=ANIME.top_tol,
    width_tol=ANIME.width_tol,
    overrun_px=ANIME.overrun_px,
    trace_px=ANIME.trace_px,
    palette={
        "fill": (ROSE, LINE),
        "line": (LINE,),
        "teeth": ("#FFFFFF",),
        "fang": ("#FFFFFF",),
        "tongue": (ROSE,),
        "shine": ("#90C1E9",),
        "drool": ("#90C1E9",),
        "blush": (PINK,),
        "cheek": (PINK,),
        "moustache": (LINE,),
    },
    sizes={  # the mouth's width at rest, by class
        "line": (34.0, 68.0),
        "short": (20.0, 40.0),
        "small": (16.0, 44.0),
        "big": (60.0, 92.0),
        "dot": (6.0, 14.0),
    },
    symbols={"kiss": ("height", 40.0), "kiss_heart": ("height", 40.0), "eps": ("height", 22.0)},
    symbol_tol=ANIME.symbol_tol,
    class_of={
        "neutral": "short",
        "smile": "line",
        "happy": "big",
        "grin": "big",
        "clenched": "big",
        "sad": "short",
        "worried": "small",
        "surprised": "small",
        "thinking": "small",
        "struggling": "short",
        "flat": "short",
        "sleepy": "dot",
        "focused": "short",
        "listening": "small",
        "sinister": "line",
        "kiss": "symbol",
        "kiss_heart": "symbol",
        "pout": "short",
        "thinking_flat": "dot",
        "moustache": "short",
        "o_mouth": "small",
        "vampire": "big",
        "terminator": "big",
        "blush": "small",
        "tease": "line",
        "tired": "small",
        "cat": "line",
        "cat_blush": "line",
        "smile_wide": "line",
        "happy_plain": "big",
        "angry": "small",
        "shocked": "small",
        "eps": "symbol",
    },
    shine=NEVER,
    crease=NEVER,
)

INK = "#1D1D1B"
COMICS = Guide(
    style="comics",
    line_px=5.0,
    detail_px=3.0,
    detail_parts=frozenset(("crease", "fold", "tick", "chin", "seam")),
    inner_parts=frozenset(("band", "dome", "bite", "gap")),
    tongue_parts=frozenset(("tongue", "crease")),
    sharp_parts=frozenset(("zigzag",)),
    min_radius={
        "fill": 3.0,
        "line": 3.0,
        "tongue": 3.0,
        "teeth": 1.0,
        "blush": 2.5,
        "moustache": 2.5,
    },
    max_turn_deg={"fill": 50.0, "tongue": 50.0, "line": 60.0},
    band_px=(14.0, 34.0),
    centre=CENTRE,
    centre_tol=4.0,
    aside={"thinking": 34.0, "thinking_flat": 34.0},
    top_tol=5.0,
    width_tol=0.2,
    overrun_px=0.5,
    trace_px=1.0,
    palette={
        "fill": (INK,),
        "line": (INK,),
        "teeth": ("#FFFFFF",),
        "fang": ("#FFFFFF",),
        "tongue": ("#DE5B55",),
        "shine": ("#C1D5FA",),
        "drool": ("#C1D5FA",),
        "blush": ("#DE5B55",),
        "cheek": ("#DE5B55",),
        "moustache": (INK,),
    },
    sizes={
        "line": (100.0, 136.0),
        "short": (60.0, 84.0),
        "small": (30.0, 90.0),
        "big": (146.0, 196.0),
        "tall": (72.0, 92.0),
    },
    symbols={
        "kiss": ("width", 60.0),
        "kiss_heart": ("width", 72.0),
        "blush": ("height", 38.0),
        "eps": ("height", 30.0),
    },
    symbol_tol=0.12,
    class_of={
        "neutral": "line",
        "smile": "line",
        "happy": "big",
        "grin": "big",
        "clenched": "big",
        "sad": "line",
        "worried": "line",
        "surprised": "tall",
        "thinking": "short",
        "struggling": "line",
        "flat": "line",
        "sleepy": "small",
        "focused": "line",
        "listening": "small",
        "sinister": "big",
        "kiss": "symbol",
        "kiss_heart": "symbol",
        "pout": "short",
        "thinking_flat": "short",
        "moustache": "line",
        "o_mouth": "small",
        "vampire": "big",
        "terminator": "big",
        "blush": "symbol",
        "tease": "line",
        "tired": "small",
        "cat": "line",
        "cat_blush": "line",
        "smile_wide": "line",
        "happy_plain": "big",
        "angry": "short",
        "shocked": "small",
        "eps": "symbol",
    },
    shine=NEVER,
    crease=FREE,
)

GUIDES = {g.style: g for g in (ANIME, KAWAII, COMICS)}


@dataclass
class Measure:
    """One keyframe's numbers on the screen: the style's scale, the stroke widths the file
    declares (the pen's) and the ones the screen gets, the smallest corner radius per slot,
    the sharpest turn of a resampled ring per slot, the band's depth, the mouth's box (fill
    and line at rest: the hooks in, the tongue, the cheeks and the details out), how far an
    inner line reaches past the outline, how far the outline strays from the fill, the
    highlights' and tongues' presence, the colours off the palette, and the faults."""

    expression: str
    key: str
    scale: float = 1.0
    widths: tuple[float, ...] = ()
    widths_px: tuple[float, ...] = ()
    radius: dict[str, float] = field(default_factory=dict)
    turn: dict[str, float] = field(default_factory=dict)
    band: float | None = None
    box: tuple[float, float, float, float] | None = None  # x0, y0, x1, y1
    lip: float = 0.0  # the upper lip's y at the mouth's centre
    where: dict[str, Point] = field(default_factory=dict)  # the smallest corner, per slot
    overrun: float = 0.0
    trace: float = 0.0
    shine: bool = False
    crease: bool = False
    tongue_out: bool = False
    off_palette: tuple[str, ...] = ()
    faults: list[str] = field(default_factory=list)

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0] if self.box else 0.0

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1] if self.box else 0.0

    @property
    def centre(self) -> Point:
        if not self.box:
            return (0.0, 0.0)
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)


# -- geometry ------------------------------------------------------------------------------------


def dense(points: tuple[Point, ...] | list[Point], closed: bool, step: float = 1.0) -> Array:
    """The polyline resampled every ``step`` px (its corners kept: the flattening's points
    are only ever moved along their own segments)."""
    pts = [p for p in points]
    if len(pts) < 2:
        return np.array(pts or [(0.0, 0.0)], dtype=float)
    ring = [*pts, pts[0]] if closed else pts
    length = sum(math.dist(ring[i], ring[i + 1]) for i in range(len(ring) - 1))
    n = max(4, round(length / step))
    return np.array(resample(pts, n, closed), dtype=float)


def min_radius(poly: Array, closed: bool, window: int, keep: Array | None = None) -> float:
    """The smallest circumradius through points ``window`` apart either side of a point (a
    corner reads as a small circle; a straight run as none); an open polyline's ends are caps,
    not corners, and only the points ``keep`` (a mask) count."""
    return min_radius_at(poly, closed, window, keep)[0]


def min_radius_at(poly: Array, closed: bool, window: int, keep: Array | None = None
                  ) -> tuple[float, Point]:  # fmt: skip
    """:func:`min_radius` and the point it was found at."""
    n = len(poly)
    if n < 2 * window + 1:
        return math.inf, (float(poly[0][0]), float(poly[0][1]))
    if closed:
        a = np.roll(poly, window, axis=0)
        b = poly
        c = np.roll(poly, -window, axis=0)
        mask = np.ones(n, dtype=bool)
    else:
        a = poly[: n - 2 * window]
        b = poly[window : n - window]
        c = poly[2 * window :]
        mask = np.ones(n - 2 * window, dtype=bool)
    if keep is not None:
        mask &= keep if closed else keep[window : n - window]
    ab = np.linalg.norm(b - a, axis=1)
    bc = np.linalg.norm(c - b, axis=1)
    ca = np.linalg.norm(a - c, axis=1)
    area2 = np.abs(
        (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(area2 > 1e-9, ab * bc * ca / (2.0 * area2), np.inf)
    r = np.where(mask, r, np.inf)
    if not len(r) or not mask.any():
        return math.inf, (float(poly[0][0]), float(poly[0][1]))
    k = int(r.argmin())
    return float(r[k]), (float(b[k][0]), float(b[k][1]))


def max_turn(ring: tuple[Point, ...] | list[Point]) -> float:
    """The sharpest turn (degrees) between a ring's consecutive segments, the degenerate ones
    skipped; a ring that runs there and back (an open stroke) is read as the open polyline,
    its two turn-backs left out."""
    pts = [p for p in ring]
    n = len(pts)
    if n < 3:
        return 0.0
    there_and_back = math.dist(pts[1], pts[-1]) < 1e-6
    if there_and_back:
        pts = pts[: n // 2 + 1]
    dirs = []
    for i in range(len(pts) - (0 if not there_and_back else 1)):
        p, q = pts[i], pts[(i + 1) % len(pts)]
        d = math.dist(p, q)
        if d > 1e-6:
            dirs.append(((q[0] - p[0]) / d, (q[1] - p[1]) / d))
    if len(dirs) < 2:
        return 0.0
    pairs = zip(dirs, dirs[1:] + ([] if there_and_back else dirs[:1]), strict=False)
    worst = 0.0
    for (ax, ay), (bx, by) in pairs:
        dot = max(-1.0, min(1.0, ax * bx + ay * by))
        worst = max(worst, math.degrees(math.acos(dot)))
    return worst


def inside(points: Array, polygon: Array) -> NDArray[np.bool_]:
    """Even-odd point in polygon, for many points."""
    x, y = points[:, 0][:, None], points[:, 1][:, None]
    x0, y0 = polygon[:, 0][None, :], polygon[:, 1][None, :]
    x1, y1 = np.roll(polygon[:, 0], -1)[None, :], np.roll(polygon[:, 1], -1)[None, :]
    crosses = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x0 + (y - y0) * (x1 - x0) / np.where(y1 == y0, 1.0, y1 - y0)
    hits = crosses & (x < xi)
    return np.asarray(hits.sum(axis=1) % 2 == 1, dtype=bool)


def distance(points: Array, polyline: Array, closed: bool) -> Array:
    """Each point's distance to the polyline."""
    a = polyline
    b = np.roll(polyline, -1, axis=0) if closed else polyline[1:]
    if not closed:
        a = polyline[:-1]
    if len(a) == 0:
        return np.asarray(np.linalg.norm(points - polyline[0], axis=1), dtype=float)
    ab = b - a
    ab2 = np.maximum((ab * ab).sum(axis=1), 1e-12)
    ap = points[:, None, :] - a[None, :, :]
    t = np.clip((ap * ab[None, :, :]).sum(axis=2) / ab2[None, :], 0.0, 1.0)
    closest = a[None, :, :] + t[:, :, None] * ab[None, :, :]
    d = np.linalg.norm(points[:, None, :] - closest, axis=2)
    return np.asarray(d.min(axis=1), dtype=float)


def outside(points: Array, polygons: list[Array]) -> Array:
    """Each point's distance outside the union of the polygons (0 inside any)."""
    if not polygons:
        return np.full(len(points), math.inf)
    out = np.full(len(points), math.inf)
    for poly in polygons:
        d = distance(points, poly, True)
        d[inside(points, poly)] = 0.0
        out = np.minimum(out, d)
    return out


def span_at(polygon: Array, x: float) -> float:
    """The polygon's vertical extent on the line ``x``."""
    x0, y0 = polygon[:, 0], polygon[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    crosses = (x0 <= x) != (x1 <= x)
    if not crosses.any():
        return 0.0
    ys = y0[crosses] + (x - x0[crosses]) * (y1[crosses] - y0[crosses]) / (x1[crosses] - x0[crosses])
    return float(ys.max() - ys.min())


# -- the measures ----------------------------------------------------------------------------------


def _is_mouth_line(s: RawShape, guide: Guide) -> bool:
    """The mouth's own line: not a tongue's, not a detail, not a line inside the mouth."""
    inner = guide.detail_parts | guide.inner_parts | guide.tongue_parts
    return s.slot == "line" and s.follow != "tongue" and s.part not in inner


def _body_shapes(shapes: list[RawShape], guide: Guide) -> list[RawShape]:
    """The shapes the mouth's box is read from: the fill, the body of teeth when there is no
    fill, and the mouth lines; a mouth of nothing but a tongue (the comics kiss: red lips)
    is the tongue."""
    has_fill = any(s.slot == "fill" for s in shapes)
    out = [
        s
        for s in shapes
        if s.slot == "fill" or (s.slot == "teeth" and not has_fill) or _is_mouth_line(s, guide)
    ]
    return out or [s for s in shapes if s.slot == "tongue"]


def _box(shapes: list[RawShape], guide: Guide) -> tuple[float, float, float, float] | None:
    """The mouth's box, the lines' half width not included."""
    xs: list[float] = []
    ys: list[float] = []
    for s in _body_shapes(shapes, guide):
        for sub, _ in s.subpaths:
            for x, y in sub:
                xs.append(x)
                ys.append(y)
    if not xs:
        return None
    return (min(xs), min(ys), max(xs), max(ys))


def _lip(shapes: list[RawShape], box: tuple[float, float, float, float], guide: Guide) -> float:
    """The upper lip's y at the mouth's centre: the body's top there, or the mouth line's."""
    x = (box[0] + box[2]) / 2.0
    ys: list[float] = []
    for s in _body_shapes(shapes, guide):
        for sub, closed in s.subpaths:
            pts = dense(sub, closed)
            near = pts[np.abs(pts[:, 0] - x) <= 1.0]
            if len(near):
                ys.append(float(near[:, 1].min()))
    return min(ys) if ys else box[1]


def _ring_turns(shapes: list[RawShape], guide: Guide, bend: float) -> dict[str, float]:
    """The sharpest turn of the resampled rings the firmware gets, per slot, the parts drawn
    sharp on purpose left out (the shapes at the table's own flattening and bend)."""
    turn: dict[str, float] = {}
    for s in shapes:
        if s.slot not in guide.max_turn_deg or s.part in guide.sharp_parts:
            continue
        slot = SLOTS[SLOT_INDEX[s.slot]]
        for sub, closed in s.subpaths:
            if slot.kind == "stroke":
                ring = stroke_ring(list(sub), closed, slot.n, bend)
            else:
                ring = polygon_ring(list(sub), slot.n, bend=bend)
            turn[s.slot] = max(turn.get(s.slot, 0.0), max_turn(ring))
    return turn


def style_of(table: FaceTable, guide: Guide) -> ShapeStyle:
    """The table's shapes style of the guide (its scale, centre and line, the guide's pen),
    or, for a table without it, one drawn as the guide's files are (no scale, the pen's
    line)."""
    spec = shape_style(table, guide.style)
    if spec is None:
        spec = ShapeStyle(guide.style, guide.style, guide.line_px, 60.0, 3.5, 1.0, guide.centre)
    return replace(spec, pen_px=guide.line_px)


def measure(path: Path, expression: str, key: str, table: FaceTable, guide: Guide) -> Measure:
    """One keyframe's numbers against ``guide``, as the screen gets it: the drawing scaled by
    the table's style (no judgement: :func:`faults` reads them)."""
    spec = style_of(table, guide)
    m = Measure(expression, key, spec.scale)
    _, raw, _ = read_svg(path, FLATTEN)
    shapes = [scaled(s, spec) for s in raw]
    line_px = spec.line_px  # the outline on the screen
    try:
        keyframe_of(path, expression, key, table, spec)
    except ValueError as e:
        m.faults.append(str(e).removeprefix(f"{path.name}: "))
    m.widths = tuple(sorted({s.width or guide.line_px for s in raw if s.slot == "line"}))
    m.widths_px = tuple(sorted({
        w for s, w in zip(shapes, stroke_widths(shapes, spec), strict=True) if s.slot == "line"
    }))  # fmt: skip
    m.off_palette = tuple(
        f"{s.slot} {s.color or '(none)'}"
        for s in shapes
        if s.slot in guide.palette and s.color not in guide.palette[s.slot]
    )
    fills = [dense(sub, True) for s in shapes if s.slot == "fill" for sub, _ in s.subpaths]
    teeth = [dense(sub, True) for s in shapes if s.slot == "teeth" and s.part != "hint"
             for sub, _ in s.subpaths]  # fmt: skip
    tongues = [dense(sub, True) for s in shapes if s.slot == "tongue" for sub, _ in s.subpaths]
    body = fills or teeth
    mouth_lines = [
        (dense(sub, closed), closed)
        for s in shapes
        if _is_mouth_line(s, guide)
        for sub, closed in s.subpaths
    ]
    window = round(RADIUS_WINDOW_PX * spec.scale)  # the window in the drawing's own pixels
    # corners: the smallest circumradius per slot; the small parts' points under a mouth line
    # do not count (the band's ends, a dome's corners on the lip); the sharp parts neither
    for s in shapes:
        if s.slot not in guide.min_radius or s.part in guide.sharp_parts:
            continue
        for sub, closed in s.subpaths:
            poly = dense(sub, closed)
            keep: Array | None = None
            if s.slot in ("teeth", "tongue", "shine") and mouth_lines:
                under = np.full(len(poly), False)
                for line, line_closed in mouth_lines:
                    under |= distance(poly, line, line_closed) <= line_px / 2.0 + 1.0
                keep = ~under
            r, at = min_radius_at(poly, closed, window, keep)
            if r < m.radius.get(s.slot, math.inf):
                m.radius[s.slot] = r
                m.where[s.slot] = at
    # the resampled rings' turns
    _, coarse, _ = read_svg(path)
    m.turn = _ring_turns([scaled(s, spec) for s in coarse], guide, BEND_PX * spec.scale)
    # the band: the teeth under an upper lip (a frame with a fill)
    if fills and teeth:
        band = max(teeth, key=lambda t: t[:, 0].max() - t[:, 0].min())
        m.band = span_at(band, float((band[:, 0].min() + band[:, 0].max()) / 2.0))
    # the tongue out of the mouth, and the highlights
    if tongues:
        t = tongues[0]
        m.tongue_out = not fills or bool((outside(t, fills) > 0.5).mean() > 0.25)
    m.shine = any(s.slot == "shine" for s in shapes)
    # the parts inside a mouth stay inside it: the band, and the tongue when it is a dome
    if fills:
        for poly in teeth + ([] if m.tongue_out else tongues):
            m.overrun = max(m.overrun, float(outside(poly, fills).max()))
    m.crease = any(s.slot == "line" and s.part == "crease" for s in shapes)
    # containment: the inner lines against the body, a tongue's lines against the tongue (an
    # untagged line after the outline is inner: a hand drawing's band edge)
    first = True
    for s in shapes:
        if s.slot != "line":
            continue
        untagged = not s.part and s.follow != "tongue" and not first
        first = False
        for sub, closed in s.subpaths:
            poly = dense(sub, closed)
            if (s.part in guide.inner_parts or untagged) and body:
                m.overrun = max(m.overrun, float(outside(poly, body + tongues).max()))
            elif s.part in guide.tongue_parts and tongues:
                m.overrun = max(m.overrun, float(outside(poly, tongues).max()))
    # the outline (the first mouth line) traces the body, except under a tongue
    if body and mouth_lines:
        line, line_closed = mouth_lines[0]
        for poly in body:
            d = distance(poly, line, line_closed)
            if tongues:
                d[inside(poly, tongues[0])] = 0.0
            m.trace = max(m.trace, float(d.max()))
        d = np.full(len(line), math.inf)
        for poly in body:
            d = np.minimum(d, distance(line, poly, True))
        m.trace = max(m.trace, float(d.max()))
    m.box = _box(shapes, guide)
    if m.box is not None:
        m.lip = _lip(shapes, m.box, guide)
    return m


def faults(m: Measure, guide: Guide, rest: Measure | None = None) -> list[str]:
    """The guide's rules ``m`` breaks (``rest``: the expression's rest keyframe, for the talk
    drawings' rules); the guide's lengths at the measure's scale."""
    out = list(m.faults)
    s = m.scale
    bad = [w for w in m.widths if w not in (guide.line_px, guide.detail_px)]
    if bad:
        out.append(f"stroke widths {bad} (the guide's: {guide.line_px:g} and {guide.detail_px:g})")
    if m.off_palette:
        out.append("off the palette: " + ", ".join(m.off_palette))
    for slot, r in sorted(m.radius.items()):
        if r < guide.min_radius[slot] * s:
            x, y = m.where.get(slot, (0.0, 0.0))
            out.append(
                f"{slot}: a corner of radius {r:.1f} px at ({x:.0f}, {y:.0f})"
                f" (at least {guide.min_radius[slot] * s:g})"
            )
    for slot, t in sorted(m.turn.items()):
        if t > guide.max_turn_deg[slot]:
            out.append(
                f"{slot}: a resampled turn of {t:.0f} deg (at most {guide.max_turn_deg[slot]:g})"
            )
    lo, hi = (v * s for v in guide.band_px)
    if m.band is not None and not (lo <= m.band <= hi):
        out.append(f"band {m.band:.1f} px deep (the guide's {lo:g}..{hi:g})")
    if m.overrun > guide.overrun_px * s:
        out.append(f"an inner line or part reaches {m.overrun:.1f} px past the outline")
    if m.trace > guide.trace_px * s:
        out.append(f"the outline strays {m.trace:.1f} px from the fill")
    if guide.shine == TONGUE_OUT and m.shine != m.tongue_out:
        out.append(
            "a highlight without a tongue out" if m.shine else "a tongue out without its highlight"
        )
    elif guide.shine == NEVER and m.shine:
        out.append("a highlight (the guide has none)")
    if guide.crease == TONGUE_OUT and m.crease != m.tongue_out:
        out.append(
            "a crease without a tongue out" if m.crease else "a tongue out without its crease"
        )
    elif guide.crease == NEVER and m.crease:
        out.append("a crease (the guide has none)")
    if m.box is None:
        out.append("no mouth")
        return out
    cx, cy = m.centre
    if m.key == KEYS[0]:
        want_x = guide.centre[0] + guide.aside.get(m.expression, 0.0) * s
        want_y = guide.centre[1]
        if abs(cx - want_x) > guide.centre_tol * s or abs(cy - want_y) > guide.centre_tol * s:
            out.append(f"centred at ({cx:.0f}, {cy:.0f}), not ({want_x:.0f}, {want_y:.0f})")
        cls = guide.class_of.get(m.expression)
        if cls == "symbol":
            axis, size = guide.symbols[m.expression]
            size *= s
            have = m.width if axis == "width" else m.height
            if abs(have - size) > guide.symbol_tol * size:
                out.append(f"{axis} {have:.0f} px, the symbol's {size:g}")
        elif cls is not None:
            lo, hi = (v * s for v in guide.sizes[cls])
            if not (lo <= m.width <= hi):
                out.append(f"width {m.width:.0f} px, the {cls} class's {lo:g}..{hi:g}")
        else:
            out.append("no size class")
    elif rest is not None and rest.box is not None:
        if abs(m.lip - rest.lip) > guide.top_tol * s:
            out.append(f"the lip at {m.lip:.0f}, the rest's at {rest.lip:.0f}: the lip moved")
        off = abs(m.width - rest.width) > guide.width_tol * rest.width
        if guide.class_of.get(m.expression) != "symbol" and rest.width > 0 and off:
            out.append(f"width {m.width:.0f} px against the rest's {rest.width:.0f}")
    return out


def measure_set(
    table: FaceTable | None = None, guide: Guide = ANIME, root: Path | None = None
) -> list[Measure]:
    """Every keyframe of the guide's directory (``config/<style>/``, or under ``root``),
    measured and judged (the gestures' extras too; the overlays are no mouths)."""
    table = table or load_face_table(REPO / "config" / "face.json")
    directory = (root or REPO / "config") / guide.style
    names = [*table.names, *sorted(
        p.name.removesuffix("_rest.svg")
        for p in directory.glob("*_rest.svg")
        if p.name.removesuffix("_rest.svg") not in (*table.names, *OVERLAYS)
    )]  # fmt: skip
    out: list[Measure] = []
    for name in names:
        rest: Measure | None = None
        for key in KEYS:
            path = directory / f"{name}_{key}.svg"
            if not path.is_file():
                continue
            m = measure(path, name, key, table, guide)
            m.faults = faults(m, guide, rest)
            if key == KEYS[0]:
                rest = m
            out.append(m)
    return out


def _fmt(v: float | None) -> str:
    if v is None:
        return "-"
    return "inf" if math.isinf(v) else f"{v:.1f}"


def main(argv: list[str]) -> int:
    only_faults = "--faults" in argv
    styles = [argv[argv.index("--style") + 1]] if "--style" in argv else list(GUIDES)
    broken = 0
    table = load_face_table(REPO / "config" / "face.json")
    for style in styles:
        guide = GUIDES[style]
        measures = measure_set(table, guide)
        spec = style_of(table, guide)
        print(
            f"== {style}: {len(measures)} keyframes, scale {spec.scale:g} about"
            f" ({spec.centre[0]:g}, {spec.centre[1]:g}), the line {spec.line_px:g} px on the"
            " screen (the numbers below are the screen's)"
        )
        print(f"{'keyframe':22} {'px':9} {'r fill':6} {'r line':6} {'r tng':6} {'r tth':6}"
              f" {'turn':5} {'band':5} {'centre':9} {'w':5} {'over':5} {'trace':5}")  # fmt: skip
        for m in measures:
            if m.faults:
                broken += 1
            if only_faults and not m.faults:
                continue
            cx, cy = m.centre
            turn = max(m.turn.values(), default=0.0)
            print(
                f"{m.expression + ' ' + m.key:22} {','.join(f'{w:g}' for w in m.widths_px):9}"
                f" {_fmt(m.radius.get('fill')):6} {_fmt(m.radius.get('line')):6}"
                f" {_fmt(m.radius.get('tongue')):6} {_fmt(m.radius.get('teeth')):6}"
                f" {turn:5.0f} {_fmt(m.band):5} {cx:4.0f},{cy:<4.0f} {m.width:5.0f}"
                f" {m.overrun:5.1f} {m.trace:5.1f}"
            )
            for f in m.faults:
                print(f"    - {f}")
    print(f"{broken} keyframes with faults")
    return 1 if broken else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
