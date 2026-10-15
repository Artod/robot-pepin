"""The shape-keyframe mouth: config/<style>/*.svg as the firmware's vertex tables.

The parametric mouth (:mod:`pepin.face`) bends one line; a style whose mouths are drawn (the
anime bowls, drops, clouds and hearts; the kawaii tiny ovals; the comics D laughs and grids)
needs silhouettes instead. Such a style (``config/face.json``
``styles.<name>.engine`` ``"shapes"``) keeps one SVG per expression and talk keyframe in
``config/<styles.name.shapes>/``, named ``<expression>_<rest|half|open>.svg`` on a canvas the
size of the screen. Each path's ``class`` names the layer it belongs to (a :data:`SLOTS`
entry: ``fill``, ``teeth``, ``tongue``, ``shine``, ``line``, ``cheek``, ``fang``,
``moustache``, ``drool``), the layers are drawn in that fixed order in their element's colour,
and ``line`` paths are strokes.

This module flattens the paths, scales every drawing of a style by its ``scale`` about the
style's common centre (:func:`centre_of`: the guide's, where every mouth of the set is centred
at rest), resamples every polygon of a layer to that layer's fixed vertex count (so the
firmware morphs any keyframe into any other vertex by vertex), turns strokes into closed rings
(a closed stroke as it is, an open one doubled back on itself, so a line opens into a bowl's
outline), and writes one table per shapes style,
``firmware/head_esp32/src/<style>_table.h`` and ``firmware/head_esp32/sim/<style>_table.js``,
with the layout they share (``src/shape_layout.h``) and the indexes the engine picks a table
from (``src/shape_tables.h``, ``sim/shape_tables.js``); ``uv run python -m pepin.face --write``
writes them with the parametric tables. The strokes' widths are the style's, not the files':
the set's pen (the widest ``line`` path of its rest drawings: the outlines') is drawn
``line_px`` wide on the screen whatever the SVG says, and a thinner stroke keeps the file's
ratio to the pen (a 2 px gap in a 3.5 px set is ``line_px * 2 / 3.5``); ``scale`` enlarges
the shapes and leaves the strokes to ``line_px``. Nothing beyond the standard library: the
SVG subset is
paths (M L H V C S Q T A Z, absolute and relative), circles, ellipses, rects, polygons,
polylines and lines, with ``translate``, ``scale``, ``rotate`` and ``matrix`` transforms on them
and their groups.
"""

from __future__ import annotations

import json
import math
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from pepin.face import (
    ELEMENTS,
    FIRMWARE,
    GENERATED,
    REPO,
    FaceTable,
    _float,
    rgb565,
)

LAYOUT_HEADER = FIRMWARE / "src" / "shape_layout.h"  # what every style's table shares
INDEX_HEADER = FIRMWARE / "src" / "shape_tables.h"  # the tables, by style
INDEX_SCRIPT = FIRMWARE / "sim" / "shape_tables.js"
KEYS = ("rest", "half", "open")  # the talk keyframes: at rest, half open, open
# The overlays: rest keyframes of no expression (``<name>_rest.svg``, like the gestures' extra
# targets) the engine draws over any look whose parameter of the same name is above 0, at
# that opacity: the style's cheek pair (``cheeks``: the blush of blush, pout, cat_blush,
# shocked, and a later {cheeks} mark over any face) and the shock's vertical stripes down one
# cheek (``stripes``). A style without the file draws none; the parametric engine draws its
# own cheeks and no stripes. An overlay keyframe needs no outline stroke.
OVERLAYS = ("cheeks", "stripes")
Point = tuple[float, float]
Matrix = tuple[float, float, float, float, float, float]  # a b c d e f: x' = ax + cy + e
IDENTITY: Matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
UNIT = 16  # table vertices in 1/16 px
WIDTH_UNIT = 8  # stroke widths in 1/8 px
SVG_NS = "{http://www.w3.org/2000/svg}"
CENTRE: Point = (160.0, 85.0)  # the screen's middle: where a set's mouths are centred at rest
# A style whose mouths share another centre (its guide's): the fixed point of its scale. The
# kawaii mouths sit high on the panel, under the eyes (2026-10-15: the laughs, 100 px tall at
# the 2.5 scale, allow no higher)
COMMON_CENTRE: dict[str, Point] = {"kawaii": (160.0, 70.0)}


def centre_of(style: str) -> Point:
    """A shapes style's common centre: where its rest drawings are centred and the fixed point
    of its ``scale`` (the screen's middle unless :data:`COMMON_CENTRE` keeps its own)."""
    return COMMON_CENTRE.get(style, CENTRE)


@dataclass(frozen=True)
class Slot:
    """One layer of a keyframe: its name (the paths' ``class``), the element whose colour it
    takes, its kind (``poly``: filled polygons; ``stroke``: centrelines the firmware sweeps with
    a width; ``anchor``: one point and a size, drawn by the firmware), the vertex count every
    polygon is resampled to, and how many polygons a keyframe may hold in it."""

    name: str
    element: str
    kind: str
    n: int
    max_polys: int
    alpha: float = 1.0  # the layer's opacity


# The layers in draw order (the firmware's AnimeSlot enum): the blush under everything (its
# soft ellipses, then its strokes), the mouth's interior, the teeth, the tongue, the highlight,
# the outline strokes, then the fangs and the moustache over the line, and the drool's anchor.
SLOTS: tuple[Slot, ...] = (
    Slot("blush", "cheek", "poly", 16, 2, 0.45),
    Slot("cheek", "cheek", "poly", 12, 6),
    Slot("fill", "cavity", "poly", 48, 2),
    Slot("teeth", "teeth", "poly", 24, 3),
    Slot("tongue", "tongue", "poly", 32, 1),
    Slot("shine", "shade", "poly", 16, 2),
    Slot("line", "lip", "stroke", 48, 8),
    Slot("fang", "teeth", "poly", 12, 2),
    Slot("moustache", "moustache", "poly", 24, 1),
    Slot("drool", "shade", "anchor", 1, 1),
)
SLOT_INDEX = {s.name: i for i, s in enumerate(SLOTS)}


@dataclass(frozen=True)
class Poly:
    """One polygon of a keyframe: its slot, its index within the slot (the firmware morphs
    index to index), its vertices (resampled, closed, screen-clockwise for an outer boundary),
    and for a stroke its width in pixels."""

    slot: int
    index: int
    points: tuple[Point, ...]
    width: float = 0.0
    follow: str = ""  # ``data-follow``: the slot whose motion (the tongue's wag) this one shares


@dataclass(frozen=True)
class Keyframe:
    """An expression at one talk key: its polygons and what the drawing shows (the SVG's
    ``<title>``)."""

    expression: str
    key: str
    polys: tuple[Poly, ...]
    title: str = ""

    def of(self, slot: str) -> tuple[Poly, ...]:
        """The polygons of ``slot``, by index."""
        i = SLOT_INDEX[slot]
        return tuple(sorted((p for p in self.polys if p.slot == i), key=lambda p: p.index))


@dataclass(frozen=True)
class ShapeStyle:
    """A shapes style's own numbers (``config/face.json`` ``styles.<name>``): the drawings'
    directory, the outline's width on the screen, the talk cut's tween, the drool's stretch
    period, the drawings' scale and the common centre it scales about, and the files' pen
    (their widest stroke, :func:`set_pen`; 0: each file's own widest)."""

    name: str
    shapes: str
    line_px: float
    talk_tween_ms: float
    drool_s: float
    scale: float = 1.0
    centre: Point = CENTRE
    pen_px: float = 0.0


@dataclass(frozen=True)
class ShapeGesture:
    """An idle gesture in shape space (``idle.gestures[].anime``): a shift (px), a scale about
    the mouth's centre, a tilt (px of y per px of x, the right side down at +), a corner curl
    (px at the corners, down at +), and a morph toward another expression's rest keyframe by
    ``amount``."""

    dx: float = 0.0
    dy: float = 0.0
    sx: float = 1.0
    sy: float = 1.0
    tilt: float = 0.0
    curl: float = 0.0
    toward: str = ""
    amount: float = 0.0


@dataclass(frozen=True)
class ShapeSet:
    """Every expression's three keyframes, in the table's id order, the extra keyframes the
    gestures morph toward and the overlays (``<name>_rest.svg`` files of no expression), the
    style's numbers, and the overlays' flat indexes in :data:`OVERLAYS` order (-1: the style
    has none). A keyframe's index in the firmware's flat list is ``3 * id + key`` for an
    expression's, ``3 * len(expressions) + k`` for an extra."""

    style: ShapeStyle
    keyframes: tuple[tuple[Keyframe, Keyframe, Keyframe], ...]
    extras: tuple[Keyframe, ...]
    gestures: tuple[ShapeGesture, ...]
    overlays: tuple[int, ...] = (-1,) * len(OVERLAYS)

    def key_index(self, name: str) -> int:
        """The flat index of ``name``'s rest keyframe (an expression or an extra)."""
        for i, trio in enumerate(self.keyframes):
            if trio[0].expression == name:
                return 3 * i
        for k, frame in enumerate(self.extras):
            if frame.expression == name:
                return 3 * len(self.keyframes) + k
        raise KeyError(name)


# -- SVG ------------------------------------------------------------------------------------


def _parse_transform(text: str) -> Matrix:
    """An SVG ``transform`` attribute as one matrix (the functions applied left to right)."""
    m = IDENTITY
    for name, args in re.findall(r"(\w+)\s*\(([^)]*)\)", text):
        v = [float(x) for x in re.split(r"[\s,]+", args.strip()) if x]
        if name == "translate":
            t: Matrix = (1.0, 0.0, 0.0, 1.0, v[0], v[1] if len(v) > 1 else 0.0)
        elif name == "scale":
            t = (v[0], 0.0, 0.0, v[1] if len(v) > 1 else v[0], 0.0, 0.0)
        elif name == "rotate":
            a = math.radians(v[0])
            c, s = math.cos(a), math.sin(a)
            t = (c, s, -s, c, 0.0, 0.0)
            if len(v) == 3:  # about a point
                cx, cy = v[1], v[2]
                t = (c, s, -s, c, cx - c * cx + s * cy, cy - s * cx - c * cy)
        elif name == "matrix":
            t = (v[0], v[1], v[2], v[3], v[4], v[5])
        else:
            raise ValueError(f"transform {name}: not supported")
        m = _multiply(m, t)
    return m


def _multiply(m: Matrix, t: Matrix) -> Matrix:
    """``m`` then ``t`` applied after it (SVG's ``m t`` order: t acts on the local points first)."""
    a, b, c, d, e, f = m
    a2, b2, c2, d2, e2, f2 = t
    return (
        a * a2 + c * b2,
        b * a2 + d * b2,
        a * c2 + c * d2,
        b * c2 + d * d2,
        a * e2 + c * f2 + e,
        b * e2 + d * f2 + f,
    )


def _apply(m: Matrix, p: Point) -> Point:
    a, b, c, d, e, f = m
    return (a * p[0] + c * p[1] + e, b * p[0] + d * p[1] + f)


SEGMENTS = 24  # a cubic's flattening (a quadratic half, an arc 12 a quarter turn)


def _bezier3(p0: Point, p1: Point, p2: Point, p3: Point, n: int = SEGMENTS) -> list[Point]:
    """A cubic's points after its start, ``n`` of them."""
    out = []
    for i in range(1, n + 1):
        t = i / n
        u = 1.0 - t
        x = u * u * u * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t * t * t * p3[0]
        y = u * u * u * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t * t * t * p3[1]
        out.append((x, y))
    return out


def _bezier2(p0: Point, p1: Point, p2: Point, n: int = 16) -> list[Point]:
    out = []
    for i in range(1, n + 1):
        t = i / n
        u = 1.0 - t
        out.append(
            (u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
             u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1])
        )  # fmt: skip
    return out


def _arc(p0: Point, rx: float, ry: float, rot: float, large: bool, sweep: bool, p1: Point,
         per_quarter: int = 12) -> list[Point]:  # fmt: skip
    """An SVG elliptical arc's points after its start (the endpoint parameterization of the
    SVG spec, appendix B.2.4)."""
    if p0 == p1:
        return []
    if rx == 0.0 or ry == 0.0:
        return [p1]
    phi = math.radians(rot)
    cp, sp = math.cos(phi), math.sin(phi)
    dx, dy = (p0[0] - p1[0]) / 2.0, (p0[1] - p1[1]) / 2.0
    x1 = cp * dx + sp * dy
    y1 = -sp * dx + cp * dy
    rx, ry = abs(rx), abs(ry)
    lam = (x1 * x1) / (rx * rx) + (y1 * y1) / (ry * ry)
    if lam > 1.0:
        rx *= math.sqrt(lam)
        ry *= math.sqrt(lam)
    num = rx * rx * ry * ry - rx * rx * y1 * y1 - ry * ry * x1 * x1
    den = rx * rx * y1 * y1 + ry * ry * x1 * x1
    coef = math.sqrt(max(num / den, 0.0)) * (-1.0 if large == sweep else 1.0)
    cx1 = coef * rx * y1 / ry
    cy1 = -coef * ry * x1 / rx
    cx = cp * cx1 - sp * cy1 + (p0[0] + p1[0]) / 2.0
    cy = sp * cx1 + cp * cy1 + (p0[1] + p1[1]) / 2.0

    def angle(ux: float, uy: float, vx: float, vy: float) -> float:
        dot = ux * vx + uy * vy
        length = math.hypot(ux, uy) * math.hypot(vx, vy)
        a = math.acos(max(-1.0, min(1.0, dot / length)))
        return -a if ux * vy - uy * vx < 0 else a

    t1 = angle(1.0, 0.0, (x1 - cx1) / rx, (y1 - cy1) / ry)
    dt = angle((x1 - cx1) / rx, (y1 - cy1) / ry, (-x1 - cx1) / rx, (-y1 - cy1) / ry)
    if not sweep and dt > 0:
        dt -= 2 * math.pi
    elif sweep and dt < 0:
        dt += 2 * math.pi
    n = max(2, int(abs(dt) / (math.pi / 2) * per_quarter))
    out = []
    for i in range(1, n + 1):
        t = t1 + dt * i / n
        x, y = rx * math.cos(t), ry * math.sin(t)
        out.append((cp * x - sp * y + cx, sp * x + cp * y + cy))
    out[-1] = p1
    return out


_NUMBER = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def path_subpaths(d: str, segments: int = SEGMENTS) -> list[tuple[list[Point], bool]]:
    """An SVG path's ``d`` flattened (``segments`` points per cubic, proportionally fewer per
    quadratic and per quarter of an arc): each subpath's points and whether it was closed
    (``Z``)."""
    tokens = re.findall(r"[MmLlHhVvCcSsQqTtAaZz]|" + _NUMBER.pattern, d)
    out: list[tuple[list[Point], bool]] = []
    pts: list[Point] = []
    cmd = ""
    cur: Point = (0.0, 0.0)
    start: Point = (0.0, 0.0)
    last_c: Point | None = None  # the previous cubic's second control point (for S)
    last_q: Point | None = None  # the previous quadratic's control point (for T)
    i = 0

    def num() -> float:
        nonlocal i
        i += 1
        return float(tokens[i - 1])

    def flag() -> bool:
        nonlocal i
        i += 1
        return tokens[i - 1] not in ("0", "0.0")

    def close() -> None:
        nonlocal pts
        if pts:
            out.append((pts, False))
        pts = []

    def pt(cur: Point, rel: bool) -> Point:
        x, y = num(), num()
        return (cur[0] + x, cur[1] + y) if rel else (x, y)

    while i < len(tokens):
        if re.fullmatch(r"[A-Za-z]", tokens[i]):
            cmd = tokens[i]
            i += 1
            if cmd in "Zz":
                if pts:
                    out.append((pts, True))
                pts = []
                cur = start
                last_c = last_q = None
                continue
        if not cmd:
            raise ValueError(f"path {d!r}: a command before the numbers")
        rel = cmd.islower()
        c = cmd.upper()
        if c == "M":
            close()
            cur = start = pt(cur, rel)
            pts = [cur]
            cmd = "l" if rel else "L"  # further pairs are lines
            last_c = last_q = None
        elif c == "L":
            cur = pt(cur, rel)
            pts.append(cur)
            last_c = last_q = None
        elif c == "H":
            x = num()
            cur = (cur[0] + x if rel else x, cur[1])
            pts.append(cur)
            last_c = last_q = None
        elif c == "V":
            y = num()
            cur = (cur[0], cur[1] + y if rel else y)
            pts.append(cur)
            last_c = last_q = None
        elif c in "CS":
            if c == "C":
                p1 = pt(cur, rel)
            else:
                p1 = (2 * cur[0] - last_c[0], 2 * cur[1] - last_c[1]) if last_c else cur
            p2 = pt(cur, rel)
            p3 = pt(cur, rel)
            pts += _bezier3(cur, p1, p2, p3, segments)
            cur, last_c, last_q = p3, p2, None
        elif c in "QT":
            if c == "Q":
                p1 = pt(cur, rel)
            else:
                p1 = (2 * cur[0] - last_q[0], 2 * cur[1] - last_q[1]) if last_q else cur
            p2 = pt(cur, rel)
            pts += _bezier2(cur, p1, p2, max(2, segments * 2 // 3))
            cur, last_q, last_c = p2, p1, None
        elif c == "A":
            rx, ry, rot = num(), num(), num()
            large, sweep = flag(), flag()
            p1 = pt(cur, rel)
            pts += _arc(cur, rx, ry, rot, large, sweep, p1, max(2, segments // 2))
            cur = p1
            last_c = last_q = None
        else:
            raise ValueError(f"path {d!r}: command {cmd}")
    close()
    return out


def _ellipse(cx: float, cy: float, rx: float, ry: float, n: int = 2 * SEGMENTS) -> list[Point]:
    return [(cx + rx * math.cos(2 * math.pi * k / n), cy + ry * math.sin(2 * math.pi * k / n))
            for k in range(n)]  # fmt: skip


@dataclass(frozen=True)
class RawShape:
    """One SVG element: its slot, its subpaths (points, closed) in canvas pixels, its stroke
    width (0: the style's default), what it follows (``data-follow``), its part in the drawing
    (``data-part``: the style guide's names, ``band``, ``dome``, ``crease``, ``fold``...) and
    its declared colour (``stroke`` for a line, ``fill`` otherwise), in document order."""

    slot: str
    subpaths: tuple[tuple[tuple[Point, ...], bool], ...]
    width: float
    follow: str = ""
    part: str = ""
    color: str = ""


def _style_attr(el: ET.Element, name: str) -> str | None:
    """``name`` as an attribute or inside ``style=".."``."""
    if name in el.attrib:
        return el.attrib[name]
    for part in el.attrib.get("style", "").split(";"):
        k, _, v = part.partition(":")
        if k.strip() == name:
            return v.strip()
    return None


def _slot_of(el: ET.Element) -> str | None:
    classes = el.attrib.get("class", "").split()
    found = [c for c in classes if c in SLOT_INDEX]
    if len(found) > 1:
        raise ValueError(f"an element in two slots: {' '.join(found)}")
    return found[0] if found else None


def read_svg(path: Path, segments: int = SEGMENTS) -> tuple[str, list[RawShape], Point]:
    """An SVG file: its title, its shapes with a slot class (transforms applied, the curves
    flattened ``segments`` points per cubic), and the canvas size (``viewBox``, or width and
    height)."""
    root = ET.fromstring(path.read_text())
    if root.tag not in ("svg", SVG_NS + "svg"):
        raise ValueError(f"{path.name}: not an SVG document")
    size: Point
    if "viewBox" in root.attrib:
        vb = [float(v) for v in re.split(r"[\s,]+", root.attrib["viewBox"].strip())]
        size = (vb[2], vb[3])
    else:
        size = (float(root.attrib["width"]), float(root.attrib["height"]))
    title = ""
    shapes: list[RawShape] = []

    def walk(el: ET.Element, m: Matrix) -> None:
        nonlocal title
        tag = el.tag.removeprefix(SVG_NS)
        if tag == "title" and not title:
            title = (el.text or "").strip()
            return
        if "transform" in el.attrib:
            m = _multiply(m, _parse_transform(el.attrib["transform"]))
        if tag in ("g", "svg"):
            for child in el:
                walk(child, m)
            return
        slot = _slot_of(el)
        if slot is None:
            return
        a = el.attrib
        subs: list[tuple[list[Point], bool]]
        if tag == "path":
            subs = path_subpaths(a["d"], segments)
        elif tag == "circle":
            r = float(a["r"])
            cx, cy = float(a.get("cx", 0)), float(a.get("cy", 0))
            subs = [(_ellipse(cx, cy, r, r, 2 * segments), True)]
        elif tag == "ellipse":
            subs = [(_ellipse(float(a.get("cx", 0)), float(a.get("cy", 0)), float(a["rx"]),
                              float(a["ry"]), 2 * segments), True)]  # fmt: skip
        elif tag == "rect":
            x, y = float(a.get("x", 0)), float(a.get("y", 0))
            w, h = float(a["width"]), float(a["height"])
            subs = [([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], True)]
        elif tag in ("polygon", "polyline"):
            nums = [float(v) for v in _NUMBER.findall(a["points"])]
            pts = [(nums[k], nums[k + 1]) for k in range(0, len(nums) - 1, 2)]
            subs = [(pts, tag == "polygon")]
        elif tag == "line":
            subs = [([(float(a["x1"]), float(a["y1"])), (float(a["x2"]), float(a["y2"]))], False)]
        else:
            raise ValueError(f"{path.name}: element {tag} in slot {slot}")
        width = _style_attr(el, "stroke-width")
        w = float(re.sub(r"[a-z]+$", "", width)) if width else 0.0
        if w:  # a transform scales the width by its mean scale
            w *= math.sqrt(abs(m[0] * m[3] - m[1] * m[2]))
        follow = el.attrib.get("data-follow", "")
        if follow and follow not in SLOT_INDEX:
            raise ValueError(f"{path.name}: data-follow {follow!r} is no slot")
        color = _style_attr(el, "stroke" if slot == "line" else "fill") or ""
        shapes.append(
            RawShape(
                slot,
                tuple((tuple(_apply(m, p) for p in pts), closed) for pts, closed in subs if pts),
                w,
                follow,
                el.attrib.get("data-part", ""),
                color.strip().upper(),
            )
        )

    walk(root, IDENTITY)
    return title, shapes, size


# -- geometry ----------------------------------------------------------------------------------


def signed_area(points: list[Point] | tuple[Point, ...]) -> float:
    """The shoelace area; positive for a screen-clockwise polygon (y down)."""
    a = 0.0
    n = len(points)
    for i in range(n):
        x0, y0 = points[i]
        x1, y1 = points[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return a / 2.0


def _inside(p: Point, poly: list[Point] | tuple[Point, ...]) -> bool:
    """Even-odd point in polygon."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def _dedupe(points: list[Point], closed: bool) -> list[Point]:
    out: list[Point] = []
    for p in points:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > 1e-6:
            out.append(p)
    if (
        closed
        and len(out) > 1
        and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) < 1e-6
    ):
        out.pop()
    return out


def resample(points: list[Point], n: int, closed: bool) -> list[Point]:
    """``n`` points along the polyline at equal arc-length steps, from its first point (a
    closed one is walked round to just before its start)."""
    if len(points) == 1:
        return [points[0]] * n
    pts = [*points, points[0]] if closed else points
    seg = [math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1])
           for i in range(len(pts) - 1)]  # fmt: skip
    total = sum(seg)
    if total <= 0.0:
        return [points[0]] * n
    steps = n if closed else n - 1
    out: list[Point] = []
    j = 0
    walked = 0.0
    for k in range(n):
        target = total * k / steps
        while j < len(seg) - 1 and walked + seg[j] < target - 1e-9:
            walked += seg[j]
            j += 1
        t = (target - walked) / seg[j] if seg[j] > 0 else 0.0
        t = min(max(t, 0.0), 1.0)
        out.append((pts[j][0] + (pts[j + 1][0] - pts[j][0]) * t,
                    pts[j][1] + (pts[j + 1][1] - pts[j][1]) * t))  # fmt: skip
    return out


BEND_PX = 16.0  # a radian of turning counts as this many px of path when a ring is resampled


def resample_bent(points: list[Point], n: int, closed: bool, bend: float = BEND_PX) -> list[Point]:
    """``n`` points along the polyline at equal steps of a length that counts each radian of
    turning as ``bend`` px (so a corner gets its share of vertices and a straight run fewer:
    the firmware draws the ring as straight segments, and a rounded corner sampled every 8 px
    would come out as a chamfer). From the first point; a closed one is walked round to just
    before its start, its first vertex's turn counted last."""
    if len(points) == 1:
        return [points[0]] * n
    ring = [*points, points[0]] if closed else list(points)
    segs = [math.hypot(ring[i + 1][0] - ring[i][0], ring[i + 1][1] - ring[i][1])
            for i in range(len(ring) - 1)]  # fmt: skip
    m = len(segs)
    turns = [0.0] * (m + 1)  # the turn on arriving at ring[i], weighted
    for i in range(1, m + (1 if closed else 0)):
        a, b = segs[i - 1], segs[i % m]
        if a <= 0.0 or b <= 0.0:
            continue
        ax = (ring[i][0] - ring[i - 1][0]) / a
        ay = (ring[i][1] - ring[i - 1][1]) / a
        j = (i + 1) % (m + 1) if closed and i == m else i + 1
        bx = (ring[j][0] - ring[i][0]) / b
        by = (ring[j][1] - ring[i][1]) / b
        dot = max(-1.0, min(1.0, ax * bx + ay * by))
        turns[i] = bend * math.acos(dot)
    if closed:
        turns[m] = turns[m]  # the start's turn, counted on arriving back at it
    # the cumulative weight at each vertex's arrival and departure
    arrive = [0.0] * (m + 1)
    depart = [0.0] * (m + 1)
    for i in range(1, m + 1):
        arrive[i] = depart[i - 1] + segs[i - 1]
        depart[i] = arrive[i] + turns[i]
    total = depart[m]
    if total <= 0.0:
        return [points[0]] * n
    steps = n if closed else n - 1
    out: list[Point] = []
    i = 0
    for k in range(n):
        target = total * k / steps
        while i < m - 1 and depart[i] + segs[i] < target - 1e-9:
            i += 1
        if target <= depart[i] + 1e-9:  # inside a vertex's turn (or on it)
            out.append(ring[i])
            continue
        t = (target - depart[i]) / segs[i] if segs[i] > 0 else 0.0
        t = min(max(t, 0.0), 1.0)
        out.append((ring[i][0] + (ring[i + 1][0] - ring[i][0]) * t,
                    ring[i][1] + (ring[i + 1][1] - ring[i][1]) * t))  # fmt: skip
    return out


def leftmost(points: list[Point]) -> int:
    """The index of the leftmost point (the topmost of those)."""
    return min(range(len(points)), key=lambda i: (points[i][0], points[i][1]))


def polygon_ring(points: list[Point], n: int, hole: bool = False, bend: float = BEND_PX
                 ) -> list[Point]:  # fmt: skip
    """A closed outline as the table's ring: screen-clockwise (a hole counter-clockwise),
    starting at its leftmost point, ``n`` vertices (``bend``: :func:`resample_bent`'s, in the
    polygon's own pixels: a scaled drawing's is scaled with it)."""
    pts = _dedupe(points, True)
    if len(pts) < 3:
        return [pts[0]] * n if pts else [(0.0, 0.0)] * n
    if (signed_area(pts) > 0) == hole:
        pts.reverse()
    k = leftmost(pts)
    pts = pts[k:] + pts[:k]
    return resample_bent(pts, n, True, bend)


def stroke_ring(points: list[Point], closed: bool, n: int, bend: float = BEND_PX) -> list[Point]:
    """A stroke's centreline as the table's ring of ``n`` vertices: a closed one as
    :func:`polygon_ring`, an open one from its left end to the right end and back (the first
    ``n/2 + 1`` vertices forward, the rest returning), so it morphs into a closed outline by
    opening like a mouth."""
    pts = _dedupe(points, closed)
    if closed:
        return polygon_ring(pts, n, bend=bend)
    if len(pts) < 2:
        return [pts[0]] * n if pts else [(0.0, 0.0)] * n
    a, b = pts[0], pts[-1]
    if b[0] < a[0] - 1e-6 or (abs(b[0] - a[0]) <= 1e-6 and b[1] < a[1]):  # from the left end
        pts.reverse()
    forward = resample_bent(pts, n // 2 + 1, False, bend)
    back = resample_bent(list(reversed(pts)), n - n // 2 + 1, False, bend)
    return forward + back[1:-1]


def scaled(shape: RawShape, style: ShapeStyle) -> RawShape:
    """The shape as the style draws it: every point scaled by the style's ``scale`` about its
    common centre (the stroke width stays the file's: the ratio :func:`keyframe_of` reads)."""
    s = style.scale
    if s == 1.0:
        return shape
    cx, cy = style.centre
    return RawShape(
        shape.slot,
        tuple(
            (tuple((cx + s * (x - cx), cy + s * (y - cy)) for x, y in sub), closed)
            for sub, closed in shape.subpaths
        ),
        shape.width,
        shape.follow,
        shape.part,
        shape.color,
    )


def stroke_widths(shapes: list[RawShape], style: ShapeStyle) -> list[float]:
    """Each stroke's width on the screen, in the shapes' order: the style's ``line_px`` for
    the pen (the style's ``pen_px``, else the file's widest stroke) and for any stroke the
    file gives no width, and for a thinner stroke the file's ratio to the pen (a set's own
    pen sets the ratios, the style the pixels)."""
    pen = style.pen_px or max((s.width for s in shapes if s.slot == "line"), default=0.0)
    pen = pen or style.line_px
    return [style.line_px * (s.width or pen) / pen for s in shapes]


def set_pen(directory: Path, names: tuple[str, ...]) -> float:
    """A set's pen: the widest stroke of the named expressions' rest drawings (0 when none
    declares a width: every stroke then draws ``line_px`` wide)."""
    widths = [
        s.width
        for name in names
        for s in read_svg(directory / f"{name}_{KEYS[0]}.svg")[1]
        if s.slot == "line"
    ]
    return max(widths, default=0.0)


def _check_canvas(size: Point, table: FaceTable, name: str) -> None:
    w, h = table.screen["width"], table.screen["height"]
    if (round(size[0]), round(size[1])) != (w, h):
        raise ValueError(f"{name}: a {w}x{h} canvas, not {size[0]:g}x{size[1]:g}")


def keyframe_of(path: Path, expression: str, key: str, table: FaceTable, style: ShapeStyle
                ) -> Keyframe:  # fmt: skip
    """One SVG file as a keyframe of ``style``: each slot's polygons in document order, scaled
    about the style's centre, resampled, the strokes ``line_px`` wide (:func:`stroke_widths`);
    ``ValueError`` when a scaled shape reaches past the screen's margin."""
    title, raw, size = read_svg(path)
    _check_canvas(size, table, path.name)
    shapes = [scaled(s, style) for s in raw]
    widths = stroke_widths(shapes, style)
    bend = BEND_PX * style.scale
    counts = dict.fromkeys(SLOT_INDEX, 0)
    polys: list[Poly] = []
    for shape, width in zip(shapes, widths, strict=True):
        slot = SLOTS[SLOT_INDEX[shape.slot]]
        if slot.kind == "anchor":
            pts = [p for sub, _ in shape.subpaths for p in sub]
            cx = sum(p[0] for p in pts) / len(pts)
            cy = sum(p[1] for p in pts) / len(pts)
            r = max(math.hypot(p[0] - cx, p[1] - cy) for p in pts)
            rings: list[tuple[list[Point], float]] = [([(cx, cy)], 2.0 * r)]
        elif slot.kind == "stroke":
            rings = [(stroke_ring(list(sub), closed, slot.n, bend), width)
                     for sub, closed in shape.subpaths]  # fmt: skip
        else:
            outers = [list(sub) for sub, _ in shape.subpaths]
            rings = []
            for k, sub in enumerate(outers):
                hole = any(_inside(sub[0], other) for j, other in enumerate(outers) if j != k
                           and len(other) >= 3)  # fmt: skip
                rings.append((polygon_ring(sub, slot.n, hole, bend), 0.0))
        for ring, width in rings:
            if counts[slot.name] >= slot.max_polys:
                raise ValueError(
                    f"{path.name}: more than {slot.max_polys} polygons in slot {slot.name}"
                )
            polys.append(
                Poly(SLOT_INDEX[slot.name], counts[slot.name], tuple(ring), width, shape.follow)
            )
            counts[slot.name] += 1
    if counts["line"] == 0 and expression not in OVERLAYS:
        raise ValueError(f"{path.name}: no line (the outline is stroke 0 of every keyframe)")
    m = table.geometry["margin_px"]
    w, h = table.screen["width"], table.screen["height"]
    for poly in polys:
        reach = poly.width / 2.0 if SLOTS[poly.slot].kind == "stroke" else 0.0
        for x, y in poly.points:
            if not (
                m <= x - reach and x + reach <= w - m and m <= y - reach and y + reach <= h - m
            ):
                raise ValueError(
                    f"{path.name}: {SLOTS[poly.slot].name} reaches ({x:.1f}, {y:.1f}),"
                    f" past the screen's margin of {m:g} px"
                )
    return Keyframe(expression, key, tuple(polys), title)


# -- the tables -----------------------------------------------------------------------------


def shape_styles(table: FaceTable) -> tuple[ShapeStyle, ...]:
    """The table's shapes styles, in the 'C' config's order (none when all are parametric)."""
    out = []
    for s in table.styles:
        if s.engine == "shapes":
            own = s.own
            scale = float(own.get("scale", 1.0))
            if not scale > 0.0:
                raise ValueError(f"style {s.name}: scale is positive, not {scale:g}")
            out.append(
                ShapeStyle(
                    s.name,
                    str(own.get("shapes", s.name)),
                    float(own.get("line_px", 2.5)),
                    float(own.get("talk_tween_ms", 60.0)),
                    float(own.get("drool_s", 3.5)),
                    scale,
                    centre_of(s.name),
                )
            )
    return tuple(out)


def shape_style(table: FaceTable, name: str | None = None) -> ShapeStyle | None:
    """The shapes style ``name`` (the first one when None), or None when there is no such."""
    styles = shape_styles(table)
    if name is None:
        return styles[0] if styles else None
    return next((s for s in styles if s.name == name), None)


def _gesture(block: dict[str, Any], names: tuple[str, ...], where: str) -> ShapeGesture:
    bad = sorted(set(block) - {"dx", "dy", "sx", "sy", "tilt", "curl", "toward", "amount", "note"})
    if bad:
        raise ValueError(f"{where}: unknown {', '.join(bad)}")
    toward = str(block.get("toward", ""))
    if toward and toward not in names:
        raise ValueError(f"{where}: toward {toward!r} is no expression and no extra keyframe")
    return ShapeGesture(
        float(block.get("dx", 0.0)),
        float(block.get("dy", 0.0)),
        float(block.get("sx", 1.0)),
        float(block.get("sy", 1.0)),
        float(block.get("tilt", 0.0)),
        float(block.get("curl", 0.0)),
        toward,
        float(block.get("amount", 0.0)),
    )


def load_shapes(
    table: FaceTable, root: Path | None = None, style: str | None = None
) -> ShapeSet | None:
    """The keyframes of the shapes style ``style`` (the first one when None) from
    ``config/<shapes>/`` (``root``: another parent of the directory), every expression's
    three; None when the table has no such style."""
    spec = shape_style(table, style)
    if spec is None:
        return None
    directory = (root or REPO / "config") / spec.shapes
    missing = [n for n in table.names if not (directory / f"{n}_{KEYS[0]}.svg").is_file()]
    if missing:
        raise ValueError(f"{spec.name}: no keyframe {missing[0]}_{KEYS[0]}.svg in {directory}")
    spec = replace(spec, pen_px=set_pen(directory, table.names))
    frames = []
    for e in table.expressions:
        trio = []
        for key in KEYS:
            path = directory / f"{e.name}_{key}.svg"
            if not path.is_file():
                raise ValueError(f"{spec.name}: no keyframe {path.name} in {directory}")
            trio.append(keyframe_of(path, e.name, key, table, spec))
        frames.append((trio[0], trio[1], trio[2]))
    # the extra rest keyframes: files of no expression, the gestures' targets
    extras = []
    for path in sorted(directory.glob("*_rest.svg")):
        name = path.name.removesuffix("_rest.svg")
        if name not in table.names:
            extras.append(keyframe_of(path, name, KEYS[0], table, spec))
    names = (*table.names, *(f.expression for f in extras))
    gestures = tuple(
        _gesture(g.shapes.get(spec.name, {}), names, f"idle gesture {g.name}: {spec.name}")
        for g in table.idle.gestures
    )
    by_name = [p.name for p in table.params]
    for name in OVERLAYS:
        if name not in by_name:
            raise ValueError(f"overlay {name}: no such parameter in the table")
    shapes = ShapeSet(spec, tuple(frames), tuple(extras), gestures)
    overlays = tuple(shapes.key_index(n) if n in names else -1 for n in OVERLAYS)
    return replace(shapes, overlays=overlays)


def load_shape_sets(table: FaceTable, root: Path | None = None) -> tuple[ShapeSet, ...]:
    """Every shapes style's keyframes, in the 'C' config's order."""
    out = []
    for spec in shape_styles(table):
        shapes = load_shapes(table, root, spec.name)
        assert shapes is not None
        out.append(shapes)
    return tuple(out)


def _q(v: float, unit: int) -> int:
    return round(v * unit)


def _camel(name: str) -> str:
    return "".join(part.capitalize() for part in name.split("_"))


def render_layout_header(table: FaceTable) -> str:
    """shape_layout.h: what every shapes style's table shares: the layers in draw order with
    their vertex counts, the units, the overlays and the parameters that show them, the
    structs of a table."""
    lines = [
        f"// {GENERATED}",
        "// The shape engine's layout, shared by every shapes style's table (<style>_table.h): the",
        "// layers and their vertex counts, the units, the structs a table is made of.",
        "#pragma once",
        "",
        "#include <stdint.h>",
        "",
        '#include "face_table.h"',
        "",
        "namespace face {",
        "",
        f"constexpr int kAnimeUnit = {UNIT};  // table vertices in 1/{UNIT} px",
        f"constexpr int kAnimeWidthUnit = {WIDTH_UNIT};  // stroke widths in 1/{WIDTH_UNIT} px",
        f"constexpr int kAnimeKeys = {len(KEYS)};  // per expression: " + ", ".join(KEYS),
        "",
        "// the layers, in draw order",
        f"constexpr int kAnimeSlotCount = {len(SLOTS)};",
        "enum AnimeSlot : uint8_t {",
        *(f"  kSlot{_camel(s.name)} = {i}," for i, s in enumerate(SLOTS)),
        "};",
        "enum AnimeKind : uint8_t { kKindPoly = 0, kKindStroke = 1, kKindAnchor = 2 };",
        "constexpr uint8_t kAnimeSlotElement[kAnimeSlotCount] = {"
        + ", ".join(f"kEl{_camel(s.element)}" for s in SLOTS)
        + "};",
        "constexpr uint8_t kAnimeSlotKind[kAnimeSlotCount] = {"
        + ", ".join(f"kKind{_camel(s.kind)}" for s in SLOTS)
        + "};",
        "constexpr uint8_t kAnimeSlotN[kAnimeSlotCount] = {"
        + ", ".join(str(s.n) for s in SLOTS)
        + "};",
        "constexpr uint8_t kAnimeSlotMax[kAnimeSlotCount] = {"
        + ", ".join(str(s.max_polys) for s in SLOTS)
        + "};",
        "constexpr float kAnimeSlotAlpha[kAnimeSlotCount] = {"
        + ", ".join(_float(s.alpha) for s in SLOTS)
        + "};",
        f"constexpr int kAnimeSlotVertices = {sum(s.n * s.max_polys for s in SLOTS)};"
        "  // every slot full",
        f"constexpr int kAnimeSlotPolys = {sum(s.max_polys for s in SLOTS)};",
        "",
        "// the overlays a table may hold (rest keyframes of no expression, ShapeTable::overlays),",
        "// each drawn over any look at the opacity of its parameter",
        f"constexpr int kAnimeOverlayCount = {len(OVERLAYS)};",
        "constexpr uint8_t kAnimeOverlayParam[kAnimeOverlayCount] = {"
        + ", ".join(f"k{_camel(n)}" for n in OVERLAYS)
        + "};",
        "",
        "// each polygon: its slot, its index in the slot, its stroke width (1/8 px), its flags",
        "// (1: it follows the tongue's wag) and its first vertex in the table's vertices (x, y",
        "// pairs in 1/16 px)",
        "struct AnimePoly {",
        "  uint8_t slot;",
        "  uint8_t index;",
        "  uint8_t width;",
        "  uint8_t flags;",
        "  uint16_t first;",
        "};",
        "// a keyframe: its first polygon in the table's polygons and its count",
        "struct AnimeKey {",
        "  uint16_t first;",
        "  uint8_t count;",
        "};",
        "// an idle gesture in shape space: shift, scale, tilt, curl, a morph toward a keyframe",
        "// (an index of the table's keyframes; -1: none) by amount",
        "struct AnimeGesture {",
        "  float dx, dy, sx, sy, tilt, curl;",
        "  int16_t toward;",
        "  float amount;",
        "};",
        "// one shapes style's table: the keyframes, flat (expression id * kAnimeKeys + key: rest,",
        "// half, open; then the extras: rest keyframes of no expression, the gestures' targets",
        "// and the overlays), their polygons and vertices, the gestures (kIdleGestureCount), the",
        "// style's numbers, the overlays' keyframes (kAnimeOverlayCount; -1: none)",
        "struct ShapeTable {",
        "  uint8_t style;          // the Style it draws",
        "  uint8_t extra_count;",
        "  uint16_t frame_count;   // kAnimeKeys * kExpressionCount + extra_count",
        "  uint16_t poly_count;",
        "  uint32_t vertex_count;",
        "  float talk_tween_ms;    // the tween between two talk drawings at a cut",
        "  float drool_s;          // the drool drop's stretch period",
        "  const AnimePoly* polys;",
        "  const int16_t* vertices;",
        "  const AnimeKey* keyframes;",
        "  const AnimeGesture* gestures;",
        "  int16_t overlays[kAnimeOverlayCount];",
        "};",
        "",
        "}  // namespace face",
        "",
    ]
    return "\n".join(lines)


def render_style_header(table: FaceTable, shapes: ShapeSet) -> str:
    """<style>_table.h: the style's keyframes' vertices, polygons, gestures and numbers as one
    ShapeTable in its own namespace."""
    style = shapes.style
    vertices: list[str] = []
    polys: list[str] = []
    keys: list[str] = []
    n_vertices = 0
    flat = [f for trio in shapes.keyframes for f in trio] + list(shapes.extras)
    for frame in flat:
        first = len(polys)
        for p in frame.polys:
            flags = 1 if p.follow == "tongue" else 0
            polys.append(
                f"  {{{p.slot}, {p.index}, {_q(p.width, WIDTH_UNIT)}, {flags}, {n_vertices}}},"
                f"  // {frame.expression} {frame.key} {SLOTS[p.slot].name}[{p.index}]"
            )
            vertices.append(
                "  " + ", ".join(f"{_q(x, UNIT)}, {_q(y, UNIT)}" for x, y in p.points) + ","
            )
            n_vertices += len(p.points)
        keys.append(
            f"  {{{first}, {len(frame.polys)}}},  // {frame.expression} {frame.key}: {frame.title}"
        )
    gestures = [
        f"  {{{_float(g.dx)}, {_float(g.dy)}, {_float(g.sx)}, {_float(g.sy)}, {_float(g.tilt)},"
        f" {_float(g.curl)}, {shapes.key_index(g.toward) if g.toward else -1},"
        f" {_float(g.amount)}}},  // {table.idle.gestures[k].name}"
        for k, g in enumerate(shapes.gestures)
    ]
    lines = [
        f"// {GENERATED}",
        f"// The shape keyframes of the {style.name} style (config/{style.shapes}/*.svg).",
        "#pragma once",
        "",
        '#include "shape_layout.h"',
        "",
        "namespace face {",
        f"namespace {style.name} {{",
        "",
        f"constexpr int kExtraCount = {len(shapes.extras)};"
        "  // rest keyframes of no expression, the gestures' targets",
        "constexpr int kFrameCount = kAnimeKeys * kExpressionCount + kExtraCount;",
        f"constexpr int kPolyCount = {len(polys)};",
        "constexpr AnimePoly kPolys[kPolyCount] = {",
        *polys,
        "};",
        f"constexpr int kVertexCount = {n_vertices};",
        "constexpr int16_t kVertices[2 * kVertexCount] = {",
        *vertices,
        "};",
        "// the keyframes, flat: expression id * kAnimeKeys + key (rest, half, open), then the",
        "// extras",
        "constexpr AnimeKey kKeyframes[kFrameCount] = {",
        *keys,
        "};",
        "constexpr AnimeGesture kGestures[kIdleGestureCount] = {",
        *gestures,
        "};",
        "constexpr ShapeTable kTable = {",
        f"    kStyle{_camel(style.name)}, kExtraCount, kFrameCount, kPolyCount, kVertexCount,",
        f"    {_float(style.talk_tween_ms)}, {_float(style.drool_s)},",
        "    kPolys, kVertices, kKeyframes, kGestures,",
        "    {" + ", ".join(str(i) for i in shapes.overlays) + "},  // " + ", ".join(OVERLAYS),
        "};",
        "",
        f"}}  // namespace {style.name}",
        "}  // namespace face",
        "",
    ]
    return "\n".join(lines)


def render_index_header(sets: tuple[ShapeSet, ...]) -> str:
    """shape_tables.h: every shapes style's table, for the engine to pick by style."""
    names = [s.style.name for s in sets]
    lines = [
        f"// {GENERATED}",
        "// The shapes styles' tables (config/face.json `styles`, engine shapes), one per style;",
        "// the shape engine picks the one of the style in force (face_anime.h shape_table_of).",
        "#pragma once",
        "",
        '#include "shape_layout.h"',
        *(f'#include "{n}_table.h"' for n in names),
        "",
        "namespace face {",
        "",
        f"constexpr int kShapeTableCount = {len(names)};",
        "constexpr const ShapeTable* kShapeTables[kShapeTableCount] = {"
        + ", ".join(f"&{n}::kTable" for n in names)
        + "};",
        "",
        "}  // namespace face",
        "",
    ]
    return "\n".join(lines)


def _script_name(style: str) -> str:
    return f"{style.upper()}_TABLE"


def render_style_script(table: FaceTable, shapes: ShapeSet) -> str:
    """<style>_table.js: the same keyframes for the simulator (a plain script)."""
    style = shapes.style
    payload = {
        "style": table.style_id_of(style.name),
        "unit": UNIT,
        "width_unit": WIDTH_UNIT,
        "line_px": style.line_px,
        "scale": style.scale,
        "talk_tween_ms": style.talk_tween_ms,
        "drool_s": style.drool_s,
        "keys": list(KEYS),
        "slots": [
            {
                "name": s.name,
                "element": ELEMENTS.index(s.element),
                "kind": s.kind,
                "n": s.n,
                "max": s.max_polys,
                "alpha": s.alpha,
            }
            for s in SLOTS
        ],
        "extras": [f.expression for f in shapes.extras],
        "overlays": [
            {"name": n, "frame": i} for n, i in zip(OVERLAYS, shapes.overlays, strict=True)
        ],
        "frames": [
            {
                "name": frame.expression,
                "key": frame.key,
                "title": frame.title,
                "polys": [
                    {
                        "slot": p.slot,
                        "index": p.index,
                        "width": _q(p.width, WIDTH_UNIT),
                        "flags": 1 if p.follow == "tongue" else 0,
                        "points": [v for x, y in p.points for v in (_q(x, UNIT), _q(y, UNIT))],
                    }
                    for p in frame.polys
                ],
            }
            for frame in [f for trio in shapes.keyframes for f in trio] + list(shapes.extras)
        ],
        "gestures": [
            {
                "dx": g.dx,
                "dy": g.dy,
                "sx": g.sx,
                "sy": g.sy,
                "tilt": g.tilt,
                "curl": g.curl,
                "toward": shapes.key_index(g.toward) if g.toward else -1,
                "amount": g.amount,
            }
            for g in shapes.gestures
        ],
    }
    body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    name = _script_name(style.name)
    return (
        f"// {GENERATED}\n"
        f"// The shape keyframes of the {style.name} style (config/{style.shapes}/*.svg).\n"
        f"const {name} = {body};\n"
        f'if (typeof module !== "undefined") module.exports = {name};\n'
    )


def render_index_script(sets: tuple[ShapeSet, ...]) -> str:
    """shape_tables.js: every shapes style's table, in the 'C' config's order (under node
    the scripts are required; in the page they are loaded before this one)."""
    names = [s.style.name for s in sets]
    required = ", ".join(f'require("./{n}_table.js")' for n in names)
    globals_ = ", ".join(_script_name(n) for n in names)
    return (
        f"// {GENERATED}\n"
        "// The shapes styles' tables (config/face.json `styles`, engine shapes), one per style.\n"
        'const SHAPE_TABLES = typeof module !== "undefined"\n'
        f"  ? [{required}]\n"
        f"  : [{globals_}];\n"
        'if (typeof module !== "undefined") module.exports = SHAPE_TABLES;\n'
    )


def style_header(style: str) -> Path:
    """The firmware's table of a shapes style."""
    return FIRMWARE / "src" / f"{style}_table.h"


def style_script(style: str) -> Path:
    """The simulator's table of a shapes style."""
    return FIRMWARE / "sim" / f"{style}_table.js"


def generated_shapes(table: FaceTable, root: Path | None = None) -> dict[Path, str]:
    """The shape engine's generated files and their text: the layout, a table per shapes
    style for the firmware and the simulator, and the two indexes (none without a shapes
    style)."""
    sets = load_shape_sets(table, root)
    if not sets:
        return {}
    out = {LAYOUT_HEADER: render_layout_header(table)}
    for shapes in sets:
        out[style_header(shapes.style.name)] = render_style_header(table, shapes)
    out[INDEX_HEADER] = render_index_header(sets)
    for shapes in sets:
        out[style_script(shapes.style.name)] = render_style_script(table, shapes)
    out[INDEX_SCRIPT] = render_index_script(sets)
    return out


__all__ = [
    "CENTRE",
    "COMMON_CENTRE",
    "INDEX_HEADER",
    "INDEX_SCRIPT",
    "KEYS",
    "LAYOUT_HEADER",
    "OVERLAYS",
    "SLOTS",
    "Keyframe",
    "Poly",
    "ShapeGesture",
    "ShapeSet",
    "ShapeStyle",
    "centre_of",
    "generated_shapes",
    "keyframe_of",
    "load_shape_sets",
    "load_shapes",
    "path_subpaths",
    "polygon_ring",
    "read_svg",
    "render_index_header",
    "render_index_script",
    "render_layout_header",
    "render_style_header",
    "render_style_script",
    "resample",
    "resample_bent",
    "rgb565",
    "scaled",
    "set_pen",
    "shape_style",
    "shape_styles",
    "signed_area",
    "stroke_ring",
    "stroke_widths",
    "style_header",
    "style_script",
]
