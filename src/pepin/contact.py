"""The camera as a lidar at floor height: the contact line, lifted onto the floor plane.

The monocular depth network gives the scene's shape with an uncertain size; the lidar gives
exact ranges, but only in its own plane 20 cm above the floor (``config/lidar.json``). What
both miss is where things *touch the floor*: a chair's legs under its seat, a box lower than
the lidar's plane, the plinth of a sofa set back from its front. In the image that place is a
line. Going up any column from the bottom, the pixels are floor — their depth grows row by row
exactly as the floor plane's would (:func:`pepin.depth.floor_depth`) — until something standing
on the floor takes over and the depth stops growing. The row where the floor ends is the
contact; the ray through that pixel, intersected with the floor plane (inverse perspective
mapping, :func:`ipm`), is a point in base_link whose range does not depend on the network's
scale at all: only the mount (height, tilt), the optics and the cart's lean (the IMU's up
vector, as :func:`pepin.depth.floor_depth` takes it) enter. The network only *classifies*
pixels as floor or not, and for that a tolerance in metres of height, widened by the depth's
own uncertainty (:class:`DepthNoise`, :func:`floor_mask`), is enough. Per column the floor's
run from the bottom is cleaned of holes and read (:func:`contact_columns`); the band's own width
is taken back off the range (:func:`band_shadow`) and, binned per half-degree of bearing like
:func:`pepin.depth.depth_to_scan`, the contact points make a LaserScan at height zero
(:func:`contact_scan`) — a second lidar, at the floor — with a :class:`ContactVerdict` for the
node's report line.

The geometry that does not depend on the frame — where each pixel's ray lands on the floor,
how far that is and which bearing it is — is a :class:`FloorPlane`, built once per mount and
lean and reused for every frame. On our mount (1.23 m up, 26 degrees down, 78 degrees across
1280x720) the image's bottom row already looks 1.02 m ahead: nothing nearer than that is in
the picture at all, which is exactly the range the lidar covers best. The two sensors are
complements, not rivals.

What this is worth, measured on a recorded drive against the lidar
(``scratch/contact_vs_lidar.py``, run 0171, 22 open-floor frames, 2026-09-11): where the lidar
returns at 1.50-1.75 m the contact scan's median difference is +1 cm (p90 28 cm) and 3 % of its
marks are false; nearer than 1.25 m it reads 65 cm too far, because it cannot see under 1.02 m
and reports the foot of whatever is behind; past 2 m every mark is false, because the network's
floor drifts out of the band (the reason for CONTACT_MAX_RANGE). And the claim it exists for
holds exactly: correcting the depth by the lidar's law — a factor of 2.2 — moved the contact
ranges by 0.0 cm.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import IntEnum

import numpy as np
import numpy.typing as npt
from numpy.lib.stride_tricks import sliding_window_view

from pepin.depth import (
    FLOOR_HEIGHT_TOLERANCE,
    NEAR_M,
    SCAN_HALF_FOV,
    SCAN_STEP,
    UP_LEVEL,
    Array,
    CameraPose,
    Intrinsics,
    Mask,
    floor_depth,
)

Ints = npt.NDArray[np.int64]

CONTACT_MIN_RUN = 8  # rows: a shorter floor run is noise, a shorter gap inside the floor is a hole
CONTACT_BOTTOM_SLACK = 16  # rows the floor may start above the bottom (the border row is an edge)
CONTACT_KTH = 2  # the k-th nearest contact of a bearing marks: a lone column never does
CONTACT_SCALE_ROWS = 40  # bottom rows whose median depth / floor ratio is the frame's floor scale
CONTACT_MAX_RANGE = 2.0  # metres, measured (scratch/contact_vs_lidar.py, run 0171, 2026-09-11):
# after the lidar's law the network's floor sits at 1.01 of the plane at 1.0-1.5 m, 0.96 at
# 1.5-2.0, 0.89 at 2.0-2.5 and 0.80 at 2.5-3.0 — and 0.89 of the plane is 13 cm of height, the
# width of the band itself. Past 2 m the floor leaves the band on its own and every column ends
# in a contact that is not there: against the lidar, 3 % false marks below 1.75 m and 100 % above
# 2.0 m. The cap is where the floor is still the floor, not where the optics run out
SCALE_BOUNDS = (0.7, 1.4)  # a floor scale outside this is not a scale: the bottom is not floor
N_BINS = round(2 * SCAN_HALF_FOV / SCAN_STEP) + 1  # the fan of pepin.depth.depth_to_scan


@dataclass(frozen=True)
class DepthNoise:
    """The network's depth error as a share of the depth, growing with distance:
    ``sigma_rel(E) = rel_at_zero + rel_per_m * E``. On a ray that meets the floor at depth ``E`` a
    relative error ``e`` is a height error of ``camera_height * e`` (similar triangles, see
    :func:`pepin.depth.floor_anchor`), so the floor's band is ``n_sigma`` such errors wide and
    never narrower than ``min_m``. The defaults are the raw floor's measured spread (sd 5.8 cm
    at the camera's 1.23 m over 1-2.5 m of floor, 2026-09-11: 4.7 % of depth)."""

    rel_at_zero: float = 0.03
    rel_per_m: float = 0.01
    n_sigma: float = 2.0
    min_m: float = FLOOR_HEIGHT_TOLERANCE

    def height_band(self, expected: Array, camera_height: float) -> Array:
        """Metres above or below the plane a pixel may stand and be floor, per pixel: the band
        widens with the floor's expected depth ``expected`` (NaN stays NaN)."""
        sigma = self.rel_at_zero + self.rel_per_m * np.asarray(expected, dtype=float)
        band: Array = np.maximum(self.min_m, self.n_sigma * camera_height * sigma)
        return band


DEPTH_NOISE = DepthNoise()  # the default band, as a singleton: a call in a default is a trap


def floor_height(depth: Array, expected: Array, camera_height: float) -> Array:
    """Metres each pixel stands above the floor plane, from its depth and the depth its ray
    would have on the plane: ``camera_height * (1 - depth / expected)``; NaN where either is
    unknown."""
    with np.errstate(invalid="ignore", divide="ignore"):
        height: Array = camera_height * (1.0 - np.asarray(depth, dtype=float) / expected)
    return height


def floor_scale(
    depth: Array,
    expected: Array,
    rows: int = CONTACT_SCALE_ROWS,
    bounds: tuple[float, float] = SCALE_BOUNDS,
) -> float:
    """The frame's floor scale: the median of depth / expected over the bottom ``rows`` rows,
    where the floor is nearest and the network surest. The law fitted on the lidar's beams is
    the room's, and a frame's own scale swings a few per cent around it (0.88-0.90 over 15
    frames at home, 2026-09-10); tested against an unscaled plane the whole floor then stands
    a few centimetres off it and the band eats that margin. 1.0 when ``rows`` is 0, when fewer
    than a tenth of those pixels are known, or when the median leaves ``bounds`` (the bottom is
    not floor then: a body parked against the bumper)."""
    if rows <= 0:
        return 1.0
    d = np.asarray(depth, dtype=float)[-rows:]
    e = np.asarray(expected, dtype=float)[-rows:]
    ok = np.isfinite(d) & np.isfinite(e) & (d > NEAR_M)
    if int(ok.sum()) < 0.1 * d.size:
        return 1.0
    ratio = float(np.median(d[ok] / e[ok]))
    return ratio if bounds[0] <= ratio <= bounds[1] else 1.0


def floor_mask(
    depth: Array, expected: Array, camera_height: float, band: Array | float, scale: float = 1.0
) -> tuple[Mask, Array]:
    """Which pixels are floor: those standing within ``band`` metres (per pixel or one number)
    of the plane, the plane's depth taken ``scale`` times ``expected``. Returns the mask and
    ``q``, the height in units of the band (0 on the plane, 1 at the band's edge, NaN where the
    depth or the plane is unknown)."""
    height = floor_height(depth, np.asarray(expected, dtype=float) * scale, camera_height)
    with np.errstate(invalid="ignore", divide="ignore"):
        q: Array = np.abs(height) / band
    mask: Mask = np.isfinite(q) & (q <= 1.0)
    return mask, q


# ---- the ray through a pixel onto the floor -----------------------------------------------------


def ipm(
    v: Array, u: Array, intr: Intrinsics, cam: CameraPose, up: Array = UP_LEVEL
) -> tuple[Array, Array]:
    """Inverse perspective mapping: the base_link (x, y) where the ray through pixel ``(u, v)``
    (continuous, :func:`pepin.depth.project`'s convention) meets the floor — the plane through
    the wheels' contact perpendicular to ``up`` (level by default, leaning with the cart when
    the accelerometer says so). NaN where the ray never meets it (at and above the horizon).
    The exact inverse of :func:`pepin.depth.project` for a point on the plane; the network's
    scale never enters."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    left = -(u - intr.cx) / intr.fx
    lift = -(v - intr.cy) / intr.fy
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    dx = c + s * lift  # the ray in base_link, per unit optical depth
    dy = left
    dz = -s + c * lift
    n = np.asarray(up, dtype=float) / np.linalg.norm(up)
    n_dot_d = n[0] * dx + n[1] * dy + n[2] * dz
    n_dot_c = n[0] * cam.x + n[1] * cam.y + n[2] * cam.z
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(n_dot_d < -1e-6, -n_dot_c / n_dot_d, np.nan)
    t = np.where(t > 0, t, np.nan)
    return cam.x + t * dx, cam.y + t * dy


def scan_bins(x: Array, y: Array) -> Ints:
    """Which half-degree bearing of :func:`pepin.depth.depth_to_scan`'s fan each base_link point
    falls in; -1 where the point is unknown, behind the camera or outside the fan."""
    xs = np.asarray(x, dtype=float)
    ys = np.asarray(y, dtype=float)
    with np.errstate(invalid="ignore"):
        bearing = np.arctan2(ys, xs)
        in_fan = np.isfinite(xs) & np.isfinite(ys) & (xs > 0.0)
        in_fan &= np.abs(bearing) <= SCAN_HALF_FOV
    index = np.rint((np.where(in_fan, bearing, 0.0) + SCAN_HALF_FOV) / SCAN_STEP)
    out: Ints = np.where(in_fan, index, -1).astype(np.int64)
    return out


@dataclass(frozen=True, eq=False)
class FloorPlane:
    """Everything the floor decides before a frame arrives, for one mount and one lean: per
    pixel the depth its ray would have on the floor (:func:`pepin.depth.floor_depth`), the
    base_link point it lands on, that point's range from the wheels' centre and its bearing bin
    (-1 outside the fan). ``height`` is the camera's perpendicular distance to the plane — the
    lever that turns a relative depth error into a height error. Built once per lean and reused
    by every frame: the trigonometry is a tenth of a second on a 1280x720 image."""

    intr: Intrinsics
    cam: CameraPose
    up: Array
    expected: Array
    x: Array
    y: Array
    range_m: Array
    bin: Ints
    height: float

    @classmethod
    def of(cls, intr: Intrinsics, cam: CameraPose, up: Array = UP_LEVEL) -> FloorPlane:
        """The plane seen by ``cam`` through ``intr`` with the cart leaning along ``up``."""
        unit: Array = np.asarray(up, dtype=float) / float(np.linalg.norm(up))
        rows, cols = np.mgrid[0 : intr.height, 0 : intr.width]
        x, y = ipm(rows.astype(float), cols.astype(float), intr, cam, unit)
        return cls(
            intr=intr,
            cam=cam,
            up=unit,
            expected=floor_depth(intr, cam, unit),
            x=x,
            y=y,
            range_m=np.hypot(x, y),
            bin=scan_bins(x, y),
            height=float(unit @ np.array([cam.x, cam.y, cam.z])),
        )

    def point(self, v: Array, u: Array) -> tuple[Array, Array]:
        """The base_link (x, y) where the ray through the continuous pixel ``(u, v)`` meets the
        plane: a contact half-way between two rows lands between their floor points."""
        return ipm(v, u, self.intr, self.cam, self.up)


# ---- the floor's run up each column ------------------------------------------------------------


class ColumnState(IntEnum):
    """What a column of the picture says about the floor in front of the cart."""

    BLIND = 0  # no floor at the bottom and nothing known there: the body, a dark edge, no depth
    NEAR = 1  # the bottom rows are something known that is not floor: nearer than the bottom row
    UNKNOWN = 2  # the floor ended in unknown depth
    CLEAR = 3  # floor all the way past the range
    CONTACT = 4  # the floor ended on something known


@dataclass(frozen=True, eq=False)
class Columns:
    """Per image column: the boundary row where the floor ends (``v``, continuous, between the
    last floor pixel and the first one above it; NaN when there is no contact), the row of the
    last floor pixel (``top``, -1 without a floor run) and the column's :class:`ColumnState`;
    ``run`` marks, per pixel, the floor the column actually verified from the bottom up; the
    bearings of that floor are the ones a column that reached a verdict may report clear."""

    v: Array
    top: Ints
    state: Ints
    run: Mask

    def count(self, state: ColumnState) -> int:
        """How many columns are in ``state``."""
        return int((self.state == int(state)).sum())


def _windows(mask: Mask, k: int) -> npt.NDArray[np.bool_]:
    """(rows, cols, k): the ``k`` rows centred on each pixel, the border rows repeated."""
    before = k // 2
    padded = np.pad(mask, ((before, k - 1 - before), (0, 0)), mode="edge")
    view: npt.NDArray[np.bool_] = sliding_window_view(padded, k, axis=0)
    return view


def _dilate(mask: Mask, k: int) -> Mask:
    """True where any of the ``k`` rows centred on the pixel is true."""
    out: Mask = np.asarray(_windows(mask, k).any(axis=2), dtype=bool)
    return out


def _erode(mask: Mask, k: int) -> Mask:
    """True where all of the ``k`` rows centred on the pixel are true."""
    out: Mask = np.asarray(_windows(mask, k).all(axis=2), dtype=bool)
    return out


def clean_runs(mask: Mask, k: int) -> Mask:
    """The mask with, along each column, gaps shorter than ``k`` rows inside a run closed and
    runs shorter than ``k`` rows removed (a closing then an opening with a ``k``-row window):
    the network's noise punches holes in the floor and sprinkles floor on a box; neither is
    a contact. The window's border rows are repeated, so a run that reaches the picture's
    bottom edge survives being shorter than ``k``."""
    if k <= 1:
        return mask
    return _dilate(_erode(_erode(_dilate(mask, k), k), k), k)


def contact_columns(
    floor: Mask,
    known: Mask,
    beyond: Mask,
    *,
    min_run: int = CONTACT_MIN_RUN,
    bottom_slack: int = CONTACT_BOTTOM_SLACK,
) -> Columns:
    """Read every column from the bottom up. ``floor`` says which pixels are floor, ``known``
    which carry a depth at all, ``beyond`` which look at floor farther than the scan's range
    (or at no floor: the horizon). After :func:`clean_runs` the floor must begin within
    ``bottom_slack`` rows of the bottom: if it does not, the column is NEAR when most of those
    rows are known non-floor (something stands nearer than the bottom row's floor) and BLIND
    otherwise. The floor's run then ends either on a pixel already past the range (CLEAR), on
    ``min_run`` rows that are mostly known non-floor (CONTACT, ``v`` at the boundary) or on
    unknown depth (UNKNOWN)."""
    h, w = floor.shape
    clean = clean_runs(floor, min_run)
    obstacle = known & ~clean
    flipped = clean[::-1]
    has_floor = flipped.any(axis=0)
    first = flipped.argmax(axis=0)  # rows under the floor's run, counted from the bottom
    idx = np.arange(h)[:, None]
    above = ~flipped & (idx >= first[None, :])
    ends = above.any(axis=0)
    run_end = np.where(ends, above.argmax(axis=0), h)  # from the bottom: first non-floor row
    top = h - run_end  # image row of the run's last floor pixel
    cols = np.arange(w)
    cumulative = np.zeros((h + 1, w), dtype=int)
    cumulative[1:] = np.cumsum(obstacle, axis=0)
    hi = np.clip(top, 0, h)
    lo = np.clip(top - min_run, 0, h)
    n_obstacle = cumulative[hi, cols] - cumulative[lo, cols]
    n_rows = hi - lo
    bottom_known = known[h - bottom_slack :].sum(axis=0)
    bottom_obstacle = obstacle[h - bottom_slack :].sum(axis=0)
    starts = has_floor & (first <= bottom_slack)
    clear = starts & beyond[np.clip(top, 0, h - 1), cols]
    contact = starts & ~clear & (n_rows > 0) & (2 * n_obstacle >= n_rows)
    unknown = starts & ~clear & ~contact
    near = ~starts & (bottom_known > 0) & (2 * bottom_obstacle > bottom_known)
    state = np.full(w, int(ColumnState.BLIND), dtype=np.int64)
    state[near] = int(ColumnState.NEAR)
    state[unknown] = int(ColumnState.UNKNOWN)
    state[clear] = int(ColumnState.CLEAR)
    state[contact] = int(ColumnState.CONTACT)
    v = np.where(contact, top - 0.5, np.nan)
    run: Mask = clean & starts[None, :] & (np.arange(h)[:, None] >= top[None, :])
    return Columns(v=v, top=np.where(has_floor, top, -1).astype(np.int64), state=state, run=run)


# ---- the columns folded into a scan -------------------------------------------------------------


def band_shadow(x: Array, y: Array, band: Array, plane: FloorPlane) -> tuple[Array, Array]:
    """The contact point the band's own width hides. The floor mask lets a pixel be floor while
    it stands within ``band`` metres of the plane, so the last floor pixel on an obstacle's face
    is not at its foot but ``band`` up it, and the ray through that pixel, carried on to the
    plane, lands *past* the foot — 10 % of the range on our mount, an obstacle reported farther
    than it is, which is the dangerous direction for a costmap. For a face standing
    perpendicular to the floor the foot is exact: with ``beta = band / height``, the seen point
    ``P`` and the camera ``C``, it is ``(1 - beta) P + beta C - band * up``. A face that leans
    away (a ramp, a sofa's skirt) is corrected less than fully, never past its foot."""
    beta = band / plane.height
    c, n = plane.cam, plane.up
    return (
        (1.0 - beta) * x + beta * c.x - band * n[0],
        (1.0 - beta) * y + beta * c.y - band * n[1],
    )


def bin_bearings(
    clear_bins: Ints, contact_bins: Ints, contact_range: Array, kth: int = CONTACT_KTH
) -> Array:
    """The bearing grid of :func:`pepin.depth.depth_to_scan` filled in: every bearing a verified
    floor pixel fell in is ``inf`` (the floor was seen along it out to the scan's range), every
    bearing whose ``kth`` nearest contact exists carries that range (fewer contacts than that
    mark nothing: a lone column is noise), and a bearing no pixel reached stays NaN — unknown,
    which the costmap neither marks nor clears. Marks are written last, so a contact always
    beats the floor its own column saw below it."""
    ranges = np.full(N_BINS, np.nan)
    if clear_bins.size:
        ranges[np.unique(clear_bins)] = np.inf
    if contact_bins.size:
        order = np.lexsort((contact_range, contact_bins))
        bins_sorted, rng_sorted = contact_bins[order], contact_range[order]
        starts = np.flatnonzero(np.r_[True, bins_sorted[1:] != bins_sorted[:-1]])
        counts = np.diff(np.r_[starts, bins_sorted.size])
        enough = counts >= kth
        ranges[bins_sorted[starts[enough]]] = rng_sorted[starts[enough] + kth - 1]
    return ranges


@dataclass(frozen=True)
class ContactVerdict:
    """One frame's contact scan in numbers, for the node's report line: the columns by state,
    the contacts' median range, the share of known pixels judged floor, the frame's floor
    scale, and the bearings marked, cleared and unseen."""

    columns: int
    contact: int
    clear: int
    near: int
    blind: int
    unknown: int
    median_range_m: float
    floor_fraction: float
    scale: float
    marked: int
    cleared: int
    unseen: int

    def __str__(self) -> str:
        median = f" (median {self.median_range_m:.2f} m)" if self.contact else ""
        return (
            f"columns: {self.contact} contact{median}, {self.clear} clear, {self.near} near,"
            f" {self.blind} blind, {self.unknown} unknown of {self.columns};"
            f" floor {self.floor_fraction * 100:.0f}% of known pixels, scale {self.scale:.2f};"
            f" bearings {self.marked} marked, {self.cleared} clear, {self.unseen} unseen"
        )


def contact_scan(
    depth: Array,
    plane: FloorPlane,
    *,
    noise: DepthNoise = DEPTH_NOISE,
    scale_rows: int = CONTACT_SCALE_ROWS,
    scale_bounds: tuple[float, float] = SCALE_BOUNDS,
    min_run: int = CONTACT_MIN_RUN,
    bottom_slack: int = CONTACT_BOTTOM_SLACK,
    kth: int = CONTACT_KTH,
    max_range: float = CONTACT_MAX_RANGE,
    shadow: bool = True,
) -> tuple[float, float, Array, ContactVerdict]:
    """The depth image as a planar scan at the floor: for every half-degree of bearing across
    the camera's view, the range to where the floor ends — the contact line of whatever stands
    on it — lifted onto ``plane`` by :func:`ipm` and pulled back by :func:`band_shadow` (off
    with ``shadow``), so the network's scale does not enter the range. ``inf`` where a column
    that reached a verdict saw floor out to ``max_range``, NaN where none could say (no floor at
    the bottom, the floor ending in unknown depth, nothing at that bearing; an obstacle so close
    that its contact is below the picture reads NaN too, and the lidar owns that metre).
    ``scale_rows`` is the frame's floor scale rows (0: off) and ``scale_bounds`` what it may
    swallow — widened, the scan reads a depth image no law has corrected yet. Returns
    (angle_min, angle_increment, ranges, verdict) — the first three ready for a LaserScan like
    :func:`pepin.depth.depth_to_scan`."""
    d = np.asarray(depth, dtype=float)
    with np.errstate(invalid="ignore"):
        beyond: Mask = ~(plane.range_m < max_range)
    scale = floor_scale(d, plane.expected, scale_rows, scale_bounds)
    band = noise.height_band(plane.expected, plane.height)
    floor, _ = floor_mask(d, plane.expected, plane.height, band, scale)
    known: Mask = np.isfinite(d) & (d > NEAR_M)
    columns = contact_columns(floor, known, beyond, min_run=min_run, bottom_slack=bottom_slack)
    cols = np.arange(plane.intr.width, dtype=float)
    x, y = plane.point(columns.v, cols)
    if shadow:
        at_contact = band[np.clip(columns.top, 0, plane.intr.height - 1), cols.astype(int)]
        x, y = band_shadow(x, y, at_contact, plane)
    reach = np.hypot(x, y)
    bins = scan_bins(x, y)
    with np.errstate(invalid="ignore"):
        marks = (columns.state == int(ColumnState.CONTACT)) & (reach < max_range) & (bins >= 0)
    decided = (columns.state == int(ColumnState.CONTACT)) | (
        columns.state == int(ColumnState.CLEAR)
    )
    verified: Mask = columns.run & decided[None, :] & ~beyond
    ranges = bin_bearings(plane.bin[verified], bins[marks], reach[marks], kth)
    seen = reach[np.isfinite(reach)]
    verdict = ContactVerdict(
        columns=plane.intr.width,
        contact=columns.count(ColumnState.CONTACT),
        clear=columns.count(ColumnState.CLEAR),
        near=columns.count(ColumnState.NEAR),
        blind=columns.count(ColumnState.BLIND),
        unknown=columns.count(ColumnState.UNKNOWN),
        median_range_m=float(np.median(seen)) if seen.size else math.nan,
        floor_fraction=float(floor.sum() / max(int(known.sum()), 1)),
        scale=scale,
        marked=int(np.isfinite(ranges).sum()),
        cleared=int(np.isinf(ranges).sum()),
        unseen=int(np.isnan(ranges).sum()),
    )
    return -SCAN_HALF_FOV, SCAN_STEP, ranges, verdict
