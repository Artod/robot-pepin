"""Is the camera turned relative to the lidar, and by how much? — from the two fans alone.

The lidar and the camera see the same room from the same cart: the lidar as a 360-degree fan of
ranges in ``base_link``, the camera as a narrow fan of obstacle ranges (``/depth_scan``) built
from its depth picture and placed into ``base_link`` through the neck's transform. If the neck's
pan reference (the servo tick that means "straight ahead") is wrong by delta, every camera fan
lands rotated by delta about the cart and the same wall is painted in two places.

The estimate is a one-dimensional search, the cheapest thing that can be believed: rotate the
camera fan by every candidate shift, and for each one take the MEDIAN absolute range difference
against the lidar over the bearings where both sensors return something near. Walls agree at the
true shift and disagree everywhere else, and the median ignores the furniture only one of them
can see. The winner is the shift; how far the score climbs one degree away says whether the
minimum is a wall's corner or a puddle (:attr:`YawOffset.depth_m`).

Nothing here knows about ROS or about the neck: a fan is bearings and ranges, and the answer is
degrees. :func:`corrected_pan_reference` turns those degrees into the tick the neck's reference
should hold (config/neck.json, ``reference.pan_ticks``).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

DEPTH_PROBE_DEG = 1.0  # how far from the winner the minimum's depth is read
DEFAULT_MAX_RANGE_M = 4.0  # beyond this the camera's fan is guesswork, not a wall
DEFAULT_SEARCH_DEG = 8.0  # the neck cannot be further off than this and still look forward
DEFAULT_STEP_DEG = 0.1
DEFAULT_GRID_DEG = 0.5  # the cell a lidar fan is medianed over time in
MAX_GAP_DEG = 2.0  # bearings further apart than this are not one surface: no interpolation
MAX_JUMP_M = 0.25  # a range step this big between neighbours is an edge: do not interpolate
SHARP_DEPTH_M = 0.01  # a minimum that climbs less than this in a degree did not measure a yaw


@dataclass(frozen=True, eq=False)
class Fan:
    """One sweep of ranges: ``bearings_rad`` CCW from the cart's nose, ``ranges_m`` beside it.

    Bearings are sorted ascending and may hold gaps; a range may be NaN where the sensor
    returned nothing. Both arrays are the same length. This is what both sensors reduce to
    before anything here compares them.
    """

    bearings_rad: Array
    ranges_m: Array

    def __post_init__(self) -> None:
        if self.bearings_rad.shape != self.ranges_m.shape:
            raise ValueError("a fan's bearings and ranges must be the same length")

    @property
    def finite(self) -> int:
        """How many bearings of this fan actually returned a range."""
        return int(np.count_nonzero(np.isfinite(self.ranges_m)))


@dataclass(frozen=True, eq=False)
class YawOffset:
    """How far the camera's fan must be turned CCW to sit on the lidar's, and how sure that is.

    ``shift_deg`` is the answer (positive: the camera fan must rotate counter-clockwise to land
    on the lidar's, because the camera really looks that much further COUNTER-CLOCKWISE — to the
    cart's left — than the projection assumed). ``score_m`` is the median
    absolute range difference there, ``depth_m`` how much that median rises one degree away (a
    sharp minimum is a wall seen by both; a flat one means the fans did not constrain the yaw),
    ``bearings`` how many bearings the winning score was taken over, ``scale`` the camera range
    scale divided out before scoring (1.0 when ``scale_free`` was off).
    """

    shift_deg: float
    score_m: float
    depth_m: float
    bearings: int
    scale: float
    scale_free: bool
    shifts_deg: Array
    scores_m: Array

    @property
    def sharp(self) -> bool:
        """Whether the minimum stands out enough to act on (:data:`SHARP_DEPTH_M` of climb one
        degree away, over at least 20 bearings) — a flat score curve fits any yaw equally.

        Necessary, not sufficient: it catches a room that does not constrain the yaw, NOT a
        camera whose ranges are wrong by a bearing-dependent factor. A ratio that slides by
        0.002 per degree across the fan, with no rotation whatever, produced a 6 deg shift with
        a 1.2 cm/deg minimum — sharp and false (scratch/fan_yaw_confounds.py, 2026-09-15). Read
        :func:`range_bias`'s ``slope_per_deg`` beside this: under a slope of a few thousandths
        the shift is the law's, not the neck's."""
        return self.depth_m >= SHARP_DEPTH_M and self.bearings >= 20


@dataclass(frozen=True, eq=False)
class RangeBias:
    """How the camera's ranges scale against the lidar's at the same bearings.

    ``ratio`` is the median camera/lidar range: 1.0 means the depth law is in metres the lidar
    agrees with. ``slope_per_deg`` is how that ratio tilts across bearing — a constant ratio
    (slope near zero) is a pure scale error of the depth law, a tilting one means the camera
    points somewhere else in pitch or pan than the transform says, so the fan's beams cut the
    room at a different height on one side than on the other. ``bearings`` is the sample size.
    """

    ratio: float
    slope_per_deg: float
    bearings: int


def bearing_grid(step_deg: float = DEFAULT_GRID_DEG) -> Array:
    """A full-circle grid of bearings in radians, ``step_deg`` apart, starting at -180 deg."""
    return np.asarray(np.radians(np.arange(-180.0, 180.0, step_deg)), dtype=np.float64)


def median_fan(sweeps: Sequence[tuple[Array, Array]], grid_rad: Array) -> Fan:
    """Many sweeps of one sensor reduced to one fan on ``grid_rad``: per-cell nearest return,
    medianed over time.

    Each sweep is (bearings, ranges) with only its valid returns. Within one sweep the NEAREST
    return in a cell wins (an obstacle is where its closest point is), across sweeps the median
    (a flicker in one frame does not move a wall). Cells no sweep lit stay NaN.
    """
    if len(grid_rad) < 2:
        raise ValueError("a bearing grid needs at least two cells")
    step = float(grid_rad[1] - grid_rad[0])
    stack = np.full((len(sweeps), len(grid_rad)), np.nan, dtype=np.float64)
    for row, (sweep_bearings, sweep_ranges) in enumerate(sweeps):
        ok = np.isfinite(sweep_bearings) & np.isfinite(sweep_ranges)
        cells = np.round((sweep_bearings[ok] - grid_rad[0]) / step).astype(int) % len(grid_rad)
        np.fmin.at(stack[row], cells, sweep_ranges[ok])  # nearest return per cell, NaN yields
    out = np.full(len(grid_rad), np.nan, dtype=np.float64)
    lit = np.any(np.isfinite(stack), axis=0) if len(sweeps) else np.zeros(len(grid_rad), bool)
    if np.any(lit):  # a median over an all-NaN cell is not asked for, so nothing warns
        out[lit] = np.nanmedian(stack[:, lit], axis=0)
    return Fan(np.asarray(grid_rad, dtype=np.float64), out)


def sample_fan(
    fan: Fan,
    query_rad: Array,
    *,
    max_gap_deg: float = MAX_GAP_DEG,
    max_jump_m: float = MAX_JUMP_M,
) -> Array:
    """The fan's range at each queried bearing, linearly interpolated between neighbours.

    A bearing gets NaN unless it lies between two returns that are close in angle
    (``max_gap_deg``) and close in range (``max_jump_m``) — across a doorway or a table's edge
    the interpolated "surface" does not exist, and a comparison there would be noise scored as
    signal. Queries outside the fan's span are NaN.
    """
    ok = np.isfinite(fan.ranges_m) & np.isfinite(fan.bearings_rad)
    b, r = fan.bearings_rad[ok], fan.ranges_m[ok]
    out = np.full(len(query_rad), np.nan, dtype=np.float64)
    if len(b) < 2:
        return out
    order = np.argsort(b)
    b, r = b[order], r[order]
    right = np.searchsorted(b, query_rad)
    inside = (right > 0) & (right < len(b))
    idx = np.clip(right, 1, len(b) - 1)
    lo, hi = idx - 1, idx
    span = b[hi] - b[lo]
    jump = np.abs(r[hi] - r[lo])
    usable = inside & (span <= math.radians(max_gap_deg)) & (jump <= max_jump_m)
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(span > 0.0, (query_rad - b[lo]) / np.where(span > 0.0, span, 1.0), 0.0)
    out[usable] = (r[lo] + t * (r[hi] - r[lo]))[usable]
    return out


def _score(
    lidar: Fan, camera: Fan, shift_deg: float, max_range_m: float, scale_free: bool
) -> tuple[float, int, float]:
    """Median |camera - lidar| range difference with the camera fan turned CCW by ``shift_deg``:
    the score in metres, the bearings it was taken over, and the camera scale divided out
    (1.0 with ``scale_free`` off). NaN, 0, NaN when the fans do not overlap there."""
    cam_ok = np.isfinite(camera.ranges_m) & (camera.ranges_m < max_range_m)
    bearings, ranges = camera.bearings_rad[cam_ok], camera.ranges_m[cam_ok]
    if not len(bearings):
        return math.nan, 0, math.nan
    # A camera beam drawn at bearing b really looks at b + shift; compare it with the lidar there.
    against = sample_fan(lidar, bearings + math.radians(shift_deg))
    both = np.isfinite(against) & (against > 0.05) & (against < max_range_m)
    if not np.any(both):
        return math.nan, 0, math.nan
    near, far = ranges[both], against[both]
    scale = float(np.median(near / far)) if scale_free else 1.0
    if not (scale > 0.0):
        return math.nan, int(np.count_nonzero(both)), math.nan
    return float(np.median(np.abs(near / scale - far))), int(np.count_nonzero(both)), scale


def estimate_yaw_offset(
    lidar: Fan,
    camera: Fan,
    *,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
    search_deg: float = DEFAULT_SEARCH_DEG,
    step_deg: float = DEFAULT_STEP_DEG,
    scale_free: bool = True,
) -> YawOffset:
    """The yaw the camera's fan must be turned CCW by to agree with the lidar's, in degrees.

    Scores every shift in +-``search_deg`` at ``step_deg`` by the median absolute range
    difference over the bearings where both fans return under ``max_range_m``, and returns the
    best one with the minimum's depth and the bearing count (see :class:`YawOffset`).

    ``scale_free`` (the switch, on by default) divides the camera's ranges by their median
    ratio to the lidar's at each shift before scoring, so the answer is about SHAPE only. It
    exists because it is the difference between an answer and a shrug: on 2026-09-15 the depth
    law was reading 0.62 of the truth, and an absolute-difference score slid to the edge of the
    search window with a flat minimum — under a scale error every shift is wrong by the scale,
    and the yaw is invisible. Off, the old absolute score is back, which is the right one when
    the depth law is known good and its scale is itself being checked (:func:`range_bias`).

    Raises ValueError when no shift had any overlapping bearing at all — the fans do not see one
    room.
    """
    shifts = np.arange(-search_deg, search_deg + 0.5 * step_deg, step_deg, dtype=np.float64)
    scored = [_score(lidar, camera, float(s), max_range_m, scale_free) for s in shifts]
    scores = np.array([s for s, _, _ in scored], dtype=np.float64)
    counts = np.array([n for _, n, _ in scored], dtype=np.int64)
    scales = np.array([k for _, _, k in scored], dtype=np.float64)
    if not np.any(np.isfinite(scores)):
        raise ValueError("no yaw shift put the two fans on a common bearing")
    best = int(np.nanargmin(scores))
    away = np.abs(shifts - shifts[best]) >= DEPTH_PROBE_DEG
    nearby = scores[away & np.isfinite(scores)]
    depth = float(np.min(nearby) - scores[best]) if len(nearby) else 0.0
    return YawOffset(
        shift_deg=float(shifts[best]),
        score_m=float(scores[best]),
        depth_m=depth,
        bearings=int(counts[best]),
        scale=float(scales[best]),
        scale_free=scale_free,
        shifts_deg=shifts,
        scores_m=scores,
    )


def range_bias(
    lidar: Fan,
    camera: Fan,
    *,
    shift_deg: float = 0.0,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
) -> RangeBias:
    """How the camera's ranges compare with the lidar's at the same bearings, once the fan is
    turned by ``shift_deg``: the median ratio and how it tilts across bearing.

    Read it as the second half of the answer the yaw cannot give: a ratio away from 1.0 with a
    flat slope is the depth law's scale, a ratio that slides across the fan is the camera
    pointing elsewhere (pitch, or a pan the yaw search could not absorb). Returns NaNs with
    zero bearings when the fans do not overlap.
    """
    cam_ok = np.isfinite(camera.ranges_m) & (camera.ranges_m < max_range_m)
    bearings, ranges = camera.bearings_rad[cam_ok], camera.ranges_m[cam_ok]
    if not len(bearings):
        return RangeBias(math.nan, math.nan, 0)
    against = sample_fan(lidar, bearings + math.radians(shift_deg))
    both = np.isfinite(against) & (against > 0.05) & (against < max_range_m)
    if np.count_nonzero(both) < 2:
        return RangeBias(math.nan, math.nan, int(np.count_nonzero(both)))
    ratio = ranges[both] / against[both]
    degrees = np.degrees(bearings[both])
    slope, _intercept = np.polyfit(degrees, ratio, 1)
    return RangeBias(float(np.median(ratio)), float(slope), int(np.count_nonzero(both)))


@dataclass(frozen=True, eq=False)
class MountFit:
    """A pan and a roll of the camera mount fitted to yaw shifts measured at several head
    pitches, and what the two of them fail to explain.

    ``pan_deg`` turns the camera about the cart's vertical (the neck's own axis) and moves every
    fan by the same angle whatever the head's pitch. ``roll_deg`` turns it about the optical
    axis, which is tilted ``pitch`` below the horizon, so its vertical component — the only one
    that shows as a yaw — is ``roll * sin(pitch)``: a roll hides at a level head and grows as
    the head looks down. ``residuals_deg`` is measured minus fitted, one per sample, and
    ``rms_deg`` their spread: bigger than the measurement's own repeatability means no rigid
    mount produced these numbers and something bearing-dependent did.
    """

    pan_deg: float
    roll_deg: float
    pitches_deg: Array
    residuals_deg: Array
    rms_deg: float

    def explains(self, spread_deg: float) -> bool:
        """Whether a rigid mount accounts for the shifts within the measurement's own spread."""
        return self.rms_deg <= spread_deg


def mount_yaw_shift(pan_deg: float, roll_deg: float, pitch_deg: float) -> float:
    """The yaw a camera fan lands at when the mount is turned ``pan_deg`` CCW about the cart's
    vertical and rolled ``roll_deg`` about its own optical axis, the head looking ``pitch_deg``
    down: ``pan + roll * sin(pitch)`` degrees.

    The roll term is the projection of a rotation about the optical axis onto base_link's z: the
    optical axis points ``(cos pitch, 0, -sin pitch)``, so a roll of rho about it carries
    ``rho * sin(pitch)`` of yaw (and ``rho * cos(pitch)`` of image rotation, which no fan sees).
    First order in the angles, which is all a few degrees need. The consequence worth keeping:
    the shift a rigid mount produces is MONOTONIC in pitch — a set of shifts that rises and
    falls again cannot be one, whatever pan and roll are chosen (:class:`MountFit`).
    """
    return pan_deg + roll_deg * math.sin(math.radians(pitch_deg))


def fit_mount(samples: Sequence[tuple[float, float]]) -> MountFit:
    """Pan and roll fitted jointly to (head pitch in degrees, measured yaw shift in degrees)
    samples by least squares on ``pan + roll * sin(pitch)``.

    Needs at least two pitches, and two that differ: a single pitch cannot tell a pan from a
    roll, they are the same degree there. Returns the two angles with the residual each sample
    is left with (:class:`MountFit`).
    """
    pitches = np.asarray([p for p, _ in samples], dtype=np.float64)
    shifts = np.asarray([s for _, s in samples], dtype=np.float64)
    if len(pitches) < 2:
        raise ValueError("a pan and a roll need at least two head pitches")
    basis = np.stack([np.ones_like(pitches), np.sin(np.radians(pitches))], axis=1)
    if np.ptp(basis[:, 1]) < 1e-9:
        raise ValueError("every sample is at the same pitch: a pan and a roll are one number there")
    (pan, roll), *_ = np.linalg.lstsq(basis, shifts, rcond=None)
    residuals = shifts - basis @ np.array([pan, roll])
    return MountFit(
        pan_deg=float(pan),
        roll_deg=float(roll),
        pitches_deg=pitches,
        residuals_deg=residuals,
        rms_deg=float(np.sqrt(np.mean(residuals**2))),
    )


def synthetic_camera_fan(
    lidar: Fan,
    *,
    shift_deg: float = 0.0,
    scale: float = 1.0,
    slope_per_deg: float = 0.0,
    half_fov_deg: float = 40.0,
    step_deg: float = 0.5,
) -> Fan:
    """A camera fan built from a lidar fan with a KNOWN yaw and a known range bias: the truth
    the estimator can be held against.

    The camera's beam drawn at bearing ``b`` really looks at ``b + shift_deg`` (the same
    convention :class:`YawOffset` answers in), and the range it reports there is the lidar's
    times ``scale + slope_per_deg * b`` — a scale error of the depth law, and a scale that
    slides across the picture, which is what the live probe measures as
    :attr:`RangeBias.slope_per_deg`. The fan spans ``+-half_fov_deg`` like
    :func:`pepin.depth.depth_to_scan`'s.

    It exists because the estimator's answer cannot be read without it: with the camera not
    turned by a single degree, a slope of a few thousandths per degree moves the minimum by
    several degrees and passes the sharpness test (:func:`slope_artefact`).
    """
    bearings = np.radians(np.arange(-half_fov_deg, half_fov_deg + 0.5 * step_deg, step_deg))
    looked_at = bearings + math.radians(shift_deg)
    ranges = sample_fan(lidar, looked_at) * (scale + slope_per_deg * np.degrees(bearings))
    return Fan(np.asarray(bearings, dtype=np.float64), np.asarray(ranges, dtype=np.float64))


def slope_artefact(
    lidar: Fan,
    slope_per_deg: float,
    *,
    scale: float = 1.0,
    shift_deg: float = 0.0,
    search_deg: float = DEFAULT_SEARCH_DEG,
    scale_free: bool = True,
) -> YawOffset:
    """What this estimator reads on a camera turned by ``shift_deg`` (0 by default: not turned
    at all) whose ranges carry ``slope_per_deg`` of bearing-dependent scale, in THIS room.

    The difference between its ``shift_deg`` and the ``shift_deg`` asked for is the artefact —
    degrees of yaw the range bias alone invents — and it depends on the room, because it is the
    room's ranges that the sliding scale distorts. Read it beside any measured shift: a shift
    smaller than the artefact its own :func:`range_bias` slope implies carries no information
    about the mount.
    """
    fake = synthetic_camera_fan(
        lidar, shift_deg=shift_deg, scale=scale, slope_per_deg=slope_per_deg
    )
    return estimate_yaw_offset(lidar, fake, search_deg=search_deg, scale_free=scale_free)


def corrected_pan_reference(
    reference_ticks: int, pan_error_deg: float, pan_sign: int, deg_per_tick: float
) -> int:
    """The pan reference tick that puts the neck's believed heading on the measured one.

    ``pan_error_deg`` is the truth MINUS what the model believes (CCW positive: positive means
    the head really points further left than /neck/state says). Where the camera fan is placed
    into base_link THROUGH the neck's transform, that is exactly :attr:`YawOffset.shift_deg`.
    Where it is not — /depth_scan today is folded onto the floor as if the head looked along the
    cart's x, whatever the pan (depth_stream ``_as_scan``, counted
    in its report line as "head panned N frames (projected as if not)") — the shift measures the
    camera's TRUE heading, and the error is ``shift - the pan /neck/state believes``.

    The neck reads ``pan = pan_sign * (ticks - reference) * deg_per_tick``, so moving the
    reference moves the believed heading by the opposite of ``pan_sign``: closing an error of
    ``pan_error_deg`` takes ``-pan_error_deg / (pan_sign * deg_per_tick)`` ticks. Nothing on the
    robot moves — the number only says where "straight ahead" really is.
    """
    if pan_sign not in (-1, 1):
        raise ValueError("pan_sign is +1 or -1")
    if deg_per_tick <= 0.0:
        raise ValueError("deg_per_tick must be positive")
    return reference_ticks - round(pan_error_deg / (pan_sign * deg_per_tick))


# ---- the pan from bearings alone, with no depth in it ------------------------------------------
EDGE_MIN_RELATIVE = 3.0  # a column is an edge at this many times the picture's median energy
EDGE_MIN_SEPARATION_PX = 12  # two peaks nearer than this are one edge seen twice
CORNER_MIN_JUMP_M = 0.15  # a step between neighbouring beams this big is an edge of something
ASSOCIATION_MAX_GAP_DEG = 6.0  # an edge and a corner further apart than this are not the same thing


@dataclass(frozen=True, eq=False)
class PanByBearings:
    """The camera's pan error measured from bearings only: no range, no depth law, no scale.

    ``pan_error_deg`` is the truth MINUS what the model believes (CCW positive), ready for
    :func:`corrected_pan_reference`. ``spread_deg`` is the median absolute deviation of the
    pairs about it — the instrument's own repeatability, not a fitted uncertainty. ``pairs`` is
    how many edge/corner associations it was taken over and ``frames`` how many pictures they
    came from.
    """

    pan_error_deg: float
    spread_deg: float
    pairs: int
    frames: int


def column_bearing(columns: Array, fx: float, cx: float, pan_rad: float = 0.0) -> Array:
    """The bearing in base_link of image columns: ``atan((cx - u) / fx) + pan``, radians CCW.

    Left of the principal point is a positive bearing, the convention
    :func:`pepin.depth.depth_to_scan` folds its fan in. ``pan_rad`` is where the neck believes
    the head points (``/neck/state``, or TF's ``base_link <- camera_link``), so the answer is in
    the cart's frame and not the camera's. Only the calibrated intrinsics enter: a bearing is a
    direction, and no depth, scale or law can move it.
    """
    return np.arctan((cx - np.asarray(columns, dtype=float)) / fx) + pan_rad


def edge_columns(
    energy: Array,
    *,
    min_relative: float = EDGE_MIN_RELATIVE,
    min_separation_px: int = EDGE_MIN_SEPARATION_PX,
) -> Array:
    """The columns of a picture that hold a vertical edge, from one column-energy profile (the
    absolute horizontal gradient summed down the image, or a Hough vertical-line count).

    A column is an edge when it stands ``min_relative`` times the profile's median and is the
    strongest within ``min_separation_px`` of itself — a door frame is two edges, not forty.
    Returns the columns, ascending. The profile, not the image, is the argument: the picture's
    filtering belongs to whoever has OpenCV, and this stays portable and testable.
    """
    e = np.asarray(energy, dtype=float)
    median = float(np.median(e[np.isfinite(e)])) if np.any(np.isfinite(e)) else 0.0
    if not (median > 0.0):
        return np.zeros(0, dtype=np.int64)
    strong = np.flatnonzero(np.isfinite(e) & (e >= min_relative * median))
    kept: list[int] = []
    for column in strong[np.argsort(-e[strong])]:
        if all(abs(int(column) - k) >= min_separation_px for k in kept):
            kept.append(int(column))
    return np.asarray(sorted(kept), dtype=np.int64)


def scan_corners(
    bearings_rad: Array,
    ranges_m: Array,
    *,
    min_jump_m: float = CORNER_MIN_JUMP_M,
    max_range_m: float = DEFAULT_MAX_RANGE_M,
) -> Array:
    """The bearings at which the lidar's range steps: a door jamb, a cupboard's corner, the edge
    of anything that occludes what is behind it.

    A step of at least ``min_jump_m`` between neighbouring beams is an edge, and the bearing
    returned is the NEARER of the two beams — the occluding side, which is the side the picture
    also draws a line at. Beams beyond ``max_range_m`` or missing do not make edges.
    """
    b = np.asarray(bearings_rad, dtype=float)
    r = np.asarray(ranges_m, dtype=float)
    ok = np.isfinite(b) & np.isfinite(r) & (r > 0.05) & (r < max_range_m)
    b, r = b[ok], r[ok]
    if len(b) < 2:
        return np.zeros(0, dtype=np.float64)
    order = np.argsort(b)
    b, r = b[order], r[order]
    step = np.abs(np.diff(r))
    at = np.flatnonzero(step >= min_jump_m)
    nearer = np.where(r[at] <= r[at + 1], b[at], b[at + 1])
    return np.asarray(nearer, dtype=np.float64)


def associate_bearings(
    edges_rad: Array, corners_rad: Array, *, max_gap_deg: float = ASSOCIATION_MAX_GAP_DEG
) -> Array:
    """Each camera edge minus the lidar corner nearest it, in radians, for the pairs that fall
    within ``max_gap_deg`` of each other — the differences, one per associated edge.

    Sign: corner MINUS edge, so a positive difference means the thing really sits further CCW
    than the picture placed it, which is the camera pointing further COUNTER-clockwise than
    believed (the picture was placed with the believed pan, so every edge sits too far clockwise).
    Unassociated edges are dropped, not guessed.
    """
    e = np.asarray(edges_rad, dtype=float)
    c = np.asarray(corners_rad, dtype=float)
    if not len(e) or not len(c):
        return np.zeros(0, dtype=np.float64)
    gaps = c[None, :] - e[:, None]
    nearest = np.argmin(np.abs(gaps), axis=1)
    picked = gaps[np.arange(len(e)), nearest]
    return np.asarray(picked[np.abs(picked) <= math.radians(max_gap_deg)], dtype=np.float64)


def pan_from_bearings(
    per_frame: Sequence[tuple[Array, Array]], *, max_gap_deg: float = ASSOCIATION_MAX_GAP_DEG
) -> PanByBearings:
    """The camera's pan error from a run of frames, each a pair of (edge bearings from the
    picture, corner bearings from the lidar) in base_link radians.

    Every edge is associated with the nearest corner within ``max_gap_deg``, and the MEDIAN of
    all the differences over all the frames is the answer, its median absolute deviation the
    spread (:class:`PanByBearings`). Nothing here has ever seen a range: this is the measurement
    a depth law cannot bias, and the one that settles a pan reference.

    Raises ValueError when no frame produced a single association.
    """
    picked = [associate_bearings(e, c, max_gap_deg=max_gap_deg) for e, c in per_frame]
    used = [p for p in picked if len(p)]
    if not used:
        return PanByBearings(math.nan, math.nan, 0, len(per_frame))
    all_diffs = np.concatenate(used)
    median = float(np.median(all_diffs))
    mad = float(np.median(np.abs(all_diffs - median)))
    return PanByBearings(
        pan_error_deg=math.degrees(median),
        spread_deg=math.degrees(mad),
        pairs=len(all_diffs),
        frames=len(used),
    )
