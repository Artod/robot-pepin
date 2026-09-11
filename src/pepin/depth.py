"""Metric depth from one camera, with the lidar setting the law.

A monocular depth network gives the shape of the scene but not its size, and the size it gets
wrong is not one number: on our lens Depth Anything V2 (metric, indoor) put a coffee table one
metre away at 2.1 m and the far wall further out still. The lidar knows distances exactly, but
only in its own plane, 20 cm above the floor (``config/lidar.json``). Projected into the image,
the scan's beams name the true depth at a few hundred pixels; those (network, true) pairs,
pooled over minutes of frames so they span the room's depths, fit an affine law in inverse
depth, 1 / z = a / D + b — what a relative-depth network is built to be right up to — and the
law corrects the whole image (:class:`AffineScale`, :func:`fit_affine`, :func:`apply_affine`).
Two more corrections act where they measure: pixels on an object's edge carry a depth blurred
between the object and what is behind it and are dropped (:func:`edge_mask`); pixels within a
few centimetres of the floor plane snap to it, the plane leaning with the accelerometer
(:func:`floor_depth`, :func:`floor_anchor`, :class:`Tilt`). The result is a depth image the
fusion turns into a 3D model and — folded onto the plane by :func:`depth_to_scan` — a scan the
costmap marks and clears with, so a table top stops the cart the way a wall does. The law is
saved next to the maps (:func:`save_law`) so a restart begins from it, not from the raw network.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]

MIN_SAMPLES = 20  # lidar points seen in the image before a frame's beams join the pool
NEAR_M = 0.20  # a depth closer than this is in front of the lens: not a scene point


@dataclass(frozen=True)
class Intrinsics:
    """A pinhole camera: focal lengths and principal point in pixels, image size."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_camera_info(cls, k: Sequence[float], width: int, height: int) -> Intrinsics:
        """The intrinsics carried by a CameraInfo: its row-major 3x3 K and the image size."""
        return cls(float(k[0]), float(k[4]), float(k[2]), float(k[5]), int(width), int(height))


def decode_rgb(data: bytes, height: int, width: int, encoding: str) -> npt.NDArray[np.uint8] | None:
    """A ROS Image's bytes as an (h, w, 3) RGB array, or ``None`` for an encoding we cannot read."""
    if encoding not in ("bgr8", "rgb8"):
        return None
    px = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
    out: npt.NDArray[np.uint8] = np.ascontiguousarray(px[:, :, ::-1] if encoding == "bgr8" else px)
    return out


@dataclass(frozen=True)
class CameraPose:
    """Where the camera_link sits in base_link: metres forward, left, up, and its pitch in
    radians (positive tilts the view down, the ROS convention)."""

    x: float
    y: float
    z: float
    pitch: float = 0.0


def scan_points(ranges: Array, angle_min: float, angle_increment: float, range_max: float) -> Array:
    """The valid beams of a scan as (n, 2) x-y points in the lidar's frame."""
    r = np.asarray(ranges, dtype=float)
    angles = angle_min + angle_increment * np.arange(r.size)
    keep = np.isfinite(r) & (r > 0.0) & (r <= range_max)
    return np.stack([r[keep] * np.cos(angles[keep]), r[keep] * np.sin(angles[keep])], axis=1)


def to_base(points_xy: Array, rotation: Array, translation: Array) -> Array:
    """Lidar-plane points lifted to (n, 3) base_link points through the lidar's static mount
    (3x3 rotation and translation of the lidar frame in base_link)."""
    xyz = np.concatenate([points_xy, np.zeros((points_xy.shape[0], 1))], axis=1)
    lifted: Array = xyz @ rotation.T + translation
    return lifted


def carry(points: Array, rotation: Array, translation: Array) -> Array:
    """(n, 3) points moved by a rigid transform: base_link points seen at the scan's moment
    expressed in base_link at the frame's moment, the cart having turned and moved in between
    (a scan 100 ms older than the frame is 2 degrees stale at 20 deg/s; anchored as is, it put
    the frame's band 2.5 degrees off during every turn, 2026-09-11)."""
    moved: Array = np.asarray(points, dtype=float) @ np.asarray(rotation).T + np.asarray(
        translation
    )
    return moved


def project(points_base: Array, cam: CameraPose, intr: Intrinsics) -> Array:
    """Pixels the base_link points land on: (m, 3) rows of column, row and depth along the
    optical axis, only for points inside the image and in front of the camera."""
    p = np.asarray(points_base, dtype=float) - np.array([cam.x, cam.y, cam.z])
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    # camera_link axes: forward, left, up; the pitch turns forward towards the floor
    forward = c * p[:, 0] - s * p[:, 2]
    left = p[:, 1]
    up = s * p[:, 0] + c * p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.fx * (-left / forward) + intr.cx
        v = intr.fy * (-up / forward) + intr.cy
    keep = (forward > NEAR_M) & (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    return np.stack([u[keep], v[keep], forward[keep]], axis=1)


def rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> Array:
    """The 3x3 rotation of a unit quaternion (x, y, z, w), as TF carries it."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz, wx, wy, wz = qx * qy, qx * qz, qy * qz, qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ]
    )


def quaternion_from_matrix(rot: Array) -> tuple[float, float, float, float]:
    """The unit quaternion (x, y, z, w) of a 3x3 rotation, for TF."""
    m = np.asarray(rot, dtype=float)
    trace = float(m[0, 0] + m[1, 1] + m[2, 2])
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        return (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s, 0.25 * s
    i = int(np.argmax([m[0, 0], m[1, 1], m[2, 2]]))
    j, k = (i + 1) % 3, (i + 2) % 3
    s = math.sqrt(1.0 + m[i, i] - m[j, j] - m[k, k]) * 2
    q = [0.0, 0.0, 0.0, 0.0]
    q[i] = 0.25 * s
    q[j] = (m[j, i] + m[i, j]) / s
    q[k] = (m[k, i] + m[i, k]) / s
    q[3] = (m[k, j] - m[j, k]) / s
    return q[0], q[1], q[2], q[3]


def invert(rotation: Array, translation: Array) -> tuple[Array, Array]:
    """The inverse rigid transform: (R^T, -R^T t)."""
    rt = np.asarray(rotation, dtype=float).T
    back: Array = -(rt @ np.asarray(translation, dtype=float))
    return rt, back


# ---- which scan anchors a frame ---------------------------------------------------------------
SCAN_WINDOW_S = 1.0  # scans kept for the pairing: the network runs 0.2-0.3 s, a stalled laptop more
SCAN_MAX_AGE_S = 0.5  # a scan farther than this from the frame's exposure does not judge it


def nearest_stamp(
    stamps: Sequence[float], target: float, max_age_s: float = SCAN_MAX_AGE_S
) -> int | None:
    """The index of the stamp nearest ``target`` (s), or ``None`` when none is within
    ``max_age_s``. The scan that anchors a frame is the one taken nearest the frame's exposure,
    not the newest: the network answers 0.2-0.3 s after the picture, so the newest scan is
    systematically younger than it, and a half-second stall of the laptop dropped every verdict."""
    if not stamps:
        return None
    ages = np.abs(np.asarray(stamps, dtype=float) - target)
    i = int(np.argmin(ages))
    return i if float(ages[i]) <= max_age_s else None


# ---- the depth folded onto the plane ----------------------------------------------------------
SCAN_MIN_Z_M = 0.15  # the scan marks from this height: the network's floor is ~5 % noisy in
# depth, and a relative depth error e is a height error of 1.23 m * e along every ray, so an 8 cm
# floor margin sat a 6.5 % depth error away from marking the open floor as a wall (a review probe,
# 2026-09-11); 15 cm is a 12 % margin, the lidar's own plane is at 20 cm, tops and seats stay in
SCAN_HALF_FOV = math.radians(40.0)  # the tilted camera's bearings reach a little past its lens
SCAN_STEP = math.radians(0.5)
SCAN_KTH = 3  # the k-th nearest point of a bearing: a flying pixel at an edge does not mark


def depth_to_scan(
    depth: Array,
    intr: Intrinsics,
    cam: CameraPose,
    *,
    stride: int = 4,
    min_z: float = SCAN_MIN_Z_M,
    max_z: float = 1.30,
    max_range: float = 3.0,
) -> tuple[float, float, Array]:
    """The depth image as a planar scan in base_link: for every half-degree of bearing across the
    camera's view, the range to the nearest thing standing between ``min_z`` and ``max_z`` above
    the floor (a table top, a seat, a leg). A bearing where finite pixels were seen but none of
    them stands in the band within ``max_range`` is ``inf``: clear that far. A bearing with no
    finite pixel at all is NaN: unknown, and the costmap neither marks nor clears it — with
    ``inf_is_valid`` an ``inf`` clears every cell out to the raytrace range, so a NaN-heavy frame
    (a region the law cannot place, a dark image) must not wipe a table top off the costmap.
    Returns (angle_min, angle_increment, ranges), ready for a LaserScan. Every ``stride``-th
    pixel is used; the k-th nearest point per bearing marks, a flying pixel does not."""
    d = np.asarray(depth, dtype=float)[::stride, ::stride]
    rows, cols = np.mgrid[0 : d.shape[0], 0 : d.shape[1]]
    u = cols * stride + 0.5
    v = rows * stride + 0.5
    ok = np.isfinite(d) & (d > NEAR_M)
    z_opt = d[ok]
    left = -(u[ok] - intr.cx) / intr.fx * z_opt
    up = -(v[ok] - intr.cy) / intr.fy * z_opt
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    px = c * z_opt + s * up + cam.x
    py = left + cam.y
    pz = -s * z_opt + c * up + cam.z
    rng = np.hypot(px, py)
    bearing = np.arctan2(py, px)
    n_bins = round(2 * SCAN_HALF_FOV / SCAN_STEP) + 1
    ranges = np.full(n_bins, np.nan)
    seen = (px > 0.0) & (np.abs(bearing) <= SCAN_HALF_FOV)
    bins = np.rint((bearing + SCAN_HALF_FOV) / SCAN_STEP).astype(int)
    ranges[np.unique(bins[seen])] = np.inf  # something was seen along the bearing: clear
    marks = seen & (pz > min_z) & (pz < max_z) & (rng <= max_range)
    order = np.lexsort((rng[marks], bins[marks]))
    bins_sorted, rng_sorted = bins[marks][order], rng[marks][order]
    starts = np.flatnonzero(np.r_[True, bins_sorted[1:] != bins_sorted[:-1]])
    counts = np.diff(np.r_[starts, bins_sorted.size])
    enough = counts >= SCAN_KTH
    ranges[bins_sorted[starts[enough]]] = rng_sorted[starts[enough] + SCAN_KTH - 1]
    return -SCAN_HALF_FOV, SCAN_STEP, ranges


# ---- the floor as a second anchor -------------------------------------------------------------
FLOOR_HEIGHT_TOLERANCE = 0.04  # metres above or below the floor plane a pixel may sit and be floor
UP_LEVEL: Array = np.array([0.0, 0.0, 1.0])
GRAVITY = 9.81
TILT_TAU_S = 10.0  # the accelerometer's up vector, low-passed: a push or a bump is not a slope
TILT_GATE_DEG = 1.0  # a sample leaning more than this from the running up is a push, not gravity


def floor_depth(intr: Intrinsics, cam: CameraPose, up: Array = UP_LEVEL) -> Array:
    """The depth every pixel would have if its ray ended on the floor: the plane through
    base_link's origin (the wheels' contact) perpendicular to ``up`` (gravity's opposite in
    base_link; level by default, tilted when the cart stands on a slipper). NaN at and above
    the horizon, where the ray never meets the floor."""
    rows, cols = np.mgrid[0 : intr.height, 0 : intr.width]
    left = -(cols - intr.cx) / intr.fx
    lift = -(rows - intr.cy) / intr.fy
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    # camera rays (forward 1, left, up) into base_link through the mount's pitch
    dx = c + s * lift
    dy = left
    dz = -s + c * lift
    n = np.asarray(up, dtype=float) / np.linalg.norm(up)
    n_dot_d = n[0] * dx + n[1] * dy + n[2] * dz
    n_dot_c = n[0] * cam.x + n[1] * cam.y + n[2] * cam.z
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(n_dot_d < -1e-6, -n_dot_c / n_dot_d, np.nan)
    expected: Array = np.where(t > 0, t, np.nan)  # depth along the axis: the ray's forward is 1
    return expected


def floor_anchor(
    depth: Array, expected: Array, camera_height: float, tolerance: float = FLOOR_HEIGHT_TOLERANCE
) -> tuple[Array, int]:
    """Pixels within ``tolerance`` metres of the floor plane are set to the floor's exact depth;
    everything else (a sofa, a shoe, a cable, a box) is left to the network. Returns the
    corrected image and how many pixels were anchored.

    The geometry: a ray leaving the camera ``camera_height`` above the plane meets it at depth
    ``E`` along the optical axis; at depth ``d`` on the same ray the point stands
    ``camera_height * (1 - d / E)`` above the plane — similar triangles, the height falls
    linearly from the lens to zero at the floor. So a tolerance on ``d / E`` is a tolerance on
    height that does not depend on range: the earlier 15 % of depth was 18 cm at the 1.23 m
    camera, and shoes, cables and low boxes snapped into the floor at every range and vanished
    from the scan. The test is on the height itself now (the camera's height above a leaning
    plane differs from its mount height by the cosine of a few degrees: ignored)."""
    d = np.asarray(depth, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        height = camera_height * (1.0 - d / expected)
        is_floor = np.isfinite(height) & (np.abs(height) < tolerance)
    out = d.copy()
    out[is_floor] = expected[is_floor]
    return out, int(is_floor.sum())


class Tilt:
    """Which way is up, from the accelerometer: the IMU's reading turned into base_link through
    its mount and low-passed. At rest the chip reads +g along up; while the cart accelerates the
    reading leans, so three gates stand before the filter: a non-finite sample is ignored, a
    norm away from 1 g (a bump, braking) is ignored, and a sample leaning more than
    TILT_GATE_DEG from the running up is a push (0.3 m/s^2 leans the apparent gravity 1.8 deg,
    which the norm cannot see) — unless the lean outlasts the time constant, which no push does:
    then the cart stands on a slope and the up vector is re-seeded from the sample."""

    def __init__(self, imu_to_base: Array, tau_s: float = TILT_TAU_S) -> None:
        self._rotation = np.asarray(imu_to_base, dtype=float)
        self._tau = tau_s
        self._up: Array = UP_LEVEL.copy()
        self._seeded = False
        self._last_t: float | None = None
        self._leaning_since: float | None = None

    def observe(self, accel_imu: Array, t: float) -> None:
        """Feed one accelerometer reading (m/s^2, the IMU's own axes) at time ``t`` (s)."""
        a = self._rotation @ np.asarray(accel_imu, dtype=float)
        if not bool(np.all(np.isfinite(a))):
            return  # a NaN would pass the norm gate and poison the up vector for good
        norm = float(np.linalg.norm(a))
        if abs(norm - GRAVITY) > 1.0:  # braking, a bump: not gravity alone
            self._last_t = t
            return
        fresh = a / norm
        if not self._seeded:
            self._up, self._seeded, self._last_t = fresh, True, t
            return
        lean = math.degrees(math.acos(float(np.clip(np.dot(fresh, self._up), -1.0, 1.0))))
        if lean > TILT_GATE_DEG:
            if self._leaning_since is None:
                self._leaning_since = t
            elif t - self._leaning_since > self._tau:
                self._up, self._leaning_since = fresh, None  # a lean that lasts is the floor
            self._last_t = t
            return
        self._leaning_since = None
        k = 1.0 - math.exp(
            -max(t - (self._last_t if self._last_t is not None else t), 0.0) / self._tau
        )
        blended = (1.0 - k) * self._up + k * fresh
        self._up = blended / np.linalg.norm(blended)
        self._last_t = t

    @property
    def up(self) -> Array:
        """The unit vector pointing up, in base_link."""
        return self._up

    @property
    def roll_pitch_deg(self) -> tuple[float, float]:
        """The cart's roll and pitch in degrees (positive: right side down, nose down)."""
        ux, uy, uz = self._up
        # nose down: the world's up leans towards the tail (-x) in the body's axes
        return math.degrees(math.atan2(uy, uz)), math.degrees(math.atan2(-ux, uz))


def imu_mount_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> Array:
    """The 3x3 rotation taking a vector from the IMU's axes to base_link, from the mount's
    roll-pitch-yaw (the ROS convention, like the static transform the board publishes)."""
    r, p, y = (math.radians(a) for a in (roll_deg, pitch_deg, yaw_deg))
    rx = np.array([[1, 0, 0], [0, math.cos(r), -math.sin(r)], [0, math.sin(r), math.cos(r)]])
    ry = np.array([[math.cos(p), 0, math.sin(p)], [0, 1, 0], [-math.sin(p), 0, math.cos(p)]])
    rz = np.array([[math.cos(y), -math.sin(y), 0], [math.sin(y), math.cos(y), 0], [0, 0, 1]])
    rot: Array = rz @ ry @ rx
    return rot


# ---- flying pixels -------------------------------------------------------------------------
EDGE_REL_STEP = 0.08  # a depth that differs from a neighbour by more than this share is an edge


def edge_mask(depth: Array, rel_step: float = EDGE_REL_STEP) -> Mask:
    """Pixels sitting on a depth discontinuity: the network blurs every object edge over a few
    pixels, so the pixels there carry depths between the object and what is behind it and land
    in mid-air when unprojected (in one frame, 21 % of the lidar-height pixels were more than
    15 cm from any lidar return, 2026-09-11). True where the depth differs from a horizontal
    or vertical neighbour by more than ``rel_step`` of itself, or where a neighbour is unknown;
    the border rows and columns are always edges. The test is relative, so the mask is the same
    on the raw depth and on the law-corrected one."""
    d = np.asarray(depth, dtype=float)
    edge = np.zeros(d.shape, dtype=bool)
    with np.errstate(invalid="ignore", divide="ignore"):
        for axis in (0, 1):
            a = np.roll(d, 1, axis=axis)
            b = np.roll(d, -1, axis=axis)
            step = np.maximum(np.abs(d - a), np.abs(d - b)) / d
            edge |= ~np.isfinite(step) | (step > rel_step)
    edge[0, :] = edge[-1, :] = True  # np.roll wraps around: the border rows and columns are edges
    edge[:, 0] = edge[:, -1] = True
    return edge


def drop_edges(depth: Array, edge: Mask | None = None) -> tuple[Array, int]:
    """The depth image with its edge pixels set to NaN (``edge`` from :func:`edge_mask`, computed
    here when not given), and how many were dropped."""
    mask = edge_mask(depth) if edge is None else edge
    out = np.asarray(depth, dtype=float).copy()
    out[mask] = np.nan
    return out, int(mask.sum())


# ---- the network's depth against the lidar: affine in inverse depth --------------------------
MIN_DEPTH_SPREAD = 2.5  # the pooled depths' 95th / 5th percentile must reach this for a shift
POOL_FRAMES = 600  # frames whose beams are pooled for the fit: minutes of views, so the law is
# the map's, not the view's — a 30-frame pool slid with the heading and layered a far wall
POOL_MIN_SAMPLES = 200  # pairs before a shift is fitted, and before any depth is published
A_BOUNDS = (0.3, 3.0)  # 1 / scale: the network is never off by more than this
B_BOUNDS = (-0.2, 0.2)  # 1/m: a shift beyond this is a broken fit, not a lens
LAW_MAX_AGE_S = 24 * 3600.0  # a saved law older than this is another day's room and lighting


def beam_pairs(
    depth: Array, samples: Array, edge: Mask | None = None
) -> tuple[Array, Array] | None:
    """The (network depth, true depth) pairs at the pixels the beams hit, or ``None`` under
    MIN_SAMPLES usable pixels. Pixels flagged in ``edge`` (:func:`edge_mask` of the same image)
    are left out: a beam landing on a blurred edge pairs the lidar's depth with a number between
    two surfaces, and such pairs bend the fit."""
    if samples.shape[0] == 0:
        return None
    cols = samples[:, 0].astype(int)
    rows = samples[:, 1].astype(int)
    predicted = np.asarray(depth, dtype=float)[rows, cols]
    ok = np.isfinite(predicted) & (predicted > NEAR_M) & (samples[:, 2] > NEAR_M)
    if edge is not None:
        ok &= ~edge[rows, cols]
    if int(ok.sum()) < MIN_SAMPLES:
        return None
    return predicted[ok], samples[ok, 2]


def _fit_noisy_on_exact(x: Array, y: Array) -> tuple[float, float]:
    """(alpha, beta) of x = alpha * y + beta by least squares, the worst quarter of the residuals
    dropped once and the fit repeated. ``x`` is the noisy variable (the network's 1 / D),
    ``y`` the exact one (the lidar's 1 / z)."""
    alpha, beta = np.polyfit(y, x, 1)
    res = np.abs(x - (alpha * y + beta))
    keep = res <= np.percentile(res, 75)
    if int(keep.sum()) >= 3:
        alpha, beta = np.polyfit(y[keep], x[keep], 1)
    return float(alpha), float(beta)


def _bounded(a: float, b: float, x: Array, y: Array) -> tuple[float, float]:
    """(a, b) of y = a x + b inside A_BOUNDS and B_BOUNDS. When a bound binds, the other
    parameter is refitted with the bound fixed (the median residual): clipping both independently
    turned a degenerate (0.27, 0.43) into a (0.3, 0.2) that fits nothing."""
    a_lo, a_hi = A_BOUNDS
    b_lo, b_hi = B_BOUNDS
    if not (math.isfinite(a) and math.isfinite(b)):
        return float(np.clip(np.median(y / x), a_lo, a_hi)), 0.0  # no slope at all: a scale
    if a < a_lo or a > a_hi:
        a = a_lo if a < a_lo else a_hi
        b = float(np.clip(np.median(y - a * x), b_lo, b_hi))
    elif b < b_lo or b > b_hi:
        b = b_lo if b < b_lo else b_hi
        a = float(np.clip(np.median((y - b) / x), a_lo, a_hi))
    return a, b


def fit_affine(d: Array, z: Array) -> tuple[float, float]:
    """The law 1 / z = a / D + b over (network, lidar) pairs, bounded to what a lens and a
    network can plausibly do.

    The regression runs the other way round, 1 / D = alpha / z + beta, and is inverted
    (a = 1 / alpha, b = -beta / alpha): least squares takes its regressor as exact and puts all
    the noise in the response, and the lidar's 1 / z is the exact one. Regressing 1 / z on the
    network's noisy 1 / D attenuates the slope towards zero and pushes the difference into the
    intercept (regression dilution) — the mechanism behind the a 0.27, b 0.43 that squeezed the
    room into two metres. A shift is fitted only when the pool spans MIN_DEPTH_SPREAD between
    its 5th and 95th depth percentiles and holds POOL_MIN_SAMPLES pairs (one reflection at 6 m
    must not enable it on a 1.3-2 m pool); otherwise the scale alone, the median ratio."""
    x, y = 1.0 / d, 1.0 / z
    lo, hi = np.percentile(z, (5, 95))
    if float(hi / lo) < MIN_DEPTH_SPREAD or d.size < POOL_MIN_SAMPLES:
        return float(np.clip(np.median(y / x), *A_BOUNDS)), 0.0
    alpha, beta = _fit_noisy_on_exact(x, y)
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = float(np.divide(1.0, alpha)), float(np.divide(-beta, alpha))
    return _bounded(a, b, x, y)


class AffineScale:
    """The affine correction fitted on the beams of the last POOL_FRAMES frames: a turn's worth
    of beams spans the room's depths, so the fit is conditioned and steady where a single
    frame's is not; a frame without beams keeps the law. Until POOL_MIN_SAMPLES pairs are
    pooled there is no law worth applying (``ready`` is false: the raw network's depth is
    1.5-2x too far and must not reach the costmap) — unless a saved law was ``seed``-ed, which
    then holds until the live pool can replace it."""

    def __init__(self, pool_frames: int = POOL_FRAMES) -> None:
        self.a = 1.0
        self.b = 0.0
        self.frames = 0
        self.held = 0
        self._pool: list[tuple[Array, Array]] = []
        self._pool_frames = pool_frames
        self._seeded = False

    def seed(self, a: float, b: float) -> None:
        """Start from a saved law (the map's): applied until POOL_MIN_SAMPLES live pairs exist."""
        self.a, self.b, self._seeded = a, b, True

    @property
    def ready(self) -> bool:
        """Whether a law worth applying exists: enough live pairs, or a seed."""
        return self._seeded or self.fitted

    @property
    def fitted(self) -> bool:
        """Whether the law rests on POOL_MIN_SAMPLES live pairs (worth saving)."""
        return self.pooled >= POOL_MIN_SAMPLES

    def observe(self, pairs: tuple[Array, Array] | None) -> tuple[float, float]:
        """Feed a frame's beam pairs (or ``None``); returns the (a, b) to apply to it."""
        self.frames += 1
        if pairs is None:
            self.held += 1
            return self.a, self.b
        self._pool.append(pairs)
        del self._pool[: -self._pool_frames]
        if self._seeded and not self.fitted:
            return self.a, self.b  # the map's law outranks a fit on a handful of pairs
        d = np.concatenate([p[0] for p in self._pool])
        z = np.concatenate([p[1] for p in self._pool])
        self.a, self.b = fit_affine(d, z)
        return self.a, self.b

    @property
    def pooled(self) -> int:
        """How many live beam pairs the pool holds."""
        return int(sum(p[0].size for p in self._pool))


def apply_affine(depth: Array, a: float, b: float) -> Array:
    """The network's depth corrected by 1 / z = a / D + b; pixels the law cannot place (a
    non-positive inverse depth) become NaN."""
    d = np.asarray(depth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = a / d + b
        z = np.where(np.isfinite(inv) & (inv > 1e-6), 1.0 / inv, np.nan)
    out: Array = z
    return out


def save_law(path: Path, a: float, b: float, pooled: int, now: float) -> None:
    """Write the law next to the maps, atomically (a temp file, then ``os.replace``): a restart
    begins from it instead of the raw network's depth. ``now`` is the wall clock in seconds."""
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps({"a": a, "b": b, "pooled": pooled, "saved_at": now}))
    os.replace(tmp, path)


def load_law(
    path: Path, now: float, max_age_s: float = LAW_MAX_AGE_S
) -> tuple[float, float, int] | None:
    """The saved (a, b, pooled) when the file is there, well-formed, inside the bounds, rests on
    POOL_MIN_SAMPLES pairs and is no older than ``max_age_s``; ``None`` otherwise."""
    try:
        data = json.loads(path.read_text())
        a, b = float(data["a"]), float(data["b"])
        pooled, saved_at = int(data["pooled"]), float(data["saved_at"])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    plausible = A_BOUNDS[0] <= a <= A_BOUNDS[1] and B_BOUNDS[0] <= b <= B_BOUNDS[1]
    if not plausible or pooled < POOL_MIN_SAMPLES or now - saved_at > max_age_s:
        return None
    return a, b, pooled
