"""Metric depth from one camera, with the lidar setting the law.

A monocular depth network gives the shape of the scene but not its size, and the size it gets
wrong is not one number: on our lens Depth Anything V2 (metric, indoor) put a coffee table one
metre away at 2.1 m and the far wall further out still. The lidar knows distances exactly, but
only in its own plane, whose height above the floor is ``config/lidar.json``'s mount and nothing
else (:func:`pepin.mounts.load_lidar_mount`). Projected into the image,
the scan's beams name the true depth at a few hundred pixels; those (network, true) pairs,
pooled over minutes of frames so they span the room's depths, fit an affine law in inverse
depth, 1 / z = a / D + b — what a relative-depth network is built to be right up to — and the
law corrects the whole image (:class:`AffineScale`, :func:`fit_affine`, :func:`apply_affine`).
One affine law is not enough for this camera: what it leaves behind tilts with range (+8.7 % at
a metre, -3.3 % at two, 2026-09-14), so the same pairs are also read per bin of the network's
own depth (:class:`RangeLaw`), which is the law the node ships live.
Two more corrections act where they measure: pixels on an object's edge carry a depth blurred
between the object and what is behind it and are dropped (:func:`edge_mask`); pixels within a
few centimetres of the floor plane snap to it, the plane leaning with the cart
(:func:`floor_depth`, :func:`floor_anchor`; how far it leans is ``pepin.lean``'s, the one
estimator every consumer of the IMU takes from). The result is a depth image the
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
from typing import Any

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

    @classmethod
    def from_optical(cls, rotation: Array, translation: Array) -> CameraPose:
        """The pose from ``base_link <- camera_optical`` (the live TF edge when the neck
        moves): the translation as is (the optical frame shares the link's origin) and the
        pitch of the optical axis. A pan of the head is not carried — the projections here
        assume the camera looks along base_link's x (:func:`optical_heading` says how far it
        does not, and :func:`depth_to_scan` takes that pan beside this pose)."""
        t = np.asarray(translation, dtype=float)
        pitch, _pan = optical_heading(rotation)
        return cls(float(t[0]), float(t[1]), float(t[2]), pitch)


def optical_heading(rotation: Array) -> tuple[float, float]:
    """Where a camera looks, from ``base_link <- camera_optical``'s rotation: the optical axis
    (the frame's z) as a pitch (radians, positive down) and a pan (radians, positive left) in
    base_link."""
    forward = np.asarray(rotation, dtype=float)[:, 2]
    pitch = math.atan2(-float(forward[2]), math.hypot(float(forward[0]), float(forward[1])))
    return pitch, math.atan2(float(forward[1]), float(forward[0]))


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


def carry_speed(translation: Array, dt_s: float) -> float:
    """Metres per second a carry implies: how far base_link travelled between the two stamps,
    over the gap between them (a gap under a millisecond reads as a millisecond).

    The sanity test on the carry itself. On 2026-09-14 a runaway EKF (43 km at 60 m/s) moved
    the scan 1-2 m across the 0.02-0.03 s between a scan and a frame; the beams landed on the
    wrong pixels, the law was refitted from those pairs and a went 1.65 -> 2.05 until the law
    file was thrown away. This cart's top speed is 0.3 m/s, so anything near a metre per second
    is the odometry talking, not the robot.
    """
    step = float(np.linalg.norm(np.asarray(translation, dtype=float)))
    return step / max(abs(dt_s), 1e-3)


def project_all(
    points_base: Array, cam: CameraPose, intr: Intrinsics
) -> tuple[Array, Array, Array]:
    """Every base_link point as (column, row, depth along the optical axis), whether or not it
    lands inside the image or in front of the camera (a point behind the lens has a negative
    depth and a meaningless pixel): for a caller that must keep the points' order."""
    p = np.asarray(points_base, dtype=float) - np.array([cam.x, cam.y, cam.z])
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    # camera_link axes: forward, left, up; the pitch turns forward towards the floor
    forward = c * p[:, 0] - s * p[:, 2]
    left = p[:, 1]
    up = s * p[:, 0] + c * p[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = intr.fx * (-left / forward) + intr.cx
        v = intr.fy * (-up / forward) + intr.cy
    return u, v, forward


def in_image(u: Array, v: Array, forward: Array, intr: Intrinsics) -> Mask:
    """Which projected points are scene points of the picture: in front of the lens (deeper
    than NEAR_M) and inside the image."""
    keep: Mask = (forward > NEAR_M) & (u >= 0) & (u < intr.width) & (v >= 0) & (v < intr.height)
    return keep


def project(points_base: Array, cam: CameraPose, intr: Intrinsics) -> Array:
    """Pixels the base_link points land on: (m, 3) rows of column, row and depth along the
    optical axis, only for points inside the image and in front of the camera."""
    u, v, forward = project_all(points_base, cam, intr)
    keep = in_image(u, v, forward, intr)
    return np.stack([u[keep], v[keep], forward[keep]], axis=1)


def plane_in_view_from(intr: Intrinsics, cam: CameraPose, plane_z: float) -> float | None:
    """How far ahead of the cart a horizontal plane at height ``plane_z`` (base_link metres)
    first shows in the picture, or ``None`` when no distance puts it there.

    The lidar's returns all lie on one such plane — ``config/lidar.json``'s mount — so this is
    the nearest range at which a beam can judge the depth at all. Parked closer than it, with
    the head where it is, the beams fall under the bottom row and the anchor is blind through
    no fault of the lidar: measured 0.71 m for the reference head pitch of 23.7 degrees, the
    lens 0.82 m above the lidar's plane and a 640x360 picture (2026-09-14).
    """
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    dz = plane_z - cam.z
    nearest: float | None = None
    for row in (0.0, float(intr.height - 1)):
        k = (row - intr.cy) / intr.fy  # -up / forward along that row
        denominator = s + k * c
        if abs(denominator) < 1e-9:
            continue  # that row is parallel to the plane: it never meets it
        dx = -dz * (c - k * s) / denominator
        forward = c * dx - s * dz
        if dx <= 0.0 or forward <= NEAR_M:
            continue  # behind the cart, or inside the lens
        if nearest is None or dx < nearest:
            nearest = dx
    return None if nearest is None else cam.x + nearest


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
# 2026-09-11); 15 cm is a 12 % margin, well under the lidar's own plane, tops and seats stay in
SCAN_HALF_FOV = math.radians(40.0)  # the tilted camera's bearings reach a little past its lens
SCAN_STEP = math.radians(0.5)
SCAN_KTH = 3  # the k-th nearest point of a bearing: a flying pixel at an edge does not mark


def depth_to_scan(
    depth: Array,
    intr: Intrinsics,
    cam: CameraPose,
    *,
    stride: int = 4,
    pan: float = 0.0,
    min_z: float | Array = SCAN_MIN_Z_M,
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
    pixel is used; the k-th nearest point per bearing marks, a flying pixel does not.

    ``pan`` is where the head looks, radians CCW from the cart's x (the yaw of ``base_link <-
    camera_optical``, :func:`optical_heading`): the rays turn with it about base_link's z before
    the mount's translation is added, and the fan's window turns with them, so ``angle_min``
    comes back at ``pan - SCAN_HALF_FOV``. The output stays a base_link scan either way — every
    bin's angle is the true bearing of what it holds, and a panned head shows as the fan sitting
    off-centre in the same frame. 0 (the default) is the old projection, which reads the picture
    as if the head looked along the cart's x.

    ``min_z`` may be one height for the whole picture or an image of heights, one per pixel: a
    floor pixel stands ``camera height * (relative depth error)`` above the plane whatever its
    range, so where the network is noisier the band's floor must rise with it or the floor marks
    itself as an obstacle (:func:`pepin.contact.fan_min_z`, the ``band`` gate)."""
    d = np.asarray(depth, dtype=float)[::stride, ::stride]
    rows, cols = np.mgrid[0 : d.shape[0], 0 : d.shape[1]]
    u = cols * stride + 0.5
    v = rows * stride + 0.5
    ok = np.isfinite(d) & (d > NEAR_M)
    z_opt = d[ok]
    left = -(u[ok] - intr.cx) / intr.fx * z_opt
    up = -(v[ok] - intr.cy) / intr.fy * z_opt
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    cp, sp = math.cos(pan), math.sin(pan)
    fwd = c * z_opt + s * up  # the ray in base_link's plane once the neck's pitch is undone
    # ...then turned by the neck's pan about base_link's z. The translation is NOT turned: it
    # comes from the same TF edge as the pan and already holds where the panned head sits.
    px = cp * fwd - sp * left + cam.x
    py = sp * fwd + cp * left + cam.y
    pz = -s * z_opt + c * up + cam.z
    rng = np.hypot(px, py)
    # The bearing measured from the head's own heading, so the fan's bins keep meaning "the i-th
    # half-degree across the picture" whatever the pan (pepin.contact's fan is on that grid too)
    # and no wrap is needed at a pan near half a turn. The bin's angle, angle_min + i * step,
    # is then the point's true bearing in base_link.
    along, across = cp * px + sp * py, -sp * px + cp * py
    bearing = np.arctan2(across, along)
    n_bins = round(2 * SCAN_HALF_FOV / SCAN_STEP) + 1
    ranges = np.full(n_bins, np.nan)
    seen = (along > 0.0) & (np.abs(bearing) <= SCAN_HALF_FOV)
    bins = np.rint((bearing + SCAN_HALF_FOV) / SCAN_STEP).astype(int)
    ranges[np.unique(bins[seen])] = np.inf  # something was seen along the bearing: clear
    floor_of = np.asarray(min_z, dtype=float)
    if floor_of.ndim:  # an image of heights: the same pixels the depths were taken from
        floor_of = floor_of[::stride, ::stride][ok]
    marks = seen & (pz > floor_of) & (pz < max_z) & (rng <= max_range)
    order = np.lexsort((rng[marks], bins[marks]))
    bins_sorted, rng_sorted = bins[marks][order], rng[marks][order]
    starts = np.flatnonzero(np.r_[True, bins_sorted[1:] != bins_sorted[:-1]])
    counts = np.diff(np.r_[starts, bins_sorted.size])
    enough = counts >= SCAN_KTH
    ranges[bins_sorted[starts[enough]]] = rng_sorted[starts[enough] + SCAN_KTH - 1]
    return pan - SCAN_HALF_FOV, SCAN_STEP, ranges


# ---- the floor as a second anchor -------------------------------------------------------------
FLOOR_HEIGHT_TOLERANCE = 0.04  # metres above or below the floor plane a pixel may sit and be floor
UP_LEVEL: Array = np.array([0.0, 0.0, 1.0])


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
SCALE_FLOOR = 0.3  # 1 / scale: the network is never nearer than this factor
SCALE_CEILING = 5.0  # 1 / scale: nor farther. The old 3.0 was not a physical limit, it was a
# ceiling the data hit: with the lidar's beams placed at the measured mount the pool asks
# a ~= 2-3 and the joint fit overshoots it, so the fit saturated at 3.00 with b pinned at -0.200
# and stopped being a fit at all (scratch/lidar_height_check.py, 2026-09-12). The rulers that
# never hear of the lidar — the wall planes and the floor plane — read the network 2.0-2.3x too
# far, so 5.0 is better than twice the largest honest ask and still refuses a pool that has gone
# degenerate. A law that lands on a bound says so in the report line (:func:`at_bound`); it is a
# symptom to read, not a number to trust. Measured on run 0171 (scratch/lidar_height_fix_report.py):
# a is fitted at 2.80 instead of pinned at 3.00, and the fused band's distance to the lidar reads
# 5.2 cm against the clipped law's 3.8 — the clipped law was the closer of the two by luck, not by
# fit, and b still lands on B_BOUNDS at -0.200, which is the next bound to question and was left
# alone here. The 3.00 is one `ros2 param set depth_stream scale_ceiling 3.0` away
# (:func:`set_scale_ceiling`), so the field can settle the two A/B without a rebuild.
A_BOUNDS = (SCALE_FLOOR, SCALE_CEILING)  # the live pair; read it through :func:`a_bounds`
B_BOUNDS = (-0.2, 0.2)  # 1/m: a shift beyond this is a broken fit, not a lens
LAW_MAX_AGE_S = 24 * 3600.0  # a saved law older than this is another day's room and lighting


def a_bounds() -> tuple[float, float]:
    """What 1 / scale is allowed to be right now: :data:`A_BOUNDS` as
    :func:`set_scale_ceiling` last left it. Every site that bounds or judges a law reads it
    here, so one switch moves all of them at once."""
    return A_BOUNDS


def set_scale_ceiling(ceiling: float) -> None:
    """Move the upper bound on 1 / scale for this process (the ``scale_ceiling`` switch of the
    depth node); ``ValueError`` for a ceiling at or under the floor changes nothing."""
    global A_BOUNDS
    if not math.isfinite(ceiling) or ceiling <= A_BOUNDS[0]:
        raise ValueError(f"scale_ceiling {ceiling} is not above the floor {A_BOUNDS[0]}")
    A_BOUNDS = (A_BOUNDS[0], float(ceiling))


def at_bound(a: float, b: float) -> str:
    """Which of the law's parameters sits on its bound — ``"a"``, ``"b"``, ``"a+b"`` or ``""``.

    A clipped law is not a fit: the pool asked for more than a lens and a network are allowed to
    be off by, and the report line must say so rather than print a round 3.00 that reads as
    measured. Empty when the law is inside both bounds."""
    a_lo, a_hi = a_bounds()
    names = []
    if a <= a_lo or a >= a_hi:
        names.append("a")
    if b <= B_BOUNDS[0] or b >= B_BOUNDS[1]:
        names.append("b")
    return "+".join(names)


def beam_hits(
    depth: Array, samples: Array, edge: Mask | None = None
) -> npt.NDArray[np.intp] | None:
    """Which beams (rows of ``samples``: column, row, true depth) judge the depth image: those
    landing on a finite scene pixel off any edge, or ``None`` under MIN_SAMPLES of them."""
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
    return np.flatnonzero(ok)


def beam_pairs(
    depth: Array, samples: Array, edge: Mask | None = None
) -> tuple[Array, Array] | None:
    """The (network depth, true depth) pairs at the pixels the beams hit, or ``None`` under
    MIN_SAMPLES usable pixels. Pixels flagged in ``edge`` (:func:`edge_mask` of the same image)
    are left out: a beam landing on a blurred edge pairs the lidar's depth with a number between
    two surfaces, and such pairs bend the fit."""
    hits = beam_hits(depth, samples, edge)
    if hits is None:
        return None
    cols = samples[hits, 0].astype(int)
    rows = samples[hits, 1].astype(int)
    return np.asarray(depth, dtype=float)[rows, cols], samples[hits, 2]


REF_SIGMA_INV = 0.005  # 1/m: the inverse-depth noise a weight of 1 stands for. It is a lidar
# beam of 2 cm at 2 m, the unit the parallax pairs were already weighed in
# (pepin.parallax.LIDAR_SIGMA_INV). Nothing depends on the value itself — a weighted fit is
# invariant to a common factor — only on every ruler being weighed against the SAME one.


def pair_weight(sigma_inv: Array, cap: float | None = None) -> Array:
    """The weight a pair deserves in a fit from its own inverse-depth noise: ``1 / sigma^2``
    expressed against :data:`REF_SIGMA_INV`, so a pair as precise as a lidar beam at 2 m weighs
    1 and one twice as noisy weighs a quarter. ``cap`` bounds it from above (the parallax
    anchor caps at 1: no triangulated corner outweighs a beam).

    The fit lives in inverse depth, so this is where a ruler's noise belongs: a ruler with a
    constant noise in METRES is not equally good at every range there — a beam of 1.5 cm at
    1 m is 0.015 in inverse depth and the same beam at 3 m is 0.0017, eight times better.

    What this is NOT is the variance of the residual being minimised. The fit regresses the
    network's noisy 1 / D on the ruler's 1 / z and takes the ruler as exact
    (:func:`fit_affine`), so the residual's own noise is the NETWORK's — 0.02-0.10 of inverse
    depth at a few per cent of range, above a beam's 0.0002-0.023 everywhere. The weight is
    therefore a RELATIVE TRUST between rulers, sound for ranking a 7-10 cm corner against a
    1.5 cm beam and unsound for ranking beam against beam by range: applied inside one ruler it
    tilts the law as z^4 and measured worse on the lidar alone
    (:data:`pepin.depth_pipeline.LIDAR_SIGMA_M`, 2026-09-15)."""
    sigma = np.asarray(sigma_inv, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(np.isfinite(sigma) & (sigma > 0.0), (REF_SIGMA_INV / sigma) ** 2, 0.0)
    out: Array = w if cap is None else np.minimum(cap, w)
    return out


def inverse_sigma(sigma_m: float | Array, z: Array) -> Array:
    """The inverse-depth noise of a ruler whose noise is ``sigma_m`` metres at depth ``z``:
    ``sigma_m / z^2``, the first-order image of a metre error in 1 / z."""
    depth = np.asarray(z, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        out: Array = np.asarray(sigma_m, dtype=float) / depth**2
    return out


def weighted_median(values: Array, weight: Array | None) -> float:
    """The median of ``values``, each counting ``weight`` times (the plain median when
    ``weight`` is ``None``): the value at which half the total weight lies below — the mean of
    the two values on either side when the half falls exactly between them, so equal weights
    give ``np.median`` to the bit."""
    if weight is None:
        return float(np.median(values))
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    cumulative = np.cumsum(weight[order])
    half = 0.5 * cumulative[-1]
    i = min(int(np.searchsorted(cumulative, half)), values.size - 1)
    if cumulative[i] == half and i + 1 < values.size:
        return float((sorted_values[i] + sorted_values[i + 1]) / 2.0)
    return float(sorted_values[i])


def _fit_noisy_on_exact(x: Array, y: Array, weight: Array | None = None) -> tuple[float, float]:
    """(alpha, beta) of x = alpha * y + beta by least squares, the worst quarter of the residuals
    dropped once and the fit repeated. ``x`` is the noisy variable (the network's 1 / D),
    ``y`` the exact one (the lidar's 1 / z). ``weight`` counts each pair that many times in
    the squares (a per-pair share, not a per-pair sigma); the residual cut is per pair."""
    w = None if weight is None else np.sqrt(weight)
    alpha, beta = np.polyfit(y, x, 1, w=w)
    res = np.abs(x - (alpha * y + beta))
    keep = res <= np.percentile(res, 75)
    if int(keep.sum()) >= 3:
        alpha, beta = np.polyfit(y[keep], x[keep], 1, w=None if w is None else w[keep])
    return float(alpha), float(beta)


def _bounded(
    a: float, b: float, x: Array, y: Array, weight: Array | None = None
) -> tuple[float, float]:
    """(a, b) of y = a x + b inside A_BOUNDS and B_BOUNDS. When a bound binds, the other
    parameter is refitted with the bound fixed (the median residual): clipping both independently
    turned a degenerate (0.27, 0.43) into a (0.3, 0.2) that fits nothing."""
    a_lo, a_hi = a_bounds()
    b_lo, b_hi = B_BOUNDS
    if not (math.isfinite(a) and math.isfinite(b)):
        with np.errstate(divide="ignore", invalid="ignore"):  # a pair at depth 0 on both rulers
            return float(np.clip(weighted_median(y / x, weight), a_lo, a_hi)), 0.0  # a scale only
    if a < a_lo or a > a_hi:
        a = a_lo if a < a_lo else a_hi
        b = float(np.clip(weighted_median(y - a * x, weight), b_lo, b_hi))
    elif b < b_lo or b > b_hi:
        b = b_lo if b < b_lo else b_hi
        with np.errstate(divide="ignore", invalid="ignore"):
            a = float(np.clip(weighted_median((y - b) / x, weight), a_lo, a_hi))
    return a, b


def fit_affine(d: Array, z: Array, weight: Array | None = None) -> tuple[float, float]:
    """The law 1 / z = a / D + b over (network, lidar) pairs, bounded to what a lens and a
    network can plausibly do.

    The regression runs the other way round, 1 / D = alpha / z + beta, and is inverted
    (a = 1 / alpha, b = -beta / alpha): least squares takes its regressor as exact and puts all
    the noise in the response, and the lidar's 1 / z is the exact one. Regressing 1 / z on the
    network's noisy 1 / D attenuates the slope towards zero and pushes the difference into the
    intercept (regression dilution) — the mechanism behind the a 0.27, b 0.43 that squeezed the
    room into two metres. A shift is fitted only when the pool spans MIN_DEPTH_SPREAD between
    its 5th and 95th depth percentiles and holds POOL_MIN_SAMPLES pairs (one reflection at 6 m
    must not enable it on a 1.3-2 m pool); otherwise the scale alone, the median ratio.
    ``weight`` (one per pair, ``None`` for equal) is each pair's share in the squares and the
    medians; the spread and count gates count pairs, not weight."""
    x, y = 1.0 / d, 1.0 / z
    lo, hi = np.percentile(z, (5, 95))
    if float(hi / lo) < MIN_DEPTH_SPREAD or d.size < POOL_MIN_SAMPLES:
        return float(np.clip(weighted_median(y / x, weight), *a_bounds())), 0.0
    alpha, beta = _fit_noisy_on_exact(x, y, weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = float(np.divide(1.0, alpha)), float(np.divide(-beta, alpha))
    return _bounded(a, b, x, y, weight)


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


def apply_affine(depth: Array, a: float | Array, b: float | Array) -> Array:
    """The network's depth corrected by 1 / z = a / D + b; pixels the law cannot place (a
    non-positive inverse depth) become NaN. ``a`` and ``b`` are one pair of numbers for the
    whole image, or an image each — one law per pixel, which is what a scale field applies
    (:class:`pepin.depth_pipeline.ScaleField`)."""
    d = np.asarray(depth, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = a / d + b
        z = np.where(np.isfinite(inv) & (inv > 1e-6), 1.0 / inv, np.nan)
    out: Array = z
    return out


# ---- the same pairs read one frame at a time --------------------------------------------------
FRAME_MIN_PAIRS = 30  # beams in a frame before it may fit its own law
FRAME_MIN_SPREAD = MIN_DEPTH_SPREAD  # this frame's own 95th / 5th of true depth before a shift.
# The pool's gate, not a looser one: a frame's beams span a fraction of the room, and a shift
# fitted on what looks like enough spread is noise. Measured on 2026-09-14
# (scratch/frame_law_eval.py, the pairs of every frame split odd / even, odd fitting, even
# judging): with the gate at 1.5 the per-frame law read -58 % on tape 0235, whose frames span
# 1.2-2.5 m — spread enough to open the shift, not enough to identify it — while at 2.5 the same
# frames read +0.9 %. On the frames that do span the room (run 0171's drive) the two gates are
# a wash (7.8 % against 7.5 % of median |residual|), so the safe gate costs nothing.
FRAME_HOLD_TAU_S = 2.0  # seconds over which a frame with no beams decays back to the pool's law
IRLS_ROUNDS = 3  # re-weightings of the per-frame fit; the third moves the law by under 0.1 %
IRLS_HUBER = 1.345  # sigmas past which a pair's weight falls off as 1 / |residual| (Huber's 95 %)


def _irls(x: Array, y: Array, weight: Array | None = None) -> tuple[float, float]:
    """(alpha, beta) of x = alpha * y + beta by iteratively reweighted least squares with
    Huber's weights: the fit is repeated :data:`IRLS_ROUNDS` times, each pair counted the less
    the further its residual sits past :data:`IRLS_HUBER` robust sigmas (the MAD's), so a beam
    that grazed an edge or landed on a moving hand bends the law by a bounded amount instead of
    by its whole residual. ``x`` is the noisy variable (the network's 1 / D), ``y`` the exact
    one (the lidar's 1 / z); ``weight`` counts each pair that many times."""
    base = np.ones_like(x) if weight is None else np.asarray(weight, dtype=float)
    alpha, beta = np.polyfit(y, x, 1, w=np.sqrt(base))
    for _ in range(IRLS_ROUNDS):
        res = x - (alpha * y + beta)
        sigma = 1.4826 * float(np.median(np.abs(res - np.median(res))))
        if not math.isfinite(sigma) or sigma <= 0.0:
            break
        huber = np.minimum(1.0, IRLS_HUBER * sigma / np.maximum(np.abs(res), 1e-12))
        alpha, beta = np.polyfit(y, x, 1, w=np.sqrt(base * huber))
    return float(alpha), float(beta)


def _finite(a: float, b: float) -> tuple[float, float] | None:
    """``(a, b)`` when both are real numbers, ``None`` when either is not — "no law", which
    every caller of a frame fit already handles (the frame is held, the field keeps the fit it
    has). A law is NOT allowed to come back NaN: :func:`_bounded`'s last resort is a weighted
    median of ``1 / z`` over ``1 / D``, which is NaN when more than half the pairs read a
    non-positive corrected depth (the prior law turns those into NaN), and a NaN law reaches the
    picture through a matrix product that does not skip a zero membership — one NaN node of nine
    takes 100 % of the published pixels with it (scratch/_field_hazards.py, 2026-09-15)."""
    return (a, b) if math.isfinite(a) and math.isfinite(b) else None


def fit_frame(
    d: Array,
    z: Array,
    weight: Array | None = None,
    min_pairs: int = FRAME_MIN_PAIRS,
    min_spread: float = FRAME_MIN_SPREAD,
) -> tuple[float, float] | None:
    """The law 1 / z = a / D + b of ONE frame's pairs, or ``None`` under ``min_pairs`` of them
    and ``None`` when the fit does not come back finite (:func:`_finite`).
    ``d`` is whatever depth the frame is to be corrected from — the raw network's, or a pool
    law's output, in which case the numbers that come back are that law's residual.

    This is what the field does with a metric monocular network: the Depth Anything V2 papers
    report metric depth after a per-image scale-and-shift alignment against sparse truth, and a
    robot with a real depth sensor aligns the monocular image against its points frame by
    frame. The pool's law describes the camera over a minute of views; this one describes the
    picture in hand, so a scene the pool never held (a corridor after a room, a new light) is
    corrected by its own beams rather than by the average of the last 64 seconds.

    The regression is the pool's — the noisy 1 / D on the exact 1 / z, inverted
    (:func:`fit_affine`) — because a frame's pairs span little range and regressing the other
    way collapses the slope towards zero exactly when the fit is weakest. The robustness is
    heavier instead (:func:`_irls`), a frame having no other frames to outvote a bad beam. A
    shift is fitted only when the frame's own depths span ``min_spread`` — the pool's own 2.5,
    not a looser gate: one picture of one wall spans little, and a shift fitted on it is noise
    (:data:`FRAME_MIN_SPREAD`) — otherwise the scale alone, the weighted median ratio. The
    result is bounded like every other law (:func:`_bounded`).

    ``weight`` is each pair's 1 / sigma^2 (:func:`pair_weight`) and is what lets two rulers
    share one fit: the lidar's beams and the parallax anchor's corners land in the same pool and
    the fit reads each by its own noise, so the beams write the law where they reach and the
    corners carry it where they do not. The weights reach the robust fit as a multiplier on
    Huber's own weight (:func:`_irls`), never as a replacement: a heavy pair that is wrong is
    still cut down.

    What the weights do NOT reach is the spread gate above: it reads the 5th and 95th
    percentile of ``z`` by count, whoever measured them. A lidar-only pool is one row of one
    room and usually stays under the gate, so the frame gets a scale and no shift; a pool with
    parallax in it spans the corners of the whole picture and opens the shift far more often —
    on a parallax-only frame, essentially always. That is the intended behaviour (the spread is
    real, and a shift is what a spread identifies) but it means switching the anchor on changes
    which TERM the frame law fits, not only its numbers."""
    d = np.asarray(d, dtype=float)
    z = np.asarray(z, dtype=float)
    if d.size < min_pairs:
        return None
    x, y = 1.0 / d, 1.0 / z
    lo, hi = np.percentile(z, (5, 95))
    if not math.isfinite(hi / lo) or float(hi / lo) < min_spread:
        return _finite(float(np.clip(weighted_median(y / x, weight), *a_bounds())), 0.0)
    alpha, beta = _irls(x, y, weight)
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = float(np.divide(1.0, alpha)), float(np.divide(-beta, alpha))
    return _finite(*_bounded(a, b, x, y, weight))


# ---- the same fit over one patch of the picture, held by what it knows already -----------------
NO_PRIOR = (0.0, 0.0, 0.0, 0.0)  # the Tikhonov rows of :func:`_weighted_line`, carrying nothing


def _weighted_line(
    x: Array,
    y: Array,
    weight: Array,
    shift: bool,
    prior: tuple[float, float, float, float] = NO_PRIOR,
) -> tuple[float, float]:
    """(alpha, beta) of ``x = alpha * y + beta`` by weighted least squares in closed form —
    ``beta`` forced to zero (a line through the origin) when ``shift`` is false. NaN in
    ``alpha`` when the rows carry no weight or do not identify a line; two 2x2 sums instead of
    :func:`numpy.polyfit`, because a scale field fits one of these per node per frame.

    ``prior`` is ``(w_alpha, alpha0, w_beta, beta0)``: two TIKHONOV rows in PARAMETER space,
    ``sqrt(w_alpha) * (alpha - alpha0) = 0`` and ``sqrt(w_beta) * (beta - beta0) = 0``, added to
    the normal equations as ``w_alpha`` on the ``alpha`` diagonal and ``w_beta`` on the
    ``beta`` one. A weight of zero is a prior that says nothing and leaves the sums untouched
    to the bit, which is what every caller but :func:`fit_node` passes."""
    w_alpha, alpha0, w_beta, beta0 = prior
    syy = float(np.sum(weight * y * y)) + w_alpha
    sxy = float(np.sum(weight * x * y)) + w_alpha * alpha0
    scale_only = (sxy / syy, 0.0) if syy > 0.0 else (math.nan, 0.0)
    if not shift:
        return scale_only
    s = float(np.sum(weight)) + w_beta
    sy = float(np.sum(weight * y))
    sx = float(np.sum(weight * x)) + w_beta * beta0
    det = s * syy - sy * sy
    if not math.isfinite(det) or abs(det) <= 1e-12 * max(abs(s * syy), 1.0):
        return scale_only
    alpha = (s * sxy - sy * sx) / det
    return alpha, (sx - alpha * sy) / s


def _huber_weights(x: Array, y: Array, weight: Array, shift: bool) -> Array:
    """Each row's Huber multiplier, judged on a fit of THESE rows alone: the line is refitted
    :data:`IRLS_ROUNDS` times and a row counts the less the further its residual sits past
    :data:`IRLS_HUBER` robust sigmas (the MAD's), exactly as :func:`_irls` re-weights.

    Their own line, not the line a prior pulls, because a pseudo-observation is not an outlier
    and must not be allowed to make the data look like one: a node's pairs sitting perfectly on
    a line a prior disagrees with would otherwise all be cut down as "outliers" and the prior
    would win a fit it holds a twentieth of the weight in (scratch/_fit_node_probe.py,
    2026-09-15: it read a scale of 1.16 where its 24 pairs said 1.10)."""
    ones: Array = np.ones_like(x)
    if x.size < 3:
        return ones
    alpha, beta = _weighted_line(x, y, weight, shift and x.size >= 2)
    huber = ones
    # A scatter under a billionth of the inverse depth itself is not scatter, it is the last
    # bits of the arithmetic: rows lying exactly on their own line would otherwise be graded
    # against each other's rounding and come back weighing 1e-5 (scratch/_fit_node_probe.py).
    floor = 1e-9 * float(np.median(np.abs(x)))
    for _ in range(IRLS_ROUNDS):
        res = x - (alpha * y + beta)
        sigma = 1.4826 * float(np.median(np.abs(res - np.median(res))))
        if not math.isfinite(sigma) or sigma <= floor:
            break
        huber = np.minimum(1.0, IRLS_HUBER * sigma / np.maximum(np.abs(res), 1e-12))
        alpha, beta = _weighted_line(x, y, weight * huber, shift and x.size >= 2)
    return huber


def node_unit(y: Array, weight: Array, fallback: float) -> float:
    """How much information about the SLOPE one pair of weight 1 carries at this node: the
    weight-mean of ``y^2`` over the node's own pairs (``y`` their true inverse depth),
    ``fallback`` when the node has none — the frame's own such mean, handed down by the caller.

    The derivation, in one line. The fit solves ``x = alpha * y + beta`` by weighted least
    squares, so its information matrix is ``sum_i w_i * [[y_i^2, y_i], [y_i, 1]]``: a pair of
    weight 1 at inverse depth ``y`` is worth ``y^2`` about ``alpha`` and 1 about ``beta``. The
    node's mean ``y^2`` is therefore the exchange rate between "one pair" and "one unit of
    slope information" AT THIS NODE — which is exactly what a prior in pairs has to be
    multiplied by to mean the same thing near the camera and across the room."""
    total = float(np.sum(weight))
    if total <= 0.0 or y.size == 0:
        return fallback
    unit = float(np.sum(weight * y * y) / total)
    return unit if math.isfinite(unit) and unit > 0.0 else fallback


def fit_node(
    d: Array,
    z: Array,
    weight: Array,
    priors: Sequence[tuple[float, float, float]] = (),
    unit: float = 1.0,
    shift: bool = True,
) -> tuple[float, float] | None:
    """The law ``1 / z = a / D + b`` of ONE NODE of a scale field: the same regression as
    :func:`fit_frame` on the pairs that belong to the node, held by a prior that says what the
    node should be where its own pairs say little. ``None`` when nothing at all constrains it
    (no pairs and no priors) and when the fit does not come back finite (:func:`_finite`) — a
    NaN node poisons every pixel of the picture, not only its own.

    ``priors`` are (a, b, weight) laws to be pulled toward — the frame's own global fit, and
    the node's previous value decayed by the time since. Each enters as two TIKHONOV rows in
    PARAMETER space, ``sqrt(w_alpha) * (alpha - alpha0) = 0`` and
    ``sqrt(w_beta) * (beta - beta0) = 0`` about the prior's own
    ``alpha0 = 1 / a``, ``beta0 = -b / a``, and NOT as pseudo-observations at two depths.
    Its weight is read in PAIRS: ``weight = N`` means "as much information about that parameter
    as N pairs of weight 1 would carry AT THIS NODE", which is ``w_alpha = N * unit`` and
    ``w_beta = N`` — the exchange rate ``unit`` being the node's own mean ``y^2``
    (:func:`node_unit`), because the fit's information matrix is
    ``sum_i w_i * [[y_i^2, y_i], [y_i, 1]]``. So a node that saw no pair comes back as the
    prior, a node that saw hundreds of beams follows them, and in between the two are averaged
    the way two rulers of different noise always are — at the same exchange rate whether the
    node looks at the far wall or at the cart's own bumper.

    What it replaces, and why (scratch/_field_refutations.py, 2026-09-15). The pull used to be
    two pseudo-observation ROWS on the prior's line at the two ends of the pairs' depth range,
    ``0.5 * weight`` each. Rows at ``+-dy/2`` about the mean carry ``w * dy^2 / 4`` of slope
    information against the data's ``W * s^2``, so their pull depended on how wide the FRAME's
    span was and on where in it the node's pairs sat: one probe (ten pairs of weight 1 at one
    depth, a prior of weight 1) measured an effective pull of 0.4 pairs with the cluster at
    0.6 m and 36 pairs with it at 6 m, and a flat ~15 pairs once the shift was open. A field
    whose top nodes hold a handful of weak parallax corners could therefore never leave the
    global fit, whatever ``field_prior`` was set to. "One beam's worth" now means one beam's
    worth everywhere.

    A prior is not an outlier: the robust re-weighting is judged on the pairs' own line and
    applied to their rows only (:func:`_huber_weights`) — the Tikhonov rows are not rows of the
    design at all and can never be reweighted.

    ``shift`` says whether the node may fit a shift at all, and belongs to the FRAME, not to
    the node: a node holds a handful of beams over half a metre of depth, and a two-parameter
    fit on that is noise. The caller passes what its own global fit decided
    (:data:`FRAME_MIN_SPREAD`), so a field never opens a term the frame's gate refused. The
    result is bounded like every other law (:func:`_bounded`), on the node's OWN pairs: when a
    bound binds, the other parameter is refitted as a median residual, and a prior in parameter
    space has no residual to take a median of. A node with no pairs is clipped instead."""
    d = np.asarray(d, dtype=float)
    z = np.asarray(z, dtype=float)
    w = np.asarray(weight, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):  # a pair at depth 0 drops out below
        x, y = 1.0 / d, 1.0 / z
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(w) & (w > 0.0)
    x, y, w = x[ok], y[ok], w[ok]
    rate = node_unit(y, w, float(unit) if math.isfinite(unit) and unit > 0.0 else 1.0)
    w_alpha, w_beta, num_alpha, num_beta = 0.0, 0.0, 0.0, 0.0
    for a_prior, b_prior, pull in priors:
        if pull <= 0.0 or not math.isfinite(a_prior) or a_prior == 0.0:
            continue
        if not math.isfinite(b_prior):
            continue
        w_alpha += pull * rate
        num_alpha += pull * rate / a_prior
        w_beta += pull
        num_beta += pull * (-b_prior / a_prior)
    if x.size == 0 and w_alpha <= 0.0:
        return None  # nothing at all constrains this node
    if x.size:
        w = w * _huber_weights(x, y, w, shift)
    prior = NO_PRIOR
    if w_alpha > 0.0:
        prior = (w_alpha, num_alpha / w_alpha, w_beta, num_beta / w_beta)
    alpha, beta = _weighted_line(x, y, w, shift and (x.size >= 2 or w_beta > 0.0), prior)
    with np.errstate(divide="ignore", invalid="ignore"):
        a, b = float(np.divide(1.0, alpha)), float(np.divide(-beta, alpha))
    if x.size == 0:  # no pairs to take a residual median over: the prior, held to the bounds
        return _finite(float(np.clip(a, *a_bounds())), float(np.clip(b, *B_BOUNDS)))
    return _finite(*_bounded(a, b, x, y, w))


# ---- the same pairs read as a curve over the network's range ----------------------------------
RANGE_NEAR = 0.3  # network metres: the network places no scene point nearer than this
RANGE_FAR = 12.0  # nor further out than this indoors; past it the pairs are reflections
RANGE_BINS = 27  # log-spaced between the two: 15 % of range per bin
RANGE_EDGES: Array = np.geomspace(RANGE_NEAR, RANGE_FAR, RANGE_BINS + 1, dtype=np.float64)
# Log-spaced, not 0.25 m steps, because the correction is a ratio: a constant *relative* bin
# width keeps the resolution constant in the quantity being fitted, and the network's depth is
# itself stretched (1.6-2.0x), so 0.25 m of true range is 0.4-0.5 m of network depth near the
# cart and more further out. Linear 0.25 m bins would leave the whole far half of the picture
# empty and held from its nearest neighbour, which is the affine law again under another name.
# 15 % a bin, not 24 %: at the measured 12 % of tilt per metre a 24 % bin leaves 1-2 % of tilt
# inside itself and 2.3 % past the outermost centre, where the law holds flat
# (scratch/range_law_bins.py); at 15 % that falls under 1 % and the pool still fills every bin
# it reaches — at rest the 62 000 pooled pairs sit in a dozen of them, hundreds to thousands
# each, against the 50 a bin needs.
RANGE_MIN_PAIRS = 50  # pairs in a bin before its ratio is its own; fewer and the bin is empty
RANGE_MIN_BINS = 2  # filled bins before the law says anything about range (one is a plain scale)


def _rising(centres: list[float], ratios: list[float], counts: list[int]) -> list[int]:
    """Which bins to keep so that the corrected depth grows with the network's depth: where two
    neighbours disagree about that, the one resting on fewer pairs is dropped, and the check
    runs again. Measured 2026-09-14 at home: the outermost bin (1 076 pairs against the next
    one's 14 301) asked for 1.74 m where the bin before it asked for 1.85, and the published
    depth came out 41 cm short over 2.0-2.5 m. A network's depth is a rising function of the
    true one whatever it gets wrong about the size, so a bin that inverts that order is a bad
    pairing — a panned head, a stale scan — not a lens."""
    kept = list(range(len(centres)))
    while len(kept) > 1:
        depth = [centres[i] * ratios[i] for i in kept]
        falls = [k for k in range(len(kept) - 1) if depth[k + 1] <= depth[k]]
        if not falls:
            break
        k = falls[0]
        kept.pop(k if counts[kept[k]] < counts[kept[k + 1]] else k + 1)
    return kept


def ratio_bounds() -> tuple[float, float]:
    """What a bin's true / network ratio is allowed to be: the reciprocal of :func:`a_bounds`,
    so the range law and the affine law are held to the same physics."""
    a_lo, a_hi = a_bounds()
    return 1.0 / a_hi, 1.0 / a_lo


@dataclass(frozen=True, eq=False)
class RangeLaw:
    """The correction as a curve over the network's own depth: per bin of network depth, the
    robust ratio true / network measured there.

    One affine law in inverse depth cannot describe this camera. Fitted on a pool that spans
    the room it leaves a residual that tilts with range — measured on 2026-09-14 at home
    (scratch/depth_scale_by_range.py, 182 frames, 18 879 beams): the published depth ran +8.7 %
    at 0.8-1.2 m, +5.2 % at 1.2-1.6 m and -3.3 % at 1.6-2.0 m, 12 % per metre, while the whole
    pool's median ratio was a healthy 0.996. Which ranges the pool happens to hold then decides
    the law: a drive brings 0.5 m and 4 m pairs, the shift term switches on, and the same camera
    is described as a 1.75 b 0 standing and a 2.3 b -0.19 driving. The volume is painted under
    one law and scored under the other.

    This law asks the pairs the question they answer — how wrong is the network *here* — and
    answers it per range: for every bin that held at least :data:`RANGE_MIN_PAIRS` pairs,
    :attr:`ratios` is the weighted median of true / network inside it and :attr:`centres` the
    median network depth of those same pairs — where the ratio was measured, not the bin's
    nominal middle, which a pool that fills a bin unevenly would put it beside (2.8 % of
    residual in the 1.2-1.6 m band on the synthetic tilt of tests/unit/test_depth.py) —
    :attr:`counts` how many pairs each rests on. Between two filled centres the
    ratio is linear in the network's depth; outside them it is held at the nearest filled
    centre's, never extrapolated. Corrected depth = network depth x ratio(network depth), and
    that corrected depth must rise from bin to bin (:func:`_rising`)."""

    centres: Array
    ratios: Array
    counts: npt.NDArray[np.intp]

    @classmethod
    def fit(
        cls,
        d: Array,
        z: Array,
        weight: Array | None = None,
        edges: Array = RANGE_EDGES,
        min_pairs: int = RANGE_MIN_PAIRS,
        min_bins: int = RANGE_MIN_BINS,
    ) -> RangeLaw | None:
        """The law over (network depth ``d``, true depth ``z``) pairs, each counting ``weight``
        times, or ``None`` when under ``min_bins`` bins survive — a pool that sees one range
        says nothing about range, and the affine law's fit over all of it is the better answer
        there. A bin needs ``min_pairs`` pairs; its ratio is clipped to :func:`ratio_bounds`;
        and a bin that puts the corrected depth below its nearer neighbour's is dropped
        (:func:`_rising`)."""
        d = np.asarray(d, dtype=float)
        z = np.asarray(z, dtype=float)
        ok = np.isfinite(d) & np.isfinite(z) & (d > NEAR_M) & (z > NEAR_M)
        lo, hi = ratio_bounds()
        index = np.digitize(d, edges) - 1
        centres, ratios, counts = [], [], []
        for b in range(edges.size - 1):
            inside = ok & (index == b)
            n = int(inside.sum())
            if n < min_pairs:
                continue
            share = None if weight is None else np.asarray(weight, dtype=float)[inside]
            centres.append(weighted_median(d[inside], share))
            ratios.append(float(np.clip(weighted_median(z[inside] / d[inside], share), lo, hi)))
            counts.append(n)
        kept = _rising(centres, ratios, counts)
        if len(kept) < min_bins:
            return None
        return cls(
            np.asarray([centres[i] for i in kept]),
            np.asarray([ratios[i] for i in kept]),
            np.asarray([counts[i] for i in kept], dtype=np.intp),
        )

    def ratio(self, depth: Array) -> Array:
        """The true / network ratio this law gives each pixel of ``depth``: interpolated
        between the filled bins' centres, held flat beyond the outermost of them."""
        out: Array = np.interp(np.asarray(depth, dtype=float), self.centres, self.ratios)
        return out

    def apply(self, depth: Array) -> Array:
        """The network's depth in metres, each pixel scaled by the ratio measured at its own
        range; pixels the law cannot place (non-finite, or non-positive) become NaN."""
        d = np.asarray(depth, dtype=float)
        with np.errstate(invalid="ignore"):
            z = d * self.ratio(d)
            out: Array = np.where(np.isfinite(z) & (z > 0.0), z, np.nan)
        return out

    def describe(self) -> str:
        """The law for the report line: every filled bin as ``centre:ratio`` over the network's
        depth, then the pairs behind each of them."""
        bins = " ".join(f"D{c:.2f}:{r:.3f}" for c, r in zip(self.centres, self.ratios, strict=True))
        return f"{bins} (n {'/'.join(str(int(n)) for n in self.counts)})"

    def state(self) -> dict[str, Any]:
        """The law as plain JSON values for :func:`save_law`."""
        return {
            "centres": [round(float(c), 4) for c in self.centres],
            "ratios": [round(float(r), 5) for r in self.ratios],
            "counts": [int(n) for n in self.counts],
        }

    @classmethod
    def restore(cls, record: Any) -> RangeLaw | None:
        """The law a past run saved (:meth:`state`), or ``None`` when the record is missing,
        malformed, too short to say anything about range, or asks for a ratio outside
        :func:`ratio_bounds` — a file is not a measurement until it passes the same gates."""
        try:
            centres = np.asarray([float(c) for c in record["centres"]])
            ratios = np.asarray([float(r) for r in record["ratios"]])
            counts = np.asarray([int(n) for n in record["counts"]], dtype=np.intp)
        except (KeyError, TypeError, ValueError):
            return None
        if not (centres.size == ratios.size == counts.size) or centres.size < RANGE_MIN_BINS:
            return None
        lo, hi = ratio_bounds()
        sane = (
            bool(np.all(np.isfinite(centres)))
            and bool(np.all(np.diff(centres) > 0.0))
            and bool(np.all(np.isfinite(ratios)))
            and bool(np.all((ratios >= lo) & (ratios <= hi)))
            and bool(np.all(counts >= RANGE_MIN_PAIRS))
        )
        return cls(centres, ratios, counts) if sane else None


LAW_VERSION = 3  # 1: the affine law alone; 2: the ray law's record beside it; 3: the range law's


def save_law(
    path: Path,
    a: float,
    b: float,
    pooled: int,
    now: float,
    ray: dict[str, Any] | None = None,
    range_law: dict[str, Any] | None = None,
) -> None:
    """Write the law next to the maps, atomically (a temp file, then ``os.replace``): a restart
    begins from it instead of the raw network's depth. ``now`` is the wall clock in seconds.
    ``ray`` is the angle-dependent law's record beside the affine one
    (:meth:`pepin.elevation.RayGain.state`) and ``range_law`` the range-dependent one's
    (:meth:`RangeLaw.state`), in the same file each under its own key so one law is
    never read with another's map: a reader of version 1 sees the affine law it expects and
    ignores the rest."""
    tmp = path.with_name(path.name + ".tmp")
    record: dict[str, Any] = {
        "version": LAW_VERSION,
        "a": a,
        "b": b,
        "pooled": pooled,
        "saved_at": now,
    }
    if ray is not None:
        record["ray"] = ray
    if range_law is not None:
        record["range"] = range_law
    tmp.write_text(json.dumps(record))
    os.replace(tmp, path)


def load_range(path: Path, now: float, max_age_s: float = LAW_MAX_AGE_S) -> RangeLaw | None:
    """The range law the last run saved when the file is there, holds one no older than
    ``max_age_s`` and it passes :meth:`RangeLaw.restore`; ``None`` otherwise — a file written
    before this law existed simply has none, and the affine law seeds the warm-up instead."""
    try:
        data = json.loads(path.read_text())
        saved_at = float(data["saved_at"])
        record = data.get("range")
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if record is None or now - saved_at > max_age_s:
        return None
    return RangeLaw.restore(record)


def load_ray(path: Path, now: float, max_age_s: float = LAW_MAX_AGE_S) -> Any | None:
    """The saved ray law's record (the dict :meth:`pepin.elevation.RayGain.state` wrote) when
    the file is there, holds one, and is no older than ``max_age_s``; ``None`` otherwise — an
    older file, or one written before the ray law existed, simply has none. The record is not
    judged here: :meth:`pepin.elevation.RayGain.restore` does that, so this module keeps no
    knowledge of the angular law's shape."""
    try:
        data = json.loads(path.read_text())
        saved_at = float(data["saved_at"])
        ray = data.get("ray")
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if ray is None or now - saved_at > max_age_s:
        return None
    return ray


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
    a_lo, a_hi = a_bounds()
    plausible = a_lo <= a <= a_hi and B_BOUNDS[0] <= b <= B_BOUNDS[1]
    if not plausible or pooled < POOL_MIN_SAMPLES or now - saved_at > max_age_s:
        return None
    return a, b, pooled
