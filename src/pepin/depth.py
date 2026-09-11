"""Metric depth from one camera, with the lidar setting the scale.

A monocular depth network gives the shape of the scene but not its size: on our 70-degree lens
Depth Anything V2 (metric, indoor) called a coffee table one metre away 2.1 m. The lidar knows
distances exactly, but only in its own plane, 20 cm above the floor (``config/lidar.json``).
Projected into the image, the scan's points name the true depth at a few hundred pixels; the
median ratio between what the lidar measured and what the network guessed there is the frame's
scale, and the whole depth image is multiplied by it. The result is a depth image RTAB-Map can
turn into a 3D map and the costmap into obstacles the lidar cannot see: table tops, seats,
shelves, cables on the floor.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

MIN_SAMPLES = 20  # lidar points seen in the image before a frame's scale is believed
MAX_SCALE_STEP = 0.05  # a frame may move the running scale by this share: the law is the slow,
# stable one; view-to-view swings of the beams' ratio (0.55..0.9) average out
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


SCAN_HALF_FOV = math.radians(40.0)  # the tilted camera's bearings reach a little past its lens
SCAN_STEP = math.radians(0.5)
SCAN_KTH = 3  # the k-th nearest point of a bearing: a flying pixel at an edge does not mark


def depth_to_scan(
    depth: Array,
    intr: Intrinsics,
    cam: CameraPose,
    *,
    stride: int = 4,
    min_z: float = 0.08,
    max_z: float = 1.30,
    max_range: float = 3.0,
) -> tuple[float, float, Array]:
    """The depth image as a planar scan in base_link: for every half-degree of bearing across the
    camera's view, the range to the nearest thing standing between ``min_z`` and ``max_z`` above
    the floor (a table top, a seat, a leg), ``inf`` where the view is clear out to ``max_range``.
    Returns (angle_min, angle_increment, ranges), ready for a LaserScan the costmap can mark and
    clear with. Every ``stride``-th pixel is used; the k-th nearest point per bearing marks."""
    d = np.asarray(depth, dtype=float)[::stride, ::stride]
    rows, cols = np.mgrid[0 : d.shape[0], 0 : d.shape[1]]
    u = cols * stride + 0.5
    v = rows * stride + 0.5
    ok = np.isfinite(d) & (d > NEAR_M)
    z_opt = d[ok]
    forward = z_opt
    left = -(u[ok] - intr.cx) / intr.fx * z_opt
    up = -(v[ok] - intr.cy) / intr.fy * z_opt
    c, s = math.cos(cam.pitch), math.sin(cam.pitch)
    px = c * forward + s * up + cam.x
    py = left + cam.y
    pz = -s * forward + c * up + cam.z
    keep = (pz > min_z) & (pz < max_z) & (px > 0.0)
    px, py = px[keep], py[keep]
    rng = np.hypot(px, py)
    bearing = np.arctan2(py, px)
    n_bins = round(2 * SCAN_HALF_FOV / SCAN_STEP) + 1
    ranges = np.full(n_bins, np.inf)
    within = (rng <= max_range) & (np.abs(bearing) <= SCAN_HALF_FOV)
    bins = np.rint((bearing[within] + SCAN_HALF_FOV) / SCAN_STEP).astype(int)
    order = np.lexsort((rng[within], bins))
    bins_sorted, rng_sorted = bins[order], rng[within][order]
    starts = np.flatnonzero(np.r_[True, bins_sorted[1:] != bins_sorted[:-1]])
    counts = np.diff(np.r_[starts, bins_sorted.size])
    enough = counts >= SCAN_KTH
    ranges[bins_sorted[starts[enough]]] = rng_sorted[starts[enough] + SCAN_KTH - 1]
    return -SCAN_HALF_FOV, SCAN_STEP, ranges


# ---- the floor as a second anchor -------------------------------------------------------------
FLOOR_TOLERANCE = 0.15  # the network's depth within this share of the floor's: it is the floor
UP_LEVEL: Array = np.array([0.0, 0.0, 1.0])
GRAVITY = 9.81
TILT_TAU_S = 1.0  # the accelerometer's up vector, low-passed: a bump is not a slope


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
    depth: Array, expected: Array, tolerance: float = FLOOR_TOLERANCE
) -> tuple[Array, int]:
    """Pixels whose depth agrees with the floor's within ``tolerance`` are set to the floor's
    exact depth; everything else (a sofa, a slipper, a box on the ray) is left to the network.
    Returns the corrected image and how many pixels were anchored."""
    d = np.asarray(depth, dtype=float)
    with np.errstate(invalid="ignore"):
        is_floor = np.isfinite(expected) & np.isfinite(d) & (np.abs(d / expected - 1.0) < tolerance)
    out = d.copy()
    out[is_floor] = expected[is_floor]
    return out, int(is_floor.sum())


class Tilt:
    """Which way is up, from the accelerometer: the IMU's reading turned into base_link through
    its mount and low-passed. At rest the chip reads +g along up; while the cart accelerates the
    reading leans, so only readings near 1 g count and a time constant smooths the rest."""

    def __init__(self, imu_to_base: Array, tau_s: float = TILT_TAU_S) -> None:
        self._rotation = np.asarray(imu_to_base, dtype=float)
        self._tau = tau_s
        self._up: Array = UP_LEVEL.copy()
        self._last_t: float | None = None

    def observe(self, accel_imu: Array, t: float) -> None:
        """Feed one accelerometer reading (m/s^2, the IMU's own axes) at time ``t`` (s)."""
        a = self._rotation @ np.asarray(accel_imu, dtype=float)
        norm = float(np.linalg.norm(a))
        if abs(norm - GRAVITY) > 1.0:  # braking, a bump: not gravity alone
            self._last_t = t
            return
        fresh = a / norm
        if self._last_t is None:
            self._up = fresh
        else:
            k = 1.0 - math.exp(-max(t - self._last_t, 0.0) / self._tau)
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


def edge_mask(depth: Array, rel_step: float = EDGE_REL_STEP) -> npt.NDArray[np.bool_]:
    """Pixels sitting on a depth discontinuity: the network blurs every object edge over a few
    pixels, so the pixels there carry depths between the object and what is behind it and land
    in mid-air when unprojected (in one frame, 21 % of the lidar-height pixels were more than
    15 cm from any lidar return, 2026-09-11). True where the depth differs from a horizontal
    or vertical neighbour by more than ``rel_step`` of itself, or where a neighbour is unknown."""
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


def drop_edges(depth: Array, rel_step: float = EDGE_REL_STEP) -> tuple[Array, int]:
    """The depth image with its edge pixels set to NaN, and how many were dropped."""
    edge = edge_mask(depth, rel_step)
    out = np.asarray(depth, dtype=float).copy()
    out[edge] = np.nan
    return out, int(edge.sum())


# ---- the network's depth against the lidar: affine in inverse depth --------------------------
MIN_DEPTH_SPREAD = 2.5  # the pooled beams must span this ratio of depths for a shift to be fitted
POOL_FRAMES = 600  # frames whose beams are pooled for the fit: minutes of views, so the law is
# the map's, not the view's — a 30-frame pool slid with the heading and layered a far wall
POOL_MIN_SAMPLES = 200
A_BOUNDS = (0.3, 3.0)  # 1 / scale: the network is never off by more than this
B_BOUNDS = (-0.2, 0.2)  # 1/m: a shift beyond this is a broken fit, not a lens


def beam_pairs(depth: Array, samples: Array) -> tuple[Array, Array] | None:
    """The (network depth, true depth) pairs at the pixels the beams hit, or ``None`` under
    MIN_SAMPLES usable pixels."""
    if samples.shape[0] == 0:
        return None
    cols = samples[:, 0].astype(int)
    rows = samples[:, 1].astype(int)
    predicted = np.asarray(depth, dtype=float)[rows, cols]
    ok = np.isfinite(predicted) & (predicted > NEAR_M) & (samples[:, 2] > NEAR_M)
    if int(ok.sum()) < MIN_SAMPLES:
        return None
    return predicted[ok], samples[ok, 2]


def fit_affine(d: Array, z: Array) -> tuple[float, float]:
    """1 / z = a / D + b over the pairs: least squares after dropping the worst quarter, a
    shift only when the true depths span MIN_DEPTH_SPREAD and POOL_MIN_SAMPLES pairs are
    there (a scale alone otherwise — a two-parameter fit on a narrow spread degenerates: one
    view of a sofa at 1.3-2 m gave a 0.27, b 0.43 and squeezed the room into two metres);
    both bounded to what a lens and a network can plausibly do."""
    x, y = 1.0 / d, 1.0 / z
    if float(z.max() / z.min()) < MIN_DEPTH_SPREAD or d.size < POOL_MIN_SAMPLES:
        return float(np.clip(np.median(y / x), *A_BOUNDS)), 0.0
    a, b = np.polyfit(x, y, 1)
    res = np.abs(y - (a * x + b))
    keep = res <= np.percentile(res, 75)
    if int(keep.sum()) >= 3:
        a, b = np.polyfit(x[keep], y[keep], 1)
    return float(np.clip(a, *A_BOUNDS)), float(np.clip(b, *B_BOUNDS))


class AffineScale:
    """The affine correction fitted on the beams of the last POOL_FRAMES frames: a turn's worth
    of beams spans the room's depths, so the fit is conditioned and steady where a single
    frame's is not; a frame without beams keeps the law."""

    def __init__(self, pool_frames: int = POOL_FRAMES) -> None:
        self.a = 1.0
        self.b = 0.0
        self.frames = 0
        self.held = 0
        self._pool: list[tuple[Array, Array]] = []
        self._pool_frames = pool_frames

    def observe(self, pairs: tuple[Array, Array] | None) -> tuple[float, float]:
        """Feed a frame's beam pairs (or ``None``); returns the (a, b) to apply to it."""
        self.frames += 1
        if pairs is None:
            self.held += 1
            return self.a, self.b
        self._pool.append(pairs)
        del self._pool[: -self._pool_frames]
        d = np.concatenate([p[0] for p in self._pool])
        z = np.concatenate([p[1] for p in self._pool])
        self.a, self.b = fit_affine(d, z)
        return self.a, self.b

    @property
    def pooled(self) -> int:
        """How many beam pairs the current law rests on."""
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
