"""Metric depth from one camera, with the lidar setting the scale.

A monocular depth network gives the shape of the scene but not its size: on our 70-degree lens
Depth Anything V2 (metric, indoor) called a coffee table one metre away 2.1 m. The lidar knows
distances exactly, but only in its own plane, 15 cm above the floor. Projected into the image,
the scan's points name the true depth at a few hundred pixels; the median ratio between what the
lidar measured and what the network guessed there is the frame's scale, and the whole depth image
is multiplied by it. The result is a depth image RTAB-Map can turn into a 3D map and the costmap
into obstacles the lidar cannot see: table tops, seats, shelves, cables on the floor.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]

MIN_SAMPLES = 20  # lidar points seen in the image before a frame's scale is believed
MAX_SCALE_STEP = 0.25  # a frame may move the running scale by this fraction, no more
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


def scale_from_samples(depth: Array, samples: Array) -> tuple[float, int] | None:
    """The factor that makes the network's depth image agree with the lidar at the sampled
    pixels (median of lidar depth over predicted depth), with the sample count; ``None`` when
    fewer than MIN_SAMPLES pixels carry a usable prediction."""
    if samples.shape[0] == 0:
        return None
    cols = samples[:, 0].astype(int)
    rows = samples[:, 1].astype(int)
    predicted = np.asarray(depth, dtype=float)[rows, cols]
    ok = np.isfinite(predicted) & (predicted > NEAR_M)
    if int(ok.sum()) < MIN_SAMPLES:
        return None
    ratio = samples[ok, 2] / predicted[ok]
    return float(np.median(ratio)), int(ok.sum())


class DepthScale:
    """The running scale of the depth network: follows the lidar's verdict frame by frame in
    bounded steps, and holds the last one while the lidar is out of the picture (a close wall,
    a doorway) — the network's scale drifts slowly, the view changes fast."""

    def __init__(self, initial: float = 1.0) -> None:
        self.value = initial
        self.frames = 0  # frames with a lidar verdict
        self.held = 0  # frames without one, since the last verdict

    def observe(self, verdict: tuple[float, int] | None) -> float:
        """Fold one frame's verdict in and return the scale to apply to it."""
        if verdict is None:
            self.held += 1
            return self.value
        scale, _count = verdict
        if self.frames == 0:
            self.value = scale
        else:
            low, high = self.value * (1 - MAX_SCALE_STEP), self.value * (1 + MAX_SCALE_STEP)
            self.value = min(max(scale, low), high)
        self.frames += 1
        self.held = 0
        return self.value


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
