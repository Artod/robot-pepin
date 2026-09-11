"""A truncated signed distance field: the room as one surface, not as a pile of clouds.

Every depth frame is integrated into a voxel grid where each voxel keeps one number — how far
it is from the nearest surface, positive in front of it, negative behind, clipped at the
truncation — and a weight, the confidence gathered so far. A new observation moves the number
by a weighted average, so three frames of one wall taken a few degrees apart make one wall
(blurred by their disagreement), never three. Observations weigh by their distance: a wall
measured from 1 m outweighs the same wall measured from 4 m, so the model sharpens when the
cart comes close and does not blur back when it leaves. The surface is where the number
crosses zero, read out at sub-voxel positions between neighbouring voxels.

Frame-to-model: before a frame is integrated, its points in the lidar's height band — exact by
construction, the lidar's own beams set them (``pepin.depth``) — are turned about the cart by a
few candidate yaws and scored against the model; the best turn corrects the pose the frame is
integrated with. The tracker's heading jitter at rest thus never reaches the model.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from pepin.depth import NEAR_M, Intrinsics

Array = npt.NDArray[np.float64]
Float32 = npt.NDArray[np.float32]


@dataclass(frozen=True)
class GridSpec:
    """The voxel grid: where it starts in the map frame, how many voxels, and the fusion law."""

    origin: tuple[float, float, float]
    shape: tuple[int, int, int]
    voxel_m: float = 0.05
    truncation_m: float = 0.10  # two voxels: a surface is felt this far on either side
    max_weight: float = 60.0  # the model stays movable when the furniture moves
    range_max_m: float = 4.0  # farther depth is noise and geometry error, not information
    weight_ref_m: float = 2.0  # an observation from this distance weighs 1; (ref/d)^2 otherwise
    weight_cap: float = 4.0  # a very near observation weighs at most this

    @classmethod
    def load(cls, path: str | Path) -> GridSpec:
        """The grid from ``config/fusion.json``."""
        with open(path) as f:
            data = json.load(f)
        return cls(
            origin=tuple(data["origin_m"]),
            shape=tuple(data["shape"]),
            voxel_m=float(data["voxel_m"]),
            truncation_m=float(data["truncation_m"]),
            max_weight=float(data["max_weight"]),
            range_max_m=float(data["range_max_m"]),
            weight_ref_m=float(data["weight_ref_m"]),
            weight_cap=float(data["weight_cap"]),
        )

    def observation_weight(self, depth: Array) -> Array:
        """How much a measurement at ``depth`` metres counts: (ref / d)^2, capped."""
        w: Array = np.minimum(self.weight_cap, (self.weight_ref_m / np.maximum(depth, 1e-3)) ** 2)
        return w


@dataclass(frozen=True)
class RigidPose:
    """A frame's placement in the map: 3x3 rotation and translation (map <- frame)."""

    rotation: Array
    translation: Array

    def inverse(self) -> RigidPose:
        r = self.rotation.T
        return RigidPose(r, -(r @ self.translation))

    def turned_about(self, pivot_xy: tuple[float, float], yaw: float) -> RigidPose:
        """The same pose after the world is turned by ``yaw`` about a vertical axis through
        ``pivot_xy`` — how a heading correction of the cart moves its camera."""
        c, s = math.cos(yaw), math.sin(yaw)
        rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        pivot = np.array([pivot_xy[0], pivot_xy[1], 0.0])
        t: Array = rz @ (self.translation - pivot) + pivot
        return RigidPose(rz @ self.rotation, t)


def backproject(depth: Array, intr: Intrinsics, stride: int = 1) -> Array:
    """The depth image as (n, 3) points in the optical frame (x right, y down, z forward),
    every ``stride``-th pixel, only finite depths beyond the lens."""
    d = np.asarray(depth, dtype=float)[::stride, ::stride]
    rows, cols = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    ok = np.isfinite(d) & (d > NEAR_M)
    z = d[ok]
    x = (cols[ok] - intr.cx) / intr.fx * z
    y = (rows[ok] - intr.cy) / intr.fy * z
    return np.stack([x, y, z], axis=1)


class Tsdf:
    """The fused model: signed distances, weights and colours on a grid, and the operations
    on it — integrate a frame, read the surface, score a set of points against the model."""

    def __init__(self, spec: GridSpec) -> None:
        self.spec = spec
        nx, ny, nz = spec.shape
        self.sdf: Float32 = np.ones((nx, ny, nz), dtype=np.float32)  # in truncation units
        self.weight: Float32 = np.zeros((nx, ny, nz), dtype=np.float32)
        self.rgb: Float32 = np.zeros((nx, ny, nz, 3), dtype=np.float32)

    def snapshot(self) -> Tsdf:
        """A copy of the model's arrays: read the surface from it while the original keeps
        integrating."""
        twin = Tsdf.__new__(Tsdf)
        twin.spec = self.spec
        twin.sdf, twin.weight, twin.rgb = self.sdf.copy(), self.weight.copy(), self.rgb.copy()
        return twin

    # ---- geometry helpers ----------------------------------------------------------------
    def _index_box(self, centre: Array, radius: float) -> tuple[slice, slice, slice] | None:
        """Voxel index ranges within ``radius`` metres of ``centre``, clipped to the grid."""
        s = self.spec
        lo = np.floor((centre - radius - np.array(s.origin)) / s.voxel_m).astype(int)
        hi = np.ceil((centre + radius - np.array(s.origin)) / s.voxel_m).astype(int) + 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array(s.shape))
        if np.any(hi <= lo):
            return None
        return slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2])

    def _centres(self, box: tuple[slice, slice, slice]) -> Array:
        s = self.spec
        ix, iy, iz = np.mgrid[box[0], box[1], box[2]]
        centres: Array = (
            np.stack([ix, iy, iz], axis=-1).reshape(-1, 3) + 0.5
        ) * s.voxel_m + np.array(s.origin)
        return centres

    def voxel_of(self, points_map: Array) -> tuple[Array, Array]:
        """Integer voxel indices of map points and a mask of the ones inside the grid."""
        s = self.spec
        idx = np.floor((np.asarray(points_map) - np.array(s.origin)) / s.voxel_m).astype(int)
        inside = np.all((idx >= 0) & (idx < np.array(s.shape)), axis=1)
        return idx, inside

    # ---- integration ---------------------------------------------------------------------
    def integrate(
        self,
        depth: Array,
        rgb: npt.NDArray[np.uint8] | None,
        intr: Intrinsics,
        pose: RigidPose,
    ) -> int:
        """Fuse one depth frame (metres, optical frame) taken from ``pose`` (map <- optical);
        returns how many voxels were updated."""
        s = self.spec
        box = self._index_box(pose.translation, s.range_max_m)
        if box is None:
            return 0
        centres = self._centres(box)
        inv = pose.inverse()
        cam = centres @ inv.rotation.T + inv.translation  # voxel centres in the optical frame
        z = cam[:, 2]
        front = z > NEAR_M
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(front, intr.fx * cam[:, 0] / z + intr.cx, -1.0)
            v = np.where(front, intr.fy * cam[:, 1] / z + intr.cy, -1.0)
        ui, vi = np.floor(u).astype(int), np.floor(v).astype(int)
        seen = front & (ui >= 0) & (ui < intr.width) & (vi >= 0) & (vi < intr.height)
        if not np.any(seen):
            return 0
        d = np.full(centres.shape[0], np.nan)
        d[seen] = np.asarray(depth, dtype=float)[vi[seen], ui[seen]]
        measured = np.isfinite(d) & (d > NEAR_M) & (d <= s.range_max_m)
        sdf = d - z  # positive: the voxel is between the camera and the surface
        touch = measured & (sdf > -s.truncation_m)
        if not np.any(touch):
            return 0
        t = np.minimum(1.0, sdf[touch] / s.truncation_m).astype(np.float32)
        w_obs = s.observation_weight(d[touch]).astype(np.float32)
        flat = np.flatnonzero(touch)
        shape = (box[0].stop - box[0].start, box[1].stop - box[1].start, box[2].stop - box[2].start)
        ix, iy, iz = np.unravel_index(flat, shape)
        ix, iy, iz = ix + box[0].start, iy + box[1].start, iz + box[2].start
        w_old = self.weight[ix, iy, iz]
        w_new = w_old + w_obs
        self.sdf[ix, iy, iz] = (self.sdf[ix, iy, iz] * w_old + t * w_obs) / w_new
        if rgb is not None:
            colour = np.asarray(rgb)[vi[touch], ui[touch]].astype(np.float32)
            self.rgb[ix, iy, iz] = (
                self.rgb[ix, iy, iz] * w_old[:, None] + colour * w_obs[:, None]
            ) / w_new[:, None]
        self.weight[ix, iy, iz] = np.minimum(s.max_weight, w_new)
        return int(flat.size)

    # ---- readout -------------------------------------------------------------------------
    def surface(self, min_weight: float = 2.0) -> tuple[Array, npt.NDArray[np.uint8]]:
        """Points where the field crosses zero between two weighted neighbours, interpolated
        to the crossing along each axis: the model's surface as (n, 3) map points and colours."""
        s = self.spec
        pts: list[Array] = []
        cols: list[Float32] = []
        known = self.weight >= min_weight
        for axis in range(3):
            a = [slice(None)] * 3
            b = [slice(None)] * 3
            a[axis] = slice(0, -1)
            b[axis] = slice(1, None)
            sa, sb = self.sdf[tuple(a)], self.sdf[tuple(b)]
            ka, kb = known[tuple(a)], known[tuple(b)]
            cross = ka & kb & (np.sign(sa) != np.sign(sb)) & (sa != 0)
            if not np.any(cross):
                continue
            idx = np.argwhere(cross).astype(float)
            frac = sa[cross] / (sa[cross] - sb[cross])
            idx[:, axis] += frac
            pts.append((idx + 0.5) * s.voxel_m + np.array(s.origin))
            ia = np.argwhere(cross)
            cols.append(self.rgb[ia[:, 0], ia[:, 1], ia[:, 2]])
        if not pts:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        return np.concatenate(pts), np.clip(np.concatenate(cols), 0, 255).astype(np.uint8)

    def fit_per_point(
        self, points_map: Array, min_weight: float = 2.0
    ) -> tuple[Array, npt.NDArray[np.bool_]]:
        """For every map point, 1 - |sdf| with the field read between voxel centres (trilinear,
        so a centimetre counts) and whether the point is on known ground (at least half of its
        eight voxels carry ``min_weight``); unknown points read 0."""
        s = self.spec
        n = int(np.asarray(points_map).shape[0])
        fit = np.zeros(n)
        known = np.zeros(n, dtype=bool)
        if n == 0:
            return fit, known
        g = (np.asarray(points_map, dtype=float) - np.array(s.origin)) / s.voxel_m - 0.5
        i0 = np.floor(g).astype(int)
        f = g - i0
        shape = np.array(s.shape)
        inside = np.all((i0 >= 0) & (i0 + 1 < shape), axis=1)
        if not np.any(inside):
            return fit, known
        i0, f = i0[inside], f[inside]
        value = np.zeros(i0.shape[0])
        mass = np.zeros(i0.shape[0])
        corners = np.zeros(i0.shape[0], dtype=int)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    ix, iy, iz = i0[:, 0] + dx, i0[:, 1] + dy, i0[:, 2] + dz
                    w = (f[:, 0] if dx else 1.0 - f[:, 0]) * (f[:, 1] if dy else 1.0 - f[:, 1])
                    w = w * (f[:, 2] if dz else 1.0 - f[:, 2])
                    here = self.weight[ix, iy, iz] >= min_weight
                    value += np.where(here, w * self.sdf[ix, iy, iz], 0.0)
                    mass += np.where(here, w, 0.0)
                    corners += here
        on_ground = (corners >= 4) & (mass > 1e-6)
        sdf = np.where(on_ground, value / np.maximum(mass, 1e-6), 1.0)
        fit[inside] = np.where(on_ground, np.maximum(0.0, 1.0 - np.abs(sdf)), 0.0)
        known[inside] = on_ground
        return fit, known

    def score(self, points_map: Array, min_weight: float = 2.0) -> tuple[float, int]:
        """The mean of ``fit_per_point`` over ALL the points (unknown ones count zero, so a
        turn that carries points out of the model loses) and how many were on known ground."""
        fit, known = self.fit_per_point(points_map, min_weight)
        if fit.size == 0:
            return 0.0, 0
        return float(fit.mean()), int(known.sum())


YAW_SEARCH = tuple(math.radians(d) for d in np.arange(-4.0, 4.01, 0.5))
ALIGN_MIN_POINTS = 200  # fewer band points on known voxels: the model has nothing to say
ALIGN_MIN_GAIN = 0.02  # the best turn must beat "no turn" by this much of the score


def align_yaw(
    model: Tsdf,
    band_map: Array,
    pivot_xy: tuple[float, float],
    candidates: tuple[float, ...] = YAW_SEARCH,
) -> tuple[float, float, int] | None:
    """The turn about ``pivot_xy`` that seats the frame's band points best on the model:
    (yaw, score gain over no turn, points on known ground at no turn), parabola-refined
    between the best three candidates. Every candidate is scored over the whole band, a point
    off known ground counting zero, so a turn cannot win by dropping points. ``None`` when the
    model knows too few of the points, when no turn beats none, or when the best candidate is
    the last one tried (the true turn may lie beyond the search, and a turn to the bound would
    bake the remainder into the model)."""
    pivot = np.array([pivot_xy[0], pivot_xy[1], 0.0])
    rel = band_map - pivot
    scores = []
    judged_at_zero = 0
    for yaw in candidates:
        c, s = math.cos(yaw), math.sin(yaw)
        turned = np.stack(
            [c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], axis=1
        )
        score, n = model.score(turned + pivot)
        scores.append(score)
        if yaw == 0.0:
            judged_at_zero = n
    if judged_at_zero < ALIGN_MIN_POINTS:
        return None
    best = int(np.argmax(scores))
    zero = candidates.index(0.0) if 0.0 in candidates else best
    gain = scores[best] - scores[zero]
    if gain < ALIGN_MIN_GAIN:
        return None
    if best == 0 or best == len(candidates) - 1:
        return None
    yaw = candidates[best]
    y0, y1, y2 = scores[best - 1], scores[best], scores[best + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom < 0.0:  # the vertex of the parabola through the three best
        step = candidates[best + 1] - candidates[best]
        yaw += 0.5 * (y0 - y2) / denom * step
    return float(yaw), float(gain), judged_at_zero


SLOW_TAU_S = 10.0  # the tracker's map->odom swings +-1.5 deg while the cart turns; frames placed
# by the smoothed correction keep the gyro's consistency between them (2026-09-11)


class SlowCorrection:
    """The tracker's map->odom correction low-passed: x, y and yaw follow it with a time
    constant, the first value taken whole. Frames placed by this correction composed with the
    odometry (gyro-smooth) stay consistent with each other while the tracker's per-match
    corrections jitter; the frame-to-model alignment absorbs what remains."""

    def __init__(self, tau_s: float = SLOW_TAU_S) -> None:
        self._tau = tau_s
        self._pose: RigidPose | None = None
        self._t: float | None = None

    def observe(self, correction: RigidPose, t: float) -> RigidPose:
        """Feed the tracker's map<-odom at time ``t`` (s); returns the smoothed one."""
        if self._pose is None or self._t is None:
            self._pose, self._t = correction, t
            return correction
        k = 1.0 - math.exp(-max(t - self._t, 0.0) / self._tau)
        old_yaw = math.atan2(self._pose.rotation[1, 0], self._pose.rotation[0, 0])
        new_yaw = math.atan2(correction.rotation[1, 0], correction.rotation[0, 0])
        d_yaw = (new_yaw - old_yaw + math.pi) % (2.0 * math.pi) - math.pi
        yaw = old_yaw + k * d_yaw
        c, s = math.cos(yaw), math.sin(yaw)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        translation = self._pose.translation + k * (correction.translation - self._pose.translation)
        self._pose, self._t = RigidPose(rotation, translation), t
        return self._pose

    @staticmethod
    def compose(a: RigidPose, b: RigidPose) -> RigidPose:
        """a then b: (map<-odom) composed with (odom<-camera) gives map<-camera."""
        return RigidPose(a.rotation @ b.rotation, a.rotation @ b.translation + a.translation)
