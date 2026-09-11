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
integrated with. The tracker's heading jitter at rest thus never reaches the model. The frame
is placed by TF at its own stamp, nothing else: a smoothed copy of the tracker's correction was
tried and lagged behind a drive across the room (2026-09-11).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.depth import NEAR_M, Intrinsics

Array = npt.NDArray[np.float64]
Float32 = npt.NDArray[np.float32]
Floats = npt.NDArray[np.floating[Any]]
Uint8 = npt.NDArray[np.uint8]


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

    def observation_weight(self, depth: Floats) -> Floats:
        """How much a measurement at ``depth`` metres counts: (ref / d)^2, capped."""
        w: Floats = np.minimum(self.weight_cap, (self.weight_ref_m / np.maximum(depth, 1e-3)) ** 2)
        return w


@dataclass(frozen=True)
class RigidPose:
    """A frame's placement in the map: 3x3 rotation and translation (map <- frame)."""

    rotation: Array
    translation: Array

    def inverse(self) -> RigidPose:
        """The same transform the other way (frame <- map)."""
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


def backproject(
    depth: Array, intr: Intrinsics, stride: int = 1, range_max: float = math.inf
) -> Array:
    """The depth image as (n, 3) points in the optical frame (x right, y down, z forward),
    every ``stride``-th pixel, only finite depths beyond the lens and within ``range_max``
    (the model integrates nothing farther, so farther points can never be on known ground)."""
    d = np.asarray(depth, dtype=float)[::stride, ::stride]
    rows, cols = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    ok = np.isfinite(d) & (d > NEAR_M) & (d <= range_max)
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
        self.rgb: Uint8 = np.zeros((nx, ny, nz, 3), dtype=np.uint8)
        # colour has its own weight: a voxel seen as free space for a while and then as a wall
        # would otherwise start its colour from black
        self.colour_weight: Float32 = np.zeros((nx, ny, nz), dtype=np.float32)

    def snapshot(self) -> Tsdf:
        """A copy of what ``surface`` reads (field, weight, colour): read the surface from it
        while the original keeps integrating."""
        twin = Tsdf.__new__(Tsdf)
        twin.spec = self.spec
        twin.sdf, twin.weight, twin.rgb = self.sdf.copy(), self.weight.copy(), self.rgb.copy()
        return twin

    # ---- geometry helpers ----------------------------------------------------------------
    def _index_box(self, lo_m: Array, hi_m: Array) -> tuple[slice, slice, slice] | None:
        """Voxel index ranges of the box ``lo_m``..``hi_m`` (map metres), clipped to the grid;
        ``None`` when the box misses the grid."""
        s = self.spec
        origin = np.array(s.origin)
        lo = np.floor((lo_m - origin) / s.voxel_m).astype(int) - 1
        hi = np.ceil((hi_m - origin) / s.voxel_m).astype(int) + 1
        lo = np.maximum(lo, 0)
        hi = np.minimum(hi, np.array(s.shape))
        if np.any(hi <= lo):
            return None
        return slice(lo[0], hi[0]), slice(lo[1], hi[1]), slice(lo[2], hi[2])

    def _frustum_box(self, intr: Intrinsics, pose: RigidPose) -> tuple[slice, slice, slice] | None:
        """The voxels a frame can touch: the bounding box of the camera and its four corner
        rays at ``range_max`` plus the truncation (a surface at the range limit is felt that
        far behind it), so a frame integrates its own view, not a 8 m cube around it."""
        s = self.spec
        far = s.range_max_m + s.truncation_m
        us, vs = (-0.5, intr.width - 0.5), (-0.5, intr.height - 0.5)  # the pixels' outer edges
        corners = np.array(
            [
                [(u - intr.cx) / intr.fx * far, (v - intr.cy) / intr.fy * far, far]
                for u in us
                for v in vs
            ]
        )
        pts = np.vstack([pose.translation, corners @ pose.rotation.T + pose.translation])
        return self._index_box(pts.min(axis=0), pts.max(axis=0))

    def _centres(self, box: tuple[slice, slice, slice]) -> Float32:
        s = self.spec
        ix, iy, iz = np.mgrid[box[0], box[1], box[2]]
        centres: Float32 = (
            (np.stack([ix, iy, iz], axis=-1).reshape(-1, 3) + 0.5) * s.voxel_m + np.array(s.origin)
        ).astype(np.float32)
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
        depth: Array | Float32,
        rgb: Uint8 | None,
        intr: Intrinsics,
        pose: RigidPose,
    ) -> int:
        """Fuse one depth frame (metres, optical frame) taken from ``pose`` (map <- optical);
        returns how many voxels were updated. Colour goes only into voxels within the
        truncation of the surface, never into the free space a ray crosses on its way."""
        s = self.spec
        box = self._frustum_box(intr, pose)
        if box is None:
            return 0
        centres = self._centres(box)
        inv = pose.inverse()
        # voxel centres in the optical frame, float32 throughout: a centimetre is 1e-2 of a
        # metre, far above float32's 1e-7, and the frame is millions of voxels
        cam = centres @ inv.rotation.T.astype(np.float32) + inv.translation.astype(np.float32)
        z = cam[:, 2]
        front = z > NEAR_M
        with np.errstate(divide="ignore", invalid="ignore"):
            u = np.where(front, intr.fx * cam[:, 0] / z + intr.cx, -1.0)
            v = np.where(front, intr.fy * cam[:, 1] / z + intr.cy, -1.0)
        # pixel i's ray passes through coordinate i (``backproject``): the nearest pixel, not the
        # one to the left, or every voxel reads the depth half a pixel aside
        ui, vi = np.rint(u).astype(int), np.rint(v).astype(int)
        seen = front & (ui >= 0) & (ui < intr.width) & (vi >= 0) & (vi < intr.height)
        if not np.any(seen):
            return 0
        d = np.full(centres.shape[0], np.nan, dtype=np.float32)
        d[seen] = np.asarray(depth, dtype=np.float32)[vi[seen], ui[seen]]
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
        self.weight[ix, iy, iz] = np.minimum(s.max_weight, w_new)
        if rgb is not None:
            near = t < 1.0  # within the truncation: the surface itself, not the ray's free run
            if np.any(near):
                self._blend_colour(
                    (ix[near], iy[near], iz[near]),
                    np.asarray(rgb)[vi[touch][near], ui[touch][near]],
                    w_obs[near],
                )
        return int(flat.size)

    def _blend_colour(self, at: tuple[Array, Array, Array], colour: Uint8, w_obs: Float32) -> None:
        """The colour of the voxels ``at`` moved toward ``colour`` by the same weighted average
        as the field, on the colour's own weight; blended in float, stored as bytes."""
        cw_old = self.colour_weight[at]
        cw_new = cw_old + w_obs
        old = self.rgb[at].astype(np.float32)
        mixed = (old * cw_old[:, None] + colour.astype(np.float32) * w_obs[:, None]) / cw_new[
            :, None
        ]
        self.rgb[at] = np.clip(np.rint(mixed), 0, 255).astype(np.uint8)
        self.colour_weight[at] = np.minimum(self.spec.max_weight, cw_new)

    # ---- readout -------------------------------------------------------------------------
    def surface(self, min_weight: float = 2.0) -> tuple[Array, Uint8]:
        """Points where the field crosses zero between two weighted neighbours, interpolated
        to the crossing along each axis: the model's surface as (n, 3) map points and colours
        (each point's colour from the neighbour nearer the surface, the smaller |sdf|)."""
        s = self.spec
        pts: list[Array] = []
        cols: list[Uint8] = []
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
            ia = np.argwhere(cross)
            va, vb = sa[cross], sb[cross]
            frac = va / (va - vb)
            idx = ia.astype(float)
            idx[:, axis] += frac
            pts.append((idx + 0.5) * s.voxel_m + np.array(s.origin))
            ib = ia.copy()
            ib[:, axis] += 1
            nearer = np.where((np.abs(vb) < np.abs(va))[:, None], ib, ia)
            cols.append(self.rgb[nearer[:, 0], nearer[:, 1], nearer[:, 2]])
        if not pts:
            return np.zeros((0, 3)), np.zeros((0, 3), dtype=np.uint8)
        return np.concatenate(pts), np.concatenate(cols)

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

    def score(
        self, points_map: Array, min_weight: float = 2.0, weights: Array | None = None
    ) -> tuple[float, int]:
        """The mean of ``fit_per_point`` over ALL the points (unknown ones count zero, so a
        turn that carries points out of the model loses), each point counting ``weights``
        (equal by default), and how many were on known ground."""
        fit, known = self.fit_per_point(points_map, min_weight)
        if fit.size == 0:
            return 0.0, 0
        if weights is None:
            return float(fit.mean()), int(known.sum())
        total = float(np.sum(weights))
        if total <= 0.0:
            return 0.0, int(known.sum())
        return float(np.dot(fit, weights) / total), int(known.sum())


YAW_SEARCH = tuple(math.radians(d) for d in np.arange(-4.0, 4.01, 0.5))
ALIGN_MIN_POINTS = 200  # fewer band points on known voxels: the model has nothing to say
ALIGN_MIN_GAIN = 0.02  # the best turn must beat "no turn" by this much of the score


class AlignReason(StrEnum):
    """Why ``align_yaw`` answered what it did. Only ALIGNED carries a turn to apply; AT_BOUND
    is the one answer a frame must not be integrated on."""

    ALIGNED = "aligned"  # a turn inside the search beat no turn: apply it
    FITS = "fits"  # no turn beats none by the minimum gain: the frame sits on the model
    UNJUDGED = "unjudged"  # too few band points on known ground: the model cannot judge
    AT_BOUND = "at_bound"  # the best turn is the search's edge: the truth may lie beyond


@dataclass(frozen=True)
class Alignment:
    """The verdict of ``align_yaw``: the turn (radians, 0 unless ALIGNED or AT_BOUND), its
    score gain over no turn, how many band points the model knew at no turn, and the reason."""

    yaw: float
    gain: float
    judged: int
    reason: AlignReason

    @property
    def aligned(self) -> bool:
        return self.reason is AlignReason.ALIGNED


def align_yaw(
    model: Tsdf,
    band_map: Array,
    pivot_xy: tuple[float, float],
    candidates: tuple[float, ...] = YAW_SEARCH,
) -> Alignment:
    """The turn about ``pivot_xy`` that seats the frame's band points best on the model,
    parabola-refined between the best three candidates (``candidates`` must include no turn).

    Every candidate is scored over the whole band, a point off known ground counting zero, so a
    turn cannot win by dropping points. Each point counts by its lever arm |p - pivot|: a turn
    of one degree moves a point 4 m out by 7 cm and one 0.5 m out by 9 mm, so the far wall
    carries the information about the turn and a sofa beside the cart, which only slides along
    itself, must not dilute it. The score's peak is a kink (a sum of |sdf| tents), so the
    parabola's vertex under-reaches it by up to a tenth of a degree — half a centimetre at 3 m.
    """
    zero = min(range(len(candidates)), key=lambda i: abs(candidates[i]))
    if abs(candidates[zero]) > 1e-12:
        raise ValueError("the yaw candidates must include no turn")
    pivot = np.array([pivot_xy[0], pivot_xy[1], 0.0])
    rel = band_map - pivot
    lever = np.hypot(rel[:, 0], rel[:, 1])  # unchanged by any turn about the pivot
    scores = []
    judged_at_zero = 0
    for i, yaw in enumerate(candidates):
        c, s = math.cos(yaw), math.sin(yaw)
        turned = np.stack(
            [c * rel[:, 0] - s * rel[:, 1], s * rel[:, 0] + c * rel[:, 1], rel[:, 2]], axis=1
        )
        score, n = model.score(turned + pivot, weights=lever)
        scores.append(score)
        if i == zero:
            judged_at_zero = n
    if judged_at_zero < ALIGN_MIN_POINTS:
        return Alignment(0.0, 0.0, judged_at_zero, AlignReason.UNJUDGED)
    best = int(np.argmax(scores))
    gain = float(scores[best] - scores[zero])
    if gain < ALIGN_MIN_GAIN:
        return Alignment(0.0, gain, judged_at_zero, AlignReason.FITS)
    if best == 0 or best == len(candidates) - 1:
        return Alignment(float(candidates[best]), gain, judged_at_zero, AlignReason.AT_BOUND)
    yaw = candidates[best]
    y0, y1, y2 = scores[best - 1], scores[best], scores[best + 1]
    denom = y0 - 2.0 * y1 + y2
    if denom < 0.0:  # the vertex of the parabola through the three best: toward the higher side
        step = candidates[best + 1] - candidates[best]
        yaw += 0.5 * (y0 - y2) / denom * step
    return Alignment(float(yaw), gain, judged_at_zero, AlignReason.ALIGNED)
