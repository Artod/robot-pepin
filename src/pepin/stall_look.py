"""What blocks the cart at a stall, where to look to check it, and what the look found.

When the controller gives up (FollowPath fails) the behaviour tree asks the gaze arbiter for a
stall look before it clears and replans. The arithmetic of that look lives here, ROS-free:

1. :func:`path_ahead` walks the global plan from the pose nearest the cart for the first metre;
2. :func:`find_blockers` sweeps the hull (:data:`pepin.footprint.HULL`, grown by a margin) along
   those poses over the local costmap and keeps the lethal cells it covers, each with the arc
   length at which the hull first reaches it, split as :mod:`pepin.marks_audit` splits them:
   lidar-backed, camera-only, unexplained. A lidar-backed cell is almost never a phantom, so the
   candidates for a look are the others, nearest first (:meth:`Blockers.candidates`);
3. the volume answers for those cells' columns (depth_fusion's ``/fusion/column``): the surface
   points standing over them with their weights; :func:`centroid` is the 3D point to look at;
4. after the look the same columns are asked again, and :func:`evidence` before against after is
   the verdict (:func:`verdict`): carved, partly carved or confirmed — and the lidar-backed cells
   are compared too, because a look that carves a cell the lidar backs is a false carve.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pepin.footprint import HULL, Footprint
from pepin.marks_audit import LETHAL, nearest_distance

Array = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]

AHEAD_M = 1.0  # how far along the plan the hull is swept
STEP_M = 0.025  # half a costmap cell between swept poses: no cell is skipped
MARGIN_M = 0.05  # the hull grown by one costmap cell: the controller's own check is not exact
CLUSTER_M = 0.25  # candidates first reached within this much of the nearest one are one blocker


def path_ahead(
    path_xy: Array, cart_xy: tuple[float, float], ahead_m: float = AHEAD_M, step_m: float = STEP_M
) -> tuple[Array, Array]:
    """Poses (k, 3: x, y, heading) every ``step_m`` along the plan from the vertex nearest the
    cart for ``ahead_m``, and their arc lengths (k,). The heading is the plan's own direction (a
    NavFn plan carries no orientation); a plan of fewer than two points gives nothing."""
    path = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if len(path) < 2:
        return np.zeros((0, 3)), np.zeros(0)
    start = int(np.argmin(np.hypot(path[:, 0] - cart_xy[0], path[:, 1] - cart_xy[1])))
    rest = path[start:]
    if len(rest) < 2:
        rest = path[-2:]
    seg = np.diff(rest, axis=0)
    seg_len = np.hypot(seg[:, 0], seg[:, 1])
    keep = seg_len > 1e-9
    if not np.any(keep):
        return np.zeros((0, 3)), np.zeros(0)
    seg, seg_len, starts = seg[keep], seg_len[keep], rest[:-1][keep]
    cum = np.concatenate([[0.0], np.cumsum(seg_len)])
    s: Array = np.arange(0.0, min(ahead_m, float(cum[-1])) + 1e-9, step_m, dtype=float)
    index = np.clip(np.searchsorted(cum, s, side="right") - 1, 0, len(seg) - 1)
    frac = (s - cum[index]) / seg_len[index]
    xy = starts[index] + seg[index] * frac[:, None]
    heading = np.arctan2(seg[index, 1], seg[index, 0])
    poses: Array = np.column_stack((xy, heading)).astype(float)
    return poses, s


def swept_cells(
    shape: tuple[int, int],
    origin: tuple[float, float],
    resolution: float,
    poses: Array,
    arc_m: Array,
    hull: Footprint = HULL,
    margin_m: float = MARGIN_M,
) -> tuple[npt.NDArray[np.int64], npt.NDArray[np.int64], Array]:
    """The grid cells (rows, cols) whose centres the hull covers at any of ``poses``, and the
    arc length at which it first does; ``shape`` is (height, width), ``origin`` its corner."""
    height, width = shape
    front, rear = hull.front_m + margin_m, hull.rear_m + margin_m
    half = hull.half_width_m + margin_m
    reach = math.hypot(max(front, rear), half)
    first: dict[tuple[int, int], float] = {}
    for (x, y, heading), s in zip(poses, arc_m, strict=True):
        c0 = max(int((x - reach - origin[0]) / resolution), 0)
        c1 = min(int((x + reach - origin[0]) / resolution) + 1, width)
        r0 = max(int((y - reach - origin[1]) / resolution), 0)
        r1 = min(int((y + reach - origin[1]) / resolution) + 1, height)
        if c0 >= c1 or r0 >= r1:
            continue
        rows, cols = np.mgrid[r0:r1, c0:c1]
        dx = origin[0] + (cols + 0.5) * resolution - x
        dy = origin[1] + (rows + 0.5) * resolution - y
        cos, sin = math.cos(heading), math.sin(heading)
        along, across = cos * dx + sin * dy, -sin * dx + cos * dy
        inside = (along <= front) & (along >= -rear) & (np.abs(across) <= half)
        for row, col in zip(rows[inside].tolist(), cols[inside].tolist(), strict=True):
            first.setdefault((row, col), float(s))
    if not first:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), np.zeros(0)
    keys = np.array(list(first.keys()), dtype=np.int64)
    return keys[:, 0], keys[:, 1], np.array(list(first.values()))


@dataclass(frozen=True)
class Blockers:
    """The lethal cells the swept hull covers: centres (n, 2) in the grid's frame, the arc
    length at which the hull first reaches each, and who can account for them."""

    xy: Array
    s_m: Array
    lidar: Mask
    camera: Mask
    resolution: float

    @property
    def count(self) -> int:
        """How many lethal cells block."""
        return len(self.xy)

    @property
    def unexplained(self) -> Mask:
        """Cells neither the lidar nor the camera's marks account for (the ToF, a stale mark)."""
        return ~self.lidar & ~self.camera

    def candidates(self, cluster_m: float = CLUSTER_M) -> Mask:
        """The cells worth a look: not the lidar's, and first reached within ``cluster_m`` of
        the nearest such cell — the one blocker the cart meets first."""
        free = ~self.lidar
        if not np.any(free):
            return free
        nearest = float(self.s_m[free].min())
        mask: Mask = free & (self.s_m <= nearest + cluster_m)
        return mask

    def describe(self, ahead_m: float) -> str:
        """The cells in words for the verdict line."""
        if self.count == 0:
            return f"no lethal cell under the hull in the first {ahead_m:.1f} m"
        return (
            f"{self.count} lethal cells under the hull in the first {ahead_m:.1f} m (lidar"
            f" {int(self.lidar.sum())}, camera-only {int(self.camera.sum())}, unexplained"
            f" {int(self.unexplained.sum())}), the first at {float(self.s_m.min()):.2f} m"
        )


def find_blockers(
    grid: npt.NDArray[np.integer],
    origin: tuple[float, float],
    resolution: float,
    path_xy: Array,
    cart_xy: tuple[float, float],
    lidar_xy: Array,
    camera_xy: Array,
    *,
    ahead_m: float = AHEAD_M,
    margin_m: float = MARGIN_M,
    match_cells: float = 1.5,
    hull: Footprint = HULL,
) -> Blockers:
    """The lethal cells of ``grid`` (ROS 0..100, rows by y) the hull covers along the first
    ``ahead_m`` of the plan, classified against the lidar's returns and the camera's marks
    (both placed in the grid's frame) as :func:`pepin.marks_audit.audit_marks` does."""
    poses, arc = path_ahead(path_xy, cart_xy, ahead_m)
    values = np.asarray(grid)
    rows, cols, s = swept_cells(values.shape, origin, resolution, poses, arc, hull, margin_m)
    hard = values[rows, cols] >= LETHAL if len(rows) else np.zeros(0, dtype=bool)
    rows, cols, s = rows[hard], cols[hard], s[hard]
    xy = np.column_stack(
        (origin[0] + (cols + 0.5) * resolution, origin[1] + (rows + 0.5) * resolution)
    ).astype(float)
    tolerance = match_cells * resolution
    lidar = nearest_distance(xy, lidar_xy) <= tolerance
    camera = ~lidar & (nearest_distance(xy, camera_xy) <= tolerance)
    return Blockers(xy.reshape(-1, 2), s, lidar, camera, resolution)


# ---- the volume's columns ----------------------------------------------------------------------
@dataclass(frozen=True)
class ColumnBox:
    """An axis-aligned box in the volume's frame: centre and side lengths, metres."""

    centre: tuple[float, float, float]
    size: tuple[float, float, float]


def column_box(cells_xy: Array, resolution: float, z_range: tuple[float, float]) -> ColumnBox:
    """The box that holds the columns of ``cells_xy`` between the two heights."""
    xy = np.asarray(cells_xy, dtype=float).reshape(-1, 2)
    lo, hi = xy.min(axis=0) - resolution, xy.max(axis=0) + resolution
    z0, z1 = z_range
    centre = (float(lo[0] + hi[0]) / 2, float(lo[1] + hi[1]) / 2, (z0 + z1) / 2)
    return ColumnBox(centre, (float(hi[0] - lo[0]), float(hi[1] - lo[1]), z1 - z0))


def in_columns(points: Array, cells_xy: Array, resolution: float) -> Mask:
    """Which points (n, 3) stand over one of the cells: within half a cell of its centre."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    cells = np.asarray(cells_xy, dtype=float).reshape(-1, 2)
    if len(pts) == 0 or len(cells) == 0:
        return np.zeros(len(pts), dtype=bool)
    half = resolution / 2 + 1e-9
    dx = np.abs(pts[:, 0, None] - cells[None, :, 0])
    dy = np.abs(pts[:, 1, None] - cells[None, :, 1])
    mask: Mask = np.any((dx <= half) & (dy <= half), axis=1)
    return mask


def centroid(points: Array, weights: Array) -> tuple[float, float, float] | None:
    """The weighted mean of the points, or None when there are none (or no weight)."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    w = np.asarray(weights, dtype=float).reshape(-1)
    total = float(w.sum()) if len(w) else 0.0
    if len(pts) == 0 or total <= 0.0:
        return None
    mean = (pts * w[:, None]).sum(axis=0) / total
    return float(mean[0]), float(mean[1]), float(mean[2])


@dataclass(frozen=True)
class Evidence:
    """What the volume holds over a set of cells: how many hold a surface, and its weight."""

    cells: int
    occupied: int
    weight: float

    def text(self) -> str:
        """``3/4 cells, weight 61``."""
        return f"{self.occupied}/{self.cells} cells, weight {self.weight:.0f}"


def evidence(points: Array, weights: Array, cells_xy: Array, resolution: float) -> Evidence:
    """How many of the cells have a surface point standing over them, and the points' weight."""
    pts = np.asarray(points, dtype=float).reshape(-1, 3)
    w = np.asarray(weights, dtype=float).reshape(-1)
    cells = np.asarray(cells_xy, dtype=float).reshape(-1, 2)
    occupied, weight = 0, 0.0
    half = resolution / 2 + 1e-9
    for cx, cy in cells:
        here = (np.abs(pts[:, 0] - cx) <= half) & (np.abs(pts[:, 1] - cy) <= half)
        if np.any(here):
            occupied += 1
            weight += float(w[here].sum())
    return Evidence(len(cells), occupied, weight)


def verdict(before: Evidence, after: Evidence) -> str:
    """``carved`` (nothing stands over the cells any more), ``confirmed`` (as many as before)
    or ``partly carved``; ``empty`` when nothing stood there to begin with."""
    if before.occupied == 0:
        return "empty"
    if after.occupied == 0:
        return "carved"
    if after.occupied >= before.occupied:
        return "confirmed"
    return "partly carved"
