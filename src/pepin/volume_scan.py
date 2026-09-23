"""The accumulated volume read out as a planar fan: what the model already holds, by bearing.

A single depth frame is one opinion. ``pepin.depth.depth_to_scan`` turns one frame into the fan
Nav2 marks with, and a stereo head on a herringbone floor puts a few blobs of wrong disparity
into every one of them: floor pixels lifted 15-25 cm, about one false bearing a frame, a
different one each time — which in the costmap is a lethal cell the lidar never saw, and on the
first stereo drive (tape ros/maps/rec/0415_*) 100-300 of them, 115 "collision ahead" a minute and
44 recoveries.

The volume is the same pixels with the disagreement taken out: every frame is fused into a TSDF
by a weighted average and every ray carves the free space it crosses (:mod:`pepin.tsdf`,
:mod:`pepin.worldmap`), so a blob that one frame invented is a weak, contradicted voxel while a
table top that ten frames agree on is a surface. This module reads THAT — the very surface
:meth:`pepin.tsdf.Tsdf.surface` publishes as ``/fusion/surface``, with the same ``min_weight``,
so a mark in the costmap and a point in Foxglove can never disagree — over a horizontal band
around the cart, and answers the nearest one per half-degree of bearing.

AND SINCE 2026-09-22 IT ALSO SAYS WHERE IT IS OPEN (:func:`free_ranges`). The fan used to mark
only: a bearing with no surface came back NaN, which a costmap neither marks nor clears, and the
sole clearing source was the single frame's own ``/depth_scan`` — the head's forward 83 degrees.
A fan that marks over the whole turn and clears over a sixth of it is a ratchet: measured on the
turn of run 0434 (scratch/one_localiser/tape_0434_turn.py), unbacked lethal cells were born at
109/s while the cart turned against 27/s standing, 52 % of them BEHIND the cart where nothing
could ever raytrace them away, 76 % still lethal three seconds later, and the count climbed from
24 to 904 in 38 s until Nav2 cleared the whole costmap. So the same walk that finds the mark also
reports how far the volume is KNOWN OPEN along that bearing, and that answer goes out as a
clearing fan of its own (``/depth_free``).

The two answers are separate topics because a LaserScan cannot carry them on one: Nav2's
ObstacleLayer marks at the END of every finite range it is given and clears up to it, so a single
source that clears a ray at 1.2 m also plants a lethal cell at 1.2 m — at the FRONTIER OF
KNOWLEDGE, which is the very defect being cured. A marking-only fan and a clearing-only fan in
one layer say the two things without either inventing the other.

A bearing the volume has not observed stays NaN in both, and NaN is the one word for "unknown":
the ray is not clear, and unknown must stay unknown.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pepin.depth import SCAN_MIN_Z_M, SCAN_STEP
from pepin.tsdf import Array, GridSpec, RigidPose, Tsdf
from pepin.worldmap import SliceLaw

# The fan: the whole turn, on /depth_scan's own half-degree grid. The whole turn because the
# volume remembers what is BEHIND the head's current view — a table the cart has driven past is
# still in the model, and the costmap layer that marks from here should hear about it — while a
# frame can only ever speak for the camera's own +-40 degrees.
MARKS_STEP = SCAN_STEP
MARKS_ANGLE_MIN = -math.pi
MARKS_BINS = round(2 * math.pi / MARKS_STEP)
# How far the fan reaches. 3.0 m is /depth_scan's own cap (pepin.depth.depth_to_scan's
# max_range) and the camera layer's raytrace range, so the two scans speak about the same
# window; the grid's own edge cuts it shorter wherever the volume ends.
MARKS_RANGE_M = 3.0
# A surface nearer than one voxel of the cart's centre is the cart itself or a quantisation
# artefact of its own footprint, and a range of zero is not a bearing at all.
MARKS_MIN_RANGE_M = 0.05
# The band's floor: the height /depth_scan marks from (pepin.depth), which is also what
# config/fusion.json's camera_band_m carries as the band the camera speaks for. The top is the
# volume's own (that same camera_band_m), because the volume is what is being read.
MARKS_MIN_Z_M = SCAN_MIN_Z_M
MARKS_MAX_Z_M = 1.30
# How open a column must be before the clearing fan may cross it: :class:`pepin.worldmap.SliceLaw`
# — the ONE place this project decides when a column of the volume is occupied, free or unknown,
# and the same law the map slice is drawn by. Half a truncation from anything is free; within half
# a voxel of a crossing is a surface; in between is the surface's halo, which is neither.
MARKS_FREE_ABOVE = SliceLaw().free_above
# The cart's own footprint, in metres: the volume can never observe the voxels the robot stands
# in (the camera's near range starts well past them) and a ray that had to begin on known-free
# ground would therefore never begin at all. Samples nearer than this are crossed unless the
# volume holds a SURFACE there — a cell nobody can see is not a cell anybody may be blocked by.
# It changes nothing about what is erased: Nav2 raytraces from the sensor's origin, so a ray
# published at all clears from the cart outward whatever this is.
MARKS_FREE_SKIP_M = 0.30


@dataclass(frozen=True)
class MarksLaw:
    """What a bearing of the fan answers with: which voxels count as a surface, the height band
    they must stand in and how far the fan reaches.

    ``min_weight`` is not this module's own idea of maturity — it is the node's ``min_weight``,
    the one number that decides what the volume calls a surface at all
    (:meth:`pepin.tsdf.Tsdf.surface`, ``/fusion/surface``). One criterion, two consumers.

    ``band_m`` is metres above the cart's own floor plane (base_link's z), the convention
    ``/depth_scan``'s ``min_z``/``max_z`` already use, so the two scans mark in one band.
    """

    min_weight: float = 2.0
    band_m: tuple[float, float] = (MARKS_MIN_Z_M, MARKS_MAX_Z_M)
    range_m: float = MARKS_RANGE_M
    step: float = MARKS_STEP
    free_above: float = MARKS_FREE_ABOVE
    free_skip_m: float = MARKS_FREE_SKIP_M

    @property
    def bins(self) -> int:
        """How many bearings the fan has: the whole turn at :attr:`step`."""
        return round(2 * math.pi / self.step)

    def column_law(self) -> SliceLaw:
        """When a column of the volume counts as occupied, free or unknown for this fan: the
        project's own :class:`pepin.worldmap.SliceLaw`, carrying this fan's ``min_weight`` so the
        marks, the clearing and the map slice can never disagree about what a surface is."""
        return SliceLaw(min_weight=self.min_weight, free_above=self.free_above)


def marks_box(
    spec: GridSpec, at: tuple[float, float], z_m: tuple[float, float], range_m: float
) -> tuple[slice, slice, slice] | None:
    """The voxel index box the fan is read out of — the square of ``range_m`` about ``at`` and
    the height band ``z_m``, in map metres, clipped to the grid; ``None`` when the cart's
    surroundings are not in this volume at all.

    One voxel of margin on every side, because a crossing is found between two neighbours: a
    wall in the last column of a box cut exactly to the range would have no partner to cross
    with and would be silent (:meth:`pepin.tsdf.Tsdf.surface`).
    """
    lo_m = np.array([at[0] - range_m, at[1] - range_m, z_m[0]])
    hi_m = np.array([at[0] + range_m, at[1] + range_m, z_m[1]])
    origin = np.array(spec.origin)
    shape = np.array(spec.shape)
    lo = np.maximum(np.floor((lo_m - origin) / spec.voxel_m).astype(int) - 1, 0)
    hi = np.minimum(np.ceil((hi_m - origin) / spec.voxel_m).astype(int) + 1, shape)
    if np.any(hi - lo < 2):  # fewer than two voxels on an axis: nothing can cross
        return None
    return (
        slice(int(lo[0]), int(hi[0])),
        slice(int(lo[1]), int(hi[1])),
        slice(int(lo[2]), int(hi[2])),
    )


def empty_marks(law: MarksLaw | None = None) -> Array:
    """A fan that says nothing: NaN on every bearing — what a cart outside its own volume
    answers, and what a costmap neither marks nor clears from."""
    law = law if law is not None else MarksLaw()
    return np.full(law.bins, np.nan)


def marks_window(volume: Tsdf, base_in_map: RigidPose, law: MarksLaw | None = None) -> Tsdf | None:
    """The few megabytes of the volume :func:`marks_ranges` will read — the box about the cart —
    copied out as a volume of its own (:meth:`pepin.tsdf.Tsdf.window`); ``None`` when the cart's
    surroundings are not in this volume at all.

    THIS is the call that belongs under the model's lock, and nothing else: the copy is a
    fraction of a millisecond while the crossing search over the same box is milliseconds, and
    the fan read from the twin is the fan the whole volume would have answered (the box carries
    its own margin, so the search inside it finds the very same crossings).
    """
    law = law if law is not None else MarksLaw()
    x0, y0, z0 = (float(v) for v in base_in_map.translation)
    box = marks_box(volume.spec, (x0, y0), (z0 + law.band_m[0], z0 + law.band_m[1]), law.range_m)
    return None if box is None else volume.window(box)


def marks_ranges(volume: Tsdf, base_in_map: RigidPose, law: MarksLaw | None = None) -> Array:
    """The volume's surface around the cart as one range per bearing, in base_link: the nearest
    surface point standing in the band, metres, NaN where the model holds none.

    ``base_in_map`` is the cart's own pose (map <- base_link), the one the observation just
    integrated was placed by. Only its yaw and its origin are used: the fan is planar, as every
    LaserScan on this robot is, so a bearing is the true bearing of what it holds however the
    body leans, and the height band is measured from the cart's own floor plane.

    Vectorised over the box (:func:`marks_box`): the crossings are found once, filtered by band
    and range, and the nearest per bearing is taken by one sort. ``volume`` may be the whole
    model or the twin :func:`marks_window` copied out of it — the answer is the same fan.
    """
    law = law if law is not None else MarksLaw()
    bins = law.bins
    ranges: Array = np.full(bins, np.nan)
    x0, y0, z0 = (float(v) for v in base_in_map.translation)
    band = (z0 + law.band_m[0], z0 + law.band_m[1])
    box = marks_box(volume.spec, (x0, y0), band, law.range_m)
    if box is None:
        return ranges
    points, _colours = volume.surface(law.min_weight, box)
    if points.shape[0] == 0:
        return ranges
    dx, dy, z = points[:, 0] - x0, points[:, 1] - y0, points[:, 2]
    rng = np.hypot(dx, dy)
    keep = (z >= band[0]) & (z <= band[1]) & (rng <= law.range_m) & (rng >= MARKS_MIN_RANGE_M)
    if not np.any(keep):
        return ranges
    yaw = math.atan2(float(base_in_map.rotation[1, 0]), float(base_in_map.rotation[0, 0]))
    bearing = np.arctan2(dy[keep], dx[keep]) - yaw
    # the bearing's bin, the fan's first bin sitting at -pi and the turn closing on itself
    index = (np.rint(np.mod(bearing + math.pi, 2 * math.pi) / law.step).astype(int)) % bins
    near = rng[keep]
    order = np.lexsort((near, index))
    index, near = index[order], near[order]
    first = np.flatnonzero(np.r_[True, index[1:] != index[:-1]])
    ranges[index[first]] = near[first]
    return ranges


def band_columns(
    volume: Tsdf, box: tuple[slice, slice, slice], band: tuple[float, float], law: MarksLaw
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]] | None:
    """The height band of ``box`` squashed onto the floor as two masks over its x-y footprint —
    (occupied, open) — by :meth:`MarksLaw.column_law`; ``None`` when the band misses the grid.

    This is :meth:`pepin.worldmap.WorldMap.slice` on one neighbourhood instead of the whole room,
    and deliberately the same three words: a column is OCCUPIED when some voxel of it sits within
    half a voxel of the zero crossing, OPEN when the nearest thing in it is at least half a
    truncation away, and neither in the halo around a surface or where nothing was observed. The
    band is cut on the grid's own z lattice exactly as the slice cuts it, so a column here and a
    cell of the map slice are the same column.
    """
    s = volume.spec
    nz = s.shape[2]
    lo = max(0, math.floor((band[0] - s.origin[2]) / s.voxel_m))
    hi = min(nz, math.ceil((band[1] - s.origin[2]) / s.voxel_m))
    hi = max(hi, lo + 1) if lo < nz else nz
    if lo >= hi:
        return None
    sel = (box[0], box[1], slice(lo, hi))
    known = volume.weight[sel] >= law.min_weight
    sdf = volume.sdf[sel]
    nearest = np.where(known, sdf, np.inf).min(axis=2)  # how open the column is
    crossing = np.where(known, np.abs(sdf), np.inf).min(axis=2)  # how near a surface it is
    column = law.column_law()
    occupied = crossing <= column.occupied_t(s)
    free = np.isfinite(nearest) & (nearest >= column.free_above) & ~occupied
    return occupied, free


def free_ranges(volume: Tsdf, base_in_map: RigidPose, law: MarksLaw | None = None) -> Array:
    """How far each bearing of the fan is KNOWN OPEN, in base_link metres: the range of the last
    column the volume has observed as free before the first column it has not, NaN where the
    volume cannot vouch for even the first step.

    THE CLEARING HALF of the fan (``/depth_free``), and it is a walk and not a sort: the ray is
    stepped outward half a voxel at a time over :func:`band_columns`, and it ends at the first
    column that is not open — a surface, the halo in front of one, a column nobody has looked
    into, or the edge of the volume. What is returned is the last OPEN sample before that end, so
    the answer is always a range the model itself observed free, never the reach it was asked
    for. A bearing whose very first column stops the ray answers NaN, and NaN clears nothing:
    beyond what the volume knows, the costmap must keep what it has.

    A bearing that also MARKS (:func:`marks_ranges`) gets an answer here too, one column short of
    its surface: the space in front of a wall the volume painted was carved by the very rays that
    painted it, and clearing it is how the camera's own stale marks die.
    """
    law = law if law is not None else MarksLaw()
    free: Array = np.full(law.bins, np.nan)
    x0, y0, z0 = (float(v) for v in base_in_map.translation)
    band = (z0 + law.band_m[0], z0 + law.band_m[1])
    box = marks_box(volume.spec, (x0, y0), band, law.range_m)
    if box is None:
        return free
    columns = band_columns(volume, box, band, law)
    if columns is None:
        return free
    occupied, open_ = columns
    s = volume.spec
    step_m = 0.5 * s.voxel_m  # half a voxel: no column of the ray can be stepped over
    samples = max(1, math.floor(law.range_m / step_m))
    along = (np.arange(samples) + 1) * step_m
    yaw = math.atan2(float(base_in_map.rotation[1, 0]), float(base_in_map.rotation[0, 0]))
    angle = yaw + MARKS_ANGLE_MIN + law.step * np.arange(law.bins)
    xs = x0 + np.cos(angle)[:, None] * along[None, :]
    ys = y0 + np.sin(angle)[:, None] * along[None, :]
    i0, j0 = box[0].indices(s.shape[0])[0], box[1].indices(s.shape[1])[0]
    ii = np.floor((xs - s.origin[0]) / s.voxel_m).astype(int) - i0
    jj = np.floor((ys - s.origin[1]) / s.voxel_m).astype(int) - j0
    nx, ny = occupied.shape
    inside = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
    ci, cj = np.clip(ii, 0, nx - 1), np.clip(jj, 0, ny - 1)
    is_open = inside & open_[ci, cj]
    is_wall = inside & occupied[ci, cj]
    # the cart's own footprint: unobservable, so not a reason to withhold a ray — but a SURFACE
    # there still ends it
    near = along <= law.free_skip_m
    blocked = ~(is_open | (near[None, :] & ~is_wall))
    ends_at = np.where(blocked.any(axis=1), blocked.argmax(axis=1), samples)
    seen = is_open & (np.arange(samples)[None, :] < ends_at[:, None])
    vouched = seen.any(axis=1)
    last = samples - 1 - np.argmax(seen[:, ::-1], axis=1)
    free[vouched] = along[last[vouched]]
    free[free < MARKS_MIN_RANGE_M] = np.nan  # a ray of no length is not a ray
    return free


def fan_counts(marks: Array, free: Array) -> tuple[int, int, int]:
    """One fan in three numbers — how many bearings MARK, how many CLEAR and how many say
    nothing at all — for a node's report line."""
    marked, cleared = np.isfinite(marks), np.isfinite(free)
    return int(marked.sum()), int(cleared.sum()), int((~marked & ~cleared).sum())
