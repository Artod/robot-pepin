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

It says nothing about free space. A bearing with no surface in the band comes back NaN, which a
costmap neither marks nor clears: CLEARING stays with the single frames (``/depth_scan``), which
is where it belongs — a frame is an eyewitness of what is open right now, and the volume
remembers what was there rather than what is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pepin.depth import SCAN_MIN_Z_M, SCAN_STEP
from pepin.tsdf import Array, GridSpec, RigidPose, Tsdf

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

    @property
    def bins(self) -> int:
        """How many bearings the fan has: the whole turn at :attr:`step`."""
        return round(2 * math.pi / self.step)


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
