"""A ToF fan as the volume reads it: a tiny depth image with its own pinhole, so the whiskers
write the TSDF through the very integrator the camera uses (:meth:`pepin.tsdf.Tsdf.integrate`).

One VL53L1X verdict is a cone of ``fov`` radians carrying ONE range: ``/tof/<name>/scan`` draws
it as a fan of equal beams (``pepin_bringup.tof_bridge.fan_scan``), +inf for "nothing within the
trusted range" (the fan's ``range_max``, :func:`pepin.tof_horizon.trusted_max_range`) and NaN
for "I do not know". Here the same cone is an n x n image whose pixels inside the cone's disc
all carry that range, looked at through a pinhole whose edges are the cone's edges. A return
then marks a surface at its range and carves the ray free up to it, exactly as a depth pixel
does; +inf becomes a depthless pixel, which :class:`pepin.tsdf.DepthLaw` carves free out to the
trusted range itself — "nothing within the ceiling" is certain, the sensor sees far past it —
and a hit writes no halo beyond that range, so no voxel a hit leaves can outlive the misses
that must carve it; "I do not know", and the corners of the square outside the cone, carry
:data:`SILENT_M`, a finite depth below the lens that the integrator writes nothing for.

THE WEIGHT. A ToF reading is a ranger's, not a picture's: its error does not shrink as the
surface comes nearer, so every fan weighs :data:`TOF_WEIGHT`, hit or miss, instead of the grid's
(ref / d)^2 (which would put a 0.4 m return at the cap, 4.0, and let five fans freeze a voxel).
With the volume's ``max_weight`` W = 20 (config/fusion.json) and the fans at ~15 Hz:

* a hit speaks in the costmap's marks once its voxel carries the marks' ``min_weight``: 4.0 on
  the fusion node (config/knobs.json), 4.0 / 0.5 = 8 fans, 0.5 s (:func:`fans_to_speak`);
* a voxel is saturated after W / 0.5 = 40 fans, 2.7 s — a pillow stared at for ten seconds is
  one;
* a saturated hit is carved by misses: each one moves the field 1 - (1 - t) * W / (W + w) toward
  free, so the deepest voxel a return writes (t -> -1, one truncation behind the surface) turns
  positive — and the zero crossing the marks read vanishes — after ln(2) / ln((W + w) / W) =
  ln 2 / ln 1.025 = 28.1, i.e. 29 fans, 1.9 s (:func:`fans_to_clear`). The surface voxel itself
  (t = 0) leaves "occupied" after ln(4/3) / ln 1.025 = 11.7 -> 12 fans, 0.8 s.

The camera's own rays through the spot carve it the same way at their own weight (a measured
pixel at 2 m weighs 1.0: 15 frames; at 1 m, 4.0: 4 frames).
"""

from __future__ import annotations

import math

import numpy as np

from pepin.depth import Intrinsics
from pepin.tsdf import Array, DepthLaw, RigidPose

TOF_WEIGHT = 0.5  # per fan, hit or miss (see the module docstring for the arithmetic)
TOF_RATE_HZ = 15.0  # what the board's ToF server delivers (pepin_bringup.tof_bridge)
SILENT_M = -1.0  # a pixel that says nothing: finite and below the lens, so nothing is written
# optical (x right, y down, z forward) axes in the sensor's frame (x forward, y left, z up):
# a ToF frame is a body frame, and the integrator thinks in a camera's
OPTICAL_IN_SENSOR: Array = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])


def tof_law(reach_m: float) -> DepthLaw:
    """How a ToF fan writes: a fixed weight per pixel, a miss that carves free out to the fan's
    trusted ``reach_m`` weighing as much as a hit (the bridge has already held a dropped frame
    for 1.2 s before calling it a miss, pepin.tof_horizon), and a hit that writes nothing
    beyond that reach."""
    return DepthLaw(
        no_depth_free=True,
        no_depth_weight=1.0,
        reach_m=reach_m,
        hit_weight=TOF_WEIGHT,
        carve_to_reach=True,
    )


def fans_to_speak(min_weight: float, weight: float = TOF_WEIGHT) -> int:
    """How many fans a fresh voxel needs before the marks may read it: ``min_weight`` in
    observations of ``weight``."""
    return math.ceil(min_weight / weight - 1e-9)


def fans_to_clear(max_weight: float, weight: float = TOF_WEIGHT, t_from: float = -1.0) -> int:
    """How many misses of ``weight`` turn a saturated voxel's field from ``t_from`` (truncation
    units; -1 is the deepest a return writes) to positive, i.e. free of any zero crossing."""
    return math.ceil(math.log(1.0 - t_from) / math.log((max_weight + weight) / max_weight) - 1e-9)


def optical_pose(sensor_in_map: RigidPose) -> RigidPose:
    """``map <- optical`` for a sensor placed at ``sensor_in_map`` (``map <- tof_*``)."""
    return RigidPose(sensor_in_map.rotation @ OPTICAL_IN_SENSOR, sensor_in_map.translation)


def fan_image(ranges: Array, angle_min: float, angle_increment: float) -> tuple[Array, Intrinsics]:
    """One fan as an n x n depth image (metres, optical frame) and the pinhole that sees it,
    n being the fan's beam count: pixel (row, col) carries the beam nearest its own bearing
    inside the cone's disc and :data:`SILENT_M` outside it; +inf becomes NaN (a depthless pixel,
    the carving kind), NaN stays silent. A return nearer than :data:`pepin.depth.NEAR_M` is
    inside the integrator's dead zone and writes nothing either."""
    r = np.asarray(ranges, dtype=float)
    n = int(r.size)
    half = (n - 1) * angle_increment / 2.0  # the cone's half-angle: the fan spans it edge to edge
    centre = (n - 1) / 2.0
    f = n / (2.0 * math.tan(half)) if half > 0.0 else float(n)  # the pixels' outer edges = cone
    intr = Intrinsics(fx=f, fy=f, cx=centre, cy=centre, width=n, height=n)
    cols = np.arange(n, dtype=float)
    # optical x grows to the right, which is the sensor's -y: a column's bearing runs backwards
    bearing = -np.arctan((cols - centre) / f)
    beam = np.clip(np.rint((bearing - angle_min) / angle_increment).astype(int), 0, n - 1)
    rb = r[beam]
    value = np.where(np.isinf(rb), np.nan, np.where(np.isnan(rb), SILENT_M, rb))
    depth = np.broadcast_to(value[None, :], (n, n)).copy()
    rows = np.arange(n, dtype=float)
    inside = ((cols[None, :] - centre) ** 2 + (rows[:, None] - centre) ** 2) <= (n / 2.0) ** 2
    depth[~inside] = SILENT_M
    return depth, intr
