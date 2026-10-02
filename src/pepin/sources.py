"""The names of the cart's sensors as every node spells them, and one shared time constant.

:data:`LIDAR` is the lidar's revolution, :data:`CAMERA` the camera's picture and depth, and
:data:`GRAPH` RTAB-Map's pose graph on the laptop. ``pepin_bringup.sensor_pack`` puts the first
two into a snapshot and ``pepin_bringup.rtabmap_frame`` names all three in its report.
:data:`RATE_TAU_S` is the time constant of a measured source rate (:class:`pepin.snapshot.Cadence`).

The board tracker's source roster and its scan feed are on the tag alt/tracker-2026-09-22.
"""

from __future__ import annotations

LIDAR = "lidar"
CAMERA = "camera"
GRAPH = "graph"
RATE_TAU_S = 2.0  # the rate's time constant: a few seconds of intervals, not the whole run
