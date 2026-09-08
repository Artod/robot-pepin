"""How far a level ToF sensor may be believed before its own cone hits the floor.

A VL53L1X mounted level at height h with a cone of half-angle fov/2 starts illuminating the
floor at h / tan(fov/2): every reading beyond that distance can be the floor rather than an
obstacle. On this cart the two low sensors sit at 0.16 m with a 0.47 rad cone, so their floor
horizon is 0.67 m — and the right sensor's steady 0.60-0.70 m returns (43% dropouts, nothing
there for the lidar) were exactly that: grazing incidence, marked into the costmap as a wall.
"""

from __future__ import annotations

import math

FLOOR_MARGIN = 0.85  # believe the cone only up to 85% of the floor distance


def floor_horizon(height_m: float, fov_rad: float, margin: float = FLOOR_MARGIN) -> float:
    """The greatest range a level sensor at ``height_m`` may be trusted for, in metres."""
    if height_m <= 0.0 or fov_rad <= 0.0:
        return math.inf
    return margin * height_m / math.tan(fov_rad / 2.0)


def trusted_max_range(height_m: float, fov_rad: float, sensor_max_m: float) -> float:
    """The sensor's own ceiling, lowered to where the floor enters its cone."""
    return min(sensor_max_m, floor_horizon(height_m, fov_rad))
