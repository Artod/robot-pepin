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


class RangeHold:
    """What to publish for one ToF sensor: the last real return, held for ``hold_s`` after it stops.

    The range layer clears its whole cone the instant a sensor reports max range, and a VL53L1X
    with a person in front of it calls only half its frames a measurement — so a mark written
    one frame was erased the next, and the cart slowed and carried on (run 0016). Three frames
    of debounce (0.2 s) were shorter than the gaps between valid frames; a time window is not.
    A real empty room stays empty far longer than a second, so clearing still happens, just late
    enough to mean it.
    """

    def __init__(self, hold_s: float = 1.2) -> None:
        self._hold_s = hold_s
        self._last: dict[str, tuple[float, float]] = {}  # name -> (range, when)

    def publish(self, name: str, seen: float | None, ceiling: float, now: float) -> float:
        """The range to put on the wire now.

        ``seen`` is a real return (metres) or None; anything above ``ceiling`` counts as none.
        Returns the return itself, the held one while it is fresh, and ``ceiling`` ("nothing")
        once the hold has expired.
        """
        if seen is not None and seen <= ceiling:
            self._last[name] = (seen, now)
            return seen
        held = self._last.get(name)
        if held is not None and now - held[1] <= self._hold_s:
            return held[0]  # the evidence stands until it is old
        return ceiling
