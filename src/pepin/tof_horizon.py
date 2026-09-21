"""How far a level ToF sensor may be believed before its own cone hits the floor, and how many
beams it takes to draw that cone on a costmap without leaving holes in it.

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


def cone_beams(ceiling_m: float, fov_rad: float, cell_m: float) -> int:
    """How many beams a cone of ``fov_rad`` needs to reach every ``cell_m`` cell inside it.

    A cone drawn as a fan of beams is a row of points on an arc, and the widest that arc ever
    gets is at the sensor's own ceiling: neighbouring beams stand ``r * fov / (n - 1)`` apart at
    range r. Asking that spacing to be at most one costmap cell at ``ceiling_m`` therefore
    covers every nearer range too, where the same fan is denser. So ``n - 1`` is the arc at the
    ceiling measured in cells, rounded up, and never below one — a cone is at least its own two
    edges. The front whisker (ceiling 0.96 m, 0.47 rad, 0.05 m cells) needs 0.45 m / 0.05 =
    9.01 -> 10 gaps, 11 beams; the two low ones (0.57 and 0.59 m) need 6 gaps, 7 beams.
    """
    if ceiling_m <= 0.0 or fov_rad <= 0.0 or cell_m <= 0.0:
        return 2
    return 1 + max(1, math.ceil(ceiling_m * fov_rad / cell_m))


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
