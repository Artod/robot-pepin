"""The cart's real outline, the one source every clearance question on the robot derives from.

The cart is 0.30 m long behind its drive wheels and 0.55 m wide, so a turn in place sweeps a
0.41 m radius at the rear. :class:`Footprint` keeps that outline as a rectangle in the robot
frame; :data:`HULL` is the cart, and :func:`hull_box` the box the lidar's scan filter cuts out.
Nav2 owns collision checking (the pre-ROS hull guard is in git history).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

# The Nav2 costmaps' resolution (tests/unit/test_nav_contract pins the YAML to it). It lives
# beside the hull because every clearance question on this cart is the hull measured in these
# cells: the contact band below is 1.5 of them, the inscribed band the point planners plan with
# is a little over one.
COSTMAP_CELL_M = 0.05

# Anything closer than this to the hull is contact, not an obstacle. The cart parks bumper-to-
# furniture on purpose, and a mark inside this band lands in the very cells of the footprint
# outline that the controller checks first: one such cell refused every command, including the
# one that drives away (run 0087, 231 s beside the printer). 8 cm is 1.5 costmap cells at 5 cm,
# so a mark just outside the band never shares a cell with the outline. Both the lidar's hull
# box filter and the ToF ranges (min_range) apply it.
CONTACT_BAND_M = 0.08


@dataclass(frozen=True)
class Footprint:
    """The hull as a rectangle in the robot frame (origin between the drive wheels, x forward).

    The defaults are the cart as measured on 2026-09-04 — the same numbers as the ``footprint``
    block of ``config/base.json``, which the base server reads; ``tests/unit/test_footprint.py``
    keeps the two equal. Everything else that needs the shape (the Nav2 polygon, the scan
    filter's box, the contact band) derives from :data:`HULL`.
    """

    front_m: float = 0.0625  # the drive wheels are the front of the cart
    rear_m: float = 0.30
    half_width_m: float = 0.275
    margin_m: float = 0.03  # the lidar's cell and the hull's own inaccuracy

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> Footprint:
        """From the ``footprint`` block of ``config/base.json``."""
        return cls(
            front_m=float(data["front_m"]),
            rear_m=float(data["rear_m"]),
            half_width_m=float(data["half_width_m"]),
            margin_m=float(data.get("margin_m", cls.margin_m)),
        )

    @property
    def swing_radius_m(self) -> float:
        """Farthest hull corner from the axle centre: what a turn in place sweeps."""
        return math.hypot(self.rear_m, self.half_width_m)

    @property
    def circumscribed_radius_m(self) -> float:
        """The circle around the whole hull, centred on base_link (Nav2's circumscribed radius)."""
        return self.swing_radius_m

    @property
    def inscribed_radius_m(self) -> float:
        """The largest circle inside the hull centred on base_link: the nearest edge. Nav2's point
        planners keep their path this far from any lethal cell and no farther — 6 cm here, since
        the axle sits at the front edge."""
        return min(self.front_m, self.rear_m, self.half_width_m)

    def polygon(self) -> list[tuple[float, float]]:
        """The four corners in base_link, the order Nav2's ``footprint`` parameter carries."""
        f, r, w = self.front_m, self.rear_m, self.half_width_m
        return [(f, w), (f, -w), (-r, -w), (-r, w)]


HULL = Footprint()  # the cart; the one instance the rest of the stack derives its shape from


def hull_box(hull: Footprint = HULL, band_m: float = CONTACT_BAND_M) -> dict[str, float]:
    """The hull grown by ``band_m`` on every side, as the box (min_x, max_x, min_y, max_y) in
    base_link that a scan filter cuts out: returns inside it are the cart or what it touches."""
    return {
        "min_x": -(hull.rear_m + band_m),
        "max_x": hull.front_m + band_m,
        "min_y": -(hull.half_width_m + band_m),
        "max_y": hull.half_width_m + band_m,
    }


def hull_cells(
    x: float,
    y: float,
    yaw_rad: float,
    origin: tuple[float, float],
    resolution: float,
    hull: Footprint = HULL,
) -> set[tuple[int, int]]:
    """The cells ``(col, row)`` of an unrotated grid (``origin`` its corner on the map, cells
    ``resolution`` wide) that the hull covers standing at map pose ``(x, y, yaw_rad)``: the
    whole rectangle, sampled every half cell with its edges included (a leg inside the outline
    counts as much as one on it)."""
    length, width = hull.front_m + hull.rear_m, 2.0 * hull.half_width_m
    along = max(1, math.ceil(length / (resolution / 2.0)))
    across = max(1, math.ceil(width / (resolution / 2.0)))
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    cells: set[tuple[int, int]] = set()
    for i in range(along + 1):
        forward = -hull.rear_m + length * i / along
        for j in range(across + 1):
            left = -hull.half_width_m + width * j / across
            mx, my = x + c * forward - s * left, y + s * forward + c * left
            col = math.floor((mx - origin[0]) / resolution)
            cells.add((col, math.floor((my - origin[1]) / resolution)))
    return cells
