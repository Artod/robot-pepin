"""Near-field safety from the lidar: is anything inside the box the robot is about to drive into?

The map may not contain a table leg seen only from a few angles; the live
scan does. This check is independent of the map and of the ToF sensors and
costs nothing: a rectangle in front of the robot, a few array compares.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pepin.kinematics import Twist


@dataclass(frozen=True)
class SafetyBox:
    """Rectangle ahead of the axle centre (m); the half width covers the cart plus a margin."""

    length_m: float = 0.30
    body_half_width_m: float = 0.32
    min_points: int = 3  # fewer hits inside the box are treated as noise


def nearest_ahead(points_robot: NDArray[np.float64], box: SafetyBox | None = None) -> float | None:
    """Distance to the nearest scan point inside the box ahead, or None when the box is clear."""
    box = box or SafetyBox()
    if len(points_robot) == 0:
        return None
    x, y = points_robot[:, 0], points_robot[:, 1]
    inside = (x > 0.0) & (x <= box.length_m) & (np.abs(y) <= box.body_half_width_m)
    if inside.sum() < box.min_points:
        return None
    return float(x[inside].min())


def guard_forward(
    command: Twist, points_robot: NDArray[np.float64], box: SafetyBox | None = None
) -> tuple[Twist, float | None]:
    """Zero the forward speed if the lidar sees something in the box; returns (twist, blocker)."""
    if command.linear <= 0.0:
        return command, None
    blocker = nearest_ahead(points_robot, box)
    if blocker is None:
        return command, None
    return Twist(0.0, command.angular), blocker
