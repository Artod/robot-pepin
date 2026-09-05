"""Will the real hull touch anything if we do this? The robot's outline swept through a command.

The planner treats the robot as a disc and the old bumper as a box ahead of
the axle; neither knows that the cart is 0.30 m long behind its drive wheels
and 0.55 m wide, so a turn in place sweeps a 0.41 m radius at the rear. This
module keeps the outline as a rectangle in the robot frame and rolls it
forward through the commanded twist for a short horizon against the newest
lidar points. :class:`FootprintGuard` then trims the command to the first
variant that stays clear — slower, turn only, straight only, or stop.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pepin.kinematics import STOP, Twist

HULL_STEP_M = 0.005  # how far any point of the hull may move between two sweep samples


@dataclass(frozen=True)
class Footprint:
    """The hull as a rectangle in the robot frame (origin between the drive wheels, x forward)."""

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

    def inside(self, points_robot: NDArray[np.float64]) -> NDArray[np.bool_]:
        """Mask of points (N, 2) that lie within the hull plus its margin."""
        x, y = points_robot[:, 0], points_robot[:, 1]
        m = self.margin_m
        return (
            (x >= -self.rear_m - m) & (x <= self.front_m + m) & (np.abs(y) <= self.half_width_m + m)
        )


def pose_after(twist: Twist, t: float) -> tuple[float, float, float]:
    """Robot frame after ``t`` seconds of a constant twist (exact arc), in the start frame."""
    v, w = twist.linear, twist.angular
    if abs(w) < 1e-9:
        return (v * t, 0.0, 0.0)
    return (v / w * math.sin(w * t), v / w * (1.0 - math.cos(w * t)), w * t)


def time_to_contact(
    points_robot: NDArray[np.float64],
    twist: Twist,
    footprint: Footprint,
    horizon_s: float = 0.6,
    dt: float = 0.05,
    min_points: int = 2,
) -> float | None:
    """Earliest time within ``horizon_s`` when the hull touches ``min_points`` points, or None.

    The points are the newest scan in the robot frame; the hull is rolled
    along the arc the twist describes and the points are viewed from each
    future pose. Standing still never touches anything.
    """
    if len(points_robot) == 0 or (twist.linear == 0.0 and twist.angular == 0.0):
        return None
    # No point of the hull moves more than half a centimetre per step: two
    # adjacent beams on a chair leg by the rear corner are only ~8 mm apart and
    # sit inside the hull together for a few hundredths of a second in a turn.
    fastest = max(abs(twist.linear), abs(twist.angular) * footprint.swing_radius_m)
    steps = max(1, math.ceil(horizon_s / min(dt, HULL_STEP_M / fastest)))
    dt = horizon_s / steps
    for k in range(1, steps + 1):
        t = k * dt
        x, y, theta = pose_after(twist, t)
        c, s = math.cos(theta), math.sin(theta)
        shifted = points_robot - np.array([x, y])
        local = shifted @ np.array([[c, -s], [s, c]])  # rotate by -theta into the future frame
        if int(footprint.inside(local).sum()) >= min_points:
            return t
    return None


class FootprintGuard:
    """Trims a twist so the hull stays clear of the lidar points for the next ``horizon_s``.

    Order of retreat: half the speed, then turn only (the planner's next
    tick will route around), then drive straight without the turn, then
    stop. Backing away is judged like any other motion, so it stays allowed
    whenever the rear is clear.
    """

    def __init__(
        self, footprint: Footprint | None = None, horizon_s: float = 0.6, min_points: int = 2
    ) -> None:
        """``footprint`` defaults to the measured cart; ``horizon_s`` is how far ahead to look.

        ``min_points`` is how many returns inside the hull count as contact:
        2 ignores a lone noisy beam, 1 trusts every return (a thin chair leg).
        """
        self.footprint = footprint or Footprint()
        self.horizon_s = horizon_s
        self.min_points = min_points

    def apply(self, command: Twist, points_robot: NDArray[np.float64]) -> tuple[Twist, str]:
        """The safe variant of ``command`` and why it changed ("" when it did not)."""
        if self._contact(points_robot, command) is None:
            return command, ""
        candidates = (
            (Twist(command.linear * 0.5, command.angular), "slowed: hull would touch"),
            (Twist(0.0, command.angular), "forward blocked: hull would touch"),
            (Twist(command.linear, 0.0), "turn blocked: hull would sweep into it"),
        )
        for candidate, reason in candidates:
            if candidate in (command, STOP):
                continue
            if self._contact(points_robot, candidate) is None:
                return candidate, reason
        return STOP, "hold: hull would touch whichever way"

    def _contact(self, points_robot: NDArray[np.float64], twist: Twist) -> float | None:
        return time_to_contact(
            points_robot, twist, self.footprint, self.horizon_s, min_points=self.min_points
        )
