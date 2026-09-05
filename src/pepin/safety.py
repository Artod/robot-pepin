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
from pepin.tof import ReflexConfig, ReflexDecision, TofRanges


@dataclass(frozen=True)
class SafetyBox:
    """Bumper: the rectangle just ahead of the hull, measured from the axle centre (m).

    The drive wheels are the front of the robot: outer wheel edges at +-0.275 m
    (0.55 m over the wheels), the wheel fronts 0.06 m ahead of the axle, the
    cart's front plane at 0.03 m; the body reaches 0.30 m back. The planner's
    inflation radius carries the margin; this box only says "you are about to
    touch it", so it must stay inside that radius or every path the planner
    accepts gets vetoed beside every obstacle.
    """

    length_m: float = 0.25
    body_half_width_m: float = 0.27
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


class Reflex:
    """ToF stop rule with hysteresis and direction, for autonomous driving.

    A sensor that tripped stays tripped until its range opens by
    ``release_margin_m`` (no chattering around the threshold), and a side hit
    only forbids moving toward it: forward, and turning to that side. Backing
    away and turning away are always allowed, so the planner can steer out.
    """

    def __init__(self, config: ReflexConfig | None = None, release_margin_m: float = 0.08) -> None:
        """``config`` holds the stop distances; ``release_margin_m`` the hysteresis band."""
        self._cfg = config or ReflexConfig()
        self._margin = release_margin_m
        self._tripped: set[str] = set()

    @property
    def tripped(self) -> frozenset[str]:
        """Sensors currently holding the robot back."""
        return frozenset(self._tripped)

    def step(self, command: Twist, ranges: TofRanges) -> ReflexDecision:
        """Trim ``command`` by what the sensors see; ``blocked`` says whether anything changed."""
        cfg = self._cfg
        if ranges.age_s > cfg.max_age_s:
            if cfg.blocked_when_stale and command.linear > 0.0:
                return ReflexDecision(Twist(0.0, command.angular), True, "no fresh ToF data")
            return ReflexDecision(command, blocked=False)
        limits = {"front": cfg.front_stop_m, "left": cfg.side_stop_m, "right": cfg.side_stop_m}
        values = {"front": ranges.front, "left": ranges.left, "right": ranges.right}
        for name, value in values.items():
            limit = limits[name] + (self._margin if name in self._tripped else 0.0)
            if value is not None and cfg.min_valid_m <= value < limit:
                self._tripped.add(name)
            else:
                self._tripped.discard(name)
        linear, angular = command.linear, command.angular
        if self._tripped and linear > 0.0:
            linear = 0.0
        if "left" in self._tripped and angular > 0.0:
            angular = 0.0  # turning left sweeps the front-left corner into it
        if "right" in self._tripped and angular < 0.0:
            angular = 0.0
        trimmed = Twist(linear, angular)
        if trimmed == command:
            return ReflexDecision(command, blocked=False)
        reason = ", ".join(
            f"{n} {values[n]:.2f} m" for n in ("front", "left", "right") if n in self._tripped
        )
        return ReflexDecision(trimmed, True, reason)
