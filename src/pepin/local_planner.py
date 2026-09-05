"""Local planner: when the follower's wish would touch something, the best arc that does not.

The A* path is the strategy and :class:`pepin.control.PathFollower` turns it
into a wish — a twist toward the lookahead point. On open floor the wish goes
through untouched. When the hull sweep says the wish meets a lidar point within
the horizon, this planner samples a lattice of twists, drops every one that
touches, and picks the survivor that leaves the robot nearest the lookahead
point (and pointing at it) after the horizon, at a small cost for deviating
from the wish so the choice stays smooth from tick to tick. Nothing here
replans: the obstacle memory and A* still route around a blocker that stays;
this only makes the meantime a manoeuvre instead of a stop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from pepin.control import ControllerConfig
from pepin.footprint import Footprint, pose_after, time_to_contact
from pepin.kinematics import STOP, Twist
from pepin.odometry import wrap_angle


@dataclass(frozen=True)
class LocalPlannerConfig:
    """Rollout horizon, the twist lattice and the cost weights."""

    horizon_s: float = 1.0  # each candidate arc is rolled out and swept this far ahead
    # ...and at least this far in meters: a slow arc covers little ground in a second and
    # would count as "clear" while inching into the obstacle. Every candidate must be
    # viable for a hull's length of travel.
    min_travel_m: float = 0.15
    # Speeds from cruise down to cruise / linear_samples. Two on purpose: a lattice with
    # a crawl in it "clears" a crawl into the obstacle and glues the cart to it.
    linear_samples: int = 2
    angular_samples: int = 9  # yaw rates from -max to +max, always including 0
    progress_weight: float = 1.0  # per meter still to go to the lookahead point
    heading_weight: float = 0.3  # per radian of heading error to it, after the horizon
    deviation_weight: float = 0.4  # per unit of |dv|/cruise + |dw|/max_yaw from the wish
    # Reverse candidates run at this share of cruise (0 = none). They rank last by the
    # costs above and are chosen only when every forward arc and turn touches: the cart
    # is wider than it is long, so 0.26 m from an object it can turn only ~25 deg and the
    # way out is backwards first.
    reverse_fraction: float = 0.5


class LocalPlanner:
    """Contact-free steering around what the map does not know, one tick at a time."""

    def __init__(
        self,
        footprint: Footprint,
        controller: ControllerConfig,
        config: LocalPlannerConfig | None = None,
        min_points: int = 2,
    ) -> None:
        """``min_points`` is the sweep's contact rule (:func:`pepin.footprint.time_to_contact`)."""
        self.footprint = footprint
        self.cfg = config or LocalPlannerConfig()
        self._cruise = controller.cruise_speed_m_s
        self._max_yaw = controller.max_yaw_rate_rad_s
        self._min_points = min_points
        speeds = np.linspace(
            self._cruise, self._cruise / self.cfg.linear_samples, self.cfg.linear_samples
        )
        yaws = np.linspace(-self._max_yaw, self._max_yaw, self.cfg.angular_samples)
        self._lattice = [Twist(float(v), float(w)) for v in speeds for w in yaws]
        self._lattice += [Twist(0.0, float(w)) for w in yaws if w != 0.0]  # turn in place
        self._reverse: list[Twist] = []
        if self.cfg.reverse_fraction > 0:
            back = -self._cruise * self.cfg.reverse_fraction
            self._reverse = [Twist(back, float(w)) for w in yaws[:: max(1, len(yaws) // 3)]]
            self._reverse.append(Twist(back, 0.0))
            self._lattice += self._reverse
        self._backing = False  # committed to backing off until there is room to turn

    def steer(
        self,
        wish: Twist,
        target_robot: tuple[float, float],
        points_robot: NDArray[np.float64],
    ) -> tuple[Twist, str]:
        """The wish if it is contact-free; otherwise the cheapest contact-free arc, or STOP.

        ``target_robot`` is the lookahead point in the robot frame. The reason is
        "" when the wish went through, else what was chosen and why.
        """
        if self._backing:
            # Once backing off, keep backing until a turn in place is free: alternating
            # a centimetre back with a centimetre forward glues the cart to the obstacle.
            if self._turn_room(points_robot) >= 0.0:
                self._backing = False
            else:
                return self._best(self._reverse, wish, target_robot, points_robot, "back off")
        # Intervene only in trouble: the wish would touch, or the cart is already too close
        # to turn and the wish brings it closer still. Everywhere else the follower rules,
        # so a good path is driven exactly as planned.
        x, y, _ = pose_after(wish, self.cfg.horizon_s)
        room_now, room_then = self._turn_room(points_robot), self._turn_room(points_robot, x, y)
        closing_in = room_now < 0.0 and room_then < room_now - 1e-3
        if not closing_in and self._clear(wish, points_robot):
            return wish, ""
        return self._best(self._lattice, wish, target_robot, points_robot, "steer around")

    def _best(
        self,
        candidates: list[Twist],
        wish: Twist,
        target: tuple[float, float],
        points_robot: NDArray[np.float64],
        verb: str,
    ) -> tuple[Twist, str]:
        """The cheapest contact-free candidate and a reason; STOP when every one touches."""
        ranked = sorted(candidates, key=lambda c: self._cost(c, wish, target, points_robot))
        for candidate in ranked:
            if candidate == wish or not self._clear(candidate, points_robot):
                continue
            if candidate.linear < 0:
                verb, self._backing = "back off", True
            return candidate, f"{verb}: v {candidate.linear:+.2f} w {candidate.angular:+.2f}"
        return STOP, "boxed in: every arc touches"

    def _turn_room(
        self, points_robot: NDArray[np.float64], x: float = 0.0, y: float = 0.0
    ) -> float:
        """Nearest point's distance from (x, y) minus what a turn in place sweeps; < 0 = cramped."""
        if len(points_robot) == 0:
            return math.inf
        nearest = float(np.hypot(points_robot[:, 0] - x, points_robot[:, 1] - y).min())
        return nearest - (self.footprint.swing_radius_m + self.footprint.margin_m)

    def _clear(self, twist: Twist, points_robot: NDArray[np.float64]) -> bool:
        horizon = self.cfg.horizon_s
        if twist.linear != 0.0:
            horizon = max(horizon, self.cfg.min_travel_m / abs(twist.linear))
        return (
            time_to_contact(
                points_robot, twist, self.footprint, horizon, min_points=self._min_points
            )
            is None
        )

    def _cost(
        self,
        candidate: Twist,
        wish: Twist,
        target: tuple[float, float],
        points_robot: NDArray[np.float64],
    ) -> float:
        """Where the arc leaves the robot: distance and heading to the lookahead point,
        plus the deviation from the wish."""
        x, y, theta = pose_after(candidate, self.cfg.horizon_s)
        remaining = math.hypot(target[0] - x, target[1] - y)
        heading = abs(wrap_angle(math.atan2(target[1] - y, target[0] - x) - theta))
        deviation = (
            abs(candidate.linear - wish.linear) / self._cruise
            + abs(candidate.angular - wish.angular) / self._max_yaw
        )
        cfg = self.cfg
        return (
            cfg.progress_weight * remaining
            + cfg.heading_weight * heading
            + cfg.deviation_weight * deviation
        )
