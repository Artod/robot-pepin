"""Differential-drive kinematics: body twist <-> wheel rates.

Conventions: x forward, y left, yaw counter-clockwise (right-hand rule).
A positive angular velocity turns the robot left, so the right wheel runs
faster than the left one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from pepin.geometry import BaseGeometry


@dataclass(frozen=True)
class Twist:
    """Body velocity: forward speed in m/s and yaw rate in rad/s."""

    linear: float
    angular: float


@dataclass(frozen=True)
class WheelRates:
    """Angular velocity of each wheel in rad/s, positive = robot forward."""

    left: float
    right: float


class DiffDriveKinematics:
    """Converts between body twists and wheel rates for a two-wheel base."""

    def __init__(self, geometry: BaseGeometry) -> None:
        self._r = geometry.wheel_radius_m
        self._half_track = geometry.track_width_m / 2.0
        self._ticks_per_rad = geometry.ticks_per_rev / (2.0 * math.pi)

    def twist_to_wheels(self, twist: Twist) -> WheelRates:
        left = (twist.linear - twist.angular * self._half_track) / self._r
        right = (twist.linear + twist.angular * self._half_track) / self._r
        return WheelRates(left=left, right=right)

    def wheels_to_twist(self, rates: WheelRates) -> Twist:
        v_left = rates.left * self._r
        v_right = rates.right * self._r
        return Twist(
            linear=(v_left + v_right) / 2.0,
            angular=(v_right - v_left) / (2.0 * self._half_track),
        )

    def rad_s_to_ticks_s(self, rad_s: float) -> int:
        """Wheel rate to the servo's native velocity unit (encoder ticks per second)."""
        return round(rad_s * self._ticks_per_rad)
