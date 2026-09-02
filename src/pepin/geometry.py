"""Physical parameters of the mobile base, loaded from ``config/base.json``.

Numbers here come from tape-measure estimates; the intended workflow is to
start with them and refine empirically (drive a known square, compare the
odometry against the lidar), so everything is plain data with no hidden
derived state.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BaseGeometry:
    """Differential-drive geometry and the wheel encoder scale."""

    wheel_diameter_m: float = 0.125
    track_width_m: float = 0.505
    ticks_per_rev: int = 4096

    @property
    def wheel_radius_m(self) -> float:
        """Rolling radius in meters — the lever arm between wheel rad/s and body m/s."""
        return self.wheel_diameter_m / 2.0

    @property
    def m_per_tick(self) -> float:
        """Linear travel of the wheel rim per encoder tick."""
        return math.pi * self.wheel_diameter_m / self.ticks_per_rev


@dataclass(frozen=True)
class WheelMotor:
    """A wheel servo on the bus and its rotation sense.

    ``direction`` is +1 when a positive velocity command drives the robot
    forward and -1 when the motor is mounted mirrored.
    """

    motor_id: int
    direction: int

    def __post_init__(self) -> None:
        """Reject any ``direction`` other than +1 or -1: it is a sign, not a gain."""
        if self.direction not in (-1, 1):
            raise ValueError(f"direction must be +1 or -1, got {self.direction}")


@dataclass(frozen=True)
class BaseConfig:
    """Everything the base driver needs to know about the hardware."""

    geometry: BaseGeometry
    left: WheelMotor
    right: WheelMotor
    max_speed_m_s: float = 0.3
    max_yaw_rate_rad_s: float = 1.0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BaseConfig:
        """Build from parsed JSON; geometry and both wheels are required, limits optional."""
        return cls(
            geometry=BaseGeometry(**data["geometry"]),
            left=WheelMotor(**data["left"]),
            right=WheelMotor(**data["right"]),
            max_speed_m_s=data.get("max_speed_m_s", cls.max_speed_m_s),
            max_yaw_rate_rad_s=data.get("max_yaw_rate_rad_s", cls.max_yaw_rate_rad_s),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> BaseConfig:
        """Load the base configuration from ``config/base.json`` or a copy of it."""
        with open(path) as f:
            return cls.from_dict(json.load(f))
