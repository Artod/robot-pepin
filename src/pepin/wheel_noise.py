"""The wheels' twist error as a law of the measured motion: /odom's twist covariance.

A constant covariance says the wheels are as good in a pivot as on a straight. Against the lidar
truth of drives 0329-0347 they are not: the 1 s mean error of their forward speed grows with the
turn rate (the cart scrubs and its casters swivel), and that of their yaw rate grows with the turn
rate much faster, while the speed itself adds nothing measurable. So, while the wheels move::

    sigma_v = v_per_yaw_rate * |w| + v_floor_m_s       (m/s, 1 s mean)
    sigma_w = yaw_per_yaw_rate * |w| + yaw_floor_rad_s (rad/s, 1 s mean)

and each sample of a ``rate_hz`` stream carries ``rate_hz * sigma^2``: a filter that takes the
samples as independent then holds one second of them to ``sigma``. At rest (both ``|v|`` and
``|w|`` under their ``moving_*`` thresholds) the law says nothing — the wheels' rest error is
below the truth's own noise — and the caller keeps its constant covariance.

``v`` and ``w`` are the MEASURED wheel twist (the encoders), never the command. The C++ twin is
ros/pepin_base_cpp/include/pepin_base_cpp/wheel_noise.hpp; tests/unit/test_base_cpp_contracts.py
holds the two to the same numbers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class WheelNoiseLaw:
    """The coefficients of the law (config/base.json ``odometry_noise``)."""

    v_per_yaw_rate: float = 0.034  # m/s of sigma_v per rad/s of |w|
    v_floor_m_s: float = 0.026
    yaw_per_yaw_rate: float = 0.20  # rad/s of sigma_w per rad/s of |w|
    yaw_floor_rad_s: float = 0.038
    moving_m_s: float = 0.03  # the fit's "moving": |v| at or above this ...
    moving_rad_s: float = 0.052  # ... or |w| at or above this (3 deg/s)

    def __post_init__(self) -> None:
        """Every coefficient is a non-negative finite number; a floor of zero is allowed."""
        for f in fields(self):
            value = getattr(self, f.name)
            if not (isinstance(value, int | float) and 0.0 <= value < float("inf")):
                raise ValueError(
                    f"odometry_noise.{f.name} must be a finite number >= 0, got {value!r}"
                )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WheelNoiseLaw:
        """From the ``odometry_noise`` block; unknown keys (the note) are ignored, missing ones
        take the fit's defaults."""
        names = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in data.items() if k in names})

    @classmethod
    def from_json(cls, path: str | Path) -> WheelNoiseLaw:
        """From config/base.json (or a copy); a file without the block gives the defaults."""
        with open(path) as f:
            return cls.from_dict(json.load(f).get("odometry_noise", {}))

    def moving(self, v: float, w: float) -> bool:
        """The wheels' measured twist is motion, not rest."""
        return abs(v) >= self.moving_m_s or abs(w) >= self.moving_rad_s

    def sigmas(self, v: float, w: float) -> tuple[float, float] | None:
        """The 1 s sigmas (vx m/s, vyaw rad/s) of the wheels at measured ``v``, ``w``; None at
        rest."""
        if not self.moving(v, w):
            return None
        turn = abs(w)
        return (
            self.v_per_yaw_rate * turn + self.v_floor_m_s,
            self.yaw_per_yaw_rate * turn + self.yaw_floor_rad_s,
        )

    def sample_variances(self, v: float, w: float, rate_hz: float) -> tuple[float, float] | None:
        """Per-sample variances (vx, vyaw) of a ``rate_hz`` stream, ``rate_hz * sigma^2``; None
        at rest."""
        s = self.sigmas(v, w)
        if s is None:
            return None
        return rate_hz * s[0] ** 2, rate_hz * s[1] ** 2

    def bridge_parameters(self, rate_hz: float) -> dict[str, float]:
        """The base bridge's parameters for this law on a ``rate_hz`` state stream
        (robot.launch.py; base_bridge.cpp declares each, read-only)."""
        return {
            "odom_law_v_per_yaw_rate": self.v_per_yaw_rate,
            "odom_law_v_floor_m_s": self.v_floor_m_s,
            "odom_law_yaw_per_yaw_rate": self.yaw_per_yaw_rate,
            "odom_law_yaw_floor_rad_s": self.yaw_floor_rad_s,
            "odom_law_moving_m_s": self.moving_m_s,
            "odom_law_moving_rad_s": self.moving_rad_s,
            "odom_law_rate_hz": float(rate_hz),
        }
