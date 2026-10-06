"""The wheels' twist error as a law of the measured motion: /odom's twist covariance.

A constant covariance says the wheels are as good in a pivot as on a straight. Against the lidar
truth of drives 0329-0347 they are not: the 1 s mean error of their forward speed grows with the
turn rate (the cart scrubs and its casters swivel), and that of their yaw rate grows with the turn
rate much faster, while the speed itself adds nothing measurable. So, while the wheels move::

    sigma_v = v_per_yaw_rate * |w| + v_floor_m_s       (m/s, 1 s mean)
    sigma_w = yaw_per_yaw_rate * |w| + yaw_floor_rad_s (rad/s, 1 s mean)

and each sample of a ``rate_hz`` stream carries ``rate_hz * sigma^2``: a filter that takes the
samples as independent then holds one second of them to ``sigma``. At rest the law says nothing —
the wheels' error on a standing cart is below the truth's own noise — and the caller keeps its
constant covariance; but rest counts only once the wheels have been still for ``hold_s`` (the
seconds just after a stop are as wrong as moving ones: :class:`WheelNoise`).

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
    hold_s: float = 2.0  # the law stays on this long after the last moving sample

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

    def sigmas(self, w: float) -> tuple[float, float]:
        """The law's 1 s sigmas (vx m/s, vyaw rad/s) at the measured yaw rate ``w``."""
        turn = abs(w)
        return (
            self.v_per_yaw_rate * turn + self.v_floor_m_s,
            self.yaw_per_yaw_rate * turn + self.yaw_floor_rad_s,
        )

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
            "odom_law_hold_s": self.hold_s,
            "odom_law_rate_hz": float(rate_hz),
        }


class WheelNoise:
    """The law applied to each sample of a ``rate_hz`` stream, with its rest gate and hold.

    A sample gets the law while the wheels move and for ``hold_s`` after their last moving sample;
    otherwise None, and the caller keeps its constant. Over the 19 drives the hold took the seconds
    that end in a stop from a mean z^2 of 5.9 (the plain gate: the standing samples' tight constant
    outvoted the stop's real error) to 1.3 (scratch/wheel_law/gate.py).
    """

    def __init__(self, law: WheelNoiseLaw, rate_hz: float) -> None:
        """``rate_hz``: the stream's sample rate, one sample's share of a second."""
        self.law = law
        self.rate_hz = rate_hz
        self._last_moving_s: float | None = None

    def reset(self) -> None:
        """Forget the last motion: the next standing sample is rest at once."""
        self._last_moving_s = None

    def update(self, v: float, w: float, stamp_s: float) -> tuple[float, float] | None:
        """Per-sample variances (vx, vyaw) for the measured ``v``, ``w`` at ``stamp_s``, or None
        for the caller's constant (rest past the hold, or a clock that went backwards)."""
        if self.law.moving(v, w):
            self._last_moving_s = stamp_s
        elif self._last_moving_s is None or not (
            0.0 <= stamp_s - self._last_moving_s <= self.law.hold_s
        ):
            return None
        s_v, s_w = self.law.sigmas(w)
        return self.rate_hz * s_v**2, self.rate_hz * s_w**2
