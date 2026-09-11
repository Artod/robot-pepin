"""Wheel slip seen from the lidar: the wheels report motion, the world around the robot does not.

Two scans a tenth of a second apart from a robot that really moved differ everywhere; from a
robot spinning its wheels on a carpet edge they are the same picture. That difference is the
only honest slip signal this cart has (no wheel-drop switch, no optical floor sensor), and it
needs no map: it compares consecutive scans beam by beam.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from pepin.odometry import Pose2D


def scan_changed(
    previous: NDArray[np.float64],
    current: NDArray[np.float64],
    threshold_m: float = 0.02,
    min_share: float = 0.15,
    min_valid: int = 60,
) -> tuple[bool, float]:
    """Did the surroundings move between two scans of equal binning?

    Returns (changed, share of comparable beams whose range moved by more than
    ``threshold_m``). A straight 4 cm step moves only the beams looking along the
    motion, so the share, not the median, is what separates driving (a quarter of
    the beams move) from slipping (none do). Fewer than ``min_valid`` comparable
    beams counts as "changed": a blind scan must not read as slip.
    """
    if previous.shape != current.shape:
        return True, math.inf
    both = np.isfinite(previous) & np.isfinite(current)
    if int(both.sum()) < min_valid:
        return True, math.inf
    share = float((np.abs(previous[both] - current[both]) > threshold_m).mean())
    return share > min_share, share


def slipping(
    motion: Pose2D,
    changed: bool,
    min_travel_m: float = 0.03,
    min_turn_deg: float = 3.0,
) -> bool:
    """Wheels claim more than ``min_travel_m`` or ``min_turn_deg`` while the scan stood still."""
    claimed = math.hypot(motion.x, motion.y) >= min_travel_m or abs(motion.theta) >= math.radians(
        min_turn_deg
    )
    return claimed and not changed


class SlipWatch:
    """Slip scan by scan, for a tracker: the wheels' step since the previous scan against
    whether the picture changed (:func:`scan_changed`, :func:`slipping`). ``streak`` counts
    the consecutive slipping scans, so a caller can say it once when the third one lands."""

    def __init__(self) -> None:
        self._last_ranges: NDArray[np.float64] | None = None
        self._last_odom: Pose2D | None = None
        self.streak = 0

    def observe(self, ranges: NDArray[np.float64], odom: Pose2D) -> bool:
        """True when the wheels claim a step since the previous scan and the ranges (equal
        binning) show the same picture. The first scan is never slip: nothing to compare."""
        from pepin.scanmatch import relative_motion

        changed = True
        if self._last_ranges is not None:
            changed, _ = scan_changed(self._last_ranges, ranges)
        step = Pose2D() if self._last_odom is None else relative_motion(self._last_odom, odom)
        slip = slipping(step, changed)
        self._last_ranges, self._last_odom = ranges, odom
        self.streak = self.streak + 1 if slip else 0
        return slip
