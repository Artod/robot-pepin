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
    threshold_m: float = 0.03,
    min_valid: int = 60,
) -> tuple[bool, float]:
    """Did the surroundings move between two scans of equal binning?

    Returns (changed, median absolute range difference over beams valid in both). Fewer than
    ``min_valid`` comparable beams counts as "changed": a blind scan must not read as slip.
    """
    if previous.shape != current.shape:
        return True, math.inf
    both = np.isfinite(previous) & np.isfinite(current)
    if int(both.sum()) < min_valid:
        return True, math.inf
    delta = float(np.median(np.abs(previous[both] - current[both])))
    return delta > threshold_m, delta


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
