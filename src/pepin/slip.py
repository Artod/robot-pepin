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
