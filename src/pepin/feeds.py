"""What every sensor or link on the laptop shares: it runs by itself and is asked, never awaited.

A :class:`Feed` starts a background reader, keeps the newest reading, says how
old it is, and can be closed. The control loop composes feeds into one
:class:`pepin.navigator.Sense` per tick without ever blocking on the network;
a feed that has nothing to say shows up as a large ``age_s``, which the
navigator's hold rules turn into "stand still" or "carry on without it"
depending on the sensor. Adding a sensor means adding a Feed and one field to
``Sense``; switching one off is a flag in ``config/robot.json``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pepin.odometry import Pose2D
from pepin.tof import TofRanges


class Feed(Protocol):
    """A background reader with a freshness clock."""

    connected: bool

    def start(self) -> Any:
        """Begin reading in a daemon thread; returns self so it chains."""
        ...

    def close(self) -> None:
        """Stop the reader and release its socket."""
        ...

    def age_s(self, now: float | None = None) -> float:
        """Seconds since the newest reading; infinite before the first."""
        ...


@dataclass(frozen=True)
class Sense:
    """One tick of sensor input, already in the robot's own units and frame."""

    now: float  # time.monotonic() of this tick
    odom_pose: Pose2D  # wheel odometry, integrated by the caller
    scans: Sequence[NDArray[np.float64]]  # robot-frame (N, 2) point sets since the last tick
    scan_age_s: float  # seconds since the newest scan ever received; inf before the first
    tof: TofRanges | None  # None when running without the ToF sensors
