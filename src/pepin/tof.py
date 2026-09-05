"""Near-field time-of-flight ranges from the board and the stop reflex built on them.

The lidar sees one horizontal slice of the world; the three VL53L1X
sensors look where it cannot (low, in front) and feed two things: a reflex
that refuses to drive into something close, and an obstacle layer for
navigation (through :class:`TofMount`, which says where each return lands
in the robot frame). Localisation never uses them.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pepin.kinematics import Twist
from pepin.streams import Connector, JsonLinesClient

TOF_PORT = 3335


@dataclass(frozen=True)
class TofRanges:
    """Latest ranges in meters (None = no return) and how old they are."""

    front: float | None
    left: float | None
    right: float | None
    age_s: float

    def by_name(self) -> dict[str, float | None]:
        """The three ranges keyed by sensor name, for code that iterates over sensors."""
        return {"front": self.front, "left": self.left, "right": self.right}


@dataclass(frozen=True)
class TofMount:
    """Where a sensor sits and looks, in the robot frame (origin between the wheels, x forward)."""

    x_m: float
    y_m: float
    yaw_deg: float  # beam direction: 0 forward, +90 left
    height_m: float

    def hit_xy(self, range_m: float) -> tuple[float, float]:
        """Robot-frame point a return at ``range_m`` corresponds to."""
        yaw = math.radians(self.yaw_deg)
        return (self.x_m + range_m * math.cos(yaw), self.y_m + range_m * math.sin(yaw))


def load_mounts(path: str | Path) -> dict[str, TofMount]:
    """Sensor mounts from ``config/tof.json``; sensors whose ``mount`` is null are left out."""
    with open(path) as f:
        sensors = json.load(f)["sensors"]
    return {
        name: TofMount(m["x_m"], m["y_m"], m["yaw_deg"], m["height_m"])
        for name, entry in sensors.items()
        if (m := entry.get("mount")) is not None
    }


class TofClient(JsonLinesClient):
    """The board's range stream (:mod:`board.tof_server`) as a :class:`pepin.feeds.Feed`.

    ``ranges()`` never blocks; its ``age_s`` is infinite until the first record,
    so a dead stream is visible instead of silently reading "nothing close".
    """

    def __init__(
        self, host: str, port: int = TOF_PORT, *, connector: Connector | None = None
    ) -> None:
        """Prepare a client for ``host:port``; nothing connects until :meth:`start`."""
        super().__init__(host, port, name="tof", connector=connector)
        self._latest: dict[str, float | None] = {"front": None, "left": None, "right": None}
        self._lock = threading.Lock()

    def _ingest(self, record: dict[str, Any]) -> None:
        """One board record: millimetres per sensor, -1 or null for no return."""
        with self._lock:
            for name in ("front", "left", "right"):
                mm = record.get(name)
                self._latest[name] = None if mm is None or mm <= 0 else float(mm) / 1000.0

    def ranges(self, now: float | None = None) -> TofRanges:
        """Latest ranges in meters plus their age; ``age_s`` is infinite before the first record."""
        with self._lock:
            return TofRanges(
                self._latest["front"], self._latest["left"], self._latest["right"], self.age_s(now)
            )


@dataclass(frozen=True)
class ReflexConfig:
    """Distances (m) below which forward motion is refused, and how stale data may be."""

    front_stop_m: float = 0.22
    # Side sensors sit level at 0.16 m; their 27-degree cone would only reach the floor
    # at ~0.67 m, so anything nearer than this is a real object beside the front wheels.
    side_stop_m: float = 0.30
    max_age_s: float = 0.5
    blocked_when_stale: bool = False
    # Below the sensor's own minimum range a value is a failure code, not an object.
    min_valid_m: float = 0.04


@dataclass(frozen=True)
class ReflexDecision:
    """What the reflex allows: the (possibly zeroed) twist and why."""

    twist: Twist
    blocked: bool
    reason: str = ""


def apply_reflex(
    command: Twist, ranges: TofRanges, config: ReflexConfig | None = None
) -> ReflexDecision:
    """Zero the forward speed when something is closer than the stop distance ahead.

    Only forward motion is blocked: backing away from an obstacle must stay
    possible. Turning in place is always allowed.
    """
    config = config or ReflexConfig()
    if command.linear <= 0:
        return ReflexDecision(command, blocked=False)
    if ranges.age_s > config.max_age_s:
        if config.blocked_when_stale:
            return ReflexDecision(Twist(0.0, command.angular), True, "no fresh ToF data")
        return ReflexDecision(command, blocked=False)
    front = ranges.front
    if front is not None and config.min_valid_m <= front < config.front_stop_m:
        return ReflexDecision(Twist(0.0, command.angular), True, f"front {front:.2f} m")
    for name, value in (("left", ranges.left), ("right", ranges.right)):
        if value is not None and config.min_valid_m <= value < config.side_stop_m:
            return ReflexDecision(Twist(0.0, command.angular), True, f"{name} {value:.2f} m")
    return ReflexDecision(command, blocked=False)
