"""Near-field time-of-flight ranges from the board and the stop reflex built on them.

The lidar sees one horizontal slice of the world; the three VL53L1X
sensors look where it cannot (low, in front). This module is the driver side:
the stream client and where each sensor sits (:class:`TofMount`). The stop
rule built on the ranges lives in :mod:`pepin.safety`. Localisation never
uses them.
"""

from __future__ import annotations

import json
import math
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    """The board's range stream (:mod:`pepin.tof_server`) as a :class:`pepin.feeds.Feed`.

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
