"""Timestamped session recording: everything the robot saw and did, replayable offline.

One session is one JSON-lines file; every line carries a monotonic timestamp
and a topic (``pose``, ``scan``, ``cmd``, ``note``). The format is deliberately
boring — greppable, diffable, readable from any language — because these
files are the raw material for mapping experiments and post-mortems.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path
from types import TracebackType
from typing import Any

import numpy as np

from pepin.kinematics import Twist
from pepin.lidar import LaserScan
from pepin.odometry import Pose2D


class SessionRecorder:
    """Appends timestamped records to ``<directory>/<timestamp>_<name>.jsonl``."""

    def __init__(self, directory: str | Path, name: str = "session") -> None:
        """Creates ``directory`` if needed and opens a fresh file; ``name`` labels the run."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"{datetime.now():%Y%m%d_%H%M%S}_{name}.jsonl"
        self._file = self.path.open("w")
        self.records = 0

    def write(self, topic: str, payload: dict[str, Any], t: float | None = None) -> None:
        """Append one record under ``topic``; ``t`` defaults to now on the monotonic clock."""
        record = {"t": time.monotonic() if t is None else t, "topic": topic, **payload}
        self._file.write(json.dumps(record, separators=(",", ":")) + "\n")
        self.records += 1

    def pose(self, pose: Pose2D, travel: tuple[float, float] | None = None) -> None:
        """Record the odometry pose (meters, radians); ``travel`` is the (left, right)
        wheel distance in meters over that step, which lets a replay re-integrate it."""
        payload: dict[str, Any] = {"x": pose.x, "y": pose.y, "theta": pose.theta}
        if travel is not None:
            payload["d_left"], payload["d_right"] = travel
        self.write("pose", payload)

    def command(self, twist: Twist) -> None:
        """Record a commanded body twist (m/s, rad/s) at the moment it was sent."""
        self.write("cmd", {"linear": twist.linear, "angular": twist.angular})

    def scan(self, scan: LaserScan) -> None:
        """Record one revolution under the scan's own stamp, not the write time.

        Angles are rounded to 0.1 mrad and ranges to the millimeter; invalid
        returns are written as ``null`` so the file stays valid JSON.
        """
        ranges = [None if math.isnan(r) else round(float(r), 3) for r in scan.ranges]
        self.write(
            "scan",
            {
                "angles": [round(float(a), 4) for a in scan.angles],
                "ranges": ranges,
                "intensities": [int(i) for i in scan.intensities],
                "speed_rps": round(scan.speed_rps, 2),
            },
            t=scan.stamp,
        )

    def note(self, text: str) -> None:
        """Record a free-text marker in the timeline (run start, an observation, a collision)."""
        self.write("note", {"text": text})

    def close(self) -> None:
        """Flush and close the session file."""
        self._file.close()

    def __enter__(self) -> SessionRecorder:
        """The file is already open by then; this only scopes the closing."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the file so a crashed run still leaves everything written on disk."""
        self.close()


def read_session(path: str | Path) -> Iterator[dict[str, Any]]:
    """Yield records of a session file in order."""
    with open(path) as f:
        for line in f:
            if line.strip():
                record: dict[str, Any] = json.loads(line)
                yield record


def scan_from_record(record: dict[str, Any]) -> LaserScan:
    """Rebuild a :class:`LaserScan` from a ``scan`` record."""
    return LaserScan(
        stamp=record["t"],
        angles=np.array(record["angles"], dtype=np.float64),
        ranges=np.array([math.nan if r is None else r for r in record["ranges"]], dtype=np.float64),
        intensities=np.array(record["intensities"], dtype=np.int64),
        speed_rps=record["speed_rps"],
    )


def pose_from_record(record: dict[str, Any]) -> Pose2D:
    """Rebuild a :class:`Pose2D` from a ``pose`` record (meters, radians)."""
    return Pose2D(x=record["x"], y=record["y"], theta=record["theta"])


def scan_record_from_ros(
    stamp: float,
    angle_min: float,
    angle_increment: float,
    ranges: list[float],
    intensities: list[float],
    range_min: float,
    range_max: float,
    scan_time: float,
    *,
    mount_yaw_rad: float,
    mount_x_m: float,
    hull: tuple[float, float, float] = (-0.30, 0.0625, 0.275),
) -> dict[str, Any]:
    """A ``scan`` record from the fields of a ROS ``sensor_msgs/LaserScan``.

    The driver reports counter-clockwise angles for an upright sensor; ours hangs
    upside down and turned by ``mount_yaw_rad``, so the robot-frame angle is
    ``-a - yaw`` (verified: scan-to-map fit 0.62-0.67 against a map built by the
    Python stack). Returns inside the cart's own hull (``hull`` = x_back, x_front,
    half_width, around the sensor's ``mount_x_m``) are written as ``null``, like
    the masked sectors of the old stack; NaN intensities become 0.
    """
    two_pi = 2.0 * math.pi
    x_back, x_front, half_width = hull
    angles: list[float] = []
    out_ranges: list[float | None] = []
    for i, r in enumerate(ranges):
        robot = (-(angle_min + i * angle_increment) - mount_yaw_rad) % two_pi
        angles.append(round(robot, 4))
        ok = math.isfinite(r) and range_min < r < range_max
        if ok:
            x, y = mount_x_m + r * math.cos(robot), r * math.sin(robot)
            if x_back <= x <= x_front and abs(y) <= half_width:
                ok = False
        out_ranges.append(round(r, 3) if ok else None)
    n = len(intensities)
    out_intensities = [
        int(intensities[i]) if i < n and math.isfinite(intensities[i]) else 0
        for i in range(len(ranges))
    ]
    speed = round(1.0 / scan_time, 2) if scan_time > 1e-3 else 10.0
    return {
        "t": stamp,
        "topic": "scan",
        "angles": angles,
        "ranges": out_ranges,
        "intensities": out_intensities,
        "speed_rps": speed,
    }
