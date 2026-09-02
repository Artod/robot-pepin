"""LDRobot LD19 lidar: frame parsing, full-revolution scans, mounting geometry.

The sensor streams 47-byte frames of 12 points each at 230400 baud. Frames
are parsed and CRC-checked, then assembled into :class:`LaserScan` objects,
one per revolution, expressed in the robot frame: angle zero is straight
ahead and angles grow counter-clockwise, matching ``pepin.kinematics``.
"""

from __future__ import annotations

import json
import math
import socket
import struct
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

BAUDRATE = 230400
FRAME_LEN = 47
FRAME_HEADER = bytes([0x54, 0x2C])  # header byte + "12 points" version/length byte
POINTS_PER_FRAME = 12

# CRC8, polynomial 0x4D, init 0 — per the LD19 development manual.
_CRC_TABLE = []
for _i in range(256):
    _c = _i
    for _ in range(8):
        _c = ((_c << 1) ^ 0x4D if _c & 0x80 else _c << 1) & 0xFF
    _CRC_TABLE.append(_c)


def crc8(data: bytes) -> int:
    """CRC8 over ``data``; for a frame, pass its first 46 bytes and compare with the 47th."""
    crc = 0
    for byte in data:
        crc = _CRC_TABLE[crc ^ byte]
    return crc


@dataclass(frozen=True)
class LidarFrame:
    """Twelve consecutive points as the sensor reports them (sensor angles, mm)."""

    speed_dps: float
    points: tuple[tuple[float, int, int], ...]  # (angle_deg, distance_mm, intensity)
    timestamp_ms: int


class FrameParser:
    """Extracts CRC-valid frames from a byte stream, resynchronising on junk."""

    def __init__(self) -> None:
        """Starts with an empty buffer; ``frames`` and ``crc_failures`` count the stream health."""
        self._buf = bytearray()
        self.frames = 0
        self.crc_failures = 0

    def feed(self, data: bytes) -> list[LidarFrame]:
        """Append raw bytes and return the frames they complete.

        Bytes that do not parse are dropped, so joining the stream mid-frame or
        losing a chunk costs one frame instead of desynchronising for good.
        """
        self._buf += data
        frames: list[LidarFrame] = []
        while True:
            start = self._buf.find(FRAME_HEADER)
            if start < 0:
                del self._buf[:-1]  # a lone trailing 0x54 may be the next header
                break
            if len(self._buf) - start < FRAME_LEN:
                del self._buf[:start]
                break
            raw = bytes(self._buf[start : start + FRAME_LEN])
            del self._buf[: start + FRAME_LEN]
            if crc8(raw[:-1]) != raw[-1]:
                self.crc_failures += 1
                continue
            self.frames += 1
            frames.append(_decode(raw))
        return frames


def _decode(raw: bytes) -> LidarFrame:
    """One 47-byte frame into 12 points; per-point angles are interpolated linearly
    between the frame's start and end angle (both in 0.01 deg units)."""
    speed, start_angle = struct.unpack_from("<HH", raw, 2)
    end_angle, timestamp = struct.unpack_from("<HH", raw, 42)
    step = ((end_angle - start_angle) % 36000) / (POINTS_PER_FRAME - 1)
    points = []
    for i in range(POINTS_PER_FRAME):
        dist, intensity = struct.unpack_from("<HB", raw, 6 + 3 * i)
        points.append((((start_angle + step * i) % 36000) / 100.0, dist, intensity))
    return LidarFrame(speed_dps=float(speed), points=tuple(points), timestamp_ms=timestamp)


@dataclass(frozen=True)
class LidarMount:
    """Where the lidar sits on the robot and how its angles map to the robot frame.

    ``masked_sectors_deg`` are in the sensor's own angle domain (before
    mirroring) because the cart posts are fixed relative to the sensor body,
    not to the calibrated forward direction.
    """

    mirror: bool = True  # upside-down mount: sensor angles run clockwise
    yaw_offset_deg: float = 0.0  # robot forward, in sensor degrees after mirroring
    x_m: float = 0.0  # sensor position relative to the wheel-axle centre
    y_m: float = 0.0
    min_range_m: float = 0.05
    max_range_m: float = 12.0
    masked_sectors_deg: tuple[tuple[float, float], ...] = ()

    @classmethod
    def from_json(cls, path: str | Path) -> LidarMount:
        """Load the mount calibration written by the alignment scripts (``config/lidar.json``)."""
        with open(path) as f:
            data = json.load(f)
        data["masked_sectors_deg"] = tuple(tuple(s) for s in data.get("masked_sectors_deg", ()))
        return cls(**data)

    def is_masked(self, sensor_angle_deg: float) -> bool:
        """True inside a blocked sector (cart posts, the mast) — those returns are the
        robot seeing itself. Sectors may wrap through zero, e.g. (350, 10)."""
        a = sensor_angle_deg % 360.0
        return any(
            (lo <= a <= hi) if lo <= hi else (a >= lo or a <= hi)
            for lo, hi in self.masked_sectors_deg
        )

    def to_robot_angle_rad(self, sensor_angle_deg: float) -> float:
        """Sensor degrees to a robot-frame bearing in radians, CCW from forward:
        undo the upside-down mount (mirror), then subtract the forward offset."""
        deg = (360.0 - sensor_angle_deg) if self.mirror else sensor_angle_deg
        return math.radians((deg - self.yaw_offset_deg) % 360.0)


@dataclass(frozen=True)
class LaserScan:
    """One revolution in the robot frame; ``ranges`` is NaN where there is no valid return."""

    stamp: float  # time.monotonic() when the revolution completed
    angles: NDArray[np.float64]  # radians, CCW from robot forward
    ranges: NDArray[np.float64]  # meters
    intensities: NDArray[np.int64]
    speed_rps: float

    def points_xy(self, mount: LidarMount) -> NDArray[np.float64]:
        """Valid returns as (N, 2) Cartesian points in the robot frame, meters.

        NaN ranges are dropped and the sensor's mounting offset is added, so the
        points are referred to the wheel-axle centre and can be mapped directly.
        """
        ok = ~np.isnan(self.ranges)
        r, a = self.ranges[ok], self.angles[ok]
        return np.column_stack((mount.x_m + r * np.cos(a), mount.y_m + r * np.sin(a)))


class ScanAssembler:
    """Groups consecutive frames into revolutions, applying the mount geometry."""

    def __init__(self, mount: LidarMount) -> None:
        """``mount`` supplies the mirroring, forward offset, range limits and masked sectors."""
        self._mount = mount
        self._angles: list[float] = []
        self._ranges: list[float] = []
        self._intensities: list[int] = []
        self._speeds: list[float] = []
        self._last_sensor_angle: float | None = None

    def feed(self, frame: LidarFrame) -> LaserScan | None:
        """Add a frame; return a scan when the sensor angle wraps past zero.

        The wrap is checked per point, not per frame: a 12-point frame spans a
        few degrees, so the revolution boundary usually falls inside one.
        """
        completed: LaserScan | None = None
        for angle_deg, dist_mm, intensity in frame.points:
            if self._last_sensor_angle is not None and angle_deg < self._last_sensor_angle - 180:
                completed = self._finish(frame.speed_dps)
            self._last_sensor_angle = angle_deg
            range_m = dist_mm / 1000.0
            usable = (
                self._mount.min_range_m <= range_m <= self._mount.max_range_m
                and not self._mount.is_masked(angle_deg)
            )
            self._angles.append(self._mount.to_robot_angle_rad(angle_deg))
            self._ranges.append(range_m if usable else math.nan)
            self._intensities.append(intensity)
        self._speeds.append(frame.speed_dps)
        return completed

    def _finish(self, speed_dps: float) -> LaserScan | None:
        """Close the accumulated revolution into a scan and start a fresh one.

        Stamped with ``time.monotonic()`` at the wrap, spin speed averaged over the
        revolution's frames. Returns None if nothing was accumulated yet.
        """
        if not self._angles:
            return None
        scan = LaserScan(
            stamp=time.monotonic(),
            angles=np.array(self._angles),
            ranges=np.array(self._ranges),
            intensities=np.array(self._intensities),
            speed_rps=(sum(self._speeds) / len(self._speeds)) / 360.0,
        )
        self._angles, self._ranges, self._intensities, self._speeds = [], [], [], []
        return scan


class ByteSource(Protocol):
    """Anything that yields raw lidar bytes: a serial port or a bridge socket."""

    def read(self, max_bytes: int) -> bytes:
        """Up to ``max_bytes`` of stream; empty when nothing arrived before the timeout."""
        ...

    def close(self) -> None:
        """Release the port or socket."""
        ...


class TcpSource:
    """Raw bytes from the board's ser2net bridge."""

    def __init__(self, host: str, port: int, timeout_s: float = 0.5) -> None:
        """Connects to ser2net at ``host:port``; ``timeout_s`` bounds a single read."""
        self._sock = socket.create_connection((host, port), timeout=2.0)
        self._sock.settimeout(timeout_s)

    def read(self, max_bytes: int) -> bytes:
        """Whatever the socket has, up to ``max_bytes``; empty on timeout, not an error."""
        try:
            return self._sock.recv(max_bytes)
        except TimeoutError:
            return b""

    def close(self) -> None:
        """Close the bridge socket; the sensor keeps spinning on the robot."""
        self._sock.close()


class SerialSource:
    """Raw bytes from a directly attached UART adapter."""

    def __init__(self, port: str, timeout_s: float = 0.5) -> None:
        """Opens the tty at ``port`` at the LD19's fixed 230400 baud."""
        import serial  # pyserial; only needed for direct USB use

        self._ser = serial.Serial(port, BAUDRATE, timeout=timeout_s)

    def read(self, max_bytes: int) -> bytes:
        """Up to ``max_bytes``, returning whatever arrived before the port timeout."""
        data: bytes = self._ser.read(max_bytes)
        return data

    def close(self) -> None:
        """Close the serial port."""
        self._ser.close()


class LidarStream:
    """Iterates full-revolution scans from a byte source."""

    def __init__(self, source: ByteSource, mount: LidarMount) -> None:
        """Wires a byte source to a parser and assembler; ``parser`` stays public so a
        caller can watch the frame and CRC-failure counts while driving."""
        self._source = source
        self.parser = FrameParser()
        self._assembler = ScanAssembler(mount)

    def scans(self) -> Iterator[LaserScan]:
        """Yield one :class:`LaserScan` per revolution, forever.

        Each read blocks up to the source's own timeout, so a silent sensor costs
        a slow poll rather than a busy loop.
        """
        while True:
            for frame in self.parser.feed(self._source.read(4096)):
                scan = self._assembler.feed(frame)
                if scan is not None:
                    yield scan

    def close(self) -> None:
        """Close the byte source; any partly assembled revolution is discarded."""
        self._source.close()
