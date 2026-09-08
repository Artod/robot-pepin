"""Wire format of the board's servers, with no ROS in sight.

The base server (:3336) and the ToF server (:3335) both speak newline-delimited
JSON over TCP: they publish one object per line forever and accept command
lines back. This module is that format and nothing else — parse a line, encode
a command — so it can be unit-tested on a laptop with neither ROS nor a robot
in the room. The nodes in this package add the sockets and the messages.

Wire format::

    base -> us   {"type": "state", "t": .., "x": .., "y": .., "theta": ..,
                  "dl": .., "dr": .., "v": .., "w": .., "moving": bool,
                  "armed": bool, "deadman": bool, "bus_ok": bool, "bus_p95_ms": ..}
    base -> us   {"type": "pong", ...}                    answer to a ping; ignored here
    us -> base   {"cmd": "twist", "v": <m/s>, "w": <rad/s>}   drive; re-arms the deadman
    us -> base   {"cmd": "stop"}                              stop the wheels now
    tof -> us    {"t": .., "front": <mm|null>, "left": .., "right": ..}

Conventions everywhere: x forward, y left, theta counter-clockwise, SI units
(the ToF server is the one exception — it speaks millimetres).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

TOF_NAMES = ("front", "left", "right")

# Covariance of one odometry sample, the diff_drive_controller defaults: wheel odometry is
# precise per tick and hopeless over a long run, and the drift is a consumer's problem (a
# filter, or SLAM correcting odom->map), not something a fixed 6x6 can express. z/roll/pitch
# get the same small number because the robot cannot leave the floor.
_POSE_VARIANCES = (0.001, 0.001, 0.001, 0.001, 0.001, 0.01)
_TWIST_VARIANCES = (0.001, 0.001, 0.001, 0.001, 0.001, 0.01)


@dataclass(frozen=True)
class BaseState:
    """One state line from the base server: where the wheels think they are, and how they feel."""

    stamp_s: float  # board clock (time.monotonic there) when the line was made
    x: float  # wheel odometry integrated on the board, odometry frame, metres
    y: float
    theta: float  # radians, counter-clockwise from x
    d_left_m: float  # left wheel travel since the previous state line
    d_right_m: float
    v: float  # twist currently applied, m/s forward
    w: float  # rad/s counter-clockwise
    moving: bool  # a non-zero twist is being applied
    armed: bool  # torque on (the wheels resist being pushed)
    deadman: bool  # the board stopped the wheels because commands stopped arriving
    bus_ok: bool  # the servos answered on the last tick
    bus_p95_ms: float  # board-local servo round trip, 95th percentile


def parse_state(message: dict[str, Any]) -> BaseState | None:
    """A ``state`` line as a :class:`BaseState`; ``None`` for any other or malformed message."""
    if message.get("type") != "state":
        return None
    try:
        return BaseState(
            stamp_s=float(message["t"]),
            x=float(message["x"]),
            y=float(message["y"]),
            theta=float(message["theta"]),
            d_left_m=float(message["dl"]),
            d_right_m=float(message["dr"]),
            v=float(message["v"]),
            w=float(message["w"]),
            moving=bool(message["moving"]),
            armed=bool(message["armed"]),
            deadman=bool(message["deadman"]),
            bus_ok=bool(message["bus_ok"]),
            bus_p95_ms=float(message.get("bus_p95_ms", 0.0)),
        )
    except (KeyError, TypeError, ValueError):
        return None


def parse_tof(message: dict[str, Any]) -> dict[str, float | None]:
    """A ToF line as metres per sensor; ``None`` where the sensor got no valid return."""
    ranges: dict[str, float | None] = {}
    for name in TOF_NAMES:
        value = message.get(name)
        ranges[name] = None if value is None else float(value) / 1000.0
    return ranges


def parse_tof_status(message: dict[str, Any]) -> dict[str, int | None]:
    """The VL53L1X range status per sensor: 0 is a measurement, anything else says why not."""
    status = message.get("status") or {}
    return {name: status.get(name) for name in TOF_NAMES}


def encode_twist(v: float, w: float) -> bytes:
    """One ``twist`` command line: ``v`` m/s forward, ``w`` rad/s counter-clockwise."""
    return _line({"cmd": "twist", "v": float(v), "w": float(w)})


def encode_stop() -> bytes:
    """One ``stop`` command line: the board cuts the wheels on receipt."""
    return _line({"cmd": "stop"})


def odometry_pose_covariance() -> list[float]:
    """Row-major 6x6 pose covariance for a nav_msgs/Odometry from wheel odometry."""
    return _diagonal(_POSE_VARIANCES)


def odometry_twist_covariance() -> list[float]:
    """Row-major 6x6 twist covariance for a nav_msgs/Odometry from wheel odometry."""
    return _diagonal(_TWIST_VARIANCES)


class LineReader:
    """Reassembles JSON objects out of arbitrary TCP chunks.

    TCP hands out bytes, not lines: one ``recv`` can hold three messages and
    half of a fourth. Feed it what arrives and take back whole objects. A line
    that is not a JSON object costs that line and nothing else, and a stream
    that never sends a newline cannot grow the buffer past ``max_line_bytes``.
    """

    def __init__(self, max_line_bytes: int = 1 << 16) -> None:
        """Prepare an empty reader that drops any line longer than ``max_line_bytes``."""
        self._max_line_bytes = max_line_bytes
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> list[dict[str, Any]]:
        """Add received bytes; return every complete JSON object they finished, in order."""
        self._buffer.extend(chunk)
        messages: list[dict[str, Any]] = []
        while b"\n" in self._buffer:
            line, _, rest = self._buffer.partition(b"\n")
            self._buffer = bytearray(rest)
            message = _decode(bytes(line))
            if message is not None:
                messages.append(message)
        if len(self._buffer) > self._max_line_bytes:
            self._buffer.clear()  # no newline in sight: the sender is not talking our language
        return messages


def _decode(line: bytes) -> dict[str, Any] | None:
    """One raw line into a JSON object, or ``None`` if it is not one."""
    if not line.strip():
        return None
    try:
        message = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    return message if isinstance(message, dict) else None


def _line(message: dict[str, Any]) -> bytes:
    """One JSON object as a single line of bytes, newline included."""
    return (json.dumps(message, separators=(",", ":")) + "\n").encode()


def _diagonal(variances: tuple[float, ...]) -> list[float]:
    """A row-major 6x6 matrix with ``variances`` on the diagonal and zeros elsewhere."""
    matrix = [0.0] * 36
    for i, variance in enumerate(variances):
        matrix[i * 6 + i] = variance
    return matrix
