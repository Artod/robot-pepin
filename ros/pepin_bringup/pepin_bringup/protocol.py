"""Wire format of the board's ToF server, with no ROS in sight.

The ToF server (:3335) speaks newline-delimited JSON over TCP: one object per line, forever.
This module is that format and nothing else — reassemble lines, parse a reading — so it can be
unit-tested on a laptop with neither ROS nor a robot in the room. The nodes in this package add
the sockets and the messages. (The base server's half lives in C++ beside the node that reads
it: ros/pepin_base_cpp/include/pepin_base_cpp/protocol.hpp.)

Wire format::

    tof -> us    {"t": .., "front": <mm|null>, "left": .., "right": ..,
                  "status": {"front": <int>, ...}}

The ToF server speaks millimetres; everything here leaves in metres.
"""

from __future__ import annotations

import json
from typing import Any

TOF_NAMES = ("front", "left", "right")


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
