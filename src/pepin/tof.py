"""Near-field time-of-flight ranges from the board and the stop reflex built on them.

The lidar sees one horizontal slice of the world; the three VL53L1X
sensors look where it cannot (low, in front) and feed two things: a reflex
that refuses to drive into something close, and — later — an obstacle
layer for navigation. Localisation never uses them.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
import time
from dataclasses import dataclass

from pepin.kinematics import Twist

logger = logging.getLogger(__name__)

TOF_PORT = 3335


@dataclass(frozen=True)
class TofRanges:
    """Latest ranges in meters (None = no return) and how old they are."""

    front: float | None
    left: float | None
    right: float | None
    age_s: float


class TofClient:
    """Reads the board's JSON-lines range stream in a background thread, reconnecting on loss.

    ``ranges()`` never blocks; ``age_s`` tells the caller how stale the data
    is (infinite until the first record), so a dead stream is visible instead
    of silently reporting "nothing close".
    """

    def __init__(self, host: str, port: int = TOF_PORT) -> None:
        self._address = (host, port)
        self._latest: dict[str, float | None] = {"front": None, "left": None, "right": None}
        self._stamp = 0.0
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.connected = False

    def start(self) -> TofClient:
        self._thread.start()
        return self

    def close(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._stream()
            except OSError as exc:
                logger.warning("tof stream lost (%s); reconnecting", exc)
            self.connected = False
            self._stop.wait(1.0)

    def _stream(self) -> None:
        sock = socket.create_connection(self._address, timeout=2.0)
        sock.settimeout(1.0)
        self.connected = True
        logger.info("tof stream connected to %s:%d", *self._address)
        buffer = b""
        with sock:
            while not self._stop.is_set():
                try:
                    chunk = sock.recv(4096)
                except TimeoutError:
                    continue
                if not chunk:
                    raise ConnectionError("stream closed")
                buffer += chunk
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    if line.strip():
                        self._ingest(json.loads(line))

    def _ingest(self, record: dict[str, float | None]) -> None:
        with self._lock:
            for name in ("front", "left", "right"):
                mm = record.get(name)
                self._latest[name] = None if mm is None or mm <= 0 else mm / 1000.0
            self._stamp = time.monotonic()

    def ranges(self) -> TofRanges:
        """Latest ranges in meters plus their age; ``age_s`` is infinite before the first record."""
        with self._lock:
            age = time.monotonic() - self._stamp if self._stamp else float("inf")
            return TofRanges(
                self._latest["front"], self._latest["left"], self._latest["right"], age
            )


@dataclass(frozen=True)
class ReflexConfig:
    """Distances (m) below which forward motion is refused, and how stale data may be."""

    front_stop_m: float = 0.22
    side_stop_m: float = 0.30  # down-tilted sensors read the floor at ~0.44 m; nearer = obstacle
    max_age_s: float = 0.5
    blocked_when_stale: bool = False


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
    if ranges.front is not None and ranges.front < config.front_stop_m:
        return ReflexDecision(Twist(0.0, command.angular), True, f"front {ranges.front:.2f} m")
    for name, value in (("left", ranges.left), ("right", ranges.right)):
        if value is not None and value < config.side_stop_m:
            return ReflexDecision(Twist(0.0, command.angular), True, f"{name} {value:.2f} m")
    return ReflexDecision(command, blocked=False)
