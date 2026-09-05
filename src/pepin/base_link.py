"""The base link: the board owns the wheels in real time; the laptop talks to it in messages.

Wifi is fast on average and frozen for half a second now and then. A control
loop that waits for a servo reply across it inherits every freeze, so the loop
that must never wait — read the encoders, integrate odometry, write the wheel
speeds, stop when nobody is talking — runs on the board next to the UART
(:mod:`pepin.base_server`), and the laptop exchanges JSON lines with it: twist
commands down, odometry state up. :meth:`BaseClient.state` never blocks; its
``age_s`` says how stale the board's last word is.

Wire format, one JSON object per line in both directions::

    laptop -> board  {"cmd": "twist", "v": <m/s>, "w": <rad/s>}   drive; re-arms the deadman
                     {"cmd": "stop"}                              stop now
                     {"cmd": "ping"}                              which servos answer on the bus
    board -> laptop  {"type": "state", ...}                       see :class:`BaseState`, ~20 Hz
                     {"type": "pong", "servos": {"7": true, ...}}
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

from pepin.kinematics import Twist
from pepin.odometry import Pose2D
from pepin.streams import Connector, JsonLinesClient

BASE_PORT = 3336
DEADMAN_S = 0.5  # the board stops the wheels when no twist arrived for this long


@dataclass(frozen=True)
class BaseState:
    """The board's last word about the wheels: odometry, what it is doing, and how it feels."""

    pose: Pose2D  # wheel odometry integrated on the board (odometry frame)
    d_left_m: float  # left wheel travel since the previous state message
    d_right_m: float
    v: float  # twist currently applied, m/s
    w: float  # rad/s
    moving: bool  # a non-zero twist is being applied
    armed: bool  # torque on (the wheels resist being pushed)
    deadman: bool  # the board stopped the wheels because commands stopped arriving
    bus_ok: bool  # the servos answered on the last tick
    bus_p95_ms: float  # board-local servo round trip, 95th percentile
    stamp_s: float  # board clock (time.monotonic there) when the message was made
    age_s: float  # laptop clock: seconds since this message arrived


def encode(message: dict[str, Any]) -> bytes:
    """One message as a JSON line ready for the socket."""
    return (json.dumps(message, separators=(",", ":")) + "\n").encode()


def decode_state(message: dict[str, Any], received_at: float) -> BaseState:
    """A ``state`` message from the board into a :class:`BaseState` (age 0 at ``received_at``)."""
    return BaseState(
        pose=Pose2D(float(message["x"]), float(message["y"]), float(message["theta"])),
        d_left_m=float(message["dl"]),
        d_right_m=float(message["dr"]),
        v=float(message["v"]),
        w=float(message["w"]),
        moving=bool(message["moving"]),
        armed=bool(message["armed"]),
        deadman=bool(message["deadman"]),
        bus_ok=bool(message["bus_ok"]),
        bus_p95_ms=float(message.get("bus_p95_ms", 0.0)),
        stamp_s=float(message["t"]),
        age_s=0.0,
    )


class BaseClient(JsonLinesClient):
    """Laptop side of the base link: non-blocking state, fire-and-forget commands, a ping."""

    def __init__(
        self, host: str, port: int = BASE_PORT, *, connector: Connector | None = None
    ) -> None:
        """Prepare a client for ``host:port``; nothing connects until :meth:`start`."""
        super().__init__(host, port, name="base", connector=connector)
        self._state: BaseState | None = None
        self._received_at = 0.0
        self._lock = threading.Lock()
        self._pong: dict[str, Any] | None = None
        self._pong_ready = threading.Event()

    def state(self, now: float | None = None) -> BaseState | None:
        """Newest state with ``age_s`` measured at ``now``; None before the first message."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._state is None:
                return None
            return replace(self._state, age_s=now - self._received_at)

    def wait_for_state(self, timeout_s: float = 5.0) -> BaseState | None:
        """Block up to ``timeout_s`` for the first message from the board (start-up only)."""
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            state = self.state()
            if state is not None:
                return state
            time.sleep(0.05)
        return None

    def set_twist(self, twist: Twist) -> None:
        """Ask the board for this body velocity; also re-arms its deadman timer."""
        self.send({"cmd": "twist", "v": twist.linear, "w": twist.angular})

    def stop(self) -> None:
        """Ask the board to stop the wheels now."""
        self.send({"cmd": "stop"})

    def ping(self, timeout_s: float = 3.0) -> dict[str, bool] | None:
        """Which servos answer on the board's bus, or None if the board did not reply in time."""
        self._pong_ready.clear()
        self.send({"cmd": "ping"})
        if not self._pong_ready.wait(timeout_s) or self._pong is None:
            return None
        return {str(k): bool(v) for k, v in self._pong.get("servos", {}).items()}

    def _ingest(self, message: dict[str, Any]) -> None:
        """Route one decoded message: states replace the newest, pongs wake :meth:`ping`."""
        kind = message.get("type")
        if kind == "state":
            now = time.monotonic()
            state = decode_state(message, now)
            with self._lock:
                self._state, self._received_at = state, now
        elif kind == "pong":
            self._pong = message
            self._pong_ready.set()
