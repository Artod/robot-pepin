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

import contextlib
import json
import logging
import socket
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

from pepin.kinematics import Twist
from pepin.odometry import Pose2D

logger = logging.getLogger(__name__)

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


class BaseClient:
    """Laptop side of the base link: reader thread, non-blocking state, fire-and-forget commands.

    Shaped like :class:`pepin.tof.TofClient`: ``start()``/``close()``, reconnects
    on loss, and the newest message is always available without waiting.
    """

    def __init__(self, host: str, port: int = BASE_PORT) -> None:
        """Prepare a client for ``host:port``; nothing connects until :meth:`start`."""
        self._address = (host, port)
        self._sock: socket.socket | None = None
        self._state: BaseState | None = None
        self._received_at = 0.0
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()
        self._pong: dict[str, Any] | None = None
        self._pong_ready = threading.Event()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._warned_disconnected = 0.0
        self.connected = False

    def start(self) -> BaseClient:
        """Begin reading in a daemon thread; returns self so it chains."""
        self._thread.start()
        return self

    def close(self) -> None:
        """Ask the reader to stop and wait (up to 2 s) for the socket to be released."""
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

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
        self._send({"cmd": "twist", "v": twist.linear, "w": twist.angular})

    def stop(self) -> None:
        """Ask the board to stop the wheels now."""
        self._send({"cmd": "stop"})

    def ping(self, timeout_s: float = 3.0) -> dict[str, bool] | None:
        """Which servos answer on the board's bus, or None if the board did not reply in time."""
        self._pong_ready.clear()
        self._send({"cmd": "ping"})
        if not self._pong_ready.wait(timeout_s) or self._pong is None:
            return None
        return {str(k): bool(v) for k, v in self._pong.get("servos", {}).items()}

    # -- internals ----------------------------------------------------------

    def _send(self, message: dict[str, Any]) -> None:
        with self._send_lock:
            sock = self._sock
            if sock is None:
                now = time.monotonic()
                if now - self._warned_disconnected > 2.0:
                    logger.warning("base link down: command %s dropped", message.get("cmd"))
                    self._warned_disconnected = now
                return
            try:
                sock.sendall(encode(message))
            except OSError as exc:
                logger.warning("base link send failed (%s); reconnecting", exc)
                self._drop_socket()

    def _drop_socket(self) -> None:
        sock, self._sock = self._sock, None
        self.connected = False
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._stream()
            except OSError as exc:
                logger.warning("base link lost (%s); reconnecting", exc)
            except Exception:
                logger.exception("base link reader crashed; reconnecting")
            self._drop_socket()
            self._stop.wait(1.0)

    def _stream(self) -> None:
        sock = socket.create_connection(self._address, timeout=2.0)
        sock.settimeout(1.0)
        with self._send_lock:
            self._sock = sock
        self.connected = True
        logger.info("base link connected to %s:%d", *self._address)
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = sock.recv(4096)
            except TimeoutError:
                continue
            if not chunk:
                raise ConnectionError("base server closed the connection")
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if line.strip():
                    self._ingest(json.loads(line))

    def _ingest(self, message: dict[str, Any]) -> None:
        """Route one decoded message: states replace the newest, pongs wake :meth:`ping`."""
        kind = message.get("type")
        if kind == "state":
            state = decode_state(message, time.monotonic())
            with self._lock:
                self._state, self._received_at = state, time.monotonic()
        elif kind == "pong":
            self._pong = message
            self._pong_ready.set()
