"""Base server: runs on the board and owns the wheels in real time.

It talks to the servo bus over the board's own loopback (ser2net on
127.0.0.1:3333, a stable sub-millisecond hop), ticks at 50 Hz — read the
encoders, integrate odometry, apply the latest twist — and publishes its
state to every connected laptop client as JSON lines (:mod:`pepin.base_link`).

Safety lives here, not on the laptop: a deadman stops the wheels when no
twist has arrived for half a second (wifi froze, the script crashed, the
laptop went to sleep); the wheels are armed (torque on) only while someone
is driving and released ten seconds after the last command, so the cart can
always be pushed by hand when idle.

Run on the board::

    python -m pepin.base_server --config /opt/pepin/config/base.json

The pure logic is :class:`BaseServerCore` (unit-tested against a fake bus);
:func:`serve` adds the sockets and the clock.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import socket
import threading
import time
from typing import Any

from pepin.base import LEFT, RIGHT, BusWatchdog, DiffDriveBase, with_suppressed_timeout
from pepin.base_link import BASE_PORT, DEADMAN_S, encode
from pepin.bus import MotorBus, verify_motors
from pepin.feetech import FeetechTcpClient
from pepin.geometry import BaseConfig
from pepin.kinematics import Twist
from pepin.odometry import DiffDriveOdometry
from pepin.telemetry import LatencyTracker

logger = logging.getLogger(__name__)

STOP = Twist(0.0, 0.0)


class BaseServerCore:
    """Wheel ownership as pure logic: commands in, ticks by the clock, state snapshots out."""

    def __init__(
        self,
        bus: MotorBus,
        config: BaseConfig,
        *,
        servo_names: list[str] | None = None,
        deadman_s: float = DEADMAN_S,
        disarm_after_s: float = 10.0,
        latency: LatencyTracker | None = None,
    ) -> None:
        """``servo_names``: the roster :meth:`command` pings; ``latency`` feeds ``bus_p95_ms``."""
        self._bus = bus
        self._base = DiffDriveBase(bus, config)
        self._odom = DiffDriveOdometry(config.geometry)
        self._servo_names = servo_names or [LEFT, RIGHT]
        self._deadman_s = deadman_s
        self._disarm_after_s = disarm_after_s
        self._latency = latency
        self._watchdog = BusWatchdog()
        self.twist = STOP
        self.armed = False
        self.deadman = False
        self.bus_ok = True
        self._last_command_at: float | None = None
        self._acc = [0.0, 0.0]  # wheel travel since the last snapshot
        self._primed = False

    @property
    def moving(self) -> bool:
        """A non-zero twist is being applied."""
        return self.twist.linear != 0.0 or self.twist.angular != 0.0

    def command(self, message: dict[str, Any], now: float) -> dict[str, Any] | None:
        """Apply one client message; returns a reply for requests that have one (``ping``)."""
        cmd = message.get("cmd")
        if cmd == "twist":
            self._last_command_at = now
            self.deadman = False
            twist = Twist(float(message.get("v", 0.0)), float(message.get("w", 0.0)))
            if not self.armed and (twist.linear or twist.angular):
                self._arm()
            self._apply(twist)
        elif cmd == "stop":
            self._last_command_at = now
            self._apply(STOP)
        elif cmd == "ping":
            answers = {name: self._bus.ping(name) is not None for name in self._servo_names}
            return {"type": "pong", "servos": answers}
        else:
            logger.warning("unknown command %r", message)
        return None

    def tick(self, now: float) -> None:
        """One control period: encoders -> odometry, then the deadman and the idle disarm."""
        try:
            travel = self._base.read_wheel_travel()
        except TimeoutError as exc:
            verdict = self._watchdog.failed(now)
            self.bus_ok = False
            if verdict == "stop":
                logger.warning("servos silent for %.1f s: stopping", self._watchdog.stop_after_s)
                with_suppressed_timeout(self._base.stop)
                self.twist = STOP
            elif verdict == "abort":
                logger.error("servos silent for %.0f s: %s", self._watchdog.give_up_after_s, exc)
            return
        if self._watchdog.recovered(now) is not None:
            self._base.reprime()  # an unseen half turn must not alias into a jump
            travel = (0.0, 0.0)
        self.bus_ok = True
        if not self._primed:
            self._primed = True  # the first read only establishes the encoder reference
            return
        self._odom.update(*travel)
        self._acc[0] += travel[0]
        self._acc[1] += travel[1]
        idle = now - self._last_command_at if self._last_command_at is not None else None
        if idle is not None and self.moving and idle > self._deadman_s:
            logger.warning("deadman: no command for %.1f s, stopping", idle)
            self._apply(STOP)
            self.deadman = True
        if idle is not None and self.armed and not self.moving and idle > self._disarm_after_s:
            self._disarm()

    def snapshot(self, now: float) -> dict[str, Any]:
        """The ``state`` message for the clients; resets the accumulated wheel travel."""
        pose = self._odom.pose
        p95 = self._latency.summary().p95_ms if self._latency is not None else 0.0
        message = {
            "type": "state",
            "t": now,
            "x": pose.x,
            "y": pose.y,
            "theta": pose.theta,
            "dl": self._acc[0],
            "dr": self._acc[1],
            "v": self.twist.linear,
            "w": self.twist.angular,
            "moving": self.moving,
            "armed": self.armed,
            "deadman": self.deadman,
            "bus_ok": self.bus_ok,
            "bus_p95_ms": p95,
        }
        self._acc = [0.0, 0.0]
        return message

    def release(self) -> None:
        """Stop and free the wheels (shutdown, or the last client left)."""
        with_suppressed_timeout(lambda: self._apply(STOP))
        if self.armed:
            with_suppressed_timeout(self._disarm)

    def _apply(self, twist: Twist) -> None:
        self._base.set_twist(twist)
        self.twist = twist

    def _arm(self) -> None:
        logger.info("arming: torque on")
        self._base.enable()
        self.armed = True

    def _disarm(self) -> None:
        logger.info("idle: torque off, the cart can be pushed")
        self._base.disable()
        self.armed = False
        self.twist = STOP


def serve(core: BaseServerCore, port: int, tick_hz: float, publish_hz: float) -> None:
    """Sockets and clock around the core: accept, tick, publish state, stop when left alone."""
    server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)  # dual-stack
    server.bind(("::", port))
    server.listen(4)
    logger.info("base server listening on %d", port)
    clients: list[socket.socket] = []
    lock = threading.Lock()
    inbox: queue.Queue[tuple[socket.socket, dict[str, Any]]] = queue.Queue()

    def reader(conn: socket.socket) -> None:
        buffer = b""
        try:
            while True:
                try:
                    chunk = conn.recv(4096)
                except TimeoutError:
                    continue  # the socket timeout is for our sends; a quiet client is fine
                if not chunk:
                    break
                buffer += chunk
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    if line.strip():
                        inbox.put((conn, json.loads(line)))
        except (OSError, ValueError) as exc:
            logger.info("client reader ended: %s", exc)
        with lock:
            if conn in clients:
                clients.remove(conn)
        conn.close()
        with lock:
            alone = not clients
        if alone:
            inbox.put((conn, {"cmd": "stop"}))  # nobody is driving any more

    def accept_loop() -> None:
        while True:
            try:
                conn, peer = server.accept()
            except OSError as exc:
                logger.warning("accept failed: %s", exc)
                time.sleep(0.5)
                continue
            conn.settimeout(0.2)  # a vanished laptop must not stall the tick
            with lock:
                clients.append(conn)
            logger.info("client %s connected", peer[0])
            threading.Thread(target=reader, args=(conn,), daemon=True).start()

    threading.Thread(target=accept_loop, daemon=True).start()
    period, publish_every = 1.0 / tick_hz, 1.0 / publish_hz
    next_publish = time.monotonic()
    try:
        while True:
            started = time.monotonic()
            while True:
                try:
                    conn, message = inbox.get_nowait()
                except queue.Empty:
                    break
                reply = core.command(message, started)
                if reply is not None:
                    _send(conn, encode(reply), clients, lock)
            core.tick(started)
            if started >= next_publish:
                next_publish = started + publish_every
                line = encode(core.snapshot(started))
                with lock:
                    targets = list(clients)
                for conn in targets:
                    _send(conn, line, clients, lock)
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        core.release()


def _send(
    conn: socket.socket, line: bytes, clients: list[socket.socket], lock: threading.Lock
) -> None:
    try:
        conn.sendall(line)
    except OSError:
        with lock:
            if conn in clients:
                clients.remove(conn)
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Own the wheels on the board; serve them over TCP."
    )
    parser.add_argument("--config", default="/opt/pepin/config/base.json")
    parser.add_argument("--bus-host", default="127.0.0.1", help="ser2net host for the servo bus")
    parser.add_argument("--bus-port", type=int, default=3333)
    parser.add_argument("--port", type=int, default=BASE_PORT)
    parser.add_argument("--tick-hz", type=float, default=50.0)
    parser.add_argument("--publish-hz", type=float, default=20.0)
    parser.add_argument(
        "--servos", default="1-10", help="bus ids the ping command checks, e.g. 1-10"
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s"
    )

    config = BaseConfig.from_json(args.config)
    motors = DiffDriveBase.motor_ids(config)
    first, last = (int(x) for x in args.servos.split("-"))
    for motor_id in range(first, last + 1):
        if motor_id not in motors.values():
            motors[f"servo{motor_id}"] = motor_id
    with FeetechTcpClient(args.bus_host, args.bus_port, motors, retries=1) as bus:
        verify_motors(bus, [LEFT, RIGHT])
        core = BaseServerCore(bus, config, servo_names=list(motors), latency=bus.latency)
        serve(core, args.port, args.tick_hz, args.publish_hz)


if __name__ == "__main__":
    main()
