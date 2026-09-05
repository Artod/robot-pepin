"""Base server: runs on the board and owns the wheels in real time.

It talks to the servo bus over the board's own loopback (ser2net on
127.0.0.1:3333, a stable sub-millisecond hop), ticks at 50 Hz — read the
encoders, integrate odometry, apply the latest twist — and publishes its
state to every connected laptop client as JSON lines (:mod:`pepin.base_link`).

Safety lives here, not on the laptop: a deadman stops the wheels when no
twist has arrived for half a second (wifi froze, the script crashed, the
laptop went to sleep); the wheels are armed (torque on) only while someone
is driving and released ten seconds after the last motion, so the cart can
always be pushed by hand when idle. Nothing that talks to a laptop runs on
the tick thread: each client has its own reader and writer threads, and a
laptop that stops reading is dropped, not waited for.

Run on the board::

    python -m pepin.base_server --config /opt/pepin/config/base.json

The pure logic is :class:`BaseServerCore` (unit-tested against a fake bus);
:func:`serve` adds the clock, :class:`pepin.streams.JsonLinesServer` the sockets.
"""

from __future__ import annotations

import argparse
import logging
import threading
import time
from typing import Any

from pepin.base import LEFT, RIGHT, BusWatchdog, DiffDriveBase, with_suppressed_timeout
from pepin.base_link import BASE_PORT, DEADMAN_S
from pepin.bus import MotorBus, verify_motors
from pepin.feetech import FeetechTcpClient
from pepin.geometry import BaseConfig
from pepin.kinematics import STOP, Twist
from pepin.odometry import DiffDriveOdometry
from pepin.streams import JsonLinesServer
from pepin.telemetry import LatencyTracker

logger = logging.getLogger(__name__)


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
        self._last_command_at: float | None = None  # any twist: feeds the deadman
        self._last_motion_at: float | None = None  # a non-zero twist: feeds the idle release
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
            if twist.linear or twist.angular:
                self._last_motion_at = now
                if not self.armed:
                    self._arm()
            self._apply(twist)
        elif cmd == "stop":
            self._last_command_at = now
            self._apply(STOP)
        elif cmd == "release":
            # The last client left: stop, but leave the deadman clock alone.
            self._apply(STOP)
        elif cmd == "ping":
            if self.moving:
                # Pinging a dozen servos blocks the bus for up to 0.4 s per silent id;
                # never while the wheels turn (the deadman and the encoders live here).
                return {"type": "pong", "busy": True}
            answers = {name: self._bus.ping(name) is not None for name in self._servo_names}
            return {"type": "pong", "servos": answers}
        else:
            logger.warning("unknown command %r", message)
        return None

    def tick(self, now: float) -> None:
        """One control period: encoders -> odometry, then the deadman and the idle disarm.

        Raises ``RuntimeError`` when the servos have been silent for the
        watchdog's give-up time: the service exits and systemd restarts it.
        """
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
                raise RuntimeError(
                    f"servos silent for {self._watchdog.give_up_after_s:.0f} s: {exc}"
                ) from exc
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
        last_cmd, last_motion = self._last_command_at, self._last_motion_at
        if self.moving and last_cmd is not None and now - last_cmd > self._deadman_s:
            logger.warning("deadman: no command for %.1f s, stopping", now - last_cmd)
            self._apply(STOP)
            self.deadman = True
        idle = self.armed and not self.moving and last_motion is not None
        if idle and last_motion is not None and now - last_motion > self._disarm_after_s:
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
        """Stop and free the wheels (shutdown)."""
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


def serve(
    core: BaseServerCore,
    server: JsonLinesServer,
    tick_hz: float,
    publish_hz: float,
    stop: threading.Event | None = None,
) -> None:
    """Clock around the core: apply commands, tick, publish state; release when left alone."""
    period, publish_every = 1.0 / tick_hz, 1.0 / publish_hz
    next_publish = time.monotonic()
    try:
        while stop is None or not stop.is_set():
            started = time.monotonic()
            for client, message in server.commands():
                try:
                    reply = core.command(message, started)
                except Exception:  # one malformed command must not take the wheels down
                    who = client.peer if client is not None else "?"
                    logger.exception("bad command %r from %s; ignored", message, who)
                    continue
                if reply is not None:
                    server.reply(client, reply)
            core.tick(started)
            if started >= next_publish:
                next_publish = started + publish_every
                server.broadcast(core.snapshot(started))
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        core.release()
        server.close()


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
        server = JsonLinesServer(args.port, on_last_client_left={"cmd": "release"}).start()
        serve(core, server, args.tick_hz, args.publish_hz)


if __name__ == "__main__":
    main()
