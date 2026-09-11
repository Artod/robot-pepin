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

The neck's two servos share the bus. A ``neck`` request answers their
encoders (:class:`NeckReader`): one extra bus round trip, cached so that a
client polling at any rate costs at most twenty reads a second, and after a
silent servo retried only at rest — a silent id costs 0.4 s of this thread,
and the wheels come first. ``neck_goto`` and ``neck_home`` move it
(:class:`NeckMover`): the goal is written in a few short transactions and the
arrival is watched one tick at a time, so a three-second head turn never makes
the deadman wait. Both are refused while the wheels turn, and the servos are
released again when the head arrives.

Run on the board::

    python -m pepin.base_server --config /opt/pepin/config/base.json

The pure logic is :class:`BaseServerCore` (unit-tested against a fake bus);
:func:`serve` adds the clock, :class:`pepin.streams.JsonLinesServer` the sockets.
"""

from __future__ import annotations

import argparse
import logging
import signal
import threading
import time
from dataclasses import dataclass
from typing import Any

from pepin.base import LEFT, RIGHT, BusWatchdog, DiffDriveBase, with_suppressed_timeout
from pepin.base_link import BASE_PORT, DEADMAN_S
from pepin.bus import MotorBus, verify_motors
from pepin.feetech import FeetechTcpClient
from pepin.geometry import BaseConfig
from pepin.kinematics import STOP, Twist
from pepin.neck import PAN, TILT, NeckConfig, neck_servo_ids
from pepin.odometry import DiffDriveOdometry
from pepin.streams import JsonLinesServer
from pepin.telemetry import LatencyTracker

logger = logging.getLogger(__name__)

# The commands that make a client a driver of the wheels; the wheels are released when the last
# such client leaves, whoever else is still connected and merely asking (pepin.streams).
DRIVING_COMMANDS = frozenset({"twist", "stop"})

POSITION_MODE = 0  # Operating_Mode of a servo that obeys Goal_Position; 1 would spin forever
NECK_PROFILE_SPEED = 400  # ticks/s, ~35 deg/s: the gentle profile speed scripts/jog.py moves at
NECK_TOLERANCE_TICKS = 4  # ~0.35 deg: arrived, as far as a 12-bit encoder is concerned
NECK_MOVE_TIMEOUT_S = 3.0  # a move that has not arrived by then is answered as not reached


class NeckReader:
    """The neck's encoders through the wheels' bus, cached: refreshed at most every ``period_s``,
    and after a silent servo retried only at rest and after ``retry_s`` — a silent id costs
    0.4 s of the tick thread, and the deadman lives there."""

    def __init__(
        self,
        bus: MotorBus,
        names: tuple[str, str] = (PAN, TILT),
        *,
        period_s: float = 0.05,
        retry_s: float = 5.0,
    ) -> None:
        """``names`` are the (pan, tilt) motor names on ``bus``."""
        self._bus = bus
        self._names = names
        self._period_s = period_s
        self._retry_s = retry_s
        self._ticks: tuple[int, int] | None = None
        self._read_at: float | None = None
        self._read_s = 0.0  # the last round trip, seconds
        self._failed_at: float | None = None
        self._error: str | None = None
        self.latency = LatencyTracker("neck.read")

    def read(self, now: float, *, at_rest: bool) -> dict[str, Any]:
        """The ``neck`` reply at ``now``: fresh from the bus when the cache is older than the
        period (a failed servo is retried only ``at_rest``), the cache otherwise."""
        fresh = self._read_at is not None and now - self._read_at < self._period_s
        failed_at = self._failed_at
        held = failed_at is not None and (not at_rest or now - failed_at < self._retry_s)
        if not fresh and not held:
            self._refresh(now)
        return self._reply(now)

    def _refresh(self, now: float) -> None:
        started = time.perf_counter()
        try:
            raw = self._bus.sync_read("Present_Position", list(self._names), normalize=False)
        except (TimeoutError, OSError) as exc:
            self._failed_at, self._error = now, str(exc)
            logger.warning("neck encoders: %s", exc)
            return
        self._read_s = time.perf_counter() - started
        self.latency.add(self._read_s)
        self._ticks = (raw[self._names[0]], raw[self._names[1]])
        self._read_at = now
        self._failed_at, self._error = None, None

    def _reply(self, now: float) -> dict[str, Any]:
        reply: dict[str, Any] = {"type": "neck"}
        if self._ticks is not None and self._read_at is not None:
            reply["pan_ticks"], reply["tilt_ticks"] = self._ticks
            reply["age_s"] = now - self._read_at
            reply["read_ms"] = self._read_s * 1000.0
        if self._error is not None:
            reply["error"] = self._error
        return reply


def neck_error(text: str) -> dict[str, Any]:
    """A refused neck move, in the shape of the answer a finished one would have had."""
    return {"type": "neck_goto", "reached": False, "error": text}


@dataclass
class NeckMove:
    """One move under way: the goals written, whether to keep the torque, when it started
    and when it gives up (monotonic seconds)."""

    targets: dict[str, int]
    hold: bool
    started: float
    deadline: float


class NeckMover:
    """The neck's write side: one move at a time, stepped from the tick thread.

    A move is never waited for. :meth:`start` writes the goal in a few short bus transactions
    and returns; :meth:`step`, once per tick, watches the same cached encoder reading the
    ``neck`` command answers from until the head is within tolerance or the deadline passes,
    and then releases the servos unless the caller asked to hold them. The wheels' deadman
    therefore never waits behind a head turn, and the move costs no bus reads of its own.

    Only the addressed servos are ever touched: the wheels' torque lives in
    :class:`pepin.base.DiffDriveBase` and is scoped to the two wheel names.
    """

    def __init__(
        self,
        bus: MotorBus,
        reader: NeckReader,
        config: NeckConfig,
        *,
        speed: int = NECK_PROFILE_SPEED,
        tolerance_ticks: int = NECK_TOLERANCE_TICKS,
        timeout_s: float = NECK_MOVE_TIMEOUT_S,
    ) -> None:
        """``reader`` is the cached encoder reader a move watches; ``speed`` the profile speed
        in ticks per second; ``config`` the limits and the reference pose of config/neck.json."""
        self._bus = bus
        self._reader = reader
        self._cfg = config
        self._speed = speed
        self._tolerance = tolerance_ticks
        self._timeout_s = timeout_s
        self._move: NeckMove | None = None
        self._energised: list[str] = []

    def home(self, *, hold: bool, now: float) -> dict[str, Any] | None:
        """Send the neck to the reference pose of config/neck.json: an error reply when those
        ticks were never read, otherwise whatever :meth:`start` answers."""
        ref = self._cfg.reference
        if ref.pan_ticks is None or ref.tilt_ticks is None:
            return neck_error("the reference ticks are unread in config/neck.json")
        return self.start(ref.pan_ticks, ref.tilt_ticks, hold=hold, now=now)

    def start(
        self, pan_ticks: int | None, tilt_ticks: int | None, *, hold: bool, now: float
    ) -> dict[str, Any] | None:
        """Begin a move to those encoder ticks; None addresses no servo on that axis.

        Returns an error reply when the move is refused — a target outside the configured
        limits (never clamped: a wrong number is a mistake, and a silently smaller move hides
        it), no target at all, a move already under way, a servo not in position mode, a bus
        that would not take the write — and None once the goals are on the bus.
        """
        if self._move is not None:
            return neck_error("a neck move is already under way")
        targets: dict[str, int] = {}
        for joint, target in ((self._cfg.pan, pan_ticks), (self._cfg.tilt, tilt_ticks)):
            if target is None:
                continue
            if not joint.within_limits(target):
                return neck_error(
                    f"{joint.name} target {target} is outside its limits "
                    f"{joint.min_ticks}..{joint.max_ticks} ticks"
                )
            targets[joint.name] = target
        if not targets:
            return neck_error("neck_goto needs pan_ticks or tilt_ticks")
        try:
            refused = self._not_position_mode(list(targets))
            if refused is not None:
                return refused
            self._bus.enable_torque(list(targets))
            self._energised = list(targets)
            for name, goal in targets.items():
                # In position mode Goal_Velocity is the profile (maximum) speed, not a command.
                self._bus.sync_write("Goal_Velocity", {name: self._speed}, normalize=False)
                self._bus.sync_write("Goal_Position", {name: goal}, normalize=False)
        except (TimeoutError, OSError) as exc:
            self.release()
            return neck_error(f"the bus refused the move: {exc}")
        self._move = NeckMove(targets, hold, now, now + self._timeout_s)
        return None

    def step(self, now: float, *, at_rest: bool) -> dict[str, Any] | None:
        """One tick of a move under way: the final reply once the head has arrived or the
        deadline passed (torque off unless the move asked to hold), None while it still moves
        and None when no move is under way."""
        move = self._move
        if move is None:
            return None
        reading = self._reader.read(now, at_rest=at_rest)
        here: dict[str, int | None] = {
            self._cfg.pan.name: reading.get("pan_ticks"),
            self._cfg.tilt.name: reading.get("tilt_ticks"),
        }

        def arrived(name: str, goal: int) -> bool:
            """Whether that servo's newest reading is within tolerance of its goal."""
            ticks = here.get(name)
            return ticks is not None and abs(ticks - goal) <= self._tolerance

        reached = all(arrived(name, goal) for name, goal in move.targets.items())
        if not reached and now < move.deadline:
            return None
        self._move = None
        if not move.hold:
            self.release()
        reply: dict[str, Any] = {
            "type": "neck_goto",
            "pan_ticks": here[self._cfg.pan.name],
            "tilt_ticks": here[self._cfg.tilt.name],
            "reached": reached,
            "ms": (now - move.started) * 1000.0,
            "hold": move.hold,
        }
        if reading.get("error") is not None:
            reply["error"] = str(reading["error"])
        return reply

    def release(self) -> None:
        """Torque off whatever this mover energised — the end of a move, or shutdown with a
        held neck — and forget any move; a bus that will not take it is logged, not raised."""
        names, self._energised = self._energised, []
        self._move = None
        if not names:
            return
        try:
            self._bus.disable_torque(names)
        except (TimeoutError, OSError) as exc:
            logger.warning("neck torque off failed: %s", exc)

    def _not_position_mode(self, names: list[str]) -> dict[str, Any] | None:
        """One bus read before every move: a servo left in velocity mode (scripts/jog.py wheel
        writes that to EEPROM) would read the profile speed as a command and turn the head
        forever. The refusal reply naming the modes, or None when both obey Goal_Position."""
        modes = self._bus.sync_read("Operating_Mode", names, normalize=False)
        wrong = {name: mode for name, mode in modes.items() if mode != POSITION_MODE}
        if wrong:
            return neck_error(
                f"not in position mode, Operating_Mode {wrong}: refusing to write a goal"
            )
        return None


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
        neck: NeckReader | None = None,
        mover: NeckMover | None = None,
    ) -> None:
        """``servo_names``: the roster :meth:`command` pings; ``latency`` feeds ``bus_p95_ms``;
        ``neck`` answers the ``neck`` command and ``mover`` the two move commands (None for
        either: that command answers an error)."""
        self._bus = bus
        self._base = DiffDriveBase(bus, config)
        self._odom = DiffDriveOdometry(config.geometry)
        self._servo_names = servo_names or [LEFT, RIGHT]
        self._deadman_s = deadman_s
        self._disarm_after_s = disarm_after_s
        self._latency = latency
        self._neck = neck
        self._mover = mover
        self._replies: list[dict[str, Any]] = []
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
        """Apply one client message; returns a reply for requests that have one (``ping``,
        ``neck``, a refused ``neck_goto``/``neck_home``).

        An accepted move answers nothing here: its reply is born when the head stops, and
        leaves through :meth:`take_replies`.
        """
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
        elif cmd == "neck":
            if self._neck is None:
                return {"type": "neck", "error": "no neck configured on this base server"}
            return self._neck.read(now, at_rest=not self.moving)
        elif cmd in ("neck_goto", "neck_home"):
            return self._start_neck_move(cmd, message, now)
        else:
            logger.warning("unknown command %r", message)
        return None

    def tick(self, now: float) -> None:
        """One control period: encoders -> odometry, the deadman and the idle disarm, and then
        one step of a neck move if one is under way — the wheels first, always.

        Raises ``RuntimeError`` when the servos have been silent for the
        watchdog's give-up time: the service exits and systemd restarts it.
        """
        self._tick_wheels(now)
        self._step_neck(now)

    def take_replies(self) -> list[dict[str, Any]]:
        """Answers that outlived the command that asked for them (a finished neck move), and
        empties the list; :func:`serve` broadcasts them — the client that asked is only one of
        the readers, and by then it may be gone."""
        replies, self._replies = self._replies, []
        return replies

    def _tick_wheels(self, now: float) -> None:
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
        """Stop and free the wheels, and the neck with them (shutdown)."""
        with_suppressed_timeout(lambda: self._apply(STOP))
        if self.armed:
            with_suppressed_timeout(self._disarm)
        if self._mover is not None:
            # A neck left holding would stay energised with nobody left to release it.
            with_suppressed_timeout(self._mover.release)

    def _start_neck_move(
        self, cmd: str, message: dict[str, Any], now: float
    ) -> dict[str, Any] | None:
        """Hand one ``neck_goto``/``neck_home`` to the mover: the reply of a refusal, None once
        the move is under way."""
        if self._mover is None:
            return neck_error("no neck configured on this base server")
        if self.moving:
            # Writing to a silent servo costs this thread 0.4 s, and the deadman lives here.
            return neck_error("the wheels are moving")
        hold = bool(message.get("hold", False))
        if cmd == "neck_home":
            return self._mover.home(hold=hold, now=now)
        try:
            pan, tilt = _target(message, "pan_ticks"), _target(message, "tilt_ticks")
        except (TypeError, ValueError) as exc:
            return neck_error(f"bad target: {exc}")
        return self._mover.start(pan, tilt, hold=hold, now=now)

    def _step_neck(self, now: float) -> None:
        """Advance a neck move by one tick; its finished answer joins :meth:`take_replies`."""
        if self._mover is None:
            return
        reply = self._mover.step(now, at_rest=not self.moving)
        if reply is not None:
            self._replies.append(reply)

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
            for late in core.take_replies():
                server.broadcast(late)  # a move's answer: the asking client reads it like a state
            if started >= next_publish:
                next_publish = started + publish_every
                server.broadcast(core.snapshot(started))
            time.sleep(max(0.0, period - (time.monotonic() - started)))
    finally:
        core.release()
        server.close()


def stop_on_sigterm() -> threading.Event:
    """An event that SIGTERM sets: systemd stops us with it, and the wheels must be released.

    Without this the default handler kills the process outright, ``serve``'s
    ``finally`` never runs and the servos keep their last velocity.
    """
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    return stop


def connect_bus(
    host: str,
    port: int,
    motors: dict[str, int],
    stop: threading.Event,
    retry_s: float = 2.0,
) -> FeetechTcpClient | None:
    """Open the servo bus and check both wheels answer; keep trying until they do.

    At boot ser2net, the USB adapter or the servo power may come up after us,
    and a crash loop (systemd restarting a process that exits at once) is not a
    state anyone can read. Returns None only when ``stop`` is set meanwhile.
    """
    attempt = 0
    while not stop.is_set():
        bus = FeetechTcpClient(host, port, motors, retries=1)
        try:
            bus.connect()
            verify_motors(bus, [LEFT, RIGHT])
            return bus
        except (OSError, RuntimeError, ValueError) as exc:
            bus.close()
            attempt += 1
            if attempt in (1, 5) or attempt % 30 == 0:  # first, then rarely: not a log flood
                logger.warning("servo bus not ready (%s); retrying every %.0f s", exc, retry_s)
            stop.wait(retry_s)
    return None


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
    parser.add_argument(
        "--neck-config",
        default="/opt/pepin/config/neck.json",
        help="the neck (config/neck.json): the ids the neck command reads and the limits the"
        " move commands obey; absent, those commands answer an error",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s"
    )

    config = BaseConfig.from_json(args.config)
    motors = DiffDriveBase.motor_ids(config)
    neck_ids = load_neck_ids(args.neck_config)
    motors.update(neck_ids)  # ids 9 and 10 by their names, before the roster fills the rest
    first, last = (int(x) for x in args.servos.split("-"))
    for motor_id in range(first, last + 1):
        if motor_id not in motors.values():
            motors[f"servo{motor_id}"] = motor_id
    stop = stop_on_sigterm()
    bus = connect_bus(args.bus_host, args.bus_port, motors, stop)
    if bus is None:
        return  # stopped before the bus ever answered
    neck_cfg = load_neck_config(args.neck_config) if neck_ids else None
    with bus:
        neck = NeckReader(bus, (PAN, TILT)) if neck_ids else None
        mover = NeckMover(bus, neck, neck_cfg) if neck and neck_cfg else None
        core = BaseServerCore(
            bus, config, servo_names=list(motors), latency=bus.latency, neck=neck, mover=mover
        )
        server = JsonLinesServer(
            args.port, on_last_client_left={"cmd": "release"}, driving_commands=DRIVING_COMMANDS
        ).start()
        serve(core, server, args.tick_hz, args.publish_hz, stop)


def load_neck_ids(path: str) -> dict[str, int]:
    """The neck's ``{name: id}`` from config/neck.json at ``path``, or ``{}`` (logged) when the
    file is missing or malformed: the wheels do not depend on the neck."""
    try:
        return neck_servo_ids(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("no neck servos (%s): the neck command will answer an error", exc)
        return {}


def load_neck_config(path: str) -> NeckConfig | None:
    """The whole config/neck.json at ``path`` — the limits and the reference pose the move
    commands need — or None (logged) when it cannot be read: reading the encoders only needs
    the ids, so a file the geometry cannot use still leaves the ``neck`` command working."""
    try:
        return NeckConfig.from_json(path)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        logger.warning("no neck limits (%s): the move commands will answer an error", exc)
        return None


def _target(message: dict[str, Any], key: str) -> int | None:
    """One encoder target out of a move command: an integer, or None when that servo is not
    addressed; raises for anything that is not a number."""
    value = message.get(key)
    return None if value is None else int(value)


if __name__ == "__main__":
    main()
