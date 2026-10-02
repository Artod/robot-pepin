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
                     {"cmd": "neck"}                              the neck's encoders (pan, tilt)
                     {"cmd": "neck_target", "pan_rad": <rad>, "tilt_rad": <rad>,
                      "speed_deg_s": <optional>, "acc_deg_s2": <optional>}
                                                                  hold the head there (the joint
                                                                  angles of pepin.neck: pan +
                                                                  left, tilt = pitch below level)
                                                                  for the lease, then it goes home
                     {"cmd": "neck_motion", "max_speed_deg_s": 120, ...}
                                                                  the head's top speed, ramp and
                                                                  lease (each key: set it until a
                                                                  restart); ros/neck.sh motion
                     {"cmd": "neck_jog", "pan": -1|0|1, "tilt": -1|0|1, "slow": false}
                                                                  walk the head (pan +1 left,
                                                                  tilt +1 down) while repeated
                                                                  within its 0.5 s deadman
                     {"cmd": "max_wheel_speed", "m_s": 0.3}        the wheel ceiling (m_s: set it
                                                                  until a restart); ros/speed.sh
                     {"cmd": "registers", "servo": "neck", "address": 85, "size": 1}
                                                                  raw control-table bytes of one
                                                                  roster servo; ros/neck.sh
                                                                  registers
    board -> laptop  {"type": "state", ...}                       see :class:`BaseState`, STATE_HZ;
                                                                  "pan_ticks"/"tilt_ticks" when the
                                                                  neck answered that read
                     {"type": "pong", "servos": {"left": true, "servo3": false, ...}}
                     {"type": "pong", "busy": true}                   moving: servos not pinged
                     {"type": "neck", "pan_ticks": 2048, "tilt_ticks": 2360, "age_s": 0.01,
                      "read_ms": 1.4}                             see :func:`pepin.neck.parse_neck`
                     {"type": "neck", "error": "..."}             a silent servo (stale ticks, if
                                                                  any, ride along)
                     {"type": "neck_target", "error": "..."}      a refused target (an accepted
                                                                  one is silent, like a twist)
                     {"type": "neck_jog", "error": "..."}         a refused jog (likewise)
                     {"type": "neck_motion", "max_speed_deg_s": .., "max_acc_deg_s2": ..,
                      "tilt_max_acc_deg_s2": .., "lease_s": .., "config": {..}, "was": {..}}
                                                                  the settings in force ("error":
                                                                  a refused value, unchanged)
                     {"type": "registers", "servo": "neck", "address": 85, "values": [254]}
                                                                  ("busy": true while the wheels
                                                                  turn; "error": no such servo, a
                                                                  silent one, a read too long)
                     {"type": "max_wheel_speed", "m_s": 0.3, "config_m_s": 0.3, "was_m_s": ..}
                                                                  the ceiling in force ("error":
                                                                  a refused value, unchanged)

The absolute moves (``neck_goto``, ``neck_home``) are board/README.md's; ``ros/neck.sh`` speaks
them. Only a client that has sent ``twist`` or ``stop`` counts as a driver: the wheels are
released when the last driver leaves, whoever is still connected only asking or aiming the head.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, replace
from typing import Any

from pepin.kinematics import Twist
from pepin.neck import NeckReading, parse_neck
from pepin.odometry import Pose2D
from pepin.streams import Connector, JsonLinesClient

BASE_PORT = 3336
DEADMAN_S = 0.5  # the board stops the wheels when no twist arrived for this long
# How often the base server broadcasts a state line: every tick of its 50 Hz loop since 2026-10-01
# (20 before, which its scheduler delivered as 16.7). Each line becomes one /odom on the board.
STATE_HZ = 50.0
NECK_MOVE_WAIT_S = 8.0  # the board gives a move 3-5 s at 120 deg/s; its reply comes a bit later
NECK_ERROR_SHOWN_S = 2.0  # a refused jog is reported for this long after it arrived


def ask(
    host: str,
    message: dict[str, Any],
    reply_type: str,
    wait_s: float,
    port: int = BASE_PORT,
) -> dict[str, Any] | None:
    """One request on a fresh connection, and the first ``{"type": reply_type}`` line back.

    The port broadcasts a state line ``STATE_HZ`` times a second to everyone connected; those
    are skipped. ``None`` when the answer did not come within ``wait_s``; ``OSError`` when nobody
    listens. A connection that only asks is not a driver, so the wheels are left alone.
    """
    with socket.create_connection((host, port), timeout=3.0) as sock:
        sock.sendall((json.dumps(message) + "\n").encode())
        sock.settimeout(1.0)
        deadline, buffer = time.monotonic() + wait_s, b""
        while time.monotonic() < deadline:
            try:
                chunk = sock.recv(4096)
            except TimeoutError:
                continue
            if not chunk:
                return None
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if not line.strip():
                    continue
                try:
                    reply = json.loads(line)
                except ValueError:
                    continue
                if isinstance(reply, dict) and reply.get("type") == reply_type:
                    return reply
    return None


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
    stamp_s: float  # board clock (time.monotonic there): the middle of the encoder read it carries
    age_s: float  # laptop clock: seconds since this message arrived
    neck_ticks: tuple[int, int] | None = None  # (pan, tilt) of the same read; None: not answered


def decode_state(message: dict[str, Any], received_at: float) -> BaseState:
    """A ``state`` message from the board into a :class:`BaseState` (age 0 at ``received_at``)."""
    pan, tilt = message.get("pan_ticks"), message.get("tilt_ticks")
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
        neck_ticks=(int(pan), int(tilt)) if pan is not None and tilt is not None else None,
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
        self._neck: NeckReading | None = None
        self._neck_error: tuple[str, float] | None = None  # text, laptop clock of its arrival

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

    def neck_jog(self, pan: int, tilt: int, *, slow: bool = False) -> None:
        """Walk the head: ``pan`` +1 left / -1 right, ``tilt`` +1 down / -1 up (the signs of
        :class:`pepin.neck.NeckAngles`), 0 holds that axis; repeat it within the board's half
        second or the head stops and lets go. Both zero stops the head where it is."""
        self.send({"cmd": "neck_jog", "pan": pan, "tilt": tilt, "slow": slow})

    def neck_target(
        self,
        pan_rad: float,
        tilt_rad: float,
        *,
        speed_deg_s: float | None = None,
        acc_deg_s2: float | None = None,
    ) -> None:
        """Hold the head at these joint angles (pepin.neck: pan positive left, tilt the pitch
        below level) for the board's lease (config/neck.json ``motion.lease_s``); repeat it
        within the lease or the head goes home and is let go. The speed and ramp are ceilings
        under the board's own; a refusal lands in :meth:`neck_error`."""
        message: dict[str, Any] = {"cmd": "neck_target", "pan_rad": pan_rad, "tilt_rad": tilt_rad}
        if speed_deg_s is not None:
            message["speed_deg_s"] = speed_deg_s
        if acc_deg_s2 is not None:
            message["acc_deg_s2"] = acc_deg_s2
        self.send(message)

    def ask_neck(self) -> None:
        """Ask for the neck's encoders; the answer lands in :meth:`neck` when it comes."""
        self.send({"cmd": "neck"})

    def neck(self) -> NeckReading | None:
        """The newest ``neck`` answer; None before the first."""
        with self._lock:
            return self._neck

    def neck_error(self, now: float | None = None) -> str | None:
        """The text of a jog or a target the board refused within the last
        ``NECK_ERROR_SHOWN_S``, else None."""
        now = time.monotonic() if now is None else now
        with self._lock:
            if self._neck_error is None or now - self._neck_error[1] > NECK_ERROR_SHOWN_S:
                return None
            return self._neck_error[0]

    def ping(self, timeout_s: float = 3.0) -> dict[str, bool] | None:
        """Servos answering on the bus; {} while driving; None when the board did not reply."""
        self._pong_ready.clear()
        self.send({"cmd": "ping"})
        if not self._pong_ready.wait(timeout_s) or self._pong is None:
            return None
        if self._pong.get("busy"):
            return {}  # the wheels are moving; the board does not ping servos then
        return {str(k): bool(v) for k, v in self._pong.get("servos", {}).items()}

    def _ingest(self, message: dict[str, Any]) -> None:
        """Route one decoded message: states replace the newest, pongs wake :meth:`ping`, neck
        readings replace the newest, a refused jog is kept with its arrival time."""
        kind = message.get("type")
        if kind == "state":
            now = time.monotonic()
            state = decode_state(message, now)
            with self._lock:
                self._state, self._received_at = state, now
        elif kind == "pong":
            self._pong = message
            self._pong_ready.set()
        elif kind == "neck":
            reading = parse_neck(message)
            with self._lock:
                self._neck = reading
        elif kind in ("neck_jog", "neck_target") and message.get("error") is not None:
            with self._lock:
                self._neck_error = (str(message["error"]), time.monotonic())
