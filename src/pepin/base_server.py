"""Base server: runs on the board and owns the wheels in real time.

It talks to the servo bus over the board's own loopback (ser2net on
127.0.0.1:3333, a stable sub-millisecond hop), ticks at 50 Hz — read the
encoders, integrate odometry, apply the latest twist — and publishes its
state to every connected client as JSON lines (:mod:`pepin.base_link`), once a
tick: the board's base bridge turns each line into one /odom, so this is the
odometry's rate. A state line costs no bus read of its own (the tick read the
encoders; the servos' temperature rides along at most every 5 s).

Safety lives here, not on the laptop: a deadman stops the wheels when no
twist has arrived for half a second (wifi froze, the script crashed, the
laptop went to sleep); the wheels are armed (torque on) only while someone
is driving and released ten seconds after the last motion, so the cart can
always be pushed by hand when idle. The encoders have the last word on that
release: ten seconds in which the wheels did not turn frees them whatever is
being commanded (``disarm_without_travel`` in config/base.json), because a
controller left commanding a twist the cart cannot follow kept the servos
locked all afternoon on 2026-09-14. Nothing that talks to a laptop runs on
the tick thread: each client has its own reader and writer threads, and a
laptop that stops reading is dropped, not waited for.

The neck's two servos share the bus, and they ride the wheels' tick: their encoders are
read in the SAME sync_read as the wheels (:class:`NeckEncoders`), after them and waited for
only a few milliseconds past them, so a dead head costs the odometry that window and never a
retry; their ticks go out in every state line under the odometry's own stamp, which is what
the board's base bridge turns into /neck/state and base_link -> camera_link. Every neck write
is an unacknowledged sync_write, at most one a tick (:class:`NeckMover`): nothing a servo can
fail to answer ever runs on this thread while the wheels turn, so the head moves while
driving. ``neck_target`` is the gaze arbiter's stream of goals, each renewing a lease; when
the lease lapses the head goes home and is let go. ``neck_goto`` and ``neck_home`` are one-shot
moves answered on arrival; ``neck_jog`` walks the head at a rate for the game-mode teleop
(:mod:`pepin.teleop`) behind a half-second deadman of its own. The speed, ramp and lease are
config/neck.json's ``motion`` block, live through ``neck_motion``.

Run on the board::

    python -m pepin.base_server --config /opt/pepin/config/base.json

The pure logic is :class:`BaseServerCore` (unit-tested against a fake bus);
:func:`serve` adds the clock, :class:`pepin.streams.JsonLinesServer` the sockets.
"""

from __future__ import annotations

import argparse
import logging
import math
import signal
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Any

from pepin.base import LEFT, RIGHT, BusWatchdog, DiffDriveBase, with_suppressed_timeout
from pepin.base_link import BASE_PORT, DEADMAN_S, STATE_HZ
from pepin.bus import MotorBus, verify_motors
from pepin.feetech import FeetechTcpClient
from pepin.geometry import BaseConfig
from pepin.kinematics import STOP, Twist
from pepin.neck import (
    MOTION_RANGES,
    PAN,
    RAD_PER_TICK,
    TILT,
    NeckAngles,
    NeckConfig,
    NeckMotion,
    acc_units,
    neck_servo_ids,
    speed_ticks,
    ticks_for,
)
from pepin.odometry import DiffDriveOdometry
from pepin.speed import check_speed
from pepin.streams import JsonLinesServer
from pepin.telemetry import LatencyTracker

logger = logging.getLogger(__name__)

# The commands that make a client a driver of the wheels; the wheels are released when the last
# such client leaves, whoever else is still connected and merely asking (pepin.streams).
DRIVING_COMMANDS = frozenset({"twist", "stop"})

POSITION_MODE = 0  # Operating_Mode of a servo that obeys Goal_Position; 1 would spin forever
NECK_TOLERANCE_TICKS = 4  # ~0.35 deg: arrived, as far as a 12-bit encoder is concerned
NECK_MOVE_TIMEOUT_S = 3.0  # the least a move is given before it is answered as not reached
NECK_JOG_FAST_DEG_S = 52.0  # a held key: the tilt range in under two seconds (40 -> 52, 2026-10-01)
NECK_JOG_SLOW_DEG_S = 8.0  # a held key with Shift: aiming
NECK_JOG_DEADMAN_S = DEADMAN_S  # no jog message for this long: the head stops, torque off
NECK_JOG_LAG_TICKS = 120  # ~10 deg: a goal this far ahead of the head waits for it
NECK_TICK_S = 0.02  # the board's 50 Hz tick: a stalled tick advances a jog by at most two of these
MODE_CHECK_WINDOW_S = 0.03  # the operating-mode read's whole wait: one reply burst, never 0.4 s
# Torque_Enable .. Goal_Velocity, eight bytes a servo: ONE packet energises, ramps, aims and paces
# both servos, so a head that starts moving costs one write, never a sequence of them.
NECK_BLOCK = ("Torque_Enable", "Acceleration", "Goal_Position", "Goal_Time", "Goal_Velocity")
# A client's outbox in state lines: ~1.3 s of the 50 Hz stream before a peer that stopped reading is
# dropped (pepin.streams' default of 24 was "about a second" at 20 Hz).
STATE_OUTBOX_LINES = 64
P95_EVERY_S = 1.0  # bus_p95_ms is re-sorted out of its 512-sample window at most this often


def jog_ticks_s(deg_s: float) -> float:
    """A jog rate in degrees per second as encoder ticks per second."""
    return math.radians(deg_s) / RAD_PER_TICK


class NeckEncoders:
    """The neck's encoders as they ride in the wheels' read of every tick.

    Both servos are asked in the same sync_read as the wheels, after them, and waited for only
    ``window_s`` past the wheels' replies (pepin.feetech's optional ids): a dead head costs the
    odometry that window and never a retry. ``silent_ticks`` reads in a row without both answers
    take the pair out of the read; it rides along again every ``retry_s`` until it answers.
    :meth:`riders` says whom to add to this tick's read, :meth:`heard` takes the answer, and
    :attr:`ticks` is this tick's reading — the one the state line carries, under the stamp of
    the same read as the odometry.
    """

    def __init__(
        self,
        names: tuple[str, str] = (PAN, TILT),
        *,
        window_s: float = 0.003,
        silent_ticks: int = 3,
        retry_s: float = 5.0,
    ) -> None:
        """``names`` are the (pan, tilt) motor names on the bus."""
        self.names = names
        self.window_s = window_s
        self._silent_ticks = silent_ticks
        self._retry_s = retry_s
        self._asked = True  # in every tick's read (until silent_ticks misses)
        self._missed = 0
        self._dropped_at = float("-inf")
        self._appeared = False
        self.ticks: tuple[int, int] | None = None  # this tick's (pan, tilt), None: not heard
        self._last: tuple[int, int] | None = None
        self._last_at = 0.0
        self._read_ms = 0.0
        self.error: str | None = None

    @property
    def live(self) -> bool:
        """The pair has answered and is in the read: a write reaches servos that listen."""
        return self._asked and self._last is not None

    def riders(self, now: float) -> list[str]:
        """The neck's names to add to this tick's read: always while the pair answers, once a
        ``retry_s`` after it fell silent, otherwise none."""
        if self._asked or now - self._dropped_at >= self._retry_s:
            return list(self.names)
        return []

    def heard(self, now: float, raw: Mapping[str, int] | None, asked: bool, read_ms: float) -> None:
        """This tick's read: ``raw`` its answer, None when the wheels' read itself failed (then
        nothing is learned about the neck); ``asked`` whether the neck rode in it."""
        self.ticks = None
        if raw is None or not asked:
            return
        self._read_ms = read_ms
        pan, tilt = raw.get(self.names[0]), raw.get(self.names[1])
        if pan is not None and tilt is not None:
            if not self.live:
                self._appeared = True
                if self._last is not None:
                    logger.info("neck servos answer again: back in the tick's read")
            self.ticks = (int(pan), int(tilt))
            self._last, self._last_at = self.ticks, now
            self._asked, self._missed, self.error = True, 0, None
            return
        silent = [name for name in self.names if raw.get(name) is None]
        self._missed += 1
        if not self._asked:
            self._dropped_at = now  # a retry that found them silent still: wait again
        elif self._missed >= self._silent_ticks:
            self._asked, self._dropped_at = False, now
            self.error = f"no reply from {silent} in {self._missed} reads"
            logger.warning(
                "neck servos %s silent for %d reads: out of the tick's read, asked again every"
                " %.0f s",
                silent,
                self._missed,
                self._retry_s,
            )

    def take_appeared(self) -> bool:
        """Whether the pair answered for the first time, or again after falling silent, since
        the last call: their operating mode must be checked, their registers are unknown."""
        appeared, self._appeared = self._appeared, False
        return appeared

    def reading(self) -> tuple[int, int] | None:
        """The newest (pan, tilt) while the pair is live: at most ``silent_ticks`` reads old."""
        return self._last if self.live else None

    def state_fields(self) -> dict[str, int]:
        """The state line's neck fields: this tick's ticks, or nothing when it did not hear them."""
        if self.ticks is None:
            return {}
        return {"pan_ticks": self.ticks[0], "tilt_ticks": self.ticks[1]}

    def reply(self, now: float) -> dict[str, Any]:
        """The ``neck`` answer: the newest reading and its age (no bus traffic of its own), and
        the error while the pair is silent — its last ticks, stale, riding along."""
        reply: dict[str, Any] = {"type": "neck"}
        if self._last is not None:
            reply["pan_ticks"], reply["tilt_ticks"] = self._last
            reply["age_s"] = now - self._last_at
            reply["read_ms"] = self._read_ms
        if not self.live:
            reply["error"] = self.error or "the neck servos have not answered yet"
        return reply


def neck_error(text: str) -> dict[str, Any]:
    """A refused neck move, in the shape of the answer a finished one would have had."""
    return {"type": "neck_goto", "reached": False, "error": text}


def jog_error(text: str) -> dict[str, Any]:
    """A refused ``neck_jog``; an accepted one answers nothing, like a twist."""
    return {"type": "neck_jog", "error": text}


def target_error(text: str) -> dict[str, Any]:
    """A refused ``neck_target``; an accepted one answers nothing, like a twist."""
    return {"type": "neck_target", "error": text}


@dataclass(frozen=True)
class ServoGoal:
    """What the two neck servos should hold: energised at these ticks with this profile
    (Goal_Velocity in ticks/s, Acceleration in register units), or let go (``torque`` False)."""

    torque: bool
    pan: int = 0
    tilt: int = 0
    speed: int = 0
    acc: int = 0


RELEASED = ServoGoal(torque=False)


@dataclass
class NeckMove:
    """A move to fixed ticks (``neck_goto``, ``neck_home``, or the lease's own way home): whether
    to keep the torque on arrival, when it started and gives up (monotonic seconds), and whether
    anyone is waiting for its answer."""

    targets: tuple[int, int]
    hold: bool
    started: float
    deadline: float
    answer: bool = True


@dataclass
class NeckLease:
    """The head held at a ``neck_target`` until ``until`` (monotonic seconds), at that target's
    own speed and ramp ceilings (None: the motion settings' maxima)."""

    targets: tuple[int, int]
    speed_deg_s: float | None
    acc_deg_s2: float | None
    until: float


@dataclass
class NeckJog:
    """One jog under way: per servo the direction in ticks (-1, 0, +1) and the goal it is walked
    towards (float ticks, so a slow rate adds up between writes), (pan, tilt); whether it is
    slow; when the last message came (the deadman) and when the goals were last advanced."""

    direction: tuple[int, int]
    goal: list[float]
    slow: bool
    heard: float
    stepped: float


class NeckMover:
    """The neck's write side: one goal for both servos, written from the tick thread, at most
    once a tick, as an UNACKNOWLEDGED sync_write — so no neck command can hold the wheels up.

    Commands only say what the servos should hold; :meth:`step`, once per tick after the shared
    encoder read, works out what that is and writes the difference from what was sent last: a
    Goal_Position packet while the head is aimed elsewhere at the same pace, the whole
    Torque_Enable..Goal_Velocity block when it is energised or its pace changes, a Torque_Enable
    0 packet when it is let go. Nothing is written to a servo whose operating mode has not been
    read as position mode since it last answered (velocity mode would turn the head forever),
    and nothing while the pair is silent; a pair that comes back is rewritten whole.

    Who holds the head, one at a time:

    * a lease (``neck_target``): the gaze arbiter's stream of goals, each renewing the lease for
      ``lease_s``; when it lapses the head goes home at full pace and is let go on arrival;
    * a move (``neck_goto``, ``neck_home``): one goal, answered when the head arrives or gives up,
      then let go unless it asked to ``hold``;
    * a jog (``neck_jog``): the game-mode teleop's held keys walk the goal at a rate within the
      limits, for as long as its messages keep coming (the jog's own half-second deadman), then
      the head stops where it is and is let go.

    A jog is the operator's and takes the head from a lease; a lease takes it from a move (which
    answers "preempted"); a move is refused while a lease or a jog holds the head; a move and a
    jog refuse each other. None of them looks at the wheels.
    """

    def __init__(
        self,
        bus: MotorBus,
        encoders: NeckEncoders,
        config: NeckConfig,
        *,
        tolerance_ticks: int = NECK_TOLERANCE_TICKS,
        jog_fast_deg_s: float = NECK_JOG_FAST_DEG_S,
        jog_slow_deg_s: float = NECK_JOG_SLOW_DEG_S,
        jog_deadman_s: float = NECK_JOG_DEADMAN_S,
        jog_lag_ticks: int = NECK_JOG_LAG_TICKS,
        tick_s: float = NECK_TICK_S,
    ) -> None:
        """``encoders`` are the shared read's neck; ``config`` the limits, the reference pose and
        the motion settings of config/neck.json (the last live through :meth:`motion_command`);
        the ``jog_*`` numbers are the two jog rates, the jog's deadman and its lag allowance;
        ``tick_s`` the tick a jog's stride is bounded by."""
        self._bus = bus
        self._encoders = encoders
        self._cfg = config
        self.motion: NeckMotion = config.motion
        self._tolerance = tolerance_ticks
        self._jog_rates = {False: jog_ticks_s(jog_fast_deg_s), True: jog_ticks_s(jog_slow_deg_s)}
        self._jog_deadman_s = jog_deadman_s
        self._jog_lag = jog_lag_ticks
        self._tick_s = tick_s
        self._owner: NeckMove | NeckLease | NeckJog | None = None
        self._held: ServoGoal | None = None  # a move that asked to hold, arrived
        self._sent: ServoGoal | None = None  # None: unknown (never written, or the pair came back)
        self._mode_ok: bool | None = None  # None: not read since the pair last answered
        self._mode_error = ""
        self._mode_asked_at = float("-inf")
        self._replies: list[dict[str, Any]] = []

    # -- commands ------------------------------------------------------------------------------

    def target(
        self,
        pan_rad: float,
        tilt_rad: float,
        *,
        speed_deg_s: float | None,
        acc_deg_s2: float | None,
        now: float,
    ) -> dict[str, Any] | None:
        """One ``neck_target``: hold the head at these joint angles (pan positive left, tilt the
        pitch below level) for the next ``lease_s``, at most at these speed and ramp ceilings.

        A refusal answers an error reply: a jog holds the head, the servos cannot be written,
        the reference ticks are unread (no angle has a tick then), a target outside the limits
        (never clamped: the arbiter knows the reach, pepin.neck.angle_limits). Accepted: None,
        and the goal is written on this tick; a move under way answers "preempted"."""
        if isinstance(self._owner, NeckJog):
            return target_error("an operator jog holds the head")
        refusal = self._unwritable()
        if refusal is not None:
            return target_error(refusal)
        try:
            targets = ticks_for(self._cfg, NeckAngles(pan_rad, tilt_rad))
        except ValueError as exc:
            return target_error(str(exc))
        outside = self._outside(targets)
        if outside is not None:
            pan_deg, tilt_deg = math.degrees(pan_rad), math.degrees(tilt_rad)
            return target_error(f"pan {pan_deg:+.1f} deg, tilt {tilt_deg:+.1f} deg: {outside}")
        self._preempt("preempted by neck_target", now)
        self._held = None
        self._owner = NeckLease(targets, speed_deg_s, acc_deg_s2, now + self.motion.lease_s)
        return None

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
        """Begin a move to those encoder ticks; None leaves that axis where it is (held there).

        Returns an error reply when the move is refused — a target outside the configured
        limits (never clamped: a wrong number is a mistake, and a silently smaller move hides
        it), no target at all, a lease, a move or a jog holding the head, servos that cannot be
        written — and None once the move is under way (its goal goes out on this tick)."""
        if pan_ticks is None and tilt_ticks is None:
            return neck_error("neck_goto needs pan_ticks or tilt_ticks")
        owner = self._owner
        if isinstance(owner, NeckJog):
            return neck_error("a neck jog is under way")
        if isinstance(owner, NeckMove) and owner.answer:
            return neck_error("a neck move is already under way")
        if isinstance(owner, NeckLease):
            left = owner.until - now
            return neck_error(f"the head is leased to neck_target for another {left:.1f} s")
        refusal = self._unwritable()
        if refusal is not None:
            return neck_error(refusal)
        for joint, target in ((self._cfg.pan, pan_ticks), (self._cfg.tilt, tilt_ticks)):
            if target is not None and not joint.within_limits(target):
                return neck_error(
                    f"{joint.name} target {target} is outside its limits "
                    f"{joint.min_ticks}..{joint.max_ticks} ticks"
                )
        here = self._encoders.reading()
        if (pan_ticks is None or tilt_ticks is None) and here is None:
            return neck_error("the encoders are unread: no pose to hold the other axis at")
        targets = (
            pan_ticks if pan_ticks is not None else here[0],  # type: ignore[index]
            tilt_ticks if tilt_ticks is not None else here[1],  # type: ignore[index]
        )
        self._held = None
        self._owner = NeckMove(targets, hold, now, now + self._move_time(targets, here))
        return None

    def jog(self, pan: int, tilt: int, *, slow: bool, now: float) -> dict[str, Any] | None:
        """One ``neck_jog`` message: walk the head in these directions (pan +1 left, tilt +1
        down — the signs of :class:`pepin.neck.NeckAngles`; 0 holds that axis) until the next
        message or the jog's deadman.

        The first message with a direction starts the jog, its goals seeded with the encoders,
        taking the head from a lease (the operator outranks the arbiter). Later messages only
        steer it; both directions zero stop the head where it is, still held, and the deadman
        lets go half a second later. Both zero with no jog under way is nothing to do. Returns
        an error reply when refused — a move under way, encoders unread, servos that cannot be
        written — and None otherwise.
        """
        owner = self._owner
        if isinstance(owner, NeckMove) and owner.answer:
            return jog_error("a neck move is under way")
        ref = self._cfg.reference
        direction = (pan * ref.pan_sign, tilt * ref.tilt_sign)
        if isinstance(owner, NeckJog):
            owner.direction, owner.heard, owner.slow = direction, now, slow
            return None
        if not any(direction):
            return None
        refusal = self._unwritable()
        if refusal is not None:
            return jog_error(refusal)
        here = self._encoders.reading()
        if here is None:
            return jog_error(f"the encoders are unread: {self._encoders.error or 'never read'}")
        self._held = None
        self._owner = NeckJog(direction, [float(here[0]), float(here[1])], slow, now, now)
        return None

    @property
    def jogging(self) -> bool:
        """A jog is under way (the servos are energised and the goals follow the messages)."""
        return isinstance(self._owner, NeckJog)

    def motion_command(self, message: dict[str, Any]) -> dict[str, Any]:
        """The ``neck_motion`` request: the motion settings in force (and the file's), each key
        of :data:`pepin.neck.MOTION_RANGES` in the message set first, until this server restarts.
        A value that is not a number in its range refuses the whole message, nothing changed."""
        changes: dict[str, float] = {}
        error = None
        for key, (low, high) in MOTION_RANGES.items():
            if key not in message:
                continue
            value = message[key]
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not (low <= float(value) <= high)
            ):
                error = f"{key} {value!r} refused: a number in {low}..{high} is required"
                break
            changes[key] = float(value)
        reply: dict[str, Any] = {"type": "neck_motion"}
        if error is not None:
            reply["error"] = error
        elif changes:
            reply["was"] = {key: getattr(self.motion, key) for key in changes}
            self.motion = replace(
                self.motion,
                max_speed_deg_s=changes.get("max_speed_deg_s", self.motion.max_speed_deg_s),
                max_acc_deg_s2=changes.get("max_acc_deg_s2", self.motion.max_acc_deg_s2),
                lease_s=changes.get("lease_s", self.motion.lease_s),
            )
            logger.info(
                "neck motion %s, live until a restart (config/neck.json: %s)",
                changes,
                {key: getattr(self._cfg.motion, key) for key in changes},
            )
        for key in MOTION_RANGES:
            reply[key] = getattr(self.motion, key)
        reply["config"] = {key: getattr(self._cfg.motion, key) for key in MOTION_RANGES}
        return reply

    # -- the tick ------------------------------------------------------------------------------

    def step(self, now: float) -> None:
        """One tick, after the shared read: the mode check of a pair that just answered, the
        lease's expiry, the jog's stride, the move's arrival, then the one write (if any)."""
        encoders = self._encoders
        if encoders.take_appeared():
            self._mode_ok, self._sent, self._mode_asked_at = None, None, float("-inf")
        if not encoders.live:
            self._end_owner(f"the neck servos fell silent: {encoders.error}", now)
            self._sent = None
            return
        if self._mode_ok is None and now - self._mode_asked_at >= self.motion.retry_s:
            self._check_mode(now)
        owner = self._owner
        if isinstance(owner, NeckLease) and now > owner.until:
            self._lapse(now)
        elif isinstance(owner, NeckJog):
            self._step_jog(owner, now)
        elif isinstance(owner, NeckMove):
            self._step_move(owner, now)
        wanted = self._wanted()
        if wanted.torque and self._mode_ok is not True:
            return  # never a goal to a servo that may be in velocity mode
        self._write(wanted)

    def take_replies(self) -> list[dict[str, Any]]:
        """The answers of moves that ended since the last call (arrived, gave up, preempted)."""
        replies, self._replies = self._replies, []
        return replies

    def release(self) -> None:
        """Torque off now — shutdown, with a held or moving head — and forget whoever held it;
        a bus that will not take it is logged, not raised."""
        self._owner, self._held = None, None
        self._write(RELEASED, force=True)

    # -- internals -----------------------------------------------------------------------------

    def _unwritable(self) -> str | None:
        """Why the servos cannot be given a goal now, or None when they can."""
        if not self._encoders.live:
            return f"the neck servos are silent: {self._encoders.error or 'never answered'}"
        if self._mode_ok is False:
            return self._mode_error
        if self._mode_ok is None:
            return "the neck's operating mode is not read yet"
        return None

    def _outside(self, targets: tuple[int, int]) -> str | None:
        """The refusal naming the axis outside its configured limits, or None."""
        for joint, target in zip((self._cfg.pan, self._cfg.tilt), targets, strict=True):
            if not joint.within_limits(target):
                return (
                    f"{joint.name} target {target} ticks is outside its limits "
                    f"{joint.min_ticks}..{joint.max_ticks} (pepin.neck.angle_limits)"
                )
        return None

    def _move_time(self, targets: tuple[int, int], here: tuple[int, int] | None) -> float:
        """How long a move is given: half again the time its longer axis needs at the motion's
        top speed, plus a second, and never under ``NECK_MOVE_TIMEOUT_S``."""
        if here is None:
            travel = max(j.max_ticks - j.min_ticks for j in (self._cfg.pan, self._cfg.tilt))
        else:
            travel = max(abs(t - h) for t, h in zip(targets, here, strict=True))
        seconds = travel / speed_ticks(self.motion.max_speed_deg_s)
        return max(NECK_MOVE_TIMEOUT_S, 1.5 * seconds + 1.0)

    def _preempt(self, why: str, now: float) -> None:
        """End a move someone is waiting for with ``why``; the lease's own way home ends quietly."""
        owner = self._owner
        if isinstance(owner, NeckMove) and owner.answer:
            self._replies.append(self._answer(owner, False, now, why))
        self._owner = None

    def _end_owner(self, why: str, now: float) -> None:
        """The pair fell silent: whoever held the head is done, a move answered with ``why``."""
        owner = self._owner
        if isinstance(owner, NeckJog):
            logger.warning("neck jog: %s; ended", why)
            self._replies.append(jog_error(f"the encoders failed mid-jog: {why}"))
        self._preempt(why, now)
        self._held = None

    def _lapse(self, now: float) -> None:
        """No ``neck_target`` for ``lease_s``: home at full pace, let go on arrival (an unread
        reference has no home: let go where it is)."""
        ref = self._cfg.reference
        if ref.pan_ticks is None or ref.tilt_ticks is None:
            self._owner = None
            return
        home = (ref.pan_ticks, ref.tilt_ticks)
        here = self._encoders.ticks
        self._owner = NeckMove(home, False, now, now + self._move_time(home, here), answer=False)

    def _step_move(self, move: NeckMove, now: float) -> None:
        """Arrival (both axes within tolerance on this tick's reading) or the deadline ends the
        move: answered if anyone asked, held or let go."""
        here = self._encoders.ticks
        reached = here is not None and all(
            abs(h - t) <= self._tolerance for h, t in zip(here, move.targets, strict=True)
        )
        if not reached and now < move.deadline:
            return
        self._owner = None
        if move.hold:
            self._held = self._goal(move.targets, None, None)
        if move.answer:
            self._replies.append(self._answer(move, reached, now))

    def _answer(
        self, move: NeckMove, reached: bool, now: float, error: str | None = None
    ) -> dict[str, Any]:
        """A move's answer: where the encoders are, whether it got there, how long it took."""
        last = self._encoders.reading()
        reply: dict[str, Any] = {
            "type": "neck_goto",
            "pan_ticks": last[0] if last is not None else None,
            "tilt_ticks": last[1] if last is not None else None,
            "reached": reached,
            "ms": (now - move.started) * 1000.0,
            "hold": move.hold,
        }
        if error is not None:
            reply["error"] = error
        return reply

    def _step_jog(self, jog: NeckJog, now: float) -> None:
        """The jog's deadman, or its goals advanced by one tick: a goal more than the lag
        allowance ahead of the encoder waits for the head instead of winding further ahead."""
        if now - jog.heard > self._jog_deadman_s:
            logger.info("neck jog: no message for %.1f s, stopping, torque off", now - jog.heard)
            self._owner = None
            return
        # A stalled tick is not one long stride: at most two ticks' worth.
        dt = min(now - jog.stepped, 2.0 * self._tick_s)
        jog.stepped = now
        rate = self._jog_rates[jog.slow]
        here = self._encoders.ticks
        for axis, joint in enumerate((self._cfg.pan, self._cfg.tilt)):
            direction = jog.direction[axis]
            if direction == 0:
                continue
            goal = jog.goal[axis]
            if here is not None and (goal - here[axis]) * direction > self._jog_lag:
                continue
            jog.goal[axis] = min(
                max(goal + direction * rate * dt, joint.min_ticks), joint.max_ticks
            )

    def _goal(
        self, targets: tuple[int, int], speed_deg_s: float | None, acc_deg_s2: float | None
    ) -> ServoGoal:
        """Energised at ``targets`` at the asked pace, capped by the motion settings in force."""
        top_speed, top_acc = self.motion.max_speed_deg_s, self.motion.max_acc_deg_s2
        speed = top_speed if speed_deg_s is None else min(speed_deg_s, top_speed)
        acc = top_acc if acc_deg_s2 is None else min(acc_deg_s2, top_acc)
        return ServoGoal(True, targets[0], targets[1], speed_ticks(speed), acc_units(acc))

    def _wanted(self) -> ServoGoal:
        """What the servos should hold this tick, from whoever holds the head."""
        owner = self._owner
        if isinstance(owner, NeckLease):
            return self._goal(owner.targets, owner.speed_deg_s, owner.acc_deg_s2)
        if isinstance(owner, NeckMove):
            return self._goal(owner.targets, None, None)
        if isinstance(owner, NeckJog):
            goal = self._goal((round(owner.goal[0]), round(owner.goal[1])), None, None)
            return replace(goal, speed=round(self._jog_rates[owner.slow]))
        return self._held if self._held is not None else RELEASED

    def _write(self, wanted: ServoGoal, *, force: bool = False) -> None:
        """The one packet that takes the servos from what was sent last to ``wanted``, if any.
        A link that will not take it leaves the servos' state unknown: the next tick rewrites
        it whole."""
        sent = self._sent
        if wanted == sent and not force:
            return
        pan, tilt = self._encoders.names
        try:
            if not wanted.torque:
                self._bus.sync_write("Torque_Enable", {pan: 0, tilt: 0}, normalize=False)
            elif (
                sent is None
                or not sent.torque
                or (sent.speed, sent.acc) != (wanted.speed, wanted.acc)
            ):
                self._bus.sync_write_block(
                    NECK_BLOCK,
                    {
                        pan: [1, wanted.acc, wanted.pan, 0, wanted.speed],
                        tilt: [1, wanted.acc, wanted.tilt, 0, wanted.speed],
                    },
                )
            else:
                goals = {pan: wanted.pan, tilt: wanted.tilt}
                self._bus.sync_write("Goal_Position", goals, normalize=False)
        except (TimeoutError, OSError) as exc:
            logger.warning("neck write failed (%s): rewritten whole on the next tick", exc)
            self._sent = None
            return
        self._sent = wanted

    def _check_mode(self, now: float) -> None:
        """Read both servos' Operating_Mode once, after they (re)appear: a servo left in
        velocity mode (scripts/jog.py wheel writes that to EEPROM) would read the profile speed
        as a command and turn the head forever. Bounded by MODE_CHECK_WINDOW_S and never
        retried within the tick: a pair that does not answer is asked again ``retry_s`` later."""
        self._mode_asked_at = now
        names = list(self._encoders.names)
        try:
            modes = self._bus.sync_read(
                "Operating_Mode",
                [],
                normalize=False,
                optional=names,
                optional_window_s=MODE_CHECK_WINDOW_S,
            )
        except (TimeoutError, OSError) as exc:
            logger.warning("neck operating mode unread: %s", exc)
            return
        if len(modes) < len(names):
            logger.warning("neck operating mode unread: only %s answered", sorted(modes))
            return
        wrong = {name: mode for name, mode in modes.items() if mode != POSITION_MODE}
        self._mode_ok = not wrong
        self._mode_error = (
            f"not in position mode, Operating_Mode {wrong}: refusing to write a goal"
            if wrong
            else ""
        )
        if wrong:
            logger.error("neck %s", self._mode_error)
        else:
            logger.info("neck servos in position mode: goals allowed")


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
        disarm_without_travel: bool = True,
        idle_travel_m: float = 0.01,
        latency: LatencyTracker | None = None,
        neck: NeckEncoders | None = None,
        mover: NeckMover | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        """``servo_names``: the roster :meth:`command` pings; ``latency`` feeds ``bus_p95_ms``;
        ``neck`` rides the tick's encoder read and answers the ``neck`` command, ``mover`` the
        commands that move the head (None for either: those commands answer an error);
        ``clock`` (the tick's own, time.monotonic on the board) stamps each state line with the
        middle of its encoder read, None with the time the line is asked for.

        ``disarm_without_travel`` releases the wheels after ``disarm_after_s`` in which the
        ENCODERS moved less than ``idle_travel_m``, whatever the commands claim; False is the
        behaviour before 2026-09-14, which counted from the last non-zero twist alone.
        """
        self._bus = bus
        self._base = DiffDriveBase(bus, config)
        self._config_wheel_speed = config.max_wheel_speed_m_s
        self._odom = DiffDriveOdometry(config.geometry)
        self._servo_names = servo_names or [LEFT, RIGHT]
        self._deadman_s = deadman_s
        self._disarm_after_s = disarm_after_s
        self._disarm_without_travel = disarm_without_travel
        self._idle_travel_m = idle_travel_m
        self._latency = latency
        self._neck = neck
        self._mover = mover
        self._watchdog = BusWatchdog()
        self.twist = STOP
        self.armed = False
        self.deadman = False
        self.bus_ok = True
        self._last_command_at: float | None = None  # any twist: feeds the deadman
        self._last_motion_at: float | None = None  # a non-zero twist: feeds the idle release
        # ...and the same clock read off the WHEELS: when the encoders last said the cart really
        # travelled. A controller that keeps commanding a twist the cart cannot follow (a stalled
        # Nav2 after a cancelled drive, 2026-09-14) refreshes the clock above for ever and left
        # the servos locked with the cart standing; this one it cannot touch.
        self._last_travel_at: float | None = None
        self._still_travel = [0.0, 0.0]  # signed wheel travel since that moment, per wheel
        self._acc = [0.0, 0.0]  # wheel travel since the last snapshot
        self._primed = False
        self._temperature: dict[str, int] | None = None
        self._temperature_at = float("-inf")  # the first state line reads it
        self._p95_ms = 0.0
        self._p95_at = float("-inf")  # the first state line computes it
        # WHEN THE ENCODERS WERE READ, which is what a state line's "t" says and what the bridge
        # differences two poses over (its measured twist). The tick's start is not that moment: a
        # twist that arrived first is written to the bus before the read, a few milliseconds the
        # tick start does not see. Over the 60 ms between lines at 16.7 Hz that was a few per cent
        # of the measured speed; over 20 ms it is three times that, both ways on alternate lines.
        self._clock = clock
        self._read_at: float | None = None

    @property
    def moving(self) -> bool:
        """A non-zero twist is being applied."""
        return self.twist.linear != 0.0 or self.twist.angular != 0.0

    def command(self, message: dict[str, Any], now: float) -> dict[str, Any] | None:
        """Apply one client message; returns a reply for requests that have one (``ping``,
        ``neck``, ``neck_motion``, ``max_wheel_speed``, a refused neck command).

        An accepted move answers nothing here: its reply is born when the head stops, and
        leaves through :meth:`take_replies`. An accepted jog or target answers nothing at all.
        None of the neck commands looks at the wheels: their writes cannot block this thread.
        """
        cmd = message.get("cmd")
        if cmd == "twist":
            self._last_command_at = now
            self.deadman = False
            twist = Twist(float(message.get("v", 0.0)), float(message.get("w", 0.0)))
            if twist.linear or twist.angular:
                self._last_motion_at = now
                if not self.armed:
                    self._arm(now)
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
            return self._neck.reply(now)
        elif cmd in ("neck_goto", "neck_home"):
            return self._start_neck_move(cmd, message, now)
        elif cmd == "neck_jog":
            return self._jog_neck(message, now)
        elif cmd == "neck_target":
            return self._target_neck(message, now)
        elif cmd == "neck_motion":
            if self._mover is None:
                return {"type": "neck_motion", "error": "no neck configured on this base server"}
            return self._mover.motion_command(message)
        elif cmd == "max_wheel_speed":
            return self._max_wheel_speed(message)
        else:
            logger.warning("unknown command %r", message)
        return None

    def tick(self, now: float) -> None:
        """One control period: the encoders (wheels and neck in one read) -> odometry, the
        deadman and the idle disarm, and then the neck's step and its one write, if any — the
        wheels first, always.

        Raises ``RuntimeError`` when the servos have been silent for the
        watchdog's give-up time: the service exits and systemd restarts it.
        """
        self._tick_wheels(now)
        self._step_neck(now)

    def take_replies(self) -> list[dict[str, Any]]:
        """Answers that outlived the command that asked for them (a finished neck move), and
        empties the list; :func:`serve` broadcasts them — the client that asked is only one of
        the readers, and by then it may be gone."""
        return self._mover.take_replies() if self._mover is not None else []

    def _read_encoders(self, riders: list[str]) -> dict[str, int]:
        """One sync_read of Present_Position: both wheels, and ``riders`` (the neck) after them
        as optional ids that only the window of :class:`NeckEncoders` is spent waiting for."""
        if not riders or self._neck is None:
            return self._bus.sync_read("Present_Position", [LEFT, RIGHT], normalize=False)
        return self._bus.sync_read(
            "Present_Position",
            [LEFT, RIGHT],
            normalize=False,
            optional=riders,
            optional_window_s=self._neck.window_s,
        )

    def _tick_wheels(self, now: float) -> None:
        before = self._clock() if self._clock is not None else now
        riders = self._neck.riders(now) if self._neck is not None else []
        started = time.perf_counter()
        try:
            raw = self._read_encoders(riders)
            travel = self._base.wheel_travel(raw)
        except TimeoutError as exc:
            if self._neck is not None:
                self._neck.heard(now, None, bool(riders), 0.0)
            self._read_at = now
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
        self._read_at = (before + self._clock()) / 2.0 if self._clock is not None else now
        if self._neck is not None:
            read_ms = (time.perf_counter() - started) * 1000.0
            self._neck.heard(now, raw, bool(riders), read_ms)
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
        # Signed, not absolute: encoder noise is zero-mean and would otherwise add up to a
        # centimetre of "travel" a minute, while a cart genuinely creeping at a millimetre a
        # second does pass the threshold within the idle time.
        self._still_travel[0] += travel[0]
        self._still_travel[1] += travel[1]
        if max(abs(self._still_travel[0]), abs(self._still_travel[1])) >= self._idle_travel_m:
            self._still_travel = [0.0, 0.0]
            self._last_travel_at = now
        last_cmd, last_motion = self._last_command_at, self._last_motion_at
        if self.moving and last_cmd is not None and now - last_cmd > self._deadman_s:
            logger.warning("deadman: no command for %.1f s, stopping", now - last_cmd)
            self._apply(STOP)
            self.deadman = True
        idle = self.armed and not self.moving and last_motion is not None
        if idle and last_motion is not None and now - last_motion > self._disarm_after_s:
            self._disarm(f"no motion commanded for {now - last_motion:.0f} s")
        elif (
            self.armed
            and self._disarm_without_travel
            and self._last_travel_at is not None
            and now - self._last_travel_at > self._disarm_after_s
        ):
            self._disarm(f"no travel for {now - self._last_travel_at:.0f} s")

    def snapshot(self, now: float) -> dict[str, Any]:
        """The ``state`` message for the clients; resets the accumulated wheel travel. Its ``t``
        is the middle of the last encoder read when the core has a clock, else ``now``."""
        pose = self._odom.pose
        stamp = self._read_at if self._clock is not None and self._read_at is not None else now
        message = {
            "type": "state",
            "t": stamp,
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
            "bus_p95_ms": self._bus_p95_ms(now),
        }
        # The neck's encoders of the same read, under the same stamp: the base bridge's
        # /neck/state and base_link -> camera_link. Absent when the neck did not answer it.
        if self._neck is not None:
            message.update(self._neck.state_fields())
        self._acc = [0.0, 0.0]
        # The servos' temperature, read at most every 5 s and only while the bus answers: the
        # speed cap was raised to what the servos can do (2026-09-30), the neck holds its torque
        # for as long as a lease lasts, and the servo's own cut-out is 70 C. None until the first
        # read; the neck rides along as optional ids, as in the tick's read.
        if self.bus_ok and now - self._temperature_at >= 5.0:
            self._temperature_at = now
            riders = self._neck.riders(now) if self._neck is not None else []
            try:
                if riders and self._neck is not None:
                    raw = self._bus.sync_read(
                        "Present_Temperature",
                        [LEFT, RIGHT],
                        normalize=False,
                        optional=riders,
                        optional_window_s=self._neck.window_s,
                    )
                else:
                    raw = self._bus.sync_read("Present_Temperature", [LEFT, RIGHT], normalize=False)
                self._temperature = {name: int(v) for name, v in raw.items()}
            except Exception as exc:  # a missed read never touches the wheels
                logger.debug("temperature read failed: %r", exc)
        message["temp_c"] = self._temperature
        return message

    def _bus_p95_ms(self, now: float) -> float:
        """The bus round trip's p95 over the tracker's window, re-sorted at most once a
        ``P95_EVERY_S``: a 512-sample statistic moves over seconds, and sorting it for every
        50 Hz state line would cost the board's Python more than the line itself."""
        if self._latency is not None and now - self._p95_at >= P95_EVERY_S:
            self._p95_at = now
            self._p95_ms = self._latency.summary().p95_ms
        return self._p95_ms

    def release(self) -> None:
        """Stop and free the wheels, and the neck with them (shutdown)."""
        with_suppressed_timeout(lambda: self._apply(STOP))
        if self.armed:
            with_suppressed_timeout(lambda: self._disarm("the last client left"))
        if self._mover is not None:
            # A neck left holding would stay energised with nobody left to release it.
            with_suppressed_timeout(self._mover.release)

    def _max_wheel_speed(self, message: dict[str, Any]) -> dict[str, Any]:
        """The ``max_wheel_speed`` request: the wheel ceiling in force, set first when the message
        carries ``m_s`` (until this server restarts; refused outside pepin.speed's range)."""
        reply: dict[str, Any] = {"type": "max_wheel_speed", "config_m_s": self._config_wheel_speed}
        if "m_s" in message:
            try:
                speed = check_speed(message["m_s"])
            except ValueError as exc:
                return {**reply, "m_s": self._base.max_wheel_speed_m_s, "error": str(exc)}
            reply["was_m_s"] = self._base.max_wheel_speed_m_s
            self._base.max_wheel_speed_m_s = speed
            logger.info(
                "wheel ceiling %.2f -> %.2f m/s, live until a restart (config/base.json: %.2f)",
                reply["was_m_s"],
                speed,
                self._config_wheel_speed,
            )
        reply["m_s"] = self._base.max_wheel_speed_m_s
        return reply

    def _start_neck_move(
        self, cmd: str, message: dict[str, Any], now: float
    ) -> dict[str, Any] | None:
        """Hand one ``neck_goto``/``neck_home`` to the mover: the reply of a refusal, None once
        the move is under way."""
        if self._mover is None:
            return neck_error("no neck configured on this base server")
        hold = bool(message.get("hold", False))
        if cmd == "neck_home":
            return self._mover.home(hold=hold, now=now)
        try:
            pan, tilt = _target(message, "pan_ticks"), _target(message, "tilt_ticks")
        except (TypeError, ValueError) as exc:
            return neck_error(f"bad target: {exc}")
        return self._mover.start(pan, tilt, hold=hold, now=now)

    def _jog_neck(self, message: dict[str, Any], now: float) -> dict[str, Any] | None:
        """Hand one ``neck_jog`` to the mover: the reply of a refusal, None when accepted."""
        if self._mover is None:
            return jog_error("no neck configured on this base server")
        try:
            pan, tilt = _direction(message, "pan"), _direction(message, "tilt")
        except (TypeError, ValueError) as exc:
            return jog_error(f"bad direction: {exc}")
        return self._mover.jog(pan, tilt, slow=bool(message.get("slow", False)), now=now)

    def _target_neck(self, message: dict[str, Any], now: float) -> dict[str, Any] | None:
        """Hand one ``neck_target`` to the mover: the reply of a refusal, None when accepted."""
        if self._mover is None:
            return target_error("no neck configured on this base server")
        try:
            pan, tilt = _angle(message, "pan_rad"), _angle(message, "tilt_rad")
            speed, acc = _ceiling(message, "speed_deg_s"), _ceiling(message, "acc_deg_s2")
        except (TypeError, ValueError) as exc:
            return target_error(f"bad target: {exc}")
        return self._mover.target(pan, tilt, speed_deg_s=speed, acc_deg_s2=acc, now=now)

    def _step_neck(self, now: float) -> None:
        """The neck's tick: a move's arrival, a jog's stride, a lease's end, the one write."""
        if self._mover is not None:
            self._mover.step(now)

    def _apply(self, twist: Twist) -> None:
        self._base.set_twist(twist)
        self.twist = twist

    def _arm(self, now: float) -> None:
        """Torque on, and the travel clock starts here: a freshly armed cart is given the whole
        idle time to move before the encoders are asked whether it did."""
        logger.info("arming: torque on")
        self._base.enable()
        self.armed = True
        self._last_travel_at = now
        self._still_travel = [0.0, 0.0]

    def _disarm(self, reason: str) -> None:
        """Torque off, so the cart can be pushed; ``reason`` is what the log says did it."""
        logger.info("idle: %s, torque off, the cart can be pushed", reason)
        self._base.disable()
        self.armed = False
        self.twist = STOP


class PublishGrid:
    """When a state line is due: on a fixed grid of ``1 / publish_hz`` seconds, asked once a tick.

    A tick counts as on time up to half a tick before its grid point, so a publish rate equal to
    the tick rate publishes on every tick and a lower one averages out to exactly its rate. The
    rule this replaces (the next line one publish period after the last) waited for the first
    tick PAST that point, and a 20 Hz stream on 20 ms ticks came out every third tick: 16.7 Hz.
    A loop that stalls resumes on a fresh grid instead of bursting out the lines it missed.
    """

    def __init__(self, publish_hz: float, tick_hz: float, start: float) -> None:
        """``start`` is the first tick's time, which is due at once."""
        self._every = 1.0 / publish_hz
        self._slack = 0.5 / tick_hz
        self._next = start

    def due(self, now: float) -> bool:
        """Whether the tick at ``now`` publishes; a True moves the grid on by one period."""
        if now + self._slack < self._next:
            return False
        self._next += self._every
        if self._next <= now:  # a stall: a new grid from here
            self._next = now + self._every
        return True


def serve(
    core: BaseServerCore,
    server: JsonLinesServer,
    tick_hz: float,
    publish_hz: float,
    stop: threading.Event | None = None,
) -> None:
    """Clock around the core: apply commands, tick, publish state; release when left alone."""
    period = 1.0 / tick_hz
    grid = PublishGrid(publish_hz, tick_hz, time.monotonic())
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
            if grid.due(started):
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
    parser.add_argument(
        "--publish-hz",
        type=float,
        default=STATE_HZ,
        help="state lines a second, at most the tick rate; each one is an /odom",
    )
    parser.add_argument(
        "--servos", default="1-10", help="bus ids the ping command checks, e.g. 1-10"
    )
    parser.add_argument(
        "--neck-config",
        default="/opt/pepin/config/neck.json",
        help="the neck (config/neck.json): the ids read with the wheels, the limits and the"
        " motion the neck commands obey; absent, those commands answer an error",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s"
    )

    config = BaseConfig.from_json(args.config)
    logger.info(
        "wheel ceiling %.2f m/s (max_wheel_speed_m_s): a twist that would run a wheel faster is"
        " slowed whole, its arc kept; axis caps %.2f m/s, %.2f rad/s",
        config.max_wheel_speed_m_s,
        config.max_speed_m_s,
        config.max_yaw_rate_rad_s,
    )
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
        neck, mover = neck_parts(bus, neck_cfg) if neck_ids else (None, None)
        core = BaseServerCore(
            bus,
            config,
            servo_names=list(motors),
            latency=bus.latency,
            neck=neck,
            mover=mover,
            disarm_after_s=config.disarm_after_s,
            disarm_without_travel=config.disarm_without_travel,
            idle_travel_m=config.idle_travel_m,
            clock=time.monotonic,  # serve()'s clock: "t" stays on the board's monotonic scale
        )
        server = JsonLinesServer(
            args.port,
            on_last_client_left={"cmd": "release"},
            driving_commands=DRIVING_COMMANDS,
            outbox_size=STATE_OUTBOX_LINES,
        ).start()
        serve(core, server, args.tick_hz, args.publish_hz, stop)


def neck_parts(bus: MotorBus, config: NeckConfig | None) -> tuple[NeckEncoders, NeckMover | None]:
    """The neck's read side, and its write side when the whole config/neck.json loaded (reading
    the encoders needs only the ids; moving the head needs the limits and the reference)."""
    motion = config.motion if config is not None else NeckMotion()
    encoders = NeckEncoders(
        (PAN, TILT),
        window_s=motion.read_window_ms / 1000.0,
        silent_ticks=motion.silent_ticks,
        retry_s=motion.retry_s,
    )
    return encoders, NeckMover(bus, encoders, config) if config is not None else None


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


def _angle(message: dict[str, Any], key: str) -> float:
    """One joint angle out of a ``neck_target``, radians: required, a finite number."""
    if key not in message:
        raise ValueError(f"{key} is required")
    value = message[key]
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{key} must be a number of radians, not {value!r}")
    return float(value)


def _ceiling(message: dict[str, Any], key: str) -> float | None:
    """An optional speed or ramp ceiling out of a ``neck_target``: a positive number, or None
    when absent (the motion settings' maximum)."""
    value = message.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float) or not value > 0:
        raise ValueError(f"{key} must be a positive number, not {value!r}")
    return float(value)


def _direction(message: dict[str, Any], key: str) -> int:
    """One jog direction out of a ``neck_jog``: -1, 0 or 1 (absent: 0); raises for anything
    else, a bool included — ``true`` is not a direction."""
    value = message.get(key, 0)
    if isinstance(value, bool) or int(value) not in (-1, 0, 1):
        raise ValueError(f"{key} must be -1, 0 or 1, not {value!r}")
    return int(value)


if __name__ == "__main__":
    main()
