"""The SO-101 arm on the board, inside the base server's 50 Hz tick (:mod:`pepin.base_server`).

The laptop plans a gesture and sends the WHOLE trajectory in one ``arm_trajectory`` line
(:mod:`pepin.expression.trajectory`); this driver plays it by itself, so a lost laptop costs
nothing: it never needs a word after the trajectory, which is finite (a lease that lapses at
its end). It refuses, never clamps: a knot outside the limits (config/arm.json's ``limits_deg``
within the servos' own EEPROM range), a step faster than the caps (``max_speed_deg_s``,
``max_acc_deg_s2``, scaled by the trajectory's ``speed``: speeds by it, accelerations by its
square), a trajectory longer than ``max_duration_s``, anything while the wheels are commanded
to move; a twist arriving while the arm moves or holds sends it to rest.

A motion, tick by tick:

1. accepted: the servos are energised at the angles they are ALREADY at, in one block write per
   servo group (Torque_Enable, Acceleration and Goal_Velocity = the caps, Goal_Position = the
   present position, Torque_Limit = ``torque_limit``), so nothing jumps; the arm then joins the
   first knot by a minimum-jerk approach within the caps (none when it is there already);
2. playing: one unacknowledged Goal_Position sync_write a tick (nothing written when the goal
   did not change), linear between the knots; the servos' own profile registers are the caps,
   so physics enforces them too;
3. the end: answered ``done``, held ``hold_s`` (``then: hold``, at most ``hold_max_s``), then
   to ``rest_deg`` by a minimum-jerk move at the caps, and ``release_after_s`` later the torque
   off. ``arm_halt`` stops the arm where its goal is and holds it ``hold_max_s``, then rests;
   ``arm_rest`` goes to rest at once.

The encoders ride in the wheels' read (:class:`pepin.base_server.Riders`): every tick while the
arm is energised, ``idle_read_hz`` while it rests with its torque off. When the group first
answers (or answers again) its Operating_Mode is read (anything but position mode refuses every
motion: velocity mode would turn a joint forever) and its EEPROM range (Min/Max_Position_Limit,
which IS lerobot's calibration on these servos), from which the angles are counted
(:class:`pepin.arm_motion.ServoRange`). Servos that fall silent end the motion (answered
``halted``) and are never written.

Cost on the board, an ESTIMATE until board_cpu.sh and bus_p95_ms measure it: six more servos in
the tick's sync_read, ~8 bytes a reply at 1 Mbaud plus the servos' return delay, about
+0.5-1 ms of bus a tick while the arm rides (every tick when energised, 5 a second at rest);
the interpolation, the conversion and one packet a tick, ~0.2-0.4 ms of Python on an A53, so
+1-3 % of one core while the arm moves and well under 1 % at rest; a few hundred bytes of RAM.
Accepting a trajectory checks every knot once on the tick thread: ~20 us a knot, so ~2-3 ms
for a gesture's ~125 knots (the laptop sends a slowed gesture proportionally fewer), one tick
late by that much at its start.
"""

from __future__ import annotations

import itertools
import logging
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pepin.arm_motion import GRIPPER, ArmMotion, ServoRange
from pepin.bus import MotorBus
from pepin.expression.mix import approach, check_knots
from pepin.expression.trajectory import (
    JOINTS,
    AnswerStatus,
    ArmAnswer,
    ArmPhase,
    ArmStateFields,
    ArmStatus,
    ArmTrajectoryCommand,
    Knot,
    Trajectory,
)
from pepin.neck import acc_units, speed_ticks

if TYPE_CHECKING:  # the base server imports this module; its rider group is only a type here
    from pepin.base_server import Riders

logger = logging.getLogger(__name__)

POSITION_MODE = 0
# Torque_Enable .. Torque_Limit, ten bytes a servo: one packet energises, ramps, aims, paces and
# limits all six.
ARM_BLOCK = (
    "Torque_Enable",
    "Acceleration",
    "Goal_Position",
    "Goal_Time",
    "Goal_Velocity",
    "Torque_Limit",
)
TORQUE_FULL = 1000  # Torque_Limit's 100 %
ARRIVED_DEG = 0.5  # a first knot this close to where the arm is needs no approach
APPROACH_HZ = 50.0  # the knots of a minimum-jerk move: one a tick
# The laptop checks its curve against the exact caps; the line rounds angles to 0.01 deg,
# which at slow speeds adds finite-difference noise: the board forgives this much over the
# caps (the servos' profile registers hold the exact caps whatever the knots say).
WIRE_SLACK = 1.05


ReadGroup = Callable[[str], dict[str, int] | None]  # one register of the group, bounded


@dataclass
class Plan:
    """A motion on the board's clock: knots (monotonic seconds, angles), whether it plays a
    trajectory (``moving``) or goes to rest (``resting``), the trajectory, and what follows."""

    knots: list[tuple[float, tuple[float, ...]]]
    phase: ArmPhase
    trajectory: Trajectory | None = None
    hold_s: float = 0.0

    @property
    def end(self) -> float:
        """When its last knot is reached."""
        return self.knots[-1][0]


class ArmDriver:
    """:class:`pepin.expression.trajectory.ArmDriver` over the base server's bus."""

    def __init__(self, bus: MotorBus, encoders: Riders, motion: ArmMotion, read: ReadGroup) -> None:
        """``encoders``: the arm's rider group (servo names in JOINTS order); ``read`` one
        register of the whole group (:func:`pepin.base_server.read_group`)."""
        self._bus = bus
        self.encoders = encoders
        self.motion = motion
        self._read = read
        self._phase: ArmPhase = "off"
        self._plan: Plan | None = None
        self._goal: tuple[float, ...] | None = None  # the angles commanded now
        self._hold_until = 0.0
        self._release_at: float | None = None
        self._scale = 1.0
        self._ranges: tuple[ServoRange, ...] | None = None
        self._limits: tuple[tuple[float, float], ...] = motion.limits()
        self._mode_ok: bool | None = None
        self._checked_at = float("-inf")
        self._profile: tuple[int, int, int] | None = None  # (speed, acc, torque) energised with
        self._sent: tuple[int, ...] | None = None
        self._answers: list[ArmAnswer] = []
        self.error = ""

    # -- commands --------------------------------------------------------------------------------

    def trajectory(self, trajectory: Trajectory, now: float) -> ArmAnswer | None:
        """Take ``trajectory`` (it starts on this tick); its refusal, or None when accepted."""
        refusal = self._unwritable() or self._invalid(trajectory)
        if refusal is not None:
            return ArmAnswer(trajectory.id, "refused", refusal)
        here = self._here()
        assert here is not None  # writable: the encoders and the range are known
        self._end_trajectory(now, "preempted", f"preempted by {trajectory.id}")
        scale = trajectory.speed_scale
        first = trajectory.knots[0]
        lead = [(0.0, here)]
        if max(abs(a - b) for a, b in zip(here, first.deg, strict=True)) > ARRIVED_DEG:
            lead = approach(here, first.deg, *self._caps(scale), APPROACH_HZ)
        lead_s = lead[-1][0]
        knots = [(now + t, deg) for t, deg in lead[:-1]]
        knots += [(now + lead_s + k.t_s - first.t_s, k.deg) for k in trajectory.knots]
        hold = min(trajectory.hold_s, self.motion.hold_max_s) if trajectory.then == "hold" else 0.0
        self._plan = Plan(knots, "moving", trajectory, hold)
        self._scale = scale
        self._phase, self._release_at, self.error = "moving", None, ""
        if self._goal is None:
            self._goal = here
        return None

    def rest(self, now: float) -> None:
        """To the rest pose at the caps, then torque off (nothing to do at rest already)."""
        if self._phase in ("off", "silent") or self._profile is None:
            return
        self._to_rest(now, "halted", "arm_rest")

    def halt(self, now: float) -> None:
        """Stop where the arm's goal is and hold it ``hold_max_s``; then rest."""
        if self._phase in ("off", "silent") or self._profile is None:
            return
        self._end_trajectory(now, "halted", "arm_halt")
        self._plan = None
        self._phase, self._release_at = "halted", None
        self._hold_until = now + self.motion.hold_max_s

    # -- the tick --------------------------------------------------------------------------------

    def step(self, now: float, wheels_moving: bool) -> None:
        """One tick after the shared read: the servos' check when they (re)appear, the drive
        rule, the motion's progress and its one write."""
        encoders = self.encoders
        if encoders.take_appeared():
            self._mode_ok, self._ranges, self._checked_at = None, None, float("-inf")
            self._profile, self._sent = None, None
        if not encoders.live:
            if self._phase not in ("off", "silent"):
                self._end_trajectory(now, "halted", f"the arm fell silent: {encoders.error}")
                logger.warning("arm: servos silent mid-motion (%s)", encoders.error)
            if encoders.error is not None:
                self._phase = "silent"
            self._plan, self._goal, self._profile, self._sent = None, None, None, None
            return
        if self._phase == "silent":
            self._phase = "off"
        if (self._mode_ok is None or self._ranges is None) and (
            now - self._checked_at >= self.motion.retry_s
        ):
            self._check_servos(now)
        if wheels_moving and self._phase in ("moving", "holding", "halted"):
            self._to_rest(now, "halted", "the wheels moved: to rest")
        self._advance(now)
        encoders.eager = self._phase != "off"
        self._write()

    def status(self, now: float = 0.0) -> ArmStatus:
        """The arm now, for the state line and the replies."""
        plan = self._plan
        trajectory = plan.trajectory if plan is not None else None
        progress = now - plan.knots[0][0] if plan is not None else 0.0
        return ArmStatus(
            now,
            self._angles(self.encoders.reading()),
            self._profile is not None,
            self._phase,
            trajectory.id if trajectory is not None else "",
            max(0.0, progress),
            self.error,
        )

    def state_fields(self, now: float) -> ArmStateFields | None:
        """The state line's ``arm``: this tick's angles and the motion; None when this tick did
        not read the arm (or its range is still unread)."""
        deg = self._angles(self.encoders.ticks)
        if deg is None:
            return None
        status = self.status(now)
        return {
            "deg": [round(d, 2) for d in deg],
            "torque": status.torque,
            "phase": status.phase,
            "id": status.trajectory,
            "progress_s": round(status.progress_s, 3),
            "error": status.error,
        }

    def read_reply(self, now: float) -> dict[str, Any]:
        """The ``arm_read`` answer: the newest angles and ticks and their age (no bus traffic
        of its own), the torque, the phase, the range, the error."""
        reading = self.encoders.reading()
        reply: dict[str, Any] = {"type": "arm_read", "phase": self._phase,
                                 "torque": self._profile is not None}  # fmt: skip
        if reading is not None:
            reply["ticks"] = list(reading)
            reply["age_s"] = round(self.encoders.age_s(now), 3)
        deg = self._angles(reading)
        if deg is not None:
            reply["deg"] = [round(d, 2) for d in deg]
        if self._ranges is not None:
            reply["range_deg"] = [[round(x, 2) for x in r.span_deg()] for r in self._ranges]
        error = self.error or (self.encoders.error or "") or self._why_unwritable()
        if error:
            reply["error"] = error
        return reply

    def take_answers(self) -> list[ArmAnswer]:
        """The answers born since the last call (done, preempted, halted)."""
        answers, self._answers = self._answers, []
        return answers

    def release(self) -> None:
        """Torque off now (the server's shutdown); a bus that will not take it is logged."""
        self._plan, self._goal, self._phase = None, None, "off"
        try:
            self._bus.sync_write("Torque_Enable", dict.fromkeys(self.encoders.names, 0),
                                 normalize=False)  # fmt: skip
        except (TimeoutError, OSError) as exc:
            logger.warning("arm torque off failed at shutdown: %s", exc)
        self._profile, self._sent = None, None

    # -- internals -------------------------------------------------------------------------------

    def _caps(self, scale: float) -> tuple[float, float]:
        """(speed, acceleration) caps at ``scale``."""
        return self.motion.max_speed_deg_s * scale, self.motion.max_acc_deg_s2 * scale * scale

    def _why_unwritable(self) -> str:
        if self._mode_ok is False:
            return self.error or "not in position mode"
        if self._mode_ok is None or self._ranges is None:
            return "the arm's mode and range are not read yet"
        return ""

    def _unwritable(self) -> str | None:
        """Why no motion may start now, None when one may."""
        if not self.encoders.live:
            return f"the arm's servos are silent: {self.encoders.error or 'never answered'}"
        why = self._why_unwritable()
        return why or None

    def _invalid(self, trajectory: Trajectory) -> str | None:
        """Why ``trajectory`` may not be played (shape, length, speed scale, limits, caps)."""
        knots = trajectory.knots
        if not knots:
            return "a trajectory needs at least one knot"
        if not 0.0 < trajectory.speed_scale <= 1.0:
            return f"speed {trajectory.speed_scale} is outside (0, 1]"
        if trajectory.then not in ("rest", "hold"):
            return f"then {trajectory.then!r}: rest or hold"
        if knots[0].t_s < 0.0 or not all(math.isfinite(k.t_s) for k in knots):
            return "knot times start at 0 or later"
        if knots[-1].t_s - knots[0].t_s > self.motion.max_duration_s:
            length = knots[-1].t_s - knots[0].t_s
            return f"{length:.1f} s is longer than max_duration_s {self.motion.max_duration_s:.0f}"
        speed, acc = self._caps(trajectory.speed_scale)
        return check_knots([(k.t_s, k.deg) for k in knots], self._limits, speed, acc, JOINTS,
                           slack=WIRE_SLACK)  # fmt: skip

    def _here(self) -> tuple[float, ...] | None:
        """Where the arm is told to be (its goal while energised), else where it is."""
        if self._profile is not None and self._goal is not None:
            return self._goal
        return self._angles(self.encoders.reading())

    def _angles(self, ticks: Sequence[int] | None) -> tuple[float, ...] | None:
        if ticks is None or self._ranges is None:
            return None
        return tuple(r.degrees(t) for r, t in zip(self._ranges, ticks, strict=True))

    def _end_trajectory(self, now: float, status: AnswerStatus, reason: str) -> None:
        """The trajectory playing (if any) answered ``status``."""
        plan = self._plan
        if plan is not None and plan.trajectory is not None and plan.phase == "moving":
            self._answers.append(ArmAnswer(plan.trajectory.id, status, reason))
            plan.trajectory = None

    def _to_rest(self, now: float, status: AnswerStatus, reason: str) -> None:
        """Answer the trajectory playing, then a minimum-jerk move to rest at the caps."""
        self._end_trajectory(now, status, reason)
        here = self._here()
        if here is None:
            self._plan, self._phase = None, "off"
            return
        knots = approach(here, self.motion.rest(), *self._caps(self._scale), APPROACH_HZ)
        self._plan = Plan([(now + t, deg) for t, deg in knots], "resting")
        self._phase, self._release_at = "resting", None

    def _advance(self, now: float) -> None:
        """The goal for this tick from the plan, and the phase it ends in."""
        plan = self._plan
        if self._phase in ("holding", "halted"):
            if now >= self._hold_until:
                self._to_rest(now, "done", "")
                plan = self._plan
            else:
                return
        if self._phase == "resting" and plan is None:
            if self._release_at is not None and now >= self._release_at:
                self._phase, self._goal, self._release_at = "off", None, None
            return
        if plan is None:
            return
        if now < plan.end:
            self._goal = interpolate(plan.knots, now)
            return
        self._goal = plan.knots[-1][1]
        if plan.phase == "moving":
            if plan.trajectory is not None:
                self._answers.append(ArmAnswer(plan.trajectory.id, "done"))
            if plan.hold_s > 0.0:
                self._plan, self._phase = None, "holding"
                self._hold_until = now + plan.hold_s
            else:
                self._plan = None
                self._phase = "holding"
                self._hold_until = now
                self._advance(now)
            return
        self._plan, self._release_at = None, now + self.motion.release_after_s

    def _write(self) -> None:
        """At most one packet: the energising block when the profile changes, else the goal
        when it changed, or the torque off at the end."""
        names = self.encoders.names
        energised = self._phase in ("moving", "holding", "halted", "resting")
        try:
            if not energised or self._goal is None or self._ranges is None or not self._mode_ok:
                if self._profile is not None:
                    self._bus.sync_write("Torque_Enable", dict.fromkeys(names, 0), normalize=False)
                    logger.info("arm: at rest, torque off")
                self._profile, self._sent = None, None
                return
            goal = tuple(r.ticks(d) for r, d in zip(self._ranges, self._goal, strict=True))
            speed, acc = self._caps(self._scale)
            profile = (
                speed_ticks(speed),
                acc_units(acc),
                round(self.motion.torque_limit * TORQUE_FULL),
            )
            if profile != self._profile:
                self._bus.sync_write_block(
                    ARM_BLOCK,
                    {
                        n: [1, profile[1], g, 0, profile[0], profile[2]]
                        for n, g in zip(names, goal, strict=True)
                    },
                )
            elif goal != self._sent:
                self._bus.sync_write(
                    "Goal_Position", dict(zip(names, goal, strict=True)), normalize=False
                )
            self._profile, self._sent = profile, goal
        except (TimeoutError, OSError) as exc:
            logger.warning("arm write failed (%s): rewritten whole on the next tick", exc)
            self._profile, self._sent = None, None

    def _check_servos(self, now: float) -> None:
        """Operating_Mode and the EEPROM range of all six, once after they (re)appear; bounded,
        retried ``retry_s`` later when one does not answer."""
        self._checked_at = now
        modes = self._read("Operating_Mode")
        lows = self._read("Min_Position_Limit") if modes is not None else None
        highs = self._read("Max_Position_Limit") if lows is not None else None
        if modes is None or lows is None or highs is None:
            return
        names = self.encoders.names
        wrong = {n: m for n, m in modes.items() if m != POSITION_MODE}
        if wrong:
            self._mode_ok = False
            self.error = f"not in position mode, Operating_Mode {wrong}: refusing every motion"
            logger.error("arm %s", self.error)
            return
        ranges = tuple(
            ServoRange(lows[n], highs[n], gripper=j == GRIPPER)
            for n, j in zip(names, JOINTS, strict=True)
        )
        limits = tuple(
            (max(lo, r.span_deg()[0]), min(hi, r.span_deg()[1]))
            for (lo, hi), r in zip(self.motion.limits(), ranges, strict=True)
        )
        outside = [j for j, (lo, hi), d in zip(JOINTS, limits, self.motion.rest(), strict=True)
                   if not lo <= d <= hi]  # fmt: skip
        if outside:
            self._mode_ok = False
            self.error = (
                f"rest_deg of {outside} is outside the servos' range: refusing every motion"
            )
            logger.error("arm %s", self.error)
            return
        self._ranges, self._limits, self._mode_ok, self.error = ranges, limits, True, ""
        logger.info("arm servos in position mode; range %s deg",
                    [tuple(round(x, 1) for x in r.span_deg()) for r in ranges])  # fmt: skip


def interpolate(knots: Sequence[tuple[float, tuple[float, ...]]], t: float) -> tuple[float, ...]:
    """The angles at ``t``, linear between the knots around it (held outside them)."""
    if t <= knots[0][0]:
        return knots[0][1]
    for (t0, a), (t1, b) in itertools.pairwise(knots):
        if t <= t1:
            share = (t - t0) / (t1 - t0) if t1 > t0 else 1.0
            return tuple(x + share * (y - x) for x, y in zip(a, b, strict=True))
    return knots[-1][1]


def trajectory_from(message: dict[str, Any]) -> Trajectory:
    """An ``arm_trajectory`` line as a :class:`Trajectory`; ``ValueError`` naming what is
    malformed (types and shapes only: the driver judges the motion)."""
    raw = message.get("knots")
    if not isinstance(raw, list) or not raw:
        raise ValueError("knots: a list of [t, six angles]")
    knots = []
    for i, row in enumerate(raw):
        if (
            not isinstance(row, list)
            or len(row) != 1 + len(JOINTS)
            or not all(isinstance(x, int | float) and not isinstance(x, bool) for x in row)
        ):
            raise ValueError(f"knots[{i}]: [t, {len(JOINTS)} angles], numbers")
        knots.append(Knot(float(row[0]), tuple(float(x) for x in row[1:])))
    then = message.get("then", "rest")
    if then not in ("rest", "hold"):
        raise ValueError("then: rest or hold")
    speed = message.get("speed", 1.0)
    hold = message.get("hold_s", 0.0)
    if isinstance(speed, bool) or not isinstance(speed, int | float):
        raise ValueError("speed: a number")
    if isinstance(hold, bool) or not isinstance(hold, int | float):
        raise ValueError("hold_s: a number")
    return Trajectory(
        str(message.get("id", "")),
        str(message.get("source", "")),
        tuple(knots),
        then,
        float(hold),
        str(message.get("name", "")),
        float(speed),
    )


def answer_line(answer: ArmAnswer) -> dict[str, Any]:
    """An answer as the ``{"type": "arm", ...}`` line every client reads."""
    return {"type": "arm", "id": answer.id, "status": answer.status, "reason": answer.reason}


def trajectory_line(trajectory: Trajectory) -> ArmTrajectoryCommand:
    """A :class:`Trajectory` as its ``arm_trajectory`` line (the laptop's end of
    :func:`trajectory_from`); angles to 0.01 deg, times to 1 ms."""
    return {
        "cmd": "arm_trajectory",
        "id": trajectory.id,
        "source": trajectory.source,
        "name": trajectory.name,
        "knots": [[round(k.t_s, 3), *(round(d, 2) for d in k.deg)] for k in trajectory.knots],
        "then": trajectory.then,
        "hold_s": trajectory.hold_s,
        "speed": trajectory.speed_scale,
    }
