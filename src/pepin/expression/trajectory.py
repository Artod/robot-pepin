"""The arm's contract: a joint trajectory the laptop sends and the board's base server plays.

The SO-101 hangs on the same Feetech bus as the wheels and the neck (ids 1..6, read since
2026-10-04 through the base server's ``registers``), so the base server is its one owner, as it
is the neck's. The laptop plans (an animation clip on top of the rest pose, time-scaled within
the caps) and sends the WHOLE trajectory in one line; the board plays it on its 50 Hz tick,
interpolating between knots, and needs the laptop for nothing after that: when the knots end it
holds for at most ``hold_s`` and then goes to its rest pose by itself and lets go of the torque.

The wire, JSON lines on the base server's port (:mod:`pepin.base_link`)::

    laptop -> board  {"cmd": "arm_trajectory", "id", "source", "name", "knots": [[t, d1..d6], ...],
                      "then": "rest" | "hold", "hold_s", "speed"}  an ArmTrajectoryCommand
                     {"cmd": "arm_rest"}   {"cmd": "arm_halt"}     an ArmCommand
                     {"cmd": "arm_read"}                           the newest reading, answered
    board -> laptop  {"type": "arm", "id", "status", "reason"}    an ArmReply, to every client
                     "arm": {...} in each state line that read the arm   ArmStateFields

``speed`` (0..1, 1 when absent) scales the board's caps for that trajectory and the rest after
it: speeds by it, accelerations by its square (the first live test runs at 0.2).

Angles are degrees in config/arm.json's convention (lerobot's: the middle of each calibrated
range is 0; the gripper's -10..100 is linear), knot times seconds from the trajectory's start.
The board starts any trajectory from where the arm IS: it energises each servo at its present
position, then joins the first knot within the caps. It refuses (never clamps) a knot outside
config/arm.json's limits or a step faster than its caps, and refuses everything while the wheels
are commanded to move; a twist arriving while the arm moves sends it to rest. Its replies say
``accepted``, ``refused``, ``done``, ``preempted`` (a newer trajectory took the arm) or
``halted``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol, TypedDict

JOINTS: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
Then = Literal["rest", "hold"]
ArmPhase = Literal["off", "moving", "holding", "resting", "halted", "silent"]
AnswerStatus = Literal["accepted", "refused", "done", "preempted", "halted"]


@dataclass(frozen=True)
class Knot:
    """One point of a trajectory: seconds from its start, and one angle per :data:`JOINTS`."""

    t_s: float
    deg: tuple[float, ...]


@dataclass(frozen=True)
class Trajectory:
    """A whole arm motion: its id and who asked (for the replies and the logs), the knots, what
    follows the last one (``rest``; ``hold`` for at most ``hold_s``, then rest), the gesture's
    name and the share of the board's caps it and its rest run at."""

    id: str
    source: str
    knots: tuple[Knot, ...]
    then: Then = "rest"
    hold_s: float = 0.0
    name: str = ""
    speed_scale: float = 1.0


@dataclass(frozen=True)
class ArmStatus:
    """The arm as the newest state line read it: its encoders' stamp (the board's clock), the
    angles (None when that line did not read the arm), the torque, the phase, the trajectory
    playing ("" none) and how far into it, and the latest error ("" none)."""

    stamp: float
    deg: tuple[float, ...] | None
    torque: bool
    phase: ArmPhase
    trajectory: str = ""
    progress_s: float = 0.0
    error: str = ""


@dataclass(frozen=True)
class ArmAnswer:
    """The board's word on one trajectory."""

    id: str
    status: AnswerStatus
    reason: str = ""


class ArmTrajectoryCommand(TypedDict):
    """``arm_trajectory`` on the wire."""

    cmd: Literal["arm_trajectory"]
    id: str
    source: str
    name: str
    knots: list[list[float]]
    then: Then
    hold_s: float
    speed: float


class ArmCommand(TypedDict):
    """``arm_rest`` (to the rest pose, then torque off), ``arm_halt`` (stop where it is,
    holding, then rest after the config's hold limit) or ``arm_read`` (the newest reading)."""

    cmd: Literal["arm_rest", "arm_halt", "arm_read"]


class ArmReply(TypedDict):
    """The board's answer, broadcast to every client."""

    type: Literal["arm"]
    id: str
    status: AnswerStatus
    reason: str


class ArmStateFields(TypedDict):
    """The ``arm`` object of a state line, under the line's stamp."""

    deg: list[float]
    torque: bool
    phase: ArmPhase
    id: str
    progress_s: float
    error: str


class ArmLink(Protocol):
    """The laptop's end: sends never block, the board's word comes back in its state lines and
    replies."""

    def send(self, trajectory: Trajectory) -> None:
        """Ask the board to play ``trajectory``."""
        ...

    def rest(self) -> None:
        """Send the arm to rest."""
        ...

    def halt(self) -> None:
        """Stop the arm where it is."""
        ...

    def status(self) -> ArmStatus | None:
        """The newest state, None before the first line that read the arm."""
        ...

    def answer(self, trajectory_id: str) -> ArmAnswer | None:
        """The board's latest word on that trajectory, None before any."""
        ...


class ArmDriver(Protocol):
    """The board's end, inside :class:`pepin.base_server.BaseServerCore`: commands in, one step a
    tick after the shared encoder read, at most one unacknowledged write a tick."""

    def trajectory(self, trajectory: Trajectory, now: float) -> ArmAnswer | None:
        """Take ``trajectory``: its refusal, or None when accepted (written from this tick)."""
        ...

    def rest(self, now: float) -> None:
        """To the rest pose at the caps, then torque off."""
        ...

    def halt(self, now: float) -> None:
        """Stop where the arm is and hold it."""
        ...

    def step(self, now: float, wheels_moving: bool) -> None:
        """One tick: progress, the hold's and the rest's ends, the drive rule, the one write."""
        ...

    def status(self) -> ArmStatus:
        """The arm now, for the state line."""
        ...

    def take_answers(self) -> list[ArmAnswer]:
        """The answers born since the last call (a trajectory done, preempted, halted)."""
        ...

    def release(self) -> None:
        """Torque off (the server's shutdown)."""
        ...
