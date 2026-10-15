"""The laptop's end of the arm (:class:`pepin.expression.trajectory.ArmLink`): one connection to
the base server's port, trajectories sent without waiting, the arm's state and the board's
answers read off the lines that come back.

Every state line that read the arm carries its ``arm`` object (angles, torque, phase, the
trajectory playing); the answers (``accepted``/``refused`` to the sender, ``done``/
``preempted``/``halted`` to everyone) arrive as ``{"type": "arm"}`` lines and are kept by
trajectory id, each with the laptop time it arrived (the cue log's "accepted"). The connection
never sends a twist or a stop, so it never counts as a driver of the wheels.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from pepin.arm_driver import trajectory_line
from pepin.base_link import BASE_PORT
from pepin.expression.trajectory import ArmAnswer, ArmStatus, Trajectory
from pepin.streams import Connector, JsonLinesClient

AnswerSink = Callable[[ArmAnswer, float], None]
REPLY_TYPES = ("arm_read", "arm_rest", "arm_halt")


class _Lines(JsonLinesClient):
    """The JSON-lines connection, every message handed to ``sink``."""

    def __init__(
        self,
        host: str,
        port: int,
        sink: Callable[[dict[str, Any]], None],
        connector: Connector | None,
    ) -> None:
        super().__init__(host, port, name="arm", connector=connector)
        self._sink = sink

    def _ingest(self, message: dict[str, Any]) -> None:
        self._sink(message)


class ArmClient:
    """The arm through the base server (TCP 3336): non-blocking sends, the newest state and
    every answer by trajectory id."""

    def __init__(
        self,
        host: str,
        port: int = BASE_PORT,
        *,
        connector: Connector | None = None,
        on_answer: AnswerSink | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Nothing connects until :meth:`start`; ``on_answer`` hears each answer on the reader
        thread with the laptop time it arrived."""
        self.clock = clock
        self.on_answer = on_answer
        self._lines = _Lines(host, port, self._ingest, connector)
        self._lock = threading.Lock()
        self._status: ArmStatus | None = None
        self._answers: dict[str, tuple[ArmAnswer, float]] = {}
        self._replies: dict[str, dict[str, Any]] = {}
        self._replied = threading.Condition(self._lock)

    def start(self) -> ArmClient:
        """Connect and read in the background; returns self so it chains."""
        self._lines.start()
        return self

    def close(self) -> None:
        """Stop reading and let the connection go."""
        self._lines.close()

    # -- the ArmLink -------------------------------------------------------------------------

    def send(self, trajectory: Trajectory) -> None:
        """Ask the board to play ``trajectory`` (dropped, logged, while the link is down)."""
        self._lines.send(dict(trajectory_line(trajectory)))

    def rest(self) -> None:
        """Send the arm to rest."""
        self._lines.send({"cmd": "arm_rest"})

    def halt(self) -> None:
        """Stop the arm where it is (held, then rest by itself)."""
        self._lines.send({"cmd": "arm_halt"})

    def status(self) -> ArmStatus | None:
        """The newest state line's arm; None before the first that read it."""
        with self._lock:
            return self._status

    def answer(self, trajectory_id: str) -> ArmAnswer | None:
        """The board's latest word on that trajectory."""
        with self._lock:
            held = self._answers.get(trajectory_id)
        return held[0] if held is not None else None

    # -- the command line's requests -------------------------------------------------------------

    def ask(self, cmd: str, timeout_s: float = 2.0) -> dict[str, Any] | None:
        """``arm_read``, ``arm_rest`` or ``arm_halt``, and its reply (None when none came)."""
        with self._lock:
            self._replies.pop(cmd, None)
        self._lines.send({"cmd": cmd})
        deadline = self.clock() + timeout_s
        with self._lock:
            while cmd not in self._replies:
                left = deadline - self.clock()
                if left <= 0:
                    return None
                self._replied.wait(min(left, 0.1))
            return self._replies.pop(cmd)

    def wait_answer(
        self, trajectory_id: str, statuses: tuple[str, ...], timeout_s: float
    ) -> ArmAnswer | None:
        """Block until the board's word on the trajectory is one of ``statuses``."""
        deadline = self.clock() + timeout_s
        with self._lock:
            while True:
                held = self._answers.get(trajectory_id)
                if held is not None and held[0].status in statuses:
                    return held[0]
                left = deadline - self.clock()
                if left <= 0:
                    return held[0] if held is not None else None
                self._replied.wait(min(left, 0.1))

    def wait_status(self, timeout_s: float = 3.0) -> ArmStatus | None:
        """The first state of the arm, waited for up to ``timeout_s``."""
        deadline = self.clock() + timeout_s
        while self.clock() < deadline:
            status = self.status()
            if status is not None:
                return status
            time.sleep(0.05)
        return self.status()

    # -- the reader ------------------------------------------------------------------------------

    def _ingest(self, message: dict[str, Any]) -> None:
        kind = message.get("type")
        if kind == "state" and isinstance(message.get("arm"), dict):
            status = status_of(message["arm"], float(message.get("t", 0.0)))
            with self._lock:
                self._status = status
        elif kind == "arm":
            answer = ArmAnswer(str(message.get("id", "")), message.get("status", "refused"),
                               str(message.get("reason", "")))  # fmt: skip
            arrived = self.clock()
            with self._lock:
                self._answers[answer.id] = (answer, arrived)
                self._replied.notify_all()
            if self.on_answer is not None:
                self.on_answer(answer, arrived)
        elif kind in REPLY_TYPES:
            with self._lock:
                self._replies[str(kind)] = message
                self._replied.notify_all()


def status_of(arm: dict[str, Any], stamp: float) -> ArmStatus:
    """A state line's ``arm`` object as an :class:`ArmStatus`."""
    deg = arm.get("deg")
    return ArmStatus(
        stamp,
        tuple(float(d) for d in deg) if isinstance(deg, list) else None,
        bool(arm.get("torque", False)),
        arm.get("phase", "off"),
        str(arm.get("id", "")),
        float(arm.get("progress_s", 0.0)),
        str(arm.get("error", "")),
    )
