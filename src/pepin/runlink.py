"""The goal server's side of the run recorder: ask for a tape by name, learn its number and path.

A drive is recorded where its sensors are. The split's first tapes were written on the laptop
and had no scans, no fused odometry and no local costmap: those topics did not cross the bridge
while the filtered scan, the wheel odometry and the ToF did (2026-09-09). So the recorder is a
node on the board, and the goal server — on whichever side — sends it one command and reads one
status. This module is that exchange without ROS: the node only carries the strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

RUN_COMMAND_TOPIC = "pepin/run"  # the goal server's word: start <name> / stop
RUN_STATUS_TOPIC = "pepin/run_status"  # the recorder's latched answer: what it is doing now
IDLE = "idle"
RECORDING = "recording"


@dataclass(frozen=True)
class RunStatus:
    """What the recorder last said: its state, the run's number and tape, the name asked for."""

    state: str
    run: int = 0
    recording: str | None = None
    name: str | None = None

    def to_json(self) -> str:
        """The status as one JSON line for the latched topic."""
        return json.dumps(
            {"state": self.state, "run": self.run, "recording": self.recording, "name": self.name}
        )

    @classmethod
    def from_json(cls, text: str) -> RunStatus | None:
        """Parse a status line; ``None`` for anything that is not one."""
        try:
            data = json.loads(text)
            return cls(
                str(data["state"]),
                int(data.get("run", 0)),
                data.get("recording"),
                data.get("name"),
            )
        except (ValueError, TypeError, KeyError):
            return None


def start_command(name: str) -> str:
    """The command that opens a tape called ``name``."""
    return json.dumps({"cmd": "start", "name": name})


def stop_command() -> str:
    """The command that closes the open tape."""
    return json.dumps({"cmd": "stop"})


def parse_command(text: str) -> tuple[str, str | None] | None:
    """The recorder's side: (``"start"``, name) or (``"stop"``, None), or ``None`` for noise."""
    try:
        data = json.loads(text)
        cmd = str(data["cmd"])
    except (ValueError, TypeError, KeyError):
        return None
    if cmd == "start" and isinstance(data.get("name"), str) and data["name"]:
        return "start", data["name"]
    if cmd == "stop":
        return "stop", None
    return None


class RunLink:
    """Tracks the run the recorder confirmed, so events can name it.

    ``run`` and ``recording`` follow the last status heard; ``started(name)`` is true once the
    recorder says it is recording the tape asked for, ``stopped()`` once it says idle. The
    waiting itself is the node's business (a thread and a timeout); this only knows the answers.
    """

    def __init__(self) -> None:
        self._status: RunStatus | None = None

    def observe(self, status: RunStatus) -> None:
        """The recorder spoke."""
        self._status = status

    @property
    def run(self) -> int:
        """The number of the run being recorded, or of the last one; 0 before any."""
        return self._status.run if self._status else 0

    @property
    def recording(self) -> str | None:
        """The tape being written, or the last one written."""
        return self._status.recording if self._status else None

    def started(self, name: str) -> bool:
        """True once the recorder is writing the tape asked for under ``name``."""
        return (
            self._status is not None
            and self._status.state == RECORDING
            and self._status.name == name
        )

    def stopped(self) -> bool:
        """True once the recorder is idle (also before it ever spoke: nothing to close)."""
        return self._status is None or self._status.state == IDLE
