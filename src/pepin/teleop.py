"""Keyboard driving: terminal key reading and the key-to-twist mapping.

The mapping is a pure state machine so it can be unit-tested; the terminal
plumbing is a thin context manager around cbreak mode.
"""

from __future__ import annotations

import select
import sys
import termios
import tty
from dataclasses import dataclass, field, replace
from types import TracebackType
from typing import Any

from pepin.kinematics import Twist


@dataclass(frozen=True)
class DriveState:
    """Commanded twist plus the speed steps the keys move it by."""

    twist: Twist = field(default_factory=lambda: Twist(0.0, 0.0))
    linear_step: float = 0.05
    angular_step: float = 0.2
    quit: bool = False


KEY_BINDINGS = {
    "w": "forward",
    "s": "backward",
    "a": "left",
    "d": "right",
    " ": "stop",
    "q": "quit",
    "\x1b[A": "forward",  # arrow keys arrive as escape sequences
    "\x1b[B": "backward",
    "\x1b[D": "left",
    "\x1b[C": "right",
}


def apply_key(state: DriveState, key: str) -> DriveState:
    """Return the state after one key press; unknown keys change nothing."""
    action = KEY_BINDINGS.get(key)
    t = state.twist
    if action == "forward":
        return replace(state, twist=Twist(t.linear + state.linear_step, t.angular))
    if action == "backward":
        return replace(state, twist=Twist(t.linear - state.linear_step, t.angular))
    if action == "left":
        return replace(state, twist=Twist(t.linear, t.angular + state.angular_step))
    if action == "right":
        return replace(state, twist=Twist(t.linear, t.angular - state.angular_step))
    if action == "stop":
        return replace(state, twist=Twist(0.0, 0.0))
    if action == "quit":
        return replace(state, twist=Twist(0.0, 0.0), quit=True)
    return state


class KeyReader:
    """Non-blocking single-key reads from the terminal (cbreak mode while open)."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._saved: list[Any] | None = None

    def __enter__(self) -> KeyReader:
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read(self) -> str | None:
        """Return one key (arrow keys as their full escape sequence) or None if none is pending."""
        if not select.select([self._fd], [], [], 0)[0]:
            return None
        key = sys.stdin.read(1)
        if key == "\x1b" and select.select([self._fd], [], [], 0.01)[0]:
            key += sys.stdin.read(2)
        return key
