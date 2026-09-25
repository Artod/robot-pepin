"""Keyboard driving: terminal key reading and the key-to-twist mapping.

Arrow keys only, so driving does not depend on the input language. A terminal gives no key-up,
so a key LATCHES its command until the next key: Up/Down drive straight, Left/Right turn in
place, always at full speed; with Shift held, at the slow speed for aiming and parking. Space
stops; any other key changes nothing.

The mapping is a pure function so it can be unit-tested; the terminal plumbing is a thin
context manager around cbreak mode.
"""

from __future__ import annotations

import os
import select
import sys
import termios
import tty
from collections.abc import Callable
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any

from pepin.kinematics import Twist

FAST_LINEAR_M_S = 0.15  # the base bridge caps forward speed here anyway
FAST_ANGULAR_RAD_S = 0.5
SLOW_LINEAR_M_S = 0.04  # Shift: aiming and parking
SLOW_ANGULAR_RAD_S = 0.15

HELP = "arrows drive, Shift+arrows slow, space stops, Ctrl-C stops and exits"

ESC = "\x1b"
SEQUENCE_GAP_S = 0.01  # the bytes of one escape sequence arrive together, well within this
MAX_SEQUENCE_CHARS = 8  # "\x1b[1;2A" is 6; a runaway sequence is cut here


@dataclass(frozen=True)
class DriveState:
    """The latched twist: what the wheels are commanded until the next key changes it."""

    twist: Twist = field(default_factory=lambda: Twist(0.0, 0.0))


# Arrow keys arrive as escape sequences; Shift+arrow adds the ";2" modifier parameter.
KEY_BINDINGS: dict[str, str] = {
    "\x1b[A": "forward",
    "\x1b[B": "backward",
    "\x1b[D": "left",
    "\x1b[C": "right",
    "\x1b[1;2A": "forward_slow",
    "\x1b[1;2B": "backward_slow",
    "\x1b[1;2D": "left_slow",
    "\x1b[1;2C": "right_slow",
    " ": "stop",
}

ACTION_TWISTS: dict[str, Twist] = {
    "forward": Twist(FAST_LINEAR_M_S, 0.0),
    "backward": Twist(-FAST_LINEAR_M_S, 0.0),
    "left": Twist(0.0, FAST_ANGULAR_RAD_S),
    "right": Twist(0.0, -FAST_ANGULAR_RAD_S),
    "forward_slow": Twist(SLOW_LINEAR_M_S, 0.0),
    "backward_slow": Twist(-SLOW_LINEAR_M_S, 0.0),
    "left_slow": Twist(0.0, SLOW_ANGULAR_RAD_S),
    "right_slow": Twist(0.0, -SLOW_ANGULAR_RAD_S),
    "stop": Twist(0.0, 0.0),
}


def apply_key(state: DriveState, key: str) -> DriveState:
    """Return the state after one key press: a bound key latches its twist, any other key
    changes nothing."""
    twist = ACTION_TWISTS.get(KEY_BINDINGS.get(key, ""))
    return state if twist is None else DriveState(twist)


def read_key(next_char: Callable[[float], str | None]) -> str | None:
    """Assemble one key from the terminal's characters: a plain character, or a whole escape
    sequence ("\\x1b[A" an arrow, "\\x1b[1;2A" Shift+Up) read up to its final letter.

    ``next_char(timeout_s)`` returns the next character, or None if none came within the
    timeout. Returns None when no key is pending.
    """
    key = next_char(0.0)
    if key != ESC:
        return key
    intro = next_char(SEQUENCE_GAP_S)
    if intro is None:
        return key  # a lone Escape
    key += intro
    if intro == "O":  # SS3: one final character follows
        final = next_char(SEQUENCE_GAP_S)
        return key + final if final is not None else key
    if intro != "[":
        return key
    while len(key) < MAX_SEQUENCE_CHARS:  # CSI: parameters, then a final byte in @..~
        char = next_char(SEQUENCE_GAP_S)
        if char is None:
            break
        key += char
        if "@" <= char <= "~":
            break
    return key


class KeyReader:
    """Non-blocking single-key reads from the terminal (cbreak mode while open)."""

    def __init__(self) -> None:
        """Binds to stdin; the terminal is only touched inside the ``with`` block."""
        self._fd = sys.stdin.fileno()
        self._saved: list[Any] | None = None

    def __enter__(self) -> KeyReader:
        """Save the terminal settings and switch to cbreak: keys arrive without Enter.

        Without a terminal (a pipe, an IDE run panel) no key is ever pending.
        """
        if not sys.stdin.isatty():
            return self
        self._saved = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Restore the terminal, so a crash never leaves the shell in cbreak mode."""
        if self._saved is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved)

    def read(self) -> str | None:
        """Return one key (arrow keys as their full escape sequence) or None if none is pending."""
        if self._saved is None:
            return None
        return read_key(self._next_char)

    def _next_char(self, timeout_s: float) -> str | None:
        """One byte straight from the descriptor, or None if none came within the timeout.

        ``os.read``, not ``sys.stdin.read``: the text wrapper pulls the whole escape sequence
        into its own buffer on the first character, and ``select`` then sees nothing pending.
        """
        if not select.select([self._fd], [], [], timeout_s)[0]:
            return None
        data = os.read(self._fd, 1)
        return data.decode("latin-1") if data else None
