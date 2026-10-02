"""Keyboard driving: the key-to-command mappings, a terminal reader and a game-mode window.

Two ways to drive from a keyboard, one mapping of the arrows between them.

The terminal (``ros/teleop.sh``, :mod:`pepin_bringup.teleop_keys`) reads arrow keys only, so
driving does not depend on the input language. A terminal gives no key-up, so a key LATCHES
its command until the next key: Up/Down drive straight, Left/Right turn in place, always at
full speed; with Shift held, at the slow speed for aiming and parking. Space stops; any other
key changes nothing.

The game mode, ``uv run python -m pepin.teleop --game [--host 10.0.0.187]``, is a small pygame
window on the Mac that talks to the base server itself (:class:`pepin.base_link.BaseClient`,
TCP 3336). Keys act while HELD and stop the moment they are released — and only while the
window has the focus: a global key listener would move the robot on a keystroke meant for
another application. Arrows are the wheels, the mapping and speeds above (Up/Down with
Left/Right combine into an arc); W/S tilt the head and A/D pan it (``neck_jog`` on the board,
which walks the goal at its own rate within config/neck.json's limits and has its own
half-second deadman); Shift makes both slow; Space stops everything; Esc or closing the window
stops the wheels, ends the jog and exits, and so does any exception. The loop runs at 20 Hz: a
twist every tick while an arrow is held (the base's deadman is 0.5 s) and one stop on release;
a jog every tick while a head key is held and one zero jog on release. The window shows the
twist, the neck's encoders, the speed and whether it has the focus.

The mappings are pure functions so they can be unit-tested; the terminal plumbing is a thin
context manager around cbreak mode, and pygame is imported only under ``--game``.
"""

from __future__ import annotations

import argparse
import logging
import os
import select
import sys
import termios
import time
import tty
from collections.abc import Callable, Collection
from dataclasses import dataclass, field
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol

from pepin.base_link import BASE_PORT
from pepin.kinematics import STOP, Twist

if TYPE_CHECKING:
    from pepin.base_link import BaseState
    from pepin.neck import NeckReading

FAST_LINEAR_M_S = 0.45  # config/base.json max_speed_m_s: the base clips anything above
FAST_ANGULAR_RAD_S = 1.0  # config/base.json max_yaw_rate_rad_s
SLOW_LINEAR_M_S = 0.064  # Shift: aiming and parking
SLOW_ANGULAR_RAD_S = 0.24

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


# -- game mode: keys act while held -----------------------------------------------------------

GAME_HZ = 20.0  # ticks of the window loop; a twist each one while held, well inside the deadman
GAME_HZ_MIN, GAME_HZ_MAX = 5.0, 50.0  # --hz is clamped: 5 Hz is still inside the deadman
EXIT_STOP_REPEATS = 3  # the exit's stops go out a few times, like teleop_keys': one may be lost
EXIT_STOP_GAP_S = 0.05
GAME_HELP = "arrows drive, W/S tilt, A/D pan, Shift slow, Space stops all, Esc quits"
WINDOW_TITLE = "Pepin teleop — keys act only while this window is focused"
DEFAULT_HOST = "10.0.0.187"


@dataclass(frozen=True)
class HeldKeys:
    """The game keys that are down this tick; the default (nothing held) is what an unfocused
    window reads, whatever the keyboard is doing."""

    up: bool = False
    down: bool = False
    left: bool = False
    right: bool = False
    tilt_up: bool = False  # W
    tilt_down: bool = False  # S
    pan_left: bool = False  # A
    pan_right: bool = False  # D
    shift: bool = False
    space: bool = False


@dataclass(frozen=True)
class GameCommand:
    """What one tick asks the base for: the wheels' twist; the head's jog, pan +1 left and
    tilt +1 down as :class:`pepin.neck.NeckAngles` signs them; slow or fast; and whether Space
    is down, which stops everything this tick and every tick it stays down."""

    twist: Twist = STOP
    pan: int = 0
    tilt: int = 0
    slow: bool = False
    stop_all: bool = False


def game_command(keys: HeldKeys) -> GameCommand:
    """The command for one tick of held keys: opposite keys cancel, Up/Down and Left/Right
    combine into an arc, Shift picks the slow speeds, Space overrides everything."""
    if keys.space:
        return GameCommand(stop_all=True, slow=keys.shift)
    linear = SLOW_LINEAR_M_S if keys.shift else FAST_LINEAR_M_S
    angular = SLOW_ANGULAR_RAD_S if keys.shift else FAST_ANGULAR_RAD_S
    return GameCommand(
        twist=Twist(
            linear * (int(keys.up) - int(keys.down)),
            angular * (int(keys.left) - int(keys.right)),
        ),
        pan=int(keys.pan_left) - int(keys.pan_right),
        tilt=int(keys.tilt_down) - int(keys.tilt_up),
        slow=keys.shift,
    )


class BaseCommands(Protocol):
    """What the game loop asks of a base link; :class:`pepin.base_link.BaseClient` has it."""

    def set_twist(self, twist: Twist) -> None:
        """Drive at this body velocity; re-arms the board's deadman."""
        ...

    def stop(self) -> None:
        """Stop the wheels now."""
        ...

    def neck_jog(self, pan: int, tilt: int, *, slow: bool = False) -> None:
        """Walk the head in these directions; both zero stops it where it is."""
        ...


class CommandStream:
    """The messages behind the held keys: a twist every tick while the wheels are asked to
    move and one stop the tick they are released; a jog every tick while the head is asked to
    move and one zero jog the tick it is released; Space sends both stops every tick it is
    down. Nothing is sent while nothing is held, so the board's idle release runs."""

    def __init__(self, link: BaseCommands) -> None:
        """``link`` receives the messages; nothing is sent until :meth:`tick`."""
        self._link = link
        self._wheels = False  # the wheels were asked to move on the previous tick
        self._head = False  # the head was asked to move on the previous tick

    def tick(self, command: GameCommand) -> None:
        """Send what this tick's command asks for, given what the previous tick asked."""
        wheels = command.twist != STOP
        if wheels:
            self._link.set_twist(command.twist)
        elif self._wheels or command.stop_all:
            self._link.stop()
        head = command.pan != 0 or command.tilt != 0
        if head:
            self._link.neck_jog(command.pan, command.tilt, slow=command.slow)
        elif self._head or command.stop_all:
            self._link.neck_jog(0, 0, slow=command.slow)
        self._wheels, self._head = wheels, head

    def stop_all(self, *, repeats: int = 1, gap_s: float = 0.0) -> None:
        """Stop the wheels and the head now, whatever the previous tick was (exit, a crash);
        ``repeats`` times, ``gap_s`` apart, when one line going astray must not matter."""
        for i in range(repeats):
            if i:
                time.sleep(gap_s)
            self.tick(GameCommand(stop_all=True))


def status_lines(
    command: GameCommand,
    state: BaseState | None,
    neck: NeckReading | None,
    neck_error: str | None,
    *,
    focused: bool,
) -> list[str]:
    """What the window shows: the help, the twist and speed, the board's word on the wheels,
    the neck's encoders and the jog, a refused jog, and the focus warning."""
    speed = "SLOW (Shift held)" if command.slow else "FAST (Shift: slow)"
    lines = [
        GAME_HELP,
        f"wheels   v {command.twist.linear:+.2f} m/s   w {command.twist.angular:+.2f} rad/s   "
        f"{speed}",
    ]
    if state is None:
        lines.append("base     no state yet: connecting")
    else:
        wheels = "moving" if state.moving else "still"
        torque = "armed" if state.armed else "free"
        deadman = ", DEADMAN fired" if state.deadman else ""
        lines.append(f"base     {wheels}, {torque}{deadman}, state {state.age_s * 1000:.0f} ms old")
    jog = f"jog pan {command.pan:+d} tilt {command.tilt:+d}" if command.pan or command.tilt else ""
    if neck is not None and neck.ticks is not None:
        lines.append(f"neck     pan {neck.pan_ticks} ticks   tilt {neck.tilt_ticks} ticks   {jog}")
    else:
        lines.append(f"neck     encoders unread   {jog}")
    if neck_error is not None:
        lines.append(f"neck     refused: {neck_error}")
    if command.stop_all:
        lines.append("STOP: everything stopped")
    if not focused:
        lines.append("UNFOCUSED: keys are ignored, everything is stopped")
    return lines


def _held_keys(held: Collection[int], pg: Any) -> HeldKeys:
    """The physical keys held, by SDL scancode, as :class:`HeldKeys` (``pg`` is the pygame
    module, for its KSCAN_* constants). Scancodes name the key's PLACE on the keyboard, so W/A/S/D
    work under any layout — a Russian layout puts a different character on the same key."""
    return HeldKeys(
        up=pg.KSCAN_UP in held,
        down=pg.KSCAN_DOWN in held,
        left=pg.KSCAN_LEFT in held,
        right=pg.KSCAN_RIGHT in held,
        tilt_up=pg.KSCAN_W in held,
        tilt_down=pg.KSCAN_S in held,
        pan_left=pg.KSCAN_A in held,
        pan_right=pg.KSCAN_D in held,
        shift=pg.KSCAN_LSHIFT in held or pg.KSCAN_RSHIFT in held,
        space=pg.KSCAN_SPACE in held,
    )


def run_game(host: str, port: int = BASE_PORT, *, hz: float = GAME_HZ) -> None:
    """The game-mode window against the base server at ``host:port``, until Esc or the window
    closes; the wheels and the head are stopped on the way out, whatever the reason."""
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    import pygame  # only the game mode pays for it; the terminal teleop runs where it is absent

    from pepin.base_link import BaseClient

    client = BaseClient(host, port).start()
    stream = CommandStream(client)
    pygame.init()
    try:
        screen = pygame.display.set_mode((760, 240))
        pygame.display.set_caption(WINDOW_TITLE)
        font = pygame.font.Font(None, 26)
        clock = pygame.time.Clock()
        running = True
        held: set[int] = set()  # scancodes down right now, from the key events
        while running:
            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    held.add(int(event.scancode))
                    if event.scancode == pygame.KSCAN_ESCAPE:
                        running = False
                elif event.type == pygame.KEYUP:
                    held.discard(int(event.scancode))
                elif event.type == pygame.QUIT:
                    running = False
            focused = bool(pygame.key.get_focused())
            if not focused:
                held.clear()  # a release while unfocused never reaches this window
            keys = _held_keys(held, pygame) if focused else HeldKeys()
            command = game_command(keys)
            stream.tick(command)
            client.ask_neck()
            lines = status_lines(
                command, client.state(), client.neck(), client.neck_error(), focused=focused
            )
            screen.fill((24, 24, 28) if focused else (72, 16, 16))
            for row, line in enumerate(lines):
                screen.blit(font.render(line, True, (230, 230, 230)), (16, 16 + 28 * row))
            pygame.display.flip()
            clock.tick(hz)
    finally:
        stream.stop_all(repeats=EXIT_STOP_REPEATS, gap_s=EXIT_STOP_GAP_S)
        client.close()
        pygame.quit()


def main(argv: list[str] | None = None) -> None:
    """``python -m pepin.teleop --game [--host HOST] [--port PORT]``: the game-mode window.

    The latching terminal teleop is not run from here: it is a ROS node (``ros/teleop.sh``).
    """
    parser = argparse.ArgumentParser(
        description="Drive Pepin from a window where keys act only while held and focused."
    )
    parser.add_argument(
        "--game",
        action="store_true",
        help="the game-mode window (the only mode here; the terminal teleop is ros/teleop.sh)",
    )
    parser.add_argument(
        "--host",
        default=os.environ.get("PEPIN_HOST", DEFAULT_HOST),
        help="the board running the base server (default: PEPIN_HOST, else 10.0.0.187)",
    )
    parser.add_argument("--port", type=int, default=BASE_PORT, help="the base server's port")
    parser.add_argument(
        "--hz",
        type=float,
        default=GAME_HZ,
        help=f"the loop rate, clamped to {GAME_HZ_MIN:.0f}..{GAME_HZ_MAX:.0f} (the board's deadman"
        " is 0.5 s)",
    )
    args = parser.parse_args(argv)
    if not args.game:
        parser.error("pass --game; the latching terminal teleop runs through ros/teleop.sh")
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    try:
        import pygame  # noqa: F401  # the one dependency this mode adds: one line, not a traceback
    except ModuleNotFoundError:
        parser.exit(2, "pygame is not installed: uv sync --group macos\n")
    logging.basicConfig(level=logging.INFO, format="%(levelname).1s %(name)s: %(message)s")
    run_game(args.host, args.port, hz=min(max(args.hz, GAME_HZ_MIN), GAME_HZ_MAX))


if __name__ == "__main__":
    main()
