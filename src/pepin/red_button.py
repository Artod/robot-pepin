"""The base's own stop, confirmed by the wheels: the half of the red button that is not Nav2's.

The base server zeroes the wheels on ``{"cmd": "stop"}`` from any client, but nothing latches
it: the base bridge forwards ``/cmd_vel`` as it comes, and a controller that is still alive
overwrites the stop within 50 ms. So a stop is only believed from the base's own state stream —
the ``moving`` field of the lines it sends 20 times a second — once the last few of them, read
for a whole window after the stop, say the wheels are commanded still. ``ros/stop.sh`` sends it
on every press, whatever the cancel said, and again after it has killed what drives the wheels.

Standard library only, like :mod:`pepin.goal_link`: the red button never waits for an
environment.

    python3 -m pepin.red_button [--host BOARD] [--port 3336] [--confirm-s 1.0]

Exit codes: 0 the wheels read still, 1 they did not (moving again, or the stream said nothing),
3 no base server answered.
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time

BASE_PORT = 3336  # = pepin.base_link.BASE_PORT
CONFIRM_S = 1.0  # how long the state stream is read after the stop
STILL_LINES = 3  # the last this many state lines must all say the wheels are still
EXIT_STILL, EXIT_MOVING, EXIT_UNREACHABLE = 0, 1, 3


def verdict(moving: list[bool]) -> tuple[bool, str]:
    """Whether the state lines read after a stop (``moving`` of each, in order) confirm it."""
    if len(moving) < STILL_LINES:
        return False, f"the base said {len(moving)} state line(s) after the stop: not confirmed"
    if any(moving[-STILL_LINES:]):
        return False, (
            f"the wheels are commanded again after the stop ({sum(moving)} of {len(moving)}"
            " state lines moving): something still drives them"
        )
    return True, f"the wheels read still ({len(moving)} state lines, the last {STILL_LINES} still)"


def base_stop(
    host: str, port: int = BASE_PORT, confirm_s: float = CONFIRM_S, connect_s: float = 2.0
) -> tuple[int, str]:
    """Send the stop and read the state stream for ``confirm_s``: an exit code and its line."""
    try:
        sock = socket.create_connection((host, port), timeout=connect_s)
    except OSError as error:
        return EXIT_UNREACHABLE, f"no base server on {host}:{port} ({error})"
    moving: list[bool] = []
    with sock:
        try:
            sock.sendall(b'{"cmd": "stop"}\n')
        except OSError as error:
            return EXIT_UNREACHABLE, f"the base server on {host}:{port} hung up ({error})"
        deadline = time.monotonic() + confirm_s
        buffer = b""
        while (left := deadline - time.monotonic()) > 0:
            sock.settimeout(left)
            try:
                chunk = sock.recv(4096)
            except OSError:  # the window ran out mid-read, or the stream broke
                break
            if not chunk:
                break
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for raw in lines:
                try:
                    message = json.loads(raw)
                except ValueError:
                    continue
                if isinstance(message, dict) and message.get("type") == "state":
                    moving.append(bool(message.get("moving")))
    still, said = verdict(moving)
    return (EXIT_STILL if still else EXIT_MOVING), said


def main(argv: list[str] | None = None) -> int:
    """Parse the command line, stop the base once, print the verdict line."""
    parser = argparse.ArgumentParser(prog="red_button", description="the base's own stop")
    parser.add_argument("--host", default="10.0.0.187")
    parser.add_argument("--port", type=int, default=BASE_PORT)
    parser.add_argument("--confirm-s", type=float, default=CONFIRM_S)
    args = parser.parse_args(argv)
    code, line = base_stop(args.host, args.port, args.confirm_s)
    print(f"base: {line}", flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
