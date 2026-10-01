"""The laptop's line to the goal server: cancel, where and a goal over its TCP socket.

Every ROS process started on the board opens a zenoh session, and a new session stalls all
laptop -> board delivery for 2.6-3.1 s about 1.5 s after it starts; an exited one stalls it again
until the router closes it. ``ros/goto.sh cancel`` started exactly such a process
(``goto_ros.py cancel``) in the middle of the drive it was stopping. The goal server
(pepin_bringup.goal_server) runs for the stack's lifetime and speaks JSON lines on port 3337, so
the same command costs one TCP connection and no session at all.

Standard library only: ``ros/goto.sh`` runs it with the laptop's own interpreter
(``PYTHONPATH=src python3 -m pepin.goal_link``), so a cancel never waits for an environment.

    python3 -m pepin.goal_link [--host H] [--port P] cancel
    python3 -m pepin.goal_link [--host H] [--port P] where
    python3 -m pepin.goal_link [--host H] [--port P] [--log FILE] go NAME | X Y [YAW_DEG]

Exit codes: 0 answered (a goal: reached), 1 refused or not reached, 3 no goal server answered,
4 the goal server cannot do what was asked (a build whose cancel reaches only its own goal), 5 a
cancel no navigator confirmed — on 3, 4 and 5 ros/goto.sh takes the old path.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import socket
import sys
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any, TextIO

PORT = 3337
# Both navigators the board's Nav2 runs; a cancel means every goal on either (goto_ros.py's
# cancel_all asks the same two, with the same zero goal id).
NAV_ACTIONS = ("navigate_to_pose", "navigate_through_poses")
# One deadline for the whole cancel, shared between the navigators: the operator is watching.
CANCEL_CONFIRM_S = 30.0  # 3 s until 2026-09-29: under load the confirmation came later, and the
# unconfirmed cancel of a Ctrl-C fell through to ros/stop.sh, whose restart zeroes the odometry
# action_msgs/CancelGoal's return codes in words, as goto_ros.py prints them.
CANCEL_OUTCOMES = {0: "accepted", 1: "rejected", 2: "no such goal", 3: "the goal had already ended"}
# A cancel or a where is answered in well under a second; past this the server is not there.
ASK_TIMEOUT_S = 5.0
# A goal's events come every second while it drives, but a whole-map search before it may take
# a minute (goal_server._find_myself: 60 s for the search, 15 s for the fit).
GOAL_SILENCE_S = 120.0
FEEDBACK_EVERY_S = 2.0  # goto_ros.py's own pace for the progress line
NAV2_SUCCEEDED, NAV2_CANCELED = 4, 5

EXIT_OK, EXIT_FAILED, EXIT_UNREACHABLE, EXIT_UNSUPPORTED, EXIT_UNCONFIRMED = 0, 1, 3, 4, 5


class GoalServerUnreachableError(OSError):
    """Nobody answered on the goal server's port: refused, timed out, or closed silently."""


def cancel_outcome(return_code: int) -> str:
    """A CancelGoal return code in the words goto_ros.py prints for it."""
    return CANCEL_OUTCOMES.get(int(return_code), str(return_code))


def navigator_phrase(action: str, said: dict[str, Any]) -> str:
    """One navigator's part of the cancel line: ``navigate_to_pose: accepted, 1 cancelling``."""
    outcome = str(said.get("outcome", "?"))
    if "cancelling" in said:
        return f"{action}: {outcome}, {int(said['cancelling'])} cancelling"
    return f"{action}: {outcome}"


def cancel_line(answer: dict[str, Any]) -> str | None:
    """The operator's cancel line from the goal server's ``cancelled`` event, in goto_ros.py's
    format; ``None`` when the answer names no navigators (a server that cancels only its own)."""
    navigators = answer.get("navigators")
    if answer.get("event") != "cancelled" or not isinstance(navigators, dict):
        return None
    return "cancel — " + "; ".join(
        navigator_phrase(action, said if isinstance(said, dict) else {})
        for action, said in navigators.items()
    )


def cancel_confirmed(answer: dict[str, Any]) -> bool:
    """True when at least one navigator answered the goal server's cancel (``cancelling`` in its
    part, zero included: no goal to cancel is a confirmed outcome, not a silent navigator)."""
    navigators = answer.get("navigators")
    if answer.get("event") != "cancelled" or not isinstance(navigators, dict):
        return False
    return any(isinstance(said, dict) and "cancelling" in said for said in navigators.values())


def find_server(board_host: str | None, port: int = PORT, timeout_s: float = 1.0) -> str | None:
    """The host whose goal server listens on ``port``: this machine first (Nav2 on the laptop,
    ``ros/laptop.sh`` or the macnav container), then the board; ``None`` when neither answers.
    The same decision ``ros/go.sh`` makes before every command."""
    hosts = ["127.0.0.1"] + ([board_host] if board_host and board_host != "127.0.0.1" else [])
    for host in hosts:
        try:
            with socket.create_connection((host, port), timeout=timeout_s):
                return host
        except OSError:
            continue
    return None


def events(
    request: dict[str, Any],
    host: str,
    port: int = PORT,
    connect_timeout_s: float = ASK_TIMEOUT_S,
    read_timeout_s: float | None = ASK_TIMEOUT_S,
) -> Iterator[dict[str, Any]]:
    """Send one request and yield the server's events until it closes the connection.

    Raises :class:`GoalServerUnreachableError` when the connection cannot be made; a line that
    is not JSON is yielded as ``{"event": "unparsed", "line": ...}`` rather than dropped.
    """
    try:
        connection = socket.create_connection((host, port), timeout=connect_timeout_s)
    except OSError as error:
        raise GoalServerUnreachableError(f"{host}:{port}: {error}") from error
    with connection:
        connection.settimeout(read_timeout_s)
        connection.sendall((json.dumps(request) + "\n").encode())
        with connection.makefile("r", encoding="utf-8") as lines:
            for line in lines:
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except ValueError:
                    yield {"event": "unparsed", "line": line.rstrip("\n")}


def ask(
    request: dict[str, Any], host: str, port: int = PORT, timeout_s: float = ASK_TIMEOUT_S
) -> dict[str, Any]:
    """The first event answering ``request``; :class:`GoalServerUnreachableError` when there is none
    (refused, timed out, or the port closed without a word)."""
    try:
        for event in events(request, host, port, timeout_s, timeout_s):
            return event
    except TimeoutError as error:
        raise GoalServerUnreachableError(
            f"{host}:{port}: no answer in {timeout_s:.0f} s"
        ) from error
    except OSError as error:
        if isinstance(error, GoalServerUnreachableError):
            raise
        raise GoalServerUnreachableError(f"{host}:{port}: {error}") from error
    raise GoalServerUnreachableError(f"{host}:{port}: closed without an answer")


def goal_request(args: list[str]) -> dict[str, Any]:
    """``NAME`` or ``X Y [YAW_DEG]`` as the goal server's go command; ValueError otherwise."""
    if not args:
        raise ValueError("a goal is NAME or X Y [YAW_DEG]")
    try:
        x = float(args[0])
    except ValueError:
        if len(args) != 1:
            raise ValueError(f"a place is one name, not {' '.join(args)!r}") from None
        return {"cmd": "go", "place": args[0]}
    if len(args) not in (2, 3):
        raise ValueError("coordinates are X Y [YAW_DEG]")
    yaw = float(args[2]) if len(args) == 3 else 0.0
    return {"cmd": "go", "x": x, "y": float(args[1]), "yaw_deg": yaw}


class DriveReport:
    """The goal server's events as the lines goto_ros.py prints for a drive.

    ``say`` receives each line; :attr:`verdict` is the exit code once the drive has ended
    (0 only for a Nav2 SUCCEEDED), and :attr:`done` says whether the ``done`` event came at all.
    """

    def __init__(self, say: Callable[[str], None]) -> None:
        self._say = say
        self._last_feedback_t: float | None = None
        self._goal: tuple[float, float, float] | None = None
        self.done = False
        self.verdict = EXIT_FAILED

    @property
    def accepted(self) -> bool:
        """Whether Nav2 took the goal: from then on it drives on the board whatever happens here."""
        return self._goal is not None

    def event(self, event: dict[str, Any]) -> None:
        """Print one event; unknown ones verbatim, so nothing the server says is lost."""
        kind = event.get("event")
        handler = getattr(self, f"_on_{kind}", None) if isinstance(kind, str) else None
        if handler is None:
            self._say(json.dumps(event))
            return
        handler(event)

    def _on_accepted(self, event: dict[str, Any]) -> None:
        x, y, yaw = float(event["x"]), float(event["y"]), float(event["yaw_deg"])
        self._goal = (x, y, yaw)
        place = event.get("place")
        self._say(
            f"goal server: run {event.get('run')}, planner {event.get('planner')}, pose from"
            f" {event.get('pose')}, sent in {event.get('sent_in_ms')} ms"
        )
        recording = event.get("recording")
        if recording:
            self._say(f"run {event.get('run')}: taped {recording}")
        else:
            self._say("no recorder confirmed this run: driving unrecorded")
        self._say(
            f"goal {str(place) + ' ' if place else ''}({x:.2f}, {y:.2f}) yaw {yaw:.0f} deg accepted"
        )

    def _on_feedback(self, event: dict[str, Any]) -> None:
        t = float(event.get("t", 0.0))
        if self._last_feedback_t is not None and t - self._last_feedback_t < FEEDBACK_EVERY_S:
            return
        self._last_feedback_t = t
        self._say(
            f"  t+{t:5.1f}s  {float(event.get('distance', math.nan)):5.2f} m left,"
            f" recoveries {int(event.get('recoveries', 0))}"
        )

    def _on_lost(self, event: dict[str, Any]) -> None:
        reading = event.get("reading") or (
            f"no SLAM correction for {event['correction_s']:.1f} s"
            if isinstance(event.get("correction_s"), (int, float))
            else "the pose is lost"
        )
        self._say(f"!! lost mid-drive at t+{event.get('t')} s: {reading}; the goal server stops")

    def _on_searching(self, event: dict[str, Any]) -> None:
        self._say(f"searching the whole map first (fit {event.get('fit')})")

    def _on_searched(self, event: dict[str, Any]) -> None:
        self._say(f"searched: {event.get('detail')} (fit {event.get('fit')})")

    def _on_resuming(self, event: dict[str, Any]) -> None:
        self._say(f"resuming the same goal (fit {event.get('fit')})")

    def _on_pivot(self, event: dict[str, Any]) -> None:
        self._say(
            f"pivot {event.get('residual_deg', '?')} deg: status {event.get('status')}"
            + (f", {event['after_deg']} deg left" if "after_deg" in event else "")
            + (f" ({event['detail']})" if "detail" in event else "")
        )

    def _on_error(self, event: dict[str, Any]) -> None:
        self._say(f"not driving: the goal server refused: {event.get('detail')}")
        self.verdict = EXIT_FAILED

    def _on_done(self, event: dict[str, Any]) -> None:
        self.done = True
        status = int(event.get("status", 0))
        word = {NAV2_SUCCEEDED: "SUCCEEDED", NAV2_CANCELED: "CANCELED"}.get(status, "FAILED")
        self._say(
            f"result: {word} after {float(event.get('seconds', 0.0)):.0f} s"
            + (f" ({event['detail']})" if event.get("detail") else "")
        )
        self._say(self.arrival(event.get("arrival")))
        reached = status == NAV2_SUCCEEDED and not event.get("detail")
        self.verdict = EXIT_OK if reached else EXIT_FAILED

    def arrival(self, pose: Any) -> str:
        """Where the cart ended against the goal, from the pose the server read at the end."""
        if not isinstance(pose, dict) or "x" not in pose:
            return "arrival: nothing answered about the pose"
        px, py, pyaw = float(pose["x"]), float(pose["y"]), float(pose.get("yaw_deg", 0.0))
        source = f"fit {pose['fit']:.2f}" if "fit" in pose else "from TF"
        if "age_s" in pose:
            source += f", map -> base_link {float(pose['age_s']) * 1e3:.0f} ms old"
        line = f"arrival: x {px:+.2f} m, y {py:+.2f} m, yaw {pyaw:+.0f} deg, {source}"
        if self._goal is None:
            return line
        x, y, yaw = self._goal
        turn = (pyaw - yaw + 180.0) % 360.0 - 180.0
        return (
            f"{line}\n         {math.hypot(px - x, py - y):.2f} m from the goal,"
            f" heading off by {turn:+.0f} deg"
        )


def run_cancel(host: str, port: int, out: TextIO) -> int:
    """Cancel every goal through the goal server and print goto_ros.py's line for it."""
    # The server confirms within CANCEL_CONFIRM_S: the answer is waited for a little longer.
    answer = ask({"cmd": "cancel"}, host, port, timeout_s=CANCEL_CONFIRM_S + 3.0)
    line = cancel_line(answer)
    if line is None:
        print(
            f"!! the goal server answered {json.dumps(answer)} without the navigators: a build"
            " whose cancel reaches only the goal it sent itself",
            file=out,
            flush=True,
        )
        return EXIT_UNSUPPORTED
    print(line, file=out, flush=True)
    if not cancel_confirmed(answer):
        # Not one navigator answered the server: a second opinion from the board is worth a
        # stall here, and the old path gives it.
        print("!! no navigator confirmed the cancel through the goal server", file=out, flush=True)
        return EXIT_UNCONFIRMED
    return EXIT_OK


def run_where(host: str, port: int, out: TextIO) -> int:
    """Print the goal server's ``where`` answer verbatim, as ``ros/go.sh where`` does."""
    answer = ask({"cmd": "where"}, host, port)
    print(json.dumps(answer), file=out, flush=True)
    return EXIT_OK if answer.get("event") == "where" else EXIT_FAILED


def run_goal(
    request: dict[str, Any], host: str, port: int, out: TextIO, log: TextIO | None = None
) -> int:
    """Drive through the goal server and print the drive as goto_ros.py does; Ctrl-C cancels.

    The goal lives on the board: a connection lost mid-drive leaves it running there, said so.
    """

    def say(line: str) -> None:
        print(line, file=out, flush=True)
        if log is not None:
            print(line, file=log, flush=True)

    report = DriveReport(say)
    try:
        for event in events(request, host, port, ASK_TIMEOUT_S, GOAL_SILENCE_S):
            report.event(event)
    except KeyboardInterrupt:
        say("Ctrl-C: cancelling every goal through the goal server...")
        try:
            run_cancel(host, port, out)
        except GoalServerUnreachableError as error:
            say(f"!! cancel NOT sent ({error}) — run ros/goto.sh cancel or ros/stop.sh NOW")
        return EXIT_FAILED
    except GoalServerUnreachableError:
        raise
    except OSError as error:
        say(f"!! the link to the goal server dropped ({error})")
    if report.accepted and not report.done:
        say("!! no result: the drive may go on on the board; ros/goto.sh cancel stops it")
    return report.verdict


def main(argv: list[str] | None = None) -> int:
    """Parse the command line, run one command, return its exit code."""
    parser = argparse.ArgumentParser(
        prog="goal_link", description="cancel, where or a goal through the goal server's socket"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=PORT)
    parser.add_argument("--log", type=Path, help="a goal's lines are also written here")
    parser.add_argument("command", choices=("cancel", "where", "go"))
    parser.add_argument("goal", nargs="*", help="go: NAME or X Y [YAW_DEG]")
    args = parser.parse_args(argv)
    try:
        if args.command == "cancel":
            return run_cancel(args.host, args.port, sys.stdout)
        if args.command == "where":
            return run_where(args.host, args.port, sys.stdout)
        try:
            request = goal_request(args.goal)
        except ValueError as error:
            parser.error(str(error))
        with contextlib.ExitStack() as stack:
            log = stack.enter_context(args.log.open("a")) if args.log else None
            return run_goal(request, args.host, args.port, sys.stdout, log)
    except GoalServerUnreachableError as error:
        print(f"!! no goal server: {error}", file=sys.stderr, flush=True)
        return EXIT_UNREACHABLE


if __name__ == "__main__":
    sys.exit(main())
