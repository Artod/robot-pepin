"""pepin.goal_link and ros/goto.sh's PEPIN_GOAL_TCP: cancel, where and a goal over the goal
server's socket instead of a fresh ROS process on the board (whose new zenoh session stalled all
laptop -> board delivery for ~3 s, journal 2026-09-25). A fake goal server on a local port
answers; the old path is a fake ``ssh`` on PATH that only writes down what it was asked."""

from __future__ import annotations

import io
import json
import os
import socket
import subprocess
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from pepin import goal_link

REPO = Path(__file__).resolve().parents[2]

NAVIGATORS = {
    "navigate_to_pose": {"outcome": "accepted", "cancelling": 1},
    "navigate_through_poses": {"outcome": "no such goal", "cancelling": 0},
}
CANCEL_LINE = (
    "cancel — navigate_to_pose: accepted, 1 cancelling;"
    " navigate_through_poses: no such goal, 0 cancelling"
)


class FakeGoalServer:
    """The goal server's socket: one request line per connection, then ``answers[cmd]``, each a
    JSON line, then the connection closes. Every request it read is kept in ``asked``."""

    def __init__(self, answers: dict[str, list[dict[str, Any]]]) -> None:
        self.answers = answers
        self.asked: list[dict[str, Any]] = []
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(4)
        self._listener.settimeout(10.0)
        self.port = int(self._listener.getsockname()[1])
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while True:
            try:
                connection, _ = self._listener.accept()
            except OSError:
                return
            with connection:
                request = json.loads(connection.makefile("r").readline())
                self.asked.append(request)
                for answer in self.answers.get(str(request.get("cmd")), []):
                    connection.sendall((json.dumps(answer) + "\n").encode())

    def close(self) -> None:
        self._listener.close()


@pytest.fixture
def served() -> Iterator[FakeGoalServer]:
    """A goal server whose cancel reaches every navigator and whose where answers from TF."""
    server = FakeGoalServer(
        {
            "cancel": [{"event": "cancelled", "had_goal": False, "navigators": NAVIGATORS}],
            "where": [{"event": "where", "pose": "tf", "x": 1.0, "y": 2.0, "yaw_deg": 90.0}],
        }
    )
    yield server
    server.close()


def closed_port() -> int:
    """A local port nobody listens on: a connection there is refused at once."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


def test_the_cancel_prints_goto_ros_s_own_line(served: FakeGoalServer) -> None:
    out = io.StringIO()
    assert goal_link.run_cancel("127.0.0.1", served.port, out) == goal_link.EXIT_OK
    assert out.getvalue().strip() == CANCEL_LINE
    assert served.asked == [{"cmd": "cancel"}]


def test_a_server_whose_cancel_reaches_only_its_own_goal_is_not_taken_for_one() -> None:
    """Before cancel_every_goal the server cancelled the goal IT sent, and goto.sh's drives are
    goto_ros.py's: its answer names no navigators, and the caller must take the old path."""
    server = FakeGoalServer({"cancel": [{"event": "cancelled", "had_goal": False}]})
    out = io.StringIO()
    try:
        assert goal_link.run_cancel("127.0.0.1", server.port, out) == goal_link.EXIT_UNSUPPORTED
    finally:
        server.close()
    assert "without the navigators" in out.getvalue()


def test_a_cancel_no_navigator_confirmed_asks_for_the_old_path_too() -> None:
    """The server is up but Nav2 did not answer it: the line is printed, and the caller is told
    to get a second opinion from the board rather than trust a cancel nobody confirmed."""
    nobody = {action: {"outcome": "no server answered"} for action in goal_link.NAV_ACTIONS}
    server = FakeGoalServer({"cancel": [{"event": "cancelled", "navigators": nobody}]})
    out = io.StringIO()
    try:
        assert goal_link.run_cancel("127.0.0.1", server.port, out) == goal_link.EXIT_UNCONFIRMED
    finally:
        server.close()
    assert "navigate_to_pose: no server answered" in out.getvalue()
    assert "no navigator confirmed" in out.getvalue()


def test_nobody_on_the_port_is_unreachable_and_says_so(capsys: pytest.CaptureFixture[str]) -> None:
    port = closed_port()
    assert goal_link.main(["--port", str(port), "cancel"]) == goal_link.EXIT_UNREACHABLE
    assert "no goal server" in capsys.readouterr().err


def test_a_port_that_closes_without_a_word_is_unreachable_too() -> None:
    server = FakeGoalServer({})
    try:
        with pytest.raises(goal_link.GoalServerUnreachableError, match="without an answer"):
            goal_link.ask({"cmd": "cancel"}, "127.0.0.1", server.port)
    finally:
        server.close()


def test_the_cancel_line_keeps_the_failure_words_of_goto_ros() -> None:
    answer = {
        "event": "cancelled",
        "navigators": {
            "navigate_to_pose": {"outcome": "no server answered"},
            "navigate_through_poses": {"outcome": "the goal had already ended", "cancelling": 0},
        },
    }
    assert goal_link.cancel_line(answer) == (
        "cancel — navigate_to_pose: no server answered;"
        " navigate_through_poses: the goal had already ended, 0 cancelling"
    )
    assert goal_link.cancel_line({"event": "error", "detail": "x"}) is None
    assert [goal_link.cancel_outcome(code) for code in (0, 1, 2, 3, 9)] == [
        "accepted",
        "rejected",
        "no such goal",
        "the goal had already ended",
        "9",
    ]


def test_where_prints_the_server_s_answer_verbatim(served: FakeGoalServer) -> None:
    out = io.StringIO()
    assert goal_link.run_where("127.0.0.1", served.port, out) == goal_link.EXIT_OK
    assert json.loads(out.getvalue()) == {
        "event": "where",
        "pose": "tf",
        "x": 1.0,
        "y": 2.0,
        "yaw_deg": 90.0,
    }


def test_a_goal_is_a_place_or_coordinates() -> None:
    assert goal_link.goal_request(["printer"]) == {"cmd": "go", "place": "printer"}
    assert goal_link.goal_request(["-1.5", "0.3"]) == {
        "cmd": "go",
        "x": -1.5,
        "y": 0.3,
        "yaw_deg": 0.0,
    }
    assert goal_link.goal_request(["1", "2", "90"])["yaw_deg"] == 90.0
    for bad in ([], ["1"], ["a", "b"], ["1", "2", "3", "4"]):
        with pytest.raises(ValueError):
            goal_link.goal_request(bad)


DRIVE = [
    {
        "event": "accepted",
        "run": 481,
        "planner": "GridBased",
        "pose": "tf",
        "recording": "/maps/rec/0481_20260925T080000Z_printer.jsonl",
        "place": "printer",
        "x": -1.0,
        "y": 0.5,
        "yaw_deg": 90.0,
        "sent_in_ms": 40,
    },
    {"event": "feedback", "t": 1.0, "distance": 2.0, "recoveries": 0},
    {"event": "feedback", "t": 2.0, "distance": 1.5, "recoveries": 0},  # under 2 s: skipped
    {"event": "feedback", "t": 3.1, "distance": 0.9, "recoveries": 1},
    {
        "event": "done",
        "run": 481,
        "planner": "navfn",
        "status": 4,
        "seconds": 12.4,
        "arrival": {"x": -1.03, "y": 0.54, "yaw_deg": 88.0, "age_s": 0.02},
        "recording": "/maps/rec/0481_20260925T080000Z_printer.jsonl",
    },
]


def test_a_drive_prints_goto_ros_s_lines_and_names_its_tape(tmp_path: Path) -> None:
    """The lines the operator knows, and the ``taped /maps/rec/...`` goto.sh's finish greps."""
    server = FakeGoalServer({"go": DRIVE})
    out, log_path = io.StringIO(), tmp_path / "goto.log"
    try:
        with log_path.open("a") as log:
            verdict = goal_link.run_goal(
                {"cmd": "go", "place": "printer"}, "127.0.0.1", server.port, out, log
            )
    finally:
        server.close()
    assert verdict == goal_link.EXIT_OK
    lines = out.getvalue().splitlines()
    assert "run 481: taped /maps/rec/0481_20260925T080000Z_printer.jsonl" in lines
    assert "goal printer (-1.00, 0.50) yaw 90 deg accepted" in lines
    assert [line for line in lines if "m left" in line] == [
        "  t+  1.0s   2.00 m left, recoveries 0",
        "  t+  3.1s   0.90 m left, recoveries 1",
    ]
    assert "result: SUCCEEDED after 12 s" in lines
    assert lines[-2].startswith("arrival: x -1.03 m, y +0.54 m, yaw +88 deg, from TF")
    assert lines[-1].strip() == "0.05 m from the goal, heading off by -2 deg"
    assert log_path.read_text().splitlines() == lines, "the log holds what the terminal showed"


def test_a_refused_goal_says_why_and_fails() -> None:
    server = FakeGoalServer({"go": [{"event": "error", "detail": "no such place: 'moon'"}]})
    out = io.StringIO()
    try:
        verdict = goal_link.run_goal({"cmd": "go", "place": "moon"}, "127.0.0.1", server.port, out)
    finally:
        server.close()
    assert verdict == goal_link.EXIT_FAILED
    assert out.getvalue().strip() == "not driving: the goal server refused: no such place: 'moon'"


def test_a_link_lost_mid_drive_says_the_drive_goes_on() -> None:
    server = FakeGoalServer({"go": DRIVE[:2]})
    out = io.StringIO()
    try:
        verdict = goal_link.run_goal(
            {"cmd": "go", "place": "printer"}, "127.0.0.1", server.port, out
        )
    finally:
        server.close()
    assert verdict == goal_link.EXIT_FAILED
    assert "the drive may go on on the board" in out.getvalue()


def test_a_cancelled_drive_is_not_a_success() -> None:
    report = goal_link.DriveReport(lambda _line: None)
    report.event(DRIVE[0])
    report.event({"event": "done", "status": 5, "seconds": 3.0, "arrival": {}})
    assert report.done and report.verdict == goal_link.EXIT_FAILED
    report.event({"event": "done", "status": 4, "seconds": 3.0, "detail": "correction stale"})
    assert report.verdict == goal_link.EXIT_FAILED, "a drive cut by the watch was not reached"


# ---- ros/goto.sh: the switch and the fallback ---------------------------------------------------

FAKE_SSH = """#!/bin/bash
printf 'ssh %s\\n' "$*" >> "$FAKE_LOG"
exit 0
"""


def goto(tmp_path: Path, port: int, *args: str, **env: str) -> tuple[str, list[str]]:
    """Run ros/goto.sh against a goal server on ``port``, with ``ssh`` faked; its output and
    the ssh commands it ran (each one a way onto the board)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (bin_dir / "ssh").write_text(FAKE_SSH)
    (bin_dir / "ssh").chmod(0o755)
    log = tmp_path / "ssh.log"
    log.write_text("")
    run = subprocess.run(
        ["bash", str(REPO / "ros/goto.sh"), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=os.environ
        | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PEPIN_HOST": "127.0.0.1",
            "PEPIN_GOAL_PORT": str(port),
            "FAKE_LOG": str(log),
        }
        | env,
    )
    return run.stdout + run.stderr, log.read_text().splitlines()


def test_goto_s_cancel_goes_over_the_socket_by_default(
    tmp_path: Path, served: FakeGoalServer
) -> None:
    output, ssh = goto(tmp_path, served.port, "cancel")
    assert CANCEL_LINE in output
    assert ssh == [], "no ssh, so no ROS process started on the board"
    assert served.asked == [{"cmd": "cancel"}]


def test_goto_s_cancel_falls_back_to_goto_ros_when_no_server_answers(tmp_path: Path) -> None:
    output, ssh = goto(tmp_path, closed_port(), "cancel")
    assert "no goal server" in output and "the old path" in output
    assert any("goto_ros.py cancel" in line for line in ssh), ssh


def test_goto_s_switch_off_is_the_old_path(tmp_path: Path, served: FakeGoalServer) -> None:
    _, ssh = goto(tmp_path, served.port, "cancel", PEPIN_GOAL_TCP="0")
    assert any("goto_ros.py cancel" in line for line in ssh), ssh
    assert served.asked == [], "the socket is not touched with the switch off"


def test_goto_s_where_goes_over_the_socket(tmp_path: Path, served: FakeGoalServer) -> None:
    output, ssh = goto(tmp_path, served.port, "where")
    assert '"event": "where"' in output and ssh == []


def test_a_cancel_is_confirmed_by_any_navigator_that_answered_even_with_nothing_to_cancel() -> None:
    """The red button's test: a navigator that answered (zero goals included) is a confirmed
    cancel; a server nobody answered, or one without navigators, is not."""
    assert goal_link.cancel_confirmed({"event": "cancelled", "navigators": NAVIGATORS})
    idle = {"navigate_to_pose": {"outcome": "no such goal", "cancelling": 0}}
    assert goal_link.cancel_confirmed({"event": "cancelled", "navigators": idle})
    nobody = {action: {"outcome": "no server answered"} for action in goal_link.NAV_ACTIONS}
    assert not goal_link.cancel_confirmed({"event": "cancelled", "navigators": nobody})
    assert not goal_link.cancel_confirmed({"event": "cancelled", "had_goal": False})
    assert not goal_link.cancel_confirmed({"event": "where"})


def test_the_goal_server_is_looked_for_on_this_machine_first_then_the_board() -> None:
    """ros/go.sh's transport decision, in Python: Nav2 on the Mac answers on 127.0.0.1, else the
    board's; the port decides, never a configuration file."""
    with socket.socket() as listener:  # a bare port: the probe connects and says nothing
        listener.bind(("127.0.0.1", 0))
        listener.listen(2)
        port = int(listener.getsockname()[1])
        assert goal_link.find_server("10.0.0.187", port, timeout_s=0.5) == "127.0.0.1"
        assert goal_link.find_server(None, port, timeout_s=0.5) == "127.0.0.1"
    assert goal_link.find_server(None, closed_port(), timeout_s=0.5) is None
    assert goal_link.find_server("127.0.0.1", closed_port(), timeout_s=0.5) is None
