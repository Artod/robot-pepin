"""pepin.goal_link and ros/goto.sh: cancel, where, planner and a goal over the goal server's
socket on this Mac instead of a fresh ROS process (a new zenoh session on the board stalled all
laptop -> board delivery for ~3 s, journal 2026-09-25). A fake goal server on a local port
answers; ``docker`` and ``ssh`` are fakes on PATH that only write down what they were asked."""

from __future__ import annotations

import contextlib
import io
import json
import os
import socket
import subprocess
import threading
import time
import zlib
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
    assert "the drive may go on; ros/goto.sh cancel stops it" in out.getvalue()


def test_a_cancelled_drive_is_not_a_success() -> None:
    report = goal_link.DriveReport(lambda _line: None)
    report.event(DRIVE[0])
    report.event({"event": "done", "status": 5, "seconds": 3.0, "arrival": {}})
    assert report.done and report.verdict == goal_link.EXIT_FAILED
    report.event({"event": "done", "status": 4, "seconds": 3.0, "detail": "correction stale"})
    assert report.verdict == goal_link.EXIT_FAILED, "a drive cut by the watch was not reached"


# ---- ros/goto.sh: the one client, against fakes --------------------------------------------------

FAKE = """#!/bin/bash
printf '%s %s\\n' "$(basename "$0")" "$*" >> "$FAKE_LOG"
case "$(basename "$0") $*" in
    "docker logs"*) printf '%s\\n' "$FAKE_NAV2_LINE"; exec sleep "$FAKE_SLEEP" ;;
    *"docker exec -i pepin-ros"*) exec sleep "$FAKE_SLEEP" ;;  # a measured motion, turning
    "docker inspect"*) printf '%s\\n' ros2 launch nav.launch.py map:=/maps/flat3.yaml ;;
esac
exit 0
"""
NAV2_LINE = (
    "[component_container_isolated-1] [WARN] [1790000000.123] [controller_server]:"
    " Failed to make progress"
)


def goto_env(tmp_path: Path, port: int, **env: str) -> dict[str, str]:
    """The fakes on PATH and the environment that points ros/goto.sh at them and at ``port``,
    with its records in ``tmp_path``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("docker", "ssh"):
        (bin_dir / name).write_text(FAKE)
        (bin_dir / name).chmod(0o755)
    (tmp_path / "fakes.log").write_text("")
    return (
        dict(os.environ)
        | {
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PEPIN_HOST": "127.0.0.1",
            "PEPIN_BASE_PORT": str(closed_port()),
            "PEPIN_GOAL_PORT": str(port),
            "PEPIN_REC_DIR": str(tmp_path / "rec"),
            "PEPIN_BT_LOG": str(tmp_path / "rec/bt_live.log"),
            "PEPIN_CAMERA_STREAM": f"http://127.0.0.1:{closed_port()}/stream",
            "PEPIN_CLIP_CHECK_S": "0.2",
            "FAKE_LOG": str(tmp_path / "fakes.log"),
            "FAKE_NAV2_LINE": NAV2_LINE,
            "FAKE_SLEEP": _sleep_of(tmp_path),
        }
        | env
    )


def _sleep_of(tmp_path: Path) -> str:
    """How long this test's fakes sleep: a number no other test's fakes use, so what is left
    behind is found by it."""
    return f"31.{zlib.crc32(str(tmp_path).encode()) % 100000:05d}"


def goto(tmp_path: Path, port: int, *args: str, **env: str) -> tuple[int, str, list[str]]:
    """Run ros/goto.sh against a goal server on ``port`` with ``docker`` and ``ssh`` faked, its
    records in ``tmp_path``; its exit status, its output and the commands the fakes were given."""
    run = subprocess.run(
        ["bash", str(REPO / "ros/goto.sh"), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env=goto_env(tmp_path, port, **env),
    )
    return (
        run.returncode,
        run.stdout + run.stderr,
        (tmp_path / "fakes.log").read_text().splitlines(),
    )


def test_goto_s_cancel_goes_over_the_socket(tmp_path: Path, served: FakeGoalServer) -> None:
    code, output, sent = goto(tmp_path, served.port, "cancel")
    assert code == 0 and CANCEL_LINE in output
    assert sent == [], "no ROS process started anywhere"
    assert served.asked == [{"cmd": "cancel"}]


def test_goto_s_cancel_falls_back_to_goto_ros_when_no_server_answers(tmp_path: Path) -> None:
    _, output, sent = goto(tmp_path, closed_port(), "cancel")
    assert "no goal server" in output and "goto_ros.py cancel in pepin-macnav" in output
    assert any(
        line.startswith("docker exec pepin-macnav") and "goto_ros.py cancel" in line
        for line in sent
    ), sent


def test_goto_s_where_and_planner_go_over_the_socket(tmp_path: Path) -> None:
    server = FakeGoalServer(
        {
            "where": [{"event": "where", "pose": "tf", "x": 1.0, "y": 2.0, "yaw_deg": 90.0}],
            "planner": [{"event": "planner", "planner": "ThetaStar"}],
        }
    )
    try:
        code, output, sent = goto(tmp_path, server.port, "where")
        assert code == 0 and '"event": "where"' in output and sent == []
        code, output, _ = goto(tmp_path, server.port, "planner", "theta")
        assert code == 0 and '"planner": "ThetaStar"' in output
    finally:
        server.close()
    assert server.asked[-1] == {"cmd": "planner", "name": "theta"}


def test_a_mark_is_goto_ros_s_in_the_nav2_container_with_the_map_s_book(tmp_path: Path) -> None:
    _, _, sent = goto(tmp_path, closed_port(), "mark", "sofa")
    assert (
        "docker exec pepin-macnav /pepin_entrypoint.sh python3 /tools/goto_ros.py"
        " --places /maps/flat3.places.yaml mark sofa"
    ) in sent, sent


@pytest.mark.slow  # a drive with its film and its streams: ~2 s
def test_a_drive_writes_its_reasons_names_its_tape_and_says_it_has_no_picture(
    tmp_path: Path,
) -> None:
    server = FakeGoalServer({"go": [DRIVE[0], DRIVE[-1]]})
    try:
        code, output, sent = goto(tmp_path, server.port, "printer")
    finally:
        server.close()
    assert code == 0, output
    assert server.asked == [{"cmd": "go", "place": "printer"}]
    logs = sorted((tmp_path / "rec").glob("*_goto.log"))
    assert len(logs) == 1 and "taped /maps/rec/0481_" in logs[0].read_text()
    reasons = logs[0].with_name(logs[0].name.replace(".log", ".nav2.log")).read_text()
    assert "nav2| [controller_server]: Failed to make progress" in reasons
    assert "numbered tape: ros/maps/rec/0481_20260925T080000Z_printer.jsonl" in output
    assert "!! the camera clip is NOT recording" in output, "a dead camera is said aloud"
    assert "!! no camera clip for this drive" in output
    assert any("logs -f --since 1s pepin-macnav" in line for line in sent), sent


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


def test_the_goal_server_is_looked_for_on_this_mac_only() -> None:
    """Nav2 and its goal server live on this Mac (ros/laptop.sh nav): 127.0.0.1 or nobody."""
    with socket.socket() as listener:  # a bare port: the probe connects and says nothing
        listener.bind(("127.0.0.1", 0))
        listener.listen(2)
        port = int(listener.getsockname()[1])
        assert goal_link.find_server(port, timeout_s=0.5) == "127.0.0.1"
    assert goal_link.find_server(closed_port(), timeout_s=0.5) is None


# ---- every way goto.sh can end cancels the goal, and leaves nothing behind ----------------------


class HoldingGoalServer:
    """A goal server whose drive goes on until a cancel arrives: ``go`` answers accepted and
    holds the connection; ``cancel`` confirms, counts, and ends the drive with Nav2's CANCELED;
    ``where`` answers ``navigating``."""

    def __init__(self, navigating: bool = False) -> None:
        self.cancels, self.navigating = 0, navigating
        self.cancelled = threading.Event()
        self._listener = socket.socket()
        self._listener.bind(("127.0.0.1", 0))
        self._listener.listen(8)
        self._listener.settimeout(0.2)
        self.port = int(self._listener.getsockname()[1])
        self._open = True
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while self._open:
            try:
                connection, _ = self._listener.accept()
            except OSError:
                continue
            threading.Thread(target=self._serve, args=(connection,), daemon=True).start()

    def _serve(self, connection: socket.socket) -> None:
        with connection:
            request = json.loads(connection.makefile("r").readline())
            answers: list[dict[str, Any]] = []
            if request["cmd"] == "go":
                connection.sendall((json.dumps(DRIVE[0]) + "\n").encode())
                self.cancelled.wait(20.0)
                answers = [{"event": "done", "status": 5, "seconds": 3.0, "arrival": {}}]
            elif request["cmd"] == "cancel":
                self.cancels += 1
                self.cancelled.set()
                answers = [{"event": "cancelled", "had_goal": True, "navigators": NAVIGATORS}]
            elif request["cmd"] == "where":
                answers = [{"event": "where", "pose": "tf", "navigating": self.navigating}]
            with contextlib.suppress(OSError):
                for answer in answers:
                    connection.sendall((json.dumps(answer) + "\n").encode())

    def close(self) -> None:
        self._open = False
        self._listener.close()


def _left_behind(tmp_path: Path) -> list[str]:
    """Every process still alive that belongs to this test's goto.sh: its fakes' sleeps, its
    film, its streams, anything naming its records."""
    ps = subprocess.run(["ps", "-A", "-o", "pid=,command="], capture_output=True, text=True)
    return [
        line
        for line in ps.stdout.splitlines()
        if (f"sleep {_sleep_of(tmp_path)}" in line or str(tmp_path) in line) and "ps -A" not in line
    ]


def _drive_then(tmp_path: Path, signal_name: str, whole_group: bool) -> tuple[int, str, int]:
    """Start a drive, wait until it is accepted, send the signal (to the whole process group
    like a terminal's Ctrl-C or hang-up, or to goto.sh alone like a kill); exit status, output,
    and the cancels the goal server received."""
    import signal

    server = HoldingGoalServer()
    try:
        process = subprocess.Popen(
            ["bash", str(REPO / "ros/goto.sh"), "printer"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=goto_env(tmp_path, server.port),
            start_new_session=True,
        )
        deadline = time.monotonic() + 15.0
        while time.monotonic() < deadline:
            logs = list((tmp_path / "rec").glob("*_goto.log"))
            if logs and " accepted" in logs[0].read_text():
                break
            time.sleep(0.1)
        sig = getattr(signal, signal_name)
        if whole_group:
            os.killpg(process.pid, sig)
        else:
            process.send_signal(sig)
        out, _ = process.communicate(timeout=30)
        return process.returncode, out, server.cancels
    finally:
        server.close()


@pytest.mark.slow  # a drive with its film, its streams and the signal: ~3 s each
@pytest.mark.parametrize(
    ("signal_name", "whole_group", "status"),
    [("SIGINT", True, 130), ("SIGHUP", True, 129), ("SIGTERM", False, 143)],
)
def test_ctrl_c_a_closed_terminal_and_a_kill_each_cancel_the_goal_and_leave_nothing(
    tmp_path: Path, signal_name: str, whole_group: bool, status: int
) -> None:
    code, out, cancels = _drive_then(tmp_path, signal_name, whole_group)
    assert code == status, out
    assert cancels >= 1, out
    assert "interrupted: cancelling every goal" in out
    time.sleep(1.5)  # a helper that was told to go has had its second
    assert _left_behind(tmp_path) == []


def test_round_refuses_while_anything_navigates_and_when_nobody_answers(tmp_path: Path) -> None:
    """The measured motions write /cmd_vel past every guard of Nav2's, and their own look at the
    action status fails open from the board: the goal server is asked, and only "idle" goes."""
    busy = HoldingGoalServer(navigating=True)
    try:
        code, out, sent = goto(tmp_path, busy.port, "round")
    finally:
        busy.close()
    assert code == 2 and "refused: a navigation goal is running" in out
    assert not [c for c in sent if c.startswith("ssh")], sent
    code, out, sent = goto(tmp_path, closed_port(), "move", "leg", "f0.40")
    assert code == 2 and "refused: the goal server did not say Nav2 is idle" in out
    assert sent == []


@pytest.mark.slow  # the motion runs until the signal
def test_ctrl_c_on_a_measured_motion_stops_it_on_the_board(tmp_path: Path) -> None:
    import signal

    idle = HoldingGoalServer(navigating=False)
    try:
        process = subprocess.Popen(
            ["bash", str(REPO / "ros/goto.sh"), "round", "lap"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=goto_env(tmp_path, idle.port),
            start_new_session=True,
        )
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            if "docker exec -i pepin-ros" in (tmp_path / "fakes.log").read_text():
                break
            time.sleep(0.1)
        os.killpg(process.pid, signal.SIGINT)
        out, _ = process.communicate(timeout=20)
    finally:
        idle.close()
    sent = (tmp_path / "fakes.log").read_text()
    assert process.returncode == 130, out
    assert "pepin_motion.pid" in sent and "kill -KILL" in sent, sent
    assert "stopping the measured motion on the board" in out and "base:" in out
    time.sleep(1.0)
    assert _left_behind(tmp_path) == []
