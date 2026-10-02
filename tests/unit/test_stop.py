"""ros/stop.sh, the red button, against fakes: a goal server on a local port, a base server that
streams its state like the board's (pepin.base_server), and ``docker``/``ssh`` on PATH that only
write down what they were asked. The wheels are believed only from the base's own state stream,
and when either the cancel or the wheels are not confirmed, what drives the wheels is killed
before the stop that is believed."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from pepin import red_button

REPO = Path(__file__).resolve().parents[2]

CONFIRMED = {
    "event": "cancelled",
    "had_goal": True,
    "navigators": {
        "navigate_to_pose": {"outcome": "accepted", "cancelling": 1},
        "navigate_through_poses": {"outcome": "no such goal", "cancelling": 0},
    },
}
UNCONFIRMED = {
    "event": "cancelled",
    "had_goal": False,
    "navigators": {
        "navigate_to_pose": {"outcome": "no server answered"},
        "navigate_through_poses": {"outcome": "no server answered"},
    },
}


def _listener() -> socket.socket:
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(0.2)
    return listener


def _closed_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class GoalServer:
    """The goal server's socket: a cancel answers ``answer``, or nothing at all (``None``)."""

    def __init__(self, answer: dict[str, object] | None) -> None:
        self.answer, self.asked = answer, 0
        self._listener = _listener()
        self.port = int(self._listener.getsockname()[1])
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._listener.accept()
            except OSError:
                continue
            threading.Thread(target=self._serve, args=(connection,), daemon=True).start()

    def _serve(self, connection: socket.socket) -> None:
        with connection:
            connection.makefile("r").readline()
            self.asked += 1
            if self.answer is None:
                self._stop.wait(10.0)  # accepts, says nothing: a hung server
                return
            connection.sendall((json.dumps(self.answer) + "\n").encode())

    def close(self) -> None:
        self._stop.set()
        self._listener.close()


class BaseServer:
    """The board's base server as the red button meets it: a state line every 50 ms. The wheels
    turn until a stop arrives — and turn again at once while ``driver_alive`` says a cmd_vel
    producer still overwrites the stop. ``silent`` streams nothing at all."""

    def __init__(self, driver_alive: Callable[[], bool], silent: bool = False) -> None:
        self.driver_alive, self.silent = driver_alive, silent
        self.stops = 0
        self._listener = _listener()
        self.port = int(self._listener.getsockname()[1])
        self._stop = threading.Event()
        threading.Thread(target=self._run, daemon=True).start()

    def moving(self) -> bool:
        return self.stops == 0 or self.driver_alive()

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                connection, _ = self._listener.accept()
            except OSError:
                continue
            threading.Thread(target=self._serve, args=(connection,), daemon=True).start()

    def _serve(self, connection: socket.socket) -> None:
        connection.settimeout(0.05)
        buffer = b""
        with connection:
            while not self._stop.is_set():
                try:
                    chunk = connection.recv(4096)
                    if not chunk:
                        return
                    buffer += chunk
                    *lines, buffer = buffer.split(b"\n")
                    self.stops += sum(json.loads(line).get("cmd") == "stop" for line in lines)
                except TimeoutError:
                    pass
                except OSError:
                    return
                if not self.silent:
                    state = {"type": "state", "moving": self.moving()}
                    try:
                        connection.sendall((json.dumps(state) + "\n").encode())
                    except OSError:
                        return

    def close(self) -> None:
        self._stop.set()
        self._listener.close()


FAKE = """#!/bin/bash
printf '%s %s\\n' "$(basename "$0")" "$*" >> "$FAKE_LOG"
case "$(basename "$0") $*" in
    *"pkill -9 -f __node:=nav2_container"*) touch "$FAKE_KILLED" ;;
    "docker stop"*) sleep "${FAKE_DOCKER_STOP_S:-0}" ;;
esac
exit 0
"""


@pytest.fixture
def fakes(tmp_path: Path) -> Iterator[Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("docker", "ssh"):
        (bin_dir / name).write_text(FAKE)
        (bin_dir / name).chmod(0o755)
    yield tmp_path


def stop(
    tmp: Path, goal_port: int, base_port: int, **env: str
) -> tuple[int, str, list[str], float]:
    """Run ros/stop.sh; its status, output, the docker/ssh calls and how long it took."""
    log = tmp / "fakes.log"
    log.write_text("")
    started = time.monotonic()
    run = subprocess.run(
        ["bash", str(REPO / "ros/stop.sh")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,  # in the order the operator reads it
        text=True,
        timeout=60,
        env=os.environ
        | {
            "PATH": f"{tmp / 'bin'}:{os.environ['PATH']}",
            "PEPIN_HOST": "127.0.0.1",
            "PEPIN_GOAL_PORT": str(goal_port),
            "PEPIN_BASE_PORT": str(base_port),
            "PEPIN_STOP_BOUND_S": "2",
            "FAKE_LOG": str(log),
            "FAKE_KILLED": str(tmp / "killed"),
        }
        | env,
    )
    return (
        run.returncode,
        run.stdout,
        log.read_text().splitlines(),
        time.monotonic() - started,
    )


def test_a_confirmed_cancel_and_still_wheels_are_the_whole_stop(fakes: Path) -> None:
    goals, base = GoalServer(CONFIRMED), BaseServer(lambda: False)
    try:
        code, out, sent, _ = stop(fakes, goals.port, base.port)
    finally:
        goals.close()
        base.close()
    assert code == 0, out
    assert out.strip().splitlines()[-1].startswith("stopped: every goal cancelled and the wheels")
    assert base.stops == 2, "the base's stop at once, and the one that is believed"
    assert sent == [], "Nav2 stays up: nothing killed, nothing stopped"


@pytest.mark.slow  # a hung goal server is waited for its 3 s
@pytest.mark.parametrize("answer", ["unconfirmed", "closed", "silent"])
def test_a_cancel_not_confirmed_kills_the_producers_then_stops_the_base(
    fakes: Path, answer: str
) -> None:
    """The base's stop is not latched: a live controller overwrites it within 50 ms. So the
    producers go first — Nav2's composed container, the board's measured motion — and only the
    stop after them is believed; then Nav2's container is stopped."""
    base = BaseServer(lambda: not (fakes / "killed").exists())
    goals = (
        None if answer == "closed" else GoalServer(UNCONFIRMED if answer == "unconfirmed" else None)
    )
    try:
        code, out, sent, took = stop(fakes, goals.port if goals else _closed_port(), base.port)
    finally:
        base.close()
        if goals:
            goals.close()
    assert code == 0, out
    assert "killing what commands the wheels" in out
    assert out.strip().splitlines()[-1].startswith("hard stop: the wheels read still"), out
    kill = sent.index("docker exec pepin-macnav pkill -9 -f __node:=nav2_container")
    motion = next(i for i, c in enumerate(sent) if c.startswith("ssh") and "pepin_motion.pid" in c)
    down = sent.index("docker stop -t 30 pepin-macnav")
    assert kill < motion < down, sent
    assert base.stops == 3, "at once, after the cancel, after the kill"
    assert took < 12.0, took


def test_wheels_that_move_again_after_a_confirmed_cancel_go_the_hard_way(fakes: Path) -> None:
    """A confirmed cancel is not a stopped cart: a producer still alive (a measured motion, a
    stuck controller) commands the wheels again, the state stream says so, and it is killed."""
    goals, base = GoalServer(CONFIRMED), BaseServer(lambda: not (fakes / "killed").exists())
    try:
        code, out, sent, _ = stop(fakes, goals.port, base.port)
    finally:
        goals.close()
        base.close()
    assert code == 0, out
    assert "cancel confirmed, wheels NOT confirmed still" in out
    assert "docker exec pepin-macnav pkill -9 -f __node:=nav2_container" in sent


def test_a_base_that_cannot_be_reached_or_says_nothing_is_never_a_stopped_cart(
    fakes: Path,
) -> None:
    goals = GoalServer(CONFIRMED)
    silent = BaseServer(lambda: False, silent=True)
    try:
        code, out, _, _ = stop(fakes, goals.port, _closed_port())
        assert code == 1, out
        assert "no base server on 127.0.0.1" in out
        assert out.strip().splitlines()[-1].startswith("!! hard stop NOT confirmed")
        code, out, _, _ = stop(fakes, goals.port, silent.port)
        assert code == 1 and "0 state line(s) after the stop" in out, out
    finally:
        goals.close()
        silent.close()


@pytest.mark.slow  # the bound itself is waited for
def test_a_hung_docker_does_not_hold_the_red_button(fakes: Path) -> None:
    goals, base = GoalServer(UNCONFIRMED), BaseServer(lambda: False)
    try:
        code, out, _, took = stop(fakes, goals.port, base.port, FAKE_DOCKER_STOP_S="5")
    finally:
        goals.close()
        base.close()
    assert code == 0, out
    assert "pepin-macnav did not stop within 2 s" in out
    assert took < 10.0, took


def test_the_verdict_believes_only_the_last_state_lines() -> None:
    assert red_button.verdict([True, True, False, False, False])[0]
    assert not red_button.verdict([False, False, False, True])[0], "moving again: a live producer"
    assert not red_button.verdict([False, False])[0], "too few lines to believe"
    assert not red_button.verdict([])[0]
