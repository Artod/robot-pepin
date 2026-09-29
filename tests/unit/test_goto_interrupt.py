"""Ctrl-C on a drive: the goal is cancelled before anything else can go wrong.

On 2026-09-17 both rocking-chair legs ended like this (ros/maps/rec/20260917_192935_goto.log and
..._201425_goto.log): rclpy's own SIGINT handler had already shut the context down, the interrupt
handler's FIRST statement created a publisher for a note, that raised "rcl node's context is
invalid", and the ``cancelTask()`` on the next line never ran — the goal stayed alive on the board.
Tested here with fakes instead of on the robot: the order of the steps, and that a failing step
cannot swallow the ones after it.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from typing import Any

import pytest
import ros_stubs

ros_stubs.install()

REPO = Path(__file__).resolve().parents[2]


class FakeNavigator:
    """The two calls the interrupt path makes on nav2_simple_commander's BasicNavigator."""

    def __init__(self, *, cancel_raises: bool = False, complete_after: int = 1) -> None:
        self.calls: list[str] = []
        self._cancel_raises = cancel_raises
        self._complete_after = complete_after
        self._asked = 0
        self.published: list[str] = []

    def cancelTask(self) -> None:  # noqa: N802 - the name Nav2 gives it
        self.calls.append("cancel")
        if self._cancel_raises:
            raise RuntimeError("context is invalid")

    def isTaskComplete(self) -> bool:  # noqa: N802 - the name Nav2 gives it
        self._asked += 1
        return self._asked >= self._complete_after

    def create_publisher(self, _type: Any, topic: str, _depth: int) -> Any:
        self.calls.append(f"create_publisher {topic}")
        published = self.published

        class Pub:
            def publish(self, msg: Any) -> None:
                published.append(str(getattr(msg, "data", msg)))

        return Pub()

    def destroy_node(self) -> None:
        self.calls.append("destroy_node")


class DeadNavigator(FakeNavigator):
    """A navigator whose context SIGINT has already torn down: every call raises."""

    def create_publisher(self, _type: Any, _topic: str, _depth: int) -> Any:
        self.calls.append("create_publisher")
        raise RuntimeError("rcl node's context is invalid")


class FakeTape:
    """The run recorder's stop word, and whether it was sent."""

    def __init__(self, *, raises: bool = False) -> None:
        self.closed = 0
        self._raises = raises

    def close(self) -> None:
        self.closed += 1
        if self._raises:
            raise RuntimeError("publisher's context is invalid")


def load_goto() -> Any:
    """ros/tools/goto_ros.py imported by path, with the two modules it needs faked."""
    sys.modules.setdefault("nav2_simple_commander", types.ModuleType("nav2_simple_commander"))
    navigator = types.ModuleType("nav2_simple_commander.robot_navigator")

    class TaskResult:
        SUCCEEDED, CANCELED, FAILED = 1, 2, 3

    navigator.BasicNavigator = FakeNavigator  # type: ignore[attr-defined]
    navigator.TaskResult = TaskResult  # type: ignore[attr-defined]
    sys.modules["nav2_simple_commander.robot_navigator"] = navigator
    signals = types.ModuleType("rclpy.signals")

    class SignalHandlerOptions:
        NO = "no"
        ALL = "all"

    signals.SignalHandlerOptions = SignalHandlerOptions  # type: ignore[attr-defined]
    sys.modules["rclpy.signals"] = signals
    cancel = types.ModuleType("action_msgs.srv")

    class CancelGoal:
        class Request:  # a message stub: the cancel service's request
            pass

    cancel.CancelGoal = CancelGoal  # type: ignore[attr-defined]
    sys.modules.setdefault("action_msgs", types.ModuleType("action_msgs"))
    sys.modules["action_msgs.srv"] = cancel
    spec = importlib.util.spec_from_file_location("goto_ros", REPO / "ros/tools/goto_ros.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def goto() -> Any:
    """The goto client as a module (:func:`load_goto`)."""
    return load_goto()


@pytest.mark.slow
def test_the_cancel_goes_out_before_the_note_and_the_tape(goto: Any, capsys: Any) -> None:
    """The order IS the safety: whatever else fails, the board is told to stop first, and the
    note creates nothing on a context SIGINT may already have torn down."""
    nav, tape = DeadNavigator(), FakeTape()
    goto.interrupted(nav, tape)
    assert nav.calls[0] == "cancel", nav.calls
    assert tape.closed == 1
    assert "create_publisher" not in nav.calls, "the note prints; it publishes nothing"
    assert "interrupted by the operator" in capsys.readouterr().out, "the operator gets the words"


def test_a_tape_that_cannot_close_does_not_swallow_the_words(goto: Any, capsys: Any) -> None:
    """2026-09-17's lesson in its present shape: a step that raises costs itself and nothing
    else."""
    nav, tape = FakeNavigator(), FakeTape(raises=True)
    goto.interrupted(nav, tape)
    assert nav.calls[0] == "cancel" and tape.closed == 1
    assert "interrupted by the operator" in capsys.readouterr().out


def test_a_cancel_that_fails_says_so_loudly(goto: Any, capsys: Any) -> None:
    """If the one step that matters fails, the operator is told to run ros/stop.sh — and the other
    steps still run."""
    nav, tape = FakeNavigator(cancel_raises=True), FakeTape()
    goto.interrupted(nav, tape)
    out = capsys.readouterr().out
    assert "cancel NOT sent" in out and "ros/stop.sh" in out
    assert tape.closed == 1


def test_guarded_runs_the_next_step_after_a_failure(goto: Any) -> None:
    """The primitive the shutdown is built of: one failure, one message, no exception."""
    done: list[str] = []
    assert goto.guarded("boom", lambda: (_ for _ in ()).throw(RuntimeError("no"))) is False
    assert goto.guarded("fine", lambda: done.append("ran")) is True
    assert done == ["ran"]


def test_rclpy_is_told_not_to_take_sigint(goto: Any) -> None:
    """The fix that makes the rest possible: with rclpy's handler installed, the context is gone
    before the first line of the handler runs."""
    source = (REPO / "ros/tools/goto_ros.py").read_text()
    assert "signal_handler_options=SignalHandlerOptions.NO" in source
    assert "rclpy.init()" not in source
