"""pepin_bringup.bag_recorder on its way out: SIGINT (the launch's stop) closes a bag still open,
and the context's shutdown is not the process's last word.

On 2026-09-23 the board's log showed the recorder dying with exit code 1 on a SIGINT:
``rclpy.shutdown()`` after ``if rclpy.ok()`` raised "rcl_shutdown already called on the given
context", because rclpy's own SIGINT handler had shut the context down from its thread in between.
rclpy is faked (``ros_stubs``); ``ros2 bag record`` is a fake process that records its signals.
"""

from __future__ import annotations

import json
import signal
from pathlib import Path
from typing import Any

import pytest
import ros_stubs

RCLPY = ros_stubs.install()

from pepin_bringup import bag_recorder  # noqa: E402


class FakeBag:
    """``ros2 bag record`` as the recorder drives it: the signals it got, and whether it ended."""

    def __init__(self, command: list[str], **_: Any) -> None:
        self.command = command
        self.signals: list[int] = []
        self.returncode: int | None = None

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.returncode = 0  # rosbag2 closes its file on SIGINT

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode or 0

    def terminate(self) -> None:
        self.signals.append(signal.SIGTERM)

    def kill(self) -> None:
        self.signals.append(signal.SIGKILL)


@pytest.fixture
def launched(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[FakeBag]:
    """Every process the recorder starts is a :class:`FakeBag` (the camera clip's curl too)."""
    started: list[FakeBag] = []

    def popen(command: list[str], **kwargs: Any) -> FakeBag:
        started.append(FakeBag(command, **kwargs))
        return started[-1]

    monkeypatch.setattr(bag_recorder.subprocess, "Popen", popen)
    with ros_stubs.parameters(record_dir=str(tmp_path), qos_overrides=str(tmp_path / "none")):
        yield started


def _statuses(node: Any) -> list[str]:
    return [json.loads(msg.data)["state"] for msg in node._status_pub.sent]


def test_sigint_mid_run_closes_the_bag_and_exits_cleanly_through_the_race(
    launched: list[FakeBag], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The spin ends with KeyboardInterrupt while a bag is open, and rclpy's handler has already
    shut the context down by the time the shutdown is asked for: the bag still gets its SIGINT
    (rosbag2 writes the summary a reader seeks by), and main() returns instead of raising."""
    nodes: list[Any] = []
    real = bag_recorder.BagRecorderNode

    def factory() -> Any:
        nodes.append(real())
        return nodes[-1]

    state = {"up": True}

    def the_handler_was_first() -> None:
        state["up"] = False
        raise RuntimeError("failed to shutdown: rcl_shutdown already called on the given context")

    def interrupted() -> None:
        nodes[0].start("printer")
        raise KeyboardInterrupt

    monkeypatch.setattr(bag_recorder, "BagRecorderNode", factory)
    monkeypatch.setattr(RCLPY, "try_shutdown", the_handler_was_first)
    monkeypatch.setattr(RCLPY, "ok", lambda: state["up"])
    RCLPY.on_spin = interrupted
    try:
        bag_recorder.main()
    finally:
        RCLPY.on_spin = None
    bag = next(p for p in launched if p.command[:3] == ["ros2", "bag", "record"])
    assert bag.signals == [signal.SIGINT], "closed on SIGINT, never killed"
    assert _statuses(nodes[0])[-1] == "idle"
    assert nodes[0].destroyed


def test_the_last_word_is_dropped_when_the_context_is_already_down(
    launched: list[FakeBag], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The status word after the bag closes has nobody to reach once the context is gone; that
    publish failing must not end the way out before the node is destroyed. With the context
    still up the same failure is real and is raised."""
    node = bag_recorder.BagRecorderNode()
    node.start("home")

    def dead(_msg: Any) -> None:
        raise RuntimeError("publisher's context is invalid")

    monkeypatch.setattr(node._status_pub, "publish", dead)
    monkeypatch.setattr(RCLPY, "ok", lambda: False)
    node.close()
    assert not node.recording
    monkeypatch.setattr(RCLPY, "ok", lambda: True)
    with pytest.raises(RuntimeError, match="context is invalid"):
        node.stop()


def test_the_camera_clip_asks_for_the_capture_stamps() -> None:
    """The board-side clip is the replay's camera (ros/tools/clip_to_bag.py): its URL asks
    ustreamer for the extra headers, the capture stamp beside the send stamp (pepin.mjpeg)."""
    from pepin_bringup.camera_clip import CAMERA_STREAM

    assert CAMERA_STREAM.endswith("/stream?extra_headers=1")
