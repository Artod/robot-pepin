"""TfLookup.expect_static: the /tf_static watch on a node's logger (pepin.static_facts glue)."""

from __future__ import annotations

from typing import Any

import pytest
import ros_stubs

RCLPY = ros_stubs.install()

from pepin_bringup import node_kit  # noqa: E402
from pepin_bringup.node_kit import TfLookup  # noqa: E402


class LookupException(Exception):  # noqa: N818 — tf2's own name
    pass


class StaticBuffer:
    """A TF buffer that holds some static edges; can_transform is what the watch asks."""

    def __init__(self) -> None:
        self.edges: set[tuple[str, str]] = set()

    def can_transform(self, parent: str, child: str, _time: Any) -> bool:
        if (parent, child) == ("boom", "boom"):
            raise LookupException("tf2 says no")
        return (parent, child) in self.edges


class Timer:
    def __init__(self, callback: Any) -> None:
        self.callback = callback
        self.cancelled = False

    def cancel(self) -> None:
        self.cancelled = True


class Node:
    """What the watch asks of a node: a timer and a logger."""

    def __init__(self) -> None:
        self.timers: list[tuple[float, Timer]] = []
        self.lines: list[tuple[str, str]] = []

    def create_timer(self, period_s: float, callback: Any) -> Timer:
        timer = Timer(callback)
        self.timers.append((period_s, timer))
        return timer

    def get_logger(self) -> Any:
        lines = self.lines

        class Logger:
            def info(self, text: str) -> None:
                lines.append(("info", text))

            def warning(self, text: str) -> None:
                lines.append(("warn", text))

        return Logger()


def test_expect_static_warns_once_due_then_infos_and_stops(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = [1000.0]
    monkeypatch.setattr(node_kit.time, "monotonic", lambda: clock[0])
    buffer = StaticBuffer()
    node = Node()
    tf = TfLookup(node, buffer=buffer)
    wait = tf.expect_static(node, [("camera_link", "camera_optical")], lambda: 15.0)
    assert tf.static_wait is wait
    ((period, timer),) = node.timers
    assert period == 1.0
    clock[0] = 1010.0
    timer.callback()
    assert node.lines == []
    clock[0] = 1015.0
    timer.callback()
    ((level, text),) = node.lines
    assert level == "warn" and "tf_static: WAITING 15 s" in text
    assert "camera_link->camera_optical" in text and "silent zenoh peer" in text
    buffer.edges.add(("camera_link", "camera_optical"))
    clock[0] = 1016.0
    timer.callback()
    assert node.lines[-1] == ("info", "tf_static: complete, 1 static edge(s) after 16.0 s")
    assert timer.cancelled


def test_has_edge_is_false_on_a_tf2_error() -> None:
    tf = TfLookup(Node(), buffer=StaticBuffer())
    assert tf.has_edge("boom", "boom") is False
    assert tf.has_edge("a", "b") is False


def test_camera_static_edges_follow_the_config_or_fall_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def broken() -> Any:
        raise OSError("no config/camera.json here")

    monkeypatch.setattr(node_kit, "load_camera_mounts", broken)
    assert node_kit.camera_static_edges() == [("camera_link", "camera_optical")]
    mounts = type("M", (), {"link_frame": "eye_link", "optical_frame": "eye_optical"})()
    monkeypatch.setattr(node_kit, "load_camera_mounts", lambda: mounts)
    assert node_kit.camera_static_edges() == [("eye_link", "eye_optical")]
