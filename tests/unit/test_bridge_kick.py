"""The board's half of the bridge kick: one message from the laptop becomes one flag file.

rclpy is faked (``ros_stubs``); the file is a real one in a tmp directory, because what the
board's systemd path unit watches is exactly that file appearing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import ros_stubs

ros_stubs.install()

from pepin_bringup.bridge_kick import BridgeKick  # noqa: E402
from ros_stubs import Node, String  # noqa: E402

from pepin.deployment import BRIDGE_KICK_FLAG, BRIDGE_KICK_TOPIC, bridged_qos  # noqa: E402


class Clock:
    """A monotonic clock the test moves by hand."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def build(tmp_path: Path, enabled: bool = True) -> tuple[Node, BridgeKick, Clock, Path, Any]:
    node, clock, flag = Node("run_recorder"), Clock(), tmp_path / "run" / "bridge_kick"
    kick = BridgeKick(node, flag=flag, enabled=lambda: enabled, clock=clock)
    return node, kick, clock, flag, node.subs[f"/{BRIDGE_KICK_TOPIC}"][1]


def test_a_kick_writes_the_flag_file_the_board_s_systemd_watches(tmp_path: Path) -> None:
    node, kick, _clock, flag, on_kick = build(tmp_path)
    on_kick(String(data="/scan has a route with no reader"))
    assert kick.kicks == 1 and flag.is_file()
    written = flag.read_text()
    assert "/scan has a route with no reader" in written, "the board's journal says who asked"
    assert written.endswith("\n") and "Z " in written, "and when, in UTC"
    assert any("bridge kick" in line for line in node.get_logger().texts("error"))


def test_two_kicks_inside_the_cooldown_are_one(tmp_path: Path) -> None:
    """The board's bridge needs ~25 s to come back; the laptop may ask again long before that."""
    _node, kick, clock, flag, on_kick = build(tmp_path)
    on_kick(String(data="first"))
    flag.unlink()  # the board's script deletes it as its first act
    clock.now += 30.0
    on_kick(String(data="second"))
    assert kick.kicks == 1 and not flag.exists(), "ignored: the first restart is still happening"
    clock.now += 100.0
    on_kick(String(data="third"))
    assert kick.kicks == 2 and flag.is_file()


def test_the_switch_off_logs_the_request_and_touches_nothing(tmp_path: Path) -> None:
    node, kick, _clock, flag, on_kick = build(tmp_path, enabled=False)
    on_kick(String(data="whatever"))
    assert kick.kicks == 0 and not flag.exists()
    assert any("bridge_kick is off" in line for line in node.get_logger().texts("warning"))


def test_a_file_that_cannot_be_written_is_said_and_never_raises(tmp_path: Path) -> None:
    """The mount may be missing on a board that has not been redeployed: the node must live."""
    node, kick, _clock, _flag, on_kick = build(tmp_path)
    blocked = tmp_path / "blocked"
    blocked.write_text("not a directory")
    kick._flag = blocked / "bridge_kick"
    on_kick(String(data="/scan"))
    assert kick.kicks == 0
    assert any("could not be written" in line for line in node.get_logger().texts("error"))


def test_a_container_without_the_board_s_mount_says_so_at_start(tmp_path: Path) -> None:
    """Without the bind mount the write succeeds into the container's own filesystem and the
    board never hears a thing: the one failure of this path that is otherwise silent."""
    from pepin_bringup.bridge_kick import mounted

    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text("31 24 0:29 / /run rw shared:5 - tmpfs tmpfs rw\n")
    assert mounted(Path("/run/pepin"), mountinfo) is False
    mountinfo.write_text(
        "31 24 0:29 / /run rw shared:5 - tmpfs tmpfs rw\n"
        "42 31 0:29 /pepin /run/pepin rw shared:5 - tmpfs tmpfs rw\n"
    )
    assert mounted(Path("/run/pepin"), mountinfo) is True
    assert mounted(Path("/run/pepin"), tmp_path / "no-proc-here") is None, "not Linux: no verdict"

    node = Node("run_recorder")
    BridgeKick(node, flag=tmp_path / "run" / "bridge_kick")
    said = [line for line in node.get_logger().texts("warning") if "not mounted" in line]
    assert said == [], "no /proc on this laptop: the watch keeps quiet rather than crying wolf"


def test_the_topic_and_its_qos_are_the_ones_both_sides_are_pinned_to(tmp_path: Path) -> None:
    """A kick is sent once, while the link is already sick: it may not lose a QoS race."""
    node, _kick, _clock, _flag, _on_kick = build(tmp_path)
    kind, depth = bridged_qos(BRIDGE_KICK_TOPIC) or ("", 0)
    assert (kind, depth) == ("reliable", 5)
    assert BRIDGE_KICK_FLAG == "/run/pepin/bridge_kick", "ros/run.sh mounts it, systemd watches it"
    assert f"/{BRIDGE_KICK_TOPIC}" in node.subs
