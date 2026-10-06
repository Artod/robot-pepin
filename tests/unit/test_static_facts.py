"""StaticWait's warnings and the preflight reading of them (pepin.static_facts)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from pepin import static_facts
from pepin.static_facts import DEBUG_RUST_LOG, StaticWait, verdict, waiting_nodes

EDGES = (("camera_link", "camera_optical"), ("camera_optical", "head_imu"))


class Buffer:
    """A fake TF buffer: the edges it holds."""

    def __init__(self) -> None:
        self.edges: set[tuple[str, str]] = set()

    def present(self, parent: str, child: str) -> bool:
        return (parent, child) in self.edges


def test_quiet_before_the_patience_then_warns_naming_edges_cause_and_recipe() -> None:
    buf = Buffer()
    wait = StaticWait(EDGES, started=100.0, after_s=15.0)
    assert wait.update(101.0, buf.present) is None
    assert wait.update(114.9, buf.present) is None
    level, text = wait.update(115.0, buf.present) or ("", "")
    assert level == "warn"
    assert (
        "WAITING 15 s for 2 static edge(s): camera_link->camera_optical, camera_optical->head_imu"
        in text
    )
    assert "silent zenoh peer" in text
    assert f"RUST_LOG={DEBUG_RUST_LOG}" in text
    assert "PEPIN_ZENOH_DEBUG=1" in text


def test_repeats_every_four_patiences_and_names_only_what_is_still_missing() -> None:
    buf = Buffer()
    wait = StaticWait(EDGES, started=0.0, after_s=15.0)
    assert wait.update(15.0, buf.present) is not None
    assert wait.update(30.0, buf.present) is None
    buf.edges.add(EDGES[0])
    assert wait.update(74.0, buf.present) is None
    level, text = wait.update(75.0, buf.present) or ("", "")
    assert level == "warn" and "1 static edge(s): camera_optical->head_imu." in text
    assert wait.clause(80.0) == "tf_static WAITING 80 s for camera_optical->head_imu"


def test_one_info_when_complete_then_silent() -> None:
    buf = Buffer()
    wait = StaticWait(EDGES, started=0.0)
    buf.edges.update(EDGES)
    assert wait.update(0.7, buf.present) == (
        "info",
        "tf_static: complete, 2 static edge(s) after 0.7 s",
    )
    assert wait.done
    assert wait.update(100.0, buf.present) is None
    assert wait.clause(100.0) == "tf_static ok"


def test_live_knob_changes_the_patience() -> None:
    buf = Buffer()
    wait = StaticWait(EDGES, started=0.0, after_s=15.0)
    assert wait.update(5.0, buf.present, after_s=4.0) is not None


def rclpy_line(level: str, node: str, text: str) -> str:
    return f"[python3-2] [{level}] [1791268325.123456789] [{node}]: {text}"


def test_preflight_counts_nodes_whose_last_word_is_waiting() -> None:
    wait = StaticWait(EDGES, started=0.0)
    warn = wait.warning(42.0)
    lines = [
        rclpy_line("WARN", "depth_fusion", warn),
        rclpy_line("WARN", "marks_audit", warn),
        rclpy_line("INFO", "marks_audit", "tf_static: complete, 1 static edge(s) after 47.0 s"),
        rclpy_line("INFO", "depth_stream", "tf_static: complete, 1 static edge(s) after 0.4 s"),
        rclpy_line("INFO", "gaze", "something else entirely"),
    ]
    waiting = waiting_nodes(lines)
    assert list(waiting) == ["depth_fusion"]
    line = verdict(waiting)
    assert line.startswith(
        "FAIL 1 node(s) waiting for /tf_static: depth_fusion 42 s "
        "(camera_link->camera_optical, camera_optical->head_imu)"
    )
    assert verdict({}) == "OK no node waiting for /tf_static"


def test_main_reads_stdin_and_exits_one_on_fail(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    text = rclpy_line("WARN", "depth_fusion", StaticWait(EDGES, started=0.0).warning(20.0))
    monkeypatch.setattr("sys.stdin", io.StringIO(text + "\n"))
    assert static_facts.main([]) == 1
    assert capsys.readouterr().out.startswith("FAIL 1 node(s)")
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert static_facts.main([]) == 0


Q = "zenoh::net::routing::dispatcher::queries:"
T = "zenoh::net::routing::dispatcher::token:"


def test_silent_peers_names_a_session_that_answers_late_or_never_by_its_node_tokens() -> None:
    lines = [
        f"2026-10-06T06:39:00.1Z DEBUG {T} Face{{3, 91d6aa}} Declare token 5 "
        "(@ros2_lv/7/91d6aa/0/0/NN/%/%/tof_bridge)",
        f"2026-10-06T06:39:01.0Z TRACE {Q} Face{{1, 4da8}}:0 Propagate query to "
        "Face{3, 91d6aa}:1",
        f"2026-10-06T06:39:01.0Z TRACE {Q} Face{{1, 4da8}}:0 Propagate query to Face{{2, 65cb}}:1",
        f"2026-10-06T06:39:01.2Z DEBUG {Q} Face{{2, 65cb}}:1 Received final reply for query "
        "Face{1, 4da8}:0",
        f"2026-10-06T06:39:02.0Z TRACE {Q} Face{{1, 4da8}}:1 Propagate query to Face{{4, f680}}:7",
        f"2026-10-06T06:39:31.0Z DEBUG {Q} Face{{4, f680}}:7 Received final reply for query "
        "Face{1, 4da8}:1",
    ]
    rows = static_facts.silent_peers(lines)
    assert rows == [("91d6aa", 1, None, ["tof_bridge"]), ("f680", 1, 29.0, [])]
    assert static_facts.silent_text(rows).splitlines() == [
        "SILENT 91d6aa (tof_bridge): 1 query never answered",
        "SILENT f680 (no ROS node token in this log: a router?): 1 query answered 29.0 s late "
        "at worst",
    ]
    assert static_facts.silent_text([]).startswith("OK")


def test_main_silent_flag(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    assert static_facts.main(["--silent"]) == 0
    assert capsys.readouterr().out.startswith("OK every query")


def test_the_laptop_script_carries_the_same_debug_recipe() -> None:
    script = (Path(__file__).parents[2] / "ros" / "laptop.sh").read_text()
    assert f'PEPIN_ZENOH_DEBUG_LOG="{DEBUG_RUST_LOG}"' in script
