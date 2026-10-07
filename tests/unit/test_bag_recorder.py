"""pepin_bringup.bag_recorder on its way out: SIGINT (the launch's stop) closes a bag still open,
and the context's shutdown is not the process's last word.

On 2026-09-23 the board's log showed the recorder dying with exit code 1 on a SIGINT:
``rclpy.shutdown()`` after ``if rclpy.ok()`` raised "rcl_shutdown already called on the given
context", because rclpy's own SIGINT handler had shut the context down from its thread in between.
rclpy is faked (``ros_stubs``); ``ros2 bag record`` is a fake process that records its signals.
"""

from __future__ import annotations

import json
import os
import signal
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import ros_stubs

RCLPY = ros_stubs.install()

from pepin_bringup import bag_recorder  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


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
        with ros_stubs.parameters(goal_bag="record"):  # ring is the default since 2026-10-06
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
    with ros_stubs.parameters(goal_bag="record"):  # ring is the default since 2026-10-06
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
    """The run's clip is the replay's camera (ros/tools/clip_to_bag.py): its URL asks ustreamer
    for the extra headers, the capture stamp beside the send stamp (pepin.mjpeg), on the board's
    address wherever the recorder runs."""
    from pepin_bringup.camera_clip import camera_stream

    url = camera_stream({"PEPIN_HOST": "10.0.0.187"}, REPO / "config/camera.json")
    assert url == "http://10.0.0.187:8080/stream?extra_headers=1"


# ---- the ring (flag goal_bag ring) -----------------------------------------------------------
@pytest.fixture
def ringed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[FakeBag]]:
    """A node under ``goal_bag ring`` in a tmp ring and rec; every process a :class:`FakeBag`."""
    started: list[FakeBag] = []

    def popen(command: list[str], **kwargs: Any) -> FakeBag:
        started.append(FakeBag(command, **kwargs))
        return started[-1]

    monkeypatch.setattr(bag_recorder.subprocess, "Popen", popen)
    with ros_stubs.parameters(
        record_dir=str(tmp_path / "rec"),
        ring_dir=str(tmp_path / "ring"),
        qos_overrides=str(tmp_path / "none"),
        goal_bag="ring",
        ring_floor_gb=1.0,
        preroll_s=3.0,
        tail_s=0.5,
    ):
        yield started


def _records(started: list[FakeBag]) -> list[FakeBag]:
    return [p for p in started if p.command[:3] == ["ros2", "bag", "record"]]


def test_the_ring_runs_from_the_start_with_the_vio_topics_and_a_goal_starts_no_recorder(
    ringed: list[FakeBag], tmp_path: Path
) -> None:
    """Under ring one recorder runs before any goal, split by the minute with the hidden topics
    and OpenVINS's outputs; a goal opens no second one and is RECORDING at once, and its end is
    IDLE at once with the window queued for the cutter."""
    node = bag_recorder.BagRecorderNode()
    rings = _records(ringed)
    assert len(rings) == 1
    command = rings[0].command
    assert "--max-bag-duration" in command and "--include-hidden-topics" in command
    assert command[-len(bag_recorder.RING_TOPICS) :] == list(bag_recorder.RING_TOPICS)
    assert {"/ov_msckf/poseimu", "/ov_msckf/odomimu"} <= set(bag_recorder.RING_TOPICS)
    assert set(bag_recorder.BAG_TOPICS) <= set(bag_recorder.RING_TOPICS)
    assert str(tmp_path / "ring") in command[command.index("--output") + 1]

    bag = node.start("home")
    assert len(_records(ringed)) == 1, "a goal starts no recorder of its own"
    assert node.recording and _statuses(node)[-1] == "recording"
    node.stop()
    assert _statuses(node)[-1] == "idle" and not node.recording
    job = node._jobs.get_nowait()
    assert job is not None and job.out == bag and job.window.duration_s >= 3.5
    assert job.window.end_s - job.goal_s >= 0.5


def test_the_mode_is_live_record_stops_the_ring_and_ring_starts_it(ringed: list[FakeBag]) -> None:
    node = bag_recorder.BagRecorderNode()
    ring = _records(ringed)[0]
    node._switches.set("goal_bag", "record")
    assert ring.signals == [signal.SIGINT], "closed on SIGINT: its file gets its summary"
    bag = node.start("printer")
    per_goal = _records(ringed)[-1]
    assert per_goal is not ring and str(bag) in per_goal.command, "the proven per-goal path"
    node.stop()
    node._switches.set("goal_bag", "ring")
    assert len(_records(ringed)) == 3 and node.ring.recording


def test_the_ring_knobs_reach_the_supervisor_live(ringed: list[FakeBag]) -> None:
    node = bag_recorder.BagRecorderNode()
    node._switches.set("ring_keep_gb", 2.5)
    node._switches.set("ring_keep_h", 0.5)
    node._switches.set("ring_floor_gb", 3.0)
    assert node.ring.cap_bytes == 2_500_000_000
    assert node.ring.keep_s == 1800.0
    assert node.ring.floor_bytes == 3_000_000_000


def test_a_run_is_cut_from_the_ring_with_the_statics_carried(
    ringed: list[FakeBag], tmp_path: Path
) -> None:
    """The cutter on a ring file written by the test: the run's directory appears whole with the
    window's messages, the statics heard on /tf_static carried to its start."""
    from mcap.reader import make_reader
    from mcap.writer import CompressionType, Writer

    from pepin.ring import SliceJob, Window

    node = bag_recorder.BagRecorderNode()
    node._latched.add(b"statics")
    now = 2_000_000_000.0
    path = tmp_path / "ring" / "20330518_033220Z" / "20330518_033220Z_0.mcap"  # now - 60 s
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        writer = Writer(handle, compression=CompressionType.NONE)
        writer.start(profile="ros2")
        schema = writer.register_schema("tf2_msgs/msg/TFMessage", "ros2msg", b"")
        tf = writer.register_channel("/tf_static", "cdr", schema, {})
        odom = writer.register_channel("/odom", "cdr", schema, {})
        writer.add_message(tf, int((now - 50) * 1e9), b"statics", int((now - 50) * 1e9))
        for i in range(100):
            t = int((now - 10 + i * 0.1) * 1e9)
            writer.add_message(odom, t, b"o", t)
        writer.finish()  # type: ignore[no-untyped-call]
    os.utime(path, (now, now))  # last written at `now`, like the ring's file at the cut
    out = tmp_path / "rec" / "0001_20330518_033320Z_home"
    node._closing.set()  # the ring is closed: cut what it holds, no waiting
    node.cut(SliceJob(out, Window.around(now - 5, now - 2, 1.0, 0.5), 1, now - 5))
    with (out / f"{out.name}_0.mcap").open("rb") as handle:
        got = [(c.topic, m.log_time) for _, c, m in make_reader(handle).iter_messages()]
    assert got[0] == ("/tf_static", int((now - 6) * 1e9))
    odom_times = [t for topic, t in got if topic == "/odom"]
    assert len(odom_times) == 46  # now-6 .. now-1.5 at 10 Hz
    assert (out / "metadata.yaml").is_file()
    assert any("from the ring" in line for line in node.logger.texts("info"))
