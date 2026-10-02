"""pepin.board_bag: the board's long-lived raw-sensor recorder and the cap that keeps the card from
filling. The recorder is a fake process; the card is a tmp directory and a free-space function."""

from __future__ import annotations

import json
import os
import signal
from collections.abc import Sequence
from pathlib import Path

import pytest

from pepin import board_bag
from pepin.board_bag import Supervisor, prune, record_command


def minute(root: Path, bag: str, index: int, size: int, mtime: float) -> Path:
    """One minute file of a recording, ``size`` bytes, last written at ``mtime``."""
    path = root / bag / f"{bag}_{index}.mcap"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


class FakeRecorder:
    """``ros2 bag record`` as the supervisor drives it: the command, the signals, its end."""

    def __init__(self, command: Sequence[str]) -> None:
        self.command = list(command)
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


def test_the_command_records_the_listed_topics_in_minute_mcap_files() -> None:
    """MCAP, a file a minute, the cache bounded, the latched statics with their QoS; never a
    camera topic and never /tf (it would pull the laptop's map -> odom over the WiFi)."""
    command = record_command(Path("/maps/board_rec/x"), Path("/params/rosbag_qos.yaml"))
    assert command[:3] == ["ros2", "bag", "record"]
    joined = " ".join(command)
    assert "--storage mcap" in joined and "--max-bag-duration 60" in joined
    assert "--qos-profile-overrides-path /params/rosbag_qos.yaml" in joined
    assert command[-len(board_bag.TOPICS) :] == list(board_bag.TOPICS)
    for topic in ("/scan", "/odom", "/imu/data_raw", "/tof/front/scan", "/neck/state"):
        assert topic in board_bag.TOPICS
    assert "/tf" not in board_bag.TOPICS
    assert not [t for t in board_bag.TOPICS if "camera" in t or "image" in t]


def test_the_qos_file_holds_the_latched_statics_the_recorder_asks_for() -> None:
    """/tf_static is published once: without the override a recorder that starts after the
    statics hears none of them."""
    qos = Path(__file__).resolve().parents[2] / "ros/params/rosbag_qos.yaml"
    assert "/tf_static:" in qos.read_text()


def test_the_oldest_minutes_go_first_until_the_directory_is_under_the_cap(tmp_path: Path) -> None:
    old = minute(tmp_path, "20261001_100000Z", 0, 100, 1000.0)
    mid = minute(tmp_path, "20261001_100000Z", 1, 100, 1060.0)
    new = minute(tmp_path, "20261001_120000Z", 0, 100, 2000.0)
    pruned = prune(tmp_path, cap_bytes=150, floor_bytes=0, free_bytes=lambda: 10**12)
    assert (pruned.deleted, pruned.held_bytes) == (2, 100)
    assert not old.exists() and not mid.exists() and new.exists()
    assert not (tmp_path / "20261001_100000Z").exists(), "an emptied recording goes with it"


def test_the_file_being_written_is_never_deleted(tmp_path: Path) -> None:
    """Over the cap with only the active recording left: its finished minutes go, its open one
    stays."""
    active = tmp_path / "20261001_120000Z"
    done = minute(tmp_path, active.name, 0, 100, 1000.0)
    writing = minute(tmp_path, active.name, 1, 100, 1060.0)
    pruned = prune(tmp_path, cap_bytes=0, floor_bytes=0, free_bytes=lambda: 10**12, active=active)
    assert pruned.deleted == 1 and not done.exists() and writing.exists() and active.is_dir()


def test_a_card_under_the_floor_is_pruned_however_small_the_directory(tmp_path: Path) -> None:
    """The floor bites first when something else fills the card: minutes go until it is met."""
    files = [minute(tmp_path, "b", i, 100, 1000.0 + i) for i in range(4)]
    card = {"free": 150}

    def free() -> int:
        return card["free"] + sum(100 for f in files if not f.exists())

    pruned = prune(tmp_path, cap_bytes=10**9, floor_bytes=300, free_bytes=free)
    assert pruned.deleted == 2 and pruned.free_bytes == 350
    assert [f.exists() for f in files] == [False, False, True, True]


def test_the_supervisor_starts_one_recorder_and_keeps_it(tmp_path: Path) -> None:
    started: list[FakeRecorder] = []

    def start(command: Sequence[str]) -> FakeRecorder:
        started.append(FakeRecorder(command))
        return started[-1]

    sup = Supervisor(
        tmp_path / "rec",
        cap_bytes=10**9,
        floor_bytes=0,
        qos_overrides=None,
        start=start,
        free_bytes=lambda: 10**12,
        clock=lambda: 1790911800.0,
    )
    sup.step()
    sup.step()
    assert len(started) == 1 and sup.recording
    assert sup.active == tmp_path / "rec" / "20261002_033000Z"
    assert started[0].command[started[0].command.index("--output") + 1] == str(sup.active)


def test_a_recorder_that_dies_is_replaced_on_a_new_directory(tmp_path: Path) -> None:
    started: list[FakeRecorder] = []
    now = [1000.0]

    def start(command: Sequence[str]) -> FakeRecorder:
        started.append(FakeRecorder(command))
        return started[-1]

    sup = Supervisor(
        tmp_path,
        cap_bytes=10**9,
        floor_bytes=0,
        qos_overrides=None,
        start=start,
        free_bytes=lambda: 10**12,
        clock=lambda: now[0],
    )
    sup.step()
    started[0].returncode = 1
    now[0] += 120.0
    sup.step()
    assert len(started) == 2 and started[0].command != started[1].command


def test_no_room_stops_the_recorder_with_sigint_and_room_starts_it_again(tmp_path: Path) -> None:
    """Nothing left to delete and the card still under the floor: the recorder is closed (SIGINT,
    so the file gets its summary) and not restarted until the card has room."""
    started: list[FakeRecorder] = []
    card = {"free": 10**12}

    def start(command: Sequence[str]) -> FakeRecorder:
        started.append(FakeRecorder(command))
        return started[-1]

    sup = Supervisor(
        tmp_path,
        cap_bytes=10**9,
        floor_bytes=1000,
        qos_overrides=None,
        start=start,
        free_bytes=lambda: card["free"],
    )
    sup.step()
    card["free"] = 10
    sup.step()
    sup.step()
    assert started[0].signals == [signal.SIGINT] and not sup.recording and len(started) == 1
    card["free"] = 10**12
    sup.step()
    assert len(started) == 2 and sup.recording


@pytest.mark.parametrize("hangs", [False, True])
def test_stop_closes_the_file_and_escalates_only_past_the_timeout(
    tmp_path: Path, hangs: bool
) -> None:
    recorder = FakeRecorder([])
    if hangs:

        def wait(timeout: float | None = None) -> int:
            raise board_bag.subprocess.TimeoutExpired("ros2", timeout or 0.0)

        recorder.wait = wait  # type: ignore[method-assign]
        recorder.send_signal = recorder.signals.append  # type: ignore[method-assign]
    sup = Supervisor(
        tmp_path,
        cap_bytes=1,
        floor_bytes=0,
        qos_overrides=None,
        start=lambda _: recorder,
        free_bytes=lambda: 10**12,
    )
    sup.step()
    sup.stop()
    expected = [signal.SIGINT, signal.SIGTERM, signal.SIGKILL] if hangs else [signal.SIGINT]
    assert recorder.signals == expected


def test_the_switch_reaches_the_launch_from_the_unit_and_feature_sh_and_defaults_off() -> None:
    """Off by default everywhere it is declared; ros/feature.sh board_bag flips the unit's
    variable; the manifest budgets both processes; the QoS file the recorder reads is synced."""
    repo = Path(__file__).resolve().parents[2]
    robot = (repo / "ros/pepin_bringup/launch/robot.launch.py").read_text()
    assert 'DeclareLaunchArgument("board_bag", default_value="false")' in robot
    assert '"pepin.board_bag"' in robot and 'IfCondition(LaunchConfiguration("board_bag"))' in robot
    bringup = (repo / "ros/pepin_bringup/launch/bringup.launch.py").read_text()
    assert '"board_bag": LaunchConfiguration("board_bag")' in bringup
    assert 'DeclareLaunchArgument("board_bag", default_value="false")' in bringup
    unit = (repo / "board/pepin-ros.service").read_text()
    assert "Environment=PEPIN_BOARD_BAG=false" in unit
    assert "board_bag:=${PEPIN_BOARD_BAG}" in unit
    assert "board_bag) VAR=PEPIN_BOARD_BAG ;;" in (repo / "ros/feature.sh").read_text()
    assert "--include 'params/rosbag_qos.yaml'" in (repo / "ros/sync.sh").read_text()
    entries = json.loads((repo / "config/board_manifest.json").read_text())["processes"]
    by_name = {entry["name"]: entry for entry in entries}
    assert by_name["board_bag"]["when"] == "sometimes"
    assert by_name["bag_record"]["budget"]["cpu_percent"] > 0
