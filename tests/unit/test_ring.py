"""pepin.ring and the ring's half of pepin.board_bag: the files a split recording leaves, which of
them hold a window, when a window is on disk, the window itself, the latched store, and the
pruning by age, size and free space. Files are a tmp directory; no recorder runs."""

from __future__ import annotations

import calendar
import os
import time
from pathlib import Path

import pytest

from pepin.board_bag import SPLIT_S, Supervisor, prune, record_command
from pepin.ring import (
    LatchedStore,
    Window,
    bag_start,
    covering,
    segment_index,
    segments,
    written_past,
)

T0 = float(calendar.timegm(time.strptime("20261005_200000Z", "%Y%m%d_%H%M%SZ")))


def segment(root: Path, bag: str, index: int, mtime: float, size: int = 100) -> Path:
    """One file of a split recording, written last at ``mtime``."""
    path = root / bag / f"{bag}_{index}.mcap"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * size)
    os.utime(path, (mtime, mtime))
    return path


def test_rosbag2_split_names_order_by_number_not_by_text() -> None:
    assert segment_index(Path("/r/20261005_200000Z/20261005_200000Z_10.mcap")) == 10
    assert segment_index(Path("/r/x/metadata.yaml")) == -1
    assert bag_start("20261005_200000Z") == T0
    assert bag_start("not_a_stamp") is None


def test_segments_span_from_the_previous_write_and_the_first_from_the_directory_name(
    tmp_path: Path,
) -> None:
    """Files 0..11 of a recording, a minute each: ordered 0, 1, ..., 10, 11 (not 0, 1, 10, 11,
    2, ...), each spanning the minute since the file before it; the first from the stamp."""
    bag = "20261005_200000Z"
    for i in range(12):
        segment(tmp_path, bag, i, T0 + 60 * (i + 1))
    (tmp_path / bag / "metadata.yaml").write_text("not a segment")
    found = segments(tmp_path)
    assert [segment_index(s.path) for s in found] == list(range(12))
    assert found[0].start == T0 and found[0].end == T0 + 60
    assert all(s.end - s.start == 60 for s in found)


def test_a_respawned_recorder_starts_a_new_directory_after_the_old_one(tmp_path: Path) -> None:
    segment(tmp_path, "20261005_200000Z", 0, T0 + 60)
    segment(tmp_path, "20261005_200000Z", 1, T0 + 95)  # closed early: the recorder died
    segment(tmp_path, "20261005_200200Z", 0, T0 + 180)
    found = segments(tmp_path)
    assert [s.path.parent.name for s in found] == [
        "20261005_200000Z",
        "20261005_200000Z",
        "20261005_200200Z",
    ]
    assert found[2].start == T0 + 120  # its own stamp, not the dead recorder's last write


def test_a_window_takes_the_files_it_touches_and_the_one_before_for_the_latched_topics(
    tmp_path: Path,
) -> None:
    bag = "20261005_200000Z"
    for i in range(6):
        segment(tmp_path, bag, i, T0 + 60 * (i + 1))
    found = segments(tmp_path)
    # a goal from 2:10 to 3:20, 15 s preroll and 2 s tail: 1:55-3:22 is in files 1, 2 and 3
    # (file i spans i:00-(i+1):00), and file 0 is read for the latched topics
    chosen = covering(found, T0 + 130 - 15, T0 + 200 + 2)
    assert [segment_index(s.path) for s in chosen] == [0, 1, 2, 3]
    chosen = covering(found, T0 + 130, T0 + 200)
    assert [segment_index(s.path) for s in chosen] == [1, 2, 3]
    # inside the first file: nothing before it to look back into
    assert [segment_index(s.path) for s in covering(found, T0 + 10, T0 + 20)] == [0]
    assert covering(found, T0 + 1000, T0 + 1010) == []


def test_the_look_back_never_crosses_into_another_recording(tmp_path: Path) -> None:
    segment(tmp_path, "20261005_200000Z", 0, T0 + 60)
    segment(tmp_path, "20261005_200200Z", 0, T0 + 180)
    chosen = covering(segments(tmp_path), T0 + 150, T0 + 170)
    assert [s.path.parent.name for s in chosen] == ["20261005_200200Z"]


def test_a_window_is_on_disk_once_a_later_write_is(tmp_path: Path) -> None:
    segment(tmp_path, "20261005_200000Z", 0, T0 + 60)
    found = segments(tmp_path)
    assert written_past(found, T0 + 58)
    assert not written_past(found, T0 + 59.5)  # within the clock slack: not yet
    assert not written_past([], T0)


def test_the_window_math() -> None:
    w = Window.around(goal_s=1000.0, end_s=1030.5, preroll_s=15.0, tail_s=2.0)
    assert (w.start_ns, w.end_ns) == (985_000_000_000, 1_032_500_000_000)
    assert w.start_s == 985.0 and w.end_s == 1032.5 and w.duration_s == 47.5
    assert Window.around(5.0, 5.0, 0.0, 0.0).duration_s == 0.0
    with pytest.raises(ValueError, match="before it starts"):
        Window.around(10.0, 9.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="durations"):
        Window.around(10.0, 11.0, -1.0, 1.0)


def test_the_latched_store_keeps_distinct_messages_newest_last_and_bounded() -> None:
    store = LatchedStore(limit=3)
    for data in (b"a", b"b", b"a", b"c", b"d"):
        store.add(data)
    assert store.messages() == [b"a", b"c", b"d"]  # b fell off; a moved behind b when repeated


def test_files_older_than_the_age_limit_go_even_under_the_size_cap(tmp_path: Path) -> None:
    old = segment(tmp_path, "20261005_200000Z", 0, T0 + 60)
    mid = segment(tmp_path, "20261005_200000Z", 1, T0 + 120)
    new = segment(tmp_path, "20261005_200000Z", 2, T0 + 180)
    pruned = prune(
        tmp_path,
        cap_bytes=10**9,
        floor_bytes=0,
        free_bytes=lambda: 10**12,
        keep_s=90.0,
        now=T0 + 200,
    )
    assert pruned.deleted == 1 and not old.exists()
    assert mid.exists() and new.exists()


def test_the_age_limit_never_takes_the_file_being_written(tmp_path: Path) -> None:
    active = tmp_path / "20261005_200000Z"
    only = segment(tmp_path, active.name, 0, T0 + 60)
    pruned = prune(tmp_path, 10**9, 0, lambda: 10**12, active, keep_s=1.0, now=T0 + 10_000)
    assert pruned.deleted == 0 and only.exists()


def test_the_size_cap_still_binds_inside_the_age_limit(tmp_path: Path) -> None:
    files = [segment(tmp_path, "20261005_200000Z", i, T0 + 60 * (i + 1)) for i in range(4)]
    pruned = prune(
        tmp_path, cap_bytes=250, floor_bytes=0, free_bytes=lambda: 10**12, keep_s=1e6, now=T0
    )
    assert pruned.deleted == 2 and pruned.held_bytes == 200
    assert [f.exists() for f in files] == [False, False, True, True]


def test_the_ring_command_adds_the_hidden_topics_and_the_chunk_config() -> None:
    command = record_command(
        Path("/maps/ring/x"),
        Path("/params/rosbag_qos.yaml"),
        ("/a", "/b"),
        hidden=True,
        storage_config=Path("/params/ring_mcap.yaml"),
    )
    joined = " ".join(command)
    assert f"--max-bag-duration {SPLIT_S}" in joined
    assert "--include-hidden-topics" in joined
    assert "--storage-config-file /params/ring_mcap.yaml" in joined
    assert command[-2:] == ["/a", "/b"]
    plain = record_command(Path("/maps/board_rec/x"))
    assert "--include-hidden-topics" not in plain and "--storage-config-file" not in plain


def test_the_ring_supervisor_records_its_topics_and_prunes_by_age(tmp_path: Path) -> None:
    started: list[list[str]] = []

    class Proc:
        returncode: int | None = None

        def poll(self) -> int | None:
            return self.returncode

        def send_signal(self, sig: int) -> None:
            self.returncode = 0

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def terminate(self) -> None: ...

        def kill(self) -> None: ...

    def start(command: list[str]) -> Proc:  # type: ignore[misc]
        started.append(list(command))
        return Proc()

    stale = segment(tmp_path, "20261005_100000Z", 0, T0 - 7200)
    sup = Supervisor(
        tmp_path,
        cap_bytes=10**9,
        floor_bytes=0,
        qos_overrides=None,
        start=start,  # type: ignore[arg-type]
        free_bytes=lambda: 10**12,
        clock=lambda: T0,
        topics=("/x", "/ov_msckf/odomimu"),
        hidden=True,
        keep_s=3600.0,
        label="ring",
    )
    sup.step()
    assert not stale.exists() and not stale.parent.exists()
    assert started and started[0][-2:] == ["/x", "/ov_msckf/odomimu"]
    assert sup.active == tmp_path / "20261005_200000Z"
    sup.keep_s = 1.0  # a live knob reaches the next pass
    sup.cap_bytes = 0
    sup.step()
    assert len(started) == 1, "one recorder, kept"
