import math
from pathlib import Path

from pepin.odometry import Pose2D
from pepin.recording import pose_from_record, read_session, scan_from_record, scan_record_from_ros
from pepin.tape import RunTape


def test_a_tape_round_trips_poses_and_scans(tmp_path: Path) -> None:
    """What the run recorder writes (pepin.tape.RunTape) reads back as the poses and scans the
    offline tools rebuild (scripts/build_map.py)."""
    path = tmp_path / "0001_test.jsonl"
    tape = RunTape(opener=lambda p: p.open("w"), sync=lambda _s: None)
    tape.start(path)
    tape.add({"t": 12.0, "topic": "pose", "x": 1.0, "y": 2.0, "theta": 0.5, "d_right": 0.02})
    tape.add(
        scan_record_from_ros(
            12.5, 0.0, 1.0, [0.75, math.nan], [200.0, 0.0], 0.05, 12.0, 0.1,
            mount_yaw_rad=0.0, mount_x_m=0.0,
        )
    )  # fmt: skip
    tape.stop()

    records = list(read_session(path))
    assert [r["topic"] for r in records] == ["pose", "scan"]
    assert pose_from_record(records[0]) == Pose2D(1.0, 2.0, 0.5)
    assert records[0]["d_right"] == 0.02
    back = scan_from_record(records[1])
    assert back.stamp == 12.5 and back.ranges[0] == 0.75 and math.isnan(back.ranges[1])
    assert back.intensities.tolist() == [200, 0]
