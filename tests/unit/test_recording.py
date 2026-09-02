import math
from pathlib import Path

import numpy as np

from pepin.kinematics import Twist
from pepin.lidar import LaserScan
from pepin.odometry import Pose2D
from pepin.recording import SessionRecorder, pose_from_record, read_session, scan_from_record


def test_session_round_trips_poses_commands_and_scans(tmp_path: Path) -> None:
    scan = LaserScan(
        stamp=12.5,
        angles=np.array([0.0, 1.0]),
        ranges=np.array([0.75, math.nan]),
        intensities=np.array([200, 0]),
        speed_rps=10.0,
    )
    with SessionRecorder(tmp_path, "test") as rec:
        rec.pose(Pose2D(1.0, 2.0, 0.5), travel=(0.01, 0.02))
        rec.command(Twist(0.1, -0.2))
        rec.scan(scan)
        rec.note("hello")
        assert rec.records == 4
    files = list(tmp_path.glob("*_test.jsonl"))
    assert len(files) == 1

    records = list(read_session(files[0]))
    assert [r["topic"] for r in records] == ["pose", "cmd", "scan", "note"]
    assert pose_from_record(records[0]) == Pose2D(1.0, 2.0, 0.5)
    assert records[0]["d_right"] == 0.02
    back = scan_from_record(records[2])
    assert back.stamp == 12.5 and back.ranges[0] == 0.75 and math.isnan(back.ranges[1])
    assert back.intensities.tolist() == [200, 0]


def test_timestamps_default_to_monotonic_and_increase(tmp_path: Path) -> None:
    with SessionRecorder(tmp_path) as rec:
        rec.note("a")
        rec.note("b")
    t = [r["t"] for r in read_session(rec.path)]
    assert t[1] >= t[0] > 0
