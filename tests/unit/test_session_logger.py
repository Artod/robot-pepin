"""What one drive writes down. The tape is the deliverable — every question about a run is
answered off it days later — so the shape of each record is a contract, tested here with rclpy
faked (ros_stubs) instead of on the robot.
"""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import pytest
import ros_stubs

ros_stubs.install()

from sensor_msgs.msg import LaserScan  # noqa: E402
from std_msgs.msg import Header, String  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
MEASUREMENT = (
    '{"x": 1.5, "y": -0.25, "yaw": 0.1, "covariance": [[0.01, 0, 0], [0, 0.01, 0],'
    ' [0, 0, 0.001]], "source": "depth", "stamp": 100.25, "fit": 0.34, "map": "flat3"}'
)


def module() -> Any:
    """ros/tools/session_logger.py imported by path: it is a tool, not a package."""
    spec = importlib.util.spec_from_file_location(
        "session_logger", REPO / "ros/tools/session_logger.py"
    )
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


def tape(node: Any) -> list[dict[str, Any]]:
    """Everything the logger has written so far, parsed."""
    node._file.flush()
    return [json.loads(line) for line in Path(node._file.name).read_text().splitlines()]


def scan() -> LaserScan:
    """A camera's virtual scan with a hole in it: 1 m, nothing, 2.5 m."""
    return LaserScan(
        header=Header(),
        angle_min=-0.7,
        angle_increment=0.01,
        ranges=[1.0, float("inf"), 2.5],
    )


@pytest.fixture
def logger(tmp_path: Path) -> Any:
    """A recorder writing to a file of its own, with the camera scans off (the default)."""
    return module().SessionLogger(str(tmp_path / "run.jsonl"))


def test_the_camera_s_word_is_taped_verbatim(logger: Any) -> None:
    """The recorder parses nothing: a measurement is on the tape exactly as it arrived, so a
    malformed message is evidence instead of a hole, and ``t`` is when it arrived here."""
    logger.subs["/localization/measurement"][1](String(data=MEASUREMENT))
    logger.subs["/localization/measurement"][1](String(data="not json at all"))
    records = tape(logger)
    assert [r["topic"] for r in records] == ["meas", "meas"]
    assert json.loads(records[0]["json"])["source"] == "depth"
    assert json.loads(records[0]["json"])["stamp"] == 100.25
    assert records[1]["json"] == "not json at all"
    assert records[0]["t"] > 1e9, "the arrival wall clock, beside the stamp inside the message"
    assert logger.measurements == 2


def test_the_tracker_s_account_of_each_update_is_taped(logger: Any) -> None:
    """/localization/sources is what tells a replay who anchored, what was fused and what each
    source claimed — including the self-check ratio."""
    report = '{"anchor": "lidar", "fused": "lidar+camera", "sources": {"lidar": {"fit": 0.7}}}'
    logger.subs["/localization/sources"][1](String(data=report))
    record = tape(logger)[0]
    assert record["topic"] == "srcs" and json.loads(record["json"])["anchor"] == "lidar"


def test_the_camera_s_scans_are_off_unless_asked_for(logger: Any) -> None:
    """The board carries what is real-time critical: /contact_scan has no other consumer there,
    so taping it opens a bridge route and is the owner's switch, not the recorder's habit."""
    assert "/depth_scan" not in logger.subs and "/contact_scan" not in logger.subs


def test_asked_for_they_are_taped_compactly(tmp_path: Path) -> None:
    """One record per scan: the first angle, the step and the ranges in millimetres, with the
    infinities as null — the shape a replay already reads the lidar's returns in."""
    logger = module().SessionLogger(str(tmp_path / "run.jsonl"), camera_scans=True)
    logger.subs["/depth_scan"][1](scan())
    logger.subs["/contact_scan"][1](scan())
    records = tape(logger)
    assert [r["topic"] for r in records] == ["depth_scan", "contact_scan"]
    assert records[0]["ranges"] == [1.0, None, 2.5]
    assert records[0]["angle_min"] == pytest.approx(-0.7)
    assert records[0]["angle_increment"] == pytest.approx(0.01)
    assert math.isfinite(records[0]["t"]) and logger.camera_scans == 2
