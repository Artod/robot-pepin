"""The marks-audit node under the ROS stubs: the topics, the JSON contract and the report line.

rclpy, tf2_ros and the messages are faked (``ros_stubs``) and the node is built and driven here as
on the laptop. Every topic name and the report line's opening are LITERALS, because they are the
contract: Foxglove is pointed at the topics by hand and ``ros/restart.sh``'s informational check
2.11 greps the line.
"""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import pytest
import ros_stubs

ros_stubs.install()

from nav_msgs.msg import OccupancyGrid  # noqa: E402
from pepin_bringup.marks_audit import (  # noqa: E402
    AUDIT_TOPIC,
    COSTMAP_TOPIC,
    MARKS_TOPIC,
    PHANTOM_TOPIC,
    SCAN_TOPIC,
    MarksAudit,
)
from sensor_msgs.msg import LaserScan  # noqa: E402

from pepin.tsdf import RigidPose  # noqa: E402

RES, SIZE, ORIGIN = 0.05, 40, -1.0
LIDAR_CELL, CAMERA_CELL, ORPHAN_CELL = (20, 30), (24, 30), (16, 30)


class FakeTf:
    """Every frame on top of every other: the arithmetic is tested in test_marks_audit.py, and
    what this file asks is whether the node wires the right message to the right lookup."""

    def __init__(self) -> None:
        self.asked: list[tuple[str, str]] = []

    def pose(self, target: str, source: str, stamp: Any = None, timeout_s: float = 0.0) -> Any:
        self.asked.append((target, source))
        return RigidPose(np.eye(3), np.zeros(3))

    def close(self) -> None:
        pass


def centre_of(row: int, col: int) -> tuple[float, float]:
    return ORIGIN + (col + 0.5) * RES, ORIGIN + (row + 0.5) * RES


def costmap(*cells: tuple[int, int]) -> OccupancyGrid:
    """A 2x2 m local grid in ``odom`` with these cells lethal."""
    msg = OccupancyGrid()
    msg.header.frame_id = "odom"
    msg.info.resolution = RES
    msg.info.width = msg.info.height = SIZE
    msg.info.origin.position.x = msg.info.origin.position.y = ORIGIN
    data = np.zeros((SIZE, SIZE), dtype=int)
    for row, col in cells:
        data[row, col] = 100
    msg.data = list(data.reshape(-1))
    return msg


def one_beam(x: float, y: float, frame: str) -> LaserScan:
    """A scan of a single return aimed at one point of ``frame``."""
    scan = LaserScan()
    scan.header.frame_id = frame
    scan.angle_min = math.atan2(y, x)
    scan.angle_increment = 0.01
    scan.range_min, scan.range_max = 0.1, 10.0
    scan.ranges = [math.hypot(x, y)]
    return scan


@pytest.fixture
def node() -> MarksAudit:
    audit = MarksAudit()
    audit._tf = FakeTf()  # type: ignore[assignment]
    return audit


def test_it_reads_the_three_topics_and_writes_the_two(node: MarksAudit) -> None:
    assert set(node.subs) == {COSTMAP_TOPIC, SCAN_TOPIC, MARKS_TOPIC}
    assert set(node.pubs) == {AUDIT_TOPIC, PHANTOM_TOPIC}


def test_a_grid_is_split_by_the_sensor_that_can_account_for_each_cell(node: MarksAudit) -> None:
    """One cell the lidar returns from, one only the camera's fan covers, one nothing explains."""
    node._scan = one_beam(*centre_of(*LIDAR_CELL), "laser")
    node._marks = one_beam(*centre_of(*CAMERA_CELL), "base_link")
    node._audit(costmap(LIDAR_CELL, CAMERA_CELL, ORPHAN_CELL))

    out = json.loads(node._audit_pub.sent[-1].data)
    assert out["frame"] == "odom", "the grid's own frame, read from the header"
    assert (out["lethal"], out["lidar_backed"], out["camera_only"], out["unexplained"]) == (
        3,
        1,
        1,
        1,
    )
    assert out["nearest_camera_only_m"] == pytest.approx(
        math.hypot(*centre_of(*CAMERA_CELL)), abs=1e-3
    )
    # each sensor is placed by ITS OWN frame, which is the whole reason the messages travel whole
    assert ("odom", "laser") in node._tf.asked and ("odom", "base_link") in node._tf.asked
    cloud = node._phantom_pub.sent[-1]
    assert cloud.header.frame_id == "odom" and cloud.width == 1, (
        "one red point, in the grid's frame"
    )


def test_the_report_line_is_the_one_the_restart_check_greps(node: MarksAudit) -> None:
    """ros/restart.sh's check 2.11 reads `]: marks audit: ` out of the container's log."""
    node._scan = one_beam(*centre_of(*LIDAR_CELL), "laser")
    node._marks = one_beam(*centre_of(*CAMERA_CELL), "base_link")
    node._audit(costmap(LIDAR_CELL, CAMERA_CELL, ORPHAN_CELL))
    node._report()
    line = next(text for level, text in node.logger.lines if text.startswith("marks audit: "))
    assert "lethal 3 (lidar 1, camera-only 1, unexplained 1)" in line
    assert "nearest camera-only 0." in line
    assert "flags: " in line and "ms median/max:" in line


def test_a_cell_nothing_can_place_is_counted_and_never_guessed_at(node: MarksAudit) -> None:
    """No transform for the grid's frame: the grid is dropped, not judged against a stale pose."""

    class NoTf(FakeTf):
        def pose(self, target: str, source: str, stamp: Any = None, timeout_s: float = 0.0) -> Any:
            return None

    node._tf = NoTf()  # type: ignore[assignment]
    node._audit(costmap(LIDAR_CELL))
    assert not node._audit_pub.sent
    assert node._verdict is None


def test_the_switches_take_effect_on_the_next_grid_without_a_restart(node: MarksAudit) -> None:
    node._marks = one_beam(*centre_of(*CAMERA_CELL), "base_link")
    node._switches.set("phantom_cloud", False)
    node._audit(costmap(CAMERA_CELL))
    assert json.loads(node._audit_pub.sent[-1].data)["camera_only"] == 1
    assert not node._phantom_pub.sent, "the cloud is off; the counts still go out"

    node._switches.set("marks_audit", False)
    node._on_costmap(costmap(CAMERA_CELL))
    assert len(node._audit_pub.sent) == 1, "switched off, nothing is even offered to the worker"
