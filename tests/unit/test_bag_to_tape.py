"""One run, two recorders, one tape: the bag converter's rows against the live recorder's.

The board can record a drive as JSON lines (``pepin_bringup.run_recorder``, 34-43 % of a core) or
as an MCAP bag (``ros2 bag record`` under ``pepin_bringup.bag_recorder``, a memcpy), and
``ros/tools/bag_to_tape.py`` turns the second into the first so every analysis script keeps
reading one format. That promise is only worth the test that holds it: the same messages go
through both paths here and the rows must be identical, field for field and decimal for decimal.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest
import ros_stubs

ros_stubs.install()

from action_msgs.msg import GoalStatus, GoalStatusArray  # noqa: E402
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist  # noqa: E402
from nav_msgs.msg import OccupancyGrid, Odometry  # noqa: E402
from nav_msgs.msg import Path as PathMsg  # noqa: E402
from pepin_bringup.bag_recorder import BAG_TOPICS, record_command  # noqa: E402
from pepin_bringup.run_recorder import RunRecorder  # noqa: E402
from sensor_msgs.msg import Imu, LaserScan, Range  # noqa: E402
from std_msgs.msg import Header, String  # noqa: E402
from tf2_msgs.msg import TFMessage  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _converter() -> Any:
    """ros/tools/bag_to_tape.py, loaded by path: /tools is not a package on the laptop."""
    spec = importlib.util.spec_from_file_location("bag_to_tape", REPO / "ros/tools/bag_to_tape.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("bag_to_tape", module)
    spec.loader.exec_module(module)
    return module


TOOL = _converter()


def header(seconds: float, frame_id: str = "") -> Header:
    """A stamped header, the way every driver on this robot fills one."""
    msg = Header(frame_id=frame_id)
    msg.stamp.sec = int(seconds)
    msg.stamp.nanosec = int((seconds - int(seconds)) * 1e9)
    return msg


def odometry(t: float, x: float, y: float, vx: float = 0.0, wz: float = 0.0) -> Odometry:
    """An odometry message at (x, y), heading 0, with a twist."""
    msg = Odometry(header=header(t))
    msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
    msg.twist.twist.linear.x, msg.twist.twist.angular.z = vx, wz
    return msg


def grid(t: float, cells: list[int], width: int = 4) -> OccupancyGrid:
    """A costmap message carrying ``cells`` row-major."""
    msg = OccupancyGrid(header=header(t))
    msg.info.resolution = 0.05
    msg.info.origin.position.x, msg.info.origin.position.y = 1.25, -0.5
    msg.info.width, msg.info.height = width, max(1, len(cells) // width)
    msg.data = cells
    return msg


def plan(t: float) -> PathMsg:
    """A three-point global plan."""
    msg = PathMsg(header=header(t))
    for x, y in ((0.0, 0.0), (0.5, 0.25), (1.0, 0.5)):
        from geometry_msgs.msg import PoseStamped

        pose = PoseStamped(header=header(t))
        pose.pose.position.x, pose.pose.position.y = x, y
        msg.poses.append(pose)
    return msg


def scan(t: float) -> LaserScan:
    """A tiny lidar scan with one invalid return, so the null rule is exercised too."""
    msg = LaserScan(header=header(t))
    msg.angle_min, msg.angle_increment = -1.5, 0.5
    msg.range_min, msg.range_max, msg.scan_time = 0.05, 12.0, 0.1
    msg.ranges = [1.2, 0.01, 3.4, 2.0, 0.9, 5.5]
    msg.intensities = [12.0, 0.0, 200.0, 7.0, 3.0, 99.0]
    return msg


def imu(t: float) -> Imu:
    """One IMU sample, gyro and accelerometer in base_link."""
    msg = Imu(header=header(t))
    msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z = 0.01, -0.02, 0.31
    msg.linear_acceleration.x = 0.12
    msg.linear_acceleration.y = -0.34
    msg.linear_acceleration.z = 9.79
    return msg


def tracker_pose(t: float) -> PoseWithCovarianceStamped:
    """The tracker's belief with a covariance that makes a readable confidence."""
    msg = PoseWithCovarianceStamped(header=header(t))
    msg.pose.pose.position.x, msg.pose.pose.position.y = 2.3456, -1.2345
    msg.pose.pose.orientation.z, msg.pose.pose.orientation.w = 0.3827, 0.9239
    msg.pose.covariance[0] = 0.04
    msg.pose.covariance[7] = 0.05
    msg.pose.covariance[35] = 0.02
    return msg


def tof(t: float) -> Range:
    """One ToF cone."""
    msg = Range(header=header(t))
    msg.range, msg.max_range = 0.4321, 2.0
    return msg


def twist() -> Twist:
    """One commanded twist."""
    msg = Twist()
    msg.linear.x, msg.angular.z = 0.1234, -0.5678
    return msg


def statuses() -> GoalStatusArray:
    """Nav2's status array: one goal executing, one aborted."""
    return GoalStatusArray(status_list=[GoalStatus(status=2), GoalStatus(status=6)])


# One message per record a drive writes, in the order a drive produces them (the plan before the
# global costmap: the grid's throttle is the plan counter in both recorders).
MESSAGES: list[tuple[str, Any, str]] = [
    ("/ldlidar_node/scan", scan(1000.0), "_on_scan"),
    ("/odom", odometry(1000.1, 1.5, 0.25), "_on_odom"),
    ("/odometry/filtered", odometry(1000.2, 1.51, 0.26, 0.2, 0.1), "_on_ekf"),
    ("/imu/data_raw", imu(1000.3), "_on_imu"),
    ("/tracker_pose", tracker_pose(1000.4), "_on_loc"),
    ("/cmd_vel", twist(), "_on_cmd"),
    ("/plan", plan(1000.5), "_on_plan"),
    ("/local_costmap/costmap", grid(1000.6, [0, 0, 50, 99, -1, -1, 100, 0]), "_on_costmap"),
    ("/global_costmap/costmap", grid(1000.7, [0, 0, 50, 99, -1, -1, 100, 0]), "_on_global_costmap"),
    ("/tof/front", tof(1000.8), "tof:front"),
    ("/localization/measurement", String(data='{"x":1.0}'), "_on_measurement"),
    ("/localization/sources", String(data='{"fit":0.9}'), "_on_sources"),
    ("/navigate_to_pose/_action/status", statuses(), "status:navigate_to_pose"),
]
# The rows dated on arrival rather than by a stamp of their own: the two recorders cannot agree
# on that number (one is the moment the message reached the node, the other the moment the bag
# wrote it), and everything else in them must still be equal.
ARRIVAL_DATED = {"cmd", "nav", "meas", "srcs"}


def live_rows(tmp_path: Path) -> list[dict[str, Any]]:
    """What ``run_recorder`` writes for MESSAGES: the tape it opens, read back."""
    from rclpy.node import Node

    recorder = RunRecorder(Node("run_recorder"), tmp_path)
    tape = recorder.start("test")
    for _, msg, how in MESSAGES:
        if how.startswith("tof:"):
            recorder._on_tof(how.split(":")[1], msg)
        elif how.startswith("status:"):
            recorder._on_action_status(how.split(":")[1], msg)
        else:
            getattr(recorder, how)(msg)
    recorder.stop()
    return [json.loads(line) for line in tape.read_text().splitlines()]


def converted_rows() -> list[dict[str, Any]]:
    """What the converter makes of the same messages out of a bag."""
    builder = TOOL.TapeBuilder()
    rows: list[dict[str, Any]] = []
    for topic, msg, _ in MESSAGES:
        rows += builder.feed(topic, msg, 2000.0)
    return rows


@pytest.fixture
def pair(tmp_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    return live_rows(tmp_path), converted_rows()


def test_the_converter_writes_the_rows_the_live_recorder_writes(
    pair: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    """Row for row, field for field: the one promise the two recorders make to every reader."""
    live, converted = pair
    assert len(live) == len(MESSAGES) and len(converted) == len(live)
    for taped, built in zip(live, converted, strict=True):
        assert taped["topic"] == built["topic"]
        if taped["topic"] in ARRIVAL_DATED:
            taped, built = dict(taped), dict(built)
            taped.pop("t"), built.pop("t")
        assert taped == built, taped["topic"]


def test_every_record_a_drive_writes_is_covered(
    pair: tuple[list[dict[str, Any]], list[dict[str, Any]]],
) -> None:
    """A record the test does not build is a record the converter may quietly get wrong."""
    live, _ = pair
    assert {row["topic"] for row in live} == {
        "scan",
        "pose",
        "ekf",
        "imu",
        "loc",
        "cmd",
        "plan",
        "costmap",
        "gcostmap",
        "tof",
        "meas",
        "srcs",
        "nav",
    }


def test_the_global_costmap_is_taped_once_per_plan() -> None:
    """The throttle is the plan counter in both recorders: a grid nobody planned on answers no
    question, and a drive that re-plans ten times a second would write ten grids a second."""
    builder = TOOL.TapeBuilder()
    cells = [0, 0, 50, 99, -1, -1, 100, 0]
    builder.feed("/plan", plan(1000.0), 1000.0)
    assert builder.feed("/global_costmap/costmap", grid(1000.1, cells), 1000.1)
    assert builder.feed("/global_costmap/costmap", grid(1000.2, cells), 1000.2) == []
    builder.feed("/plan", plan(1001.0), 1001.0)
    again = builder.feed("/global_costmap/costmap", grid(1001.1, cells), 1001.1)
    assert again and again[0]["plan"] == 2


def transforms(t: float, edges: list[tuple[str, str, float, float, float]]) -> TFMessage:
    """A /tf message carrying the named edges (parent, child, x, y, yaw quaternion z)."""
    from geometry_msgs.msg import TransformStamped

    msg = TFMessage()
    for parent, child, x, y, qz in edges:
        edge = TransformStamped(header=header(t, parent), child_frame_id=child)
        edge.transform.translation.x, edge.transform.translation.y = x, y
        edge.transform.rotation.z, edge.transform.rotation.w = qz, (1.0 - qz**2) ** 0.5
        msg.transforms.append(edge)
    return msg


def test_the_loc_rows_are_composed_from_tf_where_no_tracker_publishes_a_pose() -> None:
    """map -> odom (the laptop's correction) times odom -> base_link (the board's odometry) is
    the pose every other consumer composes; 5 Hz, and no confidence, because TF carries no
    covariance."""
    builder = TOOL.TapeBuilder(loc_from_tf=True)
    assert builder.feed("/tf", transforms(1000.0, [("map", "odom", 1.0, 2.0, 0.0)]), 1000.0) == []
    rows = builder.feed(
        "/tf",
        transforms(1000.1, [("odom", "base_link", 0.5, 0.0, 0.0)]),
        1000.1,
    )
    assert len(rows) == 1
    row = rows[0]
    assert row["topic"] == "loc" and row["source"] == "tf" and "confidence" not in row
    assert (row["x"], row["y"], row["theta"]) == (1.5, 2.0, 0.0)
    assert row["t"] == pytest.approx(1000.1, abs=1e-6)
    # 5 Hz: the next pair inside the period writes nothing, the one after it writes a row.
    soon = transforms(1000.2, [("odom", "base_link", 0.6, 0.0, 0.0)])
    assert builder.feed("/tf", soon, 1000.2) == []
    later = transforms(1000.4, [("odom", "base_link", 0.7, 0.0, 0.0)])
    assert len(builder.feed("/tf", later, 1000.4)) == 1


def test_a_tracker_pose_in_the_bag_keeps_tf_out_of_the_loc_rows() -> None:
    """Two sources of one record would double every pose in the tape."""
    builder = TOOL.TapeBuilder(loc_from_tf=False)
    builder.feed("/tf", transforms(1000.0, [("map", "odom", 1.0, 2.0, 0.0)]), 1000.0)
    assert (
        builder.feed("/tf", transforms(1000.1, [("odom", "base_link", 0.5, 0.0, 0.0)]), 1000.1)
        == []
    )
    assert builder.feed("/tracker_pose", tracker_pose(1000.2), 1000.2)[0]["topic"] == "loc"


def test_the_bag_records_every_topic_the_jsonl_recorder_subscribes_to(tmp_path: Path) -> None:
    """The two recorders must see the same drive: a topic subscribed by one and missing from the
    other's list is a record that silently disappears when the switch is flipped."""
    from rclpy.node import Node

    node = Node("run_recorder")
    recorder = RunRecorder(node, tmp_path)
    recorder.start("test")
    recorder._apply_pending()  # the run-only subscriptions, normally made by the node's timer
    recorder.stop()
    assert set(node.subs) <= set(BAG_TOPICS), set(node.subs) - set(BAG_TOPICS)
    assert {"/tf", "/tf_static", "/odom_laser"} <= set(BAG_TOPICS)


def test_the_record_command_names_the_bag_the_storage_and_the_hidden_topics() -> None:
    """Nav2's action status topics are hidden ones (a token starting with an underscore), and
    without the flag `ros2 bag record` drops exactly the records that say why a goal failed."""
    command = record_command(Path("/maps/rec/0251_20260922_141002Z_home"))
    assert command[:3] == ["ros2", "bag", "record"]
    assert "--include-hidden-topics" in command
    assert command[command.index("--storage") + 1] == "mcap"
    assert command[command.index("--output") + 1] == "/maps/rec/0251_20260922_141002Z_home"
    assert "--compression-mode" not in command, "the board's cores are the point of this recorder"
    assert command[-1].startswith("/") and "/navigate_to_pose/_action/status" in command
    with_qos = record_command(Path("/maps/rec/0251"), Path("/params/rosbag_qos.yaml"))
    assert with_qos[with_qos.index("--qos-profile-overrides-path") + 1] == "/params/rosbag_qos.yaml"
