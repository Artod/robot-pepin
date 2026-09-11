"""The two nodes that carry RTAB-Map's correction to wherever the drive needs it.

rclpy and tf2_ros are faked (``ros_stubs``); ``rtabmap_msgs`` is faked here, because only the
laptop's image carries it — which is the whole reason the board's half of this pair reads a
``geometry_msgs/TransformStamped`` and not a ``MapGraph``. What the fakes let a test see: which
topics a node opened, what it broadcast, and what it published.
"""

import sys
import types
from typing import Any

import numpy as np
import ros_stubs

from pepin.tsdf import RigidPose

ros_stubs.install()


class _MapGraph:
    """rtabmap_msgs/MapGraph as these nodes read it: the correction and nothing else."""

    def __init__(self, map_to_odom: Any) -> None:
        self.map_to_odom = map_to_odom


sys.modules.setdefault("rtabmap_msgs", types.ModuleType("rtabmap_msgs"))
sys.modules.setdefault("rtabmap_msgs.msg", types.ModuleType("rtabmap_msgs.msg"))
sys.modules["rtabmap_msgs.msg"].MapGraph = _MapGraph  # type: ignore[attr-defined]

from pepin_bringup import rtabmap_frame, slam_frame  # noqa: E402
from pepin_bringup.msgs import transform_from_pose  # noqa: E402


def _shift(x: float, y: float) -> Any:
    """A correction of x, y metres with no rotation, as a stamped transform."""
    pose = RigidPose(np.eye(3), np.array([x, y, 0.0]))
    return transform_from_pose("a", "b", pose, ros_stubs.Time())


def _xy(message: Any) -> tuple[float, float]:
    translation = message.transform.translation
    return round(translation.x, 6), round(translation.y, 6)


def test_the_board_broadcasts_identity_until_the_laptop_has_a_graph() -> None:
    """Nav2 needs map -> odom from the first second: with nothing heard yet the honest
    correction is identity — the map is born at the pose the cart starts from."""
    node = slam_frame.SlamFrame()
    assert list(node.subs) == ["/map_odom"]
    assert node.timers and node.timers[0][0] == 1.0 / slam_frame.RATE_HZ
    node.timers[0][1]()
    (sent,) = node._tf.sent
    assert (sent.header.frame_id, sent.child_frame_id) == ("map", "odom")
    assert _xy(sent) == (0.0, 0.0)


def test_the_board_holds_the_last_correction_and_re_stamps_it_every_tick() -> None:
    """A correction is not a measurement: it stands until the graph moves again, and a
    transform stamped a second ago is one Nav2's tolerance refuses."""
    node = slam_frame.SlamFrame()
    _, on_correction = node.subs["/map_odom"]
    on_correction(_shift(0.4, -0.2))
    node.clock.seconds = 5.0
    node.timers[0][1]()
    node.clock.seconds = 5.5
    node.timers[0][1]()
    first, second = node._tf.sent
    assert _xy(first) == _xy(second) == (0.4, -0.2)
    assert first.header.stamp.nanosec == 0 and second.header.stamp.nanosec == 500_000_000


def test_on_a_known_map_the_laptop_broadcasts_the_inverse_and_sends_nothing() -> None:
    """RTAB-Map's odometry is then the tracker's pose, so its correction reads rtabmap -> map
    and the edge our tree needs is the other way round: map is the root."""
    node = rtabmap_frame.RtabmapFrame()
    assert node.frames == ("map", "rtabmap") and not node.pubs
    _, on_graph = node.subs["/rtabmap/mapGraph"]
    on_graph(_MapGraph(_shift(0.3, 0.0).transform))
    node.timers[0][1]()
    (sent,) = node._tf.sent
    assert (sent.header.frame_id, sent.child_frame_id) == ("map", "rtabmap")
    assert _xy(sent) == (-0.3, 0.0), "the inverse of the correction"


def test_in_slam_the_laptop_sends_the_correction_home_and_touches_no_tf() -> None:
    """RTAB-Map is the map: the correction IS map -> odom, and it belongs on the board, where
    the reflexes look it up. Nothing is broadcast here — /tf crosses the bridge one way only."""
    with ros_stubs.parameters(slam=True):
        node = rtabmap_frame.RtabmapFrame()
    assert node.frames == ("map", "odom") and node._tf is None
    _, on_graph = node.subs["/rtabmap/mapGraph"]
    on_graph(_MapGraph(_shift(0.3, 0.1).transform))
    node.timers[0][1]()
    (sent,) = node.pubs["/map_odom"].sent
    assert (sent.header.frame_id, sent.child_frame_id) == ("map", "odom")
    assert _xy(sent) == (0.3, 0.1), "as it stands, not inverted"
    assert "slam=on" in node.logger.texts("info")[0]
