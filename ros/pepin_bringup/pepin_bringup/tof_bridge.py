"""ROS 2 node: the board's three VL53L1X sensors as sensor_msgs/Range, plus their frames.

The lidar sees one horizontal slice of the room; these three look where it
cannot — low and in front, at the height of a shoe, a cable, a cat. The board
publishes all three at ~15 Hz in millimetres and says ``null`` when a sensor
got no return; ROS wants metres and, by convention, ``+inf`` for "nothing
within max_range"; a reading equal to max_range means "nothing seen", which is
what Nav2's range layer clears the cone on.

The mounts default to the numbers measured on the robot (config/tof.json,
2026-09-04, +-1 cm) and are published once as static transforms from
base_link, so a range in ``tof_left`` lands in the right place without anyone
having to know where the shelf is.
"""

from __future__ import annotations

import contextlib
import math
import queue
from typing import Any

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Range
from tf2_ros import StaticTransformBroadcaster

from pepin_bringup.link import JsonLineLink
from pepin_bringup.protocol import TOF_NAMES, parse_tof

# VL53L1X: a ~27 deg cone, 4 cm dead zone, 1.3 m in short mode (the mode the board runs).
_FIELD_OF_VIEW_RAD = 0.47
_MIN_RANGE_M = 0.04
_MAX_RANGE_M = 1.3
_CROSSTALK_M = 0.12  # nearer than this is the sensor seeing its own surroundings

# Mount defaults, from config/tof.json: (x_m, y_m, z_m, yaw_rad) in base_link — x forward,
# y left, z up from the floor. All three look straight ahead, so every yaw is zero.
_MOUNTS: dict[str, tuple[float, float, float, float]] = {
    "front": (0.027, 0.0, 0.27, 0.0),
    "left": (0.027, 0.148, 0.16, 0.0),
    "right": (0.027, -0.155, 0.165, 0.0),
}

_DRAIN_HZ = 15.0  # readings come at ~15 Hz; a faster timer only burns the A53
_QUEUE_MAX = 100


class TofBridge(Node):
    """Bridges the ToF server to ROS: /tof/front, /tof/left, /tof/right and their static frames."""

    def __init__(self) -> None:
        """Declare parameters, publish the sensor frames, and start reading the ToF server."""
        super().__init__("tof_bridge")
        host = str(self.declare_parameter("host", "127.0.0.1").value)
        port = int(self.declare_parameter("port", 3335).value)
        self._base_frame = str(self.declare_parameter("base_frame", "base_link").value)

        self._range_pubs = {
            name: self.create_publisher(Range, f"tof/{name}", 10) for name in TOF_NAMES
        }
        self._readings: queue.Queue[dict[str, float | None]] = queue.Queue(maxsize=_QUEUE_MAX)
        self._static_tf = StaticTransformBroadcaster(self)
        self._static_tf.sendTransform([self._mount_transform(name) for name in TOF_NAMES])

        self._link = JsonLineLink(host, port, self._enqueue_ranges, name="tof server")
        self._link.start()
        self.create_timer(1.0 / _DRAIN_HZ, self._publish_pending)

    def shutdown(self) -> None:
        """Close the link, on the way out."""
        self._link.stop()

    def _mount_transform(self, name: str) -> TransformStamped:
        """Where sensor ``name`` sits on the robot, as a base_link -> tof_<name> transform."""
        x, y, z, yaw = _MOUNTS[name]
        x = float(self.declare_parameter(f"{name}_x", x).value)
        y = float(self.declare_parameter(f"{name}_y", y).value)
        z = float(self.declare_parameter(f"{name}_z", z).value)
        yaw = float(self.declare_parameter(f"{name}_yaw", yaw).value)

        transform = TransformStamped()
        transform.header.stamp = self.get_clock().now().to_msg()
        transform.header.frame_id = self._base_frame
        transform.child_frame_id = f"tof_{name}"
        transform.transform.translation.x = x
        transform.transform.translation.y = y
        transform.transform.translation.z = z
        transform.transform.rotation.z = math.sin(yaw / 2.0)
        transform.transform.rotation.w = math.cos(yaw / 2.0)
        return transform

    def _enqueue_ranges(self, message: dict[str, Any]) -> None:
        """Reader thread: hand one line of ranges to the ROS thread, dropping it if it is behind."""
        with contextlib.suppress(queue.Full):
            self._readings.put_nowait(parse_tof(message))

    def _publish_pending(self) -> None:
        """ROS thread: publish every reading the reader queued, then report link changes."""
        while True:
            try:
                ranges = self._readings.get_nowait()
            except queue.Empty:
                break
            for name, distance_m in ranges.items():
                self._publish_range(name, distance_m)
        self._log_link_status()

    def _publish_range(self, name: str, distance_m: float | None) -> None:
        """One sensor's reading as a sensor_msgs/Range; no return becomes ``max_range``."""
        message = Range()
        # Stamped 60 ms ago: the reading is at least that old (tof server -> TCP -> here), and
        # a stamp behind the newest odom->base_link transform never makes the costmap wait.
        message.header.stamp = (self.get_clock().now() - Duration(seconds=0.06)).to_msg()
        message.header.frame_id = f"tof_{name}"
        message.radiation_type = Range.INFRARED
        message.field_of_view = _FIELD_OF_VIEW_RAD
        message.min_range = _MIN_RANGE_M
        message.max_range = _MAX_RANGE_M
        # Below 12 cm the VL53L1X reports crosstalk from whatever sits at its window (the front
        # sensor flickered 0.05 <-> 1.3 m with nothing there, 2026-09-06); such a reading is
        # published below min_range, which the costmap layer drops: neither a mark nor a clear.
        if distance_m is None:
            message.range = _MAX_RANGE_M
        elif distance_m < _CROSSTALK_M:
            message.range = -1.0
        else:
            message.range = distance_m
        self._range_pubs[name].publish(message)

    def _log_link_status(self) -> None:
        """Say it once whenever the link comes up or goes down."""
        change = self._link.take_status_change()
        if change is None:
            return
        connected, detail = change
        if connected:
            self.get_logger().info(detail)
        else:
            self.get_logger().warning(detail)


def main(args: list[str] | None = None) -> None:
    """Entry point: spin the bridge until it is interrupted."""
    rclpy.init(args=args)
    node = TofBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl-C, or the SIGTERM of a docker stop
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
