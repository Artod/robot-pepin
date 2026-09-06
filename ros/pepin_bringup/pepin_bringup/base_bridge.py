"""ROS 2 node: the board's base server as /odom, odom->base_link, and a /cmd_vel sink.

The board owns the wheels in real time behind a 0.5 s deadman: it stops them
the moment commands stop arriving. Nav2 publishes a twist only when it feels
like it and expects the last one to persist, so this node does two things with
one command — forward it the instant it arrives (latency), and re-send it at
``resend_hz`` (the board stays fed while the plan is steady). When /cmd_vel
goes quiet for ``cmd_timeout_s`` we send one stop and shut up; the deadman is
the real safety net, this is the polite version that does not rely on it.

Nothing here can kill the node: the socket lives in :class:`JsonLineLink` on
its own thread and reconnects forever. State lines are published straight from
that reader thread (rclpy publishers are thread-safe): no queue, no drain timer,
no CPU spent polling — on the board's A53 a 25 Hz Python timer alone cost a
fifth of a core.
"""

from __future__ import annotations

import math
import threading
import time
from typing import Any

import rclpy
from geometry_msgs.msg import TransformStamped, Twist, TwistStamped
from nav_msgs.msg import Odometry
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from tf2_ros import TransformBroadcaster

from pepin_bringup.link import JsonLineLink
from pepin_bringup.protocol import (
    BaseState,
    encode_stop,
    encode_twist,
    odometry_pose_covariance,
    odometry_twist_covariance,
    parse_state,
)

_STATUS_HZ = 2.0  # how often link up/down transitions are logged


class BaseBridge(Node):
    """Bridges the base server to ROS: /odom and odom->base_link out, /cmd_vel down to wheels."""

    def __init__(self) -> None:
        """Declare parameters, open the link to the base server, and start publishing."""
        super().__init__("base_bridge")
        host = str(self.declare_parameter("host", "127.0.0.1").value)
        port = int(self.declare_parameter("port", 3336).value)
        self._odom_frame = str(self.declare_parameter("odom_frame", "odom").value)
        self._base_frame = str(self.declare_parameter("base_frame", "base_link").value)
        self._cmd_timeout_s = float(self.declare_parameter("cmd_timeout_s", 0.5).value)
        resend_hz = float(self.declare_parameter("resend_hz", 5.0).value)  # deadman is 0.5 s
        # Hard ceiling for whatever arrives on /cmd_vel — teleop's q key ran the cart at 0.6 m/s
        # and slam_toolbox lost the map; the planner's limits live in nav2_params.yaml.
        self._max_linear = float(self.declare_parameter("max_linear_m_s", 0.25).value)
        self._max_angular = float(self.declare_parameter("max_angular_rad_s", 0.6).value)
        # False when an EKF (robot_localization) owns odom -> base_link; /odom is still published.
        self._publish_tf = bool(self.declare_parameter("publish_tf", True).value)

        self._pose_covariance = odometry_pose_covariance()
        self._twist_covariance = odometry_twist_covariance()
        self._lock = threading.Lock()
        self._command: tuple[float, float] | None = None
        self._command_at = 0.0
        self._stop_sent = True

        self._odom_pub = self.create_publisher(Odometry, "odom", 10)
        self._tf = TransformBroadcaster(self)
        self.create_subscription(Twist, "cmd_vel", self._on_twist, 10)
        self.create_subscription(TwistStamped, "cmd_vel_stamped", self._on_twist_stamped, 10)

        self._link = JsonLineLink(host, port, self._on_state_line, name="base server")
        self._link.start()
        self.create_timer(1.0 / _STATUS_HZ, self._log_link_status)
        self.create_timer(1.0 / resend_hz, self._resend_command)

    def shutdown(self) -> None:
        """Stop the wheels and close the link, on the way out."""
        self._link.send(encode_stop())
        self._link.stop()

    def _on_state_line(self, message: dict[str, Any]) -> None:
        """Reader thread: publish a state line at once; anything else (a pong) is ignored."""
        state = parse_state(message)
        if state is not None:
            self._publish_state(state)

    def _publish_state(self, state: BaseState) -> None:
        """One state line as a nav_msgs/Odometry on /odom and an odom->base_link transform."""
        stamp = self.get_clock().now().to_msg()
        qz, qw = math.sin(state.theta / 2.0), math.cos(state.theta / 2.0)

        odom = Odometry()
        odom.header.stamp = stamp
        odom.header.frame_id = self._odom_frame
        odom.child_frame_id = self._base_frame
        odom.pose.pose.position.x = state.x
        odom.pose.pose.position.y = state.y
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.pose.covariance = self._pose_covariance
        odom.twist.twist.linear.x = state.v  # the body frame: x forward, yaw counter-clockwise
        odom.twist.twist.angular.z = state.w
        odom.twist.covariance = self._twist_covariance
        self._odom_pub.publish(odom)

        transform = TransformStamped()
        transform.header.stamp = stamp
        transform.header.frame_id = self._odom_frame
        transform.child_frame_id = self._base_frame
        transform.transform.translation.x = state.x
        transform.transform.translation.y = state.y
        transform.transform.rotation.z = qz
        transform.transform.rotation.w = qw
        if self._publish_tf:
            self._tf.sendTransform(transform)

    def _on_twist(self, message: Twist) -> None:
        """A /cmd_vel message: down to the board now, and kept as what the resend repeats."""
        self._accept_command(message.linear.x, message.angular.z)

    def _on_twist_stamped(self, message: TwistStamped) -> None:
        """A /cmd_vel_stamped message; the stamp is not needed, the board acts on receipt."""
        self._accept_command(message.twist.linear.x, message.twist.angular.z)

    def _accept_command(self, v: float, w: float) -> None:
        """Forward a twist at once (clamped to the ceiling); remember it until it goes stale."""
        v = max(-self._max_linear, min(self._max_linear, v))
        w = max(-self._max_angular, min(self._max_angular, w))
        with self._lock:
            self._command = (v, w)
            self._command_at = time.monotonic()
            self._stop_sent = False
        self._link.send(encode_twist(v, w))

    def _resend_command(self) -> None:
        """Feed the board's deadman while /cmd_vel is steady; send one stop when it goes quiet."""
        with self._lock:
            command = self._command
            age_s = time.monotonic() - self._command_at
            stop_sent = self._stop_sent
        if command is None:
            return
        if age_s < self._cmd_timeout_s:
            self._link.send(encode_twist(*command))
        elif not stop_sent:
            with self._lock:
                self._stop_sent = True
            self._link.send(encode_stop())
            self.get_logger().info(f"no cmd_vel for {self._cmd_timeout_s:.1f} s: wheels stopped")

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
    """Entry point: spin the bridge, and stop the wheels whatever happens on the way out."""
    rclpy.init(args=args)
    node = BaseBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass  # Ctrl-C, or the SIGTERM of a docker stop: the wheels are stopped below either way
    finally:
        node.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
