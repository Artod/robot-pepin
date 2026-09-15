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

from pepin.flags import Flag, FlagSet
from pepin.kinematics import Twist as KinematicTwist
from pepin.odometry import Pose2D, TwistFromPose
from pepin_bringup.link import JsonLineLink
from pepin_bringup.node_kit import Switches
from pepin_bringup.protocol import (
    BaseState,
    encode_stop,
    encode_twist,
    odometry_pose_covariance,
    odometry_twist_covariance,
    parse_state,
)

_STATUS_HZ = 2.0  # how often link up/down transitions are logged

# The live switches (CLAUDE.md rule 19), printed in the link-up line. They mute a sensor where
# it is published, so a consumer sees what a dead sensor looks like — silence — without a
# restart and without losing any other live flag (ros/sensor.sh mute imu | mute odom).
# The same two names are declared by the C++ bridge (ros/pepin_base_cpp/src/base_bridge.cpp),
# which is the node that actually runs on the board: one node name, one pair of flags, whichever
# implementation robot.launch.py picked.
FLAGS = FlagSet(
    Flag(
        "imu_publish",
        True,
        description="the MPU6050's readings leave the bridge as /imu/data_raw, where the EKF"
        " fuses index 11 (the yaw rate) and nothing else; off, the chip is still read and its"
        " bias still estimated, but no message is published. THE PYTHON BRIDGE PUBLISHES NO IMU"
        " AT ALL — here the flag only exists so the node's table is the same table whichever"
        " bridge robot.launch.py started; the C++ bridge is the one that reads the chip",
        why="on, because the gyro is the heading: the wheels over-report a turn in place by"
        " 10-25 % on carpet, and odom0's vyaw — the only other yaw-rate source, live since"
        " 2026-09-15 — carries about 4 % of the weight beside it (ros/params/ekf.yaml)",
        on_when="always, unless the point of the run is what the stack does without a gyro",
        off_when="for one test of the heading on the wheels alone, or to see an EKF meet its"
        " sensor_timeout on a source that is simply gone; unmute and the rate is back within"
        " one IMU period (50 Hz)",
    ),
    Flag(
        "odom_publish",
        True,
        description="the base server's state line leaves the bridge as /odom and, while"
        " publish_tf is on, as the odom -> base_link transform; off, the wheels are still read"
        " and still commanded, and both go silent together — a transform still broadcast from a"
        " silent /odom is a state no sensor failure produces",
        why="on, because /odom is the only source of speed this filter has: odom0 fuses vx and"
        " vy at 0.001 (m/s)^2 and, since 2026-09-15, vyaw; ax and ay are off (a mount bias of"
        " -0.229 to +0.066 m/s^2 that no covariance can answer), so with /odom silent past the"
        " EKF's sensor_timeout of 0.5 s the filter has no velocity measurement left at all",
        on_when="always, unless the run is about what the stack does with dead wheel odometry",
        off_when="to watch a consumer meet a silent odometry — the EKF's sensor_timeout, Nav2's"
        " TF lookups, the tracker's dead reckoning — without stopping the base server; unmute"
        " and /odom is back on the next state line (20 Hz)",
    ),
)


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
        # WHAT /odom's TWIST MEANS. The base server's state line reports v and w as the twist it
        # was COMMANDED to apply -- snapshot() copies self.twist, which is whatever /cmd_vel last
        # asked for (src/pepin/base_server.py:466) -- while its x/y/theta are integrated from the
        # wheel travel and ARE a measurement. Publishing the command as the twist puts a
        # controller's own output where a filter reads a sensor: robot_localization fuses
        # odom0's vx today, so the EKF has been told the cart is doing exactly what it was
        # asked to do. "measured" (the default) differences two consecutive wheel poses instead
        # (pepin.odometry.TwistFromPose); "commanded" is the old behaviour, one parameter away
        # (``ros2 param set /base_bridge odom_twist_source commanded``).
        self._odom_twist_source = str(self.declare_parameter("odom_twist_source", "measured").value)
        self._twist_from_pose = TwistFromPose()
        self._switches = Switches(self, FLAGS)  # after the last declare_parameter, by contract

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
        """One state line as a nav_msgs/Odometry on /odom and an odom->base_link transform —
        nothing at all while ``odom_publish`` is off (the state line is still read)."""
        if not self._switches.on("odom_publish"):
            self._twist_from_pose.reset()  # the gap this mute makes is not a measurement
            return
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
        twist = self._odom_twist(state)
        odom.twist.twist.linear.x = twist.linear  # body frame: x forward, yaw counter-clockwise
        odom.twist.twist.angular.z = twist.angular
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

    def _odom_twist(self, state: BaseState) -> KinematicTwist:
        """The twist /odom carries: measured off two wheel poses, or the commanded one.

        The parameter is read per sample so ``ros2 param set`` switches a live filter's input
        without a restart; the source is named in every link-up line.
        """
        source = str(self.get_parameter("odom_twist_source").value)
        self._odom_twist_source = source
        if source == "commanded":
            self._twist_from_pose.reset()
            return KinematicTwist(state.v, state.w)
        return self._twist_from_pose.update(Pose2D(state.x, state.y, state.theta), state.stamp_s)

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
            self.get_logger().info(
                f"{detail}; odom twist: {self._odom_twist_source}, {self._switches.state()}"
            )
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
