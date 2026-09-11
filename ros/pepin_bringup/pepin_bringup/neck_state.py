"""ROS 2 node on the board: the neck's encoders as ``/neck/state`` and, behind ``neck_tf``, the
live ``base_link -> camera_link`` transform.

The camera rides two servos (pan, tilt). RTAB-Map and the depth fusion on the laptop look the
camera up by TF at each frame's stamp, so an edge published here at ``poll_hz`` interpolates
cleanly between readings — and the static edge the laptop's camera node broadcasts from
config/camera.json must then stay off (camera_stream's ``static_camera_tf``, ros/laptop.sh vslam
--neck): two publishers of one edge fight. The encoders come from the base server, the bus's
owner, over the same JSON-lines socket the base bridge uses: ``{"cmd": "neck"}`` answered by
``{"type": "neck", ...}`` (pepin.neck.parse_neck; the server caches and rate-limits the bus
read, this node only asks). The geometry is pepin.neck on config/neck.json: at the reference
ticks the transform equals the static one, so flipping the switch moves nothing.

Parameters: ``host``/``port`` (the base server, 127.0.0.1:3336), ``poll_hz`` (10), ``config``
(config/neck.json beside the library, pepin.deployment.config_file); the flag ``neck_tf``
(:data:`FLAGS`, live, default **on** since 2026-09-11).

``neck_tf`` defaults on because the model is checked against the hardware: the reference ticks
in config/neck.json were read at the measured mount pose and both servo signs were verified by
moving the head by hand while watching /neck/state (config/neck.json says how, and how to redo
it after the neck is re-assembled). With the reference null again the transform is the static
mount at any head pose. The laptop's camera node must run with ``ros/laptop.sh vslam --neck``
whenever this is on, or two nodes publish base_link -> camera_link; ``ros/flags.sh set
neck_state neck_tf false`` hands the edge back to the laptop's static one.
"""

from __future__ import annotations

from typing import Any

import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.duration import Duration
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import JointState
from tf2_ros import TransformBroadcaster

from pepin.camera import quaternion_from_rpy
from pepin.deployment import config_file
from pepin.flags import Flag, FlagSet
from pepin.neck import JOINT_NAMES, NeckConfig, camera_pose, joint_angles, parse_neck
from pepin_bringup.link import JsonLineLink
from pepin_bringup.node_kit import Switches

_NECK_REQUEST = b'{"cmd":"neck"}\n'
_REPORT_S = 30.0
_STALE_S = 1.0  # a cached reading older than this (the servo fell silent) is not a pose

# The live flags (CLAUDE.md rule 19), declared last in __init__ so the kit's callback sees no
# other declaration; their state is printed in every report line. neck_tf defaults on since the
# model was checked against the hardware (see the module docstring).
FLAGS = FlagSet(
    Flag(
        "neck_tf",
        True,
        description="base_link -> camera_link is published live from the neck's encoders; the"
        " laptop's camera node must then run with ros/laptop.sh vslam --neck, or two nodes"
        " publish that edge",
    ),
)


class NeckState(Node):
    """Polls the base server for the neck's encoders; publishes joint states and the transform."""

    def __init__(self) -> None:
        super().__init__("neck_state")
        host = str(self.declare_parameter("host", "127.0.0.1").value)
        port = int(self.declare_parameter("port", 3336).value)
        self._poll_hz = float(self.declare_parameter("poll_hz", 10.0).value)
        config = str(self.declare_parameter("config", str(config_file("neck.json"))).value)
        self._switches = Switches(self, FLAGS)
        self._cfg = NeckConfig.from_json(config)
        self._joints_pub = self.create_publisher(JointState, "neck/state", 10)
        self._tf = TransformBroadcaster(self)
        self._counts = dict.fromkeys(("polls", "replies", "errors", "stale", "out_of_limits"), 0)
        self._last: tuple[int, int] | None = None
        self._last_error = ""
        self._read_ms = 0.0
        self._link = JsonLineLink(host, port, self._on_line, name="base server")
        self._link.start()
        self.create_timer(1.0 / self._poll_hz, self._poll)
        self.create_timer(_REPORT_S, self._report)
        ref = self._cfg.reference
        self.get_logger().info(
            f"neck state up: base server {host}:{port} at {self._poll_hz:.0f} Hz, reference pan"
            f" {ref.pan_ticks} tilt {ref.tilt_ticks} ticks -> pitch {ref.pitch_deg:.0f} deg at"
            f" {ref.z_m:.2f} m, signs {'verified' if ref.signs_verified else 'UNVERIFIED'};"
            f" flags: {self._switches.state()}"
        )
        if not ref.known:
            self.get_logger().warning(
                "the reference ticks are unread (config/neck.json): every pose answered is the"
                " static mount, the encoders are only reported — read them and fill them in"
            )

    def shutdown(self) -> None:
        """Close the link to the base server, on the way out."""
        self._link.stop()

    def _poll(self) -> None:
        """Ask the base server for the encoders; a dropped request (link down) is not an error."""
        self._counts["polls"] += 1
        self._link.send(_NECK_REQUEST)
        self._log_link_status()

    def _on_line(self, message: dict[str, Any]) -> None:
        """Reader thread: a neck reply becomes a joint state and, when on, the transform.

        State lines (the server broadcasts them to every client) are ignored. rclpy publishers
        are thread-safe, so this publishes at once instead of queueing for the executor.
        """
        reading = parse_neck(message)
        if reading is None:
            return
        self._counts["replies"] += 1
        if reading.error is not None:
            self._counts["errors"] += 1
            self._last_error = reading.error
        ticks = reading.ticks
        if ticks is None:
            return
        if reading.age_s > _STALE_S:
            self._counts["stale"] += 1  # the last good ticks of a servo that fell silent
            return
        pan, tilt = ticks
        self._last, self._read_ms = ticks, reading.read_ms
        if not (self._cfg.pan.within_limits(pan) and self._cfg.tilt.within_limits(tilt)):
            self._counts["out_of_limits"] += 1
        angles = joint_angles(self._cfg, pan, tilt)
        stamp = (self.get_clock().now() - Duration(seconds=reading.age_s)).to_msg()
        joints = JointState()
        joints.header.stamp = stamp
        joints.name = list(JOINT_NAMES)
        joints.position = [angles.pan_rad, angles.pitch_rad]
        self._joints_pub.publish(joints)
        if not self._switches.on("neck_tf"):
            return
        x, y, z, roll, pitch, yaw = camera_pose(self._cfg, angles)
        t = TransformStamped()
        t.header.stamp = stamp
        t.header.frame_id, t.child_frame_id = "base_link", "camera_link"
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
        qx, qy, qz, qw = quaternion_from_rpy(roll, pitch, yaw)
        t.transform.rotation.x, t.transform.rotation.y = qx, qy
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        self._tf.sendTransform(t)

    def _report(self) -> None:
        """One line per half minute: what was asked and answered, where the neck is, the cost."""
        c = self._counts
        where = "no reading yet"
        if self._last is not None:
            angles = joint_angles(self._cfg, *self._last)
            where = (
                f"pan {self._last[0]} ticks ({angles.pan_rad * 57.29578:+.1f} deg), tilt"
                f" {self._last[1]} ticks (pitch {angles.pitch_rad * 57.29578:.1f} deg)"
            )
        error = f"; last error: {self._last_error}" if c["errors"] else ""
        model = "" if self._cfg.reference.known else ", reference unread: the pose is the mount"
        self.get_logger().info(
            f"neck: polls {c['polls']}, replies {c['replies']}, errors {c['errors']}, stale"
            f" {c['stale']}, out of limits {c['out_of_limits']}; {where}; read {self._read_ms:.1f}"
            f" ms; flags: {self._switches.state()}, poll {self._poll_hz:.0f} Hz"
            f"{model}{error}"
        )
        for key in c:
            c[key] = 0

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
    """Entry point: spin the node until it is interrupted."""
    rclpy.init(args=args)
    node = NeckState()
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
