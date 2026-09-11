"""The neck camera as ROS topics, on the laptop: MJPEG over HTTP in, Image + CameraInfo out.

ustreamer on the board serves the AC310 as an MJPEG stream; nothing on the board decodes it (the
board has no core to spare and no use for pixels). This node runs on the laptop, pulls the
stream with OpenCV, and publishes ``/camera/image`` (bgr8) and ``/camera/camera_info`` with the
optics of ``config/camera.json`` (nominal until calibrated), stamped with the moment the board
captured the frame (ustreamer's X-Timestamp, the clock that stamps the lidar): a frame stamped
when the laptop decoded it was a few hundred milliseconds late, a picture placed ten degrees
wrong while the cart turns.
It also broadcasts the static ``base_link -> camera_link -> camera_optical`` transforms from
the same file, so RTAB-Map knows where the pictures were taken from — the first of them only
while ``static_camera_tf`` is true: with the board's neck node publishing base_link ->
camera_link live from the servo encoders (pepin_bringup.neck_state, ros/feature.sh neck on)
this side must not publish the same edge, and the launch passes the switch off
(``ros/laptop.sh vslam --neck``). A static transform cannot be withdrawn once sent, so that
switch is read at start, not live.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster

from pepin.camera import (
    CameraConfig,
    camera_info_arrays,
    mount_transform,
    optical_rotation,
    quaternion_from_rpy,
)
from pepin.lidar import LidarMount
from pepin.mjpeg import capture_time, parts

CONFIG = "/ws/config/camera.json"
LIDAR_CONFIG = "/ws/config/lidar.json"


class CameraStream(Node):
    """Publishes the board's MJPEG stream as images the SLAM can use."""

    def __init__(self) -> None:
        super().__init__("camera_stream")
        board = str(self.declare_parameter("board", "127.0.0.1").value)
        config = Path(str(self.declare_parameter("config", CONFIG).value))
        self._cfg = CameraConfig.load(config, board=board)
        # Half size by default: features for place recognition do not need 720p, and a reliable
        # 2.7 MB frame nine times a second is a cost with no return. The optics scale with it.
        self._scale = float(self.declare_parameter("scale", 0.5).value)
        self._size = (round(self._cfg.width * self._scale), round(self._cfg.height * self._scale))
        # Reliable, like RTAB-Map's subscribers: a best-effort image never matched them.
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._image_pub = self.create_publisher(Image, "/camera/image", reliable)
        self._info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", reliable)
        self._info = CameraInfo()
        self._info.header.frame_id = self._cfg.optical_frame
        self._info.width, self._info.height = self._size
        k, d, r, p = camera_info_arrays(self._size[0], self._size[1], self._cfg.hfov_deg)
        self._info.distortion_model = "plumb_bob"
        self._info.k, self._info.d, self._info.r, self._info.p = k, d, r, p
        if not self._cfg.calibrated:
            self.get_logger().warning(
                f"camera optics are nominal ({self._cfg.hfov_deg:.0f} deg field of view): fine for"
                " recognising places, not for measuring — calibrate with a checkerboard"
            )
        self._static = StaticTransformBroadcaster(self)
        # base_link -> camera_link is static only while the neck stands still: when the board's
        # neck node publishes it live (neck_state, flag neck_tf) this edge stays off here — two
        # publishers of one edge fight, and a static one cannot be withdrawn, so it is a launch
        # switch (vslam.launch.py static_camera_tf, ros/laptop.sh vslam --neck), not a live one.
        self._static_camera = bool(self.declare_parameter("static_camera_tf", True).value)
        # The lidar's mount as well: the board publishes it too, but a static transform does not
        # replay to a late joiner over the bridge (RTAB-Map dropped every scan for an hour after a
        # board reboot, 2026-09-10) — both sides publish the same file's numbers.
        lx, ly, lz, lroll, lpitch, lyaw = LidarMount.from_json(LIDAR_CONFIG).transform()
        transforms = [
            self._optical_tf(),
            self._tf("base_link", "laser", lx, ly, lz, lroll, lpitch, lyaw),
        ]
        if self._static_camera:
            transforms.insert(0, self._link_tf())
        else:
            self.get_logger().info(
                "base_link -> camera_link is the board's (neck_state): not broadcast from here"
            )
        self._static.sendTransform(transforms)
        self._frames = 0
        self._unstamped = 0  # frames the board sent without a capture time
        self.create_timer(30.0, self._report)
        threading.Thread(target=self._pump, daemon=True).start()
        self.get_logger().info(f"camera stream from {self._cfg.stream}")

    def _link_tf(self) -> TransformStamped:
        x, y, z, roll, pitch, yaw = mount_transform(self._cfg)
        return self._tf("base_link", self._cfg.link_frame, x, y, z, roll, pitch, yaw)

    def _optical_tf(self) -> TransformStamped:
        roll, pitch, yaw = optical_rotation()
        return self._tf(
            self._cfg.link_frame, self._cfg.optical_frame, 0.0, 0.0, 0.0, roll, pitch, yaw
        )

    def _tf(
        self,
        parent: str,
        child: str,
        x: float,
        y: float,
        z: float,
        roll: float,
        pitch: float,
        yaw: float,
    ) -> TransformStamped:
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
        t.header.frame_id, t.child_frame_id = parent, child
        t.transform.translation.x, t.transform.translation.y, t.transform.translation.z = x, y, z
        qx, qy, qz, qw = quaternion_from_rpy(roll, pitch, yaw)
        t.transform.rotation.x, t.transform.rotation.y = qx, qy
        t.transform.rotation.z, t.transform.rotation.w = qz, qw
        return t

    def _pump(self) -> None:
        """Read frames as they come; reconnect after a dropped stream (the board restarts too)."""
        import numpy as np

        while rclpy.ok():
            try:
                with urllib.request.urlopen(self._cfg.stream, timeout=5.0) as stream:
                    for headers, body in parts(stream):
                        if not rclpy.ok():
                            return
                        frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            continue
                        self._publish(frame, capture_time(headers))
                    self.get_logger().warning("camera stream ended; reconnecting")
            except Exception as error:
                self.get_logger().warning(f"camera stream not reachable ({error}); retrying in 3 s")
                time.sleep(3.0)

    def _publish(self, frame: object, taken_at: float | None) -> None:
        import numpy as np

        array = np.asarray(frame)
        if self._scale != 1.0:
            array = cv2.resize(array, self._size, interpolation=cv2.INTER_AREA)
        if taken_at is None:
            self._unstamped += 1
        stamp = (
            Time(nanoseconds=round(taken_at * 1e9)).to_msg()
            if taken_at is not None
            else self.get_clock().now().to_msg()
        )
        msg = Image()
        msg.header.stamp = stamp
        msg.header.frame_id = self._cfg.optical_frame
        msg.height, msg.width = int(array.shape[0]), int(array.shape[1])
        msg.encoding = "bgr8"
        msg.is_bigendian = 0
        msg.step = msg.width * 3
        msg.data = array.tobytes()
        self._info.header.stamp = stamp
        self._image_pub.publish(msg)
        self._info_pub.publish(self._info)
        self._frames += 1

    def _report(self) -> None:
        static = "on" if self._static_camera else "off"
        self.get_logger().info(
            f"camera: {self._frames / 30.0:.1f} frames/s; static_camera_tf {static}"
        )
        self._frames = 0


def main() -> None:
    # ustreamer's frames carry APP segments OpenCV's MJPEG decoder cannot parse; ffmpeg reports
    # that on every frame at error level, which buries the launch's log. Fatal only.
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
    rclpy.init()
    node = CameraStream()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
