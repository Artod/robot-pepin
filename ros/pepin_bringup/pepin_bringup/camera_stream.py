"""The neck camera as ROS topics, on the laptop: MJPEG over HTTP in, Image + CameraInfo out.

ustreamer on the board serves the AC310 as an MJPEG stream; nothing on the board decodes it (the
board has no core to spare and no use for pixels). This node runs on the laptop, pulls the
stream with OpenCV, and publishes ``/camera/image`` (bgr8) and ``/camera/camera_info`` with the
optics of ``config/camera.json`` (nominal until calibrated), stamped with the moment the board
captured the frame (ustreamer's X-Timestamp, the clock that stamps the lidar): a frame stamped
when the laptop decoded it was a few hundred milliseconds late, a picture placed ten degrees
wrong while the cart turns.
It also broadcasts the static ``base_link -> camera_link -> camera_optical`` and
``base_link -> laser`` transforms from the mounts of ``config/`` (:class:`pepin.mounts.Mounts`),
so RTAB-Map knows where the pictures were taken from — the camera's own edge only while
``static_camera_tf`` is true: with the board's neck node publishing base_link -> camera_link
live from the servo encoders (pepin_bringup.neck_state, ros/feature.sh neck on) this side must
not publish the same edge, and the launch passes the switch off (``ros/laptop.sh vslam --neck``).

The flags (:data:`FLAGS`, ``ros/flags.sh set camera_stream <name> <value>``): ``scale``, live
(the published picture as a fraction of the camera's own, optics included); ``static_camera_tf``,
read at start and not live — a static transform cannot be withdrawn once sent, so the other
value needs a restart. Both are printed in every report line.

The frames are pulled by one thread (:meth:`CameraStream._pump`) which :meth:`CameraStream.close`
stops and joins before the node is destroyed: a daemon thread left inside OpenCV's decoder when
the interpreter finalises is the depth node's SIGABRT (node_kit.spin_main). That thread spends
its life blocked in a socket read between frames, and what ends such a read is a shutdown of
the socket under it, not a ``close()`` of the response (:meth:`CameraStream._break_stream`).
"""

from __future__ import annotations

import contextlib
import os
import socket
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CameraInfo, Image
from tf2_ros import StaticTransformBroadcaster

from pepin.camera import CameraConfig, camera_info_arrays
from pepin.flags import Flag, FlagSet
from pepin.mjpeg import capture_time, parts
from pepin.mounts import LASER_FRAME, load_camera_mounts, load_lidar_mount
from pepin_bringup.msgs import image_from_array, stamp_from_seconds, transform_from_mount
from pepin_bringup.node_kit import STOP_PATIENCE_S, Switches, Tally, spin_main

CONFIG = "/ws/config/camera.json"
# The socket's own timeout: a stream that stops feeding raises instead of hanging. It is not
# the cost of stopping the node — close() shuts the socket down rather than waiting for it.
STREAM_TIMEOUT_S = 5.0
RETRY_S = 3.0  # between reconnections, waited on the stop event so a kick does not sit it out

# The node's flags (CLAUDE.md rule 19), declared last in __init__ so the kit's callback sees no
# other declaration; both are printed in every report line. The range's low end is inclusive
# and a scale of zero is a picture of no pixels, so _on_switch refuses that one value.
FLAGS = FlagSet(
    Flag(
        "scale",
        0.5,
        range=(0.0, 1.0),
        description="the published picture as a fraction of the camera's own 1280x720, its"
        " optics scaled with it: features for place recognition do not need 720p, and a"
        " reliable 2.7 MB frame nine times a second is a cost with no return; a change takes"
        " the next frame",
    ),
    Flag(
        "static_camera_tf",
        True,
        live=False,
        description="base_link -> camera_link is broadcast from here; it goes off (ros/laptop.sh"
        " vslam --neck) when the board's neck node publishes that edge live from the servo"
        " encoders (neck_state, flag neck_tf), because two publishers of one edge fight. Not"
        " live: a static transform cannot be withdrawn once sent",
    ),
)


@dataclass(frozen=True)
class Optics:
    """One scale's published picture: its size in pixels and the ``CameraInfo`` that describes
    it. The two travel together so a live ``scale`` never gives a frame the other size's
    optics — the pump reads the pair in one attribute read."""

    size: tuple[int, int]
    info: CameraInfo


class CameraStream(Node):
    """Publishes the board's MJPEG stream as images the SLAM can use."""

    def __init__(self) -> None:
        super().__init__("camera_stream")
        board = str(self.declare_parameter("board", "127.0.0.1").value)
        config = Path(str(self.declare_parameter("config", CONFIG).value))
        self._cfg = CameraConfig.load(config, board=board)
        # Declared after every other parameter: rclpy runs the switches' callback on
        # declarations too, and it refuses everything that is not a flag.
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        self._optics = self._optics_for(float(self._switches["scale"]))
        # Reliable, like RTAB-Map's subscribers: a best-effort image never matched them.
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._image_pub = self.create_publisher(Image, "/camera/image", reliable)
        self._info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", reliable)
        if not self._cfg.calibrated:
            self.get_logger().warning(
                f"camera optics are nominal ({self._cfg.hfov_deg:.0f} deg field of view): fine for"
                " recognising places, not for measuring — calibrate with a checkerboard"
            )
        self._static = StaticTransformBroadcaster(self)
        self._static.sendTransform(self._static_transforms(config.parent))
        self._tally = Tally()
        self.create_timer(30.0, self._report)
        self._stop = threading.Event()
        self._stream: Any | None = None  # the open response, so close() can break a blocked read
        self._thread = threading.Thread(target=self._pump, name="camera", daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"camera stream from {self._cfg.stream}; flags: {self._switches.state(live_only=False)}"
        )

    def close(self) -> None:
        """Stop the frame pump and wait for it, shutting the open stream's socket down so a read
        blocked between frames returns at once: called before the node is destroyed
        (node_kit.spin_main), so no thread is left inside OpenCV's decoder or DDS when the
        interpreter finalises.

        The stream is broken twice when needed: the pump may have been opening a new one (a
        reconnection) while the first shutdown was reaching the old one, and then it would sit
        out the socket's own timeout instead of leaving.
        """
        self._stop.set()
        self._break_stream()
        self._thread.join(0.5)
        if self._thread.is_alive():
            self._break_stream()
            self._thread.join(STOP_PATIENCE_S)
        if self._thread.is_alive():
            self.get_logger().warning(
                f"the camera pump is still in the stream after {STOP_PATIENCE_S:.0f} s;"
                " leaving anyway"
            )

    def _break_stream(self) -> None:
        """End the read the pump is blocked in, if it has a stream open: shut the socket under
        the response down, which makes that read return end-of-stream at once.

        Not ``close()``: an ``HTTPResponse`` closes through the ``BufferedReader`` whose lock
        the blocked reader holds, so the call waits out the socket's own timeout on the
        caller's thread and the read still ends on the timeout, not on the close (measured, 5 s
        of the container's ``docker stop -t 15`` budget:
        scratch/camstream_close_unblocks_a_real_read.py, 2026-09-11). ``shutdown`` returns in
        microseconds and the read comes back immediately
        (scratch/camstream_shutdown_interrupts_a_blocked_read.py). The response itself is closed
        by the pump's own ``with`` block on its way out; ``close()`` here is only the fallback
        for a stream object with no socket under it.
        """
        stream = self._stream
        if stream is None:
            return
        raw = getattr(getattr(stream, "fp", None), "raw", None)
        sock = getattr(raw, "_sock", None)
        with contextlib.suppress(Exception):  # already shut down, or gone under the reader
            if sock is None:
                stream.close()
            else:
                sock.shutdown(socket.SHUT_RDWR)

    def _static_transforms(self, config_dir: Path) -> list[Any]:
        """The static edges this node broadcasts, read from the files of ``config_dir``:
        camera_link -> camera_optical and base_link -> laser always, base_link -> camera_link
        only while ``static_camera_tf`` is on.

        Two files, camera.json and lidar.json, through the readers the board's launch uses
        (:mod:`pepin.mounts`) — not the whole-directory :meth:`pepin.mounts.Mounts.load`, which
        also parses imu.json and tof.json: a sensor this node never publishes would then take
        the camera down at start, and under the launch's RESPAWN that is a crash loop.

        The laser goes out here as well as from the board because a static transform does not
        replay to a late joiner over the bridge (RTAB-Map dropped every scan for an hour after a
        board reboot, 2026-09-10): both sides publish the same file's numbers.
        """
        stamp = self.get_clock().now().to_msg()
        camera = load_camera_mounts(config_dir)
        transforms = [
            transform_from_mount(camera.link_frame, camera.optical_frame, camera.optical, stamp),
            transform_from_mount("base_link", LASER_FRAME, load_lidar_mount(config_dir), stamp),
        ]
        if self._switches.on("static_camera_tf"):
            transforms.insert(
                0, transform_from_mount("base_link", camera.link_frame, camera.link, stamp)
            )
        else:
            self.get_logger().info(
                "base_link -> camera_link is the board's (neck_state): not broadcast from here"
            )
        return transforms

    def _optics_for(self, scale: float) -> Optics:
        """The published size for ``scale`` of the camera's own picture, and the ``CameraInfo``
        that goes with it: the nominal pinhole of the configured field of view, scaled with the
        image (half the pixels, half the focal length)."""
        size = (round(self._cfg.width * scale), round(self._cfg.height * scale))
        info = CameraInfo()
        info.header.frame_id = self._cfg.optical_frame
        info.width, info.height = size
        info.distortion_model = "plumb_bob"
        info.k, info.d, info.r, info.p = camera_info_arrays(size[0], size[1], self._cfg.hfov_deg)
        return Optics(size, info)

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed: ``scale`` rebuilds the published size and its optics for the next
        frame. ``static_camera_tf`` never reaches here — it is declared ``live=False`` and the
        kit refuses the change with that reason, because the transforms went out at start and a
        static one cannot be withdrawn."""
        if name == "scale":
            if float(new) <= 0.0:
                raise ValueError("scale is a fraction of the camera's picture, not zero")
            self._optics = self._optics_for(float(new))

    def _pump(self) -> None:
        """Read frames as they come; reconnect after a dropped stream (the board restarts too).

        Ends on :meth:`close`'s event — checked for every frame and every reconnection, and the
        blocked read between frames is ended by :meth:`_break_stream` — or when the context goes
        down; that is what lets the thread be joined instead of killed.
        """
        while not self._stop.is_set() and rclpy.ok():
            try:
                with urllib.request.urlopen(self._cfg.stream, timeout=STREAM_TIMEOUT_S) as stream:
                    self._stream = stream
                    if self._stop.is_set():  # close() ran while this one was still opening
                        return
                    for headers, body in parts(stream):
                        if self._stop.is_set() or not rclpy.ok():
                            return
                        frame = cv2.imdecode(np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR)
                        if frame is None:
                            continue
                        self._publish(frame, capture_time(headers))
                    if self._stop.is_set():
                        return  # the socket was shut down under the reader: that is the way out
                    self.get_logger().warning("camera stream ended; reconnecting")
            except Exception as error:
                if self._stop.is_set():
                    return  # the stream was closed under the reader: that is the way out
                self.get_logger().warning(
                    f"camera stream not reachable ({error}); retrying in {RETRY_S:.0f} s"
                )
                self._stop.wait(RETRY_S)
            finally:
                self._stream = None

    def _publish(self, frame: Any, taken_at: float | None) -> None:
        """One decoded frame out as ``/camera/image`` with its ``/camera/camera_info``, scaled to
        the current optics and stamped with the board's capture time (ustreamer's X-Timestamp)
        or, when it sent none, with the laptop's clock."""
        optics = self._optics
        array = np.asarray(frame)
        if optics.size != (self._cfg.width, self._cfg.height):
            array = cv2.resize(array, optics.size, interpolation=cv2.INTER_AREA)
        stamp = (
            self.get_clock().now().to_msg() if taken_at is None else stamp_from_seconds(taken_at)
        )
        optics.info.header.stamp = stamp
        self._image_pub.publish(image_from_array(array, "bgr8", stamp, self._cfg.optical_frame))
        self._info_pub.publish(optics.info)
        self._tally.count("frames")
        if taken_at is None:
            self._tally.count("unstamped")

    def _report(self) -> None:
        """Every 30 s: the period's frame rate, the frames the board sent without a capture time,
        and the flags' state, in one line. Every flag, not only the live ones: which side owns
        base_link -> camera_link is the first thing one looks for in this log."""
        w = self._tally.take()
        unstamped = (
            f", {w.counts['unstamped']} without a capture time" if w.counts["unstamped"] else ""
        )
        self.get_logger().info(
            f"camera: {w.rate('frames'):.1f} frames/s{unstamped},"
            f" flags: {self._switches.state(live_only=False)}"
        )


def main() -> None:
    # ustreamer's frames carry APP segments OpenCV's MJPEG decoder cannot parse; ffmpeg reports
    # that on every frame at error level, which buries the launch's log. Fatal only.
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
    spin_main(CameraStream)


if __name__ == "__main__":
    main()
