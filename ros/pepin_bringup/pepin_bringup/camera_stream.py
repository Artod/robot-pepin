"""The neck camera as ROS topics, on the laptop: MJPEG over HTTP in, Image + CameraInfo out.

ustreamer on the board serves the AC310 as an MJPEG stream; nothing on the board decodes it (the
board has no core to spare and no use for pixels). This node runs on the laptop, pulls the
stream with OpenCV, and publishes ``/camera/image`` (bgr8) and ``/camera/camera_info`` with the
optics of ``config/camera.json`` — the checkerboard's measured K and distortion once
``ros/calibrate.sh`` has written them (``calibrated: true``), the nominal pinhole of the
configured field of view until then, one reader deciding (:func:`pepin.camera.optics`) and the
report line saying which — stamped with the moment the board captured the frame (ustreamer's
X-Timestamp, the clock that stamps the lidar): a frame stamped when the laptop decoded it was a
few hundred milliseconds late, a picture placed ten degrees wrong while the cart turns.
It also broadcasts the static ``base_link -> camera_link -> camera_optical`` and
``base_link -> laser`` transforms from the mounts of ``config/`` (:class:`pepin.mounts.Mounts`),
so RTAB-Map knows where the pictures were taken from — the camera's own edge only while
``static_camera_tf`` is true: with the board's neck node publishing base_link -> camera_link
live from the servo encoders (pepin_bringup.neck_state, ros/feature.sh neck on) this side must
not publish the same edge, and the launch passes the switch off (``ros/laptop.sh vslam --neck``).

The flags (:data:`FLAGS`, ``ros/flags.sh set camera_stream <name> <value>``): ``scale``, live
(the published picture as a fraction of the camera's own, optics included); ``undistort``, live
(the picture is straightened by the calibration before it goes out, and its CameraInfo then
carries no distortion); ``static_camera_tf``, read at start and not live — a static transform
cannot be withdrawn once sent, so the other value needs a restart. All three are printed in
every report line.

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

from pepin.calibration import undistort_optics
from pepin.camera import CameraConfig, Optics, optics
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
        description="the published picture as a fraction of the camera's own 1280x720, its optics"
        " scaled with it; a change takes the next frame",
        why="default by design, unmeasured: the half size was chosen when the stream was made"
        " reliable for RTAB-Map (2026-09-09) and has never been compared with the full one — no"
        " feature count, no loop closure, no bandwidth measured either way. What is measured is"
        " the rate: 8.9 fps over the bridge then, 11-11.5 fps in the report lines since. A"
        " full-size bgr8 frame is 2.7 MB of arithmetic (1280 x 720 x 3), and 640x360 is what the"
        " depth network resizes to anyway",
        on_when="raise it towards 1.0 when place recognition or a calibration needs the detail"
        " and the bridge has the bandwidth to carry it",
        off_when="lower it when the bridge is the bottleneck: the optics are scaled with the"
        " picture, so nothing downstream has to be told",
        range=(0.0, 1.0),
    ),
    Flag(
        "undistort",
        False,
        description="the published picture is rectified with the checkerboard calibration"
        " (config/camera.json's intrinsics) and its camera_info then says no distortion; a no-op"
        " while the camera is uncalibrated, since there is nothing to undo. Rectifying crops to"
        " the largest all-valid rectangle, so the field of view narrows",
        why="default by design, unmeasured: the camera is calibrated (45 views, rms 0.230 px, fx"
        " 724.1, fy 726.8, cx 652.0, cy 374.0, k1 -0.150, k2 -0.129, k3 +0.092, HFOV 82.9 deg,"
        " 2026-09-13), but the straightened picture has never been compared with the raw one on"
        " the robot, and nobody has measured what the crop costs in field of view",
        on_when="when a consumer needs straight lines — a checkerboard, a marker, a recogniser"
        " that assumes a pinhole",
        off_when="wherever a consumer was measured in the raw picture's optics (the depth"
        " pipeline's law was fitted there), and wherever the field of view matters more than"
        " straight lines",
    ),
    Flag(
        "static_camera_tf",
        True,
        description="base_link -> camera_link is broadcast from here; it goes off (ros/laptop.sh"
        " vslam --neck) when the board's neck node publishes that edge live from the servo"
        " encoders (neck_state, flag neck_tf), because two publishers of one edge fight",
        why="default by design, unmeasured: an ownership rule rather than a tuning — one edge,"
        " one publisher. Not live because a static transform cannot be withdrawn once it is sent,"
        " so the choice is made at start",
        on_when="when the neck does not publish the edge: a fixed head, or the neck node down",
        off_when="whenever neck_state runs with neck_tf on — at start, since this one cannot be"
        " taken back",
        live=False,
    ),
)


@dataclass(frozen=True)
class Published:
    """One scale's published picture: its size in pixels, the ``CameraInfo`` that describes it,
    the optics those came from, and the remap tables when the picture is being rectified. They
    travel together so a live ``scale`` or ``undistort`` never gives a frame the other
    setting's optics — the pump reads the whole thing in one attribute read."""

    size: tuple[int, int]
    info: CameraInfo
    optics: Optics
    maps: tuple[Any, Any] | None = None  # cv2.remap's x and y tables, only while rectifying


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
        self._published = self._published_for(
            float(self._switches["scale"]), self._switches.on("undistort")
        )
        # Reliable, like RTAB-Map's subscribers: a best-effort image never matched them.
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._image_pub = self.create_publisher(Image, "/camera/image", reliable)
        self._info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", reliable)
        if not self._cfg.calibrated:
            self.get_logger().warning(
                f"camera optics are nominal ({self._cfg.hfov_deg:.0f} deg field of view): fine for"
                " recognising places, not for measuring — calibrate with a checkerboard"
                " (ros/calibrate.sh)"
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
            f"camera stream from {self._cfg.stream}; optics: {self._published.optics.source};"
            f" flags: {self._switches.state(live_only=False)}"
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

    def _published_for(self, scale: float, undistort: bool) -> Published:
        """The published size for ``scale`` of the camera's own picture, the ``CameraInfo`` that
        goes with it, and — while ``undistort`` is on and there is a calibration to undo — the
        remap tables that straighten the frame.

        The optics come from :func:`pepin.camera.optics`: the checkerboard's K and distortion
        scaled to the published size when config/camera.json carries a calibration, the nominal
        pinhole of the configured field of view otherwise. Rectified, the CameraInfo carries the
        straightened image's own K and no distortion, because the picture no longer has any.
        """
        size = (round(self._cfg.width * scale), round(self._cfg.height * scale))
        lens = optics(self._cfg, size[0], size[1])
        info = CameraInfo()
        info.header.frame_id = self._cfg.optical_frame
        info.width, info.height = size
        info.distortion_model = "plumb_bob"
        maps: tuple[Any, Any] | None = None
        if undistort and self._cfg.calibration is not None:
            k, d, new_k = undistort_optics(self._cfg.calibration, size[0], size[1])
            maps = cv2.initUndistortRectifyMap(k, d, None, new_k, size, cv2.CV_16SC2)
            lens = Optics(
                float(new_k[0, 0]),
                float(new_k[1, 1]),
                float(new_k[0, 2]),
                float(new_k[1, 2]),
                size[0],
                size[1],
                (),
                True,
                f"{lens.source}, rectified to {new_k[0, 0]:.0f} px focal",
            )
        info.k, info.d, info.r, info.p = lens.camera_info_arrays()
        return Published(size, info, lens, maps)

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed: ``scale`` and ``undistort`` rebuild the published size, its optics and
        the remap tables for the next frame. ``static_camera_tf`` never reaches here — it is
        declared ``live=False`` and the kit refuses the change with that reason, because the
        transforms went out at start and a static one cannot be withdrawn."""
        if name == "scale":
            if float(new) <= 0.0:
                raise ValueError("scale is a fraction of the camera's picture, not zero")
            self._published = self._published_for(float(new), self._switches.on("undistort"))
        elif name == "undistort":
            if bool(new) and self._cfg.calibration is None:
                raise ValueError(
                    "there is no calibration to undistort with: run ros/calibrate.sh first"
                )
            self._published = self._published_for(float(self._switches["scale"]), bool(new))

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
        the current optics, straightened when ``undistort`` is on, and stamped with the board's
        capture time (ustreamer's X-Timestamp) or, when it sent none, with the laptop's clock."""
        published = self._published
        array = np.asarray(frame)
        if published.size != (self._cfg.width, self._cfg.height):
            array = cv2.resize(array, published.size, interpolation=cv2.INTER_AREA)
        if published.maps is not None:
            array = cv2.remap(array, published.maps[0], published.maps[1], cv2.INTER_LINEAR)
            self._tally.count("rectified")
        stamp = (
            self.get_clock().now().to_msg() if taken_at is None else stamp_from_seconds(taken_at)
        )
        published.info.header.stamp = stamp
        self._image_pub.publish(image_from_array(array, "bgr8", stamp, self._cfg.optical_frame))
        self._info_pub.publish(published.info)
        self._tally.count("frames")
        if taken_at is None:
            self._tally.count("unstamped")

    def _report(self) -> None:
        """Every 30 s: the period's frame rate, the frames the board sent without a capture time,
        where the published optics came from, and the flags' state, in one line.

        Every flag, not only the live ones: which side owns base_link -> camera_link is the
        first thing one looks for in this log. And the optics in words, because a stack quietly
        measuring with a guessed field of view looks exactly like one measuring with a
        calibration (:attr:`pepin.camera.Optics.source`).
        """
        w = self._tally.take()
        unstamped = (
            f", {w.counts['unstamped']} without a capture time" if w.counts["unstamped"] else ""
        )
        self.get_logger().info(
            f"camera: {w.rate('frames'):.1f} frames/s{unstamped},"
            f" optics: {self._published.optics.source},"
            f" flags: {self._switches.state(live_only=False)}"
        )


def main() -> None:
    # ustreamer's frames carry APP segments OpenCV's MJPEG decoder cannot parse; ffmpeg reports
    # that on every frame at error level, which buries the launch's log. Fatal only.
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
    spin_main(CameraStream)


if __name__ == "__main__":
    main()
