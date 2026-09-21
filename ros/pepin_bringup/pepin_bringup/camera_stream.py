"""The neck camera as ROS topics, on the laptop: MJPEG over HTTP in, Image + CameraInfo out.

ustreamer on the board serves the head camera as an MJPEG stream; nothing on the board decodes
it (the board has no core to spare and no use for pixels). This node runs on the laptop, pulls
the stream with OpenCV, and publishes ``/camera/image`` (bgr8) and ``/camera/camera_info`` with
the optics of ``config/camera.json`` — the checkerboard's measured K and distortion once
``ros/calibrate.sh`` has written them (``calibrated: true``), the nominal pinhole of the
configured field of view until then, one reader deciding (:func:`pepin.camera.optics`) and the
report line saying which — stamped with the moment the board captured the frame (ustreamer's
X-Timestamp, the clock that stamps the lidar): a frame stamped when the laptop decoded it was a
few hundred milliseconds late, a picture placed ten degrees wrong while the cart turns.

WHICH CAMERA is not this node's decision and not a flag: ``config/camera.json`` holds the rigs
by name and says which is active, ``PEPIN_CAMERA`` overrides it for one process, and the
``camera`` parameter (what the launch passes) overrides both — :func:`pepin.camera.active_camera`
is the one place that order lives. The node prints the rig it got and publishes its mount.

A STEREO RIG (``config/camera.json``'s ``stereo``, a ``rig`` block: one side-by-side frame of
1600x600, two 800x600 eyes, the module taped upside down) is the same node with one more step.
The frame is decoded ONCE, cut into the robot's two eyes (:class:`pepin.stereo.SideBySide`,
which turns an upside-down module's halves back and swaps them) and then:

* with ``config/stereo_calibration.json``: both eyes go through the rectifier
  (:class:`pepin.stereo.Rectifier`, built once — it costs about a second — and rebuilt only when
  that file's mtime moves, so a calibration finished while the robot runs is picked up without a
  restart). Out go FOUR messages with ONE stamp and the LEFT eye's optical frame, as ROS stereo
  wants: ``/camera/image`` (left, bgr8) + ``/camera/camera_info`` (the rectified pinhole, no
  distortion, R identity, P with Tx 0) and ``/camera/right/image`` (right, mono8) +
  ``/camera/right/camera_info`` (the same pinhole, P[0,3] = -fx * baseline). Everything already
  reading ``/camera/image`` + ``/camera/camera_info`` sees one ordinary camera; the stereo
  matcher reads the right pair beside it.
* without that file: only the LEFT eye goes out, unrectified, with the nominal one-eye pinhole
  of ``hfov_deg``, NOTHING on the right topics, and the report line says in words that the head
  is uncalibrated and depth has no source from it.

The flags keep their meaning for the mono rig. On a stereo rig the calibration's own rectifying
replaces ``undistort`` (which is refused, with that reason) and the picture goes out at the
calibration's own size, so ``scale`` is pinned to 1.0 — a matcher wants the size the maps were
built for, and a resized picture would cost the baseline its meaning.

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
import time
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
from pepin.stereo import Rectifier, SideBySide, StereoCalibration
from pepin_bringup.msgs import image_from_array, stamp_from_seconds, transform_from_mount
from pepin_bringup.node_kit import STOP_PATIENCE_S, Switches, Tally, Window, spin_main

CONFIG = "/ws/config/camera.json"
# The socket's own timeout: a stream that stops feeding raises instead of hanging. It is not
# the cost of stopping the node — close() shuts the socket down rather than waiting for it.
STREAM_TIMEOUT_S = 5.0
RETRY_S = 3.0  # between reconnections, waited on the stop event so a kick does not sit it out
# The right eye of a stereo rig, beside /camera/image and /camera/camera_info. The names are
# ROS's stereo convention (image_pipeline's left/right namespaces), and nothing is published on
# them while the head has no calibration.
RIGHT_IMAGE_TOPIC = "/camera/right/image"
RIGHT_INFO_TOPIC = "/camera/right/camera_info"
# What a stereo frame is made of, in the order it happens; every one is timed into the tally and
# printed in the report line as median/p95 milliseconds.
STEREO_STAGES = ("decode", "split", "rectify", "publish")
# How often the pump asks whether config/stereo_calibration.json has moved. A calibration is
# finished once in an evening, so this is only about not needing a restart; the check itself is
# one stat() and is paid on the pump's own thread.
CALIBRATION_POLL_S = 2.0

# The node's flags (CLAUDE.md rule 19), declared last in __init__ so the kit's callback sees no
# other declaration; both are printed in every report line. The range's low end is inclusive
# and a scale of zero is a picture of no pixels, so _on_switch refuses that one value.
FLAGS = FlagSet(
    Flag(
        "scale",
        0.5,
        description="the published picture as a fraction of the camera's own 1280x720, its optics"
        " scaled with it; a change takes the next frame. THE MONO RIG's flag: a stereo head"
        " publishes at its calibration's own size (the size the remap tables were built for, the"
        " size a matcher's disparity is in pixels of), so the node pins this to 1.0 there and"
        " refuses any other value with that reason",
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
        " the largest all-valid rectangle, so the field of view narrows. THE MONO RIG's flag: a"
        " stereo head is rectified by its own stereo calibration (both eyes onto one pinhole with"
        " the rows aligned, which is what a disparity means at all), so the node refuses this one"
        " there rather than straighten a picture twice",
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
    """What one frame becomes: the published size in pixels, the ``CameraInfo`` that describes
    it, the optics those came from, the remap tables when a mono picture is being rectified and,
    on a stereo rig, the rectifier and the right eye's own ``CameraInfo``.

    They travel together so a live ``scale`` or ``undistort`` — or a stereo calibration that
    appeared while the node ran — never gives a frame the other setting's optics: the pump reads
    the whole thing in one attribute read.
    """

    size: tuple[int, int]
    info: CameraInfo
    optics: Optics
    maps: tuple[Any, Any] | None = None  # cv2.remap's x and y tables, only while rectifying
    rectifier: Rectifier | None = None  # a stereo rig with a calibration: both eyes' maps
    right: CameraInfo | None = None  # and the right eye's info, P[0, 3] = -fx * baseline


class CameraStream(Node):
    """Publishes the board's MJPEG stream as images the SLAM can use."""

    def __init__(self) -> None:
        super().__init__("camera_stream")
        board = str(self.declare_parameter("board", "127.0.0.1").value)
        config = Path(str(self.declare_parameter("config", CONFIG).value))
        # Which rig the head is. Empty (the default, and what the launch passes when nobody said)
        # leaves the answer to PEPIN_CAMERA and then to the file's "active".
        camera = str(self.declare_parameter("camera", "").value).strip()
        self._cfg = CameraConfig.load(config, name=camera or None, board=board)
        self._rig = self._cfg.rig
        self._split = SideBySide(self._rig.upside_down) if self._rig is not None else None
        self._calibration_file = (
            None if self._rig is None else self._rig.calibration_path(config.parent)
        )
        self._calibration_mtime: float | None = None
        self._calibration_source = ""
        self._calibration_checked = 0.0
        # Declared after every other parameter: rclpy runs the switches' callback on
        # declarations too, and it refuses everything that is not a flag.
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        rectifier = self._read_calibration()
        self._published = self._published_for(
            self._scale(), self._switches.on("undistort"), rectifier
        )
        # A stereo rig publishes at its calibration's size, so the flag is put where the picture
        # is: through the parameter server, so `ros2 param get` and the report line agree with
        # the pixels instead of printing the mono rig's 0.5 over a full-size eye.
        if self._rig is not None and float(self._switches["scale"]) != 1.0:
            self._switches.set("scale", 1.0)
        # Reliable, like RTAB-Map's subscribers: a best-effort image never matched them.
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._image_pub = self.create_publisher(Image, "/camera/image", reliable)
        self._info_pub = self.create_publisher(CameraInfo, "/camera/camera_info", reliable)
        # Only a stereo rig advertises the right eye at all: a mono run's graph is what it was.
        self._right_image_pub = (
            None if self._rig is None else self.create_publisher(Image, RIGHT_IMAGE_TOPIC, reliable)
        )
        self._right_info_pub = (
            None
            if self._rig is None
            else self.create_publisher(CameraInfo, RIGHT_INFO_TOPIC, reliable)
        )
        if self._rig is not None and self._published.rectifier is None:
            self.get_logger().warning(
                f"THE STEREO HEAD IS UNCALIBRATED ({self._calibration_source}): the left eye goes"
                f" out alone with the nominal {self._cfg.hfov_deg:.0f} deg pinhole, nothing is"
                f" published on {RIGHT_IMAGE_TOPIC}, and a stereo depth has no source until"
                f" {self._calibration_file} exists"
            )
        elif not self._cfg.calibrated:
            self.get_logger().warning(
                f"camera optics are nominal ({self._cfg.hfov_deg:.0f} deg field of view): fine for"
                " recognising places, not for measuring — calibrate with a checkerboard"
                " (ros/calibrate.sh)"
            )
        self._static = StaticTransformBroadcaster(self)
        self._static.sendTransform(self._static_transforms(config.parent))
        self._tally = Tally(STEREO_STAGES if self._rig is not None else ())
        self.create_timer(30.0, self._report)
        self._stop = threading.Event()
        self._stream: Any | None = None  # the open response, so close() can break a blocked read
        self._thread = threading.Thread(target=self._pump, name="camera", daemon=True)
        self._thread.start()
        self.get_logger().info(
            f"camera stream from {self._cfg.stream}; rig: {self._rig_report()};"
            f" optics: {self._published.optics.source};"
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

    def _scale(self) -> float:
        """The fraction of the camera's own picture that goes out: the ``scale`` flag on a mono
        rig, always 1.0 on a stereo one — the rectified eyes are published at the size their
        remap tables were built for, which is the size a disparity is in pixels of."""
        return 1.0 if self._rig is not None else float(self._switches["scale"])

    def _read_calibration(self) -> Rectifier | None:
        """The stereo calibration beside ``config/camera.json`` as remap tables, or ``None``.

        Also leaves the file's mtime and one sentence about it (the method, the day, the RMS and
        the baseline, or why there is nothing) for the report line. Building the tables costs
        about a second, which is why this is called at start and then only when that file moves.
        """
        self._calibration_checked = time.monotonic()
        file = self._calibration_file
        if file is None:
            self._calibration_source = ""
            return None
        try:
            mtime = file.stat().st_mtime
            calibration = StereoCalibration.load(file)
            rectifier = Rectifier.from_calibration(calibration)
        except FileNotFoundError:
            self._calibration_mtime, self._calibration_source = None, f"no {file.name} yet"
            return None
        except Exception as error:  # a half-written or truncated file is not a calibration
            self._calibration_mtime = None
            self._calibration_source = f"{file.name} is unreadable ({error})"
            return None
        self._calibration_mtime = mtime
        self._calibration_source = (
            f"{calibration.method} {calibration.date}, rms {calibration.rms_px:.2f} px,"
            f" {calibration.views} views, baseline {rectifier.baseline_m * 1000:.1f} mm"
        )
        if (rectifier.width, rectifier.height) != (self._cfg.width, self._cfg.height):
            self.get_logger().warning(
                f"the stereo calibration is {rectifier.width}x{rectifier.height} an eye and"
                f" config/camera.json says {self._cfg.width}x{self._cfg.height}: the rectified"
                " pictures go out at the CALIBRATION's size, which is the one the maps were"
                " built for"
            )
        return rectifier

    def _check_calibration(self) -> None:
        """Pick up a stereo calibration that appeared (or was replaced) while the node ran: the
        file's mtime is looked at every :data:`CALIBRATION_POLL_S` on the pump's own thread, and
        a move rebuilds the rectifier and both ``CameraInfo`` messages for the next frame."""
        if self._rig is None or self._calibration_file is None:
            return
        if time.monotonic() - self._calibration_checked < CALIBRATION_POLL_S:
            return
        self._calibration_checked = time.monotonic()
        try:
            mtime: float | None = self._calibration_file.stat().st_mtime
        except OSError:
            mtime = None
        if mtime == self._calibration_mtime:
            return
        had = self._published.rectifier is not None
        rectifier = self._read_calibration()
        self._published = self._published_for(
            self._scale(), self._switches.on("undistort"), rectifier
        )
        if rectifier is not None:
            self.get_logger().info(
                f"stereo calibration {'changed' if had else 'appeared'}"
                f" ({self._calibration_source}): rectifying from the next frame, both eyes at"
                f" {self._published.size[0]}x{self._published.size[1]}"
            )
        elif had:
            self.get_logger().warning(
                f"the stereo calibration is gone ({self._calibration_source}): the left eye goes"
                f" out unrectified and nothing on {RIGHT_IMAGE_TOPIC}"
            )

    def _rig_report(self) -> str:
        """The head in one phrase for the logs: the camera's name and, for a stereo rig, how its
        two eyes travel and whether they are being rectified (with the calibration's evidence)."""
        if self._rig is None:
            return f"{self._cfg.name} (mono {self._cfg.width}x{self._cfg.height})"
        rig = self._rig
        turned = ", turned upright" if rig.upside_down else ""
        where = (
            f"{self._cfg.name} ({rig.frame_width}x{rig.frame_height} {rig.layout} ->"
            f" {self._cfg.width}x{self._cfg.height} an eye{turned})"
        )
        if self._published.rectifier is None:
            return f"{where}, NOT RECTIFIED ({self._calibration_source}): depth has no source"
        return f"{where}, rectified: {self._calibration_source}"

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

        The camera's edges are the ACTIVE rig's, by name rather than by the reader's default, so
        the frames belong to the head this node is publishing: for a stereo rig that mount is
        its LEFT eye's, which is the frame all four of its messages are stamped in.
        """
        stamp = self.get_clock().now().to_msg()
        camera = load_camera_mounts(config_dir, self._cfg.name)
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

    def _published_for(
        self, scale: float, undistort: bool, rectifier: Rectifier | None = None
    ) -> Published:
        """The published size for ``scale`` of the camera's own picture, the ``CameraInfo`` that
        goes with it, and — while ``undistort`` is on and there is a calibration to undo — the
        remap tables that straighten the frame.

        The optics come from :func:`pepin.camera.optics`: the checkerboard's K and distortion
        scaled to the published size when config/camera.json carries a calibration, the nominal
        pinhole of the configured field of view otherwise. Rectified, the CameraInfo carries the
        straightened image's own K and no distortion, because the picture no longer has any.

        A ``rectifier`` (a stereo rig with a calibration) decides both instead: the size is the
        one its maps were built for, the optics are the ONE pinhole the two eyes share after
        rectification, and the right eye's ``CameraInfo`` is built beside the left one, the same
        K with ``P[0, 3] = -fx * baseline`` — which is how everything downstream reads the
        baseline off the wire rather than out of a config.
        """
        size = (round(self._cfg.width * scale), round(self._cfg.height * scale))
        lens = optics(self._cfg, size[0], size[1])
        maps: tuple[Any, Any] | None = None
        if rectifier is not None:
            size = (rectifier.width, rectifier.height)
            lens = Optics(
                rectifier.fx,
                rectifier.fy,
                rectifier.cx,
                rectifier.cy,
                size[0],
                size[1],
                (),
                True,
                f"stereo calibration ({self._calibration_source}), rectified to"
                f" {rectifier.fx:.0f} px focal",
            )
        elif undistort and self._cfg.calibration is not None:
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
        info = self._camera_info(size, lens)
        right = None
        if rectifier is not None:
            right = self._camera_info(size, lens)
            right.p = list(right.p)
            right.p[3] = rectifier.right_projection_tx()
        return Published(size, info, lens, maps, rectifier, right)

    def _camera_info(self, size: tuple[int, int], lens: Optics) -> CameraInfo:
        """One ``sensor_msgs/CameraInfo`` for a picture of ``size`` with these optics, in the
        camera's optical frame and with no stamp yet (the frame's own is put on at publish)."""
        info = CameraInfo()
        info.header.frame_id = self._cfg.optical_frame
        info.width, info.height = size
        info.distortion_model = "plumb_bob"
        info.k, info.d, info.r, info.p = lens.camera_info_arrays()
        return info

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed: ``scale`` and ``undistort`` rebuild the published size, its optics and
        the remap tables for the next frame. Both are the MONO rig's, so on a stereo head the one
        value that matches the pixels is accepted and anything else is refused with the reason
        (the eyes go out at the calibration's size, rectified by the calibration itself).
        ``static_camera_tf`` never reaches here — it is declared ``live=False`` and the kit
        refuses the change with that reason, because the transforms went out at start and a
        static one cannot be withdrawn."""
        if name == "scale":
            if float(new) <= 0.0:
                raise ValueError("scale is a fraction of the camera's picture, not zero")
            if self._rig is not None and float(new) != 1.0:
                raise ValueError(
                    "a stereo rig publishes at its calibration's own size, which is the size a"
                    " disparity is in pixels of: scale stays 1.0 here"
                )
            self._published = self._published_for(
                self._scale(), self._switches.on("undistort"), self._published.rectifier
            )
        elif name == "undistort":
            if bool(new) and self._rig is not None:
                raise ValueError(
                    "a stereo rig is rectified by its own stereo calibration (both eyes onto one"
                    " pinhole, rows aligned): undistort is the mono rig's flag"
                )
            if bool(new) and self._cfg.calibration is None:
                raise ValueError(
                    "there is no calibration to undistort with: run ros/calibrate.sh first"
                )
            self._published = self._published_for(
                self._scale(), bool(new), self._published.rectifier
            )

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
                        with self._tally.measure("decode"):
                            frame = cv2.imdecode(
                                np.frombuffer(body, dtype=np.uint8), cv2.IMREAD_COLOR
                            )
                        if frame is None:
                            continue
                        # A calibration finished while the robot runs: picked up here, on this
                        # thread, so the frame after it is already rectified (a no-op for mono).
                        self._check_calibration()
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
        capture time (ustreamer's X-Timestamp) or, when it sent none, with the laptop's clock.

        A stereo rig's frame holds two eyes and goes out through :meth:`_publish_stereo`."""
        if self._split is not None:
            self._publish_stereo(frame, taken_at)
            return
        published = self._published
        array = np.asarray(frame)
        if published.size != (self._cfg.width, self._cfg.height):
            array = cv2.resize(array, published.size, interpolation=cv2.INTER_AREA)
        if published.maps is not None:
            array = cv2.remap(array, published.maps[0], published.maps[1], cv2.INTER_LINEAR)
            self._tally.count("rectified")
        stamp = self._stamp(taken_at)
        published.info.header.stamp = stamp
        self._image_pub.publish(image_from_array(array, "bgr8", stamp, self._cfg.optical_frame))
        self._info_pub.publish(published.info)
        self._tally.count("frames")
        if taken_at is None:
            self._tally.count("unstamped")

    def _publish_stereo(self, frame: Any, taken_at: float | None) -> None:
        """One side-by-side frame out as the robot's two eyes, stamped with the board's capture
        time (ustreamer's X-Timestamp) or, when it sent none, with the laptop's clock.

        Cut into the two eyes as the robot sees them (:class:`pepin.stereo.SideBySide`: an
        upside-down module's halves are turned back and swapped) and then, with a calibration,
        both go through the rectifier onto one pinhole with the rows aligned: out go four
        messages with ONE stamp and the LEFT eye's optical frame — the left picture in colour,
        the right in grey (a matcher wants luminance, and a grey eye is a third of the remap and
        a third of the bytes; the conversion happens before the remap, where it is cheapest, and
        bilinear interpolation of a linear combination is the same picture either way).

        With no calibration only the left eye goes out, unrectified, with the nominal one-eye
        pinhole: nothing may be published on the right topics, because a right picture with no
        measured baseline beside it is a depth nobody can compute and everybody would try to.
        """
        published, stamp = self._published, self._stamp(taken_at)
        array = np.asarray(frame)
        rig = self._rig
        if rig is not None and array.shape[:2] != (rig.frame_height, rig.frame_width):
            self._tally.count("wrong_size")
            self._tally.note(
                "wrong_size",
                f"the stream is {array.shape[1]}x{array.shape[0]} and the rig says"
                f" {rig.frame_width}x{rig.frame_height}: is the board still on the other camera"
                " (/etc/default/pepin-camera)?",
            )
        with self._tally.measure("split"):
            left, right = self._split.eyes(array) if self._split is not None else (array, None)
        rectifier = published.rectifier
        if rectifier is not None and right is not None:
            with self._tally.measure("rectify"):
                grey = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
                left, right = rectifier.rectify(left, grey)
            self._tally.count("rectified")
        with self._tally.measure("publish"):
            published.info.header.stamp = stamp
            self._image_pub.publish(image_from_array(left, "bgr8", stamp, self._cfg.optical_frame))
            self._info_pub.publish(published.info)
            if (
                rectifier is not None
                and published.right is not None
                and self._right_image_pub is not None
                and self._right_info_pub is not None
            ):
                published.right.header.stamp = stamp
                self._right_image_pub.publish(
                    image_from_array(right, "mono8", stamp, self._cfg.optical_frame)
                )
                self._right_info_pub.publish(published.right)
        self._tally.count("frames")
        if taken_at is None:
            self._tally.count("unstamped")

    def _stamp(self, taken_at: float | None) -> Any:
        """The moment a frame is published under: the board's capture time when ustreamer sent
        one (the clock that stamps the lidar), the laptop's own when it did not."""
        return self.get_clock().now().to_msg() if taken_at is None else stamp_from_seconds(taken_at)

    def _report(self) -> None:
        """Every 30 s: the period's frame rate, the frames the board sent without a capture time,
        where the published optics came from, and the flags' state, in one line — and, on a
        stereo rig, the head itself (how the eyes arrive, whether they are being rectified and
        on what evidence) with the per-stage milliseconds of a frame.

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
            f"{self._stereo_report(w)}"
            f" optics: {self._published.optics.source},"
            f" flags: {self._switches.state(live_only=False)}"
        )

    def _stereo_report(self, w: Window) -> str:
        """What a stereo rig adds to the report line — the head, the stages and whatever went
        wrong — or nothing at all on a mono one, whose line is what it always was."""
        if self._rig is None:
            return ""
        stages = " ".join(
            f"{name} {s.median_ms:.1f}/{s.p95_ms:.1f}" for name, s in w.timing.items() if s.count
        )
        wrong = ""
        if w.counts["wrong_size"]:
            wrong = f" {w.counts['wrong_size']} frames of the wrong size ({w.notes['wrong_size']}),"
        timings = f" stages: {stages} ms median/p95," if stages else ""
        return f" rig: {self._rig_report()},{wrong}{timings}"


def main() -> None:
    # ustreamer's frames carry APP segments OpenCV's MJPEG decoder cannot parse; ffmpeg reports
    # that on every frame at error level, which buries the launch's log. Fatal only.
    os.environ.setdefault("OPENCV_FFMPEG_LOGLEVEL", "8")
    spin_main(CameraStream)


if __name__ == "__main__":
    main()
