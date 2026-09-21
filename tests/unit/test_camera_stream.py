"""The camera node under the ROS stubs: what it broadcasts, what it publishes, how it stops.

rclpy is faked (``ros_stubs``), the MJPEG response is a test's bytes and OpenCV is the real
thing, so the node is built and its pump run here exactly as on the laptop — including the way
out, which is the point: the pump must be joined before the node is destroyed.
"""

from __future__ import annotations

import math
import threading
import time
import urllib.request
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import ros_stubs

RCLPY = ros_stubs.install()

import cv2  # noqa: E402
from camera_configs import (  # noqa: E402
    CALIBRATION,
    camera_config,
    ideal_stereo_calibration,
    stereo_config,
)
from pepin_bringup.camera_stream import (  # noqa: E402
    FLAGS,
    RIGHT_IMAGE_TOPIC,
    RIGHT_INFO_TOPIC,
    CameraStream,
)
from pepin_bringup.msgs import stamp_seconds  # noqa: E402
from pepin_bringup.node_kit import spin_main  # noqa: E402

from pepin.camera import quaternion_from_rpy  # noqa: E402
from pepin.mounts import Mounts  # noqa: E402
from pepin.stereo import SideBySide  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "config"
CAMERA_JSON = str(CONFIG_DIR / "camera.json")


class Param:
    """What ``ros2 param set`` hands the node's callback."""

    def __init__(self, name: str, value: Any) -> None:
        self.name, self.value = name, value


class FakeSocket:
    """The socket under a response, with the one thing the node asks of it: a ``shutdown`` that
    ends the recv the pump is blocked in."""

    def __init__(self, released: threading.Event) -> None:
        self._released = released
        self.shut_down = False

    def shutdown(self, how: int) -> None:
        """End the blocked read with end-of-stream, the way a real SHUT_RDWR does."""
        self.shut_down = True
        self._released.set()


class FakeRaw:
    """urllib's SocketIO: where the socket hangs (``response.fp.raw._sock``)."""

    def __init__(self, sock: FakeSocket | None) -> None:
        self._sock = sock


class FakeBuffer:
    """The BufferedReader a response reads through (``response.fp``)."""

    def __init__(self, raw: FakeRaw) -> None:
        self.raw = raw


class FakeStream:
    """ustreamer's response as a test writes it: the queued bytes, then a read that blocks the
    way a live camera between frames does.

    It blocks the way the real one does, which is the point of the fake: only a shutdown of the
    socket (``fp.raw._sock``, where urllib keeps it) releases the reader — ``close()`` does not,
    because a real ``HTTPResponse.close()`` takes the buffer lock the blocked reader holds and
    waits out the socket's own timeout (scratch/camstream_close_unblocks_a_real_read.py). Left
    alone, the read raises ``TimeoutError`` after ``timeout_s``, as the socket's timeout does.
    """

    def __init__(self, feed: bytes, timeout_s: float = 2.0, socket_under_it: bool = True) -> None:
        self.feed = feed
        self.timeout_s = timeout_s
        self.reading = threading.Event()  # the pump is inside the blocking read
        self.released = threading.Event()
        self.closed = False
        self.socket = FakeSocket(self.released) if socket_under_it else None
        self.fp = FakeBuffer(FakeRaw(self.socket))

    def __enter__(self) -> FakeStream:
        return self

    def __exit__(self, *exc: Any) -> bool:
        self.close()
        return False

    def read(self, size: int) -> bytes:
        if self.feed:
            chunk, self.feed = self.feed[:size], self.feed[size:]
            return chunk
        self.reading.set()
        if not self.released.wait(self.timeout_s):
            raise TimeoutError("timed out")  # what the socket raises when nothing comes
        return b""  # the stream ended

    def close(self) -> None:
        self.closed = True


def jpeg(width: int, height: int) -> bytes:
    """A real JPEG of a picture with one bright corner: a decode that silently failed would
    show up as no frame at all."""
    picture = np.zeros((height, width, 3), dtype=np.uint8)
    picture[: height // 2, : width // 2] = (20, 40, 200)
    ok, buffer = cv2.imencode(".jpg", picture)
    assert ok
    return bytes(buffer)


def multipart(frames: list[tuple[float | None, bytes]]) -> bytes:
    """ustreamer's multipart stream: each part with its length and, when the board said one,
    its X-Timestamp."""
    out = b""
    for stamp, body in frames:
        head = f"Content-Type: image/jpeg\r\nContent-Length: {len(body)}\r\n"
        if stamp is not None:
            head += f"X-Timestamp: {stamp:.6f}\r\n"
        out += b"--boundarydonotcross\r\n" + head.encode() + b"\r\n" + body + b"\r\n"
    return out


def until(predicate: Callable[[], Any], timeout_s: float = 2.0) -> bool:
    """Wait for something the pump's thread does; True when it happened."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


Build = Callable[..., tuple[CameraStream, FakeStream]]


@pytest.fixture
def build(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Build]:
    """Build a camera node whose stream is the given bytes and whose parameters are the given
    overrides; every node built is closed when the test ends.

    Its config is a copy of config/camera.json with the optics pinned to the nominal pinhole, so
    what these tests see does not change the day the robot's camera is calibrated; a test that
    wants measured optics passes ``config=calibrated_config(tmp_path)``.
    """
    made: list[CameraStream] = []
    nominal = tmp_path / "nominal"
    nominal.mkdir()
    default_config = camera_config(nominal)

    def make(
        feed: bytes = b"", stream: FakeStream | None = None, **params: Any
    ) -> tuple[CameraStream, FakeStream]:
        stream = FakeStream(feed) if stream is None else stream
        monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: stream)
        with ros_stubs.parameters(**{"config": default_config, **params}):
            node = CameraStream()
        made.append(node)
        return node, stream

    yield make
    for node in made:
        node.close()


def edges(node: CameraStream) -> list[tuple[str, str]]:
    """The static transforms the node broadcast, as (parent, child)."""
    return [(t.header.frame_id, t.child_frame_id) for t in node._static.sent]


def side_by_side(width: int = 1600, height: int = 600) -> np.ndarray:
    """One transport frame of the stereo module as it arrives: the two halves marked in their
    own corners, so a swap or a missing rotation shows up as a pixel in the wrong place. The
    module is taped upside down, so the frame is the picture turned over — the mark of the eye
    that will become the robot's LEFT sits in the second half."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    half = width // 2
    frame[0, 0] = (255, 0, 0)  # the first half's corner: the RIGHT eye's bottom right
    frame[0, half] = (0, 255, 0)  # the second half's: the LEFT eye's bottom right
    frame[height - 1, width - 1] = (0, 0, 255)  # and the left eye's top left
    return frame


def stereo_jpeg(width: int = 1600, height: int = 600) -> bytes:
    """:func:`side_by_side` as the JPEG the board sends (quality 100, so the marked pixels
    survive the encoder and a test can follow them through the splitter)."""
    ok, buffer = cv2.imencode(".jpg", side_by_side(width, height), [cv2.IMWRITE_JPEG_QUALITY, 100])
    assert ok
    return bytes(buffer)


# ---- the static transforms -------------------------------------------------------------------
def test_the_node_broadcasts_the_camera_s_two_edges_and_the_laser_s(build: Build) -> None:
    """Three edges go out at start, from config/camera.json and config/lidar.json through
    pepin.mounts — the laser as well, because a static transform does not replay to a late
    joiner over the bridge. The laser's numbers are checked against the whole-config reader
    (Mounts.load, what the board's launch calls): the narrow readers are the same parser."""
    node, _ = build()
    assert edges(node) == [
        ("base_link", "camera_link"),
        ("camera_link", "camera_optical"),
        ("base_link", "laser"),
    ]
    laser = node._static.sent[2].transform
    x, y, z, roll, pitch, yaw = Mounts.load(CONFIG_DIR).lidar.transform()
    assert (laser.translation.x, laser.translation.y, laser.translation.z) == (x, y, z)
    q = laser.rotation
    assert (q.x, q.y, q.z, q.w) == pytest.approx(quaternion_from_rpy(roll, pitch, yaw))


def test_the_camera_link_edge_is_left_to_the_board_when_the_switch_is_off(build: Build) -> None:
    """With the board's neck node publishing base_link -> camera_link from the encoders, this
    side must not publish its static copy: two publishers of one edge fight."""
    node, _ = build(static_camera_tf=False)
    assert edges(node) == [("camera_link", "camera_optical"), ("base_link", "laser")]
    assert any("neck_state" in line for line in node.logger.texts("info"))


def test_the_node_reads_only_the_two_files_whose_frames_it_publishes(
    build: Build, tmp_path: Path
) -> None:
    """camera.json and lidar.json, nothing else: with imu.json missing and tof.json corrupted
    the camera node still starts and broadcasts its three edges.

    The whole-directory reader (pepin.mounts.Mounts.load) parses all four files, and a node
    that died on a sensor it never publishes would crash-loop under the launch's RESPAWN.
    """
    for name in ("camera.json", "lidar.json"):
        (tmp_path / name).write_text((CONFIG_DIR / name).read_text())
    (tmp_path / "tof.json").write_text("{ this is not json")  # and no imu.json at all
    node, _ = build(config=str(tmp_path / "camera.json"))
    assert edges(node) == [
        ("base_link", "camera_link"),
        ("camera_link", "camera_optical"),
        ("base_link", "laser"),
    ]


# ---- the frames ------------------------------------------------------------------------------
def test_a_frame_goes_out_at_the_scale_with_the_board_s_capture_time(build: Build) -> None:
    """Half of 1280x720 by default, bgr8 in the optical frame, stamped with ustreamer's
    X-Timestamp (the board's clock, the one that stamps the lidar), and the camera_info of the
    published size beside it."""
    node, _ = build(multipart([(1_750_000_000.25, jpeg(1280, 720))]))
    images = node.pubs["/camera/image"].sent
    assert until(lambda: images), "the pump published nothing"
    image = images[0]
    assert (image.width, image.height) == (640, 360)
    assert image.encoding == "bgr8" and image.step == 640 * 3
    assert len(image.data) == 640 * 360 * 3
    assert image.header.frame_id == "camera_optical"
    assert stamp_seconds(image.header.stamp) == pytest.approx(1_750_000_000.25, abs=1e-6)
    info = node.pubs["/camera/camera_info"].sent[0]
    assert (info.width, info.height) == (640, 360)
    assert info.header.stamp == image.header.stamp and info.distortion_model == "plumb_bob"
    assert (info.k[2], info.k[5]) == (320.0, 180.0), "the optics scale with the picture"


# ---- the optics ------------------------------------------------------------------------------
def calibrated_config(tmp_path: Path, calibrated: bool = True) -> str:
    """A copy of config/camera.json carrying a checkerboard calibration (and lidar.json beside
    it, which the node reads for the laser's static edge)."""
    return camera_config(tmp_path, CALIBRATION, calibrated=calibrated)


def test_a_calibrated_camera_publishes_the_measured_k_and_d_scaled_to_the_picture(
    build: Build, tmp_path: Path
) -> None:
    """With an intrinsics block in the config the CameraInfo is the checkerboard's, scaled to
    the published 640x360 (half the focal length, half the principal point) and carrying the
    lens's distortion; the start-up warning about guessed optics is gone and the report line
    says the numbers were measured."""
    node, _ = build(multipart([(1.0, jpeg(1280, 720))]), config=calibrated_config(tmp_path))
    assert until(lambda: node.pubs["/camera/camera_info"].sent)
    info = node.pubs["/camera/camera_info"].sent[0]
    assert (info.k[0], info.k[4]) == (450.0, 448.0)
    assert (info.k[2], info.k[5]) == (323.0, 177.0)
    assert info.d == pytest.approx(CALIBRATION["dist"])  # normalised: unchanged by the scale
    assert info.p[0] == info.k[0] and info.p[2] == info.k[2]
    assert not any("nominal" in line for line in node.logger.texts("warning"))
    node._report()
    line = node.logger.texts("info")[-1]
    assert "calibrated 2026-09-12" in line and "rms 0.28 px" in line and "9x6" in line


def test_the_undistort_flag_rectifies_the_picture_and_says_so_in_the_info(
    build: Build, tmp_path: Path
) -> None:
    """Flag on: the frame goes out through the remap tables and its CameraInfo carries the
    straightened image's own K with no distortion left — there is none in the picture any more.
    The flag is live, so the next frame is rectified without a restart."""
    node, stream = build(config=calibrated_config(tmp_path))
    assert node._published.maps is None and node._published.info.d[0] != 0.0
    assert node.set_parameters([Param("undistort", True)])[0].successful
    published = node._published
    assert published.maps is not None and published.size == (640, 360)
    assert published.info.d == [0.0, 0.0, 0.0, 0.0, 0.0]
    assert published.info.k[0] != 450.0, "the rectified picture has its own focal length"
    assert "rectified" in published.optics.source
    node._publish(np.zeros((720, 1280, 3), dtype=np.uint8), 2.0)
    image = node.pubs["/camera/image"].sent[-1]
    assert (image.width, image.height) == (640, 360)
    assert stream is not None


def test_undistorting_an_uncalibrated_camera_is_refused_with_the_reason(build: Build) -> None:
    """There is nothing to undo without a calibration, and a flag that silently did nothing
    would read as a measurement that had been made."""
    node, _ = build()
    refused = node.set_parameters([Param("undistort", True)])[0]
    assert not refused.successful and "ros/calibrate.sh" in refused.reason
    assert not node._switches.on("undistort") and node._published.maps is None


def test_an_intrinsics_block_with_calibrated_false_is_not_published(
    build: Build, tmp_path: Path
) -> None:
    """One boolean turns a bad calibration off: the block stays in the file as history and the
    node goes back to the nominal pinhole, warning that it did."""
    node, _ = build(config=calibrated_config(tmp_path, calibrated=False))
    node._report()
    assert "uncalibrated" in node.logger.texts("info")[-1]
    assert any("nominal" in line for line in node.logger.texts("warning"))


def test_a_frame_the_board_did_not_stamp_takes_the_laptop_s_clock_and_is_counted(
    build: Build,
) -> None:
    """No X-Timestamp: the frame is stamped here (the stub's clock stands at zero, so that is
    unmistakable) and the report line says how many such frames the period saw."""
    node, _ = build(multipart([(None, jpeg(320, 180))]))
    images = node.pubs["/camera/image"].sent
    assert until(lambda: images)
    assert (images[0].header.stamp.sec, images[0].header.stamp.nanosec) == (0, 0)
    node._report()
    assert "1 without a capture time" in node.logger.texts("info")[-1]


def test_the_report_line_carries_the_rate_the_optics_and_the_switches(build: Build) -> None:
    """The period's rate comes from the kit's Tally (the elapsed time, not a divisor of 30), the
    optics say in words whether they were measured or guessed, and every flag is printed with
    them — the live ones and the one read at start (CLAUDE.md rule 19)."""
    node, _ = build(multipart([(1.0, jpeg(320, 180)), (1.1, jpeg(320, 180))]))
    assert until(lambda: len(node.pubs["/camera/image"].sent) == 2)
    assert node.timers == [(30.0, node._report)]
    node._report()
    line = node.logger.texts("info")[-1]
    assert line.startswith("camera: ") and "frames/s" in line
    assert "optics: nominal 83 deg field of view (uncalibrated)" in line
    assert "flags: scale=0.5 undistort=off static_camera_tf=on" in line
    node._report()
    assert "camera: 0.0 frames/s" in node.logger.texts("info")[-1], "the period was emptied"


# ---- the switches ----------------------------------------------------------------------------
def test_a_live_scale_rebuilds_the_optics_and_a_nonsense_one_is_refused(build: Build) -> None:
    """``ros2 param set /camera_stream scale 0.25`` takes the next frame down to a quarter,
    optics included; a scale outside (0, 1] is refused with its reason and nothing changes."""
    node, _ = build()
    assert node._published.size == (640, 360)
    assert node.set_parameters([Param("scale", 0.25)])[0].successful
    assert node._published.size == (320, 180) and node._published.info.width == 320
    assert (node._published.info.k[2], node._published.info.k[5]) == (160.0, 90.0)
    refused = node.set_parameters([Param("scale", 0.0)])[0]
    assert not refused.successful and "fraction" in refused.reason
    assert node._published.size == (320, 180) and node._switches["scale"] == 0.25


def test_the_static_transform_switch_cannot_be_flipped_while_the_node_runs(build: Build) -> None:
    """It went out at start and a static transform cannot be withdrawn: the flag is declared
    not live, so the set is refused with that reason and the value stays. The other value is
    reached by a restart (``ros/laptop.sh vslam --neck``), which the flag's help says."""
    node, _ = build()
    refused = node.set_parameters([Param("static_camera_tf", False)])[0]
    assert not refused.successful and "not live, set at the next start" in refused.reason
    assert node._switches.on("static_camera_tf")
    entry = FLAGS.flag("static_camera_tf")
    assert "set at the next start" in entry.help()
    assert "cannot be withdrawn" in entry.paragraph(), "the reason lives in the why now"


# ---- the way out -----------------------------------------------------------------------------
def test_close_ends_the_pump_while_it_waits_to_reconnect(monkeypatch: pytest.MonkeyPatch) -> None:
    """A board that is not there: the pump waits between attempts on the stop event, not on
    sleep, so a kick does not have to sit out the three seconds."""
    attempted = threading.Event()

    def refuse(url: str, timeout: float | None = None) -> Any:
        attempted.set()
        raise ConnectionRefusedError("no board there")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    with ros_stubs.parameters(config=CAMERA_JSON):
        node = CameraStream()
    assert attempted.wait(2.0), "the pump never tried to open the stream"
    started = time.monotonic()
    node.close()
    assert not node._thread.is_alive()
    assert time.monotonic() - started < 1.0, "it sat out the retry wait"
    assert any("not reachable" in line for line in node.logger.texts("warning"))


def test_close_stops_the_pump_from_inside_a_blocked_read(build: Build) -> None:
    """The pump spends its life blocked on the socket between frames; close() shuts that socket
    down under it, so the join takes milliseconds instead of the socket's five-second timeout.

    The fake blocks the way the real response does — close() alone would not release it — so
    this passes only while the node really does shut the socket down.
    """
    node, stream = build(multipart([(1.0, jpeg(320, 180))]))
    assert stream.reading.wait(2.0), "the pump never reached the blocking read"
    started = time.monotonic()
    node.close()
    assert not node._thread.is_alive()
    assert stream.socket is not None and stream.socket.shut_down, "the socket was not shut down"
    assert stream.closed, "and the response was closed by the pump's own with-block"
    assert time.monotonic() - started < 1.0
    assert "camera stream ended" not in " ".join(node.logger.texts("warning")), (
        "the stream we broke ourselves is not a dropped stream"
    )
    assert node.logger.texts("warning")[-1:] != ["the camera pump is still in the stream"]


def test_close_falls_back_to_closing_a_stream_with_no_socket_under_it(build: Build) -> None:
    """Nothing to shut down (not a urllib response): close() is all there is, and the pump
    still leaves — on the read's own timeout, which is what STREAM_TIMEOUT_S costs when the
    shutdown is not available."""
    blind = FakeStream(multipart([(1.0, jpeg(320, 180))]), timeout_s=0.3, socket_under_it=False)
    node, stream = build(stream=blind)
    assert stream.reading.wait(2.0), "the pump never reached the blocking read"
    node.close()
    assert stream.closed and not node._thread.is_alive()


def test_the_pump_is_joined_before_the_node_is_destroyed_or_the_context_shut_down(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The SIGABRT shape, as a test: the pump was a bare daemon thread inside cv2/ffmpeg and
    urllib, and CPython ends such a thread at interpreter exit with pthread_exit, which unwinds
    through noexcept C++ frames into std::terminate (the depth node, node_kit.spin_main). Here
    the signal lands while the pump sits in the stream; the order must be pump-out, destroy,
    shutdown, with the thread gone by then."""
    RCLPY.log.clear()
    order: list[str] = []
    stream = FakeStream(multipart([(1.0, jpeg(320, 180))]))
    monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: stream)
    built: list[CameraStream] = []

    class Watched(CameraStream):
        def close(self) -> None:
            super().close()
            order.append("pump out" if not self._thread.is_alive() else "pump still running")

        def destroy_node(self) -> None:
            order.append("destroy")
            super().destroy_node()

    def factory() -> Watched:
        node = Watched()
        built.append(node)
        return node

    def end_spin() -> None:
        assert stream.reading.wait(2.0), "the pump never reached the blocking read"
        raise KeyboardInterrupt

    RCLPY.on_spin = end_spin
    try:
        with ros_stubs.parameters(config=CAMERA_JSON):
            spin_main(factory)
    finally:
        RCLPY.on_spin = None
    order.append("shutdown" if RCLPY.log[-1] == "try_shutdown" else "no shutdown")
    assert order == ["pump out", "destroy", "shutdown"]
    assert built[0].pubs["/camera/image"].sent, "the frame before the signal still went out"


# ---- the stereo rig --------------------------------------------------------------------------
def published_image(msg: Any) -> np.ndarray:
    """A published ``sensor_msgs/Image`` back as the array it was made from."""
    channels = 3 if msg.encoding == "bgr8" else 1
    array = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, channels)
    return array if channels == 3 else array[:, :, 0]


def test_a_stereo_rig_publishes_two_rectified_eyes_of_one_frame_under_one_stamp(
    build: Build, tmp_path: Path
) -> None:
    """The four messages ROS stereo wants, out of ONE side-by-side frame: the left eye in colour
    with the rectified pinhole (P's Tx zero, no distortion left, R the identity) and the right
    eye in grey with the same K and ``P[0, 3] = -fx * baseline`` — the baseline read off the
    wire, not out of a config. One stamp, the board's, on all four; one frame id, the LEFT eye's
    optical frame, because that is the frame the pair is measured in."""
    config = stereo_config(tmp_path, ideal_stereo_calibration())
    node, _ = build(multipart([(1_750_000_000.25, stereo_jpeg())]), config=config)
    images = node.pubs["/camera/image"].sent
    assert until(lambda: images), "the pump published nothing"
    left, info = images[0], node.pubs["/camera/camera_info"].sent[0]
    right = node.pubs[RIGHT_IMAGE_TOPIC].sent[0]
    right_info = node.pubs[RIGHT_INFO_TOPIC].sent[0]
    assert (left.width, left.height) == (800, 600) and left.encoding == "bgr8"
    assert (right.width, right.height) == (800, 600) and right.encoding == "mono8"
    assert len(right.data) == 800 * 600 and right.step == 800
    stamps = {(m.header.stamp.sec, m.header.stamp.nanosec) for m in (left, info, right, right_info)}
    assert len(stamps) == 1, "the four messages are one moment"
    assert stamp_seconds(left.header.stamp) == pytest.approx(1_750_000_000.25, abs=1e-6)
    frames = {m.header.frame_id for m in (left, info, right, right_info)}
    assert frames == {"camera_optical"}, "the pair lives in the left eye's optical frame"
    assert (info.width, info.height) == (800, 600) and info.d == [0.0] * 5
    assert info.r == [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert info.k[0] == pytest.approx(700.0, abs=0.01)
    assert info.k[2] == pytest.approx(400.0, abs=0.01)
    assert info.p[3] == 0.0, "the left eye is the origin of the rectified pair"
    assert list(right_info.k) == list(info.k) and right_info.d == [0.0] * 5
    assert right_info.p[3] == pytest.approx(-info.k[0] * 0.063, abs=1e-6)
    assert right_info.p[3] / -right_info.p[0] == pytest.approx(0.063, abs=1e-9)


def test_the_upside_down_module_s_eyes_are_turned_back_and_swapped(
    build: Build, tmp_path: Path
) -> None:
    """The module is taped upside down, so each half is rotated 180 degrees and the halves trade
    places: what goes out as the LEFT eye is the SECOND half of the frame, turned. The
    calibration here is two ideal pinholes, whose rectification is the identity, so the pixels on
    the wire are the splitter's own and the swap is visible in them."""
    node, _ = build(config=stereo_config(tmp_path, ideal_stereo_calibration()))
    frame = side_by_side()
    node._publish(frame, 2.0)
    left = published_image(node.pubs["/camera/image"].sent[-1])
    right = published_image(node.pubs[RIGHT_IMAGE_TOPIC].sent[-1])
    expected_left, expected_right = SideBySide(upside_down=True).eyes(frame)
    assert np.array_equal(left, expected_left), "the left eye is the second half, turned"
    assert np.array_equal(right, cv2.cvtColor(expected_right, cv2.COLOR_BGR2GRAY))
    assert tuple(left[-1, -1]) == (0, 255, 0), "the second half's mark is at the bottom right"
    assert tuple(left[0, 0]) == (0, 0, 255)
    assert not np.array_equal(left, frame[:, :800]), "the halves really did trade places"


def test_an_uncalibrated_stereo_head_sends_the_left_eye_alone_and_says_it_cannot_measure(
    build: Build, tmp_path: Path
) -> None:
    """No config/stereo_calibration.json: the left eye goes out unrectified with the nominal
    one-eye pinhole (94 degrees across 800 px), NOTHING is published on the right topics — a
    right picture with no measured baseline is a depth nobody can compute and everybody would
    try to — and both the start-up warning and the report line say so in words."""
    node, _ = build(multipart([(1.0, stereo_jpeg())]), config=stereo_config(tmp_path))
    assert until(lambda: node.pubs["/camera/image"].sent)
    assert node.pubs[RIGHT_IMAGE_TOPIC].sent == [] and node.pubs[RIGHT_INFO_TOPIC].sent == []
    image = node.pubs["/camera/image"].sent[0]
    info = node.pubs["/camera/camera_info"].sent[0]
    assert (image.width, image.height) == (800, 600) and image.encoding == "bgr8"
    assert info.k[0] == pytest.approx(400.0 / math.tan(math.radians(47.0)), abs=0.01)
    assert (info.k[2], info.k[5]) == (400.0, 300.0) and info.p[3] == 0.0
    warning = " ".join(node.logger.texts("warning"))
    assert (
        "THE STEREO HEAD IS UNCALIBRATED" in warning and "no stereo_calibration.json yet" in warning
    )
    node._report()
    line = node.logger.texts("info")[-1]
    assert "NOT RECTIFIED" in line and "depth has no source" in line


def test_a_calibration_finished_while_the_node_runs_is_picked_up_without_a_restart(
    build: Build, tmp_path: Path
) -> None:
    """The rectifier costs about a second to build, so it is built once — and rebuilt when
    config/stereo_calibration.json's mtime moves, which is what an evening's calibration is. The
    frame after it is rectified and the right eye starts, with the log saying which file and what
    it measured."""
    node, _ = build(config=stereo_config(tmp_path))
    node._publish(side_by_side(), 1.0)
    assert node.pubs[RIGHT_IMAGE_TOPIC].sent == []
    ideal_stereo_calibration().write(tmp_path / "stereo_calibration.json")
    node._calibration_checked = 0.0  # the pump looks every CALIBRATION_POLL_S; this is that look
    node._check_calibration()
    node._publish(side_by_side(), 2.0)
    assert len(node.pubs[RIGHT_IMAGE_TOPIC].sent) == 1, "the eye after the calibration"
    assert node.pubs["/camera/camera_info"].sent[-1].k[0] == pytest.approx(700.0, abs=0.01)
    assert node.pubs[RIGHT_INFO_TOPIC].sent[-1].p[3] < 0.0
    appeared = [line for line in node.logger.texts("info") if "stereo calibration appeared" in line]
    assert appeared and "opencv stereo 2026-09-20" in appeared[0] and "rms 0.21 px" in appeared[0]


def test_the_mono_rig_s_two_flags_are_refused_on_a_stereo_head_with_their_reason(
    build: Build, tmp_path: Path
) -> None:
    """``scale`` and ``undistort`` belong to the mono webcam. A stereo head publishes at its
    calibration's own size — the size its remap tables were built for and a disparity is in
    pixels of — and is rectified by that calibration, so the node pins the scale to 1.0 at start
    and refuses both changes with the reason instead of quietly doing nothing."""
    node, _ = build(config=stereo_config(tmp_path, ideal_stereo_calibration()))
    assert node._switches["scale"] == 1.0, "pinned at start, so the report line is not a lie"
    assert node._published.size == (800, 600)
    refused = node.set_parameters([Param("scale", 0.5)])[0]
    assert not refused.successful and "calibration's own size" in refused.reason
    refused = node.set_parameters([Param("undistort", True)])[0]
    assert not refused.successful and "its own stereo calibration" in refused.reason
    assert node._switches["scale"] == 1.0 and not node._switches.on("undistort")
    assert node.set_parameters([Param("scale", 1.0)])[0].successful, "the value it already has"
    # A launch override lands before the node can refuse it, so at start they are CORRECTED:
    # the flags' printed state must be what the pixels are, not what somebody asked for.
    overridden, _ = build(
        config=stereo_config(tmp_path, ideal_stereo_calibration()), scale=0.5, undistort=True
    )
    assert overridden._switches["scale"] == 1.0 and not overridden._switches.on("undistort")
    assert overridden._published.size == (800, 600) and overridden._published.maps is None
    assert any("this head is stereo" in line for line in overridden.logger.texts("warning"))


def test_the_stereo_report_line_names_the_rig_the_evidence_and_every_stage(
    build: Build, tmp_path: Path
) -> None:
    """The line an operator reads: how the eyes arrive and whether they are rectified (with the
    calibration's method, day and RMS), then the milliseconds of every stage of a frame —
    decode, split, rectify, publish — as median/p95 over the period (CLAUDE.md rule 15)."""
    config = stereo_config(tmp_path, ideal_stereo_calibration())
    node, _ = build(multipart([(1.0, stereo_jpeg()), (1.1, stereo_jpeg())]), config=config)
    assert until(lambda: len(node.pubs["/camera/image"].sent) == 2)
    node._report()
    line = node.logger.texts("info")[-1]
    assert line.startswith("camera: ") and "frames/s" in line
    assert "rig: stereo (1600x600 side_by_side -> 800x600 an eye, turned upright)" in line
    assert "rectified: opencv stereo 2026-09-20, rms 0.21 px, 24 views, baseline 63.0 mm" in line
    assert "stages: " in line and " ms median/p95" in line
    for stage in ("decode", "split", "rectify", "publish"):
        assert f"{stage} " in line.split("stages: ")[1]
    assert "flags: scale=1.0 undistort=off static_camera_tf=on" in line


def test_a_stereo_frame_of_the_wrong_size_is_counted_and_named_in_the_report(
    build: Build, tmp_path: Path
) -> None:
    """The board's half of the switch is /etc/default/pepin-camera. If it is still serving the
    mono webcam while this side is on the stereo rig, the node would be cutting a webcam picture
    down the middle: it says so, with both sizes, in every report line."""
    node, _ = build(config=stereo_config(tmp_path, ideal_stereo_calibration()))
    node._publish(np.zeros((720, 1280, 3), dtype=np.uint8), 1.0)
    node._report()
    line = node.logger.texts("info")[-1]
    assert "1 frames of the wrong size" in line and "1280x720" in line
    assert "the rig says 1600x600" in line and "/etc/default/pepin-camera" in line


def test_the_static_edges_are_the_active_rig_s_mount(build: Build, tmp_path: Path) -> None:
    """The same three edges, from the camera the node is actually publishing: a stereo head's
    base_link -> camera_link is its LEFT eye's mount, half a baseline off the centre line, and
    the mono webcam's is on it."""
    node, _ = build(config=stereo_config(tmp_path, ideal_stereo_calibration()))
    assert edges(node) == [
        ("base_link", "camera_link"),
        ("camera_link", "camera_optical"),
        ("base_link", "laser"),
    ]
    link = node._static.sent[0].transform.translation
    assert (link.x, link.y, link.z) == (0.0, 0.0315, 1.203)
    mono, _ = build()
    assert mono._static.sent[0].transform.translation.y == 0.0


def test_the_mono_rig_is_exactly_the_node_it_always_was(build: Build) -> None:
    """The contract the stereo path must not touch: with the overview camera active this node
    advertises TWO topics and no more, publishes one bgr8 picture at half the webcam's size with
    its own CameraInfo, keeps the scale flag at 0.5, and its report line carries no rig and no
    stage timings — it is the line that has been in the logs since 2026-09-09."""
    node, _ = build(multipart([(1.0, jpeg(1280, 720))]))
    assert until(lambda: node.pubs["/camera/image"].sent)
    assert set(node.pubs) == {"/camera/image", "/camera/camera_info"}
    assert node._rig is None and node._split is None and node._published.rectifier is None
    image = node.pubs["/camera/image"].sent[0]
    assert (image.width, image.height) == (640, 360) and image.encoding == "bgr8"
    assert node._switches["scale"] == 0.5
    node._report()
    line = node.logger.texts("info")[-1]
    assert "rig:" not in line and "stages:" not in line
    assert line.startswith("camera: ") and "frames/s, optics: nominal 83 deg" in line
    assert line.endswith("flags: scale=0.5 undistort=off static_camera_tf=on")
