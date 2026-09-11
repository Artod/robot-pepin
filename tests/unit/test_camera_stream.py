"""The camera node under the ROS stubs: what it broadcasts, what it publishes, how it stops.

rclpy is faked (``ros_stubs``), the MJPEG response is a test's bytes and OpenCV is the real
thing, so the node is built and its pump run here exactly as on the laptop — including the way
out, which is the point: the pump must be joined before the node is destroyed.
"""

from __future__ import annotations

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
from pepin_bringup.camera_stream import CameraStream  # noqa: E402
from pepin_bringup.msgs import stamp_seconds  # noqa: E402
from pepin_bringup.node_kit import spin_main  # noqa: E402

from pepin.camera import quaternion_from_rpy  # noqa: E402
from pepin.mounts import Mounts  # noqa: E402

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
def build(monkeypatch: pytest.MonkeyPatch) -> Iterator[Build]:
    """Build a camera node whose stream is the given bytes and whose parameters are the given
    overrides; every node built is closed when the test ends."""
    made: list[CameraStream] = []

    def make(
        feed: bytes = b"", stream: FakeStream | None = None, **params: Any
    ) -> tuple[CameraStream, FakeStream]:
        stream = FakeStream(feed) if stream is None else stream
        monkeypatch.setattr(urllib.request, "urlopen", lambda url, timeout=None: stream)
        with ros_stubs.parameters(config=CAMERA_JSON, **params):
            node = CameraStream()
        made.append(node)
        return node, stream

    yield make
    for node in made:
        node.close()


def edges(node: CameraStream) -> list[tuple[str, str]]:
    """The static transforms the node broadcast, as (parent, child)."""
    return [(t.header.frame_id, t.child_frame_id) for t in node._static.sent]


# ---- the static transforms -------------------------------------------------------------------
def test_the_node_broadcasts_the_camera_s_two_edges_and_the_laser_s(build: Build) -> None:
    """Three edges go out at start, from config/camera.json and config/lidar.json through one
    reader (pepin.mounts.Mounts) — the laser as well, because a static transform does not replay
    to a late joiner over the bridge."""
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


def test_the_report_line_carries_the_rate_and_the_switches(build: Build) -> None:
    """The period's rate comes from the kit's Tally (the elapsed time, not a divisor of 30),
    and both switches are printed with it (CLAUDE.md rule 19)."""
    node, _ = build(multipart([(1.0, jpeg(320, 180)), (1.1, jpeg(320, 180))]))
    assert until(lambda: len(node.pubs["/camera/image"].sent) == 2)
    assert node.timers == [(30.0, node._report)]
    node._report()
    line = node.logger.texts("info")[-1]
    assert line.startswith("camera: ") and "frames/s" in line
    assert "switches: scale 0.5, static_camera_tf on" in line
    node._report()
    assert "camera: 0.0 frames/s" in node.logger.texts("info")[-1], "the period was emptied"


# ---- the switches ----------------------------------------------------------------------------
def test_a_live_scale_rebuilds_the_optics_and_a_nonsense_one_is_refused(build: Build) -> None:
    """``ros2 param set /camera_stream scale 0.25`` takes the next frame down to a quarter,
    optics included; a scale outside (0, 1] is refused with its reason and nothing changes."""
    node, _ = build()
    assert node._optics.size == (640, 360)
    assert node.set_parameters([Param("scale", 0.25)])[0].successful
    assert node._optics.size == (320, 180) and node._optics.info.width == 320
    assert (node._optics.info.k[2], node._optics.info.k[5]) == (160.0, 90.0)
    refused = node.set_parameters([Param("scale", 0.0)])[0]
    assert not refused.successful and "fraction" in refused.reason
    assert node._optics.size == (320, 180) and node._switches["scale"] == 0.25


def test_the_static_transform_switch_cannot_be_flipped_while_the_node_runs(build: Build) -> None:
    """It went out at start and a static transform cannot be withdrawn: the live set is refused
    with what to do instead, and the value stays."""
    node, _ = build()
    refused = node.set_parameters([Param("static_camera_tf", False)])[0]
    assert not refused.successful and "cannot be withdrawn" in refused.reason
    assert node._switches.on("static_camera_tf")


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
