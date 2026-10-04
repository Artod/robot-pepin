"""pepin_bringup.camera_clip, the film every run's recorder takes: the stream is
config/camera.json's on the board's address (PEPIN_HOST), never the recorder's own loopback; the
copy keeps ustreamer's part headers (the capture stamps the camera replay dates frames by); a dead
stream is said aloud, retried once, and its absence named when the run closes. Until 2026-10-04
the laptop's recorders copied 127.0.0.1:8080, found nothing, and said nothing."""

from __future__ import annotations

import shutil
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest
import ros_stubs

ros_stubs.install()

from pepin_bringup.camera_clip import CameraClip, camera_stream  # noqa: E402

from pepin.mjpeg import has_grab, parts  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config/camera.json"
needs_curl = pytest.mark.skipif(shutil.which("curl") is None, reason="curl is not installed")


class Log:
    """The node logger's three levels, kept as (level, text)."""

    def __init__(self) -> None:
        self.lines: list[tuple[str, str]] = []

    def info(self, text: str) -> None:
        self.lines.append(("info", text))

    def warning(self, text: str) -> None:
        self.lines.append(("warning", text))

    def error(self, text: str) -> None:
        self.lines.append(("error", text))

    def said(self, level: str, part: str) -> bool:
        return any(lv == level and part in text for lv, text in self.lines)


class Clock:
    """A monotonic clock the test moves."""

    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_the_stream_is_the_camera_config_s_on_the_board_s_address() -> None:
    url = camera_stream({"PEPIN_HOST": "10.0.0.187"}, CONFIG)
    assert url == "http://10.0.0.187:8080/stream?extra_headers=1"
    assert camera_stream({}, CONFIG).startswith("http://127.0.0.1:8080/"), "on the board itself"


def test_an_unreadable_config_still_films_the_board_with_its_capture_stamps(
    tmp_path: Path,
) -> None:
    url = camera_stream({"PEPIN_HOST": "10.0.0.9"}, tmp_path / "missing.json")
    assert url == "http://10.0.0.9:8080/stream?extra_headers=1"


class _Ustreamer(BaseHTTPRequestHandler):
    """ustreamer with ``?extra_headers=1``: every part carries the send stamp and the grab and
    send times on the monotonic clock (pepin.mjpeg)."""

    stop: ClassVar[threading.Event] = threading.Event()

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace;boundary=boundarydonotcross")
        self.end_headers()
        i = 0
        try:
            while not self.stop.is_set():
                body = b"\xff\xd8" + bytes([i % 251]) * 600 + b"\xff\xd9"
                now = 1000.0 + 0.05 * i
                self.wfile.write(
                    b"--boundarydonotcross\r\nContent-Type: image/jpeg\r\n"
                    + f"Content-Length: {len(body)}\r\nX-Timestamp: {1.7e9 + now:.6f}\r\n"
                    f"X-UStreamer-Grab-Time: {now - 0.03:.6f}\r\n"
                    f"X-UStreamer-Send-Time: {now:.6f}\r\n\r\n".encode()
                    + body
                    + b"\r\n"
                )
                self.wfile.flush()
                i += 1
                time.sleep(0.05)
        except OSError:
            return

    def log_message(self, *_args: object) -> None:
        return


@pytest.fixture
def ustreamer() -> Iterator[str]:
    _Ustreamer.stop.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Ustreamer)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}/stream?extra_headers=1"
    _Ustreamer.stop.set()
    server.shutdown()
    server.server_close()


def _dead_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/stream?extra_headers=1"


@needs_curl
@pytest.mark.slow  # a real copy of a fake stream: ~1 s
def test_a_run_s_clip_is_a_byte_copy_with_the_capture_stamps(
    tmp_path: Path, ustreamer: str
) -> None:
    log, clock = Log(), Clock()
    clip = CameraClip(log, ustreamer, check_s=0.5, clock=clock)
    bag = tmp_path / "0304_20261004_230000Z_home"
    clip.start(bag)
    time.sleep(1.0)
    clock.now += 1.0
    clip.check()
    path = tmp_path / "0304_20261004_230000Z_home_cam.mjpeg"
    assert clip.path == path and path.stat().st_size > 0
    clip.stop()
    assert log.said("info", f"camera clip: {ustreamer} -> {path}")
    assert log.said("info", "MB, ") and not log.said("warning", "") and not log.said("error", "")
    with path.open("rb") as stream:
        frames = list(parts(stream))
    assert len(frames) >= 10 and all(has_grab(headers) for headers, _ in frames)


@needs_curl
@pytest.mark.slow  # two curl refusals waited for: ~0.4 s
def test_a_dead_stream_is_said_aloud_retried_once_and_named_at_the_close(tmp_path: Path) -> None:
    log, clock = Log(), Clock()
    clip = CameraClip(log, _dead_url(), check_s=4.0, clock=clock)
    clip.start(tmp_path / "0305_20261004_230100Z_printer")
    clip.check()
    assert not log.said("warning", "NOT recording"), "nothing is said before the check time"
    time.sleep(0.2)  # curl's refusal
    clock.now += 4.0
    clip.check()
    assert log.said("warning", "!! the camera clip is NOT recording (curl: (7)")
    time.sleep(0.2)
    clock.now += 4.0
    clip.check()
    assert log.said("error", "!! still no camera clip: this run has no picture")
    clip.stop()
    assert log.said("error", "!! no camera clip for this run")
    assert not (tmp_path / "0305_20261004_230100Z_printer_cam.mjpeg").exists()
