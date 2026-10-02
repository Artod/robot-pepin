"""ros/clip.sh, the camera clip of every drive (ros/goto.sh): against a fake MJPEG server the clip
grows while it records and the closing line gives its size; against a dead stream it says so
aloud, twice, and the closing line says there is no clip. The drives of 2026-09-28 evening had no
clip and nobody was told."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]
CLIP = REPO / "ros/clip.sh"
BOUNDARY = "frame"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg is not installed"),
]


def _jpeg(seed: int) -> bytes:
    import cv2

    noise = np.random.default_rng(seed).integers(0, 255, (120, 160, 3), dtype=np.uint8)
    ok, buffer = cv2.imencode(".jpg", noise)
    assert ok
    return bytes(buffer)


class _Stream(BaseHTTPRequestHandler):
    """The head camera's server as ustreamer speaks it: one multipart part per JPEG, 20 a second."""

    frames: ClassVar[list[bytes]] = []
    stop = threading.Event()

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", f"multipart/x-mixed-replace;boundary={BOUNDARY}")
        self.end_headers()
        i = 0
        try:
            while not self.stop.is_set():
                frame = self.frames[i % len(self.frames)]
                self.wfile.write(
                    f"--{BOUNDARY}\r\nContent-Type: image/jpeg\r\n"
                    f"Content-Length: {len(frame)}\r\n\r\n".encode()
                    + frame
                    + b"\r\n"
                )
                self.wfile.flush()
                i += 1
                time.sleep(0.05)
        except OSError:
            return  # the client went away

    def log_message(self, *_args: object) -> None:
        return


@pytest.fixture
def stream() -> Iterator[str]:
    _Stream.frames = [_jpeg(seed) for seed in range(5)]
    _Stream.stop.clear()
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Stream)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/stream"
    _Stream.stop.set()
    server.shutdown()
    server.server_close()


def _dead_url() -> str:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    return f"http://127.0.0.1:{port}/stream"


def _start(clip: Path, url: str, check_s: float) -> subprocess.Popen[str]:
    return subprocess.Popen(
        ["bash", str(CLIP), str(clip), url],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ | {"PEPIN_CLIP_CHECK_S": str(check_s)},
    )


def _size_after(clip: Path, seconds: float) -> int:
    time.sleep(seconds)
    return clip.stat().st_size if clip.exists() else 0


def test_the_clip_grows_while_it_records_and_the_closing_line_gives_its_size(
    tmp_path: Path, stream: str
) -> None:
    clip = tmp_path / "20261001_120000_goto_cam.mkv"
    process = _start(clip, stream, check_s=3.0)  # ffmpeg writes its first bytes at ~1.3 s here
    first = _size_after(clip, 3.5)
    second = _size_after(clip, 1.5)
    assert 0 < first < second, (first, second)
    process.send_signal(signal.SIGTERM)
    out, _ = process.communicate(timeout=15)
    assert process.returncode == 0, out
    size = clip.stat().st_size
    assert f"({size} bytes), {clip}" in out, out
    assert out.startswith("clip: ") and "!!" not in out, out
    assert not clip.with_suffix(".ffmpeg.log").exists(), "an empty ffmpeg log is not kept"
    if shutil.which("ffprobe"):  # a playable clip: its frames read back
        count = ["-count_packets", "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0"]
        probe = subprocess.run(
            ["ffprobe", "-v", "error", *count, str(clip)],
            capture_output=True,
            text=True,
            timeout=20,
        )
        assert int(probe.stdout.strip() or 0) >= 20, probe.stdout + probe.stderr


def test_a_dead_stream_is_said_aloud_retried_once_and_leaves_no_clip(tmp_path: Path) -> None:
    clip = tmp_path / "20261001_120000_goto_cam.mkv"
    process = _start(clip, _dead_url(), check_s=1.0)
    time.sleep(3.0)
    process.send_signal(signal.SIGTERM)
    out, _ = process.communicate(timeout=15)
    lines = out.splitlines()
    assert lines[0].startswith("!! the camera clip is NOT recording ("), out
    assert lines[0].endswith("): starting it again"), out
    assert lines[1].startswith("!! still no camera clip: this drive has no picture"), out
    assert lines[-1].startswith("!! no camera clip for this drive ("), out
    assert not clip.exists()
    assert clip.with_suffix(".ffmpeg.log").read_text().strip(), "ffmpeg's own words are kept"
