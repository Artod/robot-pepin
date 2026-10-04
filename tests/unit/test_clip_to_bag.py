"""ros/tools/clip_to_bag.py: a board-side clip into the four stereo topics with grab stamps.

The writer is a fake (the BagWriter protocol); the messages are built through the ROS stubs by
pepin_bringup.stereo_frames, the same code camera_stream publishes with.
"""

from __future__ import annotations

import importlib.util
import io
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import ros_stubs

ros_stubs.install()

from camera_configs import ideal_stereo_calibration  # noqa: E402
from pepin_bringup.msgs import stamp_seconds  # noqa: E402
from pepin_bringup.stereo_frames import STEREO_TOPICS, StereoFrames  # noqa: E402

from pepin.stereo import Rectifier, SideBySide  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def _tool() -> Any:
    """ros/tools/clip_to_bag.py, loaded by path: /tools is not a package on the laptop."""
    spec = importlib.util.spec_from_file_location("clip_to_bag", REPO / "ros/tools/clip_to_bag.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("clip_to_bag", module)
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


class FakeWriter:
    """The bag as a list of (topic, msg, stamp_ns)."""

    def __init__(self) -> None:
        self.rows: list[tuple[str, Any, int]] = []
        self.closed = False

    def write(self, topic: str, msg: Any, stamp_ns: int) -> None:
        self.rows.append((topic, msg, stamp_ns))

    def close(self) -> None:
        self.closed = True


def _jpeg() -> bytes:
    picture = np.zeros((12, 32, 3), dtype=np.uint8)  # two 16x12 eyes side by side
    picture[:, 16:] = (10, 200, 30)
    ok, buffer = cv2.imencode(".jpg", picture)
    assert ok
    return bytes(buffer)


def _clip(parts: list[tuple[float, float | None]]) -> io.BytesIO:
    """A multipart clip: (X-Timestamp, send-grab lag or None for no extra headers) per part."""
    body = _jpeg()
    out = b""
    for i, (sent, lag) in enumerate(parts):
        head = f"Content-Type: image/jpeg\r\nContent-Length: {len(body)}\r\n"
        head += f"X-Timestamp: {sent:.6f}\r\n"
        if lag is not None:
            send = 7000.0 + i * 0.1
            head += (
                f"X-UStreamer-Send-Time: {send:.6f}\r\nX-UStreamer-Grab-Time: {send - lag:.6f}\r\n"
            )
        out += b"--boundarydonotcross\r\n" + head.encode() + b"\r\n" + body + b"\r\n"
    return io.BytesIO(out)


def _frames() -> StereoFrames:
    rectifier = Rectifier.from_calibration(
        ideal_stereo_calibration(16, 12, fx=20.0, baseline_m=0.061)
    )
    return StereoFrames(SideBySide(upside_down=True), rectifier, "camera_optical")


def test_three_parts_become_twelve_messages_dated_by_the_capture() -> None:
    writer = FakeWriter()
    report = TOOL.convert(
        _clip([(1000.30, 0.050), (1000.40, 0.003), (1000.50, 0.048)]), _frames(), writer
    )
    assert report.frames == 3 and report.unstamped == 0
    assert len(writer.rows) == 12
    topics = [row[0] for row in writer.rows[:4]]
    assert topics == list(STEREO_TOPICS)
    expected = (1000.25, 1000.397, 1000.452)
    for i, stamp in enumerate(expected):
        rows = writer.rows[4 * i : 4 * i + 4]
        assert {row[2] for row in rows} == {round(stamp * 1e9)}, "one stamp across the four"
        for _topic, msg, _ns in rows:
            assert stamp_seconds(msg.header.stamp) == pytest.approx(stamp, abs=1e-6)
            assert msg.header.frame_id == "camera_optical"
    right_info = writer.rows[3][1]
    assert right_info.p[3] == pytest.approx(-20.0 * 0.061), "P[0,3] = -fx * baseline"
    assert writer.rows[2][1].encoding == "mono8" and writer.rows[0][1].encoding == "bgr8"
    assert "send-grab median/p90 48/50 ms" in report.text()


def test_a_clip_without_grab_headers_is_counted_and_refused_when_required() -> None:
    writer = FakeWriter()
    report = TOOL.convert(_clip([(5.0, None), (5.1, None)]), _frames(), writer)
    assert report.frames == 2 and report.unstamped == 2, "send stamps, counted"
    assert stamp_seconds(writer.rows[0][1].header.stamp) == pytest.approx(5.0)
    assert "no grab headers" in report.text()
    with pytest.raises(TOOL.NoGrabStampsError):
        TOOL.convert(_clip([(5.0, None)]), _frames(), FakeWriter(), require_grab=True)


def test_start_and_end_cut_by_seconds_after_the_first_frame(tmp_path: Path) -> None:
    writer = FakeWriter()
    parts = [(100.0 + 0.1 * i, 0.002) for i in range(10)]
    report = TOOL.convert(_clip(parts), _frames(), writer, start_s=0.25, end_s=0.55)
    assert report.frames == 3 and report.skipped == 7
    clip = tmp_path / "0512_cam.mjpeg"
    clip.write_bytes(_clip([(1.0, None)]).getvalue())
    assert TOOL.default_out(clip) == tmp_path / "0512_cam.bag"
    assert not TOOL.first_part_has_grab(clip)
    assert TOOL.main([str(clip), "--require-grab"]) == 3, "refused before any bag exists"
    assert not (tmp_path / "0512_cam.bag").exists()
