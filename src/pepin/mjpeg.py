"""Read an MJPEG stream part by part, keeping the headers the camera server sends with each frame.

ustreamer on the board writes a multipart stream where every JPEG comes with a time stamp
(``X-Timestamp``, the board's realtime clock, the same clock that stamps the lidar). OpenCV's
reader throws those headers away and a frame gets stamped when the laptop happened to decode it,
a few hundred milliseconds late: while the cart turns at half a radian a second, that is a
picture placed ten degrees wrong. This reader keeps the headers.

``X-Timestamp`` is the SEND time (ustreamer's ``us_get_now_real()`` at the write to the client),
1-68 ms after the V4L2 capture, bimodal (~1-5 or ~50 ms) and its median moving 4-48 ms between
windows (measured 2026-10-02, scratch/head_imu/ustreamer_stamps.py). The capture stamp comes only
with ``?extra_headers=1`` on the URL: ``X-UStreamer-Grab-Time`` (5.4; ``...-Grab-Begin-Time`` in
6.x) on the board's CLOCK_MONOTONIC beside ``X-UStreamer-Send-Time`` on the same clock, so the
capture on the realtime clock is ``grab + (X-Timestamp - send)`` (:func:`capture_time` with
``mode="grab"``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO

CHUNK = 65536


def parts(stream: IO[bytes]) -> Iterator[tuple[dict[str, str], bytes]]:
    """Yield (headers, body) for every part of a multipart stream until it ends. Header names
    are lower-cased; the boundary is whatever line starts the first part.

    The stream is read with ``read1`` where it has one (an HTTP response, a buffered file): a
    plain ``read(CHUNK)`` blocks until CHUNK bytes are in, so a frame that ends mid-chunk waited
    for the NEXT frame's first bytes — 55-70 ms of latency on every frame at 10 fps (arrival -
    send 129-130 ms with read, 58-59 with read1, 2026-10-07, scratch/vio_rate/reader_latency.py).
    """
    read = getattr(stream, "read1", stream.read)
    buffer = b""
    boundary: bytes | None = None
    while True:
        if boundary is None:
            line_end = buffer.find(b"\r\n")
            if line_end < 0:
                more = read(CHUNK)
                if not more:
                    return
                buffer += more
                continue
            candidate = buffer[:line_end].strip()
            if candidate.startswith(b"--"):
                boundary = candidate
            buffer = buffer[line_end + 2 :]
            continue
        head_end = buffer.find(b"\r\n\r\n")
        if head_end < 0:
            more = read(CHUNK)
            if not more:
                return
            buffer += more
            continue
        head = buffer[:head_end].decode("latin-1")
        headers: dict[str, str] = {}
        for line in head.split("\r\n"):
            if ":" in line:
                name, value = line.split(":", 1)
                headers[name.strip().lower()] = value.strip()
        length = int(headers.get("content-length", "0"))
        body_start = head_end + 4
        while len(buffer) < body_start + length:
            more = read(CHUNK)
            if not more:
                return
            buffer += more
        yield headers, buffer[body_start : body_start + length]
        buffer = buffer[body_start + length :]
        next_boundary = buffer.find(boundary)
        if next_boundary < 0:
            more = read(CHUNK)
            if not more:
                return
            buffer += more
            next_boundary = buffer.find(boundary)
            if next_boundary < 0:
                continue
        line_end = buffer.find(b"\r\n", next_boundary)
        buffer = (
            buffer[line_end + 2 :] if line_end >= 0 else buffer[next_boundary + len(boundary) :]
        )


STAMP_MODES = ("send", "grab")
GRAB_HEADERS = ("x-ustreamer-grab-time", "x-ustreamer-grab-begin-time")
SEND_HEADER = "x-ustreamer-send-time"


def capture_time(headers: dict[str, str], mode: str = "send") -> float | None:
    """A part's time stamp in seconds on the board's realtime clock, or ``None`` without one.

    ``send``: ``X-Timestamp`` as ustreamer wrote it (the send time). ``grab``: the V4L2 capture
    moved onto the realtime clock, ``grab + (X-Timestamp - send)``, falling back to
    ``X-Timestamp`` when the extra headers are missing (:func:`has_grab` tells which).
    """
    if mode not in STAMP_MODES:
        raise ValueError(f"a stamp mode is one of {STAMP_MODES}, not {mode!r}")
    sent = _seconds(headers, "x-timestamp")
    if mode == "send" or sent is None:
        return sent
    grab, send = _grab(headers), _seconds(headers, SEND_HEADER)
    if grab is None or send is None:
        return sent
    return grab + (sent - send)


def has_grab(headers: dict[str, str]) -> bool:
    """Whether a part carries the capture stamp (ustreamer's ``?extra_headers=1``)."""
    return (
        _grab(headers) is not None
        and _seconds(headers, SEND_HEADER) is not None
        and _seconds(headers, "x-timestamp") is not None
    )


def send_lag_s(headers: dict[str, str]) -> float | None:
    """How long after its capture a part was sent (``send - grab``, seconds), or ``None``."""
    grab, send = _grab(headers), _seconds(headers, SEND_HEADER)
    if grab is None or send is None:
        return None
    return send - grab


def _grab(headers: dict[str, str]) -> float | None:
    """The capture stamp under either of ustreamer's spellings."""
    for name in GRAB_HEADERS:
        value = _seconds(headers, name)
        if value is not None:
            return value
    return None


def _seconds(headers: dict[str, str], name: str) -> float | None:
    """One header as seconds, or ``None`` when absent or unreadable."""
    try:
        return float(headers[name])
    except (KeyError, ValueError):
        return None
