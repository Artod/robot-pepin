"""Read an MJPEG stream part by part, keeping the headers the camera server sends with each frame.

ustreamer on the board writes a multipart stream where every JPEG comes with its capture time
(``X-Timestamp``, the board's clock, the same clock that stamps the lidar). OpenCV's reader
throws those headers away and a frame gets stamped when the laptop happened to decode it, a
few hundred milliseconds late: while the cart turns at half a radian a second, that is a
picture placed ten degrees wrong. This reader keeps the headers.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import IO

CHUNK = 65536


def parts(stream: IO[bytes]) -> Iterator[tuple[dict[str, str], bytes]]:
    """Yield (headers, body) for every part of a multipart stream until it ends. Header names
    are lower-cased; the boundary is whatever line starts the first part."""
    buffer = b""
    boundary: bytes | None = None
    while True:
        if boundary is None:
            line_end = buffer.find(b"\r\n")
            if line_end < 0:
                more = stream.read(CHUNK)
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
            more = stream.read(CHUNK)
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
            more = stream.read(CHUNK)
            if not more:
                return
            buffer += more
        yield headers, buffer[body_start : body_start + length]
        buffer = buffer[body_start + length :]
        next_boundary = buffer.find(boundary)
        if next_boundary < 0:
            more = stream.read(CHUNK)
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


def capture_time(headers: dict[str, str]) -> float | None:
    """ustreamer's capture time of a part in seconds, or ``None`` when it did not say."""
    try:
        return float(headers["x-timestamp"])
    except (KeyError, ValueError):
        return None
