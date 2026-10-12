"""The sender inside camera_stream: the rectified left eye to the perception service.

camera_stream's pump hands each published frame to :meth:`FrameFeed.offer`, which costs it a
reference put into a one-slot mailbox (the rectifier returns fresh arrays, so nothing changes
under the sender) and nothing else: the frame goes out on the feed's own thread, which encodes
it to JPEG (``quality``) and POSTs it on a kept-alive connection with a short timeout. A frame
offered while the previous one is still waiting replaces it (counted ``dropped``): a slow or
busy service costs frames, never the pump's time. ``hz`` caps the rate (a frame offered sooner
than three quarters of a period after the last one taken is passed over, ``paced``). A refused
connection (no service) backs off ``backoff_s`` and counts ``refused``; a timeout or an HTTP
error costs that one frame (``failed``).

The request: ``POST /frame`` with the JPEG body and the headers ``X-Stamp`` (the board's
exposure stamp, seconds), ``X-Frame`` (this sender's number), ``X-Run`` (this sender's start),
``X-Optics`` (:meth:`pepin.perception.record.Optics.header`) and ``X-Brightness`` (the frame's
mean luma). The service answers 204 at once: the sender never waits for a detection.

No ROS and no detector here: the container imports this file alone.
"""

from __future__ import annotations

import http.client
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from pepin.telemetry import LatencyTracker

DEFAULT_HZ = 20.0
DEFAULT_QUALITY = 90
DEFAULT_TIMEOUT_S = 0.3
BACKOFF_S = 2.0
PACE_SLACK = 0.75  # a frame this share of a period after the last one taken still counts
# What a kept-alive connection the service closed raises on its next use (and what "nobody
# listens" raises through Docker Desktop's port forwarding, which accepts and then resets).
STALE = (ConnectionResetError, BrokenPipeError, ConnectionAbortedError, http.client.BadStatusLine)
# What means the service is not there: the feed backs off instead of paying a connect per frame.
AWAY = (ConnectionRefusedError, *STALE)


@dataclass(frozen=True)
class Outgoing:
    """One frame waiting in the mailbox: the picture (BGR), its board stamp, the optics header,
    its brightness and this sender's number for it."""

    image: Any
    stamp: float
    optics: str
    brightness: float
    number: int


class Post:
    """``POST /frame`` on one kept-alive connection; any failure drops the connection."""

    def __init__(self, url: str, timeout_s: float) -> None:
        """``url``: the service's ``http://host:port``."""
        if not url.startswith("http://"):
            raise ValueError(f"the perception service URL starts with http://, not {url!r}")
        host, _, port = url[len("http://") :].rstrip("/").partition(":")
        self.host, self.port = host, int(port) if port else 80
        self.timeout_s = float(timeout_s)
        self._conn: http.client.HTTPConnection | None = None

    def __call__(self, body: bytes, headers: dict[str, str]) -> int:
        """The answer's status; raises ``OSError`` / ``http.client.HTTPException``. A kept-alive
        connection the service dropped since the last frame (it restarted) is tried once more
        on a fresh one before anything is raised."""
        reused = self._conn is not None
        try:
            return self._exchange(body, headers)
        except STALE:
            if not reused:
                raise
        return self._exchange(body, headers)

    def _exchange(self, body: bytes, headers: dict[str, str]) -> int:
        if self._conn is None:
            self._conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout_s)
        try:
            self._conn.request("POST", "/frame", body=body, headers=headers)
            response = self._conn.getresponse()
            response.read()
            return response.status
        except (OSError, http.client.HTTPException):
            self.close()
            raise

    def close(self) -> None:
        """Drop the connection; the next call opens a new one."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None


def encode_jpeg(image: Any, quality: int) -> bytes:
    """A BGR picture as JPEG bytes at ``quality``."""
    import cv2

    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise ValueError("JPEG encoding failed")
    return bytes(buf.tobytes())


class FrameFeed:
    """The one-slot mailbox and the thread that empties it (see the module)."""

    def __init__(
        self,
        url: str,
        *,
        quality: int = DEFAULT_QUALITY,
        hz: float = DEFAULT_HZ,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        backoff_s: float = BACKOFF_S,
        clock: Callable[[], float] = time.monotonic,
        post: Callable[[bytes, dict[str, str]], int] | None = None,
        encode: Callable[[Any, int], bytes] = encode_jpeg,
        start: bool = True,
    ) -> None:
        """``post`` replaces the HTTP call (tests); ``start`` False leaves the thread to
        :meth:`start` (or :meth:`send_one`, which the tests call)."""
        self.url = url
        self.quality, self.hz, self.backoff_s = int(quality), float(hz), float(backoff_s)
        self._clock = clock
        self._post = post if post is not None else Post(url, timeout_s)
        self._encode = encode
        self.run = f"{time.time():.3f}"
        self._wake = threading.Condition()
        self._waiting: Outgoing | None = None
        self._stop = threading.Event()
        self._last_taken: float | None = None
        self._down_until = -float("inf")
        self._number = 0
        self._thread = threading.Thread(target=self._loop, name="perception-feed", daemon=True)
        self.counts = {"offered": 0, "sent": 0, "dropped": 0, "paced": 0, "refused": 0,
                       "failed": 0, "down": 0}  # fmt: skip
        self.last_error = ""
        self.encode_ms = LatencyTracker("encode")
        self.post_ms = LatencyTracker("post")
        self._window_since = clock()
        self._window_sent = 0
        if start:
            self.start()

    def start(self) -> None:
        """Start the sending thread."""
        self._thread.start()

    def offer(self, image: Any, stamp: float, optics: str, brightness: float = 0.0) -> None:
        """Hand over one frame (the pump's whole cost): it replaces one still waiting."""
        now = self._clock()
        with self._wake:
            self.counts["offered"] += 1
            if now < self._down_until:
                self.counts["down"] += 1
                return
            period = 1.0 / self.hz if self.hz > 0 else 0.0
            if self._last_taken is not None and now - self._last_taken < PACE_SLACK * period:
                self.counts["paced"] += 1
                return
            self._last_taken = now
            if self._waiting is not None:
                self.counts["dropped"] += 1
            self._number += 1
            self._waiting = Outgoing(image, float(stamp), optics, float(brightness), self._number)
            self._wake.notify()

    def send_one(self, timeout_s: float = 0.0) -> bool:
        """Send the frame waiting, if one comes within ``timeout_s``; whether one was sent
        (or tried)."""
        with self._wake:
            if self._waiting is None and timeout_s > 0:
                self._wake.wait_for(
                    lambda: self._waiting is not None or self._stop.is_set(), timeout=timeout_s
                )
            item, self._waiting = self._waiting, None
        if item is None:
            return False
        self._send(item)
        return True

    def close(self, timeout_s: float = 1.0) -> None:
        """Stop the thread and drop the connection."""
        self._stop.set()
        with self._wake:
            self._wake.notify_all()
        if self._thread.is_alive():
            self._thread.join(timeout_s)
        close = getattr(self._post, "close", None)
        if callable(close):
            close()

    def report(self) -> str:
        """``perception 20.0/s sent, 0 dropped, 0 refused, 0 failed, encode 0.9/1.1 ms, post
        0.4/0.6 ms`` for the window since the last call (encode and post: median/p95)."""
        now = self._clock()
        with self._wake:
            sent = self.counts["sent"]
            rate = (sent - self._window_sent) / max(now - self._window_since, 1e-6)
            self._window_since, self._window_sent = now, sent
            counts = dict(self.counts)
        enc, post = self.encode_ms.summary(), self.post_ms.summary()
        line = (
            f"perception {rate:.1f}/s sent ({counts['sent']} in all, q{self.quality},"
            f" <= {self.hz:g}/s), {counts['dropped']} dropped, {counts['refused']} refused,"
            f" {counts['failed']} failed, encode {enc.median_ms:.1f}/{enc.p95_ms:.1f} ms,"
            f" post {post.median_ms:.1f}/{post.p95_ms:.1f} ms"
        )
        if now < self._down_until:
            line += f", service down ({self.last_error}): retry in {self._down_until - now:.0f} s"
        elif counts["failed"] and self.last_error:
            line += f" (last: {self.last_error})"
        return line

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.send_one(timeout_s=0.5)

    def _send(self, item: Outgoing) -> None:
        t0 = time.perf_counter()
        try:
            body = self._encode(item.image, self.quality)
        except Exception as exc:  # a picture OpenCV cannot write costs that frame
            self.counts["failed"] += 1
            self.last_error = f"encode: {exc}"[:160]
            return
        t1 = time.perf_counter()
        headers = {
            "Content-Type": "image/jpeg",
            "X-Stamp": f"{item.stamp:.6f}",
            "X-Frame": str(item.number),
            "X-Run": self.run,
            "X-Optics": item.optics,
            "X-Brightness": f"{item.brightness:.1f}",
        }
        try:
            status = self._post(body, headers)
        except AWAY as exc:
            with self._wake:
                self.counts["refused"] += 1
                self._down_until = self._clock() + self.backoff_s
                self.last_error = f"{type(exc).__name__}"
            return
        except (OSError, http.client.HTTPException) as exc:
            self.counts["failed"] += 1
            self.last_error = f"{type(exc).__name__}: {exc}"[:160]
            return
        t2 = time.perf_counter()
        self.encode_ms.add(t1 - t0)
        self.post_ms.add(t2 - t1)
        if status in (200, 204):
            self.counts["sent"] += 1
        else:
            self.counts["failed"] += 1
            self.last_error = f"HTTP {status}"
