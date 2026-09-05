"""JSON lines over TCP with a background reader: the shape every stream from the board shares.

The board publishes ToF ranges (:3335) and base state (:3336) the same way —
one JSON object per line, forever — and the laptop wants the same things
from both: connect, reconnect on loss, never block the caller, say how old
the newest message is, and send a line back when there is something to say.
That plumbing lives here once; :class:`pepin.tof.TofClient` and
:class:`pepin.base_link.BaseClient` only decode messages.
"""

from __future__ import annotations

import contextlib
import json
import logging
import socket
import threading
import time
from collections.abc import Callable
from typing import Any, Self

logger = logging.getLogger(__name__)

Connector = Callable[[tuple[str, int]], socket.socket]


def tcp_connect(address: tuple[str, int]) -> socket.socket:
    """Default connector: a TCP socket to ``address`` with a 2 s connect timeout."""
    return socket.create_connection(address, timeout=2.0)


class JsonLinesClient:
    """Reads a JSON-lines stream in a daemon thread and hands each message to ``_ingest``.

    Subclasses implement :meth:`_ingest` (decode, store under their own lock)
    and expose typed accessors. ``age_s`` is the freshness of the newest
    message; ``send`` posts a message back and drops it, with a throttled
    warning, while the link is down. ``connector`` is replaceable so tests can
    feed a scripted socket.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        name: str,
        connector: Connector | None = None,
        retry_s: float = 1.0,
    ) -> None:
        """Prepare a client for ``host:port``; nothing connects until :meth:`start`."""
        self.name = name
        self._address = (host, port)
        self._connector = connector or tcp_connect
        self._retry_s = retry_s
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._stamp = 0.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name=f"{name}-reader")
        self._warned_down = 0.0
        self.connected = False

    def start(self) -> Self:
        """Begin reading in a daemon thread; returns self so it chains."""
        self._thread.start()
        return self

    def close(self) -> None:
        """Ask the reader to stop and wait (up to 2 s) for the socket to be released."""
        self._stop.set()
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)
        self._drop_socket()

    def age_s(self, now: float | None = None) -> float:
        """Seconds since the newest message arrived; infinite before the first."""
        now = time.monotonic() if now is None else now
        return now - self._stamp if self._stamp else float("inf")

    def send(self, message: dict[str, Any]) -> None:
        """Post one message to the board; dropped (and logged, throttled) while the link is down."""
        with self._send_lock:
            sock = self._sock
            if sock is None:
                now = time.monotonic()
                if now - self._warned_down > 2.0:
                    logger.warning("%s link down: message %s dropped", self.name, message)
                    self._warned_down = now
                return
            try:
                sock.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())
            except OSError as exc:
                logger.warning("%s link send failed (%s); reconnecting", self.name, exc)
                self._drop_socket_locked()

    def feed(self, message: dict[str, Any]) -> None:
        """Deliver one decoded message as if it came off the wire: stamp it, then ``_ingest``."""
        self._stamp = time.monotonic()
        self._ingest(message)

    def _ingest(self, message: dict[str, Any]) -> None:
        """Decode and store one message; subclasses own the lock around their state."""
        raise NotImplementedError

    # -- internals ----------------------------------------------------------

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._stream()
            except OSError as exc:
                logger.warning("%s stream lost (%s); reconnecting", self.name, exc)
            except Exception:
                logger.exception("%s reader crashed; reconnecting", self.name)
            self._drop_socket()
            self._stop.wait(self._retry_s)

    def _stream(self) -> None:
        sock = self._connector(self._address)
        sock.settimeout(1.0)
        with self._send_lock:
            self._sock = sock
        self.connected = True
        logger.info("%s stream connected to %s:%d", self.name, *self._address)
        buffer = b""
        while not self._stop.is_set():
            try:
                chunk = sock.recv(4096)
            except TimeoutError:
                continue
            if not chunk:
                raise ConnectionError("stream closed by the board")
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if line.strip():
                    self.feed(json.loads(line))

    def _drop_socket(self) -> None:
        with self._send_lock:
            self._drop_socket_locked()

    def _drop_socket_locked(self) -> None:
        sock, self._sock = self._sock, None
        self.connected = False
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()
