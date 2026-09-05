"""A reconnecting JSON-lines TCP client: the plumbing both bridges share.

The board's servers publish forever and go away without warning (a service
restart, a wedged bus, a reboot). A ROS node must survive that without dying
and without ever blocking its executor, so the socket lives here in a daemon
thread: connect, read, decode, hand each message to a callback; on any failure
sleep a growing backoff and try again. Nothing in this module imports ROS.

Ownership of the two threads is worth stating once: the reader thread calls
``on_message``, so whatever a node does there must be thread-safe (both
bridges just queue). ``send`` is called from the ROS thread and never blocks
on a dead link — it drops the line and says so.
"""

from __future__ import annotations

import contextlib
import socket
import threading
from collections.abc import Callable
from typing import Any

from pepin_bringup.protocol import LineReader

Connector = Callable[[tuple[str, int]], socket.socket]

_RECV_BYTES = 4096
_RECV_TIMEOUT_S = 0.2  # how often the reader checks whether it was asked to stop


def tcp_connect(address: tuple[str, int]) -> socket.socket:
    """Default connector: a TCP socket to ``address`` with a 2 s connect timeout."""
    return socket.create_connection(address, timeout=2.0)


class JsonLineLink:
    """A background TCP link to a JSON-lines server: messages out to a callback, lines in."""

    def __init__(
        self,
        host: str,
        port: int,
        on_message: Callable[[dict[str, Any]], None],
        *,
        name: str,
        min_backoff_s: float = 0.5,
        max_backoff_s: float = 5.0,
        connector: Connector | None = None,
    ) -> None:
        """Prepare a link to ``host:port`` called ``name`` in logs; nothing connects until start."""
        self.name = name
        self._address = (host, port)
        self._on_message = on_message
        self._min_backoff_s = min_backoff_s
        self._max_backoff_s = max_backoff_s
        self._connect = connector or tcp_connect
        self._lock = threading.Lock()
        self._socket: socket.socket | None = None
        self._connected = False
        self._reported: bool | None = None
        self._status_change: tuple[bool, str] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def connected(self) -> bool:
        """Whether a socket to the server is open right now."""
        with self._lock:
            return self._connected

    def start(self) -> None:
        """Start the reader thread; it connects, and reconnects, on its own."""
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=f"link-{self.name}", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the reader thread to finish and close the socket; safe to call twice."""
        self._stop.set()
        with self._lock:
            sock = self._socket
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
        thread, self._thread = self._thread, None
        if thread is not None:
            thread.join(timeout=1.0)

    def send(self, line: bytes) -> bool:
        """Write one line to the server; ``False`` means the link was down and it was dropped."""
        with self._lock:
            sock = self._socket
        if sock is None:
            return False
        try:
            sock.sendall(line)
        except OSError:
            return False
        return True

    def take_status_change(self) -> tuple[bool, str] | None:
        """The ``(connected, detail)`` the link last changed to, once, or ``None`` if unchanged.

        Repeated failures of the same kind report once, so a node can log every
        call and still not flood the console while the server is down.
        """
        with self._lock:
            change, self._status_change = self._status_change, None
        return change

    def _run(self) -> None:
        """Reader thread: connect, read until the link breaks, back off, repeat."""
        backoff_s = self._min_backoff_s
        while not self._stop.is_set():
            try:
                sock = self._connect(self._address)
            except OSError as exc:
                self._set_status(False, f"{self.name} unreachable at {self._where()}: {exc}")
                self._stop.wait(backoff_s)
                backoff_s = min(backoff_s * 2.0, self._max_backoff_s)
                continue
            backoff_s = self._min_backoff_s
            self._set_status(True, f"{self.name} connected at {self._where()}")
            self._read_until_closed(sock)
        self._set_status(False, f"{self.name} link closed")

    def _read_until_closed(self, sock: socket.socket) -> None:
        """Feed every line of one connection to the callback until it breaks or we stop."""
        reader = LineReader()
        with self._lock:
            self._socket = sock
        try:
            sock.settimeout(_RECV_TIMEOUT_S)
            while not self._stop.is_set():
                try:
                    chunk = sock.recv(_RECV_BYTES)
                except TimeoutError:
                    continue
                if not chunk:
                    self._set_status(False, f"{self.name} closed the connection")
                    return
                for message in reader.feed(chunk):
                    self._on_message(message)
        except OSError as exc:
            self._set_status(False, f"{self.name} read failed: {exc}")
        finally:
            with self._lock:
                self._socket = None
                self._connected = False
            with contextlib.suppress(OSError):
                sock.close()

    def _set_status(self, connected: bool, detail: str) -> None:
        """Record the link's state; keep the first ``detail`` of each change for the node to log."""
        with self._lock:
            self._connected = connected
            if connected != self._reported:
                self._reported = connected
                self._status_change = (connected, detail)

    def _where(self) -> str:
        """The address as ``host:port``, for log lines."""
        return f"{self._address[0]}:{self._address[1]}"
