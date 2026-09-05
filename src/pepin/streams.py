"""JSON lines over TCP with a background reader: the shape every stream from the board shares.

The board publishes ToF ranges (:3335) and base state (:3336) the same way —
one JSON object per line, forever — and the laptop wants the same things
from both: connect, reconnect on loss, never block the caller, say how old
the newest message is, and send a line back when there is something to say.
That plumbing lives here once; :class:`pepin.tof.TofClient` and
:class:`pepin.base_link.BaseClient` only decode messages. The board side of
the same idea is :class:`JsonLinesServer`: accept clients, read their lines
into one inbox, broadcast lines to all of them — without ever running network
I/O on the caller's thread.
"""

from __future__ import annotations

import contextlib
import json
import logging
import queue
import socket
import threading
import time
from collections.abc import Callable
from typing import Any, Self

logger = logging.getLogger(__name__)

Connector = Callable[[tuple[str, int]], socket.socket]
_DONTWAIT = getattr(socket, "MSG_DONTWAIT", 0)


def tcp_connect(address: tuple[str, int]) -> socket.socket:
    """Default connector: a TCP socket to ``address`` with a 2 s connect timeout."""
    return socket.create_connection(address, timeout=2.0)


class JsonLinesClient:
    """Reads a JSON-lines stream in a daemon thread and hands each message to ``_ingest``.

    Subclasses implement :meth:`_ingest` (decode, store under their own lock)
    and expose typed accessors. ``age_s`` is the freshness of the newest
    message; ``send`` posts a message back without ever blocking the caller
    and drops it, with a throttled warning, while the link is down. An
    unreadable or unexpected line costs that line, not the connection.
    ``connector`` is replaceable so tests can feed a scripted socket.
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
        self._drop_socket()  # unblocks a reader stuck in recv
        if self._thread.is_alive() and threading.current_thread() is not self._thread:
            self._thread.join(timeout=2.0)

    def age_s(self, now: float | None = None) -> float:
        """Seconds since the newest message arrived; infinite before the first."""
        now = time.monotonic() if now is None else now
        return now - self._stamp if self._stamp else float("inf")

    def send(self, message: dict[str, Any]) -> None:
        """Post one message to the board without blocking; dropped (logged) when it cannot go."""
        payload = (json.dumps(message, separators=(",", ":")) + "\n").encode()
        with self._send_lock:
            sock = self._sock
            if sock is None:
                self._warn_dropped(message)
                return
            try:
                sent = sock.send(payload, _DONTWAIT)
            except BlockingIOError:
                self._warn_dropped(message)  # send buffer full: the link is stalling, not dead
                return
            except OSError as exc:
                logger.warning("%s link send failed (%s); reconnecting", self.name, exc)
                self._drop_socket_locked()
                return
            if sent != len(payload):
                # A partial line would corrupt the framing on the board: start over.
                logger.warning("%s link send was partial; reconnecting", self.name)
                self._drop_socket_locked()

    def feed(self, message: dict[str, Any]) -> None:
        """Deliver one decoded message as if it came off the wire: stamp it, then ``_ingest``."""
        self._stamp = time.monotonic()
        self._ingest(message)

    def _ingest(self, message: dict[str, Any]) -> None:
        """Decode and store one message; subclasses own the lock around their state."""
        raise NotImplementedError

    # -- internals ----------------------------------------------------------

    def _warn_dropped(self, message: dict[str, Any]) -> None:
        now = time.monotonic()
        if now - self._warned_down > 2.0:
            logger.warning("%s link down: message %s dropped", self.name, message.get("cmd", "?"))
            self._warned_down = now

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self._stream()
            except OSError as exc:
                if not self._stop.is_set():
                    logger.warning("%s stream lost (%s); reconnecting", self.name, exc)
            except Exception:
                logger.exception("%s reader crashed; reconnecting", self.name)
            self._drop_socket()
            self._stop.wait(self._retry_s)

    def _stream(self) -> None:
        sock = self._connector(self._address)
        if self._stop.is_set():
            sock.close()
            return
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
                    self._deliver(line)

    def _deliver(self, line: bytes) -> None:
        """One wire line into ``feed``; bad lines are logged and skipped, the stream stays up."""
        try:
            message = json.loads(line)
        except ValueError:
            logger.warning("%s: unreadable line dropped (%d bytes)", self.name, len(line))
            return
        try:
            self.feed(message)
        except Exception:
            logger.exception("%s: message rejected: %.120s", self.name, line)

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


# -- server side (runs on the board) ---------------------------------------------------------


def encode(message: dict[str, Any]) -> bytes:
    """One message as a JSON line ready for a socket."""
    return (json.dumps(message, separators=(",", ":")) + "\n").encode()


class ClientConn:
    """One connected client: a reader thread into the shared inbox, a writer with a bounded outbox.

    Sends never run on the server's control thread. A client that stops
    reading fills its outbox (about a second of traffic) and is dropped.
    """

    def __init__(
        self,
        conn: socket.socket,
        peer: str,
        inbox: queue.Queue[tuple[ClientConn, dict[str, Any]]],
        on_close: Callable[[ClientConn], None],
        outbox_size: int = 24,
    ) -> None:
        self.conn = conn
        self.peer = peer
        self._inbox = inbox
        self._on_close = on_close
        self._outbox: queue.Queue[bytes] = queue.Queue(maxsize=outbox_size)
        self._lock = threading.Lock()
        self.alive = True
        threading.Thread(target=self._read, daemon=True, name=f"client-{peer}-r").start()
        threading.Thread(target=self._write, daemon=True, name=f"client-{peer}-w").start()

    def post(self, line: bytes) -> None:
        """Queue one line for this client; a full outbox means a dead or frozen peer."""
        try:
            self._outbox.put_nowait(line)
        except queue.Full:
            logger.warning("client %s is not reading; dropping it", self.peer)
            self.close()

    def close(self) -> None:
        """Shut the socket (unblocks the reader), stop the writer, tell the server; idempotent."""
        with self._lock:
            if not self.alive:
                return
            self.alive = False
        with contextlib.suppress(OSError):
            self.conn.shutdown(socket.SHUT_RDWR)
        self.conn.close()
        with contextlib.suppress(queue.Full):
            self._outbox.put_nowait(b"")  # wakes the writer so it can exit
        self._on_close(self)

    def _read(self) -> None:
        buffer = b""
        try:
            while self.alive:
                chunk = self.conn.recv(4096)
                if not chunk:
                    break
                buffer += chunk
                *lines, buffer = buffer.split(b"\n")
                for line in lines:
                    if not line.strip():
                        continue
                    try:
                        message = json.loads(line)
                    except ValueError:
                        logger.warning("client %s sent an unreadable line; ignored", self.peer)
                        continue
                    if isinstance(message, dict):
                        self._inbox.put((self, message))
                    else:
                        logger.warning("client %s sent a non-object line; ignored", self.peer)
        except OSError as exc:
            logger.info("client %s reader ended: %s", self.peer, exc)
        self.close()

    def _write(self) -> None:
        while True:
            line = self._outbox.get()
            if not line or not self.alive:
                return
            try:
                self.conn.sendall(line)
            except OSError as exc:
                logger.info("client %s writer ended: %s", self.peer, exc)
                self.close()
                return


class JsonLinesServer:
    """Dual-stack TCP server for JSON lines: clients' messages in one inbox, broadcasts to all.

    ``commands()`` drains what clients sent (non-blocking), ``broadcast()``
    queues a message for every client, ``reply()`` for one. When the last
    client leaves, ``on_last_client_left`` is queued as a message the owner
    sees in ``commands()`` — on its own thread, like everything else.
    """

    def __init__(self, port: int, *, on_last_client_left: dict[str, Any] | None = None) -> None:
        """``port`` 0 picks a free one (tests); ``on_last_client_left`` is queued into the inbox."""
        self._requested_port = port
        self._farewell = on_last_client_left
        self._server: socket.socket | None = None
        self._clients: list[ClientConn] = []
        self._lock = threading.Lock()
        self._inbox: queue.Queue[tuple[ClientConn | None, dict[str, Any]]] = queue.Queue()
        self.port = port

    def start(self) -> Self:
        """Bind, listen and start accepting in a daemon thread; returns self so it chains."""
        server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)  # IPv4 clients too
        server.bind(("::", self._requested_port))
        server.listen(4)
        self._server = server
        self.port = server.getsockname()[1]
        threading.Thread(target=self._accept_loop, daemon=True, name="accept").start()
        logger.info("listening on %d", self.port)
        return self

    @property
    def client_count(self) -> int:
        """How many clients are connected right now."""
        with self._lock:
            return len(self._clients)

    def commands(self) -> list[tuple[ClientConn | None, dict[str, Any]]]:
        """Everything clients sent since the last call (client None = the farewell message)."""
        out = []
        while True:
            try:
                out.append(self._inbox.get_nowait())
            except queue.Empty:
                return out

    def broadcast(self, message: dict[str, Any]) -> None:
        """Queue one message for every connected client; never blocks."""
        line = encode(message)
        with self._lock:
            targets = list(self._clients)
        for client in targets:
            client.post(line)

    def reply(self, client: ClientConn | None, message: dict[str, Any]) -> None:
        """Queue one message for one client (no-op for the farewell pseudo-client)."""
        if client is not None and client.alive:
            client.post(encode(message))

    def close(self) -> None:
        """Drop every client and stop listening."""
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            client.close()
        if self._server is not None:
            with contextlib.suppress(OSError):
                self._server.close()

    def _on_close(self, client: ClientConn) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)
            alone = not self._clients
        logger.info("client %s gone", client.peer)
        if alone and self._farewell is not None:
            self._inbox.put((None, dict(self._farewell)))

    def _accept_loop(self) -> None:
        assert self._server is not None
        while True:
            try:
                conn, peer = self._server.accept()
            except OSError:
                return  # closed
            client = ClientConn(conn, peer[0], self._inbox, self._on_close)  # type: ignore[arg-type]
            with self._lock:
                self._clients.append(client)
            logger.info("client %s connected", peer[0])
