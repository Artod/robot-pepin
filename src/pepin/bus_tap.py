"""The servo bus tap: every byte between the base server and the servos, kept.

The base server reaches the servo bus through ser2net on the board (TCP, one client at a time).
The tap sits between them as a proxy on the board: the server connects to the tap, the tap to
ser2net, and every chunk of bytes in either direction is copied, stamped, to two places at once:
a file on the board (rotated by size) and a UDP line to a sink on the laptop
(:mod:`ros.tools.bus_tap_sink`), so the record survives the robot's power switch down to the
last packets before it. The hop is loopback: microseconds on a 20 ms tick.

One line per chunk::

    <epoch s> <monotonic s> <dir> <n> <hex>

``dir`` is ``>`` for bytes the server sent to the bus and ``<`` for bytes the bus answered.
:func:`frame` turns one direction's bytes back into packets, checksum checked, and
:func:`describe` names each packet's instruction, registers and values (the decoder,
``ros/tools/bus_tap_decode.py``, prints a log that way).

    /opt/pepin/bin/python -m pepin.bus_tap --listen 3334 --upstream 3333 \\
        --log-dir /var/log/pepin/bus-tap --sink 10.0.0.167:3341

The tap serves one client, as ser2net does: a new client ends the pair that was served.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import os
import selectors
import socket
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pepin.feetech import (
    BROADCAST_ID,
    INST_PING,
    INST_READ,
    INST_SYNC_READ,
    INST_SYNC_WRITE,
    INST_WRITE,
    REGISTERS,
    checksum,
    decode_value,
)

logger = logging.getLogger(__name__)

TO_BUS = ">"
FROM_BUS = "<"
HEADER = b"\xff\xff"
INSTRUCTIONS = {
    INST_PING: "PING",
    INST_READ: "READ",
    INST_WRITE: "WRITE",
    INST_SYNC_READ: "SYNC_READ",
    INST_SYNC_WRITE: "SYNC_WRITE",
}
BY_ADDRESS = {r.address: (name, r) for name, r in REGISTERS.items()}
FSYNC_EVERY_S = 1.0
RECONNECT_S = 0.5  # the bus side reopened this often while ser2net is away


# ---- the record --------------------------------------------------------------------------------
class TapLog:
    """The two copies of the stream: a file under ``directory`` (``bus-tap.log``, rotated at
    ``max_bytes`` into ``.1`` .. ``.keep``) and one UDP datagram a line to ``sink``. Either may
    be absent. A write never blocks the proxy: the file is flushed each line and fsynced every
    second, the datagram is fire-and-forget."""

    def __init__(
        self,
        directory: Path | None,
        sink: tuple[str, int] | None,
        *,
        max_bytes: int = 200_000_000,
        keep: int = 5,
    ) -> None:
        self.directory, self.sink, self.max_bytes, self.keep = directory, sink, max_bytes, keep
        self._file: Any = None
        self._size = 0
        self._synced_at = 0.0
        self._udp: socket.socket | None = None
        if directory is not None:
            directory.mkdir(parents=True, exist_ok=True)
            self._open()
        if sink is not None:
            self._udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self._udp.setblocking(False)
        self.lines = 0

    @property
    def path(self) -> Path | None:
        """The current file."""
        return None if self.directory is None else self.directory / "bus-tap.log"

    def _open(self) -> None:
        assert self.path is not None
        self._file = open(self.path, "ab")  # noqa: SIM115 (kept open for the proxy's life)
        self._size = self.path.stat().st_size

    def _rotate(self) -> None:
        assert self.path is not None and self._file is not None
        self._file.close()
        for k in range(self.keep, 0, -1):
            older = self.path.with_suffix(f".log.{k}")
            newer = self.path if k == 1 else self.path.with_suffix(f".log.{k - 1}")
            if newer.exists():
                newer.replace(older)
        self._open()

    def write(self, direction: str, data: bytes, now: float | None = None) -> str:
        """One chunk as a line, to both copies; the line."""
        wall = time.time() if now is None else now
        line = f"{wall:.6f} {time.monotonic():.6f} {direction} {len(data)} {data.hex()}\n"
        raw = line.encode()
        if self._file is not None:
            self._file.write(raw)
            self._file.flush()
            self._size += len(raw)
            if wall - self._synced_at >= FSYNC_EVERY_S:
                with contextlib.suppress(OSError):
                    os.fsync(self._file.fileno())
                self._synced_at = wall
            if self._size >= self.max_bytes:
                self._rotate()
        if self._udp is not None and self.sink is not None:
            with contextlib.suppress(OSError):
                self._udp.sendto(raw, self.sink)
        self.lines += 1
        return line

    def close(self) -> None:
        """Flush and close the file."""
        if self._file is not None:
            self._file.close()
            self._file = None


# ---- the proxy ---------------------------------------------------------------------------------
class Tap:
    """The proxy: listens on ``listen``, opens ``upstream`` for each client, pumps bytes both
    ways through ``log``. :meth:`serve_forever` runs until :meth:`stop`; :meth:`serve_once`
    handles one round of socket events (tests drive it)."""

    def __init__(
        self, listen: tuple[str, int], upstream: tuple[str, int], log: TapLog, *, chunk: int = 4096
    ) -> None:
        self.upstream, self.log, self.chunk = upstream, log, chunk
        self._listener = socket.create_server(listen)
        self._listener.setblocking(False)
        self._sel = selectors.DefaultSelector()
        self._sel.register(self._listener, selectors.EVENT_READ, "accept")
        self._client: socket.socket | None = None
        self._bus: socket.socket | None = None
        self._pending_client: socket.socket | None = None  # a client waiting for the bus
        self._bus_retry_at = 0.0
        self._stopping = False
        self.clients = 0

    @property
    def port(self) -> int:
        """The port the listener took (0 asked for any)."""
        return int(self._listener.getsockname()[1])

    def stop(self) -> None:
        """End :meth:`serve_forever` and close every socket."""
        self._stopping = True
        self._drop("stopping")
        self._sel.unregister(self._listener)
        self._listener.close()
        self.log.close()

    def serve_forever(self) -> None:
        """Pump until :meth:`stop`."""
        while not self._stopping:
            self.serve_once(0.2)

    def serve_once(self, timeout: float = 0.0) -> None:
        """One round: accept, connect the bus for a waiting client, copy what arrived."""
        if self._pending_client is not None and time.monotonic() >= self._bus_retry_at:
            self._connect_bus()
        for key, _mask in self._sel.select(timeout):
            if key.data == "accept":
                self._accept()
            elif key.data == "client":
                self._pump(self._client, self._bus, TO_BUS)
            elif key.data == "bus":
                self._pump(self._bus, self._client, FROM_BUS)

    def _accept(self) -> None:
        try:
            client, peer = self._listener.accept()
        except OSError:
            return
        self._drop("a new client")  # ser2net's kickolduser: the newcomer takes the bus
        client.setblocking(False)
        client.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._pending_client = client
        self._bus_retry_at = 0.0
        self.clients += 1
        logger.info("client %s:%s", *peer[:2])

    def _connect_bus(self) -> None:
        assert self._pending_client is not None
        try:
            bus = socket.create_connection(self.upstream, timeout=1.0)
        except OSError as exc:
            self._bus_retry_at = time.monotonic() + RECONNECT_S
            logger.warning("the bus at %s:%s is not there (%s): retrying", *self.upstream, exc)
            return
        bus.setblocking(False)
        bus.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self._client, self._pending_client, self._bus = self._pending_client, None, bus
        self._sel.register(self._client, selectors.EVENT_READ, "client")
        self._sel.register(self._bus, selectors.EVENT_READ, "bus")
        logger.info("bus %s:%s open: the pair is served", *self.upstream)

    def _pump(self, src: socket.socket | None, dst: socket.socket | None, direction: str) -> None:
        if src is None or dst is None:
            return
        try:
            data = src.recv(self.chunk)
        except OSError:
            data = b""
        if not data:
            self._drop("the client closed" if direction == TO_BUS else "the bus closed")
            return
        self.log.write(direction, data)
        try:
            dst.sendall(data)
        except OSError:
            self._drop("the other side closed")

    def _drop(self, why: str) -> None:
        """End the served pair (both sockets), and any client still waiting for the bus."""
        for sock in (self._client, self._bus, self._pending_client):
            if sock is None:
                continue
            with contextlib.suppress(KeyError, ValueError):
                self._sel.unregister(sock)
            sock.close()
        if self._client is not None or self._pending_client is not None:
            logger.info("pair ended: %s", why)
        self._client = self._bus = self._pending_client = None


# ---- the decoder -------------------------------------------------------------------------------
@dataclass(frozen=True)
class Packet:
    """One packet of one direction: its bytes (header to checksum), its id, its instruction
    (a status packet's error byte for a reply), its parameters and whether the checksum held;
    ``noise`` is how many bytes were skipped before its header."""

    raw: bytes
    motor_id: int
    instruction: int
    params: bytes
    ok: bool
    noise: int = 0


def frame(data: bytes, carry: bytes = b"") -> tuple[list[Packet], bytes]:
    """Packets in ``carry + data``, and the tail that is not a whole packet yet. Bytes before a
    header are counted as noise on the packet that follows; a header whose length byte cannot
    be is skipped as noise too (the parser's own rule)."""
    buf = carry + data
    packets: list[Packet] = []
    noise = 0
    i = 0
    while True:
        at = buf.find(HEADER, i)
        if at < 0:
            # keep a trailing 0xFF of the unparsed rest: it may be the first half of the next
            # header (a 0xFF just consumed as a packet's checksum is not kept)
            rest = buf[i:]
            tail = rest[-1:] if rest[-1:] == b"\xff" else b""
            noise += len(rest) - len(tail)
            return packets, tail
        noise += at - i
        if at + 3 <= len(buf) and buf[at + 2] == 0xFF:
            # three 0xFF in a row: the first is a reply's checksum (or a chunk's carried byte)
            # and the header starts one byte later; 0xFF is never an id (0xFE is the broadcast)
            noise += 1
            i = at + 1
            continue
        if at + 4 > len(buf):
            return packets, buf[at:]
        length = buf[at + 3]
        if length < 2:
            i = at + 2
            noise += 2
            continue
        end = at + 4 + length
        if end > len(buf):
            return packets, buf[at:]
        body = buf[at + 2 : end - 1]
        packets.append(
            Packet(
                buf[at:end],
                buf[at + 2],
                buf[at + 4],
                buf[at + 5 : end - 1],
                checksum(body) == buf[end - 1],
                noise,
            )
        )
        noise = 0
        i = end


def _values(address: int, size: int, data: bytes) -> str:
    """``address``'s registers' values in ``data`` (``size`` bytes a servo), by name."""
    out = []
    at = address
    k = 0
    while k < size:
        named = BY_ADDRESS.get(at)
        if named is None:
            out.append(f"@{at}={data[k]}")
            at, k = at + 1, k + 1
            continue
        name, reg = named
        chunk = data[k : k + reg.size]
        whole = len(chunk) == reg.size
        value = decode_value(chunk, reg) if whole else int.from_bytes(chunk, "little")
        out.append(f"{name}={value}")
        at, k = at + reg.size, k + reg.size
    return " ".join(out)


def describe(packet: Packet, direction: str) -> str:
    """One packet in words: who, what, which registers, the values; ``CHK FAIL`` and
    ``noise N`` when the bytes were not clean."""
    flags = "" if packet.ok else " CHK FAIL"
    if packet.noise:
        flags += f" noise {packet.noise}"
    p = packet.params
    if direction == FROM_BUS:
        return f"id {packet.motor_id} status err=0x{packet.instruction:02x} {p.hex()}{flags}"
    name = INSTRUCTIONS.get(packet.instruction, f"0x{packet.instruction:02x}")
    who = "ALL" if packet.motor_id == BROADCAST_ID else f"id {packet.motor_id}"
    if packet.instruction == INST_SYNC_WRITE and len(p) >= 2:
        address, size = p[0], p[1]
        rows = []
        k = 2
        while k + 1 + size <= len(p):
            rows.append(f"id {p[k]}: {_values(address, size, p[k + 1 : k + 1 + size])}")
            k += 1 + size
        return f"{who} {name} @{address}+{size} " + "; ".join(rows) + flags
    if packet.instruction == INST_SYNC_READ and len(p) >= 2:
        reg = BY_ADDRESS.get(p[0], (f"@{p[0]}", None))[0]
        return f"{who} {name} {reg} x{p[1]} ids {list(p[2:])}{flags}"
    if packet.instruction == INST_WRITE and len(p) >= 2:
        return f"{who} {name} {_values(p[0], len(p) - 1, p[1:])}{flags}"
    if packet.instruction == INST_READ and len(p) >= 2:
        reg = BY_ADDRESS.get(p[0], (f"@{p[0]}", None))[0]
        return f"{who} {name} {reg} x{p[1]}{flags}"
    return f"{who} {name} {p.hex()}{flags}"


@dataclass
class Stream:
    """One direction's framer state across the log's lines."""

    direction: str
    carry: bytes = b""
    packets: list[tuple[float, Packet]] = field(default_factory=list)

    def feed(self, when: float, data: bytes) -> list[tuple[float, Packet]]:
        """The packets completed by ``data``, stamped ``when``."""
        found, self.carry = frame(data, self.carry)
        stamped = [(when, pk) for pk in found]
        self.packets.extend(stamped)
        return stamped


def parse_line(line: str) -> tuple[float, float, str, bytes] | None:
    """A log line as ``(epoch, monotonic, direction, bytes)``; None for anything else."""
    parts = line.split()
    if len(parts) != 5 or parts[2] not in (TO_BUS, FROM_BUS):
        return None
    try:
        return float(parts[0]), float(parts[1]), parts[2], bytes.fromhex(parts[4])
    except ValueError:
        return None


# ---- the command line --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    """Run the proxy until killed."""
    parser = argparse.ArgumentParser(description="the servo bus tap (see the module's doc)")
    parser.add_argument("--listen", type=int, default=3334, help="the port the base server uses")
    parser.add_argument("--upstream", default="127.0.0.1:3333", help="ser2net, host:port")
    parser.add_argument("--log-dir", type=Path, default=None, help="the board's copy")
    parser.add_argument("--sink", default=None, help="the laptop's copy, host:port (UDP)")
    parser.add_argument("--max-mb", type=int, default=200, help="rotate the file at this size")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname).1s %(message)s")
    host, port = args.upstream.rsplit(":", 1)
    sink = None
    if args.sink:
        sink_host, sink_port = args.sink.rsplit(":", 1)
        sink = (sink_host, int(sink_port))
    log = TapLog(args.log_dir, sink, max_bytes=args.max_mb * 1_000_000)
    tap = Tap(("0.0.0.0", args.listen), (host, int(port)), log)
    logger.info(
        "bus tap on %d -> %s; file %s; sink %s", args.listen, args.upstream, log.path, args.sink
    )
    try:
        tap.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        tap.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
