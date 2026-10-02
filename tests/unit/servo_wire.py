"""A ser2net port with Feetech servos behind it, for the real FeetechTcpClient.

The base server's tick is a budget of bus round trips, and a Python fake of the bus cannot say
what a transaction costs on the wire: this one can. It listens on a loopback port, parses the
instruction packets the client sends, keeps each servo's control table, and answers like the
servos do — a status packet per id for a read, nothing for a sync write — after a fixed delay
that stands for the USB adapter both ways (``reply_latency_s``; the board measures 4.8-5.6 ms
for the whole wheel read) plus 10 us per byte on the 1 Mbaud wire. A servo in ``silent`` never
answers. Every packet is counted, so a test can hold the number of round trips and bytes a tick
costs as well as its time.
"""

from __future__ import annotations

import socket
import threading
import time
from dataclasses import dataclass, field

from pepin.feetech import (
    BROADCAST_ID,
    HEADER,
    INST_PING,
    INST_READ,
    INST_SYNC_READ,
    INST_SYNC_WRITE,
    INST_WRITE,
    checksum,
)

BYTE_S = 10e-6  # 1 Mbaud, 8N1: ten bits a byte
ACCELERATION = 41  # the ramp a position move runs at, 100 ticks/s^2 a unit
MAX_ACCELERATION = 85  # the firmware's ceiling on it: a larger Acceleration is stored as this
DELIVERED_MAX_ACCELERATION = 50  # what the neck's STS3215 read before anyone wrote it (2026-10-02)


def status(motor_id: int, params: bytes = b"", error: int = 0) -> bytes:
    """One status packet, as a servo sends it."""
    body = bytes([motor_id, len(params) + 2, error]) + params
    return HEADER + body + bytes([checksum(body)])


@dataclass
class WireCounts:
    """What crossed the fake wire: packets by instruction, bytes each way."""

    sync_reads: int = 0
    sync_writes: int = 0
    writes: int = 0
    reads: int = 0
    pings: int = 0
    bytes_in: int = 0
    bytes_out: int = 0
    log: list[tuple[int, bytes]] = field(default_factory=list)  # (instruction, params)


class ServoWire:
    """The fake: :meth:`start` returns the port to connect a client to; :meth:`stop` ends it."""

    def __init__(self, ids: list[int], *, reply_latency_s: float = 0.0045) -> None:
        """``ids`` are the servos on the bus; each starts with a zeroed control table but for the
        acceleration ceiling the servos came with."""
        self.memory = {motor_id: bytearray(128) for motor_id in ids}
        for table in self.memory.values():
            table[MAX_ACCELERATION] = DELIVERED_MAX_ACCELERATION
        self.silent: set[int] = set()
        self.reply_latency_s = reply_latency_s
        self.counts = WireCounts()
        self._listener = socket.create_server(("127.0.0.1", 0))
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._lock = threading.Lock()

    def start(self) -> int:
        """Begin accepting the client (one at a time, like ser2net); returns the port."""
        self._thread.start()
        return int(self._listener.getsockname()[1])

    def stop(self) -> None:
        """Close the port and join the thread."""
        self._stop.set()
        self._listener.close()
        self._thread.join(timeout=2.0)

    def set_word(self, motor_id: int, address: int, value: int) -> None:
        """Put a little-endian 16-bit value into a servo's table (a position, a speed)."""
        with self._lock:
            self.memory[motor_id][address : address + 2] = value.to_bytes(2, "little")

    def word(self, motor_id: int, address: int) -> int:
        """A little-endian 16-bit value out of a servo's table."""
        with self._lock:
            return int.from_bytes(self.memory[motor_id][address : address + 2], "little")

    def byte(self, motor_id: int, address: int) -> int:
        """One byte of a servo's table."""
        with self._lock:
            return self.memory[motor_id][address]

    def _serve(self) -> None:
        self._listener.settimeout(0.1)
        while not self._stop.is_set():
            try:
                connection, _ = self._listener.accept()
            except (TimeoutError, OSError):
                continue
            with connection:
                connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
                self._talk(connection)

    def _talk(self, connection: socket.socket) -> None:
        connection.settimeout(0.1)
        buffer = bytearray()
        while not self._stop.is_set():
            try:
                chunk = connection.recv(4096)
            except TimeoutError:
                continue
            except OSError:
                return
            if not chunk:
                return
            buffer += chunk
            while True:
                packet = _take_packet(buffer)
                if packet is None:
                    break
                reply = self._handle(*packet)
                if reply:
                    time.sleep(self.reply_latency_s + BYTE_S * len(reply))
                    self.counts.bytes_out += len(reply)
                    try:
                        connection.sendall(reply)
                    except OSError:
                        return

    def _handle(self, motor_id: int, instruction: int, params: bytes) -> bytes:
        """Apply one instruction packet; the bytes the servos send back (none for a sync write)."""
        self.counts.bytes_in += len(params) + 6
        self.counts.log.append((instruction, params))
        with self._lock:
            if instruction == INST_SYNC_READ and motor_id == BROADCAST_ID:
                self.counts.sync_reads += 1
                address, size, ids = params[0], params[1], params[2:]
                return b"".join(
                    status(i, bytes(self.memory[i][address : address + size]))
                    for i in ids
                    if i in self.memory and i not in self.silent
                )
            if instruction == INST_SYNC_WRITE and motor_id == BROADCAST_ID:
                self.counts.sync_writes += 1
                address, size = params[0], params[1]
                for start in range(2, len(params), size + 1):
                    i = params[start]
                    if i in self.memory and i not in self.silent:
                        self._store(i, address, params[start + 1 : start + 1 + size])
                return b""
            if motor_id not in self.memory or motor_id in self.silent:
                return b""
            if instruction == INST_WRITE:
                self.counts.writes += 1
                self._store(motor_id, params[0], params[1:])
                return status(motor_id)
            if instruction == INST_READ:
                self.counts.reads += 1
                address, size = params[0], params[1]
                return status(motor_id, bytes(self.memory[motor_id][address : address + size]))
            if instruction == INST_PING:
                self.counts.pings += 1
                return status(motor_id)
        return b""

    def _store(self, motor_id: int, address: int, data: bytes) -> None:
        """A write into one servo's table, as the STS3215 keeps it: an Acceleration above the
        Maximum_Acceleration ceiling is stored as the ceiling (read back 50 for 114 written)."""
        table = self.memory[motor_id]
        table[address : address + len(data)] = data
        if address <= ACCELERATION < address + len(data):
            table[ACCELERATION] = min(table[ACCELERATION], table[MAX_ACCELERATION])


def _take_packet(buffer: bytearray) -> tuple[int, int, bytes] | None:
    """The first whole instruction packet in ``buffer`` as (id, instruction, params), removed
    from it; None when none is complete yet. Junk before a header is dropped."""
    while True:
        start = buffer.find(HEADER)
        if start < 0:
            del buffer[: max(0, len(buffer) - 1)]
            return None
        del buffer[:start]
        if len(buffer) < 4:
            return None
        total = 4 + buffer[3]
        if len(buffer) < total:
            return None
        frame = bytes(buffer[:total])
        del buffer[:total]
        if checksum(frame[2:-1]) == frame[-1]:
            return frame[2], frame[4], frame[5:-1]
