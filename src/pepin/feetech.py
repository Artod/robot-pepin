"""Feetech STS servo-bus protocol over a raw TCP-to-serial bridge.

Why not lerobot's bus here: it polls a local tty with timeouts tuned for a
USB adapter, and over a 15 ms wifi round trip it mistakes late replies for
the answers to later requests. This client owns the framing and waits per
request against a deadline, so one lost packet costs one retry and never
poisons the stream.

Wire format (both directions): ``FF FF id length instr|error params... checksum``
where ``length`` counts everything after itself and ``checksum`` is the
bitwise inverse of the byte sum from ``id`` to the last parameter.
Multi-byte values are little-endian; velocity and position use a sign bit
at bit 15 (sign-magnitude), not two's complement.
"""

from __future__ import annotations

import logging
import socket
import time
from dataclasses import dataclass
from types import TracebackType

from pepin.telemetry import LatencyTracker

logger = logging.getLogger(__name__)

HEADER = b"\xff\xff"
BROADCAST_ID = 0xFE
INST_PING = 0x01
INST_READ = 0x02
INST_WRITE = 0x03
INST_SYNC_READ = 0x82
INST_SYNC_WRITE = 0x83


@dataclass(frozen=True)
class Register:
    """A control-table entry: address, byte size, and whether bit 15 is a sign."""

    address: int
    size: int
    sign_magnitude: bool = False


REGISTERS = {
    "Operating_Mode": Register(33, 1),
    "Torque_Enable": Register(40, 1),
    "Goal_Velocity": Register(46, 2, sign_magnitude=True),
    "Present_Position": Register(56, 2, sign_magnitude=True),
    "Present_Velocity": Register(58, 2, sign_magnitude=True),
}


def checksum(body: bytes) -> int:
    """Low byte of the inverted sum over ``id .. last parameter`` (the header is excluded)."""
    return (~sum(body)) & 0xFF


def build_packet(motor_id: int, instruction: int, params: bytes = b"") -> bytes:
    """One instruction packet ready for the wire, checksum appended."""
    body = bytes([motor_id, len(params) + 2, instruction]) + params
    return HEADER + body + bytes([checksum(body)])


def encode_value(value: int, register: Register) -> bytes:
    """Signed value to the register's little-endian bytes, sign in bit 15 where it applies."""
    raw = value
    if register.sign_magnitude:
        raw = (abs(value) & 0x7FFF) | (0x8000 if value < 0 else 0)
    return raw.to_bytes(register.size, "little")


def decode_value(data: bytes, register: Register) -> int:
    """Register bytes back to a signed integer (ticks, or ticks/s for velocities)."""
    raw = int.from_bytes(data, "little")
    if register.sign_magnitude and raw & 0x8000:
        return -(raw & 0x7FFF)
    return raw


@dataclass(frozen=True)
class StatusPacket:
    """One servo reply: who answered, its error byte, and the raw parameter bytes."""

    motor_id: int
    error: int
    params: bytes


class PacketParser:
    """Extracts status packets from a byte stream, skipping junk between them."""

    def __init__(self) -> None:
        self._buf = bytearray()

    def reset(self) -> None:
        """Drop half-parsed bytes; after a flush they can only belong to a dead transaction."""
        self._buf.clear()

    def feed(self, data: bytes) -> list[StatusPacket]:
        """Append received bytes and return every complete, checksum-valid reply in them."""
        self._buf += data
        packets: list[StatusPacket] = []
        while True:
            start = self._buf.find(HEADER)
            if start < 0:
                # keep a trailing 0xFF: it may be the first half of the next header
                tail = self._buf[-1:] if self._buf[-1:] == b"\xff" else b""
                self._buf[:] = tail
                break
            del self._buf[:start]
            if len(self._buf) < 4:
                break
            total = 4 + self._buf[3]
            if len(self._buf) < total:
                break
            frame = bytes(self._buf[:total])
            del self._buf[:total]
            body = frame[2:-1]
            if checksum(body) != frame[-1]:
                logger.debug("dropping frame with bad checksum: %s", frame.hex(" "))
                continue
            packets.append(StatusPacket(motor_id=frame[2], error=frame[4], params=frame[5:-1]))
        return packets


class FeetechTcpClient:
    """Talks to named Feetech servos through ser2net; implements ``pepin.bus.MotorBus``.

    Values are always raw register units — no calibration or normalisation
    happens here, which is why ``normalize=True`` is rejected loudly.
    """

    def __init__(
        self,
        host: str,
        port: int,
        motors: dict[str, int],
        *,
        timeout_s: float = 0.2,
        retries: int = 2,
    ) -> None:
        """``motors`` maps names to bus ids; ``timeout_s`` is the deadline for one reply
        (~15 ms round trip over wifi) and ``retries`` the extra attempts before raising."""
        self._address = (host, port)
        self._names = dict(motors)
        self._ids = {motor_id: name for name, motor_id in motors.items()}
        self._timeout = timeout_s
        self._retries = retries
        self._sock: socket.socket | None = None
        self._parser = PacketParser()
        self.latency = LatencyTracker("feetech.transaction")

    # -- connection -------------------------------------------------------

    def connect(self) -> None:
        """Open the bridge socket with Nagle disabled and drop whatever ser2net buffered."""
        self._sock = socket.create_connection(self._address, timeout=2.0)
        self._sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        time.sleep(0.1)
        self.flush()

    def close(self) -> None:
        """Close the socket. The servos keep whatever velocity was last commanded, so
        stop the wheels before calling this."""
        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def __enter__(self) -> FeetechTcpClient:
        """Connect on entry."""
        self.connect()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        """Close the socket on exit; motor state is the caller's business."""
        self.close()

    @property
    def _socket(self) -> socket.socket:
        """The live socket, or ``ConnectionError`` if :meth:`connect` was never called."""
        if self._sock is None:
            raise ConnectionError("not connected")
        return self._sock

    def flush(self) -> None:
        """Drop anything unread: in a request/reply protocol it can only be stale."""
        sock = self._socket
        sock.settimeout(0.005)
        try:
            while sock.recv(4096):
                pass
        except (TimeoutError, BlockingIOError):
            pass
        self._parser.reset()

    # -- transactions -----------------------------------------------------

    def _collect(self, expected: set[int], deadline: float) -> dict[int, StatusPacket]:
        """Read replies until every id in ``expected`` answered or ``deadline`` passes.

        ``deadline`` is a ``time.monotonic()`` instant. Replies from other ids are
        logged and dropped; a closed bridge raises instead of timing out silently.
        """
        replies: dict[int, StatusPacket] = {}
        sock = self._socket
        while expected - replies.keys():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            sock.settimeout(remaining)
            try:
                data = sock.recv(4096)
            except TimeoutError:
                break
            if not data:
                raise ConnectionError("bridge closed the connection")
            for packet in self._parser.feed(data):
                if packet.motor_id in expected:
                    replies[packet.motor_id] = packet
                else:
                    logger.debug("ignoring reply from unexpected id %d", packet.motor_id)
        return replies

    def _transaction(self, packet: bytes, expected: set[int]) -> dict[int, StatusPacket]:
        """Send one packet and wait for a reply from every expected id, with retries."""
        for attempt in range(1, self._retries + 2):
            self.flush()
            start = time.perf_counter()
            self._socket.sendall(packet)
            replies = self._collect(expected, time.monotonic() + self._timeout)
            self.latency.add(time.perf_counter() - start)
            missing = expected - replies.keys()
            if not missing:
                return replies
            if attempt > self._retries:
                raise TimeoutError(f"no reply from ids {sorted(missing)} after {attempt} attempts")
            logger.warning("no reply from ids %s (attempt %d), retrying", sorted(missing), attempt)
        raise AssertionError("unreachable")

    def _id(self, motor: str) -> int:
        """Bus id of a motor named in the constructor's table."""
        return self._names[motor]

    @staticmethod
    def _raw_only(normalize: bool) -> None:
        """Reject ``normalize=True``: there is no calibration table here to normalise against."""
        if normalize:
            raise ValueError(
                "FeetechTcpClient speaks raw register units only; pass normalize=False"
            )

    # -- MotorBus -----------------------------------------------------------

    def ping(self, motor: str, num_retry: int = 0) -> int | None:
        """The servo's error byte (0 when healthy), or None if it stayed silent."""
        motor_id = self._id(motor)
        try:
            reply = self._transaction(build_packet(motor_id, INST_PING), {motor_id})
        except TimeoutError:
            return None
        return reply[motor_id].error

    def write(self, data_name: str, motor: str, value: int, *, normalize: bool = False) -> None:
        """Write one control-table register on one motor in raw units, waiting for its ack."""
        self._raw_only(normalize)
        register = REGISTERS[data_name]
        motor_id = self._id(motor)
        params = bytes([register.address]) + encode_value(value, register)
        self._transaction(build_packet(motor_id, INST_WRITE, params), {motor_id})

    def sync_write(self, data_name: str, values: dict[str, int], *, normalize: bool = True) -> None:
        """Broadcast one register to several motors; servos send no reply to this."""
        self._raw_only(normalize)
        register = REGISTERS[data_name]
        params = bytes([register.address, register.size]) + b"".join(
            bytes([self._id(name)]) + encode_value(value, register)
            for name, value in values.items()
        )
        self.flush()
        self._socket.sendall(build_packet(BROADCAST_ID, INST_SYNC_WRITE, params))

    def sync_read(
        self, data_name: str, motors: list[str], *, normalize: bool = True
    ) -> dict[str, int]:
        """Read one register from several motors in a single round trip, keyed by motor name.

        Raises ``TimeoutError`` if any motor stays silent through the retries, so the
        caller never gets a partially stale set of encoder readings.
        """
        self._raw_only(normalize)
        register = REGISTERS[data_name]
        ids = [self._id(name) for name in motors]
        params = bytes([register.address, register.size, *ids])
        replies = self._transaction(build_packet(BROADCAST_ID, INST_SYNC_READ, params), set(ids))
        return {self._ids[i]: decode_value(replies[i].params, register) for i in ids}

    def enable_torque(self, motors: list[str] | None = None) -> None:
        """Energise the named motors (all of them by default); they hold against being pushed."""
        for name in motors or list(self._names):
            self.write("Torque_Enable", name, 1)

    def disable_torque(self, motors: list[str] | None = None) -> None:
        """Release the named motors (all by default) so the wheels turn freely by hand."""
        for name in motors or list(self._names):
            self.write("Torque_Enable", name, 0)
