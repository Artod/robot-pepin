"""A fake head ESP32 behind a fake serial port, and the fakes of the head server's sockets.

:class:`FakeHead` is what ``pepin.head_server`` reads and writes in place of ``/dev/pepin-head``:
it answers pings, keeps every frame the host sent, and queues IMU batches and status frames that
"arrive" a transit after the ESP32 stamped them. Its micros run on the test's clock, offset and
optionally fast by some ppm, and wrap at 2^32 like the real ones.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from pepin.face import EYE_SETS
from pepin.head_link import (
    IMU,
    PING,
    PONG,
    PONG_FIELDS,
    STATUS,
    FrameDecoder,
    HeadStatus,
    ImuSample,
    encode_frame,
    encode_imu,
    encode_status,
)
from pepin.head_server import I2C_READ_S

BAUD = 921600


class FakeClock:
    """Seconds that pass when the test says so."""

    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeHead:
    """The ESP32 on the far side of the serial port."""

    def __init__(
        self,
        clock: Callable[[], float],
        *,
        offset_us: int = 7_000_000,
        fast_ppm: float = 0.0,
        transit_s: float = 0.001,
    ) -> None:
        self.clock = clock
        self.offset_us = offset_us
        self.fast_ppm = fast_ppm
        self.transit_s = transit_s
        self.decoder = FrameDecoder()
        self.received: list[tuple[int, bytes]] = []
        self.closed = False
        self._queue: list[tuple[float, bytes]] = []  # (host time it can be read, bytes)
        self.config_id = 0
        self.expression = 0
        self.version = 4  # firmware 3 drops a 'C' with the eyes' brightness
        self.eyes_brightness = 64
        self.eye_sets = True  # False: a firmware 4 built before the sets, dropping a 'C' of 10
        self.eye_set_choice = 0  # the 'C' config's eye-set byte: 0 follow the style

    def esp_us(self, host_s: float) -> int:
        """The ESP32's micros (wrapped) at host time ``host_s``."""
        return round(host_s * 1e6 * (1 + self.fast_ppm * 1e-6)) + self.offset_us

    # -- ByteLink ---------------------------------------------------------------------------

    def read(self, timeout_s: float) -> bytes:
        now = self.clock()
        ready = [data for at, data in self._queue if at <= now]
        self._queue = [(at, data) for at, data in self._queue if at > now]
        return b"".join(ready)

    def wire_s(self, frame: bytes) -> float:
        """A frame's own time on the wire at the head's 921600 baud."""
        return len(frame) * 10.0 / BAUD

    def write(self, data: bytes) -> None:
        now = self.clock()
        for kind, payload in self.decoder.feed(data):
            self.received.append((kind, payload))
            if kind == PING:
                receipt = now + self.transit_s + self.wire_s(encode_frame(PING, payload))
                pong = encode_frame(
                    PONG,
                    PONG_FIELDS.pack(
                        int.from_bytes(payload, "little"), self.esp_us(receipt) & 0xFFFFFFFF
                    ),
                )
                self._queue.append((receipt + self.wire_s(pong) + self.transit_s, pong))
            elif kind == ord("C"):
                if len(payload) > (8 if self.version < 4 else 10 if self.eye_sets else 9):
                    continue  # firmware 3 takes 7 or 8 bytes, a 4 without the sets up to 9
                self.config_id = payload[0]
                if len(payload) > 8:
                    self.eyes_brightness = payload[8]
                if len(payload) > 9:
                    self.eye_set_choice = payload[9]
            elif kind == ord("E"):
                self.expression = payload[0]

    def close(self) -> None:
        self.closed = True

    # -- what the ESP32 says -----------------------------------------------------------------

    def imu(self, host_times: list[float], cfg: int | None = None) -> list[ImuSample]:
        """A batch of samples taken at these host times: the last one read over I2C, the frame
        sent, and readable a transit after its last byte."""
        samples = [
            ImuSample(
                self.esp_us(t) & 0xFFFFFFFF,
                (0, 0, 8192),
                (i, -i, 2 * i),
                self.config_id if cfg is None else cfg,
            )
            for i, t in enumerate(host_times)
        ]
        frame = encode_frame(IMU, encode_imu(samples))
        ready = host_times[-1] + I2C_READ_S + self.wire_s(frame) + self.transit_s
        self._queue.append((ready, frame))
        return samples

    @property
    def eye_set(self) -> str:
        """The look set in force: the one the 'C' named, else classic (the fake's every style
        follows it)."""
        return EYE_SETS[self.eye_set_choice - 1] if self.eye_set_choice else EYE_SETS[0]

    def status(self, **fields: Any) -> None:
        """An 'S' frame, readable a transit from now: firmware 4's carries the eyes' fields, and
        the set in force when it has the sets (``fields`` over them)."""
        now = self.clock()
        base = dict(
            esp_us=self.esp_us(now) & 0xFFFFFFFF,
            fps=49.5,
            imu_rate_hz=200.1,
            i2c_errors=0,
            dropped=0,
            rx_errors=0,
            rx_frames=10,
            expression=self.expression,
            mode="face",
            imu_state="interrupt",
            who_am_i=0x72,
            config_id=self.config_id,
            version=self.version,
            free_heap_kb=150,
            gaps=0,
        )
        if self.version >= 4:
            base.update(eyes_ma=0, eyes_brightness=self.eyes_brightness, eyes_mode="expression",
                        eye_layout=0, eye_set=self.eye_set if self.eye_sets else None)  # fmt: skip
        base.update(fields)
        frame = encode_frame(STATUS, encode_status(HeadStatus(**base)))
        self._queue.append((now + self.transit_s, frame))

    def raw(self, data: bytes) -> None:
        """Bytes on the wire as they are (noise, a damaged frame), readable now."""
        self._queue.append((self.clock(), data))

    def sent(self, kind: str) -> list[bytes]:
        """The payloads of every frame of ``kind`` the host sent."""
        return [payload for k, payload in self.received if k == ord(kind)]


@dataclass(eq=False)  # hashed by identity, as a connection is
class FakeClient:
    """A connected client of the head server: what was posted to it."""

    peer: str = "client"
    alive: bool = True
    posted: list[bytes] = field(default_factory=list)

    def post(self, line: bytes) -> None:
        self.posted.append(line)


@dataclass
class FakeServer:
    """The head server's sockets: commands in, replies and broadcasts out."""

    inbox: list[tuple[Any, dict[str, Any]]] = field(default_factory=list)
    replies: list[tuple[Any, dict[str, Any]]] = field(default_factory=list)
    broadcasts: list[dict[str, Any]] = field(default_factory=list)
    client_count: int = 1

    def commands(self) -> list[tuple[Any, dict[str, Any]]]:
        out, self.inbox = self.inbox, []
        return out

    def reply(self, client: Any, message: dict[str, Any]) -> None:
        self.replies.append((client, message))

    def broadcast(self, message: dict[str, Any]) -> None:
        self.broadcasts.append(message)

    def close(self) -> None:
        pass
