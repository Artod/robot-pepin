"""The head's two wires: the ESP32's serial frames, and the head server's JSON lines.

THE SERIAL LINK (the head contract; ``firmware/head_esp32/src/protocol.h`` is the other end of
the same bytes). The ESP32 hangs on the board's free USB port through its CH340 at 921600 baud;
every frame both ways is::

    0xA5 | type (1) | length (2, LE) | payload | CRC-8 (poly 0x07, init 0, over type+length+payload)

=====  ===========  =====================================================================
type   direction    payload (little-endian)
=====  ===========  =====================================================================
'I'    ESP -> host  IMU samples, n x 17 bytes: micros u32, ax ay az gx gy gz i16, config id u8
'S'    ESP -> host  status once a second (:class:`HeadStatus`)
'P'    ESP -> host  pong: the ping's id u32, the ESP's micros at its receipt u32
'E'    host -> ESP  expression: id u8 (config/face.json's order), intensity u8, transition ms u16
'M'    host -> ESP  mouth openness u8, 0..255 (~25 Hz while the robot speaks)
'T'    host -> ESP  info screen: duration ms u16, count u8, items (:func:`encode_info`)
'Q'    host -> ESP  ping: id u32
'C'    host -> ESP  config: id u8, IMU rate u16, DLPF u8, accel FS u8, gyro FS u8, brightness u8
=====  ===========  =====================================================================

THE HEAD SERVER'S DOOR (``pepin.head_server``, TCP 3340, JSON lines both ways). Any client::

    {"cmd": "express", "source": "voice", "name": "thinking", "intensity": 1.0,
     "transition_ms": 280, "hold_s": null}     a source's expression; hold_s null: until replaced
    {"cmd": "event", "source": "goal", "name": "arrived", "end": true}
                                               config/face.json's event table; "end" first clears
                                               what the source held
    {"cmd": "clear", "source": "voice"}        the source shows nothing any more
    {"cmd": "mouth", "level": 0.42}            the speech level, 0..1 (the audio server)
    {"cmd": "show", "items": [...], "seconds": 8}  or  {"cmd": "show", "text": "...", ...}
    {"cmd": "lease", "name": "goal_server", "seconds": 6}   a brain is here (sleepy without one)
    {"cmd": "config", "imu_rate_hz": 1000, "dlpf": 3, "accel_fs": 1, "gyro_fs": 1,
     "brightness": 200}
    {"cmd": "subscribe", "imu": true}          this client gets the IMU lines below
    {"cmd": "status"}

and gets ``{"type": "ack", ...}`` / ``{"type": "error", "error": ...}`` for each, the status line
once a second, and, subscribed, one line per serial 'I' frame::

    {"type": "imu", "cfg": 1, "rate_hz": 1000, "acc_scale": 0.0011971, "gyro_scale": 0.00026646,
     "delay_s": 0.0048, "s": [[t, esp_us, ax, ay, az, gx, gy, gz], ...]}

``t`` is the sample's moment on the board's monotonic clock (``time.monotonic``): its data-ready
edge mapped through the clock map, less the chip's filter delay (``delay_s``); ``esp_us`` the
ESP32's own micros, unwrapped; the six values raw counts in the chip's axes, times ``acc_scale``
m/s^2 or ``gyro_scale`` rad/s.

Standard library only: this module runs on the board.
"""

from __future__ import annotations

import json
import logging
import math
import re
import socket
import struct
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from typing import Any

from pepin.streams import JsonLinesClient

logger = logging.getLogger(__name__)

HEAD_PORT = 3340
SYNC = 0xA5
MAX_PAYLOAD = 1024
OVERHEAD = 5
IMU, STATUS, PONG = ord("I"), ord("S"), ord("P")
EXPRESSION, MOUTH, INFO, PING, CONFIG = ord("E"), ord("M"), ord("T"), ord("Q"), ord("C")

IMU_SAMPLE = struct.Struct("<I6hB")
STATUS_FIELDS = struct.Struct("<IHHIIIIBBBBBBHI")
PONG_FIELDS = struct.Struct("<II")
CONFIG_FIELDS = struct.Struct("<BHBBBB")
INFO_ITEMS_MAX = 8  # protocol.h kInfoItemsMax
INFO_TEXT_MAX = 47  # bytes of UTF-8 per string (kInfoTextMax less its terminating zero)
G = 9.80665

# The MPU-6050 register map (rev 4.2): the gyro's group delay by DLPF_CFG, seconds
# (pepin.sensor_timing holds the same table for the base's IMU).
DLPF_DELAY_S = {1: 0.0019, 2: 0.0028, 3: 0.0048, 4: 0.0083, 5: 0.0134, 6: 0.0186}


def _crc_table() -> list[int]:
    table = []
    for byte in range(256):
        crc = byte
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return table


_CRC = _crc_table()


def crc8(data: bytes, crc: int = 0) -> int:
    """CRC-8, polynomial 0x07, initial value ``crc``: the frame's check byte."""
    for byte in data:
        crc = _CRC[crc ^ byte]
    return crc


def encode_frame(kind: int, payload: bytes = b"") -> bytes:
    """One frame of ``kind`` ('E', 'Q', ...) around ``payload``."""
    if len(payload) > MAX_PAYLOAD:
        raise ValueError(f"payload of {len(payload)} bytes, at most {MAX_PAYLOAD}")
    body = bytes((kind, len(payload) & 0xFF, len(payload) >> 8)) + payload
    return bytes((SYNC,)) + body + bytes((crc8(body),))


class FrameDecoder:
    """Frames out of the serial byte stream. A bad CRC or an impossible length costs that frame
    (counted); the decoder looks for the next sync byte after the one that failed, so a frame
    that follows a damaged one is not lost with it."""

    def __init__(self) -> None:
        """An empty decoder."""
        self._buffer = bytearray()
        self.frames = 0
        self.crc_errors = 0
        self.length_errors = 0
        self.skipped_bytes = 0  # bytes outside any frame: noise, the boot ROM's 115200 text

    def feed(self, data: bytes) -> list[tuple[int, bytes]]:
        """``data`` in; every complete valid frame out, as ``(type, payload)``."""
        buf = self._buffer
        buf += data
        out: list[tuple[int, bytes]] = []
        start = 0
        while True:
            sync = buf.find(SYNC, start)
            if sync < 0:
                self.skipped_bytes += len(buf) - start
                start = len(buf)
                break
            self.skipped_bytes += sync - start
            if len(buf) - sync < 4:
                start = sync
                break
            length = buf[sync + 2] | (buf[sync + 3] << 8)
            if length > MAX_PAYLOAD:
                self.length_errors += 1
                start = sync + 1
                continue
            end = sync + 4 + length + 1
            if len(buf) < end:
                start = sync
                break
            if crc8(bytes(buf[sync + 1 : end - 1])) != buf[end - 1]:
                self.crc_errors += 1
                start = sync + 1
                continue
            out.append((buf[sync + 1], bytes(buf[sync + 4 : end - 1])))
            self.frames += 1
            start = end
        del buf[:start]
        return out


# -- ESP32 -> host ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ImuSample:
    """One raw sample: the ESP32's micros at its data-ready edge (32-bit, wrapping), the six
    counts in the chip's axes, and the id of the configuration it was taken at."""

    esp_us: int
    accel: tuple[int, int, int]
    gyro: tuple[int, int, int]
    cfg: int


def decode_imu(payload: bytes) -> list[ImuSample]:
    """An 'I' payload's samples; ``ValueError`` when it is not a whole number of them."""
    if len(payload) % IMU_SAMPLE.size:
        raise ValueError(f"an IMU payload of {len(payload)} bytes is not n x {IMU_SAMPLE.size}")
    return [
        ImuSample(t, (ax, ay, az), (gx, gy, gz), cfg)
        for t, ax, ay, az, gx, gy, gz, cfg in IMU_SAMPLE.iter_unpack(payload)
    ]


def encode_imu(samples: Iterable[ImuSample]) -> bytes:
    """An 'I' payload (the fake head of the tests speaks it)."""
    return b"".join(
        IMU_SAMPLE.pack(s.esp_us & 0xFFFFFFFF, *s.accel, *s.gyro, s.cfg) for s in samples
    )


IMU_STATES = {0: "absent", 1: "interrupt", 2: "polled"}
MODES = {0: "face", 1: "info", 2: "asleep"}


@dataclass(frozen=True)
class HeadStatus:
    """The ESP32's 'S' frame: its own view of the face, the IMU and the link."""

    esp_us: int
    fps: float
    imu_rate_hz: int
    i2c_errors: int
    dropped: int
    rx_errors: int
    rx_frames: int
    expression: int
    mode: str
    imu_state: str
    who_am_i: int
    config_id: int
    version: int
    free_heap_kb: int
    duplicates: int

    def as_dict(self) -> dict[str, Any]:
        """The fields, for a JSON line."""
        return asdict(self)


def decode_status(payload: bytes) -> HeadStatus:
    """An 'S' payload; ``struct.error`` when it is the wrong size."""
    f = STATUS_FIELDS.unpack(payload)
    return HeadStatus(
        esp_us=f[0],
        fps=f[1] / 10.0,
        imu_rate_hz=f[2],
        i2c_errors=f[3],
        dropped=f[4],
        rx_errors=f[5],
        rx_frames=f[6],
        expression=f[7],
        mode=MODES.get(f[8], str(f[8])),
        imu_state=IMU_STATES.get(f[9], str(f[9])),
        who_am_i=f[10],
        config_id=f[11],
        version=f[12],
        free_heap_kb=f[13],
        duplicates=f[14],
    )


def encode_status(status: HeadStatus) -> bytes:
    """An 'S' payload (the fake head of the tests speaks it)."""
    mode = {v: k for k, v in MODES.items()}.get(status.mode, 0)
    imu = {v: k for k, v in IMU_STATES.items()}.get(status.imu_state, 0)
    return STATUS_FIELDS.pack(
        status.esp_us & 0xFFFFFFFF,
        round(status.fps * 10),
        status.imu_rate_hz,
        status.i2c_errors,
        status.dropped,
        status.rx_errors,
        status.rx_frames,
        status.expression,
        mode,
        imu,
        status.who_am_i,
        status.config_id,
        status.version,
        status.free_heap_kb,
        status.duplicates,
    )


def decode_pong(payload: bytes) -> tuple[int, int]:
    """A 'P' payload: the ping's id and the ESP32's micros at its receipt."""
    ping_id, esp_us = PONG_FIELDS.unpack(payload)
    return int(ping_id), int(esp_us)


# -- host -> ESP32 ------------------------------------------------------------------------------


def encode_expression(expression_id: int, intensity: float, transition_ms: float) -> bytes:
    """An 'E' payload."""
    level = round(min(max(intensity, 0.0), 1.0) * 255)
    ms = round(min(max(transition_ms, 0.0), 65535.0))
    return struct.pack("<BBH", expression_id, level, ms)


def encode_mouth(level: float) -> bytes:
    """An 'M' payload: the opening 0..1 as 0..255."""
    return bytes((round(min(max(level, 0.0), 1.0) * 255),))


def encode_ping(ping_id: int) -> bytes:
    """A 'Q' payload."""
    return struct.pack("<I", ping_id & 0xFFFFFFFF)


@dataclass(frozen=True)
class ImuConfig:
    """How the head's MPU6050 samples: ``id`` is echoed in every sample; ``rate_hz`` the chip's
    output rate (1000 / (1 + SMPLRT_DIV)); ``dlpf`` its DLPF_CFG (1..6); the full scales as the
    register's 0..3 (+-2/4/8/16 g, +-250/500/1000/2000 deg/s)."""

    id: int = 0
    rate_hz: int = 1000
    dlpf: int = 3
    accel_fs: int = 1
    gyro_fs: int = 1

    def __post_init__(self) -> None:
        if not 0 <= self.id <= 255:
            raise ValueError(f"config id {self.id}: 0..255")
        if not 4 <= self.rate_hz <= 1000 or 1000 % self.rate_hz:
            raise ValueError(f"imu rate {self.rate_hz} Hz: 1000 / n for n = 1..250")
        if self.dlpf not in DLPF_DELAY_S:
            raise ValueError(f"DLPF_CFG {self.dlpf}: 1..6 (0 and 7 switch the gyro to 8 kHz)")
        if not (0 <= self.accel_fs <= 3 and 0 <= self.gyro_fs <= 3):
            raise ValueError("full scales: 0..3")

    @property
    def acc_scale(self) -> float:
        """m/s^2 per count."""
        return G / (16384 >> self.accel_fs)

    @property
    def gyro_scale(self) -> float:
        """rad/s per count."""
        return math.radians(1.0 / (131.0 / (1 << self.gyro_fs)))

    @property
    def delay_s(self) -> float:
        """The gyro's group delay at this DLPF_CFG: how long before data-ready the motion was."""
        return DLPF_DELAY_S[self.dlpf]


FIRMWARE_DEFAULTS = ImuConfig()  # imu.h kImuDefaults: what samples of config id 0 were taken at


def encode_config(config: ImuConfig, brightness: int) -> bytes:
    """A 'C' payload."""
    return CONFIG_FIELDS.pack(
        config.id,
        config.rate_hz,
        config.dlpf,
        config.accel_fs,
        config.gyro_fs,
        min(max(brightness, 0), 255),
    )


def _text(value: object) -> bytes:
    """A string as at most :data:`INFO_TEXT_MAX` bytes of UTF-8, cut between characters."""
    data = str(value).encode()
    if len(data) <= INFO_TEXT_MAX:
        return data
    cut = INFO_TEXT_MAX
    while cut > 0 and (data[cut] & 0xC0) == 0x80:
        cut -= 1
    return data[:cut]


def encode_info(items: Iterable[Mapping[str, Any]], seconds: float) -> bytes:
    """A 'T' payload: at most 8 items, each ``{"text": ...}``, ``{"key": ..., "value": ...}``
    or ``{"bar": label, "frac": 0..1, "value": ...}``; ``seconds`` 0 returns to the face."""
    body = b""
    count = 0
    for item in items:
        if count == INFO_ITEMS_MAX:
            break
        if "bar" in item:
            frac = round(min(max(float(item.get("frac", 0.0)), 0.0), 1.0) * 255)
            label, value = _text(item["bar"]), _text(item.get("value", ""))
            body += bytes((2, len(label))) + label + bytes((frac, len(value))) + value
        elif "key" in item:
            key, value = _text(item["key"]), _text(item.get("value", ""))
            body += bytes((1, len(key))) + key + bytes((len(value),)) + value
        else:
            text = _text(item.get("text", ""))
            body += bytes((0, len(text))) + text
        count += 1
    ms = round(min(max(seconds, 0.0), 65.535) * 1000)
    return struct.pack("<HB", ms, count) + body


_BAR = re.compile(r"^\s*(-?\d+(?:\.\d+)?)\s*(?:/\s*(\d+(?:\.\d+)?)|(%))\s*(.*)$")


def show_items(text: str) -> list[dict[str, Any]]:
    """The show tool's markup as info-screen items, one per line: ``label: 63%`` or
    ``label: 41/70 C`` is a bar (the fraction drawn, the text kept as its value), ``key: value``
    a row, anything else a line of text (the first line is the title)."""
    items: list[dict[str, Any]] = []
    for line in text.replace("|", "\n").splitlines():
        line = line.strip()
        if not line:
            continue
        key, colon, value = line.partition(":")
        if colon and key.strip() and value.strip():
            bar = _BAR.match(value)
            if bar:
                number = float(bar.group(1))
                full = 100.0 if bar.group(3) else float(bar.group(2))
                frac = number / full if full > 0 else 0.0
                items.append({"bar": key.strip(), "frac": frac, "value": value.strip()})
            else:
                items.append({"key": key.strip(), "value": value.strip()})
        else:
            items.append({"text": line})
    return items[:INFO_ITEMS_MAX]


# -- the head server's door, as its clients use it ----------------------------------------------


class HeadClient(JsonLinesClient):
    """A long-lived connection to the head server for processes that send face commands as
    things happen (the goal server, the voice loop): every send is a non-blocking JSON line,
    dropped while the link is down, and the connection comes back by itself. Construct, then
    :meth:`start`."""

    def __init__(self, host: str, port: int = HEAD_PORT, *, source: str) -> None:
        """``source`` names this client's expressions (each source holds its own). A head
        server that is away is asked again every 5 s (each try is logged)."""
        super().__init__(host, port, name=f"head:{source}", retry_s=5.0)
        self.source = source
        self.last_status: dict[str, Any] | None = None

    def _ingest(self, message: dict[str, Any]) -> None:
        if message.get("type") == "status":
            self.last_status = message

    def express(
        self,
        name: str,
        *,
        intensity: float = 1.0,
        hold_s: float | None = None,
        transition_ms: float | None = None,
    ) -> None:
        """Show expression ``name`` (held until replaced when ``hold_s`` is None)."""
        message: dict[str, Any] = {
            "cmd": "express",
            "source": self.source,
            "name": name,
            "intensity": intensity,
            "hold_s": hold_s,
        }
        if transition_ms is not None:
            message["transition_ms"] = transition_ms
        self.send(message)

    def event(self, name: str, *, end: bool = False) -> None:
        """A robot event from config/face.json's table; ``end`` clears this source's held
        expression first."""
        self.send({"cmd": "event", "source": self.source, "name": name, "end": end})

    def clear(self) -> None:
        """This source shows nothing any more."""
        self.send({"cmd": "clear", "source": self.source})

    def mouth(self, level: float) -> None:
        """The speech level, 0..1."""
        self.send({"cmd": "mouth", "level": round(level, 3)})

    def lease(self, seconds: float) -> None:
        """This brain is here for ``seconds`` more (renew it well before it lapses)."""
        self.send({"cmd": "lease", "name": self.source, "seconds": seconds})


def ask(
    request: dict[str, Any], host: str, port: int = HEAD_PORT, timeout_s: float = 2.0
) -> dict[str, Any]:
    """One request to the head server and its answer: the ack or error line (the status line
    for ``{"cmd": "status"}``; the once-a-second broadcast in between is skipped); ``OSError``
    when nobody answers in ``timeout_s``."""
    wanted = ("status",) if request.get("cmd") == "status" else ("ack", "error")
    with socket.create_connection((host, port), timeout=timeout_s) as sock:
        sock.sendall((json.dumps(request, ensure_ascii=False) + "\n").encode())
        buffer = b""
        sock.settimeout(timeout_s)
        while True:
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionError("the head server closed the connection")
            buffer += chunk
            *lines, buffer = buffer.split(b"\n")
            for line in lines:
                if not line.strip():
                    continue
                answer = json.loads(line)
                if isinstance(answer, dict) and answer.get("type") in wanted:
                    return answer


def main(argv: list[str] | None = None) -> int:
    """Talk to the head server by hand: the bring-up's and a bench's commands."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Talk to the head server (pepin.head_server, TCP 3340).",
        epilog="e.g.  express happy 3   |   event arrived   |   show 'Temps|left: 41/70 C' 6"
        "   |   imu 5   |   status",
    )
    parser.add_argument("--host", default=os.environ.get("PEPIN_HOST", "10.0.0.187"))
    parser.add_argument("--port", type=int, default=HEAD_PORT)
    parser.add_argument("command", choices=("status", "express", "event", "clear", "show", "imu"))
    parser.add_argument("args", nargs="*")
    args = parser.parse_args(argv)
    a = args.args
    request: dict[str, Any]
    if args.command == "express":
        hold = float(a[1]) if len(a) > 1 else None
        request = {"cmd": "express", "source": "cli", "name": a[0], "hold_s": hold}
    elif args.command == "event":
        request = {"cmd": "event", "source": "cli", "name": a[0]}
    elif args.command == "clear":
        request = {"cmd": "clear", "source": "cli"}
    elif args.command == "show":
        request = {"cmd": "show", "text": a[0], "seconds": float(a[1]) if len(a) > 1 else 8.0}
    elif args.command == "imu":
        return _print_imu(args.host, args.port, int(a[0]) if a else 5)
    else:
        request = {"cmd": "status"}
    print(json.dumps(ask(request, args.host, args.port), ensure_ascii=False))
    return 0


def _print_imu(host: str, port: int, lines: int) -> int:
    """Subscribe and print ``lines`` IMU lines, each as its first sample in SI units."""
    with socket.create_connection((host, port), timeout=3.0) as sock:
        sock.sendall(b'{"cmd": "subscribe", "imu": true}\n')
        buffer = b""
        while lines > 0:
            chunk = sock.recv(65536)
            if not chunk:
                return 1
            buffer += chunk
            *done, buffer = buffer.split(b"\n")
            for line in done:
                message = json.loads(line) if line.strip() else {}
                if message.get("type") != "imu" or lines <= 0:
                    continue
                t, esp_us, *raw = message["s"][0]
                acc = [round(v * message["acc_scale"], 2) for v in raw[:3]]
                gyro = [round(v * message["gyro_scale"], 3) for v in raw[3:]]
                print(f"t {t:.6f} esp_us {esp_us} n {len(message['s'])} acc {acc} m/s2 gyro"
                      f" {gyro} rad/s")  # fmt: skip
                lines -= 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
