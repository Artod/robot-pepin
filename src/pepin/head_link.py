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
'C'    host -> ESP  config: id u8, IMU rate u16, DLPF u8, accel FS u8, gyro FS u8, brightness u8,
                    face style u8 (config/face.json's styles in their order: 0 pepin,
                    1 rose, 2 manga, 3 anime, 4 kawaii, 5 comics), eyes brightness u8 (0: the
                    eyes off), eye set u8 (:data:`pepin.face.EYE_SET_CHOICES`: 0 follow the
                    face style, 1 classic, 2 kawaii); a frame that stops before a field leaves
                    it as it is
'Y'    host -> ESP  eyes (:func:`encode_eyes`): mode u8, ms u16, brow_pos i8, brow_raise i8,
                    brow_width u8, lid_top u8, lid_bottom u8, gaze_x i8, gaze_y i8, iris r g b
                    u8, glow u8, blink u8, side u8[, effect u8, effect_ms u16]
=====  ===========  =====================================================================

'S' carries, after its 36 bytes of firmware 3, the eyes (firmware 4): their current estimate
mA u16, their brightness u8, their mode u8 (:data:`EYES_MODES`) and the eye layout's id u8, then
(a build of 2026-10-15 on) the look set in force u8 (:data:`pepin.face.EYE_SETS`: 0 classic,
1 kawaii); the shorter frame of an older firmware decodes with those fields None. A firmware 4
built before the sets drops a 'C' of ten bytes whole.

THE EYES ('Y'). Two rings of addressable LEDs around the camera's lenses, drawn by the same
ESP32; an 'E' sets their default look with the mouth's, and a 'Y' overrides it field by field.
``mode`` 0 releases the override (back to the expression's own look), 1 overrides until the next
'Y', 2 for ``ms``. Each frame is the WHOLE override of the eyes ``side`` names (0 both, 1 left,
2 right): a field not overridden carries the sentinel (0x80 for an i8, 0xFF for a u8, iris
0,0,0) and shows the expression's own value. The fields are config/face.json's eye parameters,
an i8 as -100..100 for -1..1 and a u8 as 0..254 for 0..1: ``brow_pos`` toward the nose ..
outward (0 on top), ``brow_raise`` low .. high, ``brow_width``, ``lid_top`` and ``lid_bottom``
open .. closed, ``gaze_x`` the robot's left .. its right, ``gaze_y`` down .. up, ``glow``.
``blink`` is a one-off on top of the frame: 0 none, 1 both eyes, 2 a wink of the left, 3 of the
right, whatever ``side`` says. The 19-byte form appends the inner ring's effect (config/face.json
``eyes.effects``, :data:`pepin.face.EYE_EFFECTS` by id: 0 none, which is back to the look's own
effect, 1 sparkle, 2 flash, 3 spin, 4 pulse, 5 flicker, 6 doze, 7 tear) and its life
``effect_ms`` (0: a one-shot's own length, a standing one until the next effect byte); the
16-byte form (or 0xFF in the effect byte) leaves the running effect as it is, and a release
(mode 0) ends it.

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
     "brightness": 200, "style": "anime", "eyes_brightness": 64, "eye_set": "kawaii"}
                                               any of them; style: the face's style, live;
                                               eye_set: the eyes' look set (follow, classic,
                                               kawaii), live
    {"cmd": "eyes", "source": "llm", "brow_raise": -0.6, "lid_top": 0.5, "side": "both",
     "ms": 1500}                               a source's eyes override (:class:`Eyes`' fields in
                                               config/face.json's units); ms null: until replaced
    {"cmd": "eyes_release", "source": "llm"}   the source overrides the eyes no more (its
                                               effect ends too)
    {"cmd": "eyes_effect", "source": "llm", "name": "spin", "ms": 2000}
                                               a source's effect (:data:`pepin.face.EYE_EFFECTS`)
                                               on the inner rings; ms null: until replaced (a
                                               one-shot: its own length); name null ends it
    {"cmd": "blink"}  or  {"cmd": "wink", "side": "left"}
    {"cmd": "gaze", "x": 0.4, "y": -0.1, "tracking": false}
                                               where the head looks as the eyes' gaze_x and
                                               gaze_y, -1..1 (the gaze node)
    {"cmd": "subscribe", "imu": true}          this client gets the IMU lines below
    {"cmd": "status"}

and gets ``{"type": "ack", ...}`` / ``{"type": "error", "error": ...}`` for each and the status
line once a second. A subscriber gets (the schema of the VIO design, vio.md section 2) first::

    {"type": "imu_config", "cfg": 1, "rate_hz": 200, "dlpf": 3, "gyro_fs_dps": 500,
     "accel_fs_g": 4, "filter_delay_s": 0.0048}

(again whenever the IMU's sampling changes), then one line per serial 'I' frame (every 20 ms)::

    {"type": "imu", "cfg": 1, "samples": [[t_mono_s, esp_us, gx, gy, gz, ax, ay, az], ...]}

``t_mono_s`` is the sample's data-ready edge on the board's ``time.monotonic`` (the ESP32's micros
fitted against CLOCK_MONOTONIC_RAW, carried to MONOTONIC when the line is written); the filter's
delay is NOT taken off (the consumer does, from ``filter_delay_s``). ``esp_us`` is the ESP32's own
micros, unwrapped; gyro in rad/s and accel in m/s^2, in the chip's axes, biases not removed.

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
from dataclasses import asdict, dataclass, field
from typing import Any, Protocol

from pepin.face import EYE_EFFECTS, EYE_SET_CHOICES, EYE_SETS
from pepin.streams import JsonLinesClient

logger = logging.getLogger(__name__)

HEAD_PORT = 3340
SYNC = 0xA5
MAX_PAYLOAD = 1024
OVERHEAD = 5
IMU, STATUS, PONG = ord("I"), ord("S"), ord("P")
EXPRESSION, MOUTH, INFO, PING, CONFIG = ord("E"), ord("M"), ord("T"), ord("Q"), ord("C")
EYES = ord("Y")

IMU_SAMPLE = struct.Struct("<I6hB")
STATUS_FIELDS = struct.Struct("<IHHIIIIBBBBBBHI")
STATUS_EYES = struct.Struct("<HBBB")  # appended by firmware 4
STATUS_EYE_SET = struct.Struct("<B")  # appended by firmware 4 of 2026-10-15 on: the set in force
PONG_FIELDS = struct.Struct("<II")
CONFIG_FIELDS = struct.Struct("<BHBBBBB")
EYES_FIELDS = struct.Struct("<BHbbBBBbbBBBBBB")
EYES_EFFECT = struct.Struct("<BH")  # the 19-byte 'Y' frame's tail: effect id, effect_ms
KEEP_I8, KEEP_U8 = -128, 0xFF  # a field the 'Y' frame does not override (0x80, 0xFF)
NO_EFFECT = EYE_EFFECTS[0]  # the effect byte 0: back to the look's own effect
EYES_BRIGHTNESS = 64  # the firmware's own until a 'C' carries one
EYE_RELEASE, EYE_STANDING, EYE_TIMED = 0, 1, 2  # the 'Y' frame's modes
SIDES = ("both", "left", "right")  # the 'Y' frame's side byte
BLINKS = ("none", "both", "left", "right")  # its blink byte: 2 and 3 are winks
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


IMU_STATES = {0: "absent", 1: "interrupt", 2: "polled", 3: "no_int"}  # 3: no data-ready edge
MODES = {0: "face", 1: "info", 2: "asleep"}
EYES_MODES = {0: "expression", 1: "standing", 2: "timed", 3: "self_test"}  # the eyes' look


@dataclass(frozen=True)
class HeadStatus:
    """The ESP32's 'S' frame: its own view of the face, the IMU and the link."""

    esp_us: int
    fps: float
    imu_rate_hz: float  # the chip's output rate in its own stamps' time
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
    gaps: int  # stamps more than 1.5 periods apart
    # the eyes (firmware 4; None from an older one)
    eyes_ma: int | None = None  # the rings' current, the firmware's estimate
    eyes_brightness: int | None = None
    eyes_mode: str | None = None  # EYES_MODES: the expression's look, an override, the boot sweep
    eye_layout: int | None = None  # the LED chain's layout id (pepin.face.EyeTable.layout_id)
    eye_set: str | None = None  # the look set in force (pepin.face.EYE_SETS; None: no sets yet)

    def as_dict(self) -> dict[str, Any]:
        """The fields, for a JSON line."""
        return asdict(self)


def decode_status(payload: bytes) -> HeadStatus:
    """An 'S' payload, with or without the eyes' fields and their set; ``struct.error`` when it
    is the wrong size."""
    extra = len(payload) - STATUS_FIELDS.size
    with_set = STATUS_EYES.size + STATUS_EYE_SET.size
    if extra not in (0, STATUS_EYES.size, with_set):
        raise struct.error(
            f"a status of {len(payload)} bytes: {STATUS_FIELDS.size},"
            f" {STATUS_FIELDS.size + STATUS_EYES.size} with the eyes or"
            f" {STATUS_FIELDS.size + with_set} with their set"
        )
    f = STATUS_FIELDS.unpack_from(payload)
    eyes = STATUS_EYES.unpack_from(payload, STATUS_FIELDS.size) if extra else (None,) * 4
    eye_set = None
    if extra == with_set:
        (code,) = STATUS_EYE_SET.unpack_from(payload, STATUS_FIELDS.size + STATUS_EYES.size)
        eye_set = EYE_SETS[code] if code < len(EYE_SETS) else str(code)
    return HeadStatus(
        esp_us=f[0],
        fps=f[1] / 10.0,
        imu_rate_hz=f[2] / 10.0,
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
        gaps=f[14],
        eyes_ma=eyes[0],
        eyes_brightness=eyes[1],
        eyes_mode=None if eyes[2] is None else EYES_MODES.get(eyes[2], str(eyes[2])),
        eye_layout=eyes[3],
        eye_set=eye_set,
    )


def encode_status(status: HeadStatus) -> bytes:
    """An 'S' payload (the fake head of the tests speaks it); the eyes' fields appended when
    ``eyes_ma`` is set, and their set after them when ``eye_set`` is too."""
    mode = {v: k for k, v in MODES.items()}.get(status.mode, 0)
    imu = {v: k for k, v in IMU_STATES.items()}.get(status.imu_state, 0)
    eyes = b""
    if status.eyes_ma is not None:
        eyes_mode = {v: k for k, v in EYES_MODES.items()}.get(status.eyes_mode or "", 0)
        eyes = STATUS_EYES.pack(
            status.eyes_ma, status.eyes_brightness or 0, eyes_mode, status.eye_layout or 0
        )
        if status.eye_set is not None:
            code = EYE_SETS.index(status.eye_set) if status.eye_set in EYE_SETS else 0
            eyes += STATUS_EYE_SET.pack(code)
    head = STATUS_FIELDS.pack(
        status.esp_us & 0xFFFFFFFF,
        round(status.fps * 10),
        round(status.imu_rate_hz * 10),
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
        status.gaps,
    )
    return head + eyes


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


def encode_config(
    config: ImuConfig,
    brightness: int,
    style: int = 0,
    eyes_brightness: int | None = None,
    eye_set: int | None = 0,
) -> bytes:
    """A 'C' payload; ``style`` is the face's style index (``FaceTable.style_id_of``),
    ``eyes_brightness`` the rings' (0: off), ``eye_set`` the eyes' look set
    (:func:`pepin.face.eye_set_id`: 0 follow the face style, 1 classic, 2 kawaii). The frame
    stops before the first None, the firmware leaving that field and the ones after it as they
    are: the eye set rides only behind the eyes' brightness."""
    if eye_set is not None and not 0 <= eye_set < len(EYE_SET_CHOICES):
        choices = ", ".join(f"{i} {name}" for i, name in enumerate(EYE_SET_CHOICES))
        raise ValueError(f"eye set {eye_set}: {choices}")
    payload = CONFIG_FIELDS.pack(
        config.id,
        config.rate_hz,
        config.dlpf,
        config.accel_fs,
        config.gyro_fs,
        min(max(brightness, 0), 255),
        min(max(style, 0), 255),
    )
    if eyes_brightness is None:
        return payload
    payload += bytes((min(max(eyes_brightness, 0), 255),))
    return payload if eye_set is None else payload + bytes((eye_set,))


# The eyes' scalar fields in the 'Y' frame's order, each with its range in the door's units
# (config/face.json's eye parameters); on the wire a signed one is -100..100, an unsigned one
# 0..254 (below the sentinel 0xFF).
EYE_RANGES: dict[str, tuple[float, float]] = {
    "brow_pos": (-1.0, 1.0),
    "brow_raise": (-1.0, 1.0),
    "brow_width": (0.0, 1.0),
    "lid_top": (0.0, 1.0),
    "lid_bottom": (0.0, 1.0),
    "gaze_x": (-1.0, 1.0),
    "gaze_y": (-1.0, 1.0),
    "glow": (0.0, 1.0),
}
EYE_KEYS = (*EYE_RANGES, "iris")  # every field of :class:`Eyes`
I8_FULL, U8_FULL = 100, 254  # a signed field's 1.0 on the wire, an unsigned one's


def _full(name: str) -> int:
    """Field ``name``'s 1.0 on the wire."""
    return I8_FULL if EYE_RANGES[name][0] < 0.0 else U8_FULL


@dataclass(frozen=True)
class Eyes:
    """An override of the eyes' look; a field left None is not overridden (the shown
    expression's own value). The fields hold the 'Y' frame's integers (-100..100 signed, 0..254
    unsigned); :meth:`of` and :meth:`as_dict` speak the door's units, config/face.json's.
    ``effect`` calls the inner ring's effect by name (:data:`pepin.face.EYE_EFFECTS`; ``none``:
    back to the look's own) for ``effect_ms`` (0: a one-shot's own length, a standing one until
    the next); None leaves the running effect alone. The effect is no look field: :meth:`of`,
    :meth:`over` and :meth:`as_dict` leave it out (the head server arbitrates it apart)."""

    brow_pos: int | None = None
    brow_raise: int | None = None
    brow_width: int | None = None
    lid_top: int | None = None
    lid_bottom: int | None = None
    gaze_x: int | None = None
    gaze_y: int | None = None
    glow: int | None = None
    iris: tuple[int, int, int] | None = None
    effect: str | None = None
    effect_ms: int = 0

    def __post_init__(self) -> None:
        if self.effect is not None and self.effect not in EYE_EFFECTS:
            raise ValueError(f"no eye effect {self.effect!r}: {', '.join(EYE_EFFECTS)}")

    @classmethod
    def of(cls, fields: Mapping[str, Any]) -> Eyes:
        """From a door command's fields in config/face.json's units (out of range clamped;
        ``iris`` as ``[r, g, b]`` or ``"#RRGGBB"``; null as absent); ``ValueError`` naming an
        unknown field."""
        unknown = sorted(set(fields) - set(EYE_KEYS))
        if unknown:
            raise ValueError(
                f"no eye field {', '.join(unknown)}; the eyes have {', '.join(EYE_KEYS)}"
            )
        values: dict[str, Any] = {}
        for name, (lo, hi) in EYE_RANGES.items():
            value = fields.get(name)
            if value is not None:
                values[name] = round(min(max(float(value), lo), hi) * _full(name))
        iris = fields.get("iris")
        if iris is not None:
            values["iris"] = _rgb(iris)
        return cls(**values)

    @property
    def empty(self) -> bool:
        """No look field overridden (the effect aside)."""
        return all(v is None for _, v in self.fields())

    def over(self, under: Eyes) -> Eyes:
        """This override laid over ``under``: its fields where it sets them, ``under``'s
        elsewhere."""
        return Eyes(**{k: v if v is not None else getattr(under, k) for k, v in self.fields()})

    def fields(self) -> list[tuple[str, Any]]:
        """Every field and its wire value, None included."""
        return [(name, getattr(self, name)) for name in EYE_KEYS]

    def as_dict(self) -> dict[str, Any]:
        """The overridden fields in the door's units (to 0.01), for a JSON line."""
        return {
            k: list(v) if k == "iris" else round(v / _full(k), 2)
            for k, v in self.fields()
            if v is not None
        }


def _rgb(value: Any) -> tuple[int, int, int]:
    """``[r, g, b]`` or ``"#RRGGBB"`` as three bytes; ``ValueError`` otherwise."""
    if isinstance(value, str):
        if not re.fullmatch(r"#[0-9A-Fa-f]{6}", value):
            raise ValueError(f"iris #RRGGBB, not {value!r}")
        return int(value[1:3], 16), int(value[3:5], 16), int(value[5:7], 16)
    r, g, b = (min(max(int(c), 0), 255) for c in value)
    return r, g, b


@dataclass(frozen=True)
class EyesFrame:
    """One 'Y' frame: the override of the eyes ``side`` names (0 both, 1 left, 2 right),
    ``mode`` (:data:`EYE_RELEASE`, :data:`EYE_STANDING`, :data:`EYE_TIMED` for ``ms``) and a
    one-off ``blink`` (:data:`BLINKS`' index)."""

    eyes: Eyes = field(default_factory=Eyes)
    mode: int = EYE_STANDING
    ms: int = 0
    blink: int = 0
    side: int = 0


def encode_eyes(frame: EyesFrame) -> bytes:
    """A 'Y' payload: every field not overridden as its sentinel; 19 bytes when the frame
    calls an effect (``eyes.effect``), else 16 (the running effect left alone)."""
    e = frame.eyes

    def i8(value: int | None) -> int:
        return KEEP_I8 if value is None else value

    def u8(value: int | None) -> int:
        return KEEP_U8 if value is None else value

    r, g, b = e.iris or (0, 0, 0)
    payload = EYES_FIELDS.pack(
        frame.mode,
        min(max(frame.ms, 0), 65535),
        i8(e.brow_pos),
        i8(e.brow_raise),
        u8(e.brow_width),
        u8(e.lid_top),
        u8(e.lid_bottom),
        i8(e.gaze_x),
        i8(e.gaze_y),
        r,
        g,
        b,
        u8(e.glow),
        frame.blink,
        frame.side,
    )
    if e.effect is None:
        return payload
    return payload + EYES_EFFECT.pack(EYE_EFFECTS.index(e.effect), min(max(e.effect_ms, 0), 65535))


def decode_eyes(payload: bytes) -> EyesFrame:
    """A 'Y' payload of 16 or 19 bytes (the fake head of the tests reads it); ``struct.error``
    when it is another size."""
    effect, effect_ms = None, 0
    if len(payload) == EYES_FIELDS.size + EYES_EFFECT.size:
        code, effect_ms = EYES_EFFECT.unpack_from(payload, EYES_FIELDS.size)
        known = EYE_EFFECTS[code] if code < len(EYE_EFFECTS) else NO_EFFECT  # as the firmware
        effect = None if code == KEEP_U8 else known
        payload = payload[: EYES_FIELDS.size]
    (mode, ms, brow_pos, brow_raise, brow_width, lid_top, lid_bottom, gaze_x, gaze_y, r, g, b,
     glow, blink, side) = EYES_FIELDS.unpack(payload)  # fmt: skip

    def i8(value: int) -> int | None:
        return None if value == KEEP_I8 else value

    def u8(value: int) -> int | None:
        return None if value == KEEP_U8 else value

    eyes = Eyes(
        brow_pos=i8(brow_pos),
        brow_raise=i8(brow_raise),
        brow_width=u8(brow_width),
        lid_top=u8(lid_top),
        lid_bottom=u8(lid_bottom),
        gaze_x=i8(gaze_x),
        gaze_y=i8(gaze_y),
        glow=u8(glow),
        iris=None if (r, g, b) == (0, 0, 0) else (r, g, b),
        effect=effect,
        effect_ms=effect_ms if effect is not None else 0,
    )
    return EyesFrame(eyes, mode, ms, blink, side)


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

    def style(self, name: str | None, ms: float | None = None) -> None:
        """Draw the mouth in style ``name`` (config/face.json ``styles``): for ``ms``, then
        back to the standing one, or standing when ``ms`` is None; None lets this source's
        styles go (``style_release``)."""
        if name is None:
            self.send({"cmd": "style_release", "source": self.source})
            return
        self.send({"cmd": "style", "source": self.source, "name": name, "ms": ms})

    def mouth(self, level: float) -> None:
        """The speech level, 0..1."""
        self.send({"cmd": "mouth", "level": round(level, 3)})

    def lease(self, seconds: float) -> None:
        """This brain is here for ``seconds`` more (renew it well before it lapses)."""
        self.send({"cmd": "lease", "name": self.source, "seconds": seconds})

    def eyes(self, *, ms: float | None = None, side: str = "both", **fields: Any) -> None:
        """Override the eyes' look (:class:`Eyes`' fields, config/face.json's units) on
        ``side``: until replaced when ``ms`` is None, else for ``ms``."""
        self.send({"cmd": "eyes", "source": self.source, "side": side, "ms": ms, **fields})

    def eyes_release(self) -> None:
        """This source overrides the eyes no more (its effect ends too)."""
        self.send({"cmd": "eyes_release", "source": self.source})

    def eye_set(self, name: str) -> None:
        """The eyes' look set, standing until the next or a restart: ``follow`` (the face
        style's, config/face.json ``eyes.style_sets``), ``classic`` or ``kawaii``
        (:data:`pepin.face.EYE_SET_CHOICES`)."""
        self.send({"cmd": "config", "eye_set": name})

    def eyes_effect(self, name: str | None, ms: float | None = None) -> None:
        """The inner rings' effect ``name`` (:data:`pepin.face.EYE_EFFECTS`): until replaced
        when ``ms`` is None (a one-shot: its own length), else for ``ms``; None ends this
        source's effect."""
        self.send({"cmd": "eyes_effect", "source": self.source, "name": name, "ms": ms})

    def blink(self, side: str = "both") -> None:
        """One blink: both eyes, or a wink of ``side``."""
        self.send({"cmd": "blink", "source": self.source, "side": side})

    def gaze(self, x: float, y: float, *, tracking: bool = False) -> None:
        """Where the head looks as the eyes' gaze, -1..1 each way (x + to the robot's right,
        y + up), and whether it follows a person."""
        self.send({"cmd": "gaze", "x": round(x, 3), "y": round(y, 3), "tracking": tracking})


class GazeSink(Protocol):
    """Where :class:`EyeGazeLink` sends: :class:`HeadClient` has it."""

    def gaze(self, x: float, y: float, *, tracking: bool = False) -> None:
        """Where the head looks, -1..1 each way, and whether it follows a person."""
        ...


PERIOD_SLACK = 0.9  # a send may go once 90 % of the period passed: a timer's jitter keeps the rate


class EyeGazeLink:
    """Where the head looks, to the eyes' gaze, without flooding anything: sent when it moved
    ``min_change`` or more either way (or ``tracking`` flipped), at most every
    ``min_period_s``, and once every ``refresh_s`` without such a change (a send is dropped
    while the head server is away; the head server writes the serial frame only on a
    change)."""

    def __init__(
        self,
        sink: GazeSink,
        *,
        min_period_s: float = 0.1,
        min_change: float = 0.02,
        refresh_s: float = 5.0,
    ) -> None:
        """Sends go to ``sink``."""
        self._sink = sink
        self.min_period_s = min_period_s
        self.min_change = min_change
        self.refresh_s = refresh_s
        self._sent: tuple[float, float, bool] | None = None
        self._at = -math.inf
        self.sends = 0

    def update(self, x: float, y: float, tracking: bool, now: float) -> bool:
        """The head's newest aim; whether it went out."""
        x, y = min(max(x, -1.0), 1.0), min(max(y, -1.0), 1.0)
        if now - self._at < self.min_period_s * PERIOD_SLACK:
            return False
        last = self._sent
        step = self.min_change - 1e-9  # 0.06 - 0.04 is a step of 0.02
        moved = (
            last is None
            or tracking != last[2]
            or abs(x - last[0]) >= step
            or abs(y - last[1]) >= step
        )
        if not moved and now - self._at < self.refresh_s:
            return False
        self._sent, self._at = (x, y, tracking), now
        self._sink.gaze(x, y, tracking=tracking)
        self.sends += 1
        return True


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


# The command line's short names for the eyes' fields.
EYE_ALIASES = {
    "pos": "brow_pos",
    "brow": "brow_raise",
    "width": "brow_width",
    "lid": "lid_top",
    "lower": "lid_bottom",
    "x": "gaze_x",
    "y": "gaze_y",
}
COMMANDS = (
    "status", "express", "event", "clear", "show", "style", "eyes", "blink", "wink", "gaze", "imu"
)  # fmt: skip


def cli_request(command: str, a: list[str]) -> dict[str, Any]:
    """A command line's verb and words as the head server's request; ``ValueError`` (or
    ``IndexError``) when they make none."""
    if command == "express":
        hold = float(a[1]) if len(a) > 1 else None
        return {"cmd": "express", "source": "cli", "name": a[0], "hold_s": hold}
    if command == "event":
        return {"cmd": "event", "source": "cli", "name": a[0]}
    if command == "clear":
        return {"cmd": "clear", "source": "cli"}
    if command == "show":
        return {"cmd": "show", "text": a[0], "seconds": float(a[1]) if len(a) > 1 else 8.0}
    if command == "style":  # the face's style, live (config/face.json `styles`)
        return {"cmd": "config", "style": a[0]}
    if command == "eyes":
        if a[:1] == ["release"]:
            return {"cmd": "eyes_release", "source": "cli"}
        if a[:1] == ["effect"]:  # eyes effect spin [ms] | eyes effect none
            name = None if a[1] in ("none", "off") else a[1]
            return {"cmd": "eyes_effect", "source": "cli", "name": name,
                    "ms": float(a[2]) if len(a) > 2 else None}  # fmt: skip
        if a[:1] == ["brightness"]:
            return {"cmd": "config", "eyes_brightness": int(a[1])}
        if a[:1] == ["set"]:  # eyes set kawaii | classic | follow
            return {"cmd": "config", "eye_set": a[1]}
        request: dict[str, Any] = {"cmd": "eyes", "source": "cli"}
        for word in a:
            key, equals, value = word.partition("=")
            if not equals:
                raise ValueError(f"eyes takes field=value words, not {word!r}")
            key = EYE_ALIASES.get(key, key)
            if key == "side" or key == "iris":
                request[key] = value
            elif key == "ms" or key in EYE_RANGES:
                request[key] = float(value)
            else:
                known = ", ".join([*EYE_KEYS, *EYE_ALIASES, "side", "ms"])
                raise ValueError(f"no eye field {key!r}: {known}")
        return request
    if command == "blink":
        return {"cmd": "blink", "source": "cli"}
    if command == "wink":
        return {"cmd": "wink", "source": "cli", "side": a[0] if a else "left"}
    if command == "gaze":
        x, y = float(a[0]), float(a[1]) if len(a) > 1 else 0.0
        return {"cmd": "gaze", "x": x, "y": y, "tracking": "tracking" in a[2:]}
    if command == "status":
        return {"cmd": "status"}
    raise ValueError(f"no command {command!r}: {', '.join(COMMANDS)}")


def main(argv: list[str] | None = None) -> int:
    """Talk to the head server by hand: the bring-up's and a bench's commands."""
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description="Talk to the head server (pepin.head_server, TCP 3340).",
        epilog="e.g.  express happy 3   |   event arrived   |   show 'Temps|left: 41/70 C' 6"
        "   |   style anime   |   eyes brow=-0.6 lid=0.5 ms=1500   |   eyes release   |   blink"
        "   |   wink left   |   gaze 0.4   |   eyes brightness 90   |   eyes set kawaii"
        "   |   eyes effect spin 2000   |   eyes effect none   |   imu 5   |   status",
    )
    parser.add_argument("--host", default=os.environ.get("PEPIN_HOST", "10.0.0.187"))
    parser.add_argument("--port", type=int, default=HEAD_PORT)
    parser.add_argument("command", choices=COMMANDS)
    parser.add_argument("args", nargs="*")
    args = parser.parse_args(argv)
    if args.command == "imu":
        return _print_imu(args.host, args.port, int(args.args[0]) if args.args else 5)
    try:
        request = cli_request(args.command, args.args)
    except (ValueError, IndexError) as error:
        parser.error(f"{args.command}: {error or 'a word is missing'}")
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
                if message.get("type") == "imu_config":
                    print(json.dumps(message))
                if message.get("type") != "imu" or lines <= 0:
                    continue
                t, esp_us, *values = message["samples"][0]
                gyro = [round(v, 3) for v in values[:3]]
                acc = [round(v, 2) for v in values[3:]]
                print(f"t {t:.6f} esp_us {esp_us} n {len(message['samples'])} gyro {gyro} rad/s"
                      f" acc {acc} m/s2")  # fmt: skip
                lines -= 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
