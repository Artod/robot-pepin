"""Head server: runs on the board and owns the head ESP32's serial port (TCP 3340).

The ESP32 under the stereo camera is the robot's mouth (its 1.9" screen) and eyes (two LED
rings around the lenses), and carries the head IMU (an MPU6050 glued to the camera body). This
process is the only one that opens its port (``/dev/pepin-head``, the CH340 at 921600 baud,
framing in :mod:`pepin.head_link`), and it does five things with it:

- **The face.** Any client sends expressions, robot events (config/face.json's table), speech
  levels and info screens. One arbiter decides what shows: every source (the goal server, the
  voice loop, the LLM's tools) holds at most one standing expression and one timed one, the
  newest request of all wins, and a timed one lapses back to whatever stood before it. Only a
  change goes down the wire. Speech levels ('M', the board's own audio server while it plays)
  and info screens ('T') pass straight through. The mouth's drawing style (config/face.json
  ``styles``: ``style``, ``style_release``) is arbitrated the same way beside the expression:
  per source one standing and one timed, the newest wins, a timed one lapses back to the
  newest standing one or to the base (config/face.json's ``style``, or a ``config``'s); it
  goes down in the 'C' frame when it changes, with the IMU's config and id as they stand (the
  firmware sets the IMU up again, a reset and 100 ms without samples, only when those change;
  one built before 2026-10-15 did on every 'C').
- **The eyes.** An expression sets the eyes' own look with the mouth's; any client overrides it
  field by field (brows, lids, gaze, iris, glow; :class:`pepin.head_link.Eyes`). A second
  arbiter decides what shows: every source holds at most one standing override and one timed
  one, each field of each eye shows the newest override that sets it, and a timed one lapses
  back to what stood before it. The gaze node's ``gaze`` (where the head looks) is the gaze
  layer, under every other source's override; while the head follows a person the eyes look
  straight out, at theirs. The inner rings' effect (``eyes_effect``: spin, pulse, ...) is
  arbitrated the same way beside the look (per source one standing and one timed, the newest
  wins, a timed one lapses here; a one-shot asked for without a time holds its own length,
  config/face.json ``eyes.effects``), and goes down in the 'Y' frame's 19-byte form when it
  changes. Only a change goes down the wire ('Y', the whole override each time); a blink or a
  wink goes at once. The eyes' brightness is config/face.json's, sent with the IMU's config
  ('C') to a firmware that has eyes; so is their look set (``eyes.eye_set``: follow the face
  style, classic or kawaii; live through a ``config``), to a firmware whose status carries the
  set in force (one built before the sets drops a 'C' of ten bytes whole, and its first status
  gets one of nine).
- **The head IMU.** Every 'I' frame's samples go to the subscribed clients (the C++ base bridge)
  as one JSON line, each sample with both clocks: the ESP32's micros, unwrapped, and its moment
  on this board's monotonic clock.
- **The clock map.** ESP32 micros -> board time by the lower envelope of (receive time - ESP
  micros - the frame's known wire and read time) over the pongs and the samples, against
  CLOCK_MONOTONIC_RAW (chrony slews MONOTONIC by up to 2000 ppm; RAW leaves the crystal's ratio
  alone): the smallest difference per second of ESP time, a line through the last 120 s of those
  (offset and skew; the ESP32's crystal drifts tens of ppm), less half the smallest ping round
  trip (the envelope sits one transit above the truth). A sample is carried to MONOTONIC when
  its line is written, by reading both clocks back to back.
- **The brain lease.** A laptop process (the goal server, with its face flag on) renews a lease
  every few seconds; once a lease was taken and every lease has lapsed, the laptop or the WiFi
  is gone and the face falls asleep (config/face.json's ``brain_lost``) with the sleepy look's
  own eyes: every eyes override is held back until a brain is here again. The ESP32 falls asleep
  on its own when this server goes silent (3 s without a frame).

A status line goes to every client once a second and to the journal once a minute. Run on the
board (``board/pepin-head.service``)::

    python -m pepin.head_server

Standard library only. The pure parts (clock map, arbiter, leases, the service against a fake
head) are unit-tested on the laptop.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import math
import os
import select
import signal
import struct
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol

from pepin.face import (
    EYE_EFFECTS,
    EYE_SET_CHOICES,
    FaceTable,
    eye_set_id,
    load_face_table,
)
from pepin.head_link import (
    BLINKS,
    CONFIG,
    EXPRESSION,
    EYE_RELEASE,
    EYE_STANDING,
    EYES,
    EYES_BRIGHTNESS,
    FIRMWARE_DEFAULTS,
    HEAD_PORT,
    IMU,
    IMU_SAMPLE,
    INFO,
    MOUTH,
    NO_EFFECT,
    OVERHEAD,
    PING,
    PONG,
    SIDES,
    STATUS,
    Eyes,
    EyesFrame,
    FrameDecoder,
    HeadStatus,
    ImuConfig,
    decode_pong,
    decode_status,
    encode_config,
    encode_expression,
    encode_eyes,
    encode_frame,
    encode_info,
    encode_mouth,
    encode_ping,
    show_items,
)
from pepin.overlay import read_json
from pepin.streams import JsonLinesServer
from pepin.streams import encode as encode_line

logger = logging.getLogger(__name__)

RESET_US = 1_000_000  # micros going back by more than this: the ESP32 rebooted
RETRY_S = 1.0  # a missing or broken port is opened again this often
REPORT_EVERY_S = 60.0
RTT_WINDOW = 120  # pings the smallest round trip is taken over (a minute at 2 Hz)
ENVELOPE_TOL_S = 0.0003  # a bucket's minimum this far above the fitted line is not the envelope
CONFIG_MISMATCHES = 2  # status lines naming another config id before it is sent again
OUTBOX_LINES = 400  # ~2 s of the IMU stream at 200 lines/s before a subscriber is dropped
# From an IMU sample's data-ready edge to its bytes leaving: the 14-byte burst read at 400 kHz
# (address, register, repeated start, address, 14 bytes: 18 bytes of 9 bits, ~0.4 ms, and the
# Arduino core's own overhead). An ESTIMATE until measured on the head (a scope on SCL and
# INT); an error in it shifts every IMU pair of the clock map by the same amount.
I2C_READ_S = 0.00045
PING_BYTES = 4
EYES_FIRMWARE = 4  # the first firmware with eyes: 'Y', the 'C' frame's ninth byte, the longer 'S'
GAZE_SOURCE = "gaze"  # the eyes' gaze layer ({"cmd": "gaze"}): under every other override
EYE_COMMAND_KEYS = ("cmd", "source", "side", "ms")  # an eyes command's keys beside the fields
ONE_SHOTS = ("sparkle", "flash")  # the effects that end on their own (eyes.cpp one_shot)


# -- the serial port ------------------------------------------------------------------------------


class ByteLink(Protocol):
    """The ESP32's serial port, or a fake of it."""

    def read(self, timeout_s: float) -> bytes:
        """What arrived, waiting up to ``timeout_s`` for anything; ``OSError`` when it is gone."""
        ...

    def write(self, data: bytes) -> None:
        """Send ``data``; ``OSError`` when the port is gone."""
        ...

    def close(self) -> None:
        """Release the port."""
        ...


class Subscriber(Protocol):
    """One connected client (:class:`pepin.streams.ClientConn`), as far as the IMU stream
    needs it."""

    @property
    def alive(self) -> bool:
        """Still connected."""
        ...

    def post(self, line: bytes) -> None:
        """Queue one line for it."""
        ...


class Clients(Protocol):
    """The head server's sockets (:class:`JsonLinesServer`), or a fake of them."""

    @property
    def client_count(self) -> int:
        """How many clients are connected."""
        ...

    def commands(self) -> list[tuple[Any, dict[str, Any]]]:
        """What clients sent since the last call."""
        ...

    def reply(self, client: Any, message: dict[str, Any]) -> None:
        """One line to one client."""
        ...

    def broadcast(self, message: dict[str, Any]) -> None:
        """One line to every client."""
        ...

    def close(self) -> None:
        """Drop every client."""
        ...


class SerialPort:
    """A tty in raw mode at ``baud`` (termios: no pyserial on the board). Both modem lines are
    released in one ioctl, so the ESP32's auto-reset circuit (EN and IO0 driven from DTR/RTS)
    sees them change together; opening the port may still reset the ESP32 once (~1 s dark)."""

    def __init__(self, path: str, baud: int) -> None:
        """Open ``path``; ``OSError`` when it is absent, ``ValueError`` for a baud termios lacks."""
        import fcntl
        import termios

        speed = getattr(termios, f"B{baud}", None)
        # macOS has no B921600: open at B230400, then set the real rate with IOSSIOSPEED (as
        # pyserial does), so the head can be brought up on the laptop's USB before the robot's.
        mac_speed = speed is None and sys.platform == "darwin"
        if mac_speed:
            speed = termios.B230400
        if speed is None:
            raise ValueError(f"{baud} baud is not a termios speed on this system")
        self.path = path
        self._fd = os.open(path, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            attrs = termios.tcgetattr(self._fd)
            attrs[0] = termios.IGNBRK  # iflag: raw
            attrs[1] = 0  # oflag
            attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL  # no HUPCL: close drops nothing
            attrs[3] = 0  # lflag: no echo, no canonical lines, no signals
            attrs[4] = attrs[5] = speed
            attrs[6][termios.VMIN] = 0
            attrs[6][termios.VTIME] = 0
            termios.tcsetattr(self._fd, termios.TCSANOW, attrs)
            if mac_speed:
                fcntl.ioctl(self._fd, 0x80045402, struct.pack("I", baud))  # IOSSIOSPEED
            lines = getattr(termios, "TIOCM_DTR", 0x002) | getattr(termios, "TIOCM_RTS", 0x004)
            with contextlib.suppress(OSError):
                fcntl.ioctl(self._fd, getattr(termios, "TIOCMBIC", 0x5417), struct.pack("I", lines))
            termios.tcflush(self._fd, termios.TCIOFLUSH)
        except BaseException:
            os.close(self._fd)
            raise

    def read(self, timeout_s: float) -> bytes:
        """Bytes that arrived within ``timeout_s``; ``OSError`` when the device went away."""
        ready, _, _ = select.select([self._fd], [], [], timeout_s)
        if not ready:
            return b""
        data = os.read(self._fd, 65536)
        if not data:
            raise OSError(f"{self.path} went away")
        return data

    def write(self, data: bytes) -> None:
        """All of ``data``, waiting up to 50 ms for room; ``OSError`` past that."""
        view = memoryview(data)
        deadline = time.monotonic() + 0.05
        while view:
            try:
                sent = os.write(self._fd, view)
            except BlockingIOError:
                sent = 0
            view = view[sent:]
            if view:
                left = deadline - time.monotonic()
                if left <= 0:
                    raise OSError(f"{self.path}: the write did not drain in 50 ms")
                select.select([], [self._fd], [], left)

    def close(self) -> None:
        """Release the tty."""
        with contextlib.suppress(OSError):
            os.close(self._fd)


# -- the clocks ------------------------------------------------------------------------------------


class MicrosUnwrapper:
    """The ESP32's 32-bit micros (wrapping every 71.6 min) as one growing count. Small steps
    back are normal (a pong and a batch cross on the wire); a step back of more than a second
    is a reboot, and the count starts over."""

    def __init__(self) -> None:
        """Nothing seen yet."""
        self._raw: int | None = None
        self._value = 0
        self.resets = 0

    def unwrap(self, raw: int) -> tuple[int, bool]:
        """``raw`` as the growing count, and whether it started over here."""
        if self._raw is None:
            self._raw, self._value = raw, raw
            return raw, False
        delta = (raw - self._raw) & 0xFFFFFFFF
        if delta >= 1 << 31:
            delta -= 1 << 32
        if delta < -RESET_US:
            self._raw, self._value = raw, raw
            self.resets += 1
            return raw, True
        self._raw = raw
        self._value += delta
        return self._value, False


class ClockMap:
    """ESP32 micros -> board monotonic seconds, from (receive time, ESP time) pairs.

    Each pair gives ``d = receive - esp``, which is the clocks' offset plus that frame's transit;
    the smallest ``d`` per ``bucket_s`` of ESP time is the envelope, and a line through the last
    ``window_s`` of envelope points (refitted without the points more than 0.3 ms above it)
    carries the offset and the skew. The envelope is one minimal transit late; half the
    smallest ping round trip is taken off for it.
    """

    def __init__(self, bucket_s: float = 1.0, window_s: float = 120.0) -> None:
        """The envelope's resolution and memory."""
        self.bucket_s = bucket_s
        self.window_buckets = max(3, round(window_s / bucket_s))
        self._buckets: dict[int, tuple[float, float]] = {}  # bucket -> (d, esp seconds)
        self._rtts: deque[float] = deque(maxlen=RTT_WINDOW)
        self._fit: tuple[float, float, float] | None = None  # offset, slope, reference esp s
        self._spread = 0.0
        self._kept = 0
        self.observations = 0

    def reset(self) -> None:
        """Forget everything: the ESP32 rebooted, its micros started over."""
        self._buckets.clear()
        self._rtts.clear()
        self._fit = None
        self.observations = 0

    @property
    def ready(self) -> bool:
        """Whether any pair has been seen."""
        return self._fit is not None

    def observe(self, esp_us: int, host_s: float, known_s: float = 0.0) -> None:
        """One pair: ESP micros (unwrapped) and the board time the frame carrying it was read;
        ``known_s`` is the part of its way that is known and differs between frames (the
        frame's own bytes on the wire, an IMU sample's read), taken off so that every pair
        measures the same remaining transit."""
        esp_s = esp_us * 1e-6
        d = host_s - esp_s - known_s
        key = math.floor(esp_s / self.bucket_s)
        self.observations += 1
        held = self._buckets.get(key)
        if held is not None and held[0] <= d:
            return
        self._buckets[key] = (d, esp_s)
        for old in [k for k in self._buckets if k <= key - self.window_buckets]:
            del self._buckets[old]
        self._refit()

    def observe_rtt(self, rtt_s: float) -> None:
        """One ping's round trip, seconds, without the two frames' own bytes on the wire: twice
        the remaining transit, at best. Below zero (a port with no wire, a pseudo-terminal) it
        counts as zero."""
        self._rtts.append(max(rtt_s, 0.0))

    @property
    def min_rtt_s(self) -> float | None:
        """The smallest round trip of the last :data:`RTT_WINDOW` pings."""
        return min(self._rtts) if self._rtts else None

    def to_host(self, esp_us: int) -> float:
        """Board monotonic seconds at ESP micros ``esp_us`` (unwrapped); ``RuntimeError``
        before the first pair."""
        if self._fit is None:
            raise RuntimeError("the clock map has seen no pair yet")
        offset, slope, ref = self._fit
        esp_s = esp_us * 1e-6
        bias = (self.min_rtt_s or 0.0) / 2.0
        return esp_s + offset + slope * (esp_s - ref) - bias

    def linear(self) -> tuple[float, float]:
        """:meth:`to_host` as ``(scale, shift)``: board seconds = ESP seconds * scale + shift
        (what a batch of samples is mapped with in one go)."""
        if self._fit is None:
            raise RuntimeError("the clock map has seen no pair yet")
        offset, slope, ref = self._fit
        bias = (self.min_rtt_s or 0.0) / 2.0
        return 1.0 + slope, offset - slope * ref - bias

    def state(self) -> dict[str, Any]:
        """Offset, skew (ppm the ESP32's crystal runs fast), smallest round trip, the kept
        envelope points' spread above the line: for the status line."""
        if self._fit is None:
            return {"ready": False}
        offset, slope, _ = self._fit
        rtt = self.min_rtt_s
        return {
            "ready": True,
            "offset_s": round(offset - (rtt or 0.0) / 2.0, 6),
            "esp_fast_ppm": round(-slope * 1e6, 1),
            "min_rtt_ms": None if rtt is None else round(rtt * 1000.0, 3),
            "spread_ms": round(self._spread * 1000.0, 3),
            "points": self._kept,
        }

    def _refit(self) -> None:
        points = sorted(self._buckets.values(), key=lambda p: p[1])
        ref = points[-1][1]
        if len(points) < 3:
            self._fit = (min(d for d, _ in points), 0.0, ref)
            self._spread, self._kept = 0.0, len(points)
            return
        kept = points
        offset, slope = _line(kept, ref)
        for _ in range(2):
            lower = [(d, e) for d, e in kept if d - (offset + slope * (e - ref)) <= ENVELOPE_TOL_S]
            if len(lower) < 3 or len(lower) == len(kept):
                break
            kept = lower
            offset, slope = _line(kept, ref)
        self._fit = (offset, slope, ref)
        self._kept = len(kept)
        self._spread = max(d - (offset + slope * (e - ref)) for d, e in kept)


def _line(points: list[tuple[float, float]], ref: float) -> tuple[float, float]:
    """Least squares ``d = offset + slope * (e - ref)`` through ``(d, e)`` points."""
    n = len(points)
    xs = [e - ref for _, e in points]
    ys = [d for d, _ in points]
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True)) / sxx if sxx else 0.0
    return my - slope * mx, slope


def monotonic_raw() -> float:
    """CLOCK_MONOTONIC_RAW (never slewed by chrony) where the system has it, else monotonic."""
    raw = getattr(time, "CLOCK_MONOTONIC_RAW", None)
    return time.clock_gettime(raw) if raw is not None else time.monotonic()


class ChipRate:
    """The IMU's output rate in its stamps' own time (the chip runs on its own oscillator: 198-202
    at a nominal 200 is normal), over about a second of samples."""

    def __init__(self) -> None:
        """No samples yet."""
        self._start: int | None = None
        self._n = 0
        self.hz: float | None = None

    def add(self, esp_us: list[int]) -> None:
        """Consecutive samples' unwrapped stamps."""
        for t in esp_us:
            if self._start is None:
                self._start, self._n = t, 0
                continue
            self._n += 1
            if t - self._start >= 1_000_000:
                self.hz = self._n * 1e6 / (t - self._start)
                self._start, self._n = t, 0

    def reset(self) -> None:
        """The stamps started over."""
        self._start, self._n, self.hz = None, 0, None


# -- the face ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class FaceRequest:
    """One source's wish: an expression, how strongly, how fast, until when (None: standing)."""

    source: str
    name: str
    intensity: float
    transition_ms: float
    until: float | None
    seq: int


class FaceArbiter:
    """What the face shows: each source holds one standing and one timed request, the newest
    request of all wins, and a lapsed timed one falls back to the newest still standing."""

    def __init__(self) -> None:
        """Nobody asked for anything: the face is neutral."""
        self._held: dict[tuple[str, bool], FaceRequest] = {}
        self._seq = 0

    def set(
        self,
        source: str,
        name: str,
        *,
        intensity: float,
        transition_ms: float,
        hold_s: float | None,
        now: float,
    ) -> None:
        """``source`` asks for ``name``: standing (``hold_s`` None) or for ``hold_s`` seconds."""
        self._seq += 1
        until = None if hold_s is None else now + hold_s
        self._held[(source, until is not None)] = FaceRequest(
            source, name, intensity, transition_ms, until, self._seq
        )

    def clear(self, source: str) -> None:
        """``source`` asks for nothing any more."""
        self._held.pop((source, False), None)
        self._held.pop((source, True), None)

    def showing(self, now: float) -> FaceRequest | None:
        """The winning request (None: neutral); lapsed ones are dropped here."""
        for key in [k for k, r in self._held.items() if r.until is not None and r.until <= now]:
            del self._held[key]
        return max(self._held.values(), key=lambda r: r.seq, default=None)

    def sources(self) -> list[str]:
        """Who holds a request."""
        return sorted({source for source, _ in self._held})


@dataclass(frozen=True)
class StyleRequest:
    """One source's drawing style for the mouth (config/face.json ``styles``), until when
    (None: standing)."""

    source: str
    name: str
    until: float | None
    seq: int


class StyleArbiter:
    """The mouth's drawing style: each source holds one standing and one timed request, the
    newest of all wins, and a lapsed timed one falls back to the newest still standing (none:
    the base style)."""

    def __init__(self) -> None:
        """Nobody asked for a style: the base one draws."""
        self._held: dict[tuple[str, bool], StyleRequest] = {}
        self._seq = 0

    def set(self, source: str, name: str, *, hold_s: float | None, now: float) -> None:
        """``source`` asks for style ``name``: standing (``hold_s`` None) or for ``hold_s``
        seconds."""
        self._seq += 1
        until = None if hold_s is None else now + hold_s
        self._held[(source, until is not None)] = StyleRequest(source, name, until, self._seq)

    def clear(self, source: str) -> None:
        """``source`` asks for no style any more."""
        self._held.pop((source, False), None)
        self._held.pop((source, True), None)

    def showing(self, now: float) -> StyleRequest | None:
        """The winning request (None: the base style); lapsed ones are dropped here."""
        for key in [k for k, r in self._held.items() if r.until is not None and r.until <= now]:
            del self._held[key]
        return max(self._held.values(), key=lambda r: r.seq, default=None)

    def sources(self) -> list[str]:
        """Who holds a request."""
        return sorted({source for source, _ in self._held})


@dataclass(frozen=True)
class EyesRequest:
    """One source's override of the eyes ``side`` names (0 both, 1 left, 2 right), until when
    (None: standing)."""

    source: str
    eyes: Eyes
    side: int
    until: float | None
    seq: int


@dataclass(frozen=True)
class EffectRequest:
    """One source's effect on the inner rings (:data:`pepin.face.EYE_EFFECTS`), until when
    (None: standing)."""

    source: str
    name: str
    until: float | None
    seq: int


class EyesArbiter:
    """What the eyes show over the expression's own look: each source holds one standing and
    one timed override, each field of each eye shows the newest live override that sets it
    (the gaze layer's under every other source's), and a lapsed timed one falls back to what
    stood before it. The effect beside it: each source one standing and one timed, the newest
    live one runs."""

    def __init__(self) -> None:
        """Nobody overrides anything: the expression's own eyes and effect."""
        self._held: dict[tuple[str, bool], EyesRequest] = {}
        self._effects: dict[tuple[str, bool], EffectRequest] = {}
        self._seq = 0

    def set(self, source: str, eyes: Eyes, *, side: int, hold_s: float | None, now: float) -> None:
        """``source`` overrides ``eyes``' fields on ``side``: standing (``hold_s`` None) or for
        ``hold_s`` seconds."""
        self._seq += 1
        until = None if hold_s is None else now + hold_s
        self._held[(source, until is not None)] = EyesRequest(source, eyes, side, until, self._seq)

    def set_effect(self, source: str, name: str, *, hold_s: float | None, now: float) -> None:
        """``source`` runs effect ``name``: standing (``hold_s`` None) or for ``hold_s``
        seconds."""
        self._seq += 1
        until = None if hold_s is None else now + hold_s
        self._effects[(source, until is not None)] = EffectRequest(source, name, until, self._seq)

    def end_effect(self, source: str) -> None:
        """``source`` runs no effect any more."""
        self._effects.pop((source, False), None)
        self._effects.pop((source, True), None)

    def clear(self, source: str) -> None:
        """``source`` overrides nothing any more, its effect included."""
        self._held.pop((source, False), None)
        self._held.pop((source, True), None)
        self.end_effect(source)

    def effect(self, now: float) -> EffectRequest | None:
        """The effect that runs (None: the look's own); lapsed ones are dropped here."""
        for key in [k for k, r in self._effects.items() if r.until is not None and r.until <= now]:
            del self._effects[key]
        return max(self._effects.values(), key=lambda r: r.seq, default=None)

    def showing(self, now: float) -> tuple[Eyes, Eyes]:
        """The left eye's override and the right one's; lapsed ones are dropped here."""
        for key in [k for k, r in self._held.items() if r.until is not None and r.until <= now]:
            del self._held[key]
        left = right = Eyes()
        for r in sorted(self._held.values(), key=lambda r: (r.source != GAZE_SOURCE, r.seq)):
            if r.side != 2:
                left = r.eyes.over(left)
            if r.side != 1:
                right = r.eyes.over(right)
        return left, right

    def sources(self) -> list[str]:
        """Who holds an override or an effect."""
        return sorted({source for source, _ in self._held} | {s for s, _ in self._effects})


class BrainLeases:
    """Who on the laptop says it is here, until when."""

    def __init__(self) -> None:
        """No lease yet: nobody can be missed."""
        self._until: dict[str, float] = {}

    def lease(self, name: str, seconds: float, now: float) -> None:
        """``name`` is here for ``seconds`` more."""
        self._until[name] = now + max(0.0, seconds)

    def alive(self, now: float) -> list[str]:
        """The leases standing."""
        return sorted(name for name, until in self._until.items() if until > now)

    def lost(self, now: float) -> bool:
        """A lease was taken once, and every one has lapsed."""
        return bool(self._until) and not self.alive(now)


# -- the service ------------------------------------------------------------------------------


@dataclass(frozen=True)
class HeadSettings:
    """config/head.json."""

    device: str = "/dev/pepin-head"
    baud: int = 921600
    port: int = HEAD_PORT
    imu: ImuConfig = field(default_factory=lambda: ImuConfig(id=1, rate_hz=200))
    brightness: int = 200
    ping_hz: float = 2.0
    bucket_s: float = 1.0
    window_s: float = 120.0
    brain_lease: bool = True
    resting: str = "neutral"

    @classmethod
    def from_json(cls, path: str | Path) -> HeadSettings:
        """The settings in ``path``; the head IMU's config takes id 1."""
        data = read_json(path)
        imu = data.get("imu", {})
        return cls(
            device=str(data["serial"]["device"]),
            baud=int(data["serial"]["baud"]),
            port=int(data.get("port", HEAD_PORT)),
            imu=ImuConfig(
                id=1,
                rate_hz=int(imu.get("rate_hz", 200)),
                dlpf=int(imu.get("dlpf", 3)),
                accel_fs=int(imu.get("accel_fs", 1)),
                gyro_fs=int(imu.get("gyro_fs", 1)),
            ),
            brightness=int(data.get("brightness", 200)),
            ping_hz=float(data.get("ping_hz", 2.0)),
            bucket_s=float(data.get("clock", {}).get("bucket_s", 1.0)),
            window_s=float(data.get("clock", {}).get("window_s", 120.0)),
            brain_lease=bool(data.get("brain_lease", True)),
            resting=str(data.get("resting_expression", "neutral")),
        )


class HeadService:
    """The serial port, the clients, the face's arbiter, the clock map and the leases, stepped
    on one thread (:meth:`step`); the sockets are :class:`JsonLinesServer`'s threads."""

    def __init__(
        self,
        open_link: Callable[[], ByteLink],
        server: Clients,
        table: FaceTable,
        settings: HeadSettings,
        *,
        clock: Callable[[], float] = time.monotonic,
        raw_clock: Callable[[], float] | None = None,
    ) -> None:
        """``open_link`` opens the port (``OSError`` while it is absent). ``clock`` is what the
        IMU lines are dated in (the board's monotonic, as base_server's ``t``); ``raw_clock`` what
        the ESP32's micros are fitted against: CLOCK_MONOTONIC_RAW by default, because chrony
        slews the monotonic clock (up to 2000 ppm) and a line through a slew is wrong at its
        newest end; a sample is carried from one to the other by reading both back to back."""
        self._open_link = open_link
        self._server = server
        self._table = table
        self._settings = settings
        self._clock = clock
        if raw_clock is None:
            raw_clock = monotonic_raw if clock is time.monotonic else clock
        self._raw_clock = raw_clock
        self._link: ByteLink | None = None
        self._link_error = "not opened yet"
        self._retry_at = 0.0
        self._decoder = FrameDecoder()
        self._unwrapper = MicrosUnwrapper()
        self.clock_map = ClockMap(settings.bucket_s, settings.window_s)
        self.arbiter = FaceArbiter()
        self.styles = StyleArbiter()
        self.eyes = EyesArbiter()
        self.leases = BrainLeases()
        self._moments: dict[tuple[str, str], float] = {}  # (source, event): when last shown
        self.moments_dropped = 0
        self._config = settings.imu
        self._configs: dict[int, ImuConfig] = {0: FIRMWARE_DEFAULTS, self._config.id: self._config}
        self._brightness = settings.brightness
        # the base style (config/face.json's, or a config's): what draws while no source asks
        # for one; the style that draws goes with every 'C'
        self._style = table.style
        self._sent_style: str | None = None  # the style the last 'C' carried
        # ...and the eyes' brightness (config/face.json's eyes.power.brightness)
        self._eyes_brightness = (
            EYES_BRIGHTNESS if table.eyes is None else round(table.eyes.power["brightness"])
        )
        self._config_eyes = False  # the last 'C' carried the eyes' brightness
        # ...and their look set (config/face.json's eyes.eye_set, or a config's)
        self._eye_set = EYE_SET_CHOICES[0] if table.eyes is None else table.eyes.eye_set
        self._config_sets = False  # the last 'C' carried the eye set
        self._sent_face: tuple[str, float] | None = None
        # the eyes' override and effect (its name and request) as last sent
        self._sent_eyes: tuple[Eyes, Eyes, tuple[str, int] | None] | None = None
        self._gaze: dict[str, Any] | None = None  # the newest gaze command, as it came
        self.eye_frames = 0
        self.blinks = 0
        self._subscribers: set[Subscriber] = set()
        self._rate = ChipRate()  # the IMU's rate in ESP32 time
        self._pings: dict[int, float] = {}
        self._ping_id = 0
        self._next_ping = 0.0
        self._next_report = clock() + REPORT_EVERY_S
        self._esp_status: HeadStatus | None = None
        self._esp_status_at: float | None = None
        self._mismatches = 0
        self._last_sample_us: int | None = None
        self._brain_was_lost = False
        self.imu_samples = 0
        self.imu_gaps = 0
        self.imu_lines = 0
        self.pongs = 0
        self.resyncs = 0
        self.connects = 0

    # -- the loop ---------------------------------------------------------------------------

    def step(self, timeout_s: float = 0.01) -> None:
        """One turn: read the port (up to ``timeout_s``), answer the clients, run the clocks."""
        now = self._clock()
        if self._link is None and now >= self._retry_at:
            self._connect(now)
        if self._link is not None:
            try:
                data = self._link.read(timeout_s)
            except OSError as error:
                self._drop(f"read failed: {error}")
                data = b""
            if data:
                received = self._raw_clock()
                for kind, payload in self._decoder.feed(data):
                    self._on_frame(kind, payload, received)
        else:
            time.sleep(timeout_s)
        now = self._clock()
        for client, message in self._server.commands():
            if client is None:
                continue  # the farewell pseudo-message: nothing to release
            try:
                reply = self.handle(client, message, now)
            except (KeyError, ValueError, TypeError) as error:
                reply = {"type": "error", "cmd": message.get("cmd"), "error": str(error)}
            self._server.reply(client, reply)
        self._tick(now)

    def run(self, stop: threading.Event) -> None:
        """Step until ``stop``; the port and the sockets are released on the way out."""
        try:
            while not stop.is_set():
                self.step()
        finally:
            if self._link is not None:
                self._link.close()
            self._server.close()

    # -- the port -----------------------------------------------------------------------------

    def _connect(self, now: float) -> None:
        try:
            self._link = self._open_link()
        except (OSError, ValueError) as error:
            if str(error) != self._link_error:
                logger.warning("head port %s: %s (retrying every %.0f s)", self._settings.device,
                               error, RETRY_S)  # fmt: skip
            self._link_error = str(error)
            self._retry_at = now + RETRY_S
            return
        self.connects += 1
        self._link_error = ""
        logger.info("head port %s open at %d baud", self._settings.device, self._settings.baud)
        self._decoder = FrameDecoder()
        self._resync("port opened")

    def _drop(self, why: str) -> None:
        logger.error("head port: %s; reopening", why)
        if self._link is not None:
            self._link.close()
        self._link = None
        self._link_error = why
        self._retry_at = self._clock() + RETRY_S

    def _send(self, kind: int, payload: bytes) -> bool:
        if self._link is None:
            return False
        try:
            self._link.write(encode_frame(kind, payload))
        except OSError as error:
            self._drop(f"write failed: {error}")
            return False
        return True

    def _resync(self, why: str) -> None:
        """A fresh ESP32 (a new port, a reboot): its micros, its config and its face anew."""
        self._unwrapper = MicrosUnwrapper()
        self.clock_map.reset()
        self._pings.clear()
        self._last_sample_us = None
        self._rate.reset()
        self._mismatches = 0
        self._sent_face = None
        self._sent_eyes = None
        self._sent_style = None
        self.resyncs += 1
        logger.info("head: sending config %s, style %s, the face and the eyes (%s)",
                    self._config.id, self._style_shown(self._clock())[0], why)  # fmt: skip
        self._send_config()
        self._next_ping = 0.0

    def _send_config(self) -> bool:
        """The 'C' frame: the IMU's config, the screen's brightness, the style that draws and,
        to a firmware that has eyes (or one not heard from yet), the eyes' brightness: firmware
        3 drops a 'C' of nine bytes whole, and its status then asks for the config again. The
        eye set rides behind it to a firmware whose status carries the set in force (or one not
        heard from yet): a firmware 4 built before the sets drops a 'C' of ten bytes whole, and
        its first status gets one of nine (:meth:`_on_status`)."""
        esp = self._esp_status
        eyes = esp is None or esp.version >= EYES_FIRMWARE
        sets = eyes and (esp is None or esp.eye_set is not None)
        style = self._style_shown(self._clock())[0]
        payload = encode_config(self._config, self._brightness, self._table.style_id_of(style),
                                self._eyes_brightness if eyes else None,
                                eye_set_id(self._eye_set) if sets else None)  # fmt: skip
        sent = self._send(CONFIG, payload)
        self._config_eyes = sent and eyes
        self._config_sets = sent and sets
        if sent:
            self._sent_style = style
        return sent

    # -- from the ESP32 -----------------------------------------------------------------------

    def _on_frame(self, kind: int, payload: bytes, received: float) -> None:
        try:
            if kind == IMU:
                self._on_imu(payload, received)
            elif kind == PONG:
                self._on_pong(payload, received)
            elif kind == STATUS:
                self._on_status(decode_status(payload), received)
        except (ValueError, struct.error) as error:
            logger.warning("head: a malformed %r frame (%d bytes): %s", chr(kind), len(payload),
                           error)  # fmt: skip

    def _esp_time(self, raw: int) -> int:
        value, reset = self._unwrapper.unwrap(raw)
        if reset:
            logger.warning("head: the ESP32's micros went back: it rebooted")
            self._resync("the ESP32 rebooted")
            value, _ = self._unwrapper.unwrap(raw)
        return value

    def _on_imu(self, payload: bytes, received: float) -> None:
        # The hot path (200 frames a second): raw tuples, not ImuSample objects.
        if len(payload) % IMU_SAMPLE.size:
            raise ValueError(f"an IMU payload of {len(payload)} bytes is not n x 17")
        rows = list(IMU_SAMPLE.iter_unpack(payload))
        if not rows:
            return
        times = [self._esp_time(row[0]) for row in rows]
        # The newest sample was stamped at its data-ready edge, then read over I2C, then its
        # frame went out byte by byte: both known, both taken off.
        known = I2C_READ_S + self._wire_s(len(payload) + OVERHEAD)
        self.clock_map.observe(times[-1], received, known)
        config = self._configs.get(rows[-1][7], self._config)
        limit = 1.5e6 / config.rate_hz  # a gap: stamps more than 1.5 periods apart
        previous = self._last_sample_us
        for t in times:
            if previous is not None and t - previous > limit:
                self.imu_gaps += 1
            previous = t
        self._last_sample_us = times[-1]
        self.imu_samples += len(rows)
        self._rate.add(times)
        if not self._subscribers:
            return
        self._subscribers = {c for c in self._subscribers if c.alive}
        # The data-ready moment on the board's monotonic clock: ESP seconds * scale + shift on
        # the raw clock, carried to the monotonic one by reading both back to back. The filter's
        # delay is NOT taken off here: the consumer does, from the imu_config line.
        scale, shift = self.clock_map.linear()
        shift += self._clock() - self._raw_clock()
        gs, acs = config.gyro_scale, config.acc_scale
        samples = ",".join(
            f"[{t * 1e-6 * scale + shift:.6f},{t},{r[4] * gs:.6g},{r[5] * gs:.6g},"
            f"{r[6] * gs:.6g},{r[1] * acs:.6g},{r[2] * acs:.6g},{r[3] * acs:.6g}]"
            for t, r in zip(times, rows, strict=True)
        )
        encoded = f'{{"type":"imu","cfg":{config.id},"samples":[{samples}]}}\n'.encode()
        for client in self._subscribers:
            client.post(encoded)
        self.imu_lines += 1

    def imu_config_line(self, config: ImuConfig | None = None) -> dict[str, Any]:
        """What the IMU lines' samples were taken at: sent to a subscriber when it subscribes
        and to every subscriber when the config changes."""
        config = config or self._config
        return {
            "type": "imu_config",
            "cfg": config.id,
            "rate_hz": config.rate_hz,
            "dlpf": config.dlpf,
            "gyro_fs_dps": 250 << config.gyro_fs,
            "accel_fs_g": 2 << config.accel_fs,
            "filter_delay_s": config.delay_s,
        }

    def _on_pong(self, payload: bytes, received: float) -> None:
        ping_id, raw = decode_pong(payload)
        sent = self._pings.pop(ping_id, None)
        esp = self._esp_time(raw)
        ping_wire = self._wire_s(PING_BYTES + OVERHEAD)
        pong_wire = self._wire_s(len(payload) + OVERHEAD)
        if sent is not None:
            self.clock_map.observe_rtt(received - sent - ping_wire - pong_wire)
        self.clock_map.observe(esp, received, pong_wire)
        self.pongs += 1

    def _wire_s(self, frame_bytes: int) -> float:
        """How long a frame of that many bytes takes on the wire (8N1: 10 bits a byte)."""
        return frame_bytes * 10.0 / self._settings.baud

    def _on_status(self, status: HeadStatus, received: float) -> None:
        self._esp_status, self._esp_status_at = status, received
        showing = self.arbiter.showing(received)
        wanted = self._table.id_of(showing.name if showing else self._settings.resting)
        # A config or a face the ESP32 does not show: a reboot the micros did not reveal (one
        # within a second of the last), or a frame lost while it booted. Asleep or on an info
        # screen it shows something else on purpose.
        if status.config_id != self._config.id:
            self._mismatches += 1
            if self._mismatches >= CONFIG_MISMATCHES:
                self._resync(f"the ESP32 runs config {status.config_id}")
        elif status.mode == "face" and status.expression != wanted:
            self._mismatches += 1
            if self._mismatches >= CONFIG_MISMATCHES:
                self._mismatches = 0
                self._sent_face = None  # sent again at the next tick
        else:
            self._mismatches = 0
        # a firmware with eyes that was sent the config without them, one with the eye sets
        # sent it without the set, or one without them sent a 'C' it dropped
        stale = not self._config_eyes or self._config_sets != (status.eye_set is not None)
        if status.version >= EYES_FIRMWARE and stale and self._link is not None:
            self._send_config()
        self._server.broadcast(self.status(received))

    # -- from the clients ---------------------------------------------------------------------

    def handle(self, client: Subscriber, message: dict[str, Any], now: float) -> dict[str, Any]:
        """One client command; its reply. ``KeyError``/``ValueError`` become an error reply."""
        cmd = message.get("cmd")
        source = str(message.get("source") or getattr(client, "peer", "client"))
        default_ms = self._table.timing["transition_ms"]
        if cmd == "express":
            name = message.get("name")
            if name is None:
                self.arbiter.clear(source)
                return self._ack(cmd, now)
            self._table.id_of(str(name))  # an unknown name is refused with the known ones
            hold = message.get("hold_s")
            self.arbiter.set(
                source,
                str(name),
                intensity=_unit(message.get("intensity", 1.0)),
                transition_ms=float(message.get("transition_ms", default_ms)),
                hold_s=None if hold is None else float(hold),
                now=now,
            )
            return self._ack(cmd, now)
        if cmd == "event":
            name = str(message.get("name"))
            event = self._table.event(name)
            last = self._moments.get((source, name))
            if event.min_gap_s > 0.0 and last is not None and now - last < event.min_gap_s:
                self.moments_dropped += 1  # a moment repeated too soon: the face does not flicker
                reply = self._ack(cmd, now)
                reply["dropped"] = f"{name} again within its min_gap_s {event.min_gap_s:g}"
                return reply
            self._moments[(source, name)] = now
            if message.get("end"):
                self.arbiter.clear(source)
            self.arbiter.set(
                source,
                event.expression,
                intensity=event.intensity,
                transition_ms=default_ms,
                hold_s=event.hold_s,
                now=now,
            )
            return self._ack(cmd, now)
        if cmd == "clear":
            self.arbiter.clear(source)
            return self._ack(cmd, now)
        if cmd == "style":
            name = message.get("name")
            if name is None:
                self.styles.clear(source)
                return self._style_ack(cmd, now)
            known = [s.name for s in self._table.styles]
            if name not in known:
                raise ValueError(f"no face style {name!r}: {', '.join(known)}")
            ms = message.get("ms")
            hold = None if ms is None else max(float(ms), 0.0) / 1000.0
            self.styles.set(source, str(name), hold_s=hold, now=now)
            return self._style_ack(cmd, now)
        if cmd == "style_release":
            self.styles.clear(source)
            return self._style_ack(cmd, now)
        if cmd == "eyes":
            fields = {k: v for k, v in message.items() if k not in EYE_COMMAND_KEYS}
            eyes = Eyes.of(fields)
            if eyes.empty:
                raise ValueError("an eyes override sets a field at least (eyes_release lets go)")
            ms = message.get("ms")
            hold = None if ms is None else max(float(ms), 0.0) / 1000.0
            self.eyes.set(source, eyes, side=_side(message.get("side")), hold_s=hold, now=now)
            return self._eyes_ack(cmd, now)
        if cmd == "eyes_effect":
            name = message.get("name")
            if name is None or name == NO_EFFECT:
                self.eyes.end_effect(source)
                return self._eyes_ack(cmd, now)
            if name not in EYE_EFFECTS:
                raise ValueError(f"no eye effect {name!r}: {', '.join(EYE_EFFECTS[1:])} or null")
            ms = message.get("ms")
            hold = None if ms is None else max(float(ms), 0.0) / 1000.0
            if hold is None and name in ONE_SHOTS:  # it ends by itself: so does the request
                hold = self._one_shot_s(str(name))
            self.eyes.set_effect(source, str(name), hold_s=hold, now=now)
            return self._eyes_ack(cmd, now)
        if cmd == "eyes_release":
            self.eyes.clear(source)
            if source == GAZE_SOURCE:
                self._gaze = None
            return self._eyes_ack(cmd, now)
        if cmd in ("blink", "wink"):
            side = _side(message.get("side"))
            if cmd == "wink" and side == 0:
                raise ValueError("a wink is the left eye's or the right one's: side left or right")
            blink = side + 1  # BLINKS: both, a wink of the left, of the right
            sent = self._tick_eyes(now, blink)
            self.blinks += 1
            return {"type": "ack", "cmd": cmd, "blink": BLINKS[blink], "sent": sent}
        if cmd == "gaze":
            x, y = _signed_unit(message.get("x", 0.0)), _signed_unit(message.get("y", 0.0))
            tracking = bool(message.get("tracking", False))
            self._gaze = {"x": x, "y": y, "tracking": tracking}
            # Following a person, the head points at them: the eyes look straight out, at theirs.
            eyes = Eyes.of({"gaze_x": 0.0 if tracking else x, "gaze_y": 0.0 if tracking else y})
            self.eyes.set(GAZE_SOURCE, eyes, side=0, hold_s=None, now=now)
            return self._eyes_ack(cmd, now)
        if cmd == "mouth":
            self._send(MOUTH, encode_mouth(_unit(message.get("level", 0.0))))
            return {"type": "ack", "cmd": cmd}
        if cmd == "show":
            items = message.get("items")
            if items is None:
                items = show_items(str(message.get("text", "")))
            if not isinstance(items, list):
                raise ValueError("items is a list of {text}, {key, value} or {bar, frac, value}")
            seconds = float(message.get("seconds", 8.0))
            sent = self._send(INFO, encode_info(items, seconds))
            return {"type": "ack", "cmd": cmd, "items": len(items), "sent": sent}
        if cmd == "lease":
            self.leases.lease(str(message.get("name") or source), float(message["seconds"]), now)
            return {"type": "ack", "cmd": cmd, "leases": self.leases.alive(now)}
        if cmd == "config":
            return self._configure(message)
        if cmd == "subscribe":
            if message.get("imu", True):
                self._subscribers.add(client)
                client.post(encode_line(self.imu_config_line()))  # before the first sample
            else:
                self._subscribers.discard(client)
            return {"type": "ack", "cmd": cmd, "imu": client in self._subscribers}
        if cmd == "status":
            return self.status(now)
        raise ValueError(f"unknown command {cmd!r}")

    def _ack(self, cmd: str, now: float) -> dict[str, Any]:
        self._tick_face(now)
        showing = self.arbiter.showing(now)
        return {
            "type": "ack",
            "cmd": cmd,
            "showing": showing.name if showing else self._settings.resting,
            "by": showing.source if showing else None,
        }

    def _style_ack(self, cmd: str, now: float) -> dict[str, Any]:
        sent = self._tick_style(now)
        style, by = self._style_shown(now)
        return {"type": "ack", "cmd": cmd, "style": style, "by": by, "sent": sent}

    def _style_shown(self, now: float) -> tuple[str, str | None]:
        """The style that draws and the source whose request it is (None: the base)."""
        showing = self.styles.showing(now)
        return (self._style, None) if showing is None else (showing.name, showing.source)

    def _eyes_ack(self, cmd: str, now: float) -> dict[str, Any]:
        self._tick_eyes(now)
        return {"type": "ack", "cmd": cmd, **self._eyes_shown(now)}

    def _eyes_shown(self, now: float) -> dict[str, Any]:
        """The eyes' override as it stands, both eyes in one when they are alike, and the
        effect when one runs."""
        left, right = self.eyes.showing(now)
        effect = self.eyes.effect(now)
        running = {} if effect is None else {"effect": effect.name}
        if left == right:
            return {"eyes": left.as_dict(), **running, "by": self.eyes.sources()}
        return {"left": left.as_dict(), "right": right.as_dict(), **running,
                "by": self.eyes.sources()}  # fmt: skip

    def _one_shot_s(self, name: str) -> float:
        """How long one-shot effect ``name`` plays on the firmware (config/face.json
        ``eyes.effects``: the sparkle's twinkles and their fade, the flash)."""
        fx = self._table.eyes.effects if self._table.eyes is not None else {}
        if name == "sparkle":
            return (fx.get("sparkle_ms", 1200.0) + 2.0 * fx.get("sparkle_rise_ms", 140.0)) / 1000.0
        return fx.get("flash_ms", 260.0) / 1000.0

    def _configure(self, message: dict[str, Any]) -> dict[str, Any]:
        """A ``config``: a new IMU config id only when the sampling (rate, DLPF, ranges)
        changes, so a style, an eye set or a brightness alone leaves the head IMU running (the
        firmware sets the chip up again on any other id); nothing is applied when a value is
        refused."""
        sampling = ImuConfig(
            id=self._config.id,
            rate_hz=int(message.get("imu_rate_hz", self._config.rate_hz)),
            dlpf=int(message.get("dlpf", self._config.dlpf)),
            accel_fs=int(message.get("accel_fs", self._config.accel_fs)),
            gyro_fs=int(message.get("gyro_fs", self._config.gyro_fs)),
        )
        brightness = int(message.get("brightness", self._brightness))
        eyes = int(message.get("eyes_brightness", self._eyes_brightness))
        style = str(message.get("style", self._style))
        self._table.style_id_of(style)  # KeyError naming the known styles
        eye_set = str(message.get("eye_set", self._eye_set))
        eye_set_id(eye_set)  # ValueError naming the choices
        self._brightness = min(max(brightness, 0), 255)
        self._style = style
        self._eye_set = eye_set
        self._eyes_brightness = min(max(eyes, 0), 255)
        changed = sampling != self._config
        if changed:
            self._config = replace(sampling, id=self._config.id % 255 + 1)
            self._configs[self._config.id] = self._config
            self._mismatches = 0
        config = self._config
        sent = self._send_config()
        if changed:  # the subscribers' samples are taken at it from now on
            line = encode_line(self.imu_config_line(config))
            for subscriber in self._subscribers:
                subscriber.post(line)
        style, by = self._style_shown(self._clock())
        return {"type": "ack", "cmd": "config", "config": asdict(config),
                "brightness": self._brightness, "style": style, "style_by": by,
                "eyes_brightness": self._eyes_brightness, "eye_set": self._eye_set,
                "sent": sent}  # fmt: skip

    # -- the clocks ---------------------------------------------------------------------------

    def _tick(self, now: float) -> None:
        if self._link is not None and now >= self._next_ping:
            self._ping_id = (self._ping_id + 1) & 0xFFFFFFFF
            sent = self._raw_clock()
            self._pings = {k: v for k, v in self._pings.items() if sent - v < 2.0}
            self._pings[self._ping_id] = sent
            self._send(PING, encode_ping(self._ping_id))
            period = 1.0 / self._settings.ping_hz
            self._next_ping = now + period if self._next_ping == 0.0 else self._next_ping + period
            if self._next_ping < now:
                self._next_ping = now + period
        if self._settings.brain_lease:
            lost = self.leases.lost(now)
            if lost and not self._brain_was_lost:
                logger.warning("head: every brain lease lapsed: the face falls asleep")
                event = self._table.event("brain_lost")
                self.arbiter.set("brain", event.expression, intensity=event.intensity,
                                 transition_ms=900.0, hold_s=None, now=now)  # fmt: skip
            elif not lost and self._brain_was_lost:
                logger.info("head: a brain is back: %s", ", ".join(self.leases.alive(now)))
                self.arbiter.clear("brain")
            self._brain_was_lost = lost
        self._tick_face(now)
        self._tick_style(now)
        self._tick_eyes(now)
        if now >= self._next_report:
            self._next_report = now + REPORT_EVERY_S
            logger.info("%s", self.report(now))

    def _tick_face(self, now: float) -> None:
        showing = self.arbiter.showing(now)
        name = showing.name if showing else self._settings.resting
        intensity = showing.intensity if showing else 1.0
        if (name, intensity) == self._sent_face or self._link is None:
            return
        ms = showing.transition_ms if showing else self._table.timing["transition_ms"]
        if self._send(EXPRESSION, encode_expression(self._table.id_of(name), intensity, ms)):
            self._sent_face = (name, intensity)

    def _tick_style(self, now: float) -> bool:
        """The style that draws down the wire in a 'C' when it changed; whether one went."""
        if self._link is None or self._style_shown(now)[0] == self._sent_style:
            return False
        return self._send_config()

    def _tick_eyes(self, now: float, blink: int = 0) -> bool:
        """The eyes' override down the wire when it changed (or with a ``blink``): one 'Y' for
        both eyes when they are alike, one each otherwise, every frame the whole override of its
        eyes (an eye with nothing overridden: a frame of sentinels for its side; a release, which
        the firmware applies to both eyes, only for both); the effect in the first frame's
        19-byte form when it changed (its rest of life, or none once it ended), never ended by
        a release while it runs. Asleep (every brain lease lapsed) nothing is overridden: the
        sleepy look's own eyes and effect, no laptop's override held over them; the overrides
        come back with a brain."""
        left, right = self.eyes.showing(now)
        effect = self.eyes.effect(now)
        if self._settings.brain_lease and self.leases.lost(now):
            left = right = Eyes()
            effect = None
        running = None if effect is None else (effect.name, effect.seq)
        shown = (left, right, running)
        if self._link is None or (shown == self._sent_eyes and not blink):
            return False
        before = self._sent_eyes[2] if self._sent_eyes is not None else None
        call: tuple[str, int] | None = None  # the effect byte and its life, when it changed
        if running != before and effect is not None:
            life = 0 if effect.until is None else round((effect.until - now) * 1000.0)
            call = (effect.name, life)
        elif running != before and before is not None:
            call = (NO_EFFECT, 0)
        frames = [(0, left)] if left == right else [(1, left), (2, right)]
        sent = True
        for i, (side, eyes) in enumerate(frames):
            release = side == 0 and eyes.empty and effect is None
            mode = EYE_RELEASE if release else EYE_STANDING
            if i == 0 and call is not None and not release:
                eyes = replace(eyes, effect=call[0], effect_ms=call[1])
            frame = EyesFrame(eyes, mode, 0, blink if i == 0 else 0, side)
            sent = self._send(EYES, encode_eyes(frame)) and sent
        if sent:
            self._sent_eyes = shown
            self.eye_frames += len(frames)
        return sent

    # -- what it says ---------------------------------------------------------------------------

    def status(self, now: float) -> dict[str, Any]:
        """The server's and the ESP32's numbers, for the status line."""
        showing = self.arbiter.showing(now)
        style, style_by = self._style_shown(now)
        esp = self._esp_status
        return {
            "type": "status",
            "t": round(now, 6),
            "link": "up" if self._link is not None else f"down: {self._link_error}",
            "device": self._settings.device,
            "baud": self._settings.baud,
            "esp": None if esp is None else esp.as_dict(),
            "esp_age_s": None
            if self._esp_status_at is None
            else round(now - self._esp_status_at, 3),
            "frames": self._decoder.frames,
            "crc_errors": self._decoder.crc_errors,
            "length_errors": self._decoder.length_errors,
            "skipped_bytes": self._decoder.skipped_bytes,
            "imu_samples": self.imu_samples,
            "imu_gaps": self.imu_gaps,
            "imu_lines": self.imu_lines,
            "chip_rate_hz": None if self._rate.hz is None else round(self._rate.hz, 2),
            "pongs": self.pongs,
            "clock": self.clock_map.state(),
            "config": asdict(self._config),
            "brightness": self._brightness,
            "style": style,
            "style_by": style_by,
            "style_sources": self.styles.sources(),
            "showing": showing.name if showing else self._settings.resting,
            "by": showing.source if showing else None,
            "sources": self.arbiter.sources(),
            "eyes": {
                **self._eyes_shown(now),
                "brightness": self._eyes_brightness,
                "set": self._eye_set,  # the setting the 'C' carries
                "set_shown": None if esp is None else esp.eye_set,  # the status's
                "gaze": self._gaze,
                "frames": self.eye_frames,
                "blinks": self.blinks,
            },
            "moments_dropped": self.moments_dropped,
            "leases": self.leases.alive(now),
            "brain_lost": self.leases.lost(now),
            "subscribers": len(self._subscribers),
            "clients": self._server.client_count,
            "resyncs": self.resyncs,
            "connects": self.connects,
        }

    def report(self, now: float) -> str:
        """The once-a-minute journal line."""
        s = self.status(now)
        esp = s["esp"]
        chip = (
            "no status from the ESP32"
            if esp is None
            else f"firmware {esp['version']}, face {esp['fps']:.1f} fps ({esp['mode']}), imu"
            f" {esp['imu_rate_hz']} Hz {esp['imu_state']} (who 0x{esp['who_am_i']:02x}), i2c"
            f" errors {esp['i2c_errors']}, dropped {esp['dropped']}, rx errors {esp['rx_errors']}"
        )
        if esp is not None and esp["eyes_ma"] is not None:
            shown = "" if esp["eye_set"] is None else f", set {esp['eye_set']}"
            chip += (
                f", eyes {esp['eyes_ma']} mA at brightness {esp['eyes_brightness']}"
                f" (mode {esp['eyes_mode']}, layout {esp['eye_layout']}{shown})"
            )
        eyes = s["eyes"]
        overridden = ", ".join(eyes["by"]) or "nobody"
        clock = s["clock"]
        timing = (
            f"clock {clock['offset_s']:+.6f} s, {clock['esp_fast_ppm']:+.1f} ppm,"
            f" rtt {clock['min_rtt_ms']} ms, spread {clock['spread_ms']} ms"
            if clock.get("ready")
            else "clock not ready"
        )
        return (
            f"head: link {s['link']}, {chip}; crc errors {s['crc_errors']}, samples"
            f" {s['imu_samples']} (gaps {s['imu_gaps']}), {timing}; showing {s['showing']}"
            f" ({s['by'] or 'nobody'}), style {s['style']} ({s['style_by'] or 'the base'}),"
            f" eyes overridden by {overridden} (brightness {eyes['brightness']}, set"
            f" {eyes['set']}, {eyes['frames']} frames, {eyes['blinks']} blinks), leases"
            f" {s['leases'] or 'none'}, subscribers {s['subscribers']}, clients {s['clients']}"
        )


def _unit(value: Any) -> float:
    """A number clamped to 0..1."""
    return min(max(float(value), 0.0), 1.0)


def _signed_unit(value: Any) -> float:
    """A number clamped to -1..1."""
    return min(max(float(value), -1.0), 1.0)


def _side(value: Any) -> int:
    """An eye command's side ("both", "left", "right", or 0..2; absent: both) as the 'Y' frame's
    byte; ``ValueError`` otherwise."""
    if value is None:
        return 0
    if isinstance(value, str):
        if value not in SIDES:
            raise ValueError(f"side {value!r}: {', '.join(SIDES)}")
        return SIDES.index(value)
    if isinstance(value, bool) or int(value) not in range(len(SIDES)):
        raise ValueError(f"side {value!r}: {', '.join(SIDES)}")
    return int(value)


def stop_on_sigterm() -> threading.Event:
    """An event SIGTERM sets."""
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    return stop


def main() -> None:
    parser = argparse.ArgumentParser(description="Own the head ESP32: face, head IMU, clocks.")
    parser.add_argument("--config", help="config/head.json (default: the deployed one)")
    parser.add_argument("--device", help="the serial port, over the config's")
    parser.add_argument("--port", type=int, help="the TCP port, over the config's")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s"
    )
    from pepin.deployment import config_file

    settings = HeadSettings.from_json(args.config or config_file("head.json"))
    if args.device:
        settings = replace(settings, device=args.device)
    if args.port:
        settings = replace(settings, port=args.port)
    table = load_face_table()
    server = JsonLinesServer(settings.port, outbox_size=OUTBOX_LINES).start()
    service = HeadService(
        lambda: SerialPort(settings.device, settings.baud), server, table, settings
    )
    logger.info("head server on %d; port %s at %d baud; imu %s", settings.port, settings.device,
                settings.baud, asdict(settings.imu))  # fmt: skip
    service.run(stop_on_sigterm())


if __name__ == "__main__":
    main()
