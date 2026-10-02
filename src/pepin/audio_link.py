"""The audio link: the robot's microphone array and speaker over one TCP socket.

The board (:mod:`pepin.audio_server`) owns the reSpeaker XVF3800 array: it captures the
array's processed voice, reads the direction of arrival, and plays what the laptop sends through
the array's own output (the speaker hangs on its jack, so the echo canceller hears what it must
cancel). The laptop's end is :class:`AudioClient`.

The wire is JSON lines, and a line whose object carries ``"bytes": N`` is followed by exactly N
raw bytes. A status probe with ``nc`` therefore reads plain JSON, and the audio costs one short
header per 20 ms frame::

    board -> laptop  {"type":"hello","rate":16000,"format":"s16le","channels":1,
                      "frame_samples":320,"play_rate":16000,...}          on connect
                     {"type":"pcm","seq":N,"t":<board s>,"bytes":640} + 640 bytes    listeners
                     {"type":"doa","t":<board s>,"deg":0..359,"speech":true}         listeners
                     {"type":"status",...}                                  to the asker
                     {"type":"error","error":"..."}                         to the asker
    laptop -> board  {"cmd":"listen"}                     start receiving pcm and doa
                     {"cmd":"status"}                     one status line back
                     {"cmd":"play","bytes":N} + N bytes    PCM s16le mono at hello's play_rate
                     {"cmd":"play_end"}                   the utterance is complete
                     {"cmd":"flush"}                      silence the speaker now (barge-in)

``t`` is the board's monotonic clock at the frame's first sample (at the moment of a DOA
read), so a frame and a direction compare directly. ``seq`` counts frames since the server
started; a gap in it is audio that never reached this client. An unreadable header line ends
the connection: once framing is lost, the bytes after it cannot be trusted.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import socket
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any, Protocol, Self

logger = logging.getLogger(__name__)

AUDIO_PORT = 3338
RATE = 16_000  # Hz: the XVF3800's processing rate and its USB firmware's default interface rate
SAMPLE_BYTES = 2  # s16le
FRAME_SAMPLES = 320  # 20 ms at 16 kHz
MAX_LINE = 4096  # a header longer than this is garbage, not a header
MAX_PAYLOAD = 1 << 20  # a larger "bytes" claim is a broken peer

BOARD_HOST = "10.0.0.187"  # the board's address, as ros/*.sh default it

Connector = Callable[[tuple[str, int]], socket.socket]


def board_host() -> str:
    """Where the board is: ``PEPIN_HOST`` when set, else the address the ros/ scripts use."""
    return os.environ.get("PEPIN_HOST", BOARD_HOST)


class ProtocolError(ValueError):
    """The byte stream no longer parses as the audio link's framing."""


def encode(message: dict[str, Any], payload: bytes = b"") -> bytes:
    """One message ready for the socket: its JSON line, ``bytes`` added when a payload follows."""
    if payload:
        message = {**message, "bytes": len(payload)}
    return json.dumps(message, separators=(",", ":")).encode() + b"\n" + payload


class FrameDecoder:
    """Turns received chunks, cut anywhere, back into (header, payload) messages."""

    def __init__(self) -> None:
        self._buffer = bytearray()
        self._pending: dict[str, Any] | None = None  # a header whose payload is still arriving
        self._need = 0

    def feed(self, chunk: bytes) -> list[tuple[dict[str, Any], bytes]]:
        """Every message completed by ``chunk``; raises :class:`ProtocolError` on garbage."""
        self._buffer += chunk
        out: list[tuple[dict[str, Any], bytes]] = []
        while True:
            if self._pending is not None:
                if len(self._buffer) < self._need:
                    return out
                payload = bytes(self._buffer[: self._need])
                del self._buffer[: self._need]
                out.append((self._pending, payload))
                self._pending = None
                continue
            end = self._buffer.find(b"\n")
            if end < 0:
                if len(self._buffer) > MAX_LINE:
                    raise ProtocolError(f"no header line in {len(self._buffer)} bytes")
                return out
            line = bytes(self._buffer[:end])
            del self._buffer[: end + 1]
            if not line.strip():
                continue
            header = _header(line)
            size = header.get("bytes", 0)
            if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= MAX_PAYLOAD:
                raise ProtocolError(f"bad payload size {size!r}")
            if size:
                self._pending, self._need = header, size
            else:
                out.append((header, b""))


def _header(line: bytes) -> dict[str, Any]:
    """One header line as a JSON object; raises :class:`ProtocolError` for anything else."""
    try:
        header = json.loads(line)
    except ValueError as exc:
        raise ProtocolError(f"unreadable header {line[:60]!r}") from exc
    if not isinstance(header, dict):
        raise ProtocolError(f"header is not an object: {line[:60]!r}")
    return header


@dataclass(frozen=True)
class AudioFrame:
    """20 ms of the robot's hearing: s16le mono PCM and when its first sample was captured."""

    seq: int  # frames since the server started
    stamp_s: float  # board monotonic clock at the first sample
    pcm: bytes
    rate: int = RATE

    @property
    def samples(self) -> int:
        """Samples in this frame."""
        return len(self.pcm) // SAMPLE_BYTES

    @property
    def duration_s(self) -> float:
        """How long the frame lasts."""
        return self.samples / self.rate

    def encode(self) -> bytes:
        """The frame on the wire."""
        return encode({"type": "pcm", "seq": self.seq, "t": round(self.stamp_s, 6)}, self.pcm)


@dataclass(frozen=True)
class DoaReading:
    """Where the array hears the voice: degrees in the array's own frame (Seeed's diagram:
    0 toward the USB-C connector), whether it calls it speech, and when it was read."""

    stamp_s: float  # board monotonic clock at the read
    deg: int  # 0..359
    speech: bool

    def encode(self) -> bytes:
        """The reading on the wire."""
        return encode(
            {"type": "doa", "t": round(self.stamp_s, 6), "deg": self.deg, "speech": self.speech}
        )


def parse_frame(header: dict[str, Any], payload: bytes, rate: int = RATE) -> AudioFrame:
    """A ``pcm`` message back into a frame; raises ValueError when it is malformed."""
    if len(payload) % SAMPLE_BYTES:
        raise ValueError(f"pcm payload of {len(payload)} bytes is not whole samples")
    return AudioFrame(int(header["seq"]), float(header["t"]), payload, rate)


def parse_doa(header: dict[str, Any]) -> DoaReading:
    """A ``doa`` message back into a reading; raises ValueError when it is malformed."""
    deg = int(header["deg"])
    if not 0 <= deg < 360:
        raise ValueError(f"doa angle {deg} outside 0..359")
    return DoaReading(float(header["t"]), deg, bool(header["speech"]))


class Speaker(Protocol):
    """Where speech goes: the board's playback through the array (:class:`AudioClient`)."""

    def play(self, pcm: bytes) -> None:
        """Queue s16le mono PCM at the link's play rate."""
        ...

    def play_end(self) -> None:
        """The utterance is complete: play what is queued, then close the device."""
        ...

    def flush(self) -> None:
        """Silence the speaker now, dropping whatever is queued."""
        ...


def play_paced(
    speaker: Speaker,
    pcm: bytes,
    *,
    rate: int = RATE,
    lead_s: float = 1.0,
    chunk_s: float = 0.02,
    stop: threading.Event | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
) -> float:
    """Send ``pcm`` in ``chunk_s`` pieces at real time, ``lead_s`` ahead of the speaker, then
    ``play_end``; returns the seconds of audio sent.

    The lead is the jitter buffer: the board's WiFi freezes for 300-600 ms now and then, and a
    second of audio already on the board plays through that. Sending everything at once would
    also work but would put the whole utterance beyond the reach of a cheap ``flush``: this way
    a barge-in (``stop`` set) sends ``flush`` and at most ``lead_s`` was ever queued.
    """
    step = max(SAMPLE_BYTES, int(rate * chunk_s) * SAMPLE_BYTES)
    started = clock()
    sent = 0
    while sent < len(pcm):
        if stop is not None and stop.is_set():
            speaker.flush()
            return sent / SAMPLE_BYTES / rate
        ahead = sent / SAMPLE_BYTES / rate - (clock() - started)
        if ahead > lead_s:
            sleep(min(ahead - lead_s, chunk_s))
            continue
        speaker.play(pcm[sent : sent + step])
        sent += step
    speaker.play_end()
    return len(pcm) / SAMPLE_BYTES / rate


def tcp_connect(address: tuple[str, int]) -> socket.socket:
    """Default connector: a blocking TCP socket to ``address`` (2 s to connect), without
    Nagle's delay — a ``flush`` is a few bytes that must leave at once."""
    sock = socket.create_connection(address, timeout=2.0)
    sock.settimeout(None)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    return sock


class AudioClient:
    """The laptop's end of the audio link: frames and directions in, speech out.

    :meth:`start` connects and waits for the board's hello; a reader thread then files frames
    into a bounded queue (the oldest go first when nobody reads, counted in ``overflowed``),
    keeps the last few seconds of directions, and answers :meth:`status`. Sends are blocking
    and serialised; the board never stops reading, so they only wait on the network.
    """

    def __init__(
        self,
        host: str,
        port: int = AUDIO_PORT,
        *,
        connector: Connector | None = None,
        buffer_s: float = 10.0,
        doa_history_s: float = 5.0,
    ) -> None:
        """Prepare a client for ``host:port``; nothing connects until :meth:`start`."""
        self._address = (host, port)
        self._connector = connector or tcp_connect
        self._sock: socket.socket | None = None
        self._send_lock = threading.Lock()
        self._cond = threading.Condition()
        self._frames: deque[AudioFrame] = deque(maxlen=max(1, int(buffer_s * RATE / FRAME_SAMPLES)))
        self._doa: deque[DoaReading] = deque()
        self._doa_history_s = doa_history_s
        self._doa_arrived = 0.0
        self._hello: dict[str, Any] | None = None
        self._rate = RATE
        self._status: dict[str, Any] | None = None
        self._status_count = 0
        self._last_seq: int | None = None
        self.frames_received = 0
        self.lost_frames = 0  # seq gaps: frames the board sent that never arrived here
        self.overflowed = 0  # frames dropped here because nobody consumed them
        self.connected = False

    def start(self, *, listen: bool = True, timeout_s: float = 3.0) -> Self:
        """Connect, wait for the hello, and (``listen``) subscribe to frames and directions;
        raises OSError when the board does not answer."""
        sock = self._connector(self._address)
        self._sock = sock
        self.connected = True
        threading.Thread(target=self._read, daemon=True, name="audio-reader").start()
        with self._cond:
            if not self._cond.wait_for(lambda: self._hello is not None or not self.connected,
                                       timeout_s):  # fmt: skip
                self.close()
                raise TimeoutError(f"no hello from {self._address[0]}:{self._address[1]}")
        if not self.connected:
            raise ConnectionError("the board closed the audio link at once")
        if listen:
            self._send({"cmd": "listen"})
        return self

    @property
    def hello(self) -> dict[str, Any]:
        """What the board announced on connect (rates, format, device)."""
        return dict(self._hello or {})

    @property
    def play_rate(self) -> int:
        """The sample rate the board plays at."""
        return int(self.hello.get("play_rate", RATE))

    def next_frame(self, timeout_s: float = 1.0) -> AudioFrame | None:
        """The oldest unread frame, waiting up to ``timeout_s``; None on timeout or when closed."""
        with self._cond:
            self._cond.wait_for(lambda: bool(self._frames) or not self.connected, timeout_s)
            return self._frames.popleft() if self._frames else None

    def frames(self) -> Iterator[AudioFrame]:
        """Every frame as it arrives, until the link closes."""
        while self.connected or self._frames:
            frame = self.next_frame(0.5)
            if frame is not None:
                yield frame

    def latest_doa(self) -> DoaReading | None:
        """The newest direction, or None before the first."""
        with self._cond:
            return self._doa[-1] if self._doa else None

    def doa_age_s(self) -> float:
        """Laptop seconds since the newest direction arrived; infinite before the first."""
        return time.monotonic() - self._doa_arrived if self._doa_arrived else float("inf")

    def doa_between(self, t0: float, t1: float) -> list[DoaReading]:
        """The directions read between board times ``t0`` and ``t1`` (the last few seconds)."""
        with self._cond:
            return [d for d in self._doa if t0 <= d.stamp_s <= t1]

    def status(self, timeout_s: float = 2.0) -> dict[str, Any] | None:
        """Ask the board how the audio is doing; its answer, or None after ``timeout_s``."""
        with self._cond:
            before = self._status_count
        self._send({"cmd": "status"})
        with self._cond:
            self._cond.wait_for(lambda: self._status_count > before, timeout_s)
            return dict(self._status) if self._status_count > before and self._status else None

    def play(self, pcm: bytes) -> None:
        """Queue s16le mono PCM at :attr:`play_rate` on the board's speaker."""
        if pcm:
            self._send({"cmd": "play"}, pcm)

    def play_end(self) -> None:
        """The utterance is complete: the board plays what it has, then closes the device."""
        self._send({"cmd": "play_end"})

    def flush(self) -> None:
        """Silence the speaker now."""
        self._send({"cmd": "flush"})

    def close(self) -> None:
        """Drop the connection; the reader thread ends and waiting calls return."""
        sock, self._sock = self._sock, None
        with self._cond:
            self.connected = False
            self._cond.notify_all()
        if sock is not None:
            with contextlib.suppress(OSError):
                sock.shutdown(socket.SHUT_RDWR)
            sock.close()

    # -- internals ----------------------------------------------------------

    def _send(self, message: dict[str, Any], payload: bytes = b"") -> None:
        sock = self._sock
        if sock is None:
            raise ConnectionError("the audio link is closed")
        with self._send_lock:
            sock.sendall(encode(message, payload))

    def _read(self) -> None:
        decoder = FrameDecoder()
        try:
            while True:
                sock = self._sock
                if sock is None:
                    break
                chunk = sock.recv(65536)
                if not chunk:
                    if self._sock is not None:  # not our own close()
                        logger.warning("audio link closed by the board")
                    break
                for header, payload in decoder.feed(chunk):
                    self._dispatch(header, payload)
        except (OSError, ProtocolError) as exc:
            if self._sock is not None:
                logger.warning("audio link lost: %s", exc)
        finally:
            self.close()

    def _dispatch(self, header: dict[str, Any], payload: bytes) -> None:
        kind = header.get("type")
        try:
            if kind == "pcm":
                self._take_frame(parse_frame(header, payload, self._rate))
            elif kind == "doa":
                reading = parse_doa(header)
                with self._cond:
                    self._doa.append(reading)
                    while (
                        self._doa and self._doa[0].stamp_s < reading.stamp_s - self._doa_history_s
                    ):
                        self._doa.popleft()
                self._doa_arrived = time.monotonic()
            elif kind == "hello":
                with self._cond:
                    self._rate = int(header.get("rate", RATE))
                    self._hello = header
                    self._cond.notify_all()
            elif kind == "status":
                with self._cond:
                    self._status = header
                    self._status_count += 1
                    self._cond.notify_all()
            elif kind == "error":
                logger.warning("board: %s", header.get("error"))
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("audio link: bad %s message dropped (%s)", kind, exc)

    def _take_frame(self, frame: AudioFrame) -> None:
        with self._cond:
            if self._last_seq is not None and frame.seq > self._last_seq + 1:
                self.lost_frames += frame.seq - self._last_seq - 1
            self._last_seq = frame.seq
            if len(self._frames) == self._frames.maxlen:
                self.overflowed += 1
            self._frames.append(frame)
            self.frames_received += 1
            self._cond.notify_all()
