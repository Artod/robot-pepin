"""Audio server: runs on the board and owns the reSpeaker XVF3800 microphone array.

Three jobs, one TCP port (3338, framing in :mod:`pepin.audio_link`):

- **Hearing.** ``arecord`` captures the array at 16 kHz, 2 channels, and one channel goes out as
  20 ms frames, each stamped with the board's monotonic clock at its first sample. Channel 0 by
  default: the array's processed output (echo cancelled, beamformed, noise suppressed, level
  controlled), because the robot will talk while it listens and its motors hum — the residual
  echo and noise suppression live only on that channel. Channel 1 is the ASR beam (echo
  cancelled and beamformed, fixed gain, no suppression), one flag away for an engine that wants
  it raw. A capture that delivers nothing for two seconds is logged loudly and reopened.
- **Direction.** The voice's direction (``DOA_VALUE``, :mod:`pepin.xvf3800`) is read over USB
  control at 10 Hz and sent to the listeners as JSON.
- **Speech.** PCM from the laptop plays through the array's own output, ``aplay`` on the same
  card: the speaker hangs on the array's jack, so the echo canceller's reference is exactly what
  the room hears. ``flush`` silences it at once (barge-in).

A ``status`` command reports whether capture is alive, frames per second, dropouts, the age of
the last direction and the playback's state; the same numbers go to the journal once a minute.

Run on the board (see ``board/pepin-audio.service``)::

    python -m pepin.audio_server --port 3338

Standard library only, plus ``pyusb`` for the direction (imported lazily; without it the audio
still flows and the status says why there is no direction). The pure logic — channel split,
framing, stamps, the deadman, the player's queue — is tested with fakes (tests/unit).
"""

from __future__ import annotations

import argparse
import array
import contextlib
import logging
import math
import os
import queue
import re
import select
import signal
import socket
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass
from typing import IO, Any, Protocol, Self

from pepin.audio_link import (
    AUDIO_PORT,
    FRAME_SAMPLES,
    RATE,
    SAMPLE_BYTES,
    AudioFrame,
    DoaReading,
    FrameDecoder,
    ProtocolError,
    encode,
)
from pepin.face import FaceTable, load_face_table, mouth_level
from pepin.telemetry import LatencyTracker
from pepin.xvf3800 import Xvf3800, firmware_warning, open_array

logger = logging.getLogger(__name__)

CARD_MARK = "XVF3800"  # in the array's ALSA long name: "reSpeaker XVF3800 4-Mic Array"
CARDS_FILE = "/proc/asound/cards"
DEADMAN_S = 2.0  # capture silent this long: log loudly, reopen
FPS_WINDOW_S = 5.0


# -- the sound card ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Card:
    """One ALSA card as /proc/asound/cards lists it."""

    index: int
    id: str  # "respeaker" with board/99-pepin-usb.rules, "Array" without
    name: str  # the long name: product and, for USB, where it hangs ("at usb-...-1.3")


_CARD_LINE = re.compile(r"^\s*(\d+)\s+\[(\S+)\s*\]:\s*(.*)$")


def parse_cards(text: str) -> list[Card]:
    """The cards of a /proc/asound/cards text (two lines per card)."""
    cards: list[Card] = []
    lines = text.splitlines()
    for i, line in enumerate(lines):
        match = _CARD_LINE.match(line)
        if match is None:
            continue
        long_name = lines[i + 1].strip() if i + 1 < len(lines) else ""
        cards.append(Card(int(match.group(1)), match.group(2), f"{match.group(3)} | {long_name}"))
    return cards


def find_array(text: str) -> Card | None:
    """The XVF3800 among the cards of a /proc/asound/cards text, or None."""
    return next((c for c in parse_cards(text) if CARD_MARK in c.name), None)


def resolve_device(device: str, cards_file: str = CARDS_FILE) -> tuple[str, str]:
    """The ALSA device to open and a note about where it is: ``device`` itself unless it is
    "auto", in which case the array is looked up by name; raises OSError when it is absent."""
    if device != "auto":
        return device, device
    try:
        with open(cards_file) as handle:
            text = handle.read()
    except OSError as exc:
        raise OSError(f"cannot read {cards_file}: {exc}") from exc
    card = find_array(text)
    if card is None:
        raise OSError(f"no {CARD_MARK} among the sound cards ({cards_file})")
    return f"plughw:CARD={card.id},DEV=0", card.name


# -- samples, frames and stamps ----------------------------------------------------------------


class ChannelSplitter:
    """Takes one channel out of interleaved s16le chunks cut anywhere, carrying partial frames."""

    def __init__(self, channels: int, index: int) -> None:
        """``channels`` interleaved in the stream; ``index`` the one kept (0-based)."""
        if not 0 <= index < channels:
            raise ValueError(f"channel {index} of {channels}")
        self._channels = channels
        self._index = index
        self._stride = channels * SAMPLE_BYTES
        self._rest = b""

    def take(self, chunk: bytes) -> bytes:
        """The kept channel's samples, as s16le, from every whole frame so far."""
        data = self._rest + chunk
        whole = len(data) - len(data) % self._stride
        self._rest = data[whole:]
        if self._channels == 1:
            return data[:whole]
        # Picking every n-th two-byte element moves bytes, never interprets them: byte order
        # does not matter here.
        samples = array.array("h")
        samples.frombytes(data[:whole])
        return samples[self._index :: self._channels].tobytes()

    def reset(self) -> None:
        """Forget a partial frame (a new stream starts)."""
        self._rest = b""


class SampleClock:
    """The board-monotonic time of any sample of one capture stream.

    Every chunk arrives after its last sample was captured, so ``arrival - samples_so_far/rate``
    is an upper bound on the moment the stream started; the tightest bound seen is the best
    estimate. Reads come in bursts, and taking the least-delayed one keeps that burstiness out
    of the stamps, which trail reality only by the pipeline's constant minimum latency. The bound
    leaks upward at ``leak`` (1000 ppm, beyond any crystal) so a sound card clock slower than the
    board's is followed instead of pinning the start in the past.
    """

    def __init__(self, rate: int, leak: float = 1e-3) -> None:
        """``rate`` in Hz; ``leak`` in seconds per second."""
        self._rate = rate
        self._leak = leak
        self._start: float | None = None
        self._last_arrival = 0.0

    def observe(self, arrival_s: float, samples_total: int) -> None:
        """A chunk arrived at ``arrival_s`` bringing the stream to ``samples_total`` samples."""
        bound = arrival_s - samples_total / self._rate
        if self._start is None:
            self._start = bound
        else:
            self._start = min(self._start + self._leak * (arrival_s - self._last_arrival), bound)
        self._last_arrival = arrival_s

    def time_of(self, sample_index: int) -> float:
        """When sample ``sample_index`` (0 = the stream's first) was captured."""
        if self._start is None:
            raise RuntimeError("no chunk observed yet")
        return self._start + sample_index / self._rate


class Framer:
    """Cuts a mono s16le stream into fixed frames and says where each one starts."""

    def __init__(self, frame_samples: int = FRAME_SAMPLES) -> None:
        """``frame_samples`` per frame (320 = 20 ms at 16 kHz)."""
        self._bytes = frame_samples * SAMPLE_BYTES
        self._buffer = bytearray()
        self.samples_in = 0  # samples received in this stream
        self._framed = 0  # samples handed out as frames

    def push(self, mono: bytes) -> list[tuple[int, bytes]]:
        """Add samples; every completed frame as (index of its first sample, its bytes)."""
        self._buffer += mono
        self.samples_in += len(mono) // SAMPLE_BYTES
        out = []
        while len(self._buffer) >= self._bytes:
            out.append((self._framed, bytes(self._buffer[: self._bytes])))
            del self._buffer[: self._bytes]
            self._framed += self._bytes // SAMPLE_BYTES
        return out

    def reset(self) -> None:
        """Start a new stream at sample 0."""
        self._buffer.clear()
        self.samples_in = 0
        self._framed = 0


# -- capture -----------------------------------------------------------------------------------


class PcmSource(Protocol):
    """A capture stream of interleaved s16le; the real one is :class:`ArecordSource`."""

    def open(self) -> str:
        """Start capturing; returns where from (for the logs); raises OSError when it cannot."""
        ...

    def read(self, timeout_s: float) -> bytes:
        """Whatever arrived within ``timeout_s`` (b"" if nothing); raises EOFError when the
        stream has ended."""
        ...

    def close(self) -> None:
        """Stop capturing; idempotent."""
        ...


class XrunCounter:
    """Counts the over/underruns an ALSA tool reports on stderr and logs its other complaints."""

    def __init__(self) -> None:
        self.count = 0
        self._lock = threading.Lock()

    def watch(self, stream: IO[bytes], tool: str) -> None:
        """Read ``stream`` to its end in a daemon thread."""
        threading.Thread(
            target=self._read, args=(stream, tool), daemon=True, name=f"{tool}-stderr"
        ).start()

    def line(self, text: str, tool: str) -> None:
        """One stderr line: an xrun is counted, anything else logged (the first lines of
        arecord/aplay say what they opened)."""
        text = text.removeprefix(f"{tool}: ")
        if "overrun" in text or "underrun" in text:
            with self._lock:
                self.count += 1
            logger.warning("%s: %s", tool, text)
        elif text.startswith(("Recording", "Playing")):
            logger.info("%s: %s", tool, text)
        elif text:
            logger.error("%s: %s", tool, text)

    def _read(self, stream: IO[bytes], tool: str) -> None:
        with contextlib.suppress(OSError, ValueError):
            for raw in stream:
                self.line(raw.decode(errors="replace").strip(), tool)


class ArecordSource:
    """``arecord`` from alsa-utils on the array: raw s16le to a pipe, 20 ms periods, a 0.5 s
    buffer so a busy board delays the audio instead of losing it."""

    def __init__(self, device: str, *, rate: int, channels: int, xruns: XrunCounter) -> None:
        """``device`` is an ALSA name or "auto" (the XVF3800 found by name at every open)."""
        self._device = device
        self._rate = rate
        self._channels = channels
        self._xruns = xruns
        self._proc: subprocess.Popen[bytes] | None = None
        self.where = ""

    def open(self) -> str:
        """Spawn arecord on the array; raises OSError when the card is absent."""
        device, self.where = resolve_device(self._device)
        command = [
            "arecord", "-D", device, "-t", "raw", "-f", "S16_LE", "-r", str(self._rate),
            "-c", str(self._channels), "--period-time=20000", "--buffer-time=500000",
        ]  # fmt: skip
        proc = subprocess.Popen(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        assert proc.stdout is not None and proc.stderr is not None
        self._xruns.watch(proc.stderr, "arecord")
        self._proc = proc
        return device if self.where == device else f"{device} ({self.where})"

    def read(self, timeout_s: float) -> bytes:
        """Bytes from arecord's pipe; EOFError once arecord has exited."""
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise EOFError("arecord is not running")
        fd = proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout_s)
        if not ready:
            return b""
        data = os.read(fd, 65536)
        if not data:
            code = proc.wait(timeout=1.0)
            raise EOFError(f"arecord exited with code {code}")
        return data

    def close(self) -> None:
        """Kill arecord (the card is released with it)."""
        proc, self._proc = self._proc, None
        if proc is not None:
            proc.kill()
            with contextlib.suppress(subprocess.TimeoutExpired):
                proc.wait(timeout=2.0)
            for stream in (proc.stdout, proc.stderr):
                if stream is not None:
                    stream.close()


# -- the cold-boot rescue ----------------------------------------------------------------------


@dataclass
class DeadStreamPolicy:
    """When to reboot the array's chip: after ``dead_streams`` capture streams in a row closed
    without delivering a byte, at most ``max_reboots`` times in the server's life.

    After a cold power-up the XVF3800 enumerates and answers control transfers, yet its capture
    endpoint delivers nothing: every arecord ends 0.55 s after the open (1.1x the 0.5 s buffer,
    the kernel's capture timeout) in "read error: Input/output error", for hours. The chip's
    own reboot cures it. A stream that delivered and then ended is another failure (a USB
    controller dropping the stream) and resets the count.
    """

    dead_streams: int = 2
    max_reboots: int = 1
    dead_in_row: int = 0
    reboots: int = 0

    def closed(self, delivered: bool) -> bool:
        """Record one stream that closed; True when the chip should be rebooted now."""
        if delivered:
            self.dead_in_row = 0
            return False
        self.dead_in_row += 1
        if self.dead_in_row < self.dead_streams or self.reboots >= self.max_reboots:
            return False
        self.reboots += 1
        self.dead_in_row = 0
        return True


Runner = Callable[[list[str]], subprocess.CompletedProcess[str]]


def _run(command: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, capture_output=True, text=True, timeout=5.0, check=False)


class XvfHost:
    """Seeed's ``xvf_host`` tool (``board/xvf_host_install.sh``): the chip's state in one line
    and its reboot (it re-enumerates on USB, every parameter back to default)."""

    # The GPO levels (X0D11, X0D30, X0D31, X0D33, X0D39) and the GPI levels, the first of which
    # (X1D09) is the onboard mute button, 1 = released (Seeed's xvf_host.py): the mute question
    # in the journal at every rescue. The PLL and the device-to-host FIFO: was audio clocked.
    STATE = ("GPO_READ_VALUES", "GPI_READ_VALUES", "PLL_LOCK_STATUS", "USB_D2H_BUFFER_STABLE")

    def __init__(self, tool: str = "xvf_host", *, run: Runner = _run) -> None:
        """``tool`` is the command (a path or a name on PATH)."""
        self._tool = tool
        self._run = run

    def read(self, name: str) -> str:
        """``name``'s answer as the tool prints it ("GPO_READ_VALUES 0 0 0 1 0"); raises OSError."""
        try:
            done = self._run([self._tool, name])
        except (OSError, subprocess.SubprocessError) as exc:
            raise OSError(f"{self._tool} {name}: {exc}") from exc
        for line in reversed(done.stdout.splitlines()):
            if line.startswith(name + " "):
                return line.strip()
        tail = (done.stderr or done.stdout).strip().splitlines()
        raise OSError(f"{self._tool} {name}: {tail[-1] if tail else f'exit {done.returncode}'}")

    def state(self) -> str:
        """The mute LED, the mute button, the PLL and the FIFO in one line; a failed read says
        so in its place."""
        parts = []
        for name in self.STATE:
            try:
                parts.append(self.read(name))
            except OSError as exc:
                parts.append(f"{name} ? ({exc})")
        return "; ".join(parts)

    def reboot(self) -> None:
        """``xvf_host REBOOT 1``; raises OSError when the tool cannot be run or fails."""
        try:
            done = self._run([self._tool, "REBOOT", "1"])
        except (OSError, subprocess.SubprocessError) as exc:
            raise OSError(f"{self._tool} REBOOT 1: {exc}") from exc
        if done.returncode != 0 or "expects" in done.stdout:
            raise OSError(f"{self._tool} REBOOT 1: {done.stdout.strip() or done.returncode}")


def reboot_array(
    xvf: XvfHost,
    *,
    settle_s: float = 2.0,
    wait_s: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
) -> None:
    """Log the chip's state, reboot it, wait until it answers again and log the state after.

    Blocks the caller for ``settle_s`` and up to ``wait_s`` more; raises OSError when the
    reboot itself cannot be issued."""
    logger.error("array state before its reboot: %s", xvf.state())
    xvf.reboot()
    sleep(settle_s)
    deadline = clock() + wait_s
    while True:
        try:
            xvf.read("VERSION")
            break
        except OSError as exc:
            if clock() >= deadline:
                logger.error("the array did not answer %.0f s after its reboot: %s",
                             settle_s + wait_s, exc)  # fmt: skip
                return
            sleep(0.5)
    logger.error("array state after its reboot: %s", xvf.state())


class CaptureLoop:
    """Keeps a capture stream open and turns it into stamped mono frames.

    One :meth:`step` is one read (or one open attempt). A stream that ends (arecord exits: the
    card vanished, or the "read error: Input/output error" of a USB controller that drops the
    array's isochronous stream) is reopened after ``retry_s``; a stream that stays open but
    delivers nothing for ``deadman_s`` is logged as a stall and reopened at once. Each opening
    starts a new sample clock; ``seq`` runs on across them, the stamps say where time jumped.
    With ``rescue``, streams that close without a byte reboot the array's chip as
    :class:`DeadStreamPolicy` says (the XVF3800's cold-boot state).
    """

    def __init__(
        self,
        source: PcmSource,
        on_frame: Callable[[AudioFrame], None],
        *,
        channels: int = 2,
        channel: int = 0,
        rate: int = RATE,
        frame_samples: int = FRAME_SAMPLES,
        deadman_s: float = DEADMAN_S,
        retry_s: float = 2.0,
        read_timeout_s: float = 0.2,
        clock: Callable[[], float] = time.monotonic,
        rescue: Callable[[], None] | None = None,
        policy: DeadStreamPolicy | None = None,
    ) -> None:
        """``on_frame`` receives every frame on the capture thread (it must not block);
        ``rescue`` reboots the array (blocking, on the capture thread), None never does."""
        self._source = source
        self._rescue = rescue
        self._policy = policy if policy is not None else DeadStreamPolicy()
        self._delivered = False
        self._rescued = False
        self._on_frame = on_frame
        self._splitter = ChannelSplitter(channels, channel)
        self._framer = Framer(frame_samples)
        self._rate = rate
        self._stamps = SampleClock(rate)
        self._deadman_s = deadman_s
        self._retry_s = retry_s
        self._read_timeout_s = read_timeout_s
        self._clock = clock
        self._open = False
        self._next_open = 0.0
        self._last_data = 0.0
        self._failed_opens = 0
        self._recent: deque[float] = deque()
        self._first_open: float | None = None
        self.where = ""
        self.error: str | None = None
        self.seq = 0
        self.opens = 0
        self.ends = 0  # streams that ended by themselves
        self.stalls = 0  # streams the deadman closed
        self.reboots = 0  # of the array's chip, by ``rescue``
        self.last_frame_at: float | None = None
        self.frame_age = LatencyTracker("audio.frame_age")

    def step(self) -> None:
        """One read, or one attempt to open when no stream is running."""
        if not self._open:
            self._try_open()
            return
        try:
            data = self._source.read(self._read_timeout_s)
        except (EOFError, OSError) as exc:
            self.ends += 1
            self._drop(f"capture ended: {exc}", retry=True)
            return
        now = self._clock()
        if data:
            self._last_data = now
            if not self._delivered and self._rescued:
                logger.error("capture ALIVE after the array's reboot: %s", self.where)
                self._rescued = False
            self._delivered = True
            self._ingest(data, now)
        elif now - self._last_data > self._deadman_s:
            self.stalls += 1
            logger.error(
                "CAPTURE STALLED: nothing from %s for %.1f s — reopening (stall %d). If this "
                "repeats every few seconds, the USB controller is dropping the array's stream: "
                "scratch/audio/DAY_ONE.md, the 13-pin bus",
                self.where, now - self._last_data, self.stalls,
            )  # fmt: skip
            self._drop("stalled", retry=False)

    def run(self, stop: threading.Event) -> None:
        """Step until ``stop`` is set, then close the stream."""
        try:
            while not stop.is_set():
                if not self._open and self._clock() < self._next_open:
                    stop.wait(min(0.2, self._next_open - self._clock()))
                    continue
                self.step()
        finally:
            self.close()

    def close(self) -> None:
        """Close the stream (shutdown)."""
        self._source.close()
        self._open = False

    def alive(self, now: float) -> bool:
        """A stream is open and delivered a frame within the deadman time."""
        last = self.last_frame_at
        return self._open and last is not None and now - last <= self._deadman_s

    def frames_per_s(self, now: float) -> float:
        """Frames emitted per second over the last few seconds (fewer right after the start)."""
        while self._recent and self._recent[0] < now - FPS_WINDOW_S:
            self._recent.popleft()
        if self._first_open is None:
            return 0.0
        return len(self._recent) / max(1e-3, min(FPS_WINDOW_S, now - self._first_open))

    def _try_open(self) -> None:
        now = self._clock()
        if now < self._next_open:
            return
        try:
            self.where = self._source.open()
        except OSError as exc:
            self._failed_opens += 1
            self.error = str(exc)
            self._next_open = now + self._retry_s
            if self._failed_opens in (1, 5) or self._failed_opens % 30 == 0:
                logger.warning("capture not available (%s); retrying every %.0f s",
                               exc, self._retry_s)  # fmt: skip
            return
        self._open = True
        self._delivered = False
        self.opens += 1
        if self._first_open is None:
            self._first_open = now
        self._failed_opens = 0
        self.error = None
        self._last_data = now
        self._splitter.reset()
        self._framer.reset()
        self._stamps = SampleClock(self._rate)
        logger.info("capture open: %s", self.where)

    def _drop(self, why: str, *, retry: bool) -> None:
        self._source.close()
        self._open = False
        self.error = why
        self._next_open = self._clock() + (self._retry_s if retry else 0.0)
        if retry:
            logger.error("%s; reopening in %.0f s", why, self._retry_s)
        if self._rescue is None:
            return
        policy = self._policy
        if policy.closed(self._delivered):
            logger.error(
                "ARRAY CAPTURE DEAD: %d streams in a row closed without a byte (the XVF3800's "
                "cold-boot state); rebooting the chip (xvf_host REBOOT 1), reboot %d of %d",
                policy.dead_streams, policy.reboots, policy.max_reboots,
            )  # fmt: skip
            self.reboots += 1
            self._rescued = True
            try:
                self._rescue()
            except OSError as exc:
                logger.error("the array's reboot failed: %s", exc)
            self._next_open = self._clock()
        elif self._rescued and policy.dead_in_row == policy.dead_streams:
            logger.error(
                "capture STILL DEAD after the array's reboot (%d streams without a byte); no "
                "more reboots — power-cycle the array or `xvf_host REBOOT 1` by hand",
                policy.dead_in_row,
            )  # fmt: skip

    def _ingest(self, data: bytes, now: float) -> None:
        mono = self._splitter.take(data)
        frames = self._framer.push(mono)
        self._stamps.observe(now, self._framer.samples_in)
        for first_sample, pcm in frames:
            frame = AudioFrame(self.seq, self._stamps.time_of(first_sample), pcm, self._rate)
            self.seq += 1
            self.last_frame_at = now
            self._recent.append(now)
            self.frame_age.add(max(0.0, now - frame.stamp_s - frame.duration_s))
            self._on_frame(frame)


# -- direction ---------------------------------------------------------------------------------


class DoaPoller:
    """Reads the voice direction at ``hz`` and hands each reading on; reopens a lost array."""

    def __init__(
        self,
        open_control: Callable[[], Xvf3800],
        on_reading: Callable[[DoaReading], None],
        *,
        hz: float = 10.0,
        retry_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``open_control`` finds the array (raises OSError when absent)."""
        self._open_control = open_control
        self._on_reading = on_reading
        self._period = 1.0 / hz
        self._retry_s = retry_s
        self._clock = clock
        self._control: Xvf3800 | None = None
        self._next_open = 0.0
        self._failed_opens = 0
        self._no_doa = False  # firmware older than DOA_VALUE
        self.latest: DoaReading | None = None
        self.firmware: str | None = None
        self.error: str | None = None
        self.reads = 0
        self.failures = 0
        self.latency = LatencyTracker("xvf.doa_read")

    def step(self) -> None:
        """One read, or one attempt to find the array."""
        if self._control is None:
            self._try_open()
            return
        if self._no_doa:
            return
        started = self._clock()
        try:
            deg, speech = self._control.doa()
        except OSError as exc:
            self.failures += 1
            self.error = f"DOA read failed: {exc}"
            logger.warning("%s; reopening the array's control", self.error)
            with contextlib.suppress(OSError):
                self._control.close()
            self._control = None
            self._next_open = started + self._retry_s
            return
        now = self._clock()
        self.latency.add(now - started)
        self.reads += 1
        self.error = None
        self.latest = DoaReading(started, deg, speech)
        self._on_reading(self.latest)

    def run(self, stop: threading.Event) -> None:
        """Step at the polling rate until ``stop`` is set."""
        try:
            while not stop.is_set():
                started = self._clock()
                self.step()
                wait = self._period if self._control is not None else self._retry_s
                stop.wait(max(0.0, wait - (self._clock() - started)))
        finally:
            if self._control is not None:
                with contextlib.suppress(OSError):
                    self._control.close()

    def age_s(self, now: float) -> float | None:
        """Seconds since the newest reading, None before the first."""
        return None if self.latest is None else now - self.latest.stamp_s

    def _try_open(self) -> None:
        now = self._clock()
        if now < self._next_open:
            return
        try:
            control = self._open_control()
            version = control.version()
        except (OSError, ImportError) as exc:  # ImportError: no pyusb on this machine
            self._failed_opens += 1
            self.error = f"array control unavailable: {exc}"
            self._next_open = now + self._retry_s
            if self._failed_opens in (1, 5) or self._failed_opens % 30 == 0:
                logger.warning("%s; retrying every %.0f s", self.error, self._retry_s)
            return
        self._control = control
        self._failed_opens = 0
        self.firmware = ".".join(str(v) for v in version)
        warning = firmware_warning(version)
        self._no_doa = version < (2, 0, 6)
        self.error = warning if self._no_doa else None
        logger.info("array control open: firmware %s", self.firmware)
        if warning:
            logger.error("firmware %s: %s", self.firmware, warning)


# -- speech ------------------------------------------------------------------------------------


class PcmSink(Protocol):
    """A playback device fed s16le mono; the real one is :class:`AplaySink`."""

    def write(self, pcm: bytes) -> None:
        """Queue samples; may block at the device's pace; raises OSError once killed."""
        ...

    def finish(self) -> None:
        """Play what is queued, then release the device (blocks until played)."""
        ...

    def abort(self) -> None:
        """Release the device now, dropping what is queued; safe from any thread."""
        ...


class AplaySink:
    """``aplay`` on the array: the only path to the speaker that the echo canceller hears."""

    def __init__(self, device: str, *, rate: int, xruns: XrunCounter) -> None:
        """Spawn aplay for raw s16le mono at ``rate``; raises OSError when it cannot start."""
        resolved, _ = resolve_device(device)
        command = ["aplay", "-D", resolved, "-t", "raw", "-f", "S16_LE", "-r", str(rate), "-c", "1"]
        self._proc = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        assert self._proc.stderr is not None
        xruns.watch(self._proc.stderr, "aplay")

    def write(self, pcm: bytes) -> None:
        """Into aplay's pipe; blocks while the pipe is full (the device paces it)."""
        stdin = self._proc.stdin
        if stdin is None:
            raise BrokenPipeError("aplay is closed")
        stdin.write(pcm)
        stdin.flush()

    def finish(self) -> None:
        """Close the pipe and wait for aplay to play the rest and exit."""
        stdin = self._proc.stdin
        if stdin is not None:
            with contextlib.suppress(OSError):
                stdin.close()
        try:
            self._proc.wait(timeout=30.0)
        except subprocess.TimeoutExpired:
            self.abort()

    def abort(self) -> None:
        """Kill aplay: silence within milliseconds."""
        with contextlib.suppress(OSError):
            self._proc.kill()


class LipSync:
    """The mouth's levels for the speech the speaker plays (``--lipsync``): the loudness of each
    ``window_s`` of every chunk the player hands its device, dated by when that window will be
    heard, and sent to the head server (``send``, a level 0..1) as each comes due.

    The device takes a chunk long before it sounds (aplay's pipe alone holds two seconds), so
    a window's moment is counted in samples from the utterance's first sound: the moment its
    device opened plus ``start_latency_s``. A chunk that comes after its own moment (the laptop
    fell behind and the speaker ran dry) restarts that count where the sound resumes. A flush or
    the end of the utterance closes the mouth at once. ``table`` is config/face.json: the window,
    the latency and the dB range a level spans.
    """

    def __init__(
        self,
        send: Callable[[float], None],
        table: FaceTable,
        *,
        rate: int = RATE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``send`` gets each level when it is due, on this object's own thread (:meth:`run`)."""
        self._send = send
        self._table = table
        self._rate = rate
        self._clock = clock
        lipsync = table.lipsync
        self._window = max(1, round(lipsync["window_s"] * rate))
        self._latency_s = lipsync["start_latency_s"] - lipsync["lead_s"]
        self._cond = threading.Condition()
        self._due: deque[tuple[float, float]] = deque()  # (when, level)
        self._anchor: float | None = None  # when the utterance's sample 0 is heard
        self._samples = 0  # samples of this utterance handed to the device
        self._acc = 0.0  # the open window's sum of squares...
        self._acc_n = 0  # ...and its samples
        self._open = False  # the last level sent was not a zero
        self.sent = 0
        self.late = 0  # chunks that came after their moment: the count restarted

    def chunk(self, pcm: bytes) -> None:
        """A chunk the device has just taken (whole s16le samples)."""
        now = self._clock()
        samples = array.array("h", pcm[: len(pcm) - len(pcm) % SAMPLE_BYTES])
        with self._cond:
            if self._anchor is None:
                self._anchor = now + self._latency_s
            elif self._anchor + self._samples / self._rate < now:
                self.late += 1
                self._anchor = now + self._latency_s - self._samples / self._rate
            start = 0
            while start < len(samples):
                take = min(self._window - self._acc_n, len(samples) - start)
                part = samples[start : start + take]
                self._acc += float(sum(v * v for v in part))
                self._acc_n += take
                start += take
                self._samples += take
                if self._acc_n == self._window:
                    rms = math.sqrt(self._acc / self._window) / 32768.0
                    db = 20.0 * math.log10(rms) if rms > 0.0 else -120.0
                    when = self._anchor + (self._samples - self._window) / self._rate
                    self._due.append((when, mouth_level(db, self._table)))
                    self._acc, self._acc_n = 0.0, 0
            self._cond.notify_all()

    def stop(self) -> None:
        """The utterance is over (played out, or flushed): the mouth closes now."""
        with self._cond:
            self._due.clear()
            self._anchor = None
            self._samples = 0
            self._acc, self._acc_n = 0.0, 0
            self._due.append((self._clock(), 0.0))
            self._cond.notify_all()

    def step(self, timeout_s: float = 0.05) -> None:
        """Send whatever came due; wait up to ``timeout_s`` (less when a level is due sooner)."""
        with self._cond:
            if not self._due:
                self._cond.wait(timeout_s)
            now = self._clock()
            ready = []
            while self._due and self._due[0][0] <= now:
                ready.append(self._due.popleft()[1])
            wait = self._due[0][0] - now if self._due else None
        for level in ready[-1:]:  # behind: only the newest level matters
            if level > 0.0 or self._open:
                self._send(level)
                self.sent += 1
            self._open = level > 0.0
        if wait is not None and not ready:
            time.sleep(min(max(wait, 0.0), timeout_s))

    def run(self, stop: threading.Event) -> None:
        """Step until ``stop``."""
        while not stop.is_set():
            self.step()

    def status(self) -> dict[str, Any]:
        """For the status line."""
        return {"lipsync_sent": self.sent, "lipsync_late": self.late}


class Player:
    """The speaker's queue: the laptop's chunks in, one playback device at a time out.

    :meth:`feed`, :meth:`end` and :meth:`flush` come from the command loop and never block; the
    device is written on the player's own thread (:meth:`run`). A device opens with the first
    chunk and closes when the laptop says the utterance is complete (``end``), when nothing has
    arrived for ``idle_close_s`` (a laptop that vanished mid-sentence), or at once on ``flush``.
    More than ``max_queue_s`` queued is refused and counted: a laptop that sends faster than the
    speaker plays is a bug on the laptop, and the board's memory is not the place to absorb it.
    A device that will not open costs one error and the rest of that utterance (until ``end``,
    or a pause of ``idle_close_s``), not one attempt per chunk.
    """

    def __init__(
        self,
        open_sink: Callable[[], PcmSink],
        *,
        rate: int = RATE,
        max_queue_s: float = 10.0,
        idle_close_s: float = 2.0,
        clock: Callable[[], float] = time.monotonic,
        lipsync: LipSync | None = None,
    ) -> None:
        """``open_sink`` starts a playback device (raises OSError when it cannot); ``lipsync``
        is told every chunk the device takes and every end (``--lipsync``)."""
        self._open_sink = open_sink
        self._lipsync = lipsync
        self._rate = rate
        self._max_bytes = int(max_queue_s * rate) * SAMPLE_BYTES
        self._idle_close_s = idle_close_s
        self._clock = clock
        self._cond = threading.Condition()
        self._queue: deque[bytes] = deque()
        self._queued = 0
        self._ending = False
        self._sink: PcmSink | None = None
        self._finishing: PcmSink | None = None  # playing its tail out, still cut by a flush
        self._last_feed = 0.0
        self._refusing = False  # the device would not open: drop this utterance
        self.played_s = 0.0
        self.overflows = 0
        self.flushes = 0
        self.errors = 0

    def feed(self, pcm: bytes) -> None:
        """Queue one chunk (whole samples); refused and counted past ``max_queue_s``."""
        pcm = pcm[: len(pcm) - len(pcm) % SAMPLE_BYTES]
        now = self._clock()
        with self._cond:
            if self._refusing:
                paused = now - self._last_feed > self._idle_close_s
                self._last_feed = now
                if not paused:
                    return
                self._refusing = False  # a new utterance: try the device again
            if self._queued + len(pcm) > self._max_bytes:
                self.overflows += 1
                return
            self._queue.append(pcm)
            self._queued += len(pcm)
            self._ending = False
            self._last_feed = now
            self._cond.notify_all()

    def end(self) -> None:
        """No more chunks for this utterance: play the queue out, then close the device."""
        with self._cond:
            self._ending = True
            self._refusing = False
            self._cond.notify_all()

    def flush(self) -> None:
        """Drop the queue and silence the device now."""
        with self._cond:
            self._queue.clear()
            self._queued = 0
            self._ending = False
            self.flushes += 1
            sinks = [self._sink, self._finishing]
            self._sink = self._finishing = None
            self._cond.notify_all()
        for sink in sinks:
            if sink is not None:
                sink.abort()
        if self._lipsync is not None:
            self._lipsync.stop()

    @property
    def queued_s(self) -> float:
        """Seconds of audio waiting on the board (not counting the device's own buffer)."""
        return self._queued / SAMPLE_BYTES / self._rate

    @property
    def playing(self) -> bool:
        """A playback device is open (or playing the tail of an utterance)."""
        return self._sink is not None or self._finishing is not None

    def step(self, timeout_s: float = 0.1) -> None:
        """One chunk to the device, or the device closed when the utterance is over."""
        with self._cond:
            self._cond.wait_for(lambda: bool(self._queue) or self._ending, timeout_s)
            generation = self.flushes
            chunk = self._queue.popleft() if self._queue else None
            finished = False
            if chunk is not None:
                self._queued -= len(chunk)
            elif self._ending or self._clock() - self._last_feed > self._idle_close_s:
                finished = self._sink is not None
                self._ending = False
        if chunk is not None:
            self._write(chunk, generation)
        elif finished:
            self._close_sink()

    def run(self, stop: threading.Event) -> None:
        """Step until ``stop`` is set; the device is silenced on the way out."""
        try:
            while not stop.is_set():
                self.step()
        finally:
            self.flush()

    def _write(self, chunk: bytes, generation: int) -> None:
        """``generation`` is the flush count when the chunk left the queue: a flush since then
        means the chunk was meant to die with the rest."""
        sink = self._sink
        if sink is None:
            try:
                sink = self._open_sink()
            except OSError as exc:
                self.errors += 1
                logger.error("playback cannot start: %s; this utterance is dropped", exc)
                with self._cond:
                    self._queue.clear()
                    self._queued = 0
                    self._refusing = True
                return
            with self._cond:
                flushed = self.flushes != generation
                if not flushed:
                    self._sink = sink
            if flushed:
                sink.abort()
                return
        try:
            sink.write(chunk)
        except (OSError, ValueError):  # killed by flush meanwhile: the chunk was meant to die
            return
        self.played_s += len(chunk) / SAMPLE_BYTES / self._rate
        if self._lipsync is not None and self.flushes == generation:
            self._lipsync.chunk(chunk)

    def _close_sink(self) -> None:
        with self._cond:
            sink, self._sink = self._sink, None
            self._finishing = sink
        if sink is not None:
            sink.finish()
            if self._lipsync is not None:
                self._lipsync.stop()
        with self._cond:
            if self._finishing is sink:
                self._finishing = None


# -- the socket --------------------------------------------------------------------------------


class Client:
    """One laptop connection: a reader into the shared inbox and a writer with a bounded outbox.

    Nothing blocks the capture thread: a client whose outbox is full loses messages (counted in
    ``dropped``; its ``seq`` gaps show it) rather than holding the others up, and a send that
    stalls for ``send_timeout_s`` ends the connection.
    """

    def __init__(
        self,
        conn: socket.socket,
        peer: str,
        inbox: queue.Queue[tuple[Client, dict[str, Any], bytes]],
        on_close: Callable[[Client], None],
        *,
        outbox_size: int = 200,
        send_timeout_s: float = 5.0,
    ) -> None:
        """``outbox_size`` messages ≈ 3 s of frames and directions."""
        self.conn = conn
        self.peer = peer
        self._inbox = inbox
        self._on_close = on_close
        self._outbox: queue.Queue[bytes] = queue.Queue(maxsize=outbox_size)
        self._send_timeout_s = send_timeout_s
        self._lock = threading.Lock()
        self.listening = False
        self.alive = True
        self.dropped = 0

    def start(self) -> Self:
        """Begin reading and writing."""
        threading.Thread(target=self._read, daemon=True, name=f"audio-{self.peer}-r").start()
        threading.Thread(target=self._write, daemon=True, name=f"audio-{self.peer}-w").start()
        return self

    def post(self, data: bytes) -> None:
        """Queue one encoded message; dropped (counted) when the outbox is full."""
        try:
            self._outbox.put_nowait(data)
        except queue.Full:
            self.dropped += 1

    def close(self) -> None:
        """Shut the socket and stop both threads; idempotent."""
        with self._lock:
            if not self.alive:
                return
            self.alive = False
        with contextlib.suppress(OSError):
            self.conn.shutdown(socket.SHUT_RDWR)
        self.conn.close()
        with contextlib.suppress(queue.Full):
            self._outbox.put_nowait(b"")  # wakes the writer
        self._on_close(self)

    def _read(self) -> None:
        decoder = FrameDecoder()
        try:
            while self.alive:
                chunk = self.conn.recv(65536)
                if not chunk:
                    break
                for header, payload in decoder.feed(chunk):
                    if header.get("cmd") == "listen":
                        self.listening = True
                    else:
                        self._inbox.put((self, header, payload))
        except ProtocolError as exc:
            logger.warning("client %s broke the framing (%s); dropping it", self.peer, exc)
        except OSError as exc:
            logger.info("client %s reader ended: %s", self.peer, exc)
        self.close()

    def _write(self) -> None:
        self.conn.settimeout(self._send_timeout_s)
        while True:
            data = self._outbox.get()
            if not data or not self.alive:
                return
            try:
                self.conn.sendall(data)
            except OSError as exc:
                logger.info("client %s writer ended: %s", self.peer, exc)
                self.close()
                return


class AudioServer:
    """Dual-stack TCP server of the audio link: a hello to each new client, frames and
    directions to the listeners, replies to one, and everyone's commands in one inbox."""

    def __init__(self, port: int, hello: Callable[[], dict[str, Any]]) -> None:
        """``port`` 0 picks a free one (tests); ``hello`` builds the greeting at each connect."""
        self._requested_port = port
        self._hello = hello
        self._server: socket.socket | None = None
        self._clients: list[Client] = []
        self._lock = threading.Lock()
        self._inbox: queue.Queue[tuple[Client, dict[str, Any], bytes]] = queue.Queue()
        self._dropped_gone = 0  # drops counted on clients that have left
        self.port = port

    def start(self) -> Self:
        """Bind, listen and accept in a daemon thread."""
        server = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        server.bind(("::", self._requested_port))
        server.listen(4)
        self._server = server
        self.port = server.getsockname()[1]
        threading.Thread(target=self._accept_loop, daemon=True, name="audio-accept").start()
        logger.info("listening on %d", self.port)
        return self

    def broadcast(self, data: bytes) -> None:
        """One encoded message to every listening client; never blocks."""
        with self._lock:
            targets = [c for c in self._clients if c.listening]
        for client in targets:
            client.post(data)

    def reply(self, client: Client, message: dict[str, Any]) -> None:
        """One message to one client."""
        if client.alive:
            client.post(encode(message))

    def commands(self, timeout_s: float) -> list[tuple[Client, dict[str, Any], bytes]]:
        """What clients sent, waiting up to ``timeout_s`` for the first message."""
        out = []
        try:
            out.append(self._inbox.get(timeout=timeout_s))
        except queue.Empty:
            return out
        while True:
            try:
                out.append(self._inbox.get_nowait())
            except queue.Empty:
                return out

    def stats(self) -> dict[str, int]:
        """Clients, listeners, and messages dropped to slow clients since the start."""
        with self._lock:
            clients = list(self._clients)
            gone = self._dropped_gone
        return {
            "clients": len(clients),
            "listeners": sum(c.listening for c in clients),
            "dropped_to_clients": gone + sum(c.dropped for c in clients),
        }

    def close(self) -> None:
        """Drop every client and stop listening."""
        with self._lock:
            clients = list(self._clients)
        for client in clients:
            client.close()
        if self._server is not None:
            with contextlib.suppress(OSError):
                self._server.close()

    def _on_close(self, client: Client) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)
                self._dropped_gone += client.dropped
        logger.info("client %s gone", client.peer)

    def _accept_loop(self) -> None:
        assert self._server is not None
        while True:
            try:
                conn, peer = self._server.accept()
            except OSError as exc:
                if self._server.fileno() == -1:
                    return
                logger.warning("accept failed (%s); still listening", exc)
                time.sleep(0.05)
                continue
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            client = Client(conn, f"{peer[0]}:{peer[1]}", self._inbox, self._on_close)
            client.post(encode(self._hello()))
            with self._lock:
                self._clients.append(client)
            logger.info("client %s connected", client.peer)
            client.start()


# -- the service -------------------------------------------------------------------------------


class AudioService:
    """The command loop and the status: ties capture, direction, speech and the socket."""

    def __init__(
        self,
        server: AudioServer,
        capture: CaptureLoop,
        player: Player,
        doa: DoaPoller | None,
        *,
        capture_xruns: XrunCounter,
        play_xruns: XrunCounter,
        clock: Callable[[], float] = time.monotonic,
        lipsync: LipSync | None = None,
    ) -> None:
        """``doa`` None: no direction (``--doa-hz 0``); the xrun counters are arecord's
        overruns (audio lost) and aplay's underruns (the speaker ran dry); ``lipsync`` the
        player's (``--lipsync``), stepped on a thread of its own."""
        self._server = server
        self._capture = capture
        self._player = player
        self._doa = doa
        self._lipsync = lipsync
        self._capture_xruns = capture_xruns
        self._play_xruns = play_xruns
        self._clock = clock
        self._started = clock()

    def handle(self, header: dict[str, Any], payload: bytes) -> dict[str, Any] | None:
        """One client command; the reply for those that have one."""
        cmd = header.get("cmd")
        if cmd == "play":
            self._player.feed(payload)
        elif cmd == "play_end":
            self._player.end()
        elif cmd == "flush":
            self._player.flush()
        elif cmd == "status":
            return self.status(self._clock())
        else:
            return {"type": "error", "error": f"unknown command {cmd!r}"}
        return None

    def status(self, now: float) -> dict[str, Any]:
        """Whether capture is alive, frames/s, dropouts, the direction's age, the playback."""
        cap = self._capture
        last = cap.last_frame_at
        doa = self._doa
        latest = doa.latest if doa is not None else None
        age = doa.age_s(now) if doa is not None else None
        xruns = self._capture_xruns.count
        return {
            "type": "status",
            "t": round(now, 6),
            "uptime_s": round(now - self._started, 1),
            "capture_alive": cap.alive(now),
            "device": cap.where,
            "capture_error": cap.error,
            "frames_per_s": round(cap.frames_per_s(now), 1),
            "frame_age_s": None if last is None else round(now - last, 3),
            "frame_latency_p95_ms": round(cap.frame_age.summary().p95_ms, 1),
            "dropouts": xruns + cap.stalls + cap.ends,
            "xruns": xruns,
            "stalls": cap.stalls,
            "ends": cap.ends,
            "opens": cap.opens,
            "array_reboots": cap.reboots,
            "doa_deg": None if latest is None else latest.deg,
            "doa_speech": None if latest is None else latest.speech,
            "doa_age_s": None if age is None else round(age, 3),
            "doa_error": "direction off (--doa-hz 0)" if doa is None else doa.error,
            "doa_read_p95_ms": None if doa is None else round(doa.latency.summary().p95_ms, 1),
            "firmware": None if doa is None else doa.firmware,
            "playing": self._player.playing,
            "play_queued_s": round(self._player.queued_s, 2),
            "played_s": round(self._player.played_s, 1),
            "underruns": self._play_xruns.count,
            "play_overflows": self._player.overflows,
            "play_errors": self._player.errors,
            "lipsync": self._lipsync is not None,
            **(self._lipsync.status() if self._lipsync is not None else {}),
            **self._server.stats(),
        }

    def report(self, now: float) -> str:
        """The once-a-minute journal line."""
        s = self.status(now)
        doa = "off" if s["doa_deg"] is None else f"{s['doa_deg']} deg {s['doa_age_s']:.1f} s old"
        return (
            f"audio: alive={s['capture_alive']} {s['frames_per_s']:.1f} fr/s, dropouts "
            f"{s['dropouts']} (xruns {s['xruns']}, stalls {s['stalls']}, ends {s['ends']}), "
            f"opens {s['opens']}, doa {doa}, clients {s['clients']}, played {s['played_s']} s"
        )

    def run(self, stop: threading.Event, report_every_s: float = 60.0) -> None:
        """Start the worker threads and answer commands until ``stop``; then release it all."""
        workers = [
            threading.Thread(target=self._capture.run, args=(stop,), name="capture", daemon=True),
            threading.Thread(target=self._player.run, args=(stop,), name="player", daemon=True),
        ]
        if self._doa is not None:
            workers.append(
                threading.Thread(target=self._doa.run, args=(stop,), name="doa", daemon=True)
            )
        if self._lipsync is not None:
            lips = threading.Thread(target=self._lipsync.run, args=(stop,), daemon=True)
            workers.append(lips)
        for worker in workers:
            worker.start()
        next_report = self._clock() + report_every_s
        try:
            while not stop.is_set():
                for client, header, payload in self._server.commands(0.2):
                    try:
                        reply = self.handle(header, payload)
                    except Exception:  # one malformed command must not take the audio down
                        logger.exception("bad command %r from %s; ignored", header, client.peer)
                        continue
                    if reply is not None:
                        self._server.reply(client, reply)
                now = self._clock()
                if now >= next_report:
                    next_report = now + report_every_s
                    logger.info("%s", self.report(now))
        finally:
            stop.set()
            for worker in workers:
                worker.join(timeout=3.0)
            self._server.close()


def stop_on_sigterm() -> threading.Event:
    """An event SIGTERM sets: systemd's stop then kills arecord and aplay on the way out."""
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    return stop


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve the reSpeaker XVF3800 array over TCP: voice, direction, speaker."
    )
    parser.add_argument("--port", type=int, default=AUDIO_PORT)
    parser.add_argument("--device", default="auto", help="ALSA device; auto = the XVF3800 by name")
    parser.add_argument(
        "--channel", type=int, default=0,
        help="capture channel sent: 0 processed (AEC, beams, noise suppression, AGC), 1 ASR beam",
    )  # fmt: skip
    parser.add_argument(
        "--capture-channels", type=int, default=2,
        help="channels the firmware captures: 2 (default image), 6 (the _16k6ch image)",
    )  # fmt: skip
    parser.add_argument("--rate", type=int, default=RATE, help="capture rate, Hz")
    parser.add_argument("--play-rate", type=int, default=RATE, help="playback rate, Hz")
    parser.add_argument(
        "--doa-hz", type=float, default=10.0, help="direction reads a second; 0 = off"
    )
    parser.add_argument("--deadman-s", type=float, default=DEADMAN_S)
    parser.add_argument(
        "--lipsync", action="store_true",
        help="send the speech's loudness to the head server's mouth (pepin.head_server, :3340)",
    )  # fmt: skip
    parser.add_argument("--head-port", type=int, default=3340)
    parser.add_argument(
        "--xvf-host", default="/usr/local/bin/xvf_host",
        help="Seeed's control tool, for the chip's reboot when capture streams are dead",
    )  # fmt: skip
    parser.add_argument(
        "--no-array-reboot", action="store_true",
        help="never reboot the array's chip (by default: once, after 2 streams without a byte)",
    )  # fmt: skip
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname).1s %(name)s: %(message)s"
    )

    stop = stop_on_sigterm()
    capture_xruns, play_xruns = XrunCounter(), XrunCounter()
    capture_ref: list[CaptureLoop] = []
    doa_ref: list[DoaPoller] = []

    def hello() -> dict[str, Any]:
        """The greeting: what the frames are and where they come from."""
        capture = capture_ref[0] if capture_ref else None
        return {
            "type": "hello",
            "rate": args.rate,
            "format": "s16le",
            "channels": 1,
            "frame_samples": FRAME_SAMPLES,
            "channel": args.channel,
            "play_rate": args.play_rate,
            "device": capture.where if capture else "",
            "firmware": doa_ref[0].firmware if doa_ref else None,
        }

    server = AudioServer(args.port, hello)
    xvf = XvfHost(args.xvf_host)
    capture = CaptureLoop(
        ArecordSource(
            args.device, rate=args.rate, channels=args.capture_channels, xruns=capture_xruns
        ),
        lambda frame: server.broadcast(frame.encode()),
        channels=args.capture_channels,
        channel=args.channel,
        rate=args.rate,
        deadman_s=args.deadman_s,
        rescue=None if args.no_array_reboot else lambda: reboot_array(xvf),
    )
    capture_ref.append(capture)
    doa = None
    if args.doa_hz > 0:
        doa = DoaPoller(
            open_array, lambda reading: server.broadcast(reading.encode()), hz=args.doa_hz
        )
        doa_ref.append(doa)
    lipsync = None
    if args.lipsync:
        from pepin.head_link import HeadClient

        head = HeadClient("127.0.0.1", args.head_port, source="audio").start()
        lipsync = LipSync(head.mouth, load_face_table(), rate=args.play_rate)
        logger.info("lip sync: the speech's loudness to the head server on :%d", args.head_port)
    player = Player(
        lambda: AplaySink(args.device, rate=args.play_rate, xruns=play_xruns),
        rate=args.play_rate,
        lipsync=lipsync,
    )
    service = AudioService(
        server.start(),
        capture,
        player,
        doa,
        capture_xruns=capture_xruns,
        play_xruns=play_xruns,
        lipsync=lipsync,
    )
    service.run(stop)


if __name__ == "__main__":
    main()
