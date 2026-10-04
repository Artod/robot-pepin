"""Audio plumbing of the Live loop: the model's 24 kHz speech brought to the board's play rate,
paced to the speaker as it streams, its amplitude announced as it plays, and WAV logs written
off the latency path."""

from __future__ import annotations

import contextlib
import logging
import math
import queue
import threading
import time
import wave
from collections import deque
from collections.abc import Callable
from pathlib import Path

import numpy as np

from pepin.audio_link import SAMPLE_BYTES, Speaker

logger = logging.getLogger(__name__)

LIVE_OUT_RATE = 24_000  # Hz: what the Live API speaks
LIVE_IN_RATE = 16_000  # Hz: what it hears (the array's own rate)
STALL_S = 0.15  # a partial block waits this long for more audio before it is sent as it is


class Resampler:
    """Streaming rational resampler for s16le mono (24 kHz -> 16 kHz: up 2, down 3): zero-stuff,
    a Kaiser-windowed sinc low-pass below both Nyquists, keep every ``down``-th sample. The
    filter's history crosses chunk borders, so a stream cut anywhere comes out unbroken."""

    def __init__(self, in_rate: int, out_rate: int, taps_per_phase: int = 24) -> None:
        """Rates in Hz; ``taps_per_phase`` sets the filter's length (and its sharpness)."""
        g = math.gcd(in_rate, out_rate)
        self.up, self.down = out_rate // g, in_rate // g
        self.identity = self.up == self.down == 1
        n = taps_per_phase * max(self.up, self.down) | 1
        cutoff = 0.45 / max(self.up, self.down)  # of the upsampled rate: 0.9 of the lower Nyquist
        k = np.arange(n) - (n - 1) / 2
        taps = 2 * cutoff * np.sinc(2 * cutoff * k) * np.kaiser(n, 8.0)
        self._taps = (taps / taps.sum() * self.up).astype(np.float64)
        self._history: np.ndarray[tuple[int, ...], np.dtype[np.float64]] = np.zeros(n - 1)
        self._base = -(n - 1)  # upsampled index of the history's first sample

    def process(self, pcm: bytes) -> bytes:
        """The resampled samples ``pcm`` completes (whole samples only)."""
        if self.identity or not pcm:
            return pcm
        x = np.frombuffer(pcm[: len(pcm) - len(pcm) % SAMPLE_BYTES], dtype="<i2").astype(float)
        stuffed = np.zeros(len(x) * self.up)
        stuffed[:: self.up] = x
        signal = np.concatenate([self._history, stuffed])
        n = len(self._taps)
        filtered = np.convolve(signal, self._taps, mode="valid")  # index i -> upsampled base+n-1+i
        first = self._base + n - 1
        offset = (-first) % self.down
        out = filtered[offset :: self.down]
        self._history = signal[-(n - 1) :]
        self._base += len(signal) - (n - 1)
        return np.clip(np.round(out), -32768, 32767).astype("<i2").tobytes()


def level(pcm: bytes) -> float:
    """The loudness of s16le PCM as 0..1 (-60 dBFS and below is 0, full scale is 1)."""
    if len(pcm) < SAMPLE_BYTES:
        return 0.0
    x = np.frombuffer(pcm[: len(pcm) - len(pcm) % SAMPLE_BYTES], dtype="<i2").astype(float)
    rms = math.sqrt(float(np.mean(x * x))) / 32768
    return max(0.0, min(1.0, (20 * math.log10(rms + 1e-9) + 60) / 60))


class Pacer:
    """The model's speech to the speaker at real time, ``lead_s`` ahead of it, in ``block_s``
    blocks, as it streams in; each block's amplitude is announced when the block plays.

    The lead is the jitter buffer over the board's WiFi; it also bounds what a barge-in
    (:meth:`flush`) has to silence. :meth:`end` marks the turn complete: what is queued plays,
    then the board is told the utterance ended. Thread-safe: the receive loop feeds, the
    pacer's own thread (:meth:`run`) or a test (:meth:`step`) sends.
    """

    def __init__(
        self,
        speaker: Speaker,
        *,
        rate: int = LIVE_IN_RATE,
        lead_s: float = 0.5,
        block_s: float = 0.04,
        tail_s: float = 0.3,
        on_level: Callable[[float], None] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``rate``: the speaker's play rate; ``tail_s``: the room's echo after the last block,
        counted as still speaking (the mic stays shut through it)."""
        self.speaker = speaker
        self.rate = rate
        self.lead_s, self.tail_s = lead_s, tail_s
        self._block = max(SAMPLE_BYTES, int(rate * block_s) * SAMPLE_BYTES)
        self._on_level = on_level
        self._clock = clock
        self._cond = threading.Condition()
        self._buffer = bytearray()
        self._levels: deque[tuple[float, float]] = deque()
        self._cursor = 0.0  # when the next block sent will play
        self._ending = False
        self._last_feed = 0.0
        self.sent_s = 0.0
        self.first_play_s: float | None = None  # when the first block of the session was sent

    def feed(self, pcm: bytes) -> None:
        """Queue speech at the speaker's rate."""
        with self._cond:
            self._buffer += pcm
            self._ending = False
            self._last_feed = self._clock()
            self._cond.notify_all()

    def end(self) -> None:
        """The turn is complete: play out what is queued, then end the utterance."""
        with self._cond:
            self._ending = True
            self._cond.notify_all()

    def flush(self) -> None:
        """Barge-in: drop what is queued and silence the speaker now."""
        with self._cond:
            self._buffer.clear()
            self._levels.clear()
            self._ending = False
            self._cursor = min(self._cursor, self._clock())
        self._safe(self.speaker.flush)

    def speaking(self) -> bool:
        """Speech is queued, or still playing (with the room's tail)."""
        with self._cond:
            return bool(self._buffer) or self._clock() < self._cursor + self.tail_s

    def step(self) -> float:
        """Send what is due and announce what plays now; the seconds until the next duty."""
        now = self._clock()
        due_levels: list[float] = []
        blocks: list[bytes] = []
        finish = False
        with self._cond:
            stalled = now - self._last_feed > STALL_S  # a remainder no more audio is joining
            while self._buffer and (len(self._buffer) >= self._block or self._ending or stalled):
                cursor = max(self._cursor, now)
                if cursor - now > self.lead_s:
                    break
                block = bytes(self._buffer[: self._block])
                del self._buffer[: self._block]
                self._levels.append((cursor, level(block)))
                duration = len(block) / SAMPLE_BYTES / self.rate
                self._cursor = cursor + duration
                self.sent_s += duration
                blocks.append(block)
            if self._ending and not self._buffer:
                self._ending, finish = False, True
            while self._levels and self._levels[0][0] <= now:
                due_levels.append(self._levels.popleft()[1])
            next_level = self._levels[0][0] if self._levels else math.inf
            next_send = self._cursor - self.lead_s if self._buffer else math.inf
        if blocks and self.first_play_s is None:
            self.first_play_s = now
        for block in blocks:
            self._play(block)
        if finish:
            self._safe(self.speaker.play_end)
        if self._on_level is not None:
            for value in due_levels:
                self._on_level(value)
        return max(0.002, min(0.02, next_level - now, next_send - now))

    def run(self, stop: threading.Event) -> None:
        """Step until ``stop`` is set; sleeps on the queue when there is nothing to do."""
        while not stop.is_set():
            wait = self.step()
            with self._cond:
                if not self._buffer and not self._levels:
                    self._cond.wait(0.1)
                    continue
            time.sleep(wait)

    def start(self) -> threading.Event:
        """Run on a daemon thread; set the returned event to stop it."""
        stop = threading.Event()
        threading.Thread(target=self.run, args=(stop,), name="pacer", daemon=True).start()
        return stop

    def _play(self, block: bytes) -> None:
        try:
            self.speaker.play(block)
        except (OSError, ConnectionError) as error:
            logger.warning("speaker: %s", error)

    def _safe(self, send: Callable[[], None]) -> None:
        try:
            send()
        except (OSError, ConnectionError) as error:
            logger.warning("speaker: %s", error)


class WavWriter:
    """A mono s16le WAV written on its own thread as chunks arrive; :meth:`close` finishes it."""

    def __init__(self, path: Path, rate: int) -> None:
        """Create ``path`` (and its directory)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.rate = rate
        self.bytes = 0
        self._queue: queue.Queue[bytes | None] = queue.Queue()
        self._thread = threading.Thread(target=self._write, name="wav", daemon=True)
        self._thread.start()

    @property
    def seconds(self) -> float:
        """Audio written (or queued) so far."""
        return self.bytes / SAMPLE_BYTES / self.rate

    def write(self, pcm: bytes) -> None:
        """Queue a chunk; never blocks."""
        if pcm:
            self.bytes += len(pcm)
            self._queue.put(pcm)

    def close(self) -> None:
        """Write what is queued and close the file."""
        self._queue.put(None)
        self._thread.join(timeout=5.0)

    def _write(self) -> None:
        with contextlib.closing(wave.open(str(self.path), "wb")) as out:
            out.setnchannels(1)
            out.setsampwidth(SAMPLE_BYTES)
            out.setframerate(self.rate)
            while (chunk := self._queue.get()) is not None:
                out.writeframesraw(chunk)


def save_wav(path: Path, pcm: bytes, rate: int) -> None:
    """One whole WAV (a wake segment), written by the caller's thread."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.closing(wave.open(str(path), "wb")) as out:
        out.setnchannels(1)
        out.setsampwidth(SAMPLE_BYTES)
        out.setframerate(rate)
        out.writeframes(pcm)
