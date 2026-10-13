"""The prosody listener: the syllable nuclei and the melody of the robot's own voice, found in
its audio as the player takes it, before the speaker sounds it.

A sensor inside the gesture engine (gesticulation DESIGN.md §6.1): the voice's every reply chunk
reaches :meth:`ProsodyListener.audio` on the voice's thread (one queue append, nothing computed
there); :meth:`ProsodyListener.analyse`, on the thread of whoever drives it (the engine's
listeners, :mod:`pepin.expression.listeners`), turns them into, per part of a reply:

* tracks every 10 ms: the 300-2500 Hz band energy (40 ms Hann frames, the band shaped like a
  4th-order Butterworth run forward and back), the full-band RMS and Praat's autocorrelation F0
  (75-500 Hz, parselmouth);
* syllable NUCLEI (de Jong & Wempe 2009 on the band energy): voiced local peaks, each at most
  ``REL_DB`` under the loudest frame so far, after a dip of ``DIP_DB`` and ``GAP_S`` after the
  one before; a nucleus is told once nothing later can replace it;
* each nucleus's prominence, the z-sum of its band level, its voiced length (the voiced run
  within ``LENGTH_DB`` of its peak: Russian stress is length, not pitch) and its pitch peak
  over the voice's reference, each normalised by the VOICE's own statistics
  (:class:`VoiceStats`, fitted on its recordings), so a nucleus is judged the moment it is found;
* the melody: the F0 contour in semitones round the part's median (unvoiced steps hold the last
  voiced value), low-passed at ``cutoff_hz`` forward and back, and its turns, told once the
  filter's margin is in;
* at the part's end, its voiced span and the pauses inside it.

Everything is keyed by (turn, part), never by stream second alone: a barge-in flushes the
player and the next reply reuses the flushed seconds. The nuclei are the word clock's
(:class:`pepin.expression.schedule.Nuclei`); :func:`pin_second` (a character's second, the
text's phrases pinned to the pauses heard) and :func:`snap` (to the nearest nucleus within
``SNAP_S``) are the alignment it applies; this module keeps no clock of its own.

Measured on the robot's voice (scratch/gesticulation_1012/pulse_signal.py, 423 sentences of
Qwen3-TTS): a nucleus with z >= 2 is on a stressed vowel of a content word 72 % of the time and
within 0.15 s of one 90 % (any nucleus: 44 / 59 %); pinned and snapped, a stressed vowel's time
is within 0.1 s 86 % of the time; 2.0-2.3 ms of one core per second of audio.
"""

from __future__ import annotations

import itertools
import logging
import math
import re
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from pepin.expression.conversation import VoiceChunk
from pepin.expression.schedule import Nucleus

if TYPE_CHECKING:
    from numpy.typing import NDArray

logger = logging.getLogger(__name__)

HOP_S = 0.01  # the tracks' step
WIN_S = 0.04  # the band and RMS frames
BAND_HZ = (300.0, 2500.0)  # vowels' first two formants, not the fricatives
BAND_ORDER = 4  # the Butterworth the band is shaped like (run forward and back: its square)
F0_FLOOR_HZ, F0_CEILING_HZ = 75.0, 500.0
DIP_DB, REL_DB, GAP_S = 2.0, 20.0, 0.08  # the nucleus rule (tuned on 10-10/11, pulse_signal.py)
VOICED_FRAMES = 2  # a nucleus is voiced within this many frames each side
LENGTH_DB = 10.0  # a nucleus's voiced length: its run within this of its peak
PITCH_FRAMES = 6  # a nucleus's pitch peak: the voiced F0 within this many frames each side
UNVOICED_ST = -5.0  # the pitch a nucleus without a voiced frame round it counts with
PITCH_CONTEXT_S = 0.1  # audio before the first new frame that Praat reads again
PITCH_MARGIN_S = 0.05  # frames this close to the audio's end wait for more (Praat's window)
CONTOUR_MARGIN_S = 0.3  # the contour's turns this close to the audio's end wait (the filter)
LOUD_DB = 35.0  # the voiced span: frames within this of the part's loud level (RMS p95)
PAUSE_DB = 30.0  # a pause: RMS this far under the loud level
PAUSE_S = 0.12  # for at least this long
SNAP_S = 0.25  # a predicted vowel snaps to the nearest nucleus within this
INNER = re.compile(r"[,;:\u2014\u2013]|\s-\s|[.!?\u2026]+\s")  # a phrase's end inside a sentence
VOWEL = re.compile("[аеёиоуыэюяaeiouy]", re.IGNORECASE)  # noqa: RUF001 -- Cyrillic vowels


@dataclass(frozen=True)
class VoiceStats:
    """A voice's nuclei as fitted on its recordings: the band level, the voiced length and the
    pitch peak over ``f0_ref_hz`` (the voice's median), each a (mean, standard deviation)."""

    band_db: tuple[float, float] = (-29.1, 5.7)
    length_s: tuple[float, float] = (0.145, 0.113)
    f0_st: tuple[float, float] = (1.8, 3.3)
    f0_ref_hz: float = 251.0

    def z(self, band_db: float, length_s: float, f0_st: float) -> float:
        """A nucleus's prominence: the sum of its three z-scores."""
        return (
            (band_db - self.band_db[0]) / self.band_db[1]
            + (length_s - self.length_s[0]) / self.length_s[1]
            + (f0_st - self.f0_st[0]) / self.f0_st[1]
        )

    @classmethod
    def of(cls, data: Mapping[str, Any]) -> VoiceStats:
        """config/expression.json's ``pulse.voice_stats`` (defaults for what it lacks)."""
        d = cls()

        def pair(key: str, default: tuple[float, float]) -> tuple[float, float]:
            value = data.get(key, default)
            return float(value[0]), float(value[1])

        return cls(
            pair("band_db", d.band_db),
            pair("length_s", d.length_s),
            pair("f0_st", d.f0_st),
            float(data.get("f0_ref_hz", d.f0_ref_hz)),
        )


@dataclass(frozen=True)
class Turn:
    """A turn of the part's melody: the stream second and the contour there (semitones round
    the part's median, low-passed)."""

    second: float
    st: float


@dataclass(frozen=True)
class Span:
    """A part's voice once its audio is all in: the first and last loud frame (stream seconds)
    and the pauses between (start, end)."""

    voiced: tuple[float, float]
    pauses: tuple[tuple[float, float], ...] = ()


@dataclass(frozen=True)
class PartNews:
    """What one analysis found in part ``index`` of reply ``turn`` (its first stream second
    ``start_s``): its new nuclei and melody turns, where its voice began (once known), where its
    audio ends so far, and, once complete, its :class:`Span`."""

    turn: int
    index: int
    start_s: float = 0.0
    nuclei: tuple[Nucleus, ...] = ()
    turns: tuple[Turn, ...] = ()
    onset_s: float | None = None
    end_s: float = 0.0
    complete: bool = False
    span: Span | None = None


def band_weights(rate: int, frame: int) -> NDArray[np.float64]:
    """Per rfft bin of a ``frame``-sample frame: the band's power gain (a Butterworth band-pass
    of :data:`BAND_ORDER` on :data:`BAND_HZ`, prewarped as the bilinear transform does, run
    forward and back) times the one-sided spectrum's factor over ``frame``^2, so the weighted
    sum of a windowed frame's |X|^2 is the mean square of that frame band-passed."""
    f = np.fft.rfftfreq(frame, 1.0 / rate)
    warp = 2.0 * rate * np.tan(np.pi * np.minimum(f, rate / 2.0 - 1e-6) / rate)
    lo, hi = (2.0 * rate * math.tan(math.pi * edge / rate) for edge in BAND_HZ)
    w0, bw = math.sqrt(lo * hi), hi - lo
    with np.errstate(divide="ignore", invalid="ignore"):
        x = (warp * warp - w0 * w0) / (warp * bw)
    gain = np.where(warp > 0, 1.0 / (1.0 + x ** (2 * BAND_ORDER)), 0.0)
    side = np.full(f.shape, 2.0)
    side[0] = 1.0
    if frame % 2 == 0:
        side[-1] = 1.0
    out: NDArray[np.float64] = (gain * gain * side / float(frame * frame)).astype(np.float64)
    return out


def praat_f0(x: NDArray[np.float64], rate: int) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Praat's autocorrelation pitch of ``x`` every :data:`HOP_S`: frame times (seconds from
    ``x``'s start) and F0 (Hz, 0 unvoiced)."""
    import parselmouth  # the voice group's; read lazily

    pitch = parselmouth.Sound(x, sampling_frequency=rate).to_pitch_ac(
        time_step=HOP_S, pitch_floor=F0_FLOOR_HZ, pitch_ceiling=F0_CEILING_HZ
    )
    times = np.asarray(pitch.xs(), dtype=np.float64)
    f0 = np.asarray(pitch.selected_array["frequency"], dtype=np.float64)
    return times, f0


def lowpass(x: NDArray[np.float64], cutoff_hz: float, fs: float) -> NDArray[np.float64]:
    """A 2nd-order Butterworth low-pass run forward and back (zero phase), the ends padded by
    their odd reflection and started settled, as scipy's ``filtfilt``."""
    n = len(x)
    if n < 2 or cutoff_hz <= 0.0 or cutoff_hz >= fs / 2.0:
        return x.copy()
    k = math.tan(math.pi * cutoff_hz / fs)
    norm = 1.0 / (1.0 + math.sqrt(2.0) * k + k * k)
    b0 = k * k * norm
    b1, b2 = 2.0 * b0, b0
    a1 = 2.0 * (k * k - 1.0) * norm
    a2 = (1.0 - math.sqrt(2.0) * k + k * k) * norm
    pad = min(n - 1, 9)  # scipy's padlen for a biquad
    head = 2.0 * x[0] - x[pad:0:-1]
    tail = 2.0 * x[-1] - x[-2 : -pad - 2 : -1]
    y = np.concatenate([head, x, tail]).tolist()

    def run(values: list[float]) -> list[float]:
        c = values[0]
        z1, z2 = (1.0 - b0) * c, (b2 - a2) * c
        out = []
        for v in values:
            o = b0 * v + z1
            z1 = b1 * v - a1 * o + z2
            z2 = b2 * v - a2 * o
            out.append(o)
        return out

    y = run(run(y)[::-1])[::-1]
    return np.asarray(y[pad : pad + n], dtype=np.float64)


class _Part:
    """One reply part under analysis: its audio, its tracks so far and its nuclei."""

    def __init__(self, turn: int, index: int, start_s: float, rate: int) -> None:
        self.turn, self.index, self.start_s, self.rate = turn, index, start_s, rate
        self.hop = round(HOP_S * rate)
        self.win = round(WIN_S * rate)
        self.chunks: list[NDArray[np.float64]] = []
        self.pcm: NDArray[np.float64] = np.zeros(0)
        self.band: NDArray[np.float64] = np.zeros(0)
        self.rms: NDArray[np.float64] = np.zeros(0)
        self.f0: NDArray[np.float64] = np.zeros(0)
        self.top = -math.inf  # the loudest band frame so far
        self.tested = 1  # frames below this were tested for a peak
        self.accepted: list[int] = []  # nuclei found (the last one may still be replaced)
        self.told = 0  # accepted nuclei told
        self.turns_told = -1  # the contour's frames up to this were searched for turns
        self.onset: int | None = None  # the first voiced frame
        self.end_s = start_s
        self.final = False  # all its audio is in
        self.complete = False  # and analysed

    def frame_s(self, i: int) -> float:
        """Frame ``i``'s centre, seconds from the part's first sample."""
        return (i * self.hop + self.win / 2.0) / self.rate


class ProsodyListener:
    """The voice's tap (:class:`pepin.expression.conversation.AudioTap`) and its nuclei
    (:class:`pepin.expression.schedule.Nuclei`): see the module's notes."""

    def __init__(
        self,
        *,
        stats: VoiceStats | None = None,
        cutoff_hz: float = 1.0,
        wake: Callable[[], None] | None = None,
        listening: Callable[[], bool] | None = None,
        pitch: Callable[[NDArray[np.float64], int], tuple[Any, Any]] | None = None,
    ) -> None:
        """``stats``: the voice's (:class:`VoiceStats`); ``cutoff_hz``: the melody's low-pass;
        ``wake()``: called after each chunk is queued (the analysing thread's bell; it must not
        block); ``listening()``: False drops chunks unqueued (the gestures off); ``pitch``:
        the F0 tracker (:func:`praat_f0`)."""
        self.stats = stats or VoiceStats()
        self.cutoff_hz = cutoff_hz
        self.wake = wake
        self.listening = listening or (lambda: True)
        self.pitch = pitch or praat_f0
        self._queue: deque[VoiceChunk] = deque()
        self._parts: dict[tuple[int, int], _Part] = {}
        self._found: dict[tuple[int, int], tuple[Nucleus, ...]] = {}
        self._spans: dict[tuple[int, int], Span] = {}
        self._done: set[tuple[int, int]] = set()
        self._finished: set[tuple[int, int]] = set()  # all their audio is in
        self._cut = -1  # replies up to this were cut: their late chunks are not taken
        self._lock = threading.Lock()  # the nuclei as the word clock reads them
        self._weights: dict[tuple[int, int], NDArray[np.float64]] = {}
        self.audio_s = 0.0  # seconds of audio analysed
        self.cpu_s = 0.0  # this thread's CPU on it

    def configure(self, stats: VoiceStats, cutoff_hz: float) -> None:
        """The knobs (config/expression.json ``pulse``), for parts that start from now."""
        self.stats, self.cutoff_hz = stats, cutoff_hz

    # -- the voice's thread ------------------------------------------------------------------

    def audio(self, chunk: VoiceChunk) -> None:
        """A chunk of a reply: queued for the analysis (one append; never blocks, never
        raises into the voice)."""
        if not self.listening():
            return
        self._queue.append(chunk)
        if self.wake is not None:
            self.wake()

    # -- the word clock's reads (any thread) -----------------------------------------------------

    def nuclei(self, turn: int, index: int) -> tuple[Nucleus, ...]:
        """The part's nuclei found so far, in order."""
        with self._lock:
            return self._found.get((turn, index), ())

    def complete(self, turn: int, index: int) -> bool:
        """Whether the part's whole audio has been analysed."""
        with self._lock:
            return (turn, index) in self._done

    def span(self, turn: int, index: int) -> Span | None:
        """The part's voiced span and pauses, once complete (None before)."""
        with self._lock:
            return self._spans.get((turn, index))

    # -- the analysing thread ----------------------------------------------------------------

    def pending(self) -> bool:
        """Whether chunks wait for :meth:`analyse`."""
        return bool(self._queue)

    def finish(self, turn: int, index: int) -> None:
        """All of the part's audio is in (the voice's ``part_spoken``): the next analysis
        completes it."""
        self._finished.add((turn, index))

    def drop(self, turn: int, *, only: bool = False) -> None:
        """The parts of replies up to ``turn`` (``only``: that one) forgotten, their nuclei
        with them; a cut reply's chunks still on their way are not taken."""
        if not only:
            self._cut = max(self._cut, turn)

        def gone(key: tuple[int, int]) -> bool:
            return key[0] == turn if only else key[0] <= turn

        with self._lock:
            for store in (self._found, self._spans):
                for key in [k for k in store if gone(k)]:
                    del store[key]
            self._done = {k for k in self._done if not gone(k)}
        for key in [k for k in self._parts if gone(k)]:
            del self._parts[key]

    def clear(self) -> None:
        """Everything forgotten (the conversation's close); chunks queued are dropped."""
        self._queue.clear()
        self._parts.clear()
        self._finished.clear()
        with self._lock:
            self._found.clear()
            self._spans.clear()
            self._done.clear()

    def analyse(self) -> list[PartNews]:
        """The chunks queued since the last call (and the parts finished) analysed: what each
        part gained, in the order the parts were touched."""
        cpu0 = time.thread_time()
        touched: dict[tuple[int, int], _Part] = {}
        while self._queue:
            chunk = self._queue.popleft()
            if chunk.turn <= self._cut:
                continue
            key = (chunk.turn, chunk.index)
            part = self._parts.get(key)
            if part is None:
                if key in self._done:
                    continue
                part = self._parts[key] = _Part(chunk.turn, chunk.index, chunk.start_s, chunk.rate)
            samples = np.frombuffer(chunk.pcm[: len(chunk.pcm) // 2 * 2], dtype="<i2")
            part.chunks.append(samples.astype(np.float64) / 32768.0)
            part.end_s = max(part.end_s, chunk.end_s)
            touched[key] = part
        for key in self._finished:
            part = self._parts.get(key)
            if part is not None:
                part.final = True
                touched[key] = part
            elif key[0] > self._cut:
                with self._lock:  # a part with no audio: nothing to find in it
                    self._done.add(key)
        self._finished.clear()
        news = [self._advance(part) for part in touched.values()]
        for part in touched.values():
            if part.complete:
                del self._parts[(part.turn, part.index)]
        self.cpu_s += time.thread_time() - cpu0
        return news

    def _advance(self, part: _Part) -> PartNews:
        if part.chunks:
            added = sum(len(c) for c in part.chunks)
            part.pcm = np.concatenate([part.pcm, *part.chunks])
            part.chunks = []
            self.audio_s += added / part.rate
        n_all = max(0, (len(part.pcm) - part.win) // part.hop + 1)
        if part.final:
            n = n_all
        else:
            margin = round(PITCH_MARGIN_S * part.rate)
            n = max(0, (len(part.pcm) - margin - part.win) // part.hop + 1)
        first = len(part.band)
        if n > first:
            self._tracks(part, first, n)
        found = self._peaks(part)
        turns = self._turns(part)
        complete = part.final
        span = None
        if complete:
            part.complete = True
            span = self._span(part)
        key = (part.turn, part.index)
        with self._lock:
            if found:
                self._found[key] = self._found.get(key, ()) + found
            if complete:
                self._done.add(key)
                if span is not None:
                    self._spans[key] = span
        onset = None if part.onset is None else part.start_s + part.frame_s(part.onset)
        return PartNews(
            part.turn, part.index, part.start_s, found, turns, onset, part.end_s, complete, span
        )

    def _tracks(self, part: _Part, first: int, n: int) -> None:
        """Frames ``first..n`` of the band, the RMS and the F0."""
        hop, win, rate = part.hop, part.win, part.rate
        x = part.pcm
        frames = np.lib.stride_tricks.sliding_window_view(x[: (n - 1) * hop + win], win)[::hop]
        frames = frames[first:n]
        weights = self._weights.get((rate, win))
        if weights is None:
            weights = self._weights[(rate, win)] = band_weights(rate, win)
        spectrum = np.fft.rfft(frames * np.hanning(win), axis=1)
        power = np.sum((spectrum.real**2 + spectrum.imag**2) * weights, axis=1)
        band = 10.0 * np.log10(power + 1e-12)
        rms = 10.0 * np.log10(np.mean(frames * frames, axis=1) + 1e-12)
        s0 = max(0, first * hop - round(PITCH_CONTEXT_S * rate))
        try:
            times, values = self.pitch(x[s0:], rate)
        except Exception:  # Praat refuses a sound too short to hold one period: unvoiced
            logger.debug("prosody: no pitch for %d samples", len(x) - s0, exc_info=True)
            times, values = np.zeros(0), np.zeros(0)
        t = np.array([part.frame_s(i) for i in range(first, n)]) - s0 / rate
        f0 = np.zeros(n - first)
        if len(times):
            idx = np.clip(np.round((t - times[0]) / HOP_S).astype(int), 0, len(values) - 1)
            f0 = np.asarray(values, dtype=np.float64)[idx].copy()
            f0[(t < times[0] - HOP_S) | (t > times[-1] + HOP_S)] = 0.0
        part.band = np.concatenate([part.band, band])
        part.rms = np.concatenate([part.rms, rms])
        part.f0 = np.concatenate([part.f0, f0])
        if len(band):
            part.top = max(part.top, float(band.max()))
        if part.onset is None:
            voiced = np.flatnonzero(f0 > 0)
            if voiced.size:
                part.onset = first + int(voiced[0])

    def _peaks(self, part: _Part) -> tuple[Nucleus, ...]:
        """Frames newly testable searched for nuclei; the nuclei nothing can replace now told."""
        e, f0, n = part.band, part.f0, len(part.band)
        end = n - 1 if part.final else n - VOICED_FRAMES
        floor = part.top - REL_DB
        acc = part.accepted
        for i in range(max(1, part.tested), max(1, end)):
            if not (e[i] >= e[i - 1] and e[i] > e[i + 1] and e[i] >= floor):
                continue
            if not np.any(f0[max(0, i - VOICED_FRAMES) : i + VOICED_FRAMES + 1] > 0):
                continue
            if acc:
                q = acc[-1]
                low = float(e[q : i + 1].min())
                if e[i] - low < DIP_DB or part.frame_s(i) - part.frame_s(q) < GAP_S:
                    if e[i] > e[q] and e[q] - low < DIP_DB and len(acc) > part.told:
                        acc[-1] = i
                    continue
            acc.append(i)
        part.tested = max(part.tested, end)
        told: list[Nucleus] = []
        while part.told < len(acc):
            q = acc[part.told]
            last = part.told == len(acc) - 1
            if last and not part.final and not self._settled(part, q):
                break
            told.append(self._nucleus(part, q))
            part.told += 1
        return tuple(told)

    def _settled(self, part: _Part, q: int) -> bool:
        """Whether nucleus ``q`` can no longer be replaced (the band dipped ``DIP_DB`` after it)
        and its length and pitch have all their frames."""
        e, f0, n = part.band, part.f0, len(part.band)
        if q + PITCH_FRAMES >= n:
            return False
        if not np.any(e[q + 1 : n] <= e[q] - DIP_DB):
            return False
        b = q
        while b + 1 < n and f0[b + 1] > 0 and e[b + 1] > e[q] - LENGTH_DB:
            b += 1
        return b + 1 < n

    def _nucleus(self, part: _Part, q: int) -> Nucleus:
        e, f0, n = part.band, part.f0, len(part.band)
        a = q
        while a > 0 and f0[a - 1] > 0 and e[a - 1] > e[q] - LENGTH_DB:
            a -= 1
        b = q
        while b < n - 1 and f0[b + 1] > 0 and e[b + 1] > e[q] - LENGTH_DB:
            b += 1
        length = (b - a + 1) * HOP_S
        window = f0[max(0, q - PITCH_FRAMES) : q + PITCH_FRAMES + 1]
        voiced = window[window > 0]
        st = 12.0 * math.log2(float(voiced.max()) / self.stats.f0_ref_hz) if voiced.size else None
        z = self.stats.z(float(e[q]), length, UNVOICED_ST if st is None else st)
        second = part.start_s + part.frame_s(q)
        return Nucleus(part.turn, part.index, round(second, 4), round(z, 3), st)

    def _turns(self, part: _Part) -> tuple[Turn, ...]:
        """The melody's turns found since the last call: the contour of the part's F0 (the
        frames from its voice's onset), low-passed, its local extremes; those within the
        filter's margin of the audio's end wait, unless the part is all in."""
        f0, n = part.f0, len(part.f0)
        if part.onset is None or n - part.onset < 3:
            return ()
        voiced = f0 > 0
        median = float(np.median(f0[voiced])) if voiced.any() else self.stats.f0_ref_hz
        held = np.empty(n - part.onset)
        last = 0.0
        for j, value in enumerate(f0[part.onset :]):
            if value > 0:
                last = 12.0 * math.log2(value / median)
            held[j] = last
        y = lowpass(held, self.cutoff_hz, 1.0 / HOP_S)
        limit = len(y) - 1 if part.final else len(y) - 1 - round(CONTOUR_MARGIN_S / HOP_S)
        out: list[Turn] = []
        for j in range(max(1, part.turns_told + 1), max(1, limit)):
            if (y[j] - y[j - 1]) * (y[j + 1] - y[j]) < 0:
                second = part.start_s + part.frame_s(part.onset + j)
                out.append(Turn(round(second, 4), round(float(y[j]), 3)))
        part.turns_told = max(part.turns_told, limit - 1)
        return tuple(out)

    def _span(self, part: _Part) -> Span | None:
        """The voiced span and the pauses inside it (the part's loud level: its RMS p95)."""
        rms = part.rms
        if not len(rms):
            return None
        ref = float(np.percentile(rms, 95))
        loud = np.flatnonzero(rms > ref - LOUD_DB)
        on, off = part.frame_s(int(loud[0])), part.frame_s(int(loud[-1]))
        pauses: list[tuple[float, float]] = []
        start: float | None = None
        for i, level in enumerate(rms):
            t = part.frame_s(i)
            if t <= on or t >= off:
                start = None
                continue
            if level < ref - PAUSE_DB and start is None:
                start = t
            elif level >= ref - PAUSE_DB and start is not None:
                if t - start >= PAUSE_S:
                    pauses.append((part.start_s + start, part.start_s + t))
                start = None
        return Span((part.start_s + on, part.start_s + off), tuple(pauses))


def phrases(text: str) -> tuple[tuple[int, int], ...]:
    """``text`` cut at its inner punctuation where a vowel follows: (start, end) spans."""
    cuts = [m.end() for m in INNER.finditer(text) if VOWEL.search(text[m.end() :])]
    return tuple(itertools.pairwise([0, *cuts, len(text)]))


def pin_second(text: str, char: int, span: Span) -> float:
    """The stream second character ``char`` of a part's spoken ``text`` is heard at (before
    the audio lag): its share of its phrase, the phrases pinned to the pauses heard when they
    are as many as the text's inner punctuation; else its share of the voiced span."""
    on, off = span.voiced
    cut = phrases(text)
    if len(span.pauses) == len(cut) - 1:
        starts = [on] + [b for _, b in span.pauses]
        ends = [a for a, _ in span.pauses] + [off]
        for (ca, cb), ta, tb in zip(cut, starts, ends, strict=True):
            if ca <= char <= cb:
                return ta + (char - ca) / max(1, cb - ca) * (tb - ta)
    return on + char / max(1, len(text)) * (off - on)


def snap(second: float, nuclei: Sequence[Nucleus], within_s: float = SNAP_S) -> float:
    """``second`` moved to the nearest nucleus within ``within_s`` (unmoved when none is)."""
    near = [n.second for n in nuclei if abs(n.second - second) <= within_s]
    return min(near, key=lambda s: abs(s - second)) if near else second
