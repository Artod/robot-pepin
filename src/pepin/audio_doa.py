"""The voice's direction computed on the laptop from the array's four raw microphones.

The XVF3800's own direction (``DOA_VALUE``, the chip's) names the loudest source about ten times
a second and flags "speech" by its firmware's energy and spectrum thresholds: it called the
neck's whine speech 10 of 10 times, and within one utterance its readings disagreed by 30-90 deg
while the head moved (2026-10-14). This module computes the direction itself, from the four
microphones before any processing (the 6-channel firmware's capture channels 3-6, sent by the
board as :class:`pepin.audio_link.RawFrame`), only on the chunks the ears' Silero calls voice,
and as a LIST of source peaks per chunk rather than one angle: the seed of tracking several
people by voice (the register's peaks, :meth:`pepin.expression.people_map.Register.peak`).

THE METHOD (:class:`SrpPhat`): steered-response power with PHAT weighting. Each 512-sample chunk
(Silero's, 32 ms) is windowed and FFT'd per microphone; the six pairs' cross-spectra are
normalised to unit magnitude (GCC-PHAT: every frequency votes once, whatever its level, which
makes the estimate robust to the room's colouration and reverberation); for every azimuth of a
1-degree grid the pairs' phases are steered by the delays the geometry predicts for that
direction and summed over the voice band (300-3400 Hz). The sum, normalised by the number of
pairs and bins, is the response: 1.0 for a perfect plane wave, about 0 for noise. Its local
maxima are the sources; a peak's ``strength`` is its response and the peaks kept are the
strongest and those at least ``ratio`` of it (``speaker.doa.mac_peak_ratio``): on the synthetic
stream a second voice reads 0.88-0.95 of the strongest, the square's own mirror lobe 0.36 and
the alias lobes 0.2-0.36, so 0.6 parts them (a saddle prominence does not: the response dips
below zero between the main peak and its mirror). The geometry (``mics_m``, config/audio.json)
is the firmware's own square of 66 mm (the mics at (+-33, +-33) mm): the delay between the
diagonal mics is at most 0.27 ms (4.4 samples), which the frequency-domain steering resolves to
a fraction of a sample (a clean synthetic wave lands on its grid degree; one 32 ms chunk at
0 dB of independent noise per mic is off by a median 2.5 deg, p90 6, max 12 of 40; at +6 dB at
most 5; at -6 dB it breaks down, p90 20, max 112). Above c / 2d = 2.6 kHz the 66 mm pairs
alias; summing six pairs over a wide band keeps the alias lobes below the true peak.

THE SOURCE (:class:`MicDoa`): raw frames in (:meth:`MicDoa.feed`, on the link's raw pump),
readings out as a :class:`pepin.expression.speaker.DoaSource` on the board's clock, the same
contract the chip's readings keep. A chunk's speech probability is the ears' Silero series'
(the processed channel, the ears' own model, no second instance) over the chunk's span on the
laptop's clock; a chunk at ``voice_p_min`` or more gets its peaks computed, any other chunk
costs nothing. Readings are made per ``reading_s`` (0.1 s, the chip's cadence, so that the
gaze's counts of readings keep their meaning): the span's strongest peak is the reading's angle
(``deg``, -1 when no chunk of the span was voice), ``speech`` whether any chunk was, ``voice_p``
the span's highest probability, ``peaks`` every peak of its voiced chunks (merged within
``MERGE_DEG``, strongest first). The angle is in the geometry's frame (0 along +x, growing
toward +y); its calibration into the cart's frame is the speaker gaze's
(config/expression.json ``speaker.doa.mac_readings``, measured by
``ros/tools/doa_calibrate.py --source mac``).

Laptop only (numpy); the board never imports this module.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from pepin.audio_link import RATE, SAMPLE_BYTES, DoaReading, RawFrame
from pepin.telemetry import LatencyTracker

logger = logging.getLogger(__name__)

SPEED_OF_SOUND_M_S = 343.0
CHUNK = 512  # Silero's chunk (32 ms at 16 kHz): the unit a direction is computed on
BAND_HZ = (300.0, 3400.0)  # the voice band the response sums over
MERGE_DEG = 10.0  # peaks of one span's chunks closer than this are one source
AUDIO_CONFIG = "audio.json"
Floats = npt.NDArray[np.float64]
Complexes = npt.NDArray[np.complex64]


class SpeechPeak(Protocol):
    """The ears' speech probabilities on the laptop's clock
    (:class:`pepin.voice_cascade.barge_in.SpeechSeries`)."""

    def peak(self, t0: float, t1: float) -> float | None:
        """The highest probability of the chunks within ``t0..t1``; None when none is known."""
        ...


@dataclass(frozen=True)
class ArrayGeometry:
    """The microphones' positions (x, y in metres, the geometry's own frame) in the order of
    the raw channels, and the speed of sound."""

    mics_m: tuple[tuple[float, float], ...]
    speed_m_s: float = SPEED_OF_SOUND_M_S

    @property
    def count(self) -> int:
        """Microphones."""
        return len(self.mics_m)

    @classmethod
    def from_config(cls, data: Mapping[str, Any]) -> ArrayGeometry:
        """From config/audio.json's ``array`` block; ValueError for fewer than two mics."""
        mics = tuple((float(x), float(y)) for x, y in data["mics_m"])
        if len(mics) < 2:
            raise ValueError(f"array.mics_m: {len(mics)} microphone(s), need at least two")
        return cls(mics, float(data.get("speed_of_sound_m_s", SPEED_OF_SOUND_M_S)))

    @classmethod
    def load(cls, path: str | Path | None = None) -> ArrayGeometry:
        """config/audio.json's ``array`` block (or ``path``'s)."""
        return cls.from_config(_audio_config(path)["array"])


@dataclass(frozen=True)
class Peak:
    """One source in a chunk's response: its azimuth (degrees, the geometry's frame) and its
    response (0..1, 1 a perfect plane wave)."""

    deg: float
    strength: float


def _audio_config(path: str | Path | None) -> dict[str, Any]:
    from pepin.overlay import read_json

    if path is None:
        from pepin.deployment import config_file

        path = config_file(AUDIO_CONFIG)
    data: dict[str, Any] = read_json(path)
    return data


class SrpPhat:
    """Steered-response power with PHAT weighting over a grid of azimuths (the module's notes):
    the steering matrix is built once, a chunk costs one FFT per mic and one matrix product."""

    def __init__(
        self,
        geometry: ArrayGeometry,
        *,
        rate: int = RATE,
        chunk: int = CHUNK,
        nfft: int = 1024,
        band_hz: tuple[float, float] = BAND_HZ,
        grid_deg: float = 1.0,
    ) -> None:
        """``nfft`` at least ``chunk``; ``band_hz`` the bins summed; ``grid_deg`` the step."""
        if nfft < chunk:
            raise ValueError(f"nfft {nfft} shorter than the chunk {chunk}")
        self.geometry, self.rate, self.chunk, self.nfft = geometry, rate, chunk, nfft
        self.angles_deg: Floats = np.arange(0.0, 360.0, grid_deg).astype(np.float64)
        freqs = np.fft.rfftfreq(nfft, 1.0 / rate)
        self._bins = np.flatnonzero((freqs >= band_hz[0]) & (freqs <= band_hz[1]))
        if self._bins.size == 0:
            raise ValueError(f"no FFT bin in the band {band_hz} Hz")
        mics = np.asarray(geometry.mics_m, dtype=np.float64)
        n = len(mics)
        self.pairs: list[tuple[int, int]] = [(i, j) for i in range(n) for j in range(i + 1, n)]
        baselines = np.array([mics[i] - mics[j] for i, j in self.pairs])  # (P, 2)
        radians = np.radians(self.angles_deg)
        toward = np.stack([np.cos(radians), np.sin(radians)], axis=1)  # (A, 2): unit vectors
        # A plane wave from azimuth a reaches mic i (r_i . u) / c earlier than the origin: the
        # cross-spectrum of i and j carries exp(+j 2 pi f (r_i - r_j) . u / c), undone here.
        delays = baselines @ toward.T / geometry.speed_m_s  # (P, A) seconds
        phase = -2.0 * np.pi * delays[:, :, None] * freqs[self._bins][None, None, :]  # (P, A, B)
        steering = np.exp(1j * phase).transpose(1, 0, 2).reshape(len(self.angles_deg), -1)
        # Re(S g) = Re(S) Re(g) - Im(S) Im(g): two real products (the complex BLAS path on
        # macOS Accelerate raises spurious floating-point flags that numpy would warn about).
        self._steering_re: Floats = np.ascontiguousarray(steering.real)
        self._steering_im: Floats = np.ascontiguousarray(steering.imag)
        self._window: Floats = np.hanning(chunk).astype(np.float64)
        self._norm = float(len(self.pairs) * self._bins.size)

    @classmethod
    def from_config(cls, geometry: ArrayGeometry, doa: Mapping[str, Any]) -> SrpPhat:
        """From config/audio.json's ``doa`` block."""
        band = doa.get("band_hz", BAND_HZ)
        return cls(
            geometry,
            nfft=int(doa.get("nfft", 1024)),
            band_hz=(float(band[0]), float(band[1])),
            grid_deg=float(doa.get("grid_deg", 1.0)),
        )

    @classmethod
    def load(cls, path: str | Path | None = None) -> SrpPhat:
        """config/audio.json's ``array`` and ``doa`` blocks (or ``path``'s)."""
        data = _audio_config(path)
        return cls.from_config(ArrayGeometry.from_config(data["array"]), data.get("doa", {}))

    def response(self, chunk: npt.NDArray[np.floating[Any]]) -> Floats:
        """The response at every grid azimuth of ``chunk`` (mics x samples, float, any scale);
        each value within -1..1, 1 for a perfect plane wave from that azimuth."""
        if chunk.shape != (self.geometry.count, self.chunk):
            raise ValueError(f"chunk {chunk.shape}, expected {(self.geometry.count, self.chunk)}")
        spectra = np.fft.rfft(chunk * self._window, self.nfft, axis=1)[:, self._bins]  # (M, B)
        cross = np.array([spectra[i] * np.conj(spectra[j]) for i, j in self.pairs])  # (P, B)
        magnitude = np.abs(cross)
        phasors = np.where(magnitude > 0.0, cross / np.maximum(magnitude, 1e-30), 0.0).reshape(-1)
        # macOS Accelerate's BLAS leaves spurious floating-point flags behind these products
        # (the inputs are finite unit phasors; the result equals the plain sum to 1e-15,
        # checked 2026-10-14), which numpy would report as warnings.
        with np.errstate(all="ignore"):
            steered = self._steering_re @ phasors.real - self._steering_im @ phasors.imag
        out: Floats = steered.astype(np.float64) / self._norm
        return out

    def peaks(self, response: Floats, *, ratio: float, min_strength: float = 0.0) -> list[Peak]:
        """The sources in a ``response``: its local maxima (round the circle) that reach
        ``ratio`` of the strongest one's value (the strongest always) and ``min_strength``;
        strongest first."""
        if response.size == 0:
            return []
        left, right = np.roll(response, 1), np.roll(response, -1)
        maxima = np.flatnonzero((response > left) & (response >= right))
        if maxima.size == 0:  # a flat response: no source
            return []
        strongest = float(response[maxima].max())
        floor = max(min_strength, ratio * strongest)
        found = [Peak(float(self.angles_deg[k]), float(response[k])) for k in maxima
                 if response[k] >= floor]  # fmt: skip
        found.sort(key=lambda p: p.strength, reverse=True)
        return found


@dataclass(frozen=True)
class MicReading(DoaReading):
    """A reading computed here (the chip's shape, :class:`pepin.audio_link.DoaReading`, plus
    what the chip has not): ``deg`` the span's strongest peak (-1: no chunk of the span was
    voice, nothing computed), ``speech`` whether any was, ``voice_p`` the span's highest Silero
    probability (0 when unknown), ``peaks`` every source of its voiced chunks (strongest first),
    ``chunks`` and ``voiced`` the span's counts."""

    voice_p: float = 0.0
    peaks: tuple[Peak, ...] = ()
    chunks: int = 0
    voiced: int = 0


@dataclass(frozen=True)
class _Chunk:
    stamp_s: float  # board clock, first sample
    p: float | None
    peaks: tuple[Peak, ...]


def merge_peaks(peaks: Sequence[Peak], within_deg: float = MERGE_DEG) -> tuple[Peak, ...]:
    """``peaks`` of several chunks as one list: a peak within ``within_deg`` (round the circle)
    of a stronger one kept already is the same source; strongest first."""
    kept: list[Peak] = []
    for peak in sorted(peaks, key=lambda p: p.strength, reverse=True):
        if all(abs(_wrap(peak.deg - k.deg)) > within_deg for k in kept):
            kept.append(peak)
    return tuple(kept)


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class MicDoa:
    """The laptop's direction source (the module's notes): :meth:`feed` takes the board's raw
    frames on the link's raw pump, :meth:`latest_doa` and :meth:`doa_between` answer the gaze
    on its own thread, :meth:`raw_age_s` says whether raw frames come at all."""

    def __init__(
        self,
        srp: SrpPhat,
        series: SpeechPeak | None = None,
        *,
        voice_p_min: float = 0.5,
        peak_ratio: float = 0.6,
        reading_s: float = 0.1,
        history_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``series``: the ears' Silero probabilities (None until the ears exist: no chunk is
        voice); the knobs are ``speaker.doa``'s ``mac_voice_p_min``, ``mac_peak_ratio`` and
        ``mac_reading_s`` (:meth:`configure` takes new ones)."""
        self._srp = srp
        self.series = series
        self.voice_p_min, self.peak_ratio, self.reading_s = voice_p_min, peak_ratio, reading_s
        self._history_s = history_s
        self._clock = clock
        self._lock = threading.Lock()
        self._readings: deque[MicReading] = deque()
        self._buffer: npt.NDArray[np.float32] = np.zeros((srp.geometry.count, 0), np.float32)
        self._buffer_stamp = 0.0  # board stamp of the buffer's first sample
        self._next_seq: int | None = None
        self._span: list[_Chunk] = []
        self._span_index: int | None = None
        self._origin: float | None = None  # the spans' grid origin (board clock)
        self._raw_at = 0.0  # laptop clock of the last raw frame
        self.frames = 0
        self.chunks = 0
        self.voiced = 0  # chunks whose peaks were computed
        self.gaps = 0  # seq gaps: the buffer restarted
        self.cost = LatencyTracker("audio_doa.chunk")  # seconds per voiced chunk

    def configure(self, *, voice_p_min: float, peak_ratio: float, reading_s: float) -> None:
        """New knobs (the speaker gaze's switch turned on re-reads them)."""
        self.voice_p_min, self.peak_ratio, self.reading_s = voice_p_min, peak_ratio, reading_s

    # -- the source ---------------------------------------------------------------------------

    def latest_doa(self) -> MicReading | None:
        """The newest reading, None before any."""
        with self._lock:
            return self._readings[-1] if self._readings else None

    def doa_between(self, t0: float, t1: float) -> list[DoaReading]:
        """The readings stamped within ``t0..t1`` (board clock)."""
        with self._lock:
            return [r for r in self._readings if t0 <= r.stamp_s <= t1]

    def raw_age_s(self, now: float | None = None) -> float:
        """Laptop seconds since the last raw frame; infinite before the first."""
        if not self._raw_at:
            return math.inf
        return (self._clock() if now is None else now) - self._raw_at

    def status(self) -> dict[str, Any]:
        """Counts and the cost per voiced chunk, for a status line."""
        cost = self.cost.summary()
        return {
            "raw_frames": self.frames,
            "chunks": self.chunks,
            "voiced": self.voiced,
            "gaps": self.gaps,
            "chunk_ms_median": round(cost.median_ms, 3),
            "chunk_ms_p95": round(cost.p95_ms, 3),
            "series": self.series is not None,
        }

    # -- the feed -----------------------------------------------------------------------------

    def feed(self, frame: RawFrame, arrived_s: float | None = None) -> list[MicReading]:
        """One raw frame (``arrived_s``: the laptop clock when it came, now by default); the
        readings its chunks completed."""
        now = self._clock() if arrived_s is None else arrived_s
        self._raw_at = now
        self.frames += 1
        mics = self._srp.geometry.count
        if frame.channels != mics:
            raise ValueError(f"raw frame of {frame.channels} channels, the geometry has {mics}")
        samples = np.frombuffer(frame.pcm, dtype="<i2").reshape(-1, frame.channels).T
        closed: list[MicReading] = []
        if self._next_seq is not None and frame.seq != self._next_seq:
            self.gaps += 1  # a partial chunk is dropped; the span pending cannot go on
            self._buffer = self._buffer[:, :0]
            if self._span:
                closed.append(self._close())
        if self._buffer.shape[1] == 0:
            self._buffer_stamp = frame.stamp_s
        self._next_seq = frame.seq + 1
        self._buffer = np.concatenate([self._buffer, samples.astype(np.float32) / 32768.0], axis=1)
        rate = self._srp.rate
        chunk_s = self._srp.chunk / rate
        size = self._srp.chunk
        while self._buffer.shape[1] >= size:
            chunk, self._buffer = self._buffer[:, :size], self._buffer[:, size:]
            stamp = self._buffer_stamp
            self._buffer_stamp += chunk_s
            end = now - self._buffer.shape[1] / rate  # the chunk's last sample, laptop clock
            p = None if self.series is None else self.series.peak(end - chunk_s, end)
            peaks: tuple[Peak, ...] = ()
            if p is not None and p >= self.voice_p_min:
                started = time.perf_counter()
                peaks = tuple(self._srp.peaks(self._srp.response(chunk), ratio=self.peak_ratio))
                self.cost.add(time.perf_counter() - started)
                self.voiced += 1
            self.chunks += 1
            reading = self._take(_Chunk(stamp, p, peaks))
            if reading is not None:
                closed.append(reading)
        return closed

    def _take(self, chunk: _Chunk) -> MicReading | None:
        """``chunk`` into its span; the span before it, closed, when it opens a new one."""
        if self._origin is None:
            self._origin = chunk.stamp_s
        # a microsecond of tolerance: a chunk that starts on a span's boundary belongs to it
        index = math.floor((chunk.stamp_s - self._origin) / self.reading_s + 1e-6)
        closed = None
        if self._span and index != self._span_index:
            closed = self._close()
        self._span.append(chunk)
        self._span_index = index
        return closed

    def _close(self) -> MicReading:
        span, self._span = self._span, []
        voiced = [c for c in span if c.peaks]
        peaks = merge_peaks([p for c in voiced for p in c.peaks])
        known = [c.p for c in span if c.p is not None]
        reading = MicReading(
            span[0].stamp_s,
            round(peaks[0].deg) % 360 if peaks else -1,
            bool(voiced),
            voice_p=max(known) if known else 0.0,
            peaks=peaks,
            chunks=len(span),
            voiced=len(voiced),
        )
        with self._lock:
            self._readings.append(reading)
            while self._readings and self._readings[0].stamp_s < reading.stamp_s - self._history_s:
                self._readings.popleft()
        return reading


def pcm_of(chunk: npt.NDArray[np.floating[Any]]) -> bytes:
    """``chunk`` (mics x samples, -1..1) as the interleaved s16le of a raw frame (tests, the
    bench)."""
    clipped = np.clip(chunk.T * 32767.0, -32768, 32767).astype("<i2")
    return clipped.tobytes()


def plane_wave(
    geometry: ArrayGeometry,
    source: npt.NDArray[np.floating[Any]],
    azimuth_deg: float,
    *,
    rate: int = RATE,
) -> npt.NDArray[np.float64]:
    """``source`` (samples) as the array's four microphones hear a plane wave from
    ``azimuth_deg`` (mics x samples): each mic's copy advanced by its projection on the wave's
    direction over the speed of sound (a fractional delay, through the FFT); the synthetic
    stream of the tests and the bench."""
    radians = math.radians(azimuth_deg)
    toward = np.array([math.cos(radians), math.sin(radians)])
    n = source.size
    spectrum = np.fft.rfft(source)
    freqs = np.fft.rfftfreq(n, 1.0 / rate)
    out = np.zeros((geometry.count, n))
    for m, (x, y) in enumerate(geometry.mics_m):
        advance_s = (x * toward[0] + y * toward[1]) / geometry.speed_m_s
        out[m] = np.fft.irfft(spectrum * np.exp(2j * np.pi * freqs * advance_s), n)
    return out


__all__ = [
    "BAND_HZ",
    "CHUNK",
    "SAMPLE_BYTES",
    "ArrayGeometry",
    "MicDoa",
    "MicReading",
    "Peak",
    "SpeechPeak",
    "SrpPhat",
    "merge_peaks",
    "pcm_of",
    "plane_wave",
]
