"""Who a voice is: one speaker embedding per utterance.

An utterance arrives as the ears cut it (16 kHz s16le mono, the lead-in of room sound
included). :func:`speech_span` trims it to its loud span (20 ms frames within ``span_db`` of the
utterance's 95th-percentile level, first to last), and only a span of at least ``min_s`` is
embedded: a short "да" carries too little of a voice to name anyone.

:class:`WeSpeaker` is WeSpeaker's ResNet34-LM r-vector (VoxCeleb2, 5,994 speakers; 6.6 M
parameters; CC-BY-4.0; ``voxceleb_resnet34_LM.onnx``, 26.5 MB, from
huggingface.co/Wespeaker/wespeaker-voxceleb-resnet34-LM) through onnxruntime on the CPU. Its
input is WeSpeaker's own: 80 Kaldi log-mel filterbanks (25 ms Hamming frames every 10 ms, the
samples at int16 scale, no dither) with the utterance's mean removed (:func:`fbank`, the
numpy twin of ``torchaudio.compliance.kaldi.fbank``); its output a 256-float vector, compared
by cosine.

MEASURED on this Mac (scratch/people_1011/REPORT.md): see the voice section there for the
cosines on Artem's utterances, the robot's own voice, TTS voices and LibriSpeech speakers, and
the milliseconds per utterance.
"""

from __future__ import annotations

import logging
import math
import threading
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from pepin.face_sensor.identity import Vector, unit

logger = logging.getLogger(__name__)

RATE = 16_000
FRAME_LEN = 400  # 25 ms
FRAME_SHIFT = 160  # 10 ms
FFT = 512
MEL_BINS = 80
LOW_HZ = 20.0
PREEMPHASIS = 0.97
EPS = float(np.finfo(np.float32).eps)  # torchaudio's floor under the log
SPAN_FRAME = 320  # 20 ms, for the loud span


def samples_of(pcm: bytes) -> NDArray[np.int16]:
    """16 kHz s16le mono bytes as int16 samples (an odd trailing byte dropped)."""
    return np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2")


def speech_span(samples: NDArray[np.int16], span_db: float = 35.0) -> tuple[int, int]:
    """The loud span of an utterance as sample indices ``(start, end)``: the 20 ms frames
    within ``span_db`` of the 95th-percentile frame level, first to last; ``(0, 0)`` for
    silence or less than one frame."""
    n = len(samples) // SPAN_FRAME
    if n == 0:
        return 0, 0
    frames = samples[: n * SPAN_FRAME].astype(np.float64).reshape(n, SPAN_FRAME)
    rms = np.sqrt(np.mean(frames**2, axis=1))
    db = 20.0 * np.log10(np.maximum(rms, 1.0) / 32768.0)
    top = float(np.percentile(db, 95))
    if top <= -90.0:
        return 0, 0
    loud = np.flatnonzero(db >= top - span_db)
    return int(loud[0]) * SPAN_FRAME, (int(loud[-1]) + 1) * SPAN_FRAME


def speech_seconds(pcm: bytes, span_db: float = 35.0) -> float:
    """How long the loud span of ``pcm`` lasts."""
    start, end = speech_span(samples_of(pcm), span_db)
    return (end - start) / RATE


def _mel(hz: NDArray[np.float64] | float) -> NDArray[np.float64]:
    return np.asarray(1127.0 * np.log(1.0 + np.asarray(hz, dtype=np.float64) / 700.0))


def mel_banks(bins: int = MEL_BINS, fft: int = FFT, rate: int = RATE) -> NDArray[np.float64]:
    """Kaldi's triangular mel filters, ``(bins, fft // 2 + 1)`` (the Nyquist column zero), as
    torchaudio builds them (low 20 Hz, high the Nyquist)."""
    half = fft // 2
    low, high = _mel(LOW_HZ), _mel(rate / 2.0)
    delta = (high - low) / (bins + 1)
    b = np.arange(bins, dtype=np.float64)[:, None]
    left, centre, right = low + b * delta, low + (b + 1.0) * delta, low + (b + 2.0) * delta
    mel = _mel(rate / fft * np.arange(half, dtype=np.float64))[None, :]
    up = (mel - left) / (centre - left)
    down = (right - mel) / (right - centre)
    banks = np.maximum(0.0, np.minimum(up, down))
    return np.pad(banks, ((0, 0), (0, 1)))


_BANKS = mel_banks()
_HAMMING = 0.54 - 0.46 * np.cos(2.0 * math.pi * np.arange(FRAME_LEN) / (FRAME_LEN - 1))


def fbank(samples: NDArray[np.int16]) -> NDArray[np.float32]:
    """WeSpeaker's input for int16 ``samples``: 80 Kaldi log-mel filterbanks per 10 ms
    (``snip_edges``: whole 25 ms frames only), the utterance's mean removed; ``(T, 80)``."""
    x = samples.astype(np.float64)
    n = 1 + (len(x) - FRAME_LEN) // FRAME_SHIFT
    if n <= 0:
        return np.zeros((0, MEL_BINS), dtype=np.float32)
    idx = np.arange(FRAME_LEN)[None, :] + FRAME_SHIFT * np.arange(n)[:, None]
    frames = x[idx]
    frames = frames - frames.mean(axis=1, keepdims=True)
    shifted = np.concatenate([frames[:, :1], frames[:, :-1]], axis=1)
    frames = (frames - PREEMPHASIS * shifted) * _HAMMING
    power = np.abs(np.fft.rfft(frames, n=FFT, axis=1)) ** 2
    with np.errstate(all="ignore"):  # Accelerate's matmul raises spurious FP flags on arm64
        energies = power @ _BANKS.T
    feats: NDArray[np.float64] = np.log(np.maximum(energies, EPS))
    feats = feats - feats.mean(axis=0, keepdims=True)
    return feats.astype(np.float32)


class VoiceEmbedder(Protocol):
    """The identity vector of a voice."""

    name: str  # the model's name: samples of different models never compare

    def embed(self, samples: NDArray[np.int16]) -> Vector:
        """The unit-length embedding of an utterance's speech (int16, 16 kHz, trimmed)."""
        ...


class WeSpeaker:
    """WeSpeaker's ONNX r-vector on the CPU; implements :class:`VoiceEmbedder`."""

    def __init__(self, path: Path, *, threads: int = 2) -> None:
        """Load ``voxceleb_resnet34_LM.onnx`` (or another WeSpeaker export taking ``feats``)."""
        import onnxruntime as ort

        if not path.exists():
            raise FileNotFoundError(f"the voice model is missing: {path} (config/people.json)")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        options.log_severity_level = 3
        self._session: Any = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._input = self._session.get_inputs()[0].name
        self.name = path.stem
        self._lock = threading.Lock()

    def embed(self, samples: NDArray[np.int16]) -> Vector:
        """The unit-length 256-float embedding of ``samples``."""
        feats = fbank(samples)
        if len(feats) == 0:
            raise ValueError("too short for a speaker embedding")
        with self._lock:
            out = self._session.run(None, {self._input: feats[None]})[0]
        embedding: Vector = unit(np.asarray(out, dtype=np.float32)[0])
        return embedding
