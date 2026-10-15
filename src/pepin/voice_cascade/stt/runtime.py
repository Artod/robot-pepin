"""What the engines of :mod:`pepin.voice_cascade.stt` share: the PCM as samples, the two
runtimes (onnx-asr on onnxruntime's CPU, mlx-audio on the GPU) and this process's footprint.

Each runtime is imported by its own function (:func:`onnx_runtime`, :func:`mlx_runtime`) when
an engine is built, so the cascade starts without them and the unit tests swap in fakes.

onnx-asr computes a CPU model's features with NumPy, and on this Mac every NumPy matmul raises
"divide by zero / overflow / invalid value encountered in matmul" with a correct result:
Accelerate's SME kernels (M4 and later) leave stale flags in the FP status register, which NumPy
reads after the call (numpy/numpy#29223, handled in NumPy 2.3.1; lerobot holds NumPy under
2.3). Python prints such a warning once per place, so it showed at the warm-up read only; the
reads run with those flags ignored (:func:`fp_flags_ignored`).
"""

from __future__ import annotations

import ctypes
import gc
import math
import os
import sys
from collections.abc import Callable, Mapping
from types import ModuleType
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.audio_link import RATE
from pepin.voice_cascade.stages import Transcript
from pepin.voice_live.wake import clean_transcript

WARM_S = 0.5  # the silence an engine reads once at load: its first call allocates and compiles
CPU = "CPUExecutionProvider"


def samples(pcm: bytes) -> npt.NDArray[np.float32]:
    """16 kHz s16le mono PCM as float32 samples in [-1, 1)."""
    return np.frombuffer(pcm, dtype="<i2").astype(np.float32) / np.float32(32768)


def silence(seconds: float = WARM_S) -> bytes:
    """``seconds`` of zero PCM."""
    return bytes(round(RATE * seconds) * 2)


def fp_flags_ignored() -> np.errstate:
    """NumPy's floating-point warnings off for a read (the module says why)."""
    return np.errstate(divide="ignore", over="ignore", invalid="ignore")


def onnx_runtime() -> tuple[Callable[..., Any], ModuleType]:
    """onnx-asr's ``load_model`` and onnxruntime (group ``stt-gigaam`` or ``stt-canary``)."""
    import onnx_asr
    import onnxruntime

    return onnx_asr.load_model, onnxruntime


def mlx_runtime() -> tuple[ModuleType, Callable[..., Any]]:
    """``mlx.core`` and mlx-audio's speech-to-text ``load_model`` (group ``stt-parakeet`` or
    ``stt-voxtral``)."""
    import mlx.core as mx
    from mlx_audio.stt.utils import load_model

    return mx, load_model


class OnnxAsr:
    """A model of onnx-asr on onnxruntime's CPU, loaded and warmed at construction: ``model``
    an onnx-asr name (its files come from the Hugging Face cache, fetched at the first load),
    ``threads`` onnxruntime's intra-op threads, ``options`` passed to every recognition (a
    language for the models that take one). With ``timings`` (a CTC or transducer model) a
    timed read also gives each word's time, its tokens' (``letters``: a character model's
    letters) and its vowels (:mod:`pepin.voice_cascade.timing`)."""

    def __init__(
        self,
        name: str,
        model: str,
        *,
        threads: int,
        quantization: str | None = None,
        options: Mapping[str, str] | None = None,
        timings: bool = False,
        letters: bool = False,
    ) -> None:
        """``name``: ``engine:model``, as the switch reports it."""
        load_model, onnxruntime = onnx_runtime()
        session = onnxruntime.SessionOptions()
        session.intra_op_num_threads = threads
        self.name = name
        self._options = dict(options or {})
        self._letters = letters
        self._model: Any = load_model(
            model, quantization=quantization, sess_options=session, providers=[CPU]
        )
        self._timed: Any = self._model.with_timestamps() if timings else None
        if self._timed is None:
            self.transcribe(silence())
        else:
            self.transcribe_timed(silence())

    def transcribe(self, pcm: bytes) -> str:
        """The words in ``pcm``."""
        if self._model is None:
            raise RuntimeError(f"{self.name} is closed")
        with fp_flags_ignored():
            text = self._model.recognize(samples(pcm), sample_rate=RATE, **self._options)
        return clean_transcript(str(text))

    def transcribe_timed(self, pcm: bytes) -> Transcript:
        """The words in ``pcm`` (as :meth:`transcribe` gives them) with their timings; none
        when timings are off. The same model call, onnx-asr keeping the frame each token was
        emitted at and its log-prob; with the loudness and Praat's voicing for the vowels,
        +0-4 ms on 2-4 s of speech, +10 ms on 6.8 s (2026-10-12)."""
        if self._timed is None:
            return Transcript(self.transcribe(pcm))
        if self._model is None:
            raise RuntimeError(f"{self.name} is closed")
        from pepin.voice_cascade import timing

        with fp_flags_ignored():
            result = self._timed.recognize(samples(pcm), sample_rate=RATE, **self._options)
        logprobs = result.logprobs or [None] * len(result.tokens)
        emissions = list(zip(result.tokens, result.timestamps, logprobs, strict=True))
        words = timing.words_of(emissions, pcm, letters=self._letters, voiced=timing.voicing(pcm))
        return Transcript(clean_transcript(str(result.text)), words)

    def close(self) -> None:
        """Drop the model's sessions; onnxruntime frees their memory with them."""
        self._model = self._timed = None
        gc.collect()


class MlxAudio:
    """A speech-to-text model of mlx-audio on the GPU, loaded and warmed at construction:
    ``repo`` a Hugging Face repo of MLX weights (cached after the first load), cast to ``dtype``
    when one is given (loaded lazily, so the stored copy is never in memory beside the cast),
    ``options`` passed to every ``generate`` call."""

    def __init__(
        self,
        name: str,
        repo: str,
        *,
        dtype: str | None = None,
        options: Mapping[str, Any] | None = None,
    ) -> None:
        """``name``: ``engine:model``, as the switch reports it."""
        mx, load_model = mlx_runtime()
        self.name = name
        self._mx = mx
        self._options = dict(options or {})
        model: Any = load_model(repo, lazy=True)
        if dtype is not None:
            model.set_dtype(getattr(mx, dtype))
        mx.eval(model.parameters())
        mx.clear_cache()
        self._model = model
        self.transcribe(silence())

    def transcribe(self, pcm: bytes) -> str:
        """The words in ``pcm``."""
        if self._model is None:
            raise RuntimeError(f"{self.name} is closed")
        result = self._model.generate(self._mx.array(samples(pcm)), **self._options)
        return clean_transcript(str(result.text))

    def close(self) -> None:
        """Drop the weights and hand the buffers MLX keeps for reuse back to the system."""
        self._model = None
        gc.collect()
        self._mx.clear_cache()


class _RusageInfo(ctypes.Structure):
    """``struct rusage_info_v0`` of <sys/resource.h> (macOS)."""

    _fields_ = [
        ("uuid", ctypes.c_uint8 * 16),
        *(
            (field, ctypes.c_uint64)
            for field in (
                "user_time",
                "system_time",
                "pkg_idle_wkups",
                "interrupt_wkups",
                "pageins",
                "wired_size",
                "resident_size",
                "phys_footprint",
                "proc_start_abstime",
                "proc_exit_abstime",
            )
        ),
    ]


def footprint_mb() -> float:
    """This process's physical footprint in MB (macOS: Activity Monitor's Memory column, the
    GPU's buffers included); NaN off macOS or when the call fails."""
    if sys.platform != "darwin":
        return math.nan
    info = _RusageInfo()
    libc = ctypes.CDLL("/usr/lib/libSystem.B.dylib", use_errno=True)
    if libc.proc_pid_rusage(os.getpid(), 0, ctypes.byref(info)) != 0:
        return math.nan
    return float(info.phys_footprint) / 2**20
