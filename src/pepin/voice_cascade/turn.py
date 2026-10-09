"""The end of a turn by its sound: Smart Turn v3.2 (pipecat-ai/smart-turn), on a worker thread.

The energy gate still proposes the moment (a short silence inside an utterance); the model
judges it from the utterance's last 8 s: a Whisper-tiny encoder with a linear head over the
log-mel features of 16 kHz audio, zeros in front of shorter audio so the speech sits at the end
(as it was trained), out a probability that the speaker has finished. The 8 MB int8 ONNX build
runs on the CPU in ~9 ms here (features 1.5 ms, the network 7 ms).

:class:`TurnWorker` keeps the mic's thread free: the ears hand it the audio of a pause and go on
reading frames; the worker answers on its own thread, and the ears read the latest answer on
their next frame. A newer request replaces one not yet started, so a backlog never forms.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from pepin.audio_link import RATE
from pepin.voice_cascade.config import TURN_MODELS

logger = logging.getLogger(__name__)

WINDOW_S = 8.0  # the model's input: 8 s of 16 kHz audio


class TurnDetector(Protocol):
    """The probability that a turn has ended, from its audio."""

    def probability(self, pcm: bytes) -> float:
        """For 16 kHz s16le mono ``pcm`` ending now (the speech and the pause after it)."""
        ...


class SmartTurn:
    """:class:`TurnDetector` over Smart Turn v3.2's ONNX build (onnxruntime on the CPU, the
    features by transformers' Whisper extractor). Loaded and warmed at construction."""

    def __init__(self, name: str = "smart-turn-v3.2", path: Path | None = None) -> None:
        """``name``: one of :data:`pepin.voice_cascade.config.TURN_MODELS` (fetched from Hugging
        Face once, 8 MB); ``path``: a local ONNX file instead."""
        import numpy as np
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        if path is None:
            from huggingface_hub import hf_hub_download

            repo, filename = TURN_MODELS[name]
            path = Path(hf_hub_download(repo, filename))
        options = ort.SessionOptions()
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 2  # Whisper and the voice share the machine
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self._np = np
        self._session: Any = ort.InferenceSession(str(path), sess_options=options)
        extractor: Any = WhisperFeatureExtractor  # untyped upstream
        self._features: Any = extractor(chunk_length=int(WINDOW_S))
        self.path = path
        self.probability(b"\x00\x00" * RATE)

    def probability(self, pcm: bytes) -> float:
        """The model's p for the last 8 s of ``pcm`` (zeros in front of shorter audio)."""
        np = self._np
        n = int(WINDOW_S * RATE)
        x = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(np.float32) / 32768
        x = x[-n:]
        x = np.pad(x, (n - len(x), 0))
        features = self._features(
            x,
            sampling_rate=RATE,
            return_tensors="np",
            padding="max_length",
            max_length=n,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        out = self._session.run(None, {"input_features": features})[0]
        return float(out[0][0])  # the graph ends in its sigmoid: already 0..1


@dataclass(frozen=True)
class TurnAnswer:
    """The model's answer to request ``id``: p and the milliseconds it took."""

    id: int
    p: float
    ms: float


class TurnWorker:
    """A :class:`TurnDetector` on its own thread with the latest-result pattern: :meth:`ask`
    never blocks, :attr:`latest` is the newest answer. ``inline`` answers on the caller's thread
    (the tests)."""

    def __init__(
        self,
        detector: TurnDetector,
        *,
        inline: bool = False,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        """Start the thread (unless ``inline``)."""
        self.detector = detector
        self.inline = inline
        self.clock = clock
        self.latest: TurnAnswer | None = None
        self._cond = threading.Condition()
        self._pending: tuple[int, bytes] | None = None
        self._stop = False
        if not inline:
            threading.Thread(target=self._run, name="turn", daemon=True).start()

    def ask(self, request: int, pcm: bytes) -> None:
        """Judge ``pcm`` as request ``request`` (an older one not yet started is dropped)."""
        if self.inline:
            self._answer(request, pcm)
            return
        with self._cond:
            self._pending = (request, pcm)
            self._cond.notify()

    def close(self) -> None:
        """Stop the thread."""
        with self._cond:
            self._stop = True
            self._cond.notify()

    def _run(self) -> None:
        while True:
            with self._cond:
                while self._pending is None and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                pending, self._pending = self._pending, None
            if pending is not None:
                self._answer(*pending)

    def _answer(self, request: int, pcm: bytes) -> None:
        t0 = self.clock()
        try:
            p = self.detector.probability(pcm)
        except Exception:  # a broken model leaves the fixed silence to close the turn
            logger.exception("end-of-turn model")
            return
        self.latest = TurnAnswer(request, p, (self.clock() - t0) * 1000)
