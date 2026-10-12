"""GigaAM v3, Sber's Russian conformer (MIT, github.com/salute-developers/GigaAM), on
onnxruntime's CPU through onnx-asr's export (istupakov/gigaam-v3-onnx).

Russian only, about 240 M parameters: ``gigaam-v3-e2e-rnnt`` (the default) and
``gigaam-v3-e2e-ctc`` write punctuation and capitals, ``gigaam-v3-rnnt`` and ``gigaam-v3-ctc``
lower case without. The full model takes ~0.9 GB, ``quantization: int8`` ~0.5 GB. No language
to force, no streaming: one call per utterance. CoreML fails on it in onnxruntime 1.30 and its
own PyTorch package pins torch 2.5, so the CPU is its device.
Install: ``uv sync --group stt-gigaam``.
"""

from __future__ import annotations

from pepin.voice_cascade.config import SttConfig
from pepin.voice_cascade.stt.runtime import OnnxAsr


class GigaAM(OnnxAsr):
    """GigaAM v3 on the CPU, loaded and warmed at construction."""

    def __init__(self, settings: SttConfig) -> None:
        """``settings``: the ``transcriber.gigaam`` block (model, threads, quantization)."""
        super().__init__(
            f"gigaam:{settings.model}",
            settings.model,
            threads=settings.threads,
            quantization=settings.quantization,
        )
