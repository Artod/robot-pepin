"""NVIDIA Canary 1B v2 (CC-BY-4.0, nvidia/canary-1b-v2): an attention encoder-decoder for 25
European languages, Russian among them, on onnxruntime's CPU through onnx-asr's export
(istupakov/canary-1b-v2-onnx).

The language is forced (source and target ``transcriber.language``: transcription, not
translation). ``quantization: int8`` (the default) takes ~1.4 GB, the full model ~4 GB. No
streaming: one call per utterance. Install: ``uv sync --group stt-canary``.
"""

from __future__ import annotations

from pepin.voice_cascade.config import SttConfig
from pepin.voice_cascade.stt.runtime import OnnxAsr


class Canary(OnnxAsr):
    """Canary 1B v2 on the CPU, its language forced, loaded and warmed at construction."""

    def __init__(self, settings: SttConfig, language: str) -> None:
        """``settings``: the ``transcriber.canary`` block; ``language``: an ISO code (``ru``)."""
        super().__init__(
            f"canary:{settings.model}",
            settings.model,
            threads=settings.threads,
            quantization=settings.quantization,
            options={"language": language, "target_language": language},
        )
