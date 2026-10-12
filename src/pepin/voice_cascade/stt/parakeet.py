"""NVIDIA Parakeet TDT 0.6B v3 (CC-BY-4.0, nvidia/parakeet-tdt-0.6b-v3): a transducer for 25
European languages, Russian among them, on the GPU through mlx-audio
(mlx-community/parakeet-tdt-0.6b-v3).

The model has no language input: it follows the speech, so English words come out in Latin
letters. The repo stores float32 (2.5 GB); ``dtype: bfloat16`` (the default) holds ~1.2 GB.
It can stream (mlx-audio's ``generate(stream=True)``); the cascade calls it once per utterance.
Install: ``uv sync --group stt-parakeet``.
"""

from __future__ import annotations

from pepin.voice_cascade.config import SttConfig
from pepin.voice_cascade.stt.runtime import MlxAudio


class Parakeet(MlxAudio):
    """Parakeet TDT 0.6B v3 on the GPU, loaded and warmed at construction."""

    def __init__(self, settings: SttConfig) -> None:
        """``settings``: the ``transcriber.parakeet`` block (model, dtype)."""
        super().__init__(f"parakeet:{settings.model}", settings.model, dtype=settings.dtype)
