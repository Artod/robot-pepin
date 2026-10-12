"""Mistral's Voxtral Mini 4B Realtime (Apache-2.0, mistralai/Voxtral-Mini-4B-Realtime-2602): a
streaming speech model for 13 languages, Russian among them, on the GPU through mlx-audio
(mlx-community/Voxtral-Mini-4B-Realtime-2602-4bit: 3.1 GB of 4-bit weights).

The model has no language input. It writes one token per 80 ms of audio behind a delay
(``delay_ms``, the model's own 480 by default), so a whole utterance read at once costs a
decoder step per 80 ms of it; streaming (mlx-audio's ``create_streaming_session``) would spend
that while the person talks, which the cascade's one call per utterance does not.
Install: ``uv sync --group stt-voxtral``.
"""

from __future__ import annotations

from pepin.voice_cascade.config import SttConfig
from pepin.voice_cascade.stt.runtime import MlxAudio


class Voxtral(MlxAudio):
    """Voxtral Mini 4B Realtime on the GPU, loaded and warmed at construction."""

    def __init__(self, settings: SttConfig) -> None:
        """``settings``: the ``transcriber.voxtral`` block (model, dtype, delay_ms)."""
        delay = settings.delay_ms
        super().__init__(
            f"voxtral:{settings.model}",
            settings.model,
            dtype=settings.dtype,
            options={} if delay is None else {"transcription_delay_ms": delay},
        )
