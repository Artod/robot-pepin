"""Speech-to-text engines the cascade can switch to besides Whisper, one module each.

:mod:`.gigaam` (GigaAM v3, Russian) and :mod:`.canary` (Canary 1B v2) run on onnxruntime's CPU
through onnx-asr; :mod:`.parakeet` (Parakeet TDT 0.6B v3) and :mod:`.voxtral` (Voxtral Mini 4B
Realtime) on the GPU through mlx-audio (:mod:`.runtime` holds both runtimes). Each is a
:class:`pepin.voice_cascade.stages.Transcriber` built from its ``transcriber.<engine>`` block,
loaded and warmed at construction and freed by ``close()``; its runtime comes from an optional
dependency group (``uv sync --group stt-<engine>``) and is imported only when the engine is built.
"""

from __future__ import annotations

from pepin.voice_cascade.config import SttConfig
from pepin.voice_cascade.stages import Transcriber


def build(engine: str, settings: SttConfig, language: str) -> Transcriber:
    """The engine ``engine`` (a key of ``STT_DEVICES``) with ``settings``, loaded and warmed;
    ``language`` reaches the engines that take one."""
    if engine == "gigaam":
        from pepin.voice_cascade.stt.gigaam import GigaAM

        return GigaAM(settings)
    if engine == "canary":
        from pepin.voice_cascade.stt.canary import Canary

        return Canary(settings, language)
    if engine == "parakeet":
        from pepin.voice_cascade.stt.parakeet import Parakeet

        return Parakeet(settings)
    if engine == "voxtral":
        from pepin.voice_cascade.stt.voxtral import Voxtral

        return Voxtral(settings)
    raise ValueError(f"no speech-to-text engine {engine!r}")
