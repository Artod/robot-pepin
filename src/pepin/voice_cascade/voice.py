"""The robot's voice: Qwen3-TTS on the Mac's GPU (mlx-audio), or macOS ``say`` for a test.

Qwen3-TTS 0.6B CustomVoice ("serena", Russian) streams half-second chunks; warm, its first audio
comes ~0.1 s after the text (2 s on the first call after loading, so :meth:`QwenVoice.load` says
one word to itself). The model is loaded on the thread that speaks with it: MLX keeps its
streams per thread.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pepin.audio_link import RATE
from pepin.voice_live.sources import synthesize

QWEN_RATE = 24_000  # Hz: what Qwen3-TTS's codec decodes to
WARM_UP = "Привет."


def to_pcm(audio: Any) -> bytes:
    """Float samples in -1..1 (a numpy or MLX array) as s16le."""
    import numpy as np

    samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
    return bytes((samples * 32767.0).astype("<i2").tobytes())


class QwenVoice:
    """:class:`pepin.voice_cascade.stages.Voice` over mlx-audio's Qwen3-TTS."""

    def __init__(
        self,
        model: str,
        *,
        voice: str = "serena",
        language: str = "russian",
        chunk_s: float = 0.5,
        hf_home: Path | None = None,
    ) -> None:
        """``model``: a Hugging Face repo id; ``hf_home``: an HF cache that already holds it
        (its ``hub`` folder is read, nothing downloaded), or None for the default cache."""
        self.model_id = model
        self.voice, self.language, self.chunk_s = voice, language, chunk_s
        self.hf_home = hf_home
        self.rate = QWEN_RATE
        self._model: Any = None

    def path(self) -> Path | str:
        """The model's local snapshot in ``hf_home`` when it is there, else the repo id."""
        if self.hf_home is not None:
            from huggingface_hub import snapshot_download

            try:
                return Path(
                    snapshot_download(
                        self.model_id, cache_dir=self.hf_home / "hub", local_files_only=True
                    )
                )
            except Exception:  # not in that cache: the default one (or a download) instead
                pass
        return self.model_id

    def load(self) -> None:
        """Load the model and say one word, on the calling thread."""
        from mlx_audio.tts.utils import load_model

        self._model = load_model(Path(self.path()))
        self.rate = int(getattr(self._model, "sample_rate", QWEN_RATE))
        for _ in self.stream(WARM_UP):
            pass

    def stream(self, text: str) -> Iterator[bytes]:
        """``text`` as chunks of ``chunk_s`` seconds, each yielded as soon as it is decoded."""
        if self._model is None:
            raise RuntimeError("QwenVoice.load() first, on the thread that speaks")
        for result in self._model.generate(
            text=text,
            voice=self.voice,
            lang_code=self.language,
            stream=True,
            streaming_interval=self.chunk_s,
            verbose=False,
        ):
            yield to_pcm(result.audio)


class SayVoice:
    """:class:`pepin.voice_cascade.stages.Voice` over macOS ``say``: the whole sentence at once
    (no streaming), at 16 kHz. For a test without the TTS model; not the robot's voice."""

    def __init__(self, voice: str = "Milena", rate_wpm: int = 185) -> None:
        """A ``say`` voice name and its speed in words per minute."""
        self.voice, self.rate_wpm = voice, rate_wpm
        self.rate = RATE

    def load(self) -> None:
        """Nothing to load."""

    def stream(self, text: str) -> Iterator[bytes]:
        """The whole sentence, once ``say`` has written it."""
        yield synthesize(text, self.voice, self.rate_wpm)
