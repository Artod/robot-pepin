"""The robot's local voices: Qwen3-TTS on the Mac's GPU (mlx-audio), or macOS ``say`` for a test.

Qwen3-TTS streams ``chunk_s`` chunks; warm, its first audio comes ~0.1 s after the text (2 s on
the first call after loading, so :meth:`QwenVoice.load` says one word to itself). Three models:
0.6B CustomVoice (a preset speaker, no style: upstream drops the instruction on the 0.6B and
mlx-audio passes it to a model never trained on it), 1.7B CustomVoice (a preset speaker and a
free-text ``instruct`` per sentence: the style) and 1.7B VoiceDesign (the voice described in
words, the style appended to the description). A model loaded and warmed on one thread speaks
on another (mlx 0.32, checked 2026-10-09), so a new voice loads beside the one speaking.
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


def instruction(description: str | None, style: str) -> str | None:
    """What Qwen3-TTS is told beside the text: a VoiceDesign description with the style
    appended as its own sentence, the style alone (CustomVoice), or None."""
    style = style.strip()
    if style:
        style = style[0].upper() + style[1:]
        if style[-1] not in ".!?":
            style += "."
    if description:
        return f"{description.strip()} {style}".strip()
    return style or None


class QwenVoice:
    """:class:`pepin.voice_cascade.stages.Voice` over mlx-audio's Qwen3-TTS."""

    def __init__(
        self,
        model: str,
        *,
        voice: str | None = "serena",
        language: str = "russian",
        chunk_s: float = 0.5,
        hf_home: Path | None = None,
        use_style: bool = False,
        description: str | None = None,
        seed: int | None = None,
    ) -> None:
        """``model``: a Hugging Face repo id; ``hf_home``: an HF cache that already holds it
        (its ``hub`` folder is read, nothing downloaded), or None for the default cache;
        ``voice``: a CustomVoice speaker; ``description``: a VoiceDesign voice in words;
        ``use_style``: the style reaches the model; ``seed``: the sampler seeded before each
        sentence (a designed voice keeps its timbre)."""
        self.model_id = model
        self.voice, self.language, self.chunk_s = voice, language, chunk_s
        self.hf_home = hf_home
        self.use_style, self.description, self.seed = use_style, description, seed
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
        """Load the model and say one word (once: a loaded voice returns at once)."""
        if self._model is not None:
            return
        from mlx_audio.tts.utils import load_model

        model = load_model(Path(self.path()))
        self.rate = int(getattr(model, "sample_rate", QWEN_RATE))
        self._model = model
        for _ in self.stream(WARM_UP):
            pass

    def instruct(self, style: str) -> str | None:
        """The instruction for a sentence said in ``style``."""
        return instruction(self.description, style if self.use_style else "")

    def stream(self, text: str, style: str = "") -> Iterator[bytes]:
        """``text`` as chunks of ``chunk_s`` seconds, each yielded as soon as it is decoded."""
        if self._model is None:
            raise RuntimeError("QwenVoice.load() first")
        if self.seed is not None:
            import mlx.core as mx

            mx.random.seed(self.seed)
        for result in self._model.generate(
            text=text,
            voice=self.voice,
            instruct=self.instruct(style),
            lang_code=self.language,
            stream=True,
            streaming_interval=self.chunk_s,
            verbose=False,
        ):
            yield to_pcm(result.audio)


class SayVoice:
    """:class:`pepin.voice_cascade.stages.Voice` over macOS ``say``: the whole sentence at once
    (no streaming, no style), at 16 kHz. For a test without the TTS model; not the robot's
    voice."""

    def __init__(self, voice: str = "Milena", rate_wpm: int = 185) -> None:
        """A ``say`` voice name and its speed in words per minute."""
        self.voice, self.rate_wpm = voice, rate_wpm
        self.rate = RATE

    def load(self) -> None:
        """Nothing to load."""

    def stream(self, text: str, style: str = "") -> Iterator[bytes]:
        """The whole sentence, once ``say`` has written it (the style ignored)."""
        yield synthesize(text, self.voice, self.rate_wpm)


class GainedVoice:
    """A voice's PCM scaled by ``gain_db`` (clipped at full scale): one loudness for every
    engine, so the room and the echo guard meet the same level whoever speaks. Everything
    else (the bill, the fallback's name) is the wrapped voice's."""

    def __init__(self, inner: Any, gain_db: float) -> None:
        """``inner``: the engine; ``gain_db``: negative quietens."""
        self.inner, self.gain_db = inner, gain_db
        self.rate: int = inner.rate

    def load(self) -> None:
        """See :class:`pepin.voice_cascade.stages.Voice`."""
        self.inner.load()

    def stream(self, text: str, style: str = "") -> Iterator[bytes]:
        """The engine's chunks, each scaled."""
        for pcm in self.inner.stream(text, style):
            yield with_gain(pcm, self.gain_db)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)


def with_gain(pcm: bytes, gain_db: float) -> bytes:
    """s16le ``pcm`` scaled by ``gain_db``, clipped to the sample range."""
    if not pcm or gain_db == 0.0:
        return pcm
    import numpy as np

    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) * (10 ** (gain_db / 20))
    return np.clip(samples, -32768, 32767).astype("<i2").tobytes()
