"""scripts/voice.py keeps each utterance as a 16 kHz mono WAV beside its log line."""

from __future__ import annotations

import importlib.util
import time
import wave
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]


def load_voice() -> ModuleType:
    spec = importlib.util.spec_from_file_location("voice_script", REPO / "scripts/voice.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_utterance_lands_in_its_day_folder_as_the_pcm_the_model_got(tmp_path: Path) -> None:
    voice = load_voice()
    pcm = bytes(range(256)) * 10
    t = time.mktime((2026, 10, 2, 14, 3, 7, 0, 0, -1)) + 0.042
    rel = voice.save_wav(pcm, tmp_path, t)
    assert rel == "20261002/140307_042.wav"
    assert (tmp_path / rel).read_bytes() == voice.wav_bytes(pcm)
    with wave.open(str(tmp_path / rel)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert w.readframes(w.getnframes()) == pcm
