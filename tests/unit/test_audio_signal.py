"""Level, mean direction, voice onset and WAV conversion on synthetic sound."""

import math
import wave
from pathlib import Path

import numpy as np
import pytest

from pepin.audio_link import AudioFrame, DoaReading
from pepin.audio_signal import (
    SILENCE_DBFS,
    SustainedVoice,
    circular_mean_deg,
    level_dbfs,
    wav_to_pcm,
)
from pepin.voice import run_voice_pipeline


def tone(amplitude: float, samples: int = 320) -> bytes:
    t = np.arange(samples) / 16_000
    return (amplitude * 32767 * np.sin(2 * np.pi * 440 * t)).astype("<i2").tobytes()


def test_a_full_scale_sine_is_minus_three_dbfs_and_silence_is_the_floor() -> None:
    assert level_dbfs(tone(1.0, 16_000)) == pytest.approx(-3.01, abs=0.05)
    assert level_dbfs(tone(0.1, 16_000)) == pytest.approx(-23.01, abs=0.05)
    assert level_dbfs(b"\x00\x00" * 320) == SILENCE_DBFS and level_dbfs(b"") == SILENCE_DBFS


def test_directions_average_around_the_circle() -> None:
    wrapped = circular_mean_deg([350, 10])
    assert wrapped is not None and min(wrapped, 360.0 - wrapped) == pytest.approx(0.0, abs=1e-9)
    assert circular_mean_deg([80, 100, 90]) == pytest.approx(90.0)
    assert circular_mean_deg([]) is None and circular_mean_deg([0, 180]) is None


def test_half_a_second_of_voice_fires_once_and_syllable_gaps_do_not_break_it() -> None:
    detector = SustainedVoice(threshold_db=-40.0, hold_s=0.5, gap_s=0.15)
    fired = []
    levels = [-20.0] * 15 + [-60.0] * 5 + [-20.0] * 30 + [-60.0] * 10 + [-20.0] * 30
    for i, level in enumerate(levels):  # 20 ms frames
        if detector.update(level, i * 0.02, 0.02):
            fired.append(round(i * 0.02 + 0.02, 2))
    # First episode: 0.3 s loud, a 0.1 s gap, then loud until 0.5 s have passed since the onset.
    # Second: after a 0.2 s quiet the detector rearmed; another 0.5 s of sound fires again.
    assert fired == [0.5, 1.7]


def test_a_short_sound_does_not_fire() -> None:
    detector = SustainedVoice(hold_s=0.5)
    assert not any(detector.update(-10.0, i * 0.02, 0.02) for i in range(20))  # 0.4 s


def write_wav(path: Path, data: np.ndarray, rate: int, width: int, channels: int) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(width)
        wav.setframerate(rate)
        wav.writeframes(data.tobytes())


def test_a_stereo_48k_wav_becomes_16k_mono_s16(tmp_path: Path) -> None:
    rate = 48_000
    t = np.arange(rate) / rate
    left = (0.5 * 32767 * np.sin(2 * np.pi * 300 * t)).astype("<i2")
    stereo = np.stack([left, left], axis=1)
    path = tmp_path / "hello.wav"
    write_wav(path, stereo, rate, 2, 2)
    pcm = wav_to_pcm(path, 16_000)
    assert len(pcm) == 2 * 16_000
    assert level_dbfs(pcm) == pytest.approx(20 * math.log10(0.5 / math.sqrt(2)), abs=0.1)


def test_24_bit_and_8_bit_wavs_are_read(tmp_path: Path) -> None:
    ints = np.array([0, 1 << 22, -(1 << 22)], dtype=np.int32)
    raw = np.stack([ints & 0xFF, (ints >> 8) & 0xFF, (ints >> 16) & 0xFF], axis=1).astype(np.uint8)
    write_wav(tmp_path / "a.wav", raw, 16_000, 3, 1)
    assert np.frombuffer(wav_to_pcm(tmp_path / "a.wav", 16_000), "<i2").tolist() == [
        0,
        16384,
        -16384,
    ]
    write_wav(tmp_path / "b.wav", np.array([128, 192, 64], dtype=np.uint8), 16_000, 1, 1)
    assert np.frombuffer(wav_to_pcm(tmp_path / "b.wav", 16_000), "<i2").tolist() == [
        0,
        16384,
        -16384,
    ]


class NullSpeaker:
    def play(self, pcm: bytes) -> None:
        raise AssertionError("the idle pipeline must not speak")

    def play_end(self) -> None:
        raise AssertionError

    def flush(self) -> None:
        raise AssertionError


def test_the_voice_pipeline_stub_consumes_the_stream_and_does_nothing() -> None:
    frames = [AudioFrame(i, i * 0.02, b"\x00\x00" * 320) for i in range(25)]
    assert run_voice_pipeline(iter(frames), lambda: DoaReading(0.0, 90, True), NullSpeaker()) == 25
