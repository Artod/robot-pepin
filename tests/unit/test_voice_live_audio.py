"""The model's speech on its way to the board: resampled, paced, its amplitude announced."""

import wave
from pathlib import Path

import numpy as np
import pytest

from pepin.voice_live.audio import Pacer, Resampler, WavWriter, level, save_wav


def sine(hz: float, seconds: float, rate: int, amplitude: float = 0.5) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (amplitude * 32767 * np.sin(2 * np.pi * hz * t)).astype("<i2").tobytes()


def rms(pcm: bytes) -> float:
    x = np.frombuffer(pcm, "<i2").astype(float)[200:-200]
    return float(np.sqrt(np.mean(x * x)) / 32767)


def test_24k_to_16k_keeps_speech_removes_what_would_alias_and_ignores_the_cuts() -> None:
    speech = sine(440, 1.0, 24_000)
    whole = Resampler(24_000, 16_000).process(speech)
    assert len(whole) == 16_000 * 2
    assert rms(whole) == pytest.approx(0.5 / np.sqrt(2), rel=0.02)
    chunked = Resampler(24_000, 16_000)
    sizes = [222, 960, 14, 2000, 4798]
    out, i, k = b"", 0, 0
    while i < len(speech):
        out += chunked.process(speech[i : i + sizes[k % len(sizes)]])
        i += sizes[k % len(sizes)]
        k += 1
    assert out == whole  # a stream cut anywhere comes out the same
    above_nyquist = Resampler(24_000, 16_000).process(sine(10_000, 1.0, 24_000))
    assert rms(above_nyquist) < 0.005  # 10 kHz would fold to 6 kHz; it is filtered out


def test_equal_rates_pass_through() -> None:
    assert Resampler(16_000, 16_000).process(b"\x01\x02\x03\x04") == b"\x01\x02\x03\x04"


def test_level_maps_dbfs_to_zero_one() -> None:
    assert level(b"") == 0.0
    assert level(b"\x00\x00" * 640) == 0.0
    assert 0.8 < level(sine(440, 0.04, 16_000, 0.5)) < 0.9  # -9 dBFS


class FakeSpeaker:
    def __init__(self) -> None:
        self.played: list[bytes] = []
        self.ends = 0
        self.flushes = 0

    def play(self, pcm: bytes) -> None:
        self.played.append(pcm)

    def play_end(self) -> None:
        self.ends += 1

    def flush(self) -> None:
        self.flushes += 1


def test_the_pacer_keeps_lead_s_ahead_and_announces_each_block_when_it_plays() -> None:
    now = [100.0]  # binary fractions below: no float drift in the schedule
    speaker = FakeSpeaker()
    levels: list[float] = []
    pacer = Pacer(speaker, rate=16_000, lead_s=0.125, block_s=0.0625, tail_s=0.0,
                  on_level=levels.append, clock=lambda: now[0])  # fmt: skip
    pacer.feed(sine(440, 0.625, 16_000))  # 10 blocks
    pacer.step()
    assert len(speaker.played) == 3  # playing at +0, +0.0625, +0.125: within the lead
    assert len(levels) == 1  # the first one plays now
    assert pacer.speaking() and pacer.first_play_s == 100.0
    for _ in range(8):  # the thread's rhythm, to +0.25
        now[0] += 0.03125
        pacer.step()
    assert len(speaker.played) == 7  # up to the block playing at +0.375
    assert len(levels) == 5  # those playing at +0 .. +0.25
    pacer.end()
    while pacer.speaking():
        now[0] += 0.03125
        pacer.step()
    assert len(speaker.played) == 10 and speaker.ends == 1
    assert len(levels) == 10 and all(0.8 < v < 0.9 for v in levels)
    assert pacer.sent_s == pytest.approx(0.625)


def test_a_partial_block_waits_for_more_unless_the_turn_ended() -> None:
    now = [0.0]
    speaker = FakeSpeaker()
    pacer = Pacer(speaker, rate=16_000, lead_s=1.0, block_s=0.04, clock=lambda: now[0])
    pacer.feed(b"\x01\x00" * 100)  # 100 samples, less than a 640-sample block
    pacer.step()
    assert speaker.played == []
    pacer.end()
    pacer.step()
    assert len(speaker.played) == 1 and speaker.ends == 1


def test_a_flush_drops_the_queue_and_silences_the_speaker() -> None:
    now = [0.0]
    speaker = FakeSpeaker()
    pacer = Pacer(speaker, rate=16_000, lead_s=0.1, tail_s=0.0, clock=lambda: now[0])
    pacer.feed(sine(440, 2.0, 16_000))
    pacer.step()
    pacer.flush()
    assert speaker.flushes == 1 and not pacer.speaking()
    sent = len(speaker.played)
    now[0] += 1.0
    pacer.step()
    assert len(speaker.played) == sent


def test_a_gap_in_the_stream_restarts_the_clock_instead_of_bursting() -> None:
    now = [0.0]
    speaker = FakeSpeaker()
    pacer = Pacer(speaker, rate=16_000, lead_s=0.05, block_s=0.04, clock=lambda: now[0])
    pacer.feed(sine(440, 0.04, 16_000))
    pacer.step()
    now[0] += 5.0  # a tool ran; the next words arrive later
    pacer.feed(sine(440, 0.4, 16_000))
    pacer.step()
    assert len(speaker.played) == 1 + 2  # not ten: they play from now, at real time


def test_wav_writer_and_save_wav_write_readable_files(tmp_path: Path) -> None:
    writer = WavWriter(tmp_path / "a" / "x.wav", 24_000)
    writer.write(b"\x01\x00" * 2400)
    writer.write(b"")
    writer.write(b"\x02\x00" * 2400)
    assert writer.seconds == pytest.approx(0.2)
    writer.close()
    with wave.open(str(tmp_path / "a" / "x.wav")) as w:
        assert w.getframerate() == 24_000 and w.getnframes() == 4800
    save_wav(tmp_path / "b" / "y.wav", b"\x00\x00" * 160, 16_000)
    with wave.open(str(tmp_path / "b" / "y.wav")) as w:
        assert w.getnframes() == 160
