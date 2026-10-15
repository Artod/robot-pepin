"""The barge-in detectors: loudness, a voice-activity model with the echo guard, Whisper's words;
the ears acting on them, and the player's level history they predict the echo from."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import concurrent.futures
import hashlib
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_cascade.barge_in import (
    NO_ECHO_DB,
    SILERO_V6,
    BargeInDetector,
    Frame,
    FuturePending,
    LevelDetector,
    Samples,
    SpeechSeries,
    VadDetector,
    WordsDetector,
    build_detector,
    echo_level,
    interrupting_words,
    model_path,
    own_echo,
    speech_series,
)
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.ears import Ears
from pepin.voice_cascade.player import Playback, PlayoutClock
from pepin.voice_live.wake import rms_db

FRAME_S = FRAME_SAMPLES / RATE


def tone(seconds: float, amplitude: float = 0.3) -> list[bytes]:
    t = np.arange(int(seconds * RATE)) / RATE
    pcm = (amplitude * 32767 * np.sin(2 * np.pi * 300 * t)).astype("<i2").tobytes()
    step = FRAME_SAMPLES * 2
    return [pcm[i : i + step] for i in range(0, len(pcm) - step + 1, step)]


def quiet(seconds: float) -> list[bytes]:
    return [b"\x00\x00" * FRAME_SAMPLES] * round(seconds / FRAME_S)


def frames(pcms: list[bytes], t0: float = 0.0, floor_db: float = -70.0) -> list[Frame]:
    return [Frame(p, t0 + i * FRAME_S, rms_db(p), floor_db) for i, p in enumerate(pcms)]


def first_trigger(
    detector: BargeInDetector, stream: list[Frame], speaking: bool = True
) -> float | None:
    for frame in stream:
        if detector.feed(frame, speaking) is not None:
            return frame.t
    return None


class LoudIsSpeech:
    """A fake voice-activity model: a chunk is speech when it is loud and ``speech`` says so."""

    chunk = 512

    def __init__(self) -> None:
        self.speech = True
        self.calls = 0

    def __call__(self, samples: Samples) -> float:
        self.calls += 1
        loud = float(np.sqrt(np.mean(samples**2))) > 0.01
        return 0.95 if loud and self.speech else 0.02


class Later:
    """A fake transcript: ready ``delay_s`` after it was asked for."""

    def __init__(self, text: str, ready_s: float) -> None:
        self.text, self.ready_s, self.cancelled = text, ready_s, False

    def result(self, t: float) -> str | None:
        return self.text if t >= self.ready_s else None

    def cancel(self) -> None:
        self.cancelled = True


class Whisper:
    """A fake reader: answers in turn from ``texts``, ``delay_s`` after each call."""

    def __init__(self, *texts: str, delay_s: float = 0.25) -> None:
        self.texts, self.delay_s = list(texts), delay_s
        self.asked: list[tuple[float, float]] = []  # (time, seconds of audio)

    def __call__(self, pcm: bytes, t: float) -> Later:
        self.asked.append((t, len(pcm) / 2 / RATE))
        text = self.texts.pop(0) if self.texts else ""
        return Later(text, t + self.delay_s)


# -- level ----------------------------------------------------------------------------------


def test_the_level_detector_holds_a_quarter_second_of_loud_frames() -> None:
    level = LevelDetector(over_floor_db=25.0, min_dbfs=-45.0, hold_s=0.25)
    assert level.threshold_db(-70.0) == -45.0 and level.threshold_db(-60.0) == -35.0
    stream = frames(quiet(0.5) + tone(0.5), t0=0.0)
    assert first_trigger(level, stream) == pytest.approx(0.5 + 0.24, abs=1e-6)
    assert first_trigger(level, frames(tone(0.5)), speaking=False) is None


def test_a_dip_between_syllables_costs_half_a_frame() -> None:
    level = LevelDetector(over_floor_db=25.0, min_dbfs=-45.0, hold_s=0.25)
    syllables = (tone(0.1) + quiet(0.04)) * 3
    assert first_trigger(level, frames(syllables)) is not None
    knock = tone(0.12) + quiet(0.5)
    assert (
        first_trigger(LevelDetector(over_floor_db=25, min_dbfs=-45, hold_s=0.25), frames(knock))
        is None
    )


# -- vad ------------------------------------------------------------------------------------


def test_the_vad_detector_ignores_a_loud_sound_that_is_not_speech() -> None:
    model = LoudIsSpeech()
    vad = VadDetector(model, threshold=0.5, hold_s=0.2, min_dbfs=-50.0)
    model.speech = False  # a door: loud, not speech
    assert first_trigger(vad, frames(tone(1.0))) is None
    model.speech = True
    t = first_trigger(vad, frames(tone(1.0), t0=2.0))
    assert t is not None and 2.2 <= t <= 2.3
    fed_s = 1.0 + (t - 2.0) + FRAME_S
    assert model.calls == int(fed_s * RATE) // 512  # every whole 32 ms chunk, once


def test_the_vad_detector_needs_a_level_and_clears_of_the_echo() -> None:
    model = LoudIsSpeech()
    faint = VadDetector(model, min_dbfs=-20.0)  # the tone sits at about -13 dBFS
    assert first_trigger(faint, frames(tone(1.0, amplitude=0.05))) is None
    loud_echo = VadDetector(model, echo=lambda t: -20.0, echo_margin_db=10.0)
    assert first_trigger(loud_echo, frames(tone(1.0))) is None  # within 10 dB of the echo
    no_echo = VadDetector(model, echo=lambda t: NO_ECHO_DB, echo_margin_db=10.0)
    assert first_trigger(no_echo, frames(tone(1.0))) is not None


def test_the_vad_detector_keeps_each_chunk_s_probability_on_the_laptop_clock() -> None:
    """The speaker gaze tags the array's readings with these (pepin.expression.doa): every 32 ms
    chunk's probability, stamped with the laptop time of its last sample (a frame's time is its
    last sample's), the last keep_s of them; a words detector's vad gate keeps its own, a level
    detector none."""
    vad = VadDetector(LoudIsSpeech())
    for frame in frames(quiet(1.0) + tone(1.0), t0=10.0):
        vad.voiced(frame)
    series = speech_series(vad)
    assert series is vad.series
    assert series.peak(10.0, 10.9) == pytest.approx(0.02) and series.peak(11.1, 11.9) == 0.95
    first = 10.0 + (512 - FRAME_SAMPLES) / RATE  # the first chunk ends 192 samples into frame 1
    assert series.peak(9.9, first - series.chunk_s - 1e-6) is None
    assert series.peak(9.9, first - series.chunk_s + 1e-6) == pytest.approx(0.02)
    assert series.peak(12.1, 13.0) is None, "nothing heard after the stream"
    short = SpeechSeries(keep_s=0.32)  # ten chunks
    for k in range(20):
        short.add(float(k), 0.1 * (k % 10))
    assert short.peak(0.0, 9.5) is None and short.peak(10.0, 19.0) == pytest.approx(0.9)
    base = CascadeConfig()
    words = build_detector(
        base.with_changes(barge_in_detector="words"), reader=Whisper(), vad=LoudIsSpeech()
    )
    assert isinstance(words, WordsDetector) and isinstance(words.gate, VadDetector)
    assert speech_series(words) is words.gate.series
    assert speech_series(build_detector(base)) is None and speech_series(None) is None


def test_the_echo_is_the_played_level_a_lag_earlier_less_the_loss() -> None:
    asked: list[tuple[float, float]] = []

    def played(t0: float, t1: float) -> float:
        asked.append((t0, t1))
        return -20.0 if t0 < 5.0 else NO_ECHO_DB

    echo = echo_level(played, lag_s=0.1, loss_db=40.0, spread_s=0.1)
    assert echo(5.0) == -60.0 and asked[-1] == pytest.approx((4.8, 5.0))
    assert echo(5.3) == NO_ECHO_DB


# -- words ----------------------------------------------------------------------------------


def words_detector(reader: Whisper, own: str = "", **knobs: float) -> WordsDetector:
    gate = LevelDetector(over_floor_db=25.0, min_dbfs=-45.0, hold_s=0.25)
    return WordsDetector(gate, reader, lambda: own, **knobs)


def test_a_real_word_cuts_in_once_whisper_has_read_it() -> None:
    reader = Whisper("Стоп, подожди.")
    words = words_detector(reader, listen_s=0.4)
    stream = frames(quiet(1.0) + tone(1.0))
    t = first_trigger(words, stream)
    assert len(reader.asked) == 1
    asked_t, audio_s = reader.asked[0]
    assert asked_t == pytest.approx(1.0 + 0.38, abs=1e-6)  # 0.4 s of voice
    assert audio_s == pytest.approx(0.4 + 0.3, abs=0.03)  # the voice and 0.3 s before it
    assert t == pytest.approx(asked_t + 0.26, abs=0.011)


def test_noise_lines_backchannels_and_the_robot_s_own_words_do_not_cut_in() -> None:
    own = "Я Пепин, маленький домашний робот на колёсах."
    reader = Whisper("Продолжение следует...", "Угу.", "маленький домашний робот", "")
    words = words_detector(reader, own=own, every_s=0.3)
    assert first_trigger(words, frames(tone(2.0))) is None
    # a new call while the voice holds, one at a time, at most one every 0.3 s, at most 1.5 s
    assert [round(t, 2) for t, _ in reader.asked] == [0.38, 0.68, 0.98, 1.28, 1.58, 1.88]
    assert max(seconds for _, seconds in reader.asked) == pytest.approx(1.5)


def test_the_words_detector_drops_its_call_when_the_robot_stops() -> None:
    reader = Whisper("стоп")
    words = words_detector(reader)
    stream = frames(tone(0.5))
    assert first_trigger(words, stream) is None
    words.feed(frames(quiet(0.02), t0=0.6)[0], speaking=False)
    assert first_trigger(words, frames(quiet(1.0), t0=0.62)) is None


def test_which_words_take_the_turn() -> None:
    own = "Вывел температуры сервоприводов на экран."
    assert interrupting_words("Покажи температуру.", own) == ["покажи", "температуру"]
    assert interrupting_words("температуры сервоприводов", own) == []
    assert interrupting_words("Угу. Да.", own) == []
    assert interrupting_words("Субтитры сделал DimaTorzok", own) == []
    assert interrupting_words("Хе-хе-хе. Гррр! Ух!", own) == []  # laughter, a growl
    assert interrupting_words("Эй, а сердечко?", own) == ["эй", "а", "сердечко"]
    assert own_echo(["экраны"], own) is False  # one word proves nothing
    assert own_echo(["на", "экране"], own) is True


def test_a_future_reads_as_pending_until_done_and_a_failure_as_no_words() -> None:
    future: concurrent.futures.Future[str] = concurrent.futures.Future()
    pending = FuturePending(future)
    assert pending.result(0.0) is None
    future.set_result("стоп")
    assert pending.result(0.0) == "стоп"
    failed: concurrent.futures.Future[str] = concurrent.futures.Future()
    failed.set_exception(RuntimeError("no GPU"))
    assert FuturePending(failed).result(0.0) == ""


# -- building and the ears ------------------------------------------------------------------


def test_the_config_names_the_detector() -> None:
    base = CascadeConfig()
    assert isinstance(build_detector(base), LevelDetector)
    vad = build_detector(base.with_changes(barge_in_detector="vad"), vad=LoudIsSpeech())
    assert isinstance(vad, VadDetector) and vad.echo is None
    guarded = build_detector(
        base.with_changes(barge_in_detector="vad"), vad=LoudIsSpeech(), played=lambda a, b: -20.0
    )
    assert isinstance(guarded, VadDetector) and guarded.echo is not None
    words = build_detector(
        base.with_changes(barge_in_detector="words"), reader=Whisper(), vad=LoudIsSpeech()
    )
    assert isinstance(words, WordsDetector) and isinstance(words.gate, VadDetector)
    with pytest.raises(ValueError, match="transcriber"):
        build_detector(base.with_changes(barge_in_detector="words"), vad=LoudIsSpeech())
    with pytest.raises(ValueError, match=r"barge_in\.detector"):
        base.with_changes(barge_in_detector="loudness")


def test_the_ears_cut_in_on_the_detector_s_word_with_its_preroll() -> None:
    reader = Whisper("стоп")
    config = CascadeConfig().with_changes(barge_in=True, silence_s=0.5)
    words = config.with_changes(barge_in_detector="words", barge_in_words_gate="level")
    detector = build_detector(words, reader=reader)
    mode: list[Playback] = ["reply"]
    ears = Ears(config, lambda: mode[0], detector=detector)
    events = []
    for i, pcm in enumerate(quiet(1.0) + tone(1.5)):
        events += ears.feed(pcm, i * FRAME_S)
    kinds = [e.kind for e in events]
    assert kinds[:2] == ["barge_in", "start"]
    cut = events[0]
    assert cut.why == "words стоп" and cut.t == pytest.approx(1.0 + 0.38 + 0.26, abs=0.011)
    assert events[1].t == pytest.approx(1.0 + 0.04, abs=0.03)  # heard from the voice's start


# -- the player's level history -------------------------------------------------------------


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Sink:
    def play(self, pcm: bytes) -> None:
        pass

    def play_end(self) -> None:
        pass

    def flush(self) -> None:
        pass


def test_the_playout_clock_keeps_each_block_s_level_until_a_flush() -> None:
    clock = Clock()
    playout = PlayoutClock(Sink(), RATE, clock)
    loud, soft = tone(0.5, amplitude=0.3), tone(0.5, amplitude=0.003)
    playout.play(b"".join(loud))  # 100.0 .. 100.5
    playout.play(b"".join(soft))  # 100.5 .. 101.0
    assert playout.level_between(100.1, 100.2) == pytest.approx(-13.5, abs=0.5)
    assert playout.level_between(100.6, 100.7) == pytest.approx(-53.5, abs=0.5)
    assert playout.level_between(100.4, 100.6) == pytest.approx(-13.5, abs=0.5)
    assert playout.level_between(99.0, 99.9) == NO_ECHO_DB
    clock.now = 100.2
    playout.flush()
    assert playout.level_between(100.3, 101.0) == NO_ECHO_DB  # never played
    assert playout.level_between(100.1, 100.15) == pytest.approx(-13.5, abs=0.5)


def test_the_v6_model_is_fetched_once_and_checked(tmp_path: Path) -> None:
    assert model_path(None) is None and model_path("silero-v5") is None
    assert model_path("~/vad.onnx") == Path("~/vad.onnx").expanduser()
    with pytest.raises(ValueError, match="SHA-256"):
        model_path("silero-v6", tmp_path, get=lambda url: b"not the model")
    fetched: list[str] = []

    def get(url: str) -> bytes:
        fetched.append(url)
        return b"model"

    pinned = (SILERO_V6[0], hashlib.sha256(b"model").hexdigest())
    with mock.patch("pepin.voice_cascade.barge_in.SILERO_V6", pinned):
        first = model_path("silero-v6", tmp_path, get=get)
        again = model_path("silero-v6", tmp_path, get=get)
    assert first == again and first is not None and first.read_bytes() == b"model"
    assert fetched == [SILERO_V6[0]]
