"""The offline barge-in evaluation: a recorded conversation's windows, the truth's rules, the
replay through the ears and the verdicts."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import pytest

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_cascade.barge_eval import (
    OfflineReader,
    Recording,
    Sound,
    Tally,
    Trigger,
    Window,
    frame_levels,
    ground_truth,
    is_speech,
    judge,
    replay,
    sound_events,
    spans,
    windows_of,
)
from pepin.voice_cascade.barge_in import BargeInDetector, LevelDetector
from pepin.voice_cascade.config import CascadeConfig

FRAME_S = FRAME_SAMPLES / RATE


def tone(seconds: float, amplitude: float = 0.3) -> bytes:
    t = np.arange(int(seconds * RATE)) / RATE
    return (amplitude * 32767 * np.sin(2 * np.pi * 300 * t)).astype("<i2").tobytes()


def hiss(seconds: float) -> bytes:
    rng = np.random.default_rng(0)
    return (rng.normal(0, 10, int(seconds * RATE))).astype("<i2").tobytes()  # ~ -70 dBFS


def write_wav(path: Path, pcm: bytes, rate: int) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)


def turn_row(turn: int, t: float, outcome: str, start_ms: int, end_ms: int) -> dict[str, object]:
    return {"event": "turn", "t": t, "turn": turn, "outcome": outcome, "reply": "Раз два три.",
            "ms": {"speaker_start": start_ms, "speaker_end": end_ms}}  # fmt: skip


def conversation(tmp_path: Path) -> Path:
    """Wake 1 s; a reply 2..4 s the person cut at 3.5 s; a reply 6..8 s heard to its end."""
    d = tmp_path / "cascade_120000"
    d.mkdir()
    rows = [
        {"event": "open", "t": 0.0, "wake": "name", "flags": {"barge_in": True}},
        {"event": "heard", "t": 0.0, "text": "Пепин", "seconds": 1.0},
        turn_row(1, 3.5, "interrupted", 1000, 2500),
        {"event": "heard", "t": 4.5, "text": "стоп", "seconds": 0.6},
        turn_row(2, 8.3, "committed", 3000, 5000),
    ]
    (d / "events.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    mic = tone(1.0) + hiss(3.2) + tone(0.6) + hiss(6.2)  # wake, quiet, the person at 3.2 s
    write_wav(d / "in.wav", mic, RATE)
    write_wav(d / "out.wav", tone(2.0, 0.1), 24000)
    return d


def test_a_recording_has_its_windows_on_the_mic_s_timeline(tmp_path: Path) -> None:
    rec = Recording.load(conversation(tmp_path))
    assert rec.seconds == pytest.approx(10.0, abs=0.02) and rec.barge_in and rec.wake == "name"
    first, second = rec.windows
    assert (first.turn, first.start, first.end) == (1, pytest.approx(2.0), pytest.approx(3.5))
    assert (second.start, second.end) == (pytest.approx(6.0), pytest.approx(8.0))
    assert first.replay_end() == pytest.approx(5.5) and second.replay_end() == pytest.approx(8.3)
    assert rec.voice_db == pytest.approx(-23.0, abs=0.5)
    assert rec.played(2.5, 2.6) == rec.voice_db and rec.played(4.5, 5.0) == -120.0
    levels = frame_levels(rec.pcm(3.0, 4.0))
    assert len(levels) == 50 and levels[15] > -15 and levels[5] < -60  # the person from 3.2 s


def test_a_window_needs_both_speaker_stamps() -> None:
    rows = [turn_row(1, 5.0, "committed", 1000, 2000),
            {"event": "turn", "t": 9.0, "turn": 2, "outcome": "error", "ms": {}}]  # fmt: skip
    assert windows_of(rows) == [Window(1, "committed", 3.7, 4.7, "Раз два три.")]


def test_a_replay_window_stops_before_the_next_reply() -> None:
    cut = Window(1, "interrupted", 0.0, 3.0, "")
    near = Window(2, "committed", 4.0, 6.0, "")
    far = Window(3, "interrupted", 20.0, 21.0, "")
    stops = [stop for _, stop in spans([far, cut, near])]
    assert stops == [pytest.approx(4.0 - 2 * FRAME_S), pytest.approx(6.3), pytest.approx(23.0)]


def test_loud_runs_join_across_short_gaps() -> None:
    levels = np.array([-70.0] * 10 + [-20.0] * 5 + [-70.0] * 10 + [-20.0] * 5 + [-70.0] * 20)
    levels[45] = -20.0  # one frame alone, 0.3 s on: dropped
    (run,) = sound_events(levels, 1.0, -70.0)
    assert (run.start, run.end) == (pytest.approx(1.2), pytest.approx(1.6))


def test_the_truth_s_speech_rule() -> None:
    sure = {"no_speech": 0.0, "logprob": -0.3}
    assert is_speech(Sound(1.0, 1.6, -20.0, "Покажи температуру.", **sure), "")
    assert not is_speech(Sound(1.0, 1.6, -20.0, "Продолжение следует...", **sure), "")
    assert not is_speech(Sound(1.0, 1.6, -20.0, "Угу.", **sure), "")
    assert not is_speech(Sound(1.0, 1.1, -20.0, "Покажи.", **sure), "")  # too short for it
    assert not is_speech(Sound(1.0, 1.6, -20.0, "Покажи.", no_speech=0.0, logprob=-3.0), "")
    reply = "Вывел температуры сервоприводов на экран."
    assert not is_speech(Sound(1.0, 1.6, -45.0, "сервоприводов на экран", **sure), reply)
    assert is_speech(Sound(1.0, 1.6, -15.0, "сервоприводов на экран", **sure), reply)


def test_verdicts_by_where_the_trigger_falls() -> None:
    w = Window(1, "interrupted", 0.0, 3.0, "")
    knock = Sound(1.0, 1.2, -10.0, "Спасибо.")
    voice = Sound(2.0, 2.8, -15.0, "стоп", speech=True)
    sounds = [knock, voice]
    caught = judge(w, sounds, Trigger(2.3, "level"))
    assert caught.outcome == "caught" and caught.reaction_s == pytest.approx(0.3)
    assert judge(w, sounds, Trigger(3.4, "words")).outcome == "caught"  # Whisper's late answer
    assert judge(w, sounds, Trigger(1.15, "level")).outcome == "false"
    assert judge(w, sounds, Trigger(0.5, "level")).outcome == "false"
    assert judge(w, sounds, None).outcome == "missed"
    assert judge(w, [knock], None).outcome == "quiet"
    tally = Tally()
    tally.add(caught, True)
    tally.add(judge(w, sounds, Trigger(1.15, "level")), True)
    tally.add(judge(w, [knock], None), False)
    assert tally.cell() == "1/2 1F 300ms"


def test_the_replay_plays_each_window_and_keeps_its_first_barge_in(tmp_path: Path) -> None:
    rec = Recording.load(conversation(tmp_path))
    own: list[str] = []

    def make(own_text: object) -> BargeInDetector:
        own.append("made")
        return LevelDetector(over_floor_db=25.0, min_dbfs=-45.0, hold_s=0.25)

    triggers = replay(rec, CascadeConfig(), make)
    assert own == ["made"] and list(triggers) == [1]
    assert triggers[1].t == pytest.approx(3.2 + 0.26, abs=0.03)
    assert triggers[1].why.startswith("level")


def test_the_truth_reads_each_loud_moment_once(tmp_path: Path) -> None:
    rec = Recording.load(conversation(tmp_path))
    asked: list[float] = []

    def read(pcm: bytes) -> tuple[str, float, float]:
        asked.append(len(pcm) / 2 / RATE)
        return "Стоп.", 0.01, -0.2

    truth = ground_truth(rec, read)
    (voice,) = truth[1]
    assert voice.speech and voice.start == pytest.approx(3.2, abs=0.03)
    assert truth[2] == [] and asked == [pytest.approx(0.6 + 2 * 0.3, abs=0.05)]


def test_an_offline_reader_answers_when_whisper_would_have() -> None:
    times = iter([10.0, 10.3])
    reader = OfflineReader(lambda pcm: "стоп", clock=lambda: next(times))
    pending = reader(b"", 5.0)
    assert reader.ms == [pytest.approx(300.0)]
    assert pending.result(5.29) is None and pending.result(5.31) == "стоп"
    fixed = OfflineReader(lambda pcm: "", fixed_ms=250.0, clock=lambda: 0.0)
    assert fixed(b"", 1.0).result(1.25) == ""
