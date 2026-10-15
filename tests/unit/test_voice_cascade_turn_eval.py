"""The end-of-turn bench: turns out of a conversation's record, pauses inserted, early reads on
the replay's clock, and the tally of cuts and closes."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import numpy as np
import pytest

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_cascade.barge_eval import frame_levels
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.turn_eval import (
    Clip,
    Close,
    Result,
    Tally,
    TimedReader,
    boundaries,
    fragments_of,
    loop_to,
    replay,
    run_silence_s,
    text_rows,
    text_table,
    turns_of,
    voice_end,
    with_pause,
)
from pepin.voice_cascade.turn_text import RulesJudge

FRAME_S = FRAME_SAMPLES / RATE
FRAME_BYTES = FRAME_SAMPLES * 2


def tone(seconds: float) -> bytes:
    t = np.arange(round(seconds / FRAME_S) * FRAME_SAMPLES) / RATE
    return (0.3 * 32767 * np.sin(2 * np.pi * 300 * t)).astype("<i2").tobytes()


def hiss(seconds: float, db: float = -90.0) -> bytes:
    rng = np.random.default_rng(0)
    n = round(seconds / FRAME_S) * FRAME_SAMPLES
    return (rng.standard_normal(n) * 32768 * 10 ** (db / 20)).astype("<i2").tobytes()


def clip_of(*parts: bytes, end_s: float, text: str = "") -> Clip:
    return Clip("c", b"".join(parts), -85.0, end_s, text)


def test_fragments_and_turns_come_from_the_record() -> None:
    rows = [
        {"event": "heard", "t": 0.0, "seconds": 2.0, "text": "Пепин", "whisper_ms": 3},
        {"event": "start", "t": 4.0},
        {"event": "heard", "t": 5.505, "seconds": 2.5, "text": "Как у тебя", "whisper_ms": 5},
        {"event": "start", "t": 5.9},  # 0.4 s after the close: the loop joined them
        {"event": "heard", "t": 7.0, "seconds": 1.4, "text": "дела?", "whisper_ms": 0},
        {"event": "heard", "t": 8.0, "seconds": 0.5, "text": "", "noise": True},
        {"event": "start", "t": 12.0},
        {"event": "heard", "t": 13.0, "seconds": 2.0, "text": "Ладно.", "whisper_ms": 0},
    ]
    fragments = fragments_of(rows, 0.4)
    assert [f.text for f in fragments] == ["Как у тебя", "дела?", "Ладно."]
    first = fragments[0]
    assert (first.closed, first.end, first.start, first.opened) == pytest.approx(
        (5.5, 5.1, 3.0, 4.0)
    )
    turns = turns_of(fragments, 0.8)
    assert [[f.text for f in turn] for turn in turns] == [["Как у тебя", "дела?"], ["Ладно."]]


def test_the_runs_silence_and_a_model_closed_run_is_left_out() -> None:
    turn = {"event": "turn", "kind": "person", "ms": {"closed": 400}}
    assert run_silence_s([turn, turn]) == 0.4
    model = {**turn, "eot": {"closed_by": "model"}}
    assert run_silence_s([turn, model]) is None


def test_the_voice_ends_at_the_last_loud_frame() -> None:
    pcm = hiss(0.5) + tone(1.0) + hiss(0.3) + tone(0.2) + hiss(1.0)
    assert voice_end(pcm, -85.0, 10.0, None, 10.0) == pytest.approx(2.0, abs=1e-6)
    assert voice_end(pcm, -85.0, 10.0, None, 1.7) == pytest.approx(1.5, abs=1e-6)  # limited
    assert voice_end(hiss(1.0), -85.0, 10.0, None, 1.0) is None


def test_a_pause_is_the_clips_own_quiet_and_moves_the_truth() -> None:
    clip = clip_of(hiss(0.5), tone(1.0), hiss(1.0), end_s=1.5)
    paused = with_pause(clip, 1.0, 0.5, start_db=10.0)
    assert paused.seconds == pytest.approx(clip.seconds + 0.5)
    assert paused.end_s == pytest.approx(2.0) and paused.pause_at_s == 1.0
    levels = frame_levels(paused.pcm)
    inserted = levels[round(1.0 / FRAME_S) : round(1.5 / FRAME_S)]
    assert (inserted < -85.0 + 10.0).all()
    assert len(loop_to(b"", 0.1)) == round(0.1 / FRAME_S) * FRAME_BYTES


def test_boundaries_sit_before_a_word_in_the_quiet_between() -> None:
    clip = clip_of(hiss(0.3), tone(0.5), hiss(0.06), tone(0.5), hiss(0.06), tone(0.5),
                   hiss(1.0), end_s=2.02)  # fmt: skip
    words = [("раз", 0.3), ("два", 0.86), ("три", 1.44)]
    places = boundaries(clip, words)
    assert len(places) == 2
    assert places[0] == pytest.approx(0.8, abs=0.05) and places[1] == pytest.approx(1.36, abs=0.05)
    assert boundaries(clip, words[:1]) == []


def test_an_early_read_lands_on_the_replays_clock_and_is_read_once() -> None:
    calls: list[int] = []

    def transcribe(pcm: bytes) -> str:
        calls.append(len(pcm))
        return "слово"

    reader = TimedReader(transcribe, read_ms=60)
    reader.advance(0.10)
    future = reader(b"\x01\x00" * 100)
    reader.advance(0.14)
    assert not future.done()
    reader.advance(0.16)
    assert future.result() == "слово"
    dropped = reader(b"\x01\x00" * 100)
    dropped.cancel()
    reader.advance(1.0)
    assert dropped.cancelled() and calls == [200]  # the same audio was read once
    assert reader.log == [(0.10, "слово"), (0.16, "слово")]


def test_a_replay_counts_the_cuts_the_merges_and_the_close() -> None:
    # 1 s of voice, 0.3 s of quiet, 1 s more: silence 0.25 cuts there, 0.4 does not
    clip = clip_of(hiss(0.5), tone(1.0), hiss(0.3), tone(1.0), hiss(1.5), end_s=2.8)
    short = replay(clip, CascadeConfig().with_changes(silence_s=0.25))
    (cut,) = short.cuts()
    assert cut.ended_s == pytest.approx(1.5, abs=0.03) and short.merged(cut, 0.8)
    assert short.close_ms() == pytest.approx(260, abs=25)
    base = replay(clip, CascadeConfig().with_changes(silence_s=0.4))
    assert base.cuts() == [] and base.close_ms() == pytest.approx(400, abs=25)
    tally = Tally()
    tally.add(short, 0.8)
    tally.add(base, 0.8)
    assert (tally.clips, tally.cut_clips, tally.cuts, tally.merged) == (2, 1, 1, 1)
    assert "1/2" in tally.row() and "1 merged, 0 split" in tally.row()


def test_a_cut_the_voice_does_not_come_back_to_soon_is_split() -> None:
    result = Result(clip_of(end_s=3.0), [Close(1.4, 1.0), Close(3.4, 3.0)], [0.2, 2.5])
    (cut,) = result.cuts()
    assert not result.merged(cut, 0.8) and result.final() == Close(3.4, 3.0)


def test_a_clip_never_closed_is_counted_as_such() -> None:
    tally = Tally()
    tally.add(Result(clip_of(end_s=1.0), [], [0.1]), 0.8)
    assert tally.unclosed == 1 and tally.close_ms == []


def test_the_text_table_places_reads_against_the_truth() -> None:
    clip = Clip("c@1.00+0.50", b"", -85.0, 3.0, "", pause_at_s=1.0, pause_s=0.5)
    result = Result(clip, [], [], reads=[(1.15, "Дай мне"), (3.15, "Дай мне текст."),
                                          (3.4, "Дай мне текст.")])  # fmt: skip
    rows = text_rows([result], {"rules": RulesJudge()})
    assert [(r.where, r.text) for r in rows] == [("pause", "Дай мне"), ("end", "Дай мне текст.")]
    lines = text_table(rows, ["rules"])
    assert "1/0/0 of 1" in lines[1] and "0/1/0 of 1" in lines[1]
