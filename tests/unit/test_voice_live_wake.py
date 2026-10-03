"""The wake gate's text rules and its loudness segmenter (no model is loaded)."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import struct

import pytest

from pepin.audio_link import FRAME_SAMPLES
from pepin.voice_live.wake import Segmenter, is_goodbye, is_wake, rms_db, words


@pytest.mark.parametrize(
    "text",
    [
        "Пепин",
        "Хей, Пепин, где ты?",
        "Эй, Пипин!",
        "Пепен, привет",
        "Слушай, Пеппин, который час?",
        "Hey Pepin",
        "pippin, are you there",
        "ПЕПИН.",
        "Пепину скажи",
        "Степин, где ты?",  # noisy: the first syllable lost, "пин" kept
        "Привет, Кипин",
        "Чаепепин, как дела?",
    ],
)
def test_the_name_and_the_ways_whisper_hears_it_wake(text: str) -> None:
    assert is_wake(text)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "Репин написал эту картину",
        "Пепел упал на стол",
        "Дай мне пепси",
        "Это папин стол",
        "Крепин написал эту картину",
        "пин-код",
        "Пипетка лежит на полке",
        "Продолжение следует...",
        "how do I check battery voltage",
    ],
)
def test_near_misses_and_ordinary_speech_do_not(text: str) -> None:
    assert not is_wake(text)


@pytest.mark.parametrize(
    ("text", "bye"),
    [
        ("Пока!", True),
        ("Ну пока, Пепин", True),
        ("Спасибо, пока", True),
        ("Всё, спасибо.", True),
        ("Спасибо, всё", True),
        ("До свидания", True),
        ("Пока ты едешь, расскажи что-нибудь", False),
        ("Пока что не надо", False),
        ("Спасибо", False),
        ("Где ты?", False),
        ("", False),
    ],
)
def test_goodbye_is_a_farewell_not_the_word_poka(text: str, bye: bool) -> None:
    assert is_goodbye(text) is bye


def test_words_fold_case_yo_and_punctuation() -> None:
    assert words("Всё, ПЕПИН! ok?") == ["все", "пепин", "ok"]


def tone(db: float) -> bytes:
    """One frame at about ``db`` dBFS (a square wave of that RMS)."""
    amplitude = int(32768 * 10 ** (db / 20))
    return struct.pack(f"<{FRAME_SAMPLES}h", *([amplitude, -amplitude] * (FRAME_SAMPLES // 2)))


def test_rms_db_reads_the_level() -> None:
    assert rms_db(tone(-20.0)) == pytest.approx(-20.0, abs=0.1)
    assert rms_db(b"") == -120.0


def test_a_segment_is_the_speech_with_its_lead_in_and_ends_after_the_silence() -> None:
    gate = Segmenter(silence_s=0.1, pre_s=0.1, min_s=0.1)
    t = 0.0
    out = []
    frames = [tone(-60.0)] * 20 + [tone(-20.0)] * 10 + [tone(-60.0)] * 10
    for frame in frames:
        t += 0.02
        segment = gate.feed(frame, t)
        if segment is not None:
            out.append(segment)
    assert len(out) == 1
    segment = out[0]
    # 2 quiet lead-in frames (pre_s holds 5) + 10 loud + the 5 quiet that ended it
    assert segment.duration_s == pytest.approx(0.34, abs=1e-6)
    assert segment.ended_s == pytest.approx(0.6, abs=1e-6)  # the last loud frame
    assert not gate.talking


def test_a_long_segment_is_cut_at_max_s_and_a_blip_is_dropped() -> None:
    gate = Segmenter(silence_s=0.1, max_s=0.2, min_s=0.1)
    cut = [gate.feed(tone(-20.0), i * 0.02) for i in range(1, 30)]
    assert any(s is not None for s in cut)
    blip = Segmenter(silence_s=0.04, min_s=0.5, pre_s=0.02)
    results = [blip.feed(tone(-60.0), 0.0) for _ in range(5)]
    results += [blip.feed(tone(-20.0), 0.1) for _ in range(3)]
    results += [blip.feed(tone(-60.0), 0.2) for _ in range(5)]
    assert all(r is None for r in results)
