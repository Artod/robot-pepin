"""Words timed from a model's emissions, their vowels from the sound, the drawn-out vowel the tag
line spells, and the early read that keeps the timings beside its text (synthetic sound and
emissions, no model)."""

# ruff: noqa: RUF001, RUF002, RUF003 -- Russian speech is the data here, not a lookalike of Latin

from __future__ import annotations

import concurrent.futures
import threading

import numpy as np
import pytest

from pepin.audio_link import RATE
from pepin.voice_cascade import timing
from pepin.voice_cascade.config import ProsodyConfig
from pepin.voice_cascade.stages import Prosody, TimedWord, Transcript, Unit, Vowel
from pepin.voice_cascade.tagger import MIN_VOWELS, ProsodyTagger
from pepin.voice_cascade.transcriber import read_early, read_timed, transcript_of


def tone(seconds: float, hz: float = 150.0, level: float = 0.3) -> bytes:
    t = np.arange(round(RATE * seconds)) / RATE
    x = sum(np.sin(2 * np.pi * hz * k * t) / k for k in (1, 2, 3)) / 1.8
    return (level * 32767 * x).astype("<i2").tobytes()


def silence(seconds: float) -> bytes:
    return b"\x00\x00" * round(RATE * seconds)


def word(
    text: str, starts: list[float], end: float, vowels: list[tuple[float, float, int]]
) -> TimedWord:
    """A character model's word: one letter per start, the last lasting until ``end``."""
    ends = [*starts[1:], end]
    units = tuple(Unit(c, a, b) for c, a, b in zip(text, starts, ends, strict=True))
    return TimedWord(
        text, starts[0], end, units, True, None, tuple(Vowel(a, b, i) for a, b, i in vowels)
    )


# -- units and words -------------------------------------------------------------------------


def test_bpe_tokens_make_words_and_each_lasts_until_the_next_sound_or_the_quiet() -> None:
    pcm = tone(0.3) + silence(0.2) + tone(0.3)
    emissions = [
        (" при", 0.0, -0.1),
        ("н", 0.12, -0.2),
        ("тер", 0.2, -0.3),
        (".", 0.45, -0.4),  # punctuation: no sound, ends nothing
        (" ", 0.47, 0.0),
        (" А", 0.52, -0.5),
    ]
    first, second = timing.words_of(emissions, pcm, letters=False)
    assert (first.text, second.text) == ("принтер", "А")
    assert [u.text for u in first.units] == ["при", "н", "тер"] and not first.letters
    assert first.units[0].end == pytest.approx(0.12)  # the next token's frame
    assert first.end == pytest.approx(0.31, abs=0.015)  # the quiet after the tone, not 0.52
    assert first.logprob == pytest.approx(-1.0)  # its tokens' and its full stop's
    assert second.end == pytest.approx(0.8, abs=0.015) and second.vowels == ()


def test_a_character_model_spells_its_words_and_its_spaces_part_them() -> None:
    emissions = [("п", 0.0, None), ("р", 0.04, None), ("и", 0.08, None), (" ", 0.2, None),
                 ("а", 0.3, -0.1)]  # fmt: skip
    first, second = timing.words_of(emissions, tone(0.5), letters=True)
    assert (first.text, second.text) == ("при", "а") and first.letters
    assert [u.text for u in first.units] == ["п", "р", "и"]
    assert first.units[-1].end == pytest.approx(0.3)  # the space ends nothing: the next letter
    assert first.logprob is None and second.logprob == pytest.approx(-0.1)


def test_tracks_line_up_and_praat_hears_a_tone_as_voiced() -> None:
    pcm = silence(0.2) + tone(0.6) + silence(0.2)
    level, voiced = timing.loudness(pcm), timing.voicing(pcm)
    assert len(level) == len(voiced) == 99  # 10 ms steps over 20 ms windows of 1 s
    times = np.arange(len(voiced)) * timing.HOP_S + timing.WINDOW_S / 2
    assert voiced[(times > 0.3) & (times < 0.7)].all()
    assert not voiced[times < 0.15].any() and not voiced[times > 0.85].any()
    assert timing.voicing(silence(0.05)).sum() == 0


def test_vowels_are_the_loud_voiced_runs_named_by_the_nearest_vowel_letter() -> None:
    w = word("принтер", [0.0, 0.04, 0.08, 0.12, 0.16, 0.2, 0.24], 0.7, [])
    steps = 80
    times = np.arange(steps) * timing.HOP_S + timing.WINDOW_S / 2
    voiced = ((times > 0.06) & (times < 0.4)) | ((times > 0.5) & (times < 0.6))
    level = np.where(voiced, 0.0, -30.0)
    level[(times > 0.3) & (times < 0.4)] = -15.0  # a sonorant's tail, too quiet for the vowel
    first, second = timing.vowels_of(w, level, voiced, 0.0)
    assert (first.start, first.end) == pytest.approx((0.065, 0.305), abs=0.011)
    assert first.unit == 2  # и: the vowel letter emitted nearest its start
    assert (second.start, second.end) == pytest.approx((0.505, 0.595), abs=0.011)
    assert second.unit == 5  # е


def test_a_phrase_ends_where_the_next_word_waits_and_its_last_vowel_is_final() -> None:
    a = word("да", [0.0, 0.05], 0.3, [(0.05, 0.3, 1)])
    b = word("нет", [0.32, 0.36, 0.4], 0.6, [(0.36, 0.5, 1)])  # 20 ms later: same phrase
    c = word("ну", [0.9, 0.95], 1.2, [(0.95, 1.2, 1)])  # 0.3 s later, and the last
    assert timing.kinds([a, b, c]) == [["vowel"], ["final"], ["final"]]
    assert timing.samples([a, b, c]) == [
        ("vowel", pytest.approx(0.25)), ("final", pytest.approx(0.14)),
        ("final", pytest.approx(0.25)),
    ]  # fmt: skip
    assert timing.medians({"vowel": [0.1] * 9, "final": [0.12] * 10}, 10) == {"final": 0.12}


def test_a_vowel_twice_the_usual_and_long_enough_is_spelled_drawn_out() -> None:
    w = word(
        "принтер",
        [0.0, 0.04, 0.08, 0.5, 0.54, 0.58, 0.66],
        0.75,
        [(0.06, 0.46, 2), (0.58, 0.66, 5)],
    )
    usual = {"vowel": 0.1, "final": 0.13}
    (hit,) = timing.stretches([w], usual, ratio=2.0, min_s=0.25)
    assert hit.ratio == pytest.approx(4.0) and hit.spelled() == "приииинтер"
    assert timing.stretches([w], usual, ratio=2.0, min_s=0.5) == []  # under the floor
    assert timing.stretches([w], usual, ratio=5.0, min_s=0.25) == []  # under the ratio
    assert timing.stretches([w], {"final": 0.13}, ratio=2.0, min_s=0.25) == []  # no baseline
    long = timing.Stretch(w, Vowel(0.0, 1.0, 2), 9.0)
    assert long.spelled() == "прииииинтер"  # at most MAX_REPEAT letters
    assert timing.Stretch(w, Vowel(0.0, 1.0, None), 3.0).spelled() == "принтер"
    token = TimedWord("принтер", 0.0, 0.7, (Unit("при", 0.0, 0.4), Unit("нтер", 0.4, 0.7)), False)
    assert timing.Stretch(token, Vowel(0.0, 0.3, 0), 3.0).spelled() == "прииинтер"


def test_the_log_keeps_each_word_s_span_log_prob_and_vowels() -> None:
    w = word("да", [0.0, 0.05], 0.3, [(0.05, 0.3, 1)])
    assert timing.log_rows([w]) == [{"w": "да", "t": [0.0, 0.3], "lp": None, "v": [0.25]}]


# -- the tag line ----------------------------------------------------------------------------


def utterance(seconds: float) -> tuple[TimedWord, ...]:
    """'принтер': its и lasts ``seconds``, its е (the phrase's last vowel) 0.12 s."""
    i_end = 0.06 + seconds
    return (
        word(
            "принтер",
            [0.0, 0.04, 0.06, i_end, i_end + 0.04, i_end + 0.08, i_end + 0.2],
            i_end + 0.3,
            [(0.06, i_end, 2), (i_end + 0.08, i_end + 0.2, 5)],
        ),
    )


def test_the_tag_line_spells_a_drawn_out_vowel_once_this_person_has_usual_vowels() -> None:
    tagger = ProsodyTagger(ProsodyConfig(min_baseline=100))  # the numbers say nothing yet
    plain = Prosody(1.0, 60.0, words=utterance(0.1))
    drawn = Prosody(1.0, 60.0, words=utterance(0.4))
    assert tagger.line(drawn, "принтер") == ""  # no usual vowel yet
    for _ in range(MIN_VOWELS):
        tagger.line(plain, "принтер")
    assert tagger.line(drawn, "принтер") == "[Артём: растянуто: приииинтер]"
    assert tagger.describe(drawn, "принтер", "Саша") == ""  # Саша's own vowels: none yet
    off = ProsodyTagger(ProsodyConfig(min_baseline=100, stretch_ratio=None))
    for _ in range(MIN_VOWELS):
        off.line(plain, "принтер")
    assert off.line(drawn, "принтер") == ""


def test_the_stretch_joins_the_numbers_words() -> None:
    tagger = ProsodyTagger(ProsodyConfig(min_baseline=2))
    for _ in range(MIN_VOWELS):
        tagger.line(Prosody(1.0, 60.0, words=utterance(0.1)), "принтер")
    line = tagger.line(Prosody(1.0, 50.0, words=utterance(0.4)), "принтер")
    assert line == "[Артём: тише обычного, растянуто: приииинтер]"


# -- the early read --------------------------------------------------------------------------


class Timed:
    """A transcriber that times its words: one word, 'да'."""

    def __init__(self, gate: threading.Event | None = None) -> None:
        self.gate, self.calls = gate, 0

    def transcribe(self, pcm: bytes) -> str:
        return "Да."

    def transcribe_timed(self, pcm: bytes) -> Transcript:
        self.calls += 1
        if self.gate is not None:
            self.gate.wait(5)
        return Transcript("Да.", (word("да", [0.0, 0.05], 0.3, [(0.05, 0.3, 1)]),))


class Plain:
    def transcribe(self, pcm: bytes) -> str:
        return "нет"


def test_an_early_read_keeps_the_timings_beside_its_text() -> None:
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        reading = read_early(pool, Timed(), b"")
        assert reading.result(5) == "Да."
        heard = transcript_of(reading.result(), reading)
        assert heard.words[0].text == "да"
        assert transcript_of("Да.", pool.submit(lambda: "Да.")) == Transcript("Да.")
        late = read_timed(Plain(), b"")
        assert late == Transcript("нет") and transcript_of(late, None) is late


def test_an_early_read_dropped_before_it_starts_never_runs() -> None:
    gate = threading.Event()
    busy = Timed(gate)
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        first = read_early(pool, busy, b"")
        second = read_early(pool, busy, b"")
        assert second.cancel()
        gate.set()
        assert first.result(5) == "Да."
    assert busy.calls == 1 and second.cancelled()


def test_an_early_read_that_fails_hands_the_error_on() -> None:
    class Broken(Timed):
        def transcribe_timed(self, pcm: bytes) -> Transcript:
            raise RuntimeError("closed")

    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        reading = read_early(pool, Broken(), b"")
        with pytest.raises(RuntimeError, match="closed"):
            reading.result(5)
    assert reading.transcript is None
