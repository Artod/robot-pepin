"""The word clock: when a character of a reply is heard, from its part's TEXT through START to
its own AUDIO, a filler the reply waits behind, a burst of sentences, the running median and a
cut reply's parts forgotten."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

from dataclasses import dataclass, field

import pytest

from pepin.expression.fakes import FakeNuclei
from pepin.expression.forecast import (
    CHAR_S,
    SPREAD_ACROSS_S,
    SPREAD_AUDIO_S,
    ClockKnobs,
    QueuedAudio,
    WordClock,
    filler_wait_s,
    part_length_s,
)
from pepin.expression.schedule import Nucleus

KNOBS = ClockKnobs(char_s=0.09, tts_first_s=0.10, lead_silence_s=0.16, audio_lag_s=0.15)
TEXT = "Принтер вон там, у стены."


@dataclass
class Player:
    """The player as the clock reads it: its stream seconds played (still while nothing is
    queued, as the real playout clock), the parts' marks, and what it holds ahead."""

    now: float = 0.0
    marks: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)
    held: QueuedAudio = field(default_factory=lambda: QueuedAudio(0.0, 0.0, 0.0))

    def played(self) -> float:
        return self.now

    def mark(self, turn: int, index: int) -> tuple[float, float] | None:
        return self.marks.get((turn, index))

    def queued_audio(self) -> QueuedAudio:
        return self.held


def vowel_at(text: str, word: str) -> int:
    """The first vowel's offset of ``word`` in ``text``."""
    start = text.index(word)
    return start + next(i for i, c in enumerate(word) if c.lower() in "аеёиоуыэюя")


def test_the_three_moments_each_refine_the_last() -> None:
    player = Player(now=7.0)
    clock = WordClock(player, KNOBS, FakeNuclei())
    clock.queued(1, 0, TEXT, now=100.0)
    char = vowel_at(TEXT, "там")
    into = 0.16 + char / len(TEXT) * (0.09 * len(TEXT) - 0.16)
    text = clock.at(1, 0, char, 100.0)
    assert text is not None and text.stage == "text" and text.stream is None
    assert text.heard == pytest.approx(100.0 + 0.10 + into + 0.15)
    late = clock.at(1, 0, char, 100.4)  # no first chunk 0.4 s after TEXT: it slides with the clock
    assert late is not None and late.heard == pytest.approx(100.4 + into + 0.15)
    player.marks[(1, 0)] = (7.0, 7.5)  # START: the first chunk, half a second of it
    start = clock.at(1, 0, char, 100.4)
    assert start is not None and start.stage == "start" and start.stream is not None
    assert start.heard == pytest.approx(100.4 + into + 0.15)  # still the characters' length
    player.marks[(1, 0)] = (7.0, 9.0)
    clock.spoken(1, 0)  # all of it fed: 2.0 s, the exact length
    exact = 0.16 + char / len(TEXT) * (2.0 - 0.16)
    spoken = clock.at(1, 0, char, 100.4)
    assert spoken is not None and spoken.heard == pytest.approx(100.4 + exact + 0.15)
    assert spoken.spread <= 0.26
    word = (TEXT.index("там"), TEXT.index("там") + 3)
    nuclei = clock.nuclei
    assert isinstance(nuclei, FakeNuclei)
    stressed = 7.0 + exact + 0.05
    nuclei.found[(1, 0)] = [Nucleus(1, 0, 7.0 + exact - 0.3, 3.0), Nucleus(1, 0, stressed, 1.5),
                           Nucleus(1, 0, 8.9, 0.5)]  # fmt: skip
    audio = clock.at(1, 0, char, 100.4, word)
    assert audio is not None and audio.stage == "audio" and audio.spread == SPREAD_AUDIO_S
    assert audio.heard == pytest.approx(100.4 + (stressed - 7.0) + 0.15)  # the word's nucleus


def test_the_audio_moment_waits_until_the_nuclei_cover_the_word() -> None:
    player = Player(now=1.0, marks={(1, 0): (1.0, 1.4)})
    nuclei = FakeNuclei(found={(1, 0): [Nucleus(1, 0, 1.2, 2.0)]})
    clock = WordClock(player, KNOBS, nuclei)
    clock.queued(1, 0, TEXT, now=50.0)
    word = (TEXT.index("стены"), len(TEXT) - 1)
    forecast = clock.at(1, 0, vowel_at(TEXT, "стены"), 50.0, word)
    assert forecast is not None and forecast.stage == "start"  # the word's audio is not in yet


def test_a_filler_holds_the_reply_back_only_by_what_was_sent_and_its_fade() -> None:
    assert filler_wait_s(QueuedAudio(0.5, 0.1, 0.06)) == pytest.approx(0.16)
    assert filler_wait_s(QueuedAudio(0.05, 0.05, 0.06)) == pytest.approx(0.05)
    assert filler_wait_s(QueuedAudio(0.5, 0.1, 0.0)) == pytest.approx(0.5)  # waited for whole
    player = Player(held=QueuedAudio(0.5, 0.1, 0.06))
    clock = WordClock(player, KNOBS)
    clock.queued(1, 0, TEXT, now=10.0)
    bare = WordClock(Player(), KNOBS)
    bare.queued(1, 0, TEXT, now=10.0)
    with_filler, without = clock.at(1, 0, 3, 10.0), bare.at(1, 0, 3, 10.0)
    assert with_filler is not None and without is not None
    assert with_filler.heard - without.heard == pytest.approx(0.16)


def test_a_burst_of_sentences_places_each_part_after_the_one_before_it() -> None:
    player = Player(now=3.0)
    clock = WordClock(player, KNOBS)
    first, second = "Две новости.", "Во-первых, я научился махать."
    clock.queued(1, 0, first, now=20.0)
    clock.queued(1, 1, second, now=20.08)
    ahead = clock.at(1, 1, 0, 20.08)
    assert ahead is not None and ahead.stage == "text" and ahead.spread == SPREAD_ACROSS_S
    part0_end = 20.0 + 0.10 + 0.09 * len(first)
    assert ahead.heard == pytest.approx(part0_end + 0.16 + 0.15)
    player.marks[(1, 0)] = (3.0, 3.6)  # part 0 starts: part 1 follows its forecast end
    started = clock.at(1, 1, 0, 20.2)
    assert started is not None and started.stream == pytest.approx(3.0 + 0.09 * len(first) + 0.16)
    player.marks[(1, 0)] = (3.0, 4.4)
    clock.spoken(1, 0)  # its length is known: part 1's start is exact
    exact = clock.at(1, 1, 0, 20.2)
    assert exact is not None and exact.stream == pytest.approx(4.4 + 0.16)
    assert exact.spread < SPREAD_ACROSS_S


def test_the_running_median_replaces_char_s_once_three_parts_are_spoken() -> None:
    player = Player()
    clock = WordClock(player, KNOBS)
    assert clock.sigma() == CHAR_S
    for index, (text, length) in enumerate(
        (("Короткая.", 0.9), ("Да.", 1.0), ("Вторая фраза тут.", 1.7), ("Третья тоже.", 1.2))
    ):
        clock.queued(1, index, text)
        player.marks[(1, index)] = (index * 3.0, index * 3.0 + length)
        clock.spoken(1, index)
    assert clock.sigma() == pytest.approx(0.1)  # "Да." is too short to count


def test_a_cut_reply_s_parts_are_forgotten_and_never_placed_again() -> None:
    player = Player()
    clock = WordClock(player, KNOBS)
    clock.queued(4, 0, TEXT)
    clock.queued(5, 0, TEXT)
    clock.forget(4)
    assert clock.at(4, 0, 3, 0.0) is None and clock.at(5, 0, 3, 0.0) is not None
    player.marks[(4, 0)] = (0.0, 1.0)  # the flushed stream's seconds come back to another reply
    assert clock.at(4, 0, 3, 0.0) is None


def test_a_part_s_length_is_its_fed_audio_once_spoken_else_at_least_its_characters() -> None:
    assert part_length_s((2.0, 2.3), 20, spoken=False, sigma=0.09) == pytest.approx(1.8)
    assert part_length_s((2.0, 4.0), 20, spoken=False, sigma=0.09) == pytest.approx(2.0)
    assert part_length_s((2.0, 2.3), 20, spoken=True, sigma=0.09) == pytest.approx(0.3)
