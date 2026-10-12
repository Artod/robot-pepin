"""The marker grammar: a sentence of the brain's stream as the voice's parts and the body's cues,
and where on the player's clock a cue's word starts."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import pytest

from pepin.expression.markers import (
    Cue,
    Marker,
    Part,
    Vocabulary,
    classify,
    cues,
    parse,
    part_length_s,
    sync_s,
)
from pepin.voice_cascade.loop import in_sentences
from pepin.voice_cascade.stages import Played

WORDS = Vocabulary(frozenset({"laugh", "greeting"}), frozenset({"smile", "sad"}))


def test_a_bare_intent_is_a_gesture_and_any_other_bare_name_a_style() -> None:
    assert classify("laugh", WORDS) == ("gesture", "laugh")
    assert classify(" Laugh ", WORDS) == ("gesture", "laugh")
    assert classify("whispering", WORDS) == ("style", "whispering")  # the grammar before markers
    assert classify("happy,   excited", WORDS) == ("style", "happy, excited")
    assert classify("шёпотом", WORDS) == ("style", "шёпотом")
    assert classify("smile", WORDS) == ("style", "smile")  # a face is spelled out: {face:smile}
    assert classify("", WORDS) == ("style", "")


def test_a_spelled_out_name_the_vocabulary_lacks_is_dropped() -> None:
    assert classify("gesture:laugh", WORDS) == ("gesture", "laugh")
    assert classify("face: smile", WORDS) == ("face", "smile")
    assert classify("Style : grumpy,  slow", WORDS) == ("style", "grumpy, slow")
    assert classify("style:", WORDS) == ("style", "")
    assert classify("gesture:wave", WORDS) is None and classify("face:grin", WORDS) is None
    parsed = parse("{gesture:wave} Привет, {face:grin} Артём.", WORDS)
    assert parsed.parts == (Part("Привет, Артём."),)
    assert parsed.dropped == ("{gesture:wave}", "{face:grin}")


def test_a_marker_belongs_to_the_word_after_it_and_is_never_said() -> None:
    (part,) = parse("Ну {laugh} да, это {face:sad}грустно.", WORDS).parts
    assert part.text == "Ну да, это грустно."
    assert [(m.channel, m.name, part.text[m.at :]) for m in part.markers] == [
        ("gesture", "laugh", "да, это грустно."),
        ("face", "sad", "грустно."),
    ]
    (end,) = parse("Вот и всё {laugh}", WORDS).parts
    assert end.text == "Вот и всё" and end.markers[0].at == len(end.text)  # at the part's end


def test_a_style_at_a_clause_boundary_cuts_the_sentence_into_two_voice_calls() -> None:
    parts = parse("Ну да, {style: whispering} все уже спят.", WORDS).parts
    assert parts == (Part("Ну да,"), Part("все уже спят.", "whispering"))
    parts = parse("{} Как обычно — {тихо} и тише.", WORDS).parts
    assert parts == (Part("Как обычно —", ""), Part("и тише.", "тихо"))


def test_a_style_anywhere_else_waits_for_the_next_boundary_or_sentence() -> None:
    parts = parse("Хм, ну {тихо} ладно, давай.", WORDS).parts
    assert parts == (Part("Хм, ну ладно,"), Part("давай.", "тихо"))
    parts = parse("Слово {в середине} и конец.", WORDS).parts
    assert parts == (Part("Слово и конец."), Part("", "в середине"))  # for the next sentence


def test_a_sentence_of_markers_alone_is_a_silent_part_or_only_a_style() -> None:
    laugh = Marker("gesture", "laugh", 0, "{laugh}")
    assert parse("{laugh}", WORDS).parts == (Part("", None, (laugh,)),)
    assert parse("{шёпотом}", WORDS).parts == (Part("", "шёпотом"),)
    assert parse("", WORDS).parts == ()


def test_an_unclosed_brace_is_left_to_the_voice_s_cleaning() -> None:
    (part,) = parse("Это {laugh не закрыт.", WORDS).parts
    assert part.text == "Это {laugh не закрыт." and not part.markers


def test_cues_carry_each_marker_s_share_of_its_part() -> None:
    (part,) = parse("Привет, {greeting} Артём!", WORDS).parts
    (cue,) = cues(4, 2, part)
    assert (cue.turn, cue.index, cue.marker.name) == (4, 2, "greeting")
    assert cue.share == pytest.approx(len("Привет, ") / len("Привет, Артём!"))
    assert cues(1, 0, Part("", None, (Marker("gesture", "laugh", 0),)))[0].share == 0.0


def test_a_cue_s_word_starts_at_its_share_of_the_part_on_the_player_s_clock() -> None:
    """While the part's audio is still being made its length is at least 0.07 s a character;
    once all of it is fed, the mark's own length."""
    cue = Cue(Marker("gesture", "laugh", 10), 1, 0, 0.5)
    assert part_length_s((2.0, 2.3), 20, spoken=False, char_s=0.07) == pytest.approx(1.4)
    assert sync_s(cue, (2.0, 2.3), 20, spoken=False, char_s=0.07) == pytest.approx(2.7)
    assert sync_s(cue, (2.0, 3.0), 20, spoken=True, char_s=0.07) == pytest.approx(2.5)
    assert sync_s(cue, (2.0, 4.0), 20, spoken=False, char_s=0.07) == pytest.approx(3.0)


def test_what_was_heard_counted_in_parts_is_counted_back_in_sentences() -> None:
    """The brain keeps its history by sentences; the player marks parts."""
    parts = [(0, "Ну да,"), (0, "все спят."), (1, "Ладно.")]
    assert in_sentences(parts, 2, Played(0, "Ну")) == Played(0, "Ну")
    assert in_sentences(parts, 2, Played(1, "все")) == Played(0, "Ну да, все")
    assert in_sentences(parts, 2, Played(2, "")) == Played(1, "")
    assert in_sentences(parts, 2, Played(3, "")) == Played(2, "")
    assert in_sentences([], 1, Played(0, "")) == Played(1, "")
