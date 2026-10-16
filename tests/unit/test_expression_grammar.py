"""The brain's gesture grammar (gesticulation DESIGN.md §5): a mark with its argument, the accent,
the aliases, arguments in any script, what names nothing (dropped, never a voice style), a mark
repeated in one reply, and the persona that teaches it."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import re
from pathlib import Path

from pepin.arm_motion import load_arm_motion
from pepin.expression.animation import Scene
from pepin.expression.animator import EngineConfig, GestureEngine
from pepin.expression.book import face_names, load_book, mark_lines, vocabulary, with_rest
from pepin.expression.fakes import FakePlayout
from pepin.expression.markers import (
    Cue,
    Mark,
    Marker,
    Named,
    Part,
    argument,
    cues,
    parse,
    read,
    read_mark,
)
from pepin.expression.render import BookRenderer, Resolver
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.engines import MARKS_SLOT, persona_text

CONFIG = Path(__file__).resolve().parents[2] / "config"
BOOK = with_rest(
    load_book(CONFIG / "animation.json"), load_arm_motion(CONFIG / "arm.json").rest_deg
)
WORDS = vocabulary(BOOK, face_names(CONFIG / "face.json"))
EVERYTHING = Scene(frozenset({"arm", "head", "face", "body"}))
PRIMITIVES = ("accent", "point", "arc", "size", "weigh", "count", "draw", "stop", "offer")
MARK = re.compile(r"\{([^{}]*)\}")


def named(inner: str) -> Named | None:
    return read(inner, WORDS)


def test_a_mark_with_its_argument_names_the_intent_and_the_argument() -> None:
    assert named("size:big") == ("gesture", "size", "big")
    assert named(" Arc : UP ") == ("gesture", "arc", "up")
    assert named("count:2") == ("gesture", "count", "2")
    assert named("point:you") == ("gesture", "point", "you")
    assert named("point:printer") == ("gesture", "point", "printer")
    assert named("gesture:point:me") == ("gesture", "point", "me")
    (part,) = parse("Это было очень {size:big} огромное здание.", WORDS).parts
    assert part.text == "Это было очень огромное здание."
    assert part.markers == (Marker("gesture", "size", 15, "{size:big}", "big"),)
    (cue,) = cues(1, 0, part)
    assert cue.marker.arg == "big" and part.text[cue.word[0] : cue.word[1]] == "огромное"


def test_the_accent_and_the_other_spellings_are_read_as_the_book_s_canonical_marks() -> None:
    assert named("!") == ("gesture", "accent", None)
    assert named("beat") == named("emphasis") == named("Accent") == ("gesture", "accent", None)
    for written, meant in (
        ("point:i", ("point", "me")),
        ("point:yourself", ("point", "you")),
        ("arc:rise", ("arc", "up")),
        ("arc:left", ("arc", "side")),
        ("size:huge", ("size", "big")),
        ("size:tiny", ("size", "small")),
        ("draw:circle", ("draw", "round")),
        ("count:first", ("count", "1")),
    ):
        assert named(written) == ("gesture", *meant), written
    for spelling, target in BOOK.aliases.items():  # every alias reads as what it names
        name, _, arg = target.partition(":")
        assert named(spelling) == ("gesture", name, arg or None), spelling
    (part,) = parse("Это {!} очень важно.", WORDS).parts
    assert part.text == "Это очень важно." and part.markers[0].raw == "{!}"


def test_an_argument_in_any_script_is_a_name_lower_case_single_spaced() -> None:
    assert named("point:План  А") == ("gesture", "point", "план а")
    assert named("point:Ёлка") == ("gesture", "point", "елка")
    assert argument("  Всё  ещё ") == "все еще"
    assert named("point:" + "я" * 24) is not None and named("point:" + "я" * 25) is None
    assert named("point:") is None and named("point:-план") is None


def test_what_names_nothing_is_dropped_and_never_turns_the_voice() -> None:
    for inner in (
        "bow",  # a bare name of no intent
        "wave:hello",  # an x:y whose x is no intent
        "whispering:soft",  # ... even when x is a style word
        "arc:sideways",  # an argument the intent does not take
        "size:medium",
        "count:4",
        "point",  # an intent that needs one, without it
        "arc",
        "face:nope",
        "gesture:bow",
        "nod:fast",  # a bare intent with a word it does not take: dropped, never its first
    ):
        assert named(inner) is None, inner
    parsed = parse("{bow} Цены {arc:sideways} выросли {whispering:soft} вдвое.", WORDS)
    assert parsed.parts == (Part("Цены выросли вдвое."),)  # the voice's manner untouched
    assert parsed.dropped == ("{bow}", "{arc:sideways}", "{whispering:soft}")
    assert named("whispering") == ("style", "whispering", None)
    assert named("happy, excited") == ("style", "happy, excited", None)
    assert named("") == ("style", "", None)
    assert named("laugh") == ("gesture", "laugh", None) and named("face:smile") == (
        "face", "smile", None,
    )  # fmt: skip
    # an argument the intent does not take is left out, the mark plays (2026-10-14)
    assert read_mark("laugh:loud", WORDS) == Mark("gesture", "laugh", ignored=("loud",))


def test_the_same_mark_and_argument_twice_in_one_reply_plays_once() -> None:
    """feasibility #10b: a repeat is the same primitive AND argument; the ladder's own
    renderings of a mark (a variant given) are no repeat; pointing back and the accent are."""
    renderer = BookRenderer(BOOK, resolver=Resolver())

    def clips(name: str, arg: str | None, turn: int, part: int = 0) -> int:
        cue = Cue(Marker("gesture", name, 0, "", arg), turn, part, 0.0)
        return len(renderer.render(name, EVERYTHING, 7, cue).clips)

    assert clips("size", "big", 1) and not clips("size", "big", 1, 1)
    assert clips("size", "small", 1, 2)  # another argument
    assert clips("size", "big", 2)  # the next reply
    cue = Cue(Marker("gesture", "size", 0, "", "big"), 2, 0, 0.0)
    assert renderer.render("size", EVERYTHING, 7, cue, variant=0).clips  # a rung of the ladder
    assert clips("accent", None, 2) and clips("accent", None, 2, 1)
    assert clips("weigh", None, 3) and not clips("weigh", None, 3, 1)


# -- the persona ---------------------------------------------------------------------------------


def _persona() -> str:
    config = CascadeConfig.load()
    return persona_text((CONFIG / config.brain.persona_file).read_text(), 0, mark_lines(BOOK))


def test_the_persona_s_marks_are_the_book_s_and_its_examples_leave_the_arm_time() -> None:
    """Every mark the persona shows reads as the book's (none dropped): those in its prose and
    the generated vocabulary's lines (the book's, one a line, each with at most six words); each
    primitive and feeling the brain may write is listed with ``greeting`` and ``farewell`` (its
    own hello and bye since 2026-10-14), ``emphasis`` is not (the accent's); every example's mark
    stands at least 11 characters into its sentence or in a later sentence
    (critic_architecture #6: the arm's lead from rest at the text)."""
    persona = _persona()
    assert MARKS_SLOT not in persona
    lines = mark_lines(BOOK)
    assert "\n".join(lines) in persona
    shown = MARK.findall(persona)
    for inner in shown:
        assert named(inner) is not None, inner
    listed = set()
    for line in lines:
        spelled, _, words = line.partition(" — ")
        assert 1 <= len(words.split()) <= 6, line
        name, _, choose = spelled.partition("[")[0].partition(":")
        arg = choose.split("|")[0].replace("NAME", "printer") or None  # one that chooses
        assert named(name if arg is None else f"{name}:{arg}") == ("gesture", name, arg), line
        listed.add(name)
    assert set(PRIMITIVES) <= listed and set(PRIMITIVES) <= {named(i)[1] for i in shown} | listed  # type: ignore[index]
    feelings = {"surprise", "doubt", "shrug", "sympathy", "agreement", "disagreement",
                "thinking", "evil_click", "laugh", "snap", "greeting", "farewell"}  # fmt: skip
    assert feelings <= listed
    assert "emphasis" not in listed | set(shown)
    assert "{greeting}" in persona and "{farewell}" in persona  # the brain's to decide
    assert "happen by themselves" not in persona
    examples = re.findall(r"«([^»]*)»", persona)
    marked = [e for e in examples if any(named(m) and named(m)[0] == "gesture"  # type: ignore[index]
                                         for m in MARK.findall(e))]  # fmt: skip
    assert len(marked) >= 3
    for example in marked:
        for k, sentence in enumerate(re.split(r"(?<=[.!?])\s+", example)):
            for mark in MARK.finditer(sentence):
                if named(mark.group(1))[0] == "gesture":  # type: ignore[index]
                    assert k >= 1 or mark.start() >= 11, (example, mark.group(0))


def test_the_persona_the_brain_gets_is_the_same_with_the_gestures_switch_on_and_off() -> None:
    engine = GestureEngine({}, FakePlayout(), lambda: (BOOK, EngineConfig()))
    texts = []
    for on in (True, False, True):
        assert engine.switch(on) is None and engine.on is on
        texts.append(_persona())
    assert texts[0] == texts[1] == texts[2]
