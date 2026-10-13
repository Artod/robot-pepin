"""The cascade's seam for the body: markers kept whole by the sentence splitter, a part's place
on the player's clock, and the control file's live flags."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import json
from pathlib import Path

import pytest

from pepin.audio_link import RATE
from pepin.voice_cascade.control import ControlFile, Request, flag_value, write_request
from pepin.voice_cascade.player import Player
from pepin.voice_cascade.text import SentenceSplitter, in_braces


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Sink:
    def play(self, pcm: bytes) -> None: ...
    def play_end(self) -> None: ...
    def flush(self) -> None: ...


def pcm_s(seconds: float) -> bytes:
    return b"\x01\x00" * int(seconds * RATE)


# -- the splitter -------------------------------------------------------------------------------


def test_a_stop_inside_a_marker_ends_no_sentence() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("{happy. excited} Получилось! Ура") == ["{happy. excited} Получилось!"]
    assert splitter.flush() == "Ура"


def test_a_sentence_waits_while_a_marker_is_still_open() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("Ну да {style: whisp. ") == []
    assert splitter.feed("ering} все спят. Тихо") == ["Ну да {style: whisp. ering} все спят."]


def test_the_soft_cut_never_lands_inside_a_marker() -> None:
    """The comma rule matched the ``: `` of ``{style: whispering}`` (the architect's find)."""
    splitter = SentenceSplitter(soft_chars=40)
    text = "Я сейчас стою у книжной полки {style: whispering} в комнате рядом с окном"
    assert splitter.feed(text) == []  # the only "comma" is inside the braces: no cut
    splitter = SentenceSplitter(soft_chars=40)
    out = splitter.feed("Я сейчас стою у книжной полки, {style: tired, slow} в комнате у окна")
    assert out == ["Я сейчас стою у книжной полки,"]


def test_in_braces_knows_an_open_marker_from_a_closed_one() -> None:
    assert in_braces("a {b c", 4) and not in_braces("a {b} c", 6) and not in_braces("abc", 1)


def test_the_splitter_tells_a_closed_marker_and_the_first_word_before_the_sentence_ends() -> None:
    """text_ahead: the open sentence once a marker in it closes, once when the first sentence's
    first word is whole, never twice for the same, and a new sentence counts afresh."""
    splitter = SentenceSplitter()
    assert splitter.feed("{wei") == [] and splitter.ahead() == []  # the marker still open
    assert splitter.feed("gh} Пя") == [] and splitter.ahead() == ["{weigh} Пя"]
    assert splitter.feed("тьдесят на") == [] and splitter.ahead() == ["{weigh} Пятьдесят на"]
    assert splitter.ahead() == [] and splitter.feed(" пятьдесят") == [] and splitter.ahead() == []
    assert splitter.feed(". Зонт {point:you} возьми") == ["{weigh} Пятьдесят на пятьдесят."]
    assert splitter.ahead() == ["Зонт {point:you} возьми"]  # the second sentence's marker only
    reply = SentenceSplitter()
    assert reply.feed("Да") == [] and reply.ahead() == []  # «Да» may still be «Дальше»
    assert reply.feed(", конечно") == [] and reply.ahead() == ["Да, конечно"]
    assert reply.feed(" можно") == [] and reply.ahead() == []


# -- the player's marks -------------------------------------------------------------------------


def test_a_part_s_mark_and_the_played_clock_are_public() -> None:
    clock = Clock()
    player = Player(Sink(), rate=RATE, lead_s=10.0, clock=clock)
    assert player.mark(1, 0) is None
    player.feed(pcm_s(1.0), "reply", 1, 0)
    assert player.mark(1, 0) == pytest.approx((0.0, 1.0))
    player.feed(pcm_s(0.5), "reply", 1, 0)  # the part's audio grows while it is made
    player.feed(pcm_s(1.0), "reply", 1, 1)
    assert player.mark(1, 0) == pytest.approx((0.0, 1.5))
    assert player.mark(1, 1) == pytest.approx((1.5, 2.5))
    player.pacer.step()
    clock.now += 0.7
    assert player.played() == pytest.approx(0.7)
    player.forget(1)
    assert player.mark(1, 1) is None


# -- the control file's flags -------------------------------------------------------------------


def test_a_flag_reads_as_on_or_off_in_the_ways_people_write_it() -> None:
    assert [flag_value(v) for v in (True, "on", "OFF", 1, "true", 0)] == [
        True, True, False, True, True, False,
    ]  # fmt: skip
    assert flag_value("maybe") is None and flag_value(2) is None and flag_value(None) is None


def test_a_flag_request_is_written_and_read_back_once(tmp_path: Path) -> None:
    control = ControlFile(tmp_path)
    request = write_request(tmp_path, None, None, flags={"gestures": True})
    body = json.loads((tmp_path / "control.json").read_text())
    assert body["flags"] == {"gestures": True} and "brain" not in body
    read = control.poll()
    assert read is not None and read.id == request.id and dict(read.flags) == {"gestures": True}
    assert read.brain is None and control.poll() is None
    assert control.latest("flags") == {"gestures": True}


def test_a_hand_written_flag_that_says_nothing_is_left_out() -> None:
    request = Request.from_dict({"flags": {"gestures": "off", "speaker_gaze": "perhaps"}}, "x")
    assert dict(request.flags) == {"gestures": False}
    assert dict(Request.from_dict({"flags": "gestures"}).flags) == {}
