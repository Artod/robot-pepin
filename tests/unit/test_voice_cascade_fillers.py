"""The fillers: the phrase book in the file and its old forms, the engine's rules on a fake clock
(the language by the words' script, the neutral hums, once a reply, cooldowns, silence by
chance, no repeats, clips only), the moments' times, and who says the asides (the voice itself,
a paid voice's local fallback, nobody for a paid voice without one)."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import random
import time
from collections.abc import Iterator
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from pepin.audio_link import RATE
from pepin.voice_cascade.config import CascadeConfig, FillerBook
from pepin.voice_cascade.fake import ToneVoice
from pepin.voice_cascade.fillers import (
    UNKNOWN,
    FillerEngine,
    FillerPicker,
    again_due,
    clip_lines,
    script_language,
    thinking_due,
)
from pepin.voice_cascade.player import Player
from pepin.voice_cascade.speech import (
    CLIP_TRIES,
    SpeechJob,
    SpeechWorker,
    clip_limit_s,
    make_clip,
)
from pepin.voice_cascade.stages import VoiceBill

CONFIG = Path(__file__).resolve().parents[2] / "config" / "voice_cascade.json"
BOOK = FillerBook()


# -- the config -----------------------------------------------------------------------------


def test_the_file_s_book_is_the_code_s_with_lines_per_tool_and_language() -> None:
    config = CascadeConfig.load(CONFIG)
    fillers = config.fillers
    assert fillers == BOOK  # the code's defaults are the file's
    assert fillers.for_tool("remember_person", "ru") == (
        "Дай запомнить…", "Секундочку, запоминаю.", "Сейчас, запишу.",
    )  # fmt: skip
    assert fillers.for_tool("see", "en") == ("Let me look.", "Looking.", "Let me see that.")
    assert fillers.for_tool("map_tree", "ru") == fillers.common["ru"]  # no lines of its own
    assert fillers.for_tool("see", "de") == ()  # the engine says a hum then
    assert fillers.languages() == ("ru", "en") and "Хм…" in fillers.neutral
    assert (config.conversation_language, config.filler_language_from) == ("ru", "utterance")
    assert (config.filler_thinking_s, config.filler_again_s) == (1.2, 4.0)
    assert config.filler_silent_p == {"thinking": 0.25, "backchannel": 0.25}
    assert config.filler_cooldown_s == {"backchannel": 2.0}
    lines = fillers.lines()
    assert len(lines) == len(set(lines)) == 75  # "Сейчас гляну." once for see and look
    clips = clip_lines(config)  # the backchannel's flag off: none of its lines
    assert len(clips) == 69 and "угу" not in clips
    assert clips[:4] == fillers.neutral and clips.index("Так…") < clips.index("Well…")


def test_the_old_forms_still_load() -> None:
    old = CascadeConfig.from_dict({"conversation": {"tool_filler": "Секунду."}})
    assert old.fillers == FillerBook({"ru": ("Секунду.",)}, {}, {}, BOOK.backchannel, {}, ())
    assert old.fillers.for_tool("see", "ru") == ("Секунду.",)
    empty = CascadeConfig.from_dict({"conversation": {"tool_filler": ""}}).fillers
    assert empty.lines(backchannel=False) == ()
    overridden = CascadeConfig.load(
        CONFIG, ["conversation.fillers=null", "conversation.tool_filler=Минуту."]
    )
    assert overridden.fillers.common == {"ru": ("Минуту.",)} and overridden.fillers.again == {}
    none = CascadeConfig.load(CONFIG, ["conversation.fillers={}"]).fillers
    assert none.lines(backchannel=False) == () and none.thinking == {}
    before = CascadeConfig.from_dict(  # the object of the per-tool fillers, lists alone
        {"conversation": {"fillers": {"any": ["Секунду."], "by_tool": {"see": ["Гляну."]}}}}
    ).fillers
    assert before.for_tool("see", "ru") == ("Гляну.",) and before.common == {"ru": ("Секунду.",)}
    assert before.thinking == {} and before.neutral == ()  # no thinking filler it never had
    lines = ["conversation.language=en", 'backchannel.lines=["мм"]']  # the old place wins
    english = CascadeConfig.load(CONFIG, lines)
    assert english.fillers.backchannel == {"en": ("мм",)} and english.conversation_language == "en"


@pytest.mark.parametrize(
    "bad",
    [
        "conversation.filler_again_s=-1",
        "conversation.filler_thinking_s=-0.5",
        "conversation.filler_language_from=voice",
        'conversation.filler_silent_p={"thinking": 2}',
        'conversation.filler_silent_p={"nope": 0.1}',
        "conversation.filler_silent_p=0.3",
        'conversation.filler_cooldown_s={"backchannel": -1}',
    ],
)
def test_a_fillers_knob_out_of_range_is_refused(bad: str) -> None:
    with pytest.raises(ValueError, match="filler"):
        CascadeConfig.load(CONFIG, [bad])


# -- the choice -----------------------------------------------------------------------------


def test_a_filler_comes_from_the_tool_s_own_lines_and_never_twice_in_a_row() -> None:
    own, common = BOOK.for_tool("remember_person", "ru"), BOOK.common["ru"]
    picker = FillerPicker(random.Random(7))
    said = [picker.pick(own, common) for _ in range(200)]
    assert set(said) == set(own)  # all of them, nothing else
    assert all(a != b for a, b in pairwise(said))
    lines = (*BOOK.thinking["ru"], *BOOK.neutral)  # nine: none of the last three again
    many = [picker.pick(lines) for _ in range(300)]
    assert set(many) == set(lines)
    assert all(len(set(many[i : i + 4])) == 4 for i in range(len(many) - 3))


def test_a_tool_with_one_line_takes_a_common_one_rather_than_repeat_it() -> None:
    picker = FillerPicker(random.Random(1))
    one, common = ("Хорошо, забываю.",), BOOK.common["ru"]
    first, second = picker.pick(one, common), picker.pick(one, common)
    assert first == "Хорошо, забываю." and second in common
    assert picker.pick(["Одна."]) == "Одна." and picker.pick(["Одна."]) == "Одна."  # nothing else
    assert picker.pick([]) is None


def test_the_second_filler_is_due_filler_again_s_after_the_last_on_the_clock() -> None:
    assert not again_due(None, 100.0, 4.0)  # no filler yet
    assert not again_due(10.0, 13.9, 4.0)
    assert again_due(10.0, 14.0, 4.0) and again_due(10.0, 30.0, 4.0)
    assert not again_due(10.0, 30.0, 0.0)  # 0: never


def test_the_thinking_filler_is_due_at_1_2_s_unless_the_first_token_is_in() -> None:
    assert not thinking_due(10.0, False, 11.1, 1.2)
    assert thinking_due(10.0, False, 11.2, 1.2) and thinking_due(10.0, False, 30.0, 1.2)
    assert not thinking_due(10.0, True, 11.2, 1.2)  # the first token came at 11.0
    assert not thinking_due(None, False, 30.0, 1.2)  # no close: a report, or asked early
    assert not thinking_due(10.0, False, 30.0, 0.0)  # 0: never


def test_the_thinking_filler_speaks_once_a_reply() -> None:
    engine = FillerEngine(random.Random(3))
    first = engine.choose(BOOK, "thinking", 11.2, turn=1)
    assert first in {*BOOK.thinking["ru"], *BOOK.neutral}
    assert engine.choose(BOOK, "thinking", 15.0, turn=1) is None
    assert engine.choose(BOOK, "thinking", 20.0, turn=2) not in (None, first)
    assert engine.choose(BOOK, "tool", 20.5, turn=2, tool="see") in BOOK.for_tool("see", "ru")
    assert engine.choose(BOOK, "tool", 21.0, turn=2, tool="look") is None  # one a reply


def test_the_language_follows_the_script_of_the_words() -> None:
    assert script_language("Where are you?") == "en"
    assert script_language("открой мне YouTube") == "ru"  # by words, not letters
    assert script_language("ok да") is None  # a tie: no change
    assert script_language("12, 34!") is None and script_language("こんにちは") == UNKNOWN
    engine = FillerEngine(random.Random(5))
    assert engine.hear("What do you see?") == "en"
    assert engine.choose(BOOK, "thinking", 1.0, turn=1) in {*BOOK.thinking["en"], *BOOK.neutral}
    assert engine.choose(BOOK, "tool", 1.1, turn=1, tool="find") in BOOK.for_tool("find", "en")
    assert engine.hear("...") == "en"  # no letters: no change
    assert engine.hear("А теперь по-русски") == "ru"
    assert engine.choose(BOOK, "again", 2.0, turn=2) in BOOK.again["ru"]
    engine.open("en")  # a conversation opens in conversation.language
    assert engine.language == "en"


def test_a_language_the_book_lacks_says_only_the_neutral_hums() -> None:
    engine = FillerEngine(random.Random(2), language="de")
    hums = set(BOOK.neutral)
    assert engine.choose(BOOK, "thinking", 1.0, turn=1) in hums
    assert engine.choose(BOOK, "tool", 1.0, turn=1, tool="see") in hums
    assert engine.choose(BOOK, "again", 1.0, turn=1) in hums
    assert engine.choose(BOOK, "backchannel", 1.0) is None  # no hum under the person's words
    engine.hear("こんにちは")
    assert engine.language == UNKNOWN and engine.choose(BOOK, "thinking", 2.0, turn=2) in hums


def test_a_cooldown_spaces_the_backchannels_and_a_silent_moment_counts() -> None:
    engine = FillerEngine(random.Random(4))
    gap = {"cooldown_s": 2.0}
    assert engine.choose(BOOK, "backchannel", 10.0, **gap) in BOOK.backchannel["ru"]
    assert engine.choose(BOOK, "backchannel", 11.9, **gap) is None
    assert engine.choose(BOOK, "backchannel", 12.0, **gap) is not None
    assert engine.choose(BOOK, "backchannel", 14.0, silent_p=1.0, **gap) is None  # passes...
    assert engine.choose(BOOK, "backchannel", 15.0, **gap) is None  # ...and counts
    assert engine.choose(BOOK, "thinking", 20.0, turn=7, silent_p=1.0) is None
    assert engine.choose(BOOK, "thinking", 21.0, turn=7) is None  # the reply had its moment


def test_about_silent_p_of_the_moments_pass_in_silence() -> None:
    engine = FillerEngine(random.Random(11))
    said = [engine.choose(BOOK, "thinking", float(n), turn=n, silent_p=0.25) for n in range(2000)]
    assert 0.22 < said.count(None) / len(said) < 0.28
    spoken = [line for line in said if line is not None]
    assert all(a != b for a, b in pairwise(spoken))


def test_only_a_line_with_its_clip_made_is_picked() -> None:
    engine = FillerEngine(random.Random(6))
    assert engine.choose(BOOK, "thinking", 1.0, turn=1, ready=lambda line: False) is None
    assert engine.choose(BOOK, "thinking", 1.1, turn=1, ready=lambda line: line == "Хм…") == "Хм…"
    ready = {"Сейчас."}  # none of the see's own clips yet: a common one
    assert engine.choose(BOOK, "tool", 1.2, turn=2, tool="see", ready=ready.__contains__) == (
        "Сейчас."
    )


def test_the_clips_follow_the_language_rule_and_the_backchannel_flag() -> None:
    fixed = CascadeConfig().with_changes(filler_language_from="config", backchannel=True)
    lines = clip_lines(fixed)
    assert lines == BOOK.lines("ru") and "угу" in lines and "Well…" not in lines
    assert set(clip_lines(CascadeConfig())) == set(BOOK.lines(backchannel=False))


# -- who says them --------------------------------------------------------------------------


class Sink:
    play_rate = RATE

    def __init__(self) -> None:
        self.fed = 0

    def play(self, pcm: bytes) -> None:
        self.fed += len(pcm)

    def play_end(self) -> None: ...
    def flush(self) -> None: ...


class Paid(ToneVoice):
    """A voice that pays per sentence, with (or without) a local fallback."""

    def __init__(self, fallback: ToneVoice | None) -> None:
        super().__init__(rate=24_000, seconds_per_char=0.01)
        self.fallback = fallback

    def bill(self) -> VoiceBill | None:
        return VoiceBill("cloud", usd=0.001, calls=1)


def worker_with(voice: ToneVoice, **lines: Any) -> tuple[SpeechWorker, Player, list[Any]]:
    done: list[Any] = []
    player = Player(Sink(), rate=RATE, lead_s=100.0)
    worker = SpeechWorker(voice, player, on_done=lambda job, t: done.append(job), **lines)
    worker.start()
    assert worker.ready.wait(2.0) and worker.error is None
    return worker, player, done


def wait_for(condition: Any, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.002)


def test_a_voice_makes_its_filler_clips_once_ready_and_says_them_from_memory() -> None:
    voice = ToneVoice(rate=24_000, seconds_per_char=0.01)
    worker, player, done = worker_with(
        voice, clips=["Секунду."], asides=["Секунду.", "Так, смотрю.", "так смотрю"]
    )
    fed: list[int] = []
    worker.on_audio = lambda job, pcm, t: fed.append(len(pcm))
    wait_for(lambda: "так смотрю" in worker.clips)
    assert voice.spoken == ["Секунду.", "Так, смотрю."]  # each key made once
    worker.say(SpeechJob(1, -1, "Так, смотрю.", "backchannel"))
    wait_for(lambda: len(done) == 1)
    worker.close()
    assert voice.spoken == ["Секунду.", "Так, смотрю."]  # from memory
    assert fed == [len(worker.clips["так смотрю"])] and player.playback() == "backchannel"


def test_a_paid_voice_s_fillers_are_its_local_fallback_s_and_never_billed() -> None:
    fallback = ToneVoice(rate=16_000, seconds_per_char=0.01)
    voice = Paid(fallback)
    rates: list[tuple[str, int]] = []
    bills: list[VoiceBill] = []
    worker, _, done = worker_with(voice, clips=["Секунду."], asides=["Секунду.", "Уже почти."])
    worker.on_bill = lambda job, bill: bills.append(bill)
    worker.on_audio = lambda job, pcm, t: rates.append((job.kind, worker.job_rate))
    wait_for(lambda: len(worker.clips) == 2)
    assert fallback.spoken == ["Секунду.", "Уже почти."] and voice.spoken == []  # no paid clip
    assert worker.says_asides
    worker.say(SpeechJob(1, -1, "Секунду.", "backchannel"))
    worker.say(SpeechJob(1, 0, "Секунду."))  # the same words in the reply: the paid voice's
    wait_for(lambda: len(done) == 2)
    worker.close()
    assert fallback.spoken == ["Секунду.", "Уже почти."]  # the filler from its clip
    assert voice.spoken == ["Секунду."] and len(bills) == 1  # the reply's sentence alone
    assert ("backchannel", 16_000) in rates and ("reply", 24_000) in rates


def test_a_paid_voice_without_a_fallback_says_no_filler() -> None:
    voice = Paid(None)
    bills: list[VoiceBill] = []
    fed: list[int] = []
    worker, player, done = worker_with(voice, asides=["Секунду."])
    worker.on_bill = lambda job, bill: bills.append(bill)
    worker.on_audio = lambda job, pcm, t: fed.append(len(pcm))
    worker.say(SpeechJob(1, -1, "Секунду.", "backchannel"))
    wait_for(lambda: len(done) == 1)
    worker.close()
    assert worker.clips == {} and voice.spoken == [] and bills == [] and fed == []
    assert not worker.says_asides  # the loop logs no filler it would not hear
    assert player.playback() == "silent"


def test_a_swapped_in_paid_voice_makes_its_filler_clips_with_its_fallback() -> None:
    old = ToneVoice(rate=24_000, seconds_per_char=0.01)
    worker, _, _ = worker_with(old, clips=["Да?"], asides=["Секунду."])
    wait_for(lambda: "секунду" in worker.clips)
    fallback = ToneVoice(rate=24_000, seconds_per_char=0.01)
    new = Paid(fallback)
    loaded: list[BaseException | None] = []
    worker.swap(new, loaded.append)
    wait_for(lambda: loaded == [None] and "секунду" in worker.clips)
    worker.close()
    assert fallback.spoken == ["Секунду."] and new.spoken == []  # no opener paid for either


class Babbler(ToneVoice):
    """A voice that babbles on for a minute on the first ``bad`` takes of each line."""

    def __init__(self, bad: int) -> None:
        super().__init__(rate=24_000, seconds_per_char=0.01, chunk_s=0.1)
        self.bad = bad
        self.takes: dict[str, int] = {}
        self.chunks = 0  # of babble yielded

    def stream(self, text: str, style: str = "") -> Iterator[bytes]:
        take = self.takes[text] = self.takes.get(text, 0) + 1
        if take > self.bad:
            yield from super().stream(text, style)
            return
        for _ in range(600):  # 0.1 s chunks: 60 s
            self.chunks += 1
            yield b"\x00\x01" * int(0.1 * self.rate)


def test_a_clip_that_babbles_on_is_cut_and_made_again_or_never_kept() -> None:
    once = Babbler(bad=1)
    pcm = make_clip(once, "Хм…")
    assert pcm is not None and len(pcm) <= clip_limit_s("Хм…") * once.rate * 2
    assert once.takes == {"Хм…": 2} and once.chunks <= 26  # cut at 2.5 s, not after 60
    assert clip_limit_s("Секундочку, запоминаю.") == pytest.approx(5.5)  # 0.25 s a character
    always = Babbler(bad=CLIP_TRIES)
    assert make_clip(always, "Э-э…") is None and always.takes == {"Э-э…": CLIP_TRIES}
    voice = Babbler(bad=CLIP_TRIES)
    worker, _, _ = worker_with(voice, asides=["Хм…", "Так…"])
    wait_for(lambda: voice.takes.get("Так…") == CLIP_TRIES)
    worker.close()
    assert not any(map(worker.has_clip, ["Хм…", "Так…"]))  # never said as a filler
