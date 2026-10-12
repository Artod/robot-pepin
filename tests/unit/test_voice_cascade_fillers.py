"""A slow tool's fillers: the config block and its old string form, the per-tool choice without
an immediate repeat, the second filler's time, and who says the asides (the voice itself, a paid
voice's local fallback, nobody for a paid voice without one)."""

import random
import time
from itertools import pairwise
from pathlib import Path
from typing import Any

import pytest

from pepin.audio_link import RATE
from pepin.voice_cascade.config import CascadeConfig, FillerLines
from pepin.voice_cascade.fake import ToneVoice
from pepin.voice_cascade.fillers import FillerPicker, again_due
from pepin.voice_cascade.player import Player
from pepin.voice_cascade.speech import SpeechJob, SpeechWorker
from pepin.voice_cascade.stages import VoiceBill

CONFIG = Path(__file__).resolve().parents[2] / "config" / "voice_cascade.json"


# -- the config -----------------------------------------------------------------------------


def test_the_file_gives_each_slow_tool_its_lines_and_the_rest_the_common_ones() -> None:
    config = CascadeConfig.load(CONFIG)
    fillers = config.fillers
    assert fillers.for_tool("remember_person") == (
        "Дай запомнить…", "Секундочку, запоминаю.", "Сейчас, запишу.",
    )  # fmt: skip
    assert fillers.for_tool("see") == fillers.for_tool("look") == ("Сейчас гляну.", "Так, смотрю.")
    assert fillers.for_tool("map_tree") == fillers.common  # no lines of its own
    assert "Секунду." in fillers.common and fillers.again == ("Ещё секундочку.", "Уже почти.")
    assert config.filler_again_s == 4.0
    assert fillers == FillerLines()  # the code's defaults are the file's
    lines = fillers.lines()
    assert len(lines) == len(set(lines)) == 19  # "Сейчас гляну." once for see and look


def test_the_old_string_form_reads_as_the_common_lines_alone() -> None:
    old = CascadeConfig.from_dict({"conversation": {"tool_filler": "Секунду."}})
    assert old.fillers == FillerLines(("Секунду.",), (), {})
    assert old.fillers.for_tool("see") == ("Секунду.",)
    assert CascadeConfig.from_dict({"conversation": {"tool_filler": ""}}).fillers.lines() == ()
    overridden = CascadeConfig.load(
        CONFIG, ["conversation.fillers=null", "conversation.tool_filler=Минуту."]
    )
    assert overridden.fillers.common == ("Минуту.",) and overridden.fillers.again == ()
    assert CascadeConfig.load(CONFIG, ["conversation.fillers={}"]).fillers.lines() == ()
    with pytest.raises(ValueError, match="filler_again_s"):
        CascadeConfig.load(CONFIG, ["conversation.filler_again_s=-1"])


# -- the choice -----------------------------------------------------------------------------


def test_a_filler_comes_from_the_tool_s_own_lines_and_never_twice_in_a_row() -> None:
    fillers = FillerLines()
    picker = FillerPicker(random.Random(7))
    said = [picker.pick(fillers.for_tool("remember_person"), fillers.common) for _ in range(200)]
    assert set(said) == set(fillers.for_tool("remember_person"))  # all of them, nothing else
    assert all(a != b for a, b in pairwise(said))
    common = [picker.pick(fillers.for_tool("map_tree"), fillers.common) for _ in range(200)]
    assert set(common) == set(fillers.common)
    assert all(a != b for a, b in pairwise(common))


def test_a_tool_with_one_line_takes_a_common_one_rather_than_repeat_it() -> None:
    fillers = FillerLines()
    picker = FillerPicker(random.Random(1))
    first = picker.pick(fillers.for_tool("forget_person"), fillers.common)
    second = picker.pick(fillers.for_tool("forget_person"), fillers.common)
    assert first == "Хорошо, забываю." and second in fillers.common
    assert picker.pick(["Одна."]) == "Одна." and picker.pick(["Одна."]) == "Одна."  # nothing else
    assert picker.pick([]) is None


def test_the_second_filler_is_due_filler_again_s_after_the_first_on_the_clock() -> None:
    assert not again_due(None, 100.0, 4.0)  # no first filler
    assert not again_due(10.0, 13.9, 4.0)
    assert again_due(10.0, 14.0, 4.0) and again_due(10.0, 30.0, 4.0)
    assert not again_due(10.0, 30.0, 0.0)  # 0: never


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
