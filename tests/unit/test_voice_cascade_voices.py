"""The voices: the style marker and its parse, Qwen's instruction, the Gemini voice over a fake
client (the style beside the text, streaming, its bill, the fallback on an error, a timeout and
a cap, short lines from the disk), the speech worker's styles, bills and hot swap, and the
control file of a switch."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import json
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pepin.audio_link import RATE
from pepin.voice_cascade.config import DEFAULT_PRICES, CascadeConfig
from pepin.voice_cascade.control import (
    ControlFile,
    Request,
    Switchboard,
    answer,
    answers_to,
    write_request,
)
from pepin.voice_cascade.fake import ToneVoice
from pepin.voice_cascade.player import Player
from pepin.voice_cascade.speech import SpeechJob, SpeechWorker
from pepin.voice_cascade.stages import Billed, VoiceBill
from pepin.voice_cascade.text import speakable, split_style
from pepin.voice_cascade.transcriber import (
    Choice,
    SwitchingTranscriber,
    configured,
    wanted_from,
)
from pepin.voice_cascade.voice import QwenVoice, instruction

pytest.importorskip("google.genai.types")

from pepin.voice_cascade.gemini_voice import RETRY_S, GeminiVoice, rate_of

LITE = DEFAULT_PRICES["gemini-3.8-flash-lite-tts"]
CONFIG = Path(__file__).resolve().parents[2] / "config" / "voice_cascade.json"


# -- the style marker -----------------------------------------------------------------------


def test_a_style_marker_leads_a_sentence_and_is_never_read_aloud() -> None:
    assert split_style("{шёпотом} Я тут.") == ("шёпотом", "Я тут.")
    assert split_style("{style: грустно,  тихо}Ну вот.") == ("грустно, тихо", "Ну вот.")
    assert split_style("{} Как обычно.") == ("", "Как обычно.")
    assert split_style("Без стиля.") == (None, "Без стиля.")
    assert split_style("Слово {в середине}.")[0] is None
    assert speakable("{радостно} Привет!") == "Привет!"
    assert speakable("{шёпотом}") == ""


def test_qwen_hears_a_style_only_where_it_was_trained_for_one() -> None:
    assert instruction(None, "") is None
    assert instruction(None, "шёпотом") == "Шёпотом."
    assert instruction("Мягкий женский голос.", "грустно") == "Мягкий женский голос. Грустно."
    assert instruction("Мягкий женский голос.", "") == "Мягкий женский голос."
    small = QwenVoice("m", use_style=False)
    assert small.instruct("шёпотом") is None  # the 0.6B: dropped upstream, noise here
    big = QwenVoice("m", use_style=True)
    assert big.instruct("радостно!") == "Радостно!"
    design = QwenVoice("m", voice=None, description="Тёплый голос.", use_style=True)
    assert design.instruct("") == "Тёплый голос."
    assert design.instruct("тихо") == "Тёплый голос. Тихо."


# -- the Gemini voice -----------------------------------------------------------------------


def chunk(data: bytes, usage: Any = None, mime: str = "audio/L16;codec=pcm;rate=24000") -> Any:
    part = SimpleNamespace(inline_data=SimpleNamespace(data=data, mime_type=mime))
    candidate = SimpleNamespace(content=SimpleNamespace(parts=[part]))
    return SimpleNamespace(candidates=[candidate], usage_metadata=usage)


class FakeModels:
    """Answers each call with ``plan`` in turn: a list of chunks, an exception, or a delay."""

    def __init__(self, *plan: list[Any] | BaseException | float) -> None:
        self.plan = list(plan)
        self.requests: list[dict[str, Any]] = []

    def generate_content_stream(self, **kwargs: Any) -> Iterator[Any]:
        self.requests.append(kwargs)
        step = self.plan[min(len(self.requests), len(self.plan)) - 1]
        if isinstance(step, BaseException):
            raise step
        if isinstance(step, float):
            time.sleep(step)
            return iter([chunk(b"\x01\x00" * 100)])
        return iter(step)


def gemini(models: FakeModels, tmp: Path | None = None, **kwargs: Any) -> GeminiVoice:
    voice = GeminiVoice(
        "gemini-3.8-flash-lite-tts",
        voice="Kore",
        price=LITE,
        fallback=kwargs.pop("fallback", ToneVoice(rate=16_000, seconds_per_char=0.01)),
        fallback_name="qwen3-0.6b",
        client=SimpleNamespace(models=models),
        cache_dir=tmp,
        **kwargs,
    )
    voice.load()
    return voice


def test_the_style_travels_beside_the_text_and_the_audio_streams_with_its_bill() -> None:
    usage = SimpleNamespace(prompt_token_count=20, candidates_token_count=100)
    models = FakeModels([chunk(b"\x01\x00\x02"), chunk(b"\x00\x03\x00", usage)])
    voice = gemini(models)
    assert isinstance(voice, Billed)
    pcm = list(voice.stream("Привет, Артём!", "шёпотом"))
    assert pcm == [b"\x01\x00", b"\x02\x00\x03\x00"]  # whole samples, the odd byte carried
    request = models.requests[0]
    part = request["contents"][0].parts[0]
    assert part.text == "Привет, Артём!" and part.speech_metadata.style == "шёпотом"
    config = request["config"]
    assert config.response_modalities == ["AUDIO"]
    assert config.speech_config.voice_config.prebuilt_voice_config.voice_name == "Kore"
    bill = voice.bill()
    assert bill is not None and voice.bill() is None  # taken once
    assert (bill.engine, bill.text_tokens, bill.audio_tokens, bill.calls) == (
        "gemini-3.8-flash-lite-tts", 20, 100, 1,
    )  # fmt: skip
    assert bill.usd == pytest.approx((20 * 0.50 + 100 * 6.00) / 1e6)
    assert bill.first_audio_ms is not None and bill.fallback is None
    plain = list(voice.stream("Да.", ""))
    assert plain and models.requests[1]["contents"][0].parts[0].speech_metadata is None


def test_a_failed_call_is_said_by_the_fallback_and_the_cloud_rests() -> None:
    now = [100.0]
    fallback = ToneVoice(rate=16_000, seconds_per_char=0.01)
    models = FakeModels(ConnectionError("offline"), [chunk(b"\x01\x00" * 10)])
    voice = gemini(models, fallback=fallback, clock=lambda: now[0])
    pcm = b"".join(voice.stream("Секунду, гляну.", "грустно"))
    samples_16k = int(len("Секунду, гляну.") * 0.01 * 16_000)
    assert abs(len(pcm) // 2 - samples_16k * 3 // 2) < 50  # the fallback, at 24 kHz
    assert fallback.spoken == ["Секунду, гляну."] and fallback.styles == ["грустно"]
    bill = voice.bill()
    assert bill is not None and bill.engine == "qwen3-0.6b" and bill.calls == 1
    assert bill.fallback is not None and "offline" in bill.fallback
    list(voice.stream("Ещё.", ""))  # resting: no call
    assert len(models.requests) == 1 and voice.bill().fallback.startswith("resting")  # type: ignore[union-attr]
    now[0] += RETRY_S + 1
    assert b"".join(voice.stream("Снова.", "")) == b"\x01\x00" * 10
    assert len(models.requests) == 2


def test_a_slow_call_is_left_behind_and_a_cap_stops_the_calls() -> None:
    fallback = ToneVoice(rate=24_000, seconds_per_char=0.01)
    voice = gemini(FakeModels(0.5), fallback=fallback, timeout_s=0.05)
    start = time.monotonic()
    assert b"".join(voice.stream("Медленно.", ""))
    assert time.monotonic() - start < 0.4 and fallback.spoken == ["Медленно."]
    assert "no audio" in (voice.bill() or VoiceBill("")).fallback  # type: ignore[operator]
    capped = gemini(FakeModels([chunk(b"\x01\x00")]), max_calls=1)
    list(capped.stream("Раз.", ""))
    list(capped.stream("Два.", ""))
    bill = capped.bill()
    assert bill is not None and bill.fallback == "the cap of 1 calls" and capped.calls == 1


def test_a_short_line_said_once_comes_from_the_disk_after(tmp_path: Path) -> None:
    models = FakeModels([chunk(b"\x05\x00" * 4)])
    voice = gemini(models, tmp_path)
    assert b"".join(voice.stream("Да?", "")) == b"\x05\x00" * 4
    assert b"".join(voice.stream("Да?", "")) == b"\x05\x00" * 4
    assert len(models.requests) == 1
    bill = voice.bill()
    assert bill is not None and bill.calls == 0 and bill.usd == 0.0
    list(voice.stream("Да?", "шёпотом"))  # another style is another line
    assert len(models.requests) == 2
    assert rate_of("audio/L16;codec=pcm;rate=16000") == 16_000 and rate_of("audio/L16") == 24_000


# -- the speech worker ----------------------------------------------------------------------


class Sink:
    play_rate = RATE

    def play(self, pcm: bytes) -> None: ...
    def play_end(self) -> None: ...
    def flush(self) -> None: ...


class SlowVoice(ToneVoice):
    """A tone voice whose load waits for ``gate`` (or fails)."""

    def __init__(self, gate: threading.Event, fail: bool = False) -> None:
        super().__init__(rate=24_000, seconds_per_char=0.01)
        self.gate, self.fail = gate, fail

    def load(self) -> None:
        assert self.gate.wait(5.0)
        if self.fail:
            raise RuntimeError("no weights")
        super().load()


class PaidTone(ToneVoice):
    """A tone voice that leaves a bill for every sentence."""

    def bill(self) -> VoiceBill | None:
        return VoiceBill("cloud", first_audio_ms=5, usd=0.001, calls=1)


def worker_with(voice: ToneVoice, clips: tuple[str, ...] = ()) -> tuple[SpeechWorker, list[Any]]:
    done: list[Any] = []
    player = Player(Sink(), rate=RATE, lead_s=100.0)
    worker = SpeechWorker(voice, player, clips=clips, on_done=lambda job, t: done.append(job))
    worker.start()
    assert worker.ready.wait(2.0) and worker.error is None
    return worker, done


def wait_for(condition: Any, timeout_s: float = 3.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "timed out"
        time.sleep(0.005)


def test_a_styled_sentence_reaches_the_voice_and_skips_its_plain_clip() -> None:
    voice = ToneVoice(rate=24_000, seconds_per_char=0.01)
    worker, done = worker_with(voice, ("Да?",))
    worker.say(SpeechJob(1, 0, "{шёпотом} Да?", style="шёпотом"))
    worker.say(SpeechJob(1, 1, "Да?"))
    wait_for(lambda: len(done) == 2)
    worker.close()
    assert voice.spoken == ["Да?", "Да?"]  # the clip at load, then the whispered one
    assert voice.styles == ["", "шёпотом"]


def test_a_paid_voice_makes_no_clips_and_bills_every_sentence() -> None:
    voice = PaidTone(rate=24_000, seconds_per_char=0.01)
    bills: list[tuple[int, VoiceBill]] = []
    worker, done = worker_with(voice, ("Секунду.",))
    worker.on_bill = lambda job, bill: bills.append((job.index, bill))
    assert worker.clips == {} and voice.spoken == []
    worker.say(SpeechJob(1, 0, "Секунду."))
    wait_for(lambda: len(done) == 1)
    worker.close()
    assert [i for i, _ in bills] == [0] and bills[0][1].engine == "cloud"


def test_a_new_voice_loads_beside_the_one_speaking_and_takes_the_next_sentence() -> None:
    old = ToneVoice(rate=24_000, seconds_per_char=0.01)
    worker, done = worker_with(old, ("мм",))
    gate = threading.Event()
    new = SlowVoice(gate)
    outcome: list[BaseException | None] = []
    worker.swap(new, outcome.append)
    worker.say(SpeechJob(1, 0, "Ещё старым."))
    wait_for(lambda: len(done) == 1)
    assert old.spoken[-1] == "Ещё старым." and worker.voice is old  # the load is still waiting
    gate.set()
    wait_for(lambda: outcome == [None])
    assert worker.voice is new
    wait_for(lambda: "мм" in worker.clips)  # its own clip, made after it took over
    assert new.spoken == ["мм"]
    worker.say(SpeechJob(2, 0, "Уже новым."))
    wait_for(lambda: len(done) == 2)
    assert new.spoken[-1] == "Уже новым." and "Уже новым." not in old.spoken
    broken = SlowVoice(threading.Event(), fail=True)
    broken.gate.set()
    worker.swap(broken, outcome.append)
    wait_for(lambda: len(outcome) == 2)
    assert isinstance(outcome[1], RuntimeError) and worker.voice is new  # the new one stays
    worker.close()


# -- the control file -----------------------------------------------------------------------


def test_a_switch_request_is_handed_out_once_and_an_old_one_never(tmp_path: Path) -> None:
    old = write_request(tmp_path, "sonnet", None)
    control = ControlFile(tmp_path)
    assert control.poll() is None  # an earlier run's request
    time.sleep(0.01)
    new = write_request(tmp_path, None, "gemini-lite", "whispercpp")
    got = control.poll()
    assert got == Request(new.id, None, "gemini-lite", "whispercpp") and got.id != old.id
    assert control.poll() is None
    assert control.latest("transcriber") == "whispercpp" and control.latest("brain") is None
    time.sleep(0.01)
    (tmp_path / "control.json").write_text(json.dumps({"voice_preset": "qwen3-vd"}))  # by hand
    by_hand = control.poll()
    assert by_hand is not None and by_hand.voice == "qwen3-vd" and by_hand.id.startswith("file-")
    assert control.latest("transcriber") == "whispercpp"  # left out: unchanged
    answer(tmp_path, new.id, "voice", "gemini-lite", "loading")
    answer(tmp_path, old.id, "brain", "sonnet", "applied")
    answer(tmp_path, new.id, "voice", "gemini-lite", "applied", ms=900)
    rows = answers_to(tmp_path, new.id)
    assert [r["state"] for r in rows] == ["loading", "applied"] and rows[1]["ms"] == 900


class Engine:
    """A transcriber engine that answers with its own name."""

    def __init__(self, name: str) -> None:
        self.name = name

    def transcribe(self, pcm: bytes) -> str:
        return self.name


def test_the_transcriber_key_loads_the_engine_and_answers_its_outcome(tmp_path: Path) -> None:
    config = CascadeConfig.load(CONFIG)
    control = ControlFile(tmp_path)
    built: list[Choice] = []

    def build(choice: Choice) -> Engine:
        if choice.model == "missing":
            raise FileNotFoundError("ggml-missing.bin")
        built.append(choice)
        return Engine(choice.label)

    switching = SwitchingTranscriber(
        Engine("mlx"), configured(config), build, wanted_from(control, config),
        report=lambda line: None, spawn=lambda work: work(),
    )  # fmt: skip
    board = Switchboard(
        control, configure=lambda b, v: config, build_brain=lambda c, n: None,  # type: ignore[arg-type,return-value]
        build_voice=lambda v: ToneVoice(), brain="haiku", voice="qwen3-0.6b",
        transcriber=switching,
    )  # fmt: skip

    def ask(value: str) -> list[str]:
        time.sleep(0.01)
        request = write_request(tmp_path, None, None, value)
        got = control.poll()
        assert got is not None and got.transcriber == value
        board.switch_transcriber(got.id, got.transcriber)
        return [f"{r['name']} {r['state']}" for r in answers_to(tmp_path, request.id)]

    assert ask("whispercpp") == ["whispercpp:large-v3-turbo loading",
                                 "whispercpp:large-v3-turbo applied"]  # fmt: skip
    assert switching.transcribe(b"") == "whispercpp:large-v3-turbo"
    assert ask("whispercpp") == ["whispercpp:large-v3-turbo applied"]  # already in use
    assert ask("whispercpp:missing") == ["whispercpp:missing loading", "whispercpp:missing failed"]
    assert ask("whispercpp:missing")[-1] == "whispercpp:missing failed"  # not tried again
    assert ask("vosk") == ["vosk failed"] and len(built) == 1
