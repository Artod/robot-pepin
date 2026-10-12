"""The whole cascade over fakes: the name gate, a turn's timeline and logs, the person going on
before the robot answered, a barge-in cut to what was heard, a drive acknowledged when it
starts and reported when it ends, a slow tool's fillers, and the brain asked at the early read
(kept by the close, dropped when the person goes on, billed either way)."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import asyncio
import json
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.expression.conversation import GESTURES, SLOW_TOOL, Heard
from pepin.expression.fakes import FakeFlag, RecordingObserver
from pepin.expression.markers import EMPTY, Vocabulary
from pepin.tools import TOOLS
from pepin.tools.fakes import FakeGoalServer, WallClock, fake_robot
from pepin.voice_cascade.config import BrainConfig, CascadeConfig, FillerLines, VoiceConfig
from pepin.voice_cascade.control import ControlFile, Switchboard, answers_to, write_request
from pepin.voice_cascade.engines import BrainSetup
from pepin.voice_cascade.fake import ListTranscriber, ScriptedBrain, Step, ToneVoice
from pepin.voice_cascade.loop import CascadeLoop
from pepin.voice_cascade.player import Player
from pepin.voice_cascade.speech import SpeechWorker
from pepin.voice_cascade.stages import CUT, Prosody, Speaker, TimedWord, Transcript, Unit, Vowel
from pepin.voice_live.events import Events
from pepin.voice_live.sources import FrameSink

pytestmark = pytest.mark.slow  # the player paces at real time

FRAME = FRAME_SAMPLES * 2


def tone(seconds: float) -> list[bytes]:
    t = np.arange(int(seconds * RATE)) / RATE
    pcm = (0.3 * 32767 * np.sin(2 * np.pi * 300 * t)).astype("<i2").tobytes()
    return [pcm[i : i + FRAME] for i in range(0, len(pcm) - FRAME + 1, FRAME)]


def quiet(seconds: float) -> list[bytes]:
    return [b"\x00\x00" * FRAME_SAMPLES] * int(seconds * RATE / FRAME_SAMPLES)


def utterance(loud_s: float = 0.4) -> list[bytes]:
    return quiet(0.2) + tone(loud_s) + quiet(0.7)


class PushMic:
    """A mic the test speaks into: frames delivered at once, 2 ms apart."""

    def __init__(self) -> None:
        self.sink: FrameSink | None = None

    def start(self, sink: FrameSink) -> None:
        self.sink = sink

    def push(self, frames: list[bytes]) -> None:
        assert self.sink is not None
        for frame in frames:
            self.sink(frame, time.monotonic())
            time.sleep(0.002)


class NullSpeaker:
    play_rate = RATE

    def play(self, pcm: bytes) -> None: ...
    def play_end(self) -> None: ...
    def flush(self) -> None: ...


class Rig:
    """The loop over fakes, run on the test's event loop."""

    def __init__(
        self,
        tmp: Path,
        texts: list[str],
        script: list[list[Step]] | None = None,
        *,
        tool: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        brain_delay_s: float = 0.0,
        whisper_delay_s: float = 0.0,
        seconds_per_char: float = 0.01,
        switchboard: Switchboard | None = None,
        observers: Sequence[Any] = (),
        flags: Sequence[Any] = (),
        vocabulary: Vocabulary = EMPTY,
        people: Any = None,
        clock: Callable[[], float] = time.monotonic,
        **config: Any,
    ) -> None:
        self.config = CascadeConfig().with_changes(
            prosody_tags=False, playback_lead_s=0.1, idle_close_s=30.0, **config
        )
        self.mic = PushMic()
        self.brain = ScriptedBrain(script=script or [], delay_s=brain_delay_s)
        self.voice = ToneVoice(rate=RATE, seconds_per_char=seconds_per_char, chunk_s=0.1)
        self.player = Player(NullSpeaker(), rate=RATE, lead_s=0.1)
        self.speech = SpeechWorker(self.voice, self.player)
        self.transcriber = ListTranscriber(texts, whisper_delay_s)
        self.states: list[str] = []
        events = Events(lambda e: self.states.append(e.state) if e.state != "speaking" else None)
        self.loop = CascadeLoop(
            self.config,
            mic=self.mic,
            player=self.player,
            transcriber=self.transcriber,
            tagger=None,
            brain=self.brain,
            speech=self.speech,
            tools=tool or (lambda name, args: {"ok": True, "places": [{"name": "home"}]}),
            allowed={"list_places", "go_to", "where_am_i"},
            moving={"go_to"},
            halt=lambda: "halted",
            events=events,
            logs=tmp,
            ledger=None,
            tick_s=0.01,
            switchboard=switchboard,
            observers=observers,
            flags=flags,
            vocabulary=vocabulary,
            people=people,
            clock=clock,
        )
        self.stop = threading.Event()
        self.speech.start()

    async def run(self, scenario: Callable[["Rig"], Any]) -> None:
        task = asyncio.create_task(self.loop.run(until=self.stop))
        try:
            await until(lambda: self.mic.sink is not None)
            await scenario(self)
        finally:
            self.stop.set()
            await asyncio.wait_for(task, 5.0)
            self.speech.close()

    async def say(self, frames: list[bytes]) -> None:
        await asyncio.to_thread(self.mic.push, frames)


async def until(condition: Callable[[], bool], timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition():
        if time.monotonic() > deadline:
            raise AssertionError("timed out")
        await asyncio.sleep(0.01)


def test_only_the_name_opens_and_a_turn_is_timed_and_logged(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["Продолжение следует...", "просто разговор", "Пепин, привет"])

    async def scenario(rig: Rig) -> None:
        for _ in range(3):
            await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet())
        await until(lambda: "listening" in rig.states)

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard == ["Пепин, привет"]
    assert rig.brain.outcomes == ["committed"]
    assert rig.brain.places == ["home"]
    (turn,) = rig.loop.turns
    assert turn["reply"] == "Ты сказал: Пепин, привет" and turn["outcome"] == "committed"
    ms = turn["ms"]
    assert ms["closed"] == pytest.approx(400, abs=40)
    keys = ["closed", "transcript", "first_token", "first_sentence", "tts_first_audio",
            "speaker_start", "speaker_end"]  # fmt: skip
    assert all(ms[k] is not None for k in keys)
    assert [ms[k] for k in keys] == sorted(ms[k] for k in keys)
    day = next(tmp_path.iterdir())
    heard = [json.loads(line) for line in (day / "cascade_heard.jsonl").read_text().splitlines()]
    assert [(r["noise"], r["wake"]) for r in heard] == [
        (True, False),
        (False, False),
        (False, True),
    ]
    assert {r["engine"] for r in heard} == {"fake:list"}
    (conversation,) = day.glob("cascade_*/")
    summary = json.loads((conversation / "summary.json").read_text())
    assert summary["turns"] == 1 and summary["close"] == "stopped"
    events = [json.loads(line)["event"] for line in (conversation / "events.jsonl").open()]
    assert events[0] == "open" and "turn" in events and events[-1] == "close"
    assert (conversation / "in.wav").stat().st_size > 44 and (conversation / "out.wav").exists()
    assert "thinking" in rig.states and "listening" in rig.states


def test_words_said_before_the_robot_answers_join_the_same_turn(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["Пепин, скажи", "ещё кое-что"], brain_delay_s=1.0)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.current is not None)
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.brain.outcomes == ["abandoned", "committed"]
    assert rig.brain.heard == ["Пепин, скажи", "Пепин, скажи ещё кое-что"]
    assert [t["outcome"] for t in rig.loop.turns] == ["merged", "committed"]


def test_the_rest_of_a_sentence_said_before_the_answer_plays_joins_it(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["Пепин, какой сегодня", "день недели?"], brain_delay_s=1.0)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())  # closed after its silence; the brain starts thinking
        await until(lambda: rig.loop.current is not None)
        await rig.say(tone(0.4) + quiet(0.7))  # the rest of the sentence, within 0.8 s
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard[-1] == "Пепин, какой сегодня день недели?"
    assert rig.brain.outcomes == ["abandoned", "committed"]
    merged, answered = rig.loop.turns
    assert merged["outcome"] == "merged" and answered["merged"] == 2


def test_an_utterance_still_being_read_waits_for_the_rest(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["Пепин, какой сегодня", "день недели?"], whisper_delay_s=0.3)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.hearing > 0)  # Whisper is reading the first part
        await rig.say(tone(0.4) + quiet(0.7))
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard == ["Пепин, какой сегодня день недели?"]
    assert rig.loop.turns[0]["merged"] == 2


def test_a_held_utterance_is_answered_alone_when_only_noise_follows(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["Пепин, привет", "Продолжение следует..."], whisper_delay_s=0.3)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.hearing > 0)
        await rig.say(tone(0.4) + quiet(0.7))  # a cough Whisper reads as its noise line
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard == ["Пепин, привет"]
    assert "merged" not in rig.loop.turns[0]


def test_noise_after_a_phrase_neither_holds_the_turn_nor_restarts_the_brain(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, ["Пепин, привет", "Продолжение следует..."], brain_delay_s=1.0)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())  # closed; the brain starts thinking
        await until(lambda: rig.loop.current is not None)
        await rig.say(tone(0.4) + quiet(0.7))  # a rustle within 0.8 s: Whisper's noise line
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard == ["Пепин, привет"]  # one call, never cancelled
    assert rig.brain.outcomes == ["committed"]
    (turn,) = rig.loop.turns
    assert turn["outcome"] == "committed" and "merged" not in turn
    assert rig.voice.spoken == ["Ты сказал: Пепин, привет"]
    day = next(tmp_path.iterdir())
    (conversation,) = day.glob("cascade_*/")
    kinds = [json.loads(line)["event"] for line in (conversation / "events.jsonl").open()]
    assert "held" not in kinds
    assert [k for k in kinds if k in ("muted", "resumed")] == ["muted", "resumed"]


def test_a_tool_asked_for_while_the_person_goes_on_never_runs_when_words_follow(
    tmp_path: Path,
) -> None:
    calls: list[str] = []

    def tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        calls.append(name)
        return {"ok": True, "places": [{"name": "home"}]}

    script: list[list[Step]] = [[("where_am_i", {}), "Я дома."], ["Понял, на кухню."]]
    rig = Rig(
        tmp_path,
        ["Пепин, где ты", "и поезжай на кухню"],
        script,
        tool=tool,
        brain_delay_s=0.15,
        whisper_delay_s=0.4,
    )

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.current is not None)
        await rig.say(tone(0.4) + quiet(0.7))  # the rest, read after the brain asked a tool
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    day = next(tmp_path.iterdir())
    (conversation,) = day.glob("cascade_*/")
    kinds = [json.loads(line)["event"] for line in (conversation / "events.jsonl").open()]
    assert "tool_call" in kinds and "tool_result" not in kinds  # asked for while silent
    assert [c for c in calls if c != "list_places"] == []  # it waited, then was dropped
    assert rig.brain.outcomes == ["abandoned", "committed"]
    assert rig.brain.heard[-1] == "Пепин, где ты и поезжай на кухню"
    merged, answered = rig.loop.turns
    assert merged["outcome"] == "merged" and answered["merged"] == 2
    summary = json.loads((conversation / "summary.json").read_text())
    assert summary["turns"] == 1  # the merged fragment is part of the turn after it


def test_a_barge_in_cuts_the_reply_to_what_was_heard_and_answers_the_new_words(
    tmp_path: Path,
) -> None:
    long_reply: list[Step] = ["Раз два три четыре. Пять шесть семь восемь. Девять десять."]
    rig = Rig(
        tmp_path,
        ["Пепин, считай", "хватит"],
        [long_reply, ["Хорошо."]],
        seconds_per_char=0.03,
        barge_in=True,
    )

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.player.playback() == "reply")
        await asyncio.sleep(0.8)  # the first sentence (0.6 s) has played
        await rig.say(tone(0.4) + quiet(0.7))
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.brain.outcomes == ["interrupted", "committed"]
    (played,) = rig.brain.interrupts
    assert played.sentences == 1 and played.partial.startswith("Пять")
    assert rig.brain.heard == ["Пепин, считай", "хватит"]
    cut = rig.loop.turns[0]
    assert cut["outcome"] == "interrupted" and cut["heard"]["sentences"] == 1
    assert cut["ms"]["speaker_end"] is not None
    (events,) = tmp_path.glob("*/cascade_*/events.jsonl")
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    (barge,) = [r for r in rows if r["event"] == "barge_in"]
    assert barge["turn"] == cut["turn"] and barge["why"].startswith("level")
    assert barge["played_s"] > 0.6 and barge["doa"] is None  # a fake mic has no direction


def test_a_pause_in_a_long_utterance_gets_a_backchannel_before_the_answer(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, ["Пепин", "послушай меня"], backchannel=True)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())  # the name opens the conversation (no backchannel before)
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet())
        await rig.say(quiet(0.2) + tone(1.0) + quiet(0.7))
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet())

    asyncio.run(rig.run(scenario))
    assert rig.voice.spoken[0] == "Ты сказал: Пепин"
    assert rig.voice.spoken[1] in ("угу", "мм")
    assert rig.voice.spoken[2] == "Ты сказал: послушай меня"


def test_a_brain_that_fails_before_a_word_is_covered_by_the_local_error_line(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, ["Пепин, привет"])

    async def broken(text: str, tools: Any, emit: Any) -> None:
        raise RuntimeError("the API is down")

    rig.brain.turn = broken  # type: ignore[method-assign]

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet())

    asyncio.run(rig.run(scenario))
    assert rig.voice.spoken == [rig.config.error_line]
    assert rig.loop.turns[0]["outcome"] == "error"
    assert rig.brain.outcomes == ["abandoned"]


def test_a_drive_is_acknowledged_when_it_starts_and_reported_when_it_ends(tmp_path: Path) -> None:
    clock = WallClock()
    goals = FakeGoalServer(clock, step_s=0.3, accept_s=0.05)
    robot = fake_robot(clock=clock, goals=goals)
    script: list[list[Step]] = [
        ["Еду.", ("go_to", {"place": "printer"})],
        ["Я на месте."],
    ]
    rig = Rig(
        tmp_path,
        ["Пепин, езжай к принтеру"],
        script,
        tool=lambda name, args: TOOLS.call(name, args, robot),
    )

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: len(rig.loop.turns) == 2 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    first, report = rig.loop.turns
    assert first["tools"][0]["name"] == "go_to"
    result = first["tools"][0]["result"]
    assert "driving" in result and "under way" in result and "already said" in result
    assert "follow_up" not in result
    assert report["kind"] == "robot" and report["reply"] == "Я на месте."
    assert rig.brain.heard[1].startswith("[robot] the drive go_to(place='printer') has ended")
    assert '"arrived": true' in rig.brain.heard[1]
    assert rig.loop.person_turns == 1


def test_a_style_holds_for_the_reply_and_a_switch_takes_the_next_turn(tmp_path: Path) -> None:
    late_brain = ScriptedBrain(script=[["Теперь я другой."]])
    late_voice = ToneVoice(rate=RATE, seconds_per_char=0.01, chunk_s=0.1)
    built: list[str] = []

    def build_brain(config: CascadeConfig, name: str) -> BrainSetup:
        built.append(name)
        return BrainSetup(late_brain, config, name, {"where_am_i"}, set(), paid=False)

    board = Switchboard(
        ControlFile(tmp_path),
        configure=lambda brain, voice: CascadeConfig().with_changes(
            voice=VoiceConfig(preset=voice, engine="say")
        ),
        build_brain=build_brain,
        build_voice=lambda v: late_voice,
        brain="first",
        voice="first-voice",
    )
    script: list[list[Step]] = [["{грустно} Мне жаль.", "Но всё пройдёт.", "{} Ладно."]]
    rig = Rig(tmp_path, ["Пепин, грустно", "Пепин, а теперь?"], script, switchboard=board)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)
        request = write_request(tmp_path, "second", "second-voice")

        def applied() -> int:
            return sum(r["state"] == "applied" for r in answers_to(tmp_path, request.id))

        await until(lambda: applied() == 2)
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    first, second = rig.loop.turns
    assert rig.voice.spoken == ["Мне жаль.", "Но всё пройдёт.", "Ладно."]  # markers unsaid
    assert rig.voice.styles == ["грустно", "грустно", ""]  # sticky until {} resets it
    assert first["styles"] == ["грустно", ""] and first["brain"] == "first"
    assert built == ["second"] and late_brain.heard == ["Пепин, а теперь?"]
    assert second["brain"] == "second" and second["voice"] == "second-voice"
    assert late_voice.spoken[-1] == "Теперь я другой." and rig.loop.allowed == {"where_am_i"}
    assert rig.brain.heard == ["Пепин, грустно"]  # the first brain never heard the second


class BrokenObserver(RecordingObserver):
    """An observer that breaks on every part: the voice must not notice."""

    def part_queued(self, turn: int, index: int, part: Any) -> None:
        raise RuntimeError("broken body")


def test_observers_hear_every_moment_and_a_flag_switches_live(tmp_path: Path) -> None:
    board = Switchboard(
        ControlFile(tmp_path),
        configure=lambda brain, voice: CascadeConfig(),
        build_brain=lambda config, name: BrainSetup(
            ScriptedBrain(), config, name, set(), set(), paid=False
        ),
        build_voice=lambda v: ToneVoice(rate=RATE),
        brain="first",
        voice="first-voice",
    )
    observer, flag = RecordingObserver(), FakeFlag(GESTURES)
    script: list[list[Step]] = [[("look", {"pan_deg": 10.0}), "{тихо} Вижу стол.", "Всё."]]
    rig = Rig(tmp_path, ["Пепин, что там?"], script, switchboard=board,
              observers=[BrokenObserver(), observer], flags=[flag],
              fillers=FillerLines(("Секунду.",), (), {}))  # fmt: skip

    asked: list[str] = []

    async def scenario(rig: Rig) -> None:
        request = write_request(tmp_path, None, None, flags={GESTURES: True, "nope": True})
        asked.append(request.id)
        await until(lambda: len(answers_to(tmp_path, request.id)) == 2)
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert flag.on and rig.voice.spoken == ["Секунду.", "Вижу стол.", "Всё."]  # filler first
    answers = {r["name"]: r for r in answers_to(tmp_path, asked[0])}
    assert answers[GESTURES]["state"] == "applied" and answers[GESTURES]["what"] == "flag"
    assert answers["nope"]["state"] == "failed" and "gestures" in answers["nope"]["error"]
    moments = [name for name, _ in observer.calls]
    assert moments[:3] == ["opened", "speech_ended", "moment"] and moments[-1] == "closed"
    assert moments.count("part_queued") == 2 and moments.count("part_spoken") == 2
    for index in (0, 1):  # a part is spoken after it was queued, and the reply ends after both
        queued = next(k for k, (name, args) in enumerate(observer.calls)
                      if name == "part_queued" and args[1] == index)  # fmt: skip
        assert observer.calls.index(("part_spoken", (1, index))) > queued
    assert moments.index("reply_done") > max(i for i, m in enumerate(moments) if m == "part_spoken")
    (wake,) = [args[0] for name, args in observer.calls if name == "opened"]
    assert isinstance(wake, Heard) and wake.text == "Пепин, что там?" and wake.duration_s > 0
    assert ("moment", (SLOW_TOOL,)) in observer.calls
    parts = [args for name, args in observer.calls if name == "part_queued"]
    assert [(turn, index, part.text, part.style) for turn, index, part in parts] == [
        (1, 0, "Вижу стол.", "тихо"), (1, 1, "Всё.", None),
    ]  # fmt: skip
    assert observer.calls[-1] == ("closed", ("stopped",))


class SkewClock:
    """The loop's clock: monotonic time moved on by ``skew`` seconds when the test says so."""

    def __init__(self) -> None:
        self.skew = 0.0

    def __call__(self) -> float:
        return time.monotonic() + self.skew


def slow_tool(release: threading.Event) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Tools that answer at once, but for ``look``, which waits for ``release``."""

    def run(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "look":
            assert release.wait(5.0)
        return {"ok": True, "places": [{"name": "home"}]}

    return run


def test_a_slow_tool_gets_its_own_filler_and_a_second_one_when_the_wait_goes_on(
    tmp_path: Path,
) -> None:
    release, clock = threading.Event(), SkewClock()
    script: list[list[Step]] = [[("remember_person", {"name": "Саша"}), ("look", {}), "Готово."]]
    rig = Rig(tmp_path, ["Пепин, запомни Сашу"], script, tool=slow_tool(release), clock=clock)
    rig.loop.allowed |= {"remember_person", "look"}
    said: list[list[str]] = []

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.current is not None and bool(rig.loop.current.fillers))
        reply = rig.loop.current
        assert reply is not None
        clock.skew = 3.0  # under filler_again_s (4 s) since the first filler: nothing more
        await asyncio.sleep(0.2)
        said.append(list(reply.fillers))
        clock.skew = 4.5  # past it, the tool still running and nothing said
        await until(lambda: len(reply.fillers) == 2)
        clock.skew = 20.0  # a second filler at most
        await asyncio.sleep(0.2)
        said.append(list(reply.fillers))
        release.set()
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    fillers = CascadeConfig().fillers
    (first,) = said[0]
    assert first in fillers.for_tool("remember_person")  # its own, not the look's
    again = said[1][1]
    assert said[1] == [first, again] and again in fillers.again
    assert rig.voice.spoken == [first, again, "Готово."]
    (turn,) = rig.loop.turns
    assert turn["fillers"] == [first, again]
    (events,) = tmp_path.glob("*/cascade_*/events.jsonl")
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    logged = [(r["tool"], r["text"]) for r in rows if r["event"] == "filler"]
    assert logged == [("remember_person", first), (None, again)]


def test_no_filler_speaks_once_the_reply_s_first_sentence_has_started(tmp_path: Path) -> None:
    release, clock = threading.Event(), SkewClock()
    script: list[list[Step]] = [[("see", {}), "Вижу стол.", ("look", {}), "Всё."]]
    rig = Rig(tmp_path, ["Пепин, что там?"], script, tool=slow_tool(release), clock=clock)
    rig.loop.allowed |= {"see", "look"}

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.current is not None and bool(rig.loop.current.sentences))
        clock.skew = 30.0  # the look runs on long after the sentence: no filler for it
        await asyncio.sleep(0.3)
        release.set()
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    (first, *rest) = rig.voice.spoken
    assert first in CascadeConfig().fillers.for_tool("see")
    assert rest == ["Вижу стол.", "Всё."]
    assert rig.loop.turns[0]["fillers"] == [first]


class FakeSpeakers:
    """Person recognition that names everyone Sasha; counts what it was asked."""

    def __init__(self, on: bool) -> None:
        self.on = on
        self.measured = 0
        self.named: list[str] = []

    def measure(self, pcm: bytes, end_s: float) -> Callable[[str], Speaker]:
        self.measured += 1

        def name(text: str) -> Speaker:
            self.named.append(text)
            return Speaker("говорит: Саша", "sasha", {"who": "Саша", "confidence": "sure"})

        return name


class TimedList(ListTranscriber):
    """A transcriber that times its words: each text read as one word with a 0.3 s vowel."""

    def transcribe_timed(self, pcm: bytes) -> Transcript:
        text = self.transcribe(pcm)
        units = (Unit("д", 0.0, 0.05), Unit("а", 0.05, 0.35))
        return Transcript(
            text, (TimedWord("да", 0.0, 0.35, units, True, -0.2, (Vowel(0.05, 0.35, 1),)),)
        )


class WordsTagger:
    """A tagger that keeps the words each line was given and tags with their count."""

    def __init__(self) -> None:
        self.words: list[tuple[TimedWord, ...]] = []

    def measure(self, pcm: bytes) -> Prosody:
        return Prosody(1.0, 60.0)

    def line(self, prosody: Prosody, text: str) -> str:
        self.words.append(prosody.words)
        return f"[{len(prosody.words)} timed]"

    def describe(self, prosody: Prosody, text: str, key: str) -> str:
        return ""


def test_a_timed_read_s_words_reach_the_tag_line_and_the_log(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["Пепин, привет", "да"])
    tagger = WordsTagger()
    rig.loop.transcriber = TimedList(["Пепин, привет", "да"])
    rig.loop.tagger = tagger
    rig.loop.config = rig.config.with_changes(prosody_tags=True)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet())
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet())

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard == ["[1 timed] Пепин, привет", "[1 timed] да"]
    assert [w.text for words in tagger.words for w in words] == ["да", "да"]
    (conversation,) = next(tmp_path.iterdir()).glob("cascade_*/")
    rows = [json.loads(line) for line in (conversation / "events.jsonl").open()]
    heard = [r for r in rows if r["event"] == "heard"]
    assert heard and all(r["words"] == [{"w": "да", "t": [0.0, 0.35], "lp": -0.2, "v": [0.3]}]
                         for r in heard)  # fmt: skip


def test_the_speaker_is_named_in_front_of_the_words_only_while_the_switch_is_on(
    tmp_path: Path,
) -> None:
    people = FakeSpeakers(on=True)
    rig = Rig(tmp_path, ["просто разговор", "Пепин, привет", "Продолжение следует...", "как дела"],
              people=people)  # fmt: skip

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())  # idle and no name: nobody is measured
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet())
        await rig.say(utterance())  # Whisper's noise line: measured, never named
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet())

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard == ["[говорит: Саша] Пепин, привет", "[говорит: Саша] как дела"]
    assert people.measured == 3 and people.named == ["Пепин, привет", "как дела"]
    (conversation,) = next(tmp_path.iterdir()).glob("cascade_*/")
    rows = [json.loads(line) for line in (conversation / "events.jsonl").open()]
    speakers = [r for r in rows if r["event"] == "speaker"]
    assert [(r["head"], r["who"]) for r in speakers] == [("говорит: Саша", "Саша")] * 2

    off = FakeSpeakers(on=False)
    rig = Rig(tmp_path / "off", ["Пепин, привет"], people=off)

    async def once(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet())

    asyncio.run(rig.run(once))
    assert rig.brain.heard == ["Пепин, привет"] and off.measured == 0


def test_markers_ride_their_parts_to_the_body_and_never_reach_the_voice(tmp_path: Path) -> None:
    observer = RecordingObserver()
    script: list[list[Step]] = [["Ну {laugh} да, {тихо} все спят.", "{gesture:nope} Ладно."]]
    rig = Rig(tmp_path, ["Пепин, привет"], script, observers=[observer],
              vocabulary=Vocabulary(frozenset({"laugh"}), frozenset()))  # fmt: skip

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.voice.spoken == ["Ну да,", "все спят.", "Ладно."]
    assert rig.voice.styles == ["", "тихо", "тихо"]
    parts = [args for name, args in observer.calls if name == "part_queued"]
    assert [(index, part.text, [m.name for m in part.markers]) for _, index, part in parts] == [
        (0, "Ну да,", ["laugh"]), (1, "все спят.", []), (2, "Ладно.", []),
    ]  # fmt: skip
    (turn,) = rig.loop.turns
    assert turn["reply"] == "Ну {laugh} да, {тихо} все спят. {gesture:nope} Ладно."  # as written
    (events,) = tmp_path.glob("*/cascade_*/events.jsonl")
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    sentences = [r for r in rows if r["event"] == "sentence"]
    assert [r["parts"] for r in sentences] == [[0, 1], [2]]
    assert sentences[0]["markers"] == ["{laugh}"] and sentences[1]["dropped"] == ["{gesture:nope}"]


def test_a_marker_moves_the_body_on_its_word_and_the_tool_reaches_the_same_engine(
    tmp_path: Path,
) -> None:
    """The gesture engine as the loop's observer, on the real player's clock: the head's clip
    of ``{agreement}`` goes once its part's audio has played up to the word (less the clip's
    preparation), not before the part is queued; a barge-in would drop it; the flag is live."""
    from pepin.expression.animator import EngineConfig, GestureEngine
    from pepin.expression.book import load_book
    from pepin.expression.fakes import FakeExecutor
    from pepin.expression.parts import Absent

    book = load_book(Path(__file__).resolve().parents[2] / "config" / "animation.json")
    head, face = FakeExecutor("head"), FakeExecutor("face")
    records: list[tuple[str, dict[str, Any]]] = []
    script: list[list[Step]] = [["Ну что ж. Я, {agreement} пожалуй, соглашусь с тобой."]]
    rig = Rig(tmp_path, ["Пепин, согласен?"], script, seconds_per_char=0.03)
    engine = GestureEngine(
        {"head": head, "face": face, "arm": Absent("arm", "test"), "body": Absent("body", "-")},
        rig.player,
        lambda: (book, EngineConfig(arm_speed_scale=1.0, seed=5, audio_lag_s=0.0)),
        on=True,
        record=lambda kind, fields: records.append((kind, fields)),
    ).start()
    rig.loop.observers.append(engine)
    rig.loop.vocabulary = Vocabulary(frozenset(book.compositions), frozenset())

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 10.0)

    try:
        asyncio.run(rig.run(scenario))
    finally:
        engine.close()
    assert rig.voice.spoken == ["Ну что ж.", "Я, пожалуй, соглашусь с тобой."]
    (performance,) = [f for k, f in records if k == "performance"]
    assert performance["intent"] == "agreement" and performance["part"] == 1
    cue = next(f for k, f in records if k == "cue" and f["part"] == "head")
    assert cue["ok"] and abs(cue["late_s"]) < 0.1  # sent on the player's clock, on time
    # the word: the second part starts after the first's 0.27 s, the marker 0.1 into its 0.9 s;
    # the nod's clip starts its preparation (its stroke less ~0.1 s) before the word
    assert 0.27 + 0.09 - 0.45 < cue["planned"] < 0.27 + 0.09
    assert [c[0] for c in head.calls].count("play") == 1


class BrokenVoice(ToneVoice):
    """A voice whose model never loads."""

    def load(self) -> None:
        raise RuntimeError("no weights")


def test_a_failed_switch_keeps_what_runs_and_a_later_switch_supersedes_a_slow_one(
    tmp_path: Path,
) -> None:
    slow_gate = threading.Event()
    brains = {"slow": ScriptedBrain(), "fast": ScriptedBrain()}

    def build_brain(config: CascadeConfig, name: str) -> BrainSetup:
        if name == "broken":
            raise ValueError("no such preset")
        if name == "slow":
            assert slow_gate.wait(5.0)
        return BrainSetup(brains[name], config, name, {"where_am_i"}, set(), paid=False)

    board = Switchboard(
        ControlFile(tmp_path),
        configure=lambda brain, voice: CascadeConfig().with_changes(
            voice=VoiceConfig(preset=voice, engine="say")
        ),
        build_brain=build_brain,
        build_voice=lambda v: BrokenVoice(rate=RATE),
        brain="first",
        voice="first-voice",
    )
    rig = Rig(tmp_path, [], switchboard=board)

    asked: dict[str, str] = {}

    def states(name: str) -> list[str]:
        return [f"{r['what']} {r['state']}" for r in answers_to(tmp_path, asked[name])]

    async def scenario(rig: Rig) -> None:
        try:
            asked["broken"] = write_request(tmp_path, "broken", "bad-voice").id
            await until(lambda: {"brain failed", "voice failed"} <= set(states("broken")))
            asked["slow"] = write_request(tmp_path, "slow", None).id
            await until(lambda: states("slow") == ["brain loading"])
            asked["fast"] = write_request(tmp_path, "fast", None).id
            await until(lambda: states("fast")[-1:] == ["brain applied"])
        finally:
            slow_gate.set()
        await until(lambda: states("slow")[-1:] == ["brain failed"])

    asyncio.run(rig.run(scenario))
    assert rig.speech.voice is rig.voice and rig.loop.voice_name == "qwen3"  # the old voice
    assert rig.loop.brain is brains["fast"] and rig.loop.brain_name == "fast"
    superseded = answers_to(tmp_path, asked["slow"])[-1]
    assert superseded["error"] == "superseded by a later switch"


def test_what_a_conversation_spent_reaches_the_ledger_whatever_brain_closes_it(
    tmp_path: Path,
) -> None:
    from pepin.voice_live.budget import Ledger
    from pepin.voice_live.config import LiveConfig

    rig = Rig(tmp_path, [])
    ledger = Ledger(tmp_path / "ledger.jsonl", LiveConfig())
    rig.loop.paid_ledger = ledger  # the run's ledger; the echo brain itself records nothing

    async def conversation() -> None:
        assert rig.loop.ledger is None and rig.loop.open(b"")
        assert rig.loop.conversation is not None
        rig.loop.conversation.voice_usd = 0.002  # a paid voice, or a paid brain switched out
        rig.loop.close("idle")
        assert rig.loop.open(b"")
        rig.loop.close("idle")  # nothing spent: nothing recorded

    asyncio.run(conversation())
    rig.speech.close()
    assert ledger.today_usd() == pytest.approx(0.002)
    assert len((tmp_path / "ledger.jsonl").read_text().splitlines()) == 1


# -- the brain asked at the early read (brain.speculative) ------------------------------------

EARLY = BrainConfig(speculative=True)
WINDOW_S = 0.3  # ears.continuation_s here: an utterance after it is a turn of its own


async def opened(rig: Rig) -> None:
    """The name said and answered, and the continuation window over."""
    await rig.say(utterance())
    await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)
    await asyncio.sleep(WINDOW_S)


def events_of(tmp_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The conversation's events and its summary."""
    day = next(tmp_path.iterdir())
    (conversation,) = day.glob("cascade_*/")
    events = [json.loads(line) for line in (conversation / "events.jsonl").open()]
    return events, json.loads((conversation / "summary.json").read_text())


def test_asked_early_the_reply_waits_for_the_close_and_goes_on_with_the_same_words(
    tmp_path: Path,
) -> None:
    observer = RecordingObserver()
    rig = Rig(
        tmp_path,
        ["Пепин, привет", "как у тебя дела"],
        observers=[observer],
        brain=EARLY,
        continuation_s=WINDOW_S,
    )

    async def scenario(rig: Rig) -> None:
        await opened(rig)
        await rig.say(quiet(0.2) + tone(0.4) + quiet(0.2))  # read in the pause, not yet closed
        reply = lambda: rig.loop.current  # noqa: E731
        await until(lambda: reply() is not None and reply().input.timeline.has("first_token"))
        assert rig.voice.spoken == ["Ты сказал: Пепин, привет"]  # nothing said before the close
        await rig.say(quiet(0.5))
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard == ["Пепин, привет", "как у тебя дела"]  # asked once, not again
    assert rig.brain.outcomes == ["committed", "committed"]
    first, second = rig.loop.turns
    assert "speculative" not in first  # the name is read before a conversation is open
    ms = second["ms"]
    assert second["speculative"]["asked_ms"] < ms["closed"]
    assert ms["first_token"] < ms["closed"] <= ms["tts_first_audio"]
    assert rig.voice.spoken[-1] == "Ты сказал: как у тебя дела"
    calls = observer.calls
    ended = max(i for i, (name, _) in enumerate(calls) if name == "speech_ended")
    queued = [i for i, (name, args) in enumerate(calls) if name == "part_queued" and args[0] == 2]
    assert queued and min(queued) > ended  # the body hears the part once the voice has it
    events, summary = events_of(tmp_path)
    states = [e["state"] for e in events if e["event"] == "speculative"]
    assert states == ["asked", "kept"]
    assert summary["speculative"] == {"asked": 1, "kept": 1, "dropped": 0, "wasted_usd": 0.0}


def test_asked_early_then_dropped_when_the_person_goes_on_and_asked_again_with_all_words(
    tmp_path: Path,
) -> None:
    from pepin.voice_live.budget import Ledger
    from pepin.voice_live.config import LiveConfig

    texts = ["Пепин, привет", "какой сегодня день", "какой сегодня день недели?"]
    rig = Rig(tmp_path, texts, brain_delay_s=0.3, brain=EARLY, continuation_s=WINDOW_S)
    ledger = Ledger(tmp_path / "ledger.jsonl", LiveConfig())
    rig.loop.paid_ledger = ledger  # the fake brain's normal calls are free, a cut one is not

    async def scenario(rig: Rig) -> None:
        await opened(rig)
        await rig.say(quiet(0.2) + tone(0.4) + quiet(0.25))  # a pause mid-sentence: asked
        await until(lambda: rig.loop.current is not None)
        await rig.say(tone(0.4) + quiet(0.7))  # the rest: the same utterance goes on
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard == texts
    assert rig.brain.outcomes == ["committed", "abandoned", "committed"]
    assert [t["outcome"] for t in rig.loop.turns] == ["committed", "speculative", "committed"]
    dropped = rig.loop.turns[1]
    assert dropped["usage"]["stop"] == CUT and dropped["usage"]["usd"] > 0
    assert rig.voice.spoken == ["Ты сказал: Пепин, привет", "Ты сказал: какой сегодня день недели?"]
    events, summary = events_of(tmp_path)
    (why,) = [e["why"] for e in events if e["event"] == "speculative" and e["state"] == "dropped"]
    assert why == "the person went on"
    assert summary["turns"] == 2
    early = summary["speculative"]
    assert (early["asked"], early["kept"], early["dropped"]) == (1, 0, 1)
    assert early["wasted_usd"] == pytest.approx(dropped["usage"]["usd"])
    assert ledger.today_usd() == pytest.approx(dropped["usage"]["usd"])  # the ledger has it


def test_a_brain_failing_before_the_close_is_asked_again_and_says_no_error_line(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, ["Пепин, привет", "как у тебя дела"], brain=EARLY, continuation_s=WINDOW_S)
    asked: list[str] = []
    turn = rig.brain.turn

    async def flaky(text: str, tools: Any, emit: Any) -> None:
        asked.append(text)
        if len(asked) == 2:
            raise RuntimeError("the API is down")
        await turn(text, tools, emit)

    rig.brain.turn = flaky  # type: ignore[method-assign]

    async def scenario(rig: Rig) -> None:
        await opened(rig)
        await rig.say(quiet(0.2) + tone(0.4) + quiet(0.2))
        await until(lambda: len(asked) == 2 and rig.loop.current is None)
        await rig.say(quiet(0.5))
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert asked == ["Пепин, привет", "как у тебя дела", "как у тебя дела"]
    assert rig.voice.spoken == ["Ты сказал: Пепин, привет", "Ты сказал: как у тебя дела"]
    assert [t["outcome"] for t in rig.loop.turns] == ["committed", "speculative", "committed"]
    _, summary = events_of(tmp_path)
    assert (summary["speculative"]["kept"], summary["speculative"]["dropped"]) == (0, 1)


def test_one_word_is_never_asked_early(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["Пепин, привет", "да"], brain=EARLY, continuation_s=WINDOW_S)

    async def scenario(rig: Rig) -> None:
        await opened(rig)
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard == ["Пепин, привет", "да"]
    events, summary = events_of(tmp_path)
    assert not [e for e in events if e["event"] == "speculative"]
    assert "speculative" not in summary and all("speculative" not in t for t in rig.loop.turns)
