"""The whole cascade over fakes: the name gate, a turn's timeline and logs, the person going on
before the robot answered, a barge-in cut to what was heard, and a drive acknowledged when it
starts and reported when it ends."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import asyncio
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.tools import TOOLS
from pepin.tools.fakes import FakeGoalServer, WallClock, fake_robot
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.fake import ListTranscriber, ScriptedBrain, Step, ToneVoice
from pepin.voice_cascade.loop import CascadeLoop
from pepin.voice_cascade.player import Player
from pepin.voice_cascade.speech import SpeechWorker
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
    assert ms["closed"] == pytest.approx(500, abs=40)
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
