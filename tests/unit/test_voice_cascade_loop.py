"""The whole cascade over fakes: the name gate, a turn's timeline and logs, the audio of the
utterances heard outside a conversation, the person going on before the robot answered, a
barge-in cut to what was heard, a drive acknowledged when it starts and reported when it ends
(an arrival at once and in one short line; the goal server's one rule told as it answers:
already under way, or what the new drive replaced), a robot report whose
brain call got no answer (asked again, then carried to the next turn), no cap on a
conversation's length, the fillers (a slow tool's, a late first token's, a pause's
backchannel), and the brain asked at the early read (kept by the close, dropped when the
person goes on, billed either way)."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import asyncio
import itertools
import json
import logging
import re
import threading
import time
import wave
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.expression.conversation import GESTURES, LISTENER_PAUSE, SLOW_TOOL, SPEAKER_GAZE, Heard
from pepin.expression.fakes import FakeFlag, RecordingObserver
from pepin.expression.markers import EMPTY, Vocabulary
from pepin.expression.speaker import Attention
from pepin.perception.source import VisionSource
from pepin.tools import TOOLS
from pepin.tools.fakes import FakeGaze, FakeGoalServer, WallClock, fake_robot
from pepin.voice_cascade.config import BrainConfig, CascadeConfig, FillerBook, VoiceConfig
from pepin.voice_cascade.control import ControlFile, Switchboard, answers_to, write_request
from pepin.voice_cascade.engines import BrainSetup
from pepin.voice_cascade.fake import ListTranscriber, ScriptedBrain, Step, ToneVoice
from pepin.voice_cascade.fillers import clip_lines
from pepin.voice_cascade.loop import CascadeLoop, connection_error, keep_wav
from pepin.voice_cascade.player import FillerFade, Player
from pepin.voice_cascade.speech import SpeechWorker
from pepin.voice_cascade.stages import (
    CUT,
    Prosody,
    Sentence,
    Speaker,
    TimedWord,
    Transcript,
    Unit,
    Vowel,
)
from pepin.voice_live.events import Events
from pepin.voice_live.sources import FrameSink

pytestmark = pytest.mark.slow  # the player paces at real time

FRAME = FRAME_SAMPLES * 2
# the fake goal server's arrival at the printer, as the brain is told it
ARRIVED = "[robot] arrived: printer (3 s, off by 6 cm; go_to(place='printer'), run 7)"


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
        halt: Callable[[], str] | None = None,
        brain_delay_s: float = 0.0,
        whisper_delay_s: float = 0.0,
        seconds_per_char: float = 0.01,
        switchboard: Switchboard | None = None,
        observers: Sequence[Any] = (),
        flags: Sequence[Any] = (),
        vocabulary: Vocabulary = EMPTY,
        people: Any = None,
        clock: Callable[[], float] = time.monotonic,
        clips: bool = False,
        knobs: Sequence[Any] = (),
        cells: Sequence[Callable[[], str]] = (),
        **config: Any,
    ) -> None:
        """``clips``: the voice makes the fillers' clips before the scenario (the only way a
        filler is said; what it made is then cleared from its log); without them, no filler.
        The thinking filler is off, wanted whatever the question's length, and no filler's
        moment passes in silence unless asked; the brain hears no [scene] line unless asked."""
        every_length = replace(FillerBook(), thinking_min_words=0, thinking_chance=((0.0, 1.0),))
        config = {"filler_thinking_s": 0.0, "filler_silent_p": {}, "fillers": every_length,
                  "scene_status": False, **config}  # fmt: skip
        self.config = CascadeConfig().with_changes(
            prosody_tags=False, playback_lead_s=0.1, idle_close_s=30.0, **config
        )
        self.mic = PushMic()
        self.brain = ScriptedBrain(script=script or [], delay_s=brain_delay_s)
        self.voice = ToneVoice(rate=RATE, seconds_per_char=seconds_per_char, chunk_s=0.1)
        self.player = Player(NullSpeaker(), rate=RATE, lead_s=0.1)
        self.asides = clip_lines(self.config) if clips else ()
        self.speech = SpeechWorker(self.voice, self.player, asides=self.asides, clip_idle_s=0.0)
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
            halt=halt or (lambda: "halted"),
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
            knobs=knobs,
            cells=cells,
        )
        self.stop = threading.Event()
        self.speech.start()

    async def run(self, scenario: Callable[["Rig"], Any]) -> None:
        task = asyncio.create_task(self.loop.run(until=self.stop))
        try:
            await until(lambda: self.mic.sink is not None)
            await until(lambda: all(map(self.speech.has_clip, self.asides)))
            self.voice.spoken.clear()  # the clips: from here on, what was said live
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
    assert all(r["seconds"] > 0 and isinstance(r["voice_peak_db"], float) for r in heard)
    (conversation,) = day.glob("cascade_*/")
    summary = json.loads((conversation / "summary.json").read_text())
    assert summary["turns"] == 1 and summary["close"] == "stopped"
    events = [json.loads(line)["event"] for line in (conversation / "events.jsonl").open()]
    assert events[0] == "open" and "turn" in events and events[-1] == "close"
    assert (conversation / "in.wav").stat().st_size > 44 and (conversation / "out.wav").exists()
    assert "thinking" in rig.states and "listening" in rig.states


def heard_rows(logs: Path) -> list[tuple[Path, dict[str, Any]]]:
    """Every idle row written so far, with its day folder."""
    return [
        (path.parent, json.loads(line))
        for path in sorted(logs.glob("*/cascade_heard.jsonl"))
        for line in path.read_text().splitlines()
    ]


def test_an_idle_utterance_keeps_its_audio_beside_its_row_and_a_noise_line_none(
    tmp_path: Path,
) -> None:
    # no early read: the transcriber is called once an utterance, with the whole cut
    rig = Rig(tmp_path, ["Продолжение следует...", "", "просто разговор"], early_s=0.0)

    async def scenario(rig: Rig) -> None:
        for _ in range(3):
            await rig.say(utterance())
        await until(lambda: len(heard_rows(tmp_path)) == 3)

    asyncio.run(rig.run(scenario))  # the stop waits for the files queued
    rows = heard_rows(tmp_path)
    assert [r["noise"] for _, r in rows] == [True, True, False]
    assert [("wav" in r) for _, r in rows] == [False, False, True]
    day, row = rows[2]
    assert re.fullmatch(r"heard/\d{6}_\d{3}\.wav", row["wav"])
    assert sorted(p.name for p in (day / "heard").iterdir()) == [row["wav"].removeprefix("heard/")]
    with wave.open(str(day / row["wav"])) as wav:
        assert (wav.getnchannels(), wav.getsampwidth(), wav.getframerate()) == (1, 2, RATE)
        assert wav.getnframes() * 2 == rig.transcriber.heard[2]  # the whole cut, as read
        assert wav.getnframes() / RATE == pytest.approx(row["seconds"], abs=0.01)
    # named by the local time of the utterance's end, before the row's read
    hh, mm, ss, ms = (int(x) for x in re.findall(r"(\d\d)(\d\d)(\d\d)_(\d{3})", row["wav"])[0])
    local = time.localtime(row["t"])
    row_s = local.tm_hour * 3600 + local.tm_min * 60 + local.tm_sec + row["t"] % 1
    assert (row_s - (hh * 3600 + mm * 60 + ss + ms / 1000)) % 86400 < 2.0


def test_with_heard_audio_off_the_rows_stay_as_they_were(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["просто разговор"], heard_audio=False)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: len(heard_rows(tmp_path)) == 1)

    asyncio.run(rig.run(scenario))
    ((day, row),) = heard_rows(tmp_path)
    assert row["noise"] is False and "wav" not in row
    assert not (day / "heard").exists()
    assert CascadeConfig.from_dict({"ears": {"heard_audio": False}}).heard_audio is False
    assert CascadeConfig.from_dict({}).heard_audio is True


def test_a_heard_wav_that_cannot_be_written_warns_and_leaves_no_file(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    (tmp_path / "day").write_text("a file where the day's folder should be")
    path = tmp_path / "day" / "heard" / "120000_000.wav"
    with caplog.at_level(logging.WARNING, logger="pepin.voice_cascade.loop"):
        keep_wav(path, b"\x00\x00" * 160)
    assert not path.exists()
    assert [r.levelno for r in caplog.records] == [logging.WARNING]


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
    note = "[tools of your reply cut short: where_am_i(): never ran]"  # 2026-10-14: approach
    assert rig.brain.heard[-1] == f"Пепин, где ты и поезжай на кухню\n{note}"
    merged, answered = rig.loop.turns
    assert merged["outcome"] == "merged" and answered["merged"] == 2
    assert merged["calls"] == [{"name": "where_am_i", "args": {}, "outcome": "not run"}]
    assert answered["told"] == note and "calls" not in answered
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


def test_a_tool_running_at_a_barge_in_is_told_unanswered_and_its_late_answer_comes_next(
    tmp_path: Path,
) -> None:
    """The person cuts in while a tool runs (its thread goes on): the next turn's words carry
    the call and "no answer yet"; once it answers, the turn after carries the answer."""
    release = threading.Event()

    def tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "where_am_i":
            release.wait(5.0)
            return {"ok": True, "place": "kitchen"}
        return {"ok": True, "places": [{"name": "home"}]}

    script: list[list[Step]] = [
        ["Сейчас посмотрю, где я нахожусь, одну секунду.", ("where_am_i", {}), "На кухне."],
        ["Хорошо."],
        ["Я на кухне."],
    ]
    rig = Rig(
        tmp_path,
        ["Пепин, где ты", "подожди", "ну что"],
        script,
        tool=tool,
        seconds_per_char=0.03,
        barge_in=True,
    )

    async def scenario(rig: Rig) -> None:
        try:
            await rig.say(utterance())
            await until(lambda: rig.player.playback() == "reply")
            await asyncio.sleep(0.5)  # the sentence plays while the tool runs
            await rig.say(tone(0.4) + quiet(0.7))
            await until(lambda: len(rig.brain.heard) == 2)
        finally:
            release.set()
        await until(lambda: bool(rig.loop.cut_calls))  # the answer came after the cut
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet(), 8.0)
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 3 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.brain.outcomes == ["interrupted", "committed", "committed"]
    assert rig.brain.heard[1:] == [
        "подожди\n[tools of your reply cut short: where_am_i(): no answer yet]",
        "ну что\n[tools of your reply cut short: where_am_i(): answered after the cut:"
        ' {"ok": true, "place": "kitchen"}]',
    ]
    cut, waited, answered = rig.loop.turns
    assert cut["outcome"] == "interrupted"
    assert cut["calls"] == [{"name": "where_am_i", "args": {}, "outcome": "unanswered"}]
    assert waited["told"].endswith("no answer yet]") and "calls" not in waited
    assert answered["told"].endswith('"kitchen"}]')
    (events,) = tmp_path.glob("*/cascade_*/events.jsonl")
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    (late,) = [r for r in rows if r["event"] == "late_result"]
    assert late["name"] == "where_am_i" and late["result"] == '{"ok": true, "place": "kitchen"}'
    assert rig.loop.cut_calls == []


def test_a_drive_cut_before_the_goal_server_took_it_still_reports_its_end(
    tmp_path: Path,
) -> None:
    """A barge-in while the goal server is still taking the drive: the drive goes on (its
    thread is not the brain's), the next words carry it as under way, and its end comes back
    as the robot's own turn, as any drive's; the body hears it end once, at its end."""
    clock = WallClock()
    goals = FakeGoalServer(clock, step_s=0.2, accept_s=1.5)
    robot = fake_robot(clock=clock, goals=goals)
    observer = RecordingObserver()
    script: list[list[Step]] = [
        ["Сейчас поеду к принтеру, подожди немного.", ("go_to", {"place": "printer"})],
        ["Хорошо."],
        ["Я на месте."],
    ]
    rig = Rig(
        tmp_path,
        ["Пепин, езжай к принтеру", "подожди"],
        script,
        tool=lambda name, args: TOOLS.call(name, args, robot),
        seconds_per_char=0.03,
        barge_in=True,
        observers=[observer],
    )

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.player.playback() == "reply")
        await asyncio.sleep(0.5)  # the goal server takes the drive 1.5 s after the call
        await rig.say(tone(0.4) + quiet(0.7))
        await until(lambda: len(rig.loop.turns) == 3 and rig.loop.quiet(), 10.0)

    asyncio.run(rig.run(scenario))
    cut, waited, report = rig.loop.turns
    assert cut["outcome"] == "interrupted"
    assert cut["calls"] == [
        {"name": "go_to", "args": {"place": "printer"}, "outcome": "unanswered"}
    ]
    assert rig.brain.heard[1] == (
        "подожди\n[tools of your reply cut short: go_to(place='printer'): no answer yet"
        " (a [robot] message reports the drive's end)]"
    )
    assert waited["told"] == rig.brain.heard[1].split("\n")[1]
    assert report["kind"] == "robot" and report["reply"] == "Я на месте."
    assert rig.brain.heard[2] == ARRIVED
    assert [args for name, args in observer.calls if name == "drive"] == [(True,), (False,)]
    assert not rig.loop.drives


def test_a_pause_in_a_long_utterance_gets_a_backchannel_before_the_answer(
    tmp_path: Path,
) -> None:
    rig = Rig(tmp_path, ["Пепин", "послушай меня"], backchannel=True, clips=True)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())  # the name opens the conversation (no backchannel before)
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet())
        await rig.say(quiet(0.2) + tone(1.0) + quiet(0.7))
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet())

    asyncio.run(rig.run(scenario))
    assert rig.voice.spoken == ["Ты сказал: Пепин", "Ты сказал: послушай меня"]
    (events,) = tmp_path.glob("*/cascade_*/events.jsonl")
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    kinds = [r["event"] for r in rows if r["event"] in ("backchannel", "turn")]
    assert kinds == ["turn", "backchannel", "turn"]  # through the fillers, from its clip
    (back,) = [r for r in rows if r["event"] == "backchannel"]
    assert back["text"] in CascadeConfig().fillers.backchannel["ru"] and back["language"] == "ru"


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
    assert rig.brain.heard[1] == ARRIVED
    assert rig.loop.person_turns == 1


def driving_rig(
    tmp_path: Path, texts: list[str], script: list[list[Step]], step_s: float = 0.3
) -> tuple[Rig, FakeGoalServer]:
    """A rig whose tools drive a fake goal server in real time (its halt cancels there)."""
    clock = WallClock()
    goals = FakeGoalServer(clock, step_s=step_s, accept_s=0.05)
    robot = fake_robot(clock=clock, goals=goals)
    tool = lambda name, args: TOOLS.call(name, args, robot)  # noqa: E731
    return Rig(tmp_path, texts, script, tool=tool, halt=robot.halt), goals


async def mid_drive(rig: Rig, turns: int) -> None:
    """The name and a drive asked, then the next words while the drive runs, until ``turns``
    turns are done and nothing is in flight."""
    await rig.say(utterance())
    await until(lambda: len(rig.loop.turns) == 1 and bool(rig.loop.drives))
    await rig.say(utterance())
    await until(lambda: len(rig.loop.turns) == turns and rig.loop.quiet(), 10.0)


@dataclass
class RuledGoals(FakeGoalServer):
    """The goal server's one rule over the fake, in real time (pepin_bringup.goal_server's
    ``_admit``): one run at a time; the same place asked while it runs is ``already`` with its
    run; another place cancels it (its ``done`` CANCELED, naming the request) and starts once
    that run has ended, with ``replaced``. A run is ``accepted``, ``legs`` steps of ``step_s``,
    then ``arrival`` (``on_arrival`` runs just before), and ``done`` after ``hold_s`` more (the
    pivot, the tape's close); ``read_past_arrival`` says whether anyone waited for that."""

    legs: int = 4
    hold_s: float = 2.0
    on_arrival: Callable[[], None] | None = None
    read_past_arrival: bool = False
    started: list[int] = field(default_factory=list)
    _running: dict[str, Any] | None = None
    _runs: Iterator[int] = field(default_factory=lambda: itertools.count(7))
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def go(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        self._ask(request)
        place = str(request["place"])
        with self._lock:
            ahead = self._running
            if ahead is not None and ahead["place"] == place and not ahead["stop"].is_set():
                detail = f"already under way: go {place} (run {ahead['run']})"
                yield {"event": "already", "run": ahead["run"], "source": "tool",
                       "target": f"go {place}", "detail": detail}  # fmt: skip
                return
            if ahead is not None:
                ahead["cancelled_for"] = f"go {place} (tool)"
                ahead["stop"].set()
        if ahead is not None:
            assert ahead["over"].wait(5.0), "the cancelled run ended"
        run = {"place": place, "run": next(self._runs), "stop": threading.Event(),
               "over": threading.Event(), "cancelled_for": None}  # fmt: skip
        with self._lock:
            self._running = run
        self.started.append(run["run"])
        try:
            if ahead is not None:
                yield {"event": "replaced", "run": ahead["run"], "source": "tool",
                       "target": f"go {ahead['place']}", "cancelled_run": ahead["run"],
                       "started": True}  # fmt: skip
            goal = self.book[place]
            yield {"event": "accepted", "run": run["run"], "place": place, **goal}
            for _ in range(self.legs):
                if run["stop"].wait(self.step_s):
                    yield {"event": "done", "run": run["run"], "status": 5, "seconds": 1.0,
                           "cancelled_for": run["cancelled_for"]}  # fmt: skip
                    return
            if self.on_arrival is not None:
                self.on_arrival()
            arrival = {"x": goal["x"] + 0.05, "y": goal["y"] - 0.03, "yaw_deg": goal["yaw_deg"]}
            yield {"event": "arrival", "run": run["run"], "status": 4, "seconds": 3.0,
                   "arrival": arrival}  # fmt: skip
            self.read_past_arrival = True
            time.sleep(self.hold_s)
            yield {"event": "done", "run": run["run"], "status": 4, "seconds": 3.0 + self.hold_s,
                   "arrival": arrival}  # fmt: skip
        finally:
            with self._lock:
                if self._running is run:
                    self._running = None
            run["over"].set()


def ruled_rig(
    tmp_path: Path,
    texts: list[str],
    script: list[list[Step]],
    observers: Sequence[Any] = (),
    **goals: Any,
) -> tuple[Rig, RuledGoals]:
    """A rig whose tools drive the goal server's rule (:class:`RuledGoals`) in real time."""
    clock = WallClock()
    fake = RuledGoals(clock, step_s=0.5, **goals)
    robot = fake_robot(clock=clock, goals=fake)
    tool = lambda name, args: TOOLS.call(name, args, robot)  # noqa: E731
    rig = Rig(tmp_path, texts, script, tool=tool, halt=robot.halt, observers=observers)
    return rig, fake


def test_the_same_drive_asked_again_is_the_goal_server_s_already_with_one_report(
    tmp_path: Path,
) -> None:
    """The drive under way asked for again: the goal server answers ``already`` (run 7, no
    new goal) and the call says so, its end reported by the first drive's own turn; the loop
    keeps no drives of its own to compare, and cancels nothing."""
    script: list[list[Step]] = [
        ["Еду.", ("go_to", {"place": "printer"})],
        ["Уже еду.", ("go_to", {"place": "printer"})],
        ["Я на месте."],
    ]
    rig, goals = ruled_rig(tmp_path, ["Пепин, езжай к принтеру", "к принтеру"], script)
    asyncio.run(rig.run(lambda rig: mid_drive(rig, 3)))
    assert [r.get("cmd") for r in goals.asked if r.get("cmd") in ("go", "cancel")] == ["go"] * 2
    assert goals.started == [7], "one run"
    _, again, report = rig.loop.turns
    result = again["tools"][0]["result"]
    assert "already under way: go printer (run 7), no new goal" in result
    assert "a [robot] message reports the drive's end" in result
    assert report["kind"] == "robot" and report["reply"] == "Я на месте."
    assert len(rig.brain.heard) == 3
    assert rig.brain.heard[2] == ARRIVED
    events, _ = events_of(tmp_path)
    (again_row,) = [e for e in events if e["event"] == "drive_again"]
    assert again_row["drive"] == "go_to(place='printer')" and again_row["ours"] is True


def test_another_drive_asked_while_one_runs_is_the_goal_server_s_replace(tmp_path: Path) -> None:
    """A drive to another place asked while one runs: the goal server cancels the running
    one and starts the new one after its end; the new call's answer names what it replaced,
    the cancelled drive's end is logged but not reported, the body hears one drive throughout,
    and only the new drive's arrival comes back as the robot's turn."""
    script: list[list[Step]] = [
        ["Еду.", ("go_to", {"place": "printer"})],
        ["Хорошо, к полке.", ("go_to", {"place": "bookshelf"})],
        ["Я у полки."],
    ]
    observer = RecordingObserver()
    texts = ["Пепин, езжай к принтеру", "нет, к полке"]
    rig, goals = ruled_rig(tmp_path, texts, script, observers=[observer])
    asyncio.run(rig.run(lambda rig: mid_drive(rig, 3)))
    assert [r for r in goals.asked if r.get("cmd") == "cancel"] == [], "the server's cancel"
    assert goals.started == [7, 8]
    _, replacing, report = rig.loop.turns
    result = replacing["tools"][0]["result"]
    assert "'replaced': 'go printer (run 7) cancelled for this one'" in result
    assert "'status': 'driving'" in result and "'target': 'bookshelf'" in result
    assert report["kind"] == "robot" and report["reply"] == "Я у полки."
    assert len(rig.brain.heard) == 3
    assert rig.brain.heard[2] == (
        "[robot] arrived: bookshelf (3 s, off by 6 cm; go_to(place='bookshelf'), run 8)"
    )
    events, _ = events_of(tmp_path)
    (ended,) = [e for e in events if e["event"] == "drive_ended"]
    assert ended["drive"] == "go_to(place='printer')"
    assert ended["replaced_by"] == "go bookshelf (tool)" and '"arrived": false' in ended["result"]
    assert [args for name, args in observer.calls if name == "drive"] == [(True,), (False,)]
    assert not rig.loop.drives


def test_an_arrival_reaches_the_brain_at_once_in_one_short_line(tmp_path: Path) -> None:
    """The goal server's ``arrival`` ends the drive's call (its pivot and tape, two more
    seconds, are not waited for: nobody reads past it), and the brain is asked at once while
    the speaker still plays three seconds of a line (2026-10-14: up to 10 s were waited for
    it to fall silent): its reply queues behind it. The message is one short line, the
    numbers in brackets."""
    script: list[list[Step]] = [["Еду.", ("go_to", {"place": "printer"})], ["Я на месте."]]
    rig, goals = ruled_rig(tmp_path, ["Пепин, езжай к принтеру"], script, legs=2, hold_s=2.0)
    line = b"\x10\x00" * (RATE * 3)  # three seconds still to play when the drive arrives
    goals.on_arrival = lambda: rig.player.feed(line, "backchannel", rig.loop.last_turn, -1)
    asked: list[tuple[str, bool, float]] = []
    arrived: list[float] = []
    real = rig.brain.turn

    async def turn(text: str, tools: Any, emit: Any) -> None:
        asked.append((text, rig.player.speaking(), time.monotonic()))
        await real(text, tools, emit)

    rig.brain.turn = turn  # type: ignore[method-assign]
    original = goals.on_arrival

    def arrive() -> None:
        arrived.append(time.monotonic())
        original()

    goals.on_arrival = arrive

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: len(rig.loop.turns) == 2 and rig.loop.quiet(), 10.0)

    asyncio.run(rig.run(scenario))
    (report, speaking, at) = asked[1]
    assert report == ARRIVED
    assert len(report) < 100
    assert speaking, "asked while the speaker still played the line"
    assert at - arrived[0] < 1.0, "not after the line's three seconds, nor the pivot's two"
    assert not goals.read_past_arrival, "the pivot and the tape's close were not waited for"
    assert rig.loop.turns[1]["reply"] == "Я на месте."


def test_a_drive_tried_again_after_a_failed_one_drives(tmp_path: Path) -> None:
    """A drive that failed (an unknown place) and the brain's next try in the same reply: the
    second one drives, nothing refuses it as unasked (2026-10-14: turn(me) refused 3 s after a
    failed approach)."""
    script: list[list[Step]] = [
        [("go_to", {"place": "garage"}), "Еду к принтеру.", ("go_to", {"place": "printer"})],
        ["Я на месте."],
    ]
    rig, goals = driving_rig(tmp_path, ["Пепин, езжай в гараж"], script)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: len(rig.loop.turns) == 2 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    first, report = rig.loop.turns
    failed, drove = first["tools"]
    assert "no such place" in failed["result"] and "'status': 'driving'" in drove["result"]
    assert [r["place"] for r in goals.asked if r.get("cmd") == "go"] == ["garage", "printer"]
    assert report["kind"] == "robot" and report["reply"] == "Я на месте."


def test_the_body_hears_whether_each_drive_ran(tmp_path: Path) -> None:
    """One reply: turn(me) while the speaker gaze already faces him (nothing moved), go_to of
    an unknown place (refused) and go_to(printer) (the goal server takes it). The body hears
    three drives, each begun and ended, the first two ended with ``ran`` False (no goal taken:
    the gaze lifts its body hold at once) and the third ran (2026-10-14 review: a refused drive
    held the gaze's body until the person's next words)."""
    clock = WallClock()
    goals = FakeGoalServer(clock, step_s=0.3, accept_s=0.05)
    gaze = FakeGaze(clock, view=Attention(phase="track", bearing_deg=4.0, seen_s=0.0))
    robot = fake_robot(clock=clock, goals=goals, gaze=gaze)
    observer = RecordingObserver()
    script: list[list[Step]] = [
        [("turn", {"target": "me"}), ("go_to", {"place": "garage"}), "Еду к принтеру.",
         ("go_to", {"place": "printer"})],
        ["Я на месте."],
    ]  # fmt: skip
    rig = Rig(tmp_path, ["Пепин, повернись ко мне"], script, observers=[observer],
              tool=lambda name, args: TOOLS.call(name, args, robot), halt=robot.halt)  # fmt: skip
    rig.loop.allowed, rig.loop.moving = {*rig.loop.allowed, "turn"}, {*rig.loop.moving, "turn"}

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: len(rig.loop.turns) == 2 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    faced, failed, drove = rig.loop.turns[0]["tools"]
    assert "already facing the person" in faced["result"] and "no such place" in failed["result"]
    assert "'status': 'driving'" in drove["result"] and gaze.asked == ["turn me"]
    drives = [args for name, args in observer.calls if name == "drive"]
    assert drives == [(True,), (False, False), (True,), (False, False), (True,), (False,)]
    assert not rig.loop.drives and rig.loop.drive_ran


class APIConnectionError(Exception):
    """Named as the SDKs name a call that never got an answer."""


class APITimeoutError(APIConnectionError):
    """The SDKs' timeout, a kind of it."""


class BadRequestError(Exception):
    """Named as the SDKs name a refusal (a 400)."""


def test_a_call_that_never_got_an_answer_is_told_from_a_refusal() -> None:
    assert connection_error(APIConnectionError("Connection error."))
    assert connection_error(APITimeoutError("Request timed out."))
    assert connection_error(TimeoutError()) and connection_error(ConnectionResetError())
    assert not connection_error(BadRequestError("400: messages: text content blocks must be"))
    assert not connection_error(RuntimeError("the API is down"))


def flaky(rig: Rig, failures: int) -> list[str]:
    """The brain's [robot] turns fail ``failures`` times first with a connection error; the
    texts of the calls that failed."""
    real = rig.brain.turn
    failed: list[str] = []

    async def turn(text: str, tools: Any, emit: Any) -> None:
        if text.startswith("[robot]") and len(failed) < failures:
            failed.append(text)
            raise APIConnectionError("Connection error.")
        await real(text, tools, emit)

    rig.brain.turn = turn  # type: ignore[method-assign]
    return failed


def test_a_robot_report_whose_brain_call_got_no_answer_is_asked_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A drive's end reaches the brain and its call fails on the connection: the same words
    are asked once more and answered (2026-10-14, turn 76: the arrival was dropped and the
    brain later said the report had not come)."""
    monkeypatch.setattr("pepin.voice_cascade.loop.ROBOT_RETRY_S", 0.05)
    script: list[list[Step]] = [["Еду.", ("go_to", {"place": "printer"})], ["Я на месте."]]
    rig, _ = driving_rig(tmp_path, ["Пепин, езжай к принтеру"], script)
    failed = flaky(rig, 1)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: len(rig.loop.turns) == 2 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    _, report = rig.loop.turns
    assert report["kind"] == "robot" and report["outcome"] == "committed"
    assert report["reply"] == "Я на месте."
    assert report["retried"] == "APIConnectionError: Connection error."
    assert failed == [rig.brain.heard[1]]  # the same words asked again
    assert rig.brain.outcomes == ["committed", "abandoned", "committed"]
    assert rig.config.error_line not in rig.voice.spoken
    events, _ = events_of(tmp_path)
    (retry,) = [e for e in events if e["event"] == "brain_retry"]
    assert retry["turn"] == report["turn"] and retry["error"] == report["retried"]
    assert not [e for e in events if e["event"] == "unanswered"]


def test_a_robot_report_failing_twice_goes_in_front_of_the_next_turn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The second call fails too: the turn ends in the error line and the report goes in front
    of the person's next words, marked as not answered, the result in it (so no tool is called
    again to learn it)."""
    monkeypatch.setattr("pepin.voice_cascade.loop.ROBOT_RETRY_S", 0.05)
    script: list[list[Step]] = [["Еду.", ("go_to", {"place": "printer"})], ["Я у принтера."]]
    rig, _ = driving_rig(tmp_path, ["Пепин, езжай к принтеру", "ты где"], script)
    failed = flaky(rig, 2)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: len(rig.loop.turns) == 2 and rig.loop.quiet(), 8.0)
        await rig.say(utterance())
        await until(lambda: len(rig.loop.turns) == 3 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    _, lost, answered = rig.loop.turns
    assert lost["kind"] == "robot" and lost["outcome"] == "error" and lost["retried"]
    assert len(failed) == 2 and failed[0] == failed[1]
    report = failed[0].removeprefix("[robot] ")
    assert report == ARRIVED.removeprefix("[robot] ")
    assert rig.brain.heard[1] == f"[robot, earlier, not answered] {report}\nты где"
    assert answered["carried"] == [report] and answered["reply"] == "Я у принтера."
    assert rig.config.error_line in rig.voice.spoken
    events, _ = events_of(tmp_path)
    assert [e["event"] for e in events if e["event"] in ("brain_retry", "unanswered")] == [
        "brain_retry",
        "unanswered",
    ]
    assert rig.loop.unanswered == []


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
    script: list[list[Step]] = [["{sad} Мне жаль.", "Но всё пройдёт.", "{} Ладно."]]
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
    assert rig.voice.styles == ["sad", "sad", ""]  # sticky until {} resets it
    assert first["styles"] == ["sad", ""] and first["brain"] == "first"
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
    script: list[list[Step]] = [[("look", {"pan_deg": 10.0}), "{quiet} Вижу стол.", "Всё."]]
    rig = Rig(tmp_path, ["Пепин, что там?"], script, switchboard=board,
              observers=[BrokenObserver(), observer], flags=[flag], clips=True,
              fillers=FillerBook({"ru": ("Секунду.",)}, {}, {}, {}, {}, ()))  # fmt: skip

    asked: list[str] = []

    async def scenario(rig: Rig) -> None:
        request = write_request(tmp_path, None, None, flags={GESTURES: True, "nope": True})
        asked.append(request.id)
        await until(lambda: len(answers_to(tmp_path, request.id)) == 2)
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert flag.on and rig.voice.spoken == ["Вижу стол.", "Всё."]  # the filler from its clip
    assert rig.loop.turns[0]["fillers"] == ["Секунду."]
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
        (1, 0, "Вижу стол.", "quiet"), (1, 1, "Всё.", None),
    ]  # fmt: skip
    assert observer.calls[-1] == ("closed", ("stopped",))


def test_a_knob_switches_live_and_each_turn_line_ends_with_the_cells(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """``vision.source`` set through the control file is applied at once and answered as a knob;
    an unknown knob and a value the knob does not take are answered failed; the turn's line
    carries the vision cell, and a cell that breaks is left out."""
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
    source = VisionSource()

    def broken() -> str:
        raise RuntimeError("no cell")

    rig = Rig(tmp_path, ["Пепин, что видишь?"], [["Вижу стол."]], switchboard=board,
              knobs=[source], cells=[lambda: "vision 20.0 in", broken])  # fmt: skip
    asked: list[str] = []

    async def scenario(rig: Rig) -> None:
        good = write_request(tmp_path, None, None, knobs={"vision.source": "snapshot", "x": "1"})
        asked.append(good.id)
        await until(lambda: len(answers_to(tmp_path, good.id)) == 2)
        bad = write_request(tmp_path, None, None, knobs={"vision.source": "gpu"})
        asked.append(bad.id)
        await until(lambda: len(answers_to(tmp_path, bad.id)) == 1)
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert source.value == "snapshot"
    answers = {r["name"]: r for r in answers_to(tmp_path, asked[0])}
    assert answers["vision.source"]["state"] == "applied"
    assert answers["vision.source"]["what"] == "knob"
    assert answers["vision.source"]["value"] == "snapshot"
    assert answers["x"]["state"] == "failed" and "vision.source" in answers["x"]["error"]
    (refused,) = answers_to(tmp_path, asked[1])
    assert refused["state"] == "failed" and "service, snapshot" in refused["error"]
    lines = [line for line in capsys.readouterr().out.splitlines() if line.startswith("  [")]
    assert lines and lines[-1].endswith(" | vision 20.0 in")


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
    rig = Rig(tmp_path, ["Пепин, запомни Сашу"], script, tool=slow_tool(release), clock=clock,
              clips=True)  # fmt: skip
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
    assert first in fillers.for_tool("remember_person", "ru")  # its own, not the look's
    again = said[1][1]
    assert said[1] == [first, again] and again in fillers.again["ru"]
    assert rig.voice.spoken == ["Готово."]  # the fillers from their clips
    (turn,) = rig.loop.turns
    assert turn["fillers"] == [first, again]
    (events,) = tmp_path.glob("*/cascade_*/events.jsonl")
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    logged = [(r["context"], r["tool"], r["text"]) for r in rows if r["event"] == "filler"]
    assert logged == [("tool", "remember_person", first), ("again", None, again)]


def test_no_filler_speaks_once_the_reply_s_first_sentence_has_started(tmp_path: Path) -> None:
    release, clock = threading.Event(), SkewClock()
    script: list[list[Step]] = [[("see", {}), "Вижу стол.", ("look", {}), "Всё."]]
    rig = Rig(tmp_path, ["Пепин, что там?"], script, tool=slow_tool(release), clock=clock,
              clips=True, filler_thinking_s=1.2)  # fmt: skip
    rig.loop.allowed |= {"see", "look"}

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.current is not None and bool(rig.loop.current.sentences))
        clock.skew = 30.0  # the look runs on long after the sentence: no filler for it
        await asyncio.sleep(0.3)
        release.set()
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.voice.spoken == ["Вижу стол.", "Всё."]
    (first,) = rig.loop.turns[0]["fillers"]
    assert first in CascadeConfig().fillers.for_tool("see", "ru")


def blocked_tool(release: threading.Event) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """Tools that answer at once, but for ``where_am_i`` (no filler of its own), which waits
    for ``release``: the brain's first token waits with it."""

    def run(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "where_am_i":
            assert release.wait(5.0)
        return {"ok": True, "places": [{"name": "home"}]}

    return run


def thinking_lines(language: str) -> set[str]:
    book = CascadeConfig().fillers
    return {*book.thinking.get(language, ()), *book.neutral}


def test_a_brain_late_to_its_first_token_gets_one_thinking_filler(tmp_path: Path) -> None:
    release, clock = threading.Event(), SkewClock()
    script: list[list[Step]] = [[("where_am_i", {}), "Я дома."]]
    rig = Rig(tmp_path, ["Пепин, где ты?"], script, tool=blocked_tool(release), clock=clock,
              clips=True, filler_thinking_s=1.2, filler_again_s=0.0)  # fmt: skip
    said: list[list[str]] = []

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.current is not None and rig.loop.current.acted)
        reply = rig.loop.current
        assert reply is not None
        clock.skew = 0.5  # under filler_thinking_s since the close (the ears' clock): nothing
        await asyncio.sleep(0.2)
        said.append(list(reply.fillers))
        clock.skew = 1.3
        await until(lambda: len(reply.fillers) == 1)
        clock.skew = 9.0  # once a reply
        await asyncio.sleep(0.2)
        said.append(list(reply.fillers))
        assert rig.player.on_fade is not None  # the player's word on a fade reaches the log
        rig.player.on_fade(FillerFade(reply.id, 0.06, 0.2, 0.08))
        release.set()
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert said[0] == [] and len(said[1]) == 1 and said[1][0] in thinking_lines("ru")
    assert rig.voice.spoken == ["Я дома."]
    (events,) = tmp_path.glob("*/cascade_*/events.jsonl")
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    (filler,) = [r for r in rows if r["event"] == "filler"]
    assert (filler["context"], filler["language"], filler["text"]) == ("thinking", "ru", said[1][0])
    assert filler["clip_s"] == pytest.approx(rig.speech.clip_s(said[1][0]), abs=0.01)
    assert filler["fade_ms"] == 60, "conversation.filler_fade_ms, a thinking filler's"
    (fade,) = [r for r in rows if r["event"] == "filler_fade"]
    assert (fade["turn"], fade["text"]) == (filler["turn"], said[1][0])
    assert (fade["faded_ms"], fade["cut_ms"], fade["ahead_ms"]) == (60, 200, 80)


@pytest.mark.parametrize(
    ("text", "fillers"),
    [("Пепин, где ты?", 0), ("Пепин, скажи пожалуйста, где ты сейчас стоишь?", 1)],
)
def test_a_short_question_gets_no_thinking_filler_and_a_long_one_does(
    tmp_path: Path, text: str, fillers: int
) -> None:
    """The book's minimum (4 words) holds the three-word question's moment in silence, however
    late the brain; seven words at a chance of 1 get the filler."""
    release, clock = threading.Event(), SkewClock()
    script: list[list[Step]] = [[("where_am_i", {}), "Я дома."]]
    book = replace(FillerBook(), thinking_chance=((4.0, 1.0),))
    rig = Rig(tmp_path, [text], script, tool=blocked_tool(release), clock=clock, clips=True,
              filler_thinking_s=1.2, filler_again_s=0.0, fillers=book)  # fmt: skip
    said: list[list[str]] = []

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.current is not None and rig.loop.current.acted)
        reply = rig.loop.current
        assert reply is not None
        clock.skew = 1.3  # the thinking filler's moment
        if fillers:
            await until(lambda: len(reply.fillers) == 1)
        else:
            await asyncio.sleep(0.3)
        said.append(list(reply.fillers))
        release.set()
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert len(said[0]) == fillers and rig.voice.spoken == ["Я дома."]


def test_a_thinking_clip_over_filler_max_s_is_never_said_and_the_lengths_are_told_once(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The tone voice says 10 ms a character: every thinking clip is over a 10 ms cap, so the
    late brain gets no thinking filler, and the run says once which clips and how long."""
    release, clock = threading.Event(), SkewClock()
    script: list[list[Step]] = [[("where_am_i", {}), "Я дома."]]
    rig = Rig(tmp_path, ["Пепин, где ты?"], script, tool=blocked_tool(release), clock=clock,
              clips=True, filler_thinking_s=1.2, filler_again_s=0.0,
              filler_max_s={"thinking": 0.01})  # fmt: skip

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.current is not None and rig.loop.current.acted)
        clock.skew = 1.3
        await asyncio.sleep(0.2)
        release.set()
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.loop.turns[0].get("fillers") is None and rig.voice.spoken == ["Я дома."]
    told = [line for line in capsys.readouterr().out.splitlines() if "filler_max_s" in line]
    assert len(told) == 1 and "filler thinking: every clip is over filler_max_s 0.01 s" in told[0]


def test_no_thinking_filler_once_the_first_token_came_in_time(tmp_path: Path) -> None:
    release, clock = threading.Event(), SkewClock()
    script: list[list[Step]] = [[("where_am_i", {}), "Я дома."]]
    rig = Rig(tmp_path, ["Пепин, где ты?"], script, tool=blocked_tool(release), clock=clock,
              clips=True, filler_thinking_s=1.2)  # fmt: skip

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.current is not None and rig.loop.current.acted)
        reply = rig.loop.current
        assert reply is not None
        clock.skew = 0.9
        release.set()  # the first token ~1.0 s after the close
        await until(lambda: reply.input.timeline.has("first_token"))
        clock.skew = 9.0
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.loop.turns[0].get("fillers") is None and rig.voice.spoken == ["Я дома."]


def test_the_thinking_filler_speaks_the_language_of_the_words(tmp_path: Path) -> None:
    release, clock = threading.Event(), SkewClock()
    script: list[list[Step]] = [[("where_am_i", {}), "At home."]]
    rig = Rig(tmp_path, ["Where are you?"], script, tool=blocked_tool(release), clock=clock,
              clips=True, wake="always", filler_thinking_s=1.2)  # fmt: skip

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.current is not None and rig.loop.current.acted)
        clock.skew = 1.3
        await until(lambda: rig.loop.current is not None and bool(rig.loop.current.fillers))
        release.set()
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    (filler,) = rig.loop.turns[0]["fillers"]
    assert filler in thinking_lines("en") and rig.loop.fillers.language == "en"


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
    script: list[list[Step]] = [["Ну {laugh} да, {quiet} все спят.", "{gesture:nope} Ладно."]]
    rig = Rig(tmp_path, ["Пепин, привет"], script, observers=[observer],
              vocabulary=Vocabulary(frozenset({"laugh"}), frozenset()))  # fmt: skip

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.voice.spoken == ["Ну да,", "все спят.", "Ладно."]
    assert rig.voice.styles == ["", "quiet", "quiet"]
    parts = [args for name, args in observer.calls if name == "part_queued"]
    assert [(index, part.text, [m.name for m in part.markers]) for _, index, part in parts] == [
        (0, "Ну да,", ["laugh"]), (1, "все спят.", []), (2, "Ладно.", []),
    ]  # fmt: skip
    (turn,) = rig.loop.turns
    assert turn["reply"] == "Ну {laugh} да, {quiet} все спят. {gesture:nope} Ладно."  # as written
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


def test_the_idle_close_asks_for_a_parting_line_and_the_person_may_come_back(
    tmp_path: Path,
) -> None:
    """Nobody speaks for idle_close_s: the brain gets a [robot] turn for a parting line and
    the close follows it; a person answering that line keeps the conversation open, and the
    next quiet spell asks again (2026-10-13: the robot asked a guest to look at it, she said
    nothing more, and it closed after 20 s of silence without a word)."""
    parting = "Ладно, я тут, зови если что. Буду рядом."
    script: list[list[Step]] = [["Слушаю."], [parting], ["Да?"], ["Пока."]]
    rig = Rig(tmp_path, ["Пепин", "Подожди"], script, seconds_per_char=0.03, barge_in=True)
    rig.loop.config = rig.config.with_changes(idle_close_s=0.4)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: len(rig.brain.heard) == 2, 8.0)  # the parting line asked
        await until(lambda: rig.player.playback() == "reply")
        await rig.say(tone(0.4) + quiet(0.7))  # the person comes back over the parting line
        await until(lambda: len(rig.loop.turns) == 3 and rig.loop.quiet(), 8.0)
        assert rig.loop.conversation is not None and rig.loop.conversation.closing is None
        await until(lambda: rig.loop.conversation is None, 8.0)

    asyncio.run(rig.run(scenario))
    heard = rig.brain.heard
    assert heard[0] == "Пепин" and heard[2] == "Подожди"
    for asked in (heard[1], heard[3]):
        assert asked.startswith("[robot] nobody has spoken for 0.4 s") and "parting" in asked
    assert [t["kind"] for t in rig.loop.turns] == ["person", "robot", "person", "robot"]
    assert [t["outcome"] for t in rig.loop.turns] == [
        "committed", "interrupted", "committed", "committed"
    ]  # fmt: skip
    assert rig.loop.turns[3]["reply"] == "Пока."
    _, summary = events_of(tmp_path)
    assert summary["close"] == "idle" and summary["turns"] == 2


def test_the_idle_close_is_silent_with_idle_parting_off(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ["Пепин"], [["Слушаю."]], idle_parting=False)
    rig.loop.config = rig.config.with_changes(idle_close_s=0.4)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.conversation is None, 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard == ["Пепин"]
    assert events_of(tmp_path)[1]["close"] == "idle"


def test_a_conversation_is_never_closed_by_its_length_only_by_quiet(tmp_path: Path) -> None:
    """No cap on a conversation's length (2026-10-14: max_session_s closed one at 15 min
    without a word): an hour-old conversation answers the next words and stays open; quiet
    still closes it."""
    rig = Rig(tmp_path, ["Пепин", "ты ещё тут?"], [["Слушаю."], ["Да."]], idle_parting=False)

    async def scenario(rig: Rig) -> None:
        await opened(rig)
        conversation = rig.loop.conversation
        assert conversation is not None
        conversation.t_open -= 3600.0
        await asyncio.sleep(0.2)
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet(), 8.0)
        await asyncio.sleep(0.2)
        assert rig.loop.conversation is conversation
        rig.loop.config = rig.config.with_changes(idle_close_s=0.4)
        await until(lambda: rig.loop.conversation is None, 8.0)

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard == ["Пепин", "ты ещё тут?"]
    assert events_of(tmp_path)[1]["close"] == "idle"


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
    day = next(p for p in sorted(tmp_path.iterdir()) if p.is_dir())  # the ledger file sits beside
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


def test_the_body_hears_each_part_start_its_audio_the_text_ahead_and_each_part_s_audio_record(
    tmp_path: Path,
) -> None:
    """The gesticulation contract's seams: ``part_started`` at each part's first chunk with its
    stream second, every chunk to the audio taps (the voice's thread), ``text_ahead`` from the
    brain's token stream before a sentence is whole, and one ``part_audio`` event per part
    spoken (its stream span, leading silence and chunk arrivals)."""
    from pepin.expression.fakes import RecordingTap
    from pepin.voice_cascade.stages import Ahead, BrainEvent

    observer, tap = RecordingObserver(), RecordingTap()
    script: list[list[Step]] = [["Ну {laugh} да, это было очень весело.", "Всё."]]
    rig = Rig(tmp_path, ["Пепин, привет"], script, observers=[observer],
              vocabulary=Vocabulary(frozenset({"laugh"}), frozenset()))  # fmt: skip
    rig.loop.taps.append(tap)
    scripted = rig.brain.turn

    async def ahead(text: str, tools: Any, emit: Callable[[BrainEvent], None]) -> None:
        def tee(event: BrainEvent) -> None:
            if isinstance(event, Sentence):  # the stream had its first words a moment before
                emit(Ahead(event.text.split()[0], event.round))
            emit(event)

        await scripted(text, tools, tee)

    rig.brain.turn = ahead  # type: ignore[method-assign]

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    calls = observer.calls
    assert [args for name, args in calls if name == "text_ahead"] == [(1, 0, "Ну"), (1, 1, "Всё.")]
    started = [args for name, args in calls if name == "part_started"]
    assert [(turn, index) for turn, index, _ in started] == [(1, 0), (1, 1)]
    assert 0.0 <= started[0][2] < started[1][2]
    for index in (0, 1):
        assert calls.index(("part_started", started[index])) < calls.index(
            ("part_spoken", (1, index))
        )
    by_part = {(c.turn, c.index) for c in tap.chunks}
    assert by_part == {(1, 0), (1, 1)} and all(c.rate == RATE for c in tap.chunks)
    zero, one = ([c for c in tap.chunks if c.index == i] for i in (0, 1))
    assert len(zero) > 1 and {c.start_s for c in zero} == {started[0][2]}
    assert [c.end_s for c in zero] == sorted({c.end_s for c in zero}) and zero[-1].end_s <= one[
        0
    ].start_s
    assert one[0].start_s == started[1][2]
    (events,) = tmp_path.glob("*/cascade_*/events.jsonl")
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    audio = [r for r in rows if r["event"] == "part_audio"]
    assert [(r["turn"], r["index"]) for r in audio] == [(1, 0), (1, 1)]
    assert audio[1]["start_s"] == pytest.approx(started[1][2], abs=1e-3)
    assert all(r["end_s"] > r["start_s"] and r["leading_s"] >= 0.0 for r in audio)
    assert all(r["chunks_s"][0] == 0.0 and len(r["chunks_s"]) >= 1 for r in audio)


def test_the_person_s_pause_is_a_moment_of_the_body_whatever_the_backchannel_s_flag(
    tmp_path: Path,
) -> None:
    observer = RecordingObserver()
    rig = Rig(tmp_path, ["Пепин", "послушай меня"], observers=[observer], backchannel=False)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet())
        await rig.say(quiet(0.2) + tone(1.0) + quiet(0.7))
        await until(lambda: rig.loop.person_turns == 2 and rig.loop.quiet())

    asyncio.run(rig.run(scenario))
    pauses = [a for name, a in observer.calls if name == "moment" and a[0] == LISTENER_PAUSE]
    # the person's words so far ride with it once the early read of that pause has landed
    assert pauses and all(a[1:] in ((), ("послушай меня",)) for a in pauses)
    (events,) = tmp_path.glob("*/cascade_*/events.jsonl")
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert not [r for r in rows if r["event"] == "backchannel"]  # its flag is off


# -- the body's scene before the brain's input ---------------------------------------------------


@dataclass
class SceneGaze:
    """The speaker gaze as the loop finds it among its flags: its switch and its snapshot (the
    person seen at every read)."""

    view: Attention
    flag: str = SPEAKER_GAZE
    on: bool = True

    def switch(self, on: bool) -> str | None:
        self.on = on
        return None

    def attention(self) -> Attention:
        return replace(self.view, seen_s=time.monotonic())


class SceneBody:
    """The registry as the scene reads it: where_am_i at a pose the test moves, the printer two
    metres along the map's x axis; every call kept."""

    def __init__(self) -> None:
        self.pose: dict[str, float] = {"x": 0.0, "y": 0.0, "heading_deg": 0.0}
        self.calls: list[str] = []

    def __call__(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        self.calls.append(name)
        if name == "where_am_i":
            return {"ok": True, **self.pose}
        if name == "list_places":
            return {"ok": True, "places": [{"name": "printer", "x": 2.0, "y": 0.0}]}
        return {"ok": True}


def test_the_brain_s_input_begins_with_the_body_s_scene_only_when_it_changed(
    tmp_path: Path,
) -> None:
    """2026-10-14: after turn(printer) the gaze turned the body back toward the person, and the
    brain said the printer was in front of it."""
    body, gaze = SceneBody(), SceneGaze(Attention(phase="track", bearing_deg=90.0))
    texts = ["Пепин, привет", "что впереди?", "а теперь?"]
    rig = Rig(tmp_path, texts, tool=body, flags=[gaze], scene_status=True)

    async def scenario(rig: Rig) -> None:
        for turn in (1, 2, 3):
            if turn == 3:  # meanwhile the gaze turned the body to the person
                body.pose["heading_deg"] = 90.0
                gaze.view = replace(gaze.view, bearing_deg=0.0)
            await rig.say(utterance())
            await until(lambda n=turn: rig.loop.person_turns == n and rig.loop.quiet())

    asyncio.run(rig.run(scenario))
    whole =("[scene] heading 0; position 0.0, 0.0; ahead: printer; person: left 90 deg, seen;"
             " standing still")  # fmt: skip
    turned = "[scene] heading 90; ahead: the person; person: ahead, seen"
    assert rig.brain.heard == [f"{whole}\nПепин, привет", "что впереди?", f"{turned}\nа теперь?"]
    assert body.calls.count("where_am_i") == 3  # one a turn
    (events,) = tmp_path.glob("*/cascade_*/events.jsonl")
    rows = [json.loads(line) for line in events.read_text().splitlines()]
    assert [(r["turn"], r["line"]) for r in rows if r["event"] == "scene"] == [
        (1, whole),
        (3, turned),
    ]
    opened = next(r for r in rows if r["event"] == "open")
    assert opened["flags"]["scene_status"] is True and rig.loop.flags["scene_status"].on


def test_a_turn_dropped_for_the_words_after_it_leaves_its_scene_to_their_turn(
    tmp_path: Path,
) -> None:
    body = SceneBody()
    texts = ["Пепин, скажи", "ещё кое-что"]
    rig = Rig(tmp_path, texts, tool=body, brain_delay_s=1.0, scene_status=True)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.current is not None)
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    scene = "[scene] heading 0; position 0.0, 0.0; ahead: printer; standing still"  # no gaze
    assert rig.brain.outcomes == ["abandoned", "committed"]
    assert rig.brain.heard == [f"{scene}\nПепин, скажи", f"{scene}\nПепин, скажи ещё кое-что"]


def test_a_robot_report_begins_with_what_the_drive_changed(tmp_path: Path) -> None:
    rig, _ = driving_rig(
        tmp_path, ["Пепин, езжай к принтеру"], [["Еду.", ("go_to", {"place": "printer"})]]
    )
    rig.loop.scene.switch(True)  # the rig's default is off

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: len(rig.loop.turns) == 2 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    # from home, the bookshelf 22 degrees right of the nose; to the printer's place
    assert rig.brain.heard[0] == (
        "[scene] heading 3; position 0.1, 0.1 (at home); ahead: nothing named; standing still"
        "\nПепин, езжай к принтеру"
    )
    assert rig.brain.heard[1] == f"[scene] heading 92; position -1.1, 3.4 (at printer)\n{ARRIVED}"


def test_with_the_scene_switch_off_the_brain_hears_no_line_and_nothing_is_asked(
    tmp_path: Path,
) -> None:
    body = SceneBody()
    gaze = SceneGaze(Attention(phase="track", bearing_deg=90.0))
    rig = Rig(tmp_path, ["Пепин, привет"], tool=body, flags=[gaze], scene_status=False)

    async def scenario(rig: Rig) -> None:
        await rig.say(utterance())
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet())

    asyncio.run(rig.run(scenario))
    assert rig.brain.heard == ["Пепин, привет"]
    assert "where_am_i" not in body.calls and not rig.loop.flags["scene_status"].on
    (events,) = tmp_path.glob("*/cascade_*/events.jsonl")
    assert '"scene"' not in events.read_text()
