"""The speaker gaze in the cascade: registered as an observer and as the live flag
``speaker_gaze``, switched on through the control file, turned by the wake, let go at the close,
its decisions in the conversation's events.jsonl (the loop over fakes, the gaze over the fakes of
every seam)."""

from __future__ import annotations

import asyncio
import json
import math
import time
from collections import deque
from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest
from test_speaker_gaze import Door, face, frame, view
from test_voice_cascade_loop import Rig, until, utterance

from pepin.expression.conversation import SPEAKER_GAZE
from pepin.expression.executors import NoWheels
from pepin.expression.fakes import FakeDoa, FakeFace, FakeFaceFinder, RecordingObserver
from pepin.expression.markers import Part
from pepin.expression.speaker import FaceFrame
from pepin.expression.tracker import SpeakerGaze, SpeakerGazeWorker, SpeakerSettings
from pepin.gaze import Reach
from pepin.neck import NeckConfig
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.control import ControlFile, Switchboard, answers_to, write_request
from pepin.voice_cascade.engines import BrainSetup
from pepin.voice_cascade.fake import ScriptedBrain, ToneVoice

pytestmark = pytest.mark.slow  # the player paces at real time

REPO = Path(__file__).resolve().parents[2]


def test_the_switch_turns_the_gaze_on_the_wake_turns_the_head_and_the_close_lets_it_go(
    tmp_path: Path,
) -> None:
    door = Door(view=view())
    face_channel = FakeFace()
    machine = SpeakerGaze(
        SpeakerSettings(doa_map=None),  # unmeasured: the head looks up where it points
        door=door,
        doa=FakeDoa(),
        finder=FakeFaceFinder(deque([frame(face(0.0))])),
        wheels=NoWheels(),
        face=face_channel,
        reach=Reach.of(NeckConfig.from_json(REPO / "config/neck.json")),
        report=lambda line: None,
    )
    gaze = SpeakerGazeWorker(lambda settings: machine, on=False, load=SpeakerSettings)
    board = Switchboard(
        ControlFile(tmp_path),
        configure=lambda brain, voice: CascadeConfig(),
        build_brain=lambda config, name: BrainSetup(
            ScriptedBrain(), config, name, set(), set(), paid=False
        ),
        build_voice=lambda v: ToneVoice(rate=4000),
        brain="first",
        voice="first-voice",
    )
    rig = Rig(
        tmp_path, ["Пепин, привет"], [["Привет."]], switchboard=board, observers=[gaze],
        flags=[gaze],
    )  # fmt: skip
    gaze.journal = rig.loop.record
    asked: list[str] = []

    async def scenario(rig: Rig) -> None:
        request = write_request(tmp_path, None, None, flags={SPEAKER_GAZE: True})
        asked.append(request.id)
        await until(lambda: len(answers_to(tmp_path, request.id)) == 1)
        await rig.say(utterance())
        await until(lambda: door.count("aim") >= 1 and len(face_channel.calls) == 1, 5.0)
        await until(lambda: rig.loop.person_turns == 1 and rig.loop.quiet(), 8.0)

    asyncio.run(rig.run(scenario))
    gaze.close()
    (answer,) = answers_to(tmp_path, asked[0])
    assert answer["state"] == "applied" and answer["on"] is True and gaze.on
    first = door.aims()[0][0]
    assert first.pan_rad == 0.0 and first.tilt_rad == pytest.approx(math.radians(3.0))
    assert face_channel.calls[0][1][0] == "smile"
    assert door.count("release") == 1 and machine.phase == "idle", "the run's end let it go"
    opened = [
        json.loads(line)
        for log in tmp_path.rglob("events.jsonl")
        for line in log.read_text().splitlines()
        if '"open"' in line
    ]
    assert opened and opened[0]["flags"][SPEAKER_GAZE] is True
    (log,) = tmp_path.rglob("events.jsonl")
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    gazed = [r for r in rows if r["event"].startswith("gaze_")]
    kinds = [r["event"] for r in gazed]
    assert kinds[:3] == ["gaze_wake", "gaze_reaim", "gaze_face"], "the face centred: no refine"
    assert kinds[-1] == "gaze_release", "the release after the close lands in its log"
    (heard,) = [r for r in rows if r["event"] == "heard"]
    woke = gazed[0]
    assert woke["named"] is True and woke["peak_db"] == heard["voice_peak_db"]
    assert 0.0 <= woke["t"] - heard["t"] < 1.0, "on the session's clock, beside its utterance"
    assert gazed[-1]["t"] >= next(r["t"] for r in rows if r["event"] == "close")


@dataclass
class Stamps(RecordingObserver):
    """When each reply's first part went to the voice and when its first audio came."""

    first: dict[int, float] = field(default_factory=dict)
    voiced: dict[int, float] = field(default_factory=dict)

    def part_queued(self, turn: int, index: int, part: Part) -> None:
        self.first.setdefault(turn, time.monotonic())

    def voice_started(self, turn: int) -> None:
        self.voiced[turn] = time.monotonic()


@dataclass
class StampedFinder:
    """A face straight ahead in every frame, and when each was taken."""

    times: list[float] = field(default_factory=list)

    def look(self, not_before: float | None = None) -> FaceFrame | None:
        self.times.append(time.monotonic())
        return frame(face(0.0))


def three_replies(tmp: Path, hold: bool) -> tuple[Stamps, StampedFinder, SpeakerGaze, list[int]]:
    """Three turns through the loop over fakes, the voice's first chunk 0.25 s (Qwen3-TTS takes
    0.1-0.15 s on the GPU), the gaze tracking at 10 Hz; the moments, the frames, the machine
    and each turn's tts step."""
    finder = StampedFinder()
    machine = SpeakerGaze(
        SpeakerSettings(doa_map=None),
        door=Door(view=view()),
        doa=FakeDoa(),
        finder=finder,
        wheels=NoWheels(),
        face=FakeFace(),
        reach=Reach.of(NeckConfig.from_json(REPO / "config/neck.json")),
        report=lambda line: None,
    )
    settings = replace(SpeakerSettings(), hold_for_voice=hold)
    gaze = SpeakerGazeWorker(lambda s: machine, on=True, load=lambda: settings)
    stamps = Stamps()
    texts = ["Пепин, привет", "как дела", "что нового"]
    rig = Rig(tmp, texts, [["Хорошо."]] * 3, observers=[stamps, gaze], flags=[gaze])
    rig.voice.delay_s, rig.voice.sleep = 0.25, time.sleep

    async def scenario(rig: Rig) -> None:
        for k in range(3):
            await rig.say(utterance())
            await until(lambda k=k: rig.loop.person_turns == k + 1 and rig.loop.quiet(), 8.0)
            await asyncio.sleep(0.5)

    asyncio.run(rig.run(scenario))
    gaze.close()
    tts = [t["steps"]["tts"] for t in rig.loop.turns if t["kind"] == "person"]
    return stamps, finder, machine, tts


def during_first_chunks(stamps: Stamps, finder: StampedFinder) -> list[float]:
    """The frames taken between a reply's first part (and a frame already under way) and its
    first audio."""
    grace = 0.03  # the part's moment reaches the gaze's queue; a frame already taken finishes
    return [
        t
        for turn, start in stamps.first.items()
        for t in finder.times
        if start + grace < t < stamps.voiced[turn]
    ]


def test_no_frame_is_taken_while_the_voice_makes_a_reply_s_first_chunk(tmp_path: Path) -> None:
    """From each reply's first part until its first audio no frame is taken, and frames go on
    outside those windows; with the hold switched off they fall inside."""
    stamps, finder, machine, tts = three_replies(tmp_path / "on", hold=True)
    assert sorted(stamps.voiced) == sorted(stamps.first) and len(stamps.first) == 3
    assert all(stamps.voiced[turn] - start >= 0.2 for turn, start in stamps.first.items())
    assert during_first_chunks(stamps, finder) == []
    assert len(finder.times) >= 20 and machine.counts["voice_holds"] == 3
    assert all(250 <= ms < 400 for ms in tts), tts
    stamps, finder, machine, _ = three_replies(tmp_path / "off", hold=False)
    assert len(during_first_chunks(stamps, finder)) >= 3, "at 10 Hz, two or so a reply"
