"""The speaker gaze in the cascade: registered as an observer and as the live flag
``speaker_gaze``, switched on through the control file, turned by the wake, let go at the close
(the loop over fakes, the gaze over the fakes of every seam)."""

from __future__ import annotations

import asyncio
import json
import math
from collections import deque
from pathlib import Path

import pytest
from test_speaker_gaze import Door, face, frame, view
from test_voice_cascade_loop import Rig, until, utterance

from pepin.expression.conversation import SPEAKER_GAZE
from pepin.expression.executors import NoWheels
from pepin.expression.fakes import FakeDoa, FakeFace, FakeFaceFinder
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
    assert first.pan_rad == 0.0 and first.tilt_rad == pytest.approx(math.radians(-5.0))
    assert face_channel.calls[0][1][0] == "smile"
    assert door.count("release") == 1 and machine.phase == "idle", "the run's end let it go"
    opened = [
        json.loads(line)
        for log in tmp_path.rglob("events.jsonl")
        for line in log.read_text().splitlines()
        if '"open"' in line
    ]
    assert opened and opened[0]["flags"][SPEAKER_GAZE] is True
