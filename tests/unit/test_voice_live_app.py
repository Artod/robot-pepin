"""The idle loop: segments heard, read by a fake Whisper, a session only on the name and with
the caps' consent; every segment logged."""

import asyncio
import json
import threading
import time
from pathlib import Path

import numpy as np
import pytest

from pepin.voice_live.app import VoiceLoop
from pepin.voice_live.audio import Pacer
from pepin.voice_live.budget import Ledger
from pepin.voice_live.config import LiveConfig
from pepin.voice_live.events import Events
from pepin.voice_live.fake import ScriptedLive
from pepin.voice_live.session import Conversation, Frame, ResumeStore
from pepin.voice_live.sources import FrameSink

pytestmark = pytest.mark.slow  # frames at real time and a session's idle timer

FRAME = 640  # bytes: 20 ms


def utterance_frames(loud_s: float = 0.4) -> list[bytes]:
    quiet = [b"\x00\x00" * 320] * 15
    t = np.arange(int(loud_s * 16_000)) / 16_000
    loud = (0.3 * 32767 * np.sin(2 * np.pi * 300 * t)).astype("<i2").tobytes()
    return quiet + [loud[i : i + FRAME] for i in range(0, len(loud), FRAME)] + quiet * 3


class ListMic:
    def __init__(self, frames: list[bytes]) -> None:
        self.frames = frames
        self.done = threading.Event()

    def start(self, sink: FrameSink) -> None:
        def run() -> None:
            for frame in self.frames:
                sink(frame, time.monotonic())
                time.sleep(0.002)
            self.done.set()

        threading.Thread(target=run, daemon=True).start()


class FakeWhisper:
    def __init__(self, text: str) -> None:
        self.text = text
        self.heard: list[int] = []

    def transcribe(self, pcm: bytes) -> str:
        self.heard.append(len(pcm))
        return self.text


class NullSpeaker:
    def play(self, pcm: bytes) -> None: ...
    def play_end(self) -> None: ...
    def flush(self) -> None: ...


def build(tmp: Path, text: str, ledger: Ledger | None = None) -> tuple[VoiceLoop, ScriptedLive]:
    config = LiveConfig(idle_close_s=0.1)
    live = ScriptedLive([])
    events = Events()
    pacer = Pacer(NullSpeaker(), tail_s=0.0)

    def conversation(frames: asyncio.Queue[Frame]) -> Conversation:
        return Conversation(
            live, config, frames=frames, pacer=pacer, tools=lambda n, a: {"ok": True},
            moving=set(), halt=lambda: "", events=events, logs=tmp, ledger=ledger,
            resume=ResumeStore(tmp / "resume.json", 7000.0), tick_s=0.005,
        )  # fmt: skip

    loop = VoiceLoop(
        config,
        mic=ListMic(utterance_frames()),
        pacer=pacer,
        transcriber=FakeWhisper(text),
        conversation_factory=conversation,
        events=events,
        logs=tmp,
        ledger=ledger,
    )
    return loop, live


def wake_rows(tmp: Path) -> list[dict[str, object]]:
    (day,) = [p for p in tmp.iterdir() if p.is_dir()]
    return [json.loads(line) for line in (day / "wake.jsonl").read_text().splitlines()]


def test_the_name_opens_a_session_with_the_segment_heard_first(tmp_path: Path) -> None:
    loop, live = build(tmp_path, "Пепин, где ты?")
    asyncio.run(asyncio.wait_for(loop.run(max_sessions=1), timeout=10.0))
    assert len(loop.sessions) == 1 and loop.wakes == 1
    assert len(live.sessions[0].audio) >= 0.4 * 32_000  # the wake segment went first
    (row,) = wake_rows(tmp_path)
    assert row["wake"] is True and row["text"] == "Пепин, где ты?"
    assert str(row["wav"]).startswith("wake/")


@pytest.mark.parametrize(
    ("text", "noise"),
    [("просто разговор на кухне", False), ("Субтитры сделал DimaTorzok", True)],
)
def test_other_speech_opens_nothing_and_is_still_logged(
    tmp_path: Path, text: str, noise: bool
) -> None:
    loop, live = build(tmp_path, text)
    stop = threading.Event()

    async def main() -> None:
        runner = asyncio.create_task(loop.run(until=stop))
        while not loop.mic.done.is_set():  # type: ignore[attr-defined]
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        stop.set()
        await asyncio.wait_for(runner, timeout=5.0)

    asyncio.run(main())
    assert loop.sessions == [] and live.sessions == []
    (row,) = wake_rows(tmp_path)
    assert row["wake"] is False and row["hallucination"] is noise
    assert loop.hallucinations == int(noise)


def test_a_wake_past_the_hourly_cap_is_refused(tmp_path: Path) -> None:
    config = LiveConfig(max_sessions_per_hour=1)
    ledger = Ledger(tmp_path / "ledger.jsonl", config)
    ledger.record("earlier", time.time() - 60, 0.01, closed=True)
    loop, live = build(tmp_path, "Хей, Пепин", ledger)
    stop = threading.Event()

    async def main() -> None:
        runner = asyncio.create_task(loop.run(until=stop))
        while not loop.refused:
            await asyncio.sleep(0.01)
        stop.set()
        await asyncio.wait_for(runner, timeout=5.0)

    asyncio.run(asyncio.wait_for(main(), timeout=10.0))
    assert loop.refused == 1 and live.sessions == []
