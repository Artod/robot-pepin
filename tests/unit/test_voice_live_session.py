"""A Live conversation against a scripted server: audio both ways, tools, interruption, the
caps, resumption and the logs — no network, no robot."""

# ruff: noqa: RUF001 -- Russian speech is the data here, not a lookalike of Latin

import asyncio
import contextlib
import json
import threading
import time
import wave
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from pepin.tools import progress
from pepin.voice_live.audio import Pacer
from pepin.voice_live.budget import Ledger, Usage
from pepin.voice_live.config import LiveConfig
from pepin.voice_live.events import Events, VoiceEvent
from pepin.voice_live.fake import DryRunLive, ScriptedLive, Step, beep
from pepin.voice_live.live import (
    END_CONVERSATION,
    LiveConnector,
    LiveMessage,
    LiveSession,
    ToolCall,
    ToolResponse,
)
from pepin.voice_live.session import UNASKED_DRIVE, Conversation, Frame, ResumeStore, Wake

pytestmark = pytest.mark.slow  # each session runs its idle timer and its speech in real time

SILENCE = b"\x00\x00" * 320


def speech(seconds: float, rate: int = 16_000) -> bytes:
    t = np.arange(int(seconds * rate)) / rate
    return (0.3 * 32767 * np.sin(2 * np.pi * 300 * t)).astype("<i2").tobytes()


class FakeSpeaker:
    def __init__(self) -> None:
        self.played = bytearray()
        self.ends = 0
        self.flushes = 0

    def play(self, pcm: bytes) -> None:
        self.played += pcm

    def play_end(self) -> None:
        self.ends += 1

    def flush(self) -> None:
        self.flushes += 1


class Rig:
    """A conversation over fakes, with a mic that hums silence while it runs."""

    def __init__(
        self,
        tmp: Path,
        live: LiveConnector,
        *,
        config: LiveConfig | None = None,
        ledger: bool = True,
        tools: Callable[[str, dict[str, Any]], dict[str, Any]] | None = None,
        transcriber: Any = None,
    ) -> None:
        self.config = config or LiveConfig(idle_close_s=0.15)
        self.speaker = FakeSpeaker()
        self.states: list[VoiceEvent] = []
        self.events = Events(self.states.append)
        self.pacer = Pacer(
            self.speaker,
            lead_s=1.0,
            tail_s=0.0,
            on_level=lambda value: self.events.emit("speaking", value),
        )
        self.tool_calls: list[tuple[str, dict[str, Any]]] = []
        self.halts = 0
        self.ledger = Ledger(tmp / "ledger.jsonl", self.config) if ledger else None
        self.resume = ResumeStore(tmp / "resume.json", 7000.0)
        self.tmp = tmp
        self.frames: asyncio.Queue[Frame] = asyncio.Queue()
        self.live = live

        def run_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
            self.tool_calls.append((name, args))
            return tools(name, args) if tools else {"ok": True}

        def halt() -> str:
            self.halts += 1
            return "cancel sent"

        self.conversation = Conversation(
            live,
            self.config,
            frames=self.frames,
            pacer=self.pacer,
            tools=run_tool,
            moving={"go_to"},
            halt=halt,
            events=self.events,
            logs=tmp / "voice",
            ledger=self.ledger,
            resume=self.resume,
            transcriber=transcriber,
            tick_s=0.005,
        )

    def run(
        self,
        wake_pcm: bytes | None = None,
        text: str = "Пепин, где ты?",
        voice: Callable[[float], bytes] | None = None,
    ) -> dict[str, Any]:
        """One session; ``voice(t)`` is the mic's frame ``t`` seconds after the wake (silence
        when None)."""

        async def main() -> dict[str, Any]:
            self.frames = asyncio.Queue()
            self.conversation.frames = self.frames
            start = time.monotonic()

            async def hum() -> None:
                while True:
                    now = time.monotonic()
                    pcm = voice(now - start) if voice is not None else SILENCE
                    self.frames.put_nowait(Frame(pcm, now))
                    await asyncio.sleep(0.02)  # a frame per frame: real time

            humming = asyncio.create_task(hum())
            now = time.monotonic()
            pcm = speech(0.3) if wake_pcm is None else wake_pcm
            wake = Wake(pcm, text, now - 0.6, now - 0.5, now)
            try:
                return await asyncio.wait_for(self.conversation.run(wake), timeout=10.0)
            finally:
                humming.cancel()

        stop = self.pacer.start()
        try:
            return asyncio.run(main())
        finally:
            stop.set()


def test_a_conversation_streams_both_ways_and_a_goodbye_closes_it(tmp_path: Path) -> None:
    wake = speech(0.3)
    live = ScriptedLive(
        [
            Step(
                "audio",
                [
                    LiveMessage(handle="h1"),
                    LiveMessage(input_text="Пепин, "),
                    LiveMessage(input_text="где ты?"),
                    LiveMessage(audio=beep(0.06), output_text="Я у "),
                    LiveMessage(audio=beep(0.06), output_text="принтера."),
                    LiveMessage(
                        turn_complete=True, usage=Usage(audio_in=8, audio_out=3, reports=1)
                    ),
                ],
                after_bytes=len(wake),
            ),
            Step(
                "audio",
                [
                    LiveMessage(input_text="Всё, спасибо, пока"),
                    LiveMessage(audio=beep(0.06), output_text="Пока!"),
                    LiveMessage(turn_complete=True),
                ],
                after_bytes=len(wake) + 3 * len(SILENCE),
            ),
        ]
    )
    rig = Rig(tmp_path, live)
    summary = rig.run(wake)

    assert summary["close"] == "goodbye"
    assert [(t["user"], t["model"]) for t in summary["turns"]] == [
        ("Пепин, где ты?", "Я у принтера."),
        ("Всё, спасибо, пока", "Пока!"),
    ]
    session = live.sessions[0]
    assert bytes(session.audio[: len(wake)]) == wake  # the wake segment is heard first
    # the model's 0.18 s at 24 kHz reached the speaker at 16 kHz
    assert len(rig.speaker.played) == pytest.approx(0.18 * 16_000 * 2, abs=1300)
    assert rig.speaker.ends >= 2
    assert summary["latency_ms"]["first_audio"] is not None
    assert summary["latency_ms"]["first_play"] is not None
    assert summary["usage"]["audio_in"] == 8 and summary["usd_est"] > 0

    log_dir = Path(summary["dir"])
    for name in ("in.wav", "out.wav", "events.jsonl", "summary.json"):
        assert (log_dir / name).is_file()
    with wave.open(str(log_dir / "out.wav")) as w:
        assert w.getframerate() == 24_000
        assert w.getnframes() == pytest.approx(0.18 * 24_000, abs=10)
    with wave.open(str(log_dir / "in.wav")) as w:
        assert w.getnframes() * 2 == len(session.audio)  # the log is what Live heard
    index = (log_dir.parent / "live.jsonl").read_text().splitlines()
    assert len(index) == 1 and json.loads(index[0])["close"] == "goodbye"
    lines = (log_dir / "events.jsonl").read_text().splitlines()
    kinds = [json.loads(line)["event"] for line in lines]
    assert kinds[0] == "wake" and "connected" in kinds and kinds[-1] == "close"

    assert rig.ledger is not None
    assert rig.ledger.today_usd() == pytest.approx(summary["usd_est"], abs=1e-5)
    assert rig.resume.get() == ("h1", pytest.approx(summary["audio_in_s"] + 0.36, abs=0.5))
    states = [e.state for e in rig.states]
    assert states[0] == "thinking" and "speaking" in states and states[-1] == "idle"
    assert "listening" in states


def test_tools_run_aside_and_their_results_go_back_and_end_conversation_closes(
    tmp_path: Path,
) -> None:
    live = ScriptedLive(
        [
            Step("connect", [LiveMessage(tool_calls=(ToolCall("c1", "where_am_i", {}),))]),
            Step(
                "tool_response",
                [
                    LiveMessage(audio=beep(0.05), output_text="Я дома. Пока!"),
                    LiveMessage(tool_calls=(ToolCall("c2", END_CONVERSATION, {}),)),
                ],
            ),
            Step("tool_response", [LiveMessage(turn_complete=True)]),
        ]
    )
    rig = Rig(tmp_path, live, tools=lambda name, args: {"ok": True, "at": "home"})
    summary = rig.run()
    assert rig.tool_calls == [("list_places", {}), ("where_am_i", {})]  # the book, then the call
    assert live.sessions[0].responses == [
        ToolResponse("c1", "where_am_i", {"ok": True, "at": "home"}, None),
        ToolResponse("c2", END_CONVERSATION, {"ok": True}),
    ]
    assert summary["close"] == END_CONVERSATION
    assert summary["tools"][0]["name"] == "where_am_i"
    assert summary["turns"][0]["tools"] == ["where_am_i", END_CONVERSATION]


def test_a_drive_does_not_block_and_a_cancelled_drive_is_halted(tmp_path: Path) -> None:
    release = threading.Event()
    calls: list[str] = []

    def slow_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name != "go_to":
            return {"ok": True}
        calls.append(name)
        release.wait(5.0)
        return {"ok": True, "arrived": True}

    class Connector(ScriptedLive):
        """Cancels the drive once it is running, then hangs up after the halt."""

        @contextlib.asynccontextmanager
        async def connect(self, handle: str | None, system: str) -> AsyncIterator[LiveSession]:
            async with super().connect(handle, system) as session:
                fake = self.sessions[-1]

                async def script() -> None:
                    fake.say(LiveMessage(tool_calls=(ToolCall("d1", "go_to", {"place": "home"}),)))
                    while not calls:
                        await asyncio.sleep(0.005)
                    fake.say(LiveMessage(audio=beep(0.03), output_text="Еду."))
                    fake.say(LiveMessage(cancelled=("d1",)))
                    while rig.halts == 0:
                        await asyncio.sleep(0.005)
                    release.set()
                    await asyncio.sleep(0.05)
                    fake.hang_up()

                task = asyncio.create_task(script())
                try:
                    yield session
                finally:
                    task.cancel()

    live = Connector()
    rig = Rig(tmp_path, live, tools=slow_tool)
    summary = rig.run()
    assert rig.halts == 1 and rig.conversation.moved
    assert live.sessions[0].responses == []  # the cancelled drive's result is not sent
    assert summary["close"] == "server closed"


def test_a_finished_drive_answers_when_idle(tmp_path: Path) -> None:
    live = ScriptedLive(
        [
            Step("connect", [LiveMessage(tool_calls=(ToolCall("d1", "go_to", {"place": "x"}),))]),
            Step("tool_response", [LiveMessage(turn_complete=True)]),
        ]
    )
    rig = Rig(tmp_path, live, tools=lambda name, args: {"ok": True, "arrived": True})
    rig.run()
    (response,) = live.sessions[0].responses
    assert response.scheduling == "WHEN_IDLE" and response.response == {"ok": True, "arrived": True}


def test_an_interruption_silences_the_speaker_and_silence_closes_the_session(
    tmp_path: Path,
) -> None:
    live = ScriptedLive(
        [
            Step(
                "connect",
                [
                    LiveMessage(audio=beep(0.3), output_text="Длинный ответ"),
                    LiveMessage(interrupted=True),
                    LiveMessage(turn_complete=True),
                ],
            )
        ]
    )
    rig = Rig(tmp_path, live)
    summary = rig.run()
    assert rig.speaker.flushes == 1
    assert summary["close"] == "idle"


def test_the_handle_resumes_the_next_session_and_a_stale_one_falls_back(tmp_path: Path) -> None:
    live = ScriptedLive([Step("connect", [LiveMessage(handle="h1")])], [])
    rig = Rig(tmp_path, live)
    first = rig.run()
    second = rig.run()
    assert live.handles == [None, "h1"]
    assert not first["resumed"] and second["resumed"]

    class RefusesHandles(ScriptedLive):
        @contextlib.asynccontextmanager
        async def connect(self, handle: str | None, system: str) -> AsyncIterator[LiveSession]:
            if handle is not None:
                self.handles.append(handle)
                raise ConnectionError("handle expired")
            async with super().connect(handle, system) as session:
                yield session

    stale = RefusesHandles([])
    rig2 = Rig(tmp_path / "b", stale)
    rig2.resume.save("old", 30.0)
    summary = rig2.run()
    assert stale.handles == ["old", None] and not summary["resumed"]
    assert summary["close"] == "idle" and summary["error"] is not None


def test_the_daily_budget_closes_a_session_that_crosses_it(tmp_path: Path) -> None:
    live = ScriptedLive(
        [Step("connect", [LiveMessage(usage=Usage(audio_in=10_000_000, reports=1))])]
    )
    rig = Rig(tmp_path, live, config=LiveConfig(idle_close_s=5.0, daily_budget_cad=0.5))
    summary = rig.run()
    assert summary["close"] == "daily budget"
    assert rig.ledger is not None and rig.ledger.refusal() is not None


def test_a_session_is_closed_at_max_session_s(tmp_path: Path) -> None:
    rig = Rig(tmp_path, ScriptedLive([]), config=LiveConfig(idle_close_s=5.0, max_session_s=0.2))
    summary = rig.run()
    assert summary["close"] == "max_session_s"


def test_the_mic_is_not_streamed_while_the_robot_speaks(tmp_path: Path) -> None:
    heard = {}
    for barge_in in (False, True):
        live = ScriptedLive(
            [Step("connect", [LiveMessage(audio=beep(0.4)), LiveMessage(turn_complete=True)])]
        )
        config = LiveConfig(idle_close_s=0.1, barge_in=barge_in)
        rig = Rig(tmp_path / str(barge_in), live, config=config)
        started = time.monotonic()
        summary = rig.run(speech(0.1))
        heard[barge_in] = (summary["audio_in_s"] - 0.1, time.monotonic() - started)
    gated, gated_wall = heard[False]
    open_mic, open_wall = heard[True]
    assert gated <= gated_wall - 0.4 + 0.06  # the 0.4 s of speech never went to Live
    assert open_mic >= open_wall - 0.15  # with barge-in it streams all along


def test_a_failed_connection_is_logged_without_the_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "SECRET-KEY-123456")

    class Broken(ScriptedLive):
        @contextlib.asynccontextmanager
        async def connect(self, handle: str | None, system: str) -> AsyncIterator[LiveSession]:
            raise ConnectionError("rejected key SECRET-KEY-123456")
            yield  # pragma: no cover

    rig = Rig(tmp_path, Broken())
    summary = rig.run()
    assert summary["close"] == "error"
    assert "SECRET-KEY-123456" not in json.dumps(summary)
    assert "***" in str(summary["error"])


def test_the_dry_run_beeps_after_the_speech_and_records_no_cost(tmp_path: Path) -> None:
    live = DryRunLive()
    rig = Rig(tmp_path, live, ledger=False, config=LiveConfig(idle_close_s=0.6))
    summary = rig.run(speech(0.4))
    assert summary["dry_run"] and summary["turns"][0]["model"] == "(dry run)"
    assert len(rig.speaker.played) > 0
    assert not (tmp_path / "ledger.jsonl").exists()
    assert rig.resume.get()[0] == "dry-1"


BOOK = {"ok": True, "places": [{"name": "home"}, {"name": "printer"}]}


def driving_tools(drive_s: float) -> Callable[[str, dict[str, Any]], dict[str, Any]]:
    """The places book, and a go_to that reports its start and arrives after ``drive_s``."""

    def run(name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name == "list_places":
            return BOOK
        if name == "go_to":
            progress(status="driving", target=args["place"])
            time.sleep(drive_s)
            return {"ok": True, "arrived": True, "target": args["place"]}
        return {"ok": True}

    return run


def go(call_id: str, place: str = "printer") -> LiveMessage:
    return LiveMessage(tool_calls=(ToolCall(call_id, "go_to", {"place": place}),))


def test_a_drive_answers_when_it_sets_off_and_its_end_is_the_robots_report(
    tmp_path: Path,
) -> None:
    live = ScriptedLive(
        [
            Step("connect", [LiveMessage(input_text="Пепин, к принтеру"), go("d1")]),
            Step("tool_response", [LiveMessage(audio=beep(0.05), output_text="Еду.")]),
            Step(
                "text",
                [
                    LiveMessage(audio=beep(0.05), output_text="Я на месте."),
                    LiveMessage(turn_complete=True),
                ],
            ),
        ]
    )
    config = LiveConfig(idle_close_s=5.0, idle_after_drive_s=0.2)
    rig = Rig(tmp_path, live, config=config, tools=driving_tools(0.2))
    started = time.monotonic()
    summary = rig.run()
    assert "home, printer" in live.systems[0]  # the book, read as the session opened
    session = live.sessions[0]
    (start,) = session.responses
    assert start.id == "d1" and start.scheduling == "WHEN_IDLE"
    assert start.response["status"] == "driving" and start.response["target"] == "printer"
    (report,) = session.texts
    assert report.startswith("[robot] the drive go_to(place='printer') has ended: ")
    assert json.loads(report.split("has ended: ", 1)[1])["arrived"] is True
    assert rig.tool_calls == [("list_places", {}), ("go_to", {"place": "printer"})]
    assert summary["close"] == "idle"
    assert time.monotonic() - started < 3.0  # idle_after_drive_s, not idle_close_s


def test_a_model_that_acknowledged_before_its_call_hears_the_start_silently(
    tmp_path: Path,
) -> None:
    live = ScriptedLive(
        [Step("connect", [LiveMessage(audio=beep(0.05), output_text="Слушаюсь."), go("d1")])]
    )
    rig = Rig(tmp_path, live, config=LiveConfig(idle_close_s=0.2), tools=driving_tools(0.05))
    rig.run()
    (start,) = live.sessions[0].responses
    assert start.scheduling == "SILENT" and start.response["status"] == "driving"


def test_a_drive_nobody_asked_for_is_refused_silently_until_a_person_speaks(
    tmp_path: Path,
) -> None:
    live = ScriptedLive(
        [
            Step("connect", [go("d1")]),
            Step("tool_response", [go("d2")]),  # again, at its start: nobody asked
            Step("text", [LiveMessage(input_text="А теперь домой"), go("d3", "home")]),
        ]
    )
    config = LiveConfig(idle_close_s=0.3, idle_after_drive_s=0.3)
    rig = Rig(tmp_path, live, config=config, tools=driving_tools(0.1))
    summary = rig.run()
    by_id = {r.id: r for r in live.sessions[0].responses}
    assert by_id["d2"].scheduling == "SILENT"
    assert by_id["d2"].response == {"ok": False, "why": UNASKED_DRIVE}
    assert by_id["d1"].response["status"] == by_id["d3"].response["status"] == "driving"
    assert [c for c in rig.tool_calls if c[0] == "go_to"] == [
        ("go_to", {"place": "printer"}),
        ("go_to", {"place": "home"}),
    ]
    assert summary["refused_drives"] == 1 and len(live.sessions[0].texts) == 2


class FakeEar:
    def __init__(self, text: str) -> None:
        self.text = text
        self.heard: list[int] = []

    def transcribe(self, pcm: bytes) -> str:
        self.heard.append(len(pcm))
        return self.text


def loud_seconds(pcm: bytes) -> float:
    x = np.frombuffer(pcm[: len(pcm) // 640 * 640], dtype="<i2").astype(float).reshape(-1, 320)
    return float((np.sqrt((x**2).mean(axis=1)) > 1000).sum() * 0.02)


@pytest.mark.parametrize(("heard", "forwarded", "halted"), [
    ("Пепин, стоп!", True, True),
    ("Пепин, а что ты видишь?", True, False),
    ("ну и ладно", False, False),
    ("Продолжение следует...", False, False),
])  # fmt: skip
def test_while_driving_the_mic_is_withheld_and_only_a_call_or_a_stop_reaches_live(
    tmp_path: Path, heard: str, forwarded: bool, halted: bool
) -> None:
    said = speech(0.4)

    def voice(t: float) -> bytes:  # the person speaks 0.4 s while the robot drives
        i = int((t - 0.4) * 16_000) * 2
        return said[i : i + 640].ljust(640, b"\x00") if 0 <= i < len(said) else SILENCE

    live = ScriptedLive([Step("connect", [go("d1")])])
    config = LiveConfig(idle_close_s=0.2, idle_after_drive_s=0.2)
    ear = FakeEar(heard)
    rig = Rig(tmp_path, live, config=config, tools=driving_tools(1.6), transcriber=ear)
    summary = rig.run(speech(0.3), voice=voice)
    session = live.sessions[0]
    assert len(ear.heard) == 1  # the segment, read while driving
    assert summary["heard_while_driving"][0]["text"] == heard
    assert summary["withheld_s"] > 1.2
    assert rig.halts == (1 if halted else 0)
    assert session.audio_ends == (1 if forwarded else 0)
    expected = 0.3 + (0.4 if forwarded else 0.0)  # the wake, and the segment if it went
    assert loud_seconds(bytes(session.audio)) == pytest.approx(expected, abs=0.08)


def test_room_noise_does_not_hold_the_session_open(tmp_path: Path) -> None:
    hum = speech(0.02)  # loud all along, as motors are; the server hears no one
    live = ScriptedLive([])
    config = LiveConfig(idle_close_s=0.3, max_session_s=3.0)
    rig = Rig(tmp_path, live, config=config)
    summary = rig.run(voice=lambda t: hum)
    assert summary["close"] == "idle"
