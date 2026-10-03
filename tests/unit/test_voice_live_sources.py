"""Where the voice hears and speaks, with fakes: the board link's reconnect, the scripted mic,
the WAV speaker, and the google-genai session adapter over a fake SDK session."""

import asyncio
import threading
import time
import wave
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import pytest
from google.genai import types

from pepin.audio_link import AudioFrame
from pepin.voice_live.live import ToolResponse, _GeminiSession
from pepin.voice_live.sources import BoardLink, ScriptedMic, WavSpeaker

FRAME = b"\x01\x00" * 320


class FakeClient:
    """An AudioClient that yields ``frames`` and then goes silent."""

    def __init__(self, frames: int) -> None:
        self.frames = frames
        self.sent: list[Any] = []
        self.closed = False
        self.play_rate = 16_000

    def next_frame(self, timeout_s: float = 1.0) -> AudioFrame | None:
        if self.frames > 0:
            self.frames -= 1
            return AudioFrame(0, 0.0, FRAME)
        time.sleep(0.001)
        return None

    def play(self, pcm: bytes) -> None:
        self.sent.append(("play", len(pcm)))

    def play_end(self) -> None:
        self.sent.append("play_end")

    def flush(self) -> None:
        self.sent.append("flush")

    def close(self) -> None:
        self.closed = True

    def _send(self, message: dict[str, Any]) -> None:
        self.sent.append(message)


def test_the_board_link_speaks_hears_and_replaces_a_silent_link() -> None:
    clients = [FakeClient(2), FakeClient(1)]
    opened: list[FakeClient] = []

    def connect(host: str) -> Any:
        opened.append(clients[len(opened)] if len(opened) < len(clients) else FakeClient(0))
        return opened[-1]

    link = BoardLink("board", connect)
    link.play(b"\x00\x00")
    link.play_end()
    link.flush()
    assert opened[0].sent == [("play", 2), "play_end", "flush"] and link.play_rate == 16_000
    heard: list[bytes] = []
    done = threading.Event()

    def sink(pcm: bytes, t: float) -> None:
        heard.append(pcm)
        if len(heard) == 3:
            done.set()

    link.start(sink)
    assert done.wait(2.0)  # two frames from the first link, one from its replacement
    assert opened[0].closed and len(opened) >= 2
    link.close()


def test_the_scripted_mic_waits_for_ready_and_says_each_utterance() -> None:
    ready = threading.Event()
    mic = ScriptedMic([FRAME * 3], ready.is_set, gap_s=0.0, lead_s=0.0)
    frames: list[bytes] = []
    mic.start(lambda pcm, t: frames.append(pcm))
    time.sleep(0.06)
    assert FRAME not in frames  # hiss only until ready
    ready.set()
    assert mic.done.wait(2.0)
    time.sleep(0.03)
    mic.close()
    assert frames.count(FRAME) == 3


def test_the_wav_speaker_writes_what_it_plays(tmp_path: Path) -> None:
    speaker = WavSpeaker(tmp_path / "out" / "s.wav")
    speaker.play(b"\x00\x00" * 160)
    speaker.play_end()
    speaker.flush()
    speaker.close()
    with wave.open(str(tmp_path / "out" / "s.wav")) as w:
        assert w.getnframes() == 160 and w.getframerate() == 16_000
    assert speaker.flushes == 1


class FakeSdkSession:
    """google-genai's AsyncSession, as far as the adapter uses it."""

    def __init__(self, turns: list[list[types.LiveServerMessage]]) -> None:
        self.turns = turns
        self.realtime: list[dict[str, Any]] = []
        self.tool_responses: list[Any] = []

    async def send_realtime_input(self, **kwargs: Any) -> None:
        self.realtime.append(kwargs)

    async def send_tool_response(self, *, function_responses: Any) -> None:
        self.tool_responses.append(function_responses)

    async def receive(self) -> AsyncIterator[types.LiveServerMessage]:
        if not self.turns:
            await asyncio.sleep(10)
        for message in self.turns.pop(0):
            yield message


def test_the_sdk_adapter_sends_audio_images_and_results_and_reads_across_turns() -> None:
    first = types.LiveServerMessage(
        server_content=types.LiveServerContent(output_transcription=types.Transcription(text="a"))
    )
    end = types.LiveServerMessage(server_content=types.LiveServerContent(turn_complete=True))
    second = types.LiveServerMessage(
        server_content=types.LiveServerContent(output_transcription=types.Transcription(text="b"))
    )
    sdk = FakeSdkSession([[first, end], [second]])
    session = _GeminiSession(sdk)

    async def main() -> list[str]:
        await session.send_audio(b"\x00\x00")
        await session.send_image(b"\xff\xd8")
        await session.send_tool_responses(
            [ToolResponse("1", "go_to", {"ok": True}, "WHEN_IDLE"), ToolResponse("2", "x", {})]
        )
        texts = []
        async for m in session.messages():
            texts.append(m.output_text)
            if len(texts) == 3:
                break
        return texts

    assert asyncio.run(asyncio.wait_for(main(), 5.0)) == ["a", "", "b"]
    audio, image = sdk.realtime
    assert audio["audio"].mime_type == "audio/pcm;rate=16000"
    assert image["video"].mime_type == "image/jpeg"
    (responses,) = sdk.tool_responses
    assert responses[0].scheduling == types.FunctionResponseScheduling.WHEN_IDLE
    assert responses[1].scheduling is None and responses[0].id == "1"


@pytest.mark.parametrize("missing", ["", "x"])
def test_no_key_no_client(monkeypatch: pytest.MonkeyPatch, missing: str) -> None:
    """Without a usable key the real connector cannot be built by accident in a test."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    from pepin.voice_live.config import LiveConfig
    from pepin.voice_live.live import GeminiLive

    if missing:
        monkeypatch.setenv("GEMINI_API_KEY", missing)
        live = GeminiLive(LiveConfig(), [])  # builds; nothing connects until a session opens
        assert live.config.model == "gemini-3.8-live"
    else:
        with pytest.raises(Exception):  # noqa: B017 -- the SDK's own "no key" error
            GeminiLive(LiveConfig(), [])
