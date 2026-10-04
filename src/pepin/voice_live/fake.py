"""A Live API that never leaves the laptop: scripted for the tests, a beeper for the dry run.

:class:`ScriptedLive` plays a script of server messages, each step released by a trigger (the
connection, so many bytes of audio heard, a tool response); it keeps what it was sent.
:class:`DryRunLive` answers every pause in the speech it hears with a short beep, so the whole
loop — gate, session, speaker, events, logs, caps — runs on the robot without a paid call.
"""

from __future__ import annotations

import asyncio
import contextlib
import math
import struct
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass, field
from typing import Literal

from pepin.audio_link import FRAME_SAMPLES, RATE
from pepin.voice_live.audio import LIVE_OUT_RATE
from pepin.voice_live.budget import Usage
from pepin.voice_live.live import LiveMessage, LiveSession, ToolResponse
from pepin.voice_live.wake import Segmenter

Trigger = Literal["connect", "audio", "tool_response", "image"]


@dataclass
class Step:
    """Messages released when ``trigger`` happens (``audio``: once ``after_bytes`` of audio
    were heard in all)."""

    trigger: Trigger
    messages: list[LiveMessage]
    after_bytes: int = 0


@dataclass
class FakeSession:
    """One fake connection: what it was sent, and the queue of what it says."""

    script: list[Step]
    handle: str | None
    audio: bytearray = field(default_factory=bytearray)
    images: list[bytes] = field(default_factory=list)
    responses: list[ToolResponse] = field(default_factory=list)
    closed: bool = False
    _queue: asyncio.Queue[LiveMessage | None] = field(default_factory=asyncio.Queue)

    def release(self, trigger: Trigger) -> None:
        """Queue the next step's messages when ``trigger`` satisfies it."""
        while self.script:
            step = self.script[0]
            if step.trigger != trigger:
                return
            if trigger == "audio" and len(self.audio) < step.after_bytes:
                return
            self.script.pop(0)
            for message in step.messages:
                self._queue.put_nowait(message)

    def say(self, *messages: LiveMessage) -> None:
        """Queue messages now."""
        for message in messages:
            self._queue.put_nowait(message)

    def hang_up(self) -> None:
        """The server closes the connection."""
        self._queue.put_nowait(None)

    async def send_audio(self, pcm: bytes) -> None:
        """Hear audio."""
        self.audio += pcm
        self.release("audio")

    async def send_image(self, jpeg: bytes) -> None:
        """See a picture."""
        self.images.append(jpeg)
        self.release("image")

    async def send_tool_responses(self, responses: Sequence[ToolResponse]) -> None:
        """Take tool results."""
        self.responses.extend(responses)
        self.release("tool_response")

    async def messages(self) -> AsyncIterator[LiveMessage]:
        """What the script says, until the connection is closed."""
        while not self.closed:
            message = await self._queue.get()
            if message is None:
                return
            yield message


class ScriptedLive:
    """:class:`pepin.voice_live.live.LiveConnector` over scripts: the n-th connection plays
    ``scripts[n]``; every connection is kept in :attr:`sessions`."""

    def __init__(self, *scripts: list[Step]) -> None:
        """One script per expected connection."""
        self.scripts = [list(s) for s in scripts]
        self.sessions: list[FakeSession] = []
        self.handles: list[str | None] = []

    @contextlib.asynccontextmanager
    async def connect(self, handle: str | None) -> AsyncIterator[LiveSession]:
        """The next scripted connection."""
        self.handles.append(handle)
        script = self.scripts[len(self.sessions)] if len(self.sessions) < len(self.scripts) else []
        session = FakeSession(list(script), handle)
        self.sessions.append(session)
        session.release("connect")
        try:
            yield session
        finally:
            session.closed = True


def beep(seconds: float = 0.35, hz: float = 660.0, rate: int = LIVE_OUT_RATE) -> bytes:
    """A soft sine at the Live API's output rate, faded in and out."""
    n = int(seconds * rate)
    fade = max(1, int(0.02 * rate))
    return b"".join(
        struct.pack(
            "<h",
            int(6000 * min(1.0, i / fade, (n - i) / fade) * math.sin(2 * math.pi * hz * i / rate)),
        )
        for i in range(n)
    )


class _DryRunSession(FakeSession):
    """Beeps after every pause in what it hears; counts tokens as the real one would bill."""

    def __init__(self, handle: str | None) -> None:
        super().__init__([], handle)
        self._ears = Segmenter(silence_s=0.4)
        self._t = 0.0

    async def send_audio(self, pcm: bytes) -> None:
        self.audio += pcm
        for i in range(0, len(pcm), FRAME_SAMPLES * 2):
            self._t += FRAME_SAMPLES / RATE
            segment = self._ears.feed(pcm[i : i + FRAME_SAMPLES * 2], self._t)
            if segment is not None:
                heard_tokens = int(len(self.audio) / 2 / RATE * 25)
                self.say(
                    LiveMessage(input_text=f"(dry run: {segment.duration_s:.1f} s heard)"),
                    LiveMessage(audio=beep(), output_text="(dry run)"),
                    LiveMessage(
                        turn_complete=True,
                        usage=Usage(audio_in=heard_tokens, audio_out=9, reports=1),
                    ),
                )


class DryRunLive:
    """:class:`pepin.voice_live.live.LiveConnector` that beeps instead of calling Google."""

    def __init__(self) -> None:
        """No connections yet."""
        self.sessions: list[FakeSession] = []

    @contextlib.asynccontextmanager
    async def connect(self, handle: str | None) -> AsyncIterator[LiveSession]:
        """A local session; it hands out a handle so resumption is exercised too."""
        session = _DryRunSession(handle)
        self.sessions.append(session)
        session.say(LiveMessage(handle=f"dry-{len(self.sessions)}"))
        try:
            yield session
        finally:
            session.closed = True
