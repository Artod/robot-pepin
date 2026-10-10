"""Fakes of the cascade's stages, so the whole loop runs in a test (and on the robot with no
paid call): a brain that answers from a script or repeats what it heard, a voice that hums, a
transcriber that reads from a list, and a measurer for the tagger that needs no Praat."""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from pepin.voice_cascade.stages import (
    FOLLOW_UP,
    Emit,
    FirstToken,
    Played,
    Prosody,
    Sentence,
    ToolDone,
    ToolExecutor,
    ToolUse,
    Usage,
)
from pepin.voice_cascade.text import SentenceSplitter

Step = str | tuple[str, dict[str, Any]]  # a sentence, or (tool name, its arguments)


@dataclass
class ScriptedBrain:
    """:class:`pepin.voice_cascade.stages.Brain` that answers turn ``n`` with ``script[n]``
    (sentences and tool calls in order; a tool call ends a round), or, past the script, with
    ``echo`` of what it heard. ``delay_s`` passes before every step, as a model's stream
    would take. It keeps every turn's outcome."""

    script: list[list[Step]] = field(default_factory=list)
    echo: str = "Ты сказал: {text}"
    delay_s: float = 0.0
    heard: list[str] = field(default_factory=list)
    outcomes: list[str] = field(default_factory=list)
    interrupts: list[Played] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)
    places: list[str] | None = None
    warmups: int = 0
    _said: list[str] = field(default_factory=list)
    _acted: bool = False

    def begin(self, places: list[str] | None) -> None:
        """A new conversation."""
        self.places = places
        self.history = []

    @property
    def acted(self) -> bool:
        """The open turn ran a tool."""
        return self._acted

    async def prewarm(self) -> Usage | None:
        """Counted."""
        self.warmups += 1
        return Usage(cache_write=100)

    async def turn(self, text: str, tools: ToolExecutor, emit: Emit) -> None:
        """Say the script's next reply (or the echo), running its tools."""
        n = len(self.heard)
        self.heard.append(text)
        self._said, self._acted = [], False
        steps: list[Step] = (
            self.script[n] if n < len(self.script) else [self.echo.format(text=text)]
        )
        first, round_index = True, 0
        for step in steps:
            await asyncio.sleep(self.delay_s)
            if isinstance(step, tuple):
                name, args = step
                self._acted = True
                emit(ToolUse(f"call-{n}-{round_index}", name, args))
                result = dict(await tools(name, args))
                result.pop(FOLLOW_UP, None)
                emit(ToolDone(f"call-{n}-{round_index}", name, str(result), 0))
                round_index += 1
                continue
            if first:
                first = False
                emit(FirstToken())
            splitter = SentenceSplitter(soft_chars=0)
            for sentence in [*splitter.feed(step), splitter.flush()]:
                if sentence:
                    self._said.append(sentence)
                    emit(Sentence(sentence, round_index))
        emit(Usage(input=10, output=5, stop="end_turn"))

    def commit(self) -> None:
        """Kept whole."""
        self.outcomes.append("committed")
        self.history.append((self.heard[-1], " ".join(self._said)))

    def interrupt(self, played: Played) -> None:
        """Kept as heard."""
        self.outcomes.append("interrupted")
        self.interrupts.append(played)
        heard = " ".join([*self._said[: played.sentences], played.partial]).strip()
        self.history.append((self.heard[-1], heard))

    def abandon(self) -> None:
        """Dropped."""
        self.outcomes.append("abandoned")


class ToneVoice:
    """:class:`pepin.voice_cascade.stages.Voice` that hums ``seconds_per_char`` per character
    in ``chunk_s`` chunks, at once (``delay_s`` before the first chunk)."""

    def __init__(
        self,
        rate: int = 24_000,
        *,
        seconds_per_char: float = 0.06,
        chunk_s: float = 0.5,
        delay_s: float = 0.0,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        """A 220 Hz hum at ``rate``."""
        self.rate = rate
        self.seconds_per_char, self.chunk_s, self.delay_s = seconds_per_char, chunk_s, delay_s
        self.sleep = sleep
        self.loaded = 0
        self.spoken: list[str] = []

    def load(self) -> None:
        """Counted."""
        self.loaded += 1

    def stream(self, text: str) -> Iterator[bytes]:
        """The hum, chunk by chunk."""
        self.spoken.append(text)
        if self.delay_s and self.sleep is not None:
            self.sleep(self.delay_s)
        total = max(1, int(len(text) * self.seconds_per_char * self.rate))
        step = max(1, int(self.chunk_s * self.rate))
        for start in range(0, total, step):
            n = min(step, total - start)
            t = (np.arange(n) + start) / self.rate
            yield (0.2 * 32767 * np.sin(2 * math.pi * 220 * t)).astype("<i2").tobytes()


class ListTranscriber:
    """:class:`pepin.voice_cascade.stages.Transcriber` that reads ``texts`` in order (the last
    one again once they run out)."""

    def __init__(self, texts: Sequence[str], delay_s: float = 0.0) -> None:
        """What Whisper would hear, utterance by utterance, each after ``delay_s``."""
        self.texts = list(texts)
        self.delay_s = delay_s
        self.heard: list[int] = []
        self.name = "fake:list"  # the engine named in the logs

    def transcribe(self, pcm: bytes) -> str:
        """The next text."""
        if self.delay_s:
            time.sleep(self.delay_s)
        self.heard.append(len(pcm))
        return self.texts[min(len(self.heard) - 1, len(self.texts) - 1)] if self.texts else ""


def loudness_measurer(pcm: bytes) -> Prosody:
    """A Praat stand-in: the loudness in dB of the voiced span and nothing else."""
    x = np.frombuffer(pcm[: len(pcm) - len(pcm) % 2], dtype="<i2").astype(float)
    if not len(x):
        return Prosody(active_s=0.0, loudness_db=-120.0)
    rms = math.sqrt(float(np.mean(x * x))) / 32768 + 1e-9
    return Prosody(active_s=len(x) / 16_000, loudness_db=20 * math.log10(rms) + 94)


class ScriptedTurn:
    """:class:`pepin.voice_cascade.turn.TurnDetector` that answers ``ps`` in order (the last
    one again once they run out), after ``delay_s``; it keeps how much audio it was given."""

    def __init__(self, ps: Sequence[float], delay_s: float = 0.0) -> None:
        """The probabilities to say, one a call."""
        self.ps = list(ps)
        self.delay_s = delay_s
        self.heard: list[int] = []

    def probability(self, pcm: bytes) -> float:
        """The next p."""
        if self.delay_s:
            time.sleep(self.delay_s)
        self.heard.append(len(pcm))
        return self.ps[min(len(self.heard) - 1, len(self.ps) - 1)]
