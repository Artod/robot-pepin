"""The voice loop: idle behind the local wake gate, a Live conversation when called, idle again.

Frames from the mic (the board's array, or a scripted mic) arrive on a thread and are handed to
the event loop. While idle, the gate cuts them into segments and Whisper reads each one on a
worker thread; Whisper's lines for noise are dropped
(:func:`pepin.voice_live.wake.is_hallucination`), and a segment holding the robot's name, with
the caps' consent (:meth:`pepin.voice_live.budget.Ledger.refusal`), opens a session that is
given the segment itself first, so "Пепин, где ты?" said in one breath is heard whole. Every
segment's transcript and decision go to ``data/voice/<day>/wake.jsonl`` (its audio to ``wake/``).
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from pepin.audio_link import RATE
from pepin.voice_live.audio import Pacer, save_wav
from pepin.voice_live.budget import Ledger
from pepin.voice_live.config import LiveConfig
from pepin.voice_live.events import Events
from pepin.voice_live.session import Conversation, Frame, Wake
from pepin.voice_live.sources import FrameSink
from pepin.voice_live.wake import Segmenter, Transcriber, is_hallucination, is_wake

QUEUE_FRAMES = 1500  # 30 s of mic: frames beyond it (nobody reading) are dropped, oldest first


class Mic(Protocol):
    """A source of 20 ms frames delivered from its own thread."""

    def start(self, sink: FrameSink) -> None:
        """Begin delivering frames to ``sink``."""
        ...


class VoiceLoop:
    """Idle -> wake -> conversation -> idle, until stopped or ``max_sessions`` were held."""

    def __init__(
        self,
        config: LiveConfig,
        *,
        mic: Mic,
        pacer: Pacer,
        transcriber: Transcriber,
        conversation_factory: Callable[[asyncio.Queue[Frame]], Conversation],
        events: Events,
        logs: Path,
        ledger: Ledger | None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
    ) -> None:
        """``conversation_factory`` builds the :class:`Conversation` over the loop's frames."""
        self.config, self.mic, self.pacer = config, mic, pacer
        self.transcriber = transcriber
        self.conversation_factory = conversation_factory
        self.events, self.logs, self.ledger = events, logs, ledger
        self.clock, self.wall = clock, wall
        self.sessions: list[dict[str, object]] = []
        self.wakes = 0
        self.refused = 0
        self.hallucinations = 0
        self.dropped_frames = 0

    async def run(
        self, max_sessions: int | None = None, until: threading.Event | None = None
    ) -> None:
        """Listen until ``max_sessions`` sessions were held or ``until`` is set."""
        loop = asyncio.get_running_loop()
        frames: asyncio.Queue[Frame] = asyncio.Queue()

        def put(frame: Frame) -> None:
            if frames.qsize() >= QUEUE_FRAMES:
                frames.get_nowait()
                self.dropped_frames += 1
            frames.put_nowait(frame)

        def deliver(pcm: bytes, t: float) -> None:  # on the mic's thread
            with contextlib.suppress(RuntimeError):  # the loop has ended: nobody listens
                loop.call_soon_threadsafe(put, Frame(pcm, t))

        self.mic.start(deliver)
        conversation = self.conversation_factory(frames)
        gate = Segmenter(silence_s=self.config.wake_silence_s, max_s=self.config.wake_max_s)
        whisper = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="wake")
        self.events.emit("idle")
        try:
            while max_sessions is None or len(self.sessions) < max_sessions:
                if until is not None and until.is_set():
                    return
                try:
                    frame = await asyncio.wait_for(frames.get(), timeout=0.5)
                except TimeoutError:
                    continue
                if self.pacer.speaking():
                    gate.reset()  # our own voice, the tail of a goodbye
                    continue
                segment = gate.feed(frame.pcm, frame.t)
                if segment is None:
                    continue
                cut = self.clock()
                text = await loop.run_in_executor(whisper, self.transcriber.transcribe, segment.pcm)
                detected = self.clock()
                noise = is_hallucination(text)
                woke = not noise and is_wake(text)
                self.hallucinations += noise
                self._log_segment(segment.pcm, text, woke, round((detected - cut) * 1000), noise)
                print(
                    f"heard (local, {(detected - cut) * 1000:.0f} ms): {text!r}"
                    + ("  -> WAKE" if woke else "  (Whisper's noise line)" if noise else ""),
                    flush=True,
                )
                if not woke:
                    continue
                self.wakes += 1
                refusal = self.ledger.refusal() if self.ledger is not None else None
                if refusal is not None:
                    self.refused += 1
                    print(f"  !! session refused: {refusal}", flush=True)
                    continue
                wake = Wake(segment.pcm, text, segment.ended_s, cut, detected)
                self.sessions.append(await conversation.run(wake))
                gate.reset()
        finally:
            whisper.shutdown(wait=False)

    def _log_segment(self, pcm: bytes, text: str, woke: bool, ms: int, noise: bool) -> None:
        """The segment's transcript and decision (``noise``: a Whisper hallucination); its
        audio written on a thread."""
        t = self.wall()
        stamp = time.localtime(t)
        day = self.logs / time.strftime("%Y%m%d", stamp)
        row: dict[str, object] = {
            "t": round(t, 3),
            "s": round(len(pcm) / 2 / RATE, 2),
            "text": text,
            "wake": woke,
            "hallucination": noise,
            "whisper_ms": ms,
        }
        if self.config.keep_wake_wavs:
            name = f"wake/{time.strftime('%H%M%S', stamp)}_{int(t % 1 * 1000):03d}.wav"
            row["wav"] = name
            threading.Thread(target=save_wav, args=(day / name, pcm, RATE), daemon=True).start()
        day.mkdir(parents=True, exist_ok=True)
        with (day / "wake.jsonl").open("a") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
