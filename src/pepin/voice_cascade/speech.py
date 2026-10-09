"""The voice on its own thread: sentences in, PCM into the player as each chunk is produced.

Sentences are synthesised in the order they arrive, so the next one is being made while the
one before plays. A barge-in cancels every job of the turns up to the one cut (:meth:`cut`):
under the same lock as each chunk's hand-over to the player, so no chunk of a cut turn reaches
the speaker after the flush.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

from pepin.voice_cascade.player import Player
from pepin.voice_cascade.stages import Voice
from pepin.voice_cascade.text import speakable
from pepin.voice_live.audio import Resampler

logger = logging.getLogger(__name__)

JobKind = Literal["reply", "backchannel", "end"]


@dataclass(frozen=True)
class SpeechJob:
    """Sentence ``index`` of reply ``turn``; a backchannel (``pcm`` ready, index -1); or the
    turn's end (play out what is queued)."""

    turn: int
    index: int
    text: str = ""
    kind: JobKind = "reply"
    pcm: bytes | None = None  # at the voice's rate


AudioSink = Callable[[SpeechJob, bytes, float], None]  # (job, chunk at the voice rate, time)
DoneSink = Callable[[SpeechJob, float], None]


class SpeechWorker:
    """Runs a :class:`pepin.voice_cascade.stages.Voice` on one thread for the whole run."""

    def __init__(
        self,
        voice: Voice,
        player: Player,
        *,
        on_audio: AudioSink | None = None,
        on_done: DoneSink | None = None,
        clips: Sequence[str] = (),
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """``on_audio`` hears every chunk handed to the player; ``on_done`` every job finished
        (cut or not); both on this worker's thread. ``clips``: short lines (backchannels)
        synthesised once at load and played from memory."""
        self.voice, self.player = voice, player
        self.clip_lines = tuple(clips)
        self.clips: dict[str, bytes] = {}
        self.on_audio, self.on_done = on_audio, on_done
        self.clock = clock
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.lock = threading.Lock()
        self.cut_through = -1  # jobs of turns up to this one are dropped
        self._jobs: queue.Queue[SpeechJob | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="speech", daemon=True)

    def start(self) -> None:
        """Load the voice on the worker's thread, then serve jobs; :attr:`ready` is set when
        the voice is loaded (or failed: :attr:`error`)."""
        self._thread.start()

    def say(self, job: SpeechJob) -> None:
        """Queue a job."""
        self._jobs.put(job)

    def cut(self, turn: int) -> float:
        """Barge-in: drop every job of turns up to ``turn`` and flush the speaker; the stream
        seconds that had played. Called from any thread."""
        with self.lock:
            self.cut_through = max(self.cut_through, turn)
            return self.player.flush()

    def close(self) -> None:
        """Stop after the queued jobs."""
        self._jobs.put(None)

    def _run(self) -> None:
        try:
            self.voice.load()
            for line in self.clip_lines:
                self.clips[line] = b"".join(self.voice.stream(speakable(line)))
        except BaseException as error:  # the run cannot speak: the caller reports it
            self.error = error
            self.ready.set()
            return
        self.ready.set()
        while (job := self._jobs.get()) is not None:
            try:
                self._speak(job)
            except Exception:  # one bad sentence must not silence the robot
                logger.exception("speech: %r", job.text[:80])
            if self.on_done is not None:
                self.on_done(job, self.clock())

    def _speak(self, job: SpeechJob) -> None:
        if job.turn <= self.cut_through and job.kind != "backchannel":
            return
        if job.kind == "end":
            self.player.end()
            return
        resampler = Resampler(self.voice.rate, self.player.rate)
        text = speakable(job.text)
        ready = job.pcm if job.pcm is not None else self.clips.get(job.text)
        chunks = iter([ready]) if ready is not None else self.voice.stream(text)
        player_kind: Literal["reply", "backchannel"] = (
            "backchannel" if job.kind == "backchannel" else "reply"
        )
        try:
            for pcm in chunks:
                with self.lock:
                    if job.kind == "reply" and job.turn <= self.cut_through:
                        return
                    self.player.feed(resampler.process(pcm), player_kind, job.turn, job.index)
                if self.on_audio is not None:
                    self.on_audio(job, pcm, self.clock())
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
