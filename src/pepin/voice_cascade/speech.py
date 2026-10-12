"""The voice on its own thread: sentences in, PCM into the player as each chunk is produced.

Sentences are synthesised in the order they arrive, so the next one is being made while the
one before plays. A barge-in cancels every job of the turns up to the one cut (:meth:`cut`):
under the same lock as each chunk's hand-over to the player, so no chunk of a cut turn reaches
the speaker after the flush. Another voice can be swapped in while this one speaks
(:meth:`SpeechWorker.swap`): it loads on a thread of its own and takes over at the next job.

Asides (a slow tool's fillers, backchannels) are said by the voice itself, or by the local
fallback of a voice that pays per sentence, never by a cloud call; a paid voice without a
fallback says none. Their clips are made once the voice is ready, while no job waits. A clip
(an opener's too) is kept only within a length limit for its line (:func:`make_clip`): a model
sometimes babbles on for seconds after a hum.
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
from pepin.voice_cascade.stages import Billed, Voice, VoiceBill
from pepin.voice_cascade.text import clip_key, speakable
from pepin.voice_live.audio import Resampler

logger = logging.getLogger(__name__)

JobKind = Literal["reply", "backchannel", "end"]
CLIP_IDLE_S = 0.05  # a swapped-in voice makes its next clip after this long without a job
# A clip runs at most max(CLIP_MIN_LIMIT_S, CLIP_S_PER_CHAR per character): a model sometimes
# babbles on after a short line (2026-10-12, qwen3-1.7b, scratch/fillers/: 6 of 44 takes of
# hums ran 4.7-45 s, the others 0.4-3.0 s; words run ~0.09 s a character). A take past it is
# cut there and made again, CLIP_TRIES times at most; then the line has no clip.
CLIP_MIN_LIMIT_S = 2.5
CLIP_S_PER_CHAR = 0.25
CLIP_TRIES = 3


@dataclass(frozen=True)
class SpeechJob:
    """Sentence ``index`` of reply ``turn`` (said in ``style``, "" for the plain voice); a
    backchannel (``pcm`` ready, index -1); or the turn's end (play out what is queued)."""

    turn: int
    index: int
    text: str = ""
    kind: JobKind = "reply"
    pcm: bytes | None = None  # at the voice's rate
    style: str = ""


class _Wake:
    """Wakes the worker without a job (a swapped-in voice has clips to make)."""


AudioSink = Callable[[SpeechJob, bytes, float], None]  # (job, chunk at the voice rate, time)
DoneSink = Callable[[SpeechJob, float], None]
BillSink = Callable[[SpeechJob, VoiceBill], None]
SwapSink = Callable[[BaseException | None], None]  # None: the new voice speaks from now on


def make_clips(voice: Voice, lines: Sequence[str]) -> dict[str, bytes]:
    """``lines`` said once by ``voice``, by their clip key (:func:`make_clip`: a line every take
    of which ran too long has none); none for a voice that pays per sentence (each clip would
    be a call)."""
    if isinstance(voice, Billed):
        return {}
    made = {clip_key(line): make_clip(voice, line) for line in unique(lines)}
    return {key: pcm for key, pcm in made.items() if pcm is not None}


def clip_limit_s(text: str) -> float:
    """The longest a clip of ``text`` may run (the module's notes)."""
    return max(CLIP_MIN_LIMIT_S, CLIP_S_PER_CHAR * len(text))


def make_clip(voice: Voice, line: str) -> bytes | None:
    """``line`` said by ``voice`` within :func:`clip_limit_s`: a take past it is cut there and
    made again, :data:`CLIP_TRIES` times at most (None: every take ran past it)."""
    text = speakable(line)
    limit = int(clip_limit_s(text) * voice.rate) * 2  # s16le bytes
    for _ in range(CLIP_TRIES):
        chunks = voice.stream(text)
        pcm = b""
        try:
            for chunk in chunks:
                pcm += chunk
                if len(pcm) > limit:
                    break
        finally:
            close = getattr(chunks, "close", None)
            if close is not None:
                close()
        if len(pcm) <= limit:
            return pcm
        logger.warning(
            "speech: the clip %r ran past %.1f s: made again", line, limit / 2 / voice.rate
        )
    logger.warning("speech: no clip of %r: every take ran too long", line)
    return None


def aside_voice(voice: Voice) -> Voice | None:
    """Who says the asides (fillers, backchannels) of ``voice``: the voice itself, or for a
    voice that pays per sentence its local fallback (None: it has none, no aside is said)."""
    if not isinstance(voice, Billed):
        return voice
    fallback: Voice | None = getattr(voice, "fallback", None)
    return fallback


def unique(lines: Sequence[str]) -> list[str]:
    """``lines`` with each clip key once (the first line kept), in order; no empty one."""
    first: dict[str, str] = {}
    for line in lines:
        if key := clip_key(line):
            first.setdefault(key, line)
    return list(first.values())


class SpeechWorker:
    """Runs a :class:`pepin.voice_cascade.stages.Voice` on one thread for the whole run."""

    def __init__(
        self,
        voice: Voice,
        player: Player,
        *,
        on_audio: AudioSink | None = None,
        on_done: DoneSink | None = None,
        on_bill: BillSink | None = None,
        clips: Sequence[str] = (),
        asides: Sequence[str] = (),
        clock: Callable[[], float] = time.monotonic,
        clip_idle_s: float = CLIP_IDLE_S,
    ) -> None:
        """``on_audio`` hears every chunk handed to the player; ``on_done`` every job finished
        (cut or not); ``on_bill`` the bill of every sentence a paid voice said; all on this
        worker's thread. ``clips``: short lines (the commonest openers) synthesised once at
        load and played from memory when a job says one of them without a style
        (:func:`pepin.voice_cascade.text.clip_key`); none with a paid voice. ``asides``: the
        lines of backchannel jobs (fillers, "мм"), made by :func:`aside_voice` once the voice
        is ready, in order, each after ``clip_idle_s`` without a job."""
        self.player = player
        self.clip_lines = tuple(unique(clips))
        self.aside_lines = tuple(unique(asides))
        self.on_audio, self.on_done, self.on_bill = on_audio, on_done, on_bill
        self.clock = clock
        self.clip_idle_s = clip_idle_s
        self.ready = threading.Event()
        self.error: BaseException | None = None
        self.lock = threading.Lock()
        self.cut_through = -1  # jobs of turns up to this one are dropped
        self._slot: tuple[Voice, dict[str, bytes]] = (voice, {})
        self._swaps = 0  # the latest swap asked for; an older one finishing is dropped
        self._unmade: list[str] = []  # clip lines the voice has still to make
        self._told_unsaid = False
        self.job_rate = voice.rate  # Hz of the chunks on_audio hears now (its thread reads it)
        self._jobs: queue.Queue[SpeechJob | _Wake | None] = queue.Queue()
        self._thread = threading.Thread(target=self._run, name="speech", daemon=True)

    @property
    def voice(self) -> Voice:
        """The voice that says the next job."""
        return self._slot[0]

    @property
    def clips(self) -> dict[str, bytes]:
        """The current voice's clips by key."""
        return self._slot[1]

    @property
    def says_asides(self) -> bool:
        """The current voice says asides (a paid voice without a local fallback does not)."""
        return aside_voice(self._slot[0]) is not None

    def has_clip(self, line: str) -> bool:
        """The current voice says ``line`` as an aside from memory: its clip is made (read
        from any thread)."""
        voice, clips = self._slot
        return aside_voice(voice) is not None and clip_key(speakable(line)) in clips

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

    def swap(self, voice: Voice, on_ready: SwapSink | None = None) -> threading.Thread:
        """Load ``voice`` on a thread of its own, then make it the voice of the next job; the
        voice before keeps speaking until then, and stays if this one fails. Its clips are
        made afterwards on this worker's thread while no job waits (a line asked for before
        its clip is made is synthesised). ``on_ready`` hears the outcome on the loading thread
        (a later swap supersedes this one)."""
        with self.lock:
            self._swaps += 1
            number = self._swaps

        def load() -> None:
            try:
                voice.load()
            except BaseException as error:
                logger.exception("speech: the new voice did not load")
                if on_ready is not None:
                    on_ready(error)
                return
            with self.lock:
                if number != self._swaps:
                    outcome: BaseException | None = RuntimeError("superseded by a later swap")
                else:
                    self._slot, outcome = (voice, {}), None
                    self._unmade = self._to_make(voice, {})
                    self._jobs.put(_Wake())  # the worker may be waiting with no timeout
            if on_ready is not None:
                on_ready(outcome)

        thread = threading.Thread(target=load, name="voice-load", daemon=True)
        thread.start()
        return thread

    def close(self) -> None:
        """Stop after the queued jobs."""
        self._jobs.put(None)

    def _next_job(self) -> SpeechJob | None:
        """The next job; while none is queued, the swapped-in voice's clips are made one by
        one (on this thread, the one that speaks with it)."""
        while True:
            try:
                item = self._jobs.get(timeout=self.clip_idle_s if self._unmade else None)
            except queue.Empty:
                self._make_clip()
                continue
            if not isinstance(item, _Wake):
                return item

    def _to_make(self, voice: Voice, made: dict[str, bytes]) -> list[str]:
        """The lines still to be made for ``voice``: its clips (a voice that pays: none), then
        the asides (by its aside voice; none without one), less those in ``made``."""
        lines = [] if isinstance(voice, Billed) else list(self.clip_lines)
        if aside_voice(voice) is not None:
            lines += self.aside_lines
        return [line for line in unique(lines) if clip_key(line) not in made]

    def _make_clip(self) -> None:
        with self.lock:
            if not self._unmade:
                return
            line = self._unmade.pop(0)
            voice = self._slot[0]
        maker = aside_voice(voice)  # the voice itself, or a paid voice's fallback
        if maker is None:
            return
        try:
            pcm = make_clip(maker, line)
        except Exception:
            logger.exception("speech: the clip %r was not made", line)
            return
        if pcm is None:
            return
        with self.lock:
            if self._slot[0] is voice:  # still the voice speaking
                self._slot = (voice, {**self._slot[1], clip_key(line): pcm})

    def _run(self) -> None:
        try:
            voice = self.voice
            voice.load()
            clips = make_clips(voice, self.clip_lines)
            with self.lock:
                if self._slot[0] is voice:
                    self._slot = (voice, clips)
                    self._unmade = self._to_make(voice, clips)  # the asides, made while idle
        except BaseException as error:  # the run cannot speak: the caller reports it
            self.error = error
            self.ready.set()
            return
        self.ready.set()
        while (job := self._next_job()) is not None:
            try:
                self._speak(job)
            except Exception:  # one bad sentence must not silence the robot
                logger.exception("speech: %r", job.text[:80])
            if self.on_done is not None:
                self.on_done(job, self.clock())

    def _unsaid(self, job: SpeechJob) -> None:
        """A backchannel job a paid voice without a local fallback leaves unsaid (told once)."""
        if not self._told_unsaid:
            self._told_unsaid = True
            logger.warning("speech: a paid voice without a local fallback says no %r", job.text)

    def _speak(self, job: SpeechJob) -> None:
        if job.turn <= self.cut_through and job.kind != "backchannel":
            return
        if job.kind == "end":
            self.player.end()
            return
        voice, clips = self._slot
        clipper = aside_voice(voice)  # who made the clips
        speaker = clipper if job.kind == "backchannel" else voice
        if speaker is None:
            self._unsaid(job)
            return
        self.job_rate = speaker.rate
        resampler = Resampler(speaker.rate, self.player.rate)
        text = speakable(job.text)
        if not text and job.pcm is None:
            return  # nothing to say aloud (an aside)
        ready = job.pcm if job.pcm is not None else None
        if ready is None and not job.style and speaker is clipper:
            ready = clips.get(clip_key(text))
        chunks = iter([ready]) if ready is not None else speaker.stream(text, job.style)
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
            if ready is None and speaker is voice and isinstance(voice, Billed):
                bill = voice.bill()
                if bill is not None and self.on_bill is not None:
                    self.on_bill(job, bill)
