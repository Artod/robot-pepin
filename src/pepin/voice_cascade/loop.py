"""The cascade loop: the ears on the mic's thread, Whisper and Praat on workers, the brain's
stream in asyncio, the voice on its own thread feeding the pacer as sentences arrive.

An utterance closed by the ears is read by Whisper while Praat measures it; outside a
conversation only the robot's name (``wake: name``) or anything (``wake: always``) opens one.
The words, behind the tag line, go to the brain; each sentence it finishes goes to the voice at
once, so "Сейчас гляну." plays while the tool it announces runs. A reply's turn stays open in
the brain until its audio has played: a barge-in until then keeps only what was heard. A drive
answers its tool call when the goal server takes it (the brain says its acknowledgement) and
reports its end as the robot's own turn (``[robot] ...``). "Стоп" heard during a drive halts the
robot before the brain hears it. A conversation closes after ``idle_close_s`` of quiet, on a
goodbye, on ``end_conversation`` or at a cap.

Every turn prints one line of its timeline (:mod:`pepin.voice_cascade.timeline`) with the
brain's cache tokens and price, the brain and the voice that spoke (and a cloud voice's first
audio and price), and logs it with everything else (:mod:`pepin.voice_cascade.log`).

A sentence the brain starts with ``{шёпотом}`` is said in that style, and so is the rest of the
reply until another marker (``{}``: the plain voice again). With a switchboard
(:mod:`pepin.voice_cascade.control`) the brain and the voice are switched while it runs: a
new brain is built and warmed, then takes over at the next turn (the conversation starts
afresh with it); a new voice loads beside the one speaking and takes over at its next sentence.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import random
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from pepin.tools.registry import Result, fail, ok, progress_to, render
from pepin.voice_cascade.barge_in import FuturePending, Pending, build_detector
from pepin.voice_cascade.brain import END_CONVERSATION
from pepin.voice_cascade.config import CascadeConfig, VoiceConfig
from pepin.voice_cascade.control import Request, Switchboard
from pepin.voice_cascade.ears import EarEvent, Ears
from pepin.voice_cascade.log import CascadeLog, append_line, day_dir
from pepin.voice_cascade.player import Player
from pepin.voice_cascade.speech import SpeechJob, SpeechWorker
from pepin.voice_cascade.stages import (
    FOLLOW_UP,
    Brain,
    BrainEvent,
    FirstToken,
    Played,
    Prosody,
    Sentence,
    Tagger,
    ToolDone,
    ToolUse,
    Transcriber,
    Usage,
    VoiceBill,
)
from pepin.voice_cascade.text import speakable, split_style
from pepin.voice_cascade.timeline import Timeline
from pepin.voice_cascade.turn import TurnWorker
from pepin.voice_live.audio import Resampler
from pepin.voice_live.budget import Ledger
from pepin.voice_live.events import Events, State
from pepin.voice_live.sources import FrameSink
from pepin.voice_live.wake import Segment, is_goodbye, is_hallucination, is_stop, is_wake

ToolRunner = Callable[[str, dict[str, Any]], Result]
ROBOT_REPORT = "[robot]"
UNDER_WAY = (
    f"under way: a {ROBOT_REPORT} message reports the drive's end; do not call it again meanwhile"
)
ACKNOWLEDGED = "; your acknowledgement is already said"
UNASKED_DRIVE = (
    "not driving: no one has asked for another drive since the last one; never repeat a drive"
    " unless a person asks again"
)
END_WAIT_S = 10.0  # a robot report waits this long at most for the robot's line to end
PLAYOUT_WAIT_S = 120.0  # a reply's audio is waited for this long at most
# tools that take a while: the robot says its filler when the brain calls one before a word
SLOW_TOOLS = frozenset(("see", "look", "look_around", "find", "recall", "map_tree"))
BACKCHANNEL_GAP_S = 2.0  # one "мм" per this long at most, however many pauses
HOLD_RELEASE_S = 0.3  # a held utterance waits this long after the person stopped without words


class Mic(Protocol):
    """A source of 20 ms frames delivered from its own thread."""

    def start(self, sink: FrameSink) -> None:
        """Begin delivering frames to ``sink``."""
        ...


@dataclass
class Input:
    """What a turn answers: the person's words behind their tag line, or the robot's report."""

    kind: Literal["person", "robot"]
    text: str  # what the brain reads
    timeline: Timeline
    words: str = ""
    tag: str = ""
    goodbye: bool = False
    eot: dict[str, Any] | None = None  # the end-of-turn model's verdict on the utterance
    fragments: int = 1  # utterances joined into this turn (the person went on)


@dataclass
class Reply:
    """One turn in flight."""

    id: int
    input: Input
    task: asyncio.Task[None] | None = None
    sentences: list[str] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    calls: int = 0
    tools: list[dict[str, Any]] = field(default_factory=list)
    audio: bool = False  # some of its audio reached the player
    acted: bool = False  # it ran a tool
    filled: bool = False  # the local filler was said for a slow tool
    generated: bool = False  # the brain's stream ended
    spoken: asyncio.Event = field(default_factory=asyncio.Event)  # the voice did its last job
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    outcome: str = ""  # committed, interrupted, merged, error
    played: Played | None = None
    barged_s: float | None = None
    error: str | None = None
    brain: str = ""  # the brain's preset when the turn began
    voice: str = ""  # the voice's preset when the turn began
    style: str = ""  # the style the brain asked for, in force for its next sentences
    styles: list[str] = field(default_factory=list)  # every style it asked for
    bills: list[VoiceBill] = field(default_factory=list)  # a paid voice's, one per sentence


@dataclass
class Conversation:
    """From the wake to the close."""

    sid: str
    t_open_wall: float
    t_open: float
    log: CascadeLog
    last_heard_s: float
    turns: list[dict[str, Any]] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    voice_usd: float = 0.0  # a paid voice's estimate
    closing: str | None = None

    @property
    def usd(self) -> float:
        """The conversation's estimate: the brain's calls and a paid voice's."""
        return self.usage.usd + self.voice_usd


class CascadeLoop:
    """Ears -> Whisper (+ the tag line) -> brain -> voice -> speaker, until stopped."""

    def __init__(
        self,
        config: CascadeConfig,
        *,
        mic: Mic,
        player: Player,
        transcriber: Transcriber,
        tagger: Tagger | None,
        brain: Brain,
        speech: SpeechWorker,
        tools: ToolRunner,
        allowed: set[str],
        moving: set[str],
        halt: Callable[[], str],
        events: Events,
        logs: Path,
        ledger: Ledger | None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        tick_s: float = 0.1,
        seed: int = 0,
        turn: TurnWorker | None = None,
        switchboard: Switchboard | None = None,
    ) -> None:
        """``tools`` runs a registry tool by name; ``allowed`` names those the brain may call,
        ``moving`` those that drive; ``ledger`` None spends nothing worth recording (a fake
        brain); ``switchboard``: the brain and the voice may be switched while it runs."""
        self.config = config
        self.mic, self.player, self.transcriber, self.tagger = mic, player, transcriber, tagger
        self.brain, self.speech = brain, speech
        self.tools, self.allowed, self.moving, self.halt = tools, allowed, moving, halt
        self.events, self.logs, self.ledger = events, logs, ledger
        self.clock, self.wall, self.tick_s = clock, wall, tick_s
        self._whisper = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="whisper")
        early = self._read_early if config.early_s > 0 else None
        detector = None
        if config.barge_in:
            detector = build_detector(
                config,
                reader=self._read_words,
                own_text=self._own_text,
                played=player.playout.level_between,
            )
        self.ears = Ears(config, player.playback, turn, early, detector)
        self.rng = random.Random(seed)
        self.conversation: Conversation | None = None
        self.replies: dict[int, Reply] = {}
        self.current: Reply | None = None
        self.last_turn = 0  # the id of the latest reply (the mic thread cuts up to it)
        self.inputs: asyncio.Queue[Input] = asyncio.Queue()
        self.drives: set[asyncio.Task[Result]] = set()
        self.heard_since_drive = True
        self.places: list[str] | None = None
        self.turns: list[dict[str, Any]] = []  # every finished turn (the bench reads these)
        self.person_turns = 0
        self.hearing = 0  # utterances being read
        self.last_call_s = 0.0
        self.last_reply_end_s = 0.0
        self.last_backchannel_s = -BACKCHANNEL_GAP_S
        self.last_closed_s = -1e9  # when the ears last closed an utterance
        self.held: Input | None = None  # an utterance waiting for the rest of the sentence
        self.held_s = 0.0
        self.hold_next = False  # the utterance being read is to be held, not answered
        self._ears_queue: asyncio.Queue[tuple[EarEvent, float | None]] = asyncio.Queue()
        self._order = asyncio.Lock()
        self._praat = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="prosody")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._background: set[asyncio.Task[Any]] = set()
        self.switchboard = switchboard
        self.paid_ledger = ledger  # the ledger a paid brain switched in records into
        self.brain_name = (
            switchboard.brain
            if switchboard is not None
            else config.brain.preset or config.brain.model
        )
        self._log_resampler: tuple[int, int, Resampler] | None = None
        speech.on_audio = self._on_audio
        speech.on_done = self._on_spoken
        speech.on_bill = self._on_bill

    # -- the run -----------------------------------------------------------------------------

    async def run(self, until: threading.Event | None = None, turns: int | None = None) -> None:
        """Listen until ``until`` is set or ``turns`` turns of the person have been answered."""
        self._loop = asyncio.get_running_loop()
        await asyncio.to_thread(self.speech.ready.wait)
        if self.speech.error is not None:
            raise RuntimeError(f"the voice did not load: {self.speech.error!r}")
        self.places = await asyncio.to_thread(self.read_places)
        self.brain.begin(self.places)
        await self.prewarm("start")
        stop_player = self.player.start()
        self.mic.start(self._on_frame)
        tasks = [
            asyncio.create_task(self._listen(), name="ears"),
            asyncio.create_task(self._conduct(), name="conductor"),
            asyncio.create_task(self._watch(), name="watch"),
        ]
        self.events.emit("idle")
        try:
            while not (until is not None and until.is_set()):
                if turns is not None and self.person_turns >= turns and self.quiet():
                    break
                for task in tasks:
                    if task.done():
                        task.result()  # a crashed task stops the run with its error
                await asyncio.sleep(self.tick_s)
        finally:
            for task in tasks + list(self._background):
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.conversation is not None:
                self.close("stopped")
            close = getattr(self.brain, "aclose", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    await close()
            stop_player.set()
            self._whisper.shutdown(wait=False)
            self._praat.shutdown(wait=False)

    def quiet(self) -> bool:
        """Nothing in flight: no utterance being read, no reply, no queued input, no audio, no
        drive whose report is still to come."""
        return (
            not self.drives
            and self.hearing == 0
            and self.current is None
            and self.inputs.empty()
            and self._ears_queue.empty()
            and not self.player.speaking()
            and not self.ears.gate.talking
        )

    def read_places(self) -> list[str] | None:
        """The names in the places book (list_places through the registry), None when it
        cannot be read."""
        if "list_places" not in self.allowed:
            return None
        result = self.tools("list_places", {})
        rows = result.get("places") if result.get("ok") else None
        if not isinstance(rows, list):
            return None
        names = [str(r["name"]) for r in rows if isinstance(r, dict) and "name" in r]
        return names or None

    async def prewarm(self, why: str) -> None:
        """Write the brain's prompt cache now (a failure is printed, never fatal)."""
        self.last_call_s = self.clock()
        try:
            usage = await self.brain.prewarm()
        except Exception as error:
            print(f"  !! prewarm ({why}): {type(error).__name__}: {error}"[:300], flush=True)
            return
        if usage is not None:
            print(
                f"  brain warm ({why}): cache {usage.cache_read}r/{usage.cache_write}w,"
                f" {usage.ms} ms, {usage.usd * 100:.4f} US cents",
                flush=True,
            )

    # -- the mic's thread ----------------------------------------------------------------------

    def _on_frame(self, pcm: bytes, t: float) -> None:
        conversation = self.conversation
        if conversation is not None:
            conversation.log.mic.write(pcm)
        for event in self.ears.feed(pcm, t):
            played = self.speech.cut(self.last_turn) if event.kind == "barge_in" else None
            if self._loop is not None:
                with contextlib.suppress(RuntimeError):  # the loop has ended
                    self._loop.call_soon_threadsafe(self._ears_queue.put_nowait, (event, played))

    # -- the voice's thread --------------------------------------------------------------------

    def _on_audio(self, job: SpeechJob, pcm: bytes, t: float) -> None:
        conversation = self.conversation
        if conversation is not None:
            conversation.log.voice.write(self._at_log_rate(pcm, conversation.log.voice.rate))
        reply = self.replies.get(job.turn)
        if reply is not None and job.kind == "reply":
            reply.audio = True
            reply.input.timeline.mark("tts_first_audio", t)

    def _at_log_rate(self, pcm: bytes, rate: int) -> bytes:
        """A voice chunk at the rate the conversation's out.wav was opened with (a voice
        switched in since may speak at another)."""
        voice_rate = self.speech.job_rate
        if voice_rate == rate:
            return pcm
        held = self._log_resampler
        if held is None or held[:2] != (voice_rate, rate):
            held = self._log_resampler = (voice_rate, rate, Resampler(voice_rate, rate))
        return held[2].process(pcm)

    def _on_spoken(self, job: SpeechJob, t: float) -> None:
        reply = self.replies.get(job.turn)
        if reply is not None and job.kind == "end" and self._loop is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(reply.spoken.set)

    def _on_bill(self, job: SpeechJob, bill: VoiceBill) -> None:
        """A paid voice's sentence (on the voice's thread): to the turn and the conversation."""
        if self._loop is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(self._billed, job, bill)

    def _billed(self, job: SpeechJob, bill: VoiceBill) -> None:
        conversation = self.conversation
        if conversation is not None:
            conversation.voice_usd += bill.usd
            conversation.log.event("voice_bill", self.clock(), turn=job.turn, index=job.index,
                                   style=job.style, **vars(bill))  # fmt: skip
        reply = self.replies.get(job.turn)
        if reply is not None:
            reply.bills.append(bill)
        if bill.fallback is not None:
            print(f"  (voice: {bill.engine} spoke instead: {bill.fallback})", flush=True)

    # -- the ears ------------------------------------------------------------------------------

    async def _listen(self) -> None:
        while True:
            event, played = await self._ears_queue.get()
            if event.kind == "start":
                if self.conversation is not None:
                    self.conversation.last_heard_s = event.t
                    lead_s = len(self.ears.gate.audio()) / 2 / 16_000  # the lead-in kept
                    self.conversation.log.event("start", event.t, lead_s=round(lead_s, 2))
                cfg = self.config
                if cfg.continuation_s > 0 and event.t - self.last_closed_s <= cfg.continuation_s:
                    await self._continuation()
            elif event.kind == "pause":
                self._backchannel()
            elif event.kind == "barge_in":
                await self._barge_in(event, played if played is not None else 0.0)
            elif event.kind == "segment" and event.segment is not None:
                self.last_closed_s = event.t
                self.hearing += 1
                self._spawn(self._hear(event))

    async def _continuation(self) -> None:
        """The person started again right after an utterance closed: if nothing of the reply
        has played yet, that utterance waits for the rest (its brain call cancelled)."""
        current = self.current
        if (
            current is not None
            and not current.finished.is_set()
            and current.input.kind == "person"
            and not current.audio
            and not current.acted
        ):
            await self._drop(current, "merged")
            self._hold(current.input)
        elif self.hearing > 0:
            self.hold_next = True  # still being read: held when Whisper is done

    def _hold(self, item: Input) -> None:
        held = self.held
        if held is not None:  # a third fragment: all of them wait together
            item = Input(
                "person", f"{held.text} {item.text}", item.timeline,
                f"{held.words} {item.words}", item.tag, item.goodbye, item.eot,
                held.fragments + item.fragments,
            )  # fmt: skip
        self.held, self.held_s = item, self.clock()
        self._event("held", text=item.words)
        print(f"  (the person goes on: holding {item.words!r})", flush=True)

    def _release(self) -> None:
        """The rest never came (noise, a cough): the held utterance is answered alone."""
        if self.held is not None:
            self.inputs.put_nowait(self.held)
            self.held = None

    def _backchannel(self) -> None:
        cfg = self.config
        if not (cfg.backchannel and cfg.backchannel_lines and self.conversation is not None):
            return
        now = self.clock()
        if self.player.speaking() or now - self.last_backchannel_s < BACKCHANNEL_GAP_S:
            return
        self.last_backchannel_s = now
        line = self.rng.choice(cfg.backchannel_lines)
        self.speech.say(SpeechJob(self.last_turn, -1, line, "backchannel"))
        self._event("backchannel", text=line)

    def _read_early(self, pcm: bytes) -> concurrent.futures.Future[str]:
        """Whisper on an utterance not yet closed (the ears call this from the mic's thread)."""
        return self._whisper.submit(self.transcriber.transcribe, pcm)

    def _read_words(self, pcm: bytes, t: float) -> Pending:
        """Whisper on a voice over the reply (the barge-in detector, from the mic's thread)."""
        return FuturePending(self._whisper.submit(self.transcriber.transcribe, pcm))

    def _own_text(self) -> str:
        """The sentences of the reply in play so far (read from the mic's thread)."""
        reply = self.replies.get(self.last_turn)
        return " ".join(speakable(s) for s in list(reply.sentences)) if reply is not None else ""

    async def _hear(self, event: EarEvent) -> None:
        assert event.segment is not None
        try:
            async with self._order:  # utterances are answered in the order they were said
                await self._read(event.segment, event.t, eot_of(event), event.early)
        finally:
            self.hearing -= 1

    async def _read(
        self,
        segment: Segment,
        closed_s: float,
        eot: dict[str, Any] | None = None,
        early: concurrent.futures.Future[str] | None = None,
    ) -> None:
        loop = asyncio.get_running_loop()
        timeline = Timeline(segment.ended_s)
        timeline.mark("closed", closed_s)
        if early is not None and not early.cancelled():
            reading = asyncio.wrap_future(early)  # started in the pause: done, or nearly
        else:
            early = None
            reading = loop.run_in_executor(self._whisper, self.transcriber.transcribe, segment.pcm)
        tagger = self.tagger if self.config.prosody_tags else None
        measuring = (
            loop.run_in_executor(self._praat, tagger.measure, segment.pcm)
            if tagger is not None
            else None
        )
        text = (await reading).strip()
        prosody: Prosody | None = await measuring if measuring is not None else None
        read_s = self.clock()
        noise = not text or is_hallucination(text)
        whisper_ms = round((read_s - closed_s) * 1000)
        engine = getattr(self.transcriber, "name", None)  # the engine in use (switchable)
        voice_db = round(self.ears.voice_peak_db, 1)
        if self.conversation is None:
            woke = not noise and (self.config.wake == "always" or is_wake(text))
            print(
                f"heard (idle, {whisper_ms} ms): {text!r}"
                + ("  -> WAKE" if woke else "  (Whisper's noise line)" if noise else ""),
                flush=True,
            )
            row = {
                "t": round(self.wall(), 3), "text": text, "wake": woke, "noise": noise,
                "whisper_ms": whisper_ms, "engine": engine,
            }  # fmt: skip
            append_line(day_dir(self.logs, self.wall()) / "cascade_heard.jsonl", row)
            if not woke or not self.open(segment.pcm):
                return
        conversation = self.conversation
        assert conversation is not None
        conversation.log.event(
            "heard", read_s, text=text, noise=noise, whisper_ms=whisper_ms, early=early is not None,
            seconds=round(segment.duration_s, 2), voice_peak_db=voice_db, engine=engine,
        )  # fmt: skip
        if noise:
            self.hold_next = False
            self._release()
            return
        conversation.last_heard_s = read_s
        tag = tagger.line(prosody, text) if tagger is not None and prosody is not None else ""
        timeline.mark("transcript", self.clock())
        print(f"heard> {tag + ' ' if tag else ''}{text}  ({whisper_ms} ms)", flush=True)
        if is_stop(text) and self.drives:
            answer = await asyncio.to_thread(self.halt)
            conversation.log.event("halt", self.clock(), answer=answer, why="stop heard")
            print(f"  (stop heard during a drive: {answer})", flush=True)
        self.heard_since_drive = True
        said, words, fragments = (f"{tag} {text}" if tag else text), text, 1
        current = self.current
        if current is not None and not current.audio and not current.acted:
            # the person went on before the robot said or did anything: one turn of both
            await self._drop(current, "merged")
            said, words = f"{current.input.text} {said}", f"{current.input.words} {text}"
            fragments += current.input.fragments
        held = self.held
        if held is not None:
            said, words = f"{held.text} {said}", f"{held.words} {words}"
            fragments += held.fragments
            self.held = None
        item = Input("person", said, timeline, words, tag, is_goodbye(text), eot, fragments)
        if self.hold_next:
            self.hold_next = False
            self._hold(item)
            return
        self.inputs.put_nowait(item)

    # -- turns ---------------------------------------------------------------------------------

    async def _conduct(self) -> None:
        while True:
            item = await self.inputs.get()
            if self.conversation is None and not self.open(b""):
                continue  # closed meanwhile (a drive's report, a turn heard at the close)
            if item.kind == "robot":
                deadline = self.clock() + END_WAIT_S
                while self.player.speaking() and self.clock() < deadline:
                    await asyncio.sleep(self.tick_s)
            self.last_turn += 1
            reply = Reply(self.last_turn, item, brain=self.brain_name, voice=self.voice_name)
            self.replies[reply.id] = reply
            self.current = reply
            self.events.emit("thinking")
            reply.task = asyncio.create_task(self._reply(reply), name=f"reply-{reply.id}")
            await reply.finished.wait()
            self.current = None
            self.last_reply_end_s = self.clock()
            conversation = self.conversation
            if conversation is not None and item.goodbye:
                conversation.closing = conversation.closing or "goodbye"

    async def _reply(self, reply: Reply) -> None:
        try:
            await self.brain.turn(reply.input.text, self._tool, lambda e: self._heard(reply, e))
            reply.generated = True
            self.speech.say(SpeechJob(reply.id, len(reply.sentences), kind="end"))
            await self._played_out(reply)
            self.brain.commit()
            reply.outcome = "committed"
        except asyncio.CancelledError:
            if not reply.outcome:  # cancelled by the run's end, not by a barge-in or a merge
                self.brain.abandon()
                reply.outcome = "stopped"
            raise
        except Exception as error:
            reply.error = f"{type(error).__name__}: {error}"[:300]
            reply.outcome = "error"
            print(f"  !! brain: {reply.error}", flush=True)
            self.brain.abandon()
            if not reply.audio and self.config.error_line:
                self.speech.say(SpeechJob(reply.id, len(reply.sentences), self.config.error_line))
            self.speech.say(SpeechJob(reply.id, len(reply.sentences) + 1, kind="end"))
        finally:
            if reply.outcome in ("committed", "error", "stopped"):
                self._finish(reply)

    def _heard(self, reply: Reply, event: BrainEvent) -> None:
        """A brain event of ``reply``, on the event loop."""
        now = self.clock()
        timeline = reply.input.timeline
        conversation = self.conversation
        log = conversation.log if conversation is not None else None
        if isinstance(event, FirstToken):
            timeline.mark("first_token", now)
        elif isinstance(event, Sentence):
            timeline.mark("first_sentence", now)
            index = len(reply.sentences)
            reply.sentences.append(event.text)
            style, _ = split_style(event.text)
            if style is not None:  # in force for the rest of the reply
                reply.style = style
                reply.styles.append(style)
            self.speech.say(SpeechJob(reply.id, index, event.text, style=reply.style))
            if log is not None:
                log.event("sentence", now, turn=reply.id, index=index, text=event.text,
                          style=reply.style)  # fmt: skip
        elif isinstance(event, ToolUse):
            if event.name in SLOW_TOOLS and not reply.sentences and not reply.filled:
                reply.filled = True  # nothing said yet: the wait is announced locally
                if self.config.tool_filler:
                    filler = SpeechJob(reply.id, -1, self.config.tool_filler, "backchannel")
                    self.speech.say(filler)
            reply.acted = True
            print(f"  -> {event.name}({json.dumps(event.args, ensure_ascii=False)})", flush=True)
            if log is not None:
                log.event("tool_call", now, turn=reply.id, id=event.id, name=event.name,
                          args=event.args)  # fmt: skip
        elif isinstance(event, ToolDone):
            reply.tools.append({"name": event.name, "result": event.result[:300], "ms": event.ms})
            print(f"  <- {event.result[:200]}  ({event.ms} ms)", flush=True)
            if log is not None:
                log.event("tool_result", now, turn=reply.id, id=event.id, name=event.name,
                          result=event.result, ms=event.ms)  # fmt: skip
        elif isinstance(event, Usage):
            self.last_call_s = now
            reply.usage = reply.usage.plus(event)
            reply.calls += 1
            if conversation is not None:
                conversation.usage = conversation.usage.plus(event)
            if log is not None:
                log.event("usage", now, turn=reply.id, **vars(event))

    async def _played_out(self, reply: Reply) -> None:
        """Wait until the voice has done the reply's last job and the speaker is silent."""
        try:
            await asyncio.wait_for(reply.spoken.wait(), PLAYOUT_WAIT_S)
        except TimeoutError:
            return
        deadline = self.clock() + PLAYOUT_WAIT_S
        while self.player.speaking() and self.clock() < deadline:
            await asyncio.sleep(0.02)

    async def _barge_in(self, event: EarEvent, played: float) -> None:
        """The person cut in: the reply stops where they cut it; what they heard stays."""
        reply = self.replies.get(self.last_turn)
        if reply is None or reply.finished.is_set():
            return
        reply.barged_s = event.t
        said = [split_style(s)[1] for s in reply.sentences]  # the style marker was not heard
        reply.played = self.player.heard(reply.id, played, said)
        reply.outcome = "interrupted"
        await self._cancel(reply)
        self.brain.interrupt(reply.played)
        self._event("barge_in", turn=reply.id, level_db=round(event.level_db, 1), why=event.why,
                    played_s=round(played, 3), doa=self._doa())  # fmt: skip
        print(
            f"  (barge-in at {event.level_db:.0f} dBFS, {event.why}: heard"
            f" {reply.played.sentences} sentences"
            f"{' + ' + repr(reply.played.partial) if reply.played.partial else ''})",
            flush=True,
        )
        self._finish(reply)

    async def _drop(self, reply: Reply, outcome: str) -> None:
        """Abandon a reply that has said and done nothing."""
        reply.outcome = outcome
        self.speech.cut(reply.id)
        await self._cancel(reply)
        self.brain.abandon()
        self._finish(reply)

    async def _cancel(self, reply: Reply) -> None:
        task = reply.task
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)

    def _finish(self, reply: Reply) -> None:
        """The turn's line and record; the conductor may go on (even if the record fails)."""
        if reply.finished.is_set():
            return
        try:
            self._record(reply)
        finally:
            if reply.input.kind == "person" and reply.outcome != "merged":
                self.person_turns += 1
            reply.finished.set()

    def _record(self, reply: Reply) -> None:
        timeline = reply.input.timeline
        span = self.player.span(reply.id)
        if span is not None:
            start = self.player.playout.when(span[0])
            if start is not None:
                timeline.mark("speaker_start", start)
            end = (
                reply.barged_s
                if reply.barged_s is not None
                else self.player.playout.when(span[1], end=True)
            )
            if end is not None:
                timeline.mark("speaker_end", end)
        self.player.forget(reply.id)
        self.replies.pop(reply.id, None)
        u = reply.usage
        row: dict[str, Any] = {
            "turn": reply.id,
            "kind": reply.input.kind,
            "user": reply.input.words,
            "tag": reply.input.tag,
            "reply": " ".join(reply.sentences),
            "outcome": reply.outcome,
            "ms": timeline.ms(),
            "steps": timeline.steps(),
            "usage": {**vars(u), "calls": reply.calls},
            "tools": reply.tools,
            "echo_peak_db": round(self.ears.echo_peak_db, 1),
            "error": reply.error,
            "brain": reply.brain,
            "voice": reply.voice,
        }
        if reply.styles:
            row["styles"] = reply.styles
        if reply.bills:
            row["voice_bills"] = [vars(bill) for bill in reply.bills]
        eot = reply.input.eot
        if eot is not None:
            row["eot"] = eot
        if reply.input.fragments > 1:
            row["merged"] = reply.input.fragments
        if reply.played is not None:
            row["heard"] = {"sentences": reply.played.sentences, "partial": reply.played.partial}
        self.ears.echo_peak_db = -120.0
        if reply.outcome != "merged":
            if row["reply"]:
                print(f"pepin> {row['reply']}", flush=True)
            eot_cell = f" | merged {reply.input.fragments}" if reply.input.fragments > 1 else ""
            if eot is not None:
                p = "-" if eot["p"] is None else f"{eot['p']:.2f}"
                eot_cell += f" | eot {eot['closed_by']} p {p}, saved {eot['saved_ms']:.0f} ms"
            print(
                f"  [{reply.id} {reply.outcome}] {timeline.line()} | cache {u.cache_read}r/"
                f"{u.cache_write}w in {u.input} out {u.output} | {u.usd * 100:.4f} US cents"
                f" | {voice_cell(reply)}" + eot_cell,
                flush=True,
            )
        conversation = self.conversation
        if conversation is not None:
            conversation.turns.append(row)
            conversation.log.event("turn", self.clock(), **row)
        self.turns.append(row)

    # -- tools ---------------------------------------------------------------------------------

    async def _tool(self, name: str, args: dict[str, Any]) -> Result:
        """A tool the brain called. A drive answers when the goal server takes it; its end
        comes back as the robot's own turn."""
        if name == END_CONVERSATION:
            if self.conversation is not None:
                self.conversation.closing = self.conversation.closing or END_CONVERSATION
            return ok(note="the conversation ends after this reply")
        if name not in self.allowed:
            return fail(f"{name} is not available in this conversation")
        if name not in self.moving:
            return await asyncio.to_thread(self.tools, name, args)
        if not self.heard_since_drive:
            return fail(UNASKED_DRIVE)
        self.heard_since_drive = False
        if not self.config.report_drive_start:
            return await asyncio.to_thread(self.tools, name, args)
        loop = asyncio.get_running_loop()
        started: asyncio.Future[Result] = loop.create_future()

        def interim(payload: Result) -> None:  # on the tool's thread
            def take() -> None:
                if not started.done():
                    started.set_result(payload)

            loop.call_soon_threadsafe(take)

        def drive() -> Result:
            with progress_to(interim):
                return self.tools(name, args)

        task: asyncio.Task[Result] = asyncio.ensure_future(asyncio.to_thread(drive))
        self.drives.add(task)
        await asyncio.wait({task, started}, return_when=asyncio.FIRST_COMPLETED)
        if task.done():
            self.drives.discard(task)
            return task.result()
        task.add_done_callback(lambda t: self._drive_ended(name, args, t))
        if self.current is not None and self.current.sentences:  # the acknowledgement is said
            return {**started.result(), "note": UNDER_WAY + ACKNOWLEDGED, FOLLOW_UP: False}
        return {**started.result(), "note": UNDER_WAY}

    def _drive_ended(self, name: str, args: dict[str, Any], task: asyncio.Task[Result]) -> None:
        self.drives.discard(task)
        if task.cancelled():
            return
        error = task.exception()
        result = fail(f"{type(error).__name__}: {error}") if error else task.result()
        text, _ = render(result)
        call = ", ".join(f"{k}={v!r}" for k, v in args.items())
        report = f"{ROBOT_REPORT} the drive {name}({call}) has ended: {text}"
        print(f"  <~ {report[:200]}", flush=True)
        self.inputs.put_nowait(Input("robot", report, Timeline(self.clock())))

    # -- conversations -------------------------------------------------------------------------

    def open(self, wake_pcm: bytes) -> bool:
        """Open a conversation (the caps permitting); its log starts with ``wake_pcm``."""
        refusal = self.ledger.refusal() if self.ledger is not None else None
        if refusal is not None:
            print(f"  !! conversation refused: {refusal}", flush=True)
            return False
        wall, now = self.wall(), self.clock()
        sid = "cascade-" + time.strftime("%Y%m%d-%H%M%S", time.localtime(wall))
        log = CascadeLog(self.logs, wall, now, self.speech.voice.rate)
        log.mic.write(wake_pcm)
        self.brain.begin(self.places)
        self.conversation = Conversation(sid, wall, now, log, now)
        cfg = self.config
        log.event("open", now, model=cfg.brain.model, thinking=cfg.brain.thinking,
                  effort=cfg.brain.effort, wake=cfg.wake, places=self.places,
                  brain=self.brain_name, voice=self.voice_name,
                  flags={"barge_in": cfg.barge_in, "backchannel": cfg.backchannel,
                         "prosody_tags": cfg.prosody_tags})  # fmt: skip
        if self.ledger is not None:
            self.ledger.record(sid, wall, 0.0)
        print(
            f"  conversation open ({self.brain_name} > {self.voice_name}); log {log.dir}",
            flush=True,
        )
        self._spawn(self._refresh_places())
        return True

    async def _refresh_places(self) -> None:
        places = await asyncio.to_thread(self.read_places)
        if places:
            self.places = places

    def close(self, reason: str) -> None:
        """Close the conversation: its summary, the ledger, idle."""
        conversation = self.conversation
        if conversation is None:
            return
        self.conversation = None
        now = self.clock()
        u = conversation.usage
        person = [t for t in conversation.turns if t["kind"] == "person"]
        summary = {
            "sid": conversation.sid,
            "t": round(conversation.t_open_wall, 3),
            "dir": str(conversation.log.dir),
            "model": self.config.brain.model,
            "thinking": self.config.brain.thinking,
            "effort": self.config.brain.effort,
            "close": reason,
            "seconds": round(now - conversation.t_open, 1),
            "turns": len(person),
            "usage": vars(u),
            "voice": self.voice_name,
            "voice_usd_est": round(conversation.voice_usd, 6),
            "usd_est": round(conversation.usd, 6),
            "cad_est": round(conversation.usd * self.config.usd_to_cad, 6),
        }
        conversation.log.event("close", now, reason=reason, usd_est=round(conversation.usd, 6))
        conversation.log.close(summary)
        if self.ledger is not None:
            self.ledger.record(
                conversation.sid, conversation.t_open_wall, conversation.usd, closed=True
            )
        print(
            f"  conversation closed ({reason}): {len(person)} turns, ~{conversation.usd * 100:.3f}"
            f" US cents (estimate); log {conversation.log.dir}",
            flush=True,
        )
        self.events.emit("idle")

    async def _watch(self) -> None:
        """The state shown, the idle close, the caps, the warm cache."""
        while True:
            await asyncio.sleep(self.tick_s)
            cfg = self.config  # a switch may have replaced it
            now = self.clock()
            conversation = self.conversation
            if self.switchboard is not None:
                request = self.switchboard.control.poll()
                if request is not None:
                    self._switch(request)
            if cfg.brain.keep_warm_s > 0 and now - self.last_call_s > cfg.brain.keep_warm_s:
                self._spawn(self.prewarm("keep warm"))
            if conversation is None:
                continue
            speaking = self.player.playback() == "reply"
            busy = self.current is not None or not self.inputs.empty() or self.hearing > 0
            if (
                self.held is not None
                and not self.ears.gate.talking
                and self.hearing == 0
                and self._ears_queue.empty()
                and now - self.held_s > HOLD_RELEASE_S
            ):
                self._release()  # the person stopped without a new utterance
            if not speaking:
                state: State = "acting" if self.drives else "thinking" if busy else "listening"
                self.events.emit(state)
            if busy or speaking or self.drives:
                continue
            if now - conversation.t_open > cfg.max_session_s:
                conversation.closing = conversation.closing or "max_session_s"
            if self.ledger is not None and self.ledger.over_budget(
                conversation.sid, conversation.usd
            ):
                conversation.closing = conversation.closing or "daily budget"
            quiet_since = max(conversation.last_heard_s, self.last_reply_end_s)
            if self.ears.gate.talking:
                continue
            if conversation.closing is not None:
                self.close(conversation.closing)
            elif now - quiet_since > cfg.idle_close_s:
                self.close("idle")

    # -- switching ------------------------------------------------------------------------------

    @property
    def voice_name(self) -> str:
        """The voice's preset."""
        return self.config.voice.name

    def _switch(self, request: Request) -> None:
        """A request from the control file: each part built in the background, then swapped
        in (the brain at the next turn, the voice at its next sentence, the transcriber
        between utterances)."""
        asked = (("brain", request.brain), ("voice", request.voice),
                 ("transcriber", request.transcriber))  # fmt: skip
        print("  (switch asked: " + ", ".join(f"{k} {v}" for k, v in asked if v) + ")", flush=True)
        if request.brain:
            self._spawn(self._switch_brain(request.id, request.brain))
        if request.voice:
            self._switch_voice(request.id, request.voice)
        if request.transcriber and self.switchboard is not None:
            self.switchboard.switch_transcriber(request.id, request.transcriber)

    async def _switch_brain(self, request: str, name: str) -> None:
        board = self.switchboard
        assert board is not None
        start = self.clock()
        board.answer(request, "brain", name, "loading")
        try:
            setup = await asyncio.to_thread(board.brain_setup, name)
            setup.brain.begin(self.places)
            warm = await setup.brain.prewarm()
        except Exception as error:
            why = f"{type(error).__name__}: {error}"[:300]
            board.answer(request, "brain", name, "failed", error=why)
            print(f"  !! switch to brain {name}: {why}", flush=True)
            return
        current = self.current
        while current is not None and not current.finished.is_set():
            await current.finished.wait()  # the turn in flight ends with the brain it began with
            current = self.current
        old = self.brain
        self.brain = setup.brain
        self.allowed, self.moving = setup.allowed, setup.moving
        self.config = replace(setup.config, voice=self.config.voice)
        self.ledger = self.paid_ledger if setup.paid else None
        self.brain_name = board.brain = setup.name
        self.last_call_s = self.clock()
        ms = round((self.clock() - start) * 1000)
        model = setup.config.brain.model if setup.paid else setup.name
        self._event("switch", brain=setup.name, model=model, ms=ms)
        board.answer(request, "brain", setup.name, "applied", ms=ms, model=model,
                     warm_ms=warm.ms if warm is not None else None)  # fmt: skip
        afresh = "; the conversation goes on with it afresh" if self.conversation else ""
        print(f"  (brain: {setup.name} ({model}), ready in {ms} ms{afresh})", flush=True)
        close = getattr(old, "aclose", None)
        if close is not None:
            with contextlib.suppress(Exception):
                await close()

    def _switch_voice(self, request: str, name: str) -> None:
        board = self.switchboard
        assert board is not None
        start = self.clock()
        try:
            config = board.voice_config(name)
            voice = board.build_voice(config)
        except Exception as error:
            why = f"{type(error).__name__}: {error}"[:300]
            board.answer(request, "voice", name, "failed", error=why)
            print(f"  !! switch to voice {name}: {why}", flush=True)
            return
        board.answer(request, "voice", config.name, "loading")
        loop = asyncio.get_running_loop()

        def ready(error: BaseException | None) -> None:  # on the voice's loading thread
            with contextlib.suppress(RuntimeError):
                loop.call_soon_threadsafe(self._voice_ready, request, config, error, start)

        self.speech.swap(voice, ready)

    def _voice_ready(
        self, request: str, config: VoiceConfig, error: BaseException | None, start: float
    ) -> None:
        board = self.switchboard
        assert board is not None
        ms = round((self.clock() - start) * 1000)
        if error is not None:
            why = f"{type(error).__name__}: {error}"[:300]
            board.answer(request, "voice", config.name, "failed", error=why, ms=ms)
            print(f"  !! switch to voice {config.name}: {why}", flush=True)
            return
        self.config = replace(self.config, voice=config)
        board.voice = config.name
        board.keep_voice(config)
        self._event("switch", voice=config.name, ms=ms)
        board.answer(request, "voice", config.name, "applied", ms=ms, model=config.model)
        print(f"  (voice: {config.name}, ready in {ms} ms)", flush=True)

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _doa(self) -> dict[str, Any] | None:
        """The array's latest direction of arrival, when the mic is the board's link."""
        latest = getattr(self.mic, "latest_doa", None)
        reading = latest() if callable(latest) else None
        if reading is None:
            return None
        return {"deg": reading.deg, "speech": reading.speech, "board_s": round(reading.stamp_s, 3)}

    def _event(self, kind: str, **fields: Any) -> None:
        conversation = self.conversation
        if conversation is not None:
            conversation.log.event(kind, self.clock(), **fields)


def voice_cell(reply: Reply) -> str:
    """``haiku > gemini-lite 520 ms 0.0310 US cents`` (the brain's preset, the voice's, a paid
    voice's first audio after its first call and its price, the sentences a fallback said)."""
    cell = f"{reply.brain} > {reply.voice}"
    styles = [style for style in reply.styles if style]
    if styles:
        cell += " {" + ", ".join(styles) + "}"
    bills = reply.bills
    if not bills:
        return cell
    first = next((b.first_audio_ms for b in bills if b.first_audio_ms is not None), None)
    if first is not None:
        cell += f" {first} ms"
    cell += f" {sum(b.usd for b in bills) * 100:.4f} US cents"
    stood_in = [b for b in bills if b.fallback is not None]
    if stood_in:
        cell += f", {len(stood_in)} of {len(bills)} by {stood_in[0].engine}"
    return cell


def eot_of(event: EarEvent) -> dict[str, Any] | None:
    """The end-of-turn model's verdict carried by a segment event (None without a model)."""
    if event.closed_by is None:
        return None
    return {
        "closed_by": event.closed_by,
        "p": None if event.p is None else round(event.p, 3),
        "model_ms": None if event.model_ms is None else round(event.model_ms, 1),
        "saved_ms": None if event.saved_ms is None else round(event.saved_ms),
    }
