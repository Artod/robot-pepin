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
brain's cache tokens and price, and logs it with everything else (:mod:`pepin.voice_cascade.log`).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol

from pepin.tools.registry import Result, fail, ok, progress_to, render
from pepin.voice_cascade.brain import END_CONVERSATION
from pepin.voice_cascade.config import CascadeConfig
from pepin.voice_cascade.ears import EarEvent, Ears
from pepin.voice_cascade.log import CascadeLog, append_line, day_dir
from pepin.voice_cascade.player import Player
from pepin.voice_cascade.speech import SpeechJob, SpeechWorker
from pepin.voice_cascade.stages import (
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
)
from pepin.voice_cascade.timeline import Timeline
from pepin.voice_live.budget import Ledger
from pepin.voice_live.events import Events, State
from pepin.voice_live.sources import FrameSink
from pepin.voice_live.wake import Segment, is_goodbye, is_hallucination, is_stop, is_wake

ToolRunner = Callable[[str, dict[str, Any]], Result]
ROBOT_REPORT = "[robot]"
UNDER_WAY = (
    f"under way: a {ROBOT_REPORT} message reports the drive's end; do not call it again meanwhile"
)
UNASKED_DRIVE = (
    "not driving: no one has asked for another drive since the last one; never repeat a drive"
    " unless a person asks again"
)
END_WAIT_S = 10.0  # a robot report waits this long at most for the robot's line to end
PLAYOUT_WAIT_S = 120.0  # a reply's audio is waited for this long at most


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
    generated: bool = False  # the brain's stream ended
    spoken: asyncio.Event = field(default_factory=asyncio.Event)  # the voice did its last job
    finished: asyncio.Event = field(default_factory=asyncio.Event)
    outcome: str = ""  # committed, interrupted, merged, error
    played: Played | None = None
    barged_s: float | None = None
    error: str | None = None


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
    closing: str | None = None


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
    ) -> None:
        """``tools`` runs a registry tool by name; ``allowed`` names those the brain may call,
        ``moving`` those that drive; ``ledger`` None spends nothing worth recording (a fake
        brain)."""
        self.config = config
        self.mic, self.player, self.transcriber, self.tagger = mic, player, transcriber, tagger
        self.brain, self.speech = brain, speech
        self.tools, self.allowed, self.moving, self.halt = tools, allowed, moving, halt
        self.events, self.logs, self.ledger = events, logs, ledger
        self.clock, self.wall, self.tick_s = clock, wall, tick_s
        self.ears = Ears(config, player.playback)
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
        self._ears_queue: asyncio.Queue[tuple[EarEvent, float | None]] = asyncio.Queue()
        self._order = asyncio.Lock()
        self._whisper = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="whisper")
        self._praat = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="prosody")
        self._loop: asyncio.AbstractEventLoop | None = None
        self._background: set[asyncio.Task[Any]] = set()
        speech.on_audio = self._on_audio
        speech.on_done = self._on_spoken

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
            stop_player.set()
            self._whisper.shutdown(wait=False)
            self._praat.shutdown(wait=False)

    def quiet(self) -> bool:
        """Nothing in flight: no utterance being read, no reply, no queued input, no audio."""
        return (
            self.hearing == 0
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
            conversation.log.voice.write(pcm)
        reply = self.replies.get(job.turn)
        if reply is not None and job.kind == "reply":
            reply.audio = True
            reply.input.timeline.mark("tts_first_audio", t)

    def _on_spoken(self, job: SpeechJob, t: float) -> None:
        reply = self.replies.get(job.turn)
        if reply is not None and job.kind == "end" and self._loop is not None:
            with contextlib.suppress(RuntimeError):
                self._loop.call_soon_threadsafe(reply.spoken.set)

    # -- the ears ------------------------------------------------------------------------------

    async def _listen(self) -> None:
        while True:
            event, played = await self._ears_queue.get()
            if event.kind == "start":
                if self.conversation is not None:
                    self.conversation.last_heard_s = event.t
            elif event.kind == "pause":
                self._backchannel()
            elif event.kind == "barge_in":
                await self._barge_in(event, played if played is not None else 0.0)
            elif event.kind == "segment" and event.segment is not None:
                self.hearing += 1
                self._spawn(self._hear(event.segment, event.t))

    def _backchannel(self) -> None:
        cfg = self.config
        if not (cfg.backchannel and cfg.backchannel_lines and self.conversation is not None):
            return
        if self.player.speaking():
            return
        line = self.rng.choice(cfg.backchannel_lines)
        self.speech.say(SpeechJob(self.last_turn, -1, line, "backchannel"))
        self._event("backchannel", text=line)

    async def _hear(self, segment: Segment, closed_s: float) -> None:
        try:
            async with self._order:  # utterances are answered in the order they were said
                await self._read(segment, closed_s)
        finally:
            self.hearing -= 1

    async def _read(self, segment: Segment, closed_s: float) -> None:
        loop = asyncio.get_running_loop()
        timeline = Timeline(segment.ended_s)
        timeline.mark("closed", closed_s)
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
        voice_db = round(self.ears.voice_peak_db, 1)
        if self.conversation is None:
            woke = not noise and (self.config.wake == "always" or is_wake(text))
            print(
                f"heard (idle, {whisper_ms} ms): {text!r}"
                + ("  -> WAKE" if woke else "  (Whisper's noise line)" if noise else ""),
                flush=True,
            )
            row = {"t": round(self.wall(), 3), "text": text, "wake": woke, "noise": noise}
            append_line(day_dir(self.logs, self.wall()) / "cascade_heard.jsonl", row)
            if not woke or not self.open(segment.pcm):
                return
        conversation = self.conversation
        assert conversation is not None
        conversation.log.event(
            "heard", read_s, text=text, noise=noise, whisper_ms=whisper_ms,
            seconds=round(segment.duration_s, 2), voice_peak_db=voice_db,
        )  # fmt: skip
        if noise:
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
        words = f"{tag} {text}" if tag else text
        current = self.current
        if current is not None and not current.audio and not current.acted:
            # the person went on before the robot said or did anything: one turn of both
            await self._drop(current, "merged")
            words = f"{current.input.text} {words}"
        self.inputs.put_nowait(
            Input("person", words, timeline, text, tag, goodbye=is_goodbye(text))
        )

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
            reply = Reply(self.last_turn, item)
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
                self.speech.say(SpeechJob(reply.id, 0, self.config.error_line))
                self.speech.say(SpeechJob(reply.id, 1, kind="end"))
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
            self.speech.say(SpeechJob(reply.id, index, event.text))
            if log is not None:
                log.event("sentence", now, turn=reply.id, index=index, text=event.text)
        elif isinstance(event, ToolUse):
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
        reply.played = self.player.heard(reply.id, played, list(reply.sentences))
        reply.outcome = "interrupted"
        await self._cancel(reply)
        self.brain.interrupt(reply.played)
        print(
            f"  (barge-in at {event.level_db:.0f} dBFS: heard {reply.played.sentences} sentences"
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
        }
        if reply.played is not None:
            row["heard"] = {"sentences": reply.played.sentences, "partial": reply.played.partial}
        self.ears.echo_peak_db = -120.0
        if reply.outcome != "merged":
            if row["reply"]:
                print(f"pepin> {row['reply']}", flush=True)
            print(
                f"  [{reply.id} {reply.outcome}] {timeline.line()} | cache {u.cache_read}r/"
                f"{u.cache_write}w in {u.input} out {u.output} | {u.usd * 100:.4f} US cents",
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
                  flags={"barge_in": cfg.barge_in, "backchannel": cfg.backchannel,
                         "prosody_tags": cfg.prosody_tags})  # fmt: skip
        if self.ledger is not None:
            self.ledger.record(sid, wall, 0.0)
        print(f"  conversation open ({cfg.brain.model}); log {log.dir}", flush=True)
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
            "usd_est": round(u.usd, 6),
            "cad_est": round(u.usd * self.config.usd_to_cad, 6),
        }
        conversation.log.event("close", now, reason=reason, usd_est=round(u.usd, 6))
        conversation.log.close(summary)
        if self.ledger is not None:
            self.ledger.record(conversation.sid, conversation.t_open_wall, u.usd, closed=True)
        print(
            f"  conversation closed ({reason}): {len(person)} turns, ~{u.usd * 100:.3f} US cents"
            f" (estimate); log {conversation.log.dir}",
            flush=True,
        )
        self.events.emit("idle")

    async def _watch(self) -> None:
        """The state shown, the idle close, the caps, the warm cache."""
        cfg = self.config
        while True:
            await asyncio.sleep(self.tick_s)
            now = self.clock()
            conversation = self.conversation
            if cfg.brain.keep_warm_s > 0 and now - self.last_call_s > cfg.brain.keep_warm_s:
                self._spawn(self.prewarm("keep warm"))
            if conversation is None:
                continue
            speaking = self.player.playback() == "reply"
            busy = self.current is not None or not self.inputs.empty() or self.hearing > 0
            if not speaking:
                state: State = "acting" if self.drives else "thinking" if busy else "listening"
                self.events.emit(state)
            if busy or speaking or self.drives:
                continue
            if now - conversation.t_open > cfg.max_session_s:
                conversation.closing = conversation.closing or "max_session_s"
            if self.ledger is not None and self.ledger.over_budget(
                conversation.sid, conversation.usage.usd
            ):
                conversation.closing = conversation.closing or "daily budget"
            quiet_since = max(conversation.last_heard_s, self.last_reply_end_s)
            if self.ears.gate.talking:
                continue
            if conversation.closing is not None:
                self.close(conversation.closing)
            elif now - quiet_since > cfg.idle_close_s:
                self.close("idle")

    def _spawn(self, coroutine: Any) -> None:
        task = asyncio.create_task(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _event(self, kind: str, **fields: Any) -> None:
        conversation = self.conversation
        if conversation is not None:
            conversation.log.event(kind, self.clock(), **fields)
