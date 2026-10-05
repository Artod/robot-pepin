"""One conversation: a Live session opened by the wake gate, streamed until a goodbye, silence,
or a cap closes it.

The session opens with the places book in its instructions (no list_places round trip before a
drive). The mic goes to Live only while the session is open and the robot is not speaking
(unless ``barge_in``) and not driving: during a drive (``mic_while_driving`` gated) the local
gate listens instead, and only speech naming the robot or saying stop reaches Live — a stop also
halts the robot at once, before any cloud answers. The model's audio is resampled to the board's
rate and paced to its speaker as it streams; tool calls run on worker threads through the robot's
registry while the conversation goes on, and their results go back. A drive answers its call
when the goal server takes it (``status: driving``: the model says its acknowledgement) and
reports its end as the robot's own text turn (the model says its arrival line). A drive the model
starts with nothing heard since the last one is refused. The session closes after
``idle_close_s`` without the person's speech (the server's VAD and transcript, not room noise),
``idle_after_drive_s`` after a drive's last line. Every session leaves
``data/voice/<day>/live_<HHMMSS>/`` (in.wav, out.wav, events.jsonl) and one summary line in
``data/voice/<day>/live.jsonl``: transcripts, tool calls, latencies, the estimated cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import time
from collections.abc import Callable, Coroutine
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pepin.tools.registry import Result, fail, progress_to, render
from pepin.voice_live.audio import LIVE_IN_RATE, LIVE_OUT_RATE, Pacer, Resampler, WavWriter
from pepin.voice_live.budget import Ledger, SessionCost
from pepin.voice_live.config import LiveConfig
from pepin.voice_live.events import Events
from pepin.voice_live.live import (
    END_CONVERSATION,
    ROBOT_REPORT,
    LiveConnector,
    LiveMessage,
    LiveSession,
    ToolCall,
    ToolResponse,
    system_instruction,
)
from pepin.voice_live.wake import (
    Segment,
    Segmenter,
    Transcriber,
    is_goodbye,
    is_hallucination,
    is_stop,
    is_wake,
)

logger = logging.getLogger(__name__)

ToolRunner = Callable[[str, dict[str, Any]], Result]
PREROLL_CHUNK = 3200  # bytes: the wake segment goes to Live in 100 ms pieces, as fast as it can
TICK_S = 0.1
AWAIT_REPLY_S = 8.0  # "thinking" gives up after this long without an answer (it may not answer)
DRAIN_S = 12.0  # the last words are allowed this long to play after the session closed
PROGRESS_S = 5.0  # the ledger hears the running estimate this often (a crash loses less)
END_FALLBACK_S = 3.0  # end_conversation without a turn end after it: closed after this long
# A drive's two reports (scratch/voice_1005/ack_probe.py and the --fake-robot runs, 2026-10-05).
# The start answers the call itself, WHEN_IDLE: the model, idle after its call, says its
# acknowledgement 0.5-1.0 s later (6 of 6). A model that spoke before its call (2 of 8) took that
# answer for a cue to speak again and said its arrival line early, so then it is SILENT. The end
# goes as a [robot] text turn, which starts a generation (the arrival line 0.55 s after it in the
# probe); the call's own final response after an interim (will_continue) "may" start one, and in
# 1 of 7 runs it did not (in another it took 10 s as WHEN_IDLE). The report waits for the robot's
# line in progress to end: a turn_complete text interrupts a generation.
DRIVE_STARTED = "WHEN_IDLE"
DRIVE_STARTED_SAID = "SILENT"
END_WAIT_S = 10.0  # a drive's end waits this long at most for the robot's line to end
# A drive the model starts again with nothing heard (2026-10-05: go_to home three times in a row
# after arriving home, one of them a 23 s drive; and in the measurements, a second go_to right
# after the interim or the result) is answered SILENT: the model learns it, and says nothing.
UNASKED_DRIVE = (
    "not driving: no one has asked for another drive since the last one (its report is above);"
    " never repeat a drive unless a person asks again"
)
UNDER_WAY = (
    f"under way: a {ROBOT_REPORT} message reports the drive's end; do not call it again meanwhile"
)


@dataclass(frozen=True)
class Frame:
    """20 ms of the mic and when it reached the laptop (monotonic)."""

    pcm: bytes
    t: float


@dataclass(frozen=True)
class Wake:
    """What opened the session: the segment heard, its transcript, and when (monotonic) the
    speech ended, the segment was cut, and Whisper had read it."""

    pcm: bytes
    text: str
    ended_s: float
    cut_s: float
    detected_s: float


def scrub(text: str) -> str:
    """``text`` with the API key blanked, should an error ever quote it."""
    key = os.environ.get("GEMINI_API_KEY", "")
    return text.replace(key, "***") if len(key) >= 8 else text


class ResumeStore:
    """The last resumable handle, when its session ended, and how much context it carries;
    kept in a file so a restart resumes the same conversation."""

    def __init__(self, path: Path, valid_s: float, clock: Callable[[], float] = time.time) -> None:
        """``valid_s``: how long after its session ended a handle is still offered."""
        self.path, self.valid_s, self._clock = path, valid_s, clock

    def get(self) -> tuple[str | None, float]:
        """The handle and its context in seconds of audio, or (None, 0) when none is valid."""
        try:
            data = json.loads(self.path.read_text())
            if self._clock() - float(data["saved"]) < self.valid_s and data.get("handle"):
                return str(data["handle"]), float(data.get("context_s", 0.0))
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None, 0.0

    def save(self, handle: str | None, context_s: float) -> None:
        """Keep ``handle`` (None forgets it)."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        row = {"handle": handle, "saved": self._clock(), "context_s": round(context_s, 1)}
        self.path.write_text(json.dumps(row))


class SessionLog:
    """The session's directory: both audio streams as WAV and events as JSON lines."""

    def __init__(self, root: Path, wall: float, t0: float) -> None:
        """Under ``root/<day>/live_<HHMMSS>``; ``t0`` is the monotonic zero of the events."""
        stamp = time.localtime(wall)
        self.day_dir = root / time.strftime("%Y%m%d", stamp)
        self.dir = self.day_dir / f"live_{time.strftime('%H%M%S', stamp)}"
        suffix = 1
        while self.dir.exists():
            suffix += 1
            self.dir = self.day_dir / f"live_{time.strftime('%H%M%S', stamp)}_{suffix}"
        self.dir.mkdir(parents=True)
        self.t0 = t0
        self.mic = WavWriter(self.dir / "in.wav", LIVE_IN_RATE)
        self.model = WavWriter(self.dir / "out.wav", LIVE_OUT_RATE)
        self._events = (self.dir / "events.jsonl").open("a")

    def event(self, kind: str, now: float, **fields: Any) -> None:
        """One line: what happened, seconds since the wake ended, the details."""
        row = {"event": kind, "t": round(now - self.t0, 3), **fields}
        self._events.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
        self._events.flush()

    def close(self, summary: dict[str, Any]) -> None:
        """Finish the WAVs, write the summary beside the events and into the day's index."""
        self.mic.close()
        self.model.close()
        self._events.close()
        line = json.dumps(summary, ensure_ascii=False, default=str)
        (self.dir / "summary.json").write_text(line + "\n")
        with (self.day_dir / "live.jsonl").open("a") as index:
            index.write(line + "\n")


@dataclass
class _Turn:
    user: str = ""
    model: str = ""
    user_done_s: float | None = None  # the last voiced frame of the user's turn
    first_audio_s: float | None = None
    tools: list[str] = field(default_factory=list)


class Conversation:
    """Runs sessions, one per :meth:`run`, against a :class:`LiveConnector`."""

    def __init__(
        self,
        live: LiveConnector,
        config: LiveConfig,
        *,
        frames: asyncio.Queue[Frame],
        pacer: Pacer,
        tools: ToolRunner,
        moving: set[str],
        halt: Callable[[], str],
        events: Events,
        logs: Path,
        ledger: Ledger | None,
        resume: ResumeStore,
        transcriber: Transcriber | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        tick_s: float = TICK_S,
    ) -> None:
        """``tools`` runs a tool by name (the registry bound to the robot); ``moving`` names the
        tools that drive (non-blocking; halted if the model cancels them); ``ledger`` None is
        the dry run, which spends nothing and records nothing; ``transcriber`` (Whisper) is the
        ear that listens while the robot drives with the mic withheld from Live (without it the
        mic streams as at rest)."""
        self.live, self.config = live, config
        self.frames, self.pacer = frames, pacer
        self.tools, self.moving, self.halt = tools, moving, halt
        self.events, self.logs, self.ledger, self.resume = events, logs, ledger, resume
        self.transcriber = transcriber
        self.clock, self.wall, self.tick_s = clock, wall, tick_s
        self.moved = False  # a moving tool was called in this process (Ctrl-C then halts)

    async def run(self, wake: Wake) -> dict[str, Any]:
        """One session from the wake to its close; its summary."""
        state = _Session(self, wake)
        try:
            await state.run()
        finally:
            state.finish_log()
        return state.summary


class _Session:
    """The state of one session (see :class:`Conversation`)."""

    def __init__(self, owner: Conversation, wake: Wake) -> None:
        self.o = owner
        self.cfg = owner.config
        self.wake = wake
        self.clock = owner.clock
        self.t_open_wall = owner.wall()
        self.sid = time.strftime("%Y%m%d-%H%M%S", time.localtime(self.t_open_wall))
        self.log = SessionLog(owner.logs, self.t_open_wall, wake.ended_s)
        self.cost = SessionCost(self.cfg.prices)
        self.resampler = Resampler(LIVE_OUT_RATE, owner.pacer.rate)
        owner.pacer.first_play_s = None
        self.handle: str | None = None
        self.resumed = False
        self.closing: str | None = None
        self.end_requested_s: float | None = None
        self.connected_s: float | None = None
        self.first_audio_s: float | None = None
        self.last_user_voice_s = wake.ended_s  # the local gate's: the reply latency, "thinking"
        self.last_heard_s = wake.ended_s  # the server's: its VAD or its transcript of the person
        self.heard_since_drive = True  # the wake itself asked for whatever comes
        self.drive_ended_s: float | None = None  # the last drive's final result went back
        self.last_activity_s = wake.cut_s
        self.last_speaking_s = 0.0
        self.awaiting_s: float | None = wake.cut_s  # the wake segment itself awaits an answer
        self.turn = _Turn(user_done_s=wake.ended_s)
        self.turns: list[dict[str, Any]] = []
        self.tool_log: list[dict[str, Any]] = []
        self.running: dict[str, ToolCall] = {}
        self.cancelled: set[str] = set()
        self.answered: dict[str, asyncio.Task[None]] = {}  # drives answered when they set off
        self.tool_tasks: set[asyncio.Task[None]] = set()
        self.withheld_s = 0.0  # mic seconds kept from Live while driving
        self.last_model_audio_s: float | None = None
        self.asked_s: dict[str, float] = {}  # a drive's call: when the person last spoke
        self.last_response_s: float | None = None  # a tool response went back
        self.heard_while_driving: list[dict[str, Any]] = []
        self.refused_drives = 0
        self._whisper = asyncio.Lock()
        self.last_progress_s = 0.0
        self.error: str | None = None
        self.summary: dict[str, Any] = {}

    # -- the session's life --------------------------------------------------------------------

    async def run(self) -> None:
        o = self.o
        handle, carried_s = o.resume.get()
        self.cost.carried_s = carried_s
        self.log.event("wake", self.clock(), text=self.wake.text, wake_s=len(self.wake.pcm) / 32e3)
        if o.ledger is not None:
            o.ledger.record(self.sid, self.t_open_wall, 0.0)
        o.events.emit("thinking")
        places = await asyncio.to_thread(self.places)
        system = system_instruction(self.cfg, places)
        self.log.event("places", self.clock(), places=places)
        for attempt_handle in [handle, None] if handle else [None]:
            try:
                async with o.live.connect(attempt_handle, system) as session:
                    self.resumed = attempt_handle is not None
                    self.connected_s = self.last_activity_s = self.clock()
                    self.log.event("connected", self.connected_s, resumed=self.resumed)
                    print(
                        f"  live: {'resumed' if self.resumed else 'new'} session"
                        f" ({(self.connected_s - self.wake.ended_s) * 1000:.0f} ms after the"
                        " wake)",
                        flush=True,
                    )
                    await self.converse(session)
                break
            except Exception as error:  # the connection failed or broke: logged, not fatal
                why = scrub(f"{type(error).__name__}: {error}")[:300]
                self.log.event("error", self.clock(), error=why, connected=bool(self.connected_s))
                print(f"  !! live: {why}", flush=True)
                self.error = why
                if self.connected_s is not None:
                    break  # it broke mid-session: no second session for the same wake
                self.cost.carried_s = 0.0
        if self.closing is None:
            self.closing = "error" if self.error else "server closed"
        for task in self.tool_tasks:
            task.cancel()
        o.pacer.end()  # whatever is queued plays out, a turn cut short included
        deadline = self.clock() + DRAIN_S
        while o.pacer.speaking() and self.clock() < deadline:
            await asyncio.sleep(self.o.tick_s)
        o.events.emit("idle")

    def places(self) -> list[str] | None:
        """The names in the places book (list_places through the registry), or None when it
        cannot be read: the model is then told to ask list_places."""
        result = self.o.tools("list_places", {})
        rows = result.get("places") if result.get("ok") else None
        if not isinstance(rows, list):
            return None
        names = [str(row["name"]) for row in rows if isinstance(row, dict) and "name" in row]
        return names or None

    async def converse(self, session: LiveSession) -> None:
        for i in range(0, len(self.wake.pcm), PREROLL_CHUNK):
            await self.send_mic(session, self.wake.pcm[i : i + PREROLL_CHUNK])
        tasks = [
            asyncio.create_task(self.forward_mic(session), name="mic"),
            asyncio.create_task(self.receive(session), name="receive"),
            asyncio.create_task(self.watch(), name="watch"),
        ]
        try:
            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                error = task.exception()
                if error is not None:
                    raise error
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

    async def send_mic(self, session: LiveSession, pcm: bytes) -> None:
        await session.send_audio(pcm)
        self.log.mic.write(pcm)
        self.cost.audio_in_s += len(pcm) / 2 / LIVE_IN_RATE

    def driving(self) -> bool:
        """A drive is under way and the mic is withheld from Live (``mic_while_driving``
        gated, with an ear to listen in its place)."""
        return (
            self.cfg.mic_while_driving == "gated"
            and self.o.transcriber is not None
            and any(call.name in self.o.moving for call in self.running.values())
        )

    async def forward_mic(self, session: LiveSession) -> None:
        """Mic frames to Live while the robot is silent and still; the local gate marks the
        user's voice (for the reply latency) and the end of their turn (for "thinking"). While
        it drives, the frames go to :meth:`listen_while_driving` instead."""
        ears = Segmenter(silence_s=0.5, max_s=60.0)
        gate = Segmenter(silence_s=self.cfg.wake_silence_s, max_s=self.cfg.wake_max_s)
        gated = False
        while True:
            frame = await self.o.frames.get()
            if self.o.pacer.speaking() and not self.cfg.barge_in:
                ears.reset()
                gate.reset()
                continue
            driving = self.driving()
            if driving != gated:
                gated = driving
                ears.reset()
                gate.reset()  # its floor is kept: it has followed the room all along
            segment = gate.feed(frame.pcm, frame.t)
            if driving:
                self.withheld_s += len(frame.pcm) / 2 / LIVE_IN_RATE
                if segment is not None:
                    self.spawn(self.listen_while_driving(session, segment))
                continue
            if ears.feed(frame.pcm, frame.t) is not None:
                self.awaiting_s = self.clock()
            if ears.talking:
                self.last_user_voice_s = max(self.last_user_voice_s, ears.last_voice_s)
                self.turn.user_done_s = self.last_user_voice_s
                self.awaiting_s = None
            await self.send_mic(session, frame.pcm)

    async def listen_while_driving(self, session: LiveSession, segment: Segment) -> None:
        """A segment heard while driving: Whisper reads it; a stop halts the robot here and
        now; a stop or the robot's name sends the segment to Live (and ends the turn there),
        anything else stays in the room."""
        transcriber = self.o.transcriber
        assert transcriber is not None
        async with self._whisper:
            text = await asyncio.to_thread(transcriber.transcribe, segment.pcm)
        noise = is_hallucination(text)
        stop = not noise and is_stop(text)
        named = not noise and is_wake(text)
        row = {"text": text, "stop": stop, "named": named, "noise": noise}
        self.heard_while_driving.append(row)
        self.log.event("heard_while_driving", self.clock(), **row)
        print(f"  (driving) heard: {text!r}" + ("  -> STOP" if stop else ""), flush=True)
        if stop:
            answer = await asyncio.to_thread(self.o.halt)
            self.log.event("halt", self.clock(), answer=answer, why="stop heard while driving")
            print(f"  (stop heard: {answer})", flush=True)
        if stop or named:
            for i in range(0, len(segment.pcm), PREROLL_CHUNK):
                await self.send_mic(session, segment.pcm[i : i + PREROLL_CHUNK])
            await session.end_audio()

    def spawn(self, coroutine: Coroutine[Any, Any, None]) -> asyncio.Task[None]:
        """Run ``coroutine`` beside the conversation; cancelled when the session ends."""
        task = asyncio.create_task(coroutine)
        self.tool_tasks.add(task)
        task.add_done_callback(self.tool_tasks.discard)
        return task

    async def receive(self, session: LiveSession) -> None:
        async for message in session.messages():
            await self.on_message(session, message)
        if self.closing is None:
            self.closing = "server closed"

    async def watch(self) -> None:
        """Caps, the idle timer, the state shown, the ledger's progress; returns to close."""
        o = self.o
        while True:
            await asyncio.sleep(self.o.tick_s)
            now = self.clock()
            speaking = o.pacer.speaking()
            if speaking:
                self.last_speaking_s = now
            assert self.connected_s is not None
            if now - self.connected_s > self.cfg.max_session_s:
                self.closing = self.closing or "max_session_s"
            quiet_since = max(self.last_heard_s, self.last_activity_s, self.last_speaking_s)
            idle_s = self.cfg.idle_close_s
            if self.drive_ended_s is not None and self.drive_ended_s >= self.last_heard_s:
                idle_s = min(idle_s, self.cfg.idle_after_drive_s)  # its last line was said
            if not self.running and not speaking and now - quiet_since > idle_s:
                self.closing = self.closing or "idle"
            if (
                self.end_requested_s is not None
                and not speaking
                and now - self.end_requested_s > END_FALLBACK_S
            ):
                self.closing = self.closing or END_CONVERSATION
            if self.awaiting_s is not None and now - self.awaiting_s > AWAIT_REPLY_S:
                self.awaiting_s = None
            if o.ledger is not None and now - self.last_progress_s > PROGRESS_S:
                self.last_progress_s = now
                o.ledger.record(self.sid, self.t_open_wall, self.cost.usd)
                self.check_budget()  # the audio estimate grows between usage reports
            if self.closing is not None and not speaking:
                return
            if not speaking:
                busy = self.running or self.awaiting_s is not None
                acting = any(call.name in o.moving for call in self.running.values())
                o.events.emit("acting" if acting else "thinking" if busy else "listening")

    # -- server messages -----------------------------------------------------------------------

    async def on_message(self, session: LiveSession, m: LiveMessage) -> None:
        now = self.clock()
        if m.audio:
            self.on_audio(m.audio, now)
        if m.input_text:
            self.turn.user += m.input_text
            if m.input_text.strip():
                self.last_heard_s = now
                self.heard_since_drive = True
        if m.activity is not None:
            self.last_heard_s = now
        if m.output_text:
            self.turn.model += m.output_text
        if m.raw and not m.audio:
            self.log.event("server", now, message=m.raw)
        for call in m.tool_calls:
            await self.on_tool_call(session, call, now)
        for call_id in m.cancelled:
            await self.on_cancel(call_id, now)
        if m.interrupted:
            self.o.pacer.flush()
            self.log.event("interrupted", now)
            print("  (interrupted)", flush=True)
        if m.turn_complete:
            self.on_turn_complete(now)
        if m.handle:
            self.handle = m.handle
            self.o.resume.save(self.handle, self.context_s)
        if m.go_away_s is not None:
            self.log.event("go_away", now, time_left_s=m.go_away_s)
            self.closing = self.closing or "go_away"
        if m.usage is not None:
            self.cost.usage.add(m.usage)
            self.check_budget()

    def check_budget(self) -> None:
        """Close when today's estimate with this session reaches the daily budget."""
        if self.o.ledger is not None and self.o.ledger.over_budget(self.sid, self.cost.usd):
            self.closing = self.closing or "daily budget"

    def on_audio(self, pcm: bytes, now: float) -> None:
        if self.first_audio_s is None:
            self.first_audio_s = now
            self.log.event("first_audio", now)
        if self.turn.first_audio_s is None:
            self.turn.first_audio_s = now
            self.log.event("turn_audio", now, after_response_ms=self.since_response_ms(now))
        self.awaiting_s = None
        self.last_activity_s = self.last_model_audio_s = now
        self.log.model.write(pcm)
        self.cost.audio_out_s += len(pcm) / 2 / LIVE_OUT_RATE
        self.o.pacer.feed(self.resampler.process(pcm))

    def since_response_ms(self, now: float) -> int | None:
        """Milliseconds since the last tool response went back (what a drive's line answers)."""
        if self.last_response_s is None:
            return None
        return round((now - self.last_response_s) * 1000)

    def on_turn_complete(self, now: float) -> None:
        self.o.pacer.end()
        turn = self.turn
        reply_ms = (
            round((turn.first_audio_s - turn.user_done_s) * 1000)
            if turn.first_audio_s is not None and turn.user_done_s is not None
            else None
        )
        row = {
            "user": turn.user.strip(),
            "model": turn.model.strip(),
            "reply_ms": reply_ms,
            "tools": turn.tools,
            "first_audio_t": round(turn.first_audio_s - self.log.t0, 3)
            if turn.first_audio_s is not None
            else None,
        }
        self.turns.append(row)
        self.log.event("turn", now, **row)
        if row["user"]:
            print(f"heard> {row['user']}", flush=True)
        if row["model"]:
            print(f"pepin> {row['model']}  ({reply_ms} ms)", flush=True)
        if is_goodbye(turn.user):
            self.closing = self.closing or "goodbye"
        if self.end_requested_s is not None:
            self.closing = self.closing or END_CONVERSATION
        carried = turn.user_done_s if turn.first_audio_s is None and turn.tools else None
        self.turn = _Turn(user_done_s=carried)  # a silent tool call's answer still answers them
        self.last_activity_s = now

    async def on_tool_call(self, session: LiveSession, call: ToolCall, now: float) -> None:
        self.turn.tools.append(call.name)
        self.log.event("tool_call", now, id=call.id, name=call.name, args=call.args)
        print(f"  -> {call.name}({call.args})", flush=True)
        if call.name == END_CONVERSATION:
            self.end_requested_s = now
            await session.send_tool_responses([ToolResponse(call.id, call.name, {"ok": True})])
            return
        if call.name in self.o.moving:
            if not self.heard_since_drive:
                self.refused_drives += 1
                self.log.event("drive_refused", now, id=call.id, name=call.name, args=call.args)
                print(f"  !! {call.name} refused: nothing heard since the last drive", flush=True)
                response = ToolResponse(call.id, call.name, fail(UNASKED_DRIVE), "SILENT")
                await session.send_tool_responses([response])
                return
            self.o.moved = True
            self.asked_s[call.id] = self.last_heard_s
        self.running[call.id] = call
        self.awaiting_s = None
        self.spawn(self.run_tool(session, call))

    async def run_tool(self, session: LiveSession, call: ToolCall) -> None:
        t0 = self.clock()
        loop = asyncio.get_running_loop()

        def interim(payload: Result) -> None:  # on the tool's thread, while it runs
            loop.call_soon_threadsafe(self.report_progress, session, call, payload)

        try:
            with progress_to(interim):  # asyncio.to_thread carries the context to the thread
                result = await asyncio.to_thread(self.o.tools, call.name, call.args)
        finally:
            self.running.pop(call.id, None)
            self.last_activity_s = self.awaiting_s = self.clock()  # the answer to it comes next
        started = self.answered.get(call.id)
        if started is not None:
            await started  # the start went back before the end
        text, images = render(result)
        ms = round((self.clock() - t0) * 1000)
        self.tool_log.append({"name": call.name, "args": call.args, "result": text[:500], "ms": ms})
        self.log.event("tool_result", self.clock(), id=call.id, name=call.name, result=text, ms=ms)
        print(f"  <- {text[:300]}  ({ms} ms)", flush=True)
        if call.id in self.cancelled:
            return
        try:
            payload = json.loads(text)
        except ValueError:
            payload = {"text": text}
        if call.name in self.o.moving:
            self.drive_ended_s = self.clock()
        if started is not None:  # the call was answered at the start: the end is a report
            await self.report_end(session, call, text)
            return
        scheduling = "WHEN_IDLE" if call.name in self.o.moving else None
        try:
            self.last_response_s = self.clock()
            await session.send_tool_responses(
                [ToolResponse(call.id, call.name, payload, scheduling)]
            )
            for image in images:
                if image.mime == "image/jpeg":
                    await session.send_image(image.data)
        except Exception as error:  # the session closed while the tool ran
            self.log.event("tool_unsent", self.clock(), id=call.id, error=scrub(repr(error)))

    def report_progress(self, session: LiveSession, call: ToolCall, payload: Result) -> None:
        """A drive's start (the goal server took it) answers its call: the model says its
        acknowledgement (or, when it already has, hears it silently); the drive's end follows as
        a report (:meth:`report_end`). From here on, a drive the model starts again needs a
        person's words first."""
        if call.name in self.o.moving:
            self.heard_since_drive = False
        if (
            not self.cfg.report_drive_start
            or call.name not in self.o.moving
            or call.id in self.answered
            or call.id in self.cancelled
            or call.id not in self.running
        ):
            return
        payload = {**payload, "note": UNDER_WAY}
        said = self.last_model_audio_s is not None and self.last_model_audio_s > self.asked_s.get(
            call.id, math.inf
        )  # the model spoke since it was asked: its acknowledgement is said
        scheduling = DRIVE_STARTED_SAID if said else DRIVE_STARTED
        response = ToolResponse(call.id, call.name, payload, scheduling)
        self.log.event(
            "tool_progress",
            self.clock(),
            id=call.id,
            name=call.name,
            report=payload,
            scheduling=scheduling,
        )
        print(f"  <~ {call.name}: {json.dumps(payload, ensure_ascii=False)}", flush=True)

        async def send() -> None:
            try:
                self.last_response_s = self.clock()
                await session.send_tool_responses([response])
            except Exception as error:  # the session closed meanwhile
                self.log.event("tool_unsent", self.clock(), id=call.id, error=scrub(repr(error)))

        self.answered[call.id] = self.spawn(send())

    async def report_end(self, session: LiveSession, call: ToolCall, result: str) -> None:
        """A drive answered at its start has ended: how, as the robot's own text turn, once the
        robot's line in progress (its acknowledgement) has been said."""
        deadline = self.clock() + END_WAIT_S
        while self.o.pacer.speaking() and self.clock() < deadline:
            await asyncio.sleep(self.o.tick_s)
        args = ", ".join(f"{k}={v!r}" for k, v in call.args.items())
        text = f"{ROBOT_REPORT} the drive {call.name}({args}) has ended: {result}"
        self.log.event("robot_report", self.clock(), id=call.id, text=text)
        try:
            self.last_response_s = self.clock()
            await session.send_text(text)
        except Exception as error:  # the session closed while the robot drove
            self.log.event("tool_unsent", self.clock(), id=call.id, error=scrub(repr(error)))

    async def on_cancel(self, call_id: str, now: float) -> None:
        self.cancelled.add(call_id)
        call = self.running.get(call_id)
        self.log.event("tool_cancelled", now, id=call_id, name=call.name if call else None)
        if call is not None and call.name in self.o.moving:
            answer = await asyncio.to_thread(self.o.halt)
            self.log.event("halt", self.clock(), answer=answer)
            print(f"  (cancelled {call.name}: {answer})", flush=True)

    # -- the record ----------------------------------------------------------------------------

    @property
    def context_s(self) -> float:
        """Audio seconds in the context the handle carries (capped by compression)."""
        cap_s = self.cfg.compression_trigger_tokens / self.cfg.prices.audio_tokens_per_s
        return min(cap_s, self.cost.carried_s + self.cost.audio_in_s + self.cost.audio_out_s)

    def finish_log(self) -> None:
        o, w = self.o, self.wake
        usd = self.cost.usd

        def ms(t: float | None) -> int | None:
            return round((t - w.ended_s) * 1000) if t is not None else None

        self.summary = {
            "sid": self.sid,
            "t": round(self.t_open_wall, 3),
            "dir": str(self.log.dir),
            "dry_run": o.ledger is None,
            "model": self.cfg.model,
            "wake_text": w.text,
            "resumed": self.resumed,
            "latency_ms": {
                "segment_cut": ms(w.cut_s),
                "whisper": round((w.detected_s - w.cut_s) * 1000),
                "connected": ms(self.connected_s),
                "first_audio": ms(self.first_audio_s),
                "first_play": ms(o.pacer.first_play_s),
            },
            "turns": self.turns,
            "tools": self.tool_log,
            "audio_in_s": round(self.cost.audio_in_s, 2),
            "audio_out_s": round(self.cost.audio_out_s, 2),
            "withheld_s": round(self.withheld_s, 2),
            "heard_while_driving": self.heard_while_driving,
            "refused_drives": self.refused_drives,
            "usage": vars(self.cost.usage),
            "usd_est": round(usd, 5),
            "usd_est_stream": round(self.cost.stream_usd, 5),
            "usd_est_usage": round(self.cost.usage_usd, 5),
            "cad_est": round(usd * self.cfg.prices.usd_to_cad, 5),
            "close": self.closing,
            "error": self.error,
        }
        if o.ledger is not None:
            o.ledger.record(self.sid, self.t_open_wall, usd, closed=True)
        if self.handle is not None:
            o.resume.save(self.handle, self.context_s)
        self.log.event("close", self.clock(), reason=self.closing, usd_est=round(usd, 5))
        self.log.close(self.summary)
        print(
            f"  live: closed ({self.closing}); {self.cost.audio_in_s:.1f} s heard,"
            f" {self.cost.audio_out_s:.1f} s spoken, ~{usd * self.cfg.prices.usd_to_cad:.4f} CAD"
            f" (estimate); log {self.log.dir}",
            flush=True,
        )
