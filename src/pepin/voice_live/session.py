"""One conversation: a Live session opened by the wake gate, streamed until a goodbye, silence,
or a cap closes it.

The mic goes to Live only while the session is open and the robot is not speaking (unless
``barge_in``); the model's audio is resampled to the board's rate and paced to its speaker as it
streams; tool calls run on worker threads through the robot's registry while the conversation
goes on, and their results go back; Live's interruption silences the speaker. Every session
leaves ``data/voice/<day>/live_<HHMMSS>/`` (in.wav, out.wav, events.jsonl) and one summary line in
``data/voice/<day>/live.jsonl``: transcripts, tool calls, latencies, the estimated cost.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pepin.tools.registry import Result, render
from pepin.voice_live.audio import LIVE_IN_RATE, LIVE_OUT_RATE, Pacer, Resampler, WavWriter
from pepin.voice_live.budget import Ledger, SessionCost
from pepin.voice_live.config import LiveConfig
from pepin.voice_live.events import Events
from pepin.voice_live.live import (
    END_CONVERSATION,
    LiveConnector,
    LiveMessage,
    LiveSession,
    ToolCall,
    ToolResponse,
)
from pepin.voice_live.wake import Segmenter, is_goodbye

logger = logging.getLogger(__name__)

ToolRunner = Callable[[str, dict[str, Any]], Result]
PREROLL_CHUNK = 3200  # bytes: the wake segment goes to Live in 100 ms pieces, as fast as it can
TICK_S = 0.1
AWAIT_REPLY_S = 8.0  # "thinking" gives up after this long without an answer (it may not answer)
DRAIN_S = 12.0  # the last words are allowed this long to play after the session closed
PROGRESS_S = 5.0  # the ledger hears the running estimate this often (a crash loses less)
END_FALLBACK_S = 3.0  # end_conversation without a turn end after it: closed after this long


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
        clock: Callable[[], float] = time.monotonic,
        wall: Callable[[], float] = time.time,
        tick_s: float = TICK_S,
    ) -> None:
        """``tools`` runs a tool by name (the registry bound to the robot); ``moving`` names the
        tools that drive (non-blocking; halted if the model cancels them); ``ledger`` None is
        the dry run, which spends nothing and records nothing."""
        self.live, self.config = live, config
        self.frames, self.pacer = frames, pacer
        self.tools, self.moving, self.halt = tools, moving, halt
        self.events, self.logs, self.ledger, self.resume = events, logs, ledger, resume
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
        self.cost = SessionCost(self.cfg.prices, self.cfg.compression_trigger_tokens)
        self.resampler = Resampler(LIVE_OUT_RATE, owner.pacer.rate)
        owner.pacer.first_play_s = None
        self.handle: str | None = None
        self.resumed = False
        self.closing: str | None = None
        self.end_requested_s: float | None = None
        self.connected_s: float | None = None
        self.first_audio_s: float | None = None
        self.last_user_voice_s = wake.ended_s
        self.last_activity_s = wake.cut_s
        self.last_speaking_s = 0.0
        self.awaiting_s: float | None = wake.cut_s  # the wake segment itself awaits an answer
        self.turn = _Turn(user_done_s=wake.ended_s)
        self.turns: list[dict[str, Any]] = []
        self.tool_log: list[dict[str, Any]] = []
        self.running: dict[str, ToolCall] = {}
        self.cancelled: set[str] = set()
        self.tool_tasks: set[asyncio.Task[None]] = set()
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
        for attempt_handle in [handle, None] if handle else [None]:
            try:
                async with o.live.connect(attempt_handle) as session:
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

    async def forward_mic(self, session: LiveSession) -> None:
        """Mic frames to Live while the robot is silent; the local gate marks the user's
        voice (for the idle timer) and the end of their turn (for "thinking")."""
        ears = Segmenter(silence_s=0.5, max_s=60.0)
        while True:
            frame = await self.o.frames.get()
            if self.o.pacer.speaking() and not self.cfg.barge_in:
                ears.reset()
                continue
            if ears.feed(frame.pcm, frame.t) is not None:
                self.awaiting_s = self.clock()
            if ears.talking:
                self.last_user_voice_s = max(self.last_user_voice_s, ears.last_voice_s)
                self.turn.user_done_s = self.last_user_voice_s
                self.awaiting_s = None
            await self.send_mic(session, frame.pcm)

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
            quiet_since = max(self.last_user_voice_s, self.last_activity_s, self.last_speaking_s)
            if not self.running and not speaking and now - quiet_since > self.cfg.idle_close_s:
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
                o.events.emit("thinking" if busy else "listening")

    # -- server messages -----------------------------------------------------------------------

    async def on_message(self, session: LiveSession, m: LiveMessage) -> None:
        now = self.clock()
        if m.audio:
            self.on_audio(m.audio, now)
        if m.input_text:
            self.turn.user += m.input_text
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
        self.awaiting_s = None
        self.last_activity_s = now
        self.log.model.write(pcm)
        self.cost.audio_out_s += len(pcm) / 2 / LIVE_OUT_RATE
        self.o.pacer.feed(self.resampler.process(pcm))

    def on_turn_complete(self, now: float) -> None:
        self.o.pacer.end()
        self.cost.turn()
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
        self.turn = _Turn()
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
            self.o.moved = True
        self.running[call.id] = call
        self.awaiting_s = None
        task = asyncio.create_task(self.run_tool(session, call))
        self.tool_tasks.add(task)
        task.add_done_callback(self.tool_tasks.discard)

    async def run_tool(self, session: LiveSession, call: ToolCall) -> None:
        t0 = self.clock()
        try:
            result = await asyncio.to_thread(self.o.tools, call.name, call.args)
        finally:
            self.running.pop(call.id, None)
            self.last_activity_s = self.awaiting_s = self.clock()  # the answer to it comes next
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
        scheduling = "WHEN_IDLE" if call.name in self.o.moving else None
        try:
            await session.send_tool_responses(
                [ToolResponse(call.id, call.name, payload, scheduling)]
            )
            for image in images:
                if image.mime == "image/jpeg":
                    await session.send_image(image.data)
        except Exception as error:  # the session closed while the tool ran
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
            "usage": vars(self.cost.usage),
            "usd_est": round(usd, 5),
            "usd_est_audio": round(self.cost.audio_usd, 5),
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
