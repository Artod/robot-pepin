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

The person starting again within ``continuation_s`` of an utterance's close, before the reply
to it has sent anything to the voice, keeps that reply silent: its brain goes on, its
sentences and tools wait until the new utterance is read. Words join both utterances into one
turn (the reply is dropped and the brain asked again); Whisper's noise line lets the reply go
on as if nothing had been heard.

With ``brain.speculative`` on, the brain is asked at the early read of a pause (the ears'
``words``), before the utterance closes: when the words look complete (two-gates' verdict; any
words with the silence rule or Smart Turn), hold ``brain.speculative_min_words`` at least, and
nothing else is in flight in the conversation (no reply, no utterance being read or held, not a
continuation of the one before; person recognition off, since who spoke is named at the close).
The tag line is written from the same audio without joining the baseline. That reply is kept
silent as above (its voice jobs, tools and the body's cues wait). The close with the same read
lets it go on, its first token often already there; any other close drops it (its call billed)
and the words are asked again; a brain error before the close is dropped the same way. At most
one such call an utterance; the conversation's summary counts them (asked, kept, dropped and
what the dropped ones cost).

Every turn prints one line of its timeline (:mod:`pepin.voice_cascade.timeline`) with the
brain's cache tokens and price, the brain and the voice that spoke (and a cloud voice's first
audio and price), and logs it with everything else (:mod:`pepin.voice_cascade.log`).

Each sentence is parsed for markers (:func:`pepin.expression.markers.parse`) and goes to the
voice as one part or more, by the player's part index: ``{шёпотом}`` at its start, or after a
comma, says what follows in that style, and so does the rest of the reply until another style
marker (``{}``: the plain voice again); an intent (``{laugh}``) or a face (``{face:smile}``)
rides its part to the observers, never to the voice; a spelled-out name the vocabulary lacks
is dropped and printed. With a switchboard
(:mod:`pepin.voice_cascade.control`) the brain and the voice are switched while it runs: a
new brain is built and warmed, then takes over at the next turn (the conversation starts
afresh with it); a new voice loads beside the one speaking and takes over at its next sentence.

With person recognition on (:class:`pepin.voice_cascade.stages.SpeakerId`, its switch
``people``), each utterance of a conversation (and the wake, once its name is read) is also
measured for who said it while Whisper reads, and the tag line names the speaker
(``[говорит: Саша; тише обычного]``, the prosody words against that person's own baseline);
off, the tag line is the prosody one as before.

The body hears the conversation through observers
(:class:`pepin.expression.conversation.ConversationObserver`), told each moment on the event
loop: the conversation opened, the person started and stopped speaking, a part of a reply went
to the voice and was all fed to the player, a reply was cut or played out, a slow tool's filler,
a drive's start and end, the close. None is registered by default; one that raises is logged
and never stops the voice. A feature's live switch (:class:`pepin.expression.conversation.
LiveFlag`) is flipped through the control file's ``flags``.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import contextlib
import json
import logging
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, Protocol

from pepin.expression.conversation import SLOW_TOOL, ConversationObserver, Heard, LiveFlag
from pepin.expression.markers import EMPTY, Part, Vocabulary, parse
from pepin.tools.registry import Result, fail, ok, progress_to, render
from pepin.voice_cascade.barge_in import FuturePending, Pending, build_detector
from pepin.voice_cascade.brain import END_CONVERSATION
from pepin.voice_cascade.config import CascadeConfig, VoiceConfig
from pepin.voice_cascade.control import FLAG, Request, Switchboard
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
    Speaker,
    SpeakerId,
    Tagger,
    ToolDone,
    ToolUse,
    Transcriber,
    Usage,
    VoiceBill,
)
from pepin.voice_cascade.text import speakable
from pepin.voice_cascade.timeline import Timeline
from pepin.voice_cascade.turn import TurnWorker
from pepin.voice_cascade.turn_text import TextJudge
from pepin.voice_live.audio import Resampler
from pepin.voice_live.budget import Ledger
from pepin.voice_live.events import Events, State
from pepin.voice_live.sources import FrameSink
from pepin.voice_live.wake import Segment, is_goodbye, is_hallucination, is_stop, is_wake
from pepin.voice_live.wake import words as words_of

logger = logging.getLogger(__name__)

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
# a held utterance (or a reply kept silent) waits this long after the person stopped without
# an utterance to read
HOLD_RELEASE_S = 0.3
# a reply dropped for the turn after it (the person went on; asked early on other words): it
# answered no turn of its own
UNANSWERED = frozenset(("merged", "speculative"))


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
    speculation: Speculation | None = None  # asked at the early read, before the close


@dataclass(eq=False)
class Speculation:
    """A turn asked at the early read of an utterance still open (``brain.speculative``): kept
    when the close brings the same read, dropped otherwise."""

    read: concurrent.futures.Future[str]  # the early read it answers
    input: Input
    prosody: Prosody | None  # measured on the read's audio: the close's own when kept
    asked_s: float
    reply: Reply | None = None  # once the conductor has begun it
    state: Literal["asked", "kept", "dropped"] = "asked"


@dataclass
class Speculative:
    """A conversation's calls asked early: kept by the close, dropped, what the dropped cost."""

    asked: int = 0
    kept: int = 0
    dropped: int = 0
    wasted_usd: float = 0.0


def _set_event() -> asyncio.Event:
    event = asyncio.Event()
    event.set()
    return event


@dataclass
class Reply:
    """One turn in flight."""

    id: int
    input: Input
    task: asyncio.Task[None] | None = None
    sentences: list[str] = field(default_factory=list)  # as the brain wrote them, markers kept
    # one per voice call, by the player's index: (its sentence's index, its spoken text)
    parts: list[tuple[int, str]] = field(default_factory=list)
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
    muted: bool = False  # the person may be going on: its voice jobs and tools wait
    # while muted: what it does once free, in order (its voice jobs, the body's cues)
    waiting: list[Callable[[], None]] = field(default_factory=list)
    free: asyncio.Event = field(default_factory=_set_event)  # set while not muted


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
    speculative: Speculative = field(default_factory=Speculative)

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
        observers: Sequence[ConversationObserver] = (),
        flags: Sequence[LiveFlag] = (),
        vocabulary: Vocabulary = EMPTY,
        people: SpeakerId | None = None,
        judge: TextJudge | None = None,
    ) -> None:
        """``tools`` runs a registry tool by name; ``allowed`` names those the brain may call,
        ``moving`` those that drive; ``ledger`` None spends nothing worth recording (a fake
        brain); ``switchboard``: the brain and the voice may be switched while it runs;
        ``observers`` hear the conversation's moments, ``flags`` are switched by name;
        ``vocabulary``: the intents and faces a marker may name (any other bare name is a
        voice style, a spelled-out unknown one is dropped); ``people``: who said each
        utterance, named in the tag line while its switch is on; ``judge``: two-gates' reader
        of the early words (:mod:`pepin.voice_cascade.turn_text`)."""
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
        self.ears = Ears(config, player.playback, turn, early, detector, judge, self._on_words)
        self.rng = random.Random(seed)
        self.conversation: Conversation | None = None
        self.last_log: CascadeLog | None = None  # the conversation that closed last
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
        self.continuing = 0  # utterances begun as a continuation, not yet read (replies wait)
        self.continuing_s = 0.0  # when the latest of them began
        self._continues = False  # the utterance in progress began as a continuation
        self.speculation: Speculation | None = None  # asked early, not yet settled by a close
        self._may_speculate = False  # the utterance in progress may still be asked early
        self._brain_switches = 0  # brain switches asked for (an older one finishing is dropped)
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
        self.observers: list[ConversationObserver] = list(observers)
        self.flags: dict[str, LiveFlag] = {flag.flag: flag for flag in flags}
        self.vocabulary = vocabulary
        self.people = people
        self._people = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix="people")
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
            self._people.shutdown(wait=False)

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
            first = not reply.audio
            reply.audio = True
            reply.input.timeline.mark("tts_first_audio", t)
            if first and self.observers and self._loop is not None:
                with contextlib.suppress(RuntimeError):  # the loop has ended
                    self._loop.call_soon_threadsafe(self._tell, "voice_started", job.turn)

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
        spoken = job.kind == "reply" and job.index >= 0 and job.turn > self.speech.cut_through
        if spoken and self.observers and self._loop is not None:
            with contextlib.suppress(RuntimeError):  # every chunk of the part is with the player
                self._loop.call_soon_threadsafe(self._tell, "part_spoken", job.turn, job.index)

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
                    lead_s = round(event.lead_s, 2) if event.lead_s is not None else None
                    self.conversation.log.event("start", event.t, lead_s=lead_s)
                    self._tell("speech_started", event.t)
                cfg = self.config
                self._continues = False
                joins = (
                    cfg.continuation_s > 0 and event.t - self.last_closed_s <= cfg.continuation_s
                )
                self._may_speculate = not joins
                if joins:
                    self._continues = await self._continuation()
            elif event.kind == "pause":
                self._backchannel()
            elif event.kind == "barge_in":
                await self._barge_in(event, played if played is not None else 0.0)
            elif event.kind == "words":
                self._speculate(event)
            elif event.kind == "segment" and event.segment is not None:
                self.last_closed_s = event.t
                self._may_speculate = False
                self.hearing += 1
                continues, self._continues = self._continues, False
                self._spawn(self._hear(event, continues))

    async def _continuation(self) -> bool:
        """The person started again right after an utterance closed. A reply that has sent
        nothing to the voice yet is kept silent until the new utterance is read (True: its words
        decide). One whose first sentence is on its way to the speaker is dropped and its words
        held; an utterance still being read is held when Whisper is done."""
        current = self.current
        if current is not None and self._droppable(current):
            if current.muted or not current.sentences:
                self._mute(current)
                self.continuing += 1
                self.continuing_s = self.clock()
                return True
            await self._drop(current, "merged")
            self._hold(current.input)
        elif self.hearing > 0:
            self.hold_next = True  # still being read: held when Whisper is done
        return False

    @staticmethod
    def _droppable(reply: Reply) -> bool:
        """The person's turn in flight has said and done nothing yet (a tool asked for while
        it was silent has not run)."""
        return (
            not reply.finished.is_set()
            and reply.input.kind == "person"
            and not reply.audio
            and (reply.muted or not reply.acted)
        )

    def _mute(self, reply: Reply, announce: bool = True) -> None:
        """Keep ``reply`` silent: its voice jobs, tools and the body's cues wait for
        :meth:`_unmute` (``announce``: logged and printed as the person going on)."""
        if reply.muted:
            return
        reply.muted = True
        reply.free.clear()
        if announce:
            self._event("muted", turn=reply.id, text=reply.input.words)
            print(f"  (the person goes on: the reply to {reply.input.words!r} waits)", flush=True)

    def _unmute(self, reply: Reply | None, why: str, announce: bool = True) -> None:
        """Let ``reply`` speak and act: what it waited to do is done in order."""
        if reply is None or not reply.muted:
            return
        reply.muted = False
        actions, reply.waiting = reply.waiting, []
        for action in actions:
            action()
        reply.free.set()
        if announce:
            self._event("resumed", turn=reply.id, jobs=len(actions), why=why)
            print(f"  ({why}: the reply to {reply.input.words!r} goes on)", flush=True)

    def _resume(self, why: str) -> None:
        """The reply in flight goes on once no continuation is waiting to be read."""
        current = self.current
        if self.continuing == 0 and current is not None and not current.finished.is_set():
            self._unmute(current, why)

    def _say(self, reply: Reply, job: SpeechJob, part: Part | None = None) -> None:
        """A voice job of ``reply`` (``part``: what it says, told to the observers as it goes
        to the voice): to the voice, or kept while the reply is silent."""

        def send() -> None:
            self.speech.say(job)
            if part is not None:
                self._tell("part_queued", reply.id, job.index, part)

        self._when_free(reply, send)

    @staticmethod
    def _when_free(reply: Reply, action: Callable[[], None]) -> None:
        """Do ``action`` of ``reply`` now, or once it is let go while it is silent."""
        if reply.muted:
            reply.waiting.append(action)
        else:
            action()

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

    def _on_words(self, event: EarEvent) -> None:
        """An early read landed (the transcriber's thread): in line with the mic's moments."""
        if self._loop is not None:
            with contextlib.suppress(RuntimeError):  # the loop has ended
                self._loop.call_soon_threadsafe(self._ears_queue.put_nowait, (event, None))

    def _speculate(self, event: EarEvent) -> None:
        """The early read of the pause in progress landed: with ``brain.speculative`` on, the
        brain is asked now when the words look like a whole turn and nothing else is in flight
        (at most once an utterance)."""
        b = self.config.brain
        text = (event.text or "").strip()
        if not (b.speculative and self._may_speculate and self.conversation is not None):
            return
        if event.early is None or event.segment is None:
            return
        if self.ears.two_gates and self.ears.judge is not None and event.verdict != "complete":
            return
        if is_hallucination(text) or len(words_of(text)) < b.speculative_min_words:
            return
        if self.people is not None and self.people.on:
            return  # who spoke is named at the close
        busy = self.current is not None or self.held is not None or self.speculation is not None
        if busy or self.hold_next or self.continuing or self.hearing or not self.inputs.empty():
            return
        self._may_speculate = False
        self._spawn(self._ask_early(event, text))

    async def _ask_early(self, event: EarEvent, text: str) -> None:
        """The turn of the early words, asked before the close and kept silent: the tag line
        from the read's audio, the baseline left for the close."""
        assert event.early is not None and event.segment is not None
        async with self._order:  # before the close's read, which waits for it
            tagger = self.tagger if self.config.prosody_tags else None
            prosody: Prosody | None = None
            if tagger is not None:
                loop = asyncio.get_running_loop()
                prosody = await loop.run_in_executor(self._praat, tagger.measure, event.segment.pcm)
            conversation = self.conversation
            if conversation is None:
                return
            tag = tagger.line(prosody, text, keep=False) if tagger and prosody else ""
            said = f"{tag} {text}" if tag else text
            item = Input("person", said, Timeline(event.segment.ended_s), text, tag,
                         is_goodbye(text))  # fmt: skip
            now = self.clock()
            spec = item.speculation = Speculation(event.early, item, prosody, now)
            self.speculation = spec
            conversation.speculative.asked += 1
            self._event("speculative", state="asked", text=text, verdict=event.verdict,
                        after_ms=round((now - event.segment.ended_s) * 1000))  # fmt: skip
            self.inputs.put_nowait(item)

    async def _settle(self, early: concurrent.futures.Future[str] | None) -> Speculation | None:
        """The turn asked early on this utterance: the one asked on the read that closed it
        (kept or not once its tag line is known), None when there is none; one asked on an
        earlier read of it is dropped now (the person went on after it)."""
        spec = self.speculation
        if spec is None or spec.state != "asked":
            return None
        if spec.read is early:
            return spec
        await self._drop_speculation(spec, "the person went on")
        return None

    def _keep(self, spec: Speculation, closed: Timeline, eot: dict[str, Any] | None) -> None:
        """The close brought the words the brain was asked on: its reply goes on, its turn
        counted from the close's zero with the close's stamps."""
        spec.state = "kept"
        self.speculation = None
        spec.input.timeline.take(closed)
        spec.input.eot = eot
        if self.conversation is not None:
            self.conversation.speculative.kept += 1
        head_ms = round((closed.marks["closed"] - spec.asked_s) * 1000)
        turn = spec.reply.id if spec.reply else None
        self._event("speculative", state="kept", turn=turn, head_ms=head_ms)
        if spec.reply is not None:
            self._unmute(spec.reply, "the turn closed", announce=False)

    async def _drop_speculation(self, spec: Speculation, why: str) -> None:
        """The turn asked early answers words the close did not bring: dropped (its call is
        billed), the close's words asked anew."""
        self._given_up(spec, why)
        reply = spec.reply
        if reply is not None and not reply.finished.is_set():
            await self._drop(reply, "speculative")

    def _given_up(self, spec: Speculation, why: str) -> None:
        """A turn asked early counted as dropped and settled."""
        spec.state = "dropped"
        if self.speculation is spec:
            self.speculation = None
        if self.conversation is not None:
            self.conversation.speculative.dropped += 1
        reply = spec.reply
        self._event("speculative", state="dropped", turn=reply.id if reply else None, why=why)

    @staticmethod
    def _unsettled(reply: Reply) -> bool:
        """``reply`` was asked early and its utterance has not closed yet."""
        spec = reply.input.speculation
        return spec is not None and spec.state == "asked"

    async def _hear(self, event: EarEvent, continues: bool = False) -> None:
        assert event.segment is not None
        try:
            async with self._order:  # utterances are answered in the order they were said
                await self._read(event.segment, event.t, eot_of(event), event.early, continues)
        finally:
            self.hearing -= 1

    async def _read(
        self,
        segment: Segment,
        closed_s: float,
        eot: dict[str, Any] | None = None,
        early: concurrent.futures.Future[str] | None = None,
        continues: bool = False,
    ) -> None:
        """Whisper (and Praat) on one utterance, then its fate: the name opens a conversation,
        a noise line is dropped, words become the next turn (or join the one in flight).
        ``continues``: it began as a continuation, so a silent reply waits for its words."""
        loop = asyncio.get_running_loop()
        spec = await self._settle(early)
        timeline = Timeline(segment.ended_s)
        timeline.mark("closed", closed_s)
        if early is not None and not early.cancelled():
            reading = asyncio.wrap_future(early)  # started in the pause: done, or nearly
        else:
            early = None
            reading = loop.run_in_executor(self._whisper, self.transcriber.transcribe, segment.pcm)
        tagger = self.tagger if self.config.prosody_tags else None
        measured = spec.prosody if spec is not None else None  # the same audio's
        measuring = (
            loop.run_in_executor(self._praat, tagger.measure, segment.pcm)
            if tagger is not None and measured is None
            else None
        )
        people = self.people if self.people is not None and self.people.on else None
        who: asyncio.Future[Callable[[str], Speaker]] | None = None
        if people is not None and self.conversation is not None:
            who = loop.run_in_executor(self._people, people.measure, segment.pcm, segment.ended_s)
        text = (await reading).strip()
        prosody: Prosody | None = await measuring if measuring is not None else measured
        read_s = self.clock()
        noise = not text or is_hallucination(text)
        if continues:
            self.continuing = max(0, self.continuing - 1)
        whisper_ms = round((read_s - closed_s) * 1000)
        engine = getattr(self.transcriber, "name", None)  # the engine in use (switchable)
        voice_db = round(self.ears.voice_peak_db, 1)
        if self.conversation is None:
            woke = not noise and (self.config.wake == "always" or is_wake(text))
            print(
                f"heard (idle, {whisper_ms} ms): {text!r}"
                + ("  -> WAKE" if woke else "  (noise line)" if noise else ""),
                flush=True,
            )
            row = {
                "t": round(self.wall(), 3), "text": text, "wake": woke, "noise": noise,
                "whisper_ms": whisper_ms, "engine": engine,
            }  # fmt: skip
            append_line(day_dir(self.logs, self.wall()) / "cascade_heard.jsonl", row)
            if not woke or not self.open(segment.pcm):
                return
            if people is not None and who is None:  # idle: nobody is looked at before the name
                who = loop.run_in_executor(
                    self._people, people.measure, segment.pcm, segment.ended_s
                )
            self._tell("opened", heard_of(segment, text, noise, voice_db))
        conversation = self.conversation
        assert conversation is not None
        conversation.log.event(
            "heard", read_s, text=text, noise=noise, whisper_ms=whisper_ms, early=early is not None,
            seconds=round(segment.duration_s, 2), voice_peak_db=voice_db, engine=engine,
        )  # fmt: skip
        self._tell("speech_ended", heard_of(segment, text, noise, voice_db))
        if noise:
            if spec is not None:
                await self._drop_speculation(spec, "only noise")
            self.hold_next = False
            self._release()
            self._resume("only noise followed")  # never cancelled, never asked again
            return
        conversation.last_heard_s = read_s
        if who is None:
            tag = tagger.line(prosody, text) if tagger is not None and prosody is not None else ""
        else:
            speaker = await self._speaker(who, text)
            key = speaker.key
            body = (
                tagger.describe(prosody, text, key)
                if tagger is not None and prosody is not None and key is not None
                else ""
            )
            tag = speaker.line(body)
            conversation.log.event(
                "speaker", self.clock(), head=speaker.head, **(speaker.log or {})
            )
        timeline.mark("transcript", self.clock())
        print(f"heard> {tag + ' ' if tag else ''}{text}  ({whisper_ms} ms)", flush=True)
        if is_stop(text) and self.drives:
            answer = await asyncio.to_thread(self.halt)
            conversation.log.event("halt", self.clock(), answer=answer, why="stop heard")
            print(f"  (stop heard during a drive: {answer})", flush=True)
        self.heard_since_drive = True
        said, words, fragments = (f"{tag} {text}" if tag else text), text, 1
        if spec is not None:
            if said == spec.input.text:
                self._keep(spec, timeline, eot)
                return
            await self._drop_speculation(spec, "the tag line changed")
        current = self.current
        if (
            current is not None
            and not current.finished.is_set()
            and not current.audio
            and (current.muted or not current.acted)
        ):
            # the person went on before the robot said or did anything: one turn of both (a
            # silent reply's tool call is still waiting: it never runs)
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

    @staticmethod
    async def _speaker(who: asyncio.Future[Callable[[str], Speaker]], text: str) -> Speaker:
        """The utterance's speaker once its words are known; nobody when recognition broke
        (logged: the voice goes on)."""
        try:
            return (await who)(text)
        except Exception:
            logger.exception("person recognition broke on an utterance")
            return Speaker("", log={"error": "recognition broke"})

    # -- turns ---------------------------------------------------------------------------------

    async def _conduct(self) -> None:
        while True:
            item = await self.inputs.get()
            spec = item.speculation
            if spec is not None and spec.state == "dropped":
                continue  # asked early on words the close did not bring: never begun
            if self.conversation is None and not self.open(b""):
                continue  # closed meanwhile (a drive's report, a turn heard at the close)
            if item.kind == "robot":
                deadline = self.clock() + END_WAIT_S
                while self.player.speaking() and self.clock() < deadline:
                    await asyncio.sleep(self.tick_s)
            self.last_turn += 1
            reply = Reply(self.last_turn, item, brain=self.brain_name, voice=self.voice_name)
            self.replies[reply.id] = reply
            if spec is not None:
                spec.reply = reply
            if spec is not None and spec.state == "asked":
                self._mute(reply, announce=False)  # the utterance is still open: it waits
            elif self.continuing and item.kind == "person":
                self._mute(reply)  # the person is still going on: it waits for their words
            self.current = reply
            if not self._unsettled(reply):
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
            self._say(reply, SpeechJob(reply.id, len(reply.parts), kind="end"))
            await self._played_out(reply)
            self._tell("reply_done", reply.id)
            self.brain.commit()
            reply.outcome = "committed"
        except asyncio.CancelledError:
            if not reply.outcome:  # cancelled by the run's end, not by a barge-in or a merge
                self.brain.abandon()
                reply.outcome = "stopped"
            raise
        except Exception as error:
            reply.error = f"{type(error).__name__}: {error}"[:300]
            spec = reply.input.speculation
            if spec is not None and spec.state == "asked":  # nothing said: the close asks anew
                print(f"  !! brain, asked early: {reply.error}", flush=True)
                self._given_up(spec, "the brain failed")
                reply.outcome = "speculative"
                self.speech.cut(reply.id)
                self._tell("reply_cut", reply.id)
                self.brain.abandon()
                self._finish(reply)
                return
            reply.outcome = "error"
            print(f"  !! brain: {reply.error}", flush=True)
            self.brain.abandon()
            self._unmute(reply, "the brain failed")  # what it said before the error, in order
            if not reply.audio and self.config.error_line:
                self.speech.say(SpeechJob(reply.id, len(reply.parts), self.config.error_line))
            self.speech.say(SpeechJob(reply.id, len(reply.parts) + 1, kind="end"))
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
            sentence = len(reply.sentences)
            reply.sentences.append(event.text)
            parsed = parse(event.text, self.vocabulary)
            first = len(reply.parts)
            for part in parsed.parts:
                if part.style is not None:  # in force for the rest of the reply
                    reply.style = part.style
                    reply.styles.append(part.style)
                if not part.text and not part.markers:
                    continue  # a style for the sentences after this one, nothing to say
                index = len(reply.parts)
                reply.parts.append((sentence, part.text))
                self._say(reply, SpeechJob(reply.id, index, part.text, style=reply.style), part)
            markers = [m.raw for part in parsed.parts for m in part.markers]
            if parsed.dropped:
                print(f"  (markers dropped, they name nothing: {' '.join(parsed.dropped)})",
                      flush=True)  # fmt: skip
            if log is not None:
                log.event("sentence", now, turn=reply.id, index=sentence, text=event.text,
                          style=reply.style, parts=list(range(first, len(reply.parts))),
                          markers=markers, dropped=list(parsed.dropped))  # fmt: skip
        elif isinstance(event, ToolUse):
            if event.name in SLOW_TOOLS and not reply.sentences and not reply.filled:
                reply.filled = True  # nothing said yet: the wait is announced locally
                self._when_free(reply, lambda: self._tell("moment", SLOW_TOOL))
                if self.config.tool_filler:
                    filler = SpeechJob(reply.id, -1, self.config.tool_filler, "backchannel")
                    self._say(reply, filler)
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
        by_part = self.player.heard(reply.id, played, [text for _, text in reply.parts])
        reply.played = in_sentences(reply.parts, len(reply.sentences), by_part)
        reply.outcome = "interrupted"
        self._tell("reply_cut", reply.id)
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
        self._tell("reply_cut", reply.id)
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
            if reply.input.kind == "person" and reply.outcome not in UNANSWERED:
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
        spec = reply.input.speculation
        if spec is not None:  # when the brain was asked, after the person stopped
            row["speculative"] = {"asked_ms": round((spec.asked_s - timeline.stopped) * 1000)}
        conversation = self.conversation
        self.ears.echo_peak_db = -120.0
        if reply.outcome == "speculative":
            if conversation is not None:
                conversation.speculative.wasted_usd += u.usd
            print(f"  (asked early, dropped: {u.usd * 100:.4f} US cents)", flush=True)
        elif reply.outcome != "merged":
            if row["reply"]:
                print(f"pepin> {row['reply']}", flush=True)
            eot_cell = f" | merged {reply.input.fragments}" if reply.input.fragments > 1 else ""
            if eot is not None:
                p = "-" if eot["p"] is None else f"{eot['p']:.2f}"
                eot_cell += f" | eot {eot['closed_by']} p {p}, saved {eot['saved_ms']:.0f} ms"
                if "verdict" in eot:
                    eot_cell += f", words {eot['verdict']} ({eot['why']})"
            closed_ms = row["ms"]["closed"]
            if "speculative" in row and closed_ms is not None:
                head_ms = closed_ms - row["speculative"]["asked_ms"]
                eot_cell += f" | asked early, {head_ms} ms before the cut"
            print(
                f"  [{reply.id} {reply.outcome}] {timeline.line()} | cache {u.cache_read}r/"
                f"{u.cache_write}w in {u.input} out {u.output} | {u.usd * 100:.4f} US cents"
                f" | {voice_cell(reply)}" + eot_cell,
                flush=True,
            )
        if conversation is not None:
            conversation.turns.append(row)
            conversation.log.event("turn", self.clock(), **row)
        self.turns.append(row)

    # -- tools ---------------------------------------------------------------------------------

    async def _tool(self, name: str, args: dict[str, Any]) -> Result:
        """A tool the brain called. A drive answers when the goal server takes it; its end
        comes back as the robot's own turn. While the person may be going on (the reply is
        silent) nothing runs: their words drop the reply, and the call with it."""
        reply = self.current
        if reply is not None:
            await reply.free.wait()
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
        self._tell("drive", True)
        if not self.config.report_drive_start:
            try:
                return await asyncio.to_thread(self.tools, name, args)
            finally:
                self._tell("drive", False)
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
            self._tell("drive", False)
            return task.result()
        task.add_done_callback(lambda t: self._drive_ended(name, args, t))
        if self.current is not None and self.current.sentences:  # the acknowledgement is said
            return {**started.result(), "note": UNDER_WAY + ACKNOWLEDGED, FOLLOW_UP: False}
        return {**started.result(), "note": UNDER_WAY}

    def _drive_ended(self, name: str, args: dict[str, Any], task: asyncio.Task[Result]) -> None:
        self.drives.discard(task)
        self._tell("drive", False)
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
                         "prosody_tags": cfg.prosody_tags,
                         **{name: flag.on for name, flag in self.flags.items()}})  # fmt: skip
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
        self.speculation, self._may_speculate = None, False
        person = [
            t
            for t in conversation.turns
            if t["kind"] == "person" and t["outcome"] not in UNANSWERED
        ]  # a merged fragment is part of the turn after it, a dropped early one answered none
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
        early = conversation.speculative
        if early.asked:
            summary["speculative"] = {**vars(early), "wasted_usd": round(early.wasted_usd, 6)}
        conversation.log.event("close", now, reason=reason, usd_est=round(conversation.usd, 6))
        conversation.log.close(summary)
        self.last_log = conversation.log
        ledger = self.ledger
        if ledger is None and conversation.usd > 0:
            # spent all the same: a paid brain switched to the echo meanwhile, or a paid voice
            ledger = self.paid_ledger
        if ledger is not None:
            ledger.record(conversation.sid, conversation.t_open_wall, conversation.usd, closed=True)
        asked = (
            f"; asked early {early.asked}: {early.kept} kept, {early.dropped} dropped"
            f" ({early.wasted_usd * 100:.3f} US cents)"
            if early.asked
            else ""
        )
        print(
            f"  conversation closed ({reason}): {len(person)} turns, ~{conversation.usd * 100:.3f}"
            f" US cents (estimate){asked}; log {conversation.log.dir}",
            flush=True,
        )
        self.events.emit("idle")
        self._tell("closed", reason)

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
            if (
                self.continuing
                and not self.ears.gate.talking
                and self.hearing == 0
                and self._ears_queue.empty()
                and now - self.continuing_s > HOLD_RELEASE_S
            ):
                self.continuing = 0  # the person stopped without an utterance to read
                self._resume("no words followed")
            if conversation is None:
                continue
            speaking = self.player.playback() == "reply"
            current = self.current
            answering = current is not None and not self._unsettled(current)  # not still asked
            busy = answering or not self.inputs.empty() or self.hearing > 0
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
                 ("transcriber", request.transcriber),
                 *((f"flag {name}", "on" if on else "off")
                   for name, on in request.flags.items()))  # fmt: skip
        print("  (switch asked: " + ", ".join(f"{k} {v}" for k, v in asked if v) + ")", flush=True)
        if request.brain:
            self._spawn(self._switch_brain(request.id, request.brain))
        if request.voice:
            self._switch_voice(request.id, request.voice)
        if request.transcriber and self.switchboard is not None:
            self.switchboard.switch_transcriber(request.id, request.transcriber)
        for name, on in request.flags.items():
            self._switch_flag(request.id, name, on)

    def _switch_flag(self, request: str, name: str, on: bool) -> None:
        """A feature turned on or off (on the event loop: a flag switches at once, its own
        work goes to its own thread); answered for ``request``."""
        board = self.switchboard
        assert board is not None
        flag = self.flags.get(name)
        if flag is None:
            known = ", ".join(self.flags) or "none in this run"
            board.answer(request, FLAG, name, "failed", on=on, error=f"no such flag ({known})")
            print(f"  !! flag {name}: no such flag ({known})", flush=True)
            return
        try:
            why = flag.switch(on)
        except Exception as error:  # a feature that breaks switching is reported, not fatal
            logger.exception("flag %s: the switch broke", name)
            why = f"{type(error).__name__}: {error}"[:300]
        if why is not None:
            board.answer(request, FLAG, name, "failed", on=on, error=why)
            print(f"  !! flag {name} {'on' if on else 'off'}: {why}", flush=True)
            return
        self._event("switch", flag=name, on=on)
        board.answer(request, FLAG, name, "applied", on=on)
        print(f"  (flag {name}: {'on' if on else 'off'})", flush=True)

    async def _switch_brain(self, request: str, name: str) -> None:
        board = self.switchboard
        assert board is not None
        start = self.clock()
        self._brain_switches += 1
        number = self._brain_switches
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
        if number != self._brain_switches:  # a later switch was asked meanwhile: it decides
            board.answer(request, "brain", name, "failed", error="superseded by a later switch")
            print(f"  (switch to brain {name}: superseded by a later switch)", flush=True)
            close = getattr(setup.brain, "aclose", None)
            if close is not None:
                with contextlib.suppress(Exception):
                    await close()
            return
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

    def record(self, kind: str, now: float, /, **fields: Any) -> None:
        """One event of a body part (the speaker gaze) in the conversation's events.jsonl at
        ``now`` (this loop's clock): the open conversation's, else the one that closed last (a
        release that follows its close). Safe from any thread: one locked line write."""
        conversation = self.conversation
        log = conversation.log if conversation is not None else self.last_log
        if log is not None:
            log.event(kind, now, **fields)

    def _event(self, kind: str, **fields: Any) -> None:
        conversation = self.conversation
        if conversation is not None:
            conversation.log.event(kind, self.clock(), **fields)

    def _tell(self, moment: str, *args: Any) -> None:
        """Every observer hears ``moment`` (a method of ConversationObserver) with ``args``;
        one that raises is logged, and the others and the voice go on."""
        for observer in self.observers:
            try:
                getattr(observer, moment)(*args)
            except Exception:
                logger.exception("observer %s broke on %s", type(observer).__name__, moment)


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


def in_sentences(parts: list[tuple[int, str]], sentences: int, played: Played) -> Played:
    """What was heard of a reply counted in its parts (the player's marks) counted in its
    sentences, as the brain keeps them: the sentences before the part cut, then the parts of
    the cut sentence heard whole and the words of the one cut."""
    if played.sentences >= len(parts):
        return Played(sentences, "")
    cut = parts[played.sentences][0]
    before = [text for sentence, text in parts[: played.sentences] if sentence == cut]
    partial = " ".join(t for t in [*before, played.partial] if t)
    return Played(cut, partial)


def heard_of(segment: Segment, text: str, noise: bool, peak_db: float | None = None) -> Heard:
    """An utterance as the body hears it: its start (its last voiced frame less its length,
    the lead-in included) and end on the laptop's clock, what Whisper read, its loudest frame
    and whether it calls the robot by name."""
    start = segment.ended_s - segment.duration_s
    named = not noise and is_wake(text)
    return Heard(start, segment.ended_s, text, noise, peak_db, named)


def eot_of(event: EarEvent) -> dict[str, Any] | None:
    """The end-of-turn model's verdict carried by a segment event (None without a model); with
    two-gates also the verdict on the words, why and the judge's ms."""
    if event.closed_by is None:
        return None
    eot: dict[str, Any] = {
        "closed_by": event.closed_by,
        "p": None if event.p is None else round(event.p, 3),
        "model_ms": None if event.model_ms is None else round(event.model_ms, 1),
        "saved_ms": None if event.saved_ms is None else round(event.saved_ms),
    }
    if event.verdict is not None:
        eot["verdict"] = event.verdict
        eot["why"] = event.verdict_why
        eot["text_ms"] = None if event.text_ms is None else round(event.text_ms, 1)
    return eot
