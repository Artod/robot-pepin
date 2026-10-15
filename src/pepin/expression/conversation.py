"""The voice cascade's seam for the body: the moments of a conversation, told to observers, and
the live switches that turn a feature on and off.

The cascade (:class:`pepin.voice_cascade.loop.CascadeLoop`) calls every observer at each moment
below, on its event loop: an observer never blocks it (it hands the moment to a thread of its
own) and an observer that raises is logged and never stops the voice. The gesture engine and the
speaker gaze are two observers; they meet in two places only: the engine reads the gaze's
:class:`pepin.expression.speaker.AttentionView`, and the gaze's body turn asks the engine to
rest the arm first (:class:`pepin.expression.speaker.ArmRelease`).

The voice's own audio reaches the body through an :class:`AudioTap` (each chunk of a reply as
the player takes it, on the voice's thread), and the moments of the robot (:data:`MOMENTS`)
through :meth:`ConversationObserver.moment`; config/animation.json maps a moment to an intent.

Owners after slice 0 (gesticulation DESIGN.md §11; one line each):
- A (alive): the prosody listener behind :class:`AudioTap` and the listeners that turn these
  moments into requests (``pepin.expression.prosody``, ``pepin.expression.listeners``);
- B (on the word): ``text_ahead``'s producers (the sentence splitter and the brains);
- this file, the loop's calls and the observers' no-ops are the contract: neither A nor B edits
  them.

A feature's switch is one :class:`LiveFlag`: its default is config/expression.json's ``flags``,
and ``data/voice/control.json``'s ``{"flags": {"gestures": false}}`` (written by
``scripts/voice_cascade.py switch``) changes it while the cascade runs, answered in
``control.jsonl`` like a brain or a voice switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pepin.expression.markers import Part

GESTURES = "gestures"  # the gesture engine's switch: every gesture and idle layer of every part
SPEAKER_GAZE = "speaker_gaze"  # the speaker gaze's switch: the head's base layer and its body turn
# The moments of the robot (DESIGN.md §6.3), each a request source of the gesture engine; the
# loop tells SLOW_TOOL and LISTENER_PAUSE, the engine's listeners find the others
SLOW_TOOL = "slow_tool"  # a slow tool runs before the reply says a word (the filler)
THINK_WAIT = "think_wait"  # the turn closed and no reply audio by think_glance_s
LISTENER_PAUSE = "listener_pause"  # the ears' pause inside the person's utterance
ACKNOWLEDGED = "acknowledged"  # the person finished a statement (not a question)
GREETING_DUE = "greeting_due"  # the wake or an utterance holds a greeting
FAREWELL_DUE = "farewell_due"  # an utterance holds a goodbye
YES_FIRST = "yes_first"  # the reply's first word says yes
NO_FIRST = "no_first"  # the reply's first word says no
QUESTION_END = "question_end"  # a part of the reply ends with a question mark
LONG_REPLY = "long_reply"  # an inner pause of a reply over aversion_reply_s
FACE_FOUND = "face_found"  # the attention engine's first clear face of the conversation
# Which moment goes first when two want the head at once (priority 4 of DESIGN.md §6.5, its
# order: greeting and farewell, yes and no, the thinking glance, the acknowledgement, the
# listener's nod); a moment not listed goes after these.
MOMENTS: tuple[str, ...] = (
    GREETING_DUE, FAREWELL_DUE, YES_FIRST, NO_FIRST, SLOW_TOOL, THINK_WAIT, QUESTION_END,
    LONG_REPLY, FACE_FOUND, ACKNOWLEDGED, LISTENER_PAUSE,
)  # fmt: skip


def moment_rank(name: str) -> int:
    """A moment's place in :data:`MOMENTS` (0 first); any other after them all."""
    return MOMENTS.index(name) if name in MOMENTS else len(MOMENTS)


@dataclass(frozen=True)
class Heard:
    """An utterance the ears closed: when it began and ended (laptop monotonic seconds), the
    words Whisper read, whether they were only noise, its loudest frame (dBFS, None when not
    measured) and whether it calls the robot by name."""

    start_s: float
    end_s: float
    text: str = ""
    noise: bool = False
    peak_db: float | None = None
    named: bool = False

    @property
    def words(self) -> bool:
        """Whether the transcriber read words in it (not noise, not empty)."""
        return not self.noise and bool(self.text.strip())

    @property
    def duration_s(self) -> float:
        """How long the person spoke."""
        return self.end_s - self.start_s


@dataclass(frozen=True)
class VoiceChunk:
    """One chunk of a reply's audio as the player took it: part ``index`` of reply ``turn``, the
    part's stream span after it (its first second, and its last so far: this chunk ends there),
    the PCM at the voice's ``rate`` (16-bit mono) and when it was fed (laptop monotonic)."""

    turn: int
    index: int
    start_s: float
    end_s: float
    pcm: bytes
    rate: int
    t: float


class AudioTap(Protocol):
    """Hears every chunk of a reply on the voice's thread: one append to a queue of its own,
    nothing computed there (the voice's latency is the cascade's first budget)."""

    def audio(self, chunk: VoiceChunk) -> None:
        """Take ``chunk`` (never blocks, never raises into the voice)."""
        ...


class ConversationObserver(Protocol):
    """What the body hears of a conversation, moment by moment."""

    def opened(self, wake: Heard) -> None:
        """A conversation opened on ``wake`` (the utterance with the robot's name)."""
        ...

    def speech_started(self, t: float) -> None:
        """The ears opened an utterance at ``t``: the person is speaking (in a conversation)."""
        ...

    def speech_ended(self, heard: Heard) -> None:
        """An utterance of the conversation was closed and read."""
        ...

    def part_queued(self, turn: int, index: int, part: Part) -> None:
        """Part ``index`` of reply ``turn`` went to the voice, its markers with it."""
        ...

    def voice_started(self, turn: int) -> None:
        """Reply ``turn``'s first audio is with the player: the voice has synthesised its first
        chunk (work that shares the GPU with it waits for this)."""
        ...

    def part_started(self, turn: int, index: int, stream_s: float) -> None:
        """Part ``index`` of reply ``turn`` has its first chunk with the player, at stream second
        ``stream_s`` (its mark's start: exact from here on)."""
        ...

    def text_ahead(self, turn: int, sentence: int, text: str) -> None:
        """Reply ``turn``'s sentence ``sentence`` so far, from the brain's token stream, before it
        is whole (a marker just closed, the reply's first word): its markers as written."""
        ...

    def part_spoken(self, turn: int, index: int) -> None:
        """The voice has fed all of that part's audio to the player: its mark is final."""
        ...

    def reply_cut(self, turn: int) -> None:
        """Reply ``turn`` stopped before its end (a barge-in, or dropped for a merged turn)."""
        ...

    def reply_done(self, turn: int) -> None:
        """Reply ``turn``'s audio has played out."""
        ...

    def moment(self, name: str, text: str = "") -> None:
        """A named moment of the robot (:data:`SLOW_TOOL`), which config/animation.json may map
        to an intent; ``text``: at :data:`LISTENER_PAUSE`, the person's words so far when the
        ears' early read of that pause has landed ("" otherwise, and for every other moment)."""
        ...

    def drive(self, active: bool, ran: bool = True) -> None:
        """A drive the conversation started began (True) or ended (False); at its end ``ran``
        says whether it set the body moving (the goal server took a goal of its calls, or the
        same drive was under way): False when every call was refused or found nothing to do (a
        turn already facing its target), nothing moved."""
        ...

    def closed(self, reason: str) -> None:
        """The conversation closed (idle, a goodbye, end_conversation, a cap, the run's end)."""
        ...


class EventSink(Protocol):
    """One line of the conversation's events.jsonl from a body part's own thread: what happened,
    when (the loop's monotonic clock; the log puts it on the conversation's clock) and the
    details."""

    def __call__(self, kind: str, now: float, /, **fields: Any) -> None:
        """Write the line (never raises into the caller's way: a lost line is only a lost line)."""
        ...


class LiveFlag(Protocol):
    """A feature's one switch, changed while the cascade runs."""

    @property
    def flag(self) -> str:
        """Its name in ``control.json``'s ``flags`` (:data:`GESTURES`, :data:`SPEAKER_GAZE`)."""
        ...

    @property
    def on(self) -> bool:
        """Whether the feature is on now."""
        ...

    def switch(self, on: bool) -> str | None:
        """Turn the feature on or off (turning it on re-reads its config knobs); None when
        applied, else why not."""
        ...
