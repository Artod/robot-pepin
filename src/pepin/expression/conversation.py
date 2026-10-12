"""The voice cascade's seam for the body: the moments of a conversation, told to observers, and
the live switches that turn a feature on and off.

The cascade (:class:`pepin.voice_cascade.loop.CascadeLoop`) calls every observer at each moment
below, on its event loop: an observer never blocks it (it hands the moment to a thread of its
own) and an observer that raises is logged and never stops the voice. The gesture engine and the
speaker gaze are two observers; neither knows the other.

A feature's switch is one :class:`LiveFlag`: its default is config/expression.json's ``flags``,
and ``data/voice/control.json``'s ``{"flags": {"gestures": false}}`` (written by
``scripts/voice_cascade.py switch``) changes it while the cascade runs, answered in
``control.jsonl`` like a brain or a voice switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pepin.expression.markers import Part

GESTURES = "gestures"  # the gesture engine's switch: every gesture and idle layer of every part
SPEAKER_GAZE = "speaker_gaze"  # the speaker gaze's switch: the head's base layer and its body turn
SLOW_TOOL = "slow_tool"  # a moment: a slow tool runs before the reply says a word (the filler)


@dataclass(frozen=True)
class Heard:
    """An utterance the ears closed: when it began and ended (laptop monotonic seconds), the
    words Whisper read, and whether they were only noise."""

    start_s: float
    end_s: float
    text: str = ""
    noise: bool = False

    @property
    def duration_s(self) -> float:
        """How long the person spoke."""
        return self.end_s - self.start_s


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

    def part_spoken(self, turn: int, index: int) -> None:
        """The voice has fed all of that part's audio to the player: its mark is final."""
        ...

    def reply_cut(self, turn: int) -> None:
        """Reply ``turn`` stopped before its end (a barge-in, or dropped for a merged turn)."""
        ...

    def reply_done(self, turn: int) -> None:
        """Reply ``turn``'s audio has played out."""
        ...

    def moment(self, name: str) -> None:
        """A named moment of the robot (:data:`SLOW_TOOL`), which config/animation.json may map
        to an intent."""
        ...

    def drive(self, active: bool) -> None:
        """A drive the conversation started began (True) or ended (False)."""
        ...

    def closed(self, reason: str) -> None:
        """The conversation closed (idle, a goodbye, end_conversation, a cap, the run's end)."""
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
