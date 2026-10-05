"""Robot moments on the head's face: the goal server's drives, the voice loops' turns, the gaze's
stall looks and the VIO keeper's restarts.

Each producer names moments, never expressions: config/face.json's ``events`` table (read by the
board's head server) says what each one shows and for how long, so the look of a moment changes
in one file. A producer is one source of the head server's arbiter: what it holds is replaced by
its next moment and cleared when it is done, and a timed moment (a recovery's three seconds)
lapses back to what the source held before it.

:class:`DriveFace` is the goal server's (flag ``face_events``); :class:`VoiceFace` the voice
loop's (``scripts/voice.py --face``); :class:`VoiceStateFace` the Live loop's
(``scripts/voice_live.py``, its states as they change); :class:`StallFace` the gaze's (flag
``face_events``); ``vio_restart`` the VIO keeper's. All speak through
:class:`pepin.head_link.HeadClient`, whose sends never block and are dropped while the head
server is away; the head server drops a moment repeated sooner than its ``min_gap_s``.
"""

from __future__ import annotations

from typing import Protocol

# Nav2's action_msgs/GoalStatus at the end of a drive.
SUCCEEDED, CANCELED, ABORTED = 4, 5, 6
DRIVE_END = {SUCCEEDED: "arrived", CANCELED: "goal_cancelled", ABORTED: "goal_failed"}
LEASE_S = 6.0  # a brain lease's length: renewed every 2 s, three renewals may be lost
# pepin.stall_look.verdict's words as moments: a phantom the look erased, a thing it found.
STALL_VERDICT = {
    "carved": "phantom_carved",
    "partly carved": "phantom_carved",
    "confirmed": "obstacle_confirmed",
}
VIO_RESTART = "vio_restart"
VOICE_STATES = ("listening", "thinking", "speaking")  # shown; the others hand the face back


class FaceSink(Protocol):
    """Where moments go: the head server's door as :class:`pepin.head_link.HeadClient` has it."""

    def event(self, name: str, *, end: bool = False) -> None:
        """A moment of config/face.json's table; ``end`` first clears what this source held."""
        ...

    def clear(self) -> None:
        """This source shows nothing any more."""
        ...

    def lease(self, seconds: float) -> None:
        """This brain is here for ``seconds`` more."""
        ...

    def close(self) -> None:
        """Drop the connection."""
        ...


class DriveFace:
    """A drive as the face sees it: focused from the moment Nav2 takes the goal, struggling for
    a moment at each new recovery, then happy, sad or a flat line by how the drive ended."""

    def __init__(self, sink: FaceSink) -> None:
        """Moments go to ``sink``."""
        self._sink = sink
        self._recoveries = 0
        self.driving = False

    def accepted(self) -> None:
        """Nav2 took the goal."""
        self._recoveries = 0
        self.driving = True
        self._sink.event("goal_accepted")

    def progress(self, recoveries: int) -> None:
        """Nav2's feedback: a recovery count above the last one is a new recovery."""
        if recoveries > self._recoveries:
            self._sink.event("recovery")
        self._recoveries = max(self._recoveries, recoveries)

    def done(self, status: int) -> None:
        """The drive ended with Nav2's ``status``: its moment, and the drive's focus is over."""
        self.driving = False
        moment = DRIVE_END.get(status)
        if moment is None:
            self._sink.clear()
        else:
            self._sink.event(moment, end=True)

    def refused(self) -> None:
        """Nav2 would not take the goal."""
        self.driving = False
        self._sink.event("goal_failed", end=True)

    def abandoned(self) -> None:
        """The drive's report ended without an end (the caller hung up): its focus is over."""
        if self.driving:
            self.driving = False
            self._sink.clear()

    def lease(self) -> None:
        """The goal server is here (its timer, every 2 s)."""
        self._sink.lease(LEASE_S)

    def close(self) -> None:
        """Nothing held any more, and the connection dropped."""
        self._sink.clear()
        self._sink.close()


class VoiceFace:
    """A voice turn as the face sees it: listening while someone speaks, thinking while the
    model works, speaking (the lip sync rides on top) while the answer plays, then nothing."""

    def __init__(self, sink: FaceSink) -> None:
        """Moments go to ``sink``."""
        self._sink = sink

    def listening(self) -> None:
        """An utterance started."""
        self._sink.event("listening")

    def thinking(self) -> None:
        """The utterance is with the model."""
        self._sink.event("thinking")

    def speaking(self) -> None:
        """The answer plays."""
        self._sink.event("speaking")

    def done(self) -> None:
        """The turn is over (answered, or not for the robot)."""
        self._sink.clear()


class HasState(Protocol):
    """A voice event as :mod:`pepin.voice_live.events` sends it (only its state is read)."""

    @property
    def state(self) -> str:
        """idle, listening, thinking, acting or speaking."""
        ...


class VoiceStateFace:
    """The Live voice loop's states on the face, a subscriber of
    :class:`pepin.voice_live.events.Events`: listening, thinking and speaking as their moments,
    each sent once when the state changes (speaking comes with every level sample); acting (a
    drive the voice started: the goal server's face shows it) and idle hand the face back."""

    def __init__(self, sink: FaceSink) -> None:
        """Moments go to ``sink``."""
        self._sink = sink
        self._state: str | None = None

    def __call__(self, event: HasState) -> None:
        """One voice event."""
        if event.state == self._state:
            return
        self._state = event.state
        if event.state in VOICE_STATES:
            self._sink.event(event.state)
        else:
            self._sink.clear()

    def close(self) -> None:
        """Nothing held any more, and the connection dropped."""
        self._sink.clear()
        self._sink.close()


class StallFace:
    """The gaze's stall look on the face: surprised as the head turns to the blocker, then a
    small grin when the look carved a phantom away, worried when it found a thing there."""

    def __init__(self, sink: FaceSink) -> None:
        """Moments go to ``sink``."""
        self._sink = sink

    def looking(self) -> None:
        """The head turns to look at what blocks the hull."""
        self._sink.event("stall_look")

    def verdict(self, word: str) -> None:
        """The look's verdict (pepin.stall_look.verdict); a word without a moment shows none."""
        moment = STALL_VERDICT.get(word)
        if moment is not None:
            self._sink.event(moment)

    def close(self) -> None:
        """Nothing held any more, and the connection dropped."""
        self._sink.clear()
        self._sink.close()
