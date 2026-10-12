"""Fakes of every seam of the expression layer, for tests and dry runs: each one records what it
was asked in ``calls`` and answers what its fields script, with no thread, socket or clock of its
own."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pepin.audio_link import DoaReading
from pepin.expression.animation import Clip, Layer, PartName
from pepin.expression.conversation import Heard
from pepin.expression.executors import Outcome, WheelMove, WheelReport
from pepin.expression.head import ClipLook
from pepin.expression.markers import Part
from pepin.expression.speaker import FaceFrame, GazeAnswer, HeadView
from pepin.expression.trajectory import ArmAnswer, ArmStatus, Trajectory
from pepin.gaze import Aim, Speed

Call = tuple[str, tuple[Any, ...]]


@dataclass
class FakePlayout:
    """The speaker's clock: ``now`` stream seconds played, ``marks`` by (turn, part)."""

    now: float = 0.0
    marks: dict[tuple[int, int], tuple[float, float]] = field(default_factory=dict)

    def played(self) -> float:
        """Stream seconds played."""
        return self.now

    def mark(self, turn: int, index: int) -> tuple[float, float] | None:
        """The part's span, if fed."""
        return self.marks.get((turn, index))


@dataclass
class RecordingObserver:
    """A conversation observer that only records the moments."""

    calls: list[Call] = field(default_factory=list)

    def opened(self, wake: Heard) -> None:
        """Recorded."""
        self.calls.append(("opened", (wake,)))

    def speech_started(self, t: float) -> None:
        """Recorded."""
        self.calls.append(("speech_started", (t,)))

    def speech_ended(self, heard: Heard) -> None:
        """Recorded."""
        self.calls.append(("speech_ended", (heard,)))

    def part_queued(self, turn: int, index: int, part: Part) -> None:
        """Recorded."""
        self.calls.append(("part_queued", (turn, index, part)))

    def voice_started(self, turn: int) -> None:
        """Recorded."""
        self.calls.append(("voice_started", (turn,)))

    def part_spoken(self, turn: int, index: int) -> None:
        """Recorded."""
        self.calls.append(("part_spoken", (turn, index)))

    def reply_cut(self, turn: int) -> None:
        """Recorded."""
        self.calls.append(("reply_cut", (turn,)))

    def reply_done(self, turn: int) -> None:
        """Recorded."""
        self.calls.append(("reply_done", (turn,)))

    def moment(self, name: str) -> None:
        """Recorded."""
        self.calls.append(("moment", (name,)))

    def drive(self, active: bool) -> None:
        """Recorded."""
        self.calls.append(("drive", (active,)))

    def closed(self, reason: str) -> None:
        """Recorded."""
        self.calls.append(("closed", (reason,)))


@dataclass
class FakeFlag:
    """A live switch named ``flag``; ``refusal`` set: every switch is refused with it."""

    flag: str
    on: bool = False
    refusal: str | None = None

    def switch(self, on: bool) -> str | None:
        """Set, unless scripted to refuse."""
        if self.refusal is None:
            self.on = on
        return self.refusal


@dataclass
class FakeExecutor:
    """A body part: ``why_not`` scripts :meth:`ready`; every clip played is kept."""

    part: PartName
    latency_s: float = 0.0
    why_not: str | None = None
    calls: list[Call] = field(default_factory=list)

    def ready(self) -> str | None:
        """As scripted."""
        return self.why_not

    def play(self, clip: Clip) -> Outcome:
        """Recorded; refused when not ready."""
        self.calls.append(("play", (clip,)))
        if self.why_not is not None:
            return Outcome(False, self.why_not)
        return Outcome(True, seconds=clip.duration_s)

    def fade(self, layer: Layer, seconds: float) -> None:
        """Recorded."""
        self.calls.append(("fade", (layer, seconds)))

    def rest(self) -> None:
        """Recorded."""
        self.calls.append(("rest", ()))


@dataclass
class FakeFace:
    """The face: every expression shown is kept."""

    calls: list[Call] = field(default_factory=list)

    def express(
        self,
        name: str,
        *,
        intensity: float = 1.0,
        hold_s: float | None = None,
        transition_ms: float | None = None,
    ) -> None:
        """Recorded."""
        self.calls.append(("express", (name, intensity, hold_s, transition_ms)))


@dataclass
class FakeArmLink:
    """The board's arm: ``state`` and ``answers`` script what it says back."""

    state: ArmStatus | None = None
    answers: dict[str, ArmAnswer] = field(default_factory=dict)
    calls: list[Call] = field(default_factory=list)

    def send(self, trajectory: Trajectory) -> None:
        """Recorded."""
        self.calls.append(("send", (trajectory,)))

    def rest(self) -> None:
        """Recorded."""
        self.calls.append(("rest", ()))

    def halt(self) -> None:
        """Recorded."""
        self.calls.append(("halt", ()))

    def status(self) -> ArmStatus | None:
        """As scripted."""
        return self.state

    def answer(self, trajectory_id: str) -> ArmAnswer | None:
        """As scripted."""
        return self.answers.get(trajectory_id)


@dataclass
class FakeGazeDoor:
    """The arbiter's door for one source: aims are answered ``done`` at the aim asked (or with
    the next of ``refusals``); ``view`` scripts the state; ``live`` what a renewal finds."""

    view: HeadView | None = None
    live: int = 1
    refusals: deque[GazeAnswer] = field(default_factory=deque)
    calls: list[Call] = field(default_factory=list)

    def aim(self, aim: Aim, *, speed: Speed, ttl_s: float) -> GazeAnswer:
        """Recorded; done, unless a refusal is scripted."""
        self.calls.append(("aim", (aim, speed, ttl_s)))
        if self.refusals:
            return self.refusals.popleft()
        return GazeAnswer("done", head=aim)

    def renew(self) -> int:
        """Recorded; as scripted."""
        self.calls.append(("renew", ()))
        return self.live

    def release(self) -> None:
        """Recorded."""
        self.calls.append(("release", ()))

    def state(self) -> HeadView | None:
        """As scripted."""
        return self.view


@dataclass
class FakeHeadDoor:
    """The arbiter's door for head overlays: every overlay plays, unless the next of
    ``refusals`` (a reason, or None for one that plays) is scripted."""

    refusals: deque[str | None] = field(default_factory=deque)
    calls: list[Call] = field(default_factory=list)

    def look(self, request: ClipLook) -> str | None:
        """Recorded; plays, unless a refusal is scripted."""
        self.calls.append(("look", (request,)))
        return self.refusals.popleft() if self.refusals else None

    def withdraw(self, source: str, fade_s: float) -> None:
        """Recorded."""
        self.calls.append(("withdraw", (source, fade_s)))

    def looks(self) -> list[ClipLook]:
        """The overlays asked for, in order."""
        return [args[0] for name, args in self.calls if name == "look"]


@dataclass
class FakeFaceFinder:
    """Frames in the order scripted; None once they run out."""

    frames: deque[FaceFrame | None] = field(default_factory=deque)
    looks: int = 0

    def look(self) -> FaceFrame | None:
        """The next scripted frame."""
        self.looks += 1
        return self.frames.popleft() if self.frames else None


@dataclass
class FakeDoa:
    """Direction readings as scripted (board clock)."""

    readings: list[DoaReading] = field(default_factory=list)

    def latest_doa(self) -> DoaReading | None:
        """The newest scripted reading."""
        return self.readings[-1] if self.readings else None

    def doa_between(self, t0: float, t1: float) -> list[DoaReading]:
        """The scripted readings within the window."""
        return [r for r in self.readings if t0 <= r.stamp_s <= t1]


@dataclass
class FakeWheels:
    """Wheel motions answered with the next scripted report, done in full once they run out."""

    reports: deque[WheelReport] = field(default_factory=deque)
    calls: list[Call] = field(default_factory=list)

    def move(self, request: WheelMove) -> WheelReport:
        """Recorded; as scripted."""
        self.calls.append(("move", (request,)))
        if self.reports:
            return self.reports.popleft()
        return WheelReport(True, turned_deg=request.turn_deg, attempts=1)


if TYPE_CHECKING:  # every fake, and the real clients that already exist, satisfy their seams

    def _conforms() -> None:
        from pepin.audio_link import AudioClient
        from pepin.expression.conversation import ConversationObserver, LiveFlag
        from pepin.expression.door import OverlayDoor
        from pepin.expression.executors import Executor, FaceChannel, NoBody, NoWheels, Wheels
        from pepin.expression.markers import Playout
        from pepin.expression.parts import HeadDoor
        from pepin.expression.speaker import DoaSource, FaceFinder, GazeDoor
        from pepin.expression.trajectory import ArmLink
        from pepin.head_link import HeadClient

        playout: Playout = FakePlayout()
        observer: ConversationObserver = RecordingObserver()
        flag: LiveFlag = FakeFlag("gestures")
        parts: list[Executor] = [FakeExecutor("arm"), NoBody()]
        faces: list[FaceChannel] = [FakeFace(), HeadClient("board", source="voice")]
        arm: ArmLink = FakeArmLink()
        door: GazeDoor = FakeGazeDoor()
        finder: FaceFinder = FakeFaceFinder()
        doa: list[DoaSource] = [FakeDoa(), AudioClient("board")]
        wheels: list[Wheels] = [FakeWheels(), NoWheels()]
        heads: list[HeadDoor] = [FakeHeadDoor(), OverlayDoor()]
        del playout, observer, flag, parts, faces, arm, door, finder, doa, wheels, heads
