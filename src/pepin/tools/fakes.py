"""Fakes of every owner the tools speak to, and a robot made of them.

They stand in for the services in the tests, in the chat loop run without a robot
(``scratch/llm/chat.py --fake``), and for the services that do not exist yet. Each keeps what it
was asked, so a test can say what a tool did; none of them touches a socket. Time is a
:class:`FakeClock`: a tool's two-second wait takes no time at all.
"""

from __future__ import annotations

import math
import time
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from pepin.gaze import DRIVE_REFUSAL
from pepin.tools.clients import HeadMove, HeadPose, HeadReach, ServiceDownError
from pepin.tools.registry import Image, ToolError
from pepin.tools.robot import Robot

REST = HeadPose(0.0, 23.8)  # config/neck.json's working pose
REACH = HeadReach(pan_right_deg=-156.0, pan_left_deg=155.0, tilt_up_deg=-20.0, tilt_down_deg=92.0)
FIELD_OF_VIEW_DEG = 70.0  # a thing is in the picture within half of this of where the head points
PLACES = {
    "home": {"x": 0.0, "y": 0.0, "yaw_deg": 0.0},
    "printer": {"x": -1.2, "y": 3.4, "yaw_deg": 90.0},
    "bookshelf": {"x": 2.5, "y": -0.8, "yaw_deg": 180.0},
}


class FakeClock:
    """Seconds that pass only when someone sleeps."""

    def __init__(self) -> None:
        """Time zero."""
        self.now = 0.0

    def __call__(self) -> float:
        """The time."""
        return self.now

    def sleep(self, seconds: float) -> None:
        """Let ``seconds`` pass."""
        self.now += max(0.0, seconds)


class WallClock(FakeClock):
    """Real seconds since construction, for fakes driven in real time (a voice test with
    ``scripts/voice_live.py --fake-robot``): a sleep really waits."""

    def __init__(self) -> None:
        """Time zero is now."""
        super().__init__()
        self._start = time.monotonic()

    def __call__(self) -> float:
        """The seconds since construction."""
        self.now = time.monotonic() - self._start
        return self.now

    def sleep(self, seconds: float) -> None:
        """Wait ``seconds``."""
        time.sleep(max(0.0, seconds))
        self()


@dataclass
class FakeGoalServer:
    """The goal server: a pose, a book of places, and the events of the next drive.

    ``drive`` is the script of a ``go``: its events are yielded one per ``step_s`` of the clock
    (the first, ``accepted``, after ``accept_s`` when set); after a ``cancel`` the drive ends
    with Nav2's CANCELED. ``refuse`` makes a go answer only an
    error with that detail, as the server does for an unknown place or a stale pose.
    """

    clock: FakeClock
    pose: dict[str, Any] | None = field(
        default_factory=lambda: {"x": 0.1, "y": 0.05, "yaw_deg": 3.0, "age_s": 0.05}
    )
    book: dict[str, dict[str, float]] = field(default_factory=lambda: dict(PLACES))
    drive: list[dict[str, Any]] | None = None
    refuse: str | None = None
    step_s: float = 1.0
    accept_s: float | None = None
    down: bool = False
    asked: list[dict[str, Any]] = field(default_factory=list)
    cancelled: int = 0

    def where(self) -> dict[str, Any]:
        """``where``: the pose, or ``pose: none``."""
        self._ask({"cmd": "where"})
        if self.pose is None:
            return {"event": "where", "pose": "none", "planner": "hybrid", "lidar": "ok"}
        return {"event": "where", "pose": "tf", "planner": "hybrid", "lidar": "ok", **self.pose}

    def places(self) -> dict[str, dict[str, float]]:
        """``places``: the book."""
        self._ask({"cmd": "places"})
        return dict(self.book)

    def go(self, request: dict[str, Any]) -> Iterator[dict[str, Any]]:
        """``go``: the refusal, the scripted drive, or a plain arrival at the goal."""
        self._ask(request)
        if self.refuse is not None:
            yield {"event": "error", "detail": self.refuse}
            return
        goal = self._goal(request)
        if goal is None:
            yield {"event": "error", "detail": f"no such place: {request.get('place')!r}"}
            return
        before = self.cancelled
        for event in self.drive if self.drive is not None else arrival_script(*goal):
            if self.cancelled > before:
                yield {
                    "event": "done",
                    "status": 5,
                    "seconds": self.clock.now,
                    "arrival": self.pose,
                }
                return
            wait = self.step_s
            if event.get("event") == "accepted" and self.accept_s is not None:
                wait = self.accept_s
            self.clock.sleep(wait)
            if event.get("event") == "done" and isinstance(event.get("arrival"), dict):
                self.pose = dict(event["arrival"])
            yield event

    def cancel(self) -> dict[str, Any]:
        """``cancel``: every navigator answers."""
        self._ask({"cmd": "cancel"})
        self.cancelled += 1
        return {
            "event": "cancelled",
            "had_goal": True,
            "navigators": {
                "navigate_to_pose": {"outcome": "accepted", "cancelling": 1},
                "navigate_through_poses": {"outcome": "no such goal", "cancelling": 0},
            },
        }

    def _ask(self, request: dict[str, Any]) -> None:
        self.asked.append(request)
        if self.down:
            raise ServiceDownError(
                "goal server", "127.0.0.1:3337", "connection refused", "Is it up?"
            )

    def _goal(self, request: dict[str, Any]) -> tuple[float, float, float, str | None] | None:
        if "place" in request:
            place = self.book.get(str(request["place"]))
            if place is None:
                return None
            return place["x"], place["y"], place["yaw_deg"], str(request["place"])
        return float(request["x"]), float(request["y"]), float(request["yaw_deg"]), None


def arrival_script(x: float, y: float, yaw_deg: float, place: str | None) -> list[dict[str, Any]]:
    """The events of a drive that arrives: accepted, two progress reports, done."""
    return [
        {"event": "accepted", "run": 7, "place": place, "x": x, "y": y, "yaw_deg": yaw_deg},
        {"event": "feedback", "t": 1.0, "distance": 2.0, "recoveries": 0},
        {"event": "feedback", "t": 2.0, "distance": 0.4, "recoveries": 0},
        {
            "event": "done",
            "run": 7,
            "status": 4,
            "seconds": 3.0,
            "arrival": {"x": x + 0.05, "y": y - 0.03, "yaw_deg": yaw_deg + 2.0},
        },
    ]


@dataclass
class FakeNeck:
    """The head as the gaze arbiter lends it: moves arrive at once, ``wheels_moving`` (a drive)
    refuses them in the arbiter's words, and ``kept`` counts the renewals ``see`` asks for."""

    clock: FakeClock
    pose_now: HeadPose = REST
    wheels_moving: bool = False
    down: bool = False
    moves: list[tuple[float | None, float | None]] = field(default_factory=list)
    kept: int = 0

    def reach(self) -> HeadReach:
        """The real neck's reach, rounded."""
        return REACH

    def rest(self) -> HeadPose:
        """The working pose."""
        return REST

    def pose(self) -> HeadPose:
        """Where the head points."""
        self._check()
        return self.pose_now

    def turn(self, pan_deg: float | None, tilt_deg: float | None) -> HeadMove:
        """Arrive at once, one second later, or be refused during a drive."""
        self._check()
        self.moves.append((pan_deg, tilt_deg))
        if self.wheels_moving:
            return HeadMove(False, self.pose_now, DRIVE_REFUSAL)
        self.clock.sleep(1.0)
        self.pose_now = HeadPose(
            self.pose_now.pan_deg if pan_deg is None else pan_deg,
            self.pose_now.tilt_deg if tilt_deg is None else tilt_deg,
        )
        return HeadMove(True, self.pose_now, "", 1000.0)

    def keep(self) -> None:
        """Count the renewal."""
        self._check()
        self.kept += 1

    def _check(self) -> None:
        if self.down:
            raise ServiceDownError(
                "gaze arbiter", "http://127.0.0.1:3339", "refused", "Is Nav2 up on this Mac?"
            )


@dataclass
class FakeThing:
    """Something in the flat the camera can see: from which pan of the head, between which
    times of the clock, where, and how sure the detector is."""

    label: str
    pan_deg: float = 0.0
    range_m: float = 1.5
    bearing_deg: float = 0.0
    x: float = 1.0
    y: float = 0.5
    z: float = 0.4
    score: float = 0.8
    from_s: float = 0.0
    until_s: float = math.inf


@dataclass
class FakeWorld:
    """The memory: ``things`` are seen whenever the head points at them (sightings), ``objects``
    are what it remembers, ``tree_text`` its tree; a ``remember`` is kept in ``remembered``."""

    clock: FakeClock
    neck: FakeNeck
    things: list[FakeThing] = field(default_factory=list)
    objects_known: list[dict[str, Any]] = field(default_factory=list)
    tree_text: str = "flat\n  living room: sofa (1.0, 2.0), printer place (-1.2, 3.4)"
    down: bool = False
    remembered: list[dict[str, Any]] = field(default_factory=list)
    asked: list[tuple[str, Any]] = field(default_factory=list)

    def latest(self, label: str | None, within_s: float) -> list[dict[str, Any]]:
        """Sightings of the last ``within_s`` seconds by the head's present direction."""
        self._check(("latest", label))
        now, pan = self.clock.now, self.neck.pose_now.pan_deg
        return [
            {
                "label": t.label,
                "score": t.score,
                "x": t.x,
                "y": t.y,
                "z": t.z,
                "range": t.range_m,
                "bearing": math.radians(t.bearing_deg),
            }
            for t in self.things
            if (label is None or t.label == label)
            and abs(t.pan_deg - pan) <= FIELD_OF_VIEW_DEG / 2
            and t.from_s <= now
            and t.until_s >= now - within_s
        ]

    def objects(self, label: str | None = None) -> list[dict[str, Any]]:
        """The remembered objects of that label."""
        self._check(("objects", label))
        return [o for o in self.objects_known if label is None or o.get("label") == label]

    def tree(self) -> dict[str, Any]:
        """The tree."""
        self._check(("tree", None))
        return {"text": self.tree_text, "json": {}}

    def remember(self, entry: dict[str, Any]) -> dict[str, Any]:
        """Keep the entry."""
        self._check(("remember", entry))
        self.remembered.append(entry)
        return {"ok": True, **entry}

    def health(self) -> dict[str, Any]:
        """Up."""
        self._check(("health", None))
        return {"ok": True, "objects": len(self.objects_known)}

    def _check(self, what: tuple[str, Any]) -> None:
        self.asked.append(what)
        if self.down:
            raise ServiceDownError(
                "world", "http://127.0.0.1:8798", "connection refused", "Not running."
            )


@dataclass
class FakeCamera:
    """A camera whose every picture is the same small JPEG."""

    jpeg: bytes = b"\xff\xd8\xff\xe0fake-jpeg\xff\xd9"
    down: bool = False

    def snapshot(self) -> Image:
        """The picture."""
        if self.down:
            raise ServiceDownError(
                "camera", "http://10.0.0.187:8080", "timed out", "Is the board up?"
            )
        return Image(self.jpeg, "image/jpeg", 800, 600)

    def health(self) -> dict[str, Any]:
        """Streaming."""
        return {"online": True, "fps": 15}


@dataclass
class FakeSpeech:
    """A voice that writes down what it said (60 ms a character)."""

    said: list[str] = field(default_factory=list)
    down: bool = False

    def say(self, text: str) -> float:
        """Keep the text."""
        if self.down:
            raise ServiceDownError("audio server", "10.0.0.187:3338", "refused", "Is it up?")
        self.said.append(text)
        return 0.06 * len(text)

    def health(self) -> dict[str, Any]:
        """Up, unless down."""
        if self.down:
            raise ServiceDownError("audio server", "10.0.0.187:3338", "refused", "Is it up?")
        return {"type": "status", "playing": False}


@dataclass
class FakeFace:
    """The head server's face: what was expressed and shown, in order."""

    expressed: list[tuple[str, float]] = field(default_factory=list)
    shown: list[tuple[str, float]] = field(default_factory=list)
    down: bool = False

    def express(self, name: str, seconds: float) -> dict[str, Any]:
        """Keep it; an unknown name is refused as the head server refuses it."""
        self._check()
        from pepin.face import load_face_table

        table = load_face_table()
        if name not in table.names:
            raise ToolError(f"the head server refused it: no expression {name!r}")
        self.expressed.append((name, seconds))
        return {"type": "ack", "cmd": "express", "showing": name, "by": "llm"}

    def show(self, text: str, seconds: float) -> dict[str, Any]:
        """Keep it."""
        self._check()
        self.shown.append((text, seconds))
        return {"type": "ack", "cmd": "show", "items": len(text.splitlines()), "sent": True}

    def health(self) -> dict[str, Any]:
        """Up, unless down."""
        self._check()
        return {"link": "up", "showing": "neutral", "face_fps": 49.5, "imu_hz": 1000}

    def _check(self) -> None:
        if self.down:
            raise ServiceDownError("head server", "10.0.0.187:3340", "refused", "Is it up?")


@dataclass
class FakeBody:
    """The base's servo temperatures."""

    temps: dict[str, int] = field(default_factory=lambda: {"left": 41, "right": 39})
    down: bool = False

    def temperatures(self) -> dict[str, int]:
        """The temperatures, unless down."""
        if self.down:
            raise ServiceDownError("base server", "10.0.0.187:3336", "refused", "Is it up?")
        return dict(self.temps)


def fake_robot(**replace: Any) -> Robot:
    """A robot made of fakes sharing one :class:`FakeClock`; ``replace`` swaps any part
    (``goals``, ``neck``, ``world``, ``camera``, ``speech``, ``face``, ``body``) or sets
    ``drive_timeout_s``."""
    clock = replace.pop("clock", None) or FakeClock()
    neck = replace.pop("neck", None) or FakeNeck(clock)
    parts: dict[str, Any] = {
        "goals": FakeGoalServer(clock),
        "neck": neck,
        "world": FakeWorld(clock, neck),
        "camera": FakeCamera(),
        "speech": FakeSpeech(),
        "face": FakeFace(),
        "body": FakeBody(),
    }
    parts.update(replace)
    return Robot(clock=clock, sleep=clock.sleep, **parts)
