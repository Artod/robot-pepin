"""The room's places as the gesture space points at them: a place's name as the brain writes it
in a marker (``{point:printer}``) found in the goal server's places book, and its direction and
distance in the cart's frame from where the robot stands now.

The places book and the robot's map pose are the goal server's (``places`` and ``where`` on its
socket, :class:`pepin.tools.clients.GoalServer`): nothing here holds a truth of its own. The
gesture engine's thread never waits on a socket, so :class:`GoalPlaces` keeps a snapshot that a
thread of its own refreshes (the pose every ``every_s``, the book every ``book_s``); a pose older
than ``stale_s``, or none, gives no direction (wrong localisation points wrong: no gate beyond
that). Until the book has been read once (the goal server never answered), the room knows no
names at all (:meth:`Room.known` False): a name is then neither a place nor a referent.

A name is matched leniently, the brain being told the book's names exactly: as written; then
case, spaces, hyphens and underscores aside; then Cyrillic letters transliterated
(``принтер`` is ``printer``). :class:`Room` is the one seam to the places: the resolver
(:mod:`pepin.expression.render`) reads it, :class:`GoalPlaces` and
:class:`pepin.expression.fakes.FakePlaces` are it.
"""

from __future__ import annotations

import logging
import math
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# Russian letters as Latin ones (a loanword written in Cyrillic finds its place)
_LATIN = dict(
    zip(
        "абвгдеёжзийклмнопрстуфхцчшщъыьэюя",
        ("a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m", "n", "o", "p",
         "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch", "", "y", "", "e", "yu", "ya"),
        strict=True,
    )
)  # fmt: skip
_SEPARATORS = re.compile(r"[\s_\-]+")


class Room(Protocol):
    """The room's places as the gesture space needs them (any thread, never blocking)."""

    def known(self) -> bool:
        """Whether the places book has been read (False: no name can be told apart yet)."""
        ...

    def find(self, name: str) -> str | None:
        """The places book's name that ``name`` means; None: no such place."""
        ...

    def where(self, name: str) -> tuple[float, float] | None:
        """Place ``name``'s bearing in the cart's frame (degrees, + left of the nose) and its
        distance (metres) from where the robot stands now; None: no such place, or no fresh
        pose."""
        ...


class Goals(Protocol):
    """The goal server's two answers read here (:class:`pepin.tools.clients.GoalServer`)."""

    def where(self) -> dict[str, Any]:
        """``pose`` ("tf" or "none"), ``x``, ``y``, ``yaw_deg``, ``age_s``."""
        ...

    def places(self) -> dict[str, dict[str, float]]:
        """``{name: {"x", "y", "yaw_deg"}}`` in the map frame."""
        ...


def key(name: str) -> str:
    """A name with case, spaces, hyphens and underscores aside."""
    return _SEPARATORS.sub("", name.strip().lower())


def latin(name: str) -> str:
    """``key(name)`` with its Cyrillic letters transliterated."""
    return "".join(_LATIN.get(c, c) for c in key(name))


def match(name: str, names: Mapping[str, Any]) -> str | None:
    """The one of ``names`` that ``name`` means: exact, then by :func:`key`, then by
    :func:`latin`; None: none."""
    if name in names:
        return name
    for read in (key, latin):
        wanted = read(name)
        found = [n for n in names if read(n) == wanted]
        if len(found) == 1:
            return found[0]
    return None


def bearing(here: tuple[float, float, float], there: tuple[float, float]) -> tuple[float, float]:
    """``there`` (map x, y) from ``here`` (map x, y, yaw degrees): its bearing in the cart's
    frame (degrees, + left, -180..180) and its distance."""
    x, y, yaw = here
    dx, dy = there[0] - x, there[1] - y
    angle = math.degrees(math.atan2(dy, dx)) - yaw
    return (angle + 180.0) % 360.0 - 180.0, math.hypot(dx, dy)


@dataclass(frozen=True)
class Snapshot:
    """The places book and the robot's map pose (x, y, yaw degrees) as last read, and when
    (the clock's) the pose was measured; no pose: None."""

    places: Mapping[str, tuple[float, float]]
    pose: tuple[float, float, float] | None
    measured: float


class GoalPlaces:
    """:class:`Room` over the goal server (the module's notes)."""

    def __init__(
        self,
        goals: Goals,
        *,
        every_s: float = 1.0,
        book_s: float = 30.0,
        stale_s: float = 5.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        """Nothing is read until :meth:`refresh` (or the thread :meth:`start` runs)."""
        self.goals = goals
        self.every_s, self.book_s, self.stale_s = every_s, book_s, stale_s
        self.clock = clock
        self.snapshot = Snapshot({}, None, -math.inf)
        self._book_read = -math.inf
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def refresh(self) -> None:
        """Read the pose (and the book when it is due) from the goal server; a server that
        does not answer keeps the snapshot as it was (its pose ages out)."""
        now = self.clock()
        places = self.snapshot.places
        try:
            if now - self._book_read >= self.book_s or not places:
                book = self.goals.places()
                places = {n: (float(p["x"]), float(p["y"])) for n, p in book.items()}
                self._book_read = now
            here = self.goals.where()
        except Exception as error:  # a down server is a stale pose, never an error upstream
            logger.debug("places: the goal server did not answer: %s", error)
            return
        pose = None
        measured = now
        if here.get("pose") == "tf" and all(k in here for k in ("x", "y", "yaw_deg")):
            pose = (float(here["x"]), float(here["y"]), float(here["yaw_deg"]))
            measured = now - float(here.get("age_s", 0.0))
        self.snapshot = Snapshot(places, pose, measured)

    def known(self) -> bool:
        """Whether the goal server's book has been read once."""
        return math.isfinite(self._book_read)

    def find(self, name: str) -> str | None:
        """The book's name ``name`` means."""
        return match(name, self.snapshot.places)

    def where(self, name: str) -> tuple[float, float] | None:
        """The place's bearing (degrees, + left) and distance from the robot now."""
        snap = self.snapshot
        found = match(name, snap.places)
        if found is None or snap.pose is None or self.clock() - snap.measured > self.stale_s:
            return None
        return bearing(snap.pose, snap.places[found])

    def start(self) -> GoalPlaces:
        """Refresh on a daemon thread every ``every_s``; returns self so it chains."""

        def run() -> None:
            while not self._stop.is_set():
                self.refresh()
                self._stop.wait(self.every_s)

        self._thread = threading.Thread(target=run, name="places", daemon=True)
        self._thread.start()
        return self

    def close(self) -> None:
        """Stop the thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
