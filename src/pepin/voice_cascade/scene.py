"""The body's scene for the brain: what changed since the brain last heard of it, in one line.

The brain knows the body only through its tools and the ``[robot]`` reports, so what the body
does on its own (the speaker gaze turning the cart back to the person after a ``turn`` to the
printer) leaves it believing the old scene. Each brain input therefore begins with a
``[scene]`` line naming the fields that changed since the last message the brain's history
kept, and none when nothing did (the brain's history is the log of the changes):

    [scene] heading 95; position 1.2, 3.4; ahead: printer; person: left 40 deg, seen; standing still

Every field is a coarse bucket with hysteresis, so the cart's breathing and the wheels' wobble
(a heading of 95, then 95.3) never make one: the heading (where_am_i's, degrees from the map's
x axis, counter-clockwise) in sectors ``sector_deg`` wide, sector k centred on k times that,
changed only past the sector's edge by more than ``hysteresis_deg``; the position in cells of
``cell_m`` with the place the robot stands at (where_am_i's ``at``), changed only
:data:`CELL_HYSTERESIS_M` past a cell's edge; ``ahead``: the nearest named place within
:data:`AHEAD_DEG` of the nose and under :data:`AHEAD_M` (not one underfoot), else the person
standing there, else nothing named, the one told last kept until it is past those bounds by the
hysteresis; the person engaged (the speaker gaze's snapshot): the side and sector of their
bearing, and whether the camera saw them within :data:`SEEN_S`; and whether the cart drives,
turns or stands. The first message of a conversation (or of a brain switched in) carries every
field known; a field unknown then is left out, and one lost later is told as unknown.

A report becomes what the brain knows only once its turn is kept in the brain's history
(committed, or cut by a barge-in: :meth:`SceneStatus.keep`); a turn dropped or failed leaves the
brain as it was, and the next message is told against the same state.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Hashable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from pepin.expression.places import bearing
from pepin.tools.drive import AT_PLACE_M

if TYPE_CHECKING:
    from pepin.expression.speaker import Attention

TAG = "[scene]"
SCENE_STATUS = "scene_status"  # the live switch's name in control.json's flags
AHEAD_DEG = 20.0  # a place or the person this near the nose is ahead
AHEAD_M = 2.5  # ... and a place under this far
CELL_HYSTERESIS_M = 0.1  # a cell (and ahead's distance) changes only this far past its edge
SEEN_S = 3.0  # the person is "seen" when the camera found their face or body this recently
UNTRACKED = ("idle", "paused")  # the gaze's phases with nobody engaged (a drive pauses it)
PERSON = "the person"
NOTHING = "nothing named"
MOVES = {"driving": "driving", "turning": "turning in place", "still": "standing still"}

Pose = tuple[float, float, float]  # map x, y (metres) and heading (degrees)
AttentionReader = Callable[[], "Attention"]


@dataclass(frozen=True)
class SceneKnobs:
    """The buckets' sizes (``config/voice_cascade.json`` conversation.scene_*)."""

    sector_deg: float = 15.0
    hysteresis_deg: float = 5.0
    cell_m: float = 0.25


@dataclass(frozen=True)
class Reading:
    """The body at one moment as its owners report it: the map pose and the place it stands at
    (where_am_i), the places book's positions, the speaker gaze's snapshot (None: no gaze in
    this run), whether a drive runs, and the clock the snapshot's times are on."""

    pose: Pose | None = None
    at: str | None = None
    places: Mapping[str, tuple[float, float]] = field(default_factory=dict)
    attention: Attention | None = None
    driving: bool = False
    now: float = 0.0


@dataclass(frozen=True)
class Field:
    """One field's bucket (equal keys: no change; None: unknown) and how it reads."""

    key: Hashable
    text: str


@dataclass(frozen=True, eq=False)
class Report:
    """The scene at one brain input: the changed fields' line ("" when none changed), every
    field's bucket, and whom it was told to (the conversation and the brain)."""

    line: str
    state: Mapping[str, Field]
    conversation: str
    brain: object


class SceneStatus:
    """The ``[scene]`` line of each brain input (the module's notes), and its live switch
    :data:`SCENE_STATUS` (:class:`pepin.expression.conversation.LiveFlag`)."""

    def __init__(self, knobs: SceneKnobs, *, on: bool = True) -> None:
        """Nothing told yet: the first report carries every field known."""
        self.knobs = knobs
        self._on = on
        self._told: Report | None = None  # the last report the brain's history kept

    @property
    def flag(self) -> str:
        """``scene_status``."""
        return SCENE_STATUS

    @property
    def on(self) -> bool:
        """Whether brain inputs carry the line."""
        return self._on

    def switch(self, on: bool) -> str | None:
        """On or off; turned on, the next report carries every field again."""
        if on and not self._on:
            self._told = None
        self._on = on
        return None

    def report(self, reading: Reading, conversation: str, brain: object) -> Report:
        """The scene now against what ``brain`` was last told in ``conversation``: the fields
        whose bucket changed, or every one known when it was told nothing yet."""
        told = self._told
        fresh = told is None or told.conversation != conversation or told.brain is not brain
        base: Mapping[str, Field] = {} if fresh or told is None else told.state
        state = scene_of(reading, base, self.knobs)
        texts = [
            f.text
            for name, f in state.items()
            if (base[name].key != f.key if name in base else f.key is not None)
        ]
        line = f"{TAG} {'; '.join(texts)}" if texts else ""
        return Report(line, state, conversation, brain)

    def keep(self, report: Report) -> None:
        """The brain's history kept the message ``report`` went with: it knows that scene."""
        self._told = report


def scene_of(reading: Reading, base: Mapping[str, Field], knobs: SceneKnobs) -> dict[str, Field]:
    """Every field's bucket for ``reading``, each held at ``base``'s while within its
    hysteresis; the person's only with a gaze in the run."""
    held = {name: f.key for name, f in base.items()}
    fields = {
        "heading": _heading(reading, held.get("heading"), knobs),
        "position": _position(reading, held.get("position"), knobs),
        "ahead": _ahead(reading, held.get("ahead"), knobs),
    }
    person = _person(reading, held.get("person"), knobs)
    if person is not None:
        fields["person"] = person
    driving, view = reading.driving, reading.attention
    moving = "driving" if driving else "turning" if view and view.phase == "turning" else "still"
    fields["moving"] = Field(moving, MOVES[moving])
    # an unchanged bucket keeps the field told (its text read when it changed)
    return {
        name: base[name] if name in base and base[name].key == f.key else f
        for name, f in fields.items()
    }


def sector(deg: float, width: float, hysteresis: float, held: Hashable = None) -> int:
    """The sector of ``deg`` (sector k centred on k * ``width``); ``held``, the sector told,
    is kept until ``deg`` is past its edge by more than ``hysteresis``."""
    if isinstance(held, int) and abs(_wrap(deg - held * width)) <= width / 2 + hysteresis:
        return held
    return round(deg / width) % round(360.0 / width)


def cell(value: float, size: float, hysteresis: float, held: Hashable = None) -> int:
    """The cell of ``value`` (cell i spans i * ``size`` to (i + 1) * ``size``); ``held``, the
    cell told, is kept until ``value`` is past its edge by more than ``hysteresis``."""
    if (
        isinstance(held, int)
        and held * size - hysteresis <= value <= (held + 1) * size + hysteresis
    ):
        return held
    return math.floor(value / size)


def pose_of(answer: Mapping[str, Any]) -> tuple[Pose | None, str | None]:
    """where_am_i's answer as the map pose and the place the robot stands at; (None, None)
    when it failed or has no pose."""
    keys = ("x", "y", "heading_deg")
    if not answer.get("ok") or not all(isinstance(answer.get(k), (int, float)) for k in keys):
        return None, None
    x, y, heading = (float(answer[k]) for k in keys)
    at = answer.get("at")
    return (x, y, heading), (str(at) if at else None)


def book_of(rows: object) -> dict[str, tuple[float, float]]:
    """list_places' rows as ``{name: (x, y)}``; a row without a position is left out."""
    book: dict[str, tuple[float, float]] = {}
    for row in rows if isinstance(rows, list) else []:
        if isinstance(row, dict) and "name" in row:
            x, y = row.get("x"), row.get("y")
            if isinstance(x, (int, float)) and isinstance(y, (int, float)):
                book[str(row["name"])] = (float(x), float(y))
    return book


def _heading(reading: Reading, held: Hashable, knobs: SceneKnobs) -> Field:
    if reading.pose is None:
        return Field(None, "heading unknown")
    deg = reading.pose[2]
    return Field(sector(deg, knobs.sector_deg, knobs.hysteresis_deg, held), f"heading {deg:.0f}")


def _position(reading: Reading, held: Hashable, knobs: SceneKnobs) -> Field:
    if reading.pose is None:
        return Field(None, "position unknown")
    x, y, _ = reading.pose
    i, j = held[:2] if isinstance(held, tuple) else (None, None)
    size = knobs.cell_m
    key = (cell(x, size, CELL_HYSTERESIS_M, i), cell(y, size, CELL_HYSTERESIS_M, j), reading.at)
    at = f" (at {_name(reading.at)})" if reading.at else ""
    return Field(key, f"position {x:.1f}, {y:.1f}{at}")


def _ahead(reading: Reading, held: Hashable, knobs: SceneKnobs) -> Field:
    """The nearest named place ahead (the one told kept while within the hysteresis), else the
    person, else nothing named; unknown without a pose unless the person stands ahead."""
    pose, slack_deg = reading.pose, knobs.hysteresis_deg
    if pose is not None:
        near: list[tuple[float, str]] = []
        for name, point in reading.places.items():
            angle, distance = bearing(pose, point)
            deg, m = (slack_deg, CELL_HYSTERESIS_M) if name == held else (0.0, 0.0)
            if abs(angle) <= AHEAD_DEG + deg and AT_PLACE_M - m < distance < AHEAD_M + m:
                near.append((distance, name))
        if near:
            chosen = held if any(name == held for _, name in near) else min(near)[1]
            return Field(chosen, f"ahead: {_name(str(chosen))}")
    person = _tracked(reading)
    if person is not None and abs(person) <= AHEAD_DEG + (slack_deg if held == PERSON else 0.0):
        return Field(PERSON, f"ahead: {PERSON}")
    if pose is None:
        return Field(None, "ahead: unknown")
    return Field(NOTHING, f"ahead: {NOTHING}")


def _person(reading: Reading, held: Hashable, knobs: SceneKnobs) -> Field | None:
    """The person engaged: the sector of their bearing and whether they were seen; not tracked
    while the gaze holds nobody; None without a gaze in the run."""
    view = reading.attention
    if view is None:
        return None
    deg = _tracked(reading)
    if deg is None:
        return Field(None, "person: not tracked")
    was = held[0] if isinstance(held, tuple) else None
    k = sector(deg, knobs.sector_deg, knobs.hysteresis_deg, was)
    seen = view.seen_s is not None and reading.now - view.seen_s <= SEEN_S
    count = round(360.0 / knobs.sector_deg)
    if k == 0:
        side = "ahead"
    elif 2 * k == count:
        side = "behind"
    else:
        side = f"{'left' if deg > 0 else 'right'} {abs(deg):.0f} deg"
    return Field((k, seen), f"person: {side}, {'seen' if seen else 'not seen'}")


def _tracked(reading: Reading) -> float | None:
    """The engaged person's bearing (degrees, + left of the nose); None: nobody engaged."""
    view = reading.attention
    if view is None or view.phase in UNTRACKED or view.bearing_deg is None:
        return None
    return float(view.bearing_deg)


def _name(name: str | None) -> str:
    """A place's name on one line."""
    return " ".join(str(name).split())


def _wrap(deg: float) -> float:
    """An angle wrapped to -180..180 degrees."""
    return (deg + 180.0) % 360.0 - 180.0
