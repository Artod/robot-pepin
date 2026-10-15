"""Turning the body where it stands: to the person engaged, left or right, by an angle, or toward
a named place.

One request to the goal server, ``spin`` (Nav2's Spin behaviour: a turn in place on odometry,
measured on the map's heading; the goal server's flag ``spin_requests``) with the source
``tool``, driven and reported by :func:`pepin.tools.drive.drive` like any drive: the cascade
answers the model once the turn is under way, and its end comes as the robot's own report.
Nothing here goes anywhere: coming closer or standing somewhere is ``approach``'s.

The person is the speaker gaze's (``robot.gaze``). Before anything turns, the gaze is asked to
keep them (:meth:`pepin.tools.clients.Facing.keep_person`: the cascade has just paused it for
this call, as for any drive, and the pause is given back), and its snapshot gives their bearing
in the cart's frame; while the body comes round the head stays on them. A place's bearing comes
from the goal server's book and pose, as approach reads them; standing at the place, its own
heading.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

from pepin.tools.drive import AT_PLACE_M, drive, pose_on_map
from pepin.tools.registry import Result, ToolError, fail, ok, tool
from pepin.tools.robot import Robot

if TYPE_CHECKING:  # numpy's modules behind it: the gaze's snapshot is only read here
    from pepin.expression.speaker import Attention

SPEAKER = ("me", "speaker", "person")  # approach's words for the person speaking
SIDES = {"left": 1.0, "right": -1.0}
SIDE_DEG = 90.0  # "left" or "right" without an angle
FACING_DEG = 10.0  # facing someone this nearly is facing them (the goal checker's 11.5)
MOST_DEG = 360.0  # the goal server's spin turns at most a whole turn either way
SEEN_FRESH_S = 3.0  # a person seen within this was placed by sight, else by their voice
# A person's ask: by the goal server's one rule it cancels the speaker gaze's own turn, and the
# gaze's turn never cancels it.
SOURCE = "tool"

Asked = tuple[float | None, dict[str, Any], str]  # the turn (None: none needed), found, words


@tool(moves=True)
def turn(robot: Robot, target: str, degrees: float | None = None) -> Result:
    """Turn the body in place, without driving anywhere: to face the person speaking ("повернись
    ко мне"), left or right ("повернись налево", "направо на 45 градусов"), by an angle, or
    toward a named place ("повернись к принтеру"). The head stays on the person while the body
    comes round. For "turn to me" this is the tool, not approach: approach is for coming closer
    or standing somewhere. A few seconds, reported when it ends as a drive is.

    Args:
        target: "me" for the person speaking; "left" or "right"; an angle in degrees as a number
            ("45", "-30": positive is to the robot's left, at most 360); or a place name exactly
            as list_places gives it.
        degrees: only with left or right, when the person names an angle (90 when not given).
    """
    words = target.strip()
    kept = robot.gaze.keep_person(f"turn {words}") if robot.gaze is not None else None
    deg, found, label = resolve(robot, words, degrees, kept)
    if deg is None:
        return ok(turned=False, detail=f"already facing {label}: nothing moved", found=found)
    result = drive(robot, {"cmd": "spin", "deg": round(deg, 1), "source": SOURCE}, label)
    if "unknown command" in str(result.get("why", "")):
        return fail(
            "the goal server cannot turn in place yet (started before its spin command):"
            " restart it (ros/laptop.sh kick goal_server); nothing moved"
        )
    if not result.get("already"):  # the same turn under way: its end tells whether it turned
        result["turned"] = bool(result.pop("arrived", False))
    result["asked_deg"] = round(deg, 1)
    result["found"] = found
    if isinstance(kept, str):
        result["head"] = f"not kept on anyone: {kept}"
    elif kept is not None:
        result["head"] = "kept on the person"
    return result


def resolve(robot: Robot, words: str, degrees: float | None, kept: Attention | str | None) -> Asked:
    """The turn the model asked for, degrees + left (None when already facing it), what was
    found, and its words for the drive; :class:`ToolError` saying why not. ``kept`` is the
    speaker gaze's snapshot once it keeps the person, or why it does not (None: no gaze)."""
    lowered = words.lower()
    if degrees is not None and lowered not in SIDES:
        raise ToolError(
            "degrees goes with left or right only; give any other angle as the target, e.g."
            ' "-30"; nothing moved'
        )
    if lowered in SIDES:
        size = SIDE_DEG if degrees is None else abs(degrees)
        return _checked(SIDES[lowered] * size), {"kind": "side"}, f"{lowered} {size:g} deg"
    try:
        number: float | None = float(words)
    except ValueError:
        number = None
    if number is not None:
        return _checked(number), {"kind": "angle"}, f"{number:+g} deg"
    if lowered in SPEAKER:
        return _person(robot, kept)
    book = robot.goals.places()
    names = {name.lower(): name for name in book}
    if lowered in names:
        return _place(robot, names[lowered], book[names[lowered]])
    known = ", ".join(sorted(book)) or "none yet"
    raise ToolError(
        f"target {words!r} is none of: 'me' (the person speaking), 'left' or 'right', an angle"
        f" in degrees, or a place name ({known}); nothing moved"
    )


def _checked(deg: float) -> float:
    """``deg`` when the goal server would turn it; :class:`ToolError` otherwise."""
    if not 1.0 <= abs(deg) <= MOST_DEG:
        raise ToolError(
            f"an angle must be 1 to {MOST_DEG:g} degrees either way, not {deg:g}; nothing moved"
        )
    return deg


def _person(robot: Robot, kept: Attention | str | None) -> Asked:
    """The person the speaker gaze keeps, by their bearing in the cart's frame."""
    if kept is None or isinstance(kept, str):
        why = kept or "no speaker gaze runs here"
        raise ToolError(
            f"cannot tell where the person is ({why}); nothing moved: ask them to say something"
            " and try again, or turn by an angle"
        )
    if kept.bearing_deg is None:
        raise ToolError("nobody engaged: ask the person to say something first; nothing moved")
    bearing = _wrap(kept.bearing_deg)
    seen = kept.seen_s is not None and robot.clock() - kept.seen_s <= SEEN_FRESH_S
    found = {"kind": "person", "bearing_deg": round(bearing), "by": "sight" if seen else "voice"}
    return (None if abs(bearing) <= FACING_DEG else bearing), found, "the person"


def _place(robot: Robot, name: str, place: dict[str, float]) -> Asked:
    """A place of the book: its bearing from where the robot stands; standing there (within
    :data:`pepin.tools.drive.AT_PLACE_M`), the place's own heading, which faces its thing."""
    here = pose_on_map(robot)
    x, y, yaw = float(here["x"]), float(here["y"]), float(here.get("yaw_deg", 0.0))
    px, py = float(place["x"]), float(place["y"])
    distance = math.hypot(px - x, py - y)
    if distance <= AT_PLACE_M:
        heading = float(place.get("yaw_deg", yaw))
    else:
        heading = math.degrees(math.atan2(py - y, px - x))
    deg = _wrap(heading - yaw)
    found = {"kind": "place", "name": name, "distance_m": round(distance, 2)}
    return (None if abs(deg) <= FACING_DEG else deg), found | {"bearing_deg": round(deg)}, name


def _wrap(deg: float) -> float:
    """An angle in -180..180 degrees."""
    return (deg + 180.0) % 360.0 - 180.0
