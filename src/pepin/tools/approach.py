"""Going up to someone or something and standing in a relation to it.

One tool and two words from the model, the target and the relation; the rest is the robot's.
The target becomes a map point — the speaker's face through :mod:`pepin.pixel_to_map`
(``robot.eyes``), a place from the book, or a box of the last ``see`` picture — and the relation a
standing pose with two alternatives (:mod:`pepin.stand_pose`). Each is asked of the goal server's
``plan`` (RTAB-Map's map cell and one plan: never motion, because the navigator answers a refused
goal by turning and backing up) until one would be driven, and that ONE goal is driven by
:func:`pepin.tools.drive.drive` exactly as ``go_to``'s. On arrival the head turns to the target
through the ``look`` path. Nothing here follows anyone: one goal per call.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pepin.stand_pose import Distances, Relation, Stand, Target, facing_from, stands
from pepin.tools.drive import drive, pose_on_map
from pepin.tools.registry import Result, ToolError, fail, tool
from pepin.tools.robot import Robot

if TYPE_CHECKING:  # numpy's module, loaded when a target is located
    from pepin.pixel_to_map import Box, Located, Pose2D

SPEAKER = ("me", "speaker", "person")  # the target words that mean the person speaking


@dataclass(frozen=True)
class Resolved:
    """A target the model named, as the chooser takes it: its frame, what to report about it,
    the point the head turns to on arrival (None for a place), and its words for the drive."""

    target: Target
    found: dict[str, Any]
    point: tuple[float, float, float] | None
    label: str


@tool(moves=True)
def approach(
    robot: Robot, target: str, relation: Relation = "to", distance_m: float | None = None
) -> Result:
    """Drive up to a person, a named place or a thing in the picture and stand in relation to
    it, then turn the head to it: "come to me", "stand to my right", "go behind the chair", "go
    and look at the board". The robot measures where the target is, picks a free spot and makes
    ONE drive (up to 4 minutes), reported when it ends as go_to's is. Never nearer than 0.6 m
    to a person. For a plain named place go_to is enough.

    Args:
        target: "me" for the person speaking (the face the robot sees; if it sees none, ask
            them to face it); a place name exactly as list_places gives it; or a thing in the
            picture of your last see() as its box "x0,y0,x1,y1" in pixels of that picture
            (call see() again first if the head has turned since).
        relation: to (about 0.8 m away, facing it), right_of or left_of (a person's own right
            or left; a thing's as the robot sees it), in_front_of, beside (the nearer side),
            behind.
        distance_m: only when the person names a distance, in metres.
    """
    distances = Distances.load()
    if distance_m is not None and not 0.2 <= distance_m <= 3.0:
        return fail(f"distance_m must be between 0.2 and 3 m, not {distance_m:g}")
    here = _pose(robot)
    resolved = resolve(robot, target, here, distances)
    candidates = stands(resolved.target, relation, here, distances, distance_m)
    index, stand, refusals = first_plannable(robot, candidates)
    if stand is None:
        return fail(
            f"nowhere to stand {relation} {resolved.label}: " + "; ".join(refusals) + "."
            " Nothing moved; the person may say where to stand instead",
            found=resolved.found,
        )
    request = {
        "cmd": "go",
        "x": round(stand.x, 3),
        "y": round(stand.y, 3),
        "yaw_deg": round(stand.yaw_deg, 1),
    }
    result = drive(robot, request, f"{relation} {resolved.label}")
    result["found"] = resolved.found
    result["stand"] = {
        "x": round(stand.x, 2),
        "y": round(stand.y, 2),
        "heading_deg": round(stand.yaw_deg),
        "choice": f"{index + 1} of {len(candidates)}: {stand.note}",
    }
    if result.get("ok") and resolved.point is not None:
        result["look"] = look_at(robot, resolved.point)
    return result


def resolve(robot: Robot, target: str, here: Pose2D, distances: Distances) -> Resolved:
    """The model's target as a :class:`Resolved`: a place of the book (matched first, so a name
    is never read as a box), a box of the last ``see`` picture, or the speaker's face;
    :class:`ToolError` naming the three when it is none of them."""
    from pepin.pixel_to_map import Box as Rectangle

    words = target.strip()
    book = robot.goals.places()
    names = {name.lower(): name for name in book}
    if words.lower() in names:
        name = names[words.lower()]
        place = book[name]
        x, y, yaw = float(place["x"]), float(place["y"]), float(place.get("yaw_deg", 0.0))
        found = {"kind": "place", "name": name, "x": round(x, 2), "y": round(y, 2)}
        return Resolved(Target("place", x, y, math.radians(yaw)), found, None, name)
    try:
        box: Box | None = Rectangle.parse(words)
    except ValueError:
        box = None
    if box is not None:
        return _thing(robot, box, distances)
    if words.lower() in SPEAKER:
        return _speaker(robot, here)
    known = ", ".join(sorted(book)) or "none yet"
    raise ToolError(
        f"target {target!r} is none of: 'me' (the person speaking), a place name ({known}),"
        " or a box x0,y0,x1,y1 in pixels of your last see() picture"
    )


def _speaker(robot: Robot, here: Pose2D) -> Resolved:
    """The nearest face the face sensor sees, located by depth; its facing from the head's yaw."""
    sighting = robot.sightings.face()
    if sighting is None:
        raise ToolError(
            "no face in view: ask the person to turn to the robot (it sees faces up to about"
            " 3.5 m), or turn the head toward them with look first; nothing moved"
        )
    from pepin.pixel_to_map import Box as Rectangle

    located = _locate(robot, Rectangle(*sighting.box), "the person")
    yaw = sighting.head_yaw_deg if sighting.head_yaw_deg is not None else 0.0
    facing = facing_from(located.x, located.y, here, yaw)
    found: dict[str, Any] = {"kind": "person", "who": sighting.who, **located.as_dict()}
    found["facing_deg"] = round(math.degrees(math.atan2(math.sin(facing), math.cos(facing))))
    if sighting.head_yaw_deg is None:
        found["facing_note"] = "the head's turn was not read: taken as facing the robot"
    if sighting.faces > 1:
        found["note"] = f"{sighting.faces} faces in view: the nearest one was taken"
    target = Target("person", located.x, located.y, facing, located.sigma_m)
    label = sighting.who or "the person"
    return Resolved(target, found, (located.x, located.y, located.z), label)


def _thing(robot: Robot, box: Box, distances: Distances) -> Resolved:
    """A box of the last ``see`` picture, refused once the head has turned since."""
    if robot.seen is None:
        raise ToolError(
            "a box means a spot in see()'s picture, and there is none since the last drive:"
            " call see() and give the box from its picture; nothing moved"
        )
    now = robot.neck.pose()
    turned = max(abs(now.pan_deg - robot.seen.pan_deg), abs(now.tilt_deg - robot.seen.tilt_deg))
    if turned > distances.head_moved_deg:
        raise ToolError(
            f"the head has turned {turned:.0f} deg since that picture, so the box points at"
            " something else now: call see() again and give the box from the new picture;"
            " nothing moved"
        )
    located = _locate(robot, box, "that box")
    found = {"kind": "thing", **located.as_dict()}
    label = f"the thing at ({located.x:.2f}, {located.y:.2f})"
    return Resolved(
        Target("thing", located.x, located.y, None, located.sigma_m),
        found,
        (located.x, located.y, located.z),
        label,
    )


def _locate(robot: Robot, box: Box, what: str) -> Located:
    """``robot.eyes.locate`` with its refusal in the model's words."""
    from pepin.pixel_to_map import LocateError

    try:
        return robot.eyes.locate(box)
    except LocateError as error:
        raise ToolError(f"cannot tell where {what} is: {error}; nothing moved") from error


def first_plannable(robot: Robot, candidates: list[Stand]) -> tuple[int, Stand | None, list[str]]:
    """The first candidate the goal server would drive to, its index, and the refusals of the
    ones before it; ``(len, None, refusals)`` when none. A refusal that is not the goal's (the
    planner down, the robot boxed in) ends the search: no other spot would fare better."""
    refusals: list[str] = []
    for index, stand in enumerate(candidates):
        answer = robot.goals.plan(stand.x, stand.y, stand.yaw_deg)
        if answer.get("event") == "error":
            detail = str(answer.get("detail", ""))
            if "unknown command" in detail:
                raise ToolError(
                    "the goal server cannot check a spot without driving to it (a build from"
                    " before the approach tool): restart Nav2 (ros/laptop.sh nav); nothing moved"
                )
            raise ToolError(f"the goal server could not check the spot: {detail}; nothing moved")
        if answer.get("ok"):
            return index, stand, refusals
        reason = str(answer.get("reason") or "refused")
        if answer.get("blames") != "goal":
            raise ToolError(f"the planner cannot plan from here: {reason}; nothing moved")
        refusals.append(f"{stand.note} at ({stand.x:.2f}, {stand.y:.2f}): {reason}")
    return len(candidates), None, refusals


def look_at(robot: Robot, point: tuple[float, float, float]) -> dict[str, Any]:
    """Turn the head to a map point from where the robot stands (the ``look`` path: held about
    ten seconds, then home by itself), as near as the neck reaches: a standing person's face at
    0.8 m is 25-30 deg above the camera and the neck tilts 20 up, so the face sits high in the
    picture. What happened, never a raise."""
    try:
        here = _pose(robot)
        aim = robot.eyes.aim(point[0], point[1], point[2], here)
        reach = robot.neck.reach()
        pan = min(max(aim.pan_deg, reach.pan_right_deg), reach.pan_left_deg)
        tilt = min(max(aim.tilt_deg, reach.tilt_up_deg), reach.tilt_down_deg)
        move = robot.neck.turn(pan, tilt)
    except ToolError as error:
        return {"looked": False, "why": error.why}
    from pepin.tools.head import why_not  # here: the tools register in the package's order

    out: dict[str, Any] = {"looked": move.reached, **(move.pose.as_dict() if move.pose else {})}
    if (pan, tilt) != (aim.pan_deg, aim.tilt_deg):
        out["short_by"] = (
            f"the target is at pan {aim.pan_deg:+.0f}, tilt {aim.tilt_deg:+.0f} deg: past the"
            " neck's reach, so the head stopped at its limit"
        )
    if not move.reached:
        out["why"] = why_not(move)
    return out


def _pose(robot: Robot) -> Pose2D:
    """The robot's pose on the map as :class:`pepin.pixel_to_map.Pose2D`."""
    from pepin.pixel_to_map import Pose2D

    here = pose_on_map(robot)
    return Pose2D(float(here["x"]), float(here["y"]), math.radians(float(here.get("yaw_deg", 0))))
