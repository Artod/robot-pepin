"""What is where: the view now, the memory of before, the home as a tree, and names given.

All of it is ``world``'s (CONTRACTS.md): the sightings node files what the camera sees, the
memory merges them into objects and zones. The reflex that lives here is freshness: ``find``
answers only from sightings of the last :data:`FRESH_S` seconds and waits for one rather than
answering from what was in view a minute ago.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pepin.tools.clients import HeadPose
from pepin.tools.drive import pose_on_map
from pepin.tools.head import place_of, restore, sweep, why_not
from pepin.tools.registry import Result, ToolError, fail, ok, tool
from pepin.tools.robot import Robot

FRESH_S = 1.5  # a sighting older than this is not "in view now"
SIGHT_WAIT_S = 2.5  # how long find waits for a fresh sighting at one direction
POLL_S = 0.25
MAX_MATCHES = 5


@tool
def find(robot: Robot, thing: str, look_around: bool = False) -> Result:
    """Look for a thing in view NOW and say where it is: range in metres, bearing in degrees (+
    left of the robot's nose) and map position. Answers only from sightings of the last 1.5 s,
    waiting up to 2.5 s for one. With look_around, when the thing is not in view the head sweeps
    its reach (about 15 s) and stays pointed at the thing once found. For where a thing was
    seen before, use recall.

    Args:
        thing: what to look for, one English noun as the detector names things ("chair",
            "cup", "person", "backpack").
        look_around: sweep the head when the thing is not in view now.
    """
    label = thing.strip().lower()
    rows = watch(robot, label)
    if rows:
        return found(rows, None)
    if not look_around:
        return fail(
            f"no {label} in view now (watched {SIGHT_WAIT_S:.1f} s). find with look_around=true"
            f" sweeps the head; recall('{label}') says where one was seen before"
        )
    start = robot.neck.pose()
    directions = sweep(robot, skip=start.pan_deg)
    for pan in directions:
        move = robot.neck.turn(pan, None)
        if not move.reached:
            restore(robot, start)
            return fail(f"the head could not sweep: {why_not(move)}")
        rows = watch(robot, label)
        if rows:
            return found(rows, move.pose)
    restore(robot, start)
    return fail(
        f"no {label} anywhere around here: the head looked in {len(directions) + 1} directions."
        f" recall('{label}') says where one was seen before; otherwise it is not in this room"
    )


@tool
def recall(robot: Robot, thing: str) -> Result:
    """Where a thing was seen before, from the robot's memory: each match with its id, map
    position, zone, when it was last seen, how often, the confidence, and its distance from the
    robot now. For what is in view now, use find.

    Args:
        thing: the thing, one English noun as the detector names things ("chair", "cup").
    """
    label = thing.strip().lower()
    rows = robot.world.objects(label)
    if not rows:
        return fail(
            f"nothing called {label} in the robot's memory: never seen, or forgotten."
            f" find('{label}', look_around=true) searches around here"
        )
    try:
        here = pose_on_map(robot)
    except ToolError:
        here = None
    matches = []
    for row in rows[:MAX_MATCHES]:
        match = {k: round(v, 2) if isinstance(v, float) else v for k, v in row.items()}
        if here is not None and isinstance(row.get("x"), (int, float)):
            distance = math.hypot(
                float(row["x"]) - float(here["x"]), float(row["y"]) - float(here["y"])
            )
            match["distance_m"] = round(distance, 2)
        matches.append(match)
    return ok(matches=matches, total=len(rows))


@tool
def map_tree(robot: Robot) -> Result:
    """The robot's map of the home as a text tree: rooms, the furniture and things in each, and
    the named places, with map positions. Read it to choose where to go for a request such as
    "go to the kitchen table"."""
    text = robot.world.tree().get("text")
    if not text:
        return fail("the memory answered no tree: nothing has been mapped and named yet")
    return ok(tree=str(text))


@tool
def remember(
    robot: Robot,
    name: str,
    what: Literal["zone", "object", "place"],
    object_id: str | None = None,
) -> Result:
    """Keep a name a person gives. what=zone names the area the robot stands in now ("this is
    the kitchen"); what=object names one thing from the memory, by the id recall gave ("this is
    my mug"); what=place would name a spot to drive back to, which these tools cannot do yet.

    Args:
        name: the name, as the person said it.
        what: zone, object or place.
        object_id: with what=object, the thing's id as recall gave it.
    """
    name = name.strip()
    if not name:
        return fail("the name is empty")
    if what == "place":
        return fail(
            "a place to drive back to is named in RTAB-Map's graph by its places node"
            " (ros/goto.sh mark NAME on the robot), which these tools cannot reach yet;"
            " naming the area as a zone (what=zone) works"
        )
    if what == "object":
        if not object_id:
            return fail("which thing? call recall first and pass its id as object_id")
        return ok(
            remembered=robot.world.remember({"what": "object", "name": name, "id": object_id})
        )
    here = pose_on_map(robot)
    entry = {"what": "zone", "name": name, "x": float(here["x"]), "y": float(here["y"])}
    return ok(remembered=robot.world.remember(entry))


def watch(robot: Robot, label: str) -> list[dict[str, Any]]:
    """Fresh sightings of ``label``, waiting up to :data:`SIGHT_WAIT_S` for the first."""
    end = robot.clock() + SIGHT_WAIT_S
    while True:
        rows = robot.world.latest(label, FRESH_S)
        if rows or robot.clock() >= end:
            return rows
        robot.sleep(POLL_S)


def found(rows: list[dict[str, Any]], head: HeadPose | None) -> Result:
    """The best sighting (the most confident) and how many more there were."""
    best = max(rows, key=lambda row: float(row.get("score", 0.0)))
    payload: dict[str, Any] = {
        "label": best.get("label"),
        "score": round(float(best.get("score", 0.0)), 2),
        **place_of(best),
    }
    if len(rows) > 1:
        payload["also_seen"] = len(rows) - 1
    if head is not None:
        payload["head"] = head.as_dict()
    return ok(**payload)
