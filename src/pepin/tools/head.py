"""The head and what it sees: point it, sweep it, take a picture.

The head is the base server's (its neck commands); it turns whether or not the wheels do,
and a move gives up after a few seconds. What was seen is ``world``'s: a sweep only reports the
sightings the memory filed while the head dwelt at each direction, so no picture is judged here.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterable
from typing import Any

from pepin.tools.clients import HeadMove, HeadPose
from pepin.tools.registry import Result, ToolError, fail, ok, tool
from pepin.tools.robot import Robot

SWEEP_PANS_DEG = (120.0, 60.0, 0.0, -60.0, -120.0)  # ~83 deg of view each: they overlap
DWELL_S = 2.0  # at each direction of a sweep, what the memory files in this long is what was seen
MAX_SEEN = 8  # labels reported per direction


@tool
def look(robot: Robot, pan_deg: float, tilt_deg: float | None = None) -> Result:
    """Turn the head (the camera) and wait until it gets there; see() then gives the picture.
    Angles are relative to the robot's body.

    Args:
        pan_deg: degrees left of straight ahead (+) or right of it (-); the neck reaches about
            155 each way.
        tilt_deg: degrees below level (+) or above it (-), from about 20 up to 90 down; about 24
            is the working pose the robot drives in. Leave it out to keep the current tilt.
    """
    refusal = robot.neck.reach().refusal(pan_deg, tilt_deg)
    if refusal:
        return fail(refusal)
    return moved(robot.neck.turn(pan_deg, tilt_deg))


@tool
def look_around(robot: Robot) -> Result:
    """Sweep the head across its reach (five directions, about 15 s), report what the robot saw
    in each direction (labels with range and bearing), then turn the head back where it was.
    Needs the memory service (world) to say what was seen."""
    robot.world.latest(None, 0.1)  # the memory must answer before the head is sent anywhere
    start = robot.neck.pose()
    views: list[dict[str, Any]] = []
    try:
        for pan in sweep(robot):
            move = robot.neck.turn(pan, None)
            if not move.reached:
                return fail(f"the head stopped sweeping: {why_not(move)}", views=views)
            robot.sleep(DWELL_S)
            seen = robot.world.latest(None, DWELL_S)
            views.append({"pan_deg": pan, "seen": summarise(seen)})
    finally:
        restore(robot, start)
    return ok(views=views, head=start.as_dict())


@tool
def see(robot: Robot) -> Result:
    """One picture from the head camera, as the robot sees it now, with where the head points
    (pan_deg left +, tilt_deg down +). Point the head with look first to see elsewhere."""
    picture = robot.camera.snapshot()
    try:
        head: dict[str, Any] = robot.neck.pose().as_dict()
    except ToolError as error:
        head = {"head": f"unknown: {error.why}"}
    return ok(image=picture, **head)


def sweep(robot: Robot, skip: float | None = None) -> list[float]:
    """The pans of a sweep inside the neck's reach; ``skip``: a pan already looked at."""
    reach = robot.neck.reach()
    return [
        pan
        for pan in SWEEP_PANS_DEG
        if reach.pan_right_deg <= pan <= reach.pan_left_deg
        and (skip is None or abs(pan - skip) > 15.0)
    ]


def restore(robot: Robot, pose: HeadPose) -> None:
    """Turn the head back to ``pose`` after a sweep; a failure here does not hide the sweep's."""
    with contextlib.suppress(ToolError):
        robot.neck.turn(pose.pan_deg, pose.tilt_deg)


def moved(move: HeadMove) -> Result:
    """A head move as a result: the pose it reached, or why it did not."""
    pose = move.pose.as_dict() if move.pose else {}
    if move.reached:
        return ok(**pose)
    return fail(f"the head did not get there: {why_not(move)}", **pose)


def why_not(move: HeadMove) -> str:
    """Why a head move did not arrive, in words the model can act on."""
    if "wheels are moving" in move.why:
        return "it does not move while the wheels turn: wait for the drive to end, or cancel it"
    return move.why or "it stopped short (something in the way?)"


def summarise(sightings: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """Sightings as one line per label: how many, and the nearest one's range and bearing."""
    nearest: dict[str, dict[str, Any]] = {}
    counts: dict[str, int] = {}
    for row in sightings:
        label = str(row.get("label", "?"))
        counts[label] = counts.get(label, 0) + 1
        if label not in nearest or row.get("range", 1e9) < nearest[label].get("range", 1e9):
            nearest[label] = row
    lines = [
        {"label": label, "count": counts[label], **place_of(row)} for label, row in nearest.items()
    ]
    lines.sort(key=lambda line: line.get("range_m", 1e9))
    return lines[:MAX_SEEN]


def place_of(sighting: dict[str, Any]) -> dict[str, Any]:
    """A sighting's range (m), bearing (degrees, + left of the robot's nose) and map position."""
    out: dict[str, Any] = {}
    if isinstance(sighting.get("range"), (int, float)):
        out["range_m"] = round(float(sighting["range"]), 2)
    if isinstance(sighting.get("bearing"), (int, float)):
        out["bearing_deg"] = round(math.degrees(float(sighting["bearing"])))
    for axis in ("x", "y", "z"):
        if isinstance(sighting.get(axis), (int, float)):
            out[axis] = round(float(sighting[axis]), 2)
    return out
