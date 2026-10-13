"""The head and what it sees: point it, sweep it, take a picture.

The head is the gaze arbiter's: a look is a request it grants, holds behind a more urgent one,
or refuses in words (past the reach; during a drive only on a base server that moves the neck
at rest alone). A look holds the head for ten seconds, ``see`` keeps it there, and then the head
goes home by itself, so no tool turns it back. What was seen is ``world``'s: a sweep only
reports the sightings the memory filed while the head dwelt at each direction, so no picture is
judged here.

``see`` takes the FIRST STILL FRAME: the arbiter says from which moment (the board's clock) the
head's pictures are still, and the camera (the perception service's ring) gives the newest frame
exposed after it, waiting up to the robot's ``still_wait_s`` (0.3 s) in all, the head's own way
there included; past it, the newest frame there is, marked ``still: false``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import math
from collections.abc import Iterable
from typing import Any

from pepin.tools.clients import HeadMove, HeadPose
from pepin.tools.registry import Result, ToolError, fail, ok, tool
from pepin.tools.robot import Robot

SWEEP_PANS_DEG = (120.0, 60.0, 0.0, -60.0, -120.0)  # ~83 deg of view each: they overlap
DWELL_S = 2.0  # at each direction of a sweep, what the memory files in this long is what was seen
MAX_SEEN = 8  # labels reported per direction
STILL_POLL_S = 0.05  # see: the arbiter asked this often while the head is still on its way


@tool
def look(robot: Robot, pan_deg: float, tilt_deg: float | None = None) -> Result:
    """Turn the head (the camera) somewhere else and wait until it gets there; see() then gives
    the picture. Only for a direction the person names (left, behind, down at the floor, toward
    something): to say what the robot sees now, call see() alone, without look. Angles are
    relative to the robot's body. The head stays there about ten seconds (each see() restarts
    that) and then turns home by itself; a drive may take it for the road. If the head does not
    get there, see() still gives the picture from where it points.

    Args:
        pan_deg: degrees left of straight ahead (+) or right of it (-); the neck reaches about
            155 each way.
        tilt_deg: degrees below level (+) or above it (-), from about 20 up to 90 down; about 24
            is the working pose the robot drives in. Leave it out to keep the current tilt.
    """
    reach = robot.neck.reach()
    refusal = reach.refusal(pan_deg, tilt_deg)
    if refusal:
        return fail(refusal)
    move = robot.neck.turn(pan_deg, tilt_deg)
    target = {"pan_deg": pan_deg, "tilt_deg": tilt_deg}
    return moved(move, target=target, reach=dataclasses.asdict(reach))


@tool
def look_around(robot: Robot) -> Result:
    """Sweep the head across its reach (five directions, about 15 s) and report what the robot
    saw in each direction (labels with range and bearing); the head then turns home by itself.
    Needs the memory service (world) to say what was seen."""
    robot.world.latest(None, 0.1)  # the memory must answer before the head is sent anywhere
    views: list[dict[str, Any]] = []
    for pan in sweep(robot):
        move = robot.neck.turn(pan, None)
        if not move.reached:
            return fail(f"the head stopped sweeping: {why_not(move)}", views=views)
        robot.sleep(DWELL_S)
        seen = robot.world.latest(None, DWELL_S)
        views.append({"pan_deg": pan, "seen": summarise(seen)})
    return ok(views=views)


@tool
def see(robot: Robot) -> Result:
    """One picture from the head camera, as the robot sees it now, with where the head points
    (pan_deg left +, tilt_deg down +). This is what answers "what do you see?" or "look at me":
    call it alone; the head already faces forward. Only to see another direction, turn the head
    with look first. Seeing keeps the head where look put it a while longer."""
    start = robot.clock()
    still_from: float | None = None
    robot.seen = None
    pose: HeadPose | None = None
    try:
        pose = robot.neck.pose()
        while pose.moving and robot.clock() - start < robot.still_wait_s:
            robot.sleep(STILL_POLL_S)  # the head is on its way: the first still frame is wanted
            pose = robot.neck.pose()
        head: dict[str, Any] = pose.as_dict()
        still_from = pose.still_from
    except ToolError as error:
        head = {"head": f"unknown: {error.why}"}
    left = max(0.0, robot.still_wait_s - (robot.clock() - start))
    picture = robot.camera.snapshot(not_before=still_from, wait_s=left)
    robot.seen = pose  # a box drawn on this picture means something while the head stays
    with contextlib.suppress(ToolError):
        robot.neck.keep()
    extra = {"frame": picture.frame} if picture.frame else {}
    return ok(image=picture, **head, **extra)


def sweep(robot: Robot, skip: float | None = None) -> list[float]:
    """The pans of a sweep inside the neck's reach; ``skip``: a pan already looked at."""
    reach = robot.neck.reach()
    return [
        pan
        for pan in SWEEP_PANS_DEG
        if reach.pan_right_deg <= pan <= reach.pan_left_deg
        and (skip is None or abs(pan - skip) > 15.0)
    ]


def moved(
    move: HeadMove, target: dict[str, Any] | None = None, reach: dict[str, Any] | None = None
) -> Result:
    """A head move as a result: the pose it reached, or why it did not, with the neck's report
    (the target, the reach, the arbiter's answer) and the way on (see still works)."""
    pose = move.pose.as_dict() if move.pose else {}
    if move.reached:
        return ok(**pose)
    neck: dict[str, Any] = {"took_ms": round(move.ms)}
    if target is not None:
        neck["target"] = target
    if reach is not None:
        neck["reach"] = reach
    if move.report:
        neck["arbiter"] = move.report
    return fail(
        f"the head did not get there: {why_not(move)}; see() still gives the picture from where"
        " it points now",
        **pose,
        neck=neck,
    )


def why_not(move: HeadMove) -> str:
    """Why a head move did not arrive, in the arbiter's words (written for the model)."""
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
