"""Driving: where the robot is, the named places, a drive to one of them, and the stop.

Every drive belongs to the goal server: its gate refuses a goal on a stale pose, it refuses a
second goal while one runs ("already driving: cancel first"), and its cancel stops every goal on
the robot, whoever sent it. These tools only ask it; they never command the wheels.

The reflexes live here, not in the model: a drive waits for its own end and says how it ended;
one that has not ended within :data:`pepin.tools.robot.DRIVE_TIMEOUT_S`, or whose report stream
is lost, is cancelled, so no motion outlives the call that started it; and the head is turned to
its working pose first, the pose the costmap's camera scans are measured in.
"""

from __future__ import annotations

import math
from typing import Any

from pepin.goal_link import NAV2_CANCELED, NAV2_SUCCEEDED, cancel_line
from pepin.tools.registry import Result, ToolError, fail, ok, tool
from pepin.tools.robot import Robot

AT_PLACE_M = 0.35  # "at" a place: within this distance of it
STALE_POSE_S = 2.0  # a pose older than this is reported as such
HEAD_FORWARD_DEG = 5.0  # a head further than this from its working pose is turned back to it
AFTER_CANCEL_S = 35.0  # after a cancel of our own, the drive's end is waited for this long
NAV2_ABORTED = 6


def pose_on_map(robot: Robot) -> dict[str, Any]:
    """The goal server's ``where`` answer when it carries a pose; :class:`ToolError` otherwise."""
    answer = robot.goals.where()
    if answer.get("pose") != "tf" or "x" not in answer:
        raise ToolError(
            "the robot does not know where it is: nothing publishes its pose on the map"
            " (RTAB-Map down, or this start of it not yet placed: ros/laptop.sh vslam)"
        )
    return answer


def nearest_place(
    places: dict[str, dict[str, float]], x: float, y: float
) -> tuple[str, float] | None:
    """The named place closest to ``(x, y)`` and its distance in metres; None for no places."""
    distances = {
        name: math.hypot(float(p["x"]) - x, float(p["y"]) - y) for name, p in places.items()
    }
    if not distances:
        return None
    name = min(distances, key=lambda n: distances[n])
    return name, distances[name]


@tool
def where_am_i(robot: Robot) -> Result:
    """Where the robot is now: its map position x, y in metres, its heading in degrees (0 along
    the map's x axis, counter-clockwise positive), and the nearest named place with its distance;
    ``at`` names that place when the robot stands within 0.35 m of it."""
    here = pose_on_map(robot)
    x, y = float(here["x"]), float(here["y"])
    payload: dict[str, Any] = {
        "x": round(x, 2),
        "y": round(y, 2),
        "heading_deg": round(float(here.get("yaw_deg", 0.0))),
    }
    age = here.get("age_s")
    if isinstance(age, (int, float)) and age > STALE_POSE_S:
        payload["warning"] = f"the pose is {age:.1f} s old: localisation may have stalled"
    nearest = nearest_place(robot.goals.places(), x, y)
    if nearest is not None:
        name, distance = nearest
        payload.update(
            nearest_place=name,
            distance_m=round(distance, 2),
            at=name if distance <= AT_PLACE_M else None,
        )
    return ok(**payload)


@tool
def list_places(robot: Robot) -> Result:
    """The named places the robot can drive to (go_to takes these names exactly as written),
    each with its map position and its distance from the robot, nearest first."""
    book = robot.goals.places()
    if not book:
        return ok(
            places=[],
            note="no named places yet: the book comes from RTAB-Map's graph (ros/laptop.sh"
            " vslam), and a place is named on the robot with ros/goto.sh mark NAME",
        )
    try:
        here = pose_on_map(robot)
    except ToolError:
        here = None
    rows = []
    for name, place in book.items():
        row: dict[str, Any] = {"name": name, "x": round(place["x"], 2), "y": round(place["y"], 2)}
        if here is not None:
            distance = math.hypot(place["x"] - float(here["x"]), place["y"] - float(here["y"]))
            row["distance_m"] = round(distance, 2)
        rows.append(row)
    rows.sort(key=lambda r: (r.get("distance_m", 0.0), r["name"]))
    return ok(places=rows)


@tool(moves=True)
def go_to(robot: Robot, place: str) -> Result:
    """Drive to a named place and wait until the drive ends (up to 4 minutes; a drive that has
    not ended by then is cancelled), then say how it ended: arrived, or why not and how far
    short. One drive at a time: a second one is refused until the first ends or is cancelled.

    Args:
        place: the place's name, exactly as list_places gives it.
    """
    return drive(robot, {"cmd": "go", "place": place}, place)


@tool(moves=True)
def go_to_pose(robot: Robot, x: float, y: float, yaw_deg: float) -> Result:
    """Drive to a point of the map and wait until the drive ends, as go_to does. Prefer go_to
    with a named place; this is for a position read off the map (map_tree, recall).

    Args:
        x: map x in metres.
        y: map y in metres.
        yaw_deg: the heading to end with, degrees (0 along the map's x axis, counter-clockwise
            positive).
    """
    return drive(robot, {"cmd": "go", "x": x, "y": y, "yaw_deg": yaw_deg}, f"({x:.2f}, {y:.2f})")


@tool
def cancel(robot: Robot) -> Result:
    """Stop the robot now: cancels every drive on the robot, whoever started it. Always safe;
    call it at once when a person says stop."""
    try:
        answer = robot.goals.cancel()
    except ToolError as error:
        return fail(
            f"the cancel did NOT reach the robot ({error.why}); a person must stop it:"
            " ros/goto.sh cancel, or ros/stop.sh"
        )
    if answer.get("event") != "cancelled":
        return fail(f"the goal server did not confirm the cancel: it answered {answer}")
    navigators = answer.get("navigators")
    said = (
        [v for v in navigators.values() if isinstance(v, dict)]
        if isinstance(navigators, dict)
        else []
    )
    if said and not any("cancelling" in v for v in said):
        return fail(
            "no navigator confirmed the cancel (" + (cancel_line(answer) or "") + "); a person"
            " must check the robot has stopped: ros/goto.sh cancel, or ros/stop.sh"
        )
    stopping = sum(int(v.get("cancelling", 0)) for v in said)
    stopped = bool(answer.get("had_goal")) or stopping > 0
    return ok(stopped=stopped, detail="stopping the drive" if stopped else "nothing was driving")


def drive(robot: Robot, request: dict[str, Any], target: str) -> Result:
    """One drive through the goal server, reported when it has ended: the body of go_to and
    go_to_pose, and of any later tool that drives (approach, explore)."""
    head = face_forward(robot)
    deadline = robot.clock() + robot.drive_timeout_s
    accepted: dict[str, Any] | None = None
    feedback: dict[str, Any] = {}
    lost: dict[str, Any] | None = None
    done: dict[str, Any] | None = None
    gave_up: str | None = None
    link_error: str | None = None
    events = robot.goals.go(request)
    try:
        for event in events:
            kind = event.get("event")
            if kind == "error":
                return refused(robot, str(event.get("detail", "")), head)
            if kind == "accepted":
                accepted = event
            elif kind == "feedback":
                feedback = event
            elif kind == "lost":
                lost = event
            elif kind == "done":
                done = event
                break
            now = robot.clock()
            if gave_up is None and now > deadline:
                gave_up = f"gave up after {robot.drive_timeout_s:.0f} s without arriving"
                robot.goals.cancel()
                deadline = now + AFTER_CANCEL_S
            elif gave_up is not None and now > deadline:
                break
    except OSError as error:
        link_error = str(error)
    finally:
        close = getattr(events, "close", None)
        if callable(close):
            close()
    if done is None:
        # A drive nobody is watching any more must not go on: stop it — also when the stream
        # broke before "accepted", since Nav2 may have taken the goal in that instant.
        what = (
            "lost the drive's reports"
            if accepted is not None
            else "the goal server went silent before confirming the drive"
        )
        return fail(
            f"{what} ({link_error or 'no word'}); {robot.halt()}",
            **_progress(feedback),
        )
    return outcome(accepted, done, feedback, lost, gave_up, target, head)


def outcome(
    accepted: dict[str, Any] | None,
    done: dict[str, Any],
    feedback: dict[str, Any],
    lost: dict[str, Any] | None,
    gave_up: str | None,
    target: str,
    head: str | None,
) -> Result:
    """How a drive ended, in the words and numbers the model needs to re-plan."""
    status = int(done.get("status", 0))
    payload: dict[str, Any] = {
        "target": target,
        "seconds": done.get("seconds"),
        "run": done.get("run"),
    }
    arrival = done.get("arrival")
    if isinstance(arrival, dict) and "x" in arrival:
        payload["position"] = {
            "x": round(float(arrival["x"]), 2),
            "y": round(float(arrival["y"]), 2),
            "heading_deg": round(float(arrival.get("yaw_deg", 0.0))),
        }
        if accepted is not None:
            off = math.hypot(
                float(arrival["x"]) - accepted["x"], float(arrival["y"]) - accepted["y"]
            )
            payload["off_by_m"] = round(off, 2)
    if head:
        payload["head"] = head
    if status == NAV2_SUCCEEDED and not done.get("detail"):
        return ok(arrived=True, **payload)
    payload.update(arrived=False, **_progress(feedback))
    short = f", {feedback['distance']:.1f} m short of {target}" if "distance" in feedback else ""
    tries = int(feedback.get("recoveries", 0))
    retried = f" after {tries} recovery attempts" if tries else ""
    if lost is not None:
        reading = lost.get("reading") or "the pose was lost"
        why = f"localisation was lost mid-drive ({reading}) and the goal server stopped{short}"
    elif status == NAV2_CANCELED:
        who = gave_up or "the drive was cancelled from outside this call"
        why = f"{who}{short}"
    elif status == NAV2_ABORTED:
        why = (
            f"the navigator gave up{retried}{short}: the way there is probably blocked, or the"
            " goal lies inside an obstacle"
        )
    else:
        why = f"the drive ended without arriving (navigator status {status}){short}"
    if done.get("detail"):
        why += f" ({done['detail']})"
    return fail(why, **payload)


def refused(robot: Robot, detail: str, head: str | None) -> Result:
    """A goal the goal server would not take, with what the model needs to try again."""
    why = f"not driving: {detail}"
    if "no such place" in detail or "no place" in detail:
        try:
            names = sorted(robot.goals.places())
        except ToolError:
            names = []
        why += f". Known places: {', '.join(names)}" if names else ". No places are known yet"
    elif "already driving" in detail:
        why += " (another drive is under way: wait for it to end, or cancel it)"
    return fail(why, **({"head": head} if head else {}))


def face_forward(robot: Robot) -> str | None:
    """Turn the head to its working pose before a drive; what was done, in words (None when
    it was there already). A head that cannot be checked does not stop the drive."""
    try:
        pose, rest = robot.neck.pose(), robot.neck.rest()
    except ToolError as error:
        return f"head not checked before the drive: {error.why}"
    off = max(abs(pose.pan_deg - rest.pan_deg), abs(pose.tilt_deg - rest.tilt_deg))
    if off <= HEAD_FORWARD_DEG:
        return None
    try:
        move = robot.neck.home()
    except ToolError as error:
        return f"head left {off:.0f} deg off its working pose: {error.why}"
    if move.reached:
        return "head turned to its working pose for the drive"
    return f"head left {off:.0f} deg off its working pose: {move.why or 'it did not arrive'}"


def _progress(feedback: dict[str, Any]) -> dict[str, Any]:
    """The last progress report as payload."""
    if "distance" not in feedback:
        return {}
    return {"distance_left_m": round(float(feedback["distance"]), 2)}
