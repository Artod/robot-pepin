"""Going up to someone or something and standing in a relation to it.

One tool and two words from the model, the target and the relation; the rest is the robot's.
The target becomes a map point through :mod:`pepin.pixel_to_map` (``robot.eyes``: stereo depth
in the box's own pixel space) — the person the robot talks with (``robot.sightings``: the one the
speaker gaze holds, their face or body box on the camera channel's newest still frame), a place
from the goal server's book, or a box of the last ``see`` picture — and the relation a
standing pose with two alternatives (:mod:`pepin.stand_pose`: never nearer a person than the
social minimum). Each is asked of the goal server's ``plan`` (RTAB-Map's map cell, Nav2's costmap
under the cart's whole hull standing there, then one plan: never motion, because the navigator
answers an unreachable goal by turning and backing up beside the person) until one would be
driven, and that ONE goal is driven by :func:`pepin.tools.drive.drive` exactly as ``go_to``'s;
when none would, the tool says why and nothing moves. A plan that ends short of its stand (Smac
Hybrid's goal tolerance: the planner's nearest reachable pose) makes its END the goal driven,
said in the answer: the stand itself is never reached, and on 2026-10-14 each one-second replan
ended somewhere else and MPPI chased the moving end for 80 s. A turn where the cart stands is driven
without a plan: the cart is already there, and a planner asked for its own cell answers with a
one-pose path the goal server reads as no path (2026-10-13, "повернись ко мне" on the dock:
"planner error code 0", while the same goal driven ten seconds later turned the cart). On
arrival the head turns to the target through the ``look`` path. Nothing here follows anyone:
one goal per call.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pepin.stand_pose import Distances, Relation, Stand, Target, facing_from, stands
from pepin.tools.clients import IDLE, HeadPose, HeadReach
from pepin.tools.drive import drive, pose_on_map
from pepin.tools.registry import Result, ToolError, fail, ok, tool
from pepin.tools.robot import Robot

if TYPE_CHECKING:  # numpy's module, loaded when a target is located
    from pepin.pixel_to_map import Box, Located, Pose2D, Stamped

logger = logging.getLogger(__name__)
SPEAKER = ("me", "speaker", "person")  # the target words that mean the person speaking
STILL_POLL_S = 0.05  # the head asked this often while it is still on its way (as see asks)
FACE_POLL_S = 0.1  # the gaze asked this often while the body turns toward the voice
FACING_DEG = 10.0  # a turn in place smaller than this is no turn: the goal checker's 11.5 deg
# A plan ending this near its stand ends at it: the goal checker's xy_goal_tolerance (0.10 m) and
# yaw_goal_tolerance (0.20 rad, 11.5 deg). Further off, the plan's end is the goal driven.
END_M = 0.10
END_DEG = 11.5


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
    """Go up to a person, a named place or a thing in the picture and stand in a relation to it,
    facing it: the tool for "come here", "come closer", "stand beside me", "stand to my right",
    "turn to me", "go to the printer and face it", "go and look at the board". Never use
    go_to_pose to reach or face a person: this tool measures where they are, keeps its distance
    (never nearer than 0.25 m) and takes only a spot the planner accepts; when there is none it
    says why and nothing moves. One drive (up to 4 minutes), reported when it ends as go_to's
    is; already near enough, it only turns. For a plain named place go_to is enough.

    Args:
        target: "me" for the person speaking (the one the robot looks at; seeing nobody, it
            first turns its body toward their voice and looks again); a place name exactly as
            list_places gives
            it; or a thing in the picture of your last see() as its box "x0,y0,x1,y1" in pixels
            of that picture (call see() again first if the head has turned since).
        relation: to (about 0.8 m away, facing it; already that near, it only turns to it),
            face (turn where it stands to face it), right_of or left_of (a person's own right
            or left; a thing's as the robot sees it), in_front_of, beside (the nearer side),
            behind. A place's own pose already faces its thing: "go to the printer and face
            it" is to.
        distance_m: only when the person names a distance, in metres.
    """
    distances = Distances.load()
    if distance_m is not None and not 0.2 <= distance_m <= 3.0:
        return fail(f"distance_m must be between 0.2 and 3 m, not {distance_m:g}")
    here = _pose(robot)
    resolved = resolve(robot, target, here, distances)
    candidates = stands(resolved.target, relation, here, distances, distance_m)
    if candidates[0].in_place:
        # A turn where the cart stands: no plan asked. The cart is already on the goal's cell,
        # and Smac Hybrid answers a goal on the start's cell with a one-pose path, which the
        # goal server's plan reads as no path ("planner error code 0", 2026-10-13).
        index, stand = 0, candidates[0]
        goal = stand
        if abs(_turn_deg(here, stand)) <= FACING_DEG:
            result = ok(arrived=True, moved=False, detail="already there, facing it")
        else:
            result = drive(robot, _request(stand), f"{relation} {resolved.label}")
    else:
        index, chosen, refusals, plan = first_plannable(robot, candidates)
        if chosen is None:
            return fail(
                f"nowhere to stand {relation} {resolved.label}: " + "; ".join(refusals) + "."
                " Nothing moved; the person may say where to stand instead",
                found=resolved.found,
            )
        stand = chosen
        goal, clause = drive_goal(plan, stand, resolved.target)
        label = f"{relation} {resolved.label}" + (f", {clause}" if clause else "")
        result = drive(robot, _request(goal), label)
        result["short_m"] = round(float(plan["short_m"]), 2)
    result["found"] = resolved.found
    result["stand"] = {
        "x": round(stand.x, 2),
        "y": round(stand.y, 2),
        "heading_deg": round(stand.yaw_deg),
        "choice": f"{index + 1} of {len(candidates)}: {stand.note}",
    }
    result["drive_to"] = {
        "x": round(goal.x, 2),
        "y": round(goal.y, 2),
        "heading_deg": round(goal.yaw_deg),
        "what": "the stand" if goal is stand else goal.note,
    }
    if result.get("ok") and resolved.point is not None:
        result["look"] = look_at(robot, resolved.point)
    return result


def _request(stand: Stand) -> dict[str, Any]:
    """The goal server's ``go`` for a stand, as ``go_to_pose`` sends one."""
    return {
        "cmd": "go",
        "x": round(stand.x, 3),
        "y": round(stand.y, 3),
        "yaw_deg": round(stand.yaw_deg, 1),
    }


def drive_goal(plan: dict[str, Any], stand: Stand, target: Target) -> tuple[Stand, str]:
    """Where the drive goes, from the plan asked for ``stand`` (the goal server's ``plan``
    answer: ``end``, ``short_m``, ``short_deg``): the stand itself when the plan ends within
    the goal checker's tolerance of it (:data:`END_M`, :data:`END_DEG`), else the plan's end,
    the planner's own nearest reachable pose, with the clause that says so ("" for the stand).
    The end keeps its own heading from the path; a last pose with no heading faces the target
    from there. :class:`ToolError` for a goal server that does not say where a plan ends."""
    end, short_m, said = plan.get("end"), plan.get("short_m"), plan.get("short_deg")
    if not isinstance(end, dict) or not isinstance(short_m, (int, float)):
        raise ToolError(
            "the goal server does not say where a plan ends (started before it did): restart"
            " it (ros/laptop.sh kick goal_server); nothing moved"
        )
    short_deg = float(said) if isinstance(said, (int, float)) else None
    turned = short_deg is not None and abs(short_deg) > END_DEG
    if short_m <= END_M and not turned:
        return stand, ""
    x, y, yaw = float(end["x"]), float(end["y"]), end.get("yaw_deg")
    if isinstance(yaw, (int, float)):
        heading, words = math.radians(float(yaw)), "its own heading"
    else:
        heading = math.atan2(target.y - y, target.x - x)
        words = f"facing the {target.kind} from there: the plan's last pose has no heading"
    parts = [f"{short_m:.2f} m"] if short_m > END_M else []
    parts += [f"{abs(short_deg):.0f} deg"] if short_deg is not None and turned else []
    off = " and ".join(parts)
    goal = Stand(x, y, heading, f"the plan's end, {off} short of the stand; {words}")
    logger.warning(  # warning: the voice loop logs nothing below it
        "approach: the planner stops %s short of the stand: driving to its end"
        " (%.2f, %.2f, %+.0f deg; %s)",
        off,
        x,
        y,
        goal.yaw_deg,
        words,
    )
    return goal, f"stopping {off} short of the stand: the nearest the planner reaches"


def _turn_deg(here: Pose2D, stand: Stand) -> float:
    """How far the robot would turn to the stand's heading, degrees (-180, 180]."""
    turn = stand.yaw_rad - here.yaw_rad
    return math.degrees(math.atan2(math.sin(turn), math.cos(turn)))


def resolve(robot: Robot, target: str, here: Pose2D, distances: Distances) -> Resolved:
    """The model's target as a :class:`Resolved`: a place of the book (matched first, so a name
    is never read as a box), a box of the last ``see`` picture, or the person the robot talks
    with; :class:`ToolError` naming the three when it is none of them."""
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
        return _speaker(robot, here, distances)
    known = ", ".join(sorted(book)) or "none yet"
    raise ToolError(
        f"target {target!r} is none of: 'me' (the person speaking), a place name ({known}),"
        " or a box x0,y0,x1,y1 in pixels of your last see() picture"
    )


def _speaker(robot: Robot, here: Pose2D, distances: Distances) -> Resolved:
    """The person the robot talks with (:meth:`pepin.tools.clients.Sightings.speaker`: the one
    the speaker gaze holds, on the camera channel's newest frame since the head stood still),
    located by depth in that frame's own pixels and placed with TF's poses at that frame's stamp
    (:func:`sighting_pose`); their facing from their face's turn. Refused when the sighting is
    stale, or when the camera turned between the frame and the depth pair grabbed after it."""
    head = still_head(robot)
    sighting = robot.sightings.speaker(head)
    turned: float | None = None
    if sighting is None:
        turned = face_the_voice(robot, distances)
        if turned is not None:
            head = still_head(robot)
            sighting = robot.sightings.speaker(head)
    if sighting is None:
        if turned is None:
            raise ToolError(
                "nobody in view: ask the person to step in front of the robot or turn to it"
                " (faces are seen up to about 3.5 m), or turn the head toward them with look"
                " first; nothing moved"
            )
        raise ToolError(
            f"nobody in view after the body turned {turned:+.0f} deg toward the voice: ask the"
            " person to step in front of the robot or say where they are; only the body turned"
        )
    from pepin.pixel_to_map import Box as Rectangle

    at, age, why_now = sighting_pose(robot, sighting.stamp, distances)
    located = _locate(robot, Rectangle(*sighting.box), sighting.space, "the person", at)
    camera_deg = camera_turned(robot, at.stamp) if at is not None else None
    if camera_deg is not None:
        if camera_deg > distances.head_moved_deg:
            raise ToolError(
                f"the camera turned {camera_deg:.0f} deg between the person's picture and the depth"
                " measured after it (the head or the body moved), so the two disagree: ask"
                " again; nothing moved"
            )
    else:
        moved = max(
            abs(math.degrees(located.head.pan_rad) - head.pan_deg),
            abs(math.degrees(located.head.pitch_rad) - head.tilt_deg),
        )
        if moved > distances.head_moved_deg:
            raise ToolError(
                f"the head moved {moved:.0f} deg while the person was being measured, so the"
                " picture and the depth disagree: ask again; nothing moved"
            )
    yaw = sighting.yaw_deg if sighting.yaw_deg is not None else 0.0
    facing = facing_from(located.x, located.y, here, yaw)
    found: dict[str, Any] = {"kind": "person", "seen_by": sighting.kind, **located.as_dict()}
    found["facing_deg"] = round(math.degrees(math.atan2(math.sin(facing), math.cos(facing))))
    found["frame"] = sighting.seq
    found["stamp"] = None if sighting.stamp is None else round(sighting.stamp, 3)
    found["age_s"] = None if age is None else round(age, 2)
    used = located.pose
    found["pose"] = {
        "x": round(used.x, 2),
        "y": round(used.y, 2),
        "heading_deg": round(math.degrees(used.yaw_rad)),
        "at": "the frame's stamp" if at is not None else "now",
    }
    if at is None:
        found["pose"]["why"] = why_now
    if camera_deg is not None:
        found["camera_turned_deg"] = round(camera_deg, 1)
    if sighting.yaw_deg is None:
        found["facing_note"] = "the head's turn was not read: taken as facing the robot"
    if sighting.people > 1:
        found["note"] = f"{sighting.people} people in view: {sighting.chosen} was taken"
    target = Target("person", located.x, located.y, facing, located.sigma_m)
    return Resolved(target, found, (located.x, located.y, located.z), "the person")


def sighting_pose(
    robot: Robot, stamp: float | None, distances: Distances
) -> tuple[Stamped | None, float | None, str]:
    """Where the camera and the cart were at a sighting's own ``stamp``, from the one history of
    the poses (TF, the goal server's ``pose_at``), with the sighting's age on the goal server's
    clock (TF's). ``(None, age, why)`` when the sighting has no stamp or TF has no history for
    it: the person is then placed with the neck and the cart of now, and that is logged.
    :class:`ToolError` for a sighting older than ``distances.sighting_age_s`` (stale: never
    placed) or a goal server that cannot answer."""
    if stamp is None:
        return None, None, "the sighting carries no stamp"
    camera = _pose_at(robot, "camera_link", stamp)
    said = camera.get("age_s")
    age = float(said) if isinstance(said, (int, float)) else None
    if age is not None and age > distances.sighting_age_s:
        raise ToolError(
            f"the newest sighting of the person is {age:.1f} s old (over"
            f" {distances.sighting_age_s:g} s): too old to place them, the robot or the person"
            " may have moved since; ask again; nothing moved"
        )
    base = _pose_at(robot, "base_link", stamp) if camera.get("ok") else camera
    if not base.get("ok"):
        why = f"TF has no pose at the frame's stamp ({base.get('kind')}: {base.get('reason')})"
        logger.warning("approach: %s; the person is placed with the neck and the cart of now", why)
        return None, age, why
    from pepin.pixel_to_map import Stamped as Poses

    return Poses.from_answers(camera, base), age, ""


def camera_turned(robot: Robot, stamp: float) -> float | None:
    """How far the camera turned between the frame the person was seen in (its ``stamp``) and
    now, asked after the depth pair was grabbed: the head's turn and the cart's together, from
    TF's camera link in ``odom`` at both moments (smooth: a localisation update of map -> odom
    between them is no turn), degrees; None when TF cannot say (logged)."""
    from pepin.pixel_to_map import rigid_pose, turned_deg

    then = _pose_at(robot, "camera_link", stamp, parent="odom")
    now = _pose_at(robot, "camera_link", parent="odom") if then.get("ok") else then
    if not now.get("ok"):
        logger.warning(
            "approach: TF cannot say how the camera moved since the frame (%s: %s); the"
            " neck's encoders judge instead",
            now.get("kind"),
            now.get("reason"),
        )
        return None
    return turned_deg(rigid_pose(then), rigid_pose(now))


def _pose_at(
    robot: Robot, frame: str, stamp: float | None = None, parent: str = "map"
) -> dict[str, Any]:
    """The goal server's ``pose_at`` answer; :class:`ToolError` for its error event."""
    answer = robot.goals.pose_at(frame, stamp, parent)
    if answer.get("event") == "error":
        detail = str(answer.get("detail", ""))
        if "unknown command" in detail:
            raise ToolError(
                "the goal server cannot say where the camera was when the person was seen"
                " (started before its pose_at command): restart it (ros/laptop.sh kick"
                " goal_server); nothing moved"
            )
        raise ToolError(
            f"the goal server could not say where the camera was: {detail}; nothing moved"
        )
    return answer


def face_the_voice(robot: Robot, distances: Distances) -> float | None:
    """Nobody in view: the body turned toward the voice the speaker gaze holds (its own turn,
    the whole bearing in one request), waited for; the bearing it turned to. None when there
    is no voice to go by (no gaze, no person engaged, a drive); :class:`ToolError` when the
    gaze refuses the turn or it does not end within ``distances.face_wait_s``."""
    gaze = robot.gaze
    if gaze is None:
        return None
    view = gaze.attention()
    if view.phase in IDLE or view.bearing_deg is None:
        return None
    bearing = view.bearing_deg
    if view.phase != "turning":
        why = gaze.face_person("approach: nobody in view, the voice is the lead")
        if why is not None:
            raise ToolError(
                f"nobody in view, and the body cannot turn to the voice: {why}. Nothing moved:"
                " ask the person to step in front of the robot or to say where they are, and"
                " call approach again only after they do"
            )
    start = robot.clock()
    while robot.clock() - start < distances.face_wait_s:
        robot.sleep(FACE_POLL_S)
        if gaze.attention().phase != "turning":
            return bearing
    raise ToolError(
        f"the body's turn toward the voice at {bearing:+.0f} deg did not end within"
        f" {distances.face_wait_s:.0f} s; the drive was not started"
    )


def still_head(robot: Robot) -> HeadPose:
    """The head's pose once it has stopped (or after ``robot.still_wait_s`` on its way), as
    ``see`` waits for it: the speaker's frame must be exposed after this pose's ``still_from``."""
    start = robot.clock()
    pose = robot.neck.pose()
    while pose.moving and robot.clock() - start < robot.still_wait_s:
        robot.sleep(STILL_POLL_S)
        pose = robot.neck.pose()
    return pose


def _thing(robot: Robot, box: Box, distances: Distances) -> Resolved:
    """A box of the last ``see`` picture, refused once the head has turned since."""
    if robot.seen is None:
        raise ToolError(
            "a box means a spot in see()'s picture, and there is none since the last drive:"
            " call see() and give the box from its picture; nothing moved"
        )
    now, then = robot.neck.pose(), robot.seen.head
    turned = max(abs(now.pan_deg - then.pan_deg), abs(now.tilt_deg - then.tilt_deg))
    if turned > distances.head_moved_deg:
        raise ToolError(
            f"the head has turned {turned:.0f} deg since that picture, so the box points at"
            " something else now: call see() again and give the box from the new picture;"
            " nothing moved"
        )
    located = _locate(robot, box, robot.seen.space, "that box")
    found = {"kind": "thing", **located.as_dict()}
    label = f"the thing at ({located.x:.2f}, {located.y:.2f})"
    return Resolved(
        Target("thing", located.x, located.y, None, located.sigma_m),
        found,
        (located.x, located.y, located.z),
        label,
    )


def _locate(robot: Robot, box: Box, space: str, what: str, at: Stamped | None = None) -> Located:
    """``robot.eyes.locate`` of a box in pixels of ``space`` (placed with TF's poses ``at`` its
    frame's stamp when given), its refusal in the model's words."""
    from pepin.pixel_to_map import LocateError

    try:
        return robot.eyes.locate(box, space, at)
    except LocateError as error:
        raise ToolError(f"cannot tell where {what} is: {error}; nothing moved") from error


def first_plannable(
    robot: Robot, candidates: list[Stand]
) -> tuple[int, Stand | None, list[str], dict[str, Any]]:
    """The first candidate the goal server would drive to, its index, the refusals of the ones
    before it and its ``plan`` answer (where the plan ends: :func:`drive_goal`); ``(len, None,
    refusals, {})`` when none. A refusal that is not the goal's (the planner down, the robot
    boxed in) ends the search: no other spot would fare better."""
    refusals: list[str] = []
    for index, stand in enumerate(candidates):
        answer = robot.goals.plan(stand.x, stand.y, stand.yaw_deg)
        if answer.get("event") == "error":
            detail = str(answer.get("detail", ""))
            if "unknown command" in detail:
                raise ToolError(
                    "the goal server cannot check a spot without driving to it (started before"
                    " its plan command): restart it (ros/laptop.sh kick goal_server); nothing"
                    " moved"
                )
            raise ToolError(f"the goal server could not check the spot: {detail}; nothing moved")
        if answer.get("ok"):
            return index, stand, refusals, answer
        reason = str(answer.get("reason") or "refused")
        if answer.get("blames") != "goal":
            raise ToolError(f"the planner cannot plan from here: {reason}; nothing moved")
        refusals.append(f"{stand.note} at ({stand.x:.2f}, {stand.y:.2f}): {reason}")
    return len(candidates), None, refusals, {}


def look_at(robot: Robot, point: tuple[float, float, float]) -> dict[str, Any]:
    """Turn the head to a map point from where the robot stands (the ``look`` path: held about
    ten seconds, then home by itself), as near as the neck reaches: a standing person's face at
    0.8 m is 25-30 deg above the camera and the neck tilts 20 up, so the face sits high in the
    picture. What happened, never a raise."""
    try:
        here = _pose(robot)
        aim = robot.eyes.aim(point[0], point[1], point[2], here)
        reach = robot.neck.reach()
        pan = pan_within(reach, aim.pan_deg, robot.neck.pose().pan_deg)
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


def pan_within(reach: HeadReach, pan_deg: float, here_deg: float) -> float:
    """``pan_deg`` inside the reach: past it, the end nearer the head's own pan ``here_deg``.
    A point behind reads +184 or -176 by the wrap, and the end nearest by number can be the one
    a whole sweep away through the front (:meth:`pepin.gaze.Reach.clamp`)."""
    lo, hi = reach.pan_right_deg, reach.pan_left_deg
    if lo <= pan_deg <= hi:
        return pan_deg
    return lo if abs(here_deg - lo) < abs(here_deg - hi) else hi


def _pose(robot: Robot) -> Pose2D:
    """The robot's pose on the map as :class:`pepin.pixel_to_map.Pose2D`."""
    from pepin.pixel_to_map import Pose2D

    here = pose_on_map(robot)
    return Pose2D(float(here["x"]), float(here["y"]), math.radians(float(here.get("yaw_deg", 0))))
