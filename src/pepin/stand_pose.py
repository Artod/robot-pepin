"""Where to stand for "come to me", "stand to my right", "go behind the chair": pure geometry.

A target is a point on the map with a frame: a PERSON faces somewhere (their right is their own),
a PLACE of the book is itself a standing pose (marked where the robot stood, facing its thing), a
THING has no front of its own, so its front is the side the robot sees and its right is the
viewer's right. A relation turns that frame into a standing pose, and two alternatives turn the
stand point about the target by the ring step either way at the same distance, for the planner
to refuse one by one. Distances are config/approach.json's (:class:`Distances`); a person is never
stood nearer than ``person_min_m`` plus two sigma of where they were measured. ``face`` is a turn
where the robot stands, and ``to`` is one too when the robot already stands within ``near_m`` of
its stand distance: "come closer" never backs the robot away.
"""

from __future__ import annotations

import math
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Literal

if TYPE_CHECKING:  # numpy's module: the chooser itself is plain arithmetic
    from pepin.pixel_to_map import Pose2D

Relation = Literal["to", "right_of", "left_of", "in_front_of", "beside", "behind", "face"]
RELATIONS: tuple[str, ...] = typing.get_args(Relation)
Kind = Literal["person", "place", "thing"]
CONFIG = "approach.json"


@dataclass(frozen=True)
class Target:
    """What to stand by: its kind, its map point and, for a person or a place, the way it faces
    (radians, CCW from the map's x); ``sigma_m`` is how well its point is known."""

    kind: Kind
    x: float
    y: float
    facing_rad: float | None = None
    sigma_m: float = 0.0


@dataclass(frozen=True)
class Distances:
    """How far to stand: in front/behind (``front_m``), at a side (``side_m``), never nearer a
    person than ``person_min_m``, the alternatives ``ring_deg`` either way, and ``to`` a turn in
    place when the robot already stands within ``near_m`` of its stand distance."""

    front_m: float = 0.3
    side_m: float = 0.5
    person_min_m: float = 0.25
    ring_deg: float = 30.0
    head_moved_deg: float = 2.0
    near_m: float = 0.15

    @classmethod
    def load(cls, path: str | Path | None = None) -> Distances:
        """config/approach.json (each number with its range and note, as config/knobs.json
        writes a knob), read at every call so an edit counts at the next request."""
        from pepin.flags import knobs_of, read_knobs

        if path is None:
            from pepin.deployment import config_file

            path = config_file(CONFIG)
        knobs = knobs_of({"approach": read_knobs(Path(path))}, "approach")
        return cls(**{name: float(knobs[name]) for name in knobs.names})


@dataclass(frozen=True)
class Stand:
    """One standing pose on the map, what it is (for the refusal's words), and whether it is
    where the robot already stands (a turn in place)."""

    x: float
    y: float
    yaw_rad: float
    note: str
    in_place: bool = False

    @property
    def yaw_deg(self) -> float:
        """The heading in degrees, (-180, 180]."""
        return math.degrees(math.atan2(math.sin(self.yaw_rad), math.cos(self.yaw_rad)))


def stands(
    target: Target,
    relation: Relation,
    robot: Pose2D,
    distances: Distances | None = None,
    distance_m: float | None = None,
) -> list[Stand]:
    """The standing poses for ``relation`` to ``target``, best first: the stand point, then the
    ring's alternatives (none for a place's own pose; ``beside`` is the nearer side, the other
    side, then the nearer side's ring). ``distance_m`` is a spoken distance;
    ``distances`` None are the defaults."""
    distances = distances or Distances()
    if relation == "face":
        return [_turn(target, robot, "turn where it stands to face it")]
    if target.kind == "place":
        return _place(target, relation, distances, distance_m)
    if relation == "beside":  # both sides, the nearer first: a thing's two tie, so right first
        right = stands(target, "right_of", robot, distances, distance_m)
        left = stands(target, "left_of", robot, distances, distance_m)
        gap = [math.hypot(side[0].x - robot.x, side[0].y - robot.y) for side in (right, left)]
        near, far = (right, left) if gap[0] <= gap[1] + 1e-6 else (left, right)
        return [near[0], far[0], *near[1:]]
    toward = _toward(target, robot)  # from the target to the robot
    lateral = relation in ("right_of", "left_of")
    d = distance_m if distance_m is not None else distances.side_m if lateral else distances.front_m
    if target.kind == "person":
        d = max(d, distances.person_min_m + 2.0 * target.sigma_m)
    now = math.hypot(robot.x - target.x, robot.y - target.y)
    if relation == "to" and now <= d + distances.near_m:  # never backs away from "come closer"
        return [_turn(target, robot, f"already {now:.2f} m from it: turn where it stands")]
    angle, heading = _frame(target, relation, toward)
    return _ring(target, angle, d, heading, distances.ring_deg, relation)


def facing_from(person_x: float, person_y: float, robot: Pose2D, head_yaw_deg: float) -> float:
    """A person's facing on the map: toward the robot, turned by their head's yaw (degrees,
    positive when they look to their own left)."""
    return math.atan2(robot.y - person_y, robot.x - person_x) + math.radians(head_yaw_deg)


def _turn(target: Target, robot: Pose2D, note: str) -> Stand:
    """Where the robot stands, turned to face the target (its heading kept when it stands on
    the target's point)."""
    dx, dy = target.x - robot.x, target.y - robot.y
    yaw = math.atan2(dy, dx) if math.hypot(dx, dy) > 1e-6 else robot.yaw_rad
    return Stand(robot.x, robot.y, yaw, note, in_place=True)


def _toward(target: Target, robot: Pose2D) -> float:
    """The direction from the target to the robot; the robot's back when it stands on it."""
    dx, dy = robot.x - target.x, robot.y - target.y
    if math.hypot(dx, dy) < 1e-6:
        return robot.yaw_rad + math.pi
    return math.atan2(dy, dx)


def _frame(target: Target, relation: Relation, toward: float) -> tuple[float, float | None]:
    """``(direction of the stand point from the target, heading)``; heading None means facing
    the target. A person's frame is their own: ``to`` comes from where the robot is, the rest
    from their facing, and at a side the robot looks where they look. A thing's frame is the
    viewer's: its front faces the robot, its right is the robot's right as it looks at it."""
    view = toward + math.pi  # from the robot to the target
    if target.kind == "person":
        f = target.facing_rad if target.facing_rad is not None else toward
        frames: dict[str, tuple[float, float | None]] = {
            "to": (toward, None),
            "in_front_of": (f, None),
            "behind": (f + math.pi, f),
            "right_of": (f - math.pi / 2.0, f),
            "left_of": (f + math.pi / 2.0, f),
        }
    else:
        frames = {
            "to": (toward, None),
            "in_front_of": (toward, None),
            "behind": (view, None),
            "right_of": (view - math.pi / 2.0, None),
            "left_of": (view + math.pi / 2.0, None),
        }
    return frames[relation]


def _ring(
    target: Target,
    angle: float,
    d: float,
    heading: float | None,
    ring_deg: float,
    relation: str,
) -> list[Stand]:
    """The stand point ``d`` out from the target along ``angle``, and the two alternatives
    turned ``ring_deg`` about the target either way; heading None faces the target."""
    out = []
    for turn in (0.0, ring_deg, -ring_deg):
        a = angle + math.radians(turn)
        x, y = target.x + d * math.cos(a), target.y + d * math.sin(a)
        yaw = heading if heading is not None else math.atan2(target.y - y, target.x - x)
        side = "left" if turn > 0 else "right"
        note = relation if turn == 0.0 else f"{relation} turned {abs(turn):.0f} deg {side}"
        out.append(Stand(x, y, yaw, note))
    return out


def _place(
    target: Target, relation: Relation, distances: Distances, distance_m: float | None
) -> list[Stand]:
    """A place of the book is a standing pose: to/beside/in front of it is the pose itself; at
    its sides or behind it, an offset in its own frame, facing its heading."""
    heading = target.facing_rad if target.facing_rad is not None else 0.0
    if relation in ("to", "beside", "in_front_of"):
        return [Stand(target.x, target.y, heading, "the place itself")]
    lateral = relation != "behind"
    d = distance_m if distance_m is not None else distances.side_m if lateral else distances.front_m
    angle = {
        "behind": heading + math.pi,
        "right_of": heading - math.pi / 2.0,
        "left_of": heading + math.pi / 2.0,
    }[relation]
    return _ring(target, angle, d, heading, distances.ring_deg, relation)
