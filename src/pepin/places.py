"""Named places: the words a person or a language model sends the robot to.

TWO KINDS, AND THE SECOND ONE IS WORLD R's. :class:`Place` is a name for a coordinate in a frozen
grid's frame ("kitchen" = x, y and, optionally, which way to face on arrival), kept beside the map
file as ``<map>.places.json`` so a map and its vocabulary travel together. That works for a map
that never moves. Under World R the map is RTAB-Map's loop-closed graph, and when a closure lands
the room BENDS: every coordinate written down before it now names a spot a few centimetres — or,
across a session, half a metre — off the furniture it was meant for. A coordinate cannot be right
in a map that moves.

:class:`GraphPlace` is the answer: a place is THE POSE THE CART HAD RELATIVE TO A LABELLED GRAPH
NODE when it was marked. The node is a thing in the room, the offset is a measurement of the cart
against it, and both are invariant under every optimisation the graph will ever do — so when the
graph moves the node, the place rides with it and stays on the furniture. Resolving one back into a
coordinate needs the node's CURRENT optimised pose, which is what ``/rtabmap/mapGraph`` carries
(``poses_id`` beside ``poses``, rtabmap_msgs/msg/MapGraph.msg:11-12), and that is
pepin_bringup.places' whole job.

WHY THE NODE ID IS OURS AND NOT RTAB-MAP'S TO REMEMBER. The label is set on the node as well
(``set_label``), because that is what makes the place a thing in RTAB-Map's own tools — but it is
not storage. A label put on a node that is in the working memory only flips a dirty bit
(rtabmap/core/Signature.h:76) and is written when the node leaves it; beside a LOADED database
nothing is written at all (``Mem/IncrementalMemory`` false), and with ``Mem/InitWMWithAllNodes``
true every node is in the working memory, so a label set in a localising session never reaches the
file. The id and the offset therefore live in a file of ours beside the database
(:func:`graph_places_path`), and the label is the courtesy copy.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

from pepin.measurements import compose, inverse
from pepin.odometry import Pose2D


@dataclass(frozen=True)
class Place:
    """A named pose in the map frame; ``theta_deg`` is the preferred final heading, or None."""

    name: str
    x: float
    y: float
    theta_deg: float | None = None

    @property
    def xy(self) -> tuple[float, float]:
        """The place as a planner goal."""
        return (self.x, self.y)

    @property
    def theta(self) -> float | None:
        """Preferred heading in radians, or None when any heading will do."""
        return math.radians(self.theta_deg) if self.theta_deg is not None else None


def places_path(map_path: Path) -> Path:
    """Where a map keeps its places: ``data/maps/foo.npz`` -> ``data/maps/foo.places.json``."""
    return map_path.with_suffix(".places.json")


def load_places(map_path: Path) -> dict[str, Place]:
    """Places of a map by name; an empty dict when the map has none yet."""
    path = places_path(map_path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {name: Place(name=name, **entry) for name, entry in data.get("places", {}).items()}


def save_places(map_path: Path, places: dict[str, Place]) -> Path:
    """Write the places file next to the map (sorted by name); returns its path."""
    path = places_path(map_path)
    entries = {
        name: {k: v for k, v in asdict(place).items() if k != "name"}
        for name, place in sorted(places.items())
    }
    payload = {
        "map": map_path.name,
        "frame": "map frame of that grid: meters, x/y as in the .npz, theta_deg counter-clockwise",
        "places": entries,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


def resolve_goal(tokens: list[str], map_path: Path) -> tuple[tuple[float, float], Place | None]:
    """``["kitchen"]`` or ``["-2.0", "0.5"]`` into a goal in meters (and the Place, if named).

    Raises ``ValueError`` naming the known places when a name is unknown.
    """
    if len(tokens) == 2:
        return (float(tokens[0]), float(tokens[1])), None
    if len(tokens) != 1:
        raise ValueError("a goal is either a place name or two numbers X Y")
    places = load_places(map_path)
    place = places.get(tokens[0])
    if place is None:
        known = ", ".join(sorted(places)) or "none yet"
        raise ValueError(
            f"unknown place {tokens[0]!r} for {map_path.name}; known: {known} "
            f"(add one: uv run python scripts/places.py {map_path} add NAME X Y)"
        )
    return place.xy, place


def heading_residual_deg(target_deg: float, current_deg: float) -> float:
    """How far the cart must still turn to face ``target_deg`` from ``current_deg``, in
    (-180, 180] degrees: positive is counter-clockwise, the Spin behaviour's convention."""
    residual = (target_deg - current_deg + 180.0) % 360.0 - 180.0
    return 180.0 if residual == -180.0 else residual


# Where a graph's places live: beside the database whose nodes they hang on, because a node id
# means nothing without the database it is an id in. The suffix is spelled out as a literal so a
# test can pin the name on both sides (tests/unit/test_places.py).
GRAPH_PLACES_SUFFIX = ".places.json"


def graph_places_path(database: Path) -> Path:
    """Where the places of a graph database live: ``/maps/rtabmap.db`` ->
    ``/maps/rtabmap.places.json``."""
    return database.with_suffix(GRAPH_PLACES_SUFFIX)


@dataclass(frozen=True)
class GraphPlace:
    """A named place in a graph that bends: the cart's pose RELATIVE TO the labelled node ``node``
    at the moment it was marked, in that node's own frame (metres, degrees CCW).

    Nothing here is a coordinate in the map, on purpose: the map moves. :meth:`at` is what turns it
    back into one, against whatever pose the graph gives that node now.
    """

    name: str
    node: int
    dx: float
    dy: float
    dtheta_deg: float
    marked_at: float = 0.0  # the stamp of the pose it was measured from, the board's clock

    @classmethod
    def measured(
        cls, name: str, node: int, node_pose: Pose2D, cart: Pose2D, stamp: float = 0.0
    ) -> GraphPlace:
        """The place a cart standing at ``cart`` marks against the node the graph puts at
        ``node_pose``: the cart seen from the node, which is the one part of this that a later
        optimisation cannot change."""
        offset = compose(inverse(node_pose), cart)
        return cls(
            name=name,
            node=int(node),
            dx=offset.x,
            dy=offset.y,
            dtheta_deg=math.degrees(offset.theta),
            marked_at=stamp,
        )

    def at(self, node_pose: Pose2D) -> Pose2D:
        """Where this place is in the map now, given the node's CURRENT optimised pose."""
        return compose(node_pose, Pose2D(self.dx, self.dy, math.radians(self.dtheta_deg)))

    @property
    def reach_m(self) -> float:
        """How far the cart stood from its node when the place was marked: what says whether a
        place is anchored on the node beside it or on one across the room."""
        return math.hypot(self.dx, self.dy)


def graph_place_pose(place: GraphPlace, poses: dict[int, Pose2D]) -> Pose2D | None:
    """One place as a pose in the map, or ``None`` when the graph no longer holds its node — a
    pruned node is a place nobody can drive to, and a stale coordinate is worse than a refusal."""
    node_pose = poses.get(place.node)
    return None if node_pose is None else place.at(node_pose)


def load_graph_places(path: Path) -> dict[str, GraphPlace]:
    """The places of a graph database by name; an empty book when the file is absent or is not
    one (a torn file must not take the vocabulary of the room with it)."""
    try:
        data = json.loads(path.read_text())
        entries = data["places"]
    except (OSError, ValueError, KeyError, TypeError):
        return {}
    places: dict[str, GraphPlace] = {}
    for name, entry in entries.items() if isinstance(entries, dict) else ():
        try:
            places[name] = GraphPlace(
                name=name,
                node=int(entry["node"]),
                dx=float(entry["dx"]),
                dy=float(entry["dy"]),
                dtheta_deg=float(entry["dtheta_deg"]),
                marked_at=float(entry.get("marked_at", 0.0)),
            )
        except (ValueError, KeyError, TypeError):
            continue  # one unreadable entry costs its own place and not the book
    return places


def save_graph_places(path: Path, places: dict[str, GraphPlace]) -> Path:
    """Write the book beside the database (sorted by name); returns its path.

    Written whole and not appended to: the file is small, and a half-written line in it would cost
    the room its vocabulary at the next start.
    """
    payload = {
        "frame": "each place is the cart's pose in the frame of its labelled graph node:"
        " dx, dy in metres, dtheta_deg counter-clockwise",
        "places": {
            name: {k: v for k, v in asdict(place).items() if k != "name"}
            for name, place in sorted(places.items())
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return path


# The latched topic the resolved places go out on, and the two the marking uses. Written here, with
# the rest of the vocabulary's contract, so both ends of the bridge spell one name
# (pepin_bringup.places, ros/tools/goto_ros.py).
PLACES_TOPIC = "/places"
MARK_TOPIC = "/places/mark"
MARKED_TOPIC = "/places/marked"


def resolved_places(
    places: dict[str, GraphPlace], poses: dict[int, Pose2D]
) -> dict[str, tuple[Place, GraphPlace]]:
    """Every place the graph can still answer for, as a coordinate in the map RIGHT NOW beside the
    graph place it came from. A place whose node the graph no longer holds is left out entirely."""
    answered: dict[str, tuple[Place, GraphPlace]] = {}
    for name, place in places.items():
        pose = graph_place_pose(place, poses)
        if pose is None:
            continue
        answered[name] = (
            Place(name=name, x=pose.x, y=pose.y, theta_deg=math.degrees(pose.theta)),
            place,
        )
    return answered


def places_json(places: dict[str, GraphPlace], poses: dict[int, Pose2D]) -> str:
    """The latched ``/places`` payload: every resolvable place as a coordinate, with the node it
    rides, how far the cart stood from that node when it was marked, and when.

    The coordinate is what a consumer drives to; the node and the reach are what say whether to
    believe it — a place anchored on a node the cart was standing on is worth more than one
    anchored across the room, and both are honest about which they are.
    """
    return json.dumps(
        {
            name: {
                "x": round(place.x, 3),
                "y": round(place.y, 3),
                "yaw_deg": round(place.theta_deg or 0.0, 1),
                "node": graph.node,
                "reach_m": round(graph.reach_m, 3),
                "marked_at": round(graph.marked_at, 3),
            }
            for name, (place, graph) in sorted(resolved_places(places, poses).items())
        }
    )


def places_from_json(text: str) -> dict[str, Place]:
    """One such payload back as coordinates to drive to, or an empty book when it is not one.

    Empty and not an exception, because the reader is a goal client on the board: a payload it
    cannot parse must fall back to the file beside the map, not refuse the drive.
    """
    try:
        heard = json.loads(text)
    except (TypeError, ValueError):
        return {}
    places: dict[str, Place] = {}
    for name, entry in heard.items() if isinstance(heard, dict) else ():
        try:
            places[name] = Place(
                name=name,
                x=float(entry["x"]),
                y=float(entry["y"]),
                theta_deg=float(entry["yaw_deg"]),
            )
        except (TypeError, ValueError, KeyError):
            continue
    return places
