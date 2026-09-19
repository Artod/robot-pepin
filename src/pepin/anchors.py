"""The one-seating anchor file, and the seating test every tie measurement has to pass.

A pose graph built on the filter's odometry starts wherever that odometry's origin happens to be:
its nodes know the shape of the flat and nothing about where the lidar map's origin sits. ONE
transform ties the two — ``map <- graph`` (:func:`pepin.measurements.graph_anchor`) — and it is a
property of the PAIR (this map, this graph database), not of a session. The same database opened
again beside the same map deserves the same transform; only a new database, or a new map, needs
a new one.

Two things live here. The first is the SEATING TEST (:func:`seating_refusal`): what the tracker's
own error bar must be for a moment to be worth measuring the tie from at all. A fit is not an
error bar — at the charger, along a sofa, the lidar's seatings spread up to 55 cm in y at fit
0.67-0.79 because the scan is pinned in one axis only — so the gate is the published covariance
and not the score. The second is the old ONE-SEATING FILE, ``<map id>.graph_anchor.json`` beside
the map, named by the map's own identity (``pepin_bringup.msgs.map_id``,
``239x215@-18.53,-4.38``) with the characters a file name would rather not carry replaced. The
file carries that identity inside it too: a name can collide, an identity cannot.

That file is now a STARTING POINT and nothing more. A transform fitted to one seating carries
that seating's lever arm into every word the graph ever says — re-measured from another single
seating across an RTAB-Map restart on 2026-09-17 it moved 0.40 m and 6.45 deg — and it used to
be RE-LEARNED at runtime whenever the graph and the lidar disagreed for five seconds, which is
fitting a constant to a variable and is how camera-only acquired a systematic offset. The tie is
measured over MANY places instead (:mod:`pepin.graphtie`), and this file is read only until the
first pairs exist, with the error bar that measurement gave it
(:data:`pepin.graphtie.FILE_TIE_SIGMA_M`).

What the file still buys is the wake-up. On the charger the cart's last pose is known, but a
lidar-less start has nothing that says where the cart is on the map; with a tie on disk, the
graph recognising the place IS the answer, and the tracker is told before its first scan.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from pepin.odometry import Pose2D

SUFFIX = ".graph_anchor.json"
# What a seating must be worth for the WHOLE graph's frame to be learned from it. A fit is not an
# error bar: at "home" (the charger, along a sofa) the lidar's seatings spread up to 55 cm in y
# within minutes at fit 0.67-0.79, because the scan there is pinned in one axis only — and an
# anchor learned from such a seating carries that error into every word the graph ever says
# (2026-09-14: after the first accepted closure the graph's word sat 24-28 cm from the lidar).
# The peak's own covariance is the error bar (/tracker_pose, covariance=peak, NEES-calibrated),
# so the anchor waits for a seating the scan pins in BOTH axes. 3 cm because that is where the
# gate starts to be a gate: over tapes 0293-0298 the worse of the two position sigmas has a
# median of 1.50 cm and a p90 of 3.18 cm, so 3 cm refuses the worst 11 % of seatings and 1 cm
# would refuse 79 % — an anchor that is never learned is its own failure.
ANCHOR_MAX_SIGMA_M = 0.03
ANCHOR_MAX_SIGMA_DEG = 1.0


@dataclass(frozen=True)
class Anchor:
    """``map <- graph`` for one (map, graph database) pair: the transform, the map it is for, the
    moment it was learned, where this copy came from and how many times it has been re-learned."""

    pose: Pose2D
    map_id: str
    learned_at: float = 0.0
    origin: str = "learned"
    relearns: int = 0

    def described(self) -> str:
        """The anchor in one phrase for a report line: metres, degrees and where it came from."""
        return (
            f"({self.pose.x:+.2f}, {self.pose.y:+.2f}, {math.degrees(self.pose.theta):+.1f} deg)"
            f" from {self.origin}"
            + (f" {self.relearns}" if self.relearns and self.origin != "file" else "")
        )


def describe_sigma(sigma: tuple[float, float, float] | None) -> str:
    """One seating's uncertainty for a report line: ``1.0/1.3 cm, 0.30 deg`` (x, y, heading),
    or ``unknown`` when the belief carried no covariance."""
    if sigma is None:
        return "unknown"
    return f"{sigma[0] * 100.0:.1f}/{sigma[1] * 100.0:.1f} cm, {math.degrees(sigma[2]):.2f} deg"


def seating_refusal(
    sigma: tuple[float, float, float] | None,
    max_sigma_m: float = ANCHOR_MAX_SIGMA_M,
    max_sigma_deg: float = ANCHOR_MAX_SIGMA_DEG,
) -> str | None:
    """Why this seating may not be learned from, in one phrase for a log, or ``None`` when it may.

    ``sigma`` is the tracker's own error bar at the moment — the roots of its covariance diagonal
    (x, y in metres, heading in radians) — and the anchor is a constant of the pair: whatever it
    is learned from is baked into every word the graph says until the file is rewritten. So the
    seating must be sharp in BOTH position axes, not merely well-matched: a scan pinned along a
    corridor reports an honest fit and a metre of freedom in the other axis.
    """
    if sigma is None:
        return "the tracker's belief carries no covariance"
    if max(sigma[0], sigma[1]) > max_sigma_m:
        return (
            f"the lidar's seating is soft ({describe_sigma(sigma)}, over"
            f" {max_sigma_m * 100.0:.1f} cm)"
        )
    if math.degrees(sigma[2]) > max_sigma_deg:
        return (
            f"the lidar's heading is soft ({describe_sigma(sigma)}, over {max_sigma_deg:.1f} deg)"
        )
    return None


def map_slug(map_id: str) -> str:
    """A map's identity as a file name: ``239x215@-18.53,-4.38`` -> ``239x215_-18.53_-4.38``."""
    return re.sub(r"[^A-Za-z0-9.+-]", "_", map_id)


def room_of(provenance: str) -> str:
    """The ROOM a map is of, from the provenance its identity carries: ``seed:flat3_straight`` ->
    ``flat3_straight``, ``resume:flat3_straight`` -> ``flat3_straight``, and ``""`` for a room with
    no name yet (``fresh``, :data:`pepin.worldmap.BORN_FRESH`).

    This is what every file beside the map should be named by, and the size@origin id is not. Both
    spellings were measured on the parked cart on 2026-09-18: switching the board from the seed pgm
    to the volume's exported slice changed the served id from ``239x215@-18.53,-4.38`` to
    ``280x250@-19.48,-5.48`` — the same room, the same frame, the same lattice, the cart's pose
    unchanged and the lidar's fit better (0.72 -> 0.96) — and every ``<map id>.graph_*`` file went
    invisible. The id changes again whenever the volume grows a row; the room does not.
    """
    _, _, room = provenance.partition(":")
    return room.strip()


def room_from_identity(text: str) -> str:
    """The room named by one ``/map_identity`` message (:meth:`pepin.worldmap.MapIdentity`, as
    ``pepin_bringup.depth_fusion`` publishes it), or ``""`` when it names none.

    The provenance field first (``from``: ``seed:<room>``), and the volume's own snapshot path
    second (``world_path``: ``…/flat3_straight.world.npz`` -> ``flat3_straight``), because a volume
    born fresh has no provenance to read but still keeps its files under the room's name. Anything
    that does not parse is no room at all: a name guessed from a malformed message would put the
    calibration of one flat beside another.
    """
    try:
        said = json.loads(text)
    except (TypeError, ValueError):
        return ""
    if not isinstance(said, dict):
        return ""
    room = room_of(str(said.get("from", "")))
    if room:
        return room
    path = str(said.get("world_path", ""))
    stem = Path(path).name
    for suffix in (".world.npz", ".world"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return ""


def anchor_path(directory: Path | str, map_id: str) -> Path:
    """Where the anchor of ``map_id`` lives: ``<directory>/<slug>.graph_anchor.json``."""
    return Path(directory) / f"{map_slug(map_id)}{SUFFIX}"


def load_anchor(directory: Path | str, key: str, identity: str | None = None) -> Anchor | None:
    """The anchor stored under ``key``, or ``None`` when no file has been written for it yet.

    ``identity`` is what the file must say it is about, when the caller has something to check: with
    the legacy size@origin naming the file name and the identity inside are one string, and a
    mismatch means a file of another map — a cart confidently in the wrong room. Under a ROOM name
    they are two different things (a migrated file still carries whichever grid id was served on the
    day it was written, and that id changes when the volume grows), so the caller passes ``None``
    and the room name is the identity.

    Raises ``ValueError`` when the file is there but cannot be believed — unreadable JSON, a missing
    field, or an identity that is not the one asked for — because silence is the safer answer only
    when it is said out loud.
    """
    path = anchor_path(directory, key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        pose = Pose2D(float(payload["x"]), float(payload["y"]), float(payload["theta"]))
        stored = str(payload["map"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} is not an anchor: {error}") from error
    if identity is not None and stored != identity:
        raise ValueError(f"{path} is the anchor of map {stored}, not of {identity}")
    return Anchor(
        pose=pose,
        map_id=stored,
        learned_at=float(payload.get("learned_at", 0.0)),
        origin="file",
        relearns=int(payload.get("relearns", 0)),
    )


def save_anchor(directory: Path | str, anchor: Anchor) -> Path:
    """Write ``anchor`` beside its map and answer with the path it was written to."""
    path = anchor_path(directory, anchor.map_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "map": anchor.map_id,
        "x": anchor.pose.x,
        "y": anchor.pose.y,
        "theta": anchor.pose.theta,
        "learned_at": anchor.learned_at,
        "relearns": anchor.relearns,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path
