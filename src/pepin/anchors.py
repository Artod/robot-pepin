"""The anchor between a lidar map and a pose graph, kept on disk beside the map.

A pose graph built on the filter's odometry starts wherever that odometry's origin happens to be:
its nodes know the shape of the flat and nothing about where the lidar map's origin sits. ONE
transform ties the two — ``map <- graph``, the ANCHOR (:func:`pepin.measurements.graph_anchor`) —
and it is a property of the PAIR (this map, this graph database), not of a session. The same
database opened again beside the same map deserves the same anchor; only a new database, or a new
map, needs a new one.

So it is written where the pair can find it: ``<map id>.graph_anchor.json`` beside the map, named
by the map's own identity (``pepin_bringup.msgs.map_id``, ``239x215@-18.53,-4.38``) with the
characters a file name would rather not carry replaced. The file carries that identity inside it
too: a name can collide, an identity cannot.

What the file buys is the wake-up. On the charger the cart's last pose is known, but a lidar-less
start has nothing that says where the cart is on the map; with the anchor on file, the graph
recognising the place IS the answer, and the tracker is told before its first scan.

What it costs is staleness. The odom frame the graph rides resets when the board restarts, and
the anchor learned in the previous session then points at nothing. :class:`AnchorWatch` is the
answer to that: with the lidar driving and its fit good, a disagreement that HOLDS is evidence
the anchor is stale, and one that does not hold is a closure landing.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path

from pepin.odometry import Pose2D

SUFFIX = ".graph_anchor.json"
# What "the anchor no longer holds" means. A loop closure moves the graph's word by centimetres
# to a few tens of them (measured 2026-09-14: 2.2-3.2 cm median against the lidar truth on a
# printer errand), so half a metre is past anything a closure does; 20 degrees is past anything
# but a frame that has moved under the graph. Five seconds is what tells the two apart: a closure
# lands and the word is back, a stale anchor stays wrong for as long as one looks at it.
RELEARN_GAP_M = 0.5
RELEARN_GAP_DEG = 20.0
RELEARN_HOLD_S = 5.0
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


def anchor_path(directory: Path | str, map_id: str) -> Path:
    """Where the anchor of ``map_id`` lives: ``<directory>/<slug>.graph_anchor.json``."""
    return Path(directory) / f"{map_slug(map_id)}{SUFFIX}"


def load_anchor(directory: Path | str, map_id: str) -> Anchor | None:
    """The anchor stored for ``map_id``, or ``None`` when no file has been written for it yet.

    Raises ``ValueError`` when the file is there but cannot be believed — unreadable JSON, a
    missing field, or an identity that is not this map's — because a wrong anchor is a cart
    confidently in the wrong room, and silence is the safer answer only when it is said out loud.
    """
    path = anchor_path(directory, map_id)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        pose = Pose2D(float(payload["x"]), float(payload["y"]), float(payload["theta"]))
        stored = str(payload["map"])
    except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
        raise ValueError(f"{path} is not an anchor: {error}") from error
    if stored != map_id:
        raise ValueError(f"{path} is the anchor of map {stored}, not of {map_id}")
    return Anchor(
        pose=pose,
        map_id=map_id,
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


class AnchorWatch:
    """Says when a stored anchor no longer holds, from the graph's word and the tracker's belief.

    The anchor is a constant of the pair, so it is re-learned only on evidence that survives:
    a TRUSTED tracker (the lidar driving, its fit at or above what the caller demands) and the
    graph's word disagreeing by more than ``max_gap_m`` or ``max_gap_deg`` for longer than
    ``hold_s`` without a break. With the lidar silent the clock is not even started — there is
    nothing to re-learn FROM, and a graph corrected against a drifting dead-reckoned pose would
    write its own drift into the file.
    """

    def __init__(
        self,
        max_gap_m: float = RELEARN_GAP_M,
        max_gap_deg: float = RELEARN_GAP_DEG,
        hold_s: float = RELEARN_HOLD_S,
    ) -> None:
        self._max_gap_m = max_gap_m
        self._max_gap_deg = max_gap_deg
        self._hold_s = hold_s
        self._since: float | None = None

    @property
    def since(self) -> float | None:
        """The moment the present disagreement started, or ``None`` while there is none."""
        return self._since

    def disagrees(self, gap_m: float, gap_deg: float) -> bool:
        """Whether this gap is bigger than a loop closure ever accounts for."""
        return gap_m > self._max_gap_m or abs(gap_deg) > self._max_gap_deg

    def update(self, now: float, trusted: bool, gap_m: float, gap_deg: float) -> bool:
        """Feed one word: ``True`` the moment a trusted tracker has disagreed for ``hold_s``.

        The clock restarts after a ``True`` and whenever the gap closes or the tracker stops
        being trusted, so one stale anchor produces exactly one re-learn.
        """
        if not trusted or not self.disagrees(gap_m, gap_deg):
            self._since = None
            return False
        if self._since is None:
            self._since = now
            return False
        if now - self._since < self._hold_s:
            return False
        self._since = None
        return True
