"""What a pose graph's word about the cart is worth: what the graph has RECOGNISED, not what
it has integrated.

A graph answers where the cart is at every optimisation, and the answer always looks the same
whether the graph has just recognised a place it knows or has been dead-reckoning over its own
odometry for half a session. The two are not worth the same thing, and until this module the
word travelled claiming ``fit`` 1.0 in both cases -- a claim the board then published as its
own confidence. Measured on the carry test of 2026-09-14 21:12: the cart was carried 2 m by
hand, the graph recognised nothing for the whole 64-s drive that followed (highest hypothesis
0.04, not one closure), its word was therefore the OLD anchor plus odometry -- and the tracker
published fit 1.00 while the belief was 1.5-2 m wrong, so the goal server's lost ladder saw
nothing to stop and the whole-map candidate path had nothing to fire on.

The trust here is a decay clock started by RECOGNITION. A closure accepted, or a proximity
link added, ties the node the graph is building now to an OLDER node: at that instant the
graph's word is worth 1.0, because a tie is the one thing on this robot that can undo
accumulated drift. From there it decays with the distance the cart has driven since,
``exp(-d / trust_m)``: the EKF odometry the graph is built on drifted 0.79 m and 31 degrees
over 25 m of driving on 2026-09-14, and 22 cm over 12 m, so a word five metres past its last
tie is a word about where the cart probably is, not where it is.

And a word made from an anchor READ FROM A FILE is capped (:data:`FILE_ANCHOR_TRUST`) until
the graph recognises something in this session: the wake-up anchor is a relation between two
frames that was true in another session, and nothing has confirmed it yet. It is a guess, and
a guess with a 1.0 on it is the failure this module exists to stop.

Nothing here is ROS: a mapping of RTAB-Map's statistics in (``/rtabmap/info``'s ``stats_keys``
and ``stats_values``), a number and a report line out.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

__all__ = [
    "ACCEPTED_HYPOTHESIS_ID",
    "DISTANCE_TRAVELLED_M",
    "FILE_ANCHOR_TRUST",
    "GRAPH_TRUST_M",
    "HIGHEST_HYPOTHESIS",
    "LOOP_ID",
    "PROXIMITY_ICP",
    "PROXIMITY_VISUAL",
    "GraphReport",
    "GraphTrust",
    "stat",
]

# How far the cart may drive past a tie before the graph's word is worth 1/e of it. The EKF
# odometry RTAB-Map is built on drifted 0.79 m and 31 degrees over 25 m of driving and 22 cm
# over 12 m (2026-09-14): five metres is where the drift stops being centimetres, and it is
# short enough that a carried cart's word falls under every gate downstream within a few metres
# (0.45, the whole-map candidate floor, at 4.0 m; 0.30 at 6.0 m).
GRAPH_TRUST_M = 5.0
# The most a word built on an anchor read from a file may claim before the graph has recognised
# anything in this session. The anchor ties the graph's frame to the map and was measured in
# ANOTHER session: a board restart moves the odom frame under it and the file is then stale by
# exactly that reset. 0.30 sits under the whole-map candidate floor (pepin.watch.ADMIT_FIT,
# 0.45) and just above the tracker's own lost_below (0.25): such a word may still be fused as a
# measurement, but it can neither re-seed the pose nor make a lost tracker look found.
FILE_ANCHOR_TRUST = 0.3

# RTAB-Map's statistics, by the names it publishes them under (rtabmap/Statistics.h). The keys
# carry a trailing unit segment -- "Loop/Id/", "Memory/Distance_travelled/m" -- which :func:`stat`
# tolerates, so a version that renames a unit does not silently read zero.
LOOP_ID = "Loop/Id"
ACCEPTED_HYPOTHESIS_ID = "Loop/Accepted_hypothesis_id"
PROXIMITY_VISUAL = "Proximity/Space_detections_added_visually"
PROXIMITY_ICP = "Proximity/Space_detections_added_icp_multi"
DISTANCE_TRAVELLED_M = "Memory/Distance_travelled/m"
HIGHEST_HYPOTHESIS = "Loop/Highest_hypothesis_value"
# What counts as the graph tying the node it is building now to an OLDER one. A loop closure
# and a proximity link are both links to a node that is not the previous one -- a place
# recognised by its appearance, or one the graph walked back to -- which is exactly the
# correction odometry cannot make. A neighbour link is not here: every node has one.
TIE_STATS = (LOOP_ID, ACCEPTED_HYPOTHESIS_ID, PROXIMITY_VISUAL, PROXIMITY_ICP)


def stat(stats: Mapping[str, float], name: str) -> float | None:
    """One RTAB-Map statistic by its name, tolerating the unit segment the key carries
    (``Loop/Id`` finds ``Loop/Id/``); ``None`` when this message does not carry it."""
    if name in stats:
        return stats[name]
    for key, value in stats.items():
        if key.rstrip("/") == name or key.startswith(f"{name}/"):
            return value
    return None


@dataclass(frozen=True)
class GraphReport:
    """What the graph's word is worth right now and why: the trust itself (the ``fit`` the word
    travels with), the metres driven since the graph last tied the present to an older node,
    how many such ties this session has seen, the highest loop hypothesis of the last message
    (how close the graph is to recognising something), and whether the cap for a file anchor is
    what is holding the trust down."""

    trust: float
    since_m: float
    ties: int
    hypothesis: float
    capped: bool
    heard: bool

    def text(self) -> str:
        """``0.34 trust, 5.4 m since tie 2, hypothesis 0.04`` for a report line."""
        cap = ", capped as a file anchor" if self.capped else ""
        if not self.heard:
            return f"{self.trust:.2f} trust, no /rtabmap/info yet{cap}"
        tie = f"tie {self.ties}" if self.ties else "the start, no tie yet"
        return (
            f"{self.trust:.2f} trust, {self.since_m:.1f} m since {tie},"
            f" hypothesis {self.hypothesis:.2f}{cap}"
        )


class GraphTrust:
    """The decay clock on a pose graph's word: fed every ``/rtabmap/info`` message, it answers
    how much the graph's next word is worth.

    1.0 at a tie -- a loop closure accepted or a proximity link added, the graph recognising a
    place it already has -- decaying as ``exp(-d / trust_m)`` with the distance driven since
    (:data:`GRAPH_TRUST_M`), and never above :data:`FILE_ANCHOR_TRUST` while the anchor comes
    from a file and this session has recognised nothing.

    The clock starts at the node's first message, not at the graph's birth: a node that joins a
    graph already 25 m old has no evidence about what happened before it was listening, and the
    anchor it holds was either learned from the tracker just now (worth 1.0) or read from a file
    (capped). A distance counter that goes BACKWARDS is RTAB-Map restarted, and the distance of
    the new session counts from zero rather than as a negative decay.

    Pure: the node offers statistics and reads a report; ``trust_m`` is a live flag the node
    writes straight onto the instance.
    """

    def __init__(
        self, trust_m: float = GRAPH_TRUST_M, file_anchor_trust: float = FILE_ANCHOR_TRUST
    ) -> None:
        self.trust_m = trust_m
        self.file_anchor_trust = file_anchor_trust
        self._since_m = 0.0  # metres driven since the last tie (or since the first message)
        self._ties = 0  # ties seen this session
        self._hypothesis = 0.0  # the last message's highest loop hypothesis
        self._travelled: float | None = None  # the last distance counter read, to difference it
        self._heard = 0  # messages consumed

    @property
    def ties(self) -> int:
        """How many times this session the graph has tied the present to an older node."""
        return self._ties

    @property
    def since_m(self) -> float:
        """Metres driven since the last such tie (since the first message, before any)."""
        return self._since_m

    def update(self, stats: Mapping[str, float]) -> bool:
        """One ``/rtabmap/info`` message in: the distance counter differenced onto the clock and
        the tie statistics read. Returns whether this message tied the present to an older node
        (which resets the clock and is the moment the word is worth 1.0)."""
        self._heard += 1
        hypothesis = stat(stats, HIGHEST_HYPOTHESIS)
        if hypothesis is not None:
            self._hypothesis = float(hypothesis)
        travelled = stat(stats, DISTANCE_TRAVELLED_M)
        if travelled is not None:
            distance = max(float(travelled), 0.0)
            previous = self._travelled
            # The first message is the baseline, and a counter that fell is a restarted graph:
            # what the new session has driven is the whole of the new reading.
            if previous is None:
                step = 0.0
            else:
                step = distance - previous if distance >= previous else distance
            self._since_m += max(step, 0.0)
            self._travelled = distance
        tied = any((stat(stats, name) or 0.0) > 0.0 for name in TIE_STATS)
        if tied:
            self._ties += 1
            self._since_m = 0.0
        return tied

    def trust(self, anchor_from_file: bool = False) -> float:
        """What the graph's word is worth right now, in [0, 1]: the decay since the last tie,
        capped at :attr:`file_anchor_trust` while ``anchor_from_file`` and no tie has happened
        in this session."""
        return self.report(anchor_from_file).trust

    def report(self, anchor_from_file: bool = False) -> GraphReport:
        """The trust and everything a report line says about it (:class:`GraphReport`)."""
        decay = math.exp(-self._since_m / self.trust_m) if self.trust_m > 0.0 else 0.0
        trust = min(max(decay, 0.0), 1.0)
        capped = anchor_from_file and self._ties == 0 and trust > self.file_anchor_trust
        if capped:
            trust = self.file_anchor_trust
        return GraphReport(
            trust=trust,
            since_m=self._since_m,
            ties=self._ties,
            hypothesis=self._hypothesis,
            capped=capped,
            heard=self._heard > 0,
        )
