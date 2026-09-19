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

WHAT THE DECAY CLOCK MISSED, tape 0333 of 2026-09-15 (a camera-only drive, the lidar muted):
RTAB-Map had been restarted many times that day and had recognised NOTHING against the database
it loaded since its last start, so the nodes it was building formed an unlinked segment placed
by this session's odometry alone. A word off that segment is odometry wearing a pose's clothes,
and the decay clock could not see it: the clock resets on any tie, including a tie INSIDE the
new segment, and the anchor on file (-10.33, +1.41, +53.3 deg) was still perfectly good for the
LINKED nodes it was measured on. The tracker took 19 such words, each about a metre and 150
degrees wrong (the last one 103 cm from the tracker, 102 more refused as too far), and the pose
flew. Two predicates came out of it, both here:

:class:`Recognition` — has RTAB-Map accepted a loop closure, a proximity link or a localisation
against the DATABASE IT LOADED since its own start? A closure to a node this session created is
not that: the test is the matched id against the first node id this session built. Until that
holds there is no COMMON FRAME, a word would be a number in another coordinate system, and that
-- not uncertainty -- is the only thing silence is the honest answer to. Once tied, the tie does
not expire: how old it is rides the word's covariance (:class:`GraphTrust`), not a clock.

:class:`Agreement` — how well the last words track the tracker's own pose carried forward by
odometry between them, as an rms residual over a short window, ``exp(-r / scale)``. This is
what tells a word riding a broken frame from a word riding a good one WITHOUT waiting for the
next closure: a 1 m constant offset reads as 1 m of residual and a trust of 5e-5, while a word
that follows odometry to the centimetre keeps its 1.0 whatever the metres since the last tie.
The residual is 3-DOF since 2026-09-18 — a Mahalanobis distance under the joint covariance of the
word and the belief, the same one :data:`pepin.fusion.GATE` judges — because a word can be in the
right place facing the wrong way, and until then such a word kept its 1.00.

Nothing here is ROS: a mapping of RTAB-Map's statistics in (``/rtabmap/info``'s ``stats_keys``
and ``stats_values``), the ids of the same message, a number and a report line out.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass

from pepin.fusion import SIGMA_XY_M
from pepin.measurements import GRAPH_FLOOR_XY_M

__all__ = [
    "ACCEPTED_HYPOTHESIS_ID",
    "AGREEMENT_SCALE",
    "AGREEMENT_SCALE_M",
    "AGREEMENT_WINDOW_S",
    "AGREEMENT_WORDS",
    "DISTANCE_TRAVELLED_M",
    "FILE_ANCHOR_TRUST",
    "GRAPH_TRUST_M",
    "HIGHEST_HYPOTHESIS",
    "LOOP_ID",
    "PROXIMITY_ICP",
    "PROXIMITY_VISUAL",
    "Agreement",
    "GraphReport",
    "GraphTrust",
    "InfoIds",
    "Recognition",
    "RecognitionReport",
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
# The residual at which a word is worth 1/e of itself when its trust is judged by AGREEMENT.
# 10 cm is the scale a graph word has when it is working: over tapes 0275/0276 the word sat
# 0.7-0.8 cm from the lidar truth while driving and 0-8 cm at rest, and 2.2-3.2 cm over a
# printer errand -- so 10 cm is several times the working spread and a tenth of the metre-scale
# error of tape 0333, which lands at exp(-10) and stops the word being admitted anywhere.
AGREEMENT_SCALE_M = 0.10
# ...and the same scale in the units the residual is actually judged in now: SIGMAS of the joint
# covariance of the word and the belief it is compared with, so a word that is right in position
# and 90 degrees wrong in heading is no longer a word that agrees. The number is DERIVED and not
# chosen: it is AGREEMENT_SCALE_M expressed in the joint sigma of a graph word worth its own floor
# (pepin.measurements.GRAPH_FLOOR_XY_M, 0.20 m) against a tracker publishing its own
# (pepin.fusion.SIGMA_XY_M, 0.05 m), so the curve a position-only residual rode before 2026-09-18
# is the curve it rides now, with the heading counted as well.
AGREEMENT_SCALE = AGREEMENT_SCALE_M / math.hypot(GRAPH_FLOOR_XY_M, SIGMA_XY_M)
# The window the residual is taken over: the last ten seconds, and at most this many words. Long
# enough that one word arriving mid-closure does not decide the trust, short enough that a frame
# that has just broken is not vouched for by the minute before it.
AGREEMENT_WINDOW_S = 10.0
AGREEMENT_WORDS = 10

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


@dataclass(frozen=True)
class InfoIds:
    """The graph ids of one ``/rtabmap/info``, as a node reads them off the message: the node
    RTAB-Map is building now, the node a loop closure matched, the node a proximity link matched,
    and whether the message carried a localisation pose against the loaded database. Zero is
    "not carried" -- RTAB-Map numbers its nodes from 1 -- so a message whose fields are missing
    reads as one that recognised nothing."""

    ref_id: int = 0
    loop_closure_id: int = 0
    proximity_detection_id: int = 0
    localized: bool = False


@dataclass(frozen=True)
class RecognitionReport:
    """Whether RTAB-Map's present nodes are tied to the database it loaded, and what stands
    behind the answer: how many such acceptances this start has seen, which database node the
    last one matched, and how many words the node withheld while there was no tie at all."""

    recognised: bool
    matches: int
    matched_id: int
    withheld: int

    def text(self) -> str:
        """``recognised on node 2841 (3 matches)`` for a report line, or the refusal that is
        keeping the node silent."""
        if self.recognised:
            return f"recognised on node {self.matched_id} ({self.matches} matches)"
        return f"unrecognised since start: {self.withheld} words withheld"


class Recognition:
    """Has RTAB-Map been tied to the database it LOADED since its own start -- is there a COMMON
    FRAME to speak in at all?

    Fed every ``/rtabmap/info`` -- the statistics and the message's graph ids -- it answers the
    one question no covariance can: whether the nodes the graph is building now are tied to the
    map on disk at all, or form an unlinked segment placed by this session's odometry. A word off
    such a segment is not an uncertain pose, it is a number in ANOTHER COORDINATE SYSTEM (tape
    0333, 2026-09-15: 19 words about a metre and 150 degrees wrong), and the anchor learned from
    one is worse still, because it is baked into every word that follows. That is the only thing
    silence is ever the honest answer to.

    What counts: a loop closure or a proximity link whose matched node is OLDER than the first
    node this session built (``ref_id`` of the first message: everything below it came off the
    loaded database), or a localisation pose, which RTAB-Map only publishes when it has placed
    itself on the database. A closure INSIDE the new segment is not recognition.

    ONCE TIED, A TIE DOES NOT EXPIRE. It used to, on a clock of "driving seconds" that charged
    whenever RTAB-Map's distance counter grew -- and the counter grows from VO noise on a cart
    that is standing still, so the clock ran while parked. On 2026-09-16 it reached 4279 s and
    withheld 5220 words; on 2026-09-17 it reached 991 s at a bookshelf and withheld 587, and the
    robot could not start a camera-only drive for three hours. Both times it was worked around --
    the database pruned, the flag raised by hand -- and both workarounds died at the next deploy.
    The intent behind the clock was sound and it now lives where it belongs: the word's own
    covariance grows with the DISTANCE driven since the tie (:class:`GraphTrust`), so an old tie
    makes a weak word rather than no word, and the drive gate refuses it on a number. A gate that
    demands recognition before motion also has the causality backwards: motion is what produces
    the views that make the next closure happen.

    A distance counter that goes BACKWARDS is RTAB-Map restarted, and a restart forgets
    everything here: new nodes are placed by odometry again, so the common frame is gone until a
    new tie says otherwise.

    Pure: statistics and ids in, a predicate and a report out.
    """

    def __init__(self) -> None:
        self._first_ref: int | None = None  # the first node id of this start
        self._matches = 0  # acceptances this start
        self._matched_id = 0  # ...and the database node the last one matched
        self._travelled: float | None = None  # the last distance counter read, to spot a restart
        self._starts = 0  # RTAB-Map restarts seen (a distance counter that fell)

    @property
    def matches(self) -> int:
        """How many times this start the graph has been tied to the loaded database."""
        return self._matches

    @property
    def starts(self) -> int:
        """How many times RTAB-Map has restarted under this node (its counter falling)."""
        return self._starts

    @property
    def recognised(self) -> bool:
        """Whether there is a common frame to speak in: this start has been tied to the loaded
        database at least once. How OLD that tie is does not belong here -- it is carried by the
        word's covariance (:class:`GraphTrust`)."""
        return self._matches > 0

    def update(self, stats: Mapping[str, float], ids: InfoIds, now: float) -> bool:
        """One ``/rtabmap/info`` in (``now`` is kept for the node's call signature, and nothing
        here is timed any more): a restart spotted and the ids read. Returns whether THIS message
        tied the present to the loaded database."""
        self._advance(stats, now)
        if ids.ref_id > 0 and self._first_ref is None:
            self._first_ref = ids.ref_id
        matched = self._database_match(stats, ids)
        if matched is None:
            return False
        self._matches += 1
        self._matched_id = matched
        return True

    def report(self, withheld: int = 0) -> RecognitionReport:
        """The predicate and everything a report line says about it
        (:class:`RecognitionReport`), with the words the node has withheld meanwhile."""
        return RecognitionReport(
            recognised=self.recognised,
            matches=self._matches,
            matched_id=self._matched_id,
            withheld=withheld,
        )

    def _advance(self, stats: Mapping[str, float], now: float) -> None:
        """Forget everything when RTAB-Map's distance counter falls: that is a restart, not a
        drive backwards. Nothing else is charged here -- the counter is read only to spot that."""
        travelled = stat(stats, DISTANCE_TRAVELLED_M)
        if travelled is None:
            return
        distance = max(float(travelled), 0.0)
        previous = self._travelled
        if previous is not None and distance < previous:
            self._restart()
        self._travelled = distance

    def _restart(self) -> None:
        """RTAB-Map opened a new session: its nodes are placed by the odometry again and nothing
        it recognised before belongs to them."""
        self._starts += 1
        self._first_ref = None
        self._matches = 0
        self._matched_id = 0

    def _database_match(self, stats: Mapping[str, float], ids: InfoIds) -> int | None:
        """The database node this message tied the present to, or ``None``: a closure or a
        proximity link to a node older than this start's first, or a localisation pose (reported
        as node 0, since it names no node of ours). With no ``ref_id`` on the message at all --
        an RTAB-Map that does not carry the ids -- any closure counts, which is what the
        statistics alone can tell."""
        loop = ids.loop_closure_id or int(stat(stats, LOOP_ID) or 0.0)
        for candidate in (loop, ids.proximity_detection_id):
            if candidate > 0 and (self._first_ref is None or candidate < self._first_ref):
                return candidate
        return 0 if ids.localized else None


class Agreement:
    """How well the graph's last words agree with the tracker's own pose carried forward by
    odometry between them: the rms of the last :data:`AGREEMENT_WORDS` residuals inside
    :data:`AGREEMENT_WINDOW_S`, as a trust of ``exp(-r / scale)``.

    This is the other half of what tape 0333 needed. The decay clock asks how far the cart has
    driven since the graph last recognised something; this asks whether the words the graph is
    saying RIGHT NOW land where the cart actually is. A word riding a broken frame is a metre
    out and reads as a metre of residual from the first word, with no closure to wait for; a word
    riding a good one follows odometry to the centimetre and keeps its 1.0 however many metres ago
    the last tie was.

    EVERY WORD IS TWO RESIDUALS, because a word can be in the right place facing the wrong way.
    ``add`` takes both: the distance in METRES, which is what the word's own covariance floor is
    widened by (a scatter is a spread in the map, and the fusion reads metres), and the 3-DOF
    Mahalanobis distance of the same disagreement under the joint covariance of the word and the
    belief -- the very quantity :func:`pepin.fusion.disagreement` computes and
    :data:`pepin.fusion.GATE` judges -- which is what the TRUST is made of, so a heading error
    costs the word its vote exactly as a position error does. Until 2026-09-18 only the metres
    existed and a word 90 degrees wrong at the right place kept its 1.00.

    An empty window is not a verdict: with no residual to go on the trust is 1.0, and it is the
    recognition predicate, not this, that decides whether a word may ride at all. The node feeds
    a residual only while there is something to measure against -- a belief the tracker has a
    source behind -- because a residual against a pose nobody is confirming (a carried cart, a
    lidar matching nothing) measures the tracker's error and not the graph's.
    """

    def __init__(
        self,
        scale: float = AGREEMENT_SCALE,
        window_s: float = AGREEMENT_WINDOW_S,
        words: int = AGREEMENT_WORDS,
    ) -> None:
        self.scale = scale
        self.window_s = window_s
        self.words = words
        # (moment, metres, sigmas): the same disagreement in the two languages it is read in.
        self._residuals: deque[tuple[float, float, float]] = deque()

    @property
    def count(self) -> int:
        """How many residuals the window holds right now."""
        return len(self._residuals)

    def add(self, at: float, residual_m: float, sigmas: float | None = None) -> None:
        """One word's disagreement with the odometry-propagated belief at the node's clock: how
        far in METRES, and how far in SIGMAS of the two covariances together (the square root of
        :func:`pepin.fusion.disagreement`).

        ``sigmas`` omitted means the caller has no covariance to normalise by, and the metres are
        then read on the old scale (:data:`AGREEMENT_SCALE_M` in units of the same joint sigma
        :data:`AGREEMENT_SCALE` is derived from) so that such a caller keeps the behaviour it had.
        """
        metres = max(float(residual_m), 0.0)
        normalised = (
            metres / AGREEMENT_SCALE_M * AGREEMENT_SCALE
            if sigmas is None
            else max(float(sigmas), 0.0)
        )
        self._residuals.append((at, metres, normalised))
        self._prune(at)

    def rms(self, at: float | None = None) -> float | None:
        """The rms residual over the window, in METRES, pruned to ``at`` first when given;
        ``None`` while the window holds nothing. This is the scatter a word's covariance is
        widened by, so it stays in the units the fusion reads."""
        return self._rms(1, at)

    def sigmas(self, at: float | None = None) -> float | None:
        """The same rms over the window in SIGMAS of the joint covariance -- the 3-DOF quantity
        the trust is made of; ``None`` while the window holds nothing."""
        return self._rms(2, at)

    def trust(self, at: float | None = None) -> float:
        """What the word's agreement is worth, in [0, 1]: ``exp(-r / scale)`` over the normalised
        residual, and 1.0 while there is no residual to judge it on."""
        residual = self.sigmas(at)
        if residual is None:
            return 1.0
        if self.scale <= 0.0:
            return 0.0
        return min(max(math.exp(-residual / self.scale), 0.0), 1.0)

    def _rms(self, column: int, at: float | None) -> float | None:
        """The rms of one of the two columns the window holds, pruned to ``at`` first."""
        if at is not None:
            self._prune(at)
        if not self._residuals:
            return None
        return math.sqrt(sum(r[column] ** 2 for r in self._residuals) / len(self._residuals))

    def _prune(self, at: float) -> None:
        """Drop what is older than the window or past its last :attr:`words` entries."""
        while self._residuals and at - self._residuals[0][0] > self.window_s:
            self._residuals.popleft()
        while len(self._residuals) > self.words:
            self._residuals.popleft()
