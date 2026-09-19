"""What a pose graph's word about the cart is worth: how well its last words agree with the pose
the odometry carries between them.

A word exists only where RTAB-Map has actually RECOGNISED a node of the database it loaded, and it
carries the covariance RTAB-Map measured for that localisation (pepin_bringup.rtabmap_frame). What
no covariance can see is a word riding a frame that is simply wrong — right-looking numbers in
another coordinate system. Tape 0333 of 2026-09-15 is that failure: RTAB-Map had restarted many
times that day, the nodes it was building formed an unlinked segment placed by this session's
odometry alone, and the tracker took 19 words each about a metre and 150 degrees out before the
pose flew.

:class:`Agreement` is what sees it without waiting for the next closure: the rms disagreement
between the last words and the tracker's own pose carried forward by odometry between them, as
``exp(-r / scale)``. A word riding a broken frame is a metre out and reads as a metre of residual
from the first word; a word riding a good one follows odometry to the centimetre and keeps its 1.0.
The residual is 3-DOF — a Mahalanobis distance under the joint covariance of the word and the
belief, the same one :data:`pepin.fusion.GATE` judges — because a word can be in the right place
facing the wrong way, and until 2026-09-18 such a word kept its 1.00.

Nothing here is ROS: a mapping of RTAB-Map's statistics in (``/rtabmap/info``'s ``stats_keys`` and
``stats_values``), a number and a report line out.
"""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping

from pepin.fusion import SIGMA_XY_M
from pepin.measurements import GRAPH_FLOOR_XY_M

__all__ = [
    "AGREEMENT_SCALE",
    "AGREEMENT_SCALE_M",
    "AGREEMENT_WINDOW_S",
    "AGREEMENT_WORDS",
    "HIGHEST_HYPOTHESIS",
    "Agreement",
    "stat",
]

# The residual at which a word is worth 1/e of itself. 10 cm is the scale a graph word has when it
# is working: over tapes 0275/0276 the word sat 0.7-0.8 cm from the lidar truth while driving and
# 0-8 cm at rest, and 2.2-3.2 cm over a printer errand -- so 10 cm is several times the working
# spread and a tenth of the metre-scale error of tape 0333, which lands at exp(-10) and stops the
# word being admitted anywhere.
AGREEMENT_SCALE_M = 0.10
# ...and the same scale in the units the residual is actually judged in: SIGMAS of the joint
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

# How close RTAB-Map came to recognising a place on its last update, by the name it publishes the
# statistic under (rtabmap/Statistics.h). Read for the report line and nothing else: a graph that
# has said nothing for a quarter of an hour is a different thing from one that is nearly there, and
# without this the two look identical from outside. The key carries a trailing unit segment, which
# :func:`stat` tolerates.
HIGHEST_HYPOTHESIS = "Loop/Highest_hypothesis_value"


def stat(stats: Mapping[str, float], name: str) -> float | None:
    """One RTAB-Map statistic by its name, tolerating the unit segment the key carries
    (``Loop/Id`` finds ``Loop/Id/``); ``None`` when this message does not carry it."""
    if name in stats:
        return stats[name]
    for key, value in stats.items():
        if key.rstrip("/") == name or key.startswith(f"{name}/"):
            return value
    return None


class Agreement:
    """How well the graph's last words agree with the tracker's own pose carried forward by
    odometry between them: the rms of the last :data:`AGREEMENT_WORDS` residuals inside
    :data:`AGREEMENT_WINDOW_S`, as a trust of ``exp(-r / scale)``.

    EVERY WORD IS TWO RESIDUALS, because a word can be in the right place facing the wrong way.
    ``add`` takes both: the distance in METRES, which is what the word's own covariance floor is
    widened by (a scatter is a spread in the map, and the fusion reads metres), and the 3-DOF
    Mahalanobis distance of the same disagreement under the joint covariance of the word and the
    belief -- the very quantity :func:`pepin.fusion.disagreement` computes and
    :data:`pepin.fusion.GATE` judges -- which is what the TRUST is made of, so a heading error
    costs the word its vote exactly as a position error does.

    An empty window is not a verdict: with no residual to go on the trust is 1.0, and it is the
    recognition itself — a word exists only where RTAB-Map named a node — that decides whether a
    word may ride at all. The node feeds a residual only while there is something to measure
    against — a belief the tracker has a source behind — because a residual against a pose nobody
    is confirming (a carried cart, a lidar matching nothing) measures the tracker's error and not
    the graph's.
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
