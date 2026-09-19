"""When the pose graph's database may LEARN, and when it may only RECOGNISE.

RTAB-Map has two memories. In MAPPING mode every update may become a node in the database; in
LOCALISATION mode nothing is written and the graph only recognises what it already holds. Which
one it should be in is not a launch decision but a property of the moment: a database may be
taught only from a pose that is sharp AND that does not come out of the database itself.

Both alternatives were measured and both are wrong. ALWAYS MAPPING ran until 2026-09-18: the
database grew a session per launch, its sessions ended up 1.6 m and 129 degrees apart, RTAB-Map
then rejected its own correct recognitions on ``RGBD/OptimizeMaxError``, and a parked cart kept a
node a second — 250 junk nodes in one evening. ALWAYS LOCALISING can never learn a new room.

So two conditions, in the order a person would ask them:

* SHARPNESS — the tracker's own published covariance, of this moment, under
  :func:`seating_refusal`. A lidar-held pose passes at 1-2 cm; a mono camera-only pose held by
  graph words sits at a sigma around 20 cm and fails by itself, with nothing here naming it;
* THE PUPIL IS NOT THE TEACHER — whoever HOLDS the pose must not be the graph. The holder is read
  off the board's own per-source report (:func:`pepin.watch.source_words`,
  :meth:`pepin.watch.Preflight.holding`), so no rule here spells "lidar" and a stereo matcher
  good to a few centimetres will teach the database the day it exists.

A fit is not an error bar, which is why the sharpness test reads the covariance and not the score:
at the charger, along a sofa, the lidar's seatings spread up to 55 cm in y at fit 0.67-0.79,
because the scan there is pinned in one axis only.

Nothing here is ROS: a refusal, a holder name and a clock in; a verdict out.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "ALWAYS_LOCALISE",
    "ALWAYS_MAP",
    "BY_TRUST",
    "LOCALISING",
    "MAPPING",
    "SHARP_SIGMA_DEG",
    "SHARP_SIGMA_M",
    "ModeRule",
    "ModeVerdict",
    "describe_sigma",
    "seating_refusal",
]

# What a seating must be worth for the database to be taught from it. The peak's own covariance is
# the error bar (/tracker_pose, covariance=peak, NEES-calibrated), so the test waits for a seating
# the scan pins in BOTH axes. 3 cm because that is where the gate starts to be a gate: over tapes
# 0293-0298 the worse of the two position sigmas has a median of 1.50 cm and a p90 of 3.18 cm, so
# 3 cm refuses the worst 11 % of seatings and 1 cm would refuse 79 % — a database that can never be
# taught is its own failure. One degree of heading is 7 cm at the far wall of this flat, and the
# lidar's heading sigma at a sharp seating is 0.06-1.14 deg (median 0.4) over the same tapes.
SHARP_SIGMA_M = 0.03
SHARP_SIGMA_DEG = 1.0

MAPPING, LOCALISING = "mapping", "localising"
# The three settings of the override: let the rule decide, or pin one mode.
BY_TRUST, ALWAYS_MAP, ALWAYS_LOCALISE = "trust", "map", "localise"


def describe_sigma(sigma: tuple[float, float, float] | None) -> str:
    """One seating's uncertainty for a report line: ``1.0/1.3 cm, 0.30 deg`` (x, y, heading),
    or ``unknown`` when the belief carried no covariance."""
    if sigma is None:
        return "unknown"
    return f"{sigma[0] * 100.0:.1f}/{sigma[1] * 100.0:.1f} cm, {math.degrees(sigma[2]):.2f} deg"


def seating_refusal(
    sigma: tuple[float, float, float] | None,
    max_sigma_m: float = SHARP_SIGMA_M,
    max_sigma_deg: float = SHARP_SIGMA_DEG,
) -> str | None:
    """Why this seating is not sharp enough to teach from, in one phrase for a log, or ``None``
    when it is.

    ``sigma`` is the tracker's own error bar at the moment — the roots of its covariance diagonal
    (x, y in metres, heading in radians). The seating must be sharp in BOTH position axes, not
    merely well-matched: a scan sliding along a corridor reports an honest fit and a metre of
    freedom in the other axis.
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


@dataclass(frozen=True)
class ModeVerdict:
    """Which mode RTAB-Map should be in, and the one phrase that says who decided and why."""

    mapping: bool
    why: str

    @property
    def mode(self) -> str:
        """``mapping`` or ``localising``."""
        return MAPPING if self.mapping else LOCALISING

    def text(self) -> str:
        """``mapping: lidar-held seating 1.2/0.4 cm`` for a report line."""
        return f"{self.mode}: {self.why}"


class ModeRule:
    """When the database may LEARN, and when it may only RECOGNISE — decided by trust in the pose
    and never by a sensor's name (see the module docstring for the two conditions).

    A verdict must HOLD before it is acted on, and for a length that is derived and not chosen: as
    long as the evidence it rests on takes to refresh. The seating's own freshness window is that
    length — one missed ``/tracker_pose`` then cannot flap the mode, and a real change is acted on
    as soon as it is a change and not a gap.

    Pure: seating, holder and a clock in; a verdict out, and only when it is worth a service call.
    """

    def __init__(self, hold_s: float, override: str = BY_TRUST, graph: str = "graph") -> None:
        self.hold_s = hold_s
        self.override = override
        self.graph = graph
        self._applied: bool | None = None  # the mode the service has been told, once it has been
        self._wanted: ModeVerdict | None = None  # ...and the one the rule has been asking for
        self._since = 0.0  # since when, on the caller's clock
        self._switches = 0

    @property
    def mode(self) -> str:
        """The mode this rule has asked for, or ``unknown`` before it has asked for anything."""
        return "unknown" if self._applied is None else (MAPPING if self._applied else LOCALISING)

    @property
    def switches(self) -> int:
        """How many times the rule has changed its mind and said so."""
        return self._switches

    @property
    def wanted(self) -> ModeVerdict | None:
        """The verdict the rule is asking for right now, whether or not it has been acted on."""
        return self._wanted

    def verdict(self, refusal: str | None, holder: str | None, seating: str = "") -> ModeVerdict:
        """What the rule says about this instant, with no clock and no memory: the override when
        there is one, then the two conditions in the order a person would ask them."""
        if self.override == ALWAYS_MAP:
            return ModeVerdict(True, "told to map whatever the pose is worth")
        if self.override == ALWAYS_LOCALISE:
            return ModeVerdict(False, "told to localise whatever the pose is worth")
        if refusal is not None:
            return ModeVerdict(False, f"the pose is not worth learning from: {refusal}")
        if holder is None:
            return ModeVerdict(False, "nobody is holding the pose: no source has spoken")
        if holder == self.graph:
            return ModeVerdict(False, "the pose is held by the graph: the pupil is not the teacher")
        return ModeVerdict(
            True, f"the pose is held by {holder}" + (f", seated to {seating}" if seating else "")
        )

    def update(
        self, now: float, refusal: str | None, holder: str | None, seating: str = ""
    ) -> ModeVerdict | None:
        """One instant in; the verdict to ACT on, or ``None``.

        A verdict is returned only when it differs from the mode already asked for AND has been the
        answer for :attr:`hold_s` without a break — except the very first one, which is the initial
        mode and is asked for at once. Returning it counts a switch and records it as applied, so a
        caller whose service call fails must ask again by feeding the rule the next instant.
        """
        verdict = self.verdict(refusal, holder, seating)
        if self._wanted is None or verdict.mapping != self._wanted.mapping:
            self._since = now
        self._wanted = verdict
        if self._applied is None:
            self._applied = verdict.mapping
            self._switches += 1
            return verdict
        if verdict.mapping == self._applied or now - self._since < self.hold_s:
            return None
        self._applied = verdict.mapping
        self._switches += 1
        return verdict

    def text(self) -> str:
        """The mode, who decided it and why, for a report line: ``localising (the pose is held by
        the graph: the pupil is not the teacher, 2 switches, by trust)`` — and, while a verdict is
        waiting out its hold, ``mapping (asking localising: …)``."""
        wanted = self._wanted
        if wanted is None:
            said = "nothing decided yet"
        elif wanted.mapping == self._applied:
            said = wanted.why
        else:
            said = f"asking {wanted.text()}"
        return f"{self.mode} ({said}, {self._switches} switches, by {self.override})"
