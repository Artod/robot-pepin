"""RTAB-Map's two live switches: when its database may LEARN, and which registration it runs.

Both are decided by the MOMENT and not by a launch argument, both are acted on only once a change
of verdict has HELD for as long as the evidence it rests on takes to refresh, and both are pure
here — a reading in, a verdict out, with no ROS and no clock of their own.

THE MEMORY (:class:`ModeRule`) is below; the registration is :class:`StrategyRule`.

THE REGISTRATION FOLLOWS THE SNAPSHOT. RTAB-Map's registration pipeline is ONE object for the
process and it is built from ``Reg/Strategy``, so the strategy decides which PAIRS can be linked at
all — and under World R a node carries whatever sensor was looking (:mod:`pepin.snapshot`), so no
single strategy serves every node:

* ``Reg/Strategy`` 1 (Icp) reaches the pipeline for EVERY pair, because ``Memory``'s third clause
  admits a pair with a guess when the pipeline does not require an image
  (rtabmap/core/Memory.cpp:2927-2929, with RGBD/LoopClosureIdentityGuess giving the guess) — and
  then ICP needs a scan in BOTH nodes (RegistrationIcp.cpp:459 is the positive guard; a pair with a
  scan missing falls through to "Laser scans empty ?!?" at :973-974). So a CAMERA-ONLY node forms
  no metric link under it. Measured live on 2026-09-18: a minute of camera-only snapshots logged 28
  "Missing visual features or missing raw data to compute them" (Memory.cpp:3217) and 56 "Requested
  laser scan data, but the sensor data doesn't have laser scan".
* ``Reg/Strategy`` 0 (Vis) reaches the pipeline only for a pair whose BOTH nodes carry words
  (the second clause of the same condition), which is exactly the pair a camera-only node makes
  against a database node — and RegistrationVis then wants 3D words on the database side and 2D
  words on ours (Vis/EstimationType 1, PnP), which is what a database built with depth holds. What
  it can never link is a node with no picture, which is what a LIDAR-only snapshot makes.

So the rule is the one sentence the two halves leave: scans in the snapshots -> 1, no scan -> 0.
Nothing else in the parameter table moves with it. The old SLAM_CAMERA_ONLY table
(``git show e1d3b65:ros/pepin_bringup/launch/vslam.launch.py``) differed from the lidar one in six
entries, and five of them — ``subscribe_scan``, ``Grid/Sensor``, ``Grid/3D``, ``Grid/RangeMax``,
``Grid/RayTracing`` — are about the INPUT and the GRID, which World R settled once for every node.
``Reg/Strategy`` is the sixth and the only one about registration.

A LIVE CHANGE IS HONOURED, and that was read rather than hoped for. ``update_parameters`` overwrites
rtabmap's parameter map from the node's ROS parameters and hands the WHOLE map to
``Rtabmap::parseParameters`` (rtabmap_ros/rtabmap_slam/CoreWrapper.cpp:3106-3151), which forwards it
to ``Memory::parseParameters`` (Rtabmap.cpp:719,751); there the strategy of the pipeline in hand is
INFERRED from what it requires (Memory.cpp:700-715) and, when the new value differs, the pipeline
object is deleted and re-created from the accumulated map (Memory.cpp:721-731). The stale path is
the ``else`` at :744-746, which calls ``Registration::parseParameters`` — and that re-reads only
``Reg/RepeatOnce`` and ``Reg/Force3DoF`` (Registration.cpp:79-88), so it could never change a
strategy. TWO CONDITIONS carry the whole thing: the value must be set as a STRING (every rtabmap
parameter is declared as one, CoreWrapper.cpp:364, and read back with ``as_string()`` at :3112), and
the name must be one the LAUNCH already overrode — ``uInsert(parameters_, ...)`` fires only for keys
present in the overrides (CoreWrapper.cpp:362-379), so a parameter never named in the launch table
accepts ``ros2 param set`` and is then never looked at. ``Reg/Strategy`` is in that table
(ros/pepin_bringup/launch/vslam.launch.py), which is what makes this switch possible at all.

THE MEMORY. RTAB-Map has two memories. In MAPPING mode every update may become a node in the
database; in LOCALISATION mode nothing is written and the graph only recognises what it already
holds. Which one it should be in is not a launch decision but a property of the moment: a database
may be taught only from a pose that is sharp AND that does not come out of the database itself.

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
    "REGISTRATION_PARAMETERS",
    "SHARP_SIGMA_DEG",
    "SHARP_SIGMA_M",
    "STRATEGY_ICP",
    "STRATEGY_VIS",
    "ModeRule",
    "ModeVerdict",
    "StrategyRule",
    "StrategyVerdict",
    "describe_sigma",
    "registration_verdict",
    "seating_refusal",
]

# ``Reg/Strategy``'s own values, as Parameters.h:677 names them ("0=Vis, 1=Icp, 2=VisIcp"). 2 is
# not on offer: RegistrationVis with RegistrationIcp as its CHILD, and a child's answer REPLACES
# the parent's (Registration.cpp:207-220), so a pair Vis registers and ICP has no scan for comes
# out null — and the pipeline still requires an image, so a lidar-only node never reaches it.
STRATEGY_VIS, STRATEGY_ICP = "0", "1"
STRATEGY_NAMES = {STRATEGY_VIS: "visual", STRATEGY_ICP: "ICP on the scans"}
# Everything that travels with the strategy, as strings because that is how rtabmap declares every
# one of its parameters (CoreWrapper.cpp:364). ONE entry each: the rest of what the old
# SLAM_CAMERA_ONLY table changed was the input and the grid, which World R settled once for every
# node, and each of these two names is already in the launch table — which is the condition for a
# live set to be seen at all (CoreWrapper.cpp:362-379).
REGISTRATION_PARAMETERS = {
    STRATEGY_VIS: {"Reg/Strategy": STRATEGY_VIS},
    STRATEGY_ICP: {"Reg/Strategy": STRATEGY_ICP},
}

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


@dataclass(frozen=True)
class StrategyVerdict:
    """Which registration RTAB-Map should be running, and the one phrase that says why."""

    strategy: str
    why: str

    @property
    def name(self) -> str:
        """``visual`` or ``ICP on the scans`` — the strategy in words, not as a number."""
        return STRATEGY_NAMES.get(self.strategy, self.strategy)

    @property
    def parameters(self) -> dict[str, str]:
        """Everything that travels with this strategy, as the strings rtabmap wants."""
        return dict(REGISTRATION_PARAMETERS[self.strategy])

    def text(self) -> str:
        """``visual: the snapshots carry no scan`` for a report line."""
        return f"{self.name}: {self.why}"


def registration_verdict(scan: bool, kind: str = "") -> StrategyVerdict:
    """Which registration the snapshots being packed RIGHT NOW need, with no clock and no memory.

    ``scan`` is whether a scan is in them at all (:meth:`pepin.snapshot.SnapshotState.carries`);
    ``kind`` is the last snapshot's own word for the report line. The whole rule: a pair of nodes
    with scans is registered by ICP and a pair without one cannot be, while a pair of nodes with
    pictures is registered visually and a node with no picture cannot be — so the strategy is
    chosen by what the current snapshots carry, and the module docstring holds the file:line.
    """
    said = f" (snapshots {kind})" if kind else ""
    if scan:
        return StrategyVerdict(STRATEGY_ICP, f"the snapshots carry a scan{said}")
    return StrategyVerdict(STRATEGY_VIS, f"the snapshots carry no scan{said}")


class StrategyRule:
    """Which registration RTAB-Map should be running, decided by what the snapshots carry, and
    acted on only once a change has HELD.

    The hold is not a number here: the caller passes the one the EVIDENCE carries
    (:attr:`pepin.snapshot.SnapshotState.refresh_s`, how long the packer itself takes to change its
    own answer about a source), so a sensor that stutters for one snapshot cannot rebuild the
    registration pipeline and a sensor that is really gone is believed as soon as the packer is.

    ``started`` is the strategy the LAUNCH table already set, so the rule asks for nothing until it
    has a reason to: unlike :class:`ModeRule`, whose first verdict is the initial mode.

    Pure: a reading and a clock in; the verdict to act on, or ``None``.
    """

    def __init__(self, started: str = STRATEGY_ICP) -> None:
        self._applied = started
        self._wanted: StrategyVerdict | None = None
        self._since = 0.0
        self._switches = 0

    @property
    def strategy(self) -> str:
        """``Reg/Strategy``'s value as this rule last asked for it."""
        return self._applied

    @property
    def switches(self) -> int:
        """How many times the rule has changed its mind and said so."""
        return self._switches

    @property
    def wanted(self) -> StrategyVerdict | None:
        """The verdict the rule is asking for right now, acted on or not."""
        return self._wanted

    def update(
        self, now: float, hold_s: float, scan: bool | None, kind: str = ""
    ) -> StrategyVerdict | None:
        """One instant in; the verdict to ACT on, or ``None``.

        ``scan`` ``None`` is "the packer has not said" — no snapshot state has arrived, or the one
        that did is older than its own refresh — and then nothing is asked for: an absent report is
        not evidence that the lidar is gone, and rebuilding the pipeline on silence is how a node
        that merely lost its state topic would stop linking scans.
        """
        if scan is None:
            self._wanted = None
            self._since = now
            return None
        verdict = registration_verdict(scan, kind)
        if self._wanted is None or verdict.strategy != self._wanted.strategy:
            self._since = now
        self._wanted = verdict
        if verdict.strategy == self._applied or now - self._since < hold_s:
            return None
        self._applied = verdict.strategy
        self._switches += 1
        return verdict

    def text(self) -> str:
        """The strategy, why it is that, and what is being asked for, for a report line:
        ``visual (the snapshots carry no scan (snapshots camera-only), 1 switch)``."""
        wanted = self._wanted
        name = STRATEGY_NAMES.get(self._applied, self._applied)
        if wanted is None:
            said = "nothing said about the snapshots"
        elif wanted.strategy == self._applied:
            said = wanted.why
        else:
            said = f"asking {wanted.text()}"
        return f"{name} ({said}, {self._switches} switches)"
