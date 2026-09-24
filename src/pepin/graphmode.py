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
the ``else`` at :744-746, which calls the pipeline's own ``parseParameters`` — whose base part
re-reads only ``Reg/RepeatOnce`` and ``Reg/Force3DoF`` (Registration.cpp:79-88), so it could never
change a strategy; a RegistrationVis re-reads its ``Vis/*`` there too and rebuilds its detectors
(the method is virtual, RegistrationVis.cpp:125-293), which is the path the visual feature set of
:func:`visual_parameters` rides while the strategy stays. TWO CONDITIONS carry the whole thing: the
value must be set as a STRING (every rtabmap parameter is declared as one, CoreWrapper.cpp:364,
and read back with ``as_string()`` at :3112), and the name must be one the LAUNCH already overrode
— ``uInsert(parameters_, ...)`` fires only for keys present in the overrides
(CoreWrapper.cpp:362-379), so a parameter never named in the launch table accepts ``ros2 param
set`` and is then never looked at. ``Reg/Strategy`` is in that table
(ros/pepin_bringup/launch/vslam.launch.py), which is what makes this switch possible at all.
rtabmap_slam ALSO applies every change it hears on its own ``/parameter_events`` as it arrives
(CoreWrapper.cpp:907-970, the same launch-table filter): a ``set_parameters`` request of five names
lands as five notifications and five ``parseParameters``, a ``set_parameters_atomically`` request
as one (measured 2026-09-24, scratch/xfeat_critic/atomic_set.sh) — so a set that must land whole
goes atomically (``atomic_parameter_sets`` in pepin_bringup.rtabmap_frame).

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
    "CONFIRM_AGGRESSIVE",
    "CONFIRM_PARAMETERS",
    "CONFIRM_SINGLE",
    "CONFIRM_STOCK",
    "FEATURES_ORB",
    "FEATURES_XFEAT",
    "FEATURE_PARAMETERS",
    "LOCALISING",
    "MAPPING",
    "PNP_REPROJ_PX",
    "PNP_REPROJ_RANGE_PX",
    "PROXIMITY_STOCK",
    "REGISTRATION_PARAMETERS",
    "SHARP_SIGMA_DEG",
    "SHARP_SIGMA_M",
    "STRATEGY_ICP",
    "STRATEGY_VIS",
    "XFEAT_DETECTOR_PATH",
    "XFEAT_MATCHER_PATH",
    "ModeRule",
    "ModeVerdict",
    "StrategyRule",
    "StrategyVerdict",
    "describe_sigma",
    "registration_verdict",
    "seating_refusal",
    "visual_parameters",
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
#
# THE GRID'S SENSOR TRAVELS WITH IT (2026-09-19, measured on the parked cart, fresh database both
# times, scratch/map_vs_pose_walk.py). With the grid built from BOTH sensors (Grid/Sensor 2) the
# newborn map held 1958-2207 occupied cells on 5 x 4 m — the mono network's smeared depth laid
# beside the lidar's walls — and the lidar tracker's match on it was flat: fit 0.97, published
# sigma 0.47 m / 34 deg, the pose wandering 9.8 cm and 3.9 deg in 90 s over a map that had turned
# 0.23 deg and an odometry that had turned 0.5. Built from the SCAN alone (0) the same room is
# 480-538 cells, sigma 0.00-0.01 m / 0.1 deg, 0.8 cm in 90 s. RTAB-Map's grid has no per-sensor
# weight, so a sensor that cannot vouch at the lidar's grade must not write while the lidar does:
# a node that carries a scan gives the grid its scan (0); a node with no scan gives what it has,
# the depth (1) — which only happens while MAPPING without a lidar, i.e. a wake-up on the camera
# alone. Local grids are made per node, at the node's creation, with the value set then; and in
# localisation mode nothing is written, so the value does not matter there (CoreWrapper.cpp:3153).
#
# AND SO DO THE NEIGHBOUR LINKS (same night, same cart, fresh database, parked, nothing touched).
# Unrefined — the link between two nodes is the odometry's word — the EKF's yaw creeps at rest with
# the gyro's bias while the wheels read exactly zero, RTAB-Map adds a node every few minutes at the
# crept heading, re-renders its grid, and the tracker follows it: +1 -> +28 deg in 40 min, steady
# +0.67 deg/min, at fit 0.95-0.98 and sigma 0.00 m the whole time (position held to 5 cm).
# Refined by ICP — the scan says "no motion" — -1, -3, -2, -3, -3 deg over 20 min, no trend, and
# the graph stays at two nodes. With no scan in the node there is nothing to refine with.
#
# These three are not three patches: together with Reg/Strategy they are EXACTLY the keys on which
# the two measured tables of before World R differed (SLAM_LIDAR and SLAM_CAMERA_ONLY in
# `git show e1d3b65:ros/pepin_bringup/launch/vslam.launch.py`; the fifth, Grid/RangeMax 8 / 3, now
# travels as NaN in the depth itself). One table for every situation was the over-simplification;
# what is one is the RULE — the parameters follow what the snapshots carry — and the two sets it
# chooses between are the old tables, chosen by the data instead of by the operator.
# KNOWN COST of refined links, measured 2026-09-14 beside a known map: a refined link carries
# ICP's own tiny covariance (median 0.75 cm / 0.135 deg), and RGBD/OptimizeMaxError then rejects a
# closure that asks more than ~2 cm of any one of them — every closure ACROSS two sessions did.
# Watch the log for "Rejecting all added loop closures"; the launch's neighbor_refining argument
# and graph_memory are the ways out.
#
# Grid/Sensor DOES NOT TRAVEL WITH THEM — it did for half a day and that was wrong (2026-09-19,
# first camera-only drive under World R). A change of Grid/Sensor makes RTAB-Map re-render its
# WHOLE grid from the other sensor, not just the next node: the moment the snapshots went
# camera-only the lidar-built map of 215x262 cells became a depth-built 81x52, the costmaps' static
# layer and the lidar's fit went with it, and I cancelled a drive that Artem saw arrive at the
# bookshelf. The grid's sensor is a property of the MAP — what it was built from — and stays at the
# launch's value (the scan); a node with no scan simply adds nothing to the grid.
REGISTRATION_PARAMETERS = {
    STRATEGY_VIS: {
        "Reg/Strategy": STRATEGY_VIS,
        "RGBD/NeighborLinkRefining": "false",
    },
    STRATEGY_ICP: {
        "Reg/Strategy": STRATEGY_ICP,
        # FALSE AGAIN, 2026-09-19 afternoon, on the first real drive. Refining was switched on the
        # night before to stop a parked cart's map turning with the odometry — but the root of that
        # was the gyro's bias frozen at boot, cured the same morning in the base bridge
        # (/odometry/filtered -0.01 deg/min at rest). With the root gone only the known cost was
        # left, and it came at once: over a 3 m teleop drive RTAB-Map logged "Rejecting all added
        # loop closures" 89 times, 87 of them on a NEIGHBOUR edge (type=0) whose refined covariance
        # made a 2.6 deg or 11.6 cm residual read as a ratio of 3.7-4.6 against
        # RGBD/OptimizeMaxError 3.0 — the disease of 2026-09-14, exactly. Unrefined, a neighbour
        # link carries the odometry's own honest uncertainty, which is what that check compares to.
        "RGBD/NeighborLinkRefining": "false",
    },
}

# WHICH FEATURES THE VISUAL REGISTRATION MATCHES — the third thing that travels with the strategy,
# chosen by an operator's flag and not by the snapshots. The database's words are GFTT/ORB
# (Kp/DetectorStrategy 8), and against them an evening picture registers with 0-11 PnP inliers
# where RTAB-Map asks for 20: the day map against the lamps, measured 2026-09-23
# (scratch/link_autopsy/feature_ab.py, 630 camera-only updates, 0 recognitions). XFeat keypoints
# matched by LighterGlue register the same day/evening pairs (scratch/xfeat/xfeat_bench.py).
#
# THE DATABASE IS NOT RE-PROCESSED. RGBD/LoopClosureReextractFeatures makes
# Memory::computeTransform load both nodes' stored picture and depth (getNodeData, a read,
# rtabmap/core/Memory.cpp:2902) and drop their stored words (:2950), so RegistrationVis detects
# afresh with its OWN detector — Vis/FeatureType 15, the Python detector
# (ros/xfeat/rtabmap_xfeat.py) — and matches with its own matcher — Vis/CorNNType 6, the Python
# matcher (ros/xfeat/rtabmap_lighterglue.py), which is the path of a loop closure's identity guess
# (RegistrationVis.cpp:1381-1411; Rtabmap.cpp:3057 passes the identity, which RegistrationVis does
# not count as a guess at :1012). The ORB words stay the vocabulary that FINDS the node (Kp/*,
# untouched); XFeat only decides whether it is really there, and where.
#
# ONLY WITH THE VISUAL STRATEGY AND ONLY WHILE LOCALISING, because the same flag changes what a
# NEW node stores: with it on, createSignature keeps no word descriptors and no 3D
# (Memory.cpp:6126), and a node mapped that way could later be registered by re-extraction only.
# The strategy that maps is ICP (the lidar teaches the database), and under it the set is always
# ORB's — the launch table's own values; a visual strategy that is also MAPPING (graph_memory map,
# or one day a holder that is not the graph) gets ORB's too, so the database is never written
# differently from how it was built.
FEATURES_ORB, FEATURES_XFEAT = "orb", "xfeat"
# Where the image built by ros/Dockerfile.xfeat puts the two adapters RTAB-Map loads by path. An
# image without them (the apt build, no Python in RTAB-Map) cannot run the xfeat set at all.
XFEAT_DETECTOR_PATH = "/opt/xfeat/rtabmap_xfeat.py"
XFEAT_MATCHER_PATH = "/opt/xfeat/rtabmap_lighterglue.py"
FEATURE_PARAMETERS = {
    FEATURES_ORB: {
        "Vis/FeatureType": "8",  # GFTT/ORB, RTAB-Map's default and the database's own words
        "Vis/CorNNType": "1",  # FLANN kd-tree with NNDR, RTAB-Map's default
        "RGBD/LoopClosureReextractFeatures": "false",
    },
    FEATURES_XFEAT: {
        "Vis/FeatureType": "15",  # PyDetector
        "Vis/CorNNType": "6",  # PyMatcher
        "RGBD/LoopClosureReextractFeatures": "true",
    },
}
# Vis/PnPReprojError, RTAB-Map's default (Parameters.h:684), and the widest the flag allows: a
# day-built depth seen from an evening frame lands its 3D points a pixel or two off, and 4 px
# roughly doubles the inliers (scratch/xfeat/xfeat_bench.py has both gates side by side).
PNP_REPROJ_PX = 2.0
PNP_REPROJ_RANGE_PX = (1.0, 4.0)

# HOW A LOCALISATION IS CONFIRMED — the fourth thing that travels with the visual strategy while
# localising. RTAB-Map 0.22.1 does not accept a first good localisation: it DELAYS it into the
# odometry cache (RGBD/MaxOdomCacheSize updates, Rtabmap.cpp:3650/3772) and accepts both once a
# second one lands inside that window. The first try only has to reach RGBD/AggressiveLoopThr
# (0.05) while the cache holds no localisation; the second must reach Rtabmap/LoopThr (0.11)
# (:2141-2162). Against a daylight database under the evening lamps the ORB words' hypotheses read
# 0.05-0.07, so the second try never comes: parked at the base, XFeat registered with 83-118
# inliers once every 11 updates — the cache's 10 plus the retry — and 0 of 244 updates were
# accepted (2026-09-24, scratch/link_autopsy/localisation_cadence.py; the same on the replay of
# run 0466: 21 of 22 registered, 0 accepted). Two ways out, both RTAB-Map's own parameters:
# ``aggressive`` keeps the second try at the aggressive threshold (the confirmation stays), and
# ``single`` sets the cache to 0 (the first good localisation is accepted; RGBD/OptimizeMaxError
# is already 0 while localising, so the cache's deformation check was not running anyway).
# Measured live at the base (scratch/link_autopsy/confirm_ab.sh, 240 s each): aggressive 41 of 42
# updates accepted, single 44 of 44, all within 7 cm of the seed and within 1.2 deg of the yaw at
# which the lidar's scan fits the map (scratch/link_autopsy/lidar_yaw_truth.py). Under ICP, and
# while the database maps, the set is always the stock one: the lidar's hypotheses reach 0.11 and
# its ICP from the identity guess is not a registration to try on every weak hypothesis.
CONFIRM_STOCK, CONFIRM_AGGRESSIVE, CONFIRM_SINGLE = "rtabmap", "aggressive", "single"
CONFIRM_PARAMETERS = {
    CONFIRM_STOCK: {"Rtabmap/LoopThr": "0.11", "RGBD/MaxOdomCacheSize": "10"},
    CONFIRM_AGGRESSIVE: {"Rtabmap/LoopThr": "0.05", "RGBD/MaxOdomCacheSize": "10"},
    CONFIRM_SINGLE: {"Rtabmap/LoopThr": "0.11", "RGBD/MaxOdomCacheSize": "0"},
}


# WHICH NODES A LOCALISED CAMERA REGISTERS AGAINST — the fifth thing that travels with the visual
# strategy while localising. With RGBD/ProximityBySpace true (the launch table's, and the lidar's
# path to most of this graph's links) a localised cart ALSO registers every update against the
# nodes near its pose; under ICP that is a few milliseconds of scan matching, under the visual
# strategy it is an XFeat re-extraction and a LighterGlue match per candidate. Measured
# 2026-09-24: parked at the base, a localised update cost a median 5.0-5.2 s on the reference
# BLAS and 1.77 s after RTLD_DEEPBIND, almost all of it Timing/Proximity_by_space_visual; on the
# replay of camera-only run 0466 (LoopThr 0.05) turning it off took an update from 5.3 s to
# 0.55 s and still accepted 50 of 55, map -> odom within 0.6 cm / 0.12 deg of its running median.
# Off, only the words' own hypothesis is registered (one per update at most).
PROXIMITY_STOCK = True


def visual_parameters(
    strategy: str,
    features: str,
    pnp_reproj_px: float,
    mapping: bool = False,
    confirm: str = CONFIRM_STOCK,
    proximity: bool = PROXIMITY_STOCK,
) -> dict[str, str]:
    """The feature set, PnP gate, localisation confirmation and proximity search RTAB-Map's
    registration should run under ``strategy``, as the strings rtabmap wants: the flags' values
    under the visual strategy while the database only localises, ORB's set and the stock
    confirmation and proximity under ICP or while mapping whatever the flags say (the module
    comments above say why)."""
    visual = strategy == STRATEGY_VIS and not mapping
    chosen = features if visual else FEATURES_ORB
    if chosen not in FEATURE_PARAMETERS:
        raise ValueError(f"unknown feature set {chosen!r}; sets: {sorted(FEATURE_PARAMETERS)}")
    if confirm not in CONFIRM_PARAMETERS:
        raise ValueError(f"unknown confirmation {confirm!r}; sets: {sorted(CONFIRM_PARAMETERS)}")
    return {
        **FEATURE_PARAMETERS[chosen],
        "Vis/PnPReprojError": f"{pnp_reproj_px:g}",
        **CONFIRM_PARAMETERS[confirm if visual else CONFIRM_STOCK],
        "RGBD/ProximityBySpace": "true" if (proximity if visual else PROXIMITY_STOCK) else "false",
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
        # What _applied was before the verdict update() last returned, for withdraw(); a list so
        # "nothing to hand back" (empty) differs from "the mode before was unknown" ([None]).
        self._before: list[bool | None] = []

    @property
    def mode(self) -> str:
        """The mode this rule has asked for and the caller did not hand back (:meth:`withdraw`) —
        the one RTAB-Map was told — or ``unknown`` before anything went out."""
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
        mode and is asked for at once. Returning it counts a switch and records it as applied; a
        caller whose call did not go out hands it back with :meth:`withdraw`, and the next instant
        with the same evidence returns it again.
        """
        verdict = self.verdict(refusal, holder, seating)
        if self._wanted is None or verdict.mapping != self._wanted.mapping:
            self._since = now
        self._wanted = verdict
        self._before = []
        if self._applied is not None and (
            verdict.mapping == self._applied or now - self._since < self.hold_s
        ):
            return None
        self._before = [self._applied]
        self._applied = verdict.mapping
        self._switches += 1
        return verdict

    def withdraw(self) -> None:
        """Hand back the verdict :meth:`update` has just returned, because the switch did not go
        out (the service was busy or not up): the rule's mode is again the one RTAB-Map still has,
        the switch is not counted, and the hold already served is not served twice — the next
        :meth:`update` with the same evidence returns the verdict at once."""
        if not self._before:
            raise RuntimeError("withdraw() without a verdict just returned by update()")
        self._applied = self._before.pop()
        self._switches -= 1

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
        self._before: list[str] = []  # the strategy before the verdict just returned (withdraw)

    @property
    def strategy(self) -> str:
        """``Reg/Strategy``'s value as this rule last asked for it and the caller did not hand back
        (:meth:`withdraw`): the strategy RTAB-Map was sent."""
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
        that merely lost its state topic would stop linking scans. A verdict returned is recorded
        as applied; a caller whose set did not go out hands it back with :meth:`withdraw`.
        """
        self._before = []
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
        self._before = [self._applied]
        self._applied = verdict.strategy
        self._switches += 1
        return verdict

    def withdraw(self) -> None:
        """Hand back the verdict :meth:`update` has just returned, because its parameter set did
        not go out: :attr:`strategy` is again the one RTAB-Map still runs, the switch is not
        counted, and the next :meth:`update` with the same evidence asks for it again at once."""
        if not self._before:
            raise RuntimeError("withdraw() without a verdict just returned by update()")
        self._applied = self._before.pop()
        self._switches -= 1

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
