"""RTAB-Map's side of the one map: its grid relayed as ``/map``, its memory mode, its registration,
and the word that says whether this start of it is placed.

ONE FRAME. RTAB-Map's optimised map frame IS ``map`` and RTAB-Map publishes ``map -> odom`` itself
(one localiser, 2026-09-22): this node ties nothing and broadcasts no transform. Its grid
(:data:`GRID_TOPIC`) passes through here onto :data:`MAP_TOPIC`, which both costmaps' static
layers read. The board tracker this node used to feed with a measurement per recognised update,
and the message-path owner of the frame (``slam``), are on the tag alt/tracker-2026-09-22.

THE MEMORY MODE is pinned to localising (:class:`pepin.graphmode.ModeRule` with
``ALWAYS_LOCALISE``; vslam.launch.py's rtabmap_memory says why — a restart in mapping mode opens a
session per start). This node sends RTAB-Map's own set_mode service once and carries the
parameters that mode needs with it (:data:`MODE_PARAMETERS`).

THE REGISTRATION, on the same discipline and for the same reason: it is a property of the moment and
not of the launch. RTAB-Map's registration pipeline is ONE object for the process, and which PAIRS
it can link depends on what the nodes carry — ICP links a pair of scans and nothing without one,
visual registration links a pair of pictures and nothing without one — while under World R a node
carries whatever sensor was looking. So the strategy follows the snapshots: sensor_pack says what
its snapshots carry on :data:`SNAPSHOT_STATE_TOPIC` (latched), and
:class:`pepin.graphmode.StrategyRule` turns that into ``Reg/Strategy``, acted on when a change has
held for the hold the STATE ITSELF carries — the packer's own liveness window, so this node invents
no number. It travels by the same path as the memory mode's parameters (:meth:`_set_parameters`),
which is the only one rtabmap honours: a string set on the node, then its own ``update_parameters``,
after which Memory re-creates the pipeline (the file:line is in :mod:`pepin.graphmode`). Without
this a camera-only cart cannot localise at all — measured on 2026-09-18, a minute of camera-only
snapshots under ICP formed not one metric link.

THE VISUAL REGISTRATION'S FEATURES travel with the strategy on the same path, chosen by two
operator flags rather than by the snapshots: ``visual_features`` (``xfeat`` — XFeat keypoints
matched by LighterGlue, re-extracted from both nodes' stored pictures at loop-closure time — or
``orb``, the database's own words) and ``pnp_reproj_px`` (``Vis/PnPReprojError``). Under ICP the set
is always ORB's, and so it is unless RTAB-Map is LOCALISING for certain (told so, the call answered,
no switch to mapping waiting), because the same re-extraction switch changes what a NEW node stores
(:func:`pepin.graphmode.visual_parameters` has the file:line); a switch to mapping goes out only
once ORB's set is in force and re-read. ``xfeat`` needs the image that
carries the Python adapters (ros/Dockerfile.xfeat); in any other image this node sends ORB's set and
the report line says why.

"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParametersAtomically
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rtabmap_msgs.msg import Info
from std_msgs.msg import String
from std_srvs.srv import Empty

from pepin.flags import Flag, FlagSet, load_knobs, with_knobs
from pepin.global_descriptor import (
    CENSUS_ENV,
    DESCRIPTOR_PARAMETERS,
    PLACE_DESCRIPTOR,
    PLACE_TOPIC,
    PLACE_WORDS,
    TFIDF,
    Census,
    SnapshotPlace,
    recognition_parameters,
    rtabmap_keeps_descriptors,
)
from pepin.graphmode import (
    ALWAYS_LOCALISE,
    CONFIRM_AGGRESSIVE,
    CONFIRM_PARAMETERS,
    CONFIRM_STOCK,
    FEATURES_ORB,
    FEATURES_XFEAT,
    LOCALISING,
    PNP_REPROJ_PX,
    PROXIMITY_STOCK,
    STRATEGY_ICP,
    XFEAT_DETECTOR_PATH,
    XFEAT_MATCHER_PATH,
    ModeRule,
    ModeVerdict,
    StrategyRule,
    visual_parameters,
)
from pepin.graphtrust import HIGHEST_HYPOTHESIS, stat
from pepin.live_settings import (
    BACKENDS,
    REGISTRATION_FILE,
    LiveFile,
    RegistrationSettings,
    StatusBoard,
    registration_file,
)
from pepin.snapshot import SnapshotState
from pepin.sources import CAMERA, GRAPH, LIDAR
from pepin.watch import PLACEMENT_TOPIC, Placement
from pepin_bringup.msgs import (
    stamp_seconds,
)
from pepin_bringup.node_kit import Switches, spin_main

RATE_HZ = 10.0
# What the graph has RECOGNISED on each update: the node a loop closure or a proximity link
# matched, which is what places this start of RTAB-Map (start_needs_placement) — and the
# statistics beside it, read for the report line alone (how close the last hypothesis came).
INFO_TOPIC = "/rtabmap/info"
# Where an operator tells RTAB-Map where the cart stands (ros/tools/goto_ros.py seed publishes
# it; RTAB-Map takes it in localisation mode). Heard here too: a seed since this start is one
# of the two things that PLACE it (start_needs_placement, pepin.watch.Placement).
RTABMAP_INITIAL_POSE = "/rtabmap/initialpose"
# Where RTAB-Map says the cart is IN THE DATABASE IT LOADED: counted for the report line.
LOCALIZATION_TOPIC = "/rtabmap/localization_pose"
# RTAB-Map's grid as RTAB-Map publishes it (its "map" output, remapped in vslam.launch.py), and
# the ONE map topic this node relays it onto.
GRID_TOPIC = "/rtabmap/grid"
MAP_TOPIC = "/map"
# RTAB-Map numbers nodes from 1 and a loaded database continues its own numbering, so a start
# whose first node is 1 loaded nothing: its grid is its own and there is no older map to tie to.
FIRST_ID_OF_AN_EMPTY_DATABASE = 1
# What pepin_bringup.sensor_pack's snapshots CARRY, latched: the evidence the registration strategy
# follows (pepin.snapshot.SnapshotState). The same literal is sensor_pack's own STATE_TOPIC — one
# name written on both sides of the contract, so a test can pin it. It does not cross the bridge:
# both ends of it are on this laptop.
SNAPSHOT_STATE_TOPIC = "/sensor_pack/state"
# ...and the two services rtabmap_ros offers for the switch, verified live on 2026-09-18 to take
# effect without a restart.
RTABMAP_NODE = "/rtabmap/rtabmap"
MAPPING_SERVICE = "/rtabmap/rtabmap/set_mode_mapping"
LOCALISATION_SERVICE = "/rtabmap/rtabmap/set_mode_localization"
# The parameters each mode needs, which the mode services do NOT touch. RGBD/LinearUpdate and
# RGBD/AngularUpdate 0 is what makes a PARKED cart localise (measured live: with the defaults
# Memory/Small_movement read 1 on every update at rest and not one update named a node), and it is
# exactly what must NOT hold while mapping — a node a second at a standstill put 250 junk nodes in
# the database in one evening. So they travel with the switch, as strings, through RTAB-Map's own
# parameter path. 0.05 m / 0.05 rad in mapping mode is a node every 5 cm or 3 degrees: the distance
# the graph's own neighbour links are measured over (median 0.75 cm, 0.135 deg per link,
# scratch/graph_link_sigmas.py) and small enough that a room is covered without a node per second.
MODE_PARAMETERS = {
    True: {"RGBD/LinearUpdate": "0.05", "RGBD/AngularUpdate": "0.05"},
    False: {"RGBD/LinearUpdate": "0", "RGBD/AngularUpdate": "0"},
}
# How long a memory-mode verdict must hold before it is acted on (pepin.graphmode.ModeRule).
MODE_HOLD_S = 2.0
# How old the adapters' counters (pepin.live_settings' status file) may be for the report line to
# print them as current: they write once a minute while RTAB-Map registers.
REGISTRATION_STATUS_FRESH_S = 180.0

FLAGS = FlagSet(
    Flag(
        "visual_features",
        FEATURES_XFEAT,
        choices=(FEATURES_ORB, FEATURES_XFEAT),
        description="which features RTAB-Map's VISUAL registration (Reg/Strategy 0, the camera-only"
        " strategy, and the visual half of 2) matches when it checks a node the words recognised."
        " xfeat: XFeat keypoints matched by LighterGlue, re-extracted from both nodes' stored"
        " pictures at loop-closure time (Vis/FeatureType 15, Vis/CorNNType 6,"
        " RGBD/LoopClosureReextractFeatures true); the database is only read. orb: the database's"
        " own GFTT/ORB words, the launch table's values. Sent with the strategy and changed live;"
        " under ICP alone, and unless RTAB-Map is certainly localising (told so and answered, no"
        " switch to mapping waiting), the set is always orb, and a switch to mapping waits until"
        " orb's set is in force. xfeat"
        f" needs the pepin-laptop:xfeat image ({XFEAT_DETECTOR_PATH}); in another image this"
        " node sends orb and the report line says why",
        why="xfeat, measured 2026-09-24 in RTAB-Map's OWN registration: RegistrationVis of"
        " pepin-laptop:xfeat on evening frames of runs 0455-0465, each against the daylight"
        " database's node nearest the lidar-held truth (scratch/xfeat/probe/probe_report.py;"
        " Vis/Iterations 300, 2 px, 20 inliers; judged while turning slower than 0.35 rad/s, right"
        " within 0.30 m and 10 deg). Of 219 frames ORB recognised 7 (3 %), xfeat 197 (90 %): 139"
        " right and 18 wrong — 11.5 % of the 157 judged — median error 0.12 m / 3.1 deg, p90 0.30"
        " m / 9.2 deg (scratch/xfeat_critic/rtabmap_own_figures.py). On the camera-only run 0466,"
        " against the nodes RTAB-Map itself proposed, ORB recognised 0 of 136 frames and xfeat"
        " 133. The whole node replaying 0455-0465 with the stock table"
        " (scratch/xfeat/rtabmap_replay.py, replay_summary.py) accepted 27 of 425 updates — 8"
        " loop closures, 19 proximity links — and 19 of the 21 judged were right; the 2 wrong are"
        " loop closures on node 641 (the bookshelf), all five of whose accepts sit at 0.24-0.31 m"
        " / 7-11 deg. In the rtabmap role an accepted wrong registration moves map -> odom"
        " (RGBD/OptimizeMaxError 0 there: the graph's error rejects nothing), so the 11.5 % is"
        " the number to watch on a drive. Secondary, the Python emulation over the 3 nearest"
        " nodes (scratch/xfeat/xfeat_bench.py): ORB 10 (5 %), xfeat 167 (76 %), 6 of 133 judged"
        " wrong, four of them node 492/493 where the daylight picture's rocking chair had moved"
        " by the evening (scratch/xfeat/node_consistency.py); on 0466 its 93 recognitions with"
        " an EKF sample gave a map -> odom within 4.0 cm / 0.45 deg of a running median"
        " (scratch/xfeat/bench_analyse.py)",
        on_when="xfeat whenever the camera has to localise alone on a map built in other light —"
        " the evening, lamps on, a daylight database",
        off_when="orb to reproduce RTAB-Map's stock registration (0 of 630 camera-only updates"
        " accepted on 2026-09-23 evening), in an image without the adapters, when a drive shows"
        " the wrong recognitions (map -> odom jumping by 0.3 m or 10 deg on one word), or if the"
        " laptop's CPU cannot spare a registration: 0.67-0.70 s median inside RTAB-Map on the"
        " Docker VM once torch is imported with RTLD_DEEPBIND, 1.70 s before"
        " (scratch/xfeat/data/probe/out/probe_xfeat_2px_300_6_*.csv), per loop-closure or"
        " proximity candidate; in the stock-table replay, run before that fix, an update that"
        " registered took 2.5 s median, p90 5.1 s. The benchmark's 75 + 75 + 167 ms is the"
        " Python adapters alone, not what RTAB-Map pays",
    ),
    Flag(
        "visual_confirm",
        CONFIRM_AGGRESSIVE,
        choices=tuple(CONFIRM_PARAMETERS),
        description="how RTAB-Map CONFIRMS a localisation while the camera registers alone:"
        " RTAB-Map 0.22 delays a first good localisation into its odometry cache and accepts it"
        " only with a second one inside RGBD/MaxOdomCacheSize updates, and that second try has to"
        " reach Rtabmap/LoopThr. rtabmap: its stock 0.11 and 10. aggressive: Rtabmap/LoopThr 0.05,"
        " the threshold the first try already used, so the second comes on the next update; the"
        " confirmation stays. single: RGBD/MaxOdomCacheSize 0, the first good localisation is"
        " accepted. Sent with the visual strategy while the database localises and changed live;"
        " under ICP and while it maps, always rtabmap",
        why="measured 2026-09-24 parked at the base under the evening lamps, camera only, xfeat"
        " (scratch/link_autopsy/confirm_ab.sh, 240 s each): rtabmap accepted 0 of 244 updates —"
        " XFeat registered with 83-118 inliers once every 11 updates, the night's ORB hypotheses"
        " (0.05-0.07) never reach 0.11 for the confirming try and the cache rolls the first one"
        " out (scratch/link_autopsy/localisation_cadence.py); aggressive accepted 41 of 42 and"
        " single 44 of 44, all within 7 cm of the seed and within 1.2 deg of the yaw at which the"
        " lidar's scan fits the map (scratch/link_autopsy/lidar_yaw_truth.py). aggressive keeps"
        " RTAB-Map's own two-localisation rule",
        on_when="aggressive whenever the camera has to localise alone in other light than the"
        " database's; single if a drive shows aggressive still waiting between localisations",
        off_when="rtabmap to reproduce RTAB-Map's stock confirmation, or if the extra registrations"
        " (one per update with a hypothesis over 0.05) cost the laptop more than it can spare",
    ),
    Flag(
        "visual_proximity",
        PROXIMITY_STOCK,
        description="whether a localised camera also registers every update against the database"
        " nodes near its pose (RGBD/ProximityBySpace), each an XFeat re-extraction and a"
        " LighterGlue match; off, only the words' own hypothesis is registered, one at most per"
        " update. Sent with the visual strategy while the database localises and changed live;"
        " under ICP and while it maps, always on (the lidar's proximity links are cheap and most"
        " of the graph's)",
        why="on, measured 2026-09-24. RTAB-Map in pepin-laptop:xfeat on the evening drives with"
        " the lidar-held truth (runs 0455-0465, 425 updates, LoopThr 0.05, scratch/xfeat/"
        "REPORT_replay.txt): on accepted 213 (50 %), 51 wrong at 0.25 m / 5 deg, median 0.12 m /"
        " 3.0 deg, p90 0.30 m / 8.6 deg, 775 / 1877 ms an update; off accepted 160 (38 %), the same"
        " 51 wrong, median 0.10 m / 3.9 deg, p90 0.31 m / 11.0 deg, 551 / 826 ms, and a cluster of"
        " parked accepts 49-57 deg off in 0458. Parked at the base on: 137 accepted in 300 s"
        " within +0.2..+0.8 deg of the lidar's yaw; off: the accepts walked together to 93 deg"
        " (+3.4) within minutes, each node pulling its own bias",
        on_when="always while its cost fits the laptop: every update then localises, a hypothesis"
        " the words miss is still caught by the pose, and the per-node biases average out",
        off_when="when the camera's localisations arrive seconds late, or the laptop's CPU is short"
        " under a camera-only drive",
    ),
    Flag(
        "registration_backend",
        "auto",
        choices=BACKENDS,
        env="PEPIN_REGISTRATION_BACKEND",
        description="where RTAB-Map's XFeat keypoints and LighterGlue matches are computed (the"
        " xfeat visual features): service — the localisation service on the laptop's GPU"
        " (pepin.localization_service, PEPIN_MODELS_URL), no features when it does not answer;"
        " local — in RTAB-Map's own process on the Docker VM's CPU, as before 2026-09-24; auto —"
        " the service, and a call it does not answer computed locally. Written to"
        f" {REGISTRATION_FILE}, which RTAB-Map's adapters read on every call (one stat): live,"
        " no restart of RTAB-Map",
        why="auto, measured 2026-09-24 in RTAB-Map itself against the NIGHT branch's in-process"
        " adapters under the same conditions (scratch/models/backend_control.sh and .py: the"
        " replay of camera-only run 0466, words at LoopThr 0.05, four runs back to back, the live"
        " stack beside them): the registration of a localised update took 571 ms median through"
        " the service against 1896 and 2326 ms for the night's adapters in the runs before and"
        " after it (3.3-4.1x), 2428 ms for this branch's local fallback (the night's path, within"
        " that drift), 18 localised updates of 20 in each — one run, no truth, so a speed, not a"
        " recognition result; the 0.67 s of 2026-09-23 was a quieter laptop. From inside a"
        " container XFeat answers an 800x600 picture in 41 ms, a node picture already seen in 1.5"
        " ms, LighterGlue a pair in 129 ms (scratch/models/endpoint_bench.py). auto keeps the old"
        " path as the answer to a service that is down",
        on_when="service to measure the service alone (a registration it cannot answer then finds"
        " no features, which shows as no recognition), auto always otherwise",
        off_when="local to reproduce the registration of before 2026-09-24, or when the laptop's"
        " GPU is wanted elsewhere",
    ),
    Flag(
        "place_recognition",
        PLACE_DESCRIPTOR,  # the default since 2026-09-28: shown by the drives of 09-25/26 and 09-28
        choices=(PLACE_WORDS, PLACE_DESCRIPTOR),
        description="how RTAB-Map finds WHICH database node a picture is (its likelihood, before"
        " any registration): words — the ORB bag of words' TF-IDF (Kp/TfIdfLikelihoodUsed true,"
        " Rtabmap/VirtualPlaceLikelihoodRatio 0, RTAB-Map's defaults); descriptor — the dot"
        " product of the nodes' learned place descriptors as z-scores (false and 1; rtabmap"
        " Memory::computeLikelihood -> Signature::compareTo, Rtabmap::adjustLikelihood), which"
        " sensor_pack attaches to every snapshot. descriptor is sent ONLY when it cannot abort"
        " RTAB-Map: its core carries ros/patches/rtabmap-keep-global-descriptors.patch (the"
        " marker /opt/rtabmap_patches/keep-global-descriptors), the snapshots carry one each"
        f" ({PLACE_TOPIC}) and the database's census at this start ({CENSUS_ENV}, taken by the"
        " launch before RTAB-Map opens the file) says every node carries exactly one of the same"
        " length — and only while the camera snapshots are described (descriptor_null_share);"
        " otherwise the words, and the report line says why. Live",
        why="words until a drive has shown the descriptor on the robot (a default flips after a"
        " drive). Measured 2026-09-24 in RTAB-Map itself on the replay of the evening runs"
        " against the backfilled daylight database (scratch/models/replay_place.py,"
        " replay_matrix.sh, matrix_report.py; xfeat, 2 px, proximity on, a lidar-only snapshot"
        " every fourth). Words at the stock Rtabmap/LoopThr 0.11: hypotheses 0.05-0.09, 0 camera"
        " updates localised on 0457, 0460 and the camera-only 0466; words at 0.05 (visual_confirm"
        " aggressive, this node's default): 9 of 23 (9 of 9 judged right), 3 of 17 (0 of 1) and 53"
        " of 55. Descriptor with the ratio 1 at 0.11: hypotheses 0.42-0.88 (median), 16 of 23 (15"
        " of 15 right), 14 of 17 (11 of 11) and 54 of 55; at 0.05 two of 0460's eleven judged were"
        " WRONG — so the descriptor goes with visual_confirm rtabmap, and the report line says so"
        " when it does not. Judged counts are 16 or fewer a run and 0466 has no truth. With the"
        " ratio 0 the descriptor's hypotheses read 0.01 and nothing localised. The retrieval"
        " behind it: BoQ-DINOv2 R@1 0.986 on 219 evening frames (scratch/models/place_parity.py)."
        " Without the patch RTAB-Map 0.22.1 aborted at the first comparison after a registration"
        " (Signature.cpp:252), which is why the patch is a gate",
        on_when="descriptor on a backfilled database (ros/tools/place_backfill.py) and a patched"
        " core, together with visual_confirm rtabmap: the descriptor's hypotheses reach 0.11 by"
        " themselves, and aggressive's 0.05 let the two wrong ones of 0460 through",
        off_when="words to reproduce RTAB-Map's stock place recognition, or when a drive shows the"
        " descriptor naming the wrong node; an unpatched core, a database not backfilled,"
        " snapshots without descriptors or a service that is not describing keep the words by"
        " themselves",
    ),
    Flag(
        "start_needs_placement",
        True,
        description=f"what goes out on {PLACEMENT_TOPIC} (latched) says this start of RTAB-Map"
        " is PLACED only once an update has recognised a node of the database it loaded, or an"
        f" operator's seed ({RTABMAP_INITIAL_POSE}) has been heard since its first update — or it"
        " loaded an empty database, whose start pose is the map's origin. The board's goal"
        " clients (ros/tools/goto_ros.py, pepin_bringup.goal_server) refuse a goal"
        " until then, saying to seed or to let the camera see a mapped"
        " place. Off, every start counts as placed: RTAB-Map's pose is taken as it is",
        why="on, measured 2026-09-23: after a restart RTAB-Map publishes map -> odom from the pose"
        " it SAVED at its last shutdown, before recognising anything, and the preflight took that"
        " fresh transform for a localisation — the cart was 'at home' while standing at the"
        " bookshelf (0 recognised a node in 130-198 updates, hypothesis 0.07), and after the next"
        " restart 76 cm off, inside the table, where Hybrid refused 'Start occupied' and the"
        " recoveries ran 93 times in 81 s (the journal, 19:05 and 20:57). It is the startup-zero"
        " trap a third time",
        on_when="always: a pose nobody has vouched for since the start is not a pose to drive on",
        off_when="to drive on the saved start pose anyway — a cart known to stand exactly where"
        " RTAB-Map last shut down, with the camera unable to recognise anything (darkness) and no"
        " seed at hand. A refusal of SILENCE (this node down, respawning, or older than this flag)"
        " is lifted by the goal server's flag of the same name, which both goal clients obey",
    ),
)


def _matched_id(msg: Info) -> int:
    """The database node this update RECOGNISED, or 0 for one that recognised nothing: the node a
    loop closure matched, else the node a proximity link matched.

    RTAB-Map numbers its nodes from 1, so zero is "nothing named". Read through ``getattr`` because
    a build whose Info does not carry a field must read as an update that matched nothing rather
    than crash the node.
    """
    loop = int(getattr(msg, "loop_closure_id", 0) or 0)
    proximity = int(getattr(msg, "proximity_detection_id", 0) or 0)
    return loop or proximity


@dataclass(frozen=True)
class HeldSwitch:
    """A memory-mode switch the rule asked for that has not gone out yet: to which mode, why, and
    whether the SERVICE held it back (busy, not up) rather than this node's own ordering."""

    mapping: bool
    why: str
    refused: bool


class RtabmapFrame(Node):
    """Publishes the graph's localisation as a measurement, and owns RTAB-Map's memory mode."""

    def __init__(self) -> None:
        super().__init__("rtabmap_frame")
        self._switches = Switches(self, with_knobs(FLAGS, load_knobs("rtabmap_frame")))
        # RTAB-Map's memory mode, pinned to localising (pepin.graphmode.ModeRule): a restart in
        # mapping mode opens a session per start, and the published grid is then the current
        # node's component of working memory (vslam.launch.py's rtabmap_memory).
        self._mode = ModeRule(MODE_HOLD_S, ALWAYS_LOCALISE, GRAPH)
        self._mode_pending: Any = None  # a switch the service has not answered yet
        self._mode_failed = 0  # switches that could not go out when the rule asked for them...
        self._mode_held: HeldSwitch | None = None  # ...and the one asked for now, held back
        # ...and the other live switch: which registration RTAB-Map runs, decided by what the
        # snapshots carry (pepin.graphmode.StrategyRule). It starts at the strategy the launch table
        # set, so the rule asks for nothing until it has a reason to.
        self._strategy = StrategyRule(STRATEGY_ICP)
        self._snapshots: SnapshotState | None = None  # the last state sensor_pack published...
        self._snapshots_at = -math.inf  # ...and when it reached us, by our clock
        self._strategy_failed = 0  # strategy switches the parameter path could not take...
        self._strategy_held = False  # ...and whether the one asked for now is waiting for it
        # ...and what the visual registration matches with (visual_features, pnp_reproj_px): the
        # set last sent to RTAB-Map, starting as the launch table's own (ORB, 2 px), and whether
        # this image can run xfeat at all — the two adapters RTAB-Map loads by path are here or not.
        self._xfeat_here = all(Path(p).is_file() for p in (XFEAT_DETECTOR_PATH, XFEAT_MATCHER_PATH))
        self._visual_sent = visual_parameters(STRATEGY_ICP, FEATURES_ORB, PNP_REPROJ_PX)
        # Whether RTAB-Map has READ the last set: every set counted, and the set the last re-read
        # was sent after (a re-read goes out only once its set is answered) with its future.
        self._sets_sent = 0
        self._reread_after = 0
        self._reread_pending: Any = None
        # WHERE RTAB-MAP'S PYTHON ADAPTERS COMPUTE (registration_backend and its two numbers):
        # written for them to read on every call, and their counters read back for the report.
        self._registration_file = LiveFile(registration_file())
        self._registration_status = StatusBoard()
        # HOW RTAB-MAP FINDS WHICH NODE A PICTURE IS (place_recognition): what the snapshots
        # carry, the census of the database at this start, and the likelihood last sent. None
        # at first: the first decision is ALWAYS sent, because this node may be a respawn beside
        # an RTAB-Map still holding the descriptor likelihood an earlier incarnation sent.
        self._place: SnapshotPlace | None = None
        self._census = Census.from_json(os.environ.get(CENSUS_ENV, ""))
        self._rtabmap_keeps = rtabmap_keeps_descriptors()  # the image's patch marker
        self._recognition_sent: dict[str, str] | None = None
        self._recognition_why = PLACE_WORDS
        self._infos = 0  # /rtabmap/info messages consumed...
        self._named = 0  # ...of which this many recognised a database node
        self._first_ref: int | None = None  # the first node id this start created
        # WHAT THIS START OF RTAB-MAP IS PLACED BY (start_needs_placement): its updates, how many
        # recognised a node of the loaded database, and the operator seeds heard since its first
        # update. A start is RTAB-Map's, not this node's: numbering that goes back DOWN is
        # RTAB-Map restarted under a node that kept running, and all of it starts again
        # (_new_start).
        self._last_ref = 0
        self._start_updates = 0
        self._start_recognised = 0
        self._start_seeds = 0
        self._restarts = 0  # RTAB-Map restarts seen under this node
        self._placement_said: str | None = None  # the last placement published, sans stamp
        self._grids_relayed = 0
        self._localizations = 0  # localisations heard
        self._hypothesis = 0.0  # how close the last update came to recognising something
        self.create_subscription(
            PoseWithCovarianceStamped, LOCALIZATION_TOPIC, self._on_localization, 5
        )
        self.create_subscription(Info, INFO_TOPIC, self._on_info, 5)
        self.create_subscription(PoseWithCovarianceStamped, RTABMAP_INITIAL_POSE, self._on_seed, 5)
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # RTAB-Map's grid passes through here on its way to being THE map. This subscription is
        # also what keeps RTAB-Map assembling a grid at all: it builds one only while somebody
        # listens (MapsManager's subscription-count gate).
        self._map_pub = self.create_publisher(OccupancyGridMsg, MAP_TOPIC, latched)
        # Whether this start is placed, latched: a goal client is a fresh process on the board
        # and reads it once, in its first callback, as it reads /places.
        self._placement_pub = self.create_publisher(String, PLACEMENT_TOPIC, latched)
        self.create_subscription(OccupancyGridMsg, GRID_TOPIC, self._on_grid, latched)
        # Latched, matching sensor_pack's own publisher: this node may start after it, and the
        # present state must not have to wait for the next change to arrive.
        self.create_subscription(String, SNAPSHOT_STATE_TOPIC, self._on_snapshots, latched)
        self.create_subscription(String, PLACE_TOPIC, self._on_place, latched)
        # The mode services.
        self._modes = {
            True: self.create_client(Empty, MAPPING_SERVICE),
            False: self.create_client(Empty, LOCALISATION_SERVICE),
        }
        # A set goes to RTAB-Map as ONE set_parameters_atomically request, which rtabmap_slam
        # applies as one /parameter_events notification and one parseParameters. Measured
        # 2026-09-24 (scratch/xfeat_critic/atomic_set.sh): five parameters in one set_parameters
        # request arrived as 5 events and 5 parseParameters over 61 ms, every intermediate state a
        # registration nobody chose; atomically, 1 and 1.
        self._tuner = self.create_client(
            SetParametersAtomically, f"{RTABMAP_NODE}/set_parameters_atomically"
        )
        self._reread = self.create_client(Empty, f"{RTABMAP_NODE}/update_parameters")
        self.create_timer(1.0 / RATE_HZ, self._publish)
        self.get_logger().info(
            f"rtabmap frame up: {GRID_TOPIC} -> {MAP_TOPIC}, memory pinned to localising,"
            f" placement on {PLACEMENT_TOPIC}; flags: {self._switches.state(live_only=False)}"
        )
        self.create_timer(30.0, self._report)

    def _report(self) -> None:
        """Every 30 s: how many updates arrived and how many of them recognised a node, how close
        the graph came to recognising anything, RTAB-Map's memory mode, registration and place
        recognition, the grids relayed, the placement and the switches."""
        self.get_logger().info(
            f"rtabmap frame: {self._infos} updates, {self._named} recognised a node,"
            f" {self._localizations} localisations heard;"
            f" hypothesis {self._hypothesis:.2f}; rtabmap memory {self._mode_text()};"
            f" rtabmap registration {self._strategy_text()};"
            f" adapters ({self._switches['registration_backend']}) {self._registration_text()};"
            f" place recognition {self._recognition_why};"
            f" {self._grids_relayed} grids relayed;"
            f" start {self._placement().how()}"
            + (f", {self._restarts} RTAB-Map restarts seen" if self._restarts else "")
            + f"; flags: {self._switches.state(live_only=False)}"
        )

    def _strategy_text(self) -> str:
        """RTAB-Map's registration for a report line: which strategy is set and why, what the
        snapshots say it should be, and the switches the parameter path could not take."""
        state = self._snapshots
        if state is None:
            said = f"nothing on {SNAPSHOT_STATE_TOPIC} yet"
        else:
            age = self._now() - self._snapshots_at
            stale = " STALE" if age > state.refresh_s else ""
            said = f"snapshots {state.text()}, {age:.1f} s ago{stale}"
        return f"{self._strategy.text()}; {self._visual_text()}; {said}" + (
            f", {self._strategy_failed} switches the parameter path could not take"
            if self._strategy_failed
            else ""
        )

    def _mode_text(self) -> str:
        """RTAB-Map's memory mode for a report line: the rule's own verdict, the switches that
        could not go out when asked for, and why the one asked for now is held back."""
        return (
            self._mode.text()
            + (
                f", {self._mode_failed} switches the service could not take"
                if self._mode_failed
                else ""
            )
            + (f", held back: {self._mode_held.why}" if self._mode_held else "")
        )

    def _now(self) -> float:
        """This node's clock in seconds: how long a verdict has held."""
        return stamp_seconds(self.get_clock().now().to_msg())

    # ---- inputs --------------------------------------------------------------------------
    def _on_grid(self, msg: OccupancyGridMsg) -> None:
        """RTAB-Map's grid, relayed onto the one map topic."""
        self._grids_relayed += 1
        self._map_pub.publish(msg)

    def _on_info(self, msg: Info) -> None:
        """One RTAB-Map update: WHICH NODE it recognised, if any, and how close the last hypothesis
        came. A node of the LOADED database recognised is what places this start."""
        self._infos += 1
        stats = dict(zip(msg.stats_keys, (float(v) for v in msg.stats_values), strict=False))
        hypothesis = stat(stats, HIGHEST_HYPOTHESIS)
        if hypothesis is not None:
            self._hypothesis = float(hypothesis)
        ref = int(getattr(msg, "ref_id", 0))
        if 0 < ref < self._last_ref:
            self._new_start(ref)
        if ref > 0:
            self._last_ref = ref
        if ref > 0 and self._first_ref is None:
            # the first node this start made: everything older is the loaded map
            self._first_ref = ref
        if self._first_ref is not None:
            self._start_updates += 1
        matched = _matched_id(msg)
        if matched <= 0:
            return
        if self._first_ref is not None and matched < self._first_ref:
            self._start_recognised += 1  # a node of the map this start LOADED
        self._named += 1

    def _new_start(self, ref: int) -> None:
        """RTAB-Map restarted under this node: its numbering went back down to ``ref`` (within a
        start every update's node id is one more than the last; a restart numbers on from the
        database's last saved node). That is certain only while RTAB-Map LOCALISES, which saves
        no node: a start that mapped saved its nodes, the next numbers upward and is not seen.

        The placement's first node, counts and seeds are the old start's and start again."""
        self._restarts += 1
        self.get_logger().warning(
            f"rtabmap frame: RTAB-Map restarted (node ids went from {self._last_ref} back to"
            f" {ref}): this start is not placed until it recognises a node or is seeded"
        )
        self._first_ref = None
        self._start_updates = self._start_recognised = self._start_seeds = 0

    def _on_seed(self, _msg: PoseWithCovarianceStamped) -> None:
        """An operator told RTAB-Map where the cart stands (ros/tools/goto_ros.py seed): counted
        once this start has made an update — RTAB-Map is up to take it — and never before, so a
        seed sent into a restart does not place the start that follows it."""
        if self._first_ref is None:
            self.get_logger().warning(
                f"rtabmap frame: a seed on {RTABMAP_INITIAL_POSE} before RTAB-Map's first update"
                " of this start: not counted, seed again once it runs"
            )
            return
        self._start_seeds += 1
        self.get_logger().info(
            f"rtabmap frame: operator seed {self._start_seeds} since RTAB-Map's start: placed"
        )

    def _placement(self) -> Placement:
        """What this start of RTAB-Map rests on (:class:`pepin.watch.Placement`)."""
        loaded = (
            None if self._first_ref is None else self._first_ref != FIRST_ID_OF_AN_EMPTY_DATABASE
        )
        return Placement(
            self._start_updates,
            self._start_recognised,
            self._start_seeds,
            loaded,
            required=self._switches.on("start_needs_placement"),
        )

    def _say_placement(self) -> None:
        """Publish the placement (latched) whenever it changed, the flag included."""
        placement = self._placement()
        said = placement.to_json(0.0)
        if said == self._placement_said:
            return
        self._placement_said = said
        self._placement_pub.publish(String(data=placement.to_json(self._now())))

    def _on_localization(self, _msg: PoseWithCovarianceStamped) -> None:
        """RTAB-Map placing itself in the database it loaded: counted for the report line."""
        self._localizations += 1

    def _on_snapshots(self, msg: String) -> None:
        """What pepin_bringup.sensor_pack's snapshots carry right now: the evidence RTAB-Map's
        registration strategy follows, and the hold that change must survive (the packer's own
        liveness window, carried in the message). A message that does not parse says nothing and is
        ignored — the strategy in force is never changed on a reading nobody could read."""
        state = SnapshotState.from_json(msg.data)
        if state is None:
            return
        self._snapshots, self._snapshots_at = state, self._now()

    def _on_place(self, msg: String) -> None:
        """What sensor_pack attaches to its snapshots (one descriptor each or none, their length,
        the weights of the last vector): the half of the place-recognition rule the database's
        census cannot answer. A message that does not parse is ignored."""
        place = SnapshotPlace.from_json(msg.data)
        if place is not None:
            self._place = place

    # ---- RTAB-Map's memory ----------------------------------------------------------------
    def _decide_mode(self) -> None:
        """Ask RTAB-Map to learn or only to recognise, on a change of verdict that has held.

        The rule is :class:`pepin.graphmode.ModeRule`; everything this method adds is the wiring.
        A switch is not sent while the last one is unanswered — that is the "no faster than the
        service answers" rule, exact and with no number in it — and the parameters each mode needs
        (:data:`MODE_PARAMETERS`) travel with it, because the mode services do not touch them.
        """
        verdict = self._mode.update(self._now(), None, None)
        if verdict is None:
            self._mode_held = None
            return
        wait = self._features_first(verdict)
        refused = None if wait is not None else self._send_mode(verdict)
        why = wait or refused
        if why is None:
            self._mode_held = None
            return
        # NOT SENT, SO NOT APPLIED: the rule takes the switch back and asks again next tick.
        # Recorded as applied, it was never asked again, and the visual set that follows the mode
        # (_visual_wanted) believed a mode RTAB-Map had never been told.
        self._mode.withdraw()
        if refused is not None and (self._mode_held is None or not self._mode_held.refused):
            self._mode_failed += 1  # once per switch the service held back, not once per tick
        self._mode_held = HeldSwitch(verdict.mapping, why, refused is not None)

    def _features_first(self, verdict: ModeVerdict) -> str | None:
        """Why a switch to MAPPING must wait for the visual set, or ``None`` when it may go out.

        ORB'S SET FIRST, AND ANSWERED. With RGBD/LoopClosureReextractFeatures on, a node the
        database WRITES keeps no descriptors and no 3D (Memory.cpp:6126), so the xfeat set must be
        out of force before RTAB-Map maps a single update. Sent after the mapping call, it arrived
        on a later tick while RTAB-Map's executor could run a sensor update in between (a mapping
        update with re-extraction on). So the switch waits here: the held mapping verdict makes
        :meth:`_visual_wanted` ORB's, :meth:`_decide_strategy` sends that set this same tick, and
        the switch goes out once RTAB-Map has re-read it — the re-read is sent only after the set
        was answered and runs in the group the mode services run in (CoreWrapper.cpp:665-676).
        """
        if not verdict.mapping:
            return None
        if self._visual_sent != self._visual_wanted(self._strategy.strategy):
            return "ORB's visual set goes to RTAB-Map first"
        if not self._settled():
            return "RTAB-Map has not re-read its visual set yet"
        return None

    def _send_mode(self, verdict: ModeVerdict) -> str | None:
        """Send one memory-mode switch and the parameters that travel with it; ``None`` when it
        went out, otherwise why it could not (the last switch unanswered, the service not up)."""
        if self._mode_pending is not None and not self._mode_pending.done():
            return "the last switch is unanswered"
        client = self._modes[verdict.mapping]
        if not client.service_is_ready():
            return f"{client.srv_name} is not up"
        self._mode_pending = client.call_async(Empty.Request())
        self._retune(verdict.mapping)
        # rclpy names a client's service ``srv_name``; a fake with ``.name`` let this line crash
        # the node on its first live switch (2026-09-18).
        self.get_logger().info(f"rtabmap memory: {verdict.text()} ({client.srv_name})")
        return None

    def _retune(self, mapping: bool) -> None:
        """Set the parameters the new mode needs and have RTAB-Map re-read them.

        ``RGBD/LinearUpdate`` and ``RGBD/AngularUpdate`` are the pair: 0 while LOCALISING, so a
        parked cart is recognised at all (measured live — with the defaults every update at rest
        read ``Memory/Small_movement`` 1 and named no node), and 0.05 while MAPPING, because 0
        there keeps a node a second at a standstill and put 250 junk nodes in the database in one
        evening. They are string-typed on RTAB-Map's side, and the mode services do not carry them.
        """
        self._set_parameters(MODE_PARAMETERS[mapping])

    def _set_parameters(self, values: dict[str, str]) -> bool:
        """Set these RTAB-Map parameters on its node and have it re-read them; whether the set
        went out at all (its re-read follows its answer).

        WHAT RTAB-MAP HONOURS. The values are STRINGS because rtabmap declares every one of its
        parameters as one and reads it back with ``as_string()``, and a name the LAUNCH table
        never overrode is accepted by the set and then never looked at, which is why the two
        parameter tables here (:data:`MODE_PARAMETERS`,
        :data:`pepin.graphmode.REGISTRATION_PARAMETERS`) name only parameters that table already
        carries. A set is applied TWICE on RTAB-Map's side: rtabmap_slam hears its own
        /parameter_events and hands every change to ``Rtabmap::parseParameters`` as it arrives
        (CoreWrapper.cpp:907-970) — one event for the whole set, since it goes atomically — and
        ``update_parameters`` re-reads
        the whole map synchronously, in the group the mode services run in, which is what
        :meth:`_settled` waits for. The file:line for the rest is in :mod:`pepin.graphmode`.
        """
        tuner = self._tuner
        if tuner is None or self._reread is None or not self._path_up():
            return False  # both calls or neither: a half-sent set is retried whole
        request = SetParametersAtomically.Request()
        request.parameters = [
            Parameter(
                name=name,
                value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value),
            )
            for name, value in values.items()
        ]
        self._sets_sent += 1
        serial = self._sets_sent
        # The re-read goes out when the set is ANSWERED, not beside it: the two are different
        # callback groups on RTAB-Map's side (the node's parameter services, CoreWrapper's
        # processing group), so a re-read sent beside the set could run first and re-read the old
        # values. Answered after the set, it has run parseParameters on the new ones.
        tuner.call_async(request).add_done_callback(lambda _: self._reread_once(serial))
        return True

    def _reread_once(self, serial: int) -> None:
        """Have RTAB-Map re-read its parameters now that set number ``serial`` is answered."""
        if self._reread is None or not self._reread.service_is_ready():
            return
        self._reread_pending = self._reread.call_async(Empty.Request())
        self._reread_after = serial

    def _settled(self) -> bool:
        """Whether RTAB-Map has READ every set this node sent: the last one was answered, the
        re-read sent after it was answered too (nothing sent yet: the launch table's values)."""
        answered = self._reread_pending is None or self._reread_pending.done()
        return self._reread_after == self._sets_sent and answered

    # ---- RTAB-Map's registration ----------------------------------------------------------
    def _decide_strategy(self) -> None:
        """Ask RTAB-Map for the registration the snapshots need, on a change that has held, with
        the visual features the flags choose; and re-send those features alone when a flag moves.

        The rule is :class:`pepin.graphmode.StrategyRule` and the hold is the one the STATE carries
        — how long the packer itself takes to change its mind about a source — so nothing here is a
        number. A state older than its own refresh is no evidence at all and the strategy in force
        stays: sensor_pack having gone quiet is not the lidar having gone away.
        """
        if self._tuner is None:
            return
        # Under ICP a camera-only cart cannot localise AT ALL (2026-09-18: in a minute of
        # camera-only snapshots RTAB-Map logged 28 'Missing visual features' and 56 'Requested
        # laser scan data' and not one update named a node), and the strategy is one object for
        # the process (rtabmap/core/Memory.cpp:721-731), so it follows what the snapshots carry.
        state = self._snapshots
        fresh = state is not None and self._now() - self._snapshots_at <= state.refresh_s
        verdict = self._strategy.update(
            self._now(),
            state.refresh_s if state is not None else 0.0,
            state.carries(LIDAR) if (state is not None and fresh) else None,
            state.kind if state is not None else "",
            picture=state is not None and fresh and state.carries(CAMERA),
        )
        if verdict is None:
            self._strategy_held = False
        else:
            visual = self._visual_wanted(verdict.strategy)
            # One set for the strategy and its features, applied as one (atomically):
            # the pipeline RTAB-Map re-creates on a new Reg/Strategy is built from the accumulated
            # map, so it is born with the right ones and no update runs between the two halves.
            if not self._set_parameters({**verdict.parameters, **visual}):
                # Not sent, so not applied: the rule takes it back and asks again next tick, and
                # the strategy the features follow stays the one RTAB-Map still runs.
                self._strategy.withdraw()
                if not self._strategy_held:
                    self._strategy_failed += 1  # once per switch held back, not once per tick
                self._strategy_held = True
                return
            self._strategy_held = False
            self._visual_sent = visual
            self.get_logger().info(
                f"rtabmap registration: {verdict.text()} -> Reg/Strategy {verdict.strategy},"
                f" {self._visual_text()} (set on {RTABMAP_NODE} and re-read through"
                f" {RTABMAP_NODE}/update_parameters)"
            )
            return
        visual = self._visual_wanted(self._strategy.strategy)
        if visual != self._visual_sent and self._set_parameters(visual):
            # A flag moved, or the memory mode did: the strategy in force is kept, only the
            # visual set is re-sent (RegistrationVis re-reads it and
            # rebuilds its detectors, RegistrationVis.cpp:290-293).
            self._visual_sent = visual
            self.get_logger().info(f"rtabmap registration: {self._visual_text()}")

    def _path_up(self) -> bool:
        """Whether both halves of RTAB-Map's parameter path answer: the atomic set and the re-read.
        Asked before every set, so a half-up path is sent nothing rather than half a set."""
        return all(c is not None and c.service_is_ready() for c in (self._tuner, self._reread))

    def _features(self) -> str:
        """The feature set the flag asks for, if this image can run it; ``orb`` otherwise."""
        asked = str(self._switches["visual_features"])
        return asked if (asked != FEATURES_XFEAT or self._xfeat_here) else FEATURES_ORB

    def _visual_wanted(self, strategy: str) -> dict[str, str]:
        """The visual parameters RTAB-Map should hold under ``strategy``
        (:func:`pepin.graphmode.visual_parameters`): the flags' set under the visual strategy while
        the database is certainly only read (:meth:`_database_only_read`), ORB's otherwise."""
        return visual_parameters(
            strategy,
            self._features(),
            float(self._switches["pnp_reproj_px"]),
            mapping=not self._database_only_read(),
            confirm=str(self._switches["visual_confirm"]),
            proximity=self._switches.on("visual_proximity"),
        )

    def _database_only_read(self) -> bool:
        """Whether RTAB-Map is LOCALISING for certain: this node told it so, the call is answered
        (the mode services run in the group the re-read does, CoreWrapper.cpp:675-676), and no
        switch to mapping is waiting. The only state the xfeat set may be in force in — an unknown
        mode, an unanswered switch or a mapping one on its way all read as mapping, because a node
        the database WRITES under re-extraction keeps no descriptors and no 3D (Memory.cpp:6126)."""
        if self._mode_held is not None and self._mode_held.mapping:
            return False
        answered = self._mode_pending is None or self._mode_pending.done()
        return self._mode.mode == LOCALISING and answered

    def _visual_text(self) -> str:
        """The visual registration for a report line: ``visual features xfeat, PnP 2 px`` — and
        when the flag's set is not what went out, why (``xfeat asked, no /opt/xfeat/... here``)."""
        sent = self._visual_sent
        features = FEATURES_XFEAT if sent.get("Vis/FeatureType") == "15" else FEATURES_ORB
        asked = str(self._switches["visual_features"])
        why = ""
        if asked == FEATURES_XFEAT and not self._xfeat_here:
            why = f" (xfeat asked, but this image has no {XFEAT_DETECTOR_PATH})"
        elif asked != features:
            why = f" ({asked} asked, sent with the visual strategy only and only while localising)"
        confirm = next(
            (name for name, table in CONFIRM_PARAMETERS.items() if table.items() <= sent.items()),
            "?",
        )
        asked_confirm = str(self._switches["visual_confirm"])
        confirm_why = "" if asked_confirm == confirm else f" ({asked_confirm} asked, visual only)"
        proximity = "on" if sent.get("RGBD/ProximityBySpace", "true") == "true" else "off"
        return (
            f"visual features {features}{why}, PnP {sent.get('Vis/PnPReprojError', '?')} px,"
            f" confirm {confirm}{confirm_why}, proximity {proximity}"
        )

    # ---- where the adapters compute, and how places are found --------------------------------
    def _write_registration(self) -> None:
        """Hand RTAB-Map's adapters the registration flags (pepin.live_settings), when they
        changed; a write that fails is retried on the next tick."""
        settings = RegistrationSettings(
            backend=str(self._switches["registration_backend"]),
            timeout_s=float(self._switches["registration_timeout_s"]),
            top_k=int(self._switches["xfeat_top_k"]),
        )
        try:
            self._registration_file.write_if_changed(settings.to_json())
        except OSError as exc:
            self.get_logger().error(
                f"cannot write {self._registration_file.path} for RTAB-Map's adapters: {exc}",
                throttle_duration_sec=60,
            )

    def _registration_text(self) -> str:
        """The adapters' own counters for the report line: per adapter the backend they read and
        where their answers came from, or why there is nothing to say."""
        blocks = self._registration_status.read()
        now = time.time()
        parts = []
        for name in ("xfeat", "match"):
            block = blocks.get(name)
            if not isinstance(block, dict):
                continue
            age = now - float(block.get("at", 0.0))
            stale = f", {age:.0f} s old" if age > REGISTRATION_STATUS_FRESH_S else ""
            error = f", last error {block['last_error']}" if block.get("last_error") else ""
            parts.append(
                f"{name} {block.get('backend', '?')}: service {block.get('service', 0)}, local"
                f" {block.get('local', 0)}, fallback {block.get('fallback', 0)}, failed"
                f" {block.get('failed', 0)}, {block.get('round_trip_ms', 0)} ms{stale}{error}"
            )
        return "; ".join(parts) if parts else "the adapters have not reported (no xfeat call yet)"

    def _decide_recognition(self) -> None:
        """Send RTAB-Map the likelihood place_recognition asks for, when it is safe
        (pepin.global_descriptor.recognition_parameters) and differs from the last one sent;
        through the same parameter path as the registration sets, retried while it is down."""
        if self._tuner is None:
            return
        wanted, why = recognition_parameters(
            str(self._switches["place_recognition"]),
            self._place,
            self._census,
            self._rtabmap_keeps,
            float(self._switches["descriptor_null_share"]),
        )
        if wanted == DESCRIPTOR_PARAMETERS and self._switches["visual_confirm"] != CONFIRM_STOCK:
            why += (
                f"; visual_confirm {self._switches['visual_confirm']}: with the descriptor its"
                " lower LoopThr let 2 of 11 wrong localisations through on 0460 (visual_confirm"
                " rtabmap)"
            )
        self._recognition_why = why
        if wanted == self._recognition_sent or not self._set_parameters(wanted):
            return
        self._recognition_sent = wanted
        self.get_logger().info(
            f"rtabmap place recognition: {why} ({TFIDF} {wanted[TFIDF]}, set on {RTABMAP_NODE})"
        )

    # ---- outputs -------------------------------------------------------------------------
    def _publish(self) -> None:
        """At :data:`RATE_HZ`: the placement when it changed, RTAB-Map's memory mode, its
        registration and place recognition, and the adapters' settings."""
        self._say_placement()
        self._decide_mode()
        self._decide_strategy()
        self._decide_recognition()
        self._write_registration()


def main() -> None:
    spin_main(RtabmapFrame)


if __name__ == "__main__":
    main()
