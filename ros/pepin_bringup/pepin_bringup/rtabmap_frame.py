"""Put RTAB-Map's graph correction where the mode needs it: a frame here, or a message home.

RTAB-Map computes one correction per graph optimisation and publishes it as ``map_to_odom`` on
``/rtabmap/mapGraph``: the jump its "odometry" frame takes when the graph moves under it. Where
that correction belongs depends on which map the robot drives.

On a KNOWN map the board's tracker owns ``map -> odom`` and RTAB-Map runs on the EKF's own
``odom -> base_link`` (vslam.launch.py's ``graph_odom``, the default). Its graph therefore lives
in a frame of its own that starts at the ODOM frame's origin, and this node ties that frame to
the lidar map: at the first graph it reads where the tracker says the cart is and where the
graph has it, and the difference between the two is the ANCHOR — ``map <- rtabmap``, broadcast
at 10 Hz and constant for the session. RTAB-Map cannot be told that pose itself ("Initial pose
can only be set in localization mode (Mem/IncrementalMemory=false), ignoring it",
librtabmap_core 0.22.1), so the offset is learned here instead of imposed there.

With the anchor known, the graph's answer about the cart is a measurement on the lidar map:
``anchor . map_to_odom . (odom -> base_link at the graph's stamp)`` — the graph's own opinion,
built from its own odometry and every closure it has accepted — published on
:data:`MEASUREMENT_TOPIC` for the board's fusion to weigh like any other word
(``graph_measurement``). Nothing here owns ``map -> odom``.

A measurement, though, can only correct a pose that is already nearly right: this node refuses a
word disagreeing with the tracker's belief past :data:`pepin.fusion.GATE` in 3 DOF, and the
board's information filter gates what it does take by the same chi-square. So the word after the
cart is CARRIED by hand — the graph recognising the place, metres from where the tracker thinks
it stands — is exactly the word both gates throw away. That word goes out on the whole-map
CANDIDATE channel instead (:data:`CANDIDATE_TOPIC`, source ``graph``, ``graph_candidates``), the
same door the lidar's own whole-map search uses: three agreeing candidates re-seed the tracker
(:class:`pepin.watchdog.CandidateGate`), and the board's rules for a source that is not the
lidar apply to the graph unchanged.

WHETHER THE WORD IS SAID AT ALL is RTAB-Map's own recognition of the database it LOADED. Until
it has accepted a closure, a proximity link or a localisation against a node older than the
first node of its own start, the nodes it is building are an unlinked segment placed by this
session's odometry — a word off it is not an uncertain pose but a number in another coordinate
system — and the node stays silent and counts the words withheld
(``graph_words_need_recognition``, :class:`pepin.graphtrust.Recognition`, from
:data:`INFO_TOPIC`). That is tape 0333 of 2026-09-15, a camera-only drive after a day of RTAB-Map
restarts: the anchor on file was right
for the linked nodes it was measured on, the words off the new segment were about a metre and
150 degrees out, and the tracker took 19 of them before the pose flew.

WHAT THE WORD CLAIMS, when it is said, is how well the graph's last words track the tracker's
own pose carried forward by odometry between them: ``exp(-rms residual / 10 cm)`` over the last
ten seconds (:class:`pepin.graphtrust.Agreement`), which is what tells a word riding a broken
frame from one riding a good frame WITHOUT waiting for the next closure. ``graph_trust`` keeps
the older answer reachable — ``distance``, the decay with the metres driven since the last tie
to any older node — and ``flat``, the 1.0 every word claimed before 2026-09-14. Either measured
mode is capped while the anchor came from a file and nothing has been recognised. That number is
what the board publishes as its own confidence when no scan of its own measured one, and the
score a candidate is admitted on — so a word built on nothing now READS as built on nothing,
which is what the carry test of 2026-09-14 could not see: fit 1.00 on a belief 1.5-2 m wrong,
for 64 s of driving.

With ``graph_odom`` off the old arrangement is back: RTAB-Map's "odometry" IS the tracker's
pose, ``map -> rtabmap`` is the inverse of the correction (a fixed identity in its place let
the voxels drift away from the cart after every closure — the cloud moved with the graph, the
cart did not), and the measurement is the tracker's belief moved by that correction.

THE TIE IS A CALIBRATION OF THE PAIR (this lidar map, this graph database) and not a seating of
this session: both sides are files, and a transform between two files cannot move. So it is
MEASURED OVER MANY PLACES and kept where the pair can find it — a log of
(tracker on the map, RTAB-Map in the database) pairs, ``<map id>.graph_pairs.jsonl`` beside the
map the board serves (:mod:`pepin.graphtie`, the node's ``anchor_dir``) — and one rigid fit over
the whole log is the tie the words ride (``graph_tie_from_pairs``). A pair is taken only off a
seating the lidar has PINNED IN BOTH AXES, its published covariance under ``anchor_max_sigma_m``
/ ``anchor_max_sigma_deg``, because a fit is not an error bar and a scan free to slide along a
sofa would put that slide in the log. The fit is redone whenever the log grows and REPLACES the
tie in hand only when its covariance is no larger — never because something disagreed with it.

WHAT WENT: the re-learn. Until 2026-09-17 the tie was learned from ONE seating and re-learned
whenever it disagreed with the lidar-held tracker for five seconds, which is fitting a constant
to a variable: the stored anchor took three values metres apart in one evening ((-9.22, -0.40,
-127 deg) -> (-6.66, +2.49, -117) -> (-6.18, +4.34, -113)), a single false recognition (4.4 deg
of claimed sigma at 93 deg of real error) was enough to bake a lie into every later word, and
camera-only inherited a runtime dependency on the lidar exactly where there is no lidar. Nothing
here is learned from a disagreement any more, and with no sharp seating nothing is learned at all:
the tie on file is used as it is, which IS the lidar independence.

A FRESHLY BORN MAP needs no measurement: a map and a database started at the same pose in the same
second are the same frame by construction, so ``fresh_frame`` makes the tie identity with no
uncertainty and no pairs (:func:`pepin.graphtie.identity_tie`).

THE WAKE-UP is what a tie on disk buys. The cart sleeps on the charger, the board restarts — and
the tracker's saved pose is the only thing anyone knows. With a tie on disk, the FIRST graph
message already reads as a place on the map, before the tracker has said anything at all, and the
word goes out with no belief to check it against (a word is only checked against a belief that
exists). If the graph does NOT recognise the place — an unseen corner, the camera blind — no word
is published at all: the tracker keeps its saved pose, and the owner carries the cart to a place
it knows, or to the charger.

A WORD IS REFUSED IN 3 DOF, not by position alone. The old test was the metres between the word
and the tracker's belief, so a word at exactly the right place facing 90 degrees the wrong way
passed it; the test now is the squared Mahalanobis distance between the two under the sum of their
covariances against :data:`pepin.fusion.GATE` — the very gate the board's information filter will
apply next (:func:`pepin.fusion.disagreement`). The tie's own uncertainty rides in the word's
covariance, propagated through the composition, so a tie measured at one end of the flat makes an
honestly WIDE word at the other instead of a confident one.

In online SLAM (``slam`` on, ros/laptop.sh vslam --slam) RTAB-Map IS the map and the correction
is literally ``map -> odom`` — but it must become a transform ON THE BOARD, where Nav2 and the
reflexes look it up, and ``/tf`` crosses the bridge board -> laptop only (a topic allowed as a
publisher on both sides loops until nothing crosses at all). So the correction travels as a
message on ``/map_odom`` and pepin_bringup.slam_frame broadcasts it there. This node then
publishes no transform at all: the laptop reads ``map -> odom`` back over the bridge, from the
one owner.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rtabmap_msgs.msg import Info, MapGraph
from std_msgs.msg import Float32, String
from std_srvs.srv import Empty
from tf2_ros import TransformBroadcaster

from pepin.anchors import (
    ANCHOR_MAX_SIGMA_DEG,
    ANCHOR_MAX_SIGMA_M,
    Anchor,
    anchor_path,
    describe_sigma,
    load_anchor,
    room_from_identity,
    save_anchor,
    seating_refusal,
)
from pepin.anchors import (
    SUFFIX as ANCHOR_SUFFIX,
)
from pepin.flags import Flag, FlagSet
from pepin.fusion import (
    GATE,
    SIGMA_XY_M,
    SIGMA_YAW_DEG,
    Matrix,
    PoseMeasurement,
    disagreement,
)
from pepin.graphnodes import (
    ALWAYS_LOCALISE,
    ALWAYS_MAP,
    BY_TRUST,
    NODES_SUFFIX,
    ModeRule,
    NodePose,
    NodeTable,
    NodeWord,
    append_node,
    entry_from_match,
    load_nodes,
    nodes_path,
    read_sessions,
)
from pepin.graphtie import (
    PAIRS_SUFFIX,
    Tie,
    TiePair,
    append_pair,
    file_tie,
    fit_tie,
    identity_tie,
    load_pairs,
    pairs_path,
)
from pepin.graphtrust import (
    AGREEMENT_SCALE,
    AGREEMENT_WINDOW_S,
    AGREEMENT_WORDS,
    FILE_ANCHOR_TRUST,
    GRAPH_TRUST_M,
    Agreement,
    GraphTrust,
    InfoIds,
    Recognition,
)
from pepin.measurements import (
    GRAPH_FLOOR_XY_M,
    GRAPH_FLOOR_YAW_DEG,
    RemoteMeasurement,
    compose,
    graph_anchor,
    graph_measurement,
    inverse,
)
from pepin.odometry import Pose2D, wrap_angle
from pepin.sources import GRAPH
from pepin.tsdf import RigidPose
from pepin.visual_odometry import SCALE_ERROR
from pepin.watch import SIGMA_TOPIC, Preflight, Sigma, source_words
from pepin.watchdog import GlobalCandidate, same_place
from pepin_bringup.msgs import (
    map_id,
    pose_from_transform,
    stamp_seconds,
    transform_from_pose,
    transform_from_rpy,
    yaw_of,
)
from pepin_bringup.node_kit import Switches, TfLookup, bridged_qos_profile, spin_main

RATE_HZ = 10.0
KNOWN_MAP_FRAMES = ("map", "rtabmap")  # parent, child
SLAM_FRAMES = ("map", "odom")
CORRECTION_TOPIC = "/map_odom"
# The graph's word for the board's fusion. A topic of its own, NOT the camera's
# /localization/measurement: pepin.measurements.MeasurementGate fuses everything waiting in it
# into one word named "camera", so a graph measurement dropped in there would move the pose under
# the camera's name. The board gains a gate of its own for this one, named "graph", the day its
# roster does.
MEASUREMENT_TOPIC = "/localization/graph_measurement"
# ...and the other door into the board's tracker, the one a measurement cannot open: the
# whole-map candidate channel the lidar's own search publishes on (pepin.watchdog,
# pepin_bringup.laptop_localizer). A measurement corrects a pose that is nearly right; a
# candidate MOVES a pose that is simply in the wrong place, which is what a carry leaves behind.
CANDIDATE_TOPIC = "/localization/candidate"
# What the graph knows, as opposed to what it has integrated: RTAB-Map's own statistics —
# the closures and proximity links it accepted, the distance its odometry has travelled and
# how close the last hypothesis came. The trust the word travels with is made of these
# (:mod:`pepin.graphtrust`).
INFO_TOPIC = "/rtabmap/info"
# Where RTAB-Map says the cart is IN THE DATABASE IT LOADED, published only while it is actually
# localised there, in the frame "rtabmap" and with its own covariance. This is the one number in
# the stack that is a localisation rather than an integration: /rtabmap/mapGraph carries the
# correction relative to the CURRENT SESSION, which starts at zero on every restart and is pure
# odometry between closures — 42.5 m of it had accumulated on 2026-09-17, and the word built
# from it stood 3.6 m from the tracker and drove the cart at a person. Measured parked the same
# evening (scratch/rtabmap_frame_drift.py): 33 localisations spread 0.5 cm, 0.2 cm and 0.7 deg.
LOCALIZATION_TOPIC = "/rtabmap/localization_pose"
TRACKER_POSE_TOPIC = "/tracker_pose"
TRACKED_MAP_TOPIC = "/map_tracked"  # pepin_bringup.relocalizer publishes it, latched
FIT_TOPIC = "/localization_fit"
ODOM_FRAME, BASE_FRAME = "odom", "base_link"
# Where the calibration files live, as the container sees them: the same /maps the map yaml and the
# graph database are served from, so what the pair measured travels with the pair.
ANCHOR_DIR = "/maps"
# ...and the database itself, read READ-ONLY and only for one thing: which SESSION each node id
# belongs to (pepin.graphnodes.read_sessions). A session is the piece the measurement found rigid.
DATABASE = "/maps/rtabmap.db"
# Who the map on /map is, as a room and not as a grid: the latched JSON pepin_bringup.depth_fusion
# publishes (pepin.worldmap.MapIdentity). The size@origin id of a growing volume changes whenever it
# grows a row — measured on the parked cart 2026-09-18, the served id went from 239x215@-18.53,-4.38
# to 280x250@-19.48,-5.48 for the same room at the same pose — so every file beside the map is named
# by the ROOM this names, and the size@origin id is only what a WORD carries for the board to check.
IDENTITY_TOPIC = "/map_identity"
# The board's own account of who is driving its tracker, which is what decides whether RTAB-Map may
# LEARN: the holder is read off it name-free (pepin.watch.source_words, Preflight.holding — the
# enabled source the cart has driven the least since its word), so no rule here spells "lidar" and a
# stereo matcher good to a few cm will teach the database the day it exists.
SOURCES_TOPIC = "/localization/sources"
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
# What makes the tracker's belief worth re-learning the anchor from: a lidar scan matched this
# well, this recently. /localization_fit carries only a fit a scan measured (relocalizer.py) and
# falls to 0.0 when no source has spoken, so a fresh 0.6 IS "the lidar is driving"; depth_fusion
# paints its band at 0.50 and this is the stricter half of that.
TRUSTED_FIT = 0.6
FIT_FRESH_S = 2.0
# What the board's tracker must have behind its pose for the graph's word to be a measurement
# and nothing more: a published fit at least this good and a belief no older than this. Below
# either, the tracker is holding a pose no source is confirming — dead reckoning, a lidar that
# matches nothing, a bridge that has gone quiet — and the graph's word is then not a correction
# to an almost-right pose but the only thing that knows where the cart is, which is what the
# candidate channel is for. 0.3 sits just above the tracker's own lost_below (0.25,
# pepin.localization.Localizer): a fit at that floor explains nothing.
TRACKER_TRUSTED_FIT = 0.3
# ...and the widest the tracker's own post-fusion sigma may be for its pose to stand in for a
# fit that was never measured (camera-only /localization_fit is 0.00 by construction). 0.30 m is
# above the graph word's own floor of 0.20 and under the half-width of the cart.
TRACKER_TRUSTED_SIGMA_M = 0.30
BELIEF_FRESH_S = 3.0
# How far apart in time the tracker's belief and the graph's own place may be for the two to be
# ONE PAIR of the tie's calibration: at the cart's 0.3 m/s half a second is 15 cm of frame error,
# and the tracker publishes at 10 Hz. The belief is carried over the EKF's odometry to the graph's
# own stamp first (:meth:`RtabmapFrame._predicted`), so this is the patience for a belief that has
# stopped arriving at all rather than for the tenth of a second between two topics.
ANCHOR_MAX_SKEW_S = 0.5

FLAGS = FlagSet(
    Flag(
        "slam",
        False,
        description="RTAB-Map is the map (online SLAM): its correction is map -> odom and goes to"
        " the board as a message on /map_odom, where pepin_bringup.slam_frame broadcasts it; off,"
        " the board's tracker owns map -> odom and this node broadcasts map -> rtabmap here",
        why="default by design, unmeasured: this says which edge is published — a mode, not a"
        " tunable — and the two modes are two different graphs of frames, which is also why it is"
        " not live. What the mode is worth was measured in the first session: from an empty"
        " database a room came up as a 341x341 map over 21 and then 55 graph nodes, a 1 m goal"
        " with a 90 degree turn landed within 2.8 cm and home within 6.6 cm after about 4 m of"
        " driving, one loop-closure hypothesis was rejected by the scan check (5 % against the 10"
        " % it needs) and none was accepted",
        on_when="in an unknown room, launched as one mode end to end (ros/thin.sh slam on the"
        " board, ros/laptop.sh vslam --slam): set at start, never mid-run",
        off_when="in every known-map mode, where the board's tracker owns map -> odom: the two"
        " publishers must never both run",
        live=False,
    ),
    Flag(
        "graph_odom",
        True,
        description="beside a known map, RTAB-Map is built on the EKF's odom -> base_link and"
        " this node ties its graph to the lidar map with an anchor learned once at the first"
        " graph (map -> rtabmap, constant); off, RTAB-Map's odometry is the tracker's own pose"
        " and map -> rtabmap carries the inverse of the correction, as before 2026-09-14."
        " Set by vslam.launch.py's argument of the same name: the two are one decision",
        why="ON, because the tracker's pose teleports when it relocalises and a graph cannot be"
        " built on odometry that jumps: a neighbour edge between two nodes one second apart"
        " carried 0.888 m against its 0.244 m sigma, a 3.64 error ratio over the 3.0 of"
        " RGBD/OptimizeMaxError, and on it RTAB-Map rejected EVERY loop closure it found over"
        " three hours on 2026-09-14 (5 at a time, each registered with 67 visual inliers against"
        " a Vis/MinInliers of 20). The EKF's odometry is continuous by construction and is what"
        " every other consumer already rides",
        on_when="always, beside a known map: this is what makes a closure possible at all",
        off_when="to reproduce the old graph, or to read a database recorded under it: the"
        " nodes of the two arrangements are in different frames, so the switch is a launch"
        " argument and never moves mid-session",
        live=False,
    ),
    Flag(
        "graph_measurement",
        True,
        description="beside a known map, publish where RTAB-Map's graph says the cart is as a"
        f' measurement on {MEASUREMENT_TOPIC} (source "graph") every time the graph moves, for'
        " the board's fusion to weigh like any other word; off, the graph's answer stays on this"
        " laptop and nothing reaches the pose",
        why="on since 2026-09-14 16:20: with the lidar driving (sources=lidar,graph, tapes"
        " 0275/0276)"
        " the tracker took 5 of 27 words and sat 0.7-0.8 cm from the lidar truth, and at rest the"
        " word"
        " stays 0-8 cm from the tracker; the word is what a lidar-less cart localises on (test C)."
        " Before: OFF, because on the stack as it stands the correction never moves at all: over 3"
        " h"
        " on 2026-09-14 every closure RTAB-Map found was thrown away by RGBD/OptimizeMaxError"
        " (5 links an iteration, rejected on a NEIGHBOUR edge 28042->28043 whose residual is"
        " 0.888 m against a 0.244 m sigma, ratio 3.64 over the 3.0 the parameter allows), so"
        " there is not yet one accepted correction to judge this word on. It is also OFF because"
        ' nothing on the board subscribes yet: the roster has no "graph" source and the'
        " measurement gate fuses by name",
        on_when="once a closure is accepted AND the board has a gate for it: then on, with"
        " /localization/graph_measurement recorded beside /tracker_pose for a drive, to see what"
        " the graph would have done to the pose before it is allowed to do it",
        off_when="whenever the graph's own frame may have started anywhere but the tracker's"
        " truth -- a session begun while the cart was lost puts every graph word off by that"
        " offset, since the two frames are only tied at the start pose",
    ),
    Flag(
        "graph_candidates",
        True,
        description="a graph word the board's fusion cannot act on goes out as a whole-map"
        f' CANDIDATE on {CANDIDATE_TOPIC} (source "graph", the same covariance floor, at most one'
        " per graph message): a word refused as disagreeing with the tracker's belief past the"
        f" fusion's own chi-square ({GATE:.2f}, 3 dof), and a word naming a DIFFERENT"
        " place while the tracker has no trusted source behind its pose (published fit below"
        f" {TRACKER_TRUSTED_FIT:.1f}, or no belief for {BELIEF_FRESH_S:.0f} s). Off, such a word"
        " is counted here and reaches nothing",
        why="on, because without it the graph cannot undo a carry at all — by construction, not"
        " by measurement: a measurement past the disagreement gate is refused in this"
        " node, and one that passes is gated again on the board by the information filter's"
        " chi-square (the same 11.34), so the word that is RIGHT after the cart is carried — the"
        " one the graph produces the moment it recognises the place — is exactly the word both"
        " gates throw away (2026-09-14: the carry test could not work by construction). The"
        " candidate channel is the door the lidar's own whole-map search uses for this, and the"
        " board guards it with the same rules for every source: three agreeing candidates from"
        " one sensor, no re-seed while a goal runs, and no re-seed at all from a source that is"
        " not the lidar while the lidar is alive",
        on_when="always on a known map, and above all in a carry test: it is the graph's only"
        " path to a pose that is not merely inaccurate but in the wrong room",
        off_when="whenever the tie may be wrong and the lidar cannot say so — a database read"
        " beside another map, a tie fitted from one corner of the flat: a wrong tie puts"
        " every word off by one constant offset, and three constant offsets agree with each"
        " other perfectly. Off, too, if a graph candidate is seen breaking the lidar's own"
        " re-seed streak (the board's gate holds one run, and candidates of two sources"
        " alternating end each other's)",
    ),
    Flag(
        "graph_trust",
        "agreement",
        choices=("agreement", "distance", "flat"),
        description="what the graph's word carries as its fit. agreement: how well the last"
        f" {AGREEMENT_WORDS} words inside {AGREEMENT_WINDOW_S:.0f} s track the tracker's pose"
        " carried forward by odometry between them, exp(-rms residual /"
        f" {AGREEMENT_SCALE:.2f} joint sigmas, position AND heading), times the recognition"
        " predicate"
        " (graph_words_need_recognition). distance: 1.0 the moment a loop closure or a proximity"
        " link ties the present to an older node, decaying as exp(-d / graph_trust_m) with the"
        " metres driven since, as between 2026-09-14 and 2026-09-15. flat: every word claims 1.0,"
        f" as before 2026-09-14. Never above {FILE_ANCHOR_TRUST:.1f} in either measured mode while"
        " the anchor came from a file and nothing has been recognised yet (pepin.graphtrust, fed"
        f" from {INFO_TOPIC})",
        why="agreement, because distance-since-the-last-tie cannot see a word that is simply in"
        " the wrong frame: on tape 0333 (2026-09-15, camera only) RTAB-Map had recognised nothing"
        " against the loaded database since its last start, its new nodes formed an unlinked"
        " segment, and ties INSIDE that segment kept resetting the decay clock — so words about a"
        " metre and 150 deg out (the last 103 cm from the tracker, 102 more refused as too far)"
        " rode with a high fit and the tracker took 19 of them. The residual of those same words"
        " against the odometry-propagated belief is the metre itself, which is exp(-10) on the"
        " 10 cm scale: a working word sits 0.7-0.8 cm from the lidar truth while driving, 0-8 cm"
        " at rest and 2.2-3.2 cm over a printer errand (tapes 0275/0276, 2026-09-14), so the"
        " scale is several times the working spread and a tenth of the failure. Nothing here"
        " touches the covariance: the word is still worth the remote floor geometrically, it"
        " simply stops vouching for itself",
        on_when="agreement always beside a known map — it is what lets a wrong word be SEEN as"
        " wrong by the gates that already exist, within one word instead of one closure",
        off_when="distance for a CARRY test, where the tracker's belief is wrong on purpose: the"
        " residual then measures the tracker's error and not the graph's, and the word that"
        " undoes the carry is the one that disagrees most (the node stops feeding the window"
        " while the tracker has no source behind its pose, which covers the usual case, but a"
        " carry that keeps a confident wrong pose is exactly the case it cannot cover). flat to"
        " reproduce a tape recorded before 2026-09-14",
    ),
    Flag(
        "graph_words_need_recognition",
        True,
        description="a graph word — measurement or candidate — is published only while RTAB-Map"
        " is RECOGNISED: it has accepted a loop closure, a proximity link or a localisation"
        " against the database it LOADED since its own start (the matched node id is older than"
        " the first node this start built). That tie never expires — the distance driven since"
        " it rides the word's covariance. Untied, the node stays silent, counts the words, and"
        " does not re-learn the anchor either; off, every word is published as before 2026-09-15",
        why="on, because an unlinked segment is the one failure a graph cannot report itself."
        " Tape 0333, 2026-09-15: RTAB-Map had restarted many times that day and had recognised"
        " nothing since its last start, so its current nodes were placed by this session's"
        " odometry alone — the anchor on file (-10.33, +1.41, +53.3 deg) was still right for the"
        " LINKED nodes it was measured on, the correction read (-0.44, -1.94, -151.8 deg), and"
        " the words were odometry dressed as a pose: 19 of them taken, the last 103 cm from the"
        " tracker with 102 refused as too far, and the pose flew. The predicate is the matched id"
        " against this start's first node id, which is what tells a tie to the map on disk from a"
        " tie inside the new segment. A tie NEVER EXPIRES: it fixes the frame, and how far the cart"
        " has driven since rides the word's covariance instead. The expiry used to be a clock of"
        " driving seconds, and it charged VO jitter at a standstill — 991 s at a bookshelf with"
        " 587 words withheld, 2026-09-17, twice before that in two days",
        on_when="always beside a known map, and above all after any RTAB-Map restart: it is what"
        " keeps a database that has not found itself yet from moving the pose",
        off_when="to reproduce a tape recorded before 2026-09-15, or in a room where RTAB-Map"
        " cannot close a loop at all and the word is wanted anyway — then read the trust in the"
        " report line before believing anything the graph says",
    ),
    Flag(
        "graph_word_from_localization",
        True,
        description="the word carries RTAB-Map's OWN localisation in the database it loaded"
        " (/rtabmap/localization_pose, frame 'rtabmap', with the covariance RTAB-Map measured),"
        " spent once per localisation. Off, the old arrangement: the session's correction from"
        " /rtabmap/mapGraph composed with odom -> base_link",
        why="ON since 2026-09-18, measured: the correction is relative"
        " to the CURRENT SESSION and between closures"
        " it is odometry from that session's start. On 2026-09-17 42.5 m of it had accumulated,"
        " the word stood 3.6 m from the tracker, and camera-only — where the graph is the only"
        " source — the cart drove at a person. A localisation exists only while RTAB-Map is tied"
        " to the loaded database and carries no such integral: measured parked the same evening"
        " it repeated to 0.5 cm over 33 samples (scratch/rtabmap_frame_drift.py), it sat 4-6 cm"
        " from the lidar-held tracker with its own sigma of 0.36 m, and this is also the only"
        " source that names the NODE it recognised, which is what graph_word_from_nodes hangs on."
        " With it on, the word's sigma never takes the distance RTAB-Map's counter has crept"
        " either: a localisation carries no odometry, and that counter creeping at a standstill is"
        " what grew the published sigma from 0.38 to 0.71 m on a PARKED cart and had the drive"
        " refused (2026-09-17)",
        on_when="always beside a known map: it is the only source here that is a localisation"
        " rather than an integration",
        off_when="in SLAM, where there is no loaded database to localise in, and to replay a tape"
        " recorded before 2026-09-17",
    ),
    Flag(
        "graph_trust_m",
        GRAPH_TRUST_M,
        description="how far the cart may drive past the graph's last tie to an older node"
        " before the word's fit falls to 1/e of it (the decay length of graph_trust), in metres",
        why="5 m is where the drift of the odometry the graph rides stops being centimetres:"
        " 0.79 m and 31 deg over 25 m and 22 cm over 12 m on 2026-09-14. It is also what puts a"
        " carried cart's word under the gates downstream within a few metres — the whole-map"
        " candidate floor (pepin.watch.ADMIT_FIT, 0.45) at 4.0 m past the tie, and 0.30 at 6.0 m",
        on_when="raise it in a large flat where the graph closes a loop only once a lap and the"
        " word is being refused between laps while the lidar agrees with it",
        off_when="lower it wherever the graph's odometry is worse than this one's — a slipping"
        " floor, a heavier cart — so the word stops claiming what its drift cannot hold",
        range=(0.1, 100.0),
    ),
    Flag(
        "graph_tie_from_pairs",
        True,
        description="the tie map <- rtabmap is a CALIBRATION fitted over a log of (lidar-held"
        f" tracker, RTAB-Map localisation) pairs kept beside the map (<map id>{PAIRS_SUFFIX}):"
        " every sharp seating appends one pair, the whole log is refitted whenever it grows, and"
        " the new fit replaces the tie in hand only when its covariance is smaller. Off, the old"
        " arrangement of before 2026-09-18: the tie is the one-seating anchor read from"
        " <map id>.graph_anchor.json, or learned from the first sharp seating and written there",
        why="on, because a tie fitted to ONE seating carries that seating's lever arm into every"
        " word: RTAB-Map's heading is good to 4-6 deg at one place, which is 0.4 m four metres"
        " away, and the stored anchor took three values metres apart in one evening ((-9.22,"
        " -0.40, -127 deg) -> (-6.66, +2.49, -117) -> (-6.18, +4.34, -113)). Both sides of the tie"
        " are FILES and a transform between two files cannot move: measured parked 2026-09-17,"
        " RTAB-Map's own localisation in the loaded database repeated to 0.5 cm over 33 samples"
        " (scratch/rtabmap_frame_drift.py), and across a restart the implied tie came back within"
        " the recognition's own heading noise. Fitted over 238 pairs from 19 tapes offline"
        " (scratch/graph_tie_fit.py) the tie is rigid to 3.6-5.9 cm rms inside one RTAB-Map session"
        " and repeats to 7 cm and 1.9 deg across nine drives on two evenings — while sessions of"
        " the same database sit in frames 1.6 m and 129 deg apart, which is exactly what the fit's"
        " outlier share (56 % over the whole of that database) is there to report",
        on_when="always beside a known map: it is what makes the tie lidar-independent at runtime"
        " (nothing is fitted while the lidar is silent, so camera-only uses the tie as it is)",
        off_when="to reproduce a session recorded before 2026-09-18, or beside a database for"
        " which no pairs can be taken at all and the one-seating anchor is all there is",
        live=False,
    ),
    Flag(
        "graph_word_from_nodes",
        True,
        description="the word is hung on the DATABASE NODE RTAB-Map recognised, not on a global"
        f" tie: for node N the table <room>{NODES_SUFFIX} holds where OUR map says the cart was"
        " when N was created, and the word is Table[N] . inverse(P_N) . P_current with both"
        " graph poses read from the SAME live /rtabmap/mapGraph — a relative pose between two"
        " nodes of one piece, so however the optimiser places the pieces it cancels. Off, the"
        " global tie (graph_tie_from_pairs) as before 2026-09-18",
        why="on, because ONE tie cannot serve this database and that is measured. The pieces of"
        " ros/maps/rtabmap.db are each rigid against our map to 3.6-5.9 cm rms but sit in different"
        " frames — sessions 1/38 at +52.1 deg, 0/26 1.6 m away, 31/32 129 deg away, 57 % of the"
        " pairs outside the best single fit's own gate (scratch/graph_tie_fit.py) — and the live"
        " frame is neither: the ties implied live on 2026-09-17 were ~80 deg from the offline fit,"
        " and the stored anchor moved from (-9.17, -0.29, -147.5 deg) to (-11.10, +5.71, -27.8 deg)"
        " across the 2026-09-16 prune. So the 90-degrees-wrong word at home was a TRUE recognition"
        " of a node the optimised graph misplaces. Replayed over the 171 closure links of that"
        " database whose both ends our map knows (scratch/graph_node_word_replay.py) the node word"
        " lands 4.3 cm p50 / 10.7 cm p90 / 43.5 cm worst and 0.8/2.8/7.0 deg from the lidar's"
        " truth, while the global tie on the very same recognitions is 6.7 cm p50 but 160.5 cm p90,"
        " 361 cm worst and up to 130 deg — the tail is the piece it was not fitted to",
        on_when="always beside a known map with a node table, and above all camera-only: it is the"
        " only arrangement that survives RTAB-Map re-optimising its graph",
        off_when="on the first day beside a NEW database, where the table is still empty and the"
        " global tie is all there is (the node falls back to it by itself and says so), or to"
        " reproduce a session recorded before 2026-09-18",
    ),
    Flag(
        "graph_memory",
        BY_TRUST,
        choices=(BY_TRUST, ALWAYS_MAP, ALWAYS_LOCALISE),
        description="who decides whether RTAB-Map's database may LEARN beside a known map."
        " trust: this node switches it live on the rule 'a sharp pose that does not come from the"
        " database itself' — the tracker's seating under anchor_max_sigma_m / anchor_max_sigma_deg,"
        " and a holder on /localization/sources that is not the graph — calling"
        f" {MAPPING_SERVICE} / {LOCALISATION_SERVICE} on a change of verdict that has held for"
        " the seating's own freshness window, and carrying RGBD/LinearUpdate / RGBD/AngularUpdate"
        " with it. map: always mapping. localise: always localising, whatever the pose is worth",
        why="trust, because both alternatives were measured and both are wrong. ALWAYS MAPPING is"
        " what ran until 2026-09-18: the database grew a new session every launch, the sessions sit"
        " 1.6 m and 129 deg apart (scratch/graph_tie_fit.py), RTAB-Map then rejected its own"
        " correct"
        " recognitions on RGBD/OptimizeMaxError (hypothesis 0.978 with 328 visual inliers, error"
        " ratio 5.03 against 3.0), and parked it kept a node a second — 250 junk nodes in one"
        " evening. ALWAYS LOCALISING can never learn a new room, and never covers the 43 sessions"
        " of"
        " this database no lidar-held drive has visited. The rule is the same one the volume's"
        " painting follows: teach only from a pose worth teaching from, and never from the pupil —"
        " a mono camera-only pose held by graph words sits at a sigma around 20 cm and fails the"
        " seating test by itself, with nothing naming it, while a lidar-held seating passes at"
        " 1-2 cm",
        on_when="trust always, beside a known map: it is what lets one launch both wake up in a"
        " known room and extend the map when the lidar is there to teach it",
        off_when="map while deliberately extending a database by hand with the lidar known good;"
        " localise to freeze a database completely (a measurement of what the table alone is"
        " worth, or a session where the file must not change)",
    ),
    Flag(
        "graph_rel_from_registration",
        False,
        description="the relative pose a word is built on comes from RTAB-Map's OWN raw"
        " registration of this update (Info.loop_closure_transform, flipped if that is the way"
        " the graph poses agree with); off, it is the difference of the matched node's and the"
        " current pose in the live graph. Both are computed either way and their disagreement is"
        " logged once per recognised node",
        why="OFF until one live run reads that log line. The raw registration is better in"
        " principle — after a localisation link RTAB-Map's own current pose is an optimised"
        " compromise with its odometry cache, which an earlier localisation on a bad piece"
        " can pull, while a raw transform between two frames cannot — but the field's direction"
        " convention is not documented in the message and the installed image carries no rtabmap"
        " sources to check it against, so turning it on blind would risk composing a transform"
        " backwards. The difference of two graph poses is what HAS been measured: over the 171"
        " closure links of this database whose both ends our map knows it lands 4.3 cm p50 /"
        " 10.7 cm p90 from the lidar's truth (scratch/graph_node_word_replay.py), against 16.8 cm"
        " p50 for the inverted convention — so the question is answerable, and this flag is"
        " what answers it without a code change",
        on_when="once a drive's log shows Info's registration agreeing with the graph-pose"
        " difference to centimetres in the SAME direction: then on, because it is the one that"
        " cannot be pulled by the optimiser",
        off_when="whenever that log line shows the two disagreeing by more than the registration's"
        " own sigma, or a build whose Info does not carry the transform at all",
    ),
    Flag(
        "fresh_frame",
        False,
        description="the map and the graph database were BORN TOGETHER in this session — the same"
        " pose at the same moment — so the tie is identity with no uncertainty and no pairs are"
        " needed. Off, the tie is measured (graph_tie_from_pairs) or read from disk",
        why="default by design, unmeasured: this says which of two situations the robot is in, not"
        " how well anything works. Waking up in an unknown room, the map is created at the cart's"
        " current pose and RTAB-Map's database is created at the same pose in the same second, so"
        " map and rtabmap are the same frame BY CONSTRUCTION and a measurement of the identity"
        " transform could only add noise to it. Beside a map and a database that have any history"
        " together the two frames are unrelated and the tie has to be measured",
        on_when="a wake-up in an unknown room, where this session creates both the map and the"
        " database: set by the launch that creates them, never mid-run",
        off_when="beside any map or database that existed before this session — including the same"
        " room re-entered after a restart, where the database is loaded and not born",
        live=False,
    ),
    Flag(
        "anchor_max_sigma_m",
        ANCHOR_MAX_SIGMA_M,
        range=(0.0, 1.0),
        description="the widest the tracker's own error bar may be, metres per position axis"
        " (the roots of the covariance /tracker_pose carries, which is the lidar's score peak),"
        " for that seating to become one PAIR of the tie's calibration; a softer seating is"
        " refused and the tie keeps whatever it had. 1.0 lets"
        " anything through, which is the behaviour of before 2026-09-14",
        why="0.03, because a fit is not an error bar: at home (the charger, along a sofa) the"
        " lidar's seatings spread up to 55 cm in y within minutes at fit 0.67-0.79 — the scan is"
        " pinned in one axis there — and the anchor learned from one of those carried that error"
        " into every graph word (the word then sat 23-30 cm from a SHARP lidar seating over tape"
        " 0296, scratch/graph_vs_lidar_sharp.py). 3 cm is where the gate starts to be a gate:"
        " over tapes 0293-0298 the worse of the two position sigmas has a median of 1.50 cm and"
        " a p90 of 3.18 cm, so this refuses the worst 11 % of seatings, while 1 cm would refuse"
        " 79 % and the anchor would never be learned at all",
        on_when="always: the tie is a constant of the pair and every pair it is fitted over is"
        " baked into the words the graph says",
        off_when="raise it (to 1.0) only to reproduce a session recorded before the gate, or to"
        " get a tie at all in a room where no seating is ever sharp — and then read the"
        " sigmas the report line prints beside the tie before believing a word",
    ),
    Flag(
        "anchor_max_sigma_deg",
        ANCHOR_MAX_SIGMA_DEG,
        range=(0.0, 180.0),
        description="the same gate for heading, degrees: a pair is taken only off a"
        " seating whose heading sigma is at most this",
        why="1.0, because a heading error rotates the whole graph about the cart: over tapes"
        " 0293-0298 the lidar's heading sigma at a sharp seating is 0.06-1.14 deg (median 0.4),"
        " so this is the loose end of what the peak reports when it is pinned at all, and one"
        " degree over the 4 m of the flat is 7 cm at the far wall",
        on_when="always, with anchor_max_sigma_m: a seating sharp in x and y and free in heading"
        " is a cart that knows where it stands and not which way it faces",
        off_when="raise it only with anchor_max_sigma_m, and for the same reasons",
    ),
)


def _zero_stamp() -> Any:
    """A builtin_interfaces/Time at zero, for a message that carries no header at all."""
    from builtin_interfaces.msg import Time as TimeMsg

    return TimeMsg()


def _registration_of(msg: Info) -> Pose2D | None:
    """RTAB-Map's OWN raw registration of this update in the plane
    (``Info.loop_closure_transform``), or ``None`` when the message carries none.

    An all-zero transform is "not filled": a registration between two frames that are exactly one
    point apart in every axis is not a measurement, it is an empty field. Read through ``getattr``
    because a build whose Info does not have it must read as a message that offered nothing.
    """
    transform = getattr(msg, "loop_closure_transform", None)
    translation = getattr(transform, "translation", None)
    rotation = getattr(transform, "rotation", None)
    if translation is None or rotation is None:
        return None
    x, y = float(translation.x), float(translation.y)
    qz, qw = float(rotation.z), float(rotation.w)
    if (x, y, qz, qw) == (0.0, 0.0, 0.0, 0.0):
        return None
    return Pose2D(x, y, math.atan2(2.0 * qw * qz, 1.0 - 2.0 * qz * qz))


def _info_ids(msg: Info) -> InfoIds:
    """The graph ids of one ``/rtabmap/info`` (:class:`pepin.graphtrust.InfoIds`): the node
    RTAB-Map is building now, the nodes a loop closure and a proximity link matched, and whether
    the message carried a localisation pose — a pose with a covariance on it, which RTAB-Map
    fills only when it has placed itself on the loaded database. Every field is read through
    ``getattr``: a build whose Info does not carry one reads as a message that matched nothing,
    and the statistics still answer for the closures."""
    localization = getattr(msg, "localization_pose", None)
    covariance = getattr(getattr(localization, "pose", None), "covariance", ())
    return InfoIds(
        ref_id=int(getattr(msg, "ref_id", 0) or 0),
        loop_closure_id=int(getattr(msg, "loop_closure_id", 0) or 0),
        proximity_detection_id=int(getattr(msg, "proximity_detection_id", 0) or 0),
        localized=any(float(value) > 0.0 for value in covariance),
    )


class RtabmapFrame(Node):
    """Broadcasts map -> rtabmap, or sends map -> odom to the board, from RTAB-Map's correction."""

    def __init__(self) -> None:
        super().__init__("rtabmap_frame")
        # Declared before the switches: rclpy runs the flags' callback on every declaration.
        self._anchor_dir = Path(str(self.declare_parameter("anchor_dir", ANCHOR_DIR).value))
        self._database = Path(str(self.declare_parameter("database", DATABASE).value))
        # The ROOM every file beside the map is named by, when the launch knows it. Empty is the
        # honest default: the node then takes the room from /map_identity, and falls back on the
        # served grid's size@origin id, which is what the files were named by before 2026-09-18.
        self._room = str(self.declare_parameter("room", "").value).strip()
        self._switches = Switches(self, FLAGS)
        self._slam = self._switches.on("slam")
        self._graph_odom = self._switches.on("graph_odom")
        self._pose = RigidPose(np.eye(3), np.zeros(3))
        self._graphs = 0
        self._correction2d = Pose2D()  # the same correction in the plane, as the fusion reads it
        self._belief: Pose2D | None = None  # what the board's tracker says, and when
        self._belief_stamp = 0.0
        # ...and how sharply it says it: the roots of the covariance diagonal (m, m, rad), which
        # with covariance=peak is the lidar score peak's own width per axis — the one thing that
        # tells a seating pinned in both axes from one free to slide along a sofa.
        self._belief_sigma: tuple[float, float, float] | None = None
        self._belief_at = -math.inf  # ...and when that belief reached this node, by our clock
        self._map_id = ""  # the map that belief is on; the board refuses a word about another
        # map <- rtabmap, with its error bar and its provenance: identity for a frame born with
        # this map, the fit over the pairs log, or the old one-seating file until pairs exist.
        self._tie: Tie | None = identity_tie() if self._switches.on("fresh_frame") else None
        self._pairs: list[TiePair] = []  # the calibration's log as this node has it in memory
        self._pairs_taken = 0  # ...how many of them this session appended
        self._key = ""  # what the files beside the map are named by: the room, or the legacy id
        # Where OUR map says each of the database's nodes is: the frame-free answer, which is what
        # every word is hung on (pepin.graphnodes). The sessions come from the database file.
        self._table = NodeTable()
        self._tabled = 0  # nodes this session measured and appended...
        self._from_matched = 0  # ...of which this many off a node RTAB-Map RECOGNISED
        self._last_tabled: Pose2D | None = None  # where the last entry was measured, to not repeat
        # A node this run created, measured and waiting for the live graph to show the database kept
        # it: that wait is how mapping mode is told from localisation mode.
        self._pending_node: NodePose | None = None
        self._mapping: bool | None = None  # ...and the answer, once one of them has landed or not
        self._temporary = 0  # measured nodes the database did not keep (localisation mode)
        # RTAB-Map's own RAW registration of this update (Info.loop_closure_transform), kept beside
        # the difference of two graph poses until one live run says which way it points.
        self._registration: Pose2D | None = None
        self._registration_said = 0  # ...and the node the last comparison was logged for
        # Whether the database may LEARN, decided by trust in the pose and not by a sensor's name
        # (pepin.graphnodes.ModeRule). The hold is the seating's own freshness window: a verdict is
        # acted on once it has survived as long as the evidence it rests on takes to refresh.
        self._mode = ModeRule(FIT_FRESH_S, str(self._switches["graph_memory"]), GRAPH)
        self._holder: str | None = None  # who /localization/sources says is holding the pose
        self._holder_at = -math.inf  # ...and when that report arrived, by our clock
        self._mode_pending: Any = None  # a switch the service has not answered yet
        self._mode_failed = 0  # switches the service refused or never answered
        self._node_words = 0  # words hung on a node...
        self._substituted = 0  # ...of which this many on a neighbour of the matched node
        self._unplaceable = 0  # ...and localisations whose session nothing on our map knows
        self._unplaceable_reason = ""  # ...with the last reason, said once per reason
        self._unnamed = 0  # localisations no /rtabmap/info named a matched node for
        self._hung: NodeWord | None = None  # the last word's own node, for the report line
        self._ref = 0  # the node RTAB-Map is building now, from /rtabmap/info
        self._graph_poses: dict[int, Pose2D] = {}  # the live graph's own poses, by node id
        # The node the last /rtabmap/info said RTAB-Map matched, and that message's stamp: a
        # localisation and the info of the SAME update carry the same stamp, which is what names
        # the node a localisation was made against.
        self._matched: tuple[int, float] | None = None
        self._anchor_sigma: tuple[float, float, float] | None = None  # the last pair's seating
        self._pending = 0  # graphs that found no seating sharp enough to take a pair from
        self._pending_reason = ""  # ...and why the last of them was refused, said once per reason
        self._fit, self._fit_at = 0.0, -math.inf  # the lidar's own fit, and when it last spoke
        self._sigma_xy: float | None = None  # the tracker's own post-fusion spread, where it says
        self._word: Pose2D | None = None  # the last place the graph put the cart, on the map
        self._place: Pose2D | None = None  # ...and where that was in the graph's own frame
        self._gap_m = 0.0  # ...and how far that was from the tracker's own belief
        self._gap_sigmas = 0.0  # ...in metres, and in sigmas of the two covariances together
        self._sent = 0  # graph measurements published
        self._refused = 0  # ...and words the 3-DOF gate refused as another pose entirely
        self._blind = 0  # ...and graphs with no odom -> base_link to compose with
        # RTAB-Map's own localisation in the database it loaded: the place, the moment and the
        # sigma IT measured, kept until one word has been made of it. One localisation, one word:
        # nothing else may become a word, so odometry between localisations cannot reach one.
        self._localized: tuple[Pose2D, float, float] | None = None
        # ...and the sigma of the one the last word was made of, kept for that word's covariance.
        self._localized_sigma_m: float | None = None
        self._localized_stamp = 0.0  # ...the stamp it carried, which names its matched node...
        self._localized_cov: Matrix | None = None  # ...and the planar 3x3 RTAB-Map measured for it
        self._localizations = 0  # how many have arrived
        self._unlocalized = 0  # ...and graphs that had none left to spend
        self._proposed = 0  # ...and words offered as whole-map candidates instead
        self._proposed_at = -1  # the graph whose word became one: at most one candidate each
        # What the graph has RECOGNISED, which is what its word is worth: the clock the
        # statistics feed, and the trust it answers with (pepin.graphtrust).
        self._trust = GraphTrust(float(self._switches["graph_trust_m"]))
        # ...and whether it has recognised the database it LOADED at all since its own start,
        # which is what says its nodes are a map and not an unlinked segment of odometry.
        self._recognition = Recognition()
        self._withheld = 0  # words not published because it had not
        # How well the words that DO ride track the tracker's odometry-propagated pose: the
        # window the fit is made of in the agreement mode.
        self._agreement = Agreement()
        self._belief_odom: Pose2D | None = None  # the odometry at the belief's own moment...
        self._word_odom: Pose2D | None = None  # ...and at the word's, to carry the belief over
        self._infos = 0  # /rtabmap/info messages consumed
        self._lookup = TfLookup(self)  # the EKF's odom -> base_link, at the graph's own stamp
        self._tf = None if self._slam else TransformBroadcaster(self)
        self._correction = (
            self.create_publisher(TransformStamped, CORRECTION_TOPIC, 5) if self._slam else None
        )
        self._measurement = (
            None
            if self._slam
            else self.create_publisher(
                String, MEASUREMENT_TOPIC, bridged_qos_profile(MEASUREMENT_TOPIC)
            )
        )
        # Depth 1, RELIABLE: the QoS both ends of this channel already ask for
        # (pepin_bringup.laptop_localizer's publisher, pepin_bringup.relocalizer's subscription).
        # A candidate is a snapshot of a moment and only the newest is worth judging, and two
        # publishers on one bridged topic must declare one QoS or the route's is decided by a
        # race (pepin.deployment.BRIDGED_QOS' reason, measured on /imu/data_raw 2026-09-13).
        self._candidate = (
            None
            if self._slam
            else self.create_publisher(
                String,
                CANDIDATE_TOPIC,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )
        )
        self.create_subscription(MapGraph, "/rtabmap/mapGraph", self._on_graph, 5)
        self.create_subscription(
            PoseWithCovarianceStamped, LOCALIZATION_TOPIC, self._on_localization, 5
        )
        self.create_subscription(Info, INFO_TOPIC, self._on_info, 5)
        # The map the word is about, named the way the board names it (size@origin, msgs.map_id):
        # the tracker refuses a word about another map, and a frame id is not a map id.
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # The map the tracker is ON, republished by the board itself (relocalizer's
        # TRACKED_MAP_TOPIC): its id is by construction the id the board's gates expect on a
        # word, whichever source the tracker adopted and however the volume has grown. /map is
        # still heard because a served file (SLAM modes, the first boot of a room) speaks there
        # and an older board publishes nothing else.
        self.create_subscription(OccupancyGridMsg, TRACKED_MAP_TOPIC, self._on_map, latched)
        self.create_subscription(OccupancyGridMsg, "/map", self._on_map, latched)
        self.create_subscription(String, IDENTITY_TOPIC, self._on_identity, latched)
        self.create_subscription(
            PoseWithCovarianceStamped, TRACKER_POSE_TOPIC, self._on_tracker_pose, 5
        )
        self.create_subscription(Float32, FIT_TOPIC, self._on_fit, 5)
        self.create_subscription(String, SIGMA_TOPIC, self._on_sigma, 5)
        self.create_subscription(String, SOURCES_TOPIC, self._on_sources, 5)
        # The mode services: this node owns the switch beside a known map, and touches neither in
        # SLAM (where the database IS the map being built) nor with graph_odom off.
        self._modes = (
            {}
            if self._slam or not self._graph_odom
            else {
                True: self.create_client(Empty, MAPPING_SERVICE),
                False: self.create_client(Empty, LOCALISATION_SERVICE),
            }
        )
        self._tuner = (
            None
            if self._slam or not self._graph_odom
            else self.create_client(SetParameters, f"{RTABMAP_NODE}/set_parameters")
        )
        self._reread = (
            None
            if self._slam or not self._graph_odom
            else self.create_client(Empty, f"{RTABMAP_NODE}/update_parameters")
        )
        self.create_timer(1.0 / RATE_HZ, self._publish)
        where = (
            f"map -> odom on {CORRECTION_TOPIC}, for the board"
            if self._slam
            else "map -> rtabmap on /tf, here"
            + (" (the tie, once the served map names its pairs log)" if self._graph_odom else "")
        )
        self.get_logger().info(
            f"rtabmap frame up: {where}, from /rtabmap/mapGraph"
            f" (identity until the first graph); flags: {self._switches.state(live_only=False)}"
        )
        self.create_timer(30.0, self._report)

    def _report(self) -> None:
        """Every 30 s: how many graphs arrived, how many became measurements, how many went out
        as whole-map candidates instead, how many the 3-DOF gate refused and how many found no
        odometry to compose with; then the TIE — where it points, where it came from (a fit over N
        pairs with M inliers over X metres, the one-seating file, or identity for a frame born with
        this map), what it is worth WHERE THE CART IS NOW, and the share of pairs the fit had to
        throw away, which is this database's measured false-recognition rate; then the last word's
        gap to the tracker in metres and in sigmas, what the graph's word is worth and why
        (:class:`pepin.graphtrust.GraphReport`) and the switches."""
        tie = self._tie
        if tie is None:
            described = f"not yet, pending {self._pending}" + (
                f" ({self._pending_reason})" if self._pending_reason else ""
            )
        else:
            sigma_xy, sigma_yaw = tie.sigma_at(self._place or Pose2D())
            described = (
                f"{tie.described()}; here worth {sigma_xy * 100:.0f} cm,"
                f" {math.degrees(sigma_yaw):.1f} deg"
                + (
                    f", last pair off a seating of {describe_sigma(self._anchor_sigma)}"
                    if self._anchor_sigma is not None
                    else ""
                )
                + (f", {self._pairs_taken} pairs taken" if self._pairs_taken else "")
                + (f", pending {self._pending}" if self._pending else "")
            )
        word = (
            "none yet"
            if self._word is None
            else f"({self._word.x:+.2f}, {self._word.y:+.2f},"
            f" {math.degrees(self._word.theta):+.1f} deg), {self._gap_m * 100:.0f} cm"
            f" / {self._gap_sigmas:.1f} sigmas from the tracker"
        )
        trust = self._trust.report(self._tie is not None and self._tie.origin == "file")
        self.get_logger().info(
            f"rtabmap frame: {self._graphs} graphs, {self._sent} graph measurements sent,"
            f" {self._proposed} candidates sent,"
            f" {self._refused} refused by the gate, {self._blind} without odometry,"
            f" {self._unlocalized} graphs with no localisation to spend"
            f" ({self._localizations} localisations heard, last sigma "
            + ("none" if self._localized_sigma_m is None else f"{self._localized_sigma_m:.2f} m")
            + ");"
            f" {self._table_text()};"
            f" correction ({self._correction2d.x:+.2f}, {self._correction2d.y:+.2f},"
            f" {math.degrees(self._correction2d.theta):+.1f} deg), tie {described},"
            f" last word {word}; graph {trust.text()} over {self._infos} infos;"
            f" {self._recognition.report(self._withheld).text()};"
            f" word fit {self._trust_now():.2f} by {self._switches['graph_trust']}"
            f" ({self._agreement_text()});"
            f" flags: {self._switches.state(live_only=False)}"
        )

    def _table_text(self) -> str:
        """The node table for a report line: what it is named by, how much of the database it
        places, which node the last word hung on and how far from it, and what was refused —
        a localisation nothing named a node for, and one whose session this map does not know."""
        coverage = self._table.coverage()
        hung = self._hung
        seen = (
            "mode unseen"
            if self._mapping is None
            else ("mapping seen" if self._mapping else "localisation seen")
        )
        mode = f"{seen}, asked {self._mode.text()}" + (
            f", {self._mode_failed} switches the service could not take"
            if self._mode_failed
            else ""
        )
        return (
            f"table {self._key or 'unnamed'}: {len(self._table)} nodes over"
            f" {len(coverage)} sessions"
            + ("" if self._table.sessions_known else " (sessions unread: one run is one piece)")
            + f", {mode}"
            + (
                f", {self._tabled} measured here"
                f" ({self._from_matched} off recognised nodes,"
                f" {self._tabled - self._from_matched} off new ones)"
                if self._tabled
                else ""
            )
            + (
                f", {self._temporary} measured nodes the database did not keep"
                if self._temporary
                else ""
            )
            + f"; {self._node_words} words on a node"
            + (f" ({self._substituted} on a neighbour)" if self._substituted else "")
            + (f", last on {hung.text()}" if hung is not None else "")
            + (
                f"; {self._unplaceable} unplaceable ({self._unplaceable_reason})"
                if self._unplaceable
                else ""
            )
            + (f"; {self._unnamed} localisations named no node" if self._unnamed else "")
        )

    def _agreement_text(self) -> str:
        """How well the words that rode agree with the tracker: the rms residual of the window the
        last word was judged in, in metres and in sigmas, and how many words are in it — or why
        there is none."""
        residual = self._agreement.rms(self._now())
        sigmas = self._agreement.sigmas(self._now())
        if residual is None or sigmas is None:
            return "no residual: the tracker has no source" if self._lost() else "no residual yet"
        return (
            f"residual {residual * 100:.1f} cm / {sigmas:.2f} sigmas over"
            f" {self._agreement.count} words"
        )

    def _on_tracker_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """The board's belief: the pose RTAB-Map is fed as its odometry, the one a graph
        correction is applied to — and its error bar, which is what says whether the anchor may
        be learned from this seating at all. The EKF's odometry is read at the same moment, so
        that a belief can be carried forward to a word's stamp (:meth:`_predicted`)."""
        self._belief_odom = self._planar(self._lookup.transform(ODOM_FRAME, BASE_FRAME))
        position = msg.pose.pose.position
        self._belief = Pose2D(position.x, position.y, yaw_of(msg.pose.pose.orientation))
        self._belief_stamp = stamp_seconds(msg.header.stamp)
        covariance = list(msg.pose.covariance)
        self._belief_sigma = (
            math.sqrt(max(covariance[0], 0.0)),
            math.sqrt(max(covariance[7], 0.0)),
            math.sqrt(max(covariance[35], 0.0)),
        )
        # By OUR clock, not the message's: what this says is "the board is still talking to us",
        # and a stamp compared across two machines answers a different question.
        self._belief_at = self._now()

    def _on_info(self, msg: Info) -> None:
        """RTAB-Map's own statistics and graph ids: what the graph has RECOGNISED — a closure
        accepted, a proximity link added, the node each of them matched, the distance its
        odometry has travelled and how close the last hypothesis came. Both predicates the word
        rides on are made of these: the decay clock (:class:`pepin.graphtrust.GraphTrust`) and
        the recognition of the LOADED database (:class:`pepin.graphtrust.Recognition`). A tie is
        logged because it undoes accumulated drift, and a recognition because it is what lets
        this node speak at all.

        Two more things ride on this message, both about NODES rather than about confidence: the
        node RTAB-Map is building right now (``ref_id``), which is the one this map measures and
        tables (:meth:`_table_node`), and the node a closure or a proximity link MATCHED, which is
        the one a word is hung on (:meth:`_matched_node`). The stamp is kept with the matched id
        because that is what pairs this message with the localisation of the same update."""
        self._infos += 1
        ids = _info_ids(msg)
        stamp = stamp_seconds(getattr(getattr(msg, "header", None), "stamp", None) or _zero_stamp())
        matched = ids.loop_closure_id or ids.proximity_detection_id
        if matched > 0:
            self._matched = (matched, stamp)
            self._registration = _registration_of(msg)
        if ids.ref_id > 0 and ids.ref_id != self._ref:
            self._ref = ids.ref_id
            self._table_node(ids.ref_id, stamp)
        stats = dict(zip(msg.stats_keys, (float(v) for v in msg.stats_values), strict=False))
        was = self._recognition.recognised
        if self._recognition.update(stats, ids, self._now()) and not was:
            report = self._recognition.report(self._withheld)
            self.get_logger().info(
                f"graph recognised: RTAB-Map matched database node {report.matched_id} — its"
                f" nodes are on the map again and words travel ({self._withheld} withheld"
                f" since its start, {self._recognition.starts} restarts seen)"
            )
        if not self._trust.update(stats):
            return
        self.get_logger().info(
            f"graph tie {self._trust.ties}: RTAB-Map linked the present to an older node — the"
            f" word is worth 1.00 again (hypothesis {self._trust.report().hypothesis:.2f})"
        )

    def _table_node(self, node_id: int, stamp: float) -> None:
        """Measure where OUR map says the node RTAB-Map just created is, and hold it PENDING until
        the live graph shows the database kept it (:meth:`_commit_pending`).

        The pose is the tracker's belief carried over the EKF's odometry to that node's own moment
        (:meth:`_predicted`), and it is measured only off a seating the LIDAR holds and pins in both
        axes and in heading (:meth:`_unsharp`) — the same gate the pairs pass, for the same reason:
        whatever goes into the table is what every later word hung on this node will carry.

        THE PENDING IS THE MODE TEST. In localisation mode (``Mem/IncrementalMemory`` false, the
        default beside a known map since 2026-09-18) the ids this run hands out are TEMPORARY:
        nothing is written to the database, they never appear in the graph, and an entry under
        one of them is a row nothing can ever hang on — measured live, that is exactly what
        produced "session -1 has tabled nodes and the live graph carries none of them". A node the
        database keeps appears in the next graph, and only then is it taken.

        A FRESHLY BORN FRAME is the exception to the lidar gate and needs none: the map and the
        database were created at the same pose in the same second, so our map and the graph are one
        frame by construction and every node may be tabled off whatever belief exists.
        """
        self._commit_pending()  # a pending node the graph never carried is dropped here too
        if self._slam or not self._graph_odom or not self._key:
            return
        fresh = self._tie is not None and self._tie.origin == "identity"
        predicted = self._predicted()
        if predicted is None or abs(stamp - self._belief_stamp) > ANCHOR_MAX_SKEW_S:
            return
        if not fresh and self._unsharp() is not None:
            return
        if self._last_tabled is not None and same_place(self._last_tabled, predicted):
            return
        sigma = self._belief_sigma or (
            float(self._switches["anchor_max_sigma_m"]),
            float(self._switches["anchor_max_sigma_m"]),
            math.radians(float(self._switches["anchor_max_sigma_deg"])),
        )
        self._pending_node = NodePose(
            node_id=node_id,
            session=self._table.session_of(node_id),
            stamp=stamp,
            cart=predicted,
            sigma=sigma,
        )

    def _recognised(self) -> bool:
        """Whether RTAB-Map's present nodes are tied to the database it LOADED at all — a
        closure, a proximity link or a localisation against a node older than this start's
        first, at any point since this start.

        This is the ONE thing the node stays silent about, and it is not a confidence: without a
        tie the graph's poses are in a frame of their own and a word would be a number in another
        coordinate system. A tie that happened long ago still fixes the frame; how far the cart
        has driven since rides the word's covariance instead (:meth:`_word_sigma_m`).
        ``graph_words_need_recognition`` off publishes even then, which is the behaviour of
        before 2026-09-15."""
        if not self._switches.on("graph_words_need_recognition"):
            return True
        return self._recognition.recognised

    def _trust_now(self) -> float:
        """What the graph's word may claim as its ``fit`` right now, by ``graph_trust``:

        * ``agreement`` — how well the last words track the tracker's odometry-propagated pose
          (:class:`pepin.graphtrust.Agreement`), times the recognition predicate, so a word from
          a graph that has found nothing on its map claims nothing;
        * ``distance`` — the decay since the graph's last tie to an older node, whatever that
          node was (:class:`pepin.graphtrust.GraphTrust`, the behaviour of 2026-09-14);
        * ``flat`` — 1.0, which is what every word claimed before either existed.

        Both measured modes are capped at :data:`pepin.graphtrust.FILE_ANCHOR_TRUST` while the tie
        came from the old one-seating file and nothing has been recognised yet: that transform was
        measured in another session, from one place, and nothing has confirmed it.
        """
        mode = str(self._switches["graph_trust"])
        self._trust.trust_m = float(self._switches["graph_trust_m"])
        from_file = self._tie is not None and self._tie.origin == "file"
        if mode == "flat":
            return 1.0
        if mode == "distance":
            return self._trust.trust(from_file)
        # Agreement is a measurement that is NOT ALWAYS AVAILABLE: it needs a second opinion to
        # differ from, and camera-only there is none, so :meth:`Agreement.trust` honestly answers
        # 1.0 — "nothing here says this word is wrong". Read alone that is a word claiming
        # certainty it never earned, which is why this used to be multiplied by the recognition
        # predicate; and that predicate, running on a clock, withheld every word for three hours
        # at a bookshelf (2026-09-17). The base is now the graph's OWN evidence — the decay over
        # the distance driven since its last tie, which needs nobody — and agreement may only
        # make it WORSE, never better. One source, one covariance, measured on itself.
        trust = min(self._agreement.trust(self._now()), self._trust.trust(from_file))
        if from_file and self._recognition.matches == 0:
            trust = min(trust, FILE_ANCHOR_TRUST)
        return trust

    def _predicted(self) -> Pose2D | None:
        """Where the tracker's belief says the cart is at the moment of the word being judged:
        the last belief carried forward by the EKF's odometry between its own stamp and the
        graph's (the two odometries this node already looks up). ``None`` with no belief at all;
        the belief itself when one of the two odometries is missing — a graph word and a belief
        are a tenth of a second apart at 10 Hz, and 3 cm at the cart's speed."""
        belief = self._belief
        if belief is None:
            return None
        if self._belief_odom is None or self._word_odom is None:
            return belief
        return compose(belief, compose(inverse(self._belief_odom), self._word_odom))

    def _agree(self, word: RemoteMeasurement) -> None:
        """Feed one published word's disagreement with the odometry-propagated belief to the
        agreement window (:class:`pepin.graphtrust.Agreement`), which is what its ``fit`` is made
        of in the agreement mode: the distance in metres AND the 3-DOF Mahalanobis distance under
        the two covariances together, so a word facing the wrong way at the right place counts as
        the disagreement it is.

        A residual is only taken while the tracker has a source behind its pose (:meth:`_lost`
        false): a residual against a belief nobody is confirming — a carried cart, a lidar
        matching nothing — measures the TRACKER's error, and the graph's word is then the only
        opinion anyone has rather than the suspect one."""
        predicted = self._predicted()
        if predicted is None or self._lost():
            return
        self._agreement.add(
            self._now(),
            math.hypot(word.x - predicted.x, word.y - predicted.y),
            math.sqrt(max(self._disagreement(word, predicted), 0.0)),
        )

    def _believed(self, pose: Pose2D) -> PoseMeasurement:
        """The tracker's belief as a measurement, so it can be weighed against a word on one
        scale: the pose given, with the covariance ``/tracker_pose`` published for it, or the
        tracker's own published sigmas when that message carried none
        (:data:`pepin.fusion.SIGMA_XY_M` / :data:`pepin.fusion.SIGMA_YAW_DEG`)."""
        sigma = self._belief_sigma or (
            SIGMA_XY_M,
            SIGMA_XY_M,
            math.radians(SIGMA_YAW_DEG),
        )
        covariance = np.diag([max(s, 0.0) ** 2 for s in sigma])
        return PoseMeasurement(
            pose.x, pose.y, pose.theta, covariance, "tracker", self._belief_stamp, self._fit
        )

    def _disagreement(self, word: RemoteMeasurement, predicted: Pose2D) -> float:
        """How far this word is from the odometry-carried belief, as the squared Mahalanobis
        distance under the SUM of their covariances (:func:`pepin.fusion.disagreement`): the very
        number the board's information filter will gate the word on next, and the one this node
        refuses a word by. 3-DOF, so a word 90 degrees wrong at the right place is refused."""
        return disagreement(word.measurement(), self._believed(predicted))

    def _on_fit(self, msg: Float32) -> None:
        """How well the lidar's last scan matched the map, and when: what says the tracker's
        belief is worth taking a calibration pair from."""
        self._fit, self._fit_at = float(msg.data), self._now()

    def _on_map(self, msg: OccupancyGridMsg) -> None:
        """The served map's identity (size@origin) — the name the board checks a WORD against, and
        the fallback name for the files beside the map until the room is known."""
        served = map_id(msg)
        if served == self._map_id:
            return
        self._map_id = served
        self._rekey()

    def _on_identity(self, msg: String) -> None:
        """Who the map on ``/map`` is, as a ROOM: the latched identity depth_fusion mints
        (``seed:flat3_straight`` -> ``flat3_straight``, :func:`pepin.anchors.room_from_identity`).

        This is what the calibration files are named by, because the size@origin id of a growing
        volume is not a name: measured on the parked cart on 2026-09-18, switching the board from
        the seed pgm to the volume's exported slice moved the served id from 239x215@-18.53,-4.38 to
        280x250@-19.48,-5.48 — the same room, the same lattice, the cart's pose unchanged and the
        lidar's fit better — and every ``<map id>.graph_*`` file went invisible.
        """
        room = room_from_identity(msg.data)
        if not room or room == self._room:
            return
        self._room = room
        self.get_logger().info(f"map identity: the room is {room}; the calibration is named by it")
        self._rekey()

    def _now(self) -> float:
        """This node's clock in seconds: when a fit arrived, how long a disagreement has held."""
        return stamp_seconds(self.get_clock().now().to_msg())

    def _rekey(self) -> None:
        """Take up the files this ROOM keeps, whenever the name they are found under changes: the
        room when one is known (the parameter, or ``/map_identity``), and otherwise the served
        grid's size@origin id, which is what they were named by before 2026-09-18."""
        key = self._room or self._map_id
        if not key or key == self._key or self._slam or not self._graph_odom:
            return
        self._key = key
        self._migrate(key)
        self._adopt(key)

    def _migrate(self, key: str) -> None:
        """Carry the files of the OLD naming over to the room's name, once, and say so.

        A file named by a grid's size@origin belongs to this room just as much as one named by the
        room; it simply cannot be found any more once the volume grows. So on the first start under
        a room name, each missing room-named file is copied from the legacy-named one — the served
        id's if that is there, and otherwise the one legacy file of that kind beside the map when
        there is exactly one. Several candidates is a guess, and a guess here puts the calibration
        of one flat beside another, so it is refused out loud instead.
        """
        for suffix, named in (
            (NODES_SUFFIX, nodes_path),
            (PAIRS_SUFFIX, pairs_path),
            (ANCHOR_SUFFIX, anchor_path),
        ):
            target = named(self._anchor_dir, key)
            if target.exists():
                continue
            legacy = named(self._anchor_dir, self._map_id) if self._map_id else target
            if not legacy.exists() or legacy == target:
                others = [
                    path
                    for path in sorted(self._anchor_dir.glob(f"*{suffix}"))
                    if path != target and path.is_file()
                ]
                if len(others) != 1:
                    if len(others) > 1:
                        self.get_logger().warning(
                            f"{len(others)} files named *{suffix} beside the map and none for room"
                            f" {key}: not guessing which flat they are of"
                        )
                    continue
                legacy = others[0]
            try:
                target.write_bytes(legacy.read_bytes())
            except OSError as error:
                self.get_logger().warning(f"{legacy.name} not migrated: {error}")
                continue
            self.get_logger().info(f"migrated {legacy.name} -> {target.name} (room {key})")

    def _adopt(self, key: str) -> None:
        """Take everything this room already measured about this database: the NODE TABLE first
        (what every word is hung on), then the pairs log the global tie is fitted over, and last
        the old one-seating anchor file with the error bar such a transform was measured to have
        (:func:`pepin.graphtie.file_tie`).

        This is the wake-up path — with a table or a tie on disk the first graph already reads as a
        place on the map, before the tracker has said anything at all."""
        self._table = NodeTable(load_nodes(self._anchor_dir, key), read_sessions(self._database))
        if len(self._table):
            self.get_logger().info(
                f"graph node table read for {key}: {len(self._table)} nodes over"
                f" {len(self._table.coverage())} sessions; sessions from {self._database}"
                + ("" if self._table.sessions_known else " (unreadable: one run is one piece)")
            )
        if self._switches.on("graph_tie_from_pairs"):
            self._pairs = self._read_pairs(key)
            if self._pairs and self._replace(fit_tie(self._pairs), "read"):
                return
        try:
            # The identity inside the file is checkable only under the LEGACY naming, where the file
            # name and the identity are one string; under a room name the room is the identity.
            stored = load_anchor(
                self._anchor_dir, key, self._map_id if key == self._map_id else None
            )
        except (OSError, ValueError) as error:
            self.get_logger().warning(f"stored anchor ignored: {error}")
            return
        if stored is None:
            self.get_logger().info(
                f"no tie stored for {key} in {self._anchor_dir}: it will be measured from"
                " the lidar-held tracker, one pair and one node per sharp seating"
            )
            return
        self._tie = file_tie(stored.pose)
        self.get_logger().info(
            f"graph tie read from the one-seating file: {self._tie.described()}, room {key}"
        )

    def _read_pairs(self, key: str) -> list[TiePair]:
        """The calibration's log for this room, or an empty list when it cannot be read —
        a log nobody can parse is not a reason to refuse to drive, it is a reason to say so."""
        try:
            return load_pairs(self._anchor_dir, key)
        except OSError as error:
            self.get_logger().warning(f"graph pairs not read: {error}")
            return []

    def _replace(self, fitted: Tie | None, why: str) -> bool:
        """Adopt ``fitted`` if it is the better tie, and say so; answer whether it was adopted.

        Better means a COVARIANCE THAT IS NOT LARGER — the generalised variance of the fit
        (:attr:`pepin.graphtie.Tie.volume`) — and nothing else. A refit never replaces the tie
        because something disagreed with the old one: that was the re-learn, and it is what fitted
        a constant to a variable. "Not larger" rather than "smaller" so that a refit over a longer
        log which is exactly as precise still lands: it carries the OUTLIER SHARE that longer log
        measured, and that number is how the owner learns the database holds a contradiction."""
        if fitted is None:
            return False
        held = self._tie
        if held is not None and held.origin != "file" and fitted.volume > held.volume:
            return False
        self._tie = fitted
        self.get_logger().info(f"graph tie {why}: {fitted.described()}")
        return True

    def _keep(self, anchor: Pose2D, origin: str) -> None:
        """Write a ONE-SEATING anchor beside its map, the arrangement of before 2026-09-18, reached
        only with ``graph_tie_from_pairs`` off. Nothing is written without a map to name it."""
        self._tie = file_tie(anchor)
        if not self._key:
            return
        record = Anchor(anchor, self._key, self._now(), origin, 0)
        try:
            path = save_anchor(self._anchor_dir, record)
        except OSError as error:
            self.get_logger().warning(f"anchor not stored: {error}")
            return
        self.get_logger().info(f"graph anchor stored in {path}: {record.described()}")

    @property
    def frames(self) -> tuple[str, str]:
        """The parent and child of the edge this mode publishes."""
        return SLAM_FRAMES if self._slam else KNOWN_MAP_FRAMES

    def _on_localization(self, msg: PoseWithCovarianceStamped) -> None:
        """RTAB-Map placing itself in the database it loaded: the cart's pose in the graph's own
        frame, with the covariance RTAB-Map measured for it.

        This is a LOCALISATION and not an integration, which is the whole difference: it exists
        only while RTAB-Map is tied to the loaded database, and it carries no odometry from the
        session's start. Kept here until a graph message spends it on a word.
        """
        self._localizations += 1
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        place = Pose2D(
            float(p.x),
            float(p.y),
            math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)),
        )
        sigma = math.sqrt(max(float(msg.pose.covariance[0]), 0.0))
        self._localized = (place, stamp_seconds(msg.header.stamp), sigma)
        # The planar block of the 6x6 RTAB-Map filled in (x, y, yaw -> rows 0, 1, 5): what the
        # REGISTRATION between the two nodes is worth, which is the word's own term once the word
        # is hung on a node instead of composed through a global tie.
        covariance = list(msg.pose.covariance)
        self._localized_cov = np.array(
            [[covariance[index] for index in row] for row in ((0, 1, 5), (6, 7, 11), (30, 31, 35))],
            dtype=float,
        )

    def _on_graph(self, msg: MapGraph) -> None:
        """RTAB-Map's correction, and what follows from it: in SLAM it IS ``map -> odom``; on a
        known map it says where the graph believes the odom frame sits in the graph's own frame,
        which with the EKF's ``odom -> base_link`` gives the cart's place in that frame — one more
        pair of the tie's calibration when the lidar is sharp, and one word every time."""
        pose = pose_from_transform(msg.map_to_odom)
        self._graphs += 1
        self._correction2d = Pose2D(
            float(pose.translation[0]),
            float(pose.translation[1]),
            math.atan2(float(pose.rotation[1, 0]), float(pose.rotation[0, 0])),
        )
        self._read_graph_poses(msg)
        if self._slam:
            self._pose = pose
            return
        if not self._graph_odom:  # the old arrangement: the correction IS the frame
            self._pose = pose.inverse()
            if self._belief is not None:
                # No calibration exists in this arrangement: the correction IS the transform, and
                # it carries no uncertainty of its own, so the word is worth its floor and nothing
                # is added to it.
                correction = Tie(
                    pose=self._correction2d,
                    covariance=np.zeros((3, 3)),
                    centroid=None,
                    origin="correction",
                )
                self._offer(
                    self._belief,
                    correction.pose,
                    correction.covariance,
                    GRAPH_FLOOR_YAW_DEG,
                    self._belief_stamp,
                )
            return
        cart = self._cart_in_graph(msg)
        if cart is None:
            self._blind += 1
            return
        place, stamp = cart
        self._place = place
        held = self._tie
        if held is not None and held.origin == "identity":
            pass  # a frame born with this map is identity BY CONSTRUCTION: nothing to measure
        elif self._switches.on("graph_tie_from_pairs"):
            self._collect(place, stamp)
        elif held is None:
            anchor = self._learn_anchor(place, stamp)
            if anchor is None:
                return
            self._keep(anchor, "learned")
        if self._switches.on("graph_word_from_nodes"):
            # ONE WORD SOURCE, not two. A localisation that names no node is odometry in the graph's
            # frame (measured 2026-09-18: /rtabmap/localization_pose is published on EVERY update at
            # 1 Hz, recognised or not, and parked it carried 1.14 m of accumulated
            # Loop/MapToBase_lin_std over 11 m of "travel" that was VO jitter at rest). The global
            # tie is not allowed to rescue such an update either: on that same evening the tie
            # fitted from one pair put words out 36.7 deg wrong in heading while claiming 8 deg of
            # sigma, and a fallback that lies with a confident sigma is worse than silence.
            self._hung = None
            matched = self._matched_node()
            if matched is None:
                self._unnamed += 1
                return
            # JUDGED on the table as it stands, and only then MEASURED into it. The other order is
            # circular: a node tabled from this very update would put the word exactly on the
            # tracker's own pose (measured live: node words 0.0-0.2 cm from the tracker at a place
            # tabled minutes earlier), which cannot move the pose and would feed the agreement
            # window a zero residual it never earned. A node learned now is worth a word from the
            # NEXT update on — and a recognised node whose session nothing on our map knows is
            # exactly the node worth putting on the map, whether or not a word came of it.
            hung = self._hang(matched, place)
            self._table_matched(matched, place, stamp)
            if hung is None:
                return
            self._node_words += 1
            self._substituted += 1 if hung.substituted else 0
            self._hung = hung
            self._offer(hung.rel, hung.entry.cart, hung.covariance, GRAPH_FLOOR_YAW_DEG, stamp)
            return
        tie = self._tie
        if tie is None:
            return
        self._hung = None
        self._offer(
            place,
            tie.pose,
            tie.word_covariance(place),
            self._word_sigma_deg(tie, place),
            stamp,
        )

    def _read_graph_poses(self, msg: MapGraph) -> None:
        """The LIVE graph's own poses of its nodes, by id — the only thing this node ever takes from
        RTAB-Map's global frame, and it takes them only to SUBTRACT two of them from each other
        (:meth:`_hang`). A build whose MapGraph does not carry them leaves the table as it was.

        It is also what tells MAPPING from LOCALISATION mode, which is what decides whether the node
        RTAB-Map is building now is worth tabling: a new node the database KEEPS appears in a later
        graph, and a temporary one (``Mem/IncrementalMemory`` false) never does. That signal was
        chosen over reading the other node's parameter because it is the property that actually
        matters — "will this id still mean something tomorrow" — and because it needs no service
        call and cannot race with a live ``update_parameters``."""
        ids = list(getattr(msg, "poses_id", ()) or ())
        poses = list(getattr(msg, "poses", ()) or ())
        if not ids or len(ids) != len(poses):
            return
        self._graph_poses = {
            int(node_id): Pose2D(
                float(pose.position.x), float(pose.position.y), yaw_of(pose.orientation)
            )
            for node_id, pose in zip(ids, poses, strict=True)
        }
        self._commit_pending()

    def _commit_pending(self) -> None:
        """Take the node this run created into the table once the live graph shows the database KEPT
        it, and drop it when it did not.

        A ``ref_id`` measured in mapping mode appears in the next ``/rtabmap/mapGraph``; one
        measured in localisation mode never appears, because nothing was written. That wait is
        the whole mode test: in localisation mode the table then grows ONLY through the nodes
        RTAB-Map recognises (:meth:`_table_matched`) — the only ids there that last.
        """
        pending = self._pending_node
        if pending is None:
            return
        if pending.node_id in self._graph_poses:
            self._pending_node = None
            self._take(pending, mapping=True)
            self._mapping = True
            return
        if self._ref != pending.node_id:  # a newer node came and the old one never landed
            self._pending_node = None
            self._temporary += 1
            self._mapping = False

    def _take(self, entry: NodePose, mapping: bool) -> None:
        """Put one entry in the table and on the log, if it is the sharpest for that node."""
        if not self._table.offer(entry):
            return
        self._tabled += 1
        self._from_matched += 0 if mapping else 1
        self._last_tabled = entry.cart
        try:
            append_node(self._anchor_dir, self._key, entry)
        except OSError as error:
            self.get_logger().warning(f"graph node {entry.node_id} not logged: {error}")

    def _table_matched(self, matched: int, place: Pose2D, stamp: float) -> None:
        """Measure the node RTAB-Map just RECOGNISED and put it on the map: ``Table[N] = cart .
        inverse(rel)`` with ``rel`` to THAT node (:func:`pepin.graphnodes.entry_from_match`).

        This is the only way the table grows in localisation mode — nothing is written to the
        database there, so no new node ever has a lasting id — and the only way the 43 sessions of
        this database that no lidar-held drive ever visited become placeable at all: drive past a
        place with the lidar holding the pose, let the camera recognise a node of any session, and
        that node's place on our map is known from then on.

        Nothing is measured without a sharp lidar-held seating (:meth:`_unsharp`), an entry replaces
        an existing one only when it is sharper. (The flood of 2026-09-18 — 62 -> 81 -> 96 -> 145
        entries in minutes at one spot — came from NEW ids, one a second; those are deduplicated
        by place in :meth:`_table_node`. A recognised node has one id however long the cart sits.)
        """
        node_pose = self._graph_poses.get(matched)
        predicted = self._predicted()
        if node_pose is None or predicted is None:
            return
        if abs(stamp - self._belief_stamp) > ANCHOR_MAX_SKEW_S or self._unsharp() is not None:
            return
        # No same-place check HERE, unlike for a new node: the table holds ONE entry per node id
        # and takes another only when it is sharper (:meth:`pepin.graphnodes.NodeTable.offer`), so
        # a recognised node cannot flood it — while a DIFFERENT node recognised from the place the
        # last entry was taken at is new knowledge, and the place-keyed check threw exactly that
        # away (live 2026-09-18: parked at home, node 90139 of an untabled session recognised 11
        # times, "0 off recognised nodes", 11 words unplaceable).
        seating = self._belief_sigma or (
            float(self._switches["anchor_max_sigma_m"]),
            float(self._switches["anchor_max_sigma_m"]),
            math.radians(float(self._switches["anchor_max_sigma_deg"])),
        )
        self._take(
            entry_from_match(
                node_id=matched,
                session=self._table.session_of(matched),
                stamp=stamp,
                cart=predicted,
                seating=seating,
                rel=self._rel(matched, node_pose, place),
                rel_covariance=self._localized_cov,
            ),
            mapping=False,
        )

    def _rel(self, matched: int, node_pose: Pose2D, place: Pose2D) -> Pose2D:
        """Where the cart stands RELATIVE TO the node it recognised: the pose of the current frame
        in that node's frame, the one quantity here the optimiser's global frame cannot touch.

        Two candidates, and the choice is a live flag (``graph_rel_from_registration``). The
        DIFFERENCE of the two graph poses is the default because it is the one this stack has
        measured (offline over 171 links, 4.3 cm p50 against the lidar's truth); RTAB-Map's own RAW
        registration (``Info.loop_closure_transform``) is better in principle — after a localisation
        link its ``P_current`` is an optimised compromise with its odometry cache, which an earlier
        localisation on an inconsistent piece can pull, while a raw transform cannot — but its
        direction convention is unverified here and the installed image carries no sources to check
        it against. So both are computed and the DISAGREEMENT is logged once per graph; one live run
        with that line settles which way the transform points, and then the flag goes on.
        """
        difference = compose(inverse(node_pose), place)
        raw = self._registration
        if raw is None:
            return difference
        gap_m = math.hypot(raw.x - difference.x, raw.y - difference.y)
        turn = abs(math.degrees(wrap_angle(raw.theta - difference.theta)))
        flipped = inverse(raw)
        flip_m = math.hypot(flipped.x - difference.x, flipped.y - difference.y)
        if self._registration_said != matched:
            self._registration_said = matched
            self.get_logger().info(
                f"graph rel on node {matched}: graph poses say ({difference.x:+.3f},"
                f" {difference.y:+.3f}, {math.degrees(difference.theta):+.1f} deg), Info's"
                f" registration ({raw.x:+.3f}, {raw.y:+.3f}, {math.degrees(raw.theta):+.1f} deg)"
                f" — {gap_m * 100:.1f} cm / {turn:.1f} deg apart as it stands,"
                f" {flip_m * 100:.1f} cm inverted"
            )
        if not self._switches.on("graph_rel_from_registration"):
            return difference
        return raw if gap_m <= flip_m else flipped

    def _hang(self, matched: int, place: Pose2D) -> NodeWord | None:
        """The word hung on the database NODE ``matched``, or ``None`` when it cannot be.

        The geometry is ``Table[N] . inverse(P_N) . P_current``, both graph poses read from one live
        ``/rtabmap/mapGraph``. Everything about the optimiser's own frame cancels in that
        subtraction, which is the point: the frame moves whenever the graph is re-rooted or
        re-optimised, and a relative pose inside one piece does not.

        ``None``, counted and named in the report line, when the word cannot be hung: the table is
        empty for this room, the live graph carries no poses, or nothing on our map knows that
        node's session. There is no fallback — see :meth:`_on_graph`.
        """
        if not len(self._table) or not self._graph_poses:
            return None
        if matched not in self._table and not self._table.sessions_known:
            self._table.learn_sessions(read_sessions(self._database))
        hung = self._table.hang(
            matched, place, self._graph_poses, self._localized_cov, self._graphs
        )
        if isinstance(hung, str):
            self._unplaceable += 1
            if hung != self._unplaceable_reason:
                self._unplaceable_reason = hung
                self.get_logger().info(f"graph word not placeable ({self._unplaceable}): {hung}")
            return None
        return hung

    def _matched_node(self) -> int | None:
        """The database node the localisation in hand was made against, or ``None``.

        ``/rtabmap/info`` and ``/rtabmap/localization_pose`` are published from ONE update and carry
        one stamp, so the info whose stamp is the localisation's is the one that names the node
        (``loop_closure_id`` or ``proximity_detection_id``). The tolerance is
        :data:`ANCHOR_MAX_SKEW_S` and covers float rounding and a build that stamps one of the two
        with its publish time; past it the two are different updates and no node is named.
        """
        named = self._matched
        if named is None:
            return None
        node_id, stamp = named
        if abs(stamp - self._localized_stamp) > ANCHOR_MAX_SKEW_S:
            return None
        return node_id

    def _collect(self, place: Pose2D, stamp: float) -> None:
        """One more pair of the calibration, when this instant is worth measuring the tie from, and
        a refit when the log has grown.

        The instant is worth it when the LIDAR is what the tracker is believing and its seating is
        pinned in both axes and in heading (:meth:`_unsharp`), and when the belief carried forward
        over the EKF's odometry really is of this graph's own moment
        (:data:`ANCHOR_MAX_SKEW_S`). Every refusal counts one "pending" and says why once per
        reason: camera-only, or along a sofa, nothing is measured and the tie on disk is used as it
        is — which is the whole of the lidar independence.

        A pair at THE SAME PLACE as the last one is not taken: the calibration's information is the
        spread of the places it was measured at, a parked cart would otherwise append one pair a
        second for as long as it stands there, and "the same place" already has a definition on this
        robot (:func:`pepin.watchdog.same_place`, the pair the board's own searches call agreement).

        The refit is closed-form and cheap, so it runs on every pair; it REPLACES the tie only when
        it is statistically better (:meth:`_replace`).
        """
        predicted = self._predicted()
        if predicted is None or abs(stamp - self._belief_stamp) > ANCHOR_MAX_SKEW_S:
            self._pend("the tracker's belief is not within half a second of this graph")
            return
        refusal = self._unsharp()
        if refusal is not None:
            self._pend(refusal)
            return
        if self._pairs and same_place(self._pairs[-1].cart, predicted):
            self._pend("the cart is where the last pair was taken")
            return
        sigma = self._pair_sigma()
        pair = TiePair(stamp=stamp, cart=predicted, place=place, sigma=sigma, note=self._map_id)
        self._anchor_sigma = self._belief_sigma
        self._pairs.append(pair)
        self._pairs_taken += 1
        if self._map_id:
            try:
                append_pair(self._anchor_dir, self._map_id, pair)
            except OSError as error:
                self.get_logger().warning(f"graph pair not logged: {error}")
        self._replace(fit_tie(self._pairs), f"refitted over {len(self._pairs)} pairs")

    def _pair_sigma(self) -> tuple[float, float, float]:
        """The joint error bar of one pair: the tracker's own seating (the covariance
        ``/tracker_pose`` carries) and RTAB-Map's own localisation sigma added in quadrature —
        what the fit weighs the pair by and what its residual is gated against. The seating gate's
        own limits stand in for whichever half did not say."""
        seating = self._belief_sigma or (
            float(self._switches["anchor_max_sigma_m"]),
            float(self._switches["anchor_max_sigma_m"]),
            math.radians(float(self._switches["anchor_max_sigma_deg"])),
        )
        graph_m = self._localized_sigma_m or GRAPH_FLOOR_XY_M
        return (
            math.hypot(seating[0], graph_m),
            math.hypot(seating[1], graph_m),
            math.hypot(seating[2], math.radians(GRAPH_FLOOR_YAW_DEG)),
        )

    def _cart_in_graph(self, msg: MapGraph) -> tuple[Pose2D, float] | None:
        """Where the graph has the cart, in the graph's own frame, and at which moment.

        With ``graph_word_from_localization`` on this is RTAB-Map's OWN localisation in the
        database it loaded (:meth:`_on_localization`), spent once and never reused: no
        localisation, no word. Off, it is the old arrangement — the session's correction composed
        with the EKF's ``odom -> base_link`` — which between closures is odometry from the
        session's start wearing the map's coordinates. That is what put 42.5 m into a word and
        stood it 3.6 m from the tracker (2026-09-17); the word then drove the cart at a person.
        """
        if self._switches.on("graph_word_from_localization"):
            spent = self._localized
            self._localized = None  # one localisation buys exactly one word
            if spent is None:
                self._unlocalized += 1
                return None
            place, self._localized_stamp, self._localized_sigma_m = spent
            # The POSE comes from the localisation; the STAMP must not. A localisation is stamped
            # on the LAPTOP's clock and the word is judged on the BOARD's: the two run minutes
            # apart, so a word stamped here arrives already older than the tracker's patience —
            # taken into the fusion, but counted as a source that has not spoken in two minutes,
            # and the goal gate then refuses every camera-only drive ("graph stale 128.5 s",
            # 2026-09-17). The odometry transform is published BY the board, so its stamp is the
            # board's own clock, which is what the old word source used and why it never saw this.
            transform = self._lookup.transform(ODOM_FRAME, BASE_FRAME, msg.header.stamp)
            transform = transform or self._lookup.transform(ODOM_FRAME, BASE_FRAME)
            odom = self._planar(transform)
            if transform is None or odom is None:
                self._blind += 1
                return None
            self._word_odom = odom  # for the agreement residual's own bookkeeping
            return place, stamp_seconds(transform.header.stamp)
        transform = self._lookup.transform(ODOM_FRAME, BASE_FRAME, msg.header.stamp)
        transform = (
            transform if transform is not None else self._lookup.transform(ODOM_FRAME, BASE_FRAME)
        )
        planar = self._planar(transform)
        if transform is None or planar is None:
            return None
        self._word_odom = planar
        return compose(self._correction2d, planar), stamp_seconds(transform.header.stamp)

    @staticmethod
    def _planar(transform: TransformStamped | None) -> Pose2D | None:
        """A transform read in the plane the cart drives in, or ``None`` when there is none:
        the EKF's ``odom -> base_link``, both at a belief's moment and at a graph's."""
        if transform is None:
            return None
        pose = pose_from_transform(transform)
        return Pose2D(
            float(pose.translation[0]),
            float(pose.translation[1]),
            math.atan2(float(pose.rotation[1, 0]), float(pose.rotation[0, 0])),
        )

    def _unsharp(self) -> str | None:
        """Why the tracker's present seating may not become one pair of the tie's calibration —
        one phrase for the log — or ``None`` when it may.

        Two things, in the order a person would ask them: is the LIDAR behind this belief at all
        (a fresh fit of at least :data:`TRUSTED_FIT`), and is its seating pinned in both axes and
        in heading (the covariance the belief carries, against ``anchor_max_sigma_m`` /
        ``anchor_max_sigma_deg``). A fit answers the first question and says nothing about the
        second: a scan sliding along a sofa matches beautifully everywhere it slides to.
        """
        if self._fit < TRUSTED_FIT or self._now() - self._fit_at > FIT_FRESH_S:
            return (
                f"the lidar is not driving the tracker (fit {self._fit:.2f}, last heard"
                f" {self._now() - self._fit_at:.1f} s ago)"
            )
        return seating_refusal(
            self._belief_sigma,
            float(self._switches["anchor_max_sigma_m"]),
            float(self._switches["anchor_max_sigma_deg"]),
        )

    def _learn_anchor(self, place: Pose2D, stamp: float) -> Pose2D | None:
        """The ONE-SEATING anchor of before 2026-09-18, reached only with ``graph_tie_from_pairs``
        off: ``map <- rtabmap`` from the tracker's belief and the graph's own place for the same
        cart, or ``None`` while there is nothing worth learning it from — no belief close enough in
        time, or a seating too soft (:meth:`_unsharp`). Every refusal counts one "pending" and
        leaves the frame at identity."""
        if self._belief is None or abs(stamp - self._belief_stamp) > ANCHOR_MAX_SKEW_S:
            self._pend("the tracker's belief is not within half a second of this graph")
            return None
        refusal = self._unsharp()
        if refusal is not None:
            self._pend(refusal)
            return None
        anchor = graph_anchor(self._belief, place)
        self._anchor_sigma = self._belief_sigma
        self.get_logger().info(
            f"graph anchored: map <- rtabmap = ({anchor.x:+.2f}, {anchor.y:+.2f},"
            f" {math.degrees(anchor.theta):+.1f} deg), from the tracker at"
            f" ({self._belief.x:+.2f}, {self._belief.y:+.2f}) seated to"
            f" {describe_sigma(self._belief_sigma)} at fit {self._fit:.2f} and the graph at"
            f" ({place.x:+.2f}, {place.y:+.2f}) on map {self._map_id}"
        )
        return anchor

    def _pend(self, reason: str) -> None:
        """Count one graph that found no seating worth measuring the tie from, and say why the
        first time each reason appears: a node whose calibration is not growing must say what it is
        waiting for, and a line per graph would say it 3600 times an hour."""
        self._pending += 1
        if reason == self._pending_reason:
            return
        self._pending_reason = reason
        self.get_logger().info(f"graph pair pending ({self._pending}): {reason}")

    def _offer(
        self,
        place: Pose2D,
        frame: Pose2D,
        extra: Matrix,
        floor_yaw_deg: float,
        stamp: float,
    ) -> None:
        """One graph word: ``place`` read on the tracker's own map through ``frame``, published as a
        measurement for the board's fusion. Two things wear that shape and the arithmetic is the
        same for both — the NODE word (``place`` is ``rel``, ``frame`` is ``Table[N]``) and the
        global-tie word (``place`` is the localisation, ``frame`` is the tie) — with ``extra`` the
        3x3 that frame is uncertain by, already propagated into the map frame.

        The word is remembered whatever the flag says — the report line is how a session is
        judged before it is allowed to move anything — and sent only with the flag on, a map to
        name, and a pose the 3-DOF gate accepts: the squared Mahalanobis distance between the word
        and the odometry-carried belief, under the sum of their covariances, at most
        :data:`pepin.fusion.GATE`. That is the gate the board's information filter will apply
        next, so a word this node passes is a word that can actually move the pose — and a word
        90 degrees wrong at the right place is refused here, which the old metres-only test could
        not do.

        THE FRAME'S OWN UNCERTAINTY is in that covariance, propagated through this very composition
        (:meth:`pepin.graphnodes.node_covariance`, :meth:`pepin.graphtie.Tie.word_covariance`): a
        node whose own seating was soft, or a tie measured at one end of the flat, makes an honestly
        wide word rather than silence or a lie.

        What the measurement cannot carry goes out on the candidate channel instead
        (:meth:`_propose`, ``graph_candidates``): a word the gate refuses is the very word a
        carried cart needs, and a re-seed is the only mechanism that moves a pose that is not
        almost right but simply in the wrong place.

        With NO belief at all the word still goes out, but only on a tie that came off DISK: that
        is the wake-up (a lidar-less start, nothing on ``/tracker_pose`` yet), and it is the one
        case where the graph is the only thing that knows where the cart is. A tie measured in this
        session cannot produce that word — it was measured FROM beliefs.

        NOTHING is published at all — neither measurement nor candidate — while RTAB-Map has not
        RECOGNISED the database it loaded since its own start (:meth:`_recognised`,
        ``graph_words_need_recognition``): its nodes are then an unlinked segment placed by this
        session's odometry, and a word off one is odometry dressed as a pose. The word is still
        remembered and counted, so the report line says how many were withheld and why.

        The word's ``fit`` is what the GRAPH knows, not what any scan measured: how well the
        last words track the tracker's odometry-propagated pose, or the decay since the last tie
        to an older node, capped while the tie is the one-seating file's and nothing has been
        recognised (``graph_trust``, :meth:`_trust_now`). Downstream that number is not a discount
        on the covariance — the fusion gates on chi-square and the measurement gate has no fit floor
        at all — but it is what the board publishes as its own confidence with ``local_fit`` off
        (the goal server's lost ladder then reads it), and it is the SCORE a candidate is
        admitted on (:meth:`_propose`): below :data:`pepin.watch.ADMIT_FIT` a candidate is
        "unknown map" and re-seeds nothing, which is exactly right — a graph that has recognised
        nothing has no business teleporting the cart."""
        word = compose(frame, place)
        self._word = word
        # The GEOMETRY of the word first, with no fit on it: the fit is what the last words'
        # agreement says, and this word's own residual belongs in that window before it is read.
        # The map id is the SERVED GRID's (size@origin), whatever the files are named by: it is what
        # the board's tracker checks a word against, and that is a different thing from a file name.
        remote = graph_measurement(
            place,
            frame,
            stamp,
            self._map_id,
            floor_xy_m=self._word_sigma_m(),
            floor_yaw_deg=floor_yaw_deg,
            extra=extra,
        )
        predicted = self._predicted()
        self._gap_m = (
            math.hypot(word.x - predicted.x, word.y - predicted.y)
            if predicted is not None
            else math.inf
        )
        self._gap_sigmas = (
            math.sqrt(max(self._disagreement(remote, predicted), 0.0))
            if predicted is not None
            else math.inf
        )
        if not self._recognised():
            self._withheld += 1
            return
        self._agree(remote)
        remote = replace(remote, fit=self._trust_now())
        self._propose(remote)
        if self._measurement is None or not self._switches.on("graph_measurement"):
            return
        if not self._map_id:
            return
        if predicted is None:
            # The wake-up, and the only case with nothing to check the word against. It is safe
            # exactly because a pair and a table entry both need a belief: with none ever heard,
            # whatever the word hangs on came off disk (or is the identity of a freshly born frame)
            # and not out of this session.
            if self._pairs_taken or self._tabled:
                return
        elif self._gap_sigmas**2 > GATE:
            self._refused += 1
            return
        self._sent += 1
        self._measurement.publish(String(data=remote.to_json(graphs=self._graphs)))

    def _word_sigma_deg(self, tie: Tie, place: Pose2D) -> float:
        """What this word's HEADING is worth in degrees: the graph's own measured floor
        (:data:`pepin.measurements.GRAPH_FLOOR_YAW_DEG`) and the tie's heading uncertainty added in
        quadrature, because a tie whose rotation is only known to a few degrees cannot make a word
        that knows its heading better than that."""
        _, sigma_yaw = tie.sigma_at(place)
        return math.degrees(math.hypot(math.radians(GRAPH_FLOOR_YAW_DEG), sigma_yaw))

    def _word_sigma_m(self) -> float:
        """What this word is worth in metres: the graph's measured floor, or the scatter of the
        last words when that is wider.

        The floor is what the source was last seen to be worth against the lidar (median 19 cm
        over 61 words, p90 38 cm, 2026-09-17) and it covers the part no word can see in itself —
        a whole frame sitting a little to one side. The scatter is the part it CAN see, measured
        here by :class:`pepin.graphtrust.Agreement`: when the words start disagreeing with the
        pose the odometry carries between them, the graph is coming apart, and the word must say
        so in the one language the fusion reads. Before this the word claimed the same 0.20 m
        whether it sat 1 cm from the tracker or 2.6 m (the day's worst), and every gate
        downstream — the goal's, the drive's, the volume's — believed it.

        And on top of both, the drift the word has accumulated SINCE THE TIE that placed it: the
        graph's pose is carried by odometry between closures, so a word five metres past its last
        tie is worth less than the same word at the tie, and the growth is the odometry's own
        error model — :data:`pepin.visual_odometry.SCALE_ERROR` of the distance driven, the same
        fraction the VO covariance and the tracker's own spread already use. That is what used to
        be a clock deciding to publish or not: an old tie now makes a WEAK word, not no word, and
        the drive gate refuses it on a number (2026-09-17).

        WITH THE LOCALISATION SOURCE the distance term is not here at all, and that is structural.
        A localisation carries no odometry from the session's start, so RTAB-Map's distance counter
        says nothing about it — and that counter CREEPS from visual-odometry noise on a cart that is
        standing still, which on 2026-09-17 grew the published sigma from 0.38 m to 0.71 m on a
        PARKED robot and had the drive refused. The term belongs to the old word source, where the
        word really is odometry from a session's start, and it is reached only there.

        The FRAME's own uncertainty is not here either: it is a covariance in the map frame with a
        lever arm in it, so it is added to the word's covariance as a matrix instead of squeezed
        into one radius (:func:`pepin.graphnodes.node_covariance`,
        :meth:`pepin.graphtie.Tie.word_covariance`, passed as ``extra``)."""
        spread = self._agreement.rms(self._now())
        floor = GRAPH_FLOOR_XY_M if spread is None else max(GRAPH_FLOOR_XY_M, float(spread))
        measured = self._localized_sigma_m
        if self._switches.on("graph_word_from_localization"):
            return floor if measured is None else max(floor, measured)
        if measured is not None:
            return max(floor, measured)
        return math.hypot(floor, SCALE_ERROR * self._trust.since_m)

    def _on_sources(self, msg: String) -> None:
        """Who is HOLDING the tracker's pose, off the board's own per-source report: the enabled
        source the cart has driven the least since its word (:meth:`pepin.watch.Preflight.holding`).

        Read name-free on purpose. The rule that decides whether RTAB-Map may learn is "a sharp pose
        that does not come from the database itself", and the only name it ever compares is our own
        (:data:`pepin.sources.GRAPH`): a stereo matcher that lands at a few centimetres will teach
        the database the day it exists, with nothing here changed.
        """
        try:
            report = json.loads(msg.data)
        except (TypeError, ValueError):
            return
        if not isinstance(report, dict):
            return
        holding = Preflight.holding(source_words(report))
        self._holder = None if holding is None else holding.name
        self._holder_at = self._now()

    def _decide_mode(self) -> None:
        """Ask RTAB-Map to learn or only to recognise, on a change of verdict that has held.

        The rule is :class:`pepin.graphnodes.ModeRule`; everything this method adds is the wiring.
        A switch is not sent while the last one is unanswered — that is the "no faster than the
        service answers" rule, exact and with no number in it — and the parameters each mode needs
        (:data:`MODE_PARAMETERS`) travel with it, because the mode services do not touch them.
        """
        if not self._modes or not self._key:
            return
        stale = self._now() - self._holder_at > FIT_FRESH_S
        verdict = self._mode.update(
            self._now(),
            self._unsharp() or ("the board has not said who holds the pose" if stale else None),
            None if stale else self._holder,
            describe_sigma(self._belief_sigma),
        )
        if verdict is None:
            return
        if self._mode_pending is not None and not self._mode_pending.done():
            self._mode_failed += 1
            return
        client = self._modes[verdict.mapping]
        if not client.service_is_ready():
            self._mode_failed += 1
            return
        self._mode_pending = client.call_async(Empty.Request())
        self._retune(verdict.mapping)
        # rclpy names a client's service ``srv_name``; a fake with ``.name`` let this line crash
        # the node on its first live switch (2026-09-18).
        self.get_logger().info(f"rtabmap memory: {verdict.text()} ({client.srv_name})")

    def _retune(self, mapping: bool) -> None:
        """Set the parameters the new mode needs and have RTAB-Map re-read them.

        ``RGBD/LinearUpdate`` and ``RGBD/AngularUpdate`` are the pair: 0 while LOCALISING, so a
        parked cart is recognised at all (measured live — with the defaults every update at rest
        read ``Memory/Small_movement`` 1 and named no node), and 0.05 while MAPPING, because 0
        there keeps a node a second at a standstill and put 250 junk nodes in the database in one
        evening. They are string-typed on RTAB-Map's side, and the mode services do not carry them.
        """
        if self._tuner is None or not self._tuner.service_is_ready():
            return
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name=name,
                value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value),
            )
            for name, value in MODE_PARAMETERS[mapping].items()
        ]
        self._tuner.call_async(request)
        if self._reread is not None and self._reread.service_is_ready():
            self._reread.call_async(Empty.Request())

    def _on_sigma(self, msg: String) -> None:
        """The tracker's own post-fusion spread: what its pose is worth whatever sensor held it."""
        sigma = Sigma.from_json(msg.data, age_s=0.0)
        self._sigma_xy = None if sigma is None else sigma.xy_m

    def _lost(self) -> bool:
        """Whether the board's tracker has nothing trustworthy behind the pose it publishes: a
        fit below :data:`TRACKER_TRUSTED_FIT` (or no fit at all within :data:`FIT_FRESH_S`), or
        no belief heard for :data:`BELIEF_FRESH_S`. It is what turns the graph's word from a
        correction to an almost-right pose into the only opinion anyone has."""
        now = self._now()
        fit = self._fit if now - self._fit_at <= FIT_FRESH_S else 0.0
        # A fit of zero is not a lost tracker: camera-only it publishes 0.00 by construction and
        # the pose is held by these very words. Before this the residual was therefore never
        # taken in the one mode it is needed in, and the word's trust stood at 1.00 all the way
        # into the table (2026-09-16). The sigma is what answers there: it is the tracker's own
        # post-fusion spread, and a tracker riding a graph word reports that word's floor.
        vouched = self._sigma_xy is not None and self._sigma_xy <= TRACKER_TRUSTED_SIGMA_M
        return (fit < TRACKER_TRUSTED_FIT and not vouched) or now - self._belief_at > BELIEF_FRESH_S

    def _propose(self, remote: RemoteMeasurement) -> None:
        """The same word on the whole-map candidate channel — the door a measurement cannot
        open — when it is the kind of word only that door admits:

        * disagreeing with the tracker's belief past :data:`pepin.fusion.GATE` in 3 DOF, so it is
          refused as a measurement here and would be refused again by the board's chi-square gate:
          exactly the word a carried cart produces once the graph recognises the place;
        * or naming a DIFFERENT place (:func:`pepin.watchdog.same_place`) while the tracker has
          no trusted source behind its pose (:meth:`_lost`) — including a tracker this node has
          not heard from at all, the wake-up.

        A word that says the place the tracker already holds is never sent: the board's gate
        would call it agreement, act on nothing, and end whatever streak the LIDAR's own search
        had built — the graph has no business costing the lidar its recovery.

        At most one candidate per graph message (:attr:`_proposed_at`), carrying the graph's
        own count as its scan id: a streak on the board is three DISTINCT pieces of evidence,
        and one graph message repeated is one piece. The covariance is the measurement's, floor
        and nothing else (:func:`pepin.measurements.graph_measurement`); the board re-judges the
        word against its own pose and fit, and its rules do the rest — three agreeing
        candidates, no re-seed while a goal runs, and no re-seed from a source that is not the
        lidar while the lidar is alive.

        The SCORE the board judges it on is the graph's trust (``graph_trust``), so a candidate
        sent while the graph has recognised nothing is refused there as "unknown map"
        (:data:`pepin.watch.ADMIT_FIT`, 0.45 — four metres past a tie) and a candidate sent the
        moment a closure lands claims 1.0 and beats anything a lost tracker holds. Sending it
        anyway, rather than filtering here, is deliberate: the board's report line then carries
        the refusal, and one gate judges every source.
        """
        if self._candidate is None or not self._switches.on("graph_candidates"):
            return
        if not self._map_id or self._proposed_at == self._graphs:
            return
        belief = self._belief
        elsewhere = belief is None or not same_place(remote.pose, belief)
        refused = self._gap_sigmas**2 > GATE
        if not (refused or (elsewhere and self._lost())):
            return
        candidate = GlobalCandidate(
            x=remote.x,
            y=remote.y,
            yaw=remote.yaw,
            covariance=remote.covariance,
            score=remote.fit,
            # The graph answers with one place: a closure is a recognition, not a correlation
            # surface with a runner-up to be compared against.
            ambiguity=0.0,
            stamp=remote.stamp,
            map_id=remote.map_id,
            scan_id=self._graphs,
            source=remote.source,
        )
        self._proposed += 1
        self._proposed_at = self._graphs
        self._candidate.publish(
            String(
                data=candidate.to_json(
                    reason="refused as a measurement" if refused else "the tracker has no source",
                    gap_cm=None if math.isinf(self._gap_m) else round(self._gap_m * 100.0, 1),
                    gap_sigmas=(
                        None if math.isinf(self._gap_sigmas) else round(self._gap_sigmas, 2)
                    ),
                    graphs=self._graphs,
                )
            )
        )

    def _publish(self) -> None:
        """The edge this mode owns, at :data:`RATE_HZ`: the tie (identity until one is known) on a
        known map, the correction as a message to the board in SLAM, and the correction's inverse
        with ``graph_odom`` off — each held between graphs, because none of them moves until the
        graph does."""
        self._decide_mode()
        stamp = self.get_clock().now().to_msg()
        parent, child = self.frames
        tie = self._tie
        message = (
            transform_from_rpy(
                parent,
                child,
                (tie.pose.x, tie.pose.y, 0.0),
                (0.0, 0.0, tie.pose.theta),
                stamp,
            )
            if tie is not None and self._graph_odom and not self._slam
            else transform_from_pose(parent, child, self._pose, stamp)
        )
        if self._correction is not None:
            self._correction.publish(message)
        elif self._tf is not None:
            self._tf.sendTransform(message)


def main() -> None:
    spin_main(RtabmapFrame)


if __name__ == "__main__":
    main()
