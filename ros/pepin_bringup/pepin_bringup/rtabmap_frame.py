"""RTAB-Map's loop-closed graph as ONE MORE WORD for the board's fusion, and the switch that says
when its database may learn.

ONE FRAME. RTAB-Map's optimised map frame IS ``map`` — the same frame the board's tracker, Nav2 and
the volume live in — so a localisation it publishes is already a place on our map and there is
nothing between the two. Everything that used to sit in that gap is gone: an anchor learned from
one seating, a calibration fitted over a log of pairs, a per-node table of where our map says each
database node is, and the ``map -> rtabmap`` transform they all served. Two frames were not a
problem to be solved with a better tie; two frames were the problem.

THE WORD. When an update actually RECOGNISED something — its ``/rtabmap/info`` names a node
(``loop_closure_id`` or ``proximity_detection_id``) and ``/rtabmap/localization_pose`` of the same
update carries the same header stamp — that pose goes out AS IT IS on
:data:`MEASUREMENT_TOPIC` (:class:`pepin.measurements.RemoteMeasurement`, source ``graph``) for the
board to weigh like any other source. An update that names NO node produces no word at all, and
that is measured, not cautious: ``/rtabmap/localization_pose`` is published on EVERY update,
recognised or not (2026-09-18), and without a named node it is odometry wearing the graph's
coordinates — parked, it carried 1.14 m of accumulated ``Loop/MapToBase_lin_std`` over 11 m of
"travel" that was visual-odometry jitter at rest.

WHAT THE WORD IS WORTH is RTAB-Map's own covariance from that message, floored by what this source
was last measured to be worth against the lidar (:data:`pepin.measurements.GRAPH_FLOOR_XY_M`,
0.20 m / 8 deg) and widened by the scatter of the last words
(:class:`pepin.graphtrust.Agreement`). Its ``fit`` is that scatter read as a trust: how well the
last ten words inside ten seconds track the tracker's own pose carried forward by odometry between
them. That is what tells a word riding a broken frame from one riding a good frame WITHOUT waiting
for the next closure — tape 0333 of 2026-09-15, where nineteen words about a metre and 150 degrees
out rode with fit 1.00 and the pose flew.

THE STAMP is the moment the PICTURE was taken (``word_at_picture_time``): the localisation's own
stamp, which under the snapshots is the board's clock (the picture's X-Timestamp travels through
the depth, the snapshot and RTAB-Map's update unchanged), with ``odom -> base_link`` looked up AT
that moment. The board's tracker carries the word over its odometry from there to its update
(``carry_stale_words``). Until 2026-09-19 the word was stamped with the board's NEWEST odometry
stamp instead — right on 2026-09-17, when a localisation carried the laptop's clock, minutes from
the board's ("graph stale 128.5 s") — and that said "the cart is HERE NOW" about a picture
0.1-1.4 s old: nothing on a parked cart, and on a cart turning at 25 deg/s a heading 12-35 deg
behind the truth (tapes 0386 and 0388: the heading error of EVERY word taken in a turn had the
sign of minus the turn rate; tape 0388 ended in a chair).

A WORD IS REFUSED IN 3 DOF, not by position alone: the squared Mahalanobis distance between the
word and the tracker's odometry-carried belief, under the sum of their covariances, against
:data:`pepin.fusion.GATE` — the very gate the board's information filter applies next
(:func:`pepin.fusion.disagreement`). A word at the right place facing 90 degrees the wrong way is
refused here, which the old metres-only test could not do. What the gate refuses goes out on the
whole-map CANDIDATE channel instead (:data:`CANDIDATE_TOPIC`, ``graph_candidates``), the same door
the lidar's own search uses: a measurement corrects a pose that is nearly right, a candidate MOVES
a pose that is simply in the wrong place, which is what a carry leaves behind.

THE MEMORY MODE. Beside the word this node owns one switch: whether RTAB-Map's database may LEARN
or only RECOGNISE (``graph_memory``, :class:`pepin.graphmode.ModeRule`) — a sharp pose that does not
come out of the database itself, calling RTAB-Map's own set_mode services on a change of verdict
that has held, and carrying the parameters each mode needs with it (:data:`MODE_PARAMETERS`).

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

In online SLAM (``slam`` on, ros/laptop.sh vslam --slam) RTAB-Map IS the map and its correction is
literally ``map -> odom`` — but it must become a transform ON THE BOARD, where Nav2 and the
reflexes look it up, and ``/tf`` crosses the bridge board -> laptop only (a topic allowed as a
publisher on both sides loops until nothing crosses at all). So the correction travels as a message
on ``/map_odom`` and pepin_bringup.slam_frame broadcasts it there. Beside a known map this node
broadcasts no transform at all: there is no second frame to tie.
"""

from __future__ import annotations

import json
import math
from dataclasses import replace
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

from pepin.deployment import DEFAULT_LOCALIZER
from pepin.flags import Flag, FlagSet
from pepin.fusion import (
    GATE,
    SIGMA_XY_M,
    SIGMA_YAW_DEG,
    Matrix,
    PoseMeasurement,
    disagreement,
)
from pepin.graphmode import (
    ALWAYS_LOCALISE,
    ALWAYS_MAP,
    BY_TRUST,
    SHARP_SIGMA_DEG,
    SHARP_SIGMA_M,
    STRATEGY_ICP,
    ModeRule,
    StrategyRule,
    describe_sigma,
    seating_refusal,
)
from pepin.graphtrust import HIGHEST_HYPOTHESIS, Agreement, stat
from pepin.measurements import (
    GRAPH_FLOOR_XY_M,
    GRAPH_FLOOR_YAW_DEG,
    RemoteMeasurement,
    compose,
    graph_measurement,
    inverse,
)
from pepin.odometry import Pose2D
from pepin.snapshot import SnapshotState
from pepin.sources import GRAPH, LIDAR
from pepin.tsdf import RigidPose
from pepin.watch import SIGMA_TOPIC, Preflight, Sigma, source_words
from pepin.watchdog import GlobalCandidate, same_place
from pepin_bringup.msgs import (
    map_id,
    pose_from_transform,
    stamp_from_seconds,
    stamp_seconds,
    transform_from_pose,
    yaw_of,
)
from pepin_bringup.node_kit import Switches, TfLookup, bridged_qos_profile, spin_main

RATE_HZ = 10.0
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
# What the graph has RECOGNISED on each update: the node a loop closure or a proximity link
# matched, which is the one thing that says this update is a localisation and not odometry — and
# the statistics beside it, read for the report line alone (how close the last hypothesis came).
INFO_TOPIC = "/rtabmap/info"
# Where RTAB-Map says the cart is IN THE DATABASE IT LOADED, with its own covariance. This is the
# one number in the stack that is a localisation rather than an integration: /rtabmap/mapGraph
# carries the correction relative to the CURRENT SESSION, which starts at zero on every restart
# and is pure odometry between closures — 42.5 m of it had accumulated on 2026-09-17, and the word
# built from it stood 3.6 m from the tracker and drove the cart at a person. Measured parked the
# same evening (scratch/rtabmap_frame_drift.py): 33 localisations spread 0.5 cm, 0.2 cm, 0.7 deg.
LOCALIZATION_TOPIC = "/rtabmap/localization_pose"
TRACKER_POSE_TOPIC = "/tracker_pose"
TRACKED_MAP_TOPIC = "/map_tracked"  # pepin_bringup.relocalizer publishes it, latched
# RTAB-Map's grid as RTAB-Map publishes it (its "map" output, remapped in vslam.launch.py), and
# the ONE map topic this node relays it onto once the grid is the map (grid_needs_tie).
GRID_TOPIC = "/rtabmap/grid"
MAP_TOPIC = "/map"
# RTAB-Map numbers nodes from 1 and a loaded database continues its own numbering, so a start
# whose first node is 1 loaded nothing: its grid is its own and there is no older map to tie to.
FIRST_ID_OF_AN_EMPTY_DATABASE = 1
FIT_TOPIC = "/localization_fit"
ODOM_FRAME, BASE_FRAME = "odom", "base_link"
# The board's own account of who is driving its tracker, which is what decides whether RTAB-Map may
# LEARN: the holder is read off it name-free (pepin.watch.source_words, Preflight.holding — the
# enabled source the cart has driven the least since its word), so no rule here spells "lidar" and a
# stereo matcher good to a few cm will teach the database the day it exists.
SOURCES_TOPIC = "/localization/sources"
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
# What makes the tracker's belief sharp enough to teach the database from: a lidar scan matched this
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
# How far apart in time the ``/rtabmap/info`` and the ``/rtabmap/localization_pose`` of ONE update
# may be for the two to be paired: they are published from one update and carry one stamp, so this
# covers float rounding and a build that stamps one of the two with its publish time. Past it they
# are two different updates and no node is named for that localisation.
UPDATE_MAX_SKEW_S = 0.5
# How long a word waits for odom -> base_link AT ITS PICTURE'S MOMENT. The picture is 0.13-1.35 s
# old when its localisation arrives and /tf crosses the bridge in well under 0.1 s, so the edge is
# almost always there already; the wait covers the youngest words and no more.
WORD_TF_WAIT_S = 0.2

FLAGS = FlagSet(
    Flag(
        "slam",
        False,
        description="RTAB-Map is the map (online SLAM): its correction is map -> odom and goes to"
        " the board as a message on /map_odom, where pepin_bringup.slam_frame broadcasts it; off,"
        " the board's tracker owns map -> odom and this node broadcasts no transform at all",
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
        "graph_measurement",
        True,
        description="publish where RTAB-Map's graph localised the cart as a measurement on"
        f' {MEASUREMENT_TOPIC} (source "graph") once per RECOGNISED update, for the board\'s'
        " fusion to weigh like any other word; off, the graph's answer stays on this laptop and"
        " nothing reaches the pose",
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
        " there is not yet one accepted correction to judge this word on",
        on_when="always beside a known map: the graph's loop-closed answer is the one thing in the"
        " stack that can undo accumulated drift",
        off_when="to watch a session's words in the report line and on the recorded topic before"
        " they are allowed to move the pose",
    ),
    Flag(
        "graph_candidates",
        True,
        description="a graph word the board's fusion cannot act on goes out as a whole-map"
        f' CANDIDATE on {CANDIDATE_TOPIC} (source "graph", the same covariance, at most one per'
        " recognised update): a word refused as disagreeing with the tracker's belief past the"
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
        off_when="if a graph candidate is ever seen breaking the lidar's own re-seed streak (the"
        " board's gate holds one run, and candidates of two sources alternating end each"
        " other's)",
    ),
    Flag(
        "graph_memory",
        BY_TRUST,
        choices=(BY_TRUST, ALWAYS_MAP, ALWAYS_LOCALISE),
        description="who decides whether RTAB-Map's database may LEARN beside a known map."
        " trust: this node switches it live on the rule 'a sharp pose that does not come from the"
        " database itself' — the tracker's seating under graph_memory_sigma_m /"
        " graph_memory_sigma_deg,"
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
        " evening. ALWAYS LOCALISING can never learn a new room. The rule is the same one the"
        " volume's painting follows: teach only from a pose worth teaching from, and never from"
        " the pupil — a mono camera-only pose held by graph words sits at a sigma around 20 cm and"
        " fails the seating test by itself, with nothing naming it, while a lidar-held seating"
        " passes at 1-2 cm",
        on_when="trust always, beside a known map: it is what lets one launch both wake up in a"
        " known room and extend the map when the lidar is there to teach it",
        off_when="map while deliberately extending a database by hand with the lidar known good;"
        " localise to freeze a database completely (a session where the file must not change)",
    ),
    Flag(
        "graph_memory_sigma_m",
        SHARP_SIGMA_M,
        range=(0.0, 1.0),
        description="the widest the tracker's own error bar may be, metres per position axis"
        " (the roots of the covariance /tracker_pose carries, which is the lidar's score peak),"
        " for that pose to be worth TEACHING the database from (graph_memory trust); a softer"
        " seating leaves RTAB-Map localising. 1.0 lets anything teach, which is the behaviour of"
        " before 2026-09-14",
        why="0.03, because a fit is not an error bar: at home (the charger, along a sofa) the"
        " lidar's seatings spread up to 55 cm in y within minutes at fit 0.67-0.79 — the scan is"
        " pinned in one axis there — and a database taught from one of those carries that error"
        " into every word it later says. 3 cm is where the gate starts to be a gate:"
        " over tapes 0293-0298 the worse of the two position sigmas has a median of 1.50 cm and"
        " a p90 of 3.18 cm, so this refuses the worst 11 % of seatings, while 1 cm would refuse"
        " 79 % and the database would never learn a thing",
        on_when="always: what the database is taught is baked into every word it says afterwards",
        off_when="raise it (to 1.0) only to extend a database in a room where no seating is ever"
        " sharp — and then read the sigmas the report line prints before believing a word",
    ),
    Flag(
        "graph_memory_sigma_deg",
        SHARP_SIGMA_DEG,
        range=(0.0, 180.0),
        description="the same gate for heading, degrees: the database is taught only from a"
        " seating whose heading sigma is at most this",
        why="1.0, because a heading error rotates the whole graph about the cart: over tapes"
        " 0293-0298 the lidar's heading sigma at a sharp seating is 0.06-1.14 deg (median 0.4),"
        " so this is the loose end of what the peak reports when it is pinned at all, and one"
        " degree over the 4 m of the flat is 7 cm at the far wall",
        on_when="always, with graph_memory_sigma_m: a seating sharp in x and y and free in heading"
        " is a cart that knows where it stands and not which way it faces",
        off_when="raise it only with graph_memory_sigma_m, and for the same reasons",
    ),
    Flag(
        "registration_follows_snapshots",
        True,
        description="RTAB-Map's Reg/Strategy follows what the snapshots carry"
        f" ({SNAPSHOT_STATE_TOPIC}): a scan in them means ICP (1), no scan means visual (0),"
        " switched live through the node's own parameter path on a change that has held for the"
        " hold the state carries. Off, the strategy stays whatever the launch table set and this"
        " node only reports what it would have asked for",
        why="on, because under ICP a camera-only cart cannot localise AT ALL, and that is"
        " measured rather than reasoned: in a minute of camera-only snapshots on 2026-09-18"
        " RTAB-Map logged 28 'Missing visual features or missing raw data to compute them' and 56"
        " 'Requested laser scan data, but the sensor data doesn't have laser scan', and not one"
        " update named a node. The strategy is one object for the process (the pipeline is deleted"
        " and re-created when the parsed value differs from the one in hand,"
        " rtabmap/core/Memory.cpp:721-731), so one table cannot serve a node with a scan and a node"
        " without one — and which a node has is now data, not config. What is NOT measured yet is"
        " that strategy 0 makes a camera-only link on THIS database: that is the live check",
        on_when="always beside a known map, and above all in a camera-only test — it is the whole"
        " difference between a camera that recognises a place and one that can act on it",
        off_when="to reproduce the stage-1 behaviour (ICP throughout) under the same snapshots, or"
        " if a live switch is ever seen to cost RTAB-Map its working memory",
    ),
    Flag(
        "word_at_picture_time",
        True,
        description="a graph word is stamped with the moment its PICTURE was taken — the"
        " localisation's own stamp, the board's clock under the snapshots — and odom -> base_link"
        " is looked up at that moment; the board carries the word to its update over its odometry"
        " (relocalizer carry_stale_words). Off, the word is stamped with the newest odom ->"
        " base_link stamp heard, as it was until 2026-09-19",
        why="on, measured 2026-09-19. A localisation is published 0.13-1.35 s after its picture"
        " (median 0.93 s, scratch/word_stamp_vs_board_now.py: the depth network and RTAB-Map's"
        " update), on the board's clock. Stamped 'now', every word taken in a turn was behind the"
        " truth by the turn rate times that age: tape 0388 -30.4 and -35.5 deg at +22 and +26"
        " deg/s,"
        " tape 0386 nine of nine turning words with the sign of minus the turn rate (+20 deg at -25"
        " deg/s ... -24 deg at +22 deg/s) and under 2 deg on the straights"
        " (scratch/tape_0388_word_stamp_latency.py). The camera's own stamp is good to 0.05 s"
        " against the gyro (scratch/camera_stamp_vs_gyro_lag.py), so the age is this pipeline's"
        " and nothing else's. The fusion took the -30 deg word (sigma 8 deg on both sides of the"
        " gate) and camera-only tape 0388 drove 0.5-0.7 m off its pose into a mapped obstacle",
        on_when="always under the snapshots (sensor_pack), where the picture's stamp is the"
        " board's",
        off_when="only in an arrangement whose localisations are NOT on the board's clock"
        " (sensor_pack:=false with laptop-stamped pictures): there the stamped lookup finds no"
        " odometry, the report counts the words as 'without odometry', and this switch is the way"
        " back to the old stamp",
    ),
    Flag(
        "grid_needs_tie",
        True,
        description=f"RTAB-Map's grid ({GRID_TOPIC}) is relayed onto {MAP_TOPIC} — the one map the"
        " board's tracker adopts — only once this start has recognised a node of the database it"
        " LOADED (or loaded none), and only grids stamped after that recognition. Until then the"
        " board keeps the map it cached. Off, every grid is relayed as it comes",
        why="on, measured 2026-09-19: before its first recognition RTAB-Map's graph is the current"
        " node ALONE (1 node against 254 loaded) and its grid is that node's one scan drawn where"
        " the ODOMETRY puts the cart. The tracker adopted it, matched the live scan on a picture of"
        " itself (fit 1.00, the whole-map search agreeing) and stood 1.26 m from where the graph"
        " and RTAB-Map's own scan registration put the cart; its cache kept the picture across"
        " restarts (scratch/scan_at_two_poses.py, scratch/grid_alone.py). After the first"
        " recognition the grid came back as the room (216x152), the tracker re-seated on it at fit"
        " 0.89 and RTAB-Map's word landed 1 cm from it",
        on_when="always: a grid that is not tied to the loaded graph is not the map, whatever"
        " frame id it carries",
        off_when="to reproduce the self-matching tracker of 2026-09-19, or to watch the raw grid"
        " reach the board while debugging the bridge",
    ),
)


def _zero_stamp() -> Any:
    """A builtin_interfaces/Time at zero, for a message that carries no header at all."""
    from builtin_interfaces.msg import Time as TimeMsg

    return TimeMsg()


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


class RtabmapFrame(Node):
    """Publishes the graph's localisation as a measurement, and owns RTAB-Map's memory mode."""

    def __init__(self) -> None:
        super().__init__("rtabmap_frame")
        # WHICH ROLE THIS NODE IS IN (PEPIN_LOCALIZER, pepin.deployment.localizer; the launch
        # passes the resolved word). Declared BEFORE the switches: rclpy runs their callback on
        # declarations too, and a name outside the flags table is refused there
        # (node_kit.Switches).
        #   "tracker": the graph is one voice among the board tracker's sources — a measurement
        # per recognised update, a whole-map candidate where the fusion cannot act on one — and it
        # moves RTAB-Map's memory mode live on trust in that tracker's pose.
        #   "rtabmap": THIS half owns map -> odom and the board runs no tracker, so there is
        # nobody to speak to: no measurement, no candidate, and the memory mode is pinned to
        # localising (vslam.launch.py's rtabmap_memory says why — a restart in mapping mode opens
        # a session per start and the published grid is the current node's component of working
        # memory, which is how one evening's restarts moved the map thirty times). What this node
        # still does is the relay that makes the map a map: /rtabmap/grid -> /map behind
        # grid_needs_tie.
        self._localizer = str(self.declare_parameter("localizer", DEFAULT_LOCALIZER).value)
        self._to_the_board = self._localizer == "tracker"
        self._switches = Switches(self, FLAGS)
        self._slam = self._switches.on("slam")
        self._pose = RigidPose(np.eye(3), np.zeros(3))  # the SLAM correction, map -> odom
        self._belief: Pose2D | None = None  # what the board's tracker says, and when
        self._belief_stamp = 0.0
        # ...and how sharply it says it: the roots of the covariance diagonal (m, m, rad), which
        # with covariance=peak is the lidar score peak's own width per axis — the one thing that
        # tells a seating pinned in both axes from one free to slide along a sofa.
        self._belief_sigma: tuple[float, float, float] | None = None
        self._belief_at = -math.inf  # ...and when that belief reached this node, by our clock
        self._map_id = ""  # the map the belief is on; the board refuses a word about another
        # Whether the database may LEARN, decided by trust in the pose and not by a sensor's name
        # (pepin.graphmode.ModeRule). The hold is the seating's own freshness window: a verdict is
        # acted on once it has survived as long as the evidence it rests on takes to refresh.
        # Who decides RTAB-Map's memory mode. Under "rtabmap" it is pinned to localising whatever
        # the flag says — the trust rule reads the board tracker's seating, and there is no
        # tracker — so the flag stays reachable and is simply not the authority in that role.
        mode_rule = str(self._switches["graph_memory"]) if self._to_the_board else ALWAYS_LOCALISE
        self._mode = ModeRule(FIT_FRESH_S, mode_rule, GRAPH)
        self._holder: str | None = None  # who /localization/sources says is holding the pose
        self._holder_at = -math.inf  # ...and when that report arrived, by our clock
        self._mode_pending: Any = None  # a switch the service has not answered yet
        self._mode_failed = 0  # switches the service refused or never answered
        # ...and the other live switch: which registration RTAB-Map runs, decided by what the
        # snapshots carry (pepin.graphmode.StrategyRule). It starts at the strategy the launch table
        # set, so the rule asks for nothing until it has a reason to.
        self._strategy = StrategyRule(STRATEGY_ICP)
        self._snapshots: SnapshotState | None = None  # the last state sensor_pack published...
        self._snapshots_at = -math.inf  # ...and when it reached us, by our clock
        self._strategy_failed = 0  # strategy switches the parameter path could not take
        # THE TWO HALVES OF ONE UPDATE. The node a /rtabmap/info named and that message's stamp;
        # RTAB-Map's own localisation, its stamp and the planar 3x3 it measured. A word is made when
        # the two carry the same stamp, once (`_spent`): an update recognised nothing is silence.
        self._matched: tuple[int, float] | None = None
        self._localized: tuple[Pose2D, float, Matrix] | None = None
        self._spent: float | None = None  # the stamp of the update a word was already made from
        self._infos = 0  # /rtabmap/info messages consumed...
        self._named = 0  # ...of which this many recognised a database node
        self._first_ref: int | None = None  # the first node id this start created
        self._tied = False  # whether this start has recognised a node of the loaded database
        self._tied_stamp = 0.0  # the stamp of the update that tied it: older grids are not the map
        self._grids_relayed = 0
        self._grids_withheld = 0
        self._localizations = 0  # localisations heard
        self._hypothesis = 0.0  # how close the last update came to recognising something
        self._node = 0  # the node the last word was hung on, for the report line
        self._words = 0  # words made; also the candidate's scan id (one update, one piece)
        self._sent = 0  # ...of which this many went out as measurements
        self._refused = 0  # ...and this many the 3-DOF gate refused as another pose entirely
        self._blind = 0  # ...and this many found no odom -> base_link to stamp themselves by
        self._proposed = 0  # ...and this many were offered as whole-map candidates instead
        self._word: Pose2D | None = None  # the last place the graph put the cart, on the map
        self._sigma_m = 0.0  # ...what it claimed in metres
        self._gap_m = 0.0  # ...and how far it was from the tracker's own belief
        self._gap_sigmas = 0.0  # ...in metres, and in sigmas of the two covariances together
        # How well the words that ride track the tracker's odometry-propagated pose: the window the
        # word's own fit is made of, and the scatter its covariance floor is widened by.
        self._agreement = Agreement()
        self._belief_odom: Pose2D | None = None  # the odometry at the belief's own moment...
        self._word_odom: Pose2D | None = None  # ...and at the word's, to carry the belief over
        self._fit, self._fit_at = 0.0, -math.inf  # the lidar's own fit, and when it last spoke
        self._sigma_xy: float | None = None  # the tracker's own post-fusion spread, where it says
        self._lookup = TfLookup(self)  # the EKF's odom -> base_link, for the board's own clock
        self._tf = TransformBroadcaster(self) if self._slam else None
        self._correction = (
            self.create_publisher(TransformStamped, CORRECTION_TOPIC, 5) if self._slam else None
        )
        # The two channels into the board's fusion. Not created in SLAM (the graph IS the map
        # there) and not under "rtabmap" either: there is no tracker on the board to weigh a
        # word, so a publisher here would be a topic nobody reads and a report line claiming a
        # conversation that is not happening.
        self._measurement = (
            self.create_publisher(String, MEASUREMENT_TOPIC, bridged_qos_profile(MEASUREMENT_TOPIC))
            if self._to_the_board and not self._slam
            else None
        )
        # Depth 1, RELIABLE: the QoS both ends of this channel already ask for
        # (pepin_bringup.laptop_localizer's publisher, pepin_bringup.relocalizer's subscription).
        # A candidate is a snapshot of a moment and only the newest is worth judging, and two
        # publishers on one bridged topic must declare one QoS or the route's is decided by a
        # race (pepin.deployment.BRIDGED_QOS' reason, measured on /imu/data_raw 2026-09-13).
        self._candidate = (
            self.create_publisher(
                String,
                CANDIDATE_TOPIC,
                QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
            )
            if self._to_the_board and not self._slam
            else None
        )
        # The correction, and in SLAM only: it IS map -> odom there. Beside a known map nothing
        # here reads the graph's global frame at all.
        if self._slam:
            self.create_subscription(MapGraph, "/rtabmap/mapGraph", self._on_graph, 5)
        self.create_subscription(
            PoseWithCovarianceStamped, LOCALIZATION_TOPIC, self._on_localization, 5
        )
        self.create_subscription(Info, INFO_TOPIC, self._on_info, 5)
        latched = QoSProfile(
            depth=1,
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
        )
        # The map the tracker is ON, republished by the board itself (relocalizer's
        # TRACKED_MAP_TOPIC): its id is by construction the id the board's gates expect on a
        # word, whichever source the tracker adopted. /map is still heard because a served file
        # (SLAM modes, the first boot of a room) speaks there and an older board publishes
        # nothing else.
        self.create_subscription(OccupancyGridMsg, TRACKED_MAP_TOPIC, self._on_map, latched)
        self.create_subscription(OccupancyGridMsg, MAP_TOPIC, self._on_map, latched)
        # RTAB-Map's grid passes through here on its way to being THE map (grid_needs_tie). This
        # subscription is also what keeps RTAB-Map assembling a grid at all: it builds one only
        # while somebody listens (MapsManager's subscription-count gate).
        self._map_pub = self.create_publisher(OccupancyGridMsg, MAP_TOPIC, latched)
        self.create_subscription(OccupancyGridMsg, GRID_TOPIC, self._on_grid, latched)
        self.create_subscription(
            PoseWithCovarianceStamped, TRACKER_POSE_TOPIC, self._on_tracker_pose, 5
        )
        self.create_subscription(Float32, FIT_TOPIC, self._on_fit, 5)
        self.create_subscription(String, SIGMA_TOPIC, self._on_sigma, 5)
        self.create_subscription(String, SOURCES_TOPIC, self._on_sources, 5)
        # Latched, matching sensor_pack's own publisher: this node may start after it, and the
        # present state must not have to wait for the next change to arrive.
        self.create_subscription(String, SNAPSHOT_STATE_TOPIC, self._on_snapshots, latched)
        # The mode services: this node owns the switch beside a known map, and touches neither in
        # SLAM, where the database IS the map being built.
        self._modes = (
            {}
            if self._slam
            else {
                True: self.create_client(Empty, MAPPING_SERVICE),
                False: self.create_client(Empty, LOCALISATION_SERVICE),
            }
        )
        self._tuner = (
            None
            if self._slam
            else self.create_client(SetParameters, f"{RTABMAP_NODE}/set_parameters")
        )
        self._reread = (
            None if self._slam else self.create_client(Empty, f"{RTABMAP_NODE}/update_parameters")
        )
        self.create_timer(1.0 / RATE_HZ, self._publish)
        if self._slam:
            where = f"map -> odom on {CORRECTION_TOPIC}, for the board"
        elif self._to_the_board:
            where = f"the graph's localisations on {MEASUREMENT_TOPIC}, in the one map frame"
        else:
            where = (
                f"{GRID_TOPIC} -> {MAP_TOPIC} and nothing else (localizer rtabmap: RTAB-Map owns"
                " map -> odom, no tracker on the board fuses a word), memory pinned to localising"
            )
        self.get_logger().info(
            f"rtabmap frame up: {where}; a word per RECOGNISED update ({INFO_TOPIC} names a node"
            f" and {LOCALIZATION_TOPIC} carries its stamp);"
            f" localizer={self._localizer}; flags: {self._switches.state(live_only=False)}"
        )
        self.create_timer(30.0, self._report)

    def _report(self) -> None:
        """Every 30 s: how many updates arrived and how many of them recognised a node, how many
        words were made, sent, refused by the 3-DOF gate or offered as candidates instead; then the
        last word — where it put the cart, what it claimed and how far it was from the tracker in
        metres and in sigmas — the agreement window the fit is made of, how close the graph came to
        recognising anything, RTAB-Map's memory mode and the switches."""
        word = (
            "none yet"
            if self._word is None
            else f"node {self._node} -> ({self._word.x:+.2f}, {self._word.y:+.2f},"
            f" {math.degrees(self._word.theta):+.1f} deg) +- {self._sigma_m * 100:.0f} cm,"
            f" {self._gap_m * 100:.0f} cm / {self._gap_sigmas:.1f} sigmas from the tracker"
        )
        self.get_logger().info(
            f"rtabmap frame: {self._infos} updates, {self._named} recognised a node,"
            f" {self._localizations} localisations heard, {self._words} words"
            f" ({self._sent} sent, {self._refused} refused by the gate, {self._proposed}"
            f" candidates, {self._blind} without odometry);"
            f" last word {word}; fit {self._trust_now():.2f} ({self._agreement_text()});"
            f" hypothesis {self._hypothesis:.2f}; rtabmap memory {self._mode_text()};"
            f" rtabmap registration {self._strategy_text()};"
            f" map {self._map_id or 'unknown'}, {self._grids_relayed} grids relayed,"
            f" {self._grids_withheld} withheld ({self._tie_text()});"
            f" role {self._role_text()};"
            f" flags: {self._switches.state(live_only=False)}"
        )

    def _role_text(self) -> str:
        """Which localiser arrangement this node is serving, for the report line: whether the
        graph's words go to a tracker on the board, or this half owns the frame and the words
        stay home."""
        if self._slam:
            return "slam (this node broadcasts map -> odom from the graph)"
        if self._to_the_board:
            return "tracker (the board owns map -> odom; measurements and candidates go to it)"
        return "rtabmap (RTAB-Map owns map -> odom; no words leave this laptop)"

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
        return f"{self._strategy.text()}; {said}" + (
            f", {self._strategy_failed} switches the parameter path could not take"
            if self._strategy_failed
            else ""
        )

    def _mode_text(self) -> str:
        """RTAB-Map's memory mode for a report line: the rule's own verdict, and the switches the
        service would not take."""
        return self._mode.text() + (
            f", {self._mode_failed} switches the service could not take"
            if self._mode_failed
            else ""
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

    def _now(self) -> float:
        """This node's clock in seconds: when a fit arrived, how long a verdict has held."""
        return stamp_seconds(self.get_clock().now().to_msg())

    # ---- inputs --------------------------------------------------------------------------
    def _on_tracker_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """The board's belief: the pose a graph word is checked against — and its error bar, which
        is what decides whether the database may be taught from this moment at all. The EKF's
        odometry is read at the same moment, so that a belief can be carried forward to a word's
        stamp (:meth:`_predicted`)."""
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

    def _on_fit(self, msg: Float32) -> None:
        """How well the lidar's last scan matched the map, and when: what says the tracker's
        belief is sharp enough to teach the database from."""
        self._fit, self._fit_at = float(msg.data), self._now()

    def _on_sigma(self, msg: String) -> None:
        """The tracker's own post-fusion spread: what its pose is worth whatever sensor held it."""
        sigma = Sigma.from_json(msg.data, age_s=0.0)
        self._sigma_xy = None if sigma is None else sigma.xy_m

    def _on_map(self, msg: OccupancyGridMsg) -> None:
        """The map a word is about, named the way the board names it (size@origin, msgs.map_id):
        the tracker refuses a word about another map, and a frame id is not a map id."""
        self._map_id = map_id(msg)

    def _tie_text(self) -> str:
        """One phrase for the report: whether this start stands in the map it loaded."""
        if self._first_ref is None:
            return "no update heard yet"
        if self._first_ref == FIRST_ID_OF_AN_EMPTY_DATABASE:
            return "nothing was loaded: the grid is this start's own map"
        if self._tied:
            return "this start is tied to the loaded map"
        return "this start has not recognised a node of the loaded map yet"

    def _grid_is_the_map(self, stamp: float) -> bool:
        """Whether a grid RTAB-Map stamped at ``stamp`` is assembled from the map it loaded: this
        start is tied to the loaded graph and the grid is not older than the tie, or nothing was
        loaded and the grid is this start's own map."""
        if self._first_ref == FIRST_ID_OF_AN_EMPTY_DATABASE:
            return True
        return self._tied and stamp >= self._tied_stamp

    def _on_grid(self, msg: OccupancyGridMsg) -> None:
        """RTAB-Map's grid on its way to the tracker: relayed onto the one map topic when it IS
        the map, withheld while it is only this start's own scans in the odometry's frame."""
        stamp = stamp_seconds(getattr(getattr(msg, "header", None), "stamp", None) or _zero_stamp())
        if self._switches.on("grid_needs_tie") and not self._grid_is_the_map(stamp):
            self._grids_withheld += 1
            return
        self._grids_relayed += 1
        self._map_pub.publish(msg)

    def _on_info(self, msg: Info) -> None:
        """One RTAB-Map update: WHICH NODE it recognised, if any, and how close the last hypothesis
        came.

        The matched node is the whole of "is this a localisation": a closure or a proximity link
        ties the present to a node of the database, and only then is the localisation of the same
        update a place on our map rather than odometry in the graph's coordinates. The stamp is
        kept with the id because that is what pairs this message with its localisation.
        """
        self._infos += 1
        stats = dict(zip(msg.stats_keys, (float(v) for v in msg.stats_values), strict=False))
        hypothesis = stat(stats, HIGHEST_HYPOTHESIS)
        if hypothesis is not None:
            self._hypothesis = float(hypothesis)
        ref = int(getattr(msg, "ref_id", 0))
        if ref > 0 and self._first_ref is None:
            # the first node this start made: everything older is the loaded map
            self._first_ref = ref
        matched = _matched_id(msg)
        if matched <= 0:
            return
        stamp = stamp_seconds(getattr(getattr(msg, "header", None), "stamp", None) or _zero_stamp())
        if self._first_ref is not None and matched < self._first_ref and not self._tied:
            # this start has recognised a node of the map it LOADED; grids from here on are
            # assembled from the whole graph
            self._tied, self._tied_stamp = True, stamp
        self._named += 1
        self._matched = (matched, stamp)
        self._try_word()

    def _on_localization(self, msg: PoseWithCovarianceStamped) -> None:
        """RTAB-Map placing itself in the database it loaded: the cart's pose in the ONE map frame,
        with the covariance RTAB-Map measured for it.

        Published on EVERY update, recognised or not (measured 2026-09-18), which is why it is only
        half of a word: :meth:`_try_word` waits for the info of the same update to name a node.
        """
        self._localizations += 1
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        place = Pose2D(
            float(p.x),
            float(p.y),
            math.atan2(2.0 * (q.w * q.z + q.x * q.y), 1.0 - 2.0 * (q.y * q.y + q.z * q.z)),
        )
        # The planar block of the 6x6 RTAB-Map filled in (x, y, yaw -> rows 0, 1, 5): what the
        # registration against the recognised node is worth, which is the word's own covariance.
        raw = list(msg.pose.covariance)
        covariance: Matrix = np.array(
            [[raw[index] for index in row] for row in ((0, 1, 5), (6, 7, 11), (30, 31, 35))],
            dtype=float,
        )
        self._localized = (place, stamp_seconds(msg.header.stamp), covariance)
        self._try_word()

    def _on_graph(self, msg: MapGraph) -> None:
        """RTAB-Map's correction, in SLAM only, where it IS ``map -> odom``: kept for the timer,
        which sends it to the board as a message (:data:`CORRECTION_TOPIC`)."""
        self._pose = pose_from_transform(msg.map_to_odom)

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

    def _on_snapshots(self, msg: String) -> None:
        """What pepin_bringup.sensor_pack's snapshots carry right now: the evidence RTAB-Map's
        registration strategy follows, and the hold that change must survive (the packer's own
        liveness window, carried in the message). A message that does not parse says nothing and is
        ignored — the strategy in force is never changed on a reading nobody could read."""
        state = SnapshotState.from_json(msg.data)
        if state is None:
            return
        self._snapshots, self._snapshots_at = state, self._now()

    # ---- the word ------------------------------------------------------------------------
    def _try_word(self) -> None:
        """Make the word of one update, once, as soon as both halves of that update are in hand:
        an info that NAMED a node and the localisation carrying the same stamp.

        Either message may arrive first, so both handlers call this and ``_spent`` is what keeps one
        update from producing two words. A localisation whose info named nothing is simply never
        paired — the report line's ``updates`` against ``recognised a node`` is that count — and it
        is dropped the moment the next localisation replaces it.
        """
        named, localized = self._matched, self._localized
        if named is None or localized is None:
            return
        node_id, info_stamp = named
        place, localized_stamp, covariance = localized
        if abs(info_stamp - localized_stamp) > UPDATE_MAX_SKEW_S:
            return
        if self._spent is not None and localized_stamp == self._spent:
            return
        self._spent = localized_stamp
        self._offer(node_id, place, covariance, localized_stamp)

    def _offer(self, node_id: int, place: Pose2D, covariance: Matrix, taken_at: float) -> None:
        """One graph word: the localisation ``place`` — already in the map frame — published as a
        measurement for the board's fusion, or offered on the candidate channel instead.

        THE STAMP IS THE PICTURE'S MOMENT, ``taken_at`` — the localisation's own stamp, the
        board's clock under the snapshots — and the odometry the word is compared at is looked up
        AT that moment (``word_at_picture_time``; see the module docstring for what stamping it
        "now" cost). With the switch off the newest ``odom -> base_link`` stamp is used, as before
        2026-09-19.

        The word is remembered whatever the flags say — the report line is how a session is judged
        before it is allowed to move anything — and sent only with ``graph_measurement`` on, a map
        to name, and a pose the 3-DOF gate accepts: the squared Mahalanobis distance between the
        word and the odometry-carried belief, under the sum of their covariances, at most
        :data:`pepin.fusion.GATE`. That is the gate the board's information filter applies next, so
        a word this node passes is a word that can actually move the pose — and a word 90 degrees
        wrong at the right place is refused here, which a metres-only test could not do.

        What the measurement cannot carry goes out on the candidate channel instead
        (:meth:`_propose`, ``graph_candidates``): a word the gate refuses is the very word a
        carried cart needs, and a re-seed is the only mechanism that moves a pose that is not
        almost right but simply in the wrong place.

        WITH NO BELIEF AT ALL the word still goes out: that is the wake-up (a lidar-less start,
        nothing on ``/tracker_pose`` yet), and it is the one case where the graph is the only thing
        that knows where the cart is. It is safe because the word depends on nothing this session
        measured — one frame, RTAB-Map's own recognition, and its own covariance.
        """
        at_picture = self._switches.on("word_at_picture_time")
        transform = (
            self._lookup.transform(
                ODOM_FRAME, BASE_FRAME, stamp_from_seconds(taken_at), WORD_TF_WAIT_S
            )
            if at_picture
            else self._lookup.transform(ODOM_FRAME, BASE_FRAME)
        )
        odom = self._planar(transform)
        if transform is None or odom is None:
            self._blind += 1
            return
        stamp = taken_at if at_picture else stamp_seconds(transform.header.stamp)
        self._word_odom = odom
        self._words += 1
        self._node = node_id
        self._word = place
        self._sigma_m = self._word_sigma_m()
        # The GEOMETRY of the word first, with no fit on it: the fit is what the last words'
        # agreement says, and this word's own residual belongs in that window before it is read.
        # The map id is the SERVED GRID's (size@origin): it is what the board's tracker checks a
        # word against.
        remote = graph_measurement(
            place,
            stamp,
            self._map_id,
            floor_xy_m=self._sigma_m,
            floor_yaw_deg=GRAPH_FLOOR_YAW_DEG,
            measured=covariance,
        )
        predicted = self._predicted()
        self._gap_m = (
            math.hypot(place.x - predicted.x, place.y - predicted.y)
            if predicted is not None
            else math.inf
        )
        self._gap_sigmas = (
            math.sqrt(max(self._disagreement(remote, predicted), 0.0))
            if predicted is not None
            else math.inf
        )
        self._agree(remote)
        remote = replace(remote, fit=self._trust_now())
        self._propose(remote)
        if self._measurement is None or not self._switches.on("graph_measurement"):
            return
        if not self._map_id:
            return
        if predicted is not None and self._gap_sigmas**2 > GATE:
            self._refused += 1
            return
        self._sent += 1
        self._measurement.publish(String(data=remote.to_json(words=self._words, node=node_id)))

    def _word_sigma_m(self) -> float:
        """The FLOOR under what a word claims in metres: what this source was last measured to be
        worth against the lidar (:data:`pepin.measurements.GRAPH_FLOOR_XY_M`, median 19 cm over 61
        words, p90 38 cm, 2026-09-17), or the scatter of the last words when that is wider.

        The floor covers the part no word can see in itself — a whole frame sitting a little to one
        side. The scatter is the part it CAN see (:class:`pepin.graphtrust.Agreement`): when the
        words start disagreeing with the pose the odometry carries between them, the graph is
        coming apart, and the word must say so in the one language the fusion reads. Before this
        the word claimed the same 0.20 m whether it sat 1 cm from the tracker or 2.6 m (the day's
        worst), and every gate downstream — the goal's, the drive's, the volume's — believed it.

        RTAB-Map's own covariance rides ON TOP of this floor and is never replaced by it
        (:func:`pepin.measurements.graph_measurement`): a registration it measured as wide stays
        wide.
        """
        spread = self._agreement.rms(self._now())
        return GRAPH_FLOOR_XY_M if spread is None else max(GRAPH_FLOOR_XY_M, float(spread))

    def _trust_now(self) -> float:
        """What the word claims as its ``fit``: the agreement of the last words with the tracker's
        odometry-propagated pose (:class:`pepin.graphtrust.Agreement`), and 1.0 while there is no
        residual to judge it on — camera-only there is no second opinion, and "nothing here says
        this word is wrong" is the honest answer.

        Downstream that number is not a discount on the covariance — the fusion gates on chi-square
        — but it is what the board publishes as its own confidence when no scan of its own measured
        one, and it is the SCORE a candidate is admitted on (:meth:`_propose`).
        """
        return self._agreement.trust(self._now())

    def _predicted(self) -> Pose2D | None:
        """Where the tracker's belief says the cart is at the moment of the word being judged:
        the last belief carried forward by the EKF's odometry between its own stamp and the
        word's (the two odometries this node already looks up). ``None`` with no belief at all;
        the belief itself when one of the two odometries is missing — a word and a belief are a
        tenth of a second apart at 10 Hz, and 3 cm at the cart's speed."""
        belief = self._belief
        if belief is None:
            return None
        if self._belief_odom is None or self._word_odom is None:
            return belief
        return compose(belief, compose(inverse(self._belief_odom), self._word_odom))

    def _agree(self, word: RemoteMeasurement) -> None:
        """Feed one word's disagreement with the odometry-propagated belief to the agreement window
        (:class:`pepin.graphtrust.Agreement`), which is what its ``fit`` is made of: the distance in
        metres AND the 3-DOF Mahalanobis distance under the two covariances together, so a word
        facing the wrong way at the right place counts as the disagreement it is.

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
        # into the fusion (2026-09-16). The sigma is what answers there: it is the tracker's own
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

        One candidate per recognised update, carrying the word's own count as its scan id: a
        streak on the board is three DISTINCT pieces of evidence, and one update is one piece.
        The covariance is the measurement's; the board re-judges the word against its own pose
        and fit, and its rules do the rest — three agreeing candidates, no re-seed while a goal
        runs, and no re-seed from a source that is not the lidar while the lidar is alive.

        The SCORE the board judges it on is the word's own fit (the agreement), so a candidate
        whose words have been disagreeing is refused there as "unknown map"
        (:data:`pepin.watch.ADMIT_FIT`, 0.45). Sending it anyway, rather than filtering here, is
        deliberate: the board's report line then carries the refusal, and one gate judges every
        source.
        """
        if self._candidate is None or not self._switches.on("graph_candidates"):
            return
        if not self._map_id:
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
            scan_id=self._words,
            source=remote.source,
        )
        self._proposed += 1
        self._candidate.publish(
            String(
                data=candidate.to_json(
                    reason="refused as a measurement" if refused else "the tracker has no source",
                    gap_cm=None if math.isinf(self._gap_m) else round(self._gap_m * 100.0, 1),
                    gap_sigmas=(
                        None if math.isinf(self._gap_sigmas) else round(self._gap_sigmas, 2)
                    ),
                    node=self._node,
                )
            )
        )

    # ---- RTAB-Map's memory ----------------------------------------------------------------
    def _unsharp(self) -> str | None:
        """Why the tracker's present pose may not be TAUGHT to the database — one phrase for the
        log — or ``None`` when it may.

        Two things, in the order a person would ask them: is the LIDAR behind this belief at all
        (a fresh fit of at least :data:`TRUSTED_FIT`), and is its seating pinned in both axes and
        in heading (the covariance the belief carries, against ``graph_memory_sigma_m`` /
        ``graph_memory_sigma_deg``). A fit answers the first question and says nothing about the
        second: a scan sliding along a sofa matches beautifully everywhere it slides to.
        """
        if self._fit < TRUSTED_FIT or self._now() - self._fit_at > FIT_FRESH_S:
            return (
                f"the lidar is not driving the tracker (fit {self._fit:.2f}, last heard"
                f" {self._now() - self._fit_at:.1f} s ago)"
            )
        return seating_refusal(
            self._belief_sigma,
            float(self._switches["graph_memory_sigma_m"]),
            float(self._switches["graph_memory_sigma_deg"]),
        )

    def _decide_mode(self) -> None:
        """Ask RTAB-Map to learn or only to recognise, on a change of verdict that has held.

        The rule is :class:`pepin.graphmode.ModeRule`; everything this method adds is the wiring.
        A switch is not sent while the last one is unanswered — that is the "no faster than the
        service answers" rule, exact and with no number in it — and the parameters each mode needs
        (:data:`MODE_PARAMETERS`) travel with it, because the mode services do not touch them.
        """
        if not self._modes:
            return
        stale = self._now() - self._holder_at > FIT_FRESH_S
        # NOT BEFORE THIS START IS TIED TO THE MAP IT LOADED. A switch to mapping before RTAB-Map
        # has recognised one node of the loaded database opens a new map in the ODOMETRY's frame,
        # unlinked to the old one; parked, it then never processes a frame and never ties at all.
        # Live 2026-09-19: the rule's first verdict ("the pose is held by lidar") went out a second
        # after start, /rtabmap/mapGraph carried 2 nodes instead of 173, the places had no node to
        # ride and the tracker sat in odometry coordinates 1.1 m from the bookshelf it stood at.
        # (This is what pepin.graphtrust.Recognition used to say; its job outlived the anchor.)
        untied = None
        if str(self._switches["graph_memory"]) == BY_TRUST and not self._tied:
            untied = "this start has not recognised a node of the loaded map yet"
        verdict = self._mode.update(
            self._now(),
            untied
            or self._unsharp()
            or ("the board has not said who holds the pose" if stale else None),
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
        self._set_parameters(MODE_PARAMETERS[mapping])

    def _set_parameters(self, values: dict[str, str]) -> bool:
        """Set these RTAB-Map parameters on its node and have it re-read them; whether the pair of
        calls went out at all.

        THE ONLY PATH RTAB-MAP HONOURS, and every part of it is necessary. The values are STRINGS
        because rtabmap declares every one of its parameters as one and reads it back with
        ``as_string()``; ``update_parameters`` is what copies them into rtabmap's own map and hands
        the whole map to ``Rtabmap::parseParameters``, so a set alone changes nothing (there is no
        on-set callback on that node at all); and a name the LAUNCH table never overrode is accepted
        by the set and then never looked at, which is why the two parameter tables here
        (:data:`MODE_PARAMETERS`, :data:`pepin.graphmode.REGISTRATION_PARAMETERS`) name only
        parameters that table already carries. The file:line for all of it is in
        :mod:`pepin.graphmode`.
        """
        if self._tuner is None or not self._tuner.service_is_ready():
            return False
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name=name,
                value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=value),
            )
            for name, value in values.items()
        ]
        self._tuner.call_async(request)
        if self._reread is None or not self._reread.service_is_ready():
            return False
        self._reread.call_async(Empty.Request())
        return True

    # ---- RTAB-Map's registration ----------------------------------------------------------
    def _decide_strategy(self) -> None:
        """Ask RTAB-Map for the registration the snapshots need, on a change that has held.

        The rule is :class:`pepin.graphmode.StrategyRule` and the hold is the one the STATE carries
        — how long the packer itself takes to change its mind about a source — so nothing here is a
        number. A state older than its own refresh is no evidence at all and the strategy in force
        stays: sensor_pack having gone quiet is not the lidar having gone away.
        """
        if self._tuner is None or not self._switches.on("registration_follows_snapshots"):
            return
        state = self._snapshots
        fresh = state is not None and self._now() - self._snapshots_at <= state.refresh_s
        verdict = self._strategy.update(
            self._now(),
            state.refresh_s if state is not None else 0.0,
            state.carries(LIDAR) if (state is not None and fresh) else None,
            state.kind if state is not None else "",
        )
        if verdict is None:
            return
        if not self._set_parameters(verdict.parameters):
            self._strategy_failed += 1
            return
        self.get_logger().info(
            f"rtabmap registration: {verdict.text()} -> Reg/Strategy {verdict.strategy}"
            f" (set on {RTABMAP_NODE} and re-read through {RTABMAP_NODE}/update_parameters)"
        )

    # ---- outputs -------------------------------------------------------------------------
    @staticmethod
    def _planar(transform: TransformStamped | None) -> Pose2D | None:
        """A transform read in the plane the cart drives in, or ``None`` when there is none:
        the EKF's ``odom -> base_link``, both at a belief's moment and at a word's."""
        if transform is None:
            return None
        pose = pose_from_transform(transform)
        return Pose2D(
            float(pose.translation[0]),
            float(pose.translation[1]),
            math.atan2(float(pose.rotation[1, 0]), float(pose.rotation[0, 0])),
        )

    def _publish(self) -> None:
        """At :data:`RATE_HZ`: RTAB-Map's memory mode, and in SLAM the correction as a message to
        the board (held between graphs, because it does not move until the graph does). Beside a
        known map there is no transform to publish — the graph's map frame IS ``map``."""
        self._decide_mode()
        self._decide_strategy()
        if self._correction is None:
            return
        parent, child = SLAM_FRAMES
        self._correction.publish(
            transform_from_pose(parent, child, self._pose, self.get_clock().now().to_msg())
        )


def main() -> None:
    spin_main(RtabmapFrame)


if __name__ == "__main__":
    main()
