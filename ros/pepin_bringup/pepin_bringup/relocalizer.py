"""Global relocalisation on top of AMCL: the robot always knows where it is, no human needed.

AMCL tracks well but only searches where its particles already are: carry the
cart across the room, push it, lift it onto the carpet, and it keeps believing
the old pose. This node closes that gap with the correlative whole-map search
from :mod:`pepin.localization` (a fraction of a second on the pooled grid, with
the twin check). Every second it scores how well the current scan lies on the
map at the tracked pose; when the fit stays poor it searches the whole map and,
if a clearly better pose exists, re-seeds the tracker. The operator seeds it the
same way by publishing on ``/initialpose`` (Foxglove's pose estimate); the node
never publishes there itself. The same search answers the ``/relocalize``
service on demand, and ``/where_am_i`` reports pose and fit as text.

Frames: the scan is transformed into ``base_link`` with the static laser
transform looked up once; poses are in ``map``.

THE MAP IS ONE MAP, on one topic (World R): ``/map``, RTAB-Map's loop-closed occupancy grid,
published latched by the laptop and routed here (:mod:`pepin.deployment`). There is no second
picture of the room to choose between any more — no served pgm in the loop, no volume slice — so
there is no ``map_topic`` flag either. What is still a decision is WHEN a newly arrived grid is
ADOPTED (matcher, static mask and tracker rebuilt, the episode's evidence forgotten), and that is
:class:`pepin.mapping.MapChoice`'s: the first grid, and a later one only once ``map_refresh_s``
has passed and its cells have actually changed. RTAB-Map republishes at its detection rate (1 Hz
with ``map_always_update``) and re-renders the whole grid whenever a node is added or a loop
closure moves a pose by more than a centimetre (``GridGlobal/UpdateError``,
rtabmap/core/GlobalMap.cpp: the global map is cleared and re-assembled from the per-node grids),
so adopting every publication would rebuild the matcher on four A53 cores once a second and throw
away every candidate and measurement in between.

A BENT MAP IS STILL THIS ROOM. A re-render can move the grid's origin, change its size and move
walls by tens of centimetres, and it lands here as an ordinary adoption: the pose is carried
(``carry_pose_across_maps``), the matcher and the mask are rebuilt on the new cells, the map's id
changes with its geometry — which is why the laptop stamps its words with the id of
``/map_tracked``, the grid this node accepted, and not with the one it published — and the report
line says how far the origin moved and how many cells changed (:func:`pepin.mapping.map_shift`).

And with nothing live at all this node tracks on the map it wrote down itself
(:mod:`pepin.mapcache`, the ``map_cache`` flag): the board must know where it is with the laptop
off (CLAUDE.md rule 20), and the live grid replaces the cache the moment it arrives.

Sources: the lidar's revolution is matched here, on the board, through the one trigger path
(:class:`pepin.sources.SourceFeed`) — the scan waits at its gate until the odometry covers its
whole revolution and then drives an update. The camera is a source too, but its scans are
matched on the laptop that produces them (:mod:`pepin_bringup.laptop_localizer`) and only the
POSE they measured arrives here, on ``/localization/measurement``: this node carries each
measurement to the moment of its next update over the same odometry history a riding scan would
have been carried along, and the tracker fuses it by information beside the lidar's own match
(:mod:`pepin.measurements`). Matching those scans HERE is what the day of 2026-09-13 measured
and refused: three matches a revolution, 147 ms instead of 45, every second revolution dropped
(4.7 Hz) and the live pose 50 cm p90 off the lidar's truth. With the lidar stale or absent the
measurements drive the updates by themselves, so a dead lidar still hands the tracker to the
camera without a restart; with the link down there are simply no measurements and the tracker
is the lidar-only one it always was. ``/localization/sources`` carries every source's word on
each update as JSON for the operator. The watch, the whole-map search and the first fix run on
full revolutions only (:meth:`pepin.sources.SourceFeed.full_picture`).

The watchdog: the same whole-map search runs CONTINUOUSLY on the laptop, which does it in a
tenth of a second instead of seconds (:mod:`pepin_bringup.laptop_localizer`), and its answers arrive
here on ``/localization/candidate``. Each one is first carried from the moment of its own scan
to this moment over the odometry between the two (``carry_candidates``) — a search plus a
wireless hop is a quarter of a second, and on a driving cart that is the difference between
where the cart was and where it is — and then :class:`pepin.watchdog.CandidateGate` judges it
against this tracker's own pose and fit; a streak of candidates from different scans that
disagree with the tracker and agree with each other re-seeds it through the pending seed — the
very path this node's own search uses, with the ``accept_candidates`` flag as the switch.
Nothing here depends on the laptop: with no candidate arriving, the board's own slow search is
the fallback it always was.
"""

from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from pathlib import Path as FilePath  # nav_msgs' Path owns the bare name here
from typing import Any

import numpy as np
from action_msgs.msg import GoalStatus, GoalStatusArray
from geometry_msgs.msg import PoseArray, PoseStamped, PoseWithCovarianceStamped
from nav2_msgs.msg import ParticleCloud
from nav2_msgs.srv import ClearEntireCostmap
from nav_msgs.msg import OccupancyGrid as OccupancyGridMsg
from nav_msgs.msg import Odometry, Path
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.parameter_client import AsyncParameterClient
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool, Float32, String
from std_srvs.srv import Trigger
from tf2_msgs.msg import TFMessage
from tf2_ros import Buffer, TransformBroadcaster

from pepin.dynamic import STATIC_M, StaticMask, occluded
from pepin.flags import Flag, FlagSet
from pepin.fusion import (
    COVARIANCE_CHOICES,
    EKF_YAW_PER_TURN,
    ODOM_YAW_PER_TURN,
    PEAK,
    Matrix,
    published_covariance,
)
from pepin.lastpose import FrameHold, known_room, saved_start
from pepin.localization import SWITCHES as TRACKER_SWITCHES
from pepin.localization import Localizer
from pepin.mapcache import CACHE_NAME, CacheBoot, MapCache, save
from pepin.mapping import (
    MAP_FALLBACK_S,
    MAP_TOPIC,
    MapChain,
    MapChoice,
    MapShift,
    OccupancyGrid,
    map_shift,
    scan_reach_m,
    widened,
    worth_adopting,
)
from pepin.measurements import (
    MEASUREMENT_MAX_AGE_S,
    REMOTE_FLOOR_XY_M,
    REMOTE_FLOOR_YAW_DEG,
    MeasurementGate,
    RemoteMeasurement,
    remote_update,
)
from pepin.mounts import load_lidar
from pepin.odometry import Pose2D, RunawayWatch, wrap_angle
from pepin.scanmatch import CorrelativeMatcher, SearchWindow
from pepin.slip import (
    PictureSlip,
    PictureSlipVerdict,
    PictureSpeed,
    SlipWatch,
    zero_twist_covariance,
)
from pepin.sources import (
    CAMERA,
    CONTACT,
    DEPTH,
    GRAPH,
    LIDAR,
    ScanObservation,
    SourceFeed,
    SourceRegistry,
)
from pepin.timeline import (
    MatchPacer,
    MotionEdge,
    MotionFilter,
    OdomHistory,
    TimedScan,
    deskewed,
    standing_still,
    timed_scan_from_ros,
)
from pepin.watch import (
    DRIVE_FIT,
    DRIVE_SIGMA_M,
    LOST_FIT,
    LOST_SIGMA_M,
    SIGMA_TOPIC,
    SOURCE_PATIENCE_S,
    JumpClear,
    LostWatch,
    PoseSpread,
    Sigma,
    SourceSilence,
    Verdict,
)
from pepin.watchdog import CANDIDATE_STREAK, CandidateGate, GlobalCandidate
from pepin_bringup.msgs import (
    grid_from_msg,
    map_digest,
    map_id,
    planar_mount,
    pose_with_matrix,
    transform_from_rpy,
    yaw_of,
)
from pepin_bringup.node_kit import Switches, TfLookup, bridged_qos_profile, spin_main

DUMP_DIR = (
    "/maps/rec"  # every failed whole-map search leaves its scan here, for the offline autopsy
)
LAST_POSE_FILE = "/maps/last_pose.json"  # where the robot stood when the stack last ran
# ...and where the MAP it stood on is kept, by this node, for the next cold boot: the directory the
# cache file lives in (pepin.mapcache.CACHE_NAME). The board is the one holder of its own map — no
# pgm in the loop and nothing exported from the laptop — so this file is what a boot with the laptop
# down has to track on.
MAP_CACHE_DIR = "/maps"
# ...and how often it may be rewritten. THE CACHE IS A COLD-BOOT FALLBACK AND NOTHING ELSE: its only
# reader is a board that starts with no live map, and a picture a minute old is worth as much there
# as one a second old — the live grid arrives within a second of the routes coming up, and a room
# does not change in a minute. The card is what decides the number: one write is 16 kB run-length
# encoded (this flat's 51385 cells, scratch/costmap_rle_cost.py), so a write per adoption under a
# live graph is 8 kB/s — 690 MB a day of wear for a file nobody reads. At one a minute it is 23 MB a
# day and the cache is at most a minute behind the map. The first adoption is always written: a
# board with no cache at all is the one case where being a minute late costs a boot.
MAP_CACHE_MIN_GAP_S = 60.0
# The topic this node republishes the map it is TRACKING on, latched. Nav2's static layers read it
# (ros/params/nav2_params.yaml), so the planner's static map is the tracker's map by construction
# instead of by two subscriptions to a third party that may disagree. The name is a literal here and
# in that file, and tests/unit/test_nav_contract.py holds the two equal.
TRACKED_MAP_TOPIC = "/map_tracked"
LAST_POSE_MAX_AGE_S = 3600.0
SLIP_SAID_AFTER = 3  # consecutive slipping scans before the log says it once
# The laptop's whole-map watchdog (pepin_bringup.laptop_localizer) publishes a candidate here
# about once a second, as one self-contained JSON message (pepin.watchdog.GlobalCandidate): the
# board
# never has to join two topics to judge one answer, and a message that does not parse is counted,
# not obeyed. The board's own slow search is untouched: it is the fallback when none arrives.
CANDIDATE_TOPIC = "/localization/candidate"
# ...and on this one, several times a second, the pose it measured out of a camera scan
# (pepin.measurements.RemoteMeasurement): the same shape of message, judged the same way — a
# map id that is not ours is refused, a message that does not parse is counted, not obeyed.
MEASUREMENT_TOPIC = "/localization/measurement"
# ...and on this one, whenever RTAB-Map's pose graph moves, where THAT says the cart is
# (pepin_bringup.rtabmap_frame, flag graph_measurement): the same shape of message, judged the
# same way, but a gate of its own. The camera's gate fuses everything waiting in it into one
# word named `camera`, so a graph word dropped in there would move the pose under the camera's
# name, with the camera's health and the camera's switch.
GRAPH_MEASUREMENT_TOPIC = "/localization/graph_measurement"
# How often this tracker may adopt a newer grid on :data:`pepin.mapping.MAP_TOPIC`: what the board
# can afford, since the publisher offers one every second. An adoption costs ~65 ms on an A53 and
# the rebuild duty budget allows one every 1.9 s (scratch/map_adoption_cost.py). Derived, not
# chosen; the flag's `why` carries the arithmetic.
MAP_REFRESH_DEFAULT_S = 2.0
# Nav2's own service for emptying the rolling grid the controller steers by: the marks a
# correction stranded there are erased in one call and marked again from the next scans. The
# gap is not a flag: emptying the grid costs Nav2 that rebuild, and a second is the shortest
# spacing at which the clears stay cheaper than the phantoms they remove.
CLEAR_LOCAL_COSTMAP = "/local_costmap/clear_entirely_local_costmap"
CLEAR_MIN_GAP_S = 1.0


def map_to_odom(pose: Pose2D, odom: Pose2D) -> tuple[float, float, float]:
    """The map -> odom transform (x, y, yaw) that puts the robot, seen at ``odom`` in the odom
    frame, at ``pose`` in the map frame: T_map_odom = T_map_base * inv(T_odom_base)."""
    yaw = math.atan2(math.sin(pose.theta - odom.theta), math.cos(pose.theta - odom.theta))
    c, s = math.cos(yaw), math.sin(yaw)
    return pose.x - (c * odom.x - s * odom.y), pose.y - (s * odom.x + c * odom.y), yaw


# The tracker's flags: what applies without a restart (CLAUDE.md rule 19). Each is a switch of
# the Localizer (pepin.localization: an attribute, or the roster's ``sources``) and the flag
# callback writes it there (``Localizer.switch``); the next map's tracker is built with the
# current values. Every other parameter is refused live (the answer names it), because a
# "success" that changed nothing is a lie.
FLAGS = FlagSet(
    Flag(
        "rest_lock",
        True,
        description="hold the pose while the cart stands still (wheels quiet 0.6 s and the gyro"
        " under 1.5 deg/s): a match's residual is blended in with a time constant instead of"
        " taken whole",
        why="the best-measured switch in the tracker. On tape 0182 (6 s at rest, a full turn, 8 s"
        " at rest) the published pose's rest band goes from 8.34 deg with the lattice alone to"
        " 0.51 deg with the rest lock, and on the robot it reads 0.42 deg at sd 0.13; while"
        " driving (tape 0170) it took p90 |yaw rate - gyro| from 7.74 to 4.86 deg/s and the"
        " correction's sd from 0.45 to 0.29 deg. It costs +0.04-0.08 ms a scan. Flipped live in"
        " the demo: 9 deg to 0.3",
        on_when="always: a still cart whose pose wanders is the first thing a watcher sees",
        off_when="to show what the raw match does (the demo's A/B), or where the cart is carried"
        " by hand and the wheels have no say in whether it stands still",
    ),
    Flag(
        "explained_vote",
        True,
        description="returns the static map cannot explain (a person, a moved chair) do not score"
        " the match",
        why="alone it took the rest band from 10.5 deg to 2.38, and with the rest lock and the"
        " sub-cell refinement to 0.51 (tape 0173). The mask is dropped when fewer than half the"
        " returns are explained or fewer than 60 come back, because a vote taken only on what"
        " already fits is mildly self-confirming",
        on_when="in a room with people and furniture that moves — the room this robot lives in",
        off_when="in an empty room where every return should count, or to measure what a crowd"
        " costs the match",
    ),
    Flag(
        "rest_tau_s",
        6.0,
        description="the rest lock's time constant: seconds for a residual to die at rest",
        why="6 s against the 3 s first tried: at the node's roughly 1 Hz rest cadence 3 s let"
        " 2.5x more match noise through, and 6 s halves that while still converging a nudge in"
        " seconds — the rest band reads 0.37 deg at 6 s",
        on_when="lengthen it for a cart that stands for minutes and must not drift at all",
        off_when="shorten it when a nudged cart has to take its new pose quickly — a demo where"
        " the cart is pushed by hand",
        range=(0.1, 60.0),
    ),
    Flag(
        "rest_gain",
        0.05,
        description="the rest lock's share per match when no match cadence is known",
        why="default by design, unmeasured on its own: 0.05 against the driving gain of 0.5 was a"
        " guess, kept after the robot run because the rest bands it is inside of (0.42-0.51 deg)"
        " came out right. It has never been swept",
        on_when="raise it towards the driving gain when the rest lock is too slow to accept a"
        " real correction",
        off_when="0 lets no match move the pose while the cart stands: a hard hold, and a way to"
        " see how far the odometry alone wanders",
        range=(0.0, 1.0),
    ),
    Flag(
        "sources",
        (LIDAR, GRAPH),
        description="what corrects the pose: the lidar's revolution (/scan), matched here, and"
        " the camera (`camera`), whose scans the laptop matches and whose ANSWER arrives on"
        " /localization/measurement. The lidar drives the updates while it is fresh and the"
        " camera's word rides along, carried to its moment; a stale lidar hands the updates to"
        " the measurements. `depth` and `contact` name the camera's raw scans, which this node"
        " no longer subscribes to — enabling them changes nothing here. `graph` is RTAB-Map's pose"
        " graph on the laptop, whose answer arrives on /localization/graph_measurement with a"
        " gate of its own: it rides the lidar's update, and with no scan source driving it drives"
        " one of its own exactly as the camera's word does (pepin.measurements.remote_update) —"
        " so `graph` alone is a tracker on the graph alone, and `camera,graph` is one update"
        " between the two of them, never one each",
        why="the lidar alone, because the camera cannot carry the map by itself: replayed on run"
        " 0171 against flat3 the depth band alone loses the map in 0.5 s (122 cm, 124 deg) and"
        " the contact line alone in 12 s (80 cm, 28 deg) — the camera's 0.15-1.3 m band is a"
        " different cross-section of the room than the lidar's 0.2 m map, so a look-alike place"
        " scores fit 0.90 at its own match and 0.12 at the truth. Fused with the lidar and gated"
        " on disagreement, all three together stay within 0.7/1.6/5.7 cm and 0.21/0.56/1.9 deg of"
        " lidar-only and never lose the map (scratch/camera_only_localization.py). `camera` is"
        " that same fusion with the matching moved to the laptop: on this board the raw scans"
        " took the tracker to 147 ms and 4.7 Hz and the live pose 50 cm p90 off the lidar's truth"
        " (scratch/drive_bisect.py, runs 0238-0241), while a measurement costs a matrix inverse",
        on_when="add `camera` where the lidar is blocked or blind — parked bumper to furniture,"
        " or a lidar that stopped: the fusion is measured and gated, and the board pays nothing"
        " for it. Add `graph` (e.g. lidar,graph) once the laptop's graph measurement has been"
        " watched beside /tracker_pose for a drive: a loop closure is the one correction nothing"
        " else on this robot can make",
        off_when="drop a source the moment /localization/sources shows it disagreeing with the"
        " others; the lidar alone is the safe state, and it is what the board falls back to by"
        " itself when the link dies. `depth`/`contact` stay on the roster because the library"
        " still matches those scans where there is CPU for it — an offline replay"
        " (scratch/camera_only_localization.py), another robot — not because this board will",
        choices=(LIDAR, DEPTH, CONTACT, CAMERA, GRAPH),
    ),
    Flag(
        "measurement_max_age_s",
        MEASUREMENT_MAX_AGE_S,
        description="how old a pose measurement from the laptop may be, in seconds, at the"
        " moment of the update that would take it: past this it is dropped instead of carried."
        " Read only while carry_stale_words is OFF",
        why="the number the day of 2026-09-13 asked for: the camera's word pulled the live pose"
        " 50 cm p90 off the truth while the board matched at 4.7 Hz with 147 ms per scan, and"
        " every one of those measurements was fused as if it spoke for the moment it was used"
        " at. On the new path a measurement is 0.1-0.3 s old when an update takes it (a camera"
        " frame at 5 Hz plus the link), so half a second is the slack around that, not a"
        " threshold anybody has hit; the failure it is against — a bridge that stalls and"
        " delivers a burst — is seconds",
        on_when="raise it only to see what a stale measurement does; the carry over odometry is"
        " honest for as long as the odometry is",
        off_when="lower it towards the measurement's own age (0.3 s) where the cart drives fast"
        " and a carry over a tenth of a second is already a decimetre",
        range=(0.05, 5.0),
    ),
    Flag(
        "carry_stale_words",
        True,
        description="a remote word the odometry trail can still reach is CARRIED to the update"
        " instead of being dropped for its age: what the carry costs is added to its covariance"
        " (pepin.fusion.odometry_covariance) and the trail's own reach is the only bound. Off,"
        " measurement_max_age_s decides as it did before 2026-09-18",
        why="the age budget threw away a fifth of the camera's evidence for being late by less"
        " than one carry's worth of uncertainty. Measured on the tapes of 2026-09-17"
        " (scratch/word_age.py): the camera's words arrive 272-364 ms old at the median, p90"
        " 607-802 ms, worst 1.9 s against a 500 ms budget, and 892 of 6037 `depth` and 916 of 5694"
        " `contact` words on tape 0374 alone were already over it when they arrived. A carry of"
        " 0.8 s at the cart's 0.3 m/s is 24 cm of travel, and the odometry's own error over it is"
        " 0.5-2 cm by the model the carry already applies — an order under the 8 cm floor the word"
        " carries anyway. A word that arrives late is a WIDER word, not no word; that is what an"
        " information filter is for, and what the trail cannot reach is still refused"
        " (`uncovered`)",
        on_when="always: it removes a tunable rather than adding one",
        off_when="to reproduce a tape recorded before 2026-09-18, or where the bridge delivers"
        " bursts minutes old and the trail is long enough to carry them",
    ),
    Flag(
        "remote_floor_xy_m",
        REMOTE_FLOOR_XY_M,
        description="the least position sigma, metres, a measurement from the laptop is fused"
        " with, whatever its own peak claims; 0 takes the claim as it comes",
        why="measured 2026-09-13 with the camera recorded but not fused (scratch/camera_error.py,"
        " tapes 221822 and 221909): against its own band of the volume the camera's word was"
        " 8.5-10.6 cm off the lidar's truth at the median and 12-14 cm at p90, on both legs and"
        " both sources (depth, contact). 8 cm is the median; the self-check still inflates a"
        " source that scatters beyond its claim on top of the floor",
        on_when="always while the camera's covariance is a peak at a provisional temperature:"
        " the floor is what its measured error says the word is worth",
        off_when="0, to fuse the laptop's claim untouched: only to measure what a calibrated"
        " camera temperature does on a tape",
        range=(0.0, 1.0),
    ),
    Flag(
        "remote_floor_yaw_deg",
        REMOTE_FLOOR_YAW_DEG,
        description="the least heading sigma, degrees, a measurement from the laptop is fused"
        " with; 0 takes the claim",
        why="the same tapes: the camera's heading was 1.6-4.9 deg off at the median and 7-9 deg"
        " at p90, while a fan on one wall claimed 1.06 deg. Fused on that claim (22:08, sources"
        " lidar,camera) the pose spun 14-22 cm and 33-40 deg per update and the lidar's +-9 deg"
        " window could not find the truth back. 5 deg is the median of the worse source",
        on_when="always, for the reason above",
        off_when="0, only on a tape, never on the cart",
        range=(0.0, 90.0),
    ),
    Flag(
        "fusion",
        True,
        description="fuse every enabled source's word by its information — a match made here, a"
        " measurement made on the laptop; off: the widest source corrects alone and the others"
        " only report",
        why="with all three sources the fused pose stays within 0.7-5.7 cm of lidar-only and"
        " never loses the map. One defect was found and fixed on the way: an edge-bound lidar"
        " used to be out-voted by a blind fan's plateau, so the anchor's bound is now taken alone"
        " — a 12 cm slip at rest is carried by the second match instead of held for 2.5 s, and"
        " recovery while driving is 3.2 cm against lidar-only's 2.9",
        on_when="whenever more than one source is enabled",
        off_when="to see which source is actually moving the pose: off, the others still report",
    ),
    Flag(
        "covariance",
        PEAK,
        description="how sure a match says it is: peak — the spread of its own score peak at the"
        " matcher's calibrated temperature (config/matcher.json); fit — the fit-scaled second"
        " moment of the whole surface that shipped before it. Both the covariance the lidar's"
        " match is fused by and the one /tracker_pose carries",
        why="the fit-scaled numbers were never held against an error: the published sigma was a"
        " straight line from the inlier fraction (5 cm at a perfect fit, 35 cm at none). The"
        " peak's is calibrated — scratch/peak_temperature.py over the four goto tapes of"
        " 2026-09-13 (10047 matches off the window's edge, replayed against the lidar-only"
        " trace) solves T = 0.016 for a mean NEES of 3.00 (2.99 measured), and at that"
        " temperature a fit >= 0.7 match predicts 0.9/1.2 cm and 0.47 deg against an actual"
        " 0.72/0.77 cm and 0.40 deg. That, and only that, is the reason for the default: the"
        " sigma a match reports is the error it makes. It is NOT a reason to expect the camera"
        " to weigh less — both covariances shrink about sixfold together, and on the real"
        " matcher's own lattices (scratch/peak_skeptic_fuse.py, the furnished room of the unit"
        " tests) a +-40 deg fan 5 cm off the truth pulls the fused pose 17.4 mm here against"
        " 15.9 mm on fit, its share of the across-wall information going UP, 29.8 % to 34.8 %."
        " The 0.24 mm of tests/unit/test_fusion.py is a synthetic lidar made 13 times sharper"
        " than the fan, where a real revolution is 3.5 times sharper. The balance is uneven too:"
        " position comes out 2-3x conservative and the heading optimistic (variance of"
        " error/sigma x 0.45, y 0.34, yaw 1.77)",
        on_when="on: the sigma a match reports is the error it makes, which is what an"
        " information filter needs to weigh the camera against the lidar",
        off_when="fit puts back the numbers every tape before 2026-09-13 was recorded with —"
        " for an A/B against them, or if a calibrated covariance ever misbehaves in the field."
        " Flip it TOGETHER with laptop_localizer's flag of the same name: the board weighs the"
        " laptop's fan against its own match, this path is about sixfold sharper in variance,"
        " and a board on fit with a laptop on peak hands the same fan 75 % of the across-wall"
        " information and 38.9 mm of a 5 cm pull instead of 34.8 % and 17.4 mm"
        " (scratch/peak_skeptic_fuse.py)",
        choices=COVARIANCE_CHOICES,
    ),
    Flag(
        "self_check",
        True,
        description="every source vouches for itself: its covariance is widened by how far its"
        " answers fall from where its OWN previous answer, carried over the odometry, said they"
        " would (pepin.selfcheck). A source four times out in ALL THREE directions loses sixteen"
        " times its weight; the factor is that over-claim averaged over the three, so a source"
        " out in fewer of them loses proportionally less (a camera fan bound along a wall, four"
        " times out in the two directions it measures, is widened 9.7x not 16x —"
        " scratch/selfcheck_audit.py). One that is honest, or better, is not touched. Per"
        " source, never across sources: no lidar pose enters the camera's number and no camera"
        " pose the lidar's",
        why="2026-09-13: the camera's measurements claimed 25 cm from a linear formula over the"
        " fit (pepin.fusion.sigma_from_fit: fit 0.34 -> 24.8 cm) while nobody had measured how"
        " far apart two of its own answers fall a tenth of a second apart. On that claim they"
        " took 20-45 % of the fused weight and pulled the board's pose 0.8-1.5 cm off the"
        " lidar's, whose real error at fit >= 0.7 is 0.5-0.6 cm median against the replay truth"
        " (tapes 20260913_190024/190422, scratch/drive_bisect.py). A covariance nobody measured"
        " is a claim; this makes every source pay for its weight with its own repeatability."
        " The ratio is a chi-square of 3 dof averaged over the last 20 measurements, so 1.0 is"
        " an honest covariance and the factor is capped at 25. One number for the whole matrix:"
        " an over-claim in one direction of three arrives divided by three (a depth source"
        " jumping 24 cm at rest against a 4 cm claim is widened 6x, not 36x), so the check takes"
        " back the over-claim a source's whole covariance carries, never a single direction's."
        " The prediction it judges against pays for the odometry that carried it"
        " (pepin.fusion.odometry_covariance: 2 mm + 2 % of the distance, 0.05 deg + 70 % of the"
        " turn). Without that term the peak covariance made the check accuse the lidar itself:"
        " replayed over tape 20260913_190024 (scratch/lidar_selfcheck_replay.py) the lidar's own"
        " ratio ran at a median of 1.97 and a p90 of 6.23 while the cart moved and it was widened"
        " on 172 of the 307 moving updates — a false inflation of the one measurement this robot"
        " trusts. With it the same replay reads 0.59 median / 0.97 p90 in motion and 0.30 / 0.44"
        " at rest, inflated on 25 of 2271 updates by at most 1.9x (23 of those in motion, by at"
        " most 1.09x). The 70 % is measured, not chosen: that tape's odometry turned 604 deg"
        " against the lidar's 359 (scratch/tape_odometry_error.py), the per-carry error's RMS is"
        " 0.77 of the reported turn and 0.70 with the lidar's own noise taken out — the wheels'"
        " 40-60 % in-place slip on this carpet, which is the odometry the tracker is left holding"
        " when the IMU drops (that tape carries no ekf and no imu at all). With the gyro alive it"
        " is some 2.5x conservative: on tape 0240_20260913_204114, which does carry ekf and imu,"
        " the same measurement is 678 deg of odometry against 716 of lidar and a per-carry RMS of"
        " 0.28, and the check there goes from 48 of 895 updates widened (max 1.31x) to 18"
        " (max 1.10x)",
        on_when="whenever more than one source is fused — it is the only thing standing between"
        " the fusion and a source whose covariance is a formula rather than a measurement",
        off_when="to measure what the check is worth on a tape (the ratios are still measured"
        " and printed with it off, so the A/B is one parameter set apart), or if a source that"
        " is known good is ever inflated by a real correction the odometry could not predict"
        " — a push by hand, a wheel slipping while the flag says the step was trusted",
    ),
    Flag(
        "local_fit",
        True,
        description="a fit only counts where a scan of THIS machine measured it: with no scan here"
        " at all — the camera's or the graph's words driving the tracker alone — /localization_fit"
        " carries 0.0, the value it holds before the first match, the candidate gate is given that"
        " same 0.0 to judge a whole-map answer against, and the remote source's own fit rides"
        " /localization/sources per source; off, the remote fit is published and judged against as"
        " the tracker's own",
        why="the number is read as 'how well the cart's own scan sits on the map' by everything"
        " downstream, and a remote one is neither. The camera's fit is measured on the laptop"
        " (pepin_bringup.laptop_localizer), against the very grid the painting it gates writes"
        " into: published here, that fit would bless the painting of the map it was itself"
        " measured against, a circle no drift can break out of. The replay measures what such a"
        " fit cannot"
        " see: camera-only (split-no-lidar) sits 1.1 cm from lidar-only at the median, 25.1 at"
        " p90 and 43.4 at worst over run 0171, while the fits those same matches reported were"
        " 0.41 and 0.62 (scratch/laptop_localizer_replay.txt). 0.0 and not NaN because"
        " every gate downstream compares with `<` and NaN passes them all silently"
        " (pepin.watch.reported_fit). The candidate gate was the one consumer that read the"
        " tracker's raw fit instead of this one, and camera-only that fit is the GRAPH's own claim:"
        " on 2026-09-17 21:18-21:28Z the graph claimed 1.00, so every one of the 27 whole-map"
        " answers the laptop sent per window was judged 'nothing' — no lidar score can beat"
        " 1.00 + BEAT_MARGIN — while the pose those answers disagreed with was some 90 degrees off"
        " the room (ros/maps/rec/20260917_212759_goto_board.log). A candidate's score and the fit"
        " it is weighed against have to be measured on the same machine or the comparison is void",
        on_when="always on a cart that has a lidar: a fit nothing here measured stops the goal"
        " server and the volume rather than vouching for a pose",
        off_when="to drive on the camera alone — a dead lidar, a lidar-less robot — where the"
        " laptop's fit is the only word there is; watch /localization/sources for the drift it"
        " cannot report",
    ),
    Flag(
        "map_grow",
        STATIC_M,
        description="how far a mapped obstacle's explanation reaches, metres: a return within"
        " this distance of an occupied cell of the served map is the map itself, anything"
        " farther is news (pepin.dynamic.StaticMask). It is what explained_vote silences and"
        " what tells a person beside the cart from a lost cart",
        why="0.15 m has stood since the mask was written and every number explained_vote carries"
        " was measured at it (tape 0173: the rest band 10.5 -> 2.38 deg alone, 0.51 with the rest"
        " lock). It is not a measured optimum: it is about three costmap cells, the room a"
        " wall's returns wander in at this map's 5 cm resolution plus the pose error the tracker"
        " is allowed. The flag exists because the number matters in both directions and nobody"
        " had a knob for it",
        on_when="raise it where the map is coarse or the pose is loose and honest wall returns"
        " are being called news (watch `silenced` in the tracker's report climb)",
        off_when="lower it to let the mask see smaller changes — a chair moved 10 cm is news at"
        " 0.05 and the map at 0.15. The floor is one cell: the mask always grows by at least"
        " one (0.05 m on this map), so anything below that, 0 included, is the mapped cell and"
        " its neighbours and nothing more",
        range=(0.0, 1.0),
    ),
    Flag(
        "fit_needs_a_source",
        False,
        description="/localization_fit falls to 0.00 once no enabled source has spoken for"
        " source_patience_s — no lidar revolution, no camera measurement — instead of repeating"
        " the last fit measured; off, the fit stands until a source corrects it again",
        why="off since 2026-09-19, because the rule it was written for is now enforced by a"
        " number that cannot be faked. It was added on 2026-09-14, when this node published fit"
        " 0.70 for 141 s with nothing correcting the pose and the goal server drove two goals on"
        " dead reckoning; the day after, /localization/sigma arrived (pepin.watch.PoseSpread),"
        " it grows along the odometry whenever no word lands, and every gate downstream reads it"
        " in front of the fit — so silence already shows as a widening pose. What zeroing the fit"
        " cost instead: camera-only there is no lidar to speak, the published 0.00 is then the"
        " NORMAL reading, and it fed a cascade of lidar-shaped refusals at the bookshelf on"
        " 2026-09-19 — a goal refused into a whole-map lidar search that had nothing to match",
        on_when="on a board that publishes no /localization/sigma at all (a build from before"
        " 2026-09-15), where the fit is the only number the gates have",
        off_when="off wherever the sigma is published: the fit then means what it always meant,"
        " the last lidar revolution's inlier fraction, and no gate infers silence from it",
    ),
    Flag(
        "source_patience_s",
        SOURCE_PATIENCE_S,
        description="how long every enabled source may be silent at once, in seconds, before the"
        " published fit falls to 0.00 (fit_needs_a_source)",
        why="the lidar delivers 10 revolutions a second and each camera source 5 measurements, so"
        " 3 s is thirty missed revolutions — a dead sensor or a dead link, not a hiccup. A cart"
        " standing still is not silent: its lidar keeps turning while the motion filter spares"
        " the matcher, so rest costs nothing here. Below the goal server's own 4 s blind-drive"
        " patience on purpose: the fit must have fallen before that watch starts counting",
        on_when="raise it on a link that stutters for seconds at a time and a refused goal costs"
        " more than a drive on a stale pose",
        off_when="lower it towards the sources' own stale_after_s (0.5 s lidar, 1.0 s camera)"
        " where a drive must stop the moment the sensors go quiet",
        range=(0.1, 60.0),
    ),
    Flag(
        "belief_yaw_per_turn",
        EKF_YAW_PER_TURN,
        description="the share of every reported turn the tracked pose's HEADING sigma grows by"
        " between corrections (pepin.watch.PoseSpread, accumulated step by step); the"
        " measurement carry's own term (pepin.fusion.carried, the fusion self-check) is not this"
        f" number and stays at {ODOM_YAW_PER_TURN:.2f}",
        why="0.05, measured: over three lidar-held drives of 2026-09-19 (tapes 0390/0391/0393,"
        " 539 scans matched on the evening's grid, scratch/ekf_heading_error_per_turn.py) the"
        " EKF heading's error against the lidar truth is 2.2 deg RMS over 30 deg of accumulated"
        " turn and 3.6 deg over 180 deg — it barely grows, so it is 2.2 deg of scan-matcher noise"
        " per window plus 0.016 of the turn, and 0.05 is three times that slope. The belief used"
        f" the wheels-only {ODOM_YAW_PER_TURN:.2f} until then, which is what a differential"
        " drive's two encoders are worth on carpet and not what an EKF heading with a gyro in it"
        " is. On 2026-09-19 a camera-only cart read 28 deg of heading sigma after 18 in-place"
        " recoveries and a position sigma over the start gate, and its goals were refused;"
        " replayed through the model, 18 quarter turns and half a metre of driving from a graph"
        " word's own 0.20 m / 8 deg price at 0.42 m / 42 deg with 0.70 and 0.22 m / 9.1 deg with"
        " this number",
        on_when=f"raise it towards {ODOM_YAW_PER_TURN:.2f} on a cart driving with the IMU dead —"
        " there the heading IS the two wheels and the slip is real",
        off_when="lower it only against a fresh measurement of the same kind: this number is"
        " what the tracker admits it does not know, and under the truth it is an overconfident"
        " pose that no gate can catch",
        range=(0.0, 1.0),
    ),
    Flag(
        "map_cache",
        True,
        description="the map this tracker ADOPTS is written down beside the maps"
        f" ({MAP_CACHE_DIR}/{CACHE_NAME}: the cells run-length encoded, the id and the minted"
        " identity, the digest, the stamp and the topic it came from), atomically and only when"
        " the digest changes; at start, with nothing live inside map_fallback_s, that cache is what"
        " this node tracks on. Off, the node needs a map on a topic as before 2026-09-18",
        why="the owner's rule is ONE map — the volume — and a board that cannot start without a pgm"
        " served from a file has two. This node already is the board's one holder of the map (it"
        " adopts, it rebuilds, it owns map -> odom), so it is the one that can keep it. THE CARD:"
        " one write per ADOPTION and only on a changed digest, so at map_refresh_s of 2 s the worst"
        " case is 16 kB every 2 s while the volume is actually changing (this flat's 51385 cells"
        " are 195 kB of raw JSON and 16 kB run-length encoded, scratch/costmap_rle_cost.py) — 8"
        " kB/s against the 55 kB/s a drive's tape already writes, and in practice a handful of"
        " writes a drive because depth_fusion republishes only on change. A 32 GB card rated for"
        " ~500 write cycles takes that for years; the tape, not this, is what wears it."
        " ATOMICALLY because the alternative is losing the only map to a power cut mid-write:"
        " temporary file, fsync, os.replace, fsync of the directory (pepin.mapcache.write_cache),"
        " so a reader sees the previous cache whole or the new one whole",
        on_when="always on the board: it is what makes a cold boot with the laptop down possible"
        " without a file in the loop",
        off_when="while measuring what a boot without any cache does, or on a machine whose card"
        " must not be written at all",
    ),
    Flag(
        "verify_remote",
        True,
        description="a correction made ENTIRELY of remote words — no local scan in the update, so"
        " nothing here can check them — must agree with the tracker's own belief within what the"
        " two covariances allow (pepin.fusion.GATE, the gate a fusion applies between two"
        " sources). One that does not leaves the pose where it was and the update reports that it"
        " measured nothing, so the spread grows and the drive gates read it; off, the word moves"
        " the pose as it did before 2026-09-18",
        why="the gate inside pepin.fusion.fuse compares each measurement with the SUREST one, so"
        " it needs two, and the mode that needs it most has one. Camera-only on 2026-09-17 every"
        " update carried exactly the pose graph's word: the tapes read `fused 0, rejected 0` over"
        " 40 and 50 consecutive updates (0371, 0372) and no gate ran at all. Parked at home that"
        " evening the graph's words read (-9.38, +2.49, -45 deg) while the lidar-held pose was"
        " +55 deg and the room's own answer +51 to +57 deg (the board's search after today's"
        " reboot; scratch/home_twin_search.py on the tape's last scan); with the lidar muted at"
        " 00:48:39Z the tracker went over to the graph's heading by 01:18Z, some 90 degrees, while"
        " the cart moved 4 cm in the whole half hour. The same evening the graph's words agreed"
        " with the tracker to 0.2-0.9 cm in position (scratch/graph_word_vs_tracker.py), because"
        " the anchor had been re-learned FROM the tracker and between closures the word is the"
        " tracker's own odometry: the position agreement carried no information and the heading"
        " was never checked",
        on_when="always where a remote word can be the only source of an update — camera-only,"
        " or a lidar that drops out mid-drive",
        off_when="to replay a tape recorded before 2026-09-18, or to measure how far a remote"
        " source would have taken the pose (it is still counted and reported when it is refused)",
    ),
    Flag(
        "accept_candidates",
        True,
        description=f"re-seed from the laptop watchdog's whole-map candidates ({CANDIDATE_TOPIC},"
        " pepin.watchdog): a place that disagrees with the tracked pose candidate_streak times in"
        " a row, about the same place each time, is adopted through the path the board's own"
        " search uses",
        why="measured on the kidnap tape (run 0171: the odometry jumps 1 m and 40 deg while the"
        " scans do not) this is the difference between coming back in 2.9 s over 28 scans and"
        " never coming back — the tracker's own window still had 0.69 m of error after 39 s, and"
        " the board's own searches found the truth four times (fits 0.76/0.72/0.79/0.75 against"
        " the tracker's 0.50-0.60) and died unconfirmed each time, because at a metre off this"
        " flat still fits 0.53, just under the 0.55 that declares the cart lost. Over the"
        " undisturbed tape it re-seeded 0 times, and against another flat's map 9 of 12"
        " candidates were called unknown_map",
        on_when="whenever the laptop's watchdog runs and the map is the right one",
        off_when="where a teleport is more dangerous than being lost — under a live goal, or in a"
        " room the map does not cover: off, the candidates are still judged, counted and reported",
    ),
    Flag(
        "candidate_streak",
        CANDIDATE_STREAK,
        description="how many candidates in a row must disagree with the tracker and agree with"
        " each other before one of them re-seeds it: the price of a teleport, in seconds",
        why="deliberate conservatism above a measurement that was neutral: on the kidnap tape a"
        " streak of 1 recovered in 0.8 s (8 scans) and this streak of 3 in 2.9 s (28 scans), and"
        " both re-seeded 0 times over the undisturbed tape, where the pose never left the"
        " reference by more than 0.000 m. Nothing measured prefers 3; the argument is that a"
        " look-alike keeps looking alike, so one agreement is not proof",
        on_when="raise it in a room of look-alike corners, where a wrong teleport costs more than"
        " three seconds of being lost",
        off_when="1 is the fastest recovery measured (0.8 s) and on that tape just as safe — the"
        " value to try when a demo has to show the cart coming back",
        range=(1, 10),
    ),
    Flag(
        "map_refresh_s",
        MAP_REFRESH_DEFAULT_S,
        description="the least time between two adoptions of /map: a newer grid is taken only"
        " after this many seconds AND only if its cells changed. 0 takes the first grid and no"
        " other, which is what a served file has always done",
        why=f"{MAP_REFRESH_DEFAULT_S:.0f} s, and both halves of that number are measured rather"
        " than chosen. THE COST: an adoption rebuilds the grid, the correlative matcher, the static"
        " mask and the tracker, and the bill is paid on the first match after it, when the"
        " matcher's lattice is built — 15-16 ms on the laptop's core for this flat's 239x215 cells"
        " and a 280x250 grid alike, so about 65 ms on an A53 at the 4.5x the board's own"
        " report lines give for the same match (40-50 ms there against 8-12 ms here,"
        " scratch/map_adoption_cost.py). THE BUDGET: at 10 revolutions a second and 45 ms a match"
        " the tracker already owns 45 % of a core, and after a fifth for the rest of the node a"
        " tenth of what is left is 3.5 % — which allows one adoption every 1.9 s. THE PUBLISHER"
        " offers them FASTER than that: RTAB-Map republishes its grid at its detection rate, 1 Hz"
        " with map_always_update (rtabmap_util/MapsManager.cpp: the message is rebuilt whenever a"
        " node is added or a pose moves more than GridGlobal/UpdateError, 1 cm), so this flag is"
        " what stands between a driving cart and one matcher rebuild a second. The old default"
        " was 0 — 'adopt the first map and never another' — which under a live graph would freeze"
        " the tracker on the first blob the session published",
        on_when="raise it while a room is being mapped as it is driven, where the grid changes"
        " every second and a rebuild mid-drive costs more than a slightly stale map",
        off_when="0 to pin the tracker to the first grid it sees — a served pgm's own behaviour,"
        " and the way to hold one picture still while something else is measured",
        range=(0.0, 600.0),
    ),
    Flag(
        "carry_pose_across_maps",
        True,
        description="adopting a re-rendered map keeps the pose the tracker holds instead of"
        " starting again from the saved pose or the pose the odometry gives: it is the same room"
        " a moment later, so a new picture of it is no reason to forget where the cart is",
        why="measured by its absence. On 2026-09-14 18:13 a live map swap with the cart at home"
        " restarted the tracker at (0, 0, 0) — the saved-pose file is keyed by map id and the new"
        " grid has another one — and the very next measurement-driven update published map -> odom"
        " for that origin pose: Nav2 logged 'global_costmap: Sensor origin"
        " at (0.01, -0.00) is out of map bounds' 110 times, the local costmap stopped following"
        " the cart, and no goal succeeded until the board's stack was restarted. Under World R that"
        " swap is no longer rare: every loop closure that moves a pose by a centimetre re-renders"
        " the whole grid with a new origin, a new size and a new id, and each one arrives here as"
        " an adoption. The evidence IS dropped at a switch (candidates, measurements, the graph's"
        " word, the LostWatch); the POSE is not evidence about the map, it is where the cart is",
        on_when="always, while a new grid is the same room bent by its own graph",
        off_when="a map of a DIFFERENT place arriving on the same topic, where a carried pose"
        " would be a lie: off makes the tracker find itself again before it publishes anything",
    ),
    Flag(
        "frame_needs_a_pose",
        True,
        description="on a KNOWN map — the disk holds a cached map and a pose saved on it — map ->"
        " odom is not broadcast until this tracker has a pose on a map; on a map being born"
        " (nothing on disk) the identity goes out from the first tick, as it always did",
        why="a default is a refusal, never (0, 0). The identity is the truth only in a map born"
        " under the cart (World R: that map's frame IS the odometry's). On a known map it is a lie"
        " for as long as the tracker waits for its map: on 2026-09-21 it was broadcast for 9.5 s"
        " after every start and then jumped to the saved pose 2.9 m away (the base no longer sat"
        " at the map's origin) — and a jump of the pose is what sends Nav2's RangeSensorLayer into"
        " a ~4e9-iteration loop under the costmap mutex. Every consumer already treats a missing"
        " map -> odom honestly (the ToF bridge's gate stays shut, Nav2's costmaps wait up to"
        " their initial_transform_timeout of 60 s, and the cache seats the tracker in ~10 s)",
        on_when="always",
        off_when="to reproduce a start from before this gate",
    ),
    Flag(
        "map_fallback_s",
        MAP_FALLBACK_S,
        description="how long this tracker waits for a live /map before it tracks on the map it"
        " wrote down itself (the map_cache flag, pepin.mapcache) — only while it has adopted"
        " nothing at all, and the live grid replaces the cache the moment it arrives. 0 waits for"
        " ever, which is what the tracker did before the cache existed",
        why="the board must know where it is without the laptop (CLAUDE.md rule 20), and under"
        " World R the map comes FROM the laptop: with the wifi down nothing will ever publish it,"
        " and the cache is the whole of the board's independence. Ten seconds because a latched"
        " grid arrives in the first second once the bridge's routes are up (RTAB-Map publishes it"
        " transient-local, depth 1, reliable — rtabmap_util/MapsManager.cpp) and a cache is a"
        " colder start that must not be taken while the live one is merely on its way. A map"
        " already in use needs no fallback at all: it is a grid in memory, and losing its"
        " publisher mid-drive changes nothing, which is why this only ever fires before the first"
        " adoption",
        on_when="always: it is the patience before a cold boot falls back to its own cache",
        off_when="0 to see a bring-up wait for the live grid and nothing else — where a silent"
        " /map must be visible as silence rather than papered over by yesterday's map",
        range=(0.0, 600.0),
    ),
    Flag(
        "carry_candidates",
        True,
        description="a candidate's pose is moved from the moment of its own scan to now over the"
        " odometry between the two stamps (pepin.watchdog.carried) before it is judged and fused,"
        " and one the odometry history no longer covers is dropped",
        why="by argument from a measured latency, not by a measured gain: a whole-map search"
        " takes 0.12-0.25 s plus a wireless hop, so at 0.8 m/s an uncarried pose is installed"
        " about 20 cm backwards along the drive every time — a bias, not noise. On the kidnap"
        " tape the carry changes nothing measurable (2.9 s, 28 scans, 0 false re-seeds either"
        " way), because that cart was barely moving when it was lost",
        on_when="whenever the cart may re-seed while driving",
        off_when="only to reproduce the old behaviour, where the pose the laptop measured a"
        " search and a hop ago is installed as the pose now",
    ),
    Flag(
        "odometry_guard",
        True,
        description="an odometry sample whose step from the last trusted one is impossible (over"
        " 1.5 m/s, or over 0.5 m in one sample) while its twist cannot account for it — the"
        " wheels at rest, or a twist faster than this cart can drive — is refused: it never"
        " reaches the history, so the carry keeps the last pose that made sense; off, every"
        " sample is carried, as before. The same guard watches the HEADING: with the wheels at"
        " rest, a yaw step beyond what the twist's own rate could have turned in the interval"
        " (plus 5 deg) is refused the same way",
        why="2026-09-14: a bad /vo input sent the board's EKF to 43 km from the flat at 60 m/s,"
        " and everything downstream followed — two costmaps chased the pose at 200 % CPU and the"
        " depth pipeline carried its scans metres across 25 ms and refitted the depth law from"
        " the wreckage (a 1.65 -> 2.05, the law file corrupted). The thresholds are from the tape"
        " of that evening (ros/maps/rec/0260_20260914_155145Z_home.jsonl, 146 ekf records over"
        " 7.3 s): the frame sat at x 3493.7 m with |vx| never over 0.031 m/s, and its worst"
        " single step was 0.045 m in 55 ms — 0.83 m/s, still under the 1.5 m/s limit, which is"
        " itself five times this cart's 0.3 m/s top speed. Nothing a drive does comes near it."
        " The heading arm is from the same day, 14:48-14:50: the EKF turned odom -> base_link by"
        " about 90 deg with the cart standing on its charger (x, y never left the origin) and"
        " the tracker, refusing corrections under occlusion, went round with it. At rest the"
        " gyro reads 0.3 deg/s on average and 1 deg/s at worst, so 5 deg between two samples"
        " 20-50 ms apart is already a hundred times the noise",
        on_when="always: the cart cannot move that fast or turn that quickly, so a step that"
        " says it did is the filter, not the robot",
        off_when="when the odometry frame legitimately jumps — a fresh EKF whose frame starts"
        " somewhere else while this node keeps running. The guard holds the last trusted pose"
        " until the frame comes back to somewhere reachable from it, or until this node restarts",
    ),
    Flag(
        "distinct_scans",
        True,
        description="a streak is counted in scans, not in messages: a candidate whose scan id is"
        " already in the run is a second opinion that heard the first one's scan, counted as"
        " replay and not lengthening the streak",
        why="the failure it answers is real: a frozen /scan on the laptop published the same"
        " search answer once a second and the board counted three of them as three seconds of"
        " evidence — the same replay that fooled the board's own two-search rule on 2026-09-09."
        " On the kidnap tape it costs nothing (2.9 s, 28 scans unchanged). A sender that names no"
        " scan says id 0, and a repeated 0 reads as replay too",
        on_when="wherever the candidates cross a bridge that can freeze — which is this robot's",
        off_when="only to reproduce the old counting, where one scan's answer repeated could"
        " re-seed the tracker",
    ),
    Flag(
        "graph_reseed_while_driving",
        True,
        description=f'a candidate from the pose graph (source "{GRAPH}") may re-seed the tracker'
        " WHILE a goal is running, but only when the lidar is not on the roster: with no scan"
        " source alive the graph is the only thing that knows the place, and a drive on a belief"
        " nobody can correct is worse than a teleport. With the lidar alive, and for every other"
        " source, the rule is unchanged: no re-seed mid-drive",
        why="the carry test of 2026-09-14 21:12: with the lidar off the cart drove 64 s on a"
        " belief 2 m wrong, the graph recognising the place the whole way and every candidate"
        " refused for the single reason that a goal was running. The teleport this allows is"
        " bounded by everything else in the gate — the candidate is still carried to now, still"
        " judged against this tracker's pose, fit and map, and still needs its re-seed streak",
        on_when="always on a cart that can lose its lidar mid-drive, which is this one",
        off_when="to reproduce the old rule (no re-seed of any source while navigating), or when"
        " the graph itself is suspect — a fresh database, an anchor learned off a soft seating:"
        " then a graph candidate is a confident wrong room and the drive's own watches are the"
        " better judge",
    ),
    Flag(
        "clear_costmap_on_jump",
        True,
        description="when an accepted word moves the published pose further than"
        " clear_costmap_jump_m, Nav2's local costmap is emptied"
        f' ("{CLEAR_LOCAL_COSTMAP}", asynchronously, and at most once per'
        f" {CLEAR_MIN_GAP_S:.0f} s), so the obstacles it marked at the old pose do not stand"
        " beside the ones the live scans mark at the new one",
        why="the camera-only return of 2026-09-16 13:15: the graph-held pose lagged 1.4 m behind"
        " the cart and Nav2 spent 29 recoveries fighting marks the camera had placed at the"
        " poses before each correction. Nothing takes them back — the camera layer clears only"
        " inside its own 80 deg fan and camera-only there is no lidar layer to scrub the rest —"
        " so every correction leaves a copy of the room offset by the jump and the controller"
        " spins between the copies. A cleared local costmap is marked again from the next scans"
        " within a control cycle. The behaviour tree already forgets that grid, but on a timer"
        " (RateController 0.2 Hz around ForgetStaleObstacles) and only while a goal runs: up to"
        " 5 s of driving on a stranded picture, and nothing at all between goals. This ties the"
        " clear to the correction that stranded it",
        on_when="whenever a source that corrects in jumps is on the roster — the graph, a"
        " re-seeding watchdog — and above all camera-only, where no clearing lidar layer scrubs"
        " the grid outside the fan",
        off_when="when the marks must survive a correction: a run that reads the local costmap"
        " as a memory of what the cart drove past, or a debug of the marking itself",
    ),
    Flag(
        "clear_costmap_jump_m",
        0.10,
        range=(0.0, 5.0),
        description="how far one accepted word must move the published pose before the local"
        " costmap is cleared — the step in map -> odom, which is the correction alone with the"
        " odometry's own motion taken out; 0 clears never",
        why="0.10 m is the step the fix of 2026-09-16 was written against, and it sits between"
        " the two sizes of correction this tracker makes: a scan match moves the pose by a"
        " centimetre or two, the graph's words by 0.1-0.3 m, and only the second kind strands"
        " marks worth a clear",
        on_when="raise it when a clear is paid for a correction the costmap could absorb",
        off_when="0 stops the clearing with the flag still on, for a run that wants the jumps"
        " counted without the calls",
    ),
    Flag(
        "slip_watch",
        True,
        description="while the wheels claim speed and the camera's own odometry shows the"
        " picture standing still, the wheels are muted at their source (base_bridge's"
        " odom_publish) so the EKF never fuses the metres they invent; off, the wheels are"
        " always heard and a slip enters the pose",
        why="MEASURED 2026-09-16 with the cart held by hand: the wheels reported 36 cm in 2.4 s,"
        " the EKF followed them to 34 cm, the lidar measured 3 cm. The filter's own Mahalanobis"
        " gate (odom0_twist_rejection_threshold 5.0) cannot see this — it judges each wheel"
        " sample against a prediction the wheel samples before it built. The camera can: standing"
        " still its odometry walks 0.2 cm in 42 s (worst 0.6 cm in 2 s, same day), some 0.3 cm/s"
        " of noise against the 13 cm/s the wheels were claiming, a ratio of forty"
        " (pepin.slip.PictureSlip, ratio 0.5 held for 0.4 s)",
        on_when="always on a cart whose camera half is alive: a slip is the one odometry error"
        " nothing else on board can see",
        off_when="while measuring the raw wheels, or when the camera's odometry is itself under"
        " suspicion — with no picture the watch already stands down by itself",
    ),
)
MASK_FLAGS = ("map_grow",)  # the flags that rebuild the static mask, not the tracker


class _RosLogHandler(logging.Handler):
    """Forwards the Python-side localizer's log lines to the node's ROS logger."""

    def __init__(self, node: Node) -> None:
        super().__init__()
        self._node = node

    def emit(self, record: logging.LogRecord) -> None:
        """One line of the Python-side logger at the same severity on the ROS side: an ERROR
        from pepin (a refused odometry step, say) must not reach the operator as a warning."""
        log = self._node.get_logger()
        say = {
            logging.CRITICAL: log.error,
            logging.ERROR: log.error,
            logging.WARNING: log.warning,
        }.get(record.levelno, log.info)
        say(f"[{record.name}] {record.getMessage()}")


class Relocalizer(Node):
    """Tracks the pose on the map from every enabled scan source and owns map -> odom; watches
    the fit and re-seeds itself (and AMCL) from a whole-map search when it stays poor."""

    def __init__(self) -> None:
        super().__init__("relocalizer")
        self._scan_topic = str(self.declare_parameter("scan_topic", "/scan").value)
        self._min_inliers = float(self.declare_parameter("min_global_inliers", 0.45).value)
        self._check_period_s = float(self.declare_parameter("check_period_s", 1.0).value)
        self._watch_args = dict(
            lost_fit=float(self.declare_parameter("lost_fit", LOST_FIT).value),
            lost_checks=int(self.declare_parameter("lost_checks", 3).value),
            cooldown_s=float(self.declare_parameter("search_cooldown_s", 8.0).value),
        )
        self._watch = LostWatch(**self._watch_args)  # type: ignore[arg-type]
        # One lock for the episode: the claim of _searching, every _watch call and the
        # _pending_seed swap. The worker, the executor's timers and the service thread all
        # touch these, and the single-threaded executor was the only thing serialising them.
        self._episode = threading.Lock()
        # Tracking: every scan corrects the pose by scan matching around the wheels' prediction
        # and this node owns map -> odom (AMCL then only paints particles). The wheels are trusted
        # for one scan interval, 0.1 s, where even a 25% yaw error is a fraction of a degree.
        self._track = bool(self.declare_parameter("track", True).value)
        # The three levers on the tracker's heading jitter, each switchable so a run can be
        # compared with and without it (scratch/tracker_rest_band.py measures all three offline):
        #   subcell_refine  the matcher answers between its candidates (parabola over the score)
        #   rest_lock       a standing cart averages the match in slowly instead of re-deciding
        #   explained_vote  returns the static map cannot explain do not score the match
        self._subcell_refine = bool(self.declare_parameter("subcell_refine", True).value)
        # The rest lock averages in seconds, not in matches: a standing cart is matched about
        # once a second (MotionFilter below), a replay feeds every scan, and both must settle
        # at the same speed. rest_gain is only the fallback for a caller that times nothing.
        # LIVE_PARAMS apply live (ros2 param set /relocalizer rest_lock false): a demo compares
        # them without a stack restart; subcell_refine is the matcher's construction and takes
        # effect at the next start. The switches are built at the end of this method, after the
        # last declare_parameter, and hold the values the next map's tracker is built with.
        self._odom_wz = 0.0  # newest fused yaw rate (gyro-driven): the second witness of rest
        # map -> odom is published 20 times a second, dated 0.1 s ahead (AMCL's habit, shorter):
        # a consumer asking for "now" always finds a transform and never extrapolates.
        self._tf_future_s = float(self.declare_parameter("tf_future_s", 0.1).value)
        self._tracker_initialised = False
        self._tracker_initialising = False
        # 0.05 s: the lidar turns at 10 Hz, and a gap of 0.15 s used to skip every other scan;
        # the pacer's busy rule (skip as long as a slow match took) is what protects the board.
        self._pacer = MatchPacer(
            min_gap_s=float(self.declare_parameter("min_match_gap_s", 0.05).value)
        )
        self._last_match_stamp_s: float | None = None  # scan stamp of the previous match: dt_s
        self._last_scan_age_s = 0.0
        self._last_map_odom = (0.0, 0.0, 0.0)  # the belief until the first fix: the base
        # No pose has set map -> odom yet; whether it may be said before one does is
        # pepin.lastpose.FrameHold's decision (the ``frame_needs_a_pose`` flag).
        self._frame = FrameHold(
            lambda: known_room(FilePath(LAST_POSE_FILE), FilePath(self._cache_dir))
        )
        # Nav2's local costmap keeps what the camera marked at the pose before a correction, and
        # nothing outside the fan erases it. Which step in map -> odom is a jump worth emptying
        # it for is pepin.watch.JumpClear's decision; this node only makes the call.
        self._jumps = JumpClear(self._clear_local_costmap, min_gap_s=CLEAR_MIN_GAP_S)
        self._clear_costmap = self.create_client(ClearEntireCostmap, CLEAR_LOCAL_COSTMAP)
        self._clears = 0  # costmap clears bought by jumps, since this node came up
        self._slip = SlipWatch()  # wheels claiming a step the picture does not show
        self._runaway = RunawayWatch()  # the mirror: the odometry frame flying, the wheels still
        self._runaways = 0  # odometry samples refused because of it (per report)
        self._yaw_runaways = 0  # ...and how many of those were the frame spinning on the spot
        # The map in use, as every candidate and measurement is judged against.
        self._map_id = ""
        self._shift: MapShift | None = None  # what the last adoption did to the map before it
        # Every id this tracker has adopted since the belief last started over: the laptop stamps a
        # word with the id the board had when it made it, and under a growing canvas that is one or
        # two ids back by the time it arrives (live 2026-09-19: 17 of 26 candidates refused as
        # "unknown_map"). All the same room until the frame changes (pepin.mapping.MapChain).
        self._chain = MapChain()
        self._offered: OccupancyGrid | None = None  # the grid being judged, built once
        # How far the last matched revolution reached: the radius inside which a changed cell can
        # still move this tracker's score, and so the radius that decides whether a re-rendered grid
        # is worth a rebuild at all (pepin.mapping.MapShift.matters). The lidar's own maximum until
        # the first scan, from the one file that owns it.
        self._scan_reach_m = load_lidar().max_range_m
        self._cache_dir = str(self.declare_parameter("map_cache_dir", MAP_CACHE_DIR).value)
        self._cached: MapCache | None = None  # the cache this node is tracking on, if any
        self._cache_boot = CacheBoot()  # the cold-boot read is attempted once, not every check
        self._cache_written = 0.0  # when this node last wrote one
        self._pending_seed: tuple[str, Pose2D, float] | None = None
        self._scan_id = 0
        # Time alignment (pepin.timeline): the odometry is kept as a history and every scan waits
        # at the gate until the history covers its whole revolution, so a scan is matched against
        # the pose it was taken at, beam by beam — never against the newest pose. The TF lookup at
        # the scan's stamp used to fail 118 times in 119 (the EKF runs 35-70 ms behind the lidar)
        # and fell back to "now": 1-2 degrees of false correction per scan in every pivot, with the
        # sign of the turn, and the cart steered by the wobble (runs 0080-0083, 2026-09-09).
        self._odom_topic = str(self.declare_parameter("odom_topic", "/odometry/filtered").value)
        self._picture_slip = PictureSlip()  # the wheels against the camera's own odometry
        self._vo = PictureSpeed()  # the camera's own speed, from its consecutive poses
        self._wheels_muted = False
        self._wheel_at: float | None = None  # when the wheels last said anything at all
        self._zupt_pub = self.create_publisher(Odometry, "zupt", 5)
        self._wheel_params = AsyncParameterClient(self, "base_bridge")
        self._history = OdomHistory(horizon_s=5.0)
        # Every source's scans wait at the feed (a gate per source) and one of them drives the
        # update: the lidar while it is fresh, else the camera. The roster is shared with the
        # tracker, so the ``sources`` flag switches both at once.
        self._registry = SourceRegistry()
        self._feed = SourceFeed(self._registry, max_wait_s=0.5)
        # The camera's word, measured on the laptop: the newest per source waits here until an
        # update takes it, carried to that update's moment over the same history a riding scan
        # would have been carried along (pepin.measurements).
        self._measurements = MeasurementGate(self._registry)
        # RTAB-Map's graph, one more word with one more name: its own gate, its own seat on the
        # roster (`graph` in the sources flag), the same carry, the same information filter and
        # the same disagreement gate. It rides the update a scan drives, and with no scan source
        # driving it drives one itself, beside the camera's word and through the same path
        # (pepin.measurements.remote_update): one update per word, never two.
        self._graph = MeasurementGate(self._registry, name=GRAPH)
        self._motion = MotionFilter(min_m=0.005, min_deg=0.3, max_gap_s=1.0)
        self._rested = 0  # scans left unmatched because the cart stood still (per report)
        # What the map does not explain (a person, a moved chair): the mask the match votes with
        # and the occlusion test reads (pepin.dynamic). It reaches no costmap — the lidar layer
        # marks a new object from the same return, at the same cell (2026-09-14).
        self._static_mask: StaticMask | None = None
        self._deskew_failed = 0  # scans matched raw because the history had a hole (per report)

        self._motion_edge = MotionEdge()  # odom->base_link moved since the previous check
        self._navigating = False  # a NavigateToPose goal is executing
        self._grid: OccupancyGrid | None = None
        self._matcher: CorrelativeMatcher | None = None
        logging.getLogger("pepin").addHandler(
            _RosLogHandler(self)
        )  # localizer reasons in the ROS log
        self._localizer: Localizer | None = None
        self._laser_tf: tuple[float, float, float, bool] | None = None  # x, y, yaw, mirrored
        self._searching = False
        self.fit = float("nan")
        self._fit_is_local = True  # did a scan of this machine score the pose the topic reports?
        # Is anything still telling this board where it is? With every enabled source silent for
        # longer than the patience the published fit falls to 0.00: the number downstream reads
        # as "I know where I am" may not outlive the sensors that earned it (pepin.watch).
        self._silence = SourceSilence()
        self._source_age_s = math.inf  # seconds since the freshest enabled source spoke
        # The laptop's watchdog: its candidates are judged against this tracker's own pose and
        # fit, and a streak of disagreements about one place re-seeds through _pending_seed —
        # the very path the board's own search uses. All the judging is in pepin.watchdog.
        self._candidates = CandidateGate()
        # The watch judges full revolutions only; while a fan drives it is off and the report
        # line says so (the fit is still measured and published, on the fan).
        self._watch_on = True
        # Stage one: a wide window around the pose AMCL believes in (a push, a short carry); the
        # whole map only when that fails. On four A53 cores the whole-map lattice takes ~15 s,
        # the window about a second.

        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        # ONE map topic, latched to match the publisher (RTAB-Map's grid is transient-local, depth
        # 1, reliable: a subscriber that connects late is handed the current grid at once, and a
        # VOLATILE reader would not match that writer at all). When a newly arrived grid is adopted
        # is pepin.mapping's decision; the node holds the message and does as it is told.
        self._choice = MapChoice()
        self._map_topic = f"/{MAP_TOPIC}"
        self.create_subscription(OccupancyGridMsg, self._map_topic, self._on_map_message, latched)
        # Depth 1: a match takes 40 ms and scans come every 100 ms; a deeper queue let the tracker
        # fall half a second behind reality and lose the lock in every turn (2026-09-06).
        self.create_subscription(
            LaserScan,
            self._scan_topic,
            self._on_scan,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        self.create_subscription(Odometry, self._odom_topic, self._on_odom, 20)
        # The slip watch's two witnesses: the wheels' own word (base_bridge, not the filter's
        # output — the filter follows the wheels, so it cannot testify against them) and the
        # camera's. Both are one subscription each; the board pays ~20 and ~3 messages a second.
        self.create_subscription(Odometry, "/odom", self._on_wheels, 20)
        self.create_subscription(Odometry, "/vo", self._on_vo, 10)
        # Depth 1: a candidate is a snapshot of a moment, and the newest one is the only one
        # worth judging; a backlog of them would re-seed from a scan seconds old.
        self.create_subscription(
            String,
            CANDIDATE_TOPIC,
            self._on_candidate,
            QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE),
        )
        # Depth 5, not 1: unlike a candidate — a snapshot of now, where only the newest is worth
        # judging — every measurement carries the stamp it was measured at and is carried to the
        # update that takes it, so a short queue is a few tenths of a second of the camera's
        # history rather than a backlog of stale opinions.
        self.create_subscription(
            String,
            MEASUREMENT_TOPIC,
            self._on_measurement,
            QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE),
        )
        # The graph's word, on the QoS pinned for the route (pepin.deployment.BRIDGED_QOS): both
        # ends of a bridged topic must ask for the same thing or the route's QoS is decided by a
        # race, and the loser receives nothing in silence.
        self.create_subscription(
            String,
            GRAPH_MEASUREMENT_TOPIC,
            self._on_graph_measurement,
            bridged_qos_profile(GRAPH_MEASUREMENT_TOPIC),
        )
        self._fit_pub = self.create_publisher(Float32, "localization_fit", 5)
        # Every source's word on each update, as JSON (Localizer.sources_report): the demo's
        # view of the lidar and the camera agreeing, disagreeing, or one of them gone.
        # The map this tracker is on, for Nav2's static layers: latched, so a costmap that starts
        # later still gets it, and published only when a map is adopted.
        self._tracked_map_pub = self.create_publisher(
            OccupancyGridMsg,
            TRACKED_MAP_TOPIC,
            QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self._sources_pub = self.create_publisher(String, "/localization/sources", 5)
        # The fit at the pose actually published (the blend), beside the tracker's own fit at
        # the matched pose: the two part ways while a carry is being absorbed.
        self._published_fit_pub = self.create_publisher(Float32, "localization_fit_published", 5)
        # The pose's own uncertainty, out of the fusion and grown along the odometry between
        # corrections: the ONE number a goal gate and a blind-drive watch read (pepin.watch).
        # The fit beside it stays what it always was — the LIDAR's diagnostic, and the per-source
        # fits stay on /localization/sources — because a fit is one sensor's metric and a drive
        # may be held by the camera alone.
        self._sigma_pub = self.create_publisher(String, SIGMA_TOPIC, 5)
        self._spread = PoseSpread()
        # When this node came up: before the first accepted word THAT is how long the pose has
        # gone uncorrected, and it is what the sigma message reports instead of a 0.0 that would
        # read as "a word just landed".
        self._up_since_s = self._now_s()
        # The operator's own word (Foxglove's "set pose", ros/goto.sh seed): the map has twins —
        # 2026-09-13 the whole-map search seeded the cart 6 m from its base at fit 0.77 and the
        # watchdog's true candidate (0.72 vs 0.69) could not beat it by the margin — and nothing
        # else lets a person who can see the cart say where it is. A message this node sent
        # itself (the AMCL-era echo above) re-seeds the same pose and changes nothing.
        self.create_subscription(
            PoseWithCovarianceStamped, "/initialpose", self._on_operator_seed, 5
        )
        self._tf_pub = TransformBroadcaster(self)
        self._tracker_pub = self.create_publisher(PoseWithCovarianceStamped, "/tracker_pose", 5)
        self._slip_pub = self.create_publisher(Bool, "/slip", 5)  # wheels move, the world does not
        self.create_timer(30.0, self._report_tracking)
        self.create_timer(2.0, self._remember_pose)
        self.create_timer(0.2, self._wheels_no_word)  # a mute outlives at most a fifth of a second
        self.create_timer(
            0.1, self._still_while_slipping
        )  # the zero-velocity update, while it holds
        self.create_timer(0.2, self._apply_pending_seed)  # the worker's fix, applied here
        self.create_timer(0.05, self._send_map_odom)  # the frame stays alive, scans or not
        self.create_subscription(
            String, "/pepin/note", lambda m: self.get_logger().info(f"note: {m.data}"), 5
        )
        # Eyes for the operator: AMCL's particles as arrows and its pose history as a line.
        # nav2_msgs/ParticleCloud is unknown to Foxglove; PoseArray and Path are not.
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._particles_pub = self.create_publisher(PoseArray, "/particle_poses", 1)
        self._trail_pub = self.create_publisher(Path, "/amcl_path", latched)
        self._trail = Path()
        self._trail.header.frame_id = "map"
        self.create_subscription(
            ParticleCloud, "/particle_cloud", self._on_particles, qos_profile_sensor_data
        )
        self.create_subscription(PoseWithCovarianceStamped, "/amcl_pose", self._on_amcl_pose, 10)
        self.create_service(Trigger, "relocalize", self._on_relocalize)
        self.create_service(Trigger, "where_am_i", self._on_where)
        # Static transforms only (the laser mount). /tf itself is not read here: this node owns
        # map -> odom and keeps odom -> base_link in its own history, and 40 tf messages a second
        # deserialised in Python cost a quarter of an A53 core for nothing.
        self._tf = TfLookup(self, buffer=Buffer())  # no listener: /tf_static is read below
        self.create_subscription(
            TFMessage,
            "/tf_static",
            self._on_tf_static,
            QoSProfile(
                depth=100,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            ),
        )
        self.create_timer(self._check_period_s, self._check)
        # Declared after every other parameter: rclpy runs the switches' callback on
        # declarations too, and it refuses everything that is not a flag.
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        self._registry.enable(self._switches["sources"])
        gates = (self._candidates, self._measurements, self._graph, self._choice, self._jumps)
        for gate in gates:  # a launch override too
            for name in gate.switches:
                gate.switch(name, self._switches[name])
        self.get_logger().info("relocalizer up: watching the scan-to-map fit")

    # -- inputs -------------------------------------------------------------

    def _now_s(self) -> float:
        """The node's clock in seconds: the same clock the scans are stamped with."""
        return float(self.get_clock().now().nanoseconds) * 1e-9

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A flag changed (``ros2 param set``): a mask flag re-grows the map's explanation at
        once; every other one is written through to whichever object owns it — the
        Localizer (``sources`` reaches the roster the feed shares with it, so the anchor moves
        with the flag) or the candidate gate — so the next scan, and the next candidate, are
        handled with it. Each object names its own flags (``switches``), so a set of one
        object's flag is never refused by the other."""
        if name in MASK_FLAGS:
            self._rebuild_mask()  # the switches already hold the new value
            return
        targets = (
            self._localizer,
            self._candidates,
            self._measurements,
            self._graph,
            self._choice,
            self._jumps,
        )
        for target in targets:
            if target is not None and name in target.switches:
                target.switch(name, new)

    def _rebuild_mask(self) -> None:
        """Re-grow the static map's explanation to the ``map_grow`` the switches now hold, and
        say how far it reaches. Harmless before the map arrives: the mask is built there too."""
        if self._grid is None:
            return
        grow = float(self._switches["map_grow"])
        self._static_mask = StaticMask(self._grid, grow)
        self.get_logger().info(f"static mask: the map explains a return within {grow:.2f} m")

    def _on_map_message(self, msg: OccupancyGridMsg) -> None:
        """A grid arrived on /map: offer it to the choice, which adopts it or turns it away
        (:class:`pepin.mapping.MapChoice`).

        What the choice asks back is :meth:`_unworthy`, and that is where the churn of a live graph
        is stopped. Both of its questions read the whole grid, so they are asked only of a
        publication that has already passed the cheap gate (the refresh period and the digest).
        """
        self._choice.offer(
            self._map_topic,
            lambda: map_digest(msg),
            self._now_s(),
            lambda: self._adopt(msg),
            empty=lambda: self._unworthy(msg),
        )

    def _unworthy(self, msg: OccupancyGridMsg) -> bool:
        """Whether this grid is not worth the rebuild it costs: it has no known cell at all, or
        nothing changed in it that this tracker's own scan can reach
        (:func:`pepin.mapping.worth_adopting`).

        RTAB-Map re-renders its grid at its detection rate and, parked in mapping mode, its
        probabilistic cells flicker across the occupancy threshold a handful at a time — every one
        a changed digest. The first live run adopted 59 maps in its first minutes and rebuilt the
        matcher, the mask and the tracker at each of them (2026-09-19). A cell that changed further
        away than the revolution reaches cannot move one beam's score, so the grid is turned away
        and counted like any other refusal; the NEXT publication is offered afresh, and the moment
        the cart turns towards that cell it is taken.

        The grid is built here and kept: :meth:`_on_map` uses this very object rather than
        converting the message twice (a 51000-cell ``np.where`` each time).
        """
        self._offered = grid_from_msg(msg)
        pose = self._tracked_pose()
        self._shift = map_shift(
            self._grid, self._offered, at=None if pose is None else (pose.x, pose.y)
        )
        return not worth_adopting(self._offered, self._shift, self._scan_reach_m)

    def _adopt(self, msg: OccupancyGridMsg) -> None:
        """Take ``msg`` as the map this tracker matches on: rebuild the matcher, the mask and the
        tracker on it (:meth:`_on_map`), republish it for Nav2's static layers, write it down for
        the next cold boot, and say what the change was. Called by the choice, never directly — a
        rebuild costs seconds on this board and throws away the episode's evidence.

        The cache stops being the map in use here: a live grid is the newer picture by definition,
        and a report line still naming the cache after one arrived read as a board that never got
        its map. It is cleared AFTER the rebuild because the rebuild asks whether the pose it holds
        was found on a cache (:meth:`_on_map`).
        """
        self._on_map(msg)
        self._cached = None
        self.get_logger().info(
            f"map adopted from {self._map_topic}: {msg.info.width}x{msg.info.height} cells,"
            f" id {self._map_id}, digest {self._choice.digest.split('#')[-1]}"
            + ("" if self._shift is None else f" ({self._shift.phrase()})")
        )
        self._publish_tracked_map(msg)
        self._persist_map(msg)

    def _publish_tracked_map(self, msg: OccupancyGridMsg) -> None:
        """Republish the map this tracker is on, latched, on :data:`TRACKED_MAP_TOPIC`.

        Nav2's static layers read this and nothing else, so the planner's static map IS the
        tracker's map — one map on the board, by construction rather than by two subscriptions to a
        third party that can disagree. It costs one message per adoption (at most one every
        ``map_refresh_s``), and the grid is the one already in hand.
        """
        self._tracked_map_pub.publish(msg)

    def _persist_map(self, msg: OccupancyGridMsg) -> None:
        """Write the adopted map beside the maps, for the next start with nothing live (the
        ``map_cache`` flag).

        ON A CADENCE, not on every adoption (:data:`MAP_CACHE_MIN_GAP_S`). The cache's only reader
        is a board that boots with no live map, and for that reader a picture a minute old is worth
        what one a second old is worth — while a write per adoption under a live graph is 8 kB/s of
        card wear for a file nobody reads (59 writes in the first minutes of 2026-09-19). The first
        adoption always writes: a board with no cache at all is the one case where a minute's delay
        costs a boot. A failure is logged and nothing else: a board that cannot write its cache
        still tracks, it only has a colder boot ahead of it.
        """
        now = time.time()
        due = not self._cache_written or now - self._cache_written >= MAP_CACHE_MIN_GAP_S
        cache = MapCache(
            cells=list(msg.data),
            width=int(msg.info.width),
            height=int(msg.info.height),
            resolution_m=float(msg.info.resolution),
            origin_xy=(float(msg.info.origin.position.x), float(msg.info.origin.position.y)),
            map_id=self._map_id,
            digest=self._choice.digest,
            stamp=now,
            source=self._map_topic,
        )
        self._cache_written = cache.stamp if due else self._cache_written
        self.get_logger().info(
            save(FilePath(self._cache_dir), cache, enabled=self._switches.on("map_cache"))
            if due
            else f"map cache: this adoption was not written down, the last write is"
            f" {now - self._cache_written:.0f} s old (one per {MAP_CACHE_MIN_GAP_S:.0f} s)"
        )

    def _take_cached_map(self) -> None:
        """Nothing live inside ``map_fallback_s`` and no map adopted: track on the cache.

        The one path that makes a pgm unnecessary, and the whole of the board's independence from
        the laptop (CLAUDE.md rule 20): with the wifi down nothing will ever publish /map. A board
        that has NEVER adopted anything has no cache, and then it says so loudly and waits — the
        launch's map file is the only other answer and it is behind a launch argument that is off in
        a known room (ros/nav.launch.py). A cache that exists and does not parse is refused just as
        loudly: half a map is worse than none, because the tracker would match against it and
        believe the answer.

        A CACHE IS ONLY EVER A STAND-IN. It is read once, only while nothing has been adopted, and
        the choice's ``source`` stays empty while it is in use — so the first live grid to arrive
        replaces it with no gate to pass, whatever its geometry and whichever database it came
        from. That is also the answer to a cache of a frame that no longer exists (the laptop
        started a FRESH database while this board was down): nothing on the wire identifies a
        session — not the grid, not /rtabmap/info, not /rtabmap/mapGraph — so the cache is not
        checked against one, it is simply given up the moment a live map arrives, and the pose on
        that new map comes from :meth:`_start_pose` rather than from the dead frame's coordinates,
        because a pose held on a cache is never carried across (:meth:`_on_map`).
        """
        answer = self._cache_boot.attempt(
            FilePath(self._cache_dir),
            enabled=self._switches.on("map_cache"),
            have_map=self._grid is not None,
            waited_s=self._now_s() - self._up_since_s,
            patience_s=float(self._switches["map_fallback_s"]),
        )
        for level, text in answer.words:
            getattr(self.get_logger(), level)(text)
        for cache in answer.taken:  # none or one: the decision is pepin.mapcache's, not this node's
            msg = OccupancyGridMsg()
            msg.header.frame_id = "map"
            msg.info.resolution = cache.resolution_m
            msg.info.width, msg.info.height = cache.width, cache.height
            msg.info.origin.position.x, msg.info.origin.position.y = cache.origin_xy
            msg.data = list(cache.cells)
            self._cached = cache
            self._on_map(msg)
            self._publish_tracked_map(msg)

    def _on_map(self, msg: OccupancyGridMsg) -> None:
        # The belief, before the old tracker is thrown away. A re-rendered grid is the same room a
        # moment later, so where the cart is does not change because the picture of the room did;
        # only the map id does, and the saved-pose file is keyed by that id — which is how the live
        # map swap of 2026-09-14 18:13 restarted the tracker at the ORIGIN, published map -> odom
        # for it, and left Nav2 logging "Sensor origin out of map bounds" 110 times with a rolling
        # window that never came back.
        #   A POSE HELD ON THE CACHE IS NOT CARRIED. That one is not the same room a moment later:
        # it is yesterday's map, and the live grid replacing it may belong to a database born since
        # (ros/laptop.sh vslam --fresh), whose frame has nothing to do with the coordinates the
        # cached pose is written in. Nothing on the wire says which database a grid comes from, so
        # the cache's pose is given up with the cache and :meth:`_start_pose` answers instead: the
        # saved pose when the id still matches, else the odometry's own — which in a map born under
        # the cart is exactly right, and in a mature one is a poor guess the LostWatch turns into
        # one whole-map search.
        carried = (
            self._localizer.pose
            if self._localizer is not None
            and self._tracker_initialised
            and self._cached is None
            and self._switches.on("carry_pose_across_maps")
            else None
        )
        # The grid :meth:`_unworthy` already built and judged, or a conversion of its own for the
        # cache path, which goes straight to the tracker without passing the choice.
        grid = self._offered if self._offered is not None else grid_from_msg(msg)
        self._shift = self._shift if self._offered is not None else None
        self._offered = None
        self._grid = grid
        # A CHANGE OF THE PICTURE IS NOT A REASON TO FORGET THE EPISODE. It used to be: every
        # adoption dropped the LostWatch's streak, the pending seed and all three gates' words, and
        # with a live graph re-rendering the grid every second that meant a tracker that never
        # accumulated evidence about anything — 59 adoptions in the first minutes of the first live
        # run (2026-09-19). The evidence is about the ROOM, and a re-render is the same room; what
        # makes it evidence about nothing is a change of FRAME, which is exactly the case where the
        # pose could not be carried.
        reseated = carried is None
        self._forget_episode() if reseated else None
        self._matcher = CorrelativeMatcher(self._grid)
        self._static_mask = StaticMask(self._grid, float(self._switches["map_grow"]))
        # lost_after huge: update() must never run a whole-map search in the executor thread on
        # this board (10-20 s); the 1 Hz watcher below does that in a worker and re-seeds.
        # Tracking window sized for this board: 7x7 positions x 13 headings x 120 beams is about
        # 60 ms per scan on an A53 (the laptop default, 9x9x49x200, took 360 ms: 2 Hz).
        self._map_id = map_id(msg)
        # The tracker's own flags only: the gate's (accept_candidates, candidate_streak) are
        # this node's and the mask's (MASK_FLAGS) grows the map's explanation, so neither
        # belongs to a Localizer — each object names what it owns.
        flags = {n: v for n, v in self._switches.flags.as_dict().items() if n in TRACKER_SWITCHES}
        self._registry.enable(flags.pop("sources"))  # the roster is the feed's and the tracker's
        self._localizer = Localizer(
            self._grid,
            # The pose this tracker already holds, or — at a start, where there is none — the one
            # the previous run saved, or the one the odometry gives (:meth:`_start_pose`; a restart
            # is not a trip back to the base, and neither is a map born under the cart).
            carried if carried is not None else self._start_pose(),
            window=SearchWindow(xy_m=0.09, xy_step_m=0.03, theta_deg=9.0, theta_step_deg=1.5),
            # Lost (five weak scans): a wider, coarser local search every scan re-locks after a
            # slip; the whole map stays the worker's job (global_retry False).
            recovery=SearchWindow(xy_m=0.25, xy_step_m=0.05, theta_deg=20.0, theta_step_deg=2.0),
            max_points=120,
            # Half of each match's residual per scan: one match carries about a degree of
            # noise, and at full gain that noise reaches the wheels through map -> odom ten
            # times a second (the robot weaved, 2026-09-07). Only the residual is damped, so
            # the motion itself never lags; a residual too big to be noise is taken whole.
            correction_gain=0.5,
            recovery_min_inliers=0.5,  # this flat's true pose scores 0.5-0.65 on its maps
            lost_after=3,
            global_retry=False,
            interpolate=self._subcell_refine,
            sources=self._registry,
            **flags,
        )
        self.get_logger().info(
            f"map received: {msg.info.width}x{msg.info.height} cells"
            + (
                f"; the cart stays where it was ({carried.x:+.2f}, {carried.y:+.2f},"
                f" {math.degrees(carried.theta):+.0f} deg)"
                if carried is not None
                else "; looking for the cart on it"
            )
        )
        # A carried pose keeps the tracker tracking: nothing is published for a pose nobody
        # accepted, and a carry onto a map that is NOT this room is caught by the fresh
        # LostWatch above, which re-seeds through the whole-map search like any other loss.
        self._tracker_initialised = carried is not None
        # The belief's own uncertainty survives too — the spread is not rebuilt with the tracker —
        # widened by what THIS change can account for and nothing else: the map's translation
        # (pepin.mapping.MapShift.widen_m). A bend of 0.35 m moved the room under a pose measured
        # against the room before it; six flickering cells moved nothing, and charging the pose for
        # them is how a parked cart at fit 0.97 came to report 0.52 m of sigma (2026-09-19).
        self._spread.covariance = widened(
            self._spread.covariance, 0.0 if self._shift is None else self._shift.widen_m
        )
        # The match cadence is the CART's, not the map's. Resetting these at every adoption made
        # every match "the first match on this map": dt_s None, so the rest lock fell back to its
        # gain of 0.05 instead of its 6 s time constant, at every single match, for ever (live
        # 2026-09-19: "rest-locked 1 (dt - s, gain 0.05/0.05/0.05)"). A re-seated tracker is a
        # different matter — there the cart really is somewhere else now.
        self._motion.reset() if reseated else None
        self._last_match_stamp_s = None if reseated else self._last_match_stamp_s
        # ...and every id of this room, so a word stamped with the grid the laptop had a second ago
        # is still evidence about the room the tracker is in (pepin.mapping.MapChain).
        self._chain.adopted(self._map_id, new_frame=reseated)

    def _forget_episode(self) -> None:
        """Throw away everything this tracker had gathered about WHERE it is: the loss streak, the
        pending seed and all three gates' waiting words. Only for an adoption the pose could not be
        carried across — a cold-boot cache replaced by a live grid, a database born since, a map of
        another place — because then the evidence really is about a room that is no longer here."""
        with self._episode:  # a candidate found on the old map is evidence about nothing here
            self._watch = LostWatch(**self._watch_args)  # type: ignore[arg-type]
            self._pending_seed = None
            self._candidates.forget()
            self._measurements.forget()  # nor is a pose measured against the old one
            self._graph.forget()  # nor the graph's word about it

    def _on_scan(self, msg: LaserScan) -> None:
        """The lidar's revolution, moved into base_link by the mount looked up once."""
        if self._laser_tf is None and not self._lookup_laser(msg.header.frame_id):
            return
        assert self._laser_tf is not None
        self._offer(LIDAR, msg, self._laser_tf)

    def _offer(self, name: str, msg: LaserScan, mount: tuple[float, float, float, bool]) -> None:
        """Any source's scan into the feed, and a try at matching whatever the feed releases."""
        self._scan_id += 1  # which scan a search was computed on: a second opinion needs a new one
        scan = timed_scan_from_ros(
            Time.from_msg(msg.header.stamp).nanoseconds * 1e-9,
            msg.ranges,
            msg.angle_min,
            msg.angle_increment,
            msg.range_max,
            msg.scan_time,
            mount,
            self._scan_id,
        )
        self._feed.offer(name, scan)
        self._track_pending()

    def _on_vo(self, msg: Odometry) -> None:
        """The camera's own speed, from the last pair of visual-odometry poses. The clock is the
        BOARD's arrival time, not the message's stamp: the two halves keep their own wall clocks
        and this comparison must not depend on them agreeing."""
        self._vo.feed(
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9,
            self.get_clock().now().nanoseconds * 1e-9,
        )

    def _on_wheels(self, msg: Odometry) -> None:
        """One word from the wheels, judged against the picture."""
        self._wheel_at = self.get_clock().now().nanoseconds * 1e-9
        self._wheels_speak(float(msg.twist.twist.linear.x))

    def _wheels_no_word(self) -> None:
        """A moment with no word from the wheels at all — which is what a MUTED wheel sounds
        like. Judged as "the wheels claim nothing", so a mute lasts at most one of these ticks
        and the wheels get their voice back to speak for themselves; a slip that is still going
        mutes them again within the watch's hold. Without this the mute would be permanent: a
        muted wheel publishes nothing, and a watch fed only by wheels would never hear it stop.
        While the wheels DID speak a moment ago the tick stands aside: their own words answer.
        """
        change = self._picture_slip.tick(
            self.get_clock().now().nanoseconds * 1e-9,
            self._wheel_at,
            self._vo.speed,
            self._vo.at,
            self._wheels_muted,
            watching=self._switches.on("slip_watch"),
        )
        self._mute_wheels(*change) if change is not None else None

    def _wheels_speak(self, speed: float) -> None:
        """The wheels' own speed against the picture's. While the wheels claim to drive and the
        pictures stand still, the wheels are lying (:class:`pepin.slip.PictureSlip`) and their
        voice is taken away at the source — base_bridge's ``odom_publish`` — so the EKF never
        sees the metres they invent. It comes back the moment the two agree again, or the moment
        the camera stops testifying (a stale picture is no witness)."""
        change = self._picture_slip.change(
            self.get_clock().now().nanoseconds * 1e-9,
            speed,
            self._vo.speed,
            self._vo.at,
            self._wheels_muted,
            watching=self._switches.on("slip_watch"),
        )
        self._mute_wheels(*change) if change is not None else None

    def _still_while_slipping(self) -> None:
        """Ten times a second, a zero-velocity update for as long as the wheels stand accused."""
        self._hold_still() if self._wheels_muted else None

    def _hold_still(self) -> None:
        """While the wheels are known to be lying, tell the filter what IS true: the cart is not
        moving. A muted wheel only takes a measurement away, and a filter without measurements
        coasts on the velocity it last believed — 30 cm of invented motion survived muting alone
        on 2026-09-16. This is the zero-velocity update of inertial navigation: x, y and yaw rate
        at zero with a centimetre-class sigma, published only while the verdict stands."""
        message = Odometry()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "odom"
        message.child_frame_id = "base_link"
        message.twist.covariance = zero_twist_covariance()
        self._zupt_pub.publish(message)

    def _mute_wheels(self, mute: bool, verdict: PictureSlipVerdict) -> None:
        """Set base_bridge's ``odom_publish`` to the opposite of ``mute`` and say why, once per
        change. The call is fired and not waited on: nothing on this path may block."""
        self._wheels_muted = mute
        self._wheel_params.set_parameters(
            [Parameter("odom_publish", Parameter.Type.BOOL, not mute)]
        )
        heard = "muted" if mute else "heard again"
        self.get_logger().warn(f"slip: the wheels are {heard} — {verdict.said}")

    def _on_odom(self, msg: Odometry) -> None:
        """Every fused odometry sample feeds the history; a scan waiting for it gets matched.

        A sample the guard refuses (:class:`pepin.odometry.RunawayWatch`: an impossible step
        with a twist that cannot account for it) never reaches the history, so the carry keeps
        the last pose that made sense and the tracker stands where it stood. The wheels' word is
        this message's own twist field — the EKF fuses /odom into it, and subscribing to /odom
        beside it would cost the board a second reader for a number that is already here.
        """
        p, q = msg.pose.pose.position, msg.pose.pose.orientation
        stamp = Time.from_msg(msg.header.stamp).nanoseconds * 1e-9
        pose = Pose2D(p.x, p.y, yaw_of(q))
        vx = float(msg.twist.twist.linear.x)
        # The gyro's word on whether the cart turns, taken from the filter that already fuses it:
        # subscribing to /imu/data_raw here would cost a slice of a core for a number we have.
        self._odom_wz = float(msg.twist.twist.angular.z)
        carried = self._runaway.feed(
            self._history, pose, stamp, vx, self._odom_wz, self._switches.on("odometry_guard")
        )
        self._runaways += not carried
        self._yaw_runaways += not carried and self._runaway.reason == "yaw"
        self._track_pending()

    def _on_candidate(self, msg: String) -> None:
        """The laptop watchdog's whole-map candidate (one JSON message, pepin.watchdog): carried
        to this moment, then judged by the gate against this tracker's own pose and fit, on this
        tracker's own map.

        A streak of disagreements about one place becomes a pending seed, which the 0.2 s timer
        applies through :meth:`_seed` — the same door the board's own search uses, so a
        candidate can do nothing a search could not. No re-seed while a goal runs: a teleport
        mid-drive is worse than a poor fit, and the drive's own watches stop it soon enough —
        unless the graph is the only localizer left (no lidar on the roster), where nothing else
        will ever correct the belief the cart is driving on (``graph_reseed_while_driving``).
        Nothing is decided here; the verdicts reach the operator in the report line and in the
        ``candidates`` block of /localization/sources.
        """
        try:
            candidate = GlobalCandidate.from_json(msg.data)
        except (KeyError, TypeError, ValueError) as exc:
            self._candidates.malformed(str(exc))
            return
        # A candidate found on a CAMERA fan may not move a tracker the lidar is still feeding,
        # whatever the laptop thought when it published one. The laptop has the same rule
        # (pepin.watchdog.camera_search_need), but it reads this board's own health off a topic
        # that can go quiet for reasons that have nothing to do with the lidar — and the fits of
        # the two sensors are not one scale, so a fan's saturating 1.00 beats a revolution's
        # honest 0.67 from anywhere in the flat. The roster here is the board's own word, taken
        # at the moment the candidate is judged: the candidate is still judged, counted and
        # reported, it simply cannot become a re-seed.
        lidar_alive = any(s.name == LIDAR for s in self._registry.alive(self._now_s()))
        from_a_fan = candidate.source != LIDAR
        # ...and the one candidate a running goal does NOT stop: the graph's, with no lidar on
        # the roster. The no-re-seed-while-driving rule exists because a teleport mid-drive is
        # worse than a poor fit — but that reasoning assumes something else will correct the
        # pose, and with the lidar gone nothing will: on 2026-09-14 21:12 the cart drove 64 s on
        # a belief 2 m wrong with the graph recognising the place the whole way, every candidate
        # refused for the single reason that a goal was running.
        only_localizer = (
            candidate.source == GRAPH
            and not lidar_alive
            and self._switches.on("graph_reseed_while_driving")
        )
        # The fit the candidate is weighed against must be one THIS machine measured, on the same
        # scale as the candidate's own score: the same rule /localization_fit already follows
        # (``local_fit``, ``_fit_is_local``). Camera-only the tracker's raw fit is the remote
        # source's own claim — the graph claimed 1.00 on 2026-09-17 — and no honest lidar answer
        # can beat a claim of 1.00, so every candidate was judged "nothing" exactly while the pose
        # was 90 degrees out and the candidates were right.
        answer = self._candidates.observe(
            candidate,
            self._tracked_pose(),
            self.fit if self._fit_is_local else 0.0,
            map_id=self._chain.accepted(candidate.map_id, self._map_id),
            allow=(not self._navigating or only_localizer) and not (from_a_fan and lidar_alive),
            odometry=self._history,  # the trail the candidate is carried to this moment along
        )
        with self._episode:  # the search worker writes _pending_seed from its own thread
            self._pending_seed = answer.pending(self._map_id) or self._pending_seed

    def _on_measurement(self, msg: String) -> None:
        """The laptop's pose measurement off a camera scan (one JSON message,
        pepin.measurements): it waits at the gate for the next update, which carries it from the
        moment of its own scan to that update's moment and fuses it.

        Nothing is decided here — a measurement on another map is refused by the gate, a message
        that does not parse is counted — and the try at an update is the same one every input
        makes: with the lidar alive the measurement rides its next revolution, and with the lidar
        stale or gone it drives an update of its own (:meth:`_track_on_measurements`).
        """
        try:
            remote = RemoteMeasurement.from_json(msg.data)
        except (KeyError, TypeError, ValueError) as exc:
            self._measurements.malformed(str(exc))
            return
        self._measurements.offer(remote, self._chain.accepted(remote.map_id, self._map_id))
        self._track_pending()

    def _on_graph_measurement(self, msg: String) -> None:
        """RTAB-Map's word about where the cart is (one JSON message, pepin.measurements),
        measured on the laptop out of its pose graph: it waits at its own gate for the next
        update, which carries it from the graph's moment to that update's and fuses it beside
        the lidar's and the camera's. Judged exactly as the camera's is — another map's word is
        refused by the gate, a message that does not parse is counted — and it moves nothing
        unless the sources flag names `graph`."""
        try:
            remote = RemoteMeasurement.from_json(msg.data)
        except (KeyError, TypeError, ValueError) as exc:
            self._graph.malformed(str(exc))
            return
        self._graph.offer(remote, self._chain.accepted(remote.map_id, self._map_id))
        self._track_pending()

    def _track_on_measurements(self, now: float) -> None:
        """An update driven by the remote words alone — the camera's measurements, the pose
        graph's, whichever of them is the freshest waiting — at the stamp of that word.

        This is what a dead lidar leaves: no scan waits at the feed and nothing is fresh, so the
        feed has no anchor and the only word about where the cart is comes over the link. Which
        gate drives and which ride it is :func:`pepin.measurements.remote_update`'s decision and
        each gate's own (:meth:`pepin.measurements.MeasurementGate.drive`); here it is carried
        out, once per call, so two remote sources never make two updates out of one breath. The
        rest filter that spares the matcher while the cart stands does not apply — there is
        nothing to match, the match was made on the laptop — and the rest LOCK still does its
        work inside the tracker. The tracker takes its first fix this way too, from its saved
        pose, with no whole-map search: a fan cannot find the cart, and a pose measured off one
        cannot either.
        """
        loc = self._localizer
        plan = remote_update(
            (self._measurements, self._graph), self._feed.anchor(now), self._history
        )
        if loc is None or plan is None:
            return
        if not self._tracker_initialised:
            self._tracker_initialised = True
            self.get_logger().warning(
                f"tracker starts at ({loc.pose.x:+.2f}, {loc.pose.y:+.2f}, "
                f"{math.degrees(loc.pose.theta):+.0f} deg) on the {plan.measurement.source}'s "
                "measurements without a first search: a fan cannot find the cart"
            )
        previous_stamp = self._last_match_stamp_s
        self._last_match_stamp_s = plan.stamp
        dt_s = (
            plan.stamp - previous_stamp
            if previous_stamp is not None and plan.stamp > previous_stamp
            else None
        )
        self._last_scan_age_s = now - plan.stamp
        pose = loc.update_from(
            plan.odom,
            [],
            measurements=plan.measurements,
            at_rest=standing_still(self._history, plan.stamp, self._odom_wz),
            dt_s=dt_s,
        )
        self._publish_update(loc, pose, plan.odom, plan.stamp, now)

    def _publish_update(
        self, loc: Localizer, pose: Pose2D, odom: Pose2D, stamp: float, now: float
    ) -> None:
        """What every update publishes, however it was driven: map -> odom from the pose at this
        stamp, the tracked pose with the covariance its fit buys, the fit at the pose actually
        published, and every source's word as JSON — the scans', the watchdog's and the
        camera's."""
        self._last_map_odom = map_to_odom(pose, odom)
        self._frame.define()
        self._send_map_odom()
        # The step this transform takes IS what the word moved the pose by — the cart's own
        # motion sits in odom -> base_link — so the watch reads it and clears Nav2's local
        # costmap when the marks in it no longer belong to where the cart is.
        self._jumps.moved(self._last_map_odom, now)
        # One covariance, published twice: as the pose's own on /tracker_pose and as the two
        # sigmas every gate reads. An accepted word of ANY source lands here, so this is where
        # the uncertainty collapses — the camera's measurement exactly as the lidar's match.
        covariance = published_covariance(
            loc.fused, loc.confidence, str(self._switches["covariance"])
        )
        self._publish_tracker_pose(pose, covariance, Time(nanoseconds=int(stamp * 1e9)).to_msg())
        self._spread.corrected(covariance, pose, odom, now)
        self._publish_sigma(now)
        self._published_fit_pub.publish(Float32(data=float(loc.published_fit)))
        self._publish_sources(loc, now, odom)

    def _publish_sources(self, loc: Localizer, now: float, odom: Pose2D | None) -> None:
        """Publish every source's word as JSON on ``/localization/sources`` — the scans', the
        watchdog's and the camera's — with the candidates, the camera's measurements and the
        graph beside them.

        This goes out on the check period as well as on every update, because it is the only
        place the stack says WHO is holding the pose, and a report published on updates alone
        fell silent exactly where a reader needs it most: camera-only and parked, RTAB-Map adds
        no node, no word arrives, no update runs — and the goal gate, finding nothing on the
        topic, refused every camera-only start with "the tracker published no word at all"
        while the tracker sat there healthy at 0.20 m (2026-09-17).
        """
        report = loc.sources_report(now, odom)
        report["candidates"] = self._candidates.status()  # the laptop's word, beside the scans'
        report["measurements"] = self._measurements.status()
        report["graph"] = self._graph.status()
        self._sources_pub.publish(String(data=json.dumps(report)))

    def _clear_local_costmap(self, jump_m: float) -> None:
        """Ask Nav2 to empty its local costmap, because the pose has just jumped ``jump_m`` and
        the marks in that grid were laid where the cart used to be (:class:`pepin.watch.JumpClear`
        decides when).

        The call is asynchronous and its answer is never waited for: this runs on the frame
        path, between a scan and the transform it earns.
        """
        self._clears += 1
        self._clear_costmap.call_async(ClearEntireCostmap.Request())
        self.get_logger().info(
            f"pose jumped {jump_m:.2f} m: local costmap cleared, its marks were laid at the "
            "old pose"
        )

    def _track_pending(self) -> None:
        """Match the anchor's scan at the feed once the odometry history covers its whole
        revolution, with the other sources' scans — and the camera's measurements from the
        laptop — carried to its moment riding along.

        Called on every input: a scan usually arrives before the odometry of its last beams and
        is released by the odometry sample that completes it, 30-70 ms later. Which source's
        scan is released is the feed's call — in practice the lidar's, the only scan source this
        board still matches, and when nothing of its is fresh or waiting the camera's
        measurements drive the update instead (:meth:`_track_on_measurements`). The scan is
        deskewed
        with the history (every beam moved to where the robot was at the stamp) and the pose
        the localizer predicts from is the interpolated pose at that same stamp, so the
        residual the matcher reports is odometry error and nothing else.
        """
        loc = self._localizer
        if loc is None or not self._track:  # ``track`` false: AMCL owns the pose, we only watch
            return
        now = self._now_s()
        taken = self._feed.take(self._history, now)
        if taken is None:
            self._track_on_measurements(now)  # a dead lidar: the camera's word drives instead
            return
        anchor, scan = taken
        if not self._tracker_initialised:
            if self._registry.source(anchor).partial:
                # A fan cannot find the cart (two frames of the same view agree on the same
                # look-alike): no first search on it. The tracker follows the fan from its
                # saved pose and the watch judges the pose once a full revolution drives.
                self._tracker_initialised = True
                self.get_logger().warning(
                    f"tracker starts at ({loc.pose.x:+.2f}, {loc.pose.y:+.2f}, "
                    f"{math.degrees(loc.pose.theta):+.0f} deg) on the {anchor} fan without a "
                    "first search: a fan cannot find the cart"
                )
            else:
                if not self._tracker_initialising and not self._searching:
                    self._tracker_initialising = True
                    threading.Thread(
                        target=self._initialise_tracker, args=(scan.points,), daemon=True
                    ).start()
                return
        mono = time.monotonic()
        if self._pacer.skip(mono, self._searching):
            return
        odom = self._history.at(scan.stamp)
        if odom is None:  # cannot happen past the gate; a guard, not a fallback
            return
        if not self._motion.due(odom, scan.stamp):
            self._rested += 1  # standing still: the last match still holds, and so does the pose
            return
        # Seconds of robot time since the previous match, from the scan stamps (never the wall
        # clock): the rest lock's gain is a time constant, and this cadence is what it needs.
        previous_stamp = self._last_match_stamp_s
        self._last_match_stamp_s = scan.stamp
        dt_s = (
            scan.stamp - previous_stamp
            if previous_stamp is not None and scan.stamp > previous_stamp
            else None
        )
        self._last_scan_age_s = now - scan.stamp
        # How far this revolution reached: the radius inside which a cell that changes can still
        # move a score, and so the radius that decides whether a re-rendered grid is worth a
        # rebuild (pepin.mapping.MapShift.matters). One max over the 120 matched points.
        self._scan_reach_m = scan_reach_m(scan.points, self._scan_reach_m)
        points, whole = deskewed(scan, self._history)
        self._deskew_failed += not whole
        # Slip: the wheels claim a step but the scan is the same picture as a tenth of a second ago.
        # Then the wheel step is a lie; the pose is corrected from where it was, and Nav2's progress
        # checker (map frame) sees the truth: no progress -> a recovery instead of a 60 s wheelspin.
        slip = self._slip.observe(scan.ranges, odom)
        if self._slip.streak == SLIP_SAID_AFTER:
            self.get_logger().warning(
                "wheels turning, world standing still: slip, wheel step ignored"
            )
        self._slip_pub.publish(Bool(data=slip))
        # What the sensors say, and nothing else: the wheels' and the gyro's word on rest, the
        # static map's mask. Whether either is used is the Localizer's switch (live).
        at_rest = standing_still(self._history, scan.stamp, self._odom_wz)
        # The anchor's scan and every other enabled source's waiting scan, each moved into the
        # base frame of the anchor's stamp through the odometry between the two (the feed).
        scans = [
            ScanObservation(anchor, points, scan.stamp),
            *self._feed.gather(anchor, scan.stamp, self._history),
        ]
        t0 = time.perf_counter()
        pose = loc.update_from(
            odom,
            scans,
            measurements=[
                *self._measurements.take(scan.stamp, self._history),
                *self._graph.take(scan.stamp, self._history),
            ],
            trust_odometry=not slip,
            at_rest=at_rest,
            dt_s=dt_s,
            mask=self._static_mask,
        )
        self._pacer.matched(mono, time.perf_counter() - t0)
        self._publish_update(loc, pose, odom, scan.stamp, now)

    def _start_pose(self) -> Pose2D:
        """Where to put the tracker on a map it has no pose on: the pose the previous run saved if
        it is recent, was good and was on THIS map, else the pose this board is already publishing.

        THE SECOND ANSWER IS THE ONE WORLD R CHANGED, and it matters most in a map that has just
        been born. RTAB-Map roots a fresh session's map frame at the odometry pose of its first
        node and optimises from the oldest node (``RGBD/OptimizeFromGraphEnd`` false), so until the
        first loop closure that map frame IS the board's odom frame and the cart's place in it is
        simply its odometry pose — which is exactly what this node has been broadcasting all along
        (``map -> odom`` identity until the first fix). The old answer was the map's ORIGIN, a
        corner of the grid the cart has never been at: harmless while the odometry sat near zero,
        and a lie worth metres anywhere else, published as a correction the moment the first word
        landed.
        """
        start = saved_start(
            FilePath(LAST_POSE_FILE), self._map_id, time.time(), LAST_POSE_MAX_AGE_S, LOST_FIT
        )
        self.get_logger().info(start.note)  # either way: a silent fallback is found an hour late
        return start.pose or self._tracked_pose() or Pose2D()

    def _remember_pose(self) -> None:
        """Every 2 s: write the tracked pose and its fit, so the next start knows where we are."""
        loc = self._localizer
        if loc is None or not self._tracker_initialised or self.fit < LOST_FIT:
            return
        record = {
            "x": loc.pose.x,
            "y": loc.pose.y,
            "theta": loc.pose.theta,
            "fit": self.fit,
            "time": time.time(),
            "map": self._map_id,
        }
        try:
            with open(LAST_POSE_FILE + ".tmp", "w") as f:
                json.dump(record, f)
            os.replace(LAST_POSE_FILE + ".tmp", LAST_POSE_FILE)
        except OSError:
            pass

    def _initialise_tracker(self, points: Any) -> None:
        """Worker thread: the first fix is a candidate like any other, confirmed by a second search.

        It used to seed whatever one whole-map search returned, unconditionally — even a fix the
        localizer had rejected — and that single seed could discard a candidate the watch was
        holding. Now the tracker starts from its saved pose, a whole-map search proposes, and the
        1 Hz check asks again on a fresh scan; the pose moves once two searches agree.

        AND A POSE THAT ALREADY FITS IS NOT SEARCHED FOR. The scan is scored where the tracker
        already stands first, and a fit at or above the watch's own "lost" line ends the matter:
        the cart is where it thinks it is and there is nothing for a whole-map search to propose.
        That is what makes a map born under the cart safe — a newborn grid is built FROM this very
        revolution at this very odometry pose, so the scan lies on it perfectly, while a search over
        a grid that is one scan wide has no unique answer and would happily teleport the cart across
        its own blob. It also saves a known room's cold boot the 10-20 s search it never needed
        after a restart at rest.
        """
        loc = self._localizer
        assert loc is not None
        scan = self._scan_id
        try:
            fit = 0.0 if self._matcher is None else self._matcher.inlier_fraction(loc.pose, points)
            if fit >= self._watch.lost_fit:
                self.get_logger().info(
                    f"tracker starts at ({loc.pose.x:+.2f}, {loc.pose.y:+.2f}, "
                    f"{math.degrees(loc.pose.theta):+.0f} deg): the scan already fits there at"
                    f" {fit:.2f} (over {self._watch.lost_fit:.2f}), no whole-map search"
                )
                self._tracker_initialised = True
                return
            found, confidence = loc.global_search(points, prior=loc.pose)
            yaw = math.degrees(found.pose.theta)
            where = f"({found.pose.x:+.2f}, {found.pose.y:+.2f}, {yaw:+.0f} deg)"
            self.get_logger().info(
                f"tracker starts at ({loc.pose.x:+.2f}, {loc.pose.y:+.2f}, "
                f"{math.degrees(loc.pose.theta):+.0f} deg); first search proposes {where} "
                f"at fit {confidence:.2f}"
            )
            with self._episode:
                self._watch.answer(found.pose, confidence, 0.0, scan, time.monotonic())
            self._tracker_initialised = True
        finally:
            self._tracker_initialising = False

    def _occluded(self, points: Any, pose: Pose2D) -> bool:
        """A person beside the cart rather than a lost cart (:func:`pepin.dynamic.occluded`),
        judged on the newest scan ``points`` at ``pose``."""
        return occluded(points, pose, self._static_mask, self._matcher, self._watch.lost_fit)

    def _send_map_odom(self) -> None:
        """Broadcast the current map -> odom, 20 times a second and after every match, dated
        ``tf_future_s`` (0.1 s) ahead like AMCL does.

        The contract for every consumer: compose the LATEST map -> odom with odom -> base_link
        at your own stamp. map -> odom is a slowly moving correction, not a trajectory — its
        stamp says "valid from here", never "measured here" — while odom -> base_link carries
        the motion and is exact at its stamp. A consumer that asks tf2 for map -> base_link at a
        stamp gets exactly that composition; the short future date keeps "now" inside the
        buffer so nobody extrapolates, and keeps a fresh correction from waiting half a second
        behind a stale future-dated one.
        """
        if self._frame.silent(self._switches.on("frame_needs_a_pose")):
            return  # a known map, and no pose on it yet: say nothing rather than "the origin"
        x, y, yaw = self._last_map_odom
        future = self.get_clock().now() + Duration(seconds=self._tf_future_s)
        self._tf_pub.sendTransform(
            transform_from_rpy("map", "odom", (x, y, 0.0), (0.0, 0.0, yaw), future.to_msg())
        )

    def _publish_tracker_pose(self, pose: Pose2D, covariance: Matrix, stamp: Any) -> None:
        """The tracked pose for the operator's view and the trail, with the covariance the
        ``covariance`` flag asked for — the same 3x3 :meth:`_publish_sigma` reports, so the pose
        and the sigma can never tell two stories.

        ``peak``: the full 3x3 of the last update's fused match — the spread of the score peak
        the pose was actually corrected by, anisotropic, so a corridor reads as a ridge along
        the corridor and the laptop's watchdog is judged against a real one. ``fit``: the
        isotropic pair :func:`pepin.fusion.sigma_from_fit` draws from the inlier fraction, which
        is what every tape before 2026-09-13 carries. Before the first match, and whenever the
        tracker has no fused measurement to publish (a carried belief, a re-seed), the fit's
        numbers are used either way: there is no peak to report.
        """
        msg = pose_with_matrix(pose.x, pose.y, pose.theta, covariance, stamp, "map")
        self._tracker_pub.publish(msg)
        self._append_trail(msg)

    def _publish_sigma(self, now: float) -> None:
        """Publish how sure the pose is on :data:`SIGMA_TOPIC`, as the JSON
        :class:`pepin.watch.Sigma` defines: position sigma (m), heading sigma (deg), whether
        anything has ever corrected this pose, this clock, and the seconds since the last
        accepted word.

        Sent from two places and no others: from every update, where the fusion has just
        corrected the pose, and from the once-a-second check, where nothing has and the spread
        has grown along the odometry instead. So the topic keeps carrying a number with the
        lidar dead, the link down and the camera silent — a number that grows until every gate
        downstream refuses, which is the whole point of measuring uncertainty instead of a fit.
        """
        xy, yaw = self._spread.sigma()
        age = self._spread.age_s(now)
        word_age = age if age != math.inf else max(0.0, now - self._up_since_s)
        said = Sigma(xy, yaw, 0.0, self._spread.known())
        self._sigma_pub.publish(String(data=said.to_json(stamp=now, word_age_s=word_age)))

    def _report_tracking(self) -> None:
        """Every 30 s: where every released scan went (per source, with the rides), what the
        tracker did with the matched ones (its switches, the rest lock's cadence and gain,
        carries, lost/weak, the fit at the match and at the published pose, the largest
        map -> odom step), who drives and every source's health, what the laptop's watchdog
        proposed and what came of it, how many jumps emptied Nav2's local costmap, and the
        cost."""
        loc = self._localizer
        if loc is None:
            return
        feed, pacer, track = self._feed.report(), self._pacer.report(), loc.report()
        # What the last adoption did to the map: a loop closure's bend is a moved origin and a few
        # thousand changed cells, and this is the only place it shows (pepin.mapping.map_shift).
        shift = "" if self._shift is None else f", last {self._shift.phrase()}"
        self.get_logger().info(
            f"tracker: {feed.summary()}, rested {self._rested}, {pacer.summary()}, deskew "
            f"failed {self._deskew_failed}, odometry runaway {self._runaways} "
            f"({self._yaw_runaways} yaw); "
            f"{loc.settings()}; {track.summary()}; "
            f"sources: {self._feed.status(self._now_s())}; "
            f"watch {'fit' if self._watch_on else 'off: no full-turn source, fit'} "
            f"{self.fit:.2f}"
            f"{'' if self._fit_is_local else ' (the laptop measured it: published as 0.00)'}"
            f", {self._silence.phrase(self._source_age_s)}"
            f"{' (published as 0.00)' if self._silence.held_at_zero(self._source_age_s) else ''}"
            f", scan age at match {self._last_scan_age_s * 1000:.0f} ms; "
            f"{self._spread.text(self._now_s())} "
            f"(drive under {DRIVE_SIGMA_M:.2f} m, a drive is cut over {LOST_SIGMA_M:.2f} m); "
            f"map {self._choice.source or 'none'}"
            f"{'' if self._cached is None else ' [' + self._cached.phrase() + ']'} "
            f"(id {self._map_id or 'none'}, {self._chain.text()} this frame, "
            f"{self._choice.adoptions} adopted, "
            f"{self._choice.take_ignored()} republications ignored{shift}); "
            f"{self._candidates.report()}; {self._measurements.report()}; "
            f"graph: {self._graph.report()}; "
            f"costmap cleared {self._clears} times on jumps; "
            f"flags: {self._switches.state()}"
        )
        if feed.expired:
            self.get_logger().warning(
                f"odometry ran late: {feed.expired} scans were never covered by {self._odom_topic} "
                "and matched nothing"
            )
        self._rested = self._deskew_failed = self._runaways = self._yaw_runaways = 0

    def _lookup_laser(self, frame: str) -> bool:
        """The static base_link <- laser transform, as x, y, yaw and whether roll is pi."""
        t = self._tf.transform("base_link", frame)
        if t is None:  # not yet available: try on the next scan
            return False
        x, _y, yaw, mirrored = self._laser_tf = planar_mount(t)
        self.get_logger().info(
            f"laser mount: x {x:.3f} yaw {math.degrees(yaw):.1f} deg"
            f"{' upside down' if mirrored else ''}"
        )
        self.create_subscription(
            GoalStatusArray, "/navigate_to_pose/_action/status", self._on_nav_status, 10
        )
        return True

    def _on_particles(self, msg: ParticleCloud) -> None:
        """Every AMCL particle set, thinned to 300 arrows, as a PoseArray for the 3D view."""
        step = max(1, len(msg.particles) // 300)
        out = PoseArray()
        out.header = msg.header
        out.poses = [particle.pose for particle in msg.particles[::step]]
        self._particles_pub.publish(out)

    def _on_amcl_pose(self, msg: PoseWithCovarianceStamped) -> None:
        """AMCL's pose feeds the trail only when AMCL, not this node, owns the localisation."""
        if not self._track:
            self._append_trail(msg)

    def _append_trail(self, msg: PoseWithCovarianceStamped) -> None:
        """Append a pose to the trail (last 600 poses) and republish it, latched."""
        stamped = PoseStamped()
        stamped.header = msg.header
        stamped.pose = msg.pose.pose
        self._trail.poses.append(stamped)
        del self._trail.poses[:-600]
        self._trail.header.stamp = msg.header.stamp
        self._trail_pub.publish(self._trail)

    def _on_nav_status(self, msg: GoalStatusArray) -> None:
        """Remember whether Nav2 is executing a goal: no automatic re-seeding while it drives."""
        self._navigating = any(
            s.status in (GoalStatus.STATUS_ACCEPTED, GoalStatus.STATUS_EXECUTING)
            for s in msg.status_list
        )

    def _moving(self) -> bool:
        """Did base_link move in the odom frame since the previous check (1 cm or 1 degree)?"""
        return self._motion_edge.moved(self._history.newest)

    def _on_tf_static(self, msg: TFMessage) -> None:
        """The static frames (latched): the laser mount is read from them once."""
        for transform in msg.transforms:
            self._tf.buffer.set_transform_static(transform, "static")

    def _tracked_pose(self) -> Pose2D | None:
        """map -> base_link now: this node's map -> odom over the newest odometry pose."""
        odom = self._history.newest
        if odom is None:
            return None
        x, y, yaw = self._last_map_odom
        c, s = math.cos(yaw), math.sin(yaw)
        return Pose2D(
            x + c * odom.x - s * odom.y, y + s * odom.x + c * odom.y, wrap_angle(yaw + odom.theta)
        )

    # -- the watch ------------------------------------------------------------

    def _check(self) -> None:
        """Once a second: score the fit on the newest picture of the source that drives the
        tracker; the watch decides whether to search the whole map — on a full revolution
        only: while anything narrower drives, the fit is published and the watch is off.

        With no scan of our own at all — the camera's measurements driving the tracker — there
        is nothing here to score, and what goes out on ``/localization_fit`` is 0.0 rather than
        the fit the laptop measured (``local_fit``): that number was scored against the camera's
        own band of the volume, which depth_fusion paints only while this very topic says the
        cart is localised, so publishing it here closes a circle instead of reporting anything.
        The camera's fit is on ``/localization/sources``, per source, where it says whose word
        it is.
        """
        # Before every return below: a node with no map takes them all. With nothing live inside
        # the patience, the map this board wrote down itself is what it tracks on (rule 20).
        self._take_cached_map()
        now = self._now_s()
        # The filter's prediction step, and the sigma published whatever else this tick decides:
        # nothing corrected the pose since the last update, so it grew along the odometry. Before
        # every early return below, for the same reason as the silence underneath — the pose's
        # uncertainty is a fact about the ODOMETRY since the last word, not about whether this
        # node has a map, a scan or a pose yet — and it is what a goal gate will refuse on.
        # The turn term is read off the flag every tick, so it can be moved on a standing cart.
        self._spread.yaw_per_turn = float(self._switches["belief_yaw_per_turn"])
        self._spread.carried(self._history.newest)
        self._publish_sigma(now)
        # Measured before anything can return: the silence is a fact about the SENSORS, not
        # about whether this node has a map and a pose yet. Left behind the early return below,
        # the report line said "no source ever" for as long as the map took to arrive, with the
        # lidar turning at 10 Hz the whole time — and said "(published as 0.00)" about a topic
        # nothing had published to.
        self._source_age_s = self._registry.silence_s(now)
        self._silence = SourceSilence(
            float(self._switches["source_patience_s"]), self._switches.on("fit_needs_a_source")
        )
        full = self._feed.full_picture(now)
        picture = full if full is not None else self._feed.picture(now)
        loc = self._localizer
        pose = self._tracked_pose()
        if self._matcher is None or loc is None or pose is None:
            return
        # Beside the sigma, and for the same reason: who is holding the pose is a fact a reader
        # needs most when nothing is speaking, and the only publication used to ride an update.
        self._publish_sources(loc, now, self._history.newest)
        # Sampled exactly once: _moving() is an edge detector on odometry, and a second call in
        # the same tick compared a reading with itself and told the watch the robot stood still
        # at every speed — the "never re-seed a moving robot" gate was dead (review, 2026-09-09).
        moving = self._moving()
        # The tracker's OWN confidence — the inlier fraction of its last match, scan and pose
        # from the same instant — not a re-evaluation of the TF pose against the latest scan.
        # Those two are 100-200 ms apart, and at 0.5 rad/s that is 3-6 degrees: the re-evaluated
        # fit collapsed to 0.19-0.31 during every pivot while the tracker itself sat at 0.90-0.95
        # (run 0070, second by second), and a blind-drive rule built on the false number stopped
        # healthy drives mid-turn and moved the belief by 0.4 m to "recover" from nothing.
        self.fit = (
            float(loc.confidence)
            if moving or picture is None
            else self._matcher.inlier_fraction(pose, picture.points)
        )
        # ...and whether a scan of ours scored it at all. A fit nothing here measured is not this
        # machine's fit to report: it goes out as 0.0 — "this board cannot vouch for the pose",
        # the value the topic carries before the first match — and the camera's own number rides
        # /localization/sources with its source's name on it.
        self._fit_is_local = picture is not None or not self._switches.on("local_fit")
        reported = self._watch.reported_fit(self.fit if self._fit_is_local else 0.0)
        # ...and whether anything at all still speaks. A fit measured minutes ago is not a fit:
        # on 2026-09-14 this topic carried 0.70 for 141 s with no source arriving and the goal
        # server drove two goals on dead reckoning. Silent for longer than the patience, it goes
        # out as 0.00 — under every rung of the ladder the goal server and its blind-drive watch
        # read. The watch below is untouched: it takes the tracker's OWN fit, so a silent sensor
        # cannot start a whole-map search on a scan that is not there. Both the silence and the
        # switches behind it were read at the top of this tick, before any early return.
        self._fit_pub.publish(
            Float32(data=float(self._silence.reported(reported, self._source_age_s)))
        )
        # A fan drives, or nothing here does: such a fit cannot say "lost" (LOST_FIT was tuned on
        # full revolutions) and a search on it would re-seed the tracker on a look-alike the twin
        # check cannot see — the watch is off, and the report line says so.
        self._watch_on = full is not None
        if full is None or self._searching or not self._tracker_initialised:
            return
        # The watch gets the tracker's OWN fit. reported_fit() is for the outside world: capped
        # at 0.35 while a candidate pends, and fed back here it kept a twin candidate alive at
        # fit 0.76 and ran a whole-map search every second for half an hour (18:00 today).
        occluded = self._occluded(full.points, pose)
        # Under the episode lock, like every other _watch call and every claim of _searching:
        # the worker answers a search on its own thread and the two share this state.
        with self._episode:
            search = (
                self._watch.observe(
                    self.fit,
                    moving=moving,
                    navigating=self._navigating,
                    now=time.monotonic(),
                    occluded=occluded,
                )
                and not self._searching
            )
            if search:
                self._searching = True
        if search:
            self.get_logger().warning(f"fit {self.fit:.2f}: searching the whole map")
            threading.Thread(target=self._search_and_seed, args=(full,), daemon=True).start()
        elif occluded and self.fit < self._watch.lost_fit:
            self.get_logger().info(
                f"fit {self.fit:.2f} but the scan is mostly things the map does not know: "
                "occluded, not lost"
            )

    def _dump_failure(
        self, points: Any, current: Pose2D | None, best: Pose2D | None, confidence: float
    ) -> None:
        """Write the scan and the verdict of a failed search as JSON (a few tens of kB)."""
        try:
            os.makedirs(DUMP_DIR, exist_ok=True)
            path = f"{DUMP_DIR}/reloc_fail_{time.strftime('%Y%m%d_%H%M%S')}.json"
            with open(path, "w") as f:
                json.dump(
                    {
                        "points": np.round(points, 3).tolist(),
                        "current": None
                        if current is None
                        else [current.x, current.y, current.theta],
                        "best": None if best is None else [best.x, best.y, best.theta],
                        "confidence": confidence,
                    },
                    f,
                )
            self.get_logger().info(f"search dumped to {path}")
        except OSError as exc:
            self.get_logger().warning(f"could not dump the failed search: {exc}")

    def _apply_pending_seed(self) -> None:
        """Adopt what the search worker found, unless the map changed while it was searching.

        The worker computes; the executor applies. A search runs for seconds, and a map swap in
        the middle used to hand the new localizer a pose measured on the old map.
        """
        with self._episode:
            pending, self._pending_seed = self._pending_seed, None
        if pending is None:
            return
        map_id, pose, confidence = pending
        if map_id != self._map_id:
            self.get_logger().warning("a search finished on the old map: its fix is dropped")
            return
        self._seed(pose, confidence)

    def _search_and_seed(self, picture: TimedScan) -> None:
        """Worker thread: the search must not block the executor (scan and service callbacks)."""
        try:
            self._relocalize(picture)
        finally:
            with self._episode:
                self._searching = False

    def _relocalize(self, picture: TimedScan) -> str:
        """One whole-map search on ``picture`` — the full revolution the caller decided on
        (:meth:`~pepin.sources.SourceFeed.full_picture`; never a fan); the watch decides what
        its answer is worth."""
        assert self._localizer is not None and self._matcher is not None
        # The scan's id and the map it was taken on, captured together at the start: a map
        # swap during the seconds of a search must not stamp the old scan's fix as the new map's.
        map_id = self._map_id
        points, scan = picture.points, picture.scan_id
        current = self._tracked_pose()
        current_fit = self._matcher.inlier_fraction(current, points) if current else 0.0
        started = time.monotonic()
        # Always the whole map: a "nearby first" shortcut accepted a 0.66 impostor two metres from
        # a carried robot (2026-09-06 18:14) and the whole-map stage never ran. The previous belief
        # only breaks ties between look-alikes.
        credible = current is not None and current_fit >= 0.4  # a stale belief must not break ties
        found, confidence = self._localizer.global_search(
            points,
            theta_step_deg=10.0,
            thin_to=90,
            prior=current if credible else None,
            refuse_twins=credible,
        )
        took = time.monotonic() - started
        answer = found.pose if found is not None else None
        with self._episode:
            verdict = self._watch.answer(answer, confidence, current_fit, scan, time.monotonic())
            if verdict.verdict is Verdict.APPLY and verdict.pose is not None:
                self._pending_seed = (map_id, verdict.pose, verdict.confidence)
        where = (
            "nothing"
            if answer is None
            else f"({answer.x:+.2f}, {answer.y:+.2f}, {math.degrees(answer.theta):+.0f} deg)"
        )
        fits = f"fit {confidence:.2f} vs now {current_fit:.2f}"
        text = {
            Verdict.NOTHING: f"no better pose ({took:.1f} s): {fits}",
            Verdict.CANDIDATE: f"candidate ({took:.1f} s): {where} {fits}; asking once more",
            Verdict.REPLAY: "same scan as the candidate: no second opinion yet",
            Verdict.HOLD: f"second search disagrees ({took:.1f} s): candidate {where}",
            Verdict.APPLY: f"relocalised, agreed by a second search ({took:.1f} s): to {where}",
            Verdict.GIVEN_UP: (
                f"searches keep disagreeing: the scan fits two places alike, last {where}; "
                f"tracking at fit {current_fit:.2f} until the robot moves"
            ),
        }[verdict.verdict]
        if verdict.verdict is Verdict.NOTHING:
            self._dump_failure(points, current, answer, confidence)
        # Two call sites on purpose: rclpy keys a logger call's severity on its source line and
        # raises "Logger severity cannot be changed between calls" when one line logs both.
        if verdict.verdict in (Verdict.CANDIDATE, Verdict.APPLY):
            self.get_logger().info(text)
        else:
            self.get_logger().warning(text)
        return text

    def _on_operator_seed(self, msg: PoseWithCovarianceStamped) -> None:
        """A pose handed in on /initialpose is adopted as the truth at full confidence."""
        p = msg.pose.pose
        pose = Pose2D(float(p.position.x), float(p.position.y), yaw_of(p.orientation))
        self.get_logger().info(
            f"seeded by the operator at ({pose.x:.2f}, {pose.y:.2f},"
            f" {math.degrees(pose.theta):.0f} deg)"
        )
        self._seed(pose, 1.0)

    def _seed(self, pose: Pose2D, confidence: float) -> None:
        """Adopt ``pose`` with the confidence it was measured at: the localizer, the fit, map ->
        odom and the watch follow at once.

        Nothing goes back out on /initialpose. This node listens there for the operator, and
        the old "tell AMCL" publication fed the node its own seed: every adoption came back as
        an operator seed, 20 times a second, and two hand seeds alternated for an hour with
        map -> odom flipping 16 cm at 38 Hz and the rest lock reset on every turn (2026-09-13).
        """
        if self._localizer is not None:
            self._localizer.adopt(pose, confidence)
        self.fit = confidence
        # map->odom follows the seed NOW: the next matched scan may be a second away at rest, and
        # for that second /localization_fit would say "found" while the transform still placed the
        # cart where it stood before a carry (a review probe, 2026-09-11).
        newest = self._history.newest
        self._last_map_odom = (
            map_to_odom(pose, newest) if newest is not None else self._last_map_odom
        )
        self._frame.defined = self._frame.defined or newest is not None
        self._send_map_odom()
        # A seed is an accepted word like any other — the operator's hand, or two searches that
        # agreed — so the uncertainty collapses to what that confidence buys and starts growing
        # again from here. Without this the sigma went on growing from a pose nobody holds any
        # more, and a hand seed could not clear a refusal.
        self._spread.corrected(published_covariance(None, confidence), pose, newest, self._now_s())
        self._publish_sigma(self._now_s())
        self._motion.reset()
        with self._episode:  # the worker may be inside _watch.answer() right now
            self._watch.seeded(time.monotonic())

    # -- services ---------------------------------------------------------

    def _on_relocalize(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        """Start a whole-map episode now (or report the one running); the answer is on
        /localization_fit within a few seconds. The search runs in the worker, never here: a
        loop on the executor thread froze the scan, so its "second opinion" was the first search
        replayed to the millimetre and every candidate was rubber-stamped (review, 2026-09-09)."""
        now = self._now_s()
        picture = self._feed.full_picture(now)
        if self._localizer is None or picture is None:
            res.success = False
            res.message = (
                "no map or no scan yet"
                if self._localizer is None or self._feed.picture(now) is None
                else "no full-turn scan to search with: a fan cannot find the cart"
            )
            return res
        with self._episode:
            if self._searching:
                res.success, res.message = True, "a search is already running"
                return res
            self._searching = True
        threading.Thread(target=self._search_and_seed, args=(picture,), daemon=True).start()
        res.success = True
        res.message = (
            "searching the whole map; a fix needs two searches that agree — watch /localization_fit"
        )
        return res

    def _on_where(self, _req: Trigger.Request, res: Trigger.Response) -> Trigger.Response:
        pose = self._tracked_pose()
        if pose is None:
            res.success, res.message = False, "no map->base_link transform yet"
            return res
        res.success = True
        res.message = (
            f"x {pose.x:+.2f} m, y {pose.y:+.2f} m, yaw {math.degrees(pose.theta):+.0f} deg;"
            f" scan-to-map fit {self._watch.reported_fit(self.fit):.2f}"
            f" (good > {DRIVE_FIT}, lost < {self._watch.lost_fit});"
            f" {self._spread.text(self._now_s())}, the number a drive is judged by"
            + ("" if self._watch.confirmed else "; UNCONFIRMED: waiting for a second search")
        )
        return res


def main(args: list[str] | None = None) -> None:
    """Entry point."""
    spin_main(Relocalizer, args)


if __name__ == "__main__":
    main()
