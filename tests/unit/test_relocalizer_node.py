"""The tracker node under the ROS stubs: three scan topics into one feed, the lidar's death
handing the updates to the camera on the node's own trigger path and a returning lidar
taking them back, the flags that pick the sources, and every source's word on the wire.

rclpy is faked (``ros_stubs``); the map, the scans and the odometry are the furnished room's
of test_localization, so the node is built and driven here exactly as on the board, scan by
scan, with no robot.
"""

from __future__ import annotations

import itertools
import json
import math
import time
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest
import ros_stubs

RCLPY = ros_stubs.install()

from pepin_bringup.relocalizer import FLAGS, Relocalizer  # noqa: E402
from ros_stubs import (  # noqa: E402
    Header,
    LaserScan,
    Odometry,
    Parameter,
    String,
    TransformStamped,
)
from ros_stubs import OccupancyGrid as OccupancyGridMsg  # noqa: E402
from ros_stubs import Time as TimeMsg  # noqa: E402
from synthetic import raycast_room  # noqa: E402
from test_localization import PILLAR, furnished_room_map  # noqa: E402
from test_localizer_sources import drive, error  # noqa: E402

from pepin.odometry import Pose2D  # noqa: E402
from pepin.scanmatch import apply_motion  # noqa: E402
from pepin.sources import CONTACT, DEPTH, LIDAR  # noqa: E402

BEAMS = 180
FAN = [*range(160, 180), *range(0, 21)]  # the raycast's beams within +-40 degrees, in order


def stamp(t: float) -> Any:
    whole = math.floor(t)
    return TimeMsg(sec=whole, nanosec=round((t - whole) * 1e9))


def map_msg(grid: Any = None) -> Any:
    """A grid as the map server publishes it; the furnished room by default."""
    grid = furnished_room_map() if grid is None else grid
    rows, cols = grid.spec.shape
    msg = OccupancyGridMsg()
    msg.info.resolution = grid.spec.resolution_m
    msg.info.width, msg.info.height = cols, rows
    msg.info.origin.position.x, msg.info.origin.position.y = grid.spec.x_min_m, grid.spec.y_min_m
    odds = grid.log_odds[:rows, :cols]
    msg.data = np.where(odds > 0.5, 100, np.where(odds < -0.5, 0, -1)).ravel().tolist()
    return msg


def lidar_msg(truth: Pose2D, t: float) -> Any:
    """The lidar's revolution from ``truth``, stamped ``t``, in the laser frame (identity)."""
    points = raycast_room(truth, beams=BEAMS, pillar=PILLAR)
    return LaserScan(
        header=Header(stamp=stamp(t), frame_id="laser"),
        angle_min=0.0,
        angle_increment=2.0 * math.pi / BEAMS,
        range_max=12.0,
        ranges=np.hypot(points[:, 0], points[:, 1]).tolist(),
    )


def depth_msg(truth: Pose2D, t: float) -> Any:
    """The camera's +-40 degree fan from ``truth`` as the depth node publishes it: a LaserScan
    in base_link, stamped with the frame."""
    points = raycast_room(truth, beams=BEAMS, pillar=PILLAR)[FAN]
    return LaserScan(
        header=Header(stamp=stamp(t), frame_id="base_link"),
        angle_min=math.radians(-40.0),
        angle_increment=math.radians(2.0),
        range_max=6.0,
        ranges=np.hypot(points[:, 0], points[:, 1]).tolist(),
    )


def odom_msg(pose: Pose2D, t: float) -> Any:
    msg = Odometry(header=Header(stamp=stamp(t), frame_id="odom"))
    msg.pose.pose.position.x, msg.pose.pose.position.y = pose.x, pose.y
    msg.pose.pose.orientation.z = math.sin(pose.theta / 2.0)
    msg.pose.pose.orientation.w = math.cos(pose.theta / 2.0)
    return msg


def until(predicate: Callable[[], Any], timeout_s: float = 20.0) -> bool:
    """Wait for the initialising thread; True when ``predicate`` came true in time."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def between(a: Pose2D, b: Pose2D, share: float) -> Pose2D:
    """The pose ``share`` of the way from ``a`` to ``b`` (the drive is linear in its step)."""
    return Pose2D(
        a.x + share * (b.x - a.x), a.y + share * (b.y - a.y), a.theta + share * (b.theta - a.theta)
    )


@pytest.fixture
def node() -> Relocalizer:
    """A tracker with the camera's fan enabled beside the lidar, the laser mounted at the
    base's origin, the furnished room as its map."""
    # No match gap: the pacer spaces matches on the wall clock, and a test feeds a second of
    # scans in milliseconds.
    with ros_stubs.parameters(sources="lidar,depth", min_match_gap_s=0.0):
        node = Relocalizer()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    node.subs["/map"][1](map_msg())
    return node


def test_the_node_subscribes_every_source_and_the_flags_pick_them(node: Relocalizer) -> None:
    assert {"/scan", "/depth_scan", "/contact_scan", "/odometry/filtered", "/map"} <= set(node.subs)
    assert "/localization/sources" in node.pubs and "/tracker_pose" in node.pubs
    assert FLAGS["sources"] == (LIDAR,), "the module's default: the lidar alone"
    assert node._registry.enabled == (LIDAR, DEPTH), "the launch override reached the roster"
    assert node._localizer is not None and node._localizer.sources is node._registry
    assert node.set_parameters([Parameter("sources", value="lidar")])[0].successful
    assert node._registry.enabled == (LIDAR,) and node._localizer.sources.enabled == (LIDAR,)
    refused = node.set_parameters([Parameter("sources", value="lidar,sonar")])[0]
    assert not refused.successful and "sonar" in refused.reason
    assert node.set_parameters([Parameter("fusion", value=False)])[0].successful
    assert node._localizer.fusion is False
    assert node.set_parameters([Parameter("sources", value="depth,contact")])[0].successful
    assert node._registry.enabled == (DEPTH, CONTACT)
    assert "sources=depth,contact fusion=off" in node._switches.state()


def test_a_dead_lidar_hands_the_node_to_the_camera_and_back_on_its_own_trigger(
    node: Relocalizer,
) -> None:
    """The drive of test_localizer_sources through the node's callbacks: the lidar's revolution
    on /scan, the camera's fan on /depth_scan 40 ms before it, the odometry that covers them.
    The lidar goes silent for 1.2 s in the middle: the feed waits the roster's half second,
    then the fan drives the updates and map -> odom keeps moving; when the lidar is back it
    anchors again. Every update's word goes out on /localization/sources."""
    truth, odom = drive(36)
    loc = node._localizer
    assert loc is not None
    sources_pub, pose_pub = node.pubs["/localization/sources"], node.pubs["/tracker_pose"]
    on_scan, on_depth = node.subs["/scan"][1], node.subs["/depth_scan"][1]
    on_odom = node.subs["/odometry/filtered"][1]
    anchors: list[str | None] = []
    map_odom_at: dict[int, tuple[float, float, float]] = {}
    for i, (o, t) in enumerate(zip(odom, truth, strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.02  # the node's clock at the arrival
        if i:
            on_depth(depth_msg(between(truth[i - 1], t, 0.6), ts - 0.04))
        if not 10 <= i < 22:
            on_scan(lidar_msg(t, ts))
        published = len(sources_pub.sent)
        on_odom(odom_msg(o, ts))
        if i == 0:  # the first release starts the whole-map search that seeds the watch
            assert until(lambda: node._tracker_initialised)
            assert not sources_pub.sent, "the first scan went to the search, not the tracker"
            anchors.append(None)
            continue
        if len(sources_pub.sent) == published:
            anchors.append(None)
            continue
        report = json.loads(sources_pub.sent[-1].data)
        anchors.append(report["anchor"])
        map_odom_at[i] = node._last_map_odom
        metres, degrees = error(loc, t)
        assert metres < 0.12 and degrees < 4.0, f"scan {i}: {metres * 100:.1f} cm, {degrees:.1f}"
    assert anchors[:2] == [None, LIDAR] and anchors[2:10] == [LIDAR] * 8
    assert anchors[10:14] == [None] * 4, "the roster's half second: the feed waits for the lidar"
    assert anchors[14:22] == [DEPTH] * 8, "then the camera drives"
    assert anchors[22:] == [LIDAR] * 14, "the lidar is back and anchors again"
    assert map_odom_at[21] != map_odom_at[9], "map -> odom moved on the camera's word"
    metres, degrees = error(loc, truth[-1])
    assert metres < 0.05 and degrees < 2.0
    assert len(pose_pub.sent) == len(sources_pub.sent) == 35 - 4
    camera = json.loads(sources_pub.sent[13].data)  # the first update the camera drove
    assert camera["anchor"] == DEPTH and camera["fused"] == DEPTH and camera["rejected"] == []
    assert camera["sources"][LIDAR]["health"].startswith("stale")
    assert camera["sources"][DEPTH]["health"].startswith("fresh")
    assert {"fit", "delta", "sigma", "edge"} <= set(camera["sources"][DEPTH])
    assert "fit" not in camera["sources"][LIDAR] and camera["sources"][CONTACT] == {"health": "off"}
    fused = json.loads(sources_pub.sent[-1].data)
    assert fused["anchor"] == LIDAR and fused["fused"] == "lidar+depth"
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "sources: anchor lidar; lidar fresh" in line and "depth fresh" in line
    assert "lidar: scans" in line and "attached" in line
    assert "sources lidar,depth, fusion on" in line, "the tracker's settings"
    assert "flags: rest_lock=on" in line and "sources=lidar,depth fusion=on" in line


def test_a_late_executor_matches_the_lidar_late_instead_of_calling_it_stale() -> None:
    """The regime of 2026-09-06 (every scan half a second old by the time its callback ran),
    with the default flags: the node is the old gate. Every revolution the odometry covers is
    matched, 600 ms late, none expires, no "odometry ran late", and the status names the
    lidar as the anchor while its health reads stale."""
    with ros_stubs.parameters(min_match_gap_s=0.0):
        node = Relocalizer()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    node.subs["/map"][1](map_msg())
    truth, odom = drive(12)
    loc = node._localizer
    assert loc is not None
    on_scan, on_odom = node.subs["/scan"][1], node.subs["/odometry/filtered"][1]
    for i, (o, t) in enumerate(zip(odom, truth, strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.6  # the callbacks run 0.6 s after the stamp
        on_odom(odom_msg(o, ts))  # the odometry queue drained before the scan's turn
        on_scan(lidar_msg(t, ts))
        if i == 0:
            assert until(lambda: node._tracker_initialised)
    assert len(node.pubs["/tracker_pose"].sent) == 11, "every revolution after the first"
    metres, degrees = error(loc, truth[-1])
    assert metres < 0.05 and degrees < 2.0
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "released 12, replaced 0, expired 0" in line and "scan age at match 600 ms" in line
    assert "sources: no fresh source, last release lidar 0.6 s ago; lidar stale 0.6 s" in line
    assert "watch fit" in line
    assert not any("odometry ran late" in text for text in node.logger.texts("warning"))


def test_a_fan_drives_without_a_first_search_and_the_watch_waits_for_a_full_turn() -> None:
    """Camera-only (sources=depth): the fan's first release starts the tracker from its saved
    pose with no whole-map search, every frame is matched, the fit is published on the fan,
    the watch is off and the report line says so, /relocalize refuses. The lidar switched on
    and heard from turns the watch on."""
    with ros_stubs.parameters(sources="depth", min_match_gap_s=0.0):
        node = Relocalizer()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    node.subs["/map"][1](map_msg())
    truth, odom = drive(8)
    on_scan, on_depth = node.subs["/scan"][1], node.subs["/depth_scan"][1]
    on_odom = node.subs["/odometry/filtered"][1]
    for i, (o, t) in enumerate(zip(odom, truth, strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.02
        on_depth(depth_msg(t, ts))
        on_odom(odom_msg(o, ts))
        assert node._tracker_initialised and not node._tracker_initialising
    started = node.logger.texts("warning")[0]
    assert "on the depth fan without a first search" in started
    assert not any("first search proposes" in text for text in node.logger.texts("info"))
    assert len(node.pubs["/tracker_pose"].sent) == 8, "the first frame too: no search to wait for"
    node._check()
    assert node._watch_on is False and len(node.pubs["localization_fit"].sent) == 1
    node._report_tracking()
    assert "watch off: no full-turn source, fit" in node.logger.texts("info")[-1]
    res = node.services["relocalize"][1](None, ros_stubs.Trigger.Response())
    assert not res.success and res.message.startswith("no full-turn scan to search with")
    assert not node._searching
    assert node.set_parameters([Parameter("sources", value="lidar,depth")])[0].successful
    ts = 100.0 + 0.1 * len(truth)
    node.clock.seconds = ts + 0.02
    on_scan(lidar_msg(truth[-1], ts))
    on_odom(odom_msg(odom[-1], ts))
    node._check()
    assert node._watch_on is True
    node._report_tracking()
    assert "watch fit" in node.logger.texts("info")[-1]


def test_with_nothing_fresh_the_node_holds_and_says_so(node: Relocalizer) -> None:
    node.clock.seconds = 50.0
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "sources: holding map->odom: no fresh source; lidar absent, depth absent" in line
    res = node.services["relocalize"][1](None, ros_stubs.Trigger.Response())
    assert not res.success and res.message == "no map or no scan yet"


# ---- the laptop's watchdog ------------------------------------------------------------------
CARRIED_TO = Pose2D(1.2, -0.8, math.radians(60.0))  # where the whole-map search says the cart is
SURE = np.diag([0.02**2, 0.02**2, math.radians(1.0) ** 2]).tolist()


_SCANS = itertools.count(1)  # a candidate a second, each off its own revolution


def candidate_msg(
    node: Relocalizer,
    pose: Pose2D,
    score: float = 0.70,
    stamp: float = 100.0,
    scan: int | None = None,
) -> Any:
    """What pepin_bringup.global_watch publishes: one JSON message on /localization/candidate,
    off a fresh revolution unless ``scan`` names one."""
    return String(
        data=json.dumps(
            {
                "x": pose.x,
                "y": pose.y,
                "yaw": pose.theta,
                "covariance": SURE,
                "score": score,
                "ambiguity": 0.2,
                "stamp": stamp,
                "scan": next(_SCANS) if scan is None else scan,
                "map": node._map_id,
                "verdict": "disagree",
                "search_ms": 140.0,
            }
        )
    )


def standing(node: Relocalizer) -> None:
    """One odometry sample, so the node has a tracked pose to judge a candidate against."""
    node.clock.seconds = 100.0
    node.subs["/odometry/filtered"][1](odom_msg(Pose2D(), 100.0))


def test_three_candidates_that_disagree_re_seed_the_tracker(node: Relocalizer) -> None:
    """The kidnap the board cannot see: the tracker holds the old pose, the laptop's search
    says another place three times in a row, and the tracker adopts it through the very door
    its own search uses — /initialpose goes out, map -> odom follows at once."""
    standing(node)
    on_candidate = node.subs["/localization/candidate"][1]
    loc = node._localizer
    assert loc is not None
    for _ in range(2):
        on_candidate(candidate_msg(node, CARRIED_TO))
        assert node._pending_seed is None
    on_candidate(candidate_msg(node, CARRIED_TO))
    assert node._pending_seed is not None
    node._apply_pending_seed()
    assert math.hypot(loc.pose.x - CARRIED_TO.x, loc.pose.y - CARRIED_TO.y) < 0.1
    assert node.pubs["/initialpose"].sent, "AMCL is told, as after the board's own search"
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "candidates 3 (disagree 3), re-seeds 1" in line
    assert "accept_candidates=on candidate_streak=3" in line


def rolled(node: Relocalizer, forward_m: float, at: float = 100.25) -> None:
    """One more odometry sample: the cart has rolled ``forward_m`` straight ahead by ``at``."""
    node.clock.seconds = at
    node.subs["/odometry/filtered"][1](odom_msg(Pose2D(forward_m, 0.0, 0.0), at))


def test_a_candidate_is_carried_from_the_moment_of_its_scan_to_now(node: Relocalizer) -> None:
    """The laptop's search costs 0.12-0.25 s and the link a hop on top; a cart at 0.8 m/s has
    covered 0.20 m by the time the answer lands. The answer says where the cart WAS, so it is
    moved over the odometry of those 0.25 s and the tracker is seeded where that place has
    become — never 20 cm back along the drive, which is where the uncarried pose would put it."""
    standing(node)
    on_candidate = node.subs["/localization/candidate"][1]
    for _ in range(2):
        on_candidate(candidate_msg(node, CARRIED_TO, stamp=100.0))
        assert node._pending_seed is None
    rolled(node, 0.20)
    on_candidate(candidate_msg(node, CARRIED_TO, stamp=100.0))
    node._apply_pending_seed()
    loc = node._localizer
    assert loc is not None
    ahead = apply_motion(CARRIED_TO, Pose2D(0.20, 0.0, 0.0))
    assert math.hypot(loc.pose.x - ahead.x, loc.pose.y - ahead.y) < 0.02
    assert math.hypot(loc.pose.x - CARRIED_TO.x, loc.pose.y - CARRIED_TO.y) > 0.15


def test_the_carry_switch_puts_the_uncarried_pose_back(node: Relocalizer) -> None:
    """The A/B without a restart: off, the pose measured a quarter-second ago is installed as
    the pose now, which is what this branch did before the carry."""
    standing(node)
    assert node.set_parameters([Parameter("carry_candidates", value=False)])[0].successful
    on_candidate = node.subs["/localization/candidate"][1]
    for _ in range(2):
        on_candidate(candidate_msg(node, CARRIED_TO, stamp=100.0))
    rolled(node, 0.20)
    on_candidate(candidate_msg(node, CARRIED_TO, stamp=100.0))
    node._apply_pending_seed()
    loc = node._localizer
    assert loc is not None
    assert math.hypot(loc.pose.x - CARRIED_TO.x, loc.pose.y - CARRIED_TO.y) < 0.02
    node._report_tracking()
    assert "carry_candidates=off" in node.logger.texts("info")[-1]


def test_a_candidate_the_odometry_cannot_reach_is_dropped(node: Relocalizer) -> None:
    """A candidate stamped before the history's horizon (the link stalled, the bus dropped out)
    cannot be carried to now: it is counted as stale and the streak it was part of ends."""
    standing(node)
    on_candidate = node.subs["/localization/candidate"][1]
    for _ in range(2):
        on_candidate(candidate_msg(node, CARRIED_TO, stamp=100.0))
    on_candidate(candidate_msg(node, CARRIED_TO, stamp=93.0))  # older than any odometry held
    assert node._pending_seed is None
    node._report_tracking()
    assert "stale 1" in node.logger.texts("info")[-1]
    on_candidate(candidate_msg(node, CARRIED_TO, stamp=100.0))
    assert node._pending_seed is None, "the run starts again after the gap"


def test_one_frozen_scan_cannot_re_seed_the_tracker(node: Relocalizer) -> None:
    """The laptop's /scan stops moving (the bridge wedges) and the same revolution is searched
    again and again: the answers carry one scan id, and one scan is one opinion."""
    standing(node)
    on_candidate = node.subs["/localization/candidate"][1]
    for _ in range(6):
        on_candidate(candidate_msg(node, CARRIED_TO, stamp=100.0, scan=77))
    assert node._pending_seed is None
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "replay 5" in line and "re-seeds 0" in line and "distinct_scans=on" in line


def test_a_candidate_that_agrees_changes_nothing(node: Relocalizer) -> None:
    standing(node)
    node.subs["/localization/candidate"][1](candidate_msg(node, Pose2D(0.1, 0.0, 0.0)))
    assert node._pending_seed is None
    node._report_tracking()
    assert "candidates 1 (agree 1), re-seeds 0" in node.logger.texts("info")[-1]


def test_the_flag_leaves_the_candidates_as_a_report_only(node: Relocalizer) -> None:
    """accept_candidates off: judged, counted, never acted on — the board's own slow search
    stays the only way back, exactly as before this feature."""
    standing(node)
    assert node.set_parameters([Parameter("accept_candidates", value=False)])[0].successful
    for _ in range(4):
        node.subs["/localization/candidate"][1](candidate_msg(node, CARRIED_TO))
    assert node._pending_seed is None
    node._report_tracking()
    assert "candidates 4 (disagree 4), re-seeds 0" in node.logger.texts("info")[-1]
    assert "accept_candidates=off" in node.logger.texts("info")[-1]


def test_a_candidate_on_another_map_and_a_broken_one_are_refused(node: Relocalizer) -> None:
    standing(node)
    on_candidate = node.subs["/localization/candidate"][1]
    stale = json.loads(candidate_msg(node, CARRIED_TO).data)
    stale["map"] = "1x1@0.00,0.00"
    for _ in range(4):
        on_candidate(String(data=json.dumps(stale)))
    on_candidate(String(data="not json at all"))
    assert node._pending_seed is None
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "elsewhere 4" in line and "malformed 1" in line


def test_the_map_that_does_not_fit_reaches_the_operator(node: Relocalizer) -> None:
    """No mode switch: the verdict is said in the report line and rides /localization/sources."""
    standing(node)
    on_candidate = node.subs["/localization/candidate"][1]
    for _ in range(3):
        on_candidate(candidate_msg(node, CARRIED_TO, score=0.2))
    node._report_tracking()
    assert "THE MAP DOES NOT FIT" in node.logger.texts("info")[-1]
    truth, odom = drive(2)
    for i, (o, t) in enumerate(zip(odom, truth, strict=True)):
        ts = 101.0 + 0.1 * i
        node.clock.seconds = ts + 0.02
        node.subs["/scan"][1](lidar_msg(t, ts))
        node.subs["/odometry/filtered"][1](odom_msg(o, ts))
        assert until(lambda: node._tracker_initialised)
    reported = json.loads(node.pubs["/localization/sources"].sent[-1].data)
    assert reported["candidates"] == {
        "verdict": "unknown_map",
        "map_fits": False,
        "reseeds": 0,
        "accept": True,
    }
