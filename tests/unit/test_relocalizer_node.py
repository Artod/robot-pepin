"""The tracker node under the ROS stubs: three scan topics into one feed, the lidar's death
handing the updates to the camera on the node's own trigger path and a returning lidar
taking them back, the flags that pick the sources, and every source's word on the wire.

rclpy is faked (``ros_stubs``); the map, the scans and the odometry are the furnished room's
of test_localization, so the node is built and driven here exactly as on the board, scan by
scan, with no robot.
"""

from __future__ import annotations

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
    TransformStamped,
)
from ros_stubs import OccupancyGrid as OccupancyGridMsg  # noqa: E402
from ros_stubs import Time as TimeMsg  # noqa: E402
from synthetic import raycast_room  # noqa: E402
from test_localization import PILLAR, furnished_room_map  # noqa: E402
from test_localizer_sources import drive, error  # noqa: E402

from pepin.odometry import Pose2D  # noqa: E402
from pepin.sources import CONTACT, DEPTH, LIDAR  # noqa: E402

BEAMS = 180
FAN = [*range(160, 180), *range(0, 21)]  # the raycast's beams within +-40 degrees, in order


def stamp(t: float) -> Any:
    whole = math.floor(t)
    return TimeMsg(sec=whole, nanosec=round((t - whole) * 1e9))


def map_msg() -> Any:
    """The furnished room as the map server publishes it."""
    grid = furnished_room_map()
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
