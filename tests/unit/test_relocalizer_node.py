"""The tracker node under the ROS stubs: the lidar's revolution into the feed, the camera's
poses arriving already matched from the laptop, the lidar's death handing the updates to those
measurements and a returning lidar taking them back, the flags that pick the sources, and every
source's word on the wire.

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
    PoseWithCovarianceStamped,
    String,
    TransformStamped,
)
from ros_stubs import OccupancyGrid as OccupancyGridMsg  # noqa: E402
from ros_stubs import Time as TimeMsg  # noqa: E402
from synthetic import raycast_room  # noqa: E402
from test_localization import PILLAR, furnished_room_map  # noqa: E402
from test_localizer_sources import drive, error  # noqa: E402

from pepin.odometry import Pose2D  # noqa: E402
from pepin.scanmatch import apply_motion, relative_motion  # noqa: E402
from pepin.sources import CAMERA, DEPTH, GRAPH, LIDAR  # noqa: E402
from pepin.watch import (  # noqa: E402
    DRIVE_FIT,
    DRIVE_SIGMA_M,
    LOST_SIGMA_M,
    SIGMA_TOPIC,
    SOURCE_PATIENCE_S,
    UNKNOWN_SIGMA,
    Sigma,
)

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


def odom_msg(pose: Pose2D, t: float, vx: float = 0.1, wz: float = 0.0) -> Any:
    """One /odometry/filtered sample. The twist is part of the message the EKF publishes and
    the runaway guard reads it, so it is filled here: the drives of this file cover 0.2 m per
    scan — faster than this cart really moves — and a twist of zero beside such a step is
    exactly the signature of a frame that ran away (pepin.odometry.RunawayWatch)."""
    msg = Odometry(header=Header(stamp=stamp(t), frame_id="odom"))
    msg.pose.pose.position.x, msg.pose.pose.position.y = pose.x, pose.y
    msg.pose.pose.orientation.z = math.sin(pose.theta / 2.0)
    msg.pose.pose.orientation.w = math.cos(pose.theta / 2.0)
    msg.twist.twist.linear.x, msg.twist.twist.angular.z = vx, wz
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
    """A tracker with the camera enabled beside the lidar — as measurements from the laptop, the
    only way the camera reaches this node — the laser mounted at the base's origin, the
    furnished room as its map."""
    # No match gap: the pacer spaces matches on the wall clock, and a test feeds a second of
    # scans in milliseconds.
    with ros_stubs.parameters(sources="lidar,camera", min_match_gap_s=0.0):
        node = Relocalizer()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    node.subs["/map"][1](map_msg())
    return node


SURE_CAMERA = np.diag([0.05**2, 0.05**2, math.radians(2.0) ** 2]).tolist()


def measurement_msg(
    node: Relocalizer,
    pose: Pose2D,
    t: float,
    source: str = DEPTH,
    fit: float = 0.6,
    map_id: str | None = None,
) -> Any:
    """What pepin_bringup.laptop_localizer publishes: one JSON message on
    /localization/measurement, a pose the laptop matched out of a camera scan taken at ``t``."""
    return String(
        data=json.dumps(
            {
                "x": pose.x,
                "y": pose.y,
                "yaw": pose.theta,
                "covariance": SURE_CAMERA,
                "source": source,
                "stamp": t,
                "fit": fit,
                "edge": False,
                "map": node._map_id if map_id is None else map_id,
                "belief_age_ms": 40.0,
                "matched_on": "/map",
            }
        )
    )


def graph_msg(node: Relocalizer, pose: Pose2D, t: float, fit: float = 1.0) -> Any:
    """What pepin_bringup.rtabmap_frame publishes on /localization/graph_measurement: RTAB-Map's
    own word about where the cart is on this map, out of a graph that moved at ``t``."""
    return measurement_msg(node, pose, t, source=GRAPH, fit=fit)


def test_the_node_takes_the_camera_as_a_measurement_and_never_as_a_scan(
    node: Relocalizer,
) -> None:
    """The day's architecture, as the node's own wiring: the lidar's scan and the laptop's
    measurements come in, the camera's raw scans do not — matching them here cost this board
    147 ms a revolution and 4.7 Hz (scratch/drive_bisect.py, run 0238)."""
    assert {"/scan", "/odometry/filtered", "/map", "/localization/measurement"} <= set(node.subs)
    assert "/depth_scan" not in node.subs and "/contact_scan" not in node.subs
    assert "/localization/sources" in node.pubs and "/tracker_pose" in node.pubs
    assert FLAGS["sources"] == (LIDAR, GRAPH), "the module's default: the lidar and the graph"
    assert node._registry.enabled == (LIDAR, CAMERA), "the launch override reached the roster"
    assert node._localizer is not None and node._localizer.sources is node._registry
    assert node.set_parameters([Parameter("sources", value="lidar")])[0].successful
    assert node._registry.enabled == (LIDAR,) and node._localizer.sources.enabled == (LIDAR,)
    refused = node.set_parameters([Parameter("sources", value="lidar,sonar")])[0]
    assert not refused.successful and "sonar" in refused.reason
    assert node.set_parameters([Parameter("fusion", value=False)])[0].successful
    assert node._localizer.fusion is False
    assert node.set_parameters([Parameter("sources", value="camera")])[0].successful
    assert node._registry.enabled == (CAMERA,)
    assert (
        "sources=camera measurement_max_age_s=0.5 remote_floor_xy_m=0.08"
        " remote_floor_yaw_deg=5.0 fusion=off" in node._switches.state()
    )


def test_a_dead_lidar_hands_the_node_to_the_camera_s_measurements_and_back(
    node: Relocalizer,
) -> None:
    """The drive of test_localizer_sources through the node's callbacks: the lidar's revolution
    on /scan, the camera's pose — matched on the laptop out of a frame taken 40 ms earlier — on
    /localization/measurement, the odometry that covers both. The lidar goes silent for 1.2 s in
    the middle: the feed waits the roster's half second, then the measurements drive the updates
    by themselves and map -> odom keeps moving; when the lidar is back it anchors again and they
    ride along. Every update's word goes out on /localization/sources."""
    truth, odom = drive(36)
    loc = node._localizer
    assert loc is not None
    sources_pub, pose_pub = node.pubs["/localization/sources"], node.pubs["/tracker_pose"]
    on_scan, on_measure = node.subs["/scan"][1], node.subs["/localization/measurement"][1]
    on_odom = node.subs["/odometry/filtered"][1]
    drivers: list[str | None] = []
    map_odom_at: dict[int, tuple[float, float, float]] = {}
    for i, (o, t) in enumerate(zip(odom, truth, strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.02  # the node's clock at the arrival
        if i:
            on_measure(measurement_msg(node, between(truth[i - 1], t, 0.6), ts - 0.04))
        if not 10 <= i < 22:
            on_scan(lidar_msg(t, ts))
        published = len(sources_pub.sent)
        on_odom(odom_msg(o, ts))
        if i == 0:  # the first release starts the whole-map search that seeds the watch
            assert until(lambda: node._tracker_initialised)
            assert not sources_pub.sent, "the first scan went to the search, not the tracker"
            drivers.append(None)
            continue
        if len(sources_pub.sent) == published:
            drivers.append(None)
            continue
        report = json.loads(sources_pub.sent[-1].data)
        drivers.append(report["fused"])
        map_odom_at[i] = node._last_map_odom
        metres, degrees = error(loc, t)
        assert metres < 0.12 and degrees < 4.0, f"scan {i}: {metres * 100:.1f} cm, {degrees:.1f}"
    assert (
        drivers[1].startswith(LIDAR)
        and all(  # type: ignore[union-attr]
            d is not None and d.startswith(LIDAR) for d in drivers[2:10]
        )
    )
    assert drivers[10:14] == [None] * 4, "the roster's half second: the feed waits for the lidar"
    assert drivers[14:22] == [CAMERA] * 8, "then the camera's measurements drive alone"
    assert all(d == f"{LIDAR}+{CAMERA}" for d in drivers[22:]), "the lidar anchors again"
    assert map_odom_at[21] != map_odom_at[9], "map -> odom moved on the camera's word"
    metres, degrees = error(loc, truth[-1])
    assert metres < 0.05 and degrees < 2.0
    assert len(pose_pub.sent) == len(sources_pub.sent) == 35 - 4
    camera = json.loads(sources_pub.sent[9].data)  # the first update the camera drove
    assert camera["anchor"] is None and camera["fused"] == CAMERA and camera["rejected"] == []
    assert camera["sources"][LIDAR]["health"].startswith("stale")
    assert camera["sources"][CAMERA]["health"].startswith("fresh")
    assert {"fit", "delta", "sigma", "edge"} <= set(camera["sources"][CAMERA])
    assert "fit" not in camera["sources"][LIDAR] and camera["sources"][DEPTH] == {"health": "off"}
    assert camera["measurements"]["used"] == [DEPTH]
    assert camera["measurements"]["age_ms"] == 0.0, "it drove its own update"
    fused = json.loads(sources_pub.sent[-1].data)
    assert fused["anchor"] == LIDAR and fused["fused"] == f"{LIDAR}+{CAMERA}"
    assert fused["measurements"]["age_ms"] > 0.0, "carried from its own scan to the revolution's"
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "sources: anchor lidar; lidar fresh" in line and "camera fresh" in line
    assert "tracker: scans 24, released 24" in line, "one gate again: the lidar's"
    assert "sources lidar,camera, fusion on" in line, "the tracker's settings"
    assert f"measurements 35 (received 35, replaced 4, taken 31), per source: {DEPTH} 31" in line
    # the four replaced are the ones offered while the feed still waited for the lidar:
    # each was overtaken by a newer one before any update could take it
    assert "flags: rest_lock=on" in line and "sources=lidar,camera" in line
    assert (
        "measurement_max_age_s=0.5 remote_floor_xy_m=0.08 remote_floor_yaw_deg=5.0 fusion=on"
        in line
    )


def test_a_measurement_is_carried_from_its_own_scan_to_the_update_that_takes_it(
    node: Relocalizer,
) -> None:
    """A pose measured on the laptop speaks for the moment of the camera frame it was measured
    on, and the update that fuses it happens later: it is moved over the odometry between the
    two before it is weighed, exactly as a riding scan would have been. Uncarried, it would pull
    the pose back along the drive by whatever the cart covered in the meantime."""
    truth, odom = drive(4)
    on_scan, on_measure = node.subs["/scan"][1], node.subs["/localization/measurement"][1]
    on_odom = node.subs["/odometry/filtered"][1]
    for i, (o, t) in enumerate(zip(odom[:3], truth[:3], strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.02
        on_scan(lidar_msg(t, ts))
        on_odom(odom_msg(o, ts))
        if i == 0:
            assert until(lambda: node._tracker_initialised)
    loc = node._localizer
    assert loc is not None
    step = Pose2D(0.20, 0.0, 0.0)  # what the wheels say happened between the two moments
    on_measure(measurement_msg(node, truth[2], 100.2))
    node.clock.seconds = 100.32
    on_scan(lidar_msg(truth[3], 100.3))
    on_odom(odom_msg(apply_motion(odom[2], step), 100.3))
    camera = next(m for m in loc.measurements if m.source == CAMERA)
    carried = apply_motion(truth[2], step)
    assert math.hypot(camera.x - carried.x, camera.y - carried.y) < 0.01
    assert math.hypot(camera.x - truth[2].x, camera.y - truth[2].y) > 0.15, "not the raw pose"
    assert relative_motion(odom[2], apply_motion(odom[2], step)).x > 0.19, "the trail it rode"


def test_a_measurement_older_than_the_gate_or_from_another_map_is_refused(
    node: Relocalizer,
) -> None:
    """The failure of 2026-09-13 in one rule: a camera pose whose moment the update can no
    longer honestly carry it from is dropped, not fused. So is one measured against another map,
    and a message that is not a measurement at all is counted and never obeyed."""
    truth, odom = drive(4)
    on_scan, on_measure = node.subs["/scan"][1], node.subs["/localization/measurement"][1]
    on_odom = node.subs["/odometry/filtered"][1]
    for i, (o, t) in enumerate(zip(odom[:3], truth[:3], strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.02
        on_scan(lidar_msg(t, ts))
        on_odom(odom_msg(o, ts))
        if i == 0:
            assert until(lambda: node._tracker_initialised)
    loc = node._localizer
    assert loc is not None
    on_measure(measurement_msg(node, truth[2], 99.5))  # measured 0.8 s before this update
    on_measure(measurement_msg(node, truth[2], 100.25, map_id="1x1@0.00,0.00"))
    on_measure(String(data="not json at all"))
    node.clock.seconds = 100.32
    on_scan(lidar_msg(truth[3], 100.3))
    on_odom(odom_msg(odom[3], 100.3))
    assert [m.source for m in loc.measurements] == [LIDAR], "the lidar corrected alone"
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "stale 1" in line and "elsewhere 1" in line and "malformed 1" in line
    assert "measurement_max_age_s=0.5" in line


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


def test_the_camera_alone_drives_without_a_first_search_and_the_watch_waits_for_a_turn() -> None:
    """Camera-only (sources=camera): the first measurement starts the tracker from its saved
    pose with no whole-map search — a pose measured off a +-40 degree fan cannot find the cart
    any more than the fan itself could — every measurement drives an update, the fit is the one
    the laptop measured, the watch is off and the report line says so, /relocalize refuses. The
    lidar switched on and heard from turns the watch on."""
    with ros_stubs.parameters(sources="camera", min_match_gap_s=0.0):
        node = Relocalizer()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    node.subs["/map"][1](map_msg())
    truth, odom = drive(8)
    on_scan, on_measure = node.subs["/scan"][1], node.subs["/localization/measurement"][1]
    on_odom = node.subs["/odometry/filtered"][1]
    for i, (o, t) in enumerate(zip(odom, truth, strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.02
        on_odom(odom_msg(o, ts))
        on_measure(measurement_msg(node, t, ts))
        assert node._tracker_initialised and not node._tracker_initialising
    started = node.logger.texts("warning")[0]
    assert "on the camera's measurements without a first search" in started
    assert not any("first search proposes" in text for text in node.logger.texts("info"))
    loc = node._localizer
    assert loc is not None
    metres, degrees = error(loc, truth[-1])
    assert metres < 0.05 and degrees < 2.0, "the camera's word alone carried the pose"
    assert len(node.pubs["/tracker_pose"].sent) == 8, "one update per measurement"
    node._check()
    assert node._watch_on is False and len(node.pubs["localization_fit"].sent) == 1
    # Nothing here scored that pose: what goes out is 0.0, not the laptop's own number — which
    # was measured against the very band depth_fusion paints only while this topic says 0.50.
    assert node.pubs["localization_fit"].sent[-1].data == 0.0
    assert loc.confidence > 0.5, "the tracker keeps the remote fit for itself"
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "watch off: no full-turn source, fit" in line
    assert "(the laptop measured it: published as 0.00)" in line
    assert node.set_parameters([Parameter("local_fit", value=False)])[0].successful
    node._check()
    assert node.pubs["localization_fit"].sent[-1].data == pytest.approx(loc.confidence)
    assert node.set_parameters([Parameter("local_fit", value=True)])[0].successful
    res = node.services["relocalize"][1](None, ros_stubs.Trigger.Response())
    assert not res.success and res.message.startswith("no map or no scan yet")
    assert not node._searching
    assert node.set_parameters([Parameter("sources", value="lidar,camera")])[0].successful
    ts = 100.0 + 0.1 * len(truth)
    node.clock.seconds = ts + 0.02
    on_scan(lidar_msg(truth[-1], ts))
    on_odom(odom_msg(odom[-1], ts))
    node._check()
    assert node._watch_on is True
    node._report_tracking()
    assert "watch fit" in node.logger.texts("info")[-1]


def test_the_graph_alone_drives_the_updates_it_used_to_pile_up_at_its_gate() -> None:
    """sources=graph: RTAB-Map's word is the only thing that says where the cart is, so it drives
    the updates itself, exactly as the camera's word does.

    The hole measured 2026-09-14 16:01 (test C): 28 graph words in a row received, replaced and
    never taken, `max step 0.0 cm`, the tracker's pose frozen while the cart drove, and goto
    cancelling after 15 s of "localization lost". Nothing triggered a fusion step — the lidar was
    not a source and only the camera's gate could drive one.
    """
    with ros_stubs.parameters(sources="graph", local_fit=False, min_match_gap_s=0.0):
        node = Relocalizer()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    node.subs["/map"][1](map_msg())
    truth, odom = drive(8)
    on_graph, on_odom = (
        node.subs["/localization/graph_measurement"][1],
        node.subs["/odometry/filtered"][1],
    )
    for i, (o, t) in enumerate(zip(odom, truth, strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.02
        on_odom(odom_msg(o, ts))
        on_graph(graph_msg(node, t, ts))
        assert node._tracker_initialised and not node._tracker_initialising
    assert "on the graph's measurements without a first search" in node.logger.texts("warning")[0]
    loc = node._localizer
    assert loc is not None
    metres, degrees = error(loc, truth[-1])
    assert metres < 0.05 and degrees < 2.0, "the graph's word alone carried the pose"
    assert len(node.pubs["/tracker_pose"].sent) == 8, "one update per graph word"
    node._check()
    # local_fit off, as it must be for any source whose match was made elsewhere: no scan of this
    # board's scored the pose, and the fit published is the tracker's own confidence in the
    # fusion it last made. The watch stays off — a word is not a full revolution.
    assert node._watch_on is False
    assert node.pubs["localization_fit"].sent[-1].data == pytest.approx(loc.confidence)
    assert loc.confidence > DRIVE_FIT, "a drive may start on it"
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "graph: measurements 8 (received 8, taken 8), per source: graph 8" in line


def test_the_lidar_drives_and_a_graph_word_only_rides_its_revolution() -> None:
    """sources=lidar,graph: the graph's words arrive between revolutions and none of them drives
    an update — the lidar does, and each word rides the next revolution, carried to its moment.
    Eight revolutions and eight words make seven updates (the first revolution is the search),
    not fifteen."""
    with ros_stubs.parameters(sources="lidar,graph", min_match_gap_s=0.0):
        node = Relocalizer()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    node.subs["/map"][1](map_msg())
    truth, odom = drive(8)
    on_scan, on_odom = node.subs["/scan"][1], node.subs["/odometry/filtered"][1]
    on_graph = node.subs["/localization/graph_measurement"][1]
    for i, (o, t) in enumerate(zip(odom, truth, strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.02
        on_odom(odom_msg(o, ts))
        on_scan(lidar_msg(t, ts))
        if i == 0:
            assert until(lambda: node._tracker_initialised)
        on_graph(graph_msg(node, t, ts))  # between two revolutions: it waits for the next
    assert len(node.pubs["/tracker_pose"].sent) == 7, "one update per revolution, none per word"
    loc = node._localizer
    assert loc is not None
    assert {m.source for m in loc.measurements} == {LIDAR, GRAPH}, "the word rode the revolution"
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "graph: measurements 8 (received 8, taken 7)" in line, "the last one still waits"


def test_the_camera_and_the_graph_make_one_update_between_them_never_two_per_word() -> None:
    """sources=camera,graph with no lidar: each word drives one update of its own, and two words
    waiting for the same odometry sample are taken by ONE update — fusing the same odometry step
    twice would count a step the cart never took."""
    with ros_stubs.parameters(sources="camera,graph", local_fit=False, min_match_gap_s=0.0):
        node = Relocalizer()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    node.subs["/map"][1](map_msg())
    truth, odom = drive(6)
    on_measure = node.subs["/localization/measurement"][1]
    on_graph, on_odom = (
        node.subs["/localization/graph_measurement"][1],
        node.subs["/odometry/filtered"][1],
    )
    for i, (o, t) in enumerate(zip(odom, truth, strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.02
        on_odom(odom_msg(o, ts))
        on_measure(measurement_msg(node, t, ts))
        on_graph(graph_msg(node, t, ts))
    assert len(node.pubs["/tracker_pose"].sent) == 12, "one update per word, and no more"
    # Both words stamped ahead of the newest odometry sample: neither can drive until it arrives,
    # and then one update takes both — the driver is the fresher word, the other rides it.
    ahead = 100.0 + 0.1 * len(truth)
    on_measure(measurement_msg(node, truth[-1], ahead + 0.04))
    on_graph(graph_msg(node, truth[-1], ahead + 0.05))
    assert len(node.pubs["/tracker_pose"].sent) == 12, "nothing the odometry does not reach yet"
    node.clock.seconds = ahead + 0.12
    on_odom(odom_msg(odom[-1], ahead + 0.1))
    assert len(node.pubs["/tracker_pose"].sent) == 13, "one update, not one per gate"
    report = json.loads(node.pubs["/localization/sources"].sent[-1].data)
    assert report["graph"]["used"] == [GRAPH] and report["measurements"]["used"] == [DEPTH]


def test_with_nothing_fresh_the_node_holds_and_says_so(node: Relocalizer) -> None:
    node.clock.seconds = 50.0
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "sources: holding map->odom: no fresh source; lidar absent, depth off" in line
    assert "contact off, camera absent" in line
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
    source: str = LIDAR,
) -> Any:
    """What pepin_bringup.laptop_localizer publishes: one JSON message on /localization/candidate,
    off a fresh revolution unless ``scan`` names one, found on ``source``."""
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
                "source": source,
                "verdict": "disagree",
                "search_ms": 140.0,
            }
        )
    )


def standing(node: Relocalizer) -> None:
    """One odometry sample, so the node has a tracked pose to judge a candidate against."""
    node.clock.seconds = 100.0
    node.subs["/odometry/filtered"][1](odom_msg(Pose2D(), 100.0, vx=0.0))


def test_an_operator_seed_is_adopted_once_and_never_echoed(node: Relocalizer) -> None:
    """A pose published on /initialpose (Foxglove's pose estimate) is adopted at once, and it
    stays one seed: the node has no publisher on that topic, so its adoption cannot come back
    to it as a new operator seed (the 20 Hz loop of 2026-09-13)."""
    from pepin_bringup.msgs import pose_with_covariance

    standing(node)
    loc = node._localizer
    assert loc is not None
    node.subs["/initialpose"][1](
        pose_with_covariance(
            CARRIED_TO.x, CARRIED_TO.y, CARRIED_TO.theta, 0.05, math.radians(5.0), TimeMsg(), "map"
        )
    )
    assert math.hypot(loc.pose.x - CARRIED_TO.x, loc.pose.y - CARRIED_TO.y) < 0.01
    assert node.fit == 1.0, "the operator's word is taken at full confidence"
    assert "/initialpose" not in node.pubs
    assert node.logger.texts("info")[-1].startswith(
        "seeded by the operator at (1.20, -0.80, 60 deg)"
    )


def test_three_candidates_that_disagree_re_seed_the_tracker(node: Relocalizer) -> None:
    """The kidnap the board cannot see: the tracker holds the old pose, the laptop's search
    says another place three times in a row, and the tracker adopts it through the very door
    its own search uses — map -> odom follows at once, and nothing goes out on /initialpose."""
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
    assert "/initialpose" not in node.pubs, (
        "the node listens on /initialpose: a publication there came back as an operator seed,"
        " 20 times a second (2026-09-13)"
    )
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "candidates 3 (disagree 3) from lidar 3, re-seeds 1" in line
    assert "accept_candidates=on candidate_streak=3" in line


def test_a_fan_may_not_re_seed_a_tracker_the_lidar_is_still_feeding(node: Relocalizer) -> None:
    """The last line under the camera's own whole-map search: while this board's roster says the
    lidar is fresh, a candidate found on a FAN is judged, counted and reported — and cannot move
    the pose, however long its streak. The laptop has the same rule, but it reads this board's
    health off a topic that can go quiet for reasons that have nothing to do with the lidar, and
    a fan's fit saturates at 1.00 where a revolution's honest fit is 0.67. Once the lidar really
    has gone stale, the next fan closes the streak that was waiting."""
    standing(node)
    node._registry.observe(LIDAR, 100.0)  # the board's own word: the lidar is feeding it
    on_candidate = node.subs["/localization/candidate"][1]
    for _ in range(4):
        on_candidate(candidate_msg(node, CARRIED_TO, score=1.0, source=DEPTH))
        assert node._pending_seed is None, "a fan moved a tracker the lidar was still feeding"
    node._report_tracking()
    assert "from depth 4" in node.logger.texts("info")[-1]
    node.clock.seconds = 120.0  # ...and now nothing of the lidar's is fresh any more
    on_candidate(candidate_msg(node, CARRIED_TO, score=1.0, source=DEPTH))
    assert node._pending_seed is not None, "with the lidar gone the fan is the only word there is"


def test_the_graph_may_re_seed_a_driving_cart_when_it_is_the_only_localizer(
    node: Relocalizer,
) -> None:
    """No re-seed while a goal runs, because a teleport mid-drive is worse than a poor fit — but
    that assumes something else will correct the pose, and with the lidar off the roster nothing
    will: on 2026-09-14 21:12 the cart drove 64 s on a belief 2 m wrong while the graph
    recognised the place the whole way, every candidate refused for the single reason that a
    goal was running. With the lidar alive the rule is unchanged: the graph waits."""
    standing(node)
    node._navigating = True
    on_candidate = node.subs["/localization/candidate"][1]

    node._registry.observe(LIDAR, 100.0)  # the lidar is feeding this tracker
    for _ in range(4):
        on_candidate(candidate_msg(node, CARRIED_TO, source=GRAPH))
    assert node._pending_seed is None, "the lidar is there to correct the drive; the graph waits"

    node.clock.seconds = 120.0  # ...and now nothing of the lidar's is fresh any more
    for _ in range(3):
        on_candidate(candidate_msg(node, CARRIED_TO, source=GRAPH))
    assert node._pending_seed is not None, "the graph is the only thing that knows the place"
    node._report_tracking()
    assert "graph_reseed_while_driving=on" in node.logger.texts("info")[-1]


def test_the_graph_s_re_seed_while_driving_can_be_switched_off(node: Relocalizer) -> None:
    """CLAUDE.md rule 19: the old rule stays reachable — nothing re-seeds a cart that is driving,
    which is what a fresh database or an anchor learned off a soft seating calls for."""
    standing(node)
    node._navigating = True
    assert node.set_parameters([Parameter("graph_reseed_while_driving", value=False)])[0].successful
    node.clock.seconds = 120.0  # no lidar on the roster at all
    for _ in range(4):
        node.subs["/localization/candidate"][1](candidate_msg(node, CARRIED_TO, source=GRAPH))
    assert node._pending_seed is None
    node._report_tracking()
    assert "graph_reseed_while_driving=off" in node.logger.texts("info")[-1]


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
    assert "candidates 1 (agree 1) from lidar 1, re-seeds 0" in node.logger.texts("info")[-1]


def test_the_flag_leaves_the_candidates_as_a_report_only(node: Relocalizer) -> None:
    """accept_candidates off: judged, counted, never acted on — the board's own slow search
    stays the only way back, exactly as before this feature."""
    standing(node)
    assert node.set_parameters([Parameter("accept_candidates", value=False)])[0].successful
    for _ in range(4):
        node.subs["/localization/candidate"][1](candidate_msg(node, CARRIED_TO))
    assert node._pending_seed is None
    node._report_tracking()
    assert "candidates 4 (disagree 4) from lidar 4, re-seeds 0" in node.logger.texts("info")[-1]
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
        "source": "lidar",
        "map_fits": False,
        "reseeds": 0,
        "accept": True,
    }


# -- the pace of a cart that stands still -------------------------------------


def seed_msg(pose: Pose2D) -> Any:
    """A pose on /initialpose: Foxglove's "set pose", and this node's own announcement."""
    msg = PoseWithCovarianceStamped()
    msg.pose.pose.position.x, msg.pose.pose.position.y = pose.x, pose.y
    msg.pose.pose.orientation.z = math.sin(pose.theta / 2.0)
    msg.pose.pose.orientation.w = math.cos(pose.theta / 2.0)
    return msg


def echo_initialpose(node: Relocalizer, rounds: int = 20) -> int:
    """Hand every /initialpose message the node published back to its own subscription — ROS 2
    delivers a publication to the publishing node's own subscriptions — until nothing new goes
    out; returns the rounds that took (``rounds`` means it never stopped)."""
    if "/initialpose" not in node.pubs or "/initialpose" not in node.subs:
        return 0
    pub, on_seed = node.pubs["/initialpose"], node.subs["/initialpose"][1]
    for turn in range(rounds):
        pending, pub.sent = pub.sent, []
        if not pending:
            return turn
        for msg in pending:
            on_seed(msg)
    return rounds


def stand(node: Relocalizer, t0: float, seconds: float) -> int:
    """A cart that does not move, ten revolutions a second for ``seconds``, every /initialpose
    the node publishes handed straight back to it; returns the scans that were matched."""
    poses, still = node.pubs["/tracker_pose"], Pose2D()
    before = len(poses.sent)
    for i in range(round(seconds * 10)):
        ts = t0 + 0.1 * i
        node.clock.seconds = ts + 0.02
        node.subs["/scan"][1](lidar_msg(still, ts))
        node.subs["/odometry/filtered"][1](odom_msg(still, ts))
        echo_initialpose(node)
        assert until(lambda: node._tracker_initialised)
    return len(poses.sent) - before


def test_a_standing_cart_is_matched_about_once_a_second_even_after_a_seed() -> None:
    """The regression of 2026-09-13 18:35: a standing cart was matched four to five times a
    second (``rested 0, matched 115`` against the morning's ``rested 266, matched 30``), at 90 %
    of an A53 core, with the published fit down from 0.77 to 0.51.

    ``rested`` counts the scans ``timeline.MotionFilter`` refuses — nothing moved and the last
    match is younger than a second — and every seed resets that filter, because a re-seed means
    the cart is somewhere else now. The node publishes its every seed on /initialpose so AMCL
    follows it AND listens there for the operator's, and a publication reaches the publishing
    node's own subscriptions: one seed by hand became a seed at the executor's speed. The board
    log said "seeded by the operator" 4882 times in four minutes.

    So: a still cart is matched about once a second, before a seed and after one, with whatever
    the node puts on /initialpose handed straight back to it.
    """
    with ros_stubs.parameters(min_match_gap_s=0.0):  # the pacer spaces on the wall clock
        node = Relocalizer()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    node.subs["/map"][1](map_msg())
    assert stand(node, 100.0, 4.0) <= 5, "one match a second, plus the first"
    assert node._rested >= 30, "every other scan rested: the last match still holds"

    node.clock.seconds = 104.0
    seeded = Pose2D(0.03, -0.02, math.radians(1.0))  # the operator, a few cm off the truth
    node.subs["/initialpose"][1](seed_msg(seeded))
    loc = node._localizer
    assert loc is not None
    assert math.hypot(loc.pose.x - seeded.x, loc.pose.y - seeded.y) < 1e-6, "adopted as the truth"
    assert echo_initialpose(node) <= 1, "the node answered its own seed, and kept answering"

    node._rested = 0
    assert stand(node, 104.1, 4.0) <= 6, "the seed is taken once, not once per scan"
    assert node._rested >= 30


# ---- which map the tracker matches on --------------------------------------------------------
def volume_map_msg() -> Any:
    """The lidar layer of the laptop's fused volume as depth_fusion publishes it (/map_lidar):
    the same room on a larger, differently placed grid — which is exactly why it carries another
    map id, and why the laptop's candidates about /map are no evidence about this one."""
    msg = map_msg()
    msg.info.width += 2
    msg.info.origin.position.x -= 0.10
    msg.data = (list(msg.data) + [-1] * (2 * msg.info.height))[: msg.info.width * msg.info.height]
    return msg


def test_the_tracker_matches_on_the_served_map_until_the_flag_moves_it(node: Relocalizer) -> None:
    """Both maps are subscribed always; /map_lidar arriving changes nothing while the flag says
    map, and moving the flag adopts the map already in hand — a served map is published once and
    latched, so waiting for the next publication would be waiting for ever."""
    served = node._map_id
    node.subs["/map_lidar"][1](volume_map_msg())
    assert node._map_id == served and node._choice.source == "map"

    assert node.set_parameters([Parameter("map_topic", value="map_lidar")])[0].successful
    assert node._choice.source == "map_lidar" and node._map_id != served
    assert node._grid.spec.width_m > 0.0 and node._matcher is not None, "rebuilt on the volume"

    assert node.set_parameters([Parameter("map_topic", value="map")])[0].successful
    assert node._map_id == served, "and back, without a restart"


def test_a_map_switch_keeps_the_cart_where_it_is(node: Relocalizer) -> None:
    """The incident of 2026-09-14 18:13, in a test.

    A live ``map_topic=map_lidar`` with the cart at home restarted the tracker at the ORIGIN —
    the saved-pose file is keyed by map id, and the volume's grid has an id of its own — and the
    next measurement-driven update published map -> odom for that origin pose. Nav2 logged
    "global_costmap: Sensor origin at (0.01, -0.00) is out of map bounds" 110 times, the local
    costmap stopped following the cart, and no goal succeeded until the stack was restarted.

    Both maps are the same room on grids aligned to the same file, so the pose survives the
    switch and so does the transform the node broadcasts.
    """
    stand(node, 100.0, 2.0)
    node.clock.seconds = 103.0
    here = Pose2D(0.30, -0.20, math.radians(20.0))
    node.subs["/initialpose"][1](seed_msg(here))
    echo_initialpose(node)
    stand(node, 103.1, 1.0)
    loc = node._localizer
    assert loc is not None and math.hypot(loc.pose.x, loc.pose.y) > 0.1, "not at the origin"
    before, frame = loc.pose, node._last_map_odom

    node.subs["/map_lidar"][1](volume_map_msg())
    assert node.set_parameters([Parameter("map_topic", value="map_lidar")])[0].successful
    after = node._localizer
    assert after is not None and after is not loc, "the tracker was rebuilt on the volume"
    assert (after.pose.x, after.pose.y, after.pose.theta) == (before.x, before.y, before.theta)
    assert node._tracker_initialised, "it knows where it is: no whole-map search, no origin pose"
    assert node._last_map_odom == frame, "and it broadcasts the very frame it broadcast before"


def test_a_second_map_on_the_owners_own_topic_leaves_the_cart_where_it_is(
    node: Relocalizer,
) -> None:
    """The volume takes /map over from the map server (ONE MAP, 2026-09-16): the topic's OWNER
    changes, its name does not, and the same room now arrives there more than once instead of
    being latched once and never repeated. Two gates keep the cart where it is — the choice
    refuses a republication on the topic already in use (``map_refresh_s`` 0 is "the first map
    and no other"), and where a changed one IS adopted the pose is carried across it, because
    the same frame and origin is the same room and the cart did not move because the picture of
    it did.
    """
    stand(node, 100.0, 2.0)
    node.clock.seconds = 103.0
    here = Pose2D(0.30, -0.20, math.radians(20.0))
    node.subs["/initialpose"][1](seed_msg(here))
    echo_initialpose(node)
    stand(node, 103.1, 1.0)
    loc = node._localizer
    assert loc is not None and math.hypot(loc.pose.x, loc.pose.y) > 0.1, "not at the origin"
    before, frame, served = loc.pose, node._last_map_odom, node._map_id

    node.subs["/map"][1](map_msg())  # the volume's next publication: the very same cells
    assert node._localizer is loc, "nothing adopted: the same tracker, the same matcher"
    assert node._map_id == served

    node.clock.seconds = 150.0
    assert node.set_parameters([Parameter("map_refresh_s", value=30.0)])[0].successful
    painted = map_msg()  # ...and one the fusion has painted a cell into
    painted.data[0] = 100 if painted.data[0] != 100 else 0
    node.subs["/map"][1](painted)
    after = node._localizer
    assert after is not None and after is not loc, "a changed map, old enough: adopted"
    assert node._map_id == served, "same width, height and origin: the same map id"
    assert (after.pose.x, after.pose.y, after.pose.theta) == (before.x, before.y, before.theta)
    assert node._tracker_initialised, "no whole-map search, no origin pose"
    assert node._last_map_odom == frame, "and the frame it broadcasts is the one it broadcast"


def test_the_switch_can_be_told_to_find_the_cart_again(node: Relocalizer) -> None:
    """The old behaviour stays reachable: with ``carry_pose_across_maps`` off the tracker looks
    for itself on the new map before it trusts anything (a map of another place on the same
    topic)."""
    stand(node, 100.0, 2.0)
    assert node.set_parameters([Parameter("carry_pose_across_maps", value=False)])[0].successful
    node.subs["/map_lidar"][1](volume_map_msg())
    node.set_parameters([Parameter("map_topic", value="map_lidar")])
    assert not node._tracker_initialised


def test_the_tracker_takes_the_served_map_when_the_volume_never_speaks() -> None:
    """A board that starts with the laptop down: /map_lidar has no publisher at all and the
    tracker must not sit blind waiting for it (CLAUDE.md rule 20). The served map is latched and
    already in hand; ten seconds later it is the map in use, and the volume still replaces it
    whenever it turns up."""
    with ros_stubs.parameters(map_topic="map_lidar", min_match_gap_s=0.0):
        node = Relocalizer()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    node.subs["/map"][1](map_msg())
    assert node._matcher is None, "nothing adopted: the tracker was asked for the volume"

    node.clock.seconds = 1.0
    node._check()
    node.clock.seconds = 5.0
    node._check()
    assert node._matcher is None, "five seconds is not ten"

    node.clock.seconds = 12.0
    node._check()
    assert node._choice.source == "map" and node._choice.fell_back
    assert node._matcher is not None, "matching on the served file"

    node.subs["/map_lidar"][1](volume_map_msg())
    assert node._choice.source == "map_lidar" and not node._choice.fell_back


def test_a_republished_volume_map_does_not_rebuild_the_tracker(node: Relocalizer) -> None:
    """/map_lidar arrives at the fusion's map_hz, once a second, and every adoption rebuilds the
    matcher and forgets the episode's evidence. The refresh gate is what makes it safe to point
    the tracker at a map that is still being built."""
    node.set_parameters([Parameter("map_topic", value="map_lidar")])
    node.subs["/map_lidar"][1](volume_map_msg())
    first = node._matcher
    for _ in range(3):
        node.subs["/map_lidar"][1](volume_map_msg())
    assert node._matcher is first, "the same matcher: nothing was adopted"

    node.clock.seconds = 100.0
    assert node.set_parameters([Parameter("map_refresh_s", value=30.0)])[0].successful
    changed = volume_map_msg()
    changed.data[0] = 100 if changed.data[0] != 100 else 0
    node.subs["/map_lidar"][1](changed)
    assert node._matcher is not first, "old enough and its cells differ: adopted"


def test_the_published_fit_falls_to_zero_once_every_source_has_gone_silent(
    node: Relocalizer,
) -> None:
    """No source, no confidence. On 2026-09-14 this node published fit 0.70 for 141 s with
    sources=camera and not one measurement arriving, and the goal server — which reads only that
    number — took `printer` and then `home` and drove both on dead reckoning. Silence longer than
    source_patience_s now publishes 0.00, which is under every rung of the ladder at once: a new
    goal is refused (drive_fit 0.50) and the running one is stopped (blind_fit 0.30). The watch
    that searches the whole map hears nothing of it — it reads the tracker's OWN fit — so a dead
    sensor cannot start a re-seed frenzy on a scan that is not there."""
    truth, odom = drive(3)
    on_scan, on_odom = node.subs["/scan"][1], node.subs["/odometry/filtered"][1]
    for i, (o, t) in enumerate(zip(odom, truth, strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.02
        on_scan(lidar_msg(t, ts))
        on_odom(odom_msg(o, ts))
        if i == 0:
            assert until(lambda: node._tracker_initialised)
    fit = node.pubs["localization_fit"]
    node.clock.seconds = 100.3
    node._check()  # the first search's candidate is dropped here: the tracker found its own feet
    node._check()
    measured = fit.sent[-1].data
    assert measured > DRIVE_FIT, "a source just spoke: the fit it earned goes out"
    node.clock.seconds = 100.2 + SOURCE_PATIENCE_S  # still inside the patience
    node._check()
    assert fit.sent[-1].data == pytest.approx(measured)
    node.clock.seconds = 100.2 + SOURCE_PATIENCE_S + 0.5
    node._check()
    assert fit.sent[-1].data == 0.0 < node.fit, "published 0.00, the tracker's own fit untouched"
    assert not node._searching, "silence is not evidence that the pose is wrong"
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "no source for 3.5 s (published as 0.00)" in line
    assert "fit_needs_a_source=on source_patience_s=3.0" in line
    # Off, the old behaviour: the last fit measured stands until a source corrects it again.
    assert node.set_parameters([Parameter("fit_needs_a_source", value=False)])[0].successful
    node._check()
    assert fit.sent[-1].data == pytest.approx(measured)
    node._report_tracking()
    assert "no source for 3.5 s," in node.logger.texts("info")[-1], "still said, never hidden"
    # ...and the patience is live: a link that stutters for seconds keeps its fit.
    assert node.set_parameters([Parameter("fit_needs_a_source", value=True)])[0].successful
    assert node.set_parameters([Parameter("source_patience_s", value=10.0)])[0].successful
    node._check()
    assert fit.sent[-1].data == pytest.approx(measured)


def test_a_cart_standing_still_is_not_a_cart_without_a_source(node: Relocalizer) -> None:
    """The trap the patience must not fall into: a still cart is matched about once a second
    (the motion filter spares the matcher) while its lidar keeps turning at 10 Hz. Silence is
    measured on what the SOURCES deliver, never on when the pose last moved, so ten seconds of
    standing still cost the published fit nothing."""
    truth, odom = drive(3)
    on_scan, on_odom = node.subs["/scan"][1], node.subs["/odometry/filtered"][1]
    for i, (o, t) in enumerate(zip(odom, truth, strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.02
        on_scan(lidar_msg(t, ts))
        on_odom(odom_msg(o, ts))
        if i == 0:
            assert until(lambda: node._tracker_initialised)
    node.clock.seconds = 100.3
    node._check()  # the first search's candidate is dropped: nothing is capped after this
    standing, rested = truth[-1], odom[-1]
    for i in range(100):  # ten seconds of revolutions from the same place, no motion at all
        ts = 100.3 + 0.1 * i
        node.clock.seconds = ts + 0.02
        on_scan(lidar_msg(standing, ts))
        on_odom(odom_msg(rested, ts))
    node._check()
    assert node.pubs["localization_fit"].sent[-1].data > DRIVE_FIT
    assert node._rested > 50, "the matcher was spared while the lidar went on speaking"


def test_the_silence_is_measured_before_the_tracker_is_ready() -> None:
    """The silence is a fact about the SENSORS, not about this node's readiness. While the map,
    the matcher or the first pose are still missing the fit check returns early and publishes
    nothing — and the number the report line prints beside the fit must already be the lidar's
    real age, not the `no source ever` it starts at, or every boot reads as a blind cart until
    the whole-map search lands (seconds on the board, minutes when it has to retry)."""
    with ros_stubs.parameters(sources="lidar,camera", min_match_gap_s=0.0):
        node = Relocalizer()  # no /map: _check cannot get past its early return
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    assert node._source_age_s == math.inf, "nothing has spoken yet"
    node.clock.seconds = 100.05
    node.subs["/scan"][1](lidar_msg(Pose2D(), 100.0))
    node._check()
    assert not node.pubs["localization_fit"].sent, "no map, no matcher: nothing is published"
    assert node._source_age_s == pytest.approx(0.05), "the lidar spoke 50 ms ago and is heard"
    assert node._silence.phrase(node._source_age_s) == "last source 0.1 s ago"
    assert not node._silence.held_at_zero(node._source_age_s)


def runaway(node: Relocalizer, at: float = 100.05) -> None:
    """The EKF frame 3.5 km away, one sample after the cart stood at the origin."""
    node.clock.seconds = at
    node.subs["/odometry/filtered"][1](odom_msg(Pose2D(3493.7, -395.6, 0.0), at, vx=0.01))


def test_an_odometry_sample_that_ran_away_never_reaches_the_carry(node: Relocalizer) -> None:
    """2026-09-14: the EKF flew to 43 km at 60 m/s and every consumer followed. The guard
    refuses the step, counts it, and says it once for the episode."""
    standing(node)
    runaway(node)
    assert node._runaways == 1
    assert node._history.at(100.05) is None, "the history stops at the last sample that made sense"
    runaway(node, 100.10)
    assert node._runaways == 2 and node._runaway.streak == 2, "one episode: one line, two counts"
    said = [line for line in node.get_logger().texts("error") if "ran away" in line]
    assert len(said) == 1 and "the pose stays where it was" in said[0]


def test_the_guard_off_carries_the_runaway_as_before(node: Relocalizer) -> None:
    node._switches.set("odometry_guard", False)
    standing(node)
    runaway(node)
    assert node._runaways == 0
    assert node._history.at(100.05) is not None, "the old behaviour, reachable"


def test_a_normal_drive_is_never_refused_and_the_report_line_carries_the_count(
    node: Relocalizer,
) -> None:
    standing(node)
    for i, t in enumerate((100.05, 100.10, 100.15), start=1):
        node.clock.seconds = t
        node.subs["/odometry/filtered"][1](odom_msg(Pose2D(0.015 * i, 0.0, 0.0), t, vx=0.3))
    assert node._runaways == 0
    node._report_tracking()
    line = next(line for line in node.get_logger().texts("info") if line.startswith("tracker:"))
    assert "odometry runaway 0" in line and "odometry_guard=on" in line


def said(pub: Any) -> Sigma:
    """The newest /localization/sigma message, read the way every consumer reads it."""
    heard = Sigma.from_json(pub.sent[-1].data, 0.0)
    assert heard is not None, pub.sent[-1].data
    return heard


def test_the_node_publishes_one_sigma_out_of_the_fusion_whichever_source_spoke() -> None:
    """The failure of 2026-09-15, and its fix, in one drive: with sources=camera nothing here
    scores a scan against the map, so /localization_fit goes out as 0.00 and every rule built on
    it read the cart as lost — while the fusion was holding the pose to a few centimetres. That
    fusion's own covariance is what /localization/sigma carries, so the camera's word collapses
    it exactly as the lidar's match would; with nothing correcting it, it grows along the
    odometry until every gate downstream refuses.
    """
    with ros_stubs.parameters(sources="camera", min_match_gap_s=0.0):
        node = Relocalizer()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    sigma_pub = node.pubs[SIGMA_TOPIC]
    node._check()  # before a map, a scan or a word: not localised, and the topic says so
    first = said(sigma_pub)
    assert first.xy_m == pytest.approx(UNKNOWN_SIGMA[0], abs=1e-3)
    assert first.yaw_deg == pytest.approx(UNKNOWN_SIGMA[1], abs=1e-3)
    assert first.xy_m > LOST_SIGMA_M, "no word yet may not start a drive"
    node.subs["/map"][1](map_msg())
    truth, odom = drive(8)
    on_measure = node.subs["/localization/measurement"][1]
    on_odom = node.subs["/odometry/filtered"][1]
    for i, (o, t) in enumerate(zip(odom, truth, strict=True)):
        ts = 100.0 + 0.1 * i
        node.clock.seconds = ts + 0.02
        on_odom(odom_msg(o, ts))
        on_measure(measurement_msg(node, t, ts))
    node._check()
    assert node.pubs["localization_fit"].sent[-1].data == 0.0, "the lidar's metric, and no lidar"
    word = said(sigma_pub)
    assert word.xy_m < DRIVE_SIGMA_M, f"the camera is holding the pose: {word.xy_m:.3f} m"
    assert json.loads(sigma_pub.sent[-1].data)["word_age_s"] < 0.5, "the word that did it"
    assert len(sigma_pub.sent) >= len(node.pubs["/tracker_pose"].sent), "one per update, at least"
    # ...and now nothing corrects it: the wheels turn, the camera says nothing, the check runs.
    collapsed = word.xy_m
    for i in range(1, 40):
        ts = 101.0 + 0.5 * i
        node.clock.seconds = ts
        on_odom(odom_msg(Pose2D(odom[-1].x + 0.25 * i, odom[-1].y, odom[-1].theta), ts))
        node._check()
    grown = said(sigma_pub)
    assert grown.xy_m > collapsed * 3, f"ten metres of dead reckoning: {grown.xy_m:.3f} m"
    assert grown.xy_m > LOST_SIGMA_M, "...and a drive on it is cut"
    told = json.loads(sigma_pub.sent[-1].data)
    assert told["word_age_s"] > 15.0, "the seconds since the last accepted word ride along"
    assert told["stamp"] == pytest.approx(node.clock.seconds), "...and the clock it was said at"
    node._report_tracking()
    line = node.logger.texts("info")[-1]
    assert "sigma " in line and "word " in line
    assert f"a drive is cut over {LOST_SIGMA_M:.2f} m" in line
    where = node.services["where_am_i"][1](None, ros_stubs.Trigger.Response())
    assert "the number a drive is judged by" in where.message


def test_a_seed_collapses_the_sigma_because_a_hand_is_a_word_too(node: Relocalizer) -> None:
    """An operator who can see the cart is the surest source there is: after a seed the pose is
    known to what that confidence buys, and the growth starts again from there. Without this a
    hand seed could not clear a refusal — the sigma went on growing from a pose nobody holds."""
    node.clock.seconds = 100.0
    node.subs["/odometry/filtered"][1](odom_msg(Pose2D(0.0, 0.0, 0.0), 100.0))
    node._check()
    assert said(node.pubs[SIGMA_TOPIC]).xy_m > LOST_SIGMA_M
    seed = PoseWithCovarianceStamped()
    seed.pose.pose.position.x, seed.pose.pose.position.y = -9.5, 2.4
    node.subs["/initialpose"][1](seed)
    assert said(node.pubs[SIGMA_TOPIC]).xy_m < DRIVE_SIGMA_M
    assert json.loads(node.pubs[SIGMA_TOPIC].sent[-1].data)["word_age_s"] == pytest.approx(0.0)
