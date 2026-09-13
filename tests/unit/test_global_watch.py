"""The laptop's watchdog node: a map, a scan, and a candidate on the wire once a second.

rclpy is faked (``ros_stubs``), the map and the scans are the furnished room of
test_localization, so the node is built and driven here exactly as on the laptop — the whole
search included, which is what makes this the one test that says what a candidate really
carries.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import ros_stubs

RCLPY = ros_stubs.install()

from pepin_bringup.global_watch import FLAGS, GlobalWatch  # noqa: E402
from ros_stubs import (  # noqa: E402
    Float32,
    Header,
    LaserScan,
    Parameter,
    TransformStamped,
)
from ros_stubs import PoseWithCovarianceStamped as PoseMsg  # noqa: E402
from synthetic import raycast_room  # noqa: E402
from test_localization import PILLAR, furnished_room_map, room_map  # noqa: E402
from test_relocalizer_node import map_msg, stamp  # noqa: E402

from pepin.odometry import Pose2D  # noqa: E402
from pepin.watchdog import CandidateVerdict, GlobalCandidate  # noqa: E402

# The fixture room is a 6 x 4 m rectangle with one small box in a corner, so it is very nearly
# a twin of itself turned by 180 degrees: from the middle of it the runner-up explains a scan to
# within 0.91 of the winner and the watch rightly refuses to say where the cart is (the last
# test). Beside the box it does not — the rival ranks well under the winner — and that is where the
# cart stands for the rest (0.62 at this heading).
TRUTH = Pose2D(-1.6, 1.1, math.radians(30.0))
ELSEWHERE = Pose2D(1.5, -1.2, math.radians(-120.0))
BEAMS = 180


def scan_msg(truth: Pose2D, t: float = 100.0) -> Any:
    """The lidar's revolution from ``truth``, as the board publishes it over the bridge."""
    points = raycast_room(truth, beams=BEAMS, pillar=PILLAR)
    return LaserScan(
        header=Header(stamp=stamp(t), frame_id="laser"),
        angle_min=0.0,
        angle_increment=2.0 * math.pi / BEAMS,
        range_max=12.0,
        ranges=[math.hypot(x, y) for x, y in points],
    )


def pose_msg(pose: Pose2D) -> Any:
    """What the board's tracker believes, as /tracker_pose carries it."""
    msg = PoseMsg()
    msg.pose.pose.position.x, msg.pose.pose.position.y = pose.x, pose.y
    msg.pose.pose.orientation.z = math.sin(pose.theta / 2.0)
    msg.pose.pose.orientation.w = math.cos(pose.theta / 2.0)
    return msg


def until(predicate: Callable[[], Any], timeout_s: float = 20.0) -> bool:
    """Wait for the search thread; True when ``predicate`` came true in time."""
    end = time.monotonic() + timeout_s
    while time.monotonic() < end:
        if predicate():
            return True
        time.sleep(0.005)
    return bool(predicate())


def watch(
    believes: Pose2D | None = None, fit: float = 0.9, grid: Any = None, **flags: Any
) -> GlobalWatch:
    """A node with the room as its map, the laser at the base's origin, one revolution taken
    from :data:`TRUTH` waiting, and — when given — what the board's tracker believes."""
    with ros_stubs.parameters(**flags):
        node = GlobalWatch()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    node.subs["/map"][1](map_msg(furnished_room_map() if grid is None else grid))
    node.subs["/scan"][1](scan_msg(TRUTH))
    if believes is not None:
        node.subs["/tracker_pose"][1](pose_msg(believes))
        node.subs["/localization_fit"][1](Float32(data=fit))
    return node


def published(node: GlobalWatch) -> GlobalCandidate:
    """The newest candidate, parsed back the way the board's tracker parses it."""
    return GlobalCandidate.from_json(node.pubs["/localization/candidate"].sent[-1].data)


def test_the_watch_finds_the_cart_on_the_whole_map_and_says_where() -> None:
    """One tick: the whole-map search, the winner measured in the tracking window, and one
    self-contained JSON message carrying the place, how sure it is and how alike the runner-up
    explained the scan. The pose rides a second topic for Foxglove."""
    node = watch()
    try:
        node._tick()
        assert until(lambda: node.pubs["/localization/candidate"].sent)
        candidate = published(node)
        assert math.hypot(candidate.x - TRUTH.x, candidate.y - TRUTH.y) < 0.1
        assert abs(candidate.yaw - TRUTH.theta) < math.radians(5.0)
        assert candidate.score > 0.5 and candidate.ambiguity < 0.75
        assert candidate.map_id == node._map_id and candidate.stamp == 100.0
        sx, sy, syaw = candidate.measurement().sigmas
        assert 0.0 < sx < 0.1 and 0.0 < sy < 0.1 and 0.0 < syaw < math.radians(10.0)
        pose = node.pubs["/localization/candidate_pose"].sent[-1]
        assert abs(pose.pose.pose.position.x - candidate.x) < 1e-3  # the JSON rounds
        assert pose.header.frame_id == "map"
    finally:
        node.close()


def test_the_verdict_travels_with_the_candidate() -> None:
    """The node judges against what it last heard from the board, for the operator; the board
    judges again against its own fresher pose."""
    agreeing = watch(believes=TRUTH)
    try:
        agreeing._tick()
        assert until(lambda: agreeing.pubs["/localization/candidate"].sent)
        assert f'"verdict": "{CandidateVerdict.AGREE}"' in (
            agreeing.pubs["/localization/candidate"].sent[-1].data
        )
    finally:
        agreeing.close()
    lost = watch(believes=ELSEWHERE, fit=0.2)
    try:
        lost._tick()
        assert until(lambda: lost.pubs["/localization/candidate"].sent)
        sent = lost.pubs["/localization/candidate"].sent[-1].data
        assert f'"verdict": "{CandidateVerdict.DISAGREE}"' in sent
        assert '"search_ms":' in sent
        lost._report()
        line = lost.logger.texts("info")[-1]
        assert "1 candidates from 1 scans" in line and "disagree 1" in line
        assert "ms median/max: search" in line and "measure" in line
        assert "global_watch=on watch_period_s=1.0" in line
    finally:
        lost.close()


def test_the_flag_makes_the_node_a_subscriber_that_costs_nothing() -> None:
    """Off — as the launch starts it in SLAM mode — nothing is searched and nothing published;
    on again, live, and the next tick searches."""
    node = watch(global_watch=False)
    try:
        assert FLAGS["global_watch"] is True, "the module's default: on for the laptop"
        node._tick()
        assert not node.pubs["/localization/candidate"].sent
        node._report()
        assert "off 1" in node.logger.texts("info")[-1]
        assert node.set_parameters([Parameter("global_watch", value=True)])[0].successful
        node._tick()
        assert until(lambda: node.pubs["/localization/candidate"].sent)
    finally:
        node.close()


def test_nothing_is_searched_without_a_map_or_a_scan() -> None:
    """The node waits: no map yet, a scan too thin to say anything, and the period between two
    searches are all counted and none of them is an error."""
    with ros_stubs.parameters(watch_period_s=0.2):
        node = GlobalWatch()
    try:
        node._tick()  # no map, no scan
        node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
        node.subs["/map"][1](map_msg(furnished_room_map()))
        node.subs["/scan"][1](scan_msg(TRUTH, t=101.0))
        node.subs["/scan"][1](
            LaserScan(header=Header(stamp=stamp(102.0), frame_id="laser"), ranges=[])
        )
        node._last_search = 0.0
        node._tick()  # the thin scan
        assert not node.pubs["/localization/candidate"].sent
        node.subs["/scan"][1](scan_msg(TRUTH, t=103.0))
        node._tick()  # the period has not passed since the previous tick
        node._report()
        line = node.logger.texts("info")[-1]
        assert "no map or scan 1" in line and "thin 1" in line and "0 candidates" in line
    finally:
        node.close()


def test_a_map_that_answers_with_two_places_alike_says_so() -> None:
    """The empty rectangle is exactly its own twin turned by 180 degrees: there is no fix to
    publish, and the verdict that travels is unknown_map — the "this map does not fit" the
    tracker counts and the operator is told about. Nothing re-seeds on it."""
    node = watch(believes=ELSEWHERE, fit=0.2, grid=room_map())
    try:
        node._tick()
        assert until(lambda: node.pubs["/localization/candidate"].sent)
        assert published(node).ambiguity > 0.9
        assert f'"verdict": "{CandidateVerdict.UNKNOWN_MAP}"' in (
            node.pubs["/localization/candidate"].sent[-1].data
        )
    finally:
        node.close()
