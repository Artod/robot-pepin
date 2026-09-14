"""The laptop's localizer: a candidate on the wire once a second, and the camera's own poses.

rclpy is faked (``ros_stubs``), the map and the scans are the furnished room of
test_localization, so the node is built and driven here exactly as on the laptop — the whole
search included, which is what makes this the one test that says what a candidate and a
measurement really carry.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from typing import Any

import ros_stubs

RCLPY = ros_stubs.install()

from pepin_bringup.laptop_localizer import (  # noqa: E402
    CAMERA_MAP_TOPIC,
    FLAGS,
    LaptopLocalizer,
)
from pepin_bringup.msgs import transform_from_rpy  # noqa: E402
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
from test_relocalizer_node import depth_msg, map_msg, odom_msg, stamp  # noqa: E402

from pepin.measurements import RemoteMeasurement  # noqa: E402
from pepin.odometry import Pose2D  # noqa: E402
from pepin.sources import CONTACT, DEPTH  # noqa: E402
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


def pose_msg(pose: Pose2D, t: float = 100.0) -> Any:
    """What the board's tracker believes, as /tracker_pose carries it: the pose and the moment
    it speaks for (the stamp of the scan it was matched on)."""
    msg = PoseMsg(header=Header(stamp=stamp(t), frame_id="map"))
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
) -> LaptopLocalizer:
    """A node with the room as its map, the laser at the base's origin, one revolution taken
    from :data:`TRUTH` waiting, and — when given — what the board's tracker believes."""
    with ros_stubs.parameters(**flags):
        node = LaptopLocalizer()
    node._tf.buffer.transforms[("base_link", "laser")] = TransformStamped()
    node.subs["/map"][1](map_msg(furnished_room_map() if grid is None else grid))
    node.subs["/scan"][1](scan_msg(TRUTH))
    if believes is not None:
        node.subs["/tracker_pose"][1](pose_msg(believes))
        node.subs["/localization_fit"][1](Float32(data=fit))
    return node


def standing(node: LaptopLocalizer, believes: Pose2D = TRUTH, t: float = 100.0) -> None:
    """The board's word at ``t``: one belief and the odometry around it, the cart standing
    still, so a camera scan of that moment has a pose to be matched around."""
    node.subs["/tracker_pose"][1](pose_msg(believes, t))
    node.subs["/localization_fit"][1](Float32(data=0.9))
    for k in range(16):  # a second and a half of trail: a carry never runs off its end here
        node.subs["/odometry/filtered"][1](odom_msg(Pose2D(), t - 0.5 + 0.1 * k))


def measured(node: LaptopLocalizer) -> RemoteMeasurement:
    """The newest measurement, parsed back the way the board's tracker parses it."""
    return RemoteMeasurement.from_json(node.pubs["/localization/measurement"].sent[-1].data)


def published(node: LaptopLocalizer) -> GlobalCandidate:
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


def test_one_revolution_is_one_candidate_however_often_it_arrives() -> None:
    """A message repeating the stamp in hand is the same revolution over again (a bridge
    redelivering): it is counted and dropped, so the id that travels with a candidate is the
    identity of a REVOLUTION — which is what the board counts a streak in."""
    node = watch()
    try:
        held = node._scan
        node.subs["/scan"][1](scan_msg(TRUTH))  # the same revolution, a second time
        assert node._scan is held
        node._tick()
        assert until(lambda: node.pubs["/localization/candidate"].sent)
        assert published(node).scan_id == 1
        node.subs["/scan"][1](scan_msg(TRUTH, t=100.1))
        node._last_search = 0.0
        node._tick()
        assert until(lambda: len(node.pubs["/localization/candidate"].sent) == 2)
        assert published(node).scan_id == 2
        node._report()
        assert "revolutions heard twice 1" in node.logger.texts("info")[-1]
    finally:
        node.close()


def test_a_revolution_nobody_replaces_stops_being_searched() -> None:
    """The bridge wedges and /scan freezes: searching the same revolution again would publish
    the same answer once a second as if it were news, and three of those on the board used to
    look like three seconds of evidence. Freshness is counted from when the scan ARRIVED here —
    never the node's clock minus its stamp, which is the board's and runs seconds ahead."""
    node = watch()
    try:
        node._scan_at -= 2.0  # nothing has arrived for two seconds
        node._tick()
        assert not node.pubs["/localization/candidate"].sent
        node._report()
        assert "nothing new 1" in node.logger.texts("info")[-1]
        node.subs["/scan"][1](scan_msg(TRUTH, t=101.0))
        node._tick()
        assert until(lambda: node.pubs["/localization/candidate"].sent)
    finally:
        node.close()


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
        node = LaptopLocalizer()
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


# ---- the camera's own poses -----------------------------------------------------------------
def test_a_camera_scan_around_the_board_s_belief_becomes_a_measurement() -> None:
    """The day's verdict in one test: the camera's fan is matched HERE, in a small window around
    the pose the board believes in at that fan's own stamp, and what crosses the link is the
    place it measured — with the covariance the score surface gave it, the source's name, the
    scan's stamp and the map id the board holds."""
    node = watch()
    try:
        standing(node)
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 100.1))
        answer = measured(node)
        assert answer.source == DEPTH and abs(answer.stamp - 100.1) < 1e-6
        assert answer.map_id == node._map_id, "the board's map, not the grid we matched against"
        assert math.hypot(answer.x - TRUTH.x, answer.y - TRUTH.y) < 0.1
        assert abs(answer.yaw - TRUTH.theta) < math.radians(5.0)
        assert answer.fit > 0.4 and answer.covariance.shape == (3, 3)
        sx, sy, syaw = answer.measurement().sigmas
        assert 0.0 < sx < 1.0 and 0.0 < sy < 1.0 and 0.0 < syaw < math.radians(45.0)
        sent = node.pubs["/localization/measurement"].sent[-1].data
        assert '"matched_on": "/map"' in sent and '"belief_age_ms": 100.0' in sent
        node._report()
        line = node.logger.texts("info")[-1]
        assert "measurements: depth 1 at fit" in line and "against /map" in line
        assert "camera_sources=depth,contact camera_match_hz=5.0" in line
    finally:
        node.close()


def test_each_source_is_matched_at_its_own_rate_and_the_rest_are_paced() -> None:
    """5 Hz per source, counted on the SCANS' stamps so a replay paces as the robot does: a
    second fan 50 ms after the first is not matched, one 200 ms after it is, and the contact
    line's rate is its own."""
    node = watch()
    try:
        standing(node)
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 100.0))
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 100.05))
        node.subs["/contact_scan"][1](depth_msg(TRUTH, 100.05))
        assert len(node.pubs["/localization/measurement"].sent) == 2, "depth paced, contact not"
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 100.25))
        assert len(node.pubs["/localization/measurement"].sent) == 3
        assert {
            m.source
            for m in map(
                RemoteMeasurement.from_json,
                [msg.data for msg in node.pubs["/localization/measurement"].sent],
            )
        } == {DEPTH, CONTACT}
        node._report()
        assert "paced 1" in node.logger.texts("info")[-1]
    finally:
        node.close()


def test_without_a_belief_to_start_from_nothing_is_measured() -> None:
    """A match is a refinement of the board's own pose: with no pose heard, or one old enough to
    mean the board or the bridge is gone, there is nothing to refine and the scan is counted,
    not guessed at. The same for a scan the odometry trail does not reach."""
    node = watch()
    try:
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 100.0))
        assert not node.pubs["/localization/measurement"].sent
        standing(node, t=100.0)
        node.subs["/odometry/filtered"][1](odom_msg(Pose2D(), 103.0))
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 103.0))  # the belief is 3 s old
        assert not node.pubs["/localization/measurement"].sent
        node._report()
        line = node.logger.texts("info")[-1]
        assert "no belief 1" in line and "stale belief 1" in line
    finally:
        node.close()


def test_the_camera_matches_the_volume_s_own_band_when_the_fusion_publishes_it() -> None:
    """The lidar's plane and the camera's band are two cross-sections of one room, and a scan
    must be matched against its own: once pepin_bringup.depth_fusion publishes /map_camera, that
    is the grid the camera's scans are refined on — and the measurement still carries /map's id,
    because that is the map the board holds."""
    node = watch()
    try:
        standing(node)
        node.subs[CAMERA_MAP_TOPIC][1](map_msg(furnished_room_map()))
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 100.1))
        assert node._camera_map == CAMERA_MAP_TOPIC
        assert measured(node).map_id == node._map_id
        assert '"matched_on": "/map_camera"' in (
            node.pubs["/localization/measurement"].sent[-1].data
        )
        node._report()
        assert "against /map_camera" in node.logger.texts("info")[-1]
    finally:
        node.close()


def test_the_flag_takes_the_camera_out_of_the_pose_without_touching_anything_else() -> None:
    """camera_sources empty: the scans still arrive (the costmap's copy of them is another
    matter entirely), nothing is matched, nothing crosses, and the board is on the lidar alone.
    Live, and a source at a time."""
    node = watch(camera_sources="")
    try:
        standing(node)
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 100.1))
        node.subs["/contact_scan"][1](depth_msg(TRUTH, 100.1))
        assert not node.pubs["/localization/measurement"].sent
        node._report()
        assert "source off 2" in node.logger.texts("info")[-1]
        assert node.set_parameters([Parameter("camera_sources", value="contact")])[0].successful
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 100.3))
        node.subs["/contact_scan"][1](depth_msg(TRUTH, 100.3))
        assert [measured(node).source] == [CONTACT]
    finally:
        node.close()


def test_a_match_that_explains_nothing_is_not_sent() -> None:
    """A fan of somewhere else, matched in a hand's width around where the board thinks the cart
    is, fits nothing: the pose it lands on is the window's tie-break rather than a measurement,
    so it is counted as a low fit and the board never hears it."""
    node = watch()
    try:
        standing(node)
        node.subs["/depth_scan"][1](depth_msg(ELSEWHERE, 100.1))
        assert not node.pubs["/localization/measurement"].sent
        node._report()
        assert "low fit 1" in node.logger.texts("info")[-1]
    finally:
        node.close()


def crowded(truth: Pose2D, t: float, legs: int = 14, person_m: float = 0.7) -> Any:
    """The camera's fan with a person standing in it: ``legs`` beams around the middle come
    back off a body 0.7 m ahead that the static map has no wall for."""
    msg = depth_msg(truth, t)
    ranges = list(msg.ranges)
    middle = len(ranges) // 2
    for i in range(middle - legs // 2, middle - legs // 2 + legs):
        ranges[i] = person_m
    msg.ranges = ranges
    return msg


def test_the_returns_the_map_cannot_explain_do_not_score_the_camera_match() -> None:
    """The board's vote, taken here now that the match is here: a person filling a third of the
    fan is silenced, the mask is read on the grid the match is scored against, and the pose that
    crosses the link is still the cart's. The flag off, nothing is silenced and no mask is even
    built (scratch/camera_vote_probe.py has the centimetres)."""
    node = watch()
    try:
        standing(node)
        node.subs["/depth_scan"][1](crowded(TRUTH, 100.1))
        answer = measured(node)
        assert math.hypot(answer.x - TRUTH.x, answer.y - TRUTH.y) < 0.1
        assert node._camera is not None and node._mask is not None
        assert node._mask_of == (node._camera.grid, node._camera.grid.version)
        node._report()
        assert "1 with unexplained returns silenced" in node.logger.texts("info")[-1]
    finally:
        node.close()
    node = watch(explained_vote=False)
    try:
        standing(node)
        node.subs["/depth_scan"][1](crowded(TRUTH, 100.1))
        assert node.pubs["/localization/measurement"].sent and node._mask is None
        node._report()
        line = node.logger.texts("info")[-1]
        assert "0 with unexplained returns silenced" in line and "explained_vote=off" in line
    finally:
        node.close()


def tf_belief(node: LaptopLocalizer, pose: Pose2D, t: float = 100.0) -> None:
    """``map -> base_link`` at ``t`` in the live TF tree — what the board broadcasts 20 times a
    second whatever its tracker is doing (map -> odom over odom -> base_link)."""
    node._tf_live.buffer.transforms[("map", "base_link")] = transform_from_rpy(
        "map", "base_link", (pose.x, pose.y, 0.0), (0.0, 0.0, pose.theta), stamp(t)
    )


def test_tf_answers_where_the_board_cannot_speak_yet() -> None:
    """The deadlock of 2026-09-13 night, broken: on sources=camera the board publishes
    /tracker_pose only after an update, an update needs a measurement, and a measurement needed
    a belief — 413 camera scans were rejected with "no belief" in one evening. TF has no such
    circle: map -> odom is broadcast 20 times a second whatever the tracker does, so the pose at
    the scan's own stamp is there before the board has said anything at all, and it needs no
    carry over the odometry to reach that stamp."""
    node = watch()
    try:
        tf_belief(node, TRUTH, 100.1)
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 100.1))
        answer = measured(node)
        assert math.hypot(answer.x - TRUTH.x, answer.y - TRUTH.y) < 0.1
        assert abs(answer.yaw - TRUTH.theta) < math.radians(5.0)
        sent = node.pubs["/localization/measurement"].sent[-1].data
        assert '"belief_from": "tf"' in sent and '"belief_age_ms": 0.0' in sent
        node._report()
        line = node.logger.texts("info")[-1]
        assert "belief: tracker 0, tf 1" in line and "tf_belief=on" in line
    finally:
        node.close()


def test_the_board_s_own_pose_is_preferred_while_it_is_fresh() -> None:
    """/tracker_pose is the board's fused answer and the only belief that carries a covariance:
    while it is fresher than a second it wins, TF is not even asked, and the measurement says
    how old it was. Once it goes quiet the same scan is matched around TF instead."""
    node = watch()
    try:
        tf_belief(node, ELSEWHERE, 100.1)  # there to be taken, and not taken
        standing(node, t=100.0)
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 100.1))
        sent = node.pubs["/localization/measurement"].sent[-1].data
        assert '"belief_from": "tracker"' in sent and '"belief_age_ms": 100.0' in sent
        assert math.hypot(measured(node).x - TRUTH.x, measured(node).y - TRUTH.y) < 0.1
        tf_belief(node, TRUTH, 101.5)
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 101.5))  # the board has said nothing since
        assert '"belief_from": "tf"' in node.pubs["/localization/measurement"].sent[-1].data
        node._report()
        assert "belief: tracker 1, tf 1" in node.logger.texts("info")[-1]
    finally:
        node.close()


def test_the_flag_off_leaves_the_old_rule_and_a_tf_that_answers_nothing_is_counted() -> None:
    """Off, the belief is the board's pose or none at all — the behaviour before tonight. On
    with an empty TF tree, the old rule still stands behind it: a pose younger than two seconds
    is carried as it always was, and the lookups that answered nothing are counted by tf2's own
    name for the failure."""
    node = watch(tf_belief=False)
    try:
        assert FLAGS["tf_belief"] is True, "the module's default: on"
        tf_belief(node, TRUTH, 100.1)
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 100.1))
        assert not node.pubs["/localization/measurement"].sent
        node._report()
        assert "no belief 1" in node.logger.texts("info")[-1]
        assert node.set_parameters([Parameter("tf_belief", value=True)])[0].successful
        node.subs["/depth_scan"][1](depth_msg(TRUTH, 100.3))
        assert '"belief_from": "tf"' in node.pubs["/localization/measurement"].sent[-1].data
    finally:
        node.close()
    blind = watch()
    try:
        standing(blind, t=100.0)  # the board's word, then 1.4 s of silence and no TF at all
        blind.subs["/odometry/filtered"][1](odom_msg(Pose2D(), 101.4))
        blind.subs["/depth_scan"][1](depth_msg(TRUTH, 101.4))
        sent = blind.pubs["/localization/measurement"].sent[-1].data
        assert '"belief_from": "tracker"' in sent, "TF silent: the old carry still answers"
        blind._report()
        line = blind.logger.texts("info")[-1]
        assert "no tf 1 (" in line and "belief: tracker 1, tf 0" in line
    finally:
        blind.close()
