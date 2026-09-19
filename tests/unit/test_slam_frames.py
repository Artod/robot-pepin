"""The two nodes that carry RTAB-Map's correction to wherever the drive needs it.

rclpy and tf2_ros are faked (``ros_stubs``); ``rtabmap_msgs`` is faked here, because only the
laptop's image carries it — which is the whole reason the board's half of this pair reads a
``geometry_msgs/TransformStamped`` and not a ``MapGraph``. What the fakes let a test see: which
topics a node opened, what it broadcast, and what it published.
"""

import json
import math
import sys
import types
from typing import Any

import numpy as np
import pytest
import ros_stubs

from pepin.tsdf import RigidPose

ros_stubs.install()


class _MapGraph:
    """rtabmap_msgs/MapGraph as the laptop's node still reads it — in SLAM only, where the
    correction IS ``map -> odom`` and belongs on the board."""

    def __init__(self, map_to_odom: Any, stamp: Any = None) -> None:
        self.map_to_odom = map_to_odom
        self.header = types.SimpleNamespace(stamp=stamp if stamp is not None else _stamp(0.0))


class _Info:
    """rtabmap_msgs/Info as the laptop's node reads it: RTAB-Map's statistics table (the keys and
    the values in two parallel arrays, read for the report line alone) and the node a closure or a
    proximity link MATCHED — the one thing that says this update localised at all."""

    def __init__(
        self,
        stats: dict[str, float],
        loop_closure_id: int = 0,
        proximity_detection_id: int = 0,
        stamp: float = 7.0,
    ) -> None:
        self.stats_keys = list(stats)
        self.stats_values = [float(value) for value in stats.values()]
        self.loop_closure_id = loop_closure_id
        self.proximity_detection_id = proximity_detection_id
        # The stamp is what pairs this message with the localisation of the SAME update.
        self.header = types.SimpleNamespace(stamp=_stamp(stamp))


sys.modules.setdefault("rtabmap_msgs", types.ModuleType("rtabmap_msgs"))
sys.modules.setdefault("rtabmap_msgs.msg", types.ModuleType("rtabmap_msgs.msg"))
sys.modules["rtabmap_msgs.msg"].MapGraph = _MapGraph  # type: ignore[attr-defined]
sys.modules["rtabmap_msgs.msg"].Info = _Info  # type: ignore[attr-defined]


class _Empty:
    """std_srvs/Empty: the request and the response both carry nothing, which is the whole of
    RTAB-Map's two set_mode services."""

    class Request:
        pass

    class Response:
        pass


class _SetParameters:
    """rcl_interfaces/SetParameters as this node uses it: a list of parameters in, nothing read
    back — RTAB-Map's own parameters are string-typed and re-read on update_parameters."""

    class Request:
        def __init__(self) -> None:
            self.parameters: list[Any] = []

    class Response:
        pass


sys.modules.setdefault("std_srvs", types.ModuleType("std_srvs"))
sys.modules.setdefault("std_srvs.srv", types.ModuleType("std_srvs.srv"))
sys.modules["std_srvs.srv"].Empty = _Empty  # type: ignore[attr-defined]
sys.modules.setdefault("rcl_interfaces.srv", types.ModuleType("rcl_interfaces.srv"))
sys.modules["rcl_interfaces.srv"].SetParameters = _SetParameters  # type: ignore[attr-defined]
# ...and the two message types ros_stubs does not carry, added beside its own rather than in place
# of them: ParameterType is already there with every wire constant on it.
_interfaces = sys.modules["rcl_interfaces.msg"]
_interfaces.Parameter = types.SimpleNamespace  # type: ignore[attr-defined]
_interfaces.ParameterValue = types.SimpleNamespace  # type: ignore[attr-defined]

from pepin_bringup import rtabmap_frame, slam_frame  # noqa: E402
from pepin_bringup.msgs import transform_from_pose  # noqa: E402


def _stamp(seconds: float) -> Any:
    """A builtin_interfaces/Time at ``seconds``."""
    from builtin_interfaces.msg import Time as TimeMsg

    stamp = TimeMsg()
    stamp.sec = int(seconds)
    stamp.nanosec = round((seconds - int(seconds)) * 1e9)
    return stamp


def _odom(node: Any, x: float, y: float, seconds: float = 7.0) -> None:
    """The EKF's odom -> base_link, as the laptop's node looks it up at a graph's stamp."""
    pose = RigidPose(np.eye(3), np.array([x, y, 0.0]))
    transform = transform_from_pose("odom", "base_link", pose, _stamp(seconds))
    node._lookup.buffer.transforms[("odom", "base_link")] = transform


def _shift(x: float, y: float) -> Any:
    """A correction of x, y metres with no rotation, as a stamped transform."""
    pose = RigidPose(np.eye(3), np.array([x, y, 0.0]))
    return transform_from_pose("a", "b", pose, ros_stubs.Time())


def _belief(
    node: Any,
    x: float,
    y: float,
    seconds: float = 7.0,
    sigma: tuple[float, float, float] = (0.01, 0.01, 0.005),
) -> None:
    """The board's tracker saying where the cart is, on map "flat3", and how sharply it says it:
    ``sigma`` is the score peak's width per axis (m, m, rad), the default a seating the scan
    pins in both axes."""
    from geometry_msgs.msg import PoseWithCovarianceStamped

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "map"
    msg.header.stamp = _stamp(seconds)
    msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
    covariance = list(msg.pose.covariance)
    covariance[0], covariance[7], covariance[35] = (s * s for s in sigma)
    msg.pose.covariance = covariance
    node.subs["/tracker_pose"][1](msg)


def _info(node: Any, hypothesis: float = 0.0, stamp: float = 7.0) -> None:
    """One /rtabmap/info of an update that recognised NOTHING: no node named, and how close the
    last hypothesis came (which is all the statistics are read for)."""
    node.subs[rtabmap_frame.INFO_TOPIC][1](
        _Info({"Loop/Highest_hypothesis_value/": hypothesis}, stamp=stamp)
    )


def _match(node: Any, matched: int = 41, stamp: float = 7.0, proximity: bool = False) -> None:
    """One /rtabmap/info of an update that LOCALISED: ``matched`` is the database node the closure
    (or the proximity link) landed on, and the stamp is what pairs it with the localisation of the
    same update."""
    node.subs[rtabmap_frame.INFO_TOPIC][1](
        _Info(
            {"Loop/Highest_hypothesis_value/": 0.8},
            loop_closure_id=0 if proximity else matched,
            proximity_detection_id=matched if proximity else 0,
            stamp=stamp,
        )
    )


def _localize(
    node: Any, x: float, y: float, yaw: float = 0.0, stamp: float = 7.0, sigma: float = 0.36
) -> None:
    """RTAB-Map placing itself in the database it LOADED: the cart's pose in the graph's own live
    frame, with the covariance it measured (0.36 m is what it reported on 2026-09-17)."""
    from geometry_msgs.msg import PoseWithCovarianceStamped

    msg = PoseWithCovarianceStamped()
    msg.header.frame_id = "rtabmap"
    msg.header.stamp = _stamp(stamp)
    msg.pose.pose.position.x, msg.pose.pose.position.y = x, y
    msg.pose.pose.orientation.z = math.sin(yaw / 2.0)
    msg.pose.pose.orientation.w = math.cos(yaw / 2.0)
    covariance = list(msg.pose.covariance)
    covariance[0] = covariance[7] = sigma * sigma
    covariance[35] = math.radians(4.4) ** 2
    msg.pose.covariance = covariance
    node.subs[rtabmap_frame.LOCALIZATION_TOPIC][1](msg)


def _sources(node: Any, holder: str, moved_m: float = 0.0) -> None:
    """The board's per-source report, with ``holder`` the source the cart has driven the least since
    its word — which is how pepin.watch.Preflight.holding picks the holder, name-free."""
    from std_msgs.msg import String

    others = {
        name: {"health": "fresh 1.0 Hz", "fit": 0.5, "delta": [0.0, 0.0, 0.0], "moved": 9.0}
        for name in ("lidar", "depth", "graph")
        if name != holder
    }
    report = {
        "sources": {
            holder: {
                "health": "fresh 1.0 Hz",
                "fit": 0.9,
                "delta": [0.0, 0.0, 0.0],
                "moved": moved_m,
            },
            **others,
        }
    }
    node.subs[rtabmap_frame.SOURCES_TOPIC][1](String(data=json.dumps(report)))


def _xy(message: Any) -> tuple[float, float]:
    translation = message.transform.translation
    return round(translation.x, 6), round(translation.y, 6)


def test_the_board_broadcasts_identity_until_the_laptop_has_a_graph() -> None:
    """Nav2 needs map -> odom from the first second: with nothing heard yet the honest
    correction is identity — the map is born at the pose the cart starts from."""
    node = slam_frame.SlamFrame()
    assert list(node.subs) == ["/map_odom"]
    assert node.timers and node.timers[0][0] == 1.0 / slam_frame.RATE_HZ
    node.timers[0][1]()
    (sent,) = node._tf.sent
    assert (sent.header.frame_id, sent.child_frame_id) == ("map", "odom")
    assert _xy(sent) == (0.0, 0.0)


def test_the_board_holds_the_last_correction_and_re_stamps_it_every_tick() -> None:
    """A correction is not a measurement: it stands until the graph moves again, and a
    transform stamped a second ago is one Nav2's tolerance refuses."""
    node = slam_frame.SlamFrame()
    _, on_correction = node.subs["/map_odom"]
    on_correction(_shift(0.4, -0.2))
    node.clock.seconds = 5.0
    node.timers[0][1]()
    node.clock.seconds = 5.5
    node.timers[0][1]()
    first, second = node._tf.sent
    assert _xy(first) == _xy(second) == (0.4, -0.2)
    assert first.header.stamp.nanosec == 0 and second.header.stamp.nanosec == 500_000_000


def test_the_board_says_when_the_correction_stops_and_when_it_comes_back() -> None:
    """The edge cannot show this: it is re-broadcast from the LAST correction at the same rate
    with a fresh stamp, so a laptop that went away looks exactly like one that is working. The
    log line is where a running board tells the truth about the other side of the bridge — and
    the broadcast goes on, because Nav2 here must not lose its global frame to a WiFi hiccup."""
    node = slam_frame.SlamFrame()
    _, on_correction = node.subs["/map_odom"]
    on_correction(_shift(0.4, -0.2))
    node.clock.seconds = slam_frame.SILENCE_S
    node.timers[0][1]()
    assert not node.logger.texts("warning"), "still within the patience"

    node.clock.seconds = 9.0
    node.timers[0][1]()
    node.timers[0][1]()
    warned = node.logger.texts("warning")
    assert len(warned) == 1, "said once, not ten times a second"
    assert "nothing on /map_odom for 9.0 s" in warned[0] and "1 heard" in warned[0]
    assert len(node._tf.sent) == 3, "and the edge is broadcast all the same"

    on_correction(_shift(0.5, -0.2))
    assert "the correction is back" in node.logger.texts("info")[-1]


def _map(node: Any, map_id: str = "3x4@1.00,2.00") -> None:
    """The board's served map arriving latched, spelled the way pepin_bringup.msgs.map_id does."""
    from nav_msgs.msg import OccupancyGrid

    msg = OccupancyGrid()
    msg.info.width, msg.info.height = 3, 4
    msg.info.origin.position.x, msg.info.origin.position.y = 1.0, 2.0
    assert map_id == f"{msg.info.width}x{msg.info.height}@1.00,2.00"
    node.subs["/map"][1](msg)


def _fit(node: Any, value: float, at: float) -> None:
    """The lidar's own match score arriving at ``at`` seconds on the node's clock."""
    from std_msgs.msg import Float32

    node.clock.seconds = at
    node.subs["/localization_fit"][1](Float32(data=value))


def _ready(node: Any) -> None:
    """Every service this node calls, answered by somebody."""
    for client in node.service_clients.values():
        client.ready = True


# ---- the laptop: the graph's word, in the ONE map frame ---------------------------------------
def test_on_a_known_map_the_laptop_broadcasts_no_transform_at_all() -> None:
    """RTAB-Map's optimised map frame IS ``map``, so there is no second frame to tie: the anchor,
    the pairs log, the node table and the ``map -> rtabmap`` edge they all served are gone. The
    node's timer beside a known map does one thing — RTAB-Map's memory mode."""
    node = rtabmap_frame.RtabmapFrame()
    assert node._tf is None, "nothing to broadcast: one frame"
    assert rtabmap_frame.CORRECTION_TOPIC not in node.pubs, "the board owns nothing here"
    assert "/rtabmap/mapGraph" not in node.subs, "the global frame is not read at all"
    node.timers[0][1]()
    assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent
    assert "in the one map frame" in node.logger.texts("info")[0]


def test_in_slam_the_laptop_sends_the_correction_home_and_touches_no_tf() -> None:
    """RTAB-Map is the map: the correction IS map -> odom, and it belongs on the board, where
    the reflexes look it up. Nothing is broadcast here — /tf crosses the bridge one way only."""
    with ros_stubs.parameters(slam=True):
        node = rtabmap_frame.RtabmapFrame()
    assert node._tf is not None and not node._tf.sent
    _, on_graph = node.subs["/rtabmap/mapGraph"]
    on_graph(_MapGraph(_shift(0.3, 0.1).transform))
    node.timers[0][1]()
    (sent,) = node.pubs["/map_odom"].sent
    assert (sent.header.frame_id, sent.child_frame_id) == ("map", "odom")
    assert _xy(sent) == (0.3, 0.1), "as it stands, not inverted"
    assert "slam=on" in node.logger.texts("info")[0]
    assert rtabmap_frame.MEASUREMENT_TOPIC not in node.pubs, "no word where there is no database"


def test_the_word_is_the_localisation_itself_stamped_on_the_board_s_clock() -> None:
    """One frame, so the word is RTAB-Map's own localisation carried through untouched. Its STAMP
    is not: a localisation is stamped on the laptop's clock and judged on the board's, minutes
    apart, so the stamp comes from the odom -> base_link transform the board broadcasts ("graph
    stale 128.5 s", 2026-09-17)."""
    node = rtabmap_frame.RtabmapFrame()
    _map(node)
    _odom(node, 0.2, 0.0, seconds=7.0)
    _belief(node, 1.0, 2.0)
    _match(node, matched=41, stamp=99.0)  # the laptop's own clock, minutes off the board's
    _localize(node, 1.0, 2.0, stamp=99.0)
    (sent,) = node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent
    word = json.loads(sent.data)
    assert (word["x"], word["y"]) == (1.0, 2.0), "the localisation, as it came"
    assert word["source"] == "graph" and word["map"] == "3x4@1.00,2.00"
    assert word["stamp"] == 7.0, "the board's clock, off its own odometry transform"
    assert word["node"] == 41 and node._words == 1 and node._sent == 1
    node.timers[1][1]()
    report = node.logger.texts("info")[-1]
    assert "1 updates, 1 recognised a node, 1 localisations heard, 1 words (1 sent" in report
    assert "last word node 41 -> (+1.00, +2.00, +0.0 deg)" in report
    assert "hypothesis 0.80" in report


def test_an_update_that_recognised_nothing_makes_no_word() -> None:
    """MEASURED 2026-09-18: /rtabmap/localization_pose is published on EVERY update, recognised or
    not. Without a named node it is odometry wearing the graph's coordinates — parked, it carried
    1.14 m of accumulated sigma over 11 m of "travel" that was VO jitter at rest — so silence is
    the only honest answer, and the report line is where the count of it lives."""
    node = rtabmap_frame.RtabmapFrame()
    _map(node)
    _odom(node, 0.2, 0.0)
    _belief(node, 1.0, 2.0)
    for _ in range(3):
        _info(node, hypothesis=0.04)
        _localize(node, 5.0, 5.0)
    assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent
    assert not node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent, "nor the other door"
    assert node._words == 0 and node._named == 0 and node._localizations == 3
    node.timers[1][1]()
    assert (
        "3 updates, 0 recognised a node, 3 localisations heard, 0 words"
        in (node.logger.texts("info")[-1])
    )


def test_one_update_makes_one_word_whichever_half_arrives_first() -> None:
    """The two halves of one update are two topics and the order is not ours to choose, so either
    completes the pair — and the update's own stamp is what keeps it from being spent twice."""
    for localisation_first in (False, True):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0)
        if localisation_first:
            _localize(node, 1.0, 2.0, stamp=12.0)
            _match(node, matched=41, stamp=12.0)
        else:
            _match(node, matched=41, stamp=12.0)
            _localize(node, 1.0, 2.0, stamp=12.0)
        assert node._words == 1, "one update, one word"
        _match(node, matched=41, stamp=12.0)  # the same update's info again
        assert node._words == 1, "and it is spent"
        node.close() if hasattr(node, "close") else None

    # ...and two halves of DIFFERENT updates are no pair at all
    node = rtabmap_frame.RtabmapFrame()
    _map(node)
    _odom(node, 0.2, 0.0)
    _match(node, matched=41, stamp=12.0)
    _localize(node, 1.0, 2.0, stamp=12.0 + rtabmap_frame.UPDATE_MAX_SKEW_S + 0.1)
    assert node._words == 0 and node._named == 1


def test_a_proximity_link_localises_exactly_as_a_closure_does() -> None:
    """A closure recognises a place by its appearance and a proximity link by walking back to it;
    both tie the present to a node of the loaded database, which is all a word ever needs."""
    node = rtabmap_frame.RtabmapFrame()
    _map(node)
    _odom(node, 0.2, 0.0)
    _match(node, matched=2841, proximity=True)
    _localize(node, 1.0, 2.0)
    assert node._words == 1 and node._node == 2841


def test_rtabmap_s_own_covariance_rides_the_word_and_is_floored_never_replaced() -> None:
    """What the registration against the recognised node is worth is RTAB-Map's to say. Its own
    arithmetic is useless in both directions (706 m of sigma with no closure, 8 mm right after one),
    so the measured floor of this source is raised under it — 0.20 m / 8 deg, what a graph word was
    last seen to be worth against the lidar — and a wider claim is kept exactly as it came."""
    node = rtabmap_frame.RtabmapFrame()
    _map(node)
    _odom(node, 0.2, 0.0)
    _match(node)
    _localize(node, 1.0, 2.0, sigma=0.36)  # what RTAB-Map reported parked on 2026-09-17
    word = json.loads(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1].data)
    assert word["covariance"][0][0] == pytest.approx(0.36**2), "its own claim, wider than the floor"
    assert word["covariance"][2][2] == pytest.approx(math.radians(8.0) ** 2), "8 deg is the floor"

    node = rtabmap_frame.RtabmapFrame()
    _map(node)
    _odom(node, 0.2, 0.0)
    _match(node)
    _localize(node, 1.0, 2.0, sigma=0.008)  # a graph sure of itself, which it has no right to be
    word = json.loads(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1].data)
    assert word["covariance"][0][0] == pytest.approx(0.20**2), "floored at the measured 20 cm"


def test_with_graph_measurement_off_the_word_stays_on_this_laptop() -> None:
    """CLAUDE.md rule 19: the switch, so a session's words can be read in the report line and on
    the recorded topic before they are allowed to move the pose."""
    with ros_stubs.parameters(graph_measurement=False):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0)
        _match(node)
        _localize(node, 1.0, 2.0)
    assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent
    assert node._words == 1 and node._sent == 0, "counted and remembered all the same"


def test_a_word_with_no_odometry_to_stamp_itself_by_is_not_said() -> None:
    """The stamp is the board's, so a word made while the bridge is quiet would carry no moment at
    all: it is counted blind instead of stamped on this laptop's clock."""
    node = rtabmap_frame.RtabmapFrame()
    _map(node)
    _match(node)
    _localize(node, 1.0, 2.0)
    assert node._blind == 1 and node._words == 0
    assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent


def test_the_word_the_fusion_cannot_use_goes_out_as_a_candidate() -> None:
    """A measurement can only correct a pose that is already nearly right: this node refuses a word
    past the fusion's own chi-square in 3 DOF and the board's filter gates what it does take. The
    word that undoes a CARRY is exactly that refused word, so it goes out on the whole-map
    candidate channel instead — the door the lidar's own search re-seeds through."""
    node = rtabmap_frame.RtabmapFrame()
    _map(node)
    _fit(node, 0.9, at=5.0)  # the lidar is driving and the board is talking to us
    _belief(node, 1.0, 2.0)
    _odom(node, 0.2, 0.0)
    _match(node, stamp=7.0)
    _localize(node, 1.0, 2.0, stamp=7.0)
    assert node._sent == 1, "the word is a word"
    assert not node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent, "the tracker is already there"

    # the graph recognises the place and puts the cart two metres from the tracker's belief
    _match(node, stamp=8.0)
    _localize(node, 3.0, 2.0, stamp=8.0)
    assert node._sent == 1, "refused as too far"
    (sent,) = node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent
    word = json.loads(sent.data)
    assert (word["x"], word["y"]) == (3.0, 2.0) and word["source"] == "graph"
    assert word["map"] == "3x4@1.00,2.00" and word["ambiguity"] == 0.0
    assert word["scan"] == 2, "one recognised update is one piece of evidence"
    assert node._proposed == 1 and node._refused == 1

    # ...and with the board quiet for three seconds the tracker has no source behind its pose, so
    # a word about ANOTHER place travels as a candidate even within the gate (it is a measurement
    # too: the fusion may still use what it can)
    node.clock.seconds = 5.0 + rtabmap_frame.BELIEF_FRESH_S + 0.1
    _match(node, stamp=9.0)
    _localize(node, 1.6, 2.0, stamp=9.0)
    assert node._proposed == 2 and node._sent == 2
    assert json.loads(node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent[-1].data)["x"] == 1.6
    node.timers[1][1]()
    assert "2 candidates" in node.logger.texts("info")[-1]


def test_a_word_the_tracker_can_confirm_asks_for_nothing() -> None:
    """The other half of the rule, and what keeps the graph out of the lidar's way: a word about
    the place the tracker already holds would be called agreement by the board's gate, act on
    nothing, and end whatever streak the LIDAR's own search had built."""
    node = rtabmap_frame.RtabmapFrame()
    _map(node)
    _belief(node, 1.0, 2.0)
    _odom(node, 0.2, 0.0)
    _fit(node, 0.0, at=0.0)  # nothing is confirming the tracker's pose
    _match(node, stamp=7.0)
    _localize(node, 1.3, 2.0, stamp=7.0)
    assert node._lost() and not node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent, "30 cm"
    _match(node, stamp=8.0)
    _localize(node, 1.8, 2.0, stamp=8.0)
    assert node._proposed == 1, "80 cm is another place, and nothing is confirming the pose"


def test_the_candidate_channel_can_be_switched_off() -> None:
    """CLAUDE.md rule 19: the old behaviour stays reachable. Off, a word too far to fuse is counted
    here and reaches nothing — which is what a graph whose recognitions are in doubt calls for,
    since three words off by one constant offset agree with each other perfectly."""
    with ros_stubs.parameters(graph_candidates=False):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _belief(node, 1.0, 2.0)
        _odom(node, 0.2, 0.0)
        _match(node)
        _localize(node, 3.0, 2.0)
    assert not node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent and node._proposed == 0
    assert node._refused == 1, "refused as before, and now it is the end of the road again"


def test_a_word_at_the_right_place_facing_the_wrong_way_is_refused() -> None:
    """The gate the old node did not have. A word 90 degrees out at exactly the place the tracker
    holds passed the metres-only test and went into the fusion; the test is the 3-DOF Mahalanobis
    distance the board's own filter applies (pepin.fusion.GATE), so the heading counts — and the
    refused word still goes out on the candidate channel, where a carried cart needs it."""
    node = rtabmap_frame.RtabmapFrame()
    _map(node)
    _fit(node, 0.9, at=5.0)  # the board is talking to us and the lidar holds the pose
    _belief(node, 1.0, 2.0)
    _odom(node, 0.2, 0.0)
    _match(node, stamp=7.0)
    _localize(node, 1.0, 2.0, stamp=7.0)
    assert node._sent == 1 and node._refused == 0 and node._gap_m == pytest.approx(0.0)

    _match(node, stamp=8.0)
    _localize(node, 1.0, 2.0, yaw=math.pi / 2, stamp=8.0)
    assert node._gap_m < 0.01, "the same place by metres, which is what used to be asked"
    assert node._gap_sigmas**2 > rtabmap_frame.GATE, "and another pose by the gate"
    assert node._refused == 1 and node._sent == 1
    assert node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent, "the door a carried cart needs"
    node.timers[1][1]()
    assert "1 refused by the gate" in node.logger.texts("info")[-1]


def test_the_word_s_fit_is_how_well_the_last_words_track_the_odometry() -> None:
    """What no covariance can see: a word riding a frame that no longer holds is a metre out from
    the moment it is said. The fit is exp(-rms residual / scale) over the last words against the
    tracker's pose carried forward by odometry between them — 1.0 while the word follows the cart,
    under the whole-map candidate floor the moment it stops."""
    from pepin.watch import ADMIT_FIT

    def fit(node: Any) -> float:
        return float(json.loads(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1].data)["fit"])

    node = rtabmap_frame.RtabmapFrame()
    _map(node)
    _odom(node, 0.2, 0.0)
    _belief(node, 1.0, 2.0)
    _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
    _match(node, stamp=7.0)
    _localize(node, 1.0, 2.0, stamp=7.0)
    assert fit(node) == 1.0, "the word is exactly where the tracker is"

    # the cart drives half a metre and the graph's word drives with it
    _odom(node, 0.7, 0.0)
    _belief(node, 1.5, 2.0)
    _match(node, stamp=8.0)
    _localize(node, 1.5, 2.0, stamp=8.0)
    assert fit(node) == 1.0, "the word is where the odometry says the cart is"

    # ...and now the graph's answer slips 30 cm under a cart that has not moved
    _match(node, stamp=9.0)
    _localize(node, 1.8, 2.0, stamp=9.0)
    assert fit(node) < ADMIT_FIT, "no candidate is admitted on this any more"
    node.timers[1][1]()
    report = node.logger.texts("info")[-1]
    assert "residual 17.3 cm" in report and "over 3 words" in report


def test_a_wake_up_word_travels_with_no_belief_to_check_it_against() -> None:
    """The cart sleeps on the charger and the board restarts: nothing is on /tracker_pose yet, and
    the graph recognising the place IS the answer. It is safe because the word depends on nothing
    this session measured — one frame, RTAB-Map's own recognition, and its own covariance."""
    node = rtabmap_frame.RtabmapFrame()
    _map(node)
    _odom(node, 0.2, 0.0)
    _match(node)
    _localize(node, -9.4, 2.5)
    word = json.loads(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1].data)
    assert (word["x"], word["y"]) == (-9.4, 2.5) and node._sent == 1
    assert node._proposed == 1, "and the candidate channel too: the tracker has no source at all"
    assert node._gap_m == math.inf, "nothing to compare it with, and it says so"


# ---- RTAB-Map's memory: when the database may learn -------------------------------------------
def test_the_mode_follows_trust_in_the_pose_and_not_a_sensor_s_name() -> None:
    """The database may LEARN only while a sharp pose exists that does not come from the database
    itself. Measured live on 2026-09-18: in mapping mode parked, RTAB-Map found this very place
    (hypothesis 0.978, 328 visual inliers) and rejected its own recognition on the error ratio, and
    it kept a node a second — while in localisation mode with the updates ungated every update named
    a node. So the mode is switched live, on the rule and never on a flag alone."""
    node = rtabmap_frame.RtabmapFrame()
    _ready(node)
    _map(node)
    _sources(node, holder="lidar")
    _belief(node, 1.0, 2.0)
    _fit(node, 0.7, at=0.0)  # the lidar holds the pose and its seating is sharp
    node.timers[0][1]()
    mapping = node.service_clients[rtabmap_frame.MAPPING_SERVICE]
    assert len(mapping.calls) == 1, "the initial mode is asked for at once"
    assert node._mode.mode == "mapping"
    # ...and the parameters the mode needs travel with it, as strings
    tuned = node.service_clients[f"{rtabmap_frame.RTABMAP_NODE}/set_parameters"]
    assert [p.name for p in tuned.calls[-1].parameters] == [
        "RGBD/LinearUpdate",
        "RGBD/AngularUpdate",
    ]
    assert [p.value.string_value for p in tuned.calls[-1].parameters] == ["0.05", "0.05"]

    # the lidar is still matching, but the graph's own words now hold the pose: the pupil is not
    # the teacher, whatever the seating is worth
    _sources(node, holder="graph")
    node.timers[0][1]()
    localising = node.service_clients[rtabmap_frame.LOCALISATION_SERVICE]
    assert not localising.calls, "one tick is not the freshness window"
    _fit(node, 0.7, at=rtabmap_frame.FIT_FRESH_S + 0.1)  # the clock moves, the lidar keeps up
    _belief(node, 1.0, 2.0)
    _sources(node, holder="graph")
    node.timers[0][1]()
    assert len(localising.calls) == 1 and node._mode.mode == "localising"
    assert [p.value.string_value for p in tuned.calls[-1].parameters] == ["0", "0"]
    node.timers[1][1]()
    report = node.logger.texts("info")[-1]
    assert "rtabmap memory localising (the pose is held by the graph" in report
    assert "the pupil is not the teacher, 2 switches, by trust" in report


def test_a_soft_pose_teaches_nothing_however_it_is_held() -> None:
    """The sharpness half of the rule, and the half that needs no names at all: a mono camera-only
    pose sits at a sigma around 20 cm and fails the seating test by itself, so a database is never
    taught from it — and a stereo matcher good to a few cm will teach the day it exists."""
    node = rtabmap_frame.RtabmapFrame()
    _ready(node)
    _map(node)
    _sources(node, holder="depth")  # NOT the graph: the pupil test passes
    _belief(node, 1.0, 2.0, sigma=(0.20, 0.20, 0.05))  # ...and the seating does not
    _fit(node, 0.7, at=0.0)
    node.timers[0][1]()
    assert node._mode.mode == "localising"
    assert not node.service_clients[rtabmap_frame.MAPPING_SERVICE].calls
    wanted = node._mode.wanted
    assert wanted is not None and "not worth learning from" in wanted.why


def test_the_sharpness_gate_can_be_opened_in_the_field() -> None:
    """CLAUDE.md rule 19: the gate is a live flag, so a database can still be extended in a room
    where no seating is ever sharp — with the sigmas in the report line to read first."""
    with ros_stubs.parameters(graph_memory_sigma_m=1.0, graph_memory_sigma_deg=180.0):
        node = rtabmap_frame.RtabmapFrame()
        _ready(node)
        _map(node)
        _sources(node, holder="depth")
        _belief(node, 1.0, 2.0, sigma=(0.20, 0.20, 0.05))
        _fit(node, 0.7, at=0.0)
        node.timers[0][1]()
    assert node._mode.mode == "mapping"
    assert len(node.service_clients[rtabmap_frame.MAPPING_SERVICE].calls) == 1


def test_the_mode_override_keeps_both_old_arrangements_reachable() -> None:
    """CLAUDE.md rule 19. graph_memory=map is the arrangement of before 2026-09-18 (always mapping);
    localise freezes the database for a session, which is what a measurement of what the graph's
    own recognitions are worth needs."""
    with ros_stubs.parameters(graph_memory="map"):
        node = rtabmap_frame.RtabmapFrame()
        _ready(node)
        _map(node)
        _sources(node, holder="graph")  # every reason to localise...
        node.timers[0][1]()
    assert len(node.service_clients[rtabmap_frame.MAPPING_SERVICE].calls) == 1, "...and told to map"

    with ros_stubs.parameters(graph_memory="localise"):
        frozen = rtabmap_frame.RtabmapFrame()
        _ready(frozen)
        _map(frozen)
        _sources(frozen, holder="lidar")
        _belief(frozen, 1.0, 2.0)
        _fit(frozen, 0.7, at=0.0)
        frozen.timers[0][1]()
    assert len(frozen.service_clients[rtabmap_frame.LOCALISATION_SERVICE].calls) == 1
