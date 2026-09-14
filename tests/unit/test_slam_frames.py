"""The two nodes that carry RTAB-Map's correction to wherever the drive needs it.

rclpy and tf2_ros are faked (``ros_stubs``); ``rtabmap_msgs`` is faked here, because only the
laptop's image carries it — which is the whole reason the board's half of this pair reads a
``geometry_msgs/TransformStamped`` and not a ``MapGraph``. What the fakes let a test see: which
topics a node opened, what it broadcast, and what it published.
"""

import sys
import types
from typing import Any

import numpy as np
import ros_stubs

from pepin.tsdf import RigidPose

ros_stubs.install()


class _MapGraph:
    """rtabmap_msgs/MapGraph as these nodes read it: the correction, and the stamp the
    laptop's node reads the cart's odometry at."""

    def __init__(self, map_to_odom: Any, stamp: Any = None) -> None:
        self.map_to_odom = map_to_odom
        self.header = types.SimpleNamespace(stamp=stamp if stamp is not None else _stamp(0.0))


class _Info:
    """rtabmap_msgs/Info as the laptop's node reads it: RTAB-Map's statistics table, the keys
    and the values in two parallel arrays."""

    def __init__(self, stats: dict[str, float]) -> None:
        self.stats_keys = list(stats)
        self.stats_values = [float(value) for value in stats.values()]


sys.modules.setdefault("rtabmap_msgs", types.ModuleType("rtabmap_msgs"))
sys.modules.setdefault("rtabmap_msgs.msg", types.ModuleType("rtabmap_msgs.msg"))
sys.modules["rtabmap_msgs.msg"].MapGraph = _MapGraph  # type: ignore[attr-defined]
sys.modules["rtabmap_msgs.msg"].Info = _Info  # type: ignore[attr-defined]

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


def _info(node: Any, travelled: float, loop: float = 0.0, hypothesis: float = 0.0) -> None:
    """One /rtabmap/info into the node: how far RTAB-Map's odometry has travelled, whether this
    message tied the present to an older node, and how close the last hypothesis came."""
    node.subs[rtabmap_frame.INFO_TOPIC][1](
        _Info(
            {
                "Memory/Distance_travelled/m": travelled,
                "Loop/Id/": loop,
                "Loop/Highest_hypothesis_value/": hypothesis,
            }
        )
    )


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


def test_on_a_known_map_the_laptop_broadcasts_the_anchor_it_learned_once() -> None:
    """RTAB-Map runs on the EKF's odometry, so its graph starts at the odom frame's origin: the
    edge our tree needs is where that frame sits on the map, learned from the tracker at the
    first graph and then held. Identity until it is known — a tree with a hole in it is worse
    than one edge that is not yet right."""
    node = rtabmap_frame.RtabmapFrame()
    assert node.frames == ("map", "rtabmap")
    assert rtabmap_frame.CORRECTION_TOPIC not in node.pubs, "the board owns nothing here"
    node.timers[0][1]()
    (identity,) = node._tf.sent
    assert _xy(identity) == (0.0, 0.0), "no graph, no anchor"

    _belief(node, 1.0, 2.0)
    _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
    _odom(node, 0.2, 0.0)
    node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    node.timers[0][1]()
    sent = node._tf.sent[-1]
    assert (sent.header.frame_id, sent.child_frame_id) == ("map", "rtabmap")
    assert _xy(sent) == (0.8, 2.0), "the tracker at (1, 2), the graph's cart at (0.2, 0)"

    # ...and it is learned once: a later graph moves the word, never the frame
    node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.1, 0.0).transform))
    node.timers[0][1]()
    assert _xy(node._tf.sent[-1]) == (0.8, 2.0)


def test_with_graph_odom_off_the_laptop_broadcasts_the_inverse_correction() -> None:
    """The arrangement before 2026-09-14, one launch argument away: RTAB-Map's odometry is the
    tracker's pose, its correction reads rtabmap -> map, and the edge our tree needs is the
    other way round."""
    with ros_stubs.parameters(graph_odom=False):
        node = rtabmap_frame.RtabmapFrame()
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.3, 0.0).transform))
        node.timers[0][1]()
    (sent,) = node._tf.sent
    assert (sent.header.frame_id, sent.child_frame_id) == ("map", "rtabmap")
    assert _xy(sent) == (-0.3, 0.0), "the inverse of the correction"


def test_the_graph_s_word_is_where_the_graph_puts_the_cart_on_the_map() -> None:
    """With graph_measurement off the graph's answer stays here; on, every graph that moves
    publishes where the GRAPH says the cart is — its correction composed with the EKF's
    odometry and read through the anchor — as one measurement on its own topic."""
    import json

    node = rtabmap_frame.RtabmapFrame()
    _belief(node, 1.0, 2.0)
    _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
    _odom(node, 0.2, 0.0)
    node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent, "off by default"

    with ros_stubs.parameters(graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent, "no belief to anchor to yet"
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        node._map_id = "flat3"  # what /map would have named itself (msgs.map_id)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        (sent,) = node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent
        word = json.loads(sent.data)
        # The anchor was learned from exactly this pair, so the first word IS the tracker's
        # pose: a graph that has closed no loop knows nothing the tracker does not.
        assert (word["x"], word["y"]) == (1.0, 2.0)
        assert word["source"] == "graph" and word["map"] == "flat3" and word["stamp"] == 7.0

        # the graph closes a loop and moves its odom frame 0.3 m along +x: the cart moves with it
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.3, 0.0).transform))
        word = json.loads(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1].data)
        assert (word["x"], word["y"]) == (1.3, 2.0)

        # a graph that has jumped further from the tracker than a closure ever does is refused
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(9.0, 0.0).transform))
        assert len(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent) == 2

        # ...and with no odometry to compose with (a silent bridge) there is no word at all
        node._lookup.buffer.transforms.clear()
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.1, 0.0).transform))
        assert len(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent) == 2


def test_in_slam_the_laptop_sends_the_correction_home_and_touches_no_tf() -> None:
    """RTAB-Map is the map: the correction IS map -> odom, and it belongs on the board, where
    the reflexes look it up. Nothing is broadcast here — /tf crosses the bridge one way only."""
    with ros_stubs.parameters(slam=True):
        node = rtabmap_frame.RtabmapFrame()
    assert node.frames == ("map", "odom") and node._tf is None
    _, on_graph = node.subs["/rtabmap/mapGraph"]
    on_graph(_MapGraph(_shift(0.3, 0.1).transform))
    node.timers[0][1]()
    (sent,) = node.pubs["/map_odom"].sent
    assert (sent.header.frame_id, sent.child_frame_id) == ("map", "odom")
    assert _xy(sent) == (0.3, 0.1), "as it stands, not inverted"
    assert "slam=on" in node.logger.texts("info")[0]


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


def test_the_anchor_waits_for_a_seating_the_lidar_pins_in_both_axes(tmp_path: Any) -> None:
    """A fit is not an error bar. Along a sofa the scan matches beautifully and slides in y, and
    an anchor learned off that seating carries the slide into every word the graph ever says —
    for the life of the file. So a soft seating is refused: the frame stays identity, nothing is
    published, the graphs are counted as pending, and the anchor waits for a scan the peak pins
    in BOTH axes. Nothing about the file changes but the moment it is written."""
    from pepin.anchors import anchor_path

    with ros_stubs.parameters(anchor_dir=str(tmp_path), graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0, sigma=(0.01, 0.40, 0.005))  # pinned in x, free along the sofa
        _fit(node, 0.75, at=0.0)  # ...and matching as well as it ever does
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        node.timers[0][1]()
        assert node._anchor is None and node._pending == 1
        assert _xy(node._tf.sent[-1]) == (0.0, 0.0), "identity until a seating is worth it"
        assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent, "and the graph says nothing"
        assert not anchor_path(tmp_path, "3x4@1.00,2.00").exists()

        # a soft heading is refused the same way: a cart that knows where it stands and not
        # which way it faces would rotate the whole graph about itself
        _belief(node, 1.0, 2.0, sigma=(0.01, 0.01, 0.05))
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert node._anchor is None and node._pending == 2

        _belief(node, 1.0, 2.0, sigma=(0.008, 0.012, 0.004))  # the cart reaches a corner
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        node.timers[0][1]()
    assert node._anchor is not None and node._pending == 2
    assert _xy(node._tf.sent[-1]) == (0.8, 2.0), "learned off the sharp seating, not the soft one"
    assert anchor_path(tmp_path, "3x4@1.00,2.00").exists(), "the same file, written later"
    node.timers[1][1]()
    report = node.logger.texts("info")[-1]
    assert "off a seating of 0.8/1.2 cm, 0.23 deg" in report, "the report says how sharp it was"


def test_a_lidar_that_is_not_driving_the_tracker_anchors_nothing(tmp_path: Any) -> None:
    """The other half of the gate, and the older half: the covariance of a belief the lidar is
    not behind describes nothing at all. With no fresh fit the anchor is pending, whatever the
    numbers on the belief say."""
    with ros_stubs.parameters(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.75, at=0.0)
        node.clock.seconds = rtabmap_frame.FIT_FRESH_S + 0.1  # the lidar went quiet
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert node._anchor is None and node._pending == 1
        assert "not driving the tracker" in node.logger.texts("info")[-1]
        _fit(node, 0.2, at=node.clock.seconds)  # ...and comes back matching nothing
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert node._anchor is None and node._pending == 2
    node.timers[1][1]()
    assert "not yet, pending 2" in node.logger.texts("info")[-1]


def test_the_sharpness_gate_can_be_opened_in_the_field(tmp_path: Any) -> None:
    """CLAUDE.md rule 19: the old behaviour stays reachable. With anchor_max_sigma_m wide open
    the anchor is learned off whatever seating the first graph finds, as before 2026-09-14 —
    which is what a room where no seating is ever sharp needs."""
    with ros_stubs.parameters(
        anchor_dir=str(tmp_path), anchor_max_sigma_m=1.0, anchor_max_sigma_deg=180.0
    ):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0, sigma=(0.01, 0.40, 0.05))
        _fit(node, 0.75, at=0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    assert node._anchor is not None and node._pending == 0
    assert (node._anchor.x, node._anchor.y) == (0.8, 2.0)


def test_the_anchor_is_written_beside_the_map_it_belongs_to(tmp_path: Any) -> None:
    """The anchor ties THIS graph database to THIS lidar map, so it is a property of the pair and
    not of a session: learned once from the tracker, it is written beside the map under the map's
    own identity, for the next session and the next wake-up to read."""
    from pepin.anchors import anchor_path, load_anchor

    with ros_stubs.parameters(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    stored = load_anchor(tmp_path, "3x4@1.00,2.00")
    assert stored is not None and (stored.pose.x, stored.pose.y) == (0.8, 2.0)
    assert anchor_path(tmp_path, "3x4@1.00,2.00").exists()
    node.timers[1][1]()
    assert "from learned" in node.logger.texts("info")[-1], "the report says where it came from"


def test_a_stored_anchor_lets_the_graph_speak_before_the_tracker_does(tmp_path: Any) -> None:
    """The wake-up: the cart comes off the charger with no lidar and nothing on /tracker_pose,
    and the graph recognising the place IS the answer — the anchor on file is what turns it into
    a place on the map. An anchor learned in this session could not do that: it came FROM a
    belief, and there is none."""
    import json

    from pepin.anchors import Anchor, save_anchor
    from pepin.odometry import Pose2D

    save_anchor(tmp_path, Anchor(Pose2D(0.8, 2.0, 0.0), "3x4@1.00,2.00", origin="learned"))
    with ros_stubs.parameters(anchor_dir=str(tmp_path), graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        assert node._anchor is not None, "adopted the moment the served map is known"
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    (sent,) = node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent
    word = json.loads(sent.data)
    assert (word["x"], word["y"]) == (1.0, 2.0), "no tracker said this: the graph did"
    assert word["map"] == "3x4@1.00,2.00"


def test_a_graph_that_recognises_nothing_says_nothing(tmp_path: Any) -> None:
    """The other half of the wake-up, and the reason it is safe: with no graph message there is
    no word at all. The tracker keeps its saved pose and the owner carries the cart to a place
    it knows."""
    with ros_stubs.parameters(anchor_dir=str(tmp_path), graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        node.timers[0][1]()
    assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent
    (identity,) = node._tf.sent
    assert _xy(identity) == (0.0, 0.0), "no anchor on file either: the tree keeps its hole"


def test_the_stored_anchor_is_re_learned_only_on_evidence_that_holds(tmp_path: Any) -> None:
    """A board restart moves the odom frame the graph rides and the stored anchor is then stale
    by exactly that reset. The evidence is narrow on purpose: the LIDAR driving with a good fit,
    disagreeing for five seconds. A momentary gap is a closure landing, and a gap with no lidar
    behind it is no evidence at all."""
    from pepin.anchors import load_anchor

    with ros_stubs.parameters(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert node._origin == "learned"

        # the tracker relocalises 2 m away and stays there: the graph's word no longer fits
        _belief(node, 3.0, 2.0, seconds=7.0)
        for moment in (0.0, 2.0, 4.0):  # ...but the lidar is silent, so nothing is re-learned
            node.clock.seconds = moment
            node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert node._origin == "learned" and node._relearns == 0

        _fit(node, 0.9, at=10.0)  # the lidar is driving now, and the gap holds
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        _fit(node, 0.9, at=14.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert node._relearns == 0, "four seconds is not five"
        _fit(node, 0.9, at=16.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))

    assert node._relearns == 1 and node._origin == "relearned"
    stored = load_anchor(tmp_path, "3x4@1.00,2.00")
    assert stored is not None and (stored.pose.x, stored.pose.y) == (2.8, 2.0), "rewritten"
    node.timers[1][1]()
    assert "from relearned 1" in node.logger.texts("info")[-1]


def test_the_re_learn_can_be_switched_off_in_the_field(tmp_path: Any) -> None:
    """CLAUDE.md rule 19: the old behaviour stays reachable. With anchor_relearn off the stored
    anchor is kept whatever happens and a disagreeing word is simply refused as too far."""
    with ros_stubs.parameters(anchor_dir=str(tmp_path), anchor_relearn=False):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        _belief(node, 3.0, 2.0, seconds=7.0)
        for moment in (0.0, 10.0, 20.0):
            _fit(node, 0.9, at=moment)
            node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    assert node._relearns == 0 and node._origin == "learned"


def test_the_word_the_fusion_cannot_use_goes_out_as_a_candidate(tmp_path: Any) -> None:
    """A measurement can only correct a pose that is already nearly right: this node refuses a
    word further than MAX_DISAGREEMENT_M and the board's filter gates what it does take. The
    word that undoes a CARRY is exactly that refused word, so it goes out on the whole-map
    candidate channel instead — the door the lidar's own search re-seeds through."""
    import json

    with ros_stubs.parameters(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _fit(node, 0.9, at=5.0)  # the lidar is driving and the board is talking to us
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert len(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent) == 1, "the word is a word"
        assert not node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent, "the tracker is already there"

        # the graph recognises the place and puts the cart two metres from the tracker's belief
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(2.0, 0.0).transform))
        assert len(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent) == 1, "refused as too far"
        (sent,) = node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent
        word = json.loads(sent.data)
        assert (word["x"], word["y"]) == (3.0, 2.0) and word["source"] == "graph"
        assert word["map"] == "3x4@1.00,2.00" and word["ambiguity"] == 0.0
        assert word["scan"] == 2, "the graph's own count: three graphs are three pieces of evidence"
        assert node._proposed == 1 and node._refused == 1

        # ...and with the board quiet for three seconds the tracker has no source behind its
        # pose, so a word about ANOTHER place travels as a candidate even within the gate (it is
        # a measurement too: the fusion may still use what it can)
        node.clock.seconds = 5.0 + rtabmap_frame.BELIEF_FRESH_S + 0.1
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.6, 0.0).transform))
        assert node._proposed == 2 and len(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent) == 2
        assert json.loads(node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent[-1].data)["x"] == 1.6
    node.timers[1][1]()
    assert "2 candidates sent" in node.logger.texts("info")[-1]


def test_a_word_the_tracker_can_confirm_asks_for_nothing(tmp_path: Any) -> None:
    """The other half of the rule, and what keeps the graph out of the lidar's way: a word about
    the place the tracker already holds would be called agreement by the board's gate, act on
    nothing, and end whatever streak the LIDAR's own search had built."""
    with ros_stubs.parameters(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the sharp seating the anchor is learned off
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        _fit(node, 0.0, at=0.0)  # ...and now nothing is confirming the tracker's pose
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.3, 0.0).transform))
        assert node._lost() and not node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent, "30 cm"
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.8, 0.0).transform))
        assert node._proposed == 1, "80 cm is another place, and nothing is confirming the pose"


def test_the_candidate_channel_can_be_switched_off(tmp_path: Any) -> None:
    """CLAUDE.md rule 19: the old behaviour stays reachable. Off, a word too far to fuse is
    counted here and reaches nothing — which is what a stale anchor calls for, since three
    words off by one constant offset agree with each other perfectly."""
    with ros_stubs.parameters(anchor_dir=str(tmp_path), graph_candidates=False):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(2.0, 0.0).transform))
    assert not node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent and node._proposed == 0
    assert node._refused == 1, "refused as before, and now it is the end of the road again"


def test_the_word_s_fit_is_what_the_graph_has_recognised() -> None:
    """The hole the carry test of 2026-09-14 found: the graph recognised nothing for 64 s, its
    word was the old anchor plus odometry, and it still claimed fit 1.00 — so the board
    published 1.00 as its own confidence and goto drove on a belief 1.5-2 m wrong. The word now
    carries what the GRAPH knows: 1.0 at a tie to an older node, decaying with the metres driven
    since (pepin.graphtrust), and the whole-map candidate floor and the lost ladder act on it."""
    import json

    from pepin.graphtrust import FILE_ANCHOR_TRUST
    from pepin.watch import ADMIT_FIT

    def word(node: Any) -> dict[str, Any]:
        sent = node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1]
        return dict(json.loads(sent.data))

    with ros_stubs.parameters(graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        node._map_id = "flat3"
        _belief(node, 1.0, 2.0)
        _odom(node, 0.2, 0.0)
        _info(node, 0.0, hypothesis=0.04)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert word(node)["fit"] == 1.0, "the anchor was just learned from the tracker"

        # ...and now the carry: the graph recognises nothing while the cart drives on
        _info(node, 4.0, hypothesis=0.04)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert word(node)["fit"] < ADMIT_FIT, "no candidate is admitted on this any more"
        _info(node, 6.1, hypothesis=0.04)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert word(node)["fit"] < FILE_ANCHOR_TRUST

        # ...and the closure that finds the place puts the word back where it belongs
        _info(node, 6.1, loop=41.0, hypothesis=0.81)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert word(node)["fit"] == 1.0
        assert "graph tie 1:" in node.logger.texts("info")[-1]

        node.timers[1][1]()
        report = node.logger.texts("info")[-1]
        assert "graph 1.00 trust, 0.0 m since tie 1, hypothesis 0.81 over 4 infos" in report
        assert "graph_trust=on" in report


def test_a_wake_up_word_is_a_guess_until_the_graph_recognises_something(tmp_path: Any) -> None:
    """An anchor read from a file was measured in ANOTHER session and a board restart moves the
    odom frame under it: the word it makes is capped, so it can be fused as a measurement but
    can neither re-seed the pose nor make a lost tracker look found."""
    import json

    from pepin.anchors import Anchor, save_anchor
    from pepin.graphtrust import FILE_ANCHOR_TRUST
    from pepin.odometry import Pose2D

    save_anchor(tmp_path, Anchor(Pose2D(0.8, 2.0, 0.0), "3x4@1.00,2.00", origin="learned"))
    with ros_stubs.parameters(anchor_dir=str(tmp_path), graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _info(node, 12.0, hypothesis=0.04)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        word = json.loads(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1].data)
        assert word["fit"] == FILE_ANCHOR_TRUST, "nothing has confirmed this frame yet"

        _info(node, 12.0, loop=7.0, hypothesis=0.74)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        word = json.loads(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1].data)
        assert word["fit"] == 1.0, "the graph recognised the place: the cap is gone"


def test_with_graph_trust_off_every_word_claims_what_it_claimed_before() -> None:
    """The old behaviour, one live flag away: the switch is what a regression is turned off
    with in the field, and it is what an A/B on one drive compares."""
    import json

    with ros_stubs.parameters(graph_measurement=True, graph_trust=False):
        node = rtabmap_frame.RtabmapFrame()
        node._map_id = "flat3"
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief: the anchor may be learned
        _odom(node, 0.2, 0.0)
        _info(node, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        _info(node, 30.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    word = json.loads(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1].data)
    assert word["fit"] == 1.0, "30 m past anything recognised, and still claiming everything"
