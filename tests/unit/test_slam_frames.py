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
    """rtabmap_msgs/MapGraph as these nodes read it: the correction, the stamp the laptop's node
    reads the cart's odometry at, and the LIVE graph's own poses of its nodes (``poses_id`` /
    ``poses``) — the only thing the node ever takes from RTAB-Map's global frame, and it takes
    them only to subtract two of them from each other."""

    def __init__(
        self,
        map_to_odom: Any,
        stamp: Any = None,
        nodes: dict[int, tuple[float, float, float]] | None = None,
    ) -> None:
        self.map_to_odom = map_to_odom
        self.header = types.SimpleNamespace(stamp=stamp if stamp is not None else _stamp(0.0))
        self.poses_id = list(nodes or {})
        self.poses = [_pose(*(nodes or {})[node_id]) for node_id in self.poses_id]


class _Info:
    """rtabmap_msgs/Info as the laptop's node reads it: RTAB-Map's statistics table (the keys
    and the values in two parallel arrays) and the graph ids of the message — the node it is
    building now and the node a closure or a proximity link matched, which is what tells a tie
    to the loaded database from a tie inside this start's own segment."""

    def __init__(
        self,
        stats: dict[str, float],
        ref_id: int = 0,
        loop_closure_id: int = 0,
        proximity_detection_id: int = 0,
        stamp: float = 7.0,
    ) -> None:
        self.stats_keys = list(stats)
        self.stats_values = [float(value) for value in stats.values()]
        self.ref_id = ref_id
        self.loop_closure_id = loop_closure_id
        self.proximity_detection_id = proximity_detection_id
        # The stamp is what pairs this message with the localisation of the SAME update, which is
        # how a word learns which database node it was made against.
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


def _pose(x: float, y: float, yaw: float = 0.0) -> Any:
    """A geometry_msgs/Pose in the plane, as a graph carries one of its nodes."""
    from geometry_msgs.msg import Pose

    pose = Pose()
    pose.position.x, pose.position.y = x, y
    pose.orientation.z, pose.orientation.w = math.sin(yaw / 2.0), math.cos(yaw / 2.0)
    return pose


def _odom(node: Any, x: float, y: float, seconds: float = 7.0) -> None:
    """The EKF's odom -> base_link, as the laptop's node looks it up at a graph's stamp."""
    pose = RigidPose(np.eye(3), np.array([x, y, 0.0]))
    transform = transform_from_pose("odom", "base_link", pose, _stamp(seconds))
    node._lookup.buffer.transforms[("odom", "base_link")] = transform


def _shift(x: float, y: float) -> Any:
    """A correction of x, y metres with no rotation, as a stamped transform."""
    pose = RigidPose(np.eye(3), np.array([x, y, 0.0]))
    return transform_from_pose("a", "b", pose, ros_stubs.Time())


def _turn(yaw: float, x: float = 0.0, y: float = 0.0) -> Any:
    """A correction that TURNS the graph's frame by ``yaw`` radians (and shifts it by x, y), as a
    stamped transform: what a closure does when it decides the cart was facing another way."""
    cos, sin = math.cos(yaw), math.sin(yaw)
    rotation = np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])
    return transform_from_pose(
        "a", "b", RigidPose(rotation, np.array([x, y, 0.0])), ros_stubs.Time()
    )


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


def _info(
    node: Any,
    travelled: float,
    loop: float = 0.0,
    hypothesis: float = 0.0,
    ref: int = 0,
) -> None:
    """One /rtabmap/info into the node: how far RTAB-Map's odometry has travelled, whether this
    message tied the present to an older node and which one, how close the last hypothesis came,
    and which node RTAB-Map is building now (``ref``: everything below this start's first is a
    node of the database it loaded)."""
    node.subs[rtabmap_frame.INFO_TOPIC][1](
        _Info(
            {
                "Memory/Distance_travelled/m": travelled,
                "Loop/Id/": loop,
                "Loop/Highest_hypothesis_value/": hypothesis,
            },
            ref_id=ref,
            loop_closure_id=int(loop),
        )
    )


def _recognise(node: Any, travelled: float = 0.0, ref: int = 2000) -> None:
    """RTAB-Map matching a node of the database it LOADED (id 41, far below the 2000 this start
    began at): the predicate every word rides on (graph_words_need_recognition)."""
    _info(node, travelled, loop=41.0, hypothesis=0.8, ref=ref)


def _match(node: Any, matched: int, ref: int = 2000, stamp: float = 7.0) -> None:
    """One /rtabmap/info of an update that LOCALISED: ``matched`` is the database node the closure
    landed on (the one a word is hung on), ``ref`` the node RTAB-Map is building now (the one this
    map measures and tables), and the stamp is what pairs it with the localisation."""
    node.subs[rtabmap_frame.INFO_TOPIC][1](
        _Info(
            {"Memory/Distance_travelled/m": 0.0, "Loop/Id/": float(matched)},
            ref_id=ref,
            loop_closure_id=matched,
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


def _old_source(**overrides: Any) -> Any:
    """The parameter block the tests of the CORRECTION-based word source open.

    Two defaults changed on 2026-09-18: the word is RTAB-Map's own localisation in the loaded
    database (``graph_word_from_localization``) and it is hung on the NODE it recognised
    (``graph_word_from_nodes``). The tests of the older arrangement — the session's correction
    composed with odom -> base_link, read through one global tie — say so, which is also what keeps
    that arrangement reachable (CLAUDE.md rule 19). The localisation path and the node table have
    tests of their own below.
    """
    return ros_stubs.parameters(
        graph_word_from_localization=False, graph_word_from_nodes=False, **overrides
    )


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


def test_on_a_known_map_the_laptop_broadcasts_the_tie_it_measured() -> None:
    """RTAB-Map runs on the EKF's odometry, so its graph starts at the odom frame's origin: the
    edge our tree needs is where that frame sits on the map, MEASURED from the lidar-held tracker
    one pair at a time. Identity until it is known — a tree with a hole in it is worse than one
    edge that is not yet right."""
    with _old_source():
        node = rtabmap_frame.RtabmapFrame()
    assert node.frames == ("map", "rtabmap")
    assert rtabmap_frame.CORRECTION_TOPIC not in node.pubs, "the board owns nothing here"
    node.timers[0][1]()
    (identity,) = node._tf.sent
    assert _xy(identity) == (0.0, 0.0), "no graph, no tie"

    _belief(node, 1.0, 2.0)
    _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
    _odom(node, 0.2, 0.0)
    node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    node.timers[0][1]()
    sent = node._tf.sent[-1]
    assert (sent.header.frame_id, sent.child_frame_id) == ("map", "rtabmap")
    assert _xy(sent) == (0.8, 2.0), "the tracker at (1, 2), the graph's cart at (0.2, 0)"

    # ...and a second pair of the same pose is the same tie: the frame does not follow the word
    node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.1, 0.0).transform))
    node.timers[0][1]()
    assert _xy(node._tf.sent[-1]) == (0.8, 2.0)


def test_with_graph_odom_off_the_laptop_broadcasts_the_inverse_correction() -> None:
    """The arrangement before 2026-09-14, one launch argument away: RTAB-Map's odometry is the
    tracker's pose, its correction reads rtabmap -> map, and the edge our tree needs is the
    other way round."""
    with _old_source(graph_odom=False):
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

    with _old_source():
        node = rtabmap_frame.RtabmapFrame()
    _belief(node, 1.0, 2.0)
    _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
    _odom(node, 0.2, 0.0)
    node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent, "off by default"

    with _old_source(graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _recognise(node)  # RTAB-Map knows where on the database it is: the words may travel
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
    with _old_source(slam=True):
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


def test_no_pair_is_taken_off_a_seating_the_lidar_pins_in_one_axis_only(tmp_path: Any) -> None:
    """A fit is not an error bar. Along a sofa the scan matches beautifully and slides in y, and a
    tie measured off that seating carries the slide into every word the graph says. So a soft
    seating becomes no pair at all: the frame stays identity, nothing is published, the graphs are
    counted as pending, and the calibration waits for a scan the peak pins in BOTH axes."""
    from pepin.graphtie import pairs_path

    with _old_source(anchor_dir=str(tmp_path), graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0, sigma=(0.01, 0.40, 0.005))  # pinned in x, free along the sofa
        _fit(node, 0.75, at=0.0)  # ...and matching as well as it ever does
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        node.timers[0][1]()
        assert node._tie is None and node._pending == 1
        assert _xy(node._tf.sent[-1]) == (0.0, 0.0), "identity until a seating is worth it"
        assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent, "and the graph says nothing"
        assert not pairs_path(tmp_path, "3x4@1.00,2.00").exists()

        # a soft heading is refused the same way: a cart that knows where it stands and not
        # which way it faces would rotate the whole graph about itself
        _belief(node, 1.0, 2.0, sigma=(0.01, 0.01, 0.05))
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert node._tie is None and node._pending == 2

        _belief(node, 1.0, 2.0, sigma=(0.008, 0.012, 0.004))  # the cart reaches a corner
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        node.timers[0][1]()
    assert node._tie is not None and node._pending == 2 and node._pairs_taken == 1
    assert _xy(node._tf.sent[-1]) == (0.8, 2.0), "measured off the sharp seating, not the soft one"
    assert pairs_path(tmp_path, "3x4@1.00,2.00").exists(), "and the pair is on the log"
    node.timers[1][1]()
    report = node.logger.texts("info")[-1]
    assert "last pair off a seating of 0.8/1.2 cm, 0.23 deg" in report, "how sharp it was"
    assert "from pairs 1/1 inliers" in report


def test_a_lidar_that_is_not_driving_the_tracker_measures_no_tie(tmp_path: Any) -> None:
    """The other half of the gate, and the older half: the covariance of a belief the lidar is
    not behind describes nothing at all. With no fresh fit no pair is taken, whatever the
    numbers on the belief say — which is also why camera-only never re-measures the tie."""
    with _old_source(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.75, at=0.0)
        node.clock.seconds = rtabmap_frame.FIT_FRESH_S + 0.1  # the lidar went quiet
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert node._tie is None and node._pending == 1
        assert "not driving the tracker" in node.logger.texts("info")[-1]
        _fit(node, 0.2, at=node.clock.seconds)  # ...and comes back matching nothing
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert node._tie is None and node._pending == 2
    node.timers[1][1]()
    assert "not yet, pending 2" in node.logger.texts("info")[-1]


def test_the_sharpness_gate_can_be_opened_in_the_field(tmp_path: Any) -> None:
    """CLAUDE.md rule 19: the old behaviour stays reachable. With anchor_max_sigma_m wide open a
    pair is taken off whatever seating the first graph finds, as before 2026-09-14 — which is what
    a room where no seating is ever sharp needs."""
    with _old_source(anchor_dir=str(tmp_path), anchor_max_sigma_m=1.0, anchor_max_sigma_deg=180.0):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0, sigma=(0.01, 0.40, 0.05))
        _fit(node, 0.75, at=0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    assert node._tie is not None and node._pending == 0
    assert (node._tie.pose.x, node._tie.pose.y) == (0.8, 2.0)


def test_the_pairs_are_logged_beside_the_map_they_belong_to(tmp_path: Any) -> None:
    """The tie ties THIS graph database to THIS lidar map, so its calibration is a property of the
    pair and not of a session: every pair is appended beside the map under the map's own identity,
    for the next session and the next wake-up to fit over."""
    from pepin.graphtie import load_pairs, pairs_path

    with _old_source(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        # ...the cart drives a metre and the graph's place moves with it: a second, distant pair
        _odom(node, 1.2, 0.0)
        _belief(node, 2.0, 2.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    stored = load_pairs(tmp_path, "3x4@1.00,2.00")
    assert len(stored) == 2 and pairs_path(tmp_path, "3x4@1.00,2.00").exists()
    assert (stored[0].cart.x, stored[0].cart.y) == (1.0, 2.0)
    assert (stored[1].place.x, stored[1].place.y) == (1.2, 0.0)
    assert node._tie is not None and node._tie.pose.x == pytest.approx(0.8)
    assert node._tie.extent_m == pytest.approx(0.5), "and the spread the heading now rides on"
    node.timers[1][1]()
    assert "from pairs 2/2 inliers" in node.logger.texts("info")[-1], "and the report says so"


def test_a_tie_on_disk_lets_the_graph_speak_before_the_tracker_does(tmp_path: Any) -> None:
    """The wake-up: the cart comes off the charger with no lidar and nothing on /tracker_pose,
    and the graph recognising the place IS the answer — the tie on disk is what turns it into
    a place on the map. A tie measured in this session could not do that: it would have come FROM
    a belief, and there is none."""
    import json

    from pepin.anchors import Anchor, save_anchor
    from pepin.odometry import Pose2D

    save_anchor(tmp_path, Anchor(Pose2D(0.8, 2.0, 0.0), "3x4@1.00,2.00", origin="learned"))
    with _old_source(anchor_dir=str(tmp_path), graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        assert node._tie is not None, "adopted the moment the served map is known"
        assert node._tie.origin == "file", "and it knows it is only a one-seating guess"
        _recognise(node)  # ...and the camera has found a place the database knows
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    (sent,) = node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent
    word = json.loads(sent.data)
    assert (word["x"], word["y"]) == (1.0, 2.0), "no tracker said this: the graph did"
    assert word["map"] == "3x4@1.00,2.00"
    from pepin.graphtie import FILE_TIE_SIGMA_M

    assert math.sqrt(word["covariance"][0][0]) > FILE_TIE_SIGMA_M, "and it is honestly wide"


def test_camera_only_uses_the_tie_on_disk_and_never_touches_it(tmp_path: Any) -> None:
    """The lidar independence, which is the point of the whole change. With the lidar muted the
    tracker's pose is held by these very words, so nothing about the tie may be measured from it —
    no pair, no refit, no rewrite. The calibration on disk is a constant of the pair and is simply
    read, exactly as camera extrinsics are."""
    from pepin.graphtie import TiePair, append_pair, load_pairs
    from pepin.measurements import compose, inverse
    from pepin.odometry import Pose2D

    truth = Pose2D(0.8, 2.0, 0.0)
    for x, y in ((0.0, 0.0), (1.5, 0.0), (0.0, 1.5), (1.5, 1.5)):
        cart = Pose2D(x, y, 0.0)
        append_pair(
            tmp_path,
            "3x4@1.00,2.00",
            TiePair(stamp=0.0, cart=cart, place=compose(inverse(truth), cart)),
        )
    with _old_source(anchor_dir=str(tmp_path), graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _recognise(node)
        measured = node._tie
        assert measured is not None and measured.pose.x == pytest.approx(0.8)
        _fit(node, 0.0, at=0.0)  # camera-only: /localization_fit is 0.00 by construction
        for x in (0.2, 0.9, 1.6):  # ...and the cart drives right across the flat
            _odom(node, x, 0.0)
            _belief(node, x + 0.8, 2.0)
            node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    assert node._tie is measured, "the tie on disk is what the words rode, unchanged"
    assert node._pairs_taken == 0 and len(load_pairs(tmp_path, "3x4@1.00,2.00")) == 4
    assert len(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent) == 3, "and every word travelled"
    assert "not driving the tracker" in " ".join(node.logger.texts("info"))


def test_a_log_of_pairs_is_fitted_at_start_and_beats_the_one_seating_file(tmp_path: Any) -> None:
    """The calibration outlives the session. A log written over past drives is read and fitted the
    moment the served map names it, and the fit is what the words ride — the one-seating file is
    only the fallback for a pair that has never been measured."""
    from pepin.anchors import Anchor, save_anchor
    from pepin.graphtie import TiePair, append_pair
    from pepin.measurements import compose, inverse
    from pepin.odometry import Pose2D

    truth = Pose2D(0.8, 2.0, 0.0)
    save_anchor(tmp_path, Anchor(Pose2D(5.0, -5.0, 1.0), "3x4@1.00,2.00", origin="learned"))
    for x, y in ((0.0, 0.0), (1.5, 0.0), (0.0, 1.5), (1.5, 1.5)):
        cart = Pose2D(x, y, 0.0)
        append_pair(
            tmp_path,
            "3x4@1.00,2.00",
            TiePair(stamp=0.0, cart=cart, place=compose(inverse(truth), cart)),
        )
    with _old_source(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
    assert node._tie is not None and node._tie.origin == "pairs"
    assert node._tie.pose.x == pytest.approx(truth.x) and node._tie.inliers == 4
    assert node._tie.extent_m > 1.0, "the spread over the flat is what the heading rides on"
    assert "graph tie read:" in " ".join(node.logger.texts("info"))


def test_a_frame_born_with_the_map_is_identity_and_needs_no_pair() -> None:
    """Waking up in an unknown room, this session creates the map AND the database at the same pose
    in the same second: the two frames are one by construction, so the tie is identity with no
    uncertainty and nothing is measured. The words ride from the first graph, with no lidar."""
    import json

    with _old_source(fresh_frame=True, graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        assert node._tie is not None and node._tie.origin == "identity"
        _map(node)
        _recognise(node)
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.3, 0.0).transform))
        node.timers[0][1]()
    assert _xy(node._tf.sent[-1]) == (0.0, 0.0), "identity, and not because nothing is known"
    (sent,) = node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent
    word = json.loads(sent.data)
    assert (word["x"], word["y"]) == (0.5, 0.0), "the graph's own place, read as it stands"
    assert node._pairs_taken == 0 and node._pending == 0
    node.timers[1][1]()
    assert "identity (a frame born with this map)" in node.logger.texts("info")[-1]


def test_a_graph_that_recognises_nothing_says_nothing(tmp_path: Any) -> None:
    """The other half of the wake-up, and the reason it is safe: with no graph message there is
    no word at all. The tracker keeps its saved pose and the owner carries the cart to a place
    it knows."""
    with _old_source(anchor_dir=str(tmp_path), graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        node.timers[0][1]()
    assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent
    (identity,) = node._tf.sent
    assert _xy(identity) == (0.0, 0.0), "no anchor on file either: the tree keeps its hole"


def test_a_disagreement_never_moves_the_tie_however_long_it_holds(tmp_path: Any) -> None:
    """The disease, gone. Until 2026-09-17 a word disagreeing with a lidar-held tracker for five
    seconds rewrote the whole tie — fitting a constant to a variable, which is how the stored
    anchor took three values metres apart in one evening and how a single false recognition became
    every later word's error. Now a disagreement is just a refused word: the tie is only ever
    replaced by a better FIT, and a pair the gate throws out changes nothing."""
    with _old_source(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _recognise(node)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert node._tie is not None and node._tie.pose.x == pytest.approx(0.8)
        measured = node._tie.pose

        # the tracker relocalises 2 m away on a good lidar fit and stays there for half a minute:
        # every piece of evidence the old re-learn asked for, several times over
        _belief(node, 3.0, 2.0, seconds=7.0)
        for moment in (0.0, 4.0, 10.0, 16.0, 30.0):
            _fit(node, 0.9, at=moment)
            node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert node._tie is not None and node._tie.pose == measured, (
            "the tie did not move: thirty seconds of disagreement is not a measurement"
        )
        assert node._pairs_taken == 2, "the new place did become a pair, as any new place does"
        assert node._refused >= 3, "...and its words are refused, which is all a disagreement is"
    assert node._tie.inliers == 1 and node._tie.outlier_share == pytest.approx(0.5), (
        "a pair claiming the cart moved 2 m while the database says it stood still is a"
        " contradiction, and the fit reports it as the outlier it is"
    )


def test_a_refit_replaces_the_tie_only_when_it_is_the_surer_one(tmp_path: Any) -> None:
    """A calibration is replaced by a better measurement and by nothing else. A log of four pairs
    spread over the flat is surer than one pair in a corner, so it wins; the single pair that
    follows never takes the tie back, however loudly it disagrees."""
    from pepin.graphtie import TiePair, fit_tie
    from pepin.measurements import compose, inverse
    from pepin.odometry import Pose2D

    truth = Pose2D(0.8, 2.0, 0.0)
    spread = [
        TiePair(
            stamp=0.0,
            cart=Pose2D(x, y, 0.0),
            place=compose(inverse(truth), Pose2D(x, y, 0.0)),
        )
        for x in (0.0, 1.5)
        for y in (0.0, 1.5)
    ]
    with _old_source(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        assert node._replace(fit_tie(spread), "four spread pairs") is True
        better = node._tie
        assert better is not None
        assert node._replace(fit_tie(spread[:1]), "one pair in a corner") is False
        assert node._tie is better, "a wider fit is not an argument"


def test_the_one_seating_anchor_is_still_reachable_as_the_old_behaviour(tmp_path: Any) -> None:
    """CLAUDE.md rule 19: the old behaviour stays reachable. With graph_tie_from_pairs off the tie
    is learned from the FIRST sharp seating and written to the one-seating anchor file, as before
    2026-09-18 — and nothing is logged, nothing refitted, and nothing re-learned either."""
    from pepin.anchors import load_anchor
    from pepin.graphtie import pairs_path

    with _old_source(anchor_dir=str(tmp_path), graph_tie_from_pairs=False):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _recognise(node)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert node._tie is not None and node._tie.origin == "file"
        _belief(node, 3.0, 2.0, seconds=7.0)
        for moment in (0.0, 10.0, 20.0):
            _fit(node, 0.9, at=moment)
            node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
    assert node._pairs_taken == 0 and not pairs_path(tmp_path, "3x4@1.00,2.00").exists()
    assert node._tie is not None and node._tie.pose.x == pytest.approx(0.8), "learned once, kept"
    stored = load_anchor(tmp_path, "3x4@1.00,2.00")
    assert stored is not None and (stored.pose.x, stored.pose.y) == (0.8, 2.0)
    assert node._refused > 0, "and the disagreeing words are simply refused"


def test_the_word_the_fusion_cannot_use_goes_out_as_a_candidate(tmp_path: Any) -> None:
    """A measurement can only correct a pose that is already nearly right: this node refuses a
    word further than MAX_DISAGREEMENT_M and the board's filter gates what it does take. The
    word that undoes a CARRY is exactly that refused word, so it goes out on the whole-map
    candidate channel instead — the door the lidar's own search re-seeds through."""
    import json

    with _old_source(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _recognise(node)
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
    with _old_source(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _recognise(node)
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
    with _old_source(anchor_dir=str(tmp_path), graph_candidates=False):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _recognise(node)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(2.0, 0.0).transform))
    assert not node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent and node._proposed == 0
    assert node._refused == 1, "refused as before, and now it is the end of the road again"


def test_the_word_s_fit_can_be_the_distance_since_the_graph_s_last_tie() -> None:
    """The hole the carry test of 2026-09-14 found: the graph recognised nothing for 64 s, its
    word was the old anchor plus odometry, and it still claimed fit 1.00 — so the board
    published 1.00 as its own confidence and goto drove on a belief 1.5-2 m wrong. The answer of
    that day, still one flag away (graph_trust=distance): 1.0 at a tie to an older node, decaying
    with the metres driven since (pepin.graphtrust), and the whole-map candidate floor and the
    lost ladder act on it."""
    import json

    from pepin.graphtrust import FILE_ANCHOR_TRUST
    from pepin.watch import ADMIT_FIT

    def word(node: Any) -> dict[str, Any]:
        sent = node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1]
        return dict(json.loads(sent.data))

    with _old_source(
        graph_measurement=True, graph_trust="distance", graph_words_need_recognition=False
    ):
        node = rtabmap_frame.RtabmapFrame()
        node._map_id = "flat3"
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # a lidar behind the belief: the anchor may be learned
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
        assert "graph_trust=distance" in report


def test_a_wake_up_word_is_a_guess_until_the_graph_recognises_something(tmp_path: Any) -> None:
    """An anchor read from a file was measured in ANOTHER session and a board restart moves the
    odom frame under it: the word it makes is capped, so it can be fused as a measurement but
    can neither re-seed the pose nor make a lost tracker look found."""
    import json

    from pepin.anchors import Anchor, save_anchor
    from pepin.graphtrust import FILE_ANCHOR_TRUST
    from pepin.odometry import Pose2D

    save_anchor(tmp_path, Anchor(Pose2D(0.8, 2.0, 0.0), "3x4@1.00,2.00", origin="learned"))
    with _old_source(
        anchor_dir=str(tmp_path), graph_measurement=True, graph_words_need_recognition=False
    ):
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


def test_with_graph_trust_flat_every_word_claims_what_it_claimed_before() -> None:
    """The oldest behaviour, one live flag away: the switch is what a regression is turned off
    with in the field, and it is what an A/B on one drive compares."""
    import json

    with _old_source(
        graph_measurement=True, graph_trust="flat", graph_words_need_recognition=False
    ):
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


def test_a_graph_that_has_not_found_its_database_says_nothing_at_all(tmp_path: Any) -> None:
    """Tape 0333, 2026-09-15: RTAB-Map had restarted many times and had recognised nothing
    against the database it loaded since its last start, so the nodes it was building formed an
    unlinked segment placed by this session's odometry — and a closure INSIDE that segment is not
    a place found, it is the same odometry twice. The tracker took 19 such words and the pose
    flew. Nothing rides now until a node of the LOADED database is matched: no measurement, no
    candidate, and the words are counted so the report line says why the node is quiet."""
    with _old_source(anchor_dir=str(tmp_path), graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        _info(node, 0.0, ref=2000)  # the first node of this start
        _info(node, 0.5, loop=2011.0, ref=2040)  # ...and a closure to another of its own
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent and node._withheld == 1
        assert node._tie is not None, "the tie is still measured off a sharp lidar seating"
        assert node._word is not None, "the word is remembered: the report line is the judge"

        # ...and the word a carried cart would be recovered by is withheld too: an unlinked
        # segment is the one case where the graph's disagreement means nothing
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(2.0, 0.0).transform))
        assert not node.pubs[rtabmap_frame.CANDIDATE_TOPIC].sent and node._withheld == 2
        node.timers[1][1]()
        assert "unrecognised since start: 2 words withheld" in node.logger.texts("info")[-1]

        _recognise(node, travelled=0.5)  # the camera finds a place the database knows
        assert any("graph recognised" in text for text in node.logger.texts("info"))
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert len(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent) == 1, "and the words travel"
        assert node._withheld == 2


def test_distance_since_the_tie_weakens_the_word_and_never_withholds_it(tmp_path: Any) -> None:
    """A tie fixes the frame; the drift the odometry adds after it is a COVARIANCE, not a veto.

    This used to be a clock: a recognition expired after 120 s of "driving", and the clock
    charged whenever RTAB-Map's distance counter grew — which it does from VO jitter on a parked
    cart. It reached 991 s at a bookshelf and withheld 587 words while the graph's own word sat
    4 cm from the tracker, and the robot could not start a camera-only drive for three hours
    (2026-09-17). Now every word rides; the far ones just arrive wide, and the drive gate refuses
    them on the number instead of never hearing them at all.
    """
    import json

    def sigma(node: Any) -> float:
        sent = node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1]
        return math.sqrt(float(json.loads(sent.data)["covariance"][0][0]))

    with _old_source(anchor_dir=str(tmp_path), graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        _recognise(node)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert len(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent) == 1
        at_the_tie = sigma(node)

        node.clock.seconds = 3600.0  # an hour parked: the counter does not move
        _info(node, 0.0, ref=2000)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert len(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent) == 2, "standing still is free"
        assert sigma(node) == pytest.approx(at_the_tie), "no travel, no decay: the word is as good"

        node.clock.seconds = 3800.0  # ...and then 12 m in which the counter grew
        _info(node, 12.0, ref=2000)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert len(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent) == 3, "still said, just weaker"
        assert node._withheld == 0, "nothing is ever withheld once the frame is tied"
        assert sigma(node) > at_the_tie, "12 m of odometry since the tie is in the covariance"
        node.timers[1][1]()
        assert "recognised on node 41" in node.logger.texts("info")[-1]


def test_the_word_s_fit_is_how_well_the_last_words_track_the_odometry() -> None:
    """The default trust, and what distance-since-the-last-tie could not see: a word riding a
    frame that no longer holds is a metre out from the moment it is said, whatever the metres
    since the last closure say. The fit is exp(-rms residual / 10 cm) over the last words against
    the tracker's pose carried forward by odometry between them — 1.0 while the word follows the
    cart, under the whole-map candidate floor the moment it stops."""
    import json

    from pepin.watch import ADMIT_FIT

    def fit(node: Any) -> float:
        sent = node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1]
        return float(json.loads(sent.data)["fit"])

    with _old_source(graph_measurement=True):  # graph_trust=agreement, the default
        node = rtabmap_frame.RtabmapFrame()
        node._map_id = "flat3"
        _recognise(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert fit(node) == 1.0, "the anchor was just learned from this very pair"

        # the cart drives half a metre and the graph's word drives with it
        _odom(node, 0.7, 0.0)
        _belief(node, 1.5, 2.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert fit(node) == 1.0, "the word is where the odometry says the cart is"

        # ...and now the graph's frame slips 30 cm under a cart that has not moved
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.3, 0.0).transform))
        assert fit(node) < ADMIT_FIT, "no candidate is admitted on this any more"
        node.timers[1][1]()
        report = node.logger.texts("info")[-1]
        assert "word fit 0.28 by agreement (residual 17.3 cm / 0.61 sigmas over 3 words)" in report
        assert "recognised on node 41 (1 matches)" in report


def test_a_word_at_the_right_place_facing_the_wrong_way_is_refused(tmp_path: Any) -> None:
    """The gate the old node did not have. A word 90 degrees out at exactly the place the tracker
    holds passed the metres-only test and went into the fusion; the test is now the 3-DOF
    Mahalanobis distance the board's own filter applies (pepin.fusion.GATE), so the heading counts
    — and the refused word still goes out on the candidate channel, where a carried cart needs
    it."""
    with _old_source(anchor_dir=str(tmp_path), graph_measurement=True):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _recognise(node)
        _fit(node, 0.9, at=5.0)  # the board is talking to us and the lidar holds the pose
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)
        _odom(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        assert len(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent) == 1
        assert node._refused == 0 and node._gap_m == pytest.approx(0.0)

        # the graph turns its frame 90 degrees under a cart that has not moved: the word lands
        # within a centimetre of the tracker and faces the wrong way
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_turn(math.pi / 2, 0.2, -0.2).transform))
        assert node._gap_m < 0.01, "the same place by metres, which is what used to be asked"
        assert node._gap_sigmas**2 > rtabmap_frame.GATE, "and another pose by the gate"
        assert node._refused == 1, "refused as a measurement"
        assert len(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent) == 1
    node.timers[1][1]()
    assert "1 refused by the gate" in node.logger.texts("info")[-1]


def _tabled(directory: Any, room: str, entries: dict[int, tuple[float, float, float]]) -> None:
    """A node table on disk: for each database node, where OUR map says the cart was when it was
    created. Session 0 throughout, which is what makes them one rigid piece."""
    from pepin.graphnodes import NodePose, write_nodes
    from pepin.odometry import Pose2D

    write_nodes(
        directory,
        room,
        [
            NodePose(node_id=node_id, session=0, stamp=7.0, cart=Pose2D(*pose))
            for node_id, pose in entries.items()
        ],
    )


def test_the_word_is_hung_on_the_node_rtabmap_recognised(tmp_path: Any) -> None:
    """The whole second stage. ONE global tie cannot serve this database — its pieces are rigid to
    3.6-5.9 cm each but sit 1.6 m and 129 deg apart, and the live frame moves whenever the graph is
    re-optimised — so the word is Table[N] . inverse(P_N) . P_current, a RELATIVE pose inside one
    piece. Whatever the optimiser did to the global frame cancels: here the whole live graph is
    offset by (+50, +50) m and turned 90 degrees, and the word does not move a millimetre."""
    import json

    _tabled(tmp_path, "flat3_straight", {41: (1.0, 2.0, 0.0), 44: (2.0, 2.0, 0.0)})
    with ros_stubs.parameters(
        anchor_dir=str(tmp_path), room="flat3_straight", graph_measurement=True
    ):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        assert len(node._table) == 2, "read the moment the room is known"
        _odom(node, 0.2, 0.0)
        _belief(node, 1.6, 2.0)
        _match(node, matched=41)
        # the graph's own frame, as this optimisation happens to place it: node 41 at (10, 10) and
        # the cart 0.6 m along the graph's +x from it
        _localize(node, 10.6, 10.0)
        node.subs["/rtabmap/mapGraph"][1](
            _MapGraph(
                _shift(0.0, 0.0).transform, nodes={41: (10.0, 10.0, 0.0), 44: (11.0, 10.0, 0.0)}
            )
        )
        word = json.loads(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1].data)
        assert (round(word["x"], 6), round(word["y"], 6)) == (1.6, 2.0)
        assert node._node_words == 1 and node._substituted == 0

        # ...and now the optimiser re-roots the whole graph: 50 m away and 90 degrees round
        _localize(node, 60.0, 60.6, math.pi / 2, stamp=8.0)
        _match(node, matched=41, stamp=8.0)
        node.subs["/rtabmap/mapGraph"][1](
            _MapGraph(
                _shift(0.0, 0.0).transform,
                nodes={41: (60.0, 60.0, math.pi / 2), 44: (60.0, 61.0, math.pi / 2)},
            )
        )
        moved = json.loads(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1].data)
    assert (round(moved["x"], 6), round(moved["y"], 6)) == (1.6, 2.0), "the frame cancelled"
    assert moved["map"] == "3x4@1.00,2.00", "the WORD still carries the served grid's id"
    node.timers[1][1]()
    report = node.logger.texts("info")[-1]
    assert "table flat3_straight: 2 nodes over 1 sessions" in report
    assert "2 words on a node" in report and "last on node 41 (session 0) 0.60 m away" in report


def test_a_node_with_no_entry_hangs_on_its_neighbour_of_the_same_session(tmp_path: Any) -> None:
    """A database has far more nodes than our map has measured. A session is the piece the
    measurement found rigid, so a recognition of an untabled node may still be carried by the
    nearest tabled node of that session — and the substitution costs the piece's OWN measured
    rigidity, read off the fit's residuals and never a constant."""
    import json

    _tabled(tmp_path, "flat3_straight", {41: (1.0, 2.0, 0.0), 44: (2.0, 2.0, 0.0)})
    with ros_stubs.parameters(
        anchor_dir=str(tmp_path), room="flat3_straight", graph_measurement=True
    ):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 2.1, 2.0)
        _match(node, matched=77)  # a node of session 0 that nothing has measured
        _localize(node, 11.1, 10.0)
        node.subs["/rtabmap/mapGraph"][1](
            _MapGraph(
                _shift(0.0, 0.0).transform,
                nodes={41: (10.0, 10.0, 0.0), 44: (11.0, 10.0, 0.0), 77: (11.05, 10.0, 0.0)},
            )
        )
    word = json.loads(node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1].data)
    assert (round(word["x"], 6), round(word["y"], 6)) == (2.1, 2.0), "carried by node 44"
    assert node._node_words == 1 and node._substituted == 1
    assert node._hung is not None and node._hung.entry.node_id == 44 and node._hung.matched == 77
    node.timers[1][1]()
    assert "(1 on a neighbour)" in node.logger.texts("info")[-1]
    assert "for 77" in node.logger.texts("info")[-1]


def test_a_recognition_our_map_knows_nothing_about_makes_no_word(tmp_path: Any) -> None:
    """43 of the 50 sessions of this database have no entry at all (scratch/graph_node_table.py):
    they were built camera-only or outside any recorded drive. A localisation against one of those
    is a word this map cannot place, and the honest answer is to say which session and publish
    nothing — not to hang it on a piece that sits 129 degrees away."""
    _tabled(tmp_path, "flat3_straight", {41: (1.0, 2.0, 0.0)})
    with ros_stubs.parameters(
        anchor_dir=str(tmp_path), room="flat3_straight", graph_measurement=True
    ):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.6, 2.0)
        node._table.learn_sessions({41: 0, 900: 31})  # node 900 is of a session nothing measured
        _match(node, matched=900)
        _localize(node, 10.6, 10.0)
        node.subs["/rtabmap/mapGraph"][1](
            _MapGraph(
                _shift(0.0, 0.0).transform, nodes={41: (10.0, 10.0, 0.0), 900: (3.0, 4.0, 1.0)}
            )
        )
    assert node._unplaceable == 1 and node._node_words == 0
    assert "nothing on our map knows session 31" in " ".join(node.logger.texts("info"))
    assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent, "no tie to fall back on either"


def test_a_new_node_is_tabled_only_once_the_database_keeps_it(tmp_path: Any) -> None:
    """Measured live on 2026-09-18: in LOCALISATION mode the ids RTAB-Map hands out are temporary —
    nothing is written to the database, they never appear in the graph — and the table filled with
    rows nothing could ever hang on ("session -1 has tabled nodes and the live graph carries none of
    them"). So a measured node waits for the live graph to show it was kept, and that wait IS the
    mode test: it is taken in mapping mode and dropped in localisation mode."""
    from pepin.graphnodes import load_nodes

    with ros_stubs.parameters(anchor_dir=str(tmp_path), room="flat3_straight"):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)  # the lidar is behind that belief
        _match(node, matched=41, ref=3001)
        assert node._tabled == 0 and node._pending_node is not None, "measured, not taken"
        # ...and the next graph carries 3001: the database kept it, so this is mapping mode
        node.subs["/rtabmap/mapGraph"][1](
            _MapGraph(
                _shift(0.0, 0.0).transform, nodes={41: (10.0, 10.0, 0.0), 3001: (10.6, 10.0, 0.0)}
            )
        )
        assert node._tabled == 1 and node._mapping is True

        # the cart moves on, a second node is created, and this time the graph never carries it
        _odom(node, 1.2, 0.0)
        _belief(node, 2.0, 2.0)
        _match(node, matched=41, ref=3002)
        assert node._pending_node is not None
        _match(node, matched=41, ref=3003)  # a newer id arrived and 3002 never landed
        assert node._temporary == 1 and node._mapping is False, (
            "localisation mode, told by the graph"
        )
    stored = load_nodes(tmp_path, "flat3_straight")
    assert list(stored) == [3001], "and nothing temporary reached the log"


def test_a_recognised_node_is_measured_backwards_onto_our_map(tmp_path: Any) -> None:
    """How the 43 untabled sessions of this database become placeable at all, and the only way the
    table grows where RTAB-Map keeps nothing: when a node is RECOGNISED with the lidar holding a
    sharp pose, Table[N] = cart . inverse(rel). The entry is one registration noisier than one
    measured at the node itself and says so in its sigma."""
    from pepin.graphnodes import load_nodes

    _tabled(tmp_path, "flat3_straight", {41: (1.0, 2.0, 0.0)})
    with ros_stubs.parameters(anchor_dir=str(tmp_path), room="flat3_straight"):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        node._table.learn_sessions({41: 0, 900: 49})  # node 900 is of a session nothing measured
        _odom(node, 0.2, 0.0)
        _belief(node, 4.0, 5.0)
        _fit(node, 0.7, at=0.0)
        _match(node, matched=900, ref=3001)
        _localize(node, 20.6, 20.0)
        node.subs["/rtabmap/mapGraph"][1](
            _MapGraph(
                _shift(0.0, 0.0).transform, nodes={41: (10.0, 10.0, 0.0), 900: (20.0, 20.0, 0.0)}
            )
        )
    stored = load_nodes(tmp_path, "flat3_straight")
    assert sorted(stored) == [41, 900], "the recognised node is on our map now"
    assert (round(stored[900].cart.x, 6), round(stored[900].cart.y, 6)) == (3.4, 5.0)
    assert stored[900].session == 49 and node._from_matched == 1
    assert stored[900].sigma[0] > 0.3, "one registration noisier: RTAB-Map's own 0.36 m is in it"
    assert node._unplaceable == 1, "no word this time — but next time there will be one"

    # ...and a second recognition at the SAME place adds nothing: parked, the old rule went
    # 62 -> 81 -> 96 -> 145 in minutes
    with ros_stubs.parameters(anchor_dir=str(tmp_path), room="flat3_straight"):
        again = rtabmap_frame.RtabmapFrame()
        _map(again)
        _odom(again, 0.2, 0.0)
        _belief(again, 4.0, 5.0)
        _fit(again, 0.7, at=0.0)
        for ref in (4001, 4002, 4003):
            _match(again, matched=900, ref=ref)
            _localize(again, 20.6, 20.0)
            again.subs["/rtabmap/mapGraph"][1](
                _MapGraph(_shift(0.0, 0.0).transform, nodes={900: (20.0, 20.0, 0.0)})
            )
    assert again._tabled <= 1, "one place, one entry"


def test_a_localisation_that_names_no_node_is_odometry_and_makes_no_word(tmp_path: Any) -> None:
    """Measured live on 2026-09-18: /rtabmap/localization_pose is published on EVERY update at 1 Hz,
    recognised or nothing, and with both ids zero it is map_to_odom . odom — the session's own
    odometry, 1.14 m of accumulated sigma after 11 m of "travel" that was VO jitter at rest. Such an
    update is not a word, and the global tie may not rescue it either: the tie fitted from one pair
    put words out 36.7 deg wrong while claiming 8 deg of sigma."""
    _tabled(tmp_path, "flat3_straight", {41: (1.0, 2.0, 0.0)})
    with ros_stubs.parameters(
        anchor_dir=str(tmp_path), room="flat3_straight", graph_measurement=True
    ):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)
        _recognise(node)  # something was recognised once, so the words are allowed to travel
        _odom(node, 0.2, 0.0)
        _belief(node, 1.6, 2.0)
        node._matched = None  # ...and THIS update names no node at all
        _localize(node, 10.6, 10.0)
        node.subs["/rtabmap/mapGraph"][1](
            _MapGraph(_shift(0.0, 0.0).transform, nodes={41: (10.0, 10.0, 0.0)})
        )
    assert node._unnamed == 1 and node._node_words == 0
    assert not node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent, "no word, and no tie fallback"
    node.timers[1][1]()
    assert "1 localisations named no node" in node.logger.texts("info")[-1]


def test_the_files_are_named_by_the_room_and_the_old_names_are_migrated(tmp_path: Any) -> None:
    """Measured on the parked cart on 2026-09-18: switching the board from the seed pgm to the
    volume's exported slice moved the served id from 239x215@-18.53,-4.38 to 280x250@-19.48,-5.48,
    the same room at the same pose, the lidar's fit better — and every <map id>.graph_* file went
    invisible. So the files are named by the ROOM, and the ones written under the old name are
    carried over once instead of orphaned."""
    from pepin.graphnodes import nodes_path
    from pepin.graphtie import pairs_path

    _tabled(tmp_path, "3x4@1.00,2.00", {41: (1.0, 2.0, 0.0)})  # the legacy name, as on disk today
    pairs_path(tmp_path, "3x4@1.00,2.00").write_text(
        '{"stamp": 7.0, "map": [1.0, 2.0, 0.0], "graph": [0.2, 0.0, 0.0],'
        ' "sigma": [0.03, 0.03, 0.02]}\n'
    )
    with ros_stubs.parameters(anchor_dir=str(tmp_path)):
        node = rtabmap_frame.RtabmapFrame()
        _map(node)  # only the grid is known yet: the legacy name is what there is
        assert node._key == "3x4@1.00,2.00" and len(node._table) == 1
        from std_msgs.msg import String

        node.subs[rtabmap_frame.IDENTITY_TOPIC][1](
            String(data='{"from": "seed:flat3_straight", "id": "1a2b3c4d"}')
        )
    assert node._key == "flat3_straight"
    assert nodes_path(tmp_path, "flat3_straight").exists(), "the table came with the room"
    assert pairs_path(tmp_path, "flat3_straight").exists(), "and so did the pairs"
    assert len(node._table) == 1 and 41 in node._table
    assert any("migrated" in text for text in node.logger.texts("info"))


def test_a_parked_cart_s_word_never_widens_with_the_graph_s_creeping_counter() -> None:
    """The autopsy of 2026-09-17: camera-only and PARKED, the published sigma grew 0.38 -> 0.71 m
    and the drive was refused. The suspect is the odometry term — SCALE_ERROR times the distance
    RTAB-Map's counter has travelled — fed by a counter that creeps from VO noise at a
    standstill. A localisation carries no odometry from any session's start, so with that source the
    term is not reached at all, and the word is worth its floor and RTAB-Map's own sigma."""
    import json

    def sigma(node: Any) -> float:
        sent = node.pubs[rtabmap_frame.MEASUREMENT_TOPIC].sent[-1]
        return math.sqrt(float(json.loads(sent.data)["covariance"][0][0]))

    with ros_stubs.parameters(graph_measurement=True, graph_word_from_nodes=False):
        node = rtabmap_frame.RtabmapFrame()
        node._map_id = "flat3"
        node._key = "flat3"
        _odom(node, 0.2, 0.0)
        _belief(node, 1.0, 2.0)
        _fit(node, 0.7, at=0.0)
        _recognise(node)
        _localize(node, 0.2, 0.0)
        node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
        parked = sigma(node)

        for travelled in (20.0, 60.0, 120.0):  # the counter creeps while the cart stands still
            _info(node, travelled, ref=2000)
            _localize(node, 0.2, 0.0)
            node.subs["/rtabmap/mapGraph"][1](_MapGraph(_shift(0.0, 0.0).transform))
            assert sigma(node) == pytest.approx(parked), (
                "a localisation carries no odometry: the distance counter may not touch its sigma"
            )
        assert parked > 0.36, "RTAB-Map's own sigma is in it, and the tie's on top"


def _ready(node: Any) -> None:
    """Every service this node calls, answered by somebody."""
    for client in node.service_clients.values():
        client.ready = True


def test_the_mode_follows_trust_in_the_pose_and_not_a_sensor_s_name(tmp_path: Any) -> None:
    """The database may LEARN only while a sharp pose exists that does not come from the database
    itself. Measured live on 2026-09-18: in mapping mode parked, RTAB-Map found this very place
    (hypothesis 0.978, 328 visual inliers) and rejected its own recognition on the error ratio, and
    it kept a node a second — while in localisation mode with the updates ungated every update named
    a node. So the mode is switched live, on the rule and never on a flag alone."""
    with ros_stubs.parameters(anchor_dir=str(tmp_path), room="flat3_straight"):
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

        # the lidar is still matching, but the graph's own words now hold the pose: the pupil is
        # not the teacher, whatever the seating is worth
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
    assert "asked localising (the pose is held by the graph: the pupil is not the teacher" in report
    assert "2 switches, by trust" in report


def test_a_soft_pose_teaches_nothing_however_it_is_held(tmp_path: Any) -> None:
    """The sharpness half of the rule, and the half that needs no names at all: a mono camera-only
    pose sits at a sigma around 20 cm and fails the seating test by itself, so a database is never
    taught from it — and a stereo matcher good to a few cm will teach the day it exists."""
    with ros_stubs.parameters(anchor_dir=str(tmp_path), room="flat3_straight"):
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


def test_the_mode_override_keeps_both_old_arrangements_reachable(tmp_path: Any) -> None:
    """CLAUDE.md rule 19. graph_memory=map is the arrangement of before 2026-09-18 (always mapping);
    localise freezes the database for the session, which is what a measurement of what the node
    table alone is worth needs."""
    with ros_stubs.parameters(anchor_dir=str(tmp_path), room="flat3_straight", graph_memory="map"):
        node = rtabmap_frame.RtabmapFrame()
        _ready(node)
        _map(node)
        _sources(node, holder="graph")  # every reason to localise...
        node.timers[0][1]()
    assert len(node.service_clients[rtabmap_frame.MAPPING_SERVICE].calls) == 1, "...and told to map"

    with ros_stubs.parameters(
        anchor_dir=str(tmp_path), room="flat3_straight", graph_memory="localise"
    ):
        frozen = rtabmap_frame.RtabmapFrame()
        _ready(frozen)
        _map(frozen)
        _sources(frozen, holder="lidar")
        _belief(frozen, 1.0, 2.0)
        _fit(frozen, 0.7, at=0.0)
        frozen.timers[0][1]()
    assert len(frozen.service_clients[rtabmap_frame.LOCALISATION_SERVICE].calls) == 1
