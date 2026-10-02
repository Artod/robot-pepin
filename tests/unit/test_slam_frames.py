"""rtabmap_frame: RTAB-Map's memory mode, its registration, its visual features and the parameter
path they travel by.

rclpy and tf2_ros are faked (``ros_stubs``); ``rtabmap_msgs`` is faked here, because only the
laptop's image carries it. What the fakes let a test see: which topics the node opened, what it
published, and which services it called with which parameters. The grid relay and the placement
word are held in tests/unit/test_one_localiser.py.
"""

from typing import Any

import pytest
import ros_stubs

from pepin.snapshot import SnapshotState

ros_stubs.install()


from pepin_bringup import rtabmap_frame  # noqa: E402


def _ready(node: Any) -> None:
    """Every service this node calls, answered by somebody."""
    for client in node.service_clients.values():
        client.ready = True


def _besides_recognition(calls: list[Any]) -> list[Any]:
    """The parameter sets other than place_recognition's likelihood, which rtabmap_frame sends
    once at every start whatever the table says (a respawn may sit beside a descriptor RTAB-Map)."""
    return [c for c in calls if not any(p.name == "Kp/TfIdfLikelihoodUsed" for p in c.parameters)]


# ---- the laptop: the graph's word, in the ONE map frame ---------------------------------------


# ---- RTAB-Map's memory: when the database may learn -------------------------------------------


# ---- the registration follows the snapshots ----------------------------------------------------
def _snapshots(node: Any, carrying: tuple[str, ...], kind: str, refresh_s: float = 0.5) -> None:
    """What pepin_bringup.sensor_pack's snapshots carry, on its own latched topic."""
    node.subs[rtabmap_frame.SNAPSHOT_STATE_TOPIC][1](
        ros_stubs.String(
            data=SnapshotState(
                carrying=carrying, kind=kind, refresh_s=refresh_s, stamp=1.0
            ).to_json()
        )
    )


def _tuner_ready(node: Any) -> tuple[Any, Any]:
    """RTAB-Map's parameter path, up: the set and the re-read it needs to be honoured at all."""
    tuner = node.service_clients[f"{rtabmap_frame.RTABMAP_NODE}/set_parameters_atomically"]
    reread = node.service_clients[f"{rtabmap_frame.RTABMAP_NODE}/update_parameters"]
    tuner.ready = reread.ready = True
    return tuner, reread


def _strategies(tuner: Any) -> list[str]:
    """Every Reg/Strategy that went out, in order, as the STRING rtabmap reads back."""
    out = []
    for request in tuner.calls:
        for parameter in request.parameters:
            if parameter.name == "Reg/Strategy":
                out.append(parameter.value.string_value)
    return out


def test_the_snapshot_state_is_read_latched_from_the_packer_s_own_topic() -> None:
    node = rtabmap_frame.RtabmapFrame()
    assert rtabmap_frame.SNAPSHOT_STATE_TOPIC == "/sensor_pack/state", (
        "the same literal pepin_bringup.sensor_pack publishes on"
    )
    assert rtabmap_frame.SNAPSHOT_STATE_TOPIC in node.subs


def test_camera_only_snapshots_switch_the_registration_to_visual() -> None:
    """The one thing that makes a camera-only cart able to localise at all: under ICP it forms no
    metric link (28 'Missing visual features' in a minute, 2026-09-18)."""
    node = rtabmap_frame.RtabmapFrame()
    tuner, reread = _tuner_ready(node)
    node.clock.seconds = 10.0
    _snapshots(node, ("camera",), "camera-only", refresh_s=0.5)
    node.timers[0][1]()
    assert not _strategies(tuner), "not before the change has held"
    node.clock.seconds = 10.6
    _snapshots(node, ("camera",), "camera-only", refresh_s=0.5)
    node.timers[0][1]()
    assert _strategies(tuner) == ["0"], "Vis, as a string: every rtabmap parameter is one"
    assert reread.calls, "a set alone changes nothing — update_parameters is what re-reads them"
    assert "rtabmap registration: visual" in node.logger.texts("info")[-1]
    assert "Reg/Strategy 0" in node.logger.texts("info")[-1]


def test_the_scan_coming_back_switches_it_to_visual_then_icp_and_the_camera_going_to_icp() -> None:
    """Camera only -> 0, camera+lidar -> 2 (the tuned "F+G2" set), lidar only -> 1."""
    node = rtabmap_frame.RtabmapFrame()
    tuner, _ = _tuner_ready(node)
    for seconds in (10.0, 10.6):
        node.clock.seconds = seconds
        _snapshots(node, ("camera",), "camera-only")
        node.timers[0][1]()
    for seconds in (11.0, 11.6):
        node.clock.seconds = seconds
        _snapshots(node, ("camera", "lidar"), "full")
        node.timers[0][1]()
    for seconds in (12.0, 12.6):
        node.clock.seconds = seconds
        _snapshots(node, ("lidar",), "lidar-only")
        node.timers[0][1]()
    assert _strategies(tuner) == ["0", "2", "1"]


def test_camera_and_lidar_from_the_start_switch_to_visual_then_icp_with_xfeat(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """A fresh start with both sensors needs no script: the launch's 1 becomes 2 once the
    composition has held, with the feature flag's set while localising (xfeat, as the tuned
    set_visicp_inner.sh ran it) and the stock confirmation and proximity search."""
    _adapters(monkeypatch, tmp_path, present=True)
    node = rtabmap_frame.RtabmapFrame()
    _localising(node)
    tuner, _ = _tuner_ready(node)
    for seconds in (10.0, 10.6):
        node.clock.seconds = seconds
        _snapshots(node, ("camera", "lidar"), "full")
        node.timers[0][1]()
    (sent,) = [s for s in _sent(tuner) if "Reg/Strategy" in s]
    assert sent["Reg/Strategy"] == "2" and sent["Vis/FeatureType"] == "15"
    assert sent["RGBD/LoopClosureReextractFeatures"] == "true"
    assert sent["RGBD/ProximityBySpace"] == "true" and sent["Rtabmap/LoopThr"] == "0.11"
    assert "rtabmap registration: visual then ICP" in node.logger.texts("info")[-1]


def test_the_launch_s_own_strategy_is_never_re_sent() -> None:
    """The table already set Reg/Strategy 1, and every change deletes and re-creates the
    registration pipeline (Memory.cpp:721-731): agreeing with it must cost nothing."""
    node = rtabmap_frame.RtabmapFrame()
    tuner, _ = _tuner_ready(node)
    for seconds in (10.0, 11.0, 12.0, 20.0):
        node.clock.seconds = seconds
        _snapshots(node, ("lidar",), "lidar-only")
        node.timers[0][1]()
    assert not _strategies(tuner)


def test_a_state_older_than_its_own_refresh_leaves_the_strategy_where_it_is() -> None:
    """sensor_pack going quiet is not the lidar going away."""
    node = rtabmap_frame.RtabmapFrame()
    tuner, _ = _tuner_ready(node)
    node.clock.seconds = 10.0
    _snapshots(node, ("camera",), "camera-only", refresh_s=0.5)
    node.clock.seconds = 30.0  # the state stopped arriving twenty seconds ago
    node.timers[0][1]()
    assert not _strategies(tuner)
    node._report()
    assert "STALE" in node.logger.texts("info")[-1]


def test_a_parameter_path_that_is_not_up_is_counted_and_retried() -> None:
    """A switch the path could not take must not be forgotten: it is counted once, the report says
    the pipeline is still the old one, and the moment the path is up the switch goes out — a
    camera-only session whose first switch met a path that was not up used to stay on ICP."""
    node = rtabmap_frame.RtabmapFrame()
    for seconds in (10.0, 10.6, 10.7, 10.8):
        node.clock.seconds = seconds
        _snapshots(node, ("camera",), "camera-only")
        node.timers[0][1]()
    assert node._strategy_failed == 1, "once per switch held back, not once per tick"
    assert node._strategy.strategy == rtabmap_frame.STRATEGY_ICP, "what RTAB-Map still runs"
    node._report()
    report = node.logger.texts("info")[-1]
    assert "switches the parameter path could not take" in report
    assert "asking visual" in report
    tuner, reread = _tuner_ready(node)
    node.clock.seconds = 10.9
    _snapshots(node, ("camera",), "camera-only")
    node.timers[0][1]()
    assert _strategies(tuner) == ["0"] and reread.calls, "retried the tick the path came up"
    assert node._strategy.strategy == "0"


def test_a_half_up_parameter_path_is_sent_nothing() -> None:
    """The set and its re-read go out together or not at all: a set whose re-read could not follow
    was counted as failed and then sent again whole on the retry."""
    node = rtabmap_frame.RtabmapFrame()
    tuner = node.service_clients[f"{rtabmap_frame.RTABMAP_NODE}/set_parameters_atomically"]
    tuner.ready = True  # update_parameters is not up
    _go_visual(node)
    assert not tuner.calls and node._strategy_failed == 1


def test_a_mode_switch_the_service_could_not_take_is_asked_for_again() -> None:
    """The rule used to record a switch as applied when it returned it, so a switch dropped
    because the service was not up (or the last one unanswered) was never asked again — and the
    visual features that follow the mode believed a mode RTAB-Map had never been told."""
    node = rtabmap_frame.RtabmapFrame()
    localising = node.service_clients[rtabmap_frame.LOCALISATION_SERVICE]
    for seconds in (1.0, 1.1, 1.2):
        node.clock.seconds = seconds
        node.timers[0][1]()
    assert not localising.calls and node._mode.mode == "unknown"
    assert node._mode_failed == 1, "once per switch held back, not once per tick"
    node._report()
    assert (
        f"held back: {rtabmap_frame.LOCALISATION_SERVICE} is not up"
        in (node.logger.texts("info")[-1])
    )
    _ready(node)
    node.timers[0][1]()
    assert len(localising.calls) == 1 and node._mode.mode == "localising"
    node.timers[0][1]()
    assert len(localising.calls) == 1, "and once it went out, once"
    assert node._mode_held is None and node._mode_failed == 1


# ---- the visual registration's features ---------------------------------------------------------
def _sent(tuner: Any) -> list[dict[str, str]]:
    """Every parameter set that carried the visual features, in order, as name -> the STRING
    rtabmap reads back (the memory mode's own pair, RGBD/Linear/AngularUpdate, is left out)."""
    sets = [{p.name: p.value.string_value for p in request.parameters} for request in tuner.calls]
    return [values for values in sets if "Vis/FeatureType" in values]


def _localising(node: Any) -> None:
    """Every service up and one tick: the rule's first verdict (localise, this start not yet tied
    to the loaded map) goes out and is answered, so RTAB-Map is LOCALISING for certain."""
    _ready(node)
    node.clock.seconds = 1.0
    node.timers[0][1]()
    assert node._mode.mode == "localising" and node._database_only_read()


class _Unanswered:
    """A service call whose response has not arrived: rclpy's future, not done yet."""

    def __init__(self) -> None:
        self.answered = False
        self.callbacks: list[Any] = []

    def done(self) -> bool:
        return self.answered

    def add_done_callback(self, callback: Any) -> None:
        self.callbacks.append(callback)

    def answer(self) -> None:
        self.answered = True
        for callback in self.callbacks:
            callback(self)


def _adapters(monkeypatch: pytest.MonkeyPatch, where: Any, present: bool) -> None:
    """The image: the two adapters RTAB-Map loads by path are there (pepin-laptop:xfeat) or not."""
    for name in ("XFEAT_DETECTOR_PATH", "XFEAT_MATCHER_PATH"):
        path = where / f"{name}.py"
        if present:
            path.write_text("")
        monkeypatch.setattr(rtabmap_frame, name, str(path))


def _go_visual(node: Any) -> None:
    """Camera-only snapshots for as long as the change has to hold: the rule switches to visual."""
    for seconds in (10.0, 10.6):
        node.clock.seconds = seconds
        _snapshots(node, ("camera",), "camera-only")
        node.timers[0][1]()


def test_the_visual_strategy_goes_out_with_the_xfeat_set_in_the_xfeat_image(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """ONE set, strategy and features together: the pipeline RTAB-Map re-creates on a new
    Reg/Strategy is built from the accumulated map, so it must be born with its features."""
    _adapters(monkeypatch, tmp_path, present=True)
    node = rtabmap_frame.RtabmapFrame()
    _localising(node)
    tuner, _ = _tuner_ready(node)
    _go_visual(node)
    sets = _sent(tuner)
    assert len(sets) == 1
    assert {k: sets[0][k] for k in ("Reg/Strategy", "Vis/FeatureType", "Vis/CorNNType")} == {
        "Reg/Strategy": "0",
        "Vis/FeatureType": "15",
        "Vis/CorNNType": "6",
    }
    assert sets[0]["RGBD/LoopClosureReextractFeatures"] == "true", "the database is only read"
    assert "visual features xfeat" in node.logger.texts("info")[-1]


def test_without_the_adapters_the_visual_strategy_keeps_orb_and_says_why(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The apt image has no Python in RTAB-Map: asking it for type 15 would silently give GFTT/ORB
    under another name. The flag stays xfeat; what goes out is ORB, and the report says so."""
    _adapters(monkeypatch, tmp_path, present=False)
    node = rtabmap_frame.RtabmapFrame()
    tuner, _ = _tuner_ready(node)
    _go_visual(node)
    sets = _sent(tuner)
    assert sets[0]["Reg/Strategy"] == "0" and sets[0]["Vis/FeatureType"] == "8"
    assert sets[0]["RGBD/LoopClosureReextractFeatures"] == "false"
    node._report()
    assert "xfeat asked, but this image has no" in node.logger.texts("info")[-1]


def test_a_flag_moved_under_the_visual_strategy_re_sends_the_features_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The features ride the pipeline's own parseParameters (RegistrationVis rebuilds its
    detectors) while the strategy stays: Reg/Strategy is not sent again."""
    _adapters(monkeypatch, tmp_path, present=True)
    node = rtabmap_frame.RtabmapFrame()
    _localising(node)
    tuner, _ = _tuner_ready(node)
    _go_visual(node)
    node._switches.set("visual_features", "orb")
    node.timers[0][1]()
    node._switches.set("pnp_reproj_px", 4.0)
    node.timers[0][1]()
    node.timers[0][1]()  # nothing moved: nothing goes out
    sets = _sent(tuner)
    assert len(sets) == 3
    assert "Reg/Strategy" not in sets[1] and "Reg/Strategy" not in sets[2]
    assert sets[1]["Vis/FeatureType"] == "8" and sets[1]["Vis/PnPReprojError"] == "2"
    assert sets[2]["Vis/FeatureType"] == "8" and sets[2]["Vis/PnPReprojError"] == "4"


def test_under_icp_the_feature_flag_sends_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """Under ICP the set is ORB's, which is the launch table's: agreeing with it costs nothing."""
    _adapters(monkeypatch, tmp_path, present=True)
    node = rtabmap_frame.RtabmapFrame()
    tuner, _ = _tuner_ready(node)
    for seconds in (10.0, 11.0):
        node.clock.seconds = seconds
        _snapshots(node, ("lidar",), "lidar-only")
        node.timers[0][1]()
    node._switches.set("visual_features", "orb")
    node.timers[0][1]()
    assert not _besides_recognition(tuner.calls)


def test_the_scan_coming_back_takes_orb_back_with_icp(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """ICP is the strategy that maps, and a node mapped under re-extraction keeps no descriptors
    and no 3D (Memory.cpp:6126): ORB's set comes back in the same set as Reg/Strategy 1."""
    _adapters(monkeypatch, tmp_path, present=True)
    node = rtabmap_frame.RtabmapFrame()
    _localising(node)
    tuner, _ = _tuner_ready(node)
    _go_visual(node)
    assert _sent(tuner)[-1]["Vis/FeatureType"] == "15"
    for seconds in (11.0, 11.6):
        node.clock.seconds = seconds
        _snapshots(node, ("lidar",), "lidar-only")
        node.timers[0][1]()
    back = _sent(tuner)[-1]
    assert back["Reg/Strategy"] == "1" and back["Vis/FeatureType"] == "8"
    assert back["RGBD/LoopClosureReextractFeatures"] == "false"
    node._report()
    assert (
        "visual features orb (xfeat asked, sent with the visual strategy only and only while"
        in (node.logger.texts("info")[-1])
    )


def test_an_unknown_or_unanswered_mode_keeps_orb_and_xfeat_follows_the_answer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any
) -> None:
    """The xfeat set is in force only while RTAB-Map is LOCALISING for certain: before this node
    has told it anything, and while the localisation call is unanswered, what goes out with the
    visual strategy is ORB's — and xfeat follows on the tick the answer is in."""
    _adapters(monkeypatch, tmp_path, present=True)
    node = rtabmap_frame.RtabmapFrame()
    tuner, _ = _tuner_ready(node)  # the parameter path is up, the mode services are not
    _go_visual(node)
    assert node._mode.mode == "unknown"
    assert _sent(tuner)[-1]["Vis/FeatureType"] == "8", "nobody has told RTAB-Map to localise"

    localisation = node.service_clients[rtabmap_frame.LOCALISATION_SERVICE]
    pending = _Unanswered()
    localisation.ready = True
    localisation.call_async = lambda request: (localisation.calls.append(request), pending)[1]
    node.timers[0][1]()
    assert len(localisation.calls) == 1 and node._mode.mode == "localising"
    node.timers[0][1]()
    assert _sent(tuner)[-1]["Vis/FeatureType"] == "8", "told, but not answered yet"
    pending.answer()
    node.timers[0][1]()
    assert _sent(tuner)[-1]["Vis/FeatureType"] == "15"
    assert _sent(tuner)[-1]["RGBD/LoopClosureReextractFeatures"] == "true"


def test_a_set_lands_whole(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> None:
    """rtabmap_slam applies each /parameter_events notification as it arrives: a set_parameters
    request of five names landed as five parseParameters, Reg/Strategy first, re-extraction
    fourth (scratch/xfeat_critic/atomic_set.sh). Atomically it lands as one, and that is the only
    request this node makes."""
    _adapters(monkeypatch, tmp_path, present=True)
    node = rtabmap_frame.RtabmapFrame()
    _localising(node)
    atomic, _ = _tuner_ready(node)
    _go_visual(node)
    assert _sent(atomic)[-1]["Vis/FeatureType"] == "15"
    assert f"{rtabmap_frame.RTABMAP_NODE}/set_parameters" not in node.service_clients
