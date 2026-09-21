"""The ToF bridge under the ROS stubs: the precondition it owns before a reading leaves it, and
the shape it leaves in.

A reading whose frame cannot be placed in the global frame is not a lost measurement, it is a
wedge: Nav2's RangeSensorLayer blocks a whole ``transform_tolerance`` per message on any
transform failure, three layers per costmap receive 15 Hz each, and the costmap's first update
then never ends (scratch/nav2_hang/, and the node's own docstring). So this file holds the
contract of the two flags that answer it — ``dynamic_mounts`` (the mount goes out with every
reading, so no late joiner can miss the frame) and ``tf_gate`` (no reading leaves while the
chain that must place it is broken or stale) — including the two properties that keep the gate
from becoming its own failure: it judges ``map <- base_link``, never the mount it suppresses
itself, and with every flag the other way the node is the one that shipped before them.

...and the contract of ``range_as``, the answer to that plugin's OTHER defect (an unbounded
loop over unsigned cell bounds, range_sensor_layer.cpp:362-369, which no publisher can gate):
every cone also leaves as a small LaserScan fan for an ObstacleLayer, and what that fan says
for a return, for "nothing within my trusted range" and for "I do not know" is exactly what the
layer reads as a mark, a clear and a silence.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import ros_stubs

ros_stubs.install()

from pepin_bringup import tof_bridge as module  # noqa: E402
from pepin_bringup.tof_bridge import FLAGS, ChainState, Gate, TfChainProbe, TofBridge  # noqa: E402
from ros_stubs import Buffer, Header, TransformStamped  # noqa: E402
from ros_stubs import Time as TimeMsg  # noqa: E402

from pepin.tof_horizon import cone_beams  # noqa: E402

READING = {"front": 500, "left": 900, "right": None, "status": {"front": 0, "left": 0, "right": 2}}
NAMES = ("front", "left", "right")


class Param:
    """What ``ros2 param set`` hands the node's callback."""

    def __init__(self, name: str, value: Any) -> None:
        self.name, self.value = name, value


class FakeLink:
    """The ToF server's socket: nothing is opened, nothing is read."""

    def __init__(self, host: str, port: int, on_line: Any, name: str = "") -> None:
        self.host, self.port, self.on_line, self.name = host, port, on_line, name

    def start(self) -> None:
        """The reader thread the real one starts; here the test feeds the node by hand."""

    def stop(self) -> None:
        """The clean close."""

    def take_status_change(self) -> tuple[bool, str] | None:
        """No link, no transitions."""
        return None


class FakeProbe:
    """A :class:`pepin_bringup.tof_bridge.ChainProbe`: it answers whatever the test set, and
    remembers every frame it was asked about."""

    def __init__(self, state: ChainState | None = None) -> None:
        self.state = state if state is not None else ChainState(-0.04)
        self.asked: list[str] = []

    def chain(self, frame: str, stamp_s: float) -> ChainState:
        self.asked.append(frame)
        return self.state


def bridge(monkeypatch: Any, probe: Any = None, **flags: Any) -> TofBridge:
    """A bridge on a fake link, with a fake probe unless the test wants the node's own."""
    monkeypatch.setattr(module, "JsonLineLink", FakeLink)
    with ros_stubs.parameters(**flags):
        return TofBridge(probe)


def feed(node: TofBridge, reading: dict[str, Any] | None = None) -> None:
    """One line of the ToF server through the node, as the reader thread and the drain timer
    deliver it."""
    node._enqueue_ranges(reading if reading is not None else READING)
    node._publish_pending()


def published(node: TofBridge) -> list[Any]:
    """Every Range that left the node, over all three topics."""
    return [msg for name in NAMES for msg in node.pubs[f"tof/{name}"].sent]


def scans(node: TofBridge, name: str = "") -> list[Any]:
    """Every LaserScan that left the node, over one sensor's fan topic or over all three."""
    wanted = (name,) if name else NAMES
    return [msg for n in wanted for msg in node.pubs[f"tof/{n}/scan"].sent]


def test_a_fresh_chain_lets_the_reading_out_with_its_mount_on_the_same_stamp(
    monkeypatch: Any,
) -> None:
    """The shipping state: three Ranges, three mounts on /tf, one stamp for all six — a consumer
    that takes the mount before the Range can place the cone without waiting for anything."""
    node = bridge(monkeypatch, FakeProbe(ChainState(-0.04)))
    feed(node)
    messages = published(node)
    assert len(messages) == 3
    mounts = node._mount_tf.sent
    assert [t.child_frame_id for t in mounts] == ["tof_front", "tof_left", "tof_right"]
    assert [t.header.frame_id for t in mounts] == ["base_link"] * 3
    assert len(node._mount_tf.batches) == 1, "three mounts in one message: 15 a second, not 45"
    stamps = {(m.header.stamp.sec, m.header.stamp.nanosec) for m in messages}
    assert len(stamps) == 1, "one stamp per line of readings"
    assert {(t.header.stamp.sec, t.header.stamp.nanosec) for t in mounts} == stamps


def test_a_broken_chain_publishes_nothing_at_all_and_counts_it(monkeypatch: Any) -> None:
    """Neither the Range nor its mount: what the gate withholds must not feed the gate."""
    node = bridge(monkeypatch, FakeProbe(ChainState(None, 'Lookup: "map" does not exist')))
    feed(node)
    feed(node)
    assert published(node) == [] and node._mount_tf.sent == []
    assert node._gate.withheld == dict.fromkeys(NAMES, 2)
    assert node._gate.withheld_total == dict.fromkeys(NAMES, 2)
    assert not node._gate.open


def test_a_stale_chain_shuts_the_gate_and_the_bound_is_live(monkeypatch: Any) -> None:
    """A chain that resolves but stopped moving is as bad as no chain: the costmap would still
    block. How stale is too stale is a live flag, so it is raised on the robot, not in a build."""
    node = bridge(monkeypatch, FakeProbe(ChainState(0.9)))
    feed(node)
    assert published(node) == []
    assert node.set_parameters([Param("tf_gate_max_lag_s", 1.0)])[0].successful
    feed(node)
    assert len(published(node)) == 3


def test_the_gate_judges_the_chain_the_bridge_does_not_own(monkeypatch: Any) -> None:
    """map <- base_link, never map <- tof_*: the mount is THIS node's own transform and the gate
    suppresses it, so a gate judged on it would shut once and never open again."""
    probe = FakeProbe()
    node = bridge(monkeypatch, probe)
    feed(node)
    assert set(probe.asked) == {"base_link"}


def test_the_gate_says_it_once_when_it_shuts_and_once_when_it_opens(monkeypatch: Any) -> None:
    """A shut gate is never silent: it is the reason /tof/* went quiet, and a log that did not
    say so would send the next reader to the sensors."""
    probe = FakeProbe(ChainState(-0.04))
    node = bridge(monkeypatch, probe)
    feed(node)
    assert any("tf gate open" in line for line in node.logger.texts("info"))
    probe.state = ChainState(None, "Lookup: no chain")
    feed(node)
    feed(node)
    shut = [line for line in node.logger.texts("warning") if "tf gate SHUT" in line]
    assert len(shut) == 1 and "Lookup: no chain" in shut[0]
    probe.state = ChainState(0.0)
    feed(node)
    assert len([line for line in node.logger.texts("info") if "tf gate open" in line]) == 2


def test_the_report_line_carries_the_gate_its_counts_and_the_flags(monkeypatch: Any) -> None:
    """CLAUDE.md rule 19: the switches and what they did are in the node's own report line."""
    node = bridge(monkeypatch, FakeProbe(ChainState(None, "Lookup: no chain")))
    feed(node)
    node._report_status()
    line = node.logger.texts("info")[-1]
    assert "gate SHUT" in line and "Lookup: no chain" in line
    assert "withheld front 1 left 1 right 1" in line and "since start 3" in line
    assert "dynamic_mounts=on" in line and "tf_gate=on" in line and "tf_gate_max_lag_s=0.5" in line
    node._report_status()
    assert "withheld front 0 left 0 right 0" in node.logger.texts("info")[-1], "a fresh period"


def test_with_both_switches_off_the_node_is_the_one_that_shipped_before(monkeypatch: Any) -> None:
    """The old behaviour stays reachable (rule 19): the static mounts alone, every reading
    published whatever TF says — and no TF listener is built at all, so the board pays nothing."""
    node = bridge(monkeypatch, None, tf_gate=False, dynamic_mounts=False)
    assert node._probe is None
    feed(node)
    assert len(published(node)) == 3
    assert node._mount_tf.sent == []
    static = [t.child_frame_id for t in node._static_tf.sent]
    assert static == ["tof_front", "tof_left", "tof_right"]
    node._report_status()
    assert "gate off, every reading leaves" in node.logger.texts("info")[-1]


def test_the_mounts_leave_one_way_at_a_time_never_both(monkeypatch: Any) -> None:
    """tf2 re-allocates a frame's cache whenever the same edge arrives as the other kind, and a
    lookup landing just after throws an exception Nav2's range layer does not catch (the Nav2
    container aborted that way on 2026-09-21). So the shipping node never touches /tf_static,
    the old node never touches /tf, and a live flip moves from one to the other."""
    node = bridge(monkeypatch, FakeProbe())
    feed(node)
    assert node._static_tf is None, "dynamic mounts: no static message has ever left"
    assert node._mount_tf.sent
    old = bridge(monkeypatch, FakeProbe(), dynamic_mounts=False)
    feed(old)
    assert old._static_tf is not None and len(old._static_tf.sent) == 3
    assert not old._mount_tf.sent
    node._on_switch("dynamic_mounts", True, False)
    assert node._static_tf is not None and len(node._static_tf.sent) == 3
    node._on_switch("dynamic_mounts", False, True)
    before = len(node._static_tf.sent)
    node._resend_static()
    assert len(node._static_tf.sent) == before, "the re-send stops with the static way"


def test_a_gate_turned_on_live_gets_its_listener_then(monkeypatch: Any) -> None:
    """The switch is live in both directions: turning it on builds the probe the node did not
    have, and the next reading is judged rather than let through blind."""
    node = bridge(monkeypatch, None, tf_gate=False)
    assert node._probe is None
    assert node.set_parameters([Param("tf_gate", True)])[0].successful
    assert isinstance(node._probe, TfChainProbe)


def test_a_withheld_reading_still_counts_as_the_sensor_being_alive(monkeypatch: Any) -> None:
    """A shut gate must not be reported as three dead sensors: the silence warning is about the
    hardware, and the hardware is answering."""
    node = bridge(monkeypatch, FakeProbe(ChainState(None, "Lookup: no chain")))
    node._last_valid = dict.fromkeys(NAMES, 0.0)  # long ago: the next check would warn
    feed(node)
    assert [line for line in node.logger.texts("warning") if "nothing inside its trusted" in line]
    assert node._last_valid["front"] > 0.0, "the front sensor answered, gate or no gate"


def test_the_real_probe_reads_the_chain_s_latest_common_time(monkeypatch: Any) -> None:
    """The one lookup that says both things at once: tf2's "latest" (time 0, timeout 0) resolves
    to the chain's latest common time and carries it in the answer's own stamp."""
    node = bridge(monkeypatch, None, tf_gate=False)
    probe = TfChainProbe(node, "map", 2.0)
    buffer: Any = probe._lookup.buffer
    assert isinstance(buffer, Buffer) and buffer.cache_time is not None
    transform = TransformStamped(header=Header(stamp=TimeMsg(sec=100, nanosec=0), frame_id="map"))
    buffer.transforms[("map", "base_link")] = transform
    assert probe.chain("base_link", 100.10).lag_s == pytest.approx(0.10)
    assert probe.chain("base_link", 99.90).lag_s == pytest.approx(-0.10)
    assert buffer.calls[-1][3].nanoseconds == 0, "timeout 0: the gate never waits"
    buffer.error = KeyError("no such frame")
    state = probe.chain("base_link", 100.0)
    assert state.lag_s is None and "no such frame" in state.detail


def test_the_gate_is_shut_until_something_proves_otherwise() -> None:
    """Pure bookkeeping, no ROS: a gate that started open would let the first readings of a cold
    start through — which is exactly the window the wedge is built in."""
    gate = Gate(NAMES, max_lag_s=0.5)
    assert not gate.open and gate.take_change() is None
    assert gate.judge(ChainState(0.1), 10.0) and gate.open
    assert "tf gate open" in (gate.take_change() or "")
    assert gate.take_change() is None, "said once"
    assert not gate.judge(ChainState(None, "Lookup: gone"), 20.0)
    assert "10 s" in (gate.take_change() or ""), "how long it stood the other way"


def test_the_flags_are_the_features_own_names_and_live() -> None:
    """Rule 19: a flag is named after its feature, every one of these takes effect at once, and
    each carries the measurement its default rests on."""
    assert FLAGS.names == ("dynamic_mounts", "tf_gate", "tf_gate_max_lag_s", "range_as")
    assert all(flag.live and flag.measured for flag in FLAGS)
    assert FLAGS["dynamic_mounts"] is True and FLAGS["tf_gate"] is True
    assert FLAGS["tf_gate_max_lag_s"] == 0.5
    assert FLAGS["range_as"] == "scan"


# ---- the fan: one cone as points, because a RangeSensorLayer is not safe to run ---------------
def test_a_return_is_marked_across_the_whole_cone_with_no_gap_in_it(monkeypatch: Any) -> None:
    """The fan IS the cone. Every beam carries the one distance the sensor measured — a whisker
    cannot say where across its 27 degrees the thing stands — the fan spans the whole field of
    view symmetrically about the sensor's own axis, and it has enough beams that neighbours land
    at most one costmap cell apart where the arc is widest, at the sensor's ceiling. The front
    whisker's 0.96 m ceiling needs 11 beams, the two low ones 7."""
    node = bridge(monkeypatch, FakeProbe())
    feed(node)
    fov = module._FIELD_OF_VIEW_RAD
    for name in NAMES:
        scan = scans(node, name)[-1]
        ceiling = node._ceiling[name]
        assert scan.header.frame_id == f"tof_{name}", "the cone's origin is the sensor"
        assert scan.range_min == module._MIN_RANGE_M, "the contact band is dropped by min_range"
        assert scan.range_max == ceiling, "the fan never announces reach the sensor has not got"
        assert len(scan.ranges) == cone_beams(ceiling, fov, module._COSTMAP_CELL_M)
        assert scan.angle_min == pytest.approx(-fov / 2)
        assert scan.angle_max == pytest.approx(fov / 2)
        assert ceiling * scan.angle_increment <= module._COSTMAP_CELL_M, "a gap in the cone"
    assert [round(r, 3) for r in scans(node, "front")[-1].ranges] == [0.5] * 11
    assert len(scans(node, "left")[-1].ranges) == 7


def test_nothing_in_range_is_an_inf_on_every_beam_so_the_cone_clears(monkeypatch: Any) -> None:
    """A whisker clears by saying +inf: Nav2's laserScanValidInfCallback puts that point at the
    fan's own range_max less a tenth of a millimetre and raytraces the cone free. The two
    readings that mean "nothing" — none at all, and one past the sensor's floor horizon — must
    both come out that way, and the Range beside it still says the ceiling as it always did."""
    node = bridge(monkeypatch, FakeProbe())
    feed(node, {"front": None, "left": 1200, "right": None, "status": dict.fromkeys(NAMES, 2)})
    for name in ("front", "right"):
        assert all(math.isinf(r) and r > 0 for r in scans(node, name)[-1].ranges), name
        assert node.pubs[f"tof/{name}"].sent[-1].range == node._ceiling[name]
    # left read 1.20 m, past its 0.57 m floor horizon: the carpet, not a wall
    assert all(math.isinf(r) for r in scans(node, "left")[-1].ranges)


def test_a_sensor_that_did_not_answer_neither_marks_nor_clears(monkeypatch: Any) -> None:
    """ "I do not know" is NaN on every beam: the callback leaves it alone (it only rewrites
    positive infinities) and laser_geometry's projector drops it, so the cone is neither marked
    nor cleared — which is what the Range's -1.0 has always meant. A sensor off the bus is not
    an empty room, and neither is a reading from the sensor's own window."""
    node = bridge(monkeypatch, FakeProbe())
    reading = {"front": 60, "left": 500, "right": 500, "status": {"front": 0, "right": 0}}
    feed(node, reading)  # left has no verdict at all: the sensor did not answer
    front, left = scans(node, "front")[-1], scans(node, "left")[-1]
    assert all(math.isnan(r) for r in front.ranges), "0.06 m is the sensor's own window"
    assert all(math.isnan(r) for r in left.ranges), "a sensor that did not answer at all"
    assert node.pubs["tof/front"].sent[-1].range == -1.0, "the Range says the same thing"
    assert not any(math.isnan(r) for r in scans(node, "right")[-1].ranges)


def test_the_fan_rides_with_the_mount_and_the_range_on_one_stamp(monkeypatch: Any) -> None:
    """One gate decision, one stamp, one set of mounts for all six messages of a line: a
    consumer that takes the mount first can place every cone without waiting for anything."""
    node = bridge(monkeypatch, FakeProbe(ChainState(-0.04)))
    feed(node)
    stamps = {(m.header.stamp.sec, m.header.stamp.nanosec) for m in published(node)}
    assert len(stamps) == 1
    assert {(s.header.stamp.sec, s.header.stamp.nanosec) for s in scans(node)} == stamps
    assert {(t.header.stamp.sec, t.header.stamp.nanosec) for t in node._mount_tf.sent} == stamps


def test_a_shut_gate_withholds_the_fans_too(monkeypatch: Any) -> None:
    """The gate is about the CHAIN that must place a cone, and a fan needs it exactly as a Range
    does: nothing on either topic while it is shut."""
    node = bridge(monkeypatch, FakeProbe(ChainState(None, "Lookup: no chain")))
    feed(node)
    assert published(node) == [] and scans(node) == []


def test_the_old_plugin_is_one_live_flag_away(monkeypatch: Any) -> None:
    """Rule 19: ``range_as=range`` is the node before 2026-09-21 — the three Range topics and
    nothing on the fans — and the switch works in both directions on a running robot, because a
    field that has to be reverted is not a switch."""
    node = bridge(monkeypatch, FakeProbe(), range_as="range")
    feed(node)
    assert len(published(node)) == 3 and scans(node) == []
    assert node.set_parameters([Param("range_as", "scan")])[0].successful
    feed(node)
    assert len(scans(node)) == 3, "a flag flipped live puts the cones on the wire at once"
    assert not node.set_parameters([Param("range_as", "cone")])[0].successful
