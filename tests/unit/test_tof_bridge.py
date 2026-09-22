"""The ToF bridge under the ROS stubs: what leaves it, in what shape, and on which frames.

The whiskers reach Nav2 as small LaserScan fans read by an ``ObstacleLayer`` (``range_as``,
2026-09-21). That is the answer to the two defects of ``nav2_costmap_2d::RangeSensorLayer``
that have each stopped this robot — a ``canTransform`` that blocks a whole
``transform_tolerance`` per message it cannot place, and an unbounded loop over unsigned cell
bounds (range_sensor_layer.cpp:362-369) — because that layer drops what it cannot place
instead of blocking on it and walks points instead of a cell rectangle. So this file holds the
contract of the fan: what it says for a return, for "nothing within my trusted range" and for
"I do not know" is exactly what the layer reads as a mark, a clear and a silence.

The bridge itself guards nothing on the consumer's behalf any more (2026-09-22): the ``tf_gate``
and the ``dynamic_mounts`` of 2026-09-21 are gone with the range layers they were written for,
and the mounts are the static broadcast this node always made, re-sent for a window because a
one-shot can be missed under rmw_zenoh.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
import ros_stubs

ros_stubs.install()

from pepin_bringup import tof_bridge as module  # noqa: E402
from pepin_bringup.tof_bridge import FLAGS, TofBridge  # noqa: E402

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


def bridge(monkeypatch: Any, **flags: Any) -> TofBridge:
    """A bridge on a fake link, built with the launch overrides a test asks for."""
    monkeypatch.setattr(module, "JsonLineLink", FakeLink)
    with ros_stubs.parameters(**flags):
        return TofBridge()


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


def test_the_mounts_are_static_and_only_static(monkeypatch: Any) -> None:
    """One edge, one kind of message. base_link -> tof_<name> goes out on /tf_static and never
    on /tf: tf2 re-allocates a frame's cache whenever the same edge arrives as the other kind,
    and a lookup landing just after throws an exception Nav2's range layer does not catch (the
    Nav2 container aborted that way on 2026-09-21). The mount riding on /tf with every reading
    was also what made the ObstacleLayer's message filter drop every fan — 723 drops in one
    drive, the whiskers neither marking nor clearing — so it came out on 2026-09-22."""
    node = bridge(monkeypatch)
    feed(node)
    mounts = node._static_tf.sent
    assert [t.child_frame_id for t in mounts] == ["tof_front", "tof_left", "tof_right"]
    assert [t.header.frame_id for t in mounts] == ["base_link"] * 3
    assert len(node._static_tf.batches) == 1, "the three mounts in one message"
    assert not hasattr(node, "_mount_tf"), "nothing of this node's goes out on /tf"


def test_the_static_mounts_are_re_sent_for_their_window_and_then_stop(monkeypatch: Any) -> None:
    """A static transform is published ONCE, and under rmw_zenoh a subscriber that matched a
    moment too early or too late never gets it (Nav2's container did not see the ToF mounts for
    157 s on 4 of 7 board starts of 2026-09-21). So the three mounts are re-sent once a second
    while the window lasts, and the timer cancels itself when it is over."""
    node = bridge(monkeypatch, static_tf_resend_s=5.0)
    assert (1.0, node._resend_static) in node.timers
    node._resend_static()
    node._resend_static()
    assert len(node._static_tf.batches) == 3, "one at start, one per second after it"
    node._static_resend_until = 0.0  # the window is over
    node._resend_static()
    assert len(node._static_tf.batches) == 3 and node._static_resend.cancelled


def test_every_reading_leaves_and_the_whole_line_carries_one_stamp(monkeypatch: Any) -> None:
    """Nothing is withheld here. The three sensors are read together, so one line of readings
    is stamped once and every message of it — Range and fan — carries that moment; the stamp
    sits behind ``now`` because the reading is at least that old and a stamp behind the newest
    odom -> base_link never makes a costmap wait."""
    node = bridge(monkeypatch)
    feed(node)
    messages = published(node)
    assert len(messages) == 3
    stamps = {(m.header.stamp.sec, m.header.stamp.nanosec) for m in messages}
    assert len(stamps) == 1, "one stamp per line of readings"
    assert {(s.header.stamp.sec, s.header.stamp.nanosec) for s in scans(node)} == stamps


def test_the_report_line_carries_the_sensors_verdicts_and_the_flags(monkeypatch: Any) -> None:
    """CLAUDE.md rule 19: the switches are in the node's own report line, beside the raw
    VL53L1X statuses that say which sensor is answering at all."""
    node = bridge(monkeypatch)
    feed(node)
    node._report_status()
    line = node.logger.texts("info")[-1]
    assert "front [0:100%]" in line and "right [2:100%]" in line
    assert "range_as=scan" in line
    assert "gate" not in line, "the gate came out on 2026-09-22"


def test_the_flags_are_the_features_own_names_and_live() -> None:
    """Rule 19: a flag is named after its feature, it takes effect at once, and it carries the
    measurement its default rests on. The gate's three came out with the range layers."""
    assert FLAGS.names == ("range_as",)
    assert all(flag.live and flag.measured for flag in FLAGS)
    assert FLAGS["range_as"] == "scan"


# ---- the fan: one cone as points, because a RangeSensorLayer is not safe to run ---------------
def test_a_return_is_marked_across_the_whole_cone_with_no_gap_in_it(monkeypatch: Any) -> None:
    """The fan IS the cone. Every beam carries the one distance the sensor measured — a whisker
    cannot say where across its 27 degrees the thing stands — the fan spans the whole field of
    view symmetrically about the sensor's own axis, and it has enough beams that neighbours land
    at most one costmap cell apart where the arc is widest, at the sensor's ceiling. The front
    whisker's 0.96 m ceiling needs 11 beams, the two low ones 7."""
    node = bridge(monkeypatch)
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
    node = bridge(monkeypatch)
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
    node = bridge(monkeypatch)
    reading = {"front": 60, "left": 500, "right": 500, "status": {"front": 0, "right": 0}}
    feed(node, reading)  # left has no verdict at all: the sensor did not answer
    front, left = scans(node, "front")[-1], scans(node, "left")[-1]
    assert all(math.isnan(r) for r in front.ranges), "0.06 m is the sensor's own window"
    assert all(math.isnan(r) for r in left.ranges), "a sensor that did not answer at all"
    assert node.pubs["tof/front"].sent[-1].range == -1.0, "the Range says the same thing"
    assert not any(math.isnan(r) for r in scans(node, "right")[-1].ranges)


def test_a_silent_sensor_is_said_once_in_the_log(monkeypatch: Any) -> None:
    """A dead sensor and a sensor staring at an empty room look alike on the wire — both report
    nothing valid — so the run's own log must name the one that has been quiet for too long
    (front: 801 of 802 frames invalid, 2026-09-08), and name it once."""
    node = bridge(monkeypatch)
    quiet = {"front": None, "left": None, "right": None, "status": dict.fromkeys(NAMES, 2)}

    def said() -> list[str]:
        """The silence warnings in the node's log so far."""
        return [t for t in node.logger.texts("warning") if "nothing inside its trusted" in t]

    node._last_valid = dict.fromkeys(NAMES, 0.0)  # long ago: the next check warns
    feed(node, quiet)
    assert len(said()) == 3, "one per sensor"
    feed(node, quiet)
    assert len(said()) == 3, "said once, not once per reading"


def test_the_old_plugin_is_one_live_flag_away(monkeypatch: Any) -> None:
    """Rule 19: ``range_as=range`` is the node before 2026-09-21 — the three Range topics and
    nothing on the fans — and the switch works in both directions on a running robot, because a
    field that has to be reverted is not a switch."""
    node = bridge(monkeypatch, range_as="range")
    feed(node)
    assert len(published(node)) == 3 and scans(node) == []
    assert node.set_parameters([Param("range_as", "scan")])[0].successful
    feed(node)
    assert len(scans(node)) == 3, "a flag flipped live puts the cones on the wire at once"
    assert not node.set_parameters([Param("range_as", "cone")])[0].successful
