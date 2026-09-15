"""The Python base bridge under the ROS stubs: what a muted sensor publishes, which is nothing.

The mute is the publisher end of a sensor switch (ros/sensor.sh mute odom): the base server is
still read, the state line still arrives, and the node simply stops sending — so a consumer
meets silence, the way it would if the sensor had died, and no restart loses the other live
flags. The same two flags are declared by the C++ bridge that actually runs on the board
(ros/pepin_base_cpp/src/base_bridge.cpp); this file holds the contract on the Python twin.
"""

from __future__ import annotations

from typing import Any

import ros_stubs

RCLPY = ros_stubs.install()

from pepin_bringup import base_bridge as module  # noqa: E402
from pepin_bringup.base_bridge import FLAGS, BaseBridge  # noqa: E402

STATE = {
    "type": "state",
    "t": 100.0,
    "x": 1.0,
    "y": 0.0,
    "theta": 0.0,
    "dl": 0.0,
    "dr": 0.0,
    "v": 0.1,
    "w": 0.0,
    "moving": True,
    "armed": True,
    "deadman": False,
    "bus_ok": True,
}


class Param:
    """What ``ros2 param set`` hands the node's callback."""

    def __init__(self, name: str, value: Any) -> None:
        self.name, self.value = name, value


class FakeLink:
    """The base server's socket: nothing is opened, every line sent down is kept."""

    def __init__(self, host: str, port: int, on_line: Any, name: str = "") -> None:
        self.host, self.port, self.on_line, self.name = host, port, on_line, name
        self.sent: list[Any] = []
        self.status: tuple[bool, str] | None = None

    def start(self) -> None:
        """The reader thread the real one starts; here there is nothing to read."""

    def stop(self) -> None:
        """The clean close."""

    def send(self, line: Any) -> None:
        """Keep what the node would have sent to the base server."""
        self.sent.append(line)

    def take_status_change(self) -> tuple[bool, str] | None:
        """One link up/down transition, once, the way the real link hands it over."""
        change, self.status = self.status, None
        return change


def _bridge(monkeypatch: Any) -> BaseBridge:
    """A bridge on a fake link, ready to be fed state lines."""
    monkeypatch.setattr(module, "JsonLineLink", FakeLink)
    return BaseBridge()


def test_both_sensors_publish_by_default_and_the_flags_are_live(monkeypatch: Any) -> None:
    """The shipping state: a state line is /odom and the odom -> base_link transform."""
    node = _bridge(monkeypatch)
    node._on_state_line(STATE)
    assert len(node.pubs["odom"].sent) == 1
    assert len(node._tf.sent) == 1
    assert FLAGS.names == ("imu_publish", "odom_publish")
    assert all(FLAGS.flag(name).live and FLAGS[name] is True for name in FLAGS.names)


def test_muting_odom_stops_the_message_and_the_transform_with_it(monkeypatch: Any) -> None:
    """Wheel odometry that keeps broadcasting a transform while /odom is silent is a state no
    sensor failure produces, so both go together — and the link is still read and still fed."""
    node = _bridge(monkeypatch)
    assert node.set_parameters([Param("odom_publish", False)])[0].successful
    node._on_state_line(STATE)
    assert node.pubs["odom"].sent == []
    assert node._tf.sent == []
    node._accept_command(0.1, 0.0)
    assert node._link.sent, "the wheels still obey while the sensor is muted"


def test_unmuting_puts_the_next_state_line_back_on_the_topic(monkeypatch: Any) -> None:
    """No restart: the very next state line is published again (20 Hz, so within 1 s)."""
    node = _bridge(monkeypatch)
    node.set_parameters([Param("odom_publish", False)])
    node._on_state_line(STATE)
    assert node.set_parameters([Param("odom_publish", True)])[0].successful
    node._on_state_line(STATE)
    assert len(node.pubs["odom"].sent) == 1


def test_the_mute_is_not_a_measurement_the_twist_estimator_may_use(monkeypatch: Any) -> None:
    """The measured twist differences two consecutive wheel poses; the gap a mute makes is not
    a step the cart took, so the estimator is reset instead of jumping over it."""
    node = _bridge(monkeypatch)
    node._on_state_line(STATE)
    node.set_parameters([Param("odom_publish", False)])
    node._on_state_line({**STATE, "t": 101.0, "x": 5.0})
    node.set_parameters([Param("odom_publish", True)])
    node._on_state_line({**STATE, "t": 101.1, "x": 5.0})
    assert node.pubs["odom"].sent[-1].twist.twist.linear.x == 0.0


def test_the_report_line_names_both_switches(monkeypatch: Any) -> None:
    """CLAUDE.md rule 19: the link-up line says what is muted, so a log answers the question."""
    node = _bridge(monkeypatch)
    node.set_parameters([Param("odom_publish", False)])
    node._link.status = (True, "base server up")
    node._log_link_status()
    line = node.logger.lines[-1][1]
    assert "imu_publish=on" in line and "odom_publish=off" in line, line
