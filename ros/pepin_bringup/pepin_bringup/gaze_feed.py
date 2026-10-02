"""The gaze gate's ROS half: ``/gaze/state`` and the IMU's yaw rate into one
:class:`pepin.gaze_gate.FrameGate`, for the nodes that drop a frame by its stamp.

A module of its own, not a piece of :mod:`pepin_bringup.node_kit`: only the gated camera
consumers (depth_stream, sensor_pack, visual_odometry) import it, so a change here kicks those
three and nothing on the board.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

from rclpy.qos import QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from pepin.gaze_gate import GAZE_STATE_TOPIC, FrameGate, GazeState
from pepin_bringup.msgs import stamp_seconds
from pepin_bringup.node_kit import BASE_FRAME, bridged_qos_profile

IMU_TOPIC = "/imu/data_raw"


class GazeFeed:
    """``/gaze/state`` and the IMU's yaw rate feeding one :class:`pepin.gaze_gate.FrameGate`: a
    node asks :meth:`verdict` by a frame's stamp and counts what it drops (the ``gaze_gate`` flag
    and the ``gate_*`` knobs are the node's own, :data:`pepin.gaze_gate.GAZE_GATE`).

    ``/imu/data_raw`` is subscribed only once ``gate_yaw_dps`` is above zero, so a node with the
    yaw gate off carries exactly the subscriptions it had. Only readings in base_link (the C++
    bridge's) are used: a reading in another frame is counted and ignored, never guessed at.
    """

    def __init__(
        self,
        node: Any,
        *,
        exposure_s: float,
        settle_s: float,
        yaw_dps: float,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._node = node
        self._now = clock
        self.gate = FrameGate(exposure_s=exposure_s, settle_s=settle_s, yaw_dps=yaw_dps)
        self._imu = False
        self.bad_states = 0  # messages that were not a state (counted for the report line)
        self.foreign_imu = 0  # IMU readings outside base_link, ignored
        node.create_subscription(
            String,
            GAZE_STATE_TOPIC,
            self._on_state,
            QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE),
        )
        self._listen_imu()

    def set(self, name: str, value: float) -> None:
        """One of :data:`pepin.gaze_gate.GATE_KNOBS` changed live."""
        if name == "gate_exposure_s":
            self.gate.exposure_s = float(value)
        elif name == "gate_settle_s":
            self.gate.settle_s = float(value)
        elif name == "gate_yaw_dps":
            self.gate.yaw_dps = float(value)
            self._listen_imu()

    def verdict(self, stamp: float) -> str | None:
        """Why the frame stamped ``stamp`` (board seconds) is for nothing, or ``None``."""
        return self.gate.verdict(stamp, self._now())

    def text(self) -> str:
        """The gate's state for the owning node's report line."""
        extra = f", {self.bad_states} unreadable states" if self.bad_states else ""
        if self.foreign_imu:
            extra += f", {self.foreign_imu} IMU readings outside base_link ignored"
        return self.gate.text(self._now()) + extra

    def _listen_imu(self) -> None:
        if self._imu or self.gate.yaw_dps <= 0.0:
            return
        self._node.create_subscription(Imu, IMU_TOPIC, self._on_imu, bridged_qos_profile(IMU_TOPIC))
        self._imu = True

    def _on_state(self, msg: Any) -> None:
        state = GazeState.from_json(msg.data)
        if state is None:
            self.bad_states += 1
            return
        self.gate.observe_state(state, self._now())

    def _on_imu(self, msg: Any) -> None:
        if msg.header.frame_id != BASE_FRAME:
            self.foreign_imu += 1
            return
        self.gate.observe_yaw(stamp_seconds(msg.header.stamp), float(msg.angular_velocity.z))


def gate_counts(counts: Mapping[str, int], frames: int) -> str:
    """``12 blind, 3 spinning of 300 frames``: what a node's gate dropped in a report window,
    from its tally's ``gaze_blind`` and ``gaze_spinning`` counts."""
    blind, spinning = counts.get("gaze_blind", 0), counts.get("gaze_spinning", 0)
    return f"{blind} blind, {spinning} spinning of {frames} frames"


__all__ = ["GazeFeed", "gate_counts"]
