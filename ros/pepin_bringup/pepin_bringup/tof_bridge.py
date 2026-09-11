"""ROS 2 node: the board's three VL53L1X sensors as sensor_msgs/Range, plus their frames.

The lidar sees one horizontal slice of the room; these three look where it
cannot — low and in front, at the height of a shoe, a cable, a cat. The board
publishes all three at ~15 Hz in millimetres and says ``null`` when a sensor
got no return; ROS wants metres and, by convention, ``+inf`` for "nothing
within max_range"; a reading equal to max_range means "nothing seen", which is
what Nav2's range layer clears the cone on.

The mounts are read from config/tof.json through :class:`pepin.mounts.Mounts` (measured
2026-09-04, +-1 cm; a parameter may still nudge one) and published once as static transforms
from base_link, so a range in ``tof_left`` lands in the right place without anyone having to
know where the shelf is.
"""

from __future__ import annotations

import contextlib
import math
import queue
import time
from typing import Any

from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import Range
from tf2_ros import StaticTransformBroadcaster

from pepin.footprint import CONTACT_BAND_M
from pepin.mounts import TOF_FRAME, Mount, Mounts
from pepin.tof_horizon import RangeHold, trusted_max_range
from pepin_bringup.link import JsonLineLink
from pepin_bringup.msgs import transform_from_rpy
from pepin_bringup.node_kit import spin_main
from pepin_bringup.protocol import TOF_NAMES, parse_tof, parse_tof_status

# VL53L1X: a ~27 deg cone, 4 cm dead zone, 1.3 m in short mode (the mode the board runs).
_FIELD_OF_VIEW_RAD = 0.47
# Readings inside the contact band are dropped by the range layer (below min_range): a printer
# 5 cm from the bumper is what the cart parked against, not a wall to refuse.
_MIN_RANGE_M = CONTACT_BAND_M
_MAX_RANGE_M = 1.3
_CROSSTALK_M = 0.12  # nearer than this is the sensor seeing its own surroundings

_DRAIN_HZ = 15.0  # readings come at ~15 Hz; a faster timer only burns the A53
_SILENCE_WARN_S = 20.0  # a sensor with nothing valid for this long is reported, not trusted
_HOLD_S = 1.2  # a real return is held this long after it stops: see pepin.tof_horizon.RangeHold
_STATUS_REPORT_S = 15.0  # how often the run's log gets the raw sensor verdicts
_QUEUE_MAX = 100


def _status_key(item: tuple[int | None, int]) -> tuple[int, int]:
    """Sort statuses with the unknown one last."""
    status, _count = item
    return (1, 0) if status is None else (0, status)


class TofBridge(Node):
    """Bridges the ToF server to ROS: /tof/front, /tof/left, /tof/right and their static frames."""

    def __init__(self) -> None:
        """Declare parameters, publish the sensor frames, and start reading the ToF server."""
        super().__init__("tof_bridge")
        host = str(self.declare_parameter("host", "127.0.0.1").value)
        port = int(self.declare_parameter("port", 3335).value)
        self._base_frame = str(self.declare_parameter("base_frame", "base_link").value)

        self._range_pubs = {
            name: self.create_publisher(Range, f"tof/{name}", 10) for name in TOF_NAMES
        }
        self._readings: queue.Queue[tuple[dict[str, float | None], dict[str, int | None]]] = (
            queue.Queue(maxsize=_QUEUE_MAX)
        )
        # Each sensor is believed only as far as its cone stays off the floor: the two low ones
        # graze the carpet at 0.67 m, and the right sensor's steady 0.60-0.70 m returns (with
        # nothing there for the lidar) were being marked into the costmap as a wall, 2026-09-08.
        # The mounts are resolved once, here, so the ceiling and the published frame agree: the
        # ceiling used to come from the hard-coded height while the frame came from the
        # parameter, and overriding one moved the other.
        measured = Mounts.load().tof
        self._mounts = {
            name: self._resolve_mount(name, measured.get(name, Mount())) for name in TOF_NAMES
        }
        self._ceiling = {
            name: trusted_max_range(self._mounts[name][2], _FIELD_OF_VIEW_RAD, _MAX_RANGE_M)
            for name in TOF_NAMES
        }
        self._hold = RangeHold(_HOLD_S)
        self._static_tf = StaticTransformBroadcaster(self)
        self._static_tf.sendTransform([self._mount_transform(name) for name in TOF_NAMES])
        self._last_valid = dict.fromkeys(TOF_NAMES, time.monotonic())  # judged from startup
        self._warned = dict.fromkeys(TOF_NAMES, False)
        self._status_counts: dict[str, dict[int | None, int]] = {n: {} for n in TOF_NAMES}
        self.create_timer(_STATUS_REPORT_S, self._report_status)
        self.get_logger().info(
            "tof ceilings: " + ", ".join(f"{n} {self._ceiling[n]:.2f} m" for n in TOF_NAMES)
        )

        self._link = JsonLineLink(host, port, self._enqueue_ranges, name="tof server")
        self._link.start()
        self.create_timer(1.0 / _DRAIN_HZ, self._publish_pending)

    def close(self) -> None:
        """Close the link, on the way out."""
        self._link.stop()

    def _resolve_mount(self, name: str, measured: Mount) -> tuple[float, float, float, float]:
        """The mount of sensor ``name`` as ``(x_m, y_m, z_m, yaw_rad)``: what config/tof.json
        measured, each number overridable by a parameter (``left_z`` and the like)."""
        return (
            float(self.declare_parameter(f"{name}_x", measured.x_m).value),
            float(self.declare_parameter(f"{name}_y", measured.y_m).value),
            float(self.declare_parameter(f"{name}_z", measured.z_m).value),
            float(self.declare_parameter(f"{name}_yaw", math.radians(measured.yaw_deg)).value),
        )

    def _mount_transform(self, name: str) -> Any:
        """Where sensor ``name`` sits on the robot, as a base_link -> tof_<name> transform."""
        x, y, z, yaw = self._mounts[name]
        return transform_from_rpy(
            self._base_frame,
            TOF_FRAME.format(name=name),
            (x, y, z),
            (0.0, 0.0, yaw),
            self.get_clock().now().to_msg(),
        )

    def _enqueue_ranges(self, message: dict[str, Any]) -> None:
        """Reader thread: hand one line of ranges to the ROS thread, dropping it if it is behind."""
        statuses = parse_tof_status(message)
        for name, status in statuses.items():
            counts = self._status_counts[name]
            counts[status] = counts.get(status, 0) + 1
        with contextlib.suppress(queue.Full):
            self._readings.put_nowait((parse_tof(message), statuses))

    def _report_status(self) -> None:
        """Put the raw VL53L1X verdicts in the run's own log, so a dead sensor is visible there.

        0 = measured, 1 = sigma too high, 2 = signal too weak (usually "nothing in range"),
        4 = out of bounds, 7 = wraparound, none = the sensor did not answer at all.
        """
        report = []
        for name in TOF_NAMES:
            counts = self._status_counts[name]
            total = sum(counts.values()) or 1
            share = ", ".join(
                f"{status}:{100 * n // total}%"
                for status, n in sorted(counts.items(), key=_status_key)
            )
            report.append(f"{name} [{share}]")
            self._status_counts[name] = {}
        self.get_logger().info("tof status " + "; ".join(report))

    def _publish_pending(self) -> None:
        """ROS thread: publish every reading the reader queued, then report link changes."""
        while True:
            try:
                ranges, statuses = self._readings.get_nowait()
            except queue.Empty:
                break
            for name, distance_m in ranges.items():
                self._publish_range(name, distance_m, statuses.get(name))
        self._log_link_status()

    def _publish_range(self, name: str, distance_m: float | None, status: int | None) -> None:
        """One sensor's reading as a sensor_msgs/Range; no return becomes its own ``max_range``.

        A reading past the sensor's floor horizon is reported as "nothing seen" rather than as an
        obstacle: past that distance the cone is looking at the carpet.
        """
        message = Range()
        # Stamped 60 ms ago: the reading is at least that old (tof server -> TCP -> here), and
        # a stamp behind the newest odom->base_link transform never makes the costmap wait.
        message.header.stamp = (self.get_clock().now() - Duration(seconds=0.06)).to_msg()
        message.header.frame_id = TOF_FRAME.format(name=name)
        message.radiation_type = Range.INFRARED
        message.field_of_view = _FIELD_OF_VIEW_RAD
        message.min_range = _MIN_RANGE_M
        message.max_range = self._ceiling[name]
        # Below 12 cm the VL53L1X reports crosstalk from whatever sits at its window (the front
        # sensor flickered 0.05 <-> 1.3 m with nothing there, 2026-09-06); such a reading is
        # published below min_range, which the costmap layer drops: neither a mark nor a clear.
        now = time.monotonic()
        if status is None or status == 255:
            # A sensor that has left the bus is not an empty room: below min_range the layer
            # neither marks nor clears, which is the honest "I do not know".
            message.range = -1.0
        elif distance_m is not None and distance_m < _CROSSTALK_M:
            message.range = -1.0  # below min_range: the layer neither marks nor clears
        else:
            message.range = self._hold.publish(name, distance_m, self._ceiling[name], now)
            if distance_m is not None and distance_m <= self._ceiling[name]:
                self._last_valid[name] = now
                self._warned[name] = False
        self._range_pubs[name].publish(message)
        self._warn_if_silent(name)

    def _warn_if_silent(self, name: str) -> None:
        """Say once when a sensor has produced no valid measurement for a long time.

        It cannot be silenced automatically — a sensor staring at an empty room reports nothing
        valid too, and its 'nothing there' is what clears the costmap. But a dead one looks
        exactly like this in the log (front: 801 of 802 frames invalid, 2026-09-08), so the run's
        own log must say it.
        """
        idle = time.monotonic() - self._last_valid[name]
        if idle > _SILENCE_WARN_S and not self._warned[name]:
            self._warned[name] = True
            self.get_logger().warning(
                f"tof {name}: nothing inside its trusted range ({self._ceiling[name]:.2f} m) "
                f"for {idle:.0f} s — it can only clear the costmap, never mark it"
            )

    def _log_link_status(self) -> None:
        """Say it once whenever the link comes up or goes down."""
        change = self._link.take_status_change()
        if change is None:
            return
        connected, detail = change
        if connected:
            self.get_logger().info(detail)
        else:
            self.get_logger().warning(detail)


def main(args: list[str] | None = None) -> None:
    """Entry point: spin the bridge until it is interrupted."""
    spin_main(TofBridge, args)


if __name__ == "__main__":
    main()
