"""ROS 2 node: the board's three VL53L1X sensors as sensor_msgs/Range and as scan fans, plus
their frames.

The lidar sees one horizontal slice of the room; these three look where it
cannot — low and in front, at the height of a shoe, a cable, a cat. The board
publishes all three at ~15 Hz in millimetres and says ``null`` when a sensor
got no return; ROS wants metres and, by convention, ``+inf`` for "nothing
within max_range"; a reading equal to max_range means "nothing seen", which is
what a costmap clears the cone on.

EACH CONE ALSO LEAVES AS A LaserScan (2026-09-21, the ``range_as`` flag), and that is how Nav2
is fed now: ``/tof/<name>/scan``, a small fan in the sensor's own frame, read by an
``ObstacleLayer``. Nav2 Jazzy's ``RangeSensorLayer`` — the plugin written for exactly this
message — carries two unfixed defects that have each taken this robot's navigation down:

1. it calls ``canTransform(..., the message's stamp, timeout=transform_tolerance)`` once per
   message, and tf2 blocks the WHOLE timeout on any failure, so a broken chain buries
   ``updateMap()`` in a backlog it never drains (4 of 7 board starts on 2026-09-21;
   scratch/nav2_hang/wedge_gain.py has the arithmetic and the measurement);
2. ``range_sensor_layer.cpp:362-369`` (copy in scratch/nav2_hang/src/) clamps its cell bounds
   with ``bx0 = std::max(0, bx0); bx1 = std::min(size_x_, bx1)`` and then walks
   ``for (unsigned int x = bx0; x <= (unsigned int)bx1; x++)``. When the cone lies entirely
   left of or below the grid, ``bx1``/``by1`` stay NEGATIVE, the cast makes them about 4e9, and
   the loop runs for ever holding the costmap's mutex: one thread at 100 %, "Pose Goes Off
   Grid", every service timing out, zero plans. It takes a jump of the pose in ``map`` between
   a Range's stamp and the costmap update — a tracker restart, a relocalisation, the cart
   carried by hand — and it was reproduced on this robot on 2026-09-21 with
   ``ros/thin.sh kick relocalizer`` (tid 191 of the Nav2 container: 415 s of CPU in 700 s).
   The jump happens AFTER the reading has left, so no publisher could hold it off.

An ``ObstacleLayer`` has neither defect: its MessageFilter drops what it cannot place instead
of blocking on it, its queue is bounded, and it walks the points it was given rather than a
cell rectangle. What it costs is that a cone must arrive as points — hence the fan, one beam
every cell-width of arc at the sensor's ceiling (:func:`pepin.tof_horizon.cone_beams`), every
beam carrying the one distance the sensor measured, because a whisker cannot say where across
its 27 degrees the thing stands and the honest mark is the whole arc. The old layers stay in
ros/params/nav2_params.yaml, unlisted; ``range_as:=range`` and one line of ``plugins:`` put
them back (CLAUDE.md rule 19).

THE PUBLISHER DOES NOT GUARD THE CONSUMER (2026-09-22). For one day this node carried two more
switches against defect 1: a ``tf_gate`` that withheld every reading while ``map <- base_link``
did not resolve in a TF buffer of its own, and ``dynamic_mounts`` that put
``base_link -> tof_<name>`` on ``/tf`` beside each reading. Defect 1 left with the range layers
on the very same day, so the gate guarded nothing — and both cost: the rclpy TF listener took
this process from 13 % to 20-37 % of an A53 core (config/board_manifest.json), and a mount
arriving on ``/tf`` with the reading's own stamp made the ObstacleLayer's message filter DROP
every fan ("timestamp on the message is earlier than all the data in the transform cache", 723
drops in one drive), so the whiskers neither marked nor cleared. Both are out; the code stays
reachable on branch ``stereo`` (commits 3d8c966, e4535f3). What a costmap cannot place it now
drops by itself, in microseconds, which is the whole point of the layer the ToF moved to.

The mounts are read from config/tof.json through :class:`pepin.mounts.Mounts` (measured
2026-09-04, +-1 cm; a parameter may still nudge one) and published as static transforms from
base_link — at start and again once a second for ``static_tf_resend_s``, because under
rmw_zenoh a subscriber that matched a moment too early or too late never gets a one-shot
(Nav2's container did not see the ToF mounts for 157 s on 4 of 7 board starts of 2026-09-21).
So a range in ``tof_left`` lands in the right place without anyone having to know where the
shelf is.

One live flag is left (:data:`FLAGS`, ``ros/flags.sh set tof_bridge ...``):

``range_as``
    ``scan`` (the default) publishes each cone on ``/tof/<name>/scan`` as well, for the
    ObstacleLayer that feeds Nav2 now; ``range`` is the node as it was, the Range topics alone.
    The sensor_msgs/Range is published EITHER WAY and unchanged: it is what the run recorder
    tapes (run_recorder.py:261) and what Foxglove draws, and the three of them together cost
    less than one lidar revolution.
"""

from __future__ import annotations

import contextlib
import math
import queue
import time
from typing import Any

from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Range
from tf2_ros import StaticTransformBroadcaster

from pepin.flags import Flag, FlagSet
from pepin.footprint import CONTACT_BAND_M
from pepin.mounts import TOF_FRAME, Mount, Mounts
from pepin.tof_horizon import RangeHold, cone_beams, trusted_max_range
from pepin_bringup.link import JsonLineLink
from pepin_bringup.msgs import scan_from_ranges, transform_from_rpy
from pepin_bringup.node_kit import Switches, spin_main
from pepin_bringup.protocol import TOF_NAMES, parse_tof, parse_tof_status

# VL53L1X: a ~27 deg cone, 4 cm dead zone, 1.3 m in short mode (the mode the board runs).
_FIELD_OF_VIEW_RAD = 0.47
# Readings inside the contact band are dropped before any layer sees them — the fan's own
# range_min, which laser_geometry's projector filters on, and the Range's: a printer 5 cm from
# the bumper is what the cart parked against, not a wall to refuse.
_MIN_RANGE_M = CONTACT_BAND_M
_MAX_RANGE_M = 1.3
_CROSSTALK_M = 0.12  # nearer than this is the sensor seeing its own surroundings
_UNKNOWN_RANGE_M = -1.0  # below any min_range: a reading that must neither mark nor clear
# The cell of the costmap the fan is drawn on (ros/params/nav2_params.yaml, local_costmap's
# resolution): it sets how many beams a cone needs (:func:`pepin.tof_horizon.cone_beams`).
_COSTMAP_CELL_M = 0.05
_SCAN_TOPIC = "tof/{name}/scan"

_DRAIN_HZ = 15.0  # readings come at ~15 Hz; a faster timer only burns the A53
_SILENCE_WARN_S = 20.0  # a sensor with nothing valid for this long is reported, not trusted
_HOLD_S = 1.2  # a real return is held this long after it stops: see pepin.tof_horizon.RangeHold
_STATIC_TF_RESEND_S = 120.0  # how long the static mounts are re-sent after start, seconds
_STATUS_REPORT_S = 15.0  # how often the run's log gets the raw sensor verdicts
_QUEUE_MAX = 100
# A reading is stamped this far behind "now": it is at least that old (sensor -> ToF server ->
# TCP -> here), and a stamp behind the newest odom -> base_link never makes a costmap wait.
_STAMP_LAG_S = 0.06

# The live flags (CLAUDE.md rule 19), declared last in __init__ so the kit's callback sees no
# other declaration, and printed in every report line.
FLAGS = FlagSet(
    Flag(
        "range_as",
        "scan",
        choices=("scan", "range"),
        description="what Nav2 is fed with: scan also publishes each cone on /tof/<name>/scan as"
        " a small LaserScan fan for an ObstacleLayer; range publishes nothing there, which is the"
        " node before 2026-09-21 and needs the three RangeSensorLayer blocks back in the local"
        " costmap's plugins list. The sensor_msgs/Range topics are published either way",
        why="nav2_costmap_2d::RangeSensorLayer carries two defects that are both unfixed on main"
        " and have each stopped this robot. One: it asks tf2 to transform every message at the"
        " message's own stamp with transform_tolerance as the timeout, and tf2 blocks the whole"
        " timeout on any failure, so a broken chain amplifies 4.5x per update cycle until"
        " updateMap never returns (4 of 7 board starts on 2026-09-21; scratch/nav2_hang/"
        "wedge_gain.py). Two: after clamping its cell"
        " bounds to the grid (range_sensor_layer.cpp:362-369) bx1/by1 stay NEGATIVE when the cone"
        " falls off the left or bottom edge, and the loops cast them to unsigned: about 4e9"
        " iterations holding the costmap mutex, one thread at 100 % for ever, 'Pose Goes Off"
        " Grid', every service timing out and zero plans. It takes one jump of the pose in map"
        " between a reading's stamp and the update — a tracker restart, a relocalisation, the"
        " cart carried by hand — and no publisher can gate it, because the jump happens after"
        " the reading has left. Reproduced on 2026-09-21 with ros/thin.sh kick relocalizer (tid"
        " 191 of the Nav2 container: 415 s of CPU in 700 s). An ObstacleLayer drops what it"
        " cannot place instead of blocking on it and walks points instead of a cell rectangle, so"
        " it has neither; the fan is what turns one distance into points, one beam per costmap"
        " cell of arc at the sensor's ceiling (pepin.tof_horizon.cone_beams: 11 beams front, 7"
        " and 7 at the sides), every beam carrying that distance",
        on_when="always while Nav2 reads the ToF: it is the arrangement with no unbounded loop"
        " and no blocking transform in it",
        off_when="to compare against the old plugin, or if an ObstacleLayer ever proves worse at"
        " a cone than the range layer was — put tof_front_layer, tof_left_layer and"
        " tof_right_layer back into the local costmap's plugins list at the same time, or the"
        " whiskers reach no costmap at all",
    ),
)


def _status_key(item: tuple[int | None, int]) -> tuple[int, int]:
    """Sort statuses with the unknown one last."""
    status, _count = item
    return (1, 0) if status is None else (0, status)


class TofBridge(Node):
    """Bridges the ToF server to ROS: /tof/front, /tof/left, /tof/right, the scan fan of each
    (/tof/<name>/scan, with ``range_as`` at ``scan``) and the three sensor frames."""

    def __init__(self) -> None:
        """Declare parameters, publish the sensor frames, and start reading the ToF server."""
        super().__init__("tof_bridge")
        host = str(self.declare_parameter("host", "127.0.0.1").value)
        port = int(self.declare_parameter("port", 3335).value)
        self._base_frame = str(self.declare_parameter("base_frame", "base_link").value)

        self._range_pubs = {
            name: self.create_publisher(Range, f"tof/{name}", 10) for name in TOF_NAMES
        }
        # The fans exist whatever ``range_as`` says: a publisher nobody writes to costs an entry
        # in the graph, and the flag has to be live in BOTH directions — turning it back to
        # ``scan`` on a running robot must put the cones on the wire on the next reading.
        self._scan_pubs = {
            name: self.create_publisher(LaserScan, _SCAN_TOPIC.format(name=name), 10)
            for name in TOF_NAMES
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
        # The fan each cone is drawn as, fixed once from that sensor's own ceiling: enough beams
        # that neighbours land at most one costmap cell apart where the arc is widest, which is
        # at the ceiling (:func:`pepin.tof_horizon.cone_beams`). The cone is symmetric about the
        # sensor's own x axis, and the mount carries the yaw, so the fan needs none.
        self._fan = {name: self._fan_for(name) for name in TOF_NAMES}
        self._hold = RangeHold(_HOLD_S)
        # The mounts are ONE edge published ONE way: base_link -> tof_<name> on /tf_static, sent
        # now and then again, once a second, for the first ``static_tf_resend_s`` seconds (under
        # rmw_zenoh a subscriber that matched a moment too early or too late never gets a
        # one-shot: Nav2's container did not, on 4 of 7 board starts of 2026-09-21); 0 is a
        # single shot. Never also on /tf: tf2 re-allocates a frame's cache whenever the same edge
        # arrives as the other kind (static <-> dynamic), wiping its history, and a lookup that
        # lands just after sees a one-sample cache and throws NoDataForExtrapolationException —
        # the whole Nav2 container aborted that way 66 s into a start on 2026-09-21.
        self._static_resend_s = float(
            self.declare_parameter("static_tf_resend_s", _STATIC_TF_RESEND_S).value
        )
        self._static_tf = StaticTransformBroadcaster(self)
        self._static_resend_until = time.monotonic() + self._static_resend_s
        self._send_static_mounts()
        self._static_resend = self.create_timer(1.0, self._resend_static)
        self._last_valid = dict.fromkeys(TOF_NAMES, time.monotonic())  # judged from startup
        self._warned = dict.fromkeys(TOF_NAMES, False)
        self._status_counts: dict[str, dict[int | None, int]] = {n: {} for n in TOF_NAMES}
        # The switches are built after the LAST ordinary declare_parameter: the kit's callback
        # runs on declarations too and refuses every name that is not a flag.
        self._switches = Switches(self, FLAGS)
        self.create_timer(_STATUS_REPORT_S, self._report_status)
        self.get_logger().info(
            "tof ceilings: "
            + ", ".join(
                f"{n} {self._ceiling[n]:.2f} m ({self._fan[n][2]} beams)" for n in TOF_NAMES
            )
            + f"; flags: {self._switches.state()}"
        )

        self._link = JsonLineLink(host, port, self._enqueue_ranges, name="tof server")
        self._link.start()
        self.create_timer(1.0 / _DRAIN_HZ, self._publish_pending)

    def close(self) -> None:
        """Close the link, on the way out."""
        self._link.stop()

    def _fan_for(self, name: str) -> tuple[float, float, int]:
        """Sensor ``name``'s cone as a fan: ``(angle_min_rad, angle_increment_rad, beams)``,
        symmetric about the sensor's own x axis and spanning the whole field of view."""
        beams = cone_beams(self._ceiling[name], _FIELD_OF_VIEW_RAD, _COSTMAP_CELL_M)
        return -_FIELD_OF_VIEW_RAD / 2.0, _FIELD_OF_VIEW_RAD / (beams - 1), beams

    def _resolve_mount(self, name: str, measured: Mount) -> tuple[float, float, float, float]:
        """The mount of sensor ``name`` as ``(x_m, y_m, z_m, yaw_rad)``: what config/tof.json
        measured, each number overridable by a parameter (``left_z`` and the like)."""
        return (
            float(self.declare_parameter(f"{name}_x", measured.x_m).value),
            float(self.declare_parameter(f"{name}_y", measured.y_m).value),
            float(self.declare_parameter(f"{name}_z", measured.z_m).value),
            float(self.declare_parameter(f"{name}_yaw", math.radians(measured.yaw_deg)).value),
        )

    def _send_static_mounts(self) -> None:
        """The three mounts on /tf_static, as they stand at this moment."""
        self._static_tf.sendTransform([self._mount_transform(name) for name in TOF_NAMES])

    def _resend_static(self) -> None:
        """Send the three mounts again while the start-up window lasts, then stop the timer."""
        if time.monotonic() >= self._static_resend_until:
            self._static_resend.cancel()
            return
        self._send_static_mounts()

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
        self.get_logger().info(
            "tof status " + "; ".join(report) + f"; flags: {self._switches.state()}"
        )

    def _publish_pending(self) -> None:
        """ROS thread: publish every reading the reader queued, then report link changes.

        One stamp per line of readings: the three sensors are read together, and a consumer must
        see every message of one line carrying the same moment.
        """
        while True:
            try:
                ranges, statuses = self._readings.get_nowait()
            except queue.Empty:
                break
            stamp = (self.get_clock().now() - Duration(seconds=_STAMP_LAG_S)).to_msg()
            for name, distance_m in ranges.items():
                self._publish_reading(name, distance_m, statuses.get(name), stamp)
                self._warn_if_silent(name)
        self._log_link_status()

    def _publish_reading(
        self, name: str, distance_m: float | None, status: int | None, stamp: Any
    ) -> None:
        """One sensor's reading on both faces of this bridge at ``stamp`` (the whole line's, so
        the three sensors agree about the moment): the sensor_msgs/Range it has always
        published, and — with ``range_as`` at ``scan`` — the same cone as a fan for Nav2's
        ObstacleLayer.

        The verdict is taken ONCE, here, so the two messages can never disagree about what the
        sensor said.
        """
        value = self._verdict(name, distance_m, status)
        self._publish_range(name, value, stamp)
        if self._switches["range_as"] == "scan":
            self._publish_scan(name, value, stamp)

    def _verdict(self, name: str, distance_m: float | None, status: int | None) -> float:
        """What sensor ``name`` says right now, in metres: the return itself, the last one while
        the hold lasts, its ceiling for "nothing within the trusted range", or
        :data:`_UNKNOWN_RANGE_M` for "I do not know" — a sensor that did not answer, or a
        reading so near that it is the sensor's own window.

        A reading past the sensor's floor horizon counts as "nothing seen" rather than as an
        obstacle: past that distance the cone is looking at the carpet.
        """
        if status is None or status == 255:
            return _UNKNOWN_RANGE_M  # a sensor that has left the bus is not an empty room
        # Below 12 cm the VL53L1X reports crosstalk from whatever sits at its window (the front
        # sensor flickered 0.05 <-> 1.3 m with nothing there, 2026-09-06), and neither is that.
        if distance_m is not None and distance_m < _CROSSTALK_M:
            return _UNKNOWN_RANGE_M
        now = time.monotonic()
        value = self._hold.publish(name, distance_m, self._ceiling[name], now)
        if distance_m is not None and distance_m <= self._ceiling[name]:
            self._last_valid[name] = now
            self._warned[name] = False
        return value

    def _publish_range(self, name: str, value: float, stamp: Any) -> None:
        """The verdict as a sensor_msgs/Range: no return becomes the sensor's own ``max_range``,
        and "I do not know" goes out below ``min_range``, which a costmap neither marks nor
        clears on."""
        message = Range()
        message.header.stamp = stamp
        message.header.frame_id = TOF_FRAME.format(name=name)
        message.radiation_type = Range.INFRARED
        message.field_of_view = _FIELD_OF_VIEW_RAD
        message.min_range = _MIN_RANGE_M
        message.max_range = self._ceiling[name]
        message.range = value
        self._range_pubs[name].publish(message)

    def _publish_scan(self, name: str, value: float, stamp: Any) -> None:
        """The same verdict as a sensor_msgs/LaserScan fan on ``/tof/<name>/scan``, in the
        sensor's own frame — what Nav2's ObstacleLayer marks and clears with.

        Three answers, and the fan says each of them the way that layer reads it. A RETURN: every
        beam carries it, because a whisker cannot say where across its 27 degrees the thing
        stands and the whole arc is the honest mark. NOTHING within the trusted range: every beam
        is ``+inf``, which the layer's ``inf_is_valid`` turns into a clear out to the fan's own
        ``range_max`` (Nav2's laserScanValidInfCallback puts the point at ``range_max`` minus a
        tenth of a millimetre — which is why ``obstacle_max_range`` in the yaml sits a cell below
        it, or that clearing point would MARK a lethal ring at the ceiling). I DO NOT KNOW: every
        beam is NaN, which the projector drops — neither a mark nor a clear, the silence the
        Range's ``-1.0`` has always meant.
        """
        angle_min, increment, beams = self._fan[name]
        ceiling = self._ceiling[name]
        if value < _MIN_RANGE_M:
            beam = math.nan
        elif value >= ceiling:
            beam = math.inf
        else:
            beam = value
        self._scan_pubs[name].publish(
            scan_from_ranges(
                [beam] * beams,
                angle_min,
                increment,
                stamp,
                TOF_FRAME.format(name=name),
                _MIN_RANGE_M,
                ceiling,
            )
        )

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
