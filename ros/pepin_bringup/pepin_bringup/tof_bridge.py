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
message — carries two unfixed defects that have each taken this robot's navigation down, and
the second one cannot be held off by any publisher:

1. it calls ``canTransform(..., the message's stamp, timeout=transform_tolerance)`` once per
   message, and tf2 blocks the WHOLE timeout on any failure, so a broken chain buries
   ``updateMap()`` in a backlog it never drains (the ``tf_gate`` below, and the paragraphs
   after it);
2. ``range_sensor_layer.cpp:362-369`` (copy in scratch/nav2_hang/src/) clamps its cell bounds
   with ``bx0 = std::max(0, bx0); bx1 = std::min(size_x_, bx1)`` and then walks
   ``for (unsigned int x = bx0; x <= (unsigned int)bx1; x++)``. When the cone lies entirely
   left of or below the grid, ``bx1``/``by1`` stay NEGATIVE, the cast makes them about 4e9, and
   the loop runs for ever holding the costmap's mutex: one thread at 100 %, "Pose Goes Off
   Grid", every service timing out, zero plans. It takes a jump of the pose in ``map`` between
   a Range's stamp and the costmap update — a tracker restart, a relocalisation, the cart
   carried by hand — and it was reproduced on this robot on 2026-09-21 with
   ``ros/thin.sh kick relocalizer`` (tid 191 of the Nav2 container: 415 s of CPU in 700 s).
   The jump happens AFTER the reading has left, so this bridge cannot gate it.

An ``ObstacleLayer`` has neither defect: its MessageFilter drops what it cannot place instead
of blocking on it, its queue is bounded, and it walks the points it was given rather than a
cell rectangle. What it costs is that a cone must arrive as points — hence the fan, one beam
every cell-width of arc at the sensor's ceiling (:func:`pepin.tof_horizon.cone_beams`), every
beam carrying the one distance the sensor measured, because a whisker cannot say where across
its 27 degrees the thing stands and the honest mark is the whole arc. The old layers stay in
ros/params/nav2_params.yaml, unlisted; ``range_as:=range`` and one line of ``plugins:`` put
them back (CLAUDE.md rule 19).

The mounts are read from config/tof.json through :class:`pepin.mounts.Mounts` (measured
2026-09-04, +-1 cm; a parameter may still nudge one) and published once as static transforms
from base_link, so a range in ``tof_left`` lands in the right place without anyone having to
know where the shelf is.

THE BRIDGE OWNS THE PRECONDITION OF ITS OWN MESSAGES (2026-09-21). A Range whose frame cannot
be placed in the global frame is not a harmless message: Nav2's ``RangeSensorLayer`` calls
``canTransform(global_frame, frame, THE MESSAGE'S STAMP, timeout=transform_tolerance)`` once
per message, and tf2's ``canTransform`` blocks the WHOLE timeout on any failure — a missing
frame, a stamp outside the cache, a chain that does not resolve — with no short-circuit
(scratch/nav2_hang/src/buffer.cpp:121-130, range_sensor_layer.cpp:220-232). Three ToF layers
per costmap, 15 Hz each, tolerance 0.3 s local and 1.0 s global: the per-cycle amplification
is 4.5x and 15x, so any window in which readings flow while the chain is broken buries
``updateMap()`` in a backlog it can never drain — the messages age out of the 10 s TF cache
before they are reached, the costmap mutex is held, ``Costmap2DROS::start()`` never returns
and ``planner_server`` hangs in *Activating* (4 of 7 board starts on 2026-09-21;
scratch/nav2_hang/wedge_gain.py has the arithmetic and the measurement). ``enabled: false``
does not help — the drain runs before the check — and the tolerance is shared with
``getRobotPose`` and is not live. So the two halves of that precondition are held here, at the
one publisher, behind two live flags (:data:`FLAGS`, ``ros/flags.sh set tof_bridge ...``):

``dynamic_mounts``
    besides the static broadcast, each sensor's ``base_link -> tof_<name>`` goes out on ``/tf``
    with the Range's own stamp, beside that Range. A static transform is sent once, and a
    late-joining subscriber that matched a moment too early or too late never gets it (Nav2's
    container did not see the ToF mounts for 157 s on 2026-09-21); a transform republished with
    every reading cannot be missed.

``tf_gate``
    a Range leaves only while the chain that will have to place it is alive and fresh. THE
    RULE, and why it is this one: the gate asks this node's OWN TF buffer for
    ``map <- base_link`` at tf2's "latest" (time 0, timeout 0 — never a blocking wait), which
    resolves to the chain's latest common time and carries it in the answer's own stamp, and it
    opens when that moment is no more than ``tf_gate_max_lag_s`` behind the reading's stamp.
    Two deliberate choices. (1) ``base_link``, not ``tof_<name>``: the mount is THIS node's own
    transform, published here and suppressed here when the gate shuts, so a gate that judged
    itself on it would shut once and never reopen — a guard must never be fed by what it
    suppresses. (2) The latest common time, not the reading's exact stamp: the reading is
    stamped :data:`_STAMP_LAG_S` behind now while the EKF stamps ``odom -> base_link`` at its
    own 20 Hz (ros/params/ekf.yaml:26), so an exact-stamp lookup would fail on the ordinary gap
    between two filter ticks and throw away healthy readings. In health the chain is AHEAD of
    the reading — ``map -> odom`` is dated 0.1 s into the future and re-sent every 0.05 s
    (relocalizer.py:980, :1186, :2042-2058) and the EKF's newest stamp is at most one 0.05 s
    period old — so the measured lag sits at or below zero and essentially nothing is withheld.
    Silence is safe on the other side too: every ToF layer runs without a rate to live up to
    (``expected_update_rate: 0.0`` on the scan sources, ``no_readings_timeout: 0.0`` on the old
    range layers, ros/params/nav2_params.yaml), so a topic that stops says nothing rather than
    stalling a costmap — which is the whole reason withholding is a legitimate answer here.

``range_as``
    ``scan`` (the default) publishes each cone on ``/tof/<name>/scan`` as well, for the
    ObstacleLayer that feeds Nav2 now; ``range`` is the node as it was, the Range topics alone.
    The sensor_msgs/Range is published EITHER WAY and unchanged: it is what the run recorder
    tapes (run_recorder.py:261) and what Foxglove draws, and the three of them together cost
    less than one lidar revolution.

With every flag off the node behaves exactly as it did before: the static mounts alone, every
reading published whatever TF says, and nothing but the three Range topics.
"""

from __future__ import annotations

import contextlib
import math
import queue
import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Protocol

from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan, Range
from tf2_ros import Buffer, StaticTransformBroadcaster, TransformBroadcaster, TransformListener

from pepin.flags import Flag, FlagSet
from pepin.footprint import CONTACT_BAND_M
from pepin.mounts import TOF_FRAME, Mount, Mounts
from pepin.tof_horizon import RangeHold, cone_beams, trusted_max_range
from pepin_bringup.link import JsonLineLink
from pepin_bringup.msgs import scan_from_ranges, transform_from_rpy
from pepin_bringup.node_kit import Switches, TfLookup, spin_main, stamp_seconds
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
_GLOBAL_FRAME = "map"  # the frame a costmap places a Range in; the ``global_frame`` parameter
_TF_CACHE_S = 2.0  # the gate asks only about the newest moment; the board pays for the rest
_GATE_MAX_LAG_S = 0.5  # see the tf_gate_max_lag_s flag for the arithmetic behind the number
_GATE_DETAIL_CHARS = 90  # how much of tf2's own complaint fits in a report line

# The live flags (CLAUDE.md rule 19), declared last in __init__ so the kit's callback sees no
# other declaration, and printed in every report line. Every default is the 2026-09-21 answer to
# the two RangeSensorLayer defects (the module docstring), and with all of them the other way
# this node is byte for byte the one that shipped before them.
FLAGS = FlagSet(
    Flag(
        "dynamic_mounts",
        True,
        description="each sensor's mount base_link -> tof_<name> is published on /tf with the"
        " Range's own stamp, beside that Range, as well as once on /tf_static; off: the static"
        " broadcast alone, which is every consumer's only chance to learn the frame",
        why="a static transform is published ONCE, and under rmw_zenoh a subscriber that matched"
        " a moment too early or too late never gets it: on 2026-09-21 Nav2's container did not"
        " see the tof_* mounts for 157 s on 4 of 7 board starts (scratch/nav2_hang/timeline.py on"
        " board_full_1550.log), and re-sending the static transforms for the first 120 s did not"
        " cure it — any window longer than a second is enough. A missing frame is what makes"
        " RangeSensorLayer block its whole transform_tolerance per message (4.5x local, 15x"
        " global amplification, scratch/nav2_hang/wedge_gain.py) until updateMap never returns"
        " and planner_server hangs in Activating. A transform republished with every reading"
        " cannot be missed by a late joiner, and it costs three TransformStamped at 15 Hz",
        on_when="always on this robot: every consumer of /tof/* needs the mount to place a cone,"
        " and nothing else publishes that edge",
        off_when="to reproduce the pre-2026-09-21 node, or if another publisher ever owns"
        " base_link -> tof_* — two publishers of one edge fight. The static broadcast keeps"
        " running either way, so the frames still exist",
    ),
    Flag(
        "tf_gate",
        True,
        description="a Range leaves only while map <- base_link resolves in this node's own TF"
        " buffer right now and its latest common time is no more than tf_gate_max_lag_s behind"
        " the reading; off: every reading is published whatever TF says",
        why="a Range published while its chain is broken is not a lost measurement, it is a wedge:"
        " tf2's canTransform blocks the FULL timeout on any failure and RangeSensorLayer calls it"
        " once per message with the message's own stamp, so 15 Hz against a 0.3 s (local) / 1.0 s"
        " (global) tolerance means 4.5 and 15 messages arrive per message drained, the backlog"
        " passes the 10 s TF cache, every message then fails although TF is healthy, and the"
        " costmap's first update never ends — 4 of 7 board starts on 2026-09-21, measured at"
        " exactly 1/tolerance (3.262/s and 0.996/s) for 600 s (scratch/nav2_hang/wedge_gain.py,"
        " range_drain.py). The check here is the cheap opposite of that one: tf2 with timeout 0,"
        " which answers from the buffer and never waits. In health it withholds nothing — the"
        " chain is AHEAD of the reading (map -> odom dated 0.1 s ahead, relocalizer.py:2042; the"
        " EKF at 20 Hz, ros/params/ekf.yaml:26; the reading stamped 0.06 s behind now)",
        on_when="always while Nav2's costmaps read /tof/*: it is the one place that can refuse to"
        " feed the wedge, and a closed gate says so in every report line",
        off_when="to reproduce the pre-2026-09-21 node, or when the ToF cones must be watched in"
        " Foxglove with no tracker running at all (no map -> odom, so the gate would be shut and"
        " nothing would reach the topic)",
    ),
    Flag(
        "tf_gate_max_lag_s",
        _GATE_MAX_LAG_S,
        range=(0.0, 10.0),
        description="how far behind a reading's stamp the newest moment of map <- base_link may"
        " be and still let that reading out; 0 demands a chain at least as new as the reading",
        why="both links of the chain are published at 20 Hz — map -> odom every 0.05 s dated 0.1 s"
        " ahead (relocalizer.py:1186, :2055) and odom -> base_link by the EKF at frequency 20.0"
        " (ros/params/ekf.yaml:26), stamped at the filter's own time — while a reading is stamped"
        " 0.06 s behind now and drained at 15 Hz. So in health the newest moment of the chain is"
        " within one filter period of the reading and the measured lag sits at or below zero;"
        " 0.5 s is ten missed periods of both publishers at once, which is a publisher that"
        " stopped (a tracker restarting, an EKF whose sources all went quiet), not jitter. It is"
        " also well inside tf2's 10 s cache, so nothing is withheld for a reason that would have"
        " healed by itself a moment later",
        on_when="raise it on a board so loaded that healthy readings are withheld — the report"
        " line's withheld counts and the lag it prints are the measurement to raise it by",
        off_when="lower it towards 0.1 s to prove that a suspected wedge is a stale chain: the"
        " gate then shuts on exactly the windows the costmap would have blocked in",
    ),
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
        "wedge_gain.py) — that one the tf_gate above holds off. Two: after clamping its cell"
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


@dataclass(frozen=True)
class ChainState:
    """What TF says about one chain right now: ``lag_s`` is how far behind the reading the newest
    moment the chain covers is (negative when the chain is ahead of it, which is the healthy
    case), or ``None`` with tf2's own words in ``detail`` when the chain does not resolve."""

    lag_s: float | None
    detail: str = ""


class ChainProbe(Protocol):
    """The one question the bridge asks TF before it lets a reading out, and it must not block.

    A protocol so the node is driven in a unit test by a four-line fake, and the only piece that
    needs tf2 at all is :class:`TfChainProbe`.
    """

    def chain(self, frame: str, stamp_s: float) -> ChainState:
        """The state of ``<global frame> <- frame`` in the prober's own buffer right now, judged
        against a reading stamped ``stamp_s`` (seconds). Answers from what is already there."""
        ...


class TfChainProbe:
    """:class:`ChainProbe` over tf2: a short buffer fed by a listener on the node's own executor.

    The buffer keeps ``cache_s`` of history because the gate only ever asks about the newest
    moment, and every second kept is board memory. The listener runs WITHOUT a thread of its own
    (``spin_thread=False``): this node already spins, and a second executor buys nothing on four
    A53 cores. The lookup is tf2's "latest" (time 0) with timeout 0, which resolves to the
    chain's latest common time and carries that moment in the answer's own stamp — so one
    non-blocking call says both whether the chain exists and how fresh it is.
    """

    def __init__(
        self, node: Any, global_frame: str = _GLOBAL_FRAME, cache_s: float = _TF_CACHE_S
    ) -> None:
        """Start the listener and the buffer that answer for ``global_frame <- ...``."""
        self.global_frame = global_frame
        self._detail = ""
        buffer = Buffer(cache_time=Duration(seconds=cache_s))
        self._listener = TransformListener(buffer, node, spin_thread=False)
        self._lookup = TfLookup(node, buffer=buffer, on_failure=self._failed)

    def chain(self, frame: str, stamp_s: float) -> ChainState:
        """:class:`ChainProbe`: the chain's lag behind a reading stamped ``stamp_s``, or the
        reason tf2 gives for having no chain at all."""
        self._detail = ""
        transform = self._lookup.transform(self.global_frame, frame, None, 0.0)
        if transform is None:
            return ChainState(None, self._detail or f"{self.global_frame} <- {frame}: no answer")
        return ChainState(stamp_s - stamp_seconds(transform.header.stamp))

    def _failed(self, kind: str, text: str) -> None:
        """tf2's own complaint, kept for the gate's report line."""
        self._detail = f"{kind}: {text}"


class Gate:
    """The bridge's precondition as a switch with a memory: open or shut, since when, why, and
    how many readings it has withheld — everything the report line and the log need.

    Pure bookkeeping over what a :class:`ChainProbe` answered, so the decision is unit-testable
    with no ROS and no clock of its own. It starts SHUT: nothing has been proven yet, and the
    first reading judged opens it within one drain period when the chain is up.
    """

    def __init__(
        self, names: Iterable[str], max_lag_s: float = _GATE_MAX_LAG_S, now_s: float = 0.0
    ) -> None:
        """A gate that counts the readings of ``names`` it withholds, born shut at monotonic
        ``now_s`` — so "shut for N s" counts from the node's start, not from the clock's zero."""
        self.max_lag_s = max_lag_s
        self.open = False
        self.since_s = now_s
        self.detail = "no reading judged yet"
        self.lag_s: float | None = None
        self.withheld = dict.fromkeys(names, 0)  # this report period
        self.withheld_total = dict.fromkeys(names, 0)  # since the node started
        self._change: str | None = None

    def judge(self, state: ChainState, now_s: float) -> bool:
        """Take one probe answer at monotonic ``now_s``: True when a reading may leave. A change
        of state is remembered for :meth:`take_change`, so the log says it once."""
        self.lag_s = state.lag_s
        allowed = state.lag_s is not None and state.lag_s <= self.max_lag_s
        if state.lag_s is None:
            self.detail = state.detail
        elif allowed:
            self.detail = f"lag {state.lag_s:+.2f} s"
        else:
            self.detail = f"the chain is {state.lag_s:.2f} s behind the reading"
        if allowed != self.open:
            self._change = self._transition(allowed, now_s)
            self.open, self.since_s = allowed, now_s
        return allowed

    def withhold(self, name: str) -> None:
        """One reading of ``name`` the gate did not let out."""
        self.withheld[name] = self.withheld.get(name, 0) + 1
        self.withheld_total[name] = self.withheld_total.get(name, 0) + 1

    def take_change(self) -> str | None:
        """The one line to log about the last change of state, once; ``None`` when nothing
        changed — the link's own habit (:class:`pepin_bringup.link.JsonLineLink`)."""
        change, self._change = self._change, None
        return change

    def report(self, now_s: float, enabled: bool) -> str:
        """The gate for the node's report line, and a fresh counting period starts: its state and
        for how long, the lag or the reason, and what it withheld in this period and since the
        node started. Never silent — a shut gate that said nothing would look exactly like three
        dead sensors."""
        held = " ".join(f"{name} {n}" for name, n in self.withheld.items())
        total = sum(self.withheld_total.values())
        self.withheld = dict.fromkeys(self.withheld, 0)
        since = f" (since start {total})" if total else ""
        if not enabled:
            return f"gate off, every reading leaves; withheld {held}{since}"
        state = "open" if self.open else "SHUT"
        return (
            f"gate {state} for {max(now_s - self.since_s, 0.0):.0f} s"
            f" ({self.detail[:_GATE_DETAIL_CHARS]}); withheld {held}{since}"
        )

    def _transition(self, opening: bool, now_s: float) -> str:
        """The sentence for the log when the gate moves, with how long it stood the other way."""
        stood = f"{max(now_s - self.since_s, 0.0):.0f} s"
        if opening:
            held = sum(self.withheld_total.values())
            return (
                f"tf gate open: {self.detail}; it was shut for {stood} and withheld {held}"
                " readings — the chain that places a cone is back"
            )
        return (
            f"tf gate SHUT after {stood} open: {self.detail}. No /tof/* message leaves while the"
            " chain is broken — a Range Nav2 cannot place blocks its range layer for a whole"
            " transform_tolerance and wedges the costmap (scratch/nav2_hang/)"
        )


class TofBridge(Node):
    """Bridges the ToF server to ROS: /tof/front, /tof/left, /tof/right, the scan fan of each
    (/tof/<name>/scan, with ``range_as`` at ``scan``) and the three sensor frames."""

    def __init__(self, probe: ChainProbe | None = None) -> None:
        """Declare parameters, publish the sensor frames, and start reading the ToF server.

        ``probe`` is the gate's window on TF; left out, the node builds the real one
        (:class:`TfChainProbe`) the first time the gate needs it — with ``tf_gate`` off it
        builds none at all, so the node costs exactly what it cost before the gate existed.
        """
        super().__init__("tof_bridge")
        host = str(self.declare_parameter("host", "127.0.0.1").value)
        port = int(self.declare_parameter("port", 3335).value)
        self._base_frame = str(self.declare_parameter("base_frame", "base_link").value)
        # The frame the costmaps place a cone in, and so the frame the gate judges the chain to.
        self._global_frame = str(self.declare_parameter("global_frame", _GLOBAL_FRAME).value)

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
        # The mounts go out ONE way at a time. With ``dynamic_mounts`` (the default) they ride on
        # /tf with every reading's own stamp, which no late joiner can miss. With the flag off
        # they are the static broadcast this node always made — sent once and then again, once a
        # second, for the first ``static_tf_resend_s`` seconds (under rmw_zenoh a subscriber that
        # matched a moment too early or too late never gets a one-shot: Nav2's container did not,
        # on 4 of 7 board starts of 2026-09-21); 0 is the single shot it was.
        # NEVER BOTH: tf2 re-allocates a frame's cache whenever the same edge arrives as the
        # other kind (static <-> dynamic), wiping its history; a lookup that lands just after
        # sees a one-sample cache and throws NoDataForExtrapolationException, which Nav2's
        # RangeSensorLayer does not catch — the whole Nav2 container aborted that way 66 s into
        # a start on 2026-09-21, while the re-send and the dynamic mounts ran side by side.
        self._static_resend_s = float(
            self.declare_parameter("static_tf_resend_s", _STATIC_TF_RESEND_S).value
        )
        self._static_tf: StaticTransformBroadcaster | None = None
        self._static_resend: Any = None
        self._static_resend_until = 0.0
        self._mount_tf = TransformBroadcaster(self)
        self._last_valid = dict.fromkeys(TOF_NAMES, time.monotonic())  # judged from startup
        self._warned = dict.fromkeys(TOF_NAMES, False)
        self._status_counts: dict[str, dict[int | None, int]] = {n: {} for n in TOF_NAMES}
        # The switches are built after the LAST ordinary declare_parameter: the kit's callback
        # runs on declarations too and refuses every name that is not a flag.
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        if not self._switches.on("dynamic_mounts"):
            self._start_static_mounts()
        self._gate = Gate(
            TOF_NAMES, float(self._switches["tf_gate_max_lag_s"]), now_s=time.monotonic()
        )
        self._probe = probe
        if self._probe is None and self._switches.on("tf_gate"):
            self._probe = self._build_probe()
        self.create_timer(_STATUS_REPORT_S, self._report_status)
        self.get_logger().info(
            "tof ceilings: "
            + ", ".join(
                f"{n} {self._ceiling[n]:.2f} m ({self._fan[n][2]} beams)" for n in TOF_NAMES
            )
            + f"; gate chain {self._global_frame} <- {self._base_frame}"
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

    def _build_probe(self) -> ChainProbe:
        """The gate's window on TF: a short buffer and a listener on this node's own executor."""
        self.get_logger().info(
            f"tf gate: watching {self._global_frame} <- {self._base_frame}, at most"
            f" {self._switches['tf_gate_max_lag_s']:.2f} s behind a reading"
        )
        return TfChainProbe(self, self._global_frame, _TF_CACHE_S)

    def _on_switch(self, name: str, old: Any, new: Any) -> None:
        """A flag moved: the gate takes its new bound, and a gate turned on gets its listener."""
        if name == "dynamic_mounts":
            if new:
                self._stop_static_mounts()
            else:
                self._start_static_mounts()
        elif name == "tf_gate_max_lag_s":
            self._gate.max_lag_s = float(new)
        elif name == "tf_gate" and new and self._probe is None:
            self._probe = self._build_probe()

    def _resolve_mount(self, name: str, measured: Mount) -> tuple[float, float, float, float]:
        """The mount of sensor ``name`` as ``(x_m, y_m, z_m, yaw_rad)``: what config/tof.json
        measured, each number overridable by a parameter (``left_z`` and the like)."""
        return (
            float(self.declare_parameter(f"{name}_x", measured.x_m).value),
            float(self.declare_parameter(f"{name}_y", measured.y_m).value),
            float(self.declare_parameter(f"{name}_z", measured.z_m).value),
            float(self.declare_parameter(f"{name}_yaw", math.radians(measured.yaw_deg)).value),
        )

    def _start_static_mounts(self) -> None:
        """The static way (``dynamic_mounts`` off): the three mounts on /tf_static now, and again
        once a second while the re-send window lasts."""
        if self._static_tf is None:
            self._static_tf = StaticTransformBroadcaster(self)
        self._static_tf.sendTransform([self._mount_transform(name) for name in TOF_NAMES])
        self._static_resend_until = time.monotonic() + self._static_resend_s
        if self._static_resend is None:
            self._static_resend = self.create_timer(1.0, self._resend_static)
        else:
            self._static_resend.reset()

    def _stop_static_mounts(self) -> None:
        """The dynamic way took over: no further static message leaves this node. (One already
        latched stays latched for late joiners until the node restarts — a flag flipped at run
        time is an A/B, the default start never publishes a static mount at all.)"""
        if self._static_resend is not None:
            self._static_resend.cancel()

    def _resend_static(self) -> None:
        """Send the three mounts again while the start-up window lasts, then stop the timer."""
        if (
            self._static_tf is None
            or self._switches.on("dynamic_mounts")
            or time.monotonic() >= self._static_resend_until
        ):
            self._stop_static_mounts()
            return
        self._static_tf.sendTransform([self._mount_transform(name) for name in TOF_NAMES])

    def _mount_transform(self, name: str, stamp: Any = None) -> Any:
        """Where sensor ``name`` sits on the robot, as a base_link -> tof_<name> transform, at
        ``stamp`` (the reading's own, for the dynamic mount) or at this moment."""
        x, y, z, yaw = self._mounts[name]
        return transform_from_rpy(
            self._base_frame,
            TOF_FRAME.format(name=name),
            (x, y, z),
            (0.0, 0.0, yaw),
            self.get_clock().now().to_msg() if stamp is None else stamp,
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
            "tof status "
            + "; ".join(report)
            + f"; {self._gate.report(time.monotonic(), self._switches.on('tf_gate'))}"
            + f"; flags: {self._switches.state()}"
        )

    def _publish_pending(self) -> None:
        """ROS thread: publish every reading the reader queued, then report link changes.

        One stamp and one gate decision per line of readings: the three sensors are read
        together, the same chain places all three, and a consumer must see the mount and the
        Range it belongs to carrying the same moment.
        """
        while True:
            try:
                ranges, statuses = self._readings.get_nowait()
            except queue.Empty:
                break
            stamp = (self.get_clock().now() - Duration(seconds=_STAMP_LAG_S)).to_msg()
            allowed = self._gate_allows(stamp)
            if allowed and self._switches.on("dynamic_mounts"):
                # The three mounts first and in ONE message, then the readings: a consumer that
                # takes them in that order has every frame in its buffer before it tries to place
                # a cone, and the board pays for 15 TF messages a second instead of 45.
                self._mount_tf.sendTransform([self._mount_transform(n, stamp) for n in ranges])
            for name, distance_m in ranges.items():
                if allowed:
                    self._publish_reading(name, distance_m, statuses.get(name), stamp)
                else:
                    self._withhold(name, distance_m)
                self._warn_if_silent(name)
        self._log_link_status()

    def _gate_allows(self, stamp: Any) -> bool:
        """Whether the chain that will have to place a reading stamped ``stamp`` is alive and
        fresh right now (``tf_gate``); always true with the flag off, and every change of the
        gate's mind is said once in the log."""
        if not self._switches.on("tf_gate") or self._probe is None:
            return True
        state = self._probe.chain(self._base_frame, stamp_seconds(stamp))
        allowed = self._gate.judge(state, time.monotonic())
        change = self._gate.take_change()
        if change is not None:
            # Two call sites on purpose: rclpy refuses one call site that changes its severity
            # between calls (ValueError), and that killed this node the first time the gate shut
            # on the robot (2026-09-21) — nothing a test on the ROS stubs could see.
            if allowed:
                self.get_logger().info(change)
            else:
                self.get_logger().warning(change)
        return allowed

    def _withhold(self, name: str, distance_m: float | None) -> None:
        """A reading the gate did not let out: counted, and the sensor's own liveness clock kept
        running — a shut gate must not be reported as three sensors that went blind."""
        self._gate.withhold(name)
        if distance_m is not None and distance_m <= self._ceiling[name]:
            self._last_valid[name] = time.monotonic()
            self._warned[name] = False

    def _publish_reading(
        self, name: str, distance_m: float | None, status: int | None, stamp: Any
    ) -> None:
        """One sensor's reading on both faces of this bridge at ``stamp`` (the whole line's, so
        it matches the mount sent with it): the sensor_msgs/Range it has always published, and —
        with ``range_as`` at ``scan`` — the same cone as a fan for Nav2's ObstacleLayer.

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
