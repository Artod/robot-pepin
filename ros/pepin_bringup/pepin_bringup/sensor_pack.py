"""One snapshot of the room per moment, out of whatever sensor was looking: RTAB-Map's SensorData.

WHY THIS NODE EXISTS. RTAB-Map used to subscribe to a SYNCHRONISED TRIPLE — picture, depth,
scan — so the mapper only ever saw a moment all three spoke for. The camera is the slow and
fragile one (a network on the CPU, a neck that moves, a laptop that stalls), and when it stopped
the synchroniser stopped firing: the lidar went on delivering ten revolutions a second into a
mapper that was told nothing. The three "modes" the launch file used to carry — grid from the
scan, grid from the depth, no scan at all — were the same fact said three times: a mode is which
sensors are alive, and that belongs in the data, not in three configs. rtabmap_ros has one input
for exactly this: ``subscribe_sensor_data`` takes a single ``rtabmap_msgs/SensorData`` carrying
picture, depth, optics and scan together, and is mutually exclusive with every other subscription
(rtabmap_sync/CommonDataSubscriber.cpp:453-509 of 0.22.1-jazzy). This node fills it.

ON THE LAPTOP, and that is CLAUDE.md rule 20 rather than convenience: it consumes the camera —
the picture, the depth the network computes from it, the optics — and a feature that only works
with the laptop alive belongs on the laptop by definition. The board keeps its tracker, its
odometry and its reflexes and never waits for this.

WHAT GOES IN ONE SNAPSHOT is :mod:`pepin.snapshot`'s arithmetic, not this node's: each source
measures its own period from the stamps it delivers, the alive source whose newest stamp is
OLDEST drives (the latest moment everybody has spoken for), and every other alive source joins
with its own message nearest that stamp. The snapshot's stamp is the driver's own message stamp —
a BOARD-clock stamp — and never this laptop's now: the board stamps the picture (camera_stream
copies the frame's X-Timestamp) and stamps the scan, the depth carries the picture's stamp bit for
bit, and RTAB-Map looks up ``odom -> base_link`` at exactly the stamp in this message's header
(rtabmap_slam/CoreWrapper.cpp:1905). A laptop stamp there would place every node at the wrong
moment of the drive.

The camera's own three topics are paired by an EXACT stamp and nothing else: depth_stream
publishes ``/camera/depth`` with the picture's ``header.stamp`` and ``frame_id`` unchanged, and
1024 of 1024 depth frames matched a picture bit for bit over 120 s on the wire
(scratch/vo_stamp_pairs.py, 2026-09-14) — the same measurement the visual odometry's exact sync
stands on. The optics are the exception and deliberately so: a ``CameraInfo`` describes the lens,
not the moment, and RTAB-Map reads only its size and matrices out of it
(rtabmap_conversions::cameraModelFromROS), so the newest one whose size matches the PICTURE is
used whatever its stamp.

EVERY FIELD CONVENTION BELOW WAS READ, NOT GUESSED, in rtabmap_ros 0.22.1-jazzy and rtabmap
0.22.1 — the versions installed in ros/Dockerfile.laptop's image, confirmed by
``Parameters.h`` in the image being byte-identical to the 0.22.1 tag:

* ``left`` is the RGB picture and ``right`` is the DEPTH ("For RGB-D, left corresponds to rgb
  camera, and right corresponds to depth camera", rtabmap_msgs/msg/SensorData.msg). ``left``
  accepts mono8/mono16/rgb8/bgr8/bgra8/rgba8, ``right`` accepts 32FC1/16UC1/mono16 when the
  message is NOT stereo (rtabmap_conversions/MsgConversion.cpp:1135-1195). bgr8 and 32FC1, which
  is what camera_stream and depth_stream already publish, go through untouched.
* ``right_camera_info`` MUST STAY EMPTY: ``isStereo`` is exactly ``!msg.right_camera_info.empty()``
  (MsgConversion.cpp:1096), and a stereo message would read our depth image as a right picture.
* ``left_camera_info`` and ``local_transform`` must be the same length, or the camera model is
  dropped in silence (MsgConversion.cpp:1116-1125). One camera: one of each.
* ``left_camera_info`` DESCRIBES THE PICTURE, at the picture's exact size, and is taken verbatim:
  ``sensorDataFromROS`` scales it to neither image (MsgConversion.cpp:1144-1241 calls
  ``cameraModelFromROS`` and then ``setRGBDImage``, with no ``scaled`` and no resize in the path),
  and rtabmap's own reprojection asserts the model against the PICTURE
  (``model.imageHeight()==imageRgb.rows``, rtabmap/core/util3d.cpp:497) while deriving the depth's
  own intrinsics from the integer ratio between the two (util3d.cpp:324-328,402).
* THE DEPTH MAY BE THE PICTURE'S SIZE OR AN INTEGER FACTOR SMALLER, NEVER LARGER, and a violation
  is FATAL rather than warned about: ``Memory.cpp:4579`` is a UASSERT — "For RGB-D, depth can be X
  times smaller than RGB (where X is an integer)" — and ``util3d.cpp:324`` asserts the modulo. So
  both are checked HERE, one hop before RTAB-Map would abort the process, and a frame that fails is
  refused and counted (``no_optics``, ``depth_not_a_factor``). It is not hypothetical: camera_stream
  publishes at a live ``scale`` and depth_stream at the network's own size, so a scale of 0.25 makes
  a 320x180 picture beside a 640x360 depth.
* ``local_transform[i]`` is ``base_link <- the frame of the picture`` (the camera's OPTICAL
  frame), taken from TF at the picture's stamp with no optical rotation added — exactly what the
  old subscription path computed for itself (MsgConversion.cpp:2179).
* Both raw images must be present for either to be used: ``setRGBDImage(left, right, ...)`` is
  called only ``if(!left.empty() && !right.empty())`` (MsgConversion.cpp:1215-1218). A picture
  with no depth is not a camera member.
* ``laser_scan`` is a ``PointCloud2`` of the scan's returns IN THE LASER'S OWN FRAME, and
  ``laser_scan_local_transform`` is ``base_link <- the frame of the scan`` — the arrangement the
  old path built by hand (MsgConversion.cpp:2632-2745).
* ``laser_scan_format`` must equal what RTAB-Map itself infers from the cloud's fields, or it
  ABORTS on an assertion: ``UASSERT((Format)msg.laser_scan_format == s.laserScanRaw().format())``
  (MsgConversion.cpp:1241). The inference reads the field names (rtabmap/core/impl/util3d.hpp
  :37-170, called with its default ``is2D=false``): x and y alone give ``kXY`` = 1, an added z gives
  ``kXYZ`` = 5. This node publishes x and y only (:data:`LASER_SCAN_FORMAT_XY`), which is the
  format the old path produced (MsgConversion.cpp:2726) — and the format is what decides
  everything downstream, because ``LaserScan::is2d()`` reads the FORMAT and not the angles
  (rtabmap/core/LaserScan.h:131 with ``isScan2d``). It is why a 2D scan's whole return set is
  taken as obstacle instead of being sent through the ground-height segmentation
  (rtabmap/core/LocalGridMaker.cpp:264-270); a z of 0.0 would put the lidar's walls 0.383 m up,
  above Grid/MaxGroundHeight, so this is not the difference between working and not — it is the
  difference between the grid the known map was built with and another one.
* ``laser_scan_max_pts`` has to be given here, because the SensorData message carries no angles
  and the constructor that takes none leaves the count at whatever it is handed
  (rtabmap/core/LaserScan.cpp:352-362). The value that reproduces the old path exactly is the
  beam count the angles imply, which is the length of ``ranges``; RegistrationIcp reads it to make
  Icp/CorrespondenceRatio absolute instead of relative. ``laser_scan_max_range`` is the scan's own
  ``range_max``.
* ``header.frame_id`` is not read on the way in at all (``sensorDataFromROS`` uses only
  ``header.stamp``, MsgConversion.cpp:1086-1272). It is set to the base frame because that is what
  RTAB-Map writes there itself when it publishes one.

The one thing the old path had and this cannot: the scan's angles. RTAB-Map's own point order for
an upside-down laser (ours hangs at roll 180, config/lidar.json) was a ``cv::flip`` of the row of
points (MsgConversion.cpp:2729-2735) — the ORDER only, which neither ICP nor the occupancy grid
reads, so it is not reproduced.

WHAT THE SNAPSHOTS CARRY IS SAID OUT LOUD, latched, on :data:`STATE_TOPIC`
(:class:`pepin.snapshot.SnapshotState`). Only this node can answer it — the packer's liveness rule
is here — and RTAB-Map's registration depends on the answer: ICP links a pair of nodes with scans
and can link nothing without one, visual registration links a pair with pictures and nothing
without one, and the pipeline is ONE object for the process
(:class:`pepin.graphmode.StrategyRule`, which pepin_bringup.rtabmap_frame owns). Latched, because a
reader that starts later must not have to wait for the next change to learn the present state.

A MEMBER WAITS FOR ITS TRANSFORM INSTEAD OF BEING DROPPED. Every member has to be placed on the
cart from TF at its own stamp, and the neck's dynamic edge lands tens of milliseconds behind the
frame it belongs to: about 5 % of camera frames used to answer "Lookup would require extrapolation
into the future" and the moment went out as a lidar-only snapshot (measured live 2026-09-18). The
wait for it is no longer a blocking one — a fifth of a second spent inside a subscription callback
is a fifth of a second in which no scan and no picture is read — but a RETRY: the moment is put
back (:meth:`pepin.snapshot.SnapshotPacker.rewind`) and offered again on the next arrival, of which
there are eighteen a second, until it has left that member's own pairing patience
(:data:`pepin.snapshot.PAIR_PERIODS` of its measured period, 176 ms for the camera at 8.5 Hz). Past
that the snapshot goes out with whatever could be placed, exactly as before, and the report line
counts both the waits and the give-ups.

THE REPORT LINE says what went into the last snapshot, every source's measured cadence and the
counts of full, lidar-only and camera-only snapshots; the flags (:data:`FLAGS`, ``ros/flags.sh
set sensor_pack <flag> <value>``) are ``sensor_pack``, ``sources``, ``pack_hz``,
``pair_periods`` and ``tf_retry``. The arrangement of before this node — RTAB-Map on its own
synchronised triple — is one launch argument away: ``ros/laptop.sh vslam`` passing
``sensor_pack:=false`` does not start this node and puts ``subscribe_depth`` and ``subscribe_scan``
back on RTAB-Map.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from rtabmap_msgs.msg import SensorData
from sensor_msgs.msg import CameraInfo, Image, LaserScan, PointCloud2, PointField
from std_msgs.msg import String

from pepin.flags import Flag, FlagSet
from pepin.snapshot import (
    LIVE_PERIODS,
    PAIR_PERIODS,
    Snapshot,
    SnapshotPacker,
    SnapshotState,
    StampRing,
)
from pepin.sources import CAMERA, LIDAR
from pepin_bringup.msgs import header, stamp_seconds
from pepin_bringup.node_kit import Switches, Tally, TfLookup, Window, spin_main

SENSOR_DATA_TOPIC = "/rtabmap/sensor_data"  # rtabmap's own relative "sensor_data" in its namespace
# ...and what those snapshots CARRY, latched, for the one reader that must follow it: rtabmap_frame
# chooses RTAB-Map's registration strategy by it (pepin.graphmode.StrategyRule).
STATE_TOPIC = "/sensor_pack/state"
IMAGE_TOPIC = "/camera/image"
DEPTH_TOPIC = "/camera/depth"
CAMERA_INFO_TOPIC = "/camera/camera_info"
SCAN_TOPIC = "/scan"
BASE_FRAME = "base_link"
STAGES = ("pack", "scan", "publish")
# rtabmap_msgs/SensorData's own accepted encodings, read in MsgConversion.cpp:1135-1195. Ours
# are bgr8 and 32FC1; the rest are listed so a frame this node refuses is refused for the same
# reason RTAB-Map would have refused it, one hop earlier and counted.
RGB_ENCODINGS = ("mono8", "8UC1", "mono16", "bgr8", "rgb8", "bgra8", "rgba8")
DEPTH_ENCODINGS = ("32FC1", "16UC1", "mono16")
# rtabmap::LaserScan::kXY (rtabmap/core/LaserScan.h:41): a cloud of x and y alone, which is what
# rtabmap infers from these two fields and asserts this number against (MsgConversion.cpp:1241).
LASER_SCAN_FORMAT_XY = 1
XY_POINT_STEP = 8  # two float32
# TF IS NEVER WAITED FOR ON THIS THREAD. Every lookup here happens inside a subscription callback,
# so a blocking wait is a wait in which no picture, no depth and no revolution is read: the fifth of
# a second this node used to pass (depth_stream's CARRY_WAIT_S) stalled the executor on every frame
# whose neck edge was late, and still dropped about 5 % of them. The member waits by RETRY instead
# (:meth:`SensorPack._retry`), so this is zero and means it.
TF_WAIT_S = 0.0
# Rtabmap/DetectionRate in vslam.launch.py's table: RTAB-Map adds at most one node a second and
# drops the rest, so a faster snapshot is megabytes of DDS traffic for a message nobody reads.
DETECTION_RATE_HZ = 1.0

FLAGS = FlagSet(
    Flag(
        "sensor_pack",
        True,
        description="snapshots are published; off, the node subscribes and counts and RTAB-Map"
        " is fed nothing at all",
        why="on by design: with subscribe_sensor_data there is no other input, so off is a mapper"
        " that stops adding nodes, and at Rtabmap/DetectionRate 1.0 that shows in RTAB-Map's own"
        " report within 1 s and in this node's within the 30 s of a window. It is the switch for"
        " isolating this node during a live test, not a behaviour A/B: the A/B against the old"
        " synchronised triple is the launch argument sensor_pack:=false, which also puts"
        " subscribe_depth and subscribe_scan back on RTAB-Map",
        on_when="always, whenever the map is meant to grow or to localise",
        off_when="to prove that a node RTAB-Map reports is this node's and not a leftover"
        " subscription, and to take the camera's bytes off DDS while something else is measured",
    ),
    Flag(
        "sources",
        (CAMERA, LIDAR),
        description="which sensors may enter a snapshot: the live A/B for camera-only and"
        " lidar-only mapping, with no restart and without muting a publisher",
        why="both, because that is the whole point of the message: a node carries the scan the"
        " grid's plane is exact from AND the picture the place is recognised by. Dropping one"
        " here is exactly what the stack used to need a different launch and a different"
        " parameter table for (the SLAM_LIDAR and SLAM_CAMERA_ONLY tables of before 2026-09-19)",
        on_when="lidar alone to reproduce the old lidar SLAM, camera alone for the honest"
        " camera-only test — the map is then only as true as the network's scale",
        off_when="never empty: with no source there is nothing to pack and RTAB-Map starves",
        choices=(CAMERA, LIDAR),
    ),
    Flag(
        "pack_hz",
        DETECTION_RATE_HZ,
        description="at most this many snapshots a second of SENSOR time (the stamps' own clock,"
        " not this laptop's)",
        why=f"{DETECTION_RATE_HZ} is Rtabmap/DetectionRate in vslam.launch.py's table: RTAB-Map"
        " creates at most one node a second and throws the rest away after paying for the"
        " conversion, and one snapshot is 6.5 MB at 1280x720 (2.8 MB of bgr8 plus 3.7 MB of"
        " 32FC1). Packing at the camera's 8.5 Hz would put 55 MB/s on DDS for seven messages in"
        " eight that RTAB-Map drops. It is not a node-rate choice: the node rate was already this"
        " number before the snapshots existed",
        on_when="raise it only together with Rtabmap/DetectionRate, and watch the laptop's CPU"
        " and RTAB-Map's own ms per node in its report",
        off_when="lower it on a laptop that cannot keep up; the cost is a sparser graph",
        range=(0.1, 15.0),
    ),
    Flag(
        "pair_periods",
        PAIR_PERIODS,
        description="how many of its OWN measured periods a source's message may be from the"
        " snapshot's stamp and still be paired with it (pepin.snapshot)",
        why=f"{PAIR_PERIODS} is derived, not chosen: the nearest message of a source running at"
        " period T is at most T/2 from any instant, and one lost message doubles the far side, so"
        " 1.5 T is the worst case of a source that is live and has missed one. Past it two"
        " messages in a row are missing. At the measured 9.9 Hz that is 152 ms of patience against"
        " a healthy 51 ms of offset, which at 0.2 m/s and 17 deg/s is 1.0 cm and 0.86 deg of"
        " placement inside one node (scratch/pairing_bound.py). Liveness is a different and much"
        f" looser question, {LIVE_PERIODS} periods, and is not a flag",
        on_when="raise it where a source is known to stutter and a stale member is better than a"
        " node without it — a scan 0.3 s off still fixes a wall to 6 cm at walking pace",
        off_when="lower it to 0.5 to admit only the genuinely nearest message, which is the right"
        " test while a placement error inside a node is being hunted",
        range=(0.5, 10.0),
    ),
    Flag(
        "tf_retry",
        True,
        description="a member whose transform TF cannot answer for yet does not cost the snapshot:"
        " the moment is put back and offered again on the next arrival, until it has left that"
        " member's own pairing patience (pair_periods of its measured period). Off, the member is"
        " dropped at once and the snapshot goes out without it — the behaviour of before"
        " 2026-09-19",
        why="about 5 % of camera frames answered 'base_link<-camera_optical: Lookup would require"
        " extrapolation into the future' and became lidar-only snapshots (measured live"
        " 2026-09-18): the neck's dynamic edge is published on the board and arrives over the"
        " bridge tens of milliseconds behind the frame it belongs to. The old answer was a 0.2 s"
        " BLOCKING wait inside the subscription callback, which both stalled the executor on every"
        " late frame — no picture, no depth and no revolution read while it waited — and still"
        " dropped those 5 %. A retry costs nothing and waits longer: arrivals come eighteen a"
        " second, so the patience is spent on the bridge rather than on this thread. The bound is"
        " the pairing bound itself and not a new number (176 ms for the camera at 8.5 Hz)",
        on_when="always: a camera frame is the only thing in a snapshot that can recognise a place,"
        " and it is the one whose transform is late",
        off_when="to measure what the retry is worth — the report line's 'tf waits' against its"
        " 'frames TF could not place' is the same comparison with it on",
    ),
)


@dataclass(frozen=True)
class Frame:
    """The camera's member of a snapshot: one picture and the depth image computed from it, which
    carry the same ``header.stamp`` bit for bit. ``stamp`` is that stamp as the message, kept so
    the snapshot's header can be a real sensor stamp instead of a float rounded back."""

    image: Any
    depth: Any

    @property
    def stamp(self) -> Any:
        """The picture's ``builtin_interfaces/Time``, which is also the depth's."""
        return self.image.header.stamp


def same_stamp(left: Any, right: Any) -> bool:
    """Whether two stamp messages are the same moment to the nanosecond. Compared on the fields
    and not on seconds as a float: at an epoch stamp a double's step is 238 ns, so two different
    moments can round to one number."""
    return (left.sec, left.nanosec) == (right.sec, right.nanosec)


@dataclass(frozen=True)
class Packed:
    """One snapshot turned into a message: the message (``None`` when not one member could be
    placed on the cart) and the members TF could not place YET, which is what decides whether the
    moment is put back for another try (:meth:`SensorPack._retry`)."""

    msg: Any | None
    unplaced: tuple[str, ...] = ()


def depth_is_a_factor(image: Any, depth: Any) -> bool:
    """Whether this depth image is the picture's size divided by ONE whole number — RTAB-Map's own
    contract for an RGB-D pair, asserted FATALLY on its side (``Memory.cpp:4579`` for "never
    bigger", ``util3d.cpp:324`` for the modulo). Equal sizes are the normal case and pass.

    One number and not one per axis, which is stricter than the assertion: rtabmap turns the ratio
    into a single ``decimation`` and prints it as one (``util3d.cpp:328``, "Decimation from model
    (%d)"), so a pair halved in width alone would pass the assert and then be reprojected through
    the wrong factor in the other axis.
    """
    if depth.width <= 0 or depth.height <= 0:
        return False
    if depth.width > image.width or depth.height > image.height:
        return False
    if image.width % depth.width or image.height % depth.height:
        return False
    return bool(image.width // depth.width == image.height // depth.height)


def xy_cloud(scan: Any) -> Any:
    """A ``LaserScan`` as the ``PointCloud2`` rtabmap_msgs/SensorData wants: the returns as x and
    y float32 IN THE SCAN'S OWN FRAME, non-finite ranges and ranges outside
    ``[range_min, range_max]`` dropped (what laser_geometry's projection dropped for the old
    path), no z field — which is what makes RTAB-Map read the cloud as
    :data:`LASER_SCAN_FORMAT_XY`.

    The cloud keeps the SCAN's own header, not the snapshot's: RTAB-Map reads only the fields and
    the bytes out of it (MsgConversion.cpp:1231-1242), and a recorded message then carries the
    pairing offset itself — ``header.stamp`` is the snapshot's moment, ``laser_scan.header.stamp``
    is the revolution's."""
    ranges = np.asarray(scan.ranges, dtype=np.float64)
    angles = scan.angle_min + np.arange(ranges.size, dtype=np.float64) * scan.angle_increment
    keep = np.isfinite(ranges) & (ranges >= scan.range_min) & (ranges <= scan.range_max)
    xy = np.empty((int(np.count_nonzero(keep)), 2), dtype=np.float32)
    xy[:, 0] = (ranges[keep] * np.cos(angles[keep])).astype(np.float32)
    xy[:, 1] = (ranges[keep] * np.sin(angles[keep])).astype(np.float32)
    return PointCloud2(
        header=header(scan.header.stamp, scan.header.frame_id),
        height=1,
        width=int(xy.shape[0]),
        fields=[
            PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
            PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        ],
        is_bigendian=False,
        point_step=XY_POINT_STEP,
        row_step=XY_POINT_STEP * int(xy.shape[0]),
        data=xy.tobytes(),
        is_dense=True,
    )


class SensorPack(Node):
    """Publishes one ``rtabmap_msgs/SensorData`` per moment out of whatever sensor is alive."""

    def __init__(self) -> None:
        super().__init__("sensor_pack")
        self._switches = Switches(self, FLAGS, on_change=self._on_switch)
        self._packer = SnapshotPacker[Any](
            (CAMERA, LIDAR), pair_periods=float(self._switches["pair_periods"])
        )
        self._packer.enable(self._switches["sources"])
        # Two deep for the pictures: a camera member is two messages paired here rather than by a
        # synchroniser, so the pair must both be deliverable, and one of them is 3.7 MB.
        pair = QoSProfile(depth=2, reliability=ReliabilityPolicy.RELIABLE)
        reliable = QoSProfile(depth=5, reliability=ReliabilityPolicy.RELIABLE)
        self._pub = self.create_publisher(SensorData, SENSOR_DATA_TOPIC, reliable)
        # Pictures waiting for their depth and the other way round, over pepin.depth's measured
        # SCAN_WINDOW_S. The price of pairing on stamps instead of on arrival order: about 60 MB
        # of this laptop's memory at 1280x720 (11 pictures of 2.8 MB, 9 depths of 3.7 MB).
        self._images = StampRing[Any]()
        self._depths = StampRing[Any]()
        self._info: Any | None = None  # the newest optics; a lens, not a moment
        self._tally = Tally(STAGES)
        self._last: Snapshot[Any] | None = None  # what went out last, for the report line
        # What the snapshots CARRY, latched: the one fact only this node can answer, and the one
        # RTAB-Map's registration strategy follows (pepin_bringup.rtabmap_frame).
        self._state_pub = self.create_publisher(
            String,
            STATE_TOPIC,
            QoSProfile(
                depth=1,
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
            ),
        )
        self._state: SnapshotState | None = None  # ...as it was last published
        self.create_subscription(CameraInfo, CAMERA_INFO_TOPIC, self._on_info, reliable)
        self.create_subscription(Image, IMAGE_TOPIC, self._on_image, pair)
        self.create_subscription(Image, DEPTH_TOPIC, self._on_depth, pair)
        self.create_subscription(LaserScan, SCAN_TOPIC, self._on_scan, reliable)
        self._tf = TfLookup(self, on_failure=self._on_tf_failure)
        # The state is published on every CHANGE (in _pack) and once a second whatever happens, so
        # a reader can tell a state that has stopped moving from one that simply has not changed.
        self.create_timer(1.0, self._publish_state)
        self.create_timer(30.0, self._report)
        self.get_logger().info(
            f"sensor pack up: {IMAGE_TOPIC} + {DEPTH_TOPIC} + {CAMERA_INFO_TOPIC} and"
            f" {SCAN_TOPIC} -> one {SENSOR_DATA_TOPIC} per moment, at most"
            f" {float(self._switches['pack_hz']):.1f} a second of sensor time; every source's"
            " freshness is its own measured period; flags:"
            f" {self._switches.state()}"
        )

    def close(self) -> None:
        """Stop the TF listener's thread before the node is destroyed under it."""
        self._tf.close()

    # ---- inputs ------------------------------------------------------------------------------
    def _on_switch(self, name: str, _old: object, new: object) -> None:
        """A flag changed: ``sources`` is the packer's roster, ``pair_periods`` its patience; the
        rest are read where they are used."""
        if name == "sources":
            if not new:
                raise ValueError("sources: at least one sensor, or there is nothing to pack")
            self._packer.enable(tuple(new))  # type: ignore[arg-type]
        elif name == "pair_periods":
            self._packer.set_pair_periods(float(new))  # type: ignore[arg-type]

    def _on_info(self, msg: CameraInfo) -> None:
        self._info = msg
        self._tally.count("info_in")

    def _on_image(self, msg: Image) -> None:
        """A picture in: kept, and offered as a camera member if its depth is already here."""
        self._tally.count("image_in")
        if msg.encoding not in RGB_ENCODINGS:
            self._tally.count("bad_image")
            self.get_logger().error(
                f"{IMAGE_TOPIC} is {msg.encoding}; rtabmap_msgs/SensorData's left image must be"
                f" one of {', '.join(RGB_ENCODINGS)}",
                throttle_duration_sec=30,
            )
            return
        self._images.offer(stamp_seconds(msg.header.stamp), msg)
        self._pair(msg, self._depths, arrival_is_image=True)

    def _on_depth(self, msg: Image) -> None:
        """A depth image in: kept, and offered as a camera member if its picture is already
        here (it carries that picture's stamp, so the match is exact)."""
        self._tally.count("depth_in")
        if msg.encoding not in DEPTH_ENCODINGS:
            self._tally.count("bad_depth")
            self.get_logger().error(
                f"{DEPTH_TOPIC} is {msg.encoding}; rtabmap_msgs/SensorData's right image must be"
                f" one of {', '.join(DEPTH_ENCODINGS)}",
                throttle_duration_sec=30,
            )
            return
        self._depths.offer(stamp_seconds(msg.header.stamp), msg)
        self._pair(msg, self._images, arrival_is_image=False)

    def _pair(self, msg: Any, others: StampRing[Any], *, arrival_is_image: bool) -> None:
        """The message that just arrived against the other half's ring: on a bit-equal stamp the
        two become one camera member, and a snapshot is attempted. Which of the two is the picture
        is told by the topic it came on and never by its encoding — ``mono16`` is a legal encoding
        for both halves (MsgConversion.cpp:1135-1195)."""
        for _stamp, other in others.items():
            if not same_stamp(msg.header.stamp, other.header.stamp):
                continue
            image, depth = (msg, other) if arrival_is_image else (other, msg)
            self._packer.offer(CAMERA, stamp_seconds(image.header.stamp), Frame(image, depth))
            self._tally.count("frames")
            self._pack()
            return
        self._tally.count("unpaired")

    def _on_scan(self, msg: LaserScan) -> None:
        """A revolution in: offered as the lidar's member, and a snapshot attempted. A scan whose
        angle_increment is zero carries no bearings at all — every return would land on one ray —
        and is refused here for the reason rtabmap refuses it itself
        (MsgConversion.cpp:2602-2605)."""
        self._tally.count("scan_in")
        if msg.angle_increment == 0.0 or msg.range_min > msg.range_max:
            self._tally.count("bad_scan")
            self.get_logger().error(
                f"{SCAN_TOPIC}: angle_increment {msg.angle_increment} and range"
                f" {msg.range_min}..{msg.range_max} describe no revolution",
                throttle_duration_sec=30,
            )
            return
        self._packer.offer(LIDAR, stamp_seconds(msg.header.stamp), msg)
        self._pack()

    # ---- the snapshot ------------------------------------------------------------------------
    def _pack(self) -> None:
        """One snapshot out, or nothing: the packer decides who is alive and who pairs, this
        turns the answer into a message. Called on every arrival; ``pack_hz`` of sensor time is
        what keeps that from being a publish per message."""
        if not self._switches.on("sensor_pack"):
            self._tally.count("off")
            return
        with self._tally.measure("pack"):
            snapshot = self._packer.plan(min_gap_s=1.0 / float(self._switches["pack_hz"]))
            if snapshot is None:
                return
            packed = self._sensor_data(snapshot)
        if self._retry(snapshot, packed):
            return
        if packed.msg is None:
            return
        with self._tally.measure("publish"):
            self._pub.publish(packed.msg)
        self._last = snapshot
        self._tally.count("packs")
        self._tally.count(f"kind {snapshot.kind}")
        for name in snapshot.members:
            if name != snapshot.driver:
                self._tally.sample(f"{name} offset ms", snapshot.offset_s(name) * 1e3)
        self._publish_state(on_change=True)

    def _retry(self, snapshot: Snapshot[Any], packed: Packed) -> bool:
        """Whether this moment is PUT BACK for the next arrival instead of published, because a
        member's transform has not reached TF yet and the moment is still inside that member's own
        pairing patience (:data:`pepin.snapshot.PAIR_PERIODS` of its measured period).

        Nothing waits on a thread here. The moment goes back to the packer
        (:meth:`pepin.snapshot.SnapshotPacker.rewind`) and the next arriving picture, depth or
        revolution — eighteen a second between them — asks for it again, by which time the neck's
        edge has landed. Past the patience the snapshot goes out with whatever could be placed,
        which is the behaviour this replaces, and the give-up is counted under its own name.
        """
        if not packed.unplaced or not self._switches.on("tf_retry"):
            return False
        for name in packed.unplaced:
            patience = self._packer.cadence(name).patience_s
            if patience is not None and self._packer.how_old_s(snapshot.stamp) <= patience:
                self._packer.rewind()
                self._tally.count(f"tf_wait {name}")
                return True
        return False

    def _sensor_data(self, snapshot: Snapshot[Any]) -> Packed:
        """One snapshot as ``rtabmap_msgs/SensorData`` — and which of its members TF could not
        place yet, for :meth:`_retry`.

        ``Packed.msg`` is ``None`` when every member fell away while it was being built (no optics
        for the picture, a depth that is not the picture's size over a whole number, no transform
        for either). The header's stamp is the driver's own message stamp, so the moment RTAB-Map
        looks the odometry up at is a moment the board named."""
        driver = snapshot.members[snapshot.driver][1]
        stamp = driver.stamp if isinstance(driver, Frame) else driver.header.stamp
        msg = SensorData(header=header(stamp, BASE_FRAME))
        camera = snapshot.members.get(CAMERA)
        scan = snapshot.members.get(LIDAR)
        filled, unplaced = False, []
        if camera is not None:
            placed, waiting = self._camera_into(msg, camera[1])
            filled = filled or placed
            if waiting:
                unplaced.append(CAMERA)
        if scan is not None:
            placed, waiting = self._scan_into(msg, scan[1])
            filled = filled or placed
            if waiting:
                unplaced.append(LIDAR)
        return Packed(msg if filled else None, tuple(unplaced))

    def _camera_into(self, msg: Any, frame: Frame) -> tuple[bool, bool]:
        """The picture, its depth, its optics and its place on the cart into ``msg``: whether it
        went in, and whether the only thing missing was the TRANSFORM (which is worth another try).

        Two refusals that are not worth a retry, and both are RTAB-Map's own contract checked one
        hop before it would abort the process:

        * the optics must describe THE PICTURE at its exact size — the model is taken verbatim and
          asserted against the picture on rtabmap's side (util3d.cpp:497), not against the depth;
        * the depth must be the picture's size divided by a whole number (``depth_is_a_factor``):
          bigger is a FATAL UASSERT there (Memory.cpp:4579) and a non-integer ratio is another
          (util3d.cpp:324).

        The transform is ``base_link <- the picture's own frame`` at the picture's stamp, looked up
        without waiting (:data:`TF_WAIT_S`) — the wait is :meth:`_retry`'s business.
        """
        info, image, depth = self._info, frame.image, frame.depth
        if info is None or (info.width, info.height) != (image.width, image.height):
            self._tally.count("no_optics")
            return False, False
        if not depth_is_a_factor(image, depth):
            self._tally.count("depth_not_a_factor")
            self.get_logger().error(
                f"{DEPTH_TOPIC} is {depth.width}x{depth.height} against a {image.width}x"
                f"{image.height} picture: rtabmap needs the depth to be the picture's size over a"
                " whole number and ABORTS otherwise (Memory.cpp:4579)",
                throttle_duration_sec=30,
            )
            return False, False
        local = self._tf.transform(BASE_FRAME, image.header.frame_id, image.header.stamp, TF_WAIT_S)
        if local is None:
            self._tally.count("no_camera_tf")
            return False, True
        msg.left = image
        msg.right = depth
        msg.left_camera_info = [info]  # right_camera_info stays empty: empty is what "not stereo"
        msg.local_transform = [local.transform]  # means (MsgConversion.cpp:1096)
        return True, False

    def _scan_into(self, msg: Any, scan: Any) -> tuple[bool, bool]:
        """The revolution into ``msg`` as a cloud in its own frame with the transform that places
        it on the cart: whether it went in, and whether only the transform was missing."""
        local = self._tf.transform(BASE_FRAME, scan.header.frame_id, scan.header.stamp, TF_WAIT_S)
        if local is None:
            self._tally.count("no_scan_tf")
            return False, True
        with self._tally.measure("scan"):
            msg.laser_scan = xy_cloud(scan)
        msg.laser_scan_format = LASER_SCAN_FORMAT_XY
        # The beam count the angles imply, which is what the old path's LaserScan carried
        # (rtabmap/core/LaserScan.cpp:352) and what makes Icp/CorrespondenceRatio absolute.
        msg.laser_scan_max_pts = len(scan.ranges)
        msg.laser_scan_max_range = float(scan.range_max)
        msg.laser_scan_local_transform = local.transform
        return True, False

    # ---- what the snapshots carry ------------------------------------------------------------
    def _publish_state(self, on_change: bool = False) -> None:
        """Say what the snapshots carry on :data:`STATE_TOPIC`, latched: on every change (from
        :meth:`_pack`) and once a second whatever happens, so a reader can tell a state that has
        stopped moving from one that simply has not changed.

        ``on_change`` publishes only when the sources or the kind actually differ, which is what
        keeps the packing path from serialising a message per snapshot."""
        state = self._packer.state(self._last.kind if self._last is not None else "none yet")
        last = self._state
        if (
            on_change
            and last is not None
            and (state.carrying, state.kind)
            == (
                last.carrying,
                last.kind,
            )
        ):
            return
        self._state = state
        self._state_pub.publish(String(data=state.to_json()))

    def _on_tf_failure(self, kind: str, text: str) -> None:
        self._tally.count(f"tf_{kind}")
        self._tally.note(kind, text)

    # ---- the report --------------------------------------------------------------------------
    def _report(self) -> None:
        """The window's snapshots by kind, every source's measured cadence, what fell away and
        the flags, in one line."""
        w = self._tally.take()
        c = w.counts
        kinds = ", ".join(
            f"{c[name]} {name.removeprefix('kind ')}"
            for name in sorted(c)
            if name.startswith("kind ")
        )
        state = self._state
        self.get_logger().info(
            f"sensor pack: {w.rate('packs'):.2f} snapshots/s of {c['scan_in']} scans and"
            f" {c['frames']} camera frames ({kinds or 'none'});"
            f" {self._packer.report()}; {self._last_note()}{self._losses(w)};"
            f" carrying {state.text() if state is not None else 'nothing said yet'}"
            f" on {STATE_TOPIC};"
            f" flags: {self._switches.state()}; ms median/max: {w.stages()}"
        )
        if self._switches.on("sensor_pack") and c["packs"] == 0:
            self.get_logger().warning(
                "no snapshot in this window: RTAB-Map is being told nothing. Is"
                f" {SCAN_TOPIC} arriving over the bridge, and does {DEPTH_TOPIC} come"
                " (the depth law needs the lidar)?"
            )

    def _last_note(self) -> str:
        """What the last snapshot was made of: ``last: full, driven by camera, scan 23 ms off``."""
        snapshot = self._last
        if snapshot is None:
            return "nothing packed yet"
        offsets = "".join(
            f", {name} {snapshot.offset_s(name) * 1e3:.0f} ms off"
            for name in snapshot.members
            if name != snapshot.driver
        )
        silent = f", silent {'+'.join(snapshot.silent)}" if snapshot.silent else ""
        return f"last: {snapshot.kind}, driven by {snapshot.driver}{offsets}{silent}"

    def _losses(self, w: Window) -> str:
        """The parts of the line a healthy window has nothing to say about: the messages that
        never became a member, and the last TF failure of each kind."""
        c, out = w.counts, ""
        for count, what in (
            (c["unpaired"], "images or depths with no partner at their stamp"),
            (c["no_optics"], "frames whose camera_info does not describe the picture"),
            (c["depth_not_a_factor"], "depths that are not the picture's size over a whole number"),
            (c[f"tf_wait {CAMERA}"], "moments put back waiting for the camera's transform"),
            (c[f"tf_wait {LIDAR}"], "moments put back waiting for the laser's transform"),
            (c["no_camera_tf"], "frames TF could not place"),
            (c["no_scan_tf"], "scans TF could not place"),
            (c["bad_image"], f"pictures not in {'/'.join(RGB_ENCODINGS)}"),
            (c["bad_depth"], f"depth images not in {'/'.join(DEPTH_ENCODINGS)}"),
            (c["bad_scan"], "scans describing no revolution"),
            (c["off"], "arrivals with the switch off"),
        ):
            if count:
                out += f", {count} {what}"
        if w.notes:
            out += ", tf: " + "; ".join(
                f"{kind} {c['tf_' + kind]}: {text}" for kind, text in w.notes.items()
            )
        return out


def main() -> None:
    spin_main(SensorPack)


if __name__ == "__main__":
    main()
