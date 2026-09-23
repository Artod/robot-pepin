#!/usr/bin/env python3
"""One run's rosbag2 bag turned into the numbered JSONL tape every analysis script reads.

The board can record a drive two ways (ros/README.md, "Two recorders"): as JSON lines written by
``pepin_bringup.run_recorder``, or as an MCAP bag written by ``ros2 bag record`` under
``pepin_bringup.bag_recorder`` — the second costs the board a memcpy instead of 34-43 % of a
core, because nothing there deserialises a message. This tool pays that cost later and on the
laptop: it reads the bag, deserialises every message once, and writes the very same rows the
JSONL recorder would have written, from the same functions (:mod:`pepin.tape_rows`). So
``scratch/*``, ``ros/tools/*`` and ``scripts/build_map.py`` read a bag run without knowing it
was one.

What differs from a live tape, and cannot be otherwise:

- the records with no stamp of their own (``cmd``, ``nav``, ``meas``, ``srcs``) are dated by the
  bag's receive time instead of the recorder's ``time.time()`` — the same clock, one hop later;
- there is no prelude: a bag starts when the goal's word does, a tape starts 15 seconds earlier
  (``pepin.tape.RunTape``);
- the ``loc`` rows are the tracker's where the bag holds ``/tracker_pose``, and otherwise are
  composed from ``/tf`` (``map -> odom``, ``odom -> base_link``) at 5 Hz, the rate and the
  composition the live recorder uses where no tracker publishes a pose.

Runs in the laptop's ROS container, where rosbag2_py and rclpy's serialization live:

    docker exec pepin-vslam /pepin_entrypoint.sh \\
        python3 /tools/bag_to_tape.py /maps/rec/0251_20260922_141002Z_home

With no output path the tape is written beside the bag as ``<bag>.jsonl``, which is the name
ros/goto.sh fetches. ``--force`` overwrites an existing tape.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from pepin.mounts import Mounts
from pepin.recording import imu_record, scan_record_from_ros
from pepin.tape_rows import (
    TOPIC_RECORDS,
    cmd_row,
    costmap_row,
    ekf_row,
    gcostmap_row,
    laser_odom_row,
    loc_row,
    loc_row_from_transform,
    meas_row,
    nav_row,
    plan_row,
    pose_row,
    srcs_row,
    stamp,
    tof_row,
)

# How often ``loc`` is written when it is composed from TF, and the rate the live recorder reads
# ``map -> base_link`` at (RunRecorder.LOC_TF_PERIOD_S / IDLE_PERIOD_S): 5 Hz, so a tape looks the
# same whoever wrote it.
LOC_PERIOD_S = 0.2


class _Transform:
    """A ``map -> base_link`` transform composed here, with the fields the row builder reads."""

    def __init__(self, x: float, y: float, theta: float, stamp_s: float) -> None:
        self.header = _Header(stamp_s)
        self.transform = _TransformBody(x, y, theta)


class _Header:
    def __init__(self, seconds: float) -> None:
        self.stamp = _Time(seconds)


class _Time:
    def __init__(self, seconds: float) -> None:
        self.sec = int(seconds)
        self.nanosec = int((seconds - int(seconds)) * 1e9)


class _TransformBody:
    def __init__(self, x: float, y: float, theta: float) -> None:
        self.translation = _Vector(x, y)
        self.rotation = _Quaternion(theta)


class _Vector:
    def __init__(self, x: float, y: float) -> None:
        self.x, self.y, self.z = x, y, 0.0


class _Quaternion:
    def __init__(self, theta: float) -> None:
        self.x = self.y = 0.0
        self.z, self.w = math.sin(theta / 2.0), math.cos(theta / 2.0)


def _yaw_of(rotation: Any) -> float:
    """Yaw in radians of a quaternion message (the same formula pepin.tape_rows.yaw uses)."""
    x, y, z, w = rotation.x, rotation.y, rotation.z, rotation.w
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


class TapeBuilder:
    """Turns a bag's messages into a tape's rows, in the order they were received.

    ``feed`` takes one message with the time the bag stamped it and returns the rows it makes
    (none for a topic that carries no record of its own, several never). The two throttles a live
    tape has live here too: one global costmap per new plan, and 5 Hz for ``loc`` rows composed
    from TF.
    """

    def __init__(self, *, loc_from_tf: bool = False) -> None:
        lidar = Mounts.load().lidar
        self._mount_yaw_rad = -math.radians(lidar.yaw_deg)
        self._mount_x_m = lidar.x_m
        self._loc_from_tf = loc_from_tf
        self._plan_seq = 0
        self._gcostmap_seq = -1
        self._map_odom: tuple[float, float, float] | None = None
        self._odom_base: tuple[float, float, float] | None = None
        self._last_loc = 0.0

    def feed(self, topic: str, msg: Any, received_s: float) -> list[dict[str, Any]]:
        """The rows one recorded message makes; ``received_s`` is the bag's own receive time."""
        record = TOPIC_RECORDS.get(topic)
        if topic in ("/tf", "/tf_static"):
            return self._on_tf(msg, received_s)
        if record == "scan":
            return [
                scan_record_from_ros(
                    stamp(msg.header, received_s),
                    msg.angle_min,
                    msg.angle_increment,
                    list(msg.ranges),
                    list(msg.intensities),
                    msg.range_min,
                    msg.range_max,
                    msg.scan_time,
                    mount_yaw_rad=self._mount_yaw_rad,
                    mount_x_m=self._mount_x_m,
                )
            ]
        if record == "pose":
            return [pose_row(msg, received_s)]
        if record == "ekf":
            return [ekf_row(msg, received_s)]
        if record == "laser_odom":
            return [laser_odom_row(msg, received_s)]
        if record == "imu":
            w, a = msg.angular_velocity, msg.linear_acceleration
            t = stamp(msg.header, received_s)
            return [imu_record(t, (w.x, w.y, w.z), (a.x, a.y, a.z))]
        if record == "cmd":
            return [cmd_row(msg, received_s)]
        if record == "loc":
            return [loc_row(msg, received_s)]
        if record == "plan":
            self._plan_seq += 1
            return [plan_row(msg, received_s)]
        if record == "costmap":
            return [costmap_row(msg, received_s)]
        if record == "gcostmap":
            if self._plan_seq == self._gcostmap_seq:
                return []
            self._gcostmap_seq = self._plan_seq
            return [gcostmap_row(msg, received_s, self._plan_seq)]
        if record == "tof":
            return [tof_row(topic.rsplit("/", 1)[-1], msg, received_s)]
        if record == "meas":
            return [meas_row(msg.data, received_s)]
        if record == "srcs":
            return [srcs_row(msg.data, received_s)]
        if record == "nav":
            return [nav_row(topic.split("/")[1], msg, received_s)]
        return []

    def _on_tf(self, msg: Any, received_s: float) -> list[dict[str, Any]]:
        """Keep the two edges a pose is made of, and write a ``loc`` row at 5 Hz from them."""
        newest = 0.0
        for transform in msg.transforms:
            parent, child = transform.header.frame_id.lstrip("/"), transform.child_frame_id
            edge = (
                transform.transform.translation.x,
                transform.transform.translation.y,
                _yaw_of(transform.transform.rotation),
            )
            if (parent, child.lstrip("/")) == ("map", "odom"):
                self._map_odom = edge
            elif (parent, child.lstrip("/")) == ("odom", "base_link"):
                self._odom_base = edge
            else:
                continue
            newest = max(newest, stamp(transform.header, received_s))
        if not self._loc_from_tf or self._map_odom is None or self._odom_base is None:
            return []
        when = newest or received_s
        if when - self._last_loc < LOC_PERIOD_S:
            return []
        self._last_loc = when
        mx, my, mtheta = self._map_odom
        ox, oy, otheta = self._odom_base
        x = mx + ox * math.cos(mtheta) - oy * math.sin(mtheta)
        y = my + ox * math.sin(mtheta) + oy * math.cos(mtheta)
        return [loc_row_from_transform(_Transform(x, y, mtheta + otheta, when), received_s)]


def read_bag(bag: Path) -> Iterator[tuple[str, Any, float]]:
    """Every message of ``bag`` as (topic, message, receive time in seconds), in bag order.

    Needs the ROS runtime (rosbag2_py, rclpy.serialization), which is why this runs in the
    laptop's container and why the import is here and not at the top: the row builders above are
    pure and the unit tests import them with no ROS at all.
    """
    import rosbag2_py
    from rclpy.serialization import deserialize_message
    from rosidl_runtime_py.utilities import get_message

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id=""),
        rosbag2_py.ConverterOptions(
            input_serialization_format="cdr", output_serialization_format="cdr"
        ),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    while reader.has_next():
        topic, payload, nanoseconds = reader.read_next()
        kind = types.get(topic)
        if kind is None:
            continue
        yield topic, deserialize_message(payload, get_message(kind)), nanoseconds * 1e-9


def has_tracker_pose(bag: Path) -> bool:
    """Whether the bag holds a tracker pose, which decides where the ``loc`` rows come from."""
    import rosbag2_py

    info = rosbag2_py.Info().read_metadata(str(bag), "")
    return any(
        t.topic_metadata.name == "/tracker_pose" and t.message_count > 0
        for t in info.topics_with_message_count
    )


def convert(bag: Path, tape: Path, *, loc_from_tf: bool) -> int:
    """Write ``bag`` out as a tape; returns how many rows it holds."""
    builder = TapeBuilder(loc_from_tf=loc_from_tf)
    written = 0
    with tape.open("w") as out:
        for topic, msg, received_s in read_bag(bag):
            for row in builder.feed(topic, msg, received_s):
                out.write(json.dumps(row, separators=(",", ":")) + "\n")
                written += 1
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("bag", type=Path, help="the run's bag directory (/maps/rec/NNNN_...)")
    parser.add_argument(
        "tape", type=Path, nargs="?", help="where to write it (default <bag>.jsonl)"
    )
    parser.add_argument("--force", action="store_true", help="overwrite an existing tape")
    args = parser.parse_args(argv)
    bag: Path = args.bag
    # The bag's own name plus the suffix, never with_suffix: a place called "flat3.1" would lose
    # its last part, and the name is what ros/goto.sh prints and the operator greps for.
    tape: Path = args.tape or bag.parent / f"{bag.name}.jsonl"
    if not bag.is_dir():
        print(f"no bag at {bag}", file=sys.stderr)
        return 2
    if tape.exists() and not args.force:
        print(f"{tape} is already there (--force overwrites)", file=sys.stderr)
        return 2
    rows = convert(bag, tape, loc_from_tf=not has_tracker_pose(bag))
    print(f"{tape}: {rows} records from {bag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
