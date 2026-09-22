"""The rows of a numbered tape, built from ROS messages: one format, two recorders.

A drive used to be taped by one process only (``pepin_bringup.run_recorder``, which subscribes
to every topic and writes the JSON lines itself). Since the board also records with rosbag2, the
same rows have to be produced a second time — offline, from the bag, on the laptop
(``ros/tools/bag_to_tape.py``) — and two hand-written copies of "round x to 4 decimals" drift
within a week. So every row lives here, in one pure function per record, and both recorders call
the same one: whatever the tape format is, it is what this module says.

The functions read ROS messages by attribute and import no ROS: any object with the fields of
the message works, which is how the unit tests drive them (``tests/unit/ros_stubs``). Nothing
here has state; the two throttles a tape has (the global costmap's one-grid-per-plan and the
5 Hz of the ``loc`` rows) belong to the recorder, because only it knows what it has already
written.

Times: a record's ``t`` is the message's own stamp where it has one, and the moment it arrived
where it has none (``cmd``, ``nav``, ``meas``, ``srcs``) — the caller passes that moment in, so
a live recorder can hand it ``time.time()`` and a converter the bag's receive time.
"""

from __future__ import annotations

import math
from typing import Any

from pepin.mapcache import run_length_encode

# How a taped grid's cells are encoded: :func:`pepin.mapcache.run_length_encode`, the one
# implementation, shared with the map the relocalizer persists.
RLE = "rle"
# The four values a feasibility question needs, and the only ones the global grid is taped with:
# Nav2's own unknown and free, everything inflated between them, and the 99-100 band that stops a
# footprint. A record says which it is (``classes``), so no reader guesses.
UNKNOWN, FREE, INFLATED, LETHAL_BAND = -1, 0, 1, 99

# Which ROS topic carries which record, and the only list of a drive's topics there is: the bag
# recorder records these, the converter turns them back into rows, and the JSONL recorder
# subscribes to exactly them. ``/tf`` and ``/tf_static`` carry no record of their own — they are
# what the ``loc`` rows are composed from where no tracker publishes a pose — and the camera's
# two String topics share the ``meas`` record.
TOPIC_RECORDS: dict[str, str] = {
    "/ldlidar_node/scan": "scan",
    "/odom": "pose",
    "/odometry/filtered": "ekf",
    "/odom_laser": "laser_odom",
    "/imu/data_raw": "imu",
    "/cmd_vel": "cmd",
    "/tracker_pose": "loc",
    "/plan": "plan",
    "/local_costmap/costmap": "costmap",
    "/global_costmap/costmap": "gcostmap",
    "/tof/front": "tof",
    "/tof/left": "tof",
    "/tof/right": "tof",
    "/localization/measurement": "meas",
    "/localization/graph_measurement": "meas",
    "/localization/sources": "srcs",
    "/navigate_to_pose/_action/status": "nav",
    "/compute_path_to_pose/_action/status": "nav",
    "/follow_path/_action/status": "nav",
}
# The frames the ``loc`` rows are composed from when the tape carries no tracker pose: the
# laptop's correction and the board's own odometry, the two edges every other consumer composes.
TF_EDGES = (("map", "odom"), ("odom", "base_link"))


def yaw(orientation: Any) -> float:
    """Yaw in radians from a quaternion message."""
    x, y, z, w = (getattr(orientation, name) for name in ("x", "y", "z", "w"))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def stamp(header: Any, now: float) -> float:
    """The message's own time in seconds; ``now`` when it carries none (an unstamped header is
    zero, and a record dated 1970 is worse than one dated on arrival)."""
    time_field = getattr(header, "stamp", None)
    seconds = time_field.sec + time_field.nanosec * 1e-9 if time_field else 0.0
    return seconds if seconds > 0 else now


def feasibility_classes(cells: list[int]) -> list[int]:
    """One costmap's cells reduced to the four values that decide whether the cart fits
    (:data:`UNKNOWN`, :data:`FREE`, :data:`INFLATED`, :data:`LETHAL_BAND`)."""
    return [
        UNKNOWN
        if value < 0
        else (FREE if value == 0 else (LETHAL_BAND if value >= 99 else INFLATED))
        for value in cells
    ]


def pose_row(msg: Any, now: float) -> dict[str, Any]:
    """The wheels' own odometry (``nav_msgs/Odometry`` on /odom): where the cart thinks it is."""
    return {
        "t": stamp(msg.header, now),
        "topic": "pose",
        "x": round(msg.pose.pose.position.x, 4),
        "y": round(msg.pose.pose.position.y, 4),
        "theta": round(yaw(msg.pose.pose.orientation), 5),
    }


def ekf_row(msg: Any, now: float) -> dict[str, Any]:
    """The fused odometry (odom -> base_link) the tracker pairs scans with, pose and rates."""
    return {
        "t": stamp(msg.header, now),
        "topic": "ekf",
        "x": round(msg.pose.pose.position.x, 4),
        "y": round(msg.pose.pose.position.y, 4),
        "theta": round(yaw(msg.pose.pose.orientation), 5),
        "vx": round(msg.twist.twist.linear.x, 4),
        "wz": round(msg.twist.twist.angular.z, 4),
    }


def laser_odom_row(msg: Any, now: float) -> dict[str, Any]:
    """The laser scan-matcher's twist in base_link (rf2o, /odom_laser), stamped by the scan."""
    return {
        "t": stamp(msg.header, now),
        "topic": "laser_odom",
        "vx": round(msg.twist.twist.linear.x, 4),
        "vy": round(msg.twist.twist.linear.y, 4),
        "wz": round(msg.twist.twist.angular.z, 4),
    }


def plan_row(msg: Any, now: float) -> dict[str, Any]:
    """Nav2's global plan as a polyline (at most 200 points), so a replay can draw it."""
    step = max(1, len(msg.poses) // 200)
    return {
        "t": stamp(msg.header, now),
        "topic": "plan",
        "points": [
            [round(p.pose.position.x, 3), round(p.pose.position.y, 3)] for p in msg.poses[::step]
        ],
    }


def costmap_row(msg: Any, now: float) -> dict[str, Any]:
    """The local costmap as the controller sees it, cells verbatim (ROS 0-100 plus -1 unknown;
    99-100 is the lethal/inscribed band)."""
    info = msg.info
    return {
        "t": stamp(msg.header, now),
        "topic": "costmap",
        "origin": [round(info.origin.position.x, 3), round(info.origin.position.y, 3)],
        "resolution": round(info.resolution, 3),
        "width": int(info.width),
        "height": int(info.height),
        "data": list(msg.data),
    }


def gcostmap_row(msg: Any, now: float, plan_seq: int) -> dict[str, Any]:
    """The grid the PLANNER plans on, reduced to the four feasibility classes and run-length
    encoded (16 kB instead of 195 kB for this flat's 239x215 cells); ``plan_seq`` is the plan it
    belonged to, which is also the throttle its recorder keeps."""
    info = msg.info
    return {
        "t": stamp(msg.header, now),
        "topic": "gcostmap",
        "origin": [round(info.origin.position.x, 3), round(info.origin.position.y, 3)],
        "resolution": round(info.resolution, 3),
        "width": int(info.width),
        "height": int(info.height),
        "encoding": RLE,
        "classes": [UNKNOWN, FREE, INFLATED, LETHAL_BAND],
        "plan": plan_seq,
        "data": run_length_encode(feasibility_classes(list(msg.data))),
    }


def tof_row(sensor: str, msg: Any, now: float) -> dict[str, Any]:
    """One ToF reading and its ceiling: a beam at max range means the sensor saw nothing."""
    return {
        "t": stamp(msg.header, now),
        "topic": "tof",
        "sensor": sensor,
        "range": round(float(msg.range), 3),
        "max": round(float(msg.max_range), 3),
    }


def nav_row(action: str, msg: Any, now: float) -> dict[str, Any]:
    """What became of one of Nav2's actions: every goal's status code in the array
    (action_msgs/GoalStatus: 2 executing, 4 succeeded, 5 canceled, 6 aborted)."""
    return {
        "t": now,
        "topic": "nav",
        "action": action,
        "status": [int(s.status) for s in msg.status_list],
    }


def meas_row(text: str, now: float) -> dict[str, Any]:
    """One pose measured out of a camera scan or out of the pose graph, kept verbatim: ``t`` is
    when it ARRIVED, the moment it speaks for is ``stamp`` inside the JSON."""
    return {"t": now, "topic": "meas", "json": text}


def srcs_row(text: str, now: float) -> dict[str, Any]:
    """The tracker's own account of one update (Localizer.sources_report), verbatim."""
    return {"t": now, "topic": "srcs", "json": text}


def cmd_row(msg: Any, now: float) -> dict[str, Any]:
    """What the controller asked the wheels for: the only record of the command side."""
    return {
        "t": now,
        "topic": "cmd",
        "linear": round(msg.linear.x, 4),
        "angular": round(msg.angular.z, 4),
    }


def loc_row(msg: Any, now: float) -> dict[str, Any]:
    """The tracker's belief in the map frame (/tracker_pose); covariance trace as a stand-in
    confidence."""
    cov = msg.pose.covariance
    return {
        "t": stamp(msg.header, now),
        "topic": "loc",
        "x": round(msg.pose.pose.position.x, 4),
        "y": round(msg.pose.pose.position.y, 4),
        "theta": round(yaw(msg.pose.pose.orientation), 5),
        "confidence": round(1.0 / (1.0 + cov[0] + cov[7] + cov[35]), 3),
    }


def loc_row_from_transform(transform: Any, now: float) -> dict[str, Any]:
    """The same record read from ``map -> base_link`` where no tracker publishes a pose.

    No ``confidence``: TF carries no covariance, and a made-up number in a tape is worse than a
    missing one — ``source`` says which of the two wrote the record.
    """
    return {
        "t": stamp(transform.header, now),
        "topic": "loc",
        "source": "tf",
        "x": round(transform.transform.translation.x, 4),
        "y": round(transform.transform.translation.y, 4),
        "theta": round(yaw(transform.transform.rotation), 5),
    }
