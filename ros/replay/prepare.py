"""One recorded drive made ready for costmap_replay: what the board's costmaps were fed, rebuilt.

QUESTION: the board bag (``ros/maps/rec/NNNN_*/``, pepin_bringup.bag_recorder) keeps the RAW lidar,
the ToF as sensor_msgs/Range and no map; Nav2 was fed the hull-filtered ``/scan``, the ToF as
LaserScan fans, RTAB-Map's ``/map`` and a ClearEntireCostmap every few seconds by the tree. What
exactly did each costmap receive, and when?

METHOD, per drive, in the laptop-side replay container (rosbag2_py, rclpy, this repo's pepin):
    * ``/ldlidar_node/scan`` -> ``/scan``: laser_filters' LaserScanBoxFilter as robot.launch.py
      runs it — every return laser_geometry would project (range_min <= r < range_max) is placed
      in base_link through the bag's own base_link -> laser and set NaN when it falls strictly
      inside :func:`pepin.footprint.hull_box` (z within +-1 m);
    * ``/tof/<name>`` (Range) -> ``/tof/<name>/scan``: :func:`pepin_bringup.tof_bridge.fan_scan`,
      the bridge's own function, from the Range's verdict and ceiling;
    * ``/tf``, ``/tf_static``, ``/depth_marks``, ``/depth_free`` and, where recorded,
      ``/depth_scan`` and ``/contact_scan`` pass through byte for byte;
    * ``/map``: the bag's own where it has one; otherwise the stand's static grid (RTAB-Map's
      saved grid, :func:`stand_from_db`) once at the bag's first instant, in frame ``map``;
    * ``/replay/clear``: the behaviour tree's ClearEntireCostmap calls, which no topic records,
      rebuilt from the action status topics (:func:`clear_schedule`): each RateController-wrapped
      clear of ros/params/pepin_nav_to_pose.xml at the goal's acceptance and then every period
      plus one tree tick; the pipeline's restarts (a planner or controller goal CANCELLED by the
      tree, then the next planner goal) re-anchor both and clear both; a local clear per FollowPath
      failure and a global one per ComputePathToPose failure (an abort with no other goal of that
      server active — the ones beside a new goal are preemptions). A pipeline that failed with
      nothing running to cancel is invisible here.
All of it goes to ``<cache>/prepared/NNNN/bag`` with every message at its recorded receive time
(the board's clock, which is also what the costmaps' MessageFilters waited in).

ANSWER: ``prepared/NNNN/bag`` (MCAP) and ``prepared/NNNN/drive.json``: the goal pose and window, the
clears, the recorded /cmd_vel and /plan, and ``recorded_local.npz`` (the board's own
/local_costmap/costmap, what ros/replay/score.py measures the replay's fidelity against).
"""

from __future__ import annotations

import base64
import itertools
import json
import math
import re
import sqlite3
import struct
import xml.etree.ElementTree as ET
import zlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

# What the prepared bag carries through unchanged: the frames and the camera's words.
PASS_THROUGH = ("/tf", "/tf_static", "/depth_marks", "/depth_free", "/depth_scan", "/contact_scan")
RAW_SCAN = "/ldlidar_node/scan"
SCAN = "/scan"
TOF = ("/tof/front", "/tof/left", "/tof/right")
NAVIGATE = "/navigate_to_pose/_action/status"
FOLLOW = "/follow_path/_action/status"
COMPUTE = "/compute_path_to_pose/_action/status"
CLEAR_TOPIC = "/replay/clear"
MAP_TOPIC = "/map"
ACCEPTED, EXECUTING, CANCELING, SUCCEEDED, CANCELED, ABORTED = 1, 2, 3, 4, 5, 6
TERMINAL = (SUCCEEDED, CANCELED, ABORTED)
# The version of what this module writes: a prepared drive of another version is prepared again.
PREPARE_VERSION = 6
Matrix = npt.NDArray[np.float64]


@dataclass(frozen=True)
class Stand:
    """What every drive of a replay is judged against: the static grid and the named places."""

    grid: npt.NDArray[np.int8]  # int8 (height, width), ROS values: -1 unknown, 0 free, 100 occupied
    origin: tuple[float, float]
    resolution: float
    goals: dict[str, tuple[float, float, float]]  # place -> (x, y, yaw) in map
    source: str

    def to_json(self) -> dict[str, Any]:
        """The stand as it is kept inside a baseline: the grid zlib-compressed, base64."""
        packed = base64.b64encode(zlib.compress(self.grid.astype(np.int8).tobytes(), 9))
        return {
            "source": self.source,
            "origin": list(self.origin),
            "resolution": self.resolution,
            "shape": list(self.grid.shape),
            "grid_zlib_b64": packed.decode(),
            "goals": {k: list(v) for k, v in sorted(self.goals.items())},
        }

    @staticmethod
    def from_json(d: dict[str, Any]) -> Stand:
        """The stand a baseline was scored on."""
        raw = zlib.decompress(base64.b64decode(d["grid_zlib_b64"]))
        grid = np.frombuffer(raw, dtype=np.int8).reshape(d["shape"]).copy()
        goals = {k: (float(v[0]), float(v[1]), float(v[2])) for k, v in d["goals"].items()}
        return Stand(grid, (d["origin"][0], d["origin"][1]), d["resolution"], goals, d["source"])

    def fingerprint(self) -> str:
        """A short hash of the grid and the goals: two stands with the same one are the same."""
        h = zlib.crc32(self.grid.tobytes())
        h = zlib.crc32(json.dumps(sorted(self.goals.items())).encode(), h)
        return f"{h:08x}"


def _uncompress(blob: bytes) -> tuple[int, int, int, bytes]:
    """RTAB-Map's compressData2 format: zlib data, then rows, cols and the cv type as int32."""
    rows, cols, cv_type = struct.unpack("<iii", blob[-12:])
    return rows, cols, cv_type, zlib.decompress(blob[:-12])


def stand_from_db(db: Path, places: Path) -> Stand:
    """RTAB-Map's own saved grid (the Admin table's opt_map: what /map was, read-only) and each
    place of ``places`` (rtabmap.places.json: a pose in the frame of a graph node) resolved on
    the saved optimised graph — the same arithmetic the goal server resolves a place with."""
    con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        row = con.execute(
            "select opt_map, opt_map_x_min, opt_map_y_min, opt_map_resolution, opt_ids, opt_poses"
            " from Admin"
        ).fetchone()
    finally:
        con.close()
    if row is None or row[0] is None:
        raise SystemExit(f"{db}: no saved grid in the Admin table (RTAB-Map never closed it?)")
    rows, cols, _, data = _uncompress(row[0])
    grid = np.frombuffer(data, dtype=np.int8).reshape(rows, cols).copy()
    ids = np.frombuffer(_uncompress(row[4])[3], dtype=np.int32)
    poses = np.frombuffer(_uncompress(row[5])[3], dtype=np.float32).reshape(-1, 12)
    goals: dict[str, tuple[float, float, float]] = {}
    for name, place in json.loads(places.read_text())["places"].items():
        hit = np.nonzero(ids == int(place["node"]))[0]
        if not len(hit):
            continue
        p = poses[hit[0]]
        yaw = math.atan2(float(p[4]), float(p[0]))
        dx, dy = float(place["dx"]), float(place["dy"])
        goals[name] = (
            float(p[3]) + dx * math.cos(yaw) - dy * math.sin(yaw),
            float(p[7]) + dx * math.sin(yaw) + dy * math.cos(yaw),
            yaw + math.radians(float(place["dtheta_deg"])),
        )
    source = f"{db.name} saved grid, {places.name}"
    return Stand(grid, (float(row[1]), float(row[2])), float(row[3]), goals, source)


def goal_name(bag: Path) -> str:
    """The place a drive went to: the last word of its name (``0496_..._printer``)."""
    return bag.name.rsplit("_", 1)[-1]


def run_number(bag: Path) -> int:
    """The drive's number, the name's first field."""
    return int(bag.name.split("_", 1)[0])


def bt_clear_periods(tree: Path) -> dict[str, float]:
    """Seconds between the tree's periodic ClearEntireCostmap calls per costmap: each
    RateController whose child is a ClearEntireCostmap (``local`` / ``global`` by service name)."""
    periods: dict[str, float] = {}
    for rate in ET.parse(tree).iter("RateController"):
        for child in rate:
            if child.tag == "ClearEntireCostmap":
                which = "local" if "local" in child.get("service_name", "") else "global"
                periods[which] = 1.0 / float(rate.get("hz", "1"))
    return periods


@dataclass
class ActionGoals:
    """The goals of one Nav2 action server as its status topic told them: for each goal the
    moment it was accepted and the first receive time of every status it went through."""

    goals: dict[str, dict[str, Any]]

    @staticmethod
    def empty() -> ActionGoals:
        return ActionGoals({})

    def feed(self, t: float, statuses: list[tuple[str, int, float]]) -> None:
        """One GoalStatusArray, received at ``t``: (uuid, status, accepted stamp) per goal."""
        active = sum(status in (ACCEPTED, EXECUTING) for _, status, _ in statuses)
        for uuid, status, accepted in statuses:
            g = self.goals.setdefault(uuid, {"accepted": accepted, "seen": {}, "first": status})
            if status not in g["seen"]:
                g["seen"][status] = t
                # Nav2's SimpleActionServer ABORTS the goal a new one preempts
                # (accept_pending_goal): FollowPath "aborts" once a second as the planner hands it
                # each new path. A failure is an abort with no other goal of the server active.
                if status == ABORTED:
                    g["preempted"] = active > 0

    def failures(self, bag_start: float) -> list[float]:
        """When a goal failed inside the bag (not preempted, not already over at its start)."""
        out = []
        for g in self.goals.values():
            if ABORTED not in g["seen"] or g.get("preempted", False):
                continue
            if g["first"] == ABORTED and g["accepted"] < bag_start:
                continue  # ended before the recording began
            out.append(g["seen"][ABORTED])
        return sorted(out)

    def cancels(self, bag_start: float) -> list[float]:
        """When the tree cancelled a goal inside the bag (CANCELING or CANCELED first seen)."""
        out = []
        for g in self.goals.values():
            times = [g["seen"][s] for s in (CANCELING, CANCELED) if s in g["seen"]]
            if times and not (g["first"] in (CANCELING, CANCELED) and g["accepted"] < bag_start):
                out.append(min(times))
        return sorted(out)

    def outcomes(self) -> list[tuple[float, int]]:
        """Every goal as (acceptance stamp, last status seen), sorted by acceptance."""
        return sorted(
            (float(g["accepted"]), max(g["seen"], key=lambda s: g["seen"][s]))
            for g in self.goals.values()
        )


def bt_planner_period(tree: Path) -> float:
    """Seconds between the tree's ComputePathToPose ticks: the RateController around it."""
    for rate in ET.parse(tree).iter("RateController"):
        if any(node.tag == "ComputePathToPose" for node in rate.iter()):
            return 1.0 / float(rate.get("hz", "1"))
    return 1.0


# A planner goal sooner than this share of the planner's period after the last one that did not
# fail can only come from a RateController ticking for the first time: a restarted pipeline.
EARLY_PLANNER_SHARE = 0.8


def pipeline_restarts(
    cancels: list[float], planner_goals: list[tuple[float, int]], planner_period_s: float
) -> list[float]:
    """When the tree's PipelineSequence started again after it failed. The restart ticks every
    RateController for the first time, the planner's among them, so a ComputePathToPose comes at
    once: after a cancel (the failing pipeline halted a running planner or controller goal), or
    sooner than a period after the last planner goal — unless that one failed, whose retry comes
    from the planner's own RecoveryNode. ``planner_goals`` is (accepted, final status), sorted."""
    out: set[float] = set()
    accepted = [a for a, _ in planner_goals]
    for c in cancels:
        after = [a for a in accepted if a > c]
        if after:
            out.add(after[0])
    for (prev, status), (cur, _) in itertools.pairwise(planner_goals):
        if status != ABORTED and cur - prev < EARLY_PLANNER_SHARE * planner_period_s:
            out.add(cur)
    return sorted(out)


def clear_schedule(
    start: float,
    end: float,
    periods: dict[str, float],
    loop_s: float,
    restarts: list[float],
    follow_failures: list[float],
    compute_failures: list[float],
) -> list[tuple[float, str]]:
    """The ClearEntireCostmap calls of one goal, sorted. A RateController ticks its clear at once
    when the pipeline (re)starts — the goal's acceptance and every restart — and then whenever a
    period has passed since the last clear succeeded, which on a tree ticked every ``loop_s``
    is a period plus a tick; a restart re-anchors it and is itself a clear of both (the recovery's
    ClearingActions or the RateControllers' first tick). Beside them: a local clear at every
    FollowPath failure (ClearLocalCostmap-Context) and a global one at every ComputePathToPose
    failure (ClearGlobalCostmap-Context)."""
    out: list[tuple[float, str]] = []
    anchors = sorted({start, *(r for r in restarts if start < r <= end)})
    for k, anchor in enumerate(anchors):
        stop = anchors[k + 1] if k + 1 < len(anchors) else end
        for which, period in periods.items():
            t = anchor
            while t < stop or (t == anchor and t <= end):
                out.append((t, which))
                t += period + loop_s
    out += [(t, "local") for t in follow_failures if start <= t <= end]
    out += [(t, "global") for t in compute_failures if start <= t <= end]
    return sorted(out)


def bt_loop_s(params: Path) -> float:
    """bt_navigator's bt_loop_duration (ms in the file) in seconds; Nav2's 10 ms if unset."""
    import yaml

    doc = yaml.safe_load(params.read_text()) or {}
    ms = doc.get("bt_navigator", {}).get("ros__parameters", {}).get("bt_loop_duration", 10)
    return float(ms) / 1000.0


def bag_topics(bag: Path) -> dict[str, str]:
    """Every topic of a bag directory and its type."""
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    return {t.name: t.type for t in reader.get_all_topics_and_types()}


def read_bag(
    bag: Path, topics: tuple[str, ...] | None = None
) -> Iterator[tuple[str, str, bytes, int]]:
    """Every message of a bag directory in receive order: (topic, type, CDR bytes, receive ns)."""
    import rosbag2_py

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    types = {t.name: t.type for t in reader.get_all_topics_and_types()}
    if topics is not None:
        reader.set_filter(rosbag2_py.StorageFilter(topics=[t for t in topics if t in types]))
    while reader.has_next():
        topic, data, t_ns = reader.read_next()
        yield topic, types[topic], data, t_ns


def _quaternion_matrix(q: Any) -> Matrix:
    """3x3 rotation of a geometry_msgs quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def hull_filtered(scan: Any, rotation: Matrix, translation: Matrix, box: dict[str, float]) -> Any:
    """``scan`` with every return inside the hull box set NaN, as LaserScanBoxFilter does it."""
    r = np.asarray(scan.ranges, dtype=np.float64)
    a = scan.angle_min + np.arange(r.size) * scan.angle_increment
    with np.errstate(invalid="ignore"):
        kept = (r >= scan.range_min) & (r < scan.range_max)
    pts = (
        np.stack([r * np.cos(a), r * np.sin(a), np.zeros_like(r)], axis=1) @ rotation.T
        + translation
    )
    inside = (
        kept
        & (pts[:, 0] > box["min_x"])
        & (pts[:, 0] < box["max_x"])
        & (pts[:, 1] > box["min_y"])
        & (pts[:, 1] < box["max_y"])
        & (pts[:, 2] > -1.0)
        & (pts[:, 2] < 1.0)
    )
    out = list(scan.ranges)
    for i in np.nonzero(inside)[0]:
        out[int(i)] = math.nan
    scan.ranges = out
    return scan


def _uuid(status: Any) -> str:
    return bytes(status.goal_info.goal_id.uuid).hex()


def _stamp(msg_stamp: Any) -> float:
    return float(msg_stamp.sec) + float(msg_stamp.nanosec) * 1e-9


def prepare(bag: Path, out: Path, stand: Stand, tree: Path, params: Path) -> dict[str, Any]:
    """Write ``out/bag`` and ``out/drive.json`` for one recorded drive; returns the drive's
    description (what drive.json holds)."""
    import rosbag2_py
    from nav_msgs.msg import OccupancyGrid
    from pepin_bringup.tof_bridge import fan_scan
    from rclpy.serialization import deserialize_message, serialize_message
    from rosidl_runtime_py.utilities import get_message
    from std_msgs.msg import String

    from pepin.footprint import hull_box

    box = hull_box()
    out.mkdir(parents=True, exist_ok=True)
    target = out / "bag"
    if target.exists():
        for f in target.iterdir():
            f.unlink()
        target.rmdir()
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=str(target), storage_id="mcap"),
        rosbag2_py.ConverterOptions("cdr", "cdr"),
    )
    created: set[str] = set()

    def write(topic: str, type_name: str, data: bytes, t_ns: int) -> None:
        if topic not in created:
            writer.create_topic(rosbag2_py.TopicMetadata(len(created), topic, type_name, "cdr"))
            created.add(topic)
        writer.write(topic, data, t_ns)

    laser: tuple[Matrix, Matrix] | None = None
    waiting_scans: list[tuple[Any, int]] = []
    navigate: dict[str, dict[str, float]] = {}
    follow = ActionGoals.empty()
    compute = ActionGoals.empty()
    cmds: list[list[float]] = []
    plans: list[dict[str, Any]] = []
    local: list[tuple[float, float, float, float, int, int, bytes]] = []
    first_ns = last_ns = None
    msg_types: dict[str, Any] = {}

    def decode(type_name: str, data: bytes) -> Any:
        if type_name not in msg_types:
            msg_types[type_name] = get_message(type_name)
        return deserialize_message(data, msg_types[type_name])

    # The map the costmaps had, where the bag holds it; the stand's grid where it does not.
    map_recorded = MAP_TOPIC in bag_topics(bag)
    passed = (*PASS_THROUGH, MAP_TOPIC) if map_recorded else PASS_THROUGH
    for topic, type_name, data, t_ns in read_bag(bag):
        if first_ns is None:
            first_ns = t_ns
        if not map_recorded and t_ns == first_ns and MAP_TOPIC not in created:
            grid = OccupancyGrid()
            grid.header.frame_id = "map"
            grid.header.stamp.sec, grid.header.stamp.nanosec = divmod(t_ns, 1_000_000_000)
            grid.info.resolution = stand.resolution
            grid.info.height, grid.info.width = stand.grid.shape
            grid.info.origin.position.x, grid.info.origin.position.y = stand.origin
            grid.info.origin.orientation.w = 1.0
            grid.data = stand.grid.astype(np.int8).ravel().tolist()
            write(MAP_TOPIC, "nav_msgs/msg/OccupancyGrid", serialize_message(grid), t_ns)
        last_ns = t_ns
        t = t_ns * 1e-9
        if topic in passed:
            write(topic, type_name, data, t_ns)
            if topic == "/tf_static" and laser is None:
                for tr in decode(type_name, data).transforms:
                    if tr.header.frame_id == "base_link" and tr.child_frame_id == "laser":
                        v = tr.transform.translation
                        laser = (
                            _quaternion_matrix(tr.transform.rotation),
                            np.array([v.x, v.y, v.z]),
                        )
                if laser is not None:
                    for scan, t_scan in waiting_scans:
                        write(
                            SCAN,
                            "sensor_msgs/msg/LaserScan",
                            serialize_message(hull_filtered(scan, *laser, box)),
                            t_scan,
                        )
                    waiting_scans.clear()
        elif topic == RAW_SCAN:
            scan = decode(type_name, data)
            if laser is None:
                waiting_scans.append((scan, t_ns))
            else:
                write(
                    SCAN,
                    "sensor_msgs/msg/LaserScan",
                    serialize_message(hull_filtered(scan, *laser, box)),
                    t_ns,
                )
        elif topic in TOF:
            r = decode(type_name, data)
            fan = fan_scan(r.range, r.max_range, r.header.stamp, r.header.frame_id)
            write(f"{topic}/scan", "sensor_msgs/msg/LaserScan", serialize_message(fan), t_ns)
        elif topic == NAVIGATE:
            for s in decode(type_name, data).status_list:
                g = navigate.setdefault(_uuid(s), {"accepted": _stamp(s.goal_info.stamp)})
                if s.status in TERMINAL and "end" not in g:
                    g["end"], g["status"] = t, float(s.status)
        elif topic in (FOLLOW, COMPUTE):
            (follow if topic == FOLLOW else compute).feed(
                t,
                [
                    (_uuid(s), int(s.status), _stamp(s.goal_info.stamp))
                    for s in decode(type_name, data).status_list
                ],
            )
        elif topic == "/cmd_vel":
            m = decode(type_name, data)
            cmds.append([t, m.linear.x, m.angular.z])
        elif topic == "/plan":
            m = decode(type_name, data)
            pts = [[p.pose.position.x, p.pose.position.y] for p in m.poses]
            plans.append({"t": t, "frame": m.header.frame_id, "points": pts})
        elif topic == "/local_costmap/costmap":
            m = decode(type_name, data)
            local.append(
                (
                    _stamp(m.header.stamp),
                    m.info.origin.position.x,
                    m.info.origin.position.y,
                    m.info.resolution,
                    m.info.width,
                    m.info.height,
                    bytes(np.asarray(m.data, dtype=np.int8).tobytes()),
                )
            )
    if first_ns is None or last_ns is None:
        raise SystemExit(f"{bag}: empty bag")
    bag_start, bag_end = first_ns * 1e-9, last_ns * 1e-9
    # The goal the drive was about: the one accepted last before the bag's end.
    goals = sorted(navigate.values(), key=lambda g: g["accepted"])
    goal = goals[-1] if goals else {"accepted": bag_start}
    start, end = goal["accepted"], goal.get("end", bag_end)
    follow_failures = follow.failures(bag_start)
    compute_failures = compute.failures(bag_start)
    cancels = [c for c in follow.cancels(bag_start) + compute.cancels(bag_start) if c < end]
    restarts = [
        r
        for r in pipeline_restarts(sorted(cancels), compute.outcomes(), bt_planner_period(tree))
        if r < end
    ]
    clears = clear_schedule(
        start,
        end,
        bt_clear_periods(tree),
        bt_loop_s(params),
        restarts,
        follow_failures,
        compute_failures,
    )
    # Written last, out of order: an MCAP is read back by receive time whatever the order of the
    # writes (costmap_replay refuses a bag that comes back out of order). A clear the tree issued
    # before the bag began lands on its first instant: what it wiped, the replay never had.
    for t_clear, which in clears:
        t_ns = max(first_ns, round(t_clear * 1e9))
        write(CLEAR_TOPIC, "std_msgs/msg/String", serialize_message(String(data=which)), t_ns)
    close = getattr(writer, "close", None)  # otherwise the file closes with the writer
    if close is not None:
        close()
    name = goal_name(bag)
    description: dict[str, Any] = {
        "version": PREPARE_VERSION,
        "run": run_number(bag),
        "bag": bag.name,
        "goal_name": name,
        "goal": list(stand.goals[name]) if name in stand.goals else None,
        "stand": stand.fingerprint(),
        "map": "recorded /map" if map_recorded else stand.source,
        "bag_start": bag_start,
        "bag_end": bag_end,
        "goal_accepted": start,
        "goal_end": end,
        "goal_status": int(goal.get("status", 0)),
        "clears": clears,
        "follow_failures": follow_failures,
        "compute_failures": compute_failures,
        "pipeline_restarts": restarts,
        "cmd_vel": cmds,
        "plans": plans,
    }
    (out / "drive.json").write_text(json.dumps(description))
    np.savez_compressed(
        out / "recorded_local.npz",
        t=np.array([x[0] for x in local]),
        origin=np.array([[x[1], x[2]] for x in local]).reshape(-1, 2),
        resolution=np.array([x[3] for x in local]),
        shape=np.array([[x[5], x[4]] for x in local], dtype=np.int64).reshape(-1, 2),
        data=np.frombuffer(b"".join(x[6] for x in local), dtype=np.int8),
    )
    return description


def bag_dirs(rec: Path, runs: list[int]) -> list[Path]:
    """The bag directory of each run number, in order; a run with no bag is said and skipped."""
    out = []
    for n in runs:
        hits = sorted(p for p in rec.glob(f"{n:04d}_*") if p.is_dir() and any(p.glob("*.mcap")))
        if hits:
            out.append(hits[0])
        else:
            print(f"run {n:04d}: no bag under {rec}, skipped")
    return out


def parse_runs(spec: list[str]) -> list[int]:
    """``483-498 500`` -> [483, ..., 498, 500]."""
    runs: list[int] = []
    for part in spec:
        m = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not m:
            raise SystemExit(f"not a run or a range of runs: {part}")
        lo = int(m.group(1))
        hi = int(m.group(2) or lo)
        runs += list(range(lo, hi + 1))
    return runs
