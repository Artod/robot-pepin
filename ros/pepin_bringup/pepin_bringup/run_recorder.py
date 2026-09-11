"""ROS wiring for `pepin.tape.RunTape`: the topics of a drive, turned into records.

The node that owns the goal owns the tape, so a run's file opens the moment the goal does and
closes with it — and, because the tape always holds the last seconds, it opens with the seconds
*before* the goal on it. The decisions live in `pepin.tape`; this file only converts messages.

Topics: the raw lidar scan, wheel odometry, the tracker's pose, Nav2's plan, the commanded
twist, the three ToF ranges and the local costmap — what `scripts/build_map.py` and the replays
read, plus the proof of what the robot itself believed: which cells it held for occupied when it
refused to move (a question a run could not answer before, and the one every stall raises).
"""

from __future__ import annotations

import math
import subprocess
import time
from pathlib import Path
from typing import Any, ClassVar

from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from nav_msgs.msg import OccupancyGrid, Odometry
from nav_msgs.msg import Path as PathMsg
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy, qos_profile_sensor_data
from sensor_msgs.msg import Imu, LaserScan, Range
from std_msgs.msg import String

from pepin.mounts import Mounts
from pepin.recording import scan_record_from_ros
from pepin.runlink import (
    IDLE,
    RECORDING,
    RUN_COMMAND_TOPIC,
    RUN_STATUS_TOPIC,
    RunStatus,
    parse_command,
)
from pepin.tape import RunTape, camera_clip_path, next_run_number


def _yaw(orientation: object) -> float:
    """Yaw in radians from a quaternion message."""
    q = orientation
    x, y, z, w = (getattr(q, name) for name in ("x", "y", "z", "w"))
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _stamp(header: object) -> float:
    """The message's own time in seconds; wall time when it carries none."""
    stamp = getattr(header, "stamp", None)
    seconds = stamp.sec + stamp.nanosec * 1e-9 if stamp else 0.0
    return seconds if seconds > 0 else time.time()


class RunRecorder:
    """Subscribes a node to the topics of a drive and feeds them to the run's tape."""

    # While no run is open the tape only keeps a sketch: turning every scan into a record costs
    # 40% of a core on this board (measured 2026-09-08, load average 9 with the controller loop
    # down to 3 Hz), and a prelude does not need 10 Hz. During a run nothing is thinned.
    IDLE_PERIOD_S: ClassVar[dict[str, float]] = {"pose": 0.2, "loc": 0.2}

    def __init__(self, node: Node, directory: Path, tape: RunTape | None = None) -> None:
        self._node = node
        self._last_kept: dict[str, float] = {}
        self._directory = directory
        self._tape = tape or RunTape()
        self.number = 0  # the run's number, said aloud instead of a timestamp
        # The laser's place on the cart, from config/lidar.json: the replays turn the driver's
        # own bearings into robot-frame ones with it (pepin.recording.scan_record_from_ros
        # wants the offset the mount's yaw is the negative of).
        lidar = Mounts.load().lidar
        self._mount_yaw_rad = -math.radians(lidar.yaw_deg)
        self._mount_x_m = lidar.x_m
        scan_qos = QoSProfile(depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self._scan_qos = scan_qos
        # Always on: small messages, and they are what the prelude is made of.
        node.create_subscription(Odometry, "/odom", self._on_odom, 20)
        node.create_subscription(PoseWithCovarianceStamped, "/tracker_pose", self._on_loc, 10)
        node.create_subscription(Twist, "/cmd_vel", self._on_cmd, 20)
        # Only while a run is open: rclpy turns every LaserScan into Python objects BEFORE our
        # callback can decline it, and that deserialisation alone cost 38% of a core between
        # goals on this board (measured 2026-09-08, load average 8.4). A drive gets them from
        # the moment its tape opens, which is before the goal is even sent.
        self._during_run: list[Any] = []
        # rclpy subscriptions may be created and destroyed ONLY on the thread that spins the
        # executor. Doing it from the socket thread killed the node mid-run with "cannot use
        # Destroyable because destruction was requested" — the executor was building its wait set
        # out of the very objects being torn down (run 0027). So the socket thread asks, and this
        # timer, which runs where the executor runs, does it.
        node.create_timer(0.1, self._apply_pending)

    @property
    def recording(self) -> bool:
        """True while a run is being written."""
        return self._tape.recording

    def start(self, name: str) -> Path:
        """Open a numbered tape for this run; returns the path, prelude already in it."""
        self.number = next_run_number(self._directory)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        return self._tape.start(self._directory / f"{self.number:04d}_{stamp}_{name}.jsonl")

    def stop(self) -> None:
        """Close the run's tape and give the board back the cores the lidar stream was costing."""
        self._tape.stop()

    def _apply_pending(self) -> None:
        """Executor thread: keep the run-only subscriptions in step with whether a tape is open.

        Level-triggered on the tape's own state, not on a flag set from the socket thread: an
        edge written there and cleared here could be lost, leaving a run with no scans, and a
        tape that closed itself on its own limit left the subscriptions attached for good.
        """
        want = "listen" if self._tape.recording else "deafen"
        if want == "listen" and not self._during_run:
            self._during_run = [
                self._node.create_subscription(
                    LaserScan, "/ldlidar_node/scan", self._on_scan, self._scan_qos
                ),
                self._node.create_subscription(PathMsg, "/plan", self._on_plan, 5),
                # What the tracker pairs scans with, and the gyro it is built from: without
                # them a wobble of map -> odom cannot be told from wheel slip after the fact.
                self._node.create_subscription(Odometry, "/odometry/filtered", self._on_ekf, 20),
                self._node.create_subscription(
                    Imu, "/imu/data_raw", self._on_imu, qos_profile_sensor_data
                ),
                self._node.create_subscription(
                    OccupancyGrid, "/local_costmap/costmap", self._on_costmap, 1
                ),
                # The ToF too: three sensors at 14 Hz are 42 messages a second that rclpy
                # deserialises before the tape can decline them — a third of a core between
                # goals, for a prelude nobody reads. A drive gets them from its first moment.
                *(
                    self._node.create_subscription(
                        Range, f"/tof/{sensor}", lambda msg, s=sensor: self._on_tof(s, msg), 10
                    )
                    for sensor in ("front", "left", "right")
                ),
            ]
        elif want == "deafen" and self._during_run:
            for subscription in self._during_run:
                self._node.destroy_subscription(subscription)
            self._during_run = []

    def _keep(self, topic: str) -> bool:
        """True when this record is worth building: everything during a run, a sketch when idle."""
        if self._tape.recording:
            return True
        period = self.IDLE_PERIOD_S.get(topic)
        if period is None:
            return False  # the plan and the costmap only matter while a run is being driven
        now = time.monotonic()
        if now - self._last_kept.get(topic, 0.0) < period:
            return False
        self._last_kept[topic] = now
        return True

    def _on_scan(self, msg: LaserScan) -> None:
        if not self._keep("scan"):
            return
        self._tape.add(
            scan_record_from_ros(
                _stamp(msg.header),
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
        )

    def _on_odom(self, msg: Odometry) -> None:
        if not self._keep("pose"):
            return
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "pose",
                "x": round(msg.pose.pose.position.x, 4),
                "y": round(msg.pose.pose.position.y, 4),
                "theta": round(_yaw(msg.pose.pose.orientation), 5),
            }
        )

    def _on_ekf(self, msg: Odometry) -> None:
        """The fused odometry (odom -> base_link) the tracker pairs scans with, pose and rates."""
        if not self._keep("ekf"):
            return
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "ekf",
                "x": round(msg.pose.pose.position.x, 4),
                "y": round(msg.pose.pose.position.y, 4),
                "theta": round(_yaw(msg.pose.pose.orientation), 5),
                "vx": round(msg.twist.twist.linear.x, 4),
                "wz": round(msg.twist.twist.angular.z, 4),
            }
        )

    def _on_imu(self, msg: Imu) -> None:
        """The gyro's yaw rate, raw: the heading truth the wheels are checked against."""
        if not self._keep("imu"):
            return
        self._tape.add(
            {"t": _stamp(msg.header), "topic": "imu", "wz": round(msg.angular_velocity.z, 4)}
        )

    def _on_plan(self, msg: PathMsg) -> None:
        """Nav2's global plan as a polyline (at most 200 points), so a replay can draw it."""
        if not self._keep("plan"):
            return
        step = max(1, len(msg.poses) // 200)
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "plan",
                "points": [
                    [round(p.pose.position.x, 3), round(p.pose.position.y, 3)]
                    for p in msg.poses[::step]
                ],
            }
        )

    def _on_costmap(self, msg: OccupancyGrid) -> None:
        """The local costmap as the controller sees it (1 Hz, 3x3 m at 5 cm: about 14 kB/s).

        Costs are the ROS 0-100 scale plus -1 for unknown; 99-100 is the lethal/inscribed band.
        """
        if not self._keep("costmap"):
            return
        info = msg.info
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "costmap",
                "origin": [round(info.origin.position.x, 3), round(info.origin.position.y, 3)],
                "resolution": round(info.resolution, 3),
                "width": int(info.width),
                "height": int(info.height),
                "data": list(msg.data),
            }
        )

    def _on_tof(self, sensor: str, msg: Range) -> None:
        """One ToF reading and its ceiling: a beam at max range means the sensor saw nothing."""
        if not self._keep("tof"):
            return
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "tof",
                "sensor": sensor,
                "range": round(float(msg.range), 3),
                "max": round(float(msg.max_range), 3),
            }
        )

    def _on_cmd(self, msg: Twist) -> None:
        """What the controller asked the wheels for: the only record of the command side."""
        self._tape.add(
            {
                "t": time.time(),
                "topic": "cmd",
                "linear": round(msg.linear.x, 4),
                "angular": round(msg.angular.z, 4),
            }
        )

    def _on_loc(self, msg: PoseWithCovarianceStamped) -> None:
        """The tracker's belief in the map frame; covariance trace as a stand-in confidence."""
        if not self._keep("loc"):
            return
        cov = msg.pose.covariance
        self._tape.add(
            {
                "t": _stamp(msg.header),
                "topic": "loc",
                "x": round(msg.pose.pose.position.x, 4),
                "y": round(msg.pose.pose.position.y, 4),
                "theta": round(_yaw(msg.pose.pose.orientation), 5),
                "confidence": round(1.0 / (1.0 + cov[0] + cov[7] + cov[35]), 3),
            }
        )


CAMERA_STREAM = "http://127.0.0.1:8080/stream"  # ustreamer on the board's host network


class RunRecorderNode(Node):
    """Records every drive where its sensors are, on the goal server's word.

    The goal server (on the board or on the laptop) publishes one command on
    ``pepin/run``; this node opens or closes the numbered tape, copies the camera stream next
    to it, and answers on the latched ``pepin/run_status`` with the run's number and path.
    """

    def __init__(self) -> None:
        super().__init__("run_recorder")
        self._record_dir = Path(str(self.declare_parameter("record_dir", "/maps/rec").value))
        self._recorder = RunRecorder(self, self._record_dir)
        self._camera: subprocess.Popen[bytes] | None = None  # curl copying the stream
        self._camera_clip: Path | None = None
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._status_pub = self.create_publisher(String, RUN_STATUS_TOPIC, latched)
        self.create_subscription(String, RUN_COMMAND_TOPIC, self._on_command, 10)
        self._say(RunStatus(IDLE))
        self.get_logger().info(f"run recorder ready: tapes in {self._record_dir}")

    def _say(self, status: RunStatus) -> None:
        self._status_pub.publish(String(data=status.to_json()))

    def _on_command(self, msg: String) -> None:
        parsed = parse_command(msg.data)
        if parsed is None:
            self.get_logger().warning(f"run command not understood: {msg.data[:80]}")
            return
        cmd, name = parsed
        if cmd == "start" and name is not None:
            if self._recorder.recording:
                self.stop()
            path = self._recorder.start(name)
            self._start_camera(path)
            self.get_logger().info(f"run {self._recorder.number}: recording {path}")
            self._say(RunStatus(RECORDING, self._recorder.number, str(path), name))
        else:
            self.stop()

    def stop(self) -> None:
        """Close the tape (flushed and synced) and the clip; harmless when no run is open."""
        was = self._recorder.recording
        self._recorder.stop()
        self._stop_camera()
        if was:
            self.get_logger().info(f"run {self._recorder.number}: closed")
        self._say(RunStatus(IDLE, self._recorder.number, None, None))

    def _start_camera(self, tape: Path) -> None:
        """Copy the camera's MJPEG stream next to the tape: no laptop in the loop.

        curl writes the stream as-is (a few percent of a core); the laptop converts after.
        """
        self._stop_camera()
        clip = camera_clip_path(tape)
        self._camera_clip = clip
        try:
            self._camera = subprocess.Popen(
                ["curl", "-s", "-m", "1800", CAMERA_STREAM, "-o", str(clip)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except OSError as exc:
            self._camera = None
            self.get_logger().warning(f"camera clip not started: {exc}")

    def _stop_camera(self) -> None:
        """End the clip; a stream that was never reachable leaves an empty file, removed here."""
        proc, self._camera = self._camera, None
        if proc is None:
            return
        proc.terminate()
        try:
            proc.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            proc.kill()
        clip, self._camera_clip = self._camera_clip, None
        if clip is not None and clip.exists() and clip.stat().st_size == 0:
            clip.unlink()


def main() -> None:
    import rclpy

    rclpy.init()
    node = RunRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
