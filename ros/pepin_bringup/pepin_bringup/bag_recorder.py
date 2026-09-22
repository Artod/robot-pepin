"""The drive recorded by rosbag2 instead of by a Python node: this process only opens and closes.

``pepin_bringup.run_recorder`` subscribes to every topic of a drive and turns each message into
a JSON line on the board: rclpy deserialises 450 floats ten times a second, the TF buffer runs,
and ``json.dumps`` writes them — 34-43 % of one A53 core, paid on the machine that also runs the
controller loop. ``ros2 bag record`` copies the SAME topics without deserialising anything (a
generic subscription hands it the serialised bytes, which go to an MCAP file as they are), so
the cost is a memcpy and the card's write.

This node is the thin half of that: it hears the goal's word on ``pepin/run`` exactly as the
JSONL recorder does — the goal server on either side publishes ``{"cmd": "start", "name": ...}``
and the recorder answers on the latched ``pepin/run_status`` — and all it does is spawn and
terminate the ``ros2 bag record`` subprocess. It subscribes to no topic of the drive itself.
The bag is one directory per run in the same place and under the same stem as a tape,
``/maps/rec/NNNN_<utc>Z_<goal>``, so a run has one name on both recorders, and
``ros/tools/bag_to_tape.py`` turns it into the very JSONL tape every analysis script reads.

NO IDLE SKETCH. The JSONL recorder keeps the last 15 seconds of a few topics in memory so a tape
begins before the goal did (``pepin.tape.RunTape``) and thins them to 5 Hz while nothing drives.
Here there is nothing to thin: between runs no process is subscribed at all, and a second bag
running all day to catch the seconds before a goal would cost the board a permanent recorder and
the card a permanent write. The price is those seconds — the bag starts when ``ros2 bag record``
has discovered the publishers, about a second or two after the word — and it is the reason the
switch defaults to ``jsonl`` until the two are compared on the robot.

The camera clip is kept: the same copy of the MJPEG stream lands beside the bag under the run's
stem (``pepin_bringup.camera_clip``), so a bag run and a tape run leave the same video.
"""

from __future__ import annotations

import signal
import subprocess
import time
from pathlib import Path

from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String

from pepin.runlink import (
    IDLE,
    RECORDING,
    RUN_COMMAND_TOPIC,
    RUN_STATUS_TOPIC,
    RunStatus,
    parse_command,
)
from pepin.tape import MAX_RUN_S, next_run_number
from pepin.tape_rows import TOPIC_RECORDS
from pepin_bringup.bridge_kick import BridgeKick
from pepin_bringup.camera_clip import CameraClip

# What a run is made of, on the wire: every topic the JSONL recorder subscribes to, plus the two
# the ``loc`` records are composed from where no tracker publishes a pose (/tf, /tf_static). The
# list is TOPIC_RECORDS' own, so a topic added to the tape is recorded here without a second edit.
BAG_TOPICS: tuple[str, ...] = (*sorted(TOPIC_RECORDS), "/tf", "/tf_static")
# MCAP, no compression: the storage rosbag2 ships with in Jazzy, read by rosbag2_py on the laptop
# and by every MCAP tool. Compression would cost this board's cores exactly what we are taking
# off them.
STORAGE = "mcap"
# QoS the recorder must ask for where the publisher is latched: /global_costmap/costmap is
# published whole ONCE and then only as updates, so a volatile subscription that starts mid-drive
# hears nothing at all (the same reason the JSONL recorder subscribes to it transient-local).
QOS_OVERRIDES = "/params/rosbag_qos.yaml"
# How long `ros2 bag record` is given to close its file on SIGINT before it is killed. An MCAP
# file is written as it goes; the close writes the summary a reader seeks by.
BAG_STOP_TIMEOUT_S = 15.0


def record_command(bag: Path, qos_overrides: Path | None = None) -> list[str]:
    """The ``ros2 bag record`` command line for one run's bag; ``--include-hidden-topics``
    because Nav2's action status topics (``/navigate_to_pose/_action/status``) are hidden ones."""
    command = [
        "ros2",
        "bag",
        "record",
        "--storage",
        STORAGE,
        "--output",
        str(bag),
        "--include-hidden-topics",
    ]
    if qos_overrides is not None:
        command += ["--qos-profile-overrides-path", str(qos_overrides)]
    return [*command, *BAG_TOPICS]


class BagRecorderNode(Node):
    """Starts and stops one ``ros2 bag record`` per drive, on the goal server's word."""

    def __init__(self) -> None:
        super().__init__("bag_recorder")
        self._record_dir = Path(str(self.declare_parameter("record_dir", "/maps/rec").value))
        overrides = Path(str(self.declare_parameter("qos_overrides", QOS_OVERRIDES).value))
        self._qos_overrides = overrides if overrides.is_file() else None
        # The laptop's one way to restart this board's zenoh bridge without an ssh key: the JSONL
        # recorder hosts the same handler, and one of the two is always the process that runs on
        # the board in every mode. Always on here — this node has no flag table, and the A/B for
        # it is the recorder switch itself (PEPIN_RECORDER=jsonl brings the flagged one back).
        self._kick = BridgeKick(self, enabled=lambda: True)
        self._clip = CameraClip(self.get_logger())
        self._process: subprocess.Popen[bytes] | None = None
        self._bag: Path | None = None
        self._number = 0
        self._started = 0.0
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._status_pub = self.create_publisher(String, RUN_STATUS_TOPIC, latched)
        self.create_subscription(String, RUN_COMMAND_TOPIC, self._on_command, 10)
        self.create_timer(1.0, self._watch)
        self._say(RunStatus(IDLE))
        self.get_logger().info(
            f"bag recorder ready: {STORAGE} bags in {self._record_dir},"
            f" {len(BAG_TOPICS)} topics, qos overrides"
            f" {self._qos_overrides or 'none'}; run recorder (jsonl) is not running"
        )

    def _say(self, status: RunStatus) -> None:
        self._status_pub.publish(String(data=status.to_json()))

    @property
    def recording(self) -> bool:
        """True while a bag is being written."""
        return self._process is not None and self._process.poll() is None

    def _on_command(self, msg: String) -> None:
        parsed = parse_command(msg.data)
        if parsed is None:
            self.get_logger().warning(f"run command not understood: {msg.data[:80]}")
            return
        command, name = parsed
        if command == "start" and name is not None:
            self.start(name)
        else:
            self.stop()

    def start(self, name: str) -> Path:
        """Open a numbered bag for this run and say so on the status topic; returns its path.

        The stamp is UTC and says so with a trailing ``Z``, like a tape's: this container's clock
        is UTC while the board's shell and the laptop's files are on the flat's local time.
        """
        self.stop()
        self._number = next_run_number(self._record_dir)
        stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime()) + "Z"
        bag = self._record_dir / f"{self._number:04d}_{stamp}_{name}"
        self._record_dir.mkdir(parents=True, exist_ok=True)
        try:
            self._process = subprocess.Popen(record_command(bag, self._qos_overrides))
        except OSError as exc:  # no ros2bag in the image: say it, do not hold the drive
            self._process = None
            self.get_logger().error(f"bag recorder could not start `ros2 bag record`: {exc}")
            self._say(RunStatus(IDLE, self._number, None, None))
            return bag
        self._bag, self._started = bag, time.monotonic()
        self._clip.start(bag)
        self.get_logger().info(f"run {self._number}: recording {bag}")
        self._say(RunStatus(RECORDING, self._number, str(bag), name))
        return bag

    def stop(self) -> None:
        """End the bag (SIGINT, so rosbag2 closes the file) and the clip; harmless when idle."""
        process, self._process = self._process, None
        bag, self._bag = self._bag, None
        self._clip.stop()
        if process is not None:
            self._end(process)
            self.get_logger().info(f"run {self._number}: closed {bag}")
        self._say(RunStatus(IDLE, self._number, None, None))

    def _end(self, process: subprocess.Popen[bytes]) -> None:
        """SIGINT, then TERM, then KILL: a bag whose summary was never written still reads, but
        only sequentially, so the recorder is given the whole window to close it."""
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=BAG_STOP_TIMEOUT_S)
            return
        except subprocess.TimeoutExpired:
            self.get_logger().warning("`ros2 bag record` did not close on SIGINT; terminating")
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()

    def _watch(self) -> None:
        """Once a second: end a run that outlived the limit, and notice a recorder that died.

        A recorder nobody stops is a bug, not a feature (``pepin.tape.MAX_RUN_S``, the JSONL
        tape's own limit), and a bag process that exits on its own must not leave the goal server
        believing a run is being written.
        """
        if self._process is None:
            return
        if self._process.poll() is not None:
            self.get_logger().error(
                f"`ros2 bag record` exited with {self._process.returncode} mid-run: {self._bag}"
            )
            self._process = None
            self.stop()
        elif time.monotonic() - self._started > MAX_RUN_S:
            self.get_logger().warning(f"run {self._number}: past {MAX_RUN_S:.0f} s, closing")
            self.stop()


def main() -> None:
    import rclpy

    rclpy.init()
    node = BagRecorderNode()
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
