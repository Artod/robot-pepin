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

THE RING (flag ``goal_bag ring``). Since the recorder moved beside Nav2 on the Mac, a recorder
that runs all day costs the laptop, not the board: this node then keeps ONE ``ros2 bag record``
of :data:`RING_TOPICS` running for as long as it lives (:class:`pepin.board_bag.Supervisor`, a
file a minute under ``ring_dir``, pruned by the knobs ``ring_keep_h``, ``ring_keep_gb`` and
``ring_floor_gb``), and a goal's word only marks the run's start and end. Once the ring is
written past ``end + tail_s`` a thread cuts ``[goal - preroll_s, end + tail_s]`` out of it
(:mod:`pepin.bag_slice`) into the run's directory, the same ``/maps/rec/NNNN_<utc>Z_<goal>``
with the same ``<stem>_0.mcap`` and ``metadata.yaml``, written whole and renamed into place.
So the bag holds the robot at rest before the command (a cold OpenVINS needs a second of it)
and whatever pushed it there. The status says ``idle`` at once; the bag follows ``tail_s`` and
a chunk (~1 s) later, which ``ros/goto.sh`` waits for (``pepin_bag_to_tape``).

The camera clip is kept: the same copy of the MJPEG stream lands beside the bag under the run's
stem (``pepin_bringup.camera_clip``), so a bag run and a tape run leave the same video.
"""

from __future__ import annotations

import importlib.util
import queue
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import String
from tf2_msgs.msg import TFMessage

from pepin.board_bag import GB, Supervisor
from pepin.flags import Flag, FlagSet, load_knobs, with_knobs
from pepin.ring import LatchedStore, SliceJob, Window, covering, segments, written_past
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
from pepin_bringup.camera_clip import CameraClip
from pepin_bringup.node_kit import Switches, spin_main

# What a run is made of, on the wire: every topic the JSONL recorder subscribes to, plus the two
# the ``loc`` records are composed from where no tracker publishes a pose (/tf, /tf_static). The
# list is TOPIC_RECORDS' own, so a topic added to the tape is recorded here without a second edit.
# One topic the tape has no row for: the lidar as it left the driver, before the hull filter cut
# the cart AND its 8 cm contact band out of it (robot.launch.py). The band hides the cart's own
# posts and cables (up to 6.5 cm past the hull, 2026-09-08) together with whatever it stands
# against, and only this topic, recorded while the cart moves, separates the two: a return that
# travels with the cart is the cart. bag_to_tape skips it; it is read from the bag directly.
RAW_SCAN_TOPIC = "/ldlidar_node/scan"
# ...and what the CAMERA told the costmaps: the fused volume's 720-bearing marking fan and its
# clearing fan (pepin_bringup.depth_fusion). The tape keeps only the merged costmap, so a lethal
# cell with no lidar return beside it could not be pinned on a camera voxel after the fact (the
# "hill" on the carpet at the base, tape 0458). bag_to_tape skips them too.
CAMERA_MARK_TOPICS = ("/depth_marks", "/depth_free")
# ...and every odometry the board's EKF fuses (ros/params/ekf.yaml) that the tape has no row for.
# The tape keeps the wheels' own /odom (odom0, published raw by the base bridge), the lidar's
# /odom_laser (odom3), the gyro (imu0) and the filter's output — but not the camera's visual
# odometry (/vo, odom1: pepin_bringup.visual_odometry on the laptop, the gated rgbd_odometry) nor
# the zero-velocity update (/zupt, odom2: the base bridge's at rest, the tracker's on a slip), so a
# fused pose that went wrong could not be taken apart into what each source said while it happened.
# bag_to_tape skips them.
ODOMETRY_TOPICS = ("/vo", "/vo_twist", "/zupt")  # /vo_twist: the VIO as twist0 (vo_output twist)
# ...and the head: its IMU (the bridge's /head/imu, 200 Hz) and the mast's sway (/mast/state), so a
# drive's bag carries what the VIO and the sway number (vio.md section 6) are computed from; the
# camera itself is the board-side clip (ros/tools/clip_to_bag.py). Neither exists without
# head_imu:=true, and a topic nobody publishes costs the recorder nothing. bag_to_tape skips them.
HEAD_TOPICS = ("/head/imu", "/mast/state")
BAG_TOPICS: tuple[str, ...] = (
    *sorted(TOPIC_RECORDS),
    "/tf",
    "/tf_static",
    RAW_SCAN_TOPIC,
    *CAMERA_MARK_TOPICS,
    *ODOMETRY_TOPICS,
    *HEAD_TOPICS,
)
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

# THE RING (flag goal_bag ring): what it records is a goal's bag and OpenVINS's two outputs, the
# per-image pose the relay reads and the IMU-rate prediction, so a replay can be held against the
# VIO as it ran. odomimu is published only while somebody subscribes (OpenVINS's subscriber
# check), so the ring is what makes OpenVINS compute it: ~170 Odometry messages a second,
# ~7 MB a minute of the ring's ~26 (pepin.ring; scratch/ring_recorder/topic_sizes.py). With them
# its health per frame (openvins-reset.patch: the keeper's verdicts are read from it) and the
# keeper's velocity seed, so a reset is replayed with what decided it.
VIO_TOPICS = (
    "/ov_msckf/poseimu",
    "/ov_msckf/odomimu",
    "/ov_msckf/health",
    "/vio/seed_twist",
)
# What the controllers were steering by: the plan as the controller received it, its local plan
# and RPP's collision arc (2026-10-06: the "white arc" of drive 0337 and the reverse runs of
# 0342/0344/0347 could not be read from a recording without them).
PLAN_TOPICS = (
    "/plan",
    "/received_global_plan",
    "/local_plan",
    "/lookahead_collision_arc",
)
RING_TOPICS: tuple[str, ...] = (*BAG_TOPICS, *VIO_TOPICS, *PLAN_TOPICS)
RING_DIR = "/maps/ring"  # ros/maps/ring on the Mac: beside rec/, gitignored with it
# The MCAP writer's options for the ring: a 256 KiB chunk instead of rosbag2's 768 KiB, so the
# newest file is on disk within ~1 s of a message instead of 3-4 s, which is how long a cut waits
# after its window's end (measured: 2.7-4.0 s per 768 KiB chunk at a drive's rates).
RING_STORAGE_CONFIG = "/params/ring_mcap.yaml"
RING_EVERY_S = 10.0  # the ring's pass: the recorder kept running, the files pruned
SLICE_WAIT_S = 20.0  # past the window's end, the longest a cut waits for the ring to be written
LATCHED_TOPIC = "/tf_static"  # heard once at the ring's start: carried into every cut
# Latched, but republished while Nav2 runs (1.5 Hz in drive bags): the last grid before the window
# is in the file read before it, and is carried to the window's start like a fresh subscription's
CARRY_LAST = ("/global_costmap/costmap",)

FLAGS = FlagSet(
    Flag(
        "goal_bag",
        "record",
        choices=("record", "ring"),
        description="how a goal's bag is made. record: one `ros2 bag record` of BAG_TOPICS per"
        " goal, started on the goal's word and closed at its end (its first message 0.3-0.4 s"
        " after the word). ring: one `ros2 bag record` of RING_TOPICS (BAG_TOPICS and OpenVINS's"
        " /ov_msckf/poseimu and /ov_msckf/odomimu) runs as long as this node, a file a minute"
        " under ring_dir pruned by the knobs ring_keep_h, ring_keep_gb and ring_floor_gb, and a"
        " goal's bag is cut out of it, [goal - preroll_s, end + tail_s], into the same"
        " /maps/rec/<run>/ (pepin.ring, pepin.bag_slice), whole once the ring is written past"
        " the window (tail_s and ~1 s after the goal). A change starts or stops the ring at once;"
        " a run keeps the way it began",
        why="record until the ring is measured on the robot (a default flips on a measurement):"
        " estimated 3 % of a core for its recorder (a replayed drive bag, a throwaway container)"
        " and ~19 MB a minute of the Mac's disk, ~26 with odomimu",
        on_when="for bags that begin with the robot at rest before the goal (a cold OpenVINS"
        " replay needs init_window_time, 1 s, of it: the replays of 2026-10-05 never initialised)"
        " and that show what happened before a command (a shove, a carry); needs the mcap"
        " library in the image (ros/Dockerfile.laptop), refused without it",
        off_when="the Mac's disk is short (the ring stops itself under ring_floor_gb), the"
        " ring's recorder is seen to cost the stack, or a cut goes wrong: record is the proven"
        " per-goal path",
    ),
)


def mcap_available() -> bool:
    """Whether the ``mcap`` library a cut needs is installed (the image's pip, not ROS's)."""
    return importlib.util.find_spec("mcap") is not None


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
    """Makes one bag per drive on the goal server's word: its own ``ros2 bag record`` (flag
    ``goal_bag record``) or a window cut out of the always-on ring (``goal_bag ring``)."""

    def __init__(self) -> None:
        super().__init__("bag_recorder")
        self._record_dir = Path(str(self.declare_parameter("record_dir", "/maps/rec").value))
        overrides = Path(str(self.declare_parameter("qos_overrides", QOS_OVERRIDES).value))
        self._qos_overrides = overrides if overrides.is_file() else None
        self._ring_dir = Path(str(self.declare_parameter("ring_dir", RING_DIR).value))
        config = Path(str(self.declare_parameter("ring_storage_config", RING_STORAGE_CONFIG).value))
        self._clip = CameraClip(self.get_logger())
        self._process: subprocess.Popen[bytes] | None = None
        self._bag: Path | None = None
        self._number = 0
        self._started = 0.0
        self._ring_run = False  # the open run is the ring's: cut at its end, nothing to close
        self._goal_s = 0.0  # its goal's wall time, the window's anchor
        # After the last declare_parameter: rclpy runs the switches' callback on declarations too
        self._switches = Switches(
            self, with_knobs(FLAGS, load_knobs("bag_recorder")), on_change=self._on_switch
        )
        self._ring = Supervisor(
            self._ring_dir,
            cap_bytes=int(float(self._switches["ring_keep_gb"]) * GB),
            floor_bytes=int(float(self._switches["ring_floor_gb"]) * GB),
            keep_s=float(self._switches["ring_keep_h"]) * 3600.0,
            qos_overrides=self._qos_overrides,
            topics=RING_TOPICS,
            hidden=True,
            storage_config=config if config.is_file() else None,
            label="ring",
        )
        self._latched = LatchedStore()
        self._jobs: queue.Queue[SliceJob | None] = queue.Queue()
        self._closing = threading.Event()
        self._cutter = threading.Thread(target=self._cut_loop, name="ring cutter", daemon=True)
        self._cutter.start()
        latched = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        statics = QoSProfile(
            depth=100,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._status_pub = self.create_publisher(String, RUN_STATUS_TOPIC, latched)
        self.create_subscription(String, RUN_COMMAND_TOPIC, self._on_command, 10)
        # The statics as a fresh subscription hears them, bytes as they came (nothing parsed):
        # a cut carries them to its start (pepin.ring.LatchedStore)
        self.create_subscription(TFMessage, LATCHED_TOPIC, self._latched.add, statics, raw=True)
        self.create_timer(1.0, self._watch)
        self.create_timer(RING_EVERY_S, self._ring_step)
        self._say(RunStatus(IDLE))
        if self._switches["goal_bag"] == "ring" and not mcap_available():
            self.get_logger().error("goal_bag ring: no mcap library in this image; record instead")
            self._switches.set("goal_bag", "record")
        self._ring_step()
        self.get_logger().info(
            f"bag recorder ready: {STORAGE} bags in {self._record_dir},"
            f" {len(BAG_TOPICS)} topics, qos overrides"
            f" {self._qos_overrides or 'none'}; camera clip from {self._clip.stream};"
            f" {self._switches.state()} (ring {len(RING_TOPICS)} topics in {self._ring_dir});"
            " run recorder (jsonl) is not running"
        )

    def _say(self, status: RunStatus) -> None:
        """The recorder's state on the latched status topic. On the way out rclpy's SIGINT
        handler may already have shut the context down from its own thread: the last word then
        has nobody left to reach and is dropped rather than raised over the bag it follows."""
        try:
            self._status_pub.publish(String(data=status.to_json()))
        except Exception:
            if rclpy.ok():
                raise

    def close(self) -> None:
        """On the node's way out (:func:`pepin_bringup.node_kit.spin_main`): a run still open is
        ended exactly as a goal's stop ends it — the bag closed on SIGINT, the clip stopped; the
        ring is closed (its last file gets its summary) and the cuts still owed are made from
        what it holds."""
        self.stop()
        self._closing.set()
        self._ring.stop()
        self._jobs.put(None)
        self._cutter.join(timeout=60.0)

    @property
    def recording(self) -> bool:
        """True while a run is open: its bag being written, or its window in the ring."""
        if self._ring_run:
            return self._bag is not None
        return self._process is not None and self._process.poll() is None

    @property
    def ring(self) -> Supervisor:
        """The always-on recorder (running under ``goal_bag ring``)."""
        return self._ring

    def _on_switch(self, name: str, _old: Any, new: Any) -> None:
        """A live change: the mode starts or stops the ring, a knob reaches it at once."""
        if name == "goal_bag":
            if new == "ring" and not mcap_available():
                raise ValueError("no mcap library in this image (ros/Dockerfile.laptop)")
            if new == "ring":
                self._ring.step()
            else:
                self._ring.stop()
        elif name == "ring_keep_gb":
            self._ring.cap_bytes = int(float(new) * GB)
        elif name == "ring_floor_gb":
            self._ring.floor_bytes = int(float(new) * GB)
        elif name == "ring_keep_h":
            self._ring.keep_s = float(new) * 3600.0

    def _ring_step(self) -> None:
        """Every RING_EVERY_S: under ``ring`` the recorder kept running and the files pruned."""
        if self._switches["goal_bag"] == "ring":
            self._ring.step()

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
        if self._switches["goal_bag"] == "ring":
            self._ring_run, self._goal_s = True, time.time()
            self._bag, self._started = bag, time.monotonic()
            self._clip.start(bag)
            self.get_logger().info(
                f"run {self._number}: {bag} will be cut from the ring,"
                f" {float(self._switches['preroll_s']):.0f} s before the goal on"
            )
            self._say(RunStatus(RECORDING, self._number, str(bag), name))
            return bag
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
        """End the bag (SIGINT, so rosbag2 closes the file) and the clip, or hand a ring run's
        window to the cutter; harmless when idle."""
        process, self._process = self._process, None
        bag, self._bag = self._bag, None
        ring_run, self._ring_run = self._ring_run, False
        self._clip.stop()
        if process is not None:
            self._end(process)
            self.get_logger().info(f"run {self._number}: closed {bag}")
        if ring_run and bag is not None:
            window = Window.around(
                self._goal_s,
                time.time(),
                float(self._switches["preroll_s"]),
                float(self._switches["tail_s"]),
            )
            self._jobs.put(SliceJob(bag, window, self._number, self._goal_s))
            self.get_logger().info(
                f"run {self._number}: closed; {bag.name} is cut from the ring once it is"
                f" written past {window.end_s - self._goal_s:.1f} s after the goal"
            )
        self._say(RunStatus(IDLE, self._number, None, None))

    def _cut_loop(self) -> None:
        """The cutter's thread: one job at a time, in the order the runs ended."""
        while True:
            job = self._jobs.get()
            if job is None:
                return
            try:
                self.cut(job)
            except Exception as exc:  # a cut gone wrong must not end the thread or the node
                self.get_logger().error(f"run {job.run}: the cut of {job.out.name} failed: {exc!r}")

    def cut(self, job: SliceJob) -> None:
        """Wait until the ring is written past the window (SLICE_WAIT_S at most, or the node's
        way out), then cut the window into the run's bag and say what it holds."""
        from pepin.bag_slice import Carry
        from pepin.bag_slice import cut as cut_window

        window = job.window
        deadline = window.end_s + SLICE_WAIT_S
        while not self._closing.is_set():
            now = time.time()
            if now >= window.end_s and written_past(segments(self._ring_dir), window.end_s):
                break
            if now > deadline:
                self.get_logger().warning(
                    f"run {job.run}: the ring was not written past the window in"
                    f" {SLICE_WAIT_S:.0f} s (its recorder down?); cutting what it holds"
                )
                break
            self._closing.wait(0.5)
        found = covering(segments(self._ring_dir), window.start_s, window.end_s)
        if not found:
            self.get_logger().error(
                f"run {job.run}: the ring holds no file of the window; {job.out.name} has no bag"
            )
            return
        began = time.monotonic()
        carry = [Carry(LATCHED_TOPIC, data) for data in self._latched.messages()]
        sliced = cut_window(
            [s.path for s in found],
            window.start_ns,
            window.end_ns,
            job.out,
            carry=carry,
            carry_last=CARRY_LAST,
        )
        lead = job.goal_s - (sliced.first_ns or window.end_ns) / 1e9
        short = (window.end_ns - sliced.newest_ns) / 1e9
        self.get_logger().info(
            f"run {job.run}: cut {job.out} from the ring in {time.monotonic() - began:.2f} s:"
            f" {sliced.messages} messages, {sliced.bytes / 1e6:.1f} MB, from {lead:.1f} s before"
            f" the goal ({window.duration_s:.1f} s window), {sliced.files} files"
            f" ({sliced.open_files} open)"
            + (f"; the ring ends {short:.1f} s before the window does" if short > 0 else "")
        )

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
        """Once a second: end a run that outlived the limit, notice a recorder that died, and let
        the camera clip say it is not growing (:meth:`CameraClip.check`).

        A recorder nobody stops is a bug, not a feature (``pepin.tape.MAX_RUN_S``, the JSONL
        tape's own limit), and a bag process that exits on its own must not leave the goal server
        believing a run is being written.
        """
        if self._bag is None:
            return
        self._clip.check()
        if self._process is not None and self._process.poll() is not None:
            self.get_logger().error(
                f"`ros2 bag record` exited with {self._process.returncode} mid-run: {self._bag}"
            )
            self._process = None
            self.stop()
        elif time.monotonic() - self._started > MAX_RUN_S:
            self.get_logger().warning(f"run {self._number}: past {MAX_RUN_S:.0f} s, closing")
            self.stop()


def main() -> None:
    """The node's life through the kit's one exit path: SIGINT (the launch's stop) ends the spin,
    ``close`` ends a run still open, and the context is shut down once, whoever gets there first
    (:func:`pepin_bringup.node_kit.end_context`)."""
    spin_main(BagRecorderNode)


if __name__ == "__main__":
    main()
