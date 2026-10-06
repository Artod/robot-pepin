"""The board's own recording of its raw sensors: one long-lived ``ros2 bag record``, split by the
minute, under a size cap that keeps the SD card from ever filling.

Why on the board: a drive's bag is written on the laptop (``pepin_bringup.bag_recorder`` beside
Nav2), out of whatever crossed the WiFi, so a link that stalls leaves its hole in the laptop's
bag (2.6-3.8 s stalls inside drives, 2026-09-25). This one is written next to the sensors, and a
drive the laptop lost is still here.

Why one long-lived recorder and not one per drive: a new zenoh session on the board was followed
within two seconds by delivery stalls (3 of 4 stall episodes, 2026-09-25), and a recorder started
per goal cost the board's control loop six missed cycles (drive 0568, 2026-09-29). This one opens
its session when the stack starts and keeps it.

What it records is :data:`TOPICS`: the sensors as the board publishes them and the two inputs it
receives (the commands, the camera's odometry), never a picture. MCAP, uncompressed, one file a
minute (:data:`SPLIT_S`) in ``<dir>/<UTC start>/``; a respawned recorder starts a new directory.

The cap: every ``every_s`` seconds, and before the recorder starts, whole minute files are deleted
oldest first while the directory holds more than ``cap_gb`` or the card has less than ``floor_gb``
free; the file being written is never touched. A card that stays under the floor with nothing
left to delete (something else is filling it) stops the recorder until there is room again.

The laptop's ring (``pepin_bringup.bag_recorder``, flag ``goal_bag ring``) is this supervisor
with its own topics, the hidden ones, a smaller MCAP chunk and an age limit as well.

Run in the board's container (robot.launch.py, ``board_bag:=true``)::

    python3 -m pepin.board_bag --dir /maps/board_rec
"""

from __future__ import annotations

import argparse
import ctypes
import logging
import shutil
import signal
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# The board's sensors as they leave it, and the two things it is told: the hull-filtered lidar
# scan, the wheels (50 Hz) and the gyro (100 Hz), the lidar's scan-to-scan odometry, the
# zero-velocity update, the three ToF ranges and their fans, the neck's encoders and the static
# mounts; /cmd_vel and /vo as they ARRIVED, which is what a lost-command or a late-pose question
# needs (both already cross to the board for the bridge and the EKF, so recording them adds no
# WiFi traffic). Every input of the board's EKF is here, so its output is not: /odometry/filtered
# is 50 messages a second of the recorder's ~300, and a replay of the inputs through the same
# ekf.yaml gives it back (measured: 15 % of the recorder's CPU). Not /tf: the board's own dynamic
# edges are that output and /neck/state again, and subscribing would pull the laptop's map ->
# odom across the WiFi for nothing. Not the camera: the laptop films it.
TOPICS: tuple[str, ...] = (
    "/cmd_vel",
    "/imu/data_raw",
    "/neck/state",
    "/odom",
    "/odom_laser",
    "/scan",
    "/tf_static",
    "/tof/front",
    "/tof/front/scan",
    "/tof/left",
    "/tof/left/scan",
    "/tof/right",
    "/tof/right/scan",
    "/vo",
    "/vo_twist",
    "/zupt",
)
SPLIT_S = 60  # one MCAP file a minute: the unit the cap deletes, and all a crash can cost
CAP_GB = 20.0  # what the recordings may hold: ~38 h at the 0.14 MB/s these topics measure
FLOOR_GB = 10.0  # the card's free space is never taken under this by a recording
EVERY_S = 30.0  # how often the cap is enforced: a minute file is ~9 MB
STATUS_EVERY_S = 600.0  # a line in the stack's log saying what is held and what is free
# rosbag2's message cache, per buffer: it writes as soon as anything is cached, so this only
# bounds the RAM a stalled card can take (messages past it are dropped and counted by rosbag2).
CACHE_BYTES = 8 * 1024 * 1024
# How often rosbag2 looks for a requested topic that is not up yet (/tof/* with the ToF off, /vo
# without the laptop): a graph query every 100 ms by default, for as long as one is missing.
POLL_MS = 1000
STOP_TIMEOUT_S = 15.0  # SIGINT to a closed file (MCAP writes its summary on close), then TERM
QOS_OVERRIDES = Path("/params/rosbag_qos.yaml")  # /tf_static is latched: transient_local
GB = 1_000_000_000


def bag_name(now: float) -> str:
    """A recording's directory name: its UTC start, ``20261001_231500Z``."""
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime(now)) + "Z"


def record_command(
    output: Path,
    qos_overrides: Path | None = None,
    topics: Sequence[str] = TOPICS,
    *,
    hidden: bool = False,
    storage_config: Path | None = None,
) -> list[str]:
    """The ``ros2 bag record`` command line for one recording directory; ``hidden`` adds the
    hidden topics (Nav2's action status), ``storage_config`` the MCAP writer's options file."""
    command = [
        "ros2",
        "bag",
        "record",
        "--storage",
        "mcap",
        "--output",
        str(output),
        "--max-bag-duration",
        str(SPLIT_S),
        "--max-cache-size",
        str(CACHE_BYTES),
        "--polling-interval",
        str(POLL_MS),
    ]
    if hidden:
        command.append("--include-hidden-topics")
    if storage_config is not None:
        command += ["--storage-config-file", str(storage_config)]
    if qos_overrides is not None:
        command += ["--qos-profile-overrides-path", str(qos_overrides)]
    return [*command, *topics]


@dataclass(frozen=True)
class Split:
    """One minute file on the card."""

    path: Path
    size: int
    mtime: float


def splits(root: Path) -> list[Split]:
    """Every recording's minute files under ``root``, oldest first (by modification time)."""
    found = []
    for path in root.glob("*/*.mcap"):
        try:
            stat = path.stat()
        except FileNotFoundError:  # deleted between the listing and the stat
            continue
        found.append(Split(path, stat.st_size, stat.st_mtime))
    return sorted(found, key=lambda split: (split.mtime, split.path.name))


@dataclass(frozen=True)
class Pruned:
    """What one pass of the cap did and left: files and bytes deleted, bytes held, bytes free."""

    deleted: int
    deleted_bytes: int
    held_bytes: int
    free_bytes: int


def prune(
    root: Path,
    cap_bytes: int,
    floor_bytes: int,
    free_bytes: Callable[[], int],
    active: Path | None = None,
    *,
    keep_s: float | None = None,
    now: float | None = None,
) -> Pruned:
    """Delete minute files oldest first while ``root`` holds more than ``cap_bytes``, the card
    has less than ``floor_bytes`` free or (with ``keep_s``) the file was last written more than
    ``keep_s`` before ``now``; never the newest file of ``active`` (the recording being
    written), and a recording directory left with no minute file goes with its last one."""
    files = splits(root)
    held = sum(split.size for split in files)
    open_file = max(
        (split for split in files if active is not None and split.path.parent == active),
        key=lambda split: (split.mtime, split.path.name),
        default=None,
    )
    deleted, deleted_bytes = 0, 0
    free = free_bytes()
    oldest = float("-inf")
    if keep_s is not None:
        oldest = (time.time() if now is None else now) - keep_s
    for split in files:
        if held <= cap_bytes and free >= floor_bytes and split.mtime >= oldest:
            break
        if split == open_file:
            continue
        split.path.unlink(missing_ok=True)
        deleted += 1
        deleted_bytes += split.size
        held -= split.size
        free += split.size  # an estimate until the next pass asks the card again
        if split.path.parent != active and not any(split.path.parent.glob("*.mcap")):
            shutil.rmtree(split.path.parent, ignore_errors=True)
    return Pruned(deleted, deleted_bytes, held, free_bytes() if deleted else free)


class Process(Protocol):
    """The part of :class:`subprocess.Popen` the supervisor uses (a fake in the tests)."""

    returncode: int | None

    def poll(self) -> int | None: ...

    def send_signal(self, sig: int) -> None: ...

    def wait(self, timeout: float | None = None) -> int: ...

    def terminate(self) -> None: ...

    def kill(self) -> None: ...


def _die_with_parent() -> None:
    """In the recorder's child, before exec: SIGINT when the supervisor dies however it dies, so a
    respawned supervisor never finds a second recorder still writing (prctl PR_SET_PDEATHSIG)."""
    ctypes.CDLL(None).prctl(1, signal.SIGINT)


def spawn(command: Sequence[str]) -> Process:
    """Start the recorder as a child that ends with this process."""
    linux = sys.platform.startswith("linux")
    return subprocess.Popen(list(command), preexec_fn=_die_with_parent if linux else None)


class Supervisor:
    """Keeps one recorder running while there is room for it, and the cap enforced."""

    def __init__(
        self,
        root: Path,
        *,
        cap_bytes: int,
        floor_bytes: int,
        qos_overrides: Path | None,
        start: Callable[[Sequence[str]], Process] = spawn,
        free_bytes: Callable[[], int] | None = None,
        clock: Callable[[], float] = time.time,
        topics: Sequence[str] = TOPICS,
        hidden: bool = False,
        storage_config: Path | None = None,
        keep_s: float | None = None,
        label: str = "board bag",
    ) -> None:
        """``start`` launches a command, ``free_bytes`` answers the card's free space and
        ``clock`` names a new recording and dates the files (all three are the real ones on the
        board); ``topics``, ``hidden`` and ``storage_config`` go to :func:`record_command`,
        ``keep_s`` is :func:`prune`'s age limit and ``label`` begins every log line.
        ``cap_bytes``, ``floor_bytes`` and ``keep_s`` are attributes a caller may change."""
        self._root = root
        self.cap_bytes = cap_bytes
        self.floor_bytes = floor_bytes
        self.keep_s = keep_s
        self._qos = qos_overrides
        self._start = start
        self._free = free_bytes or (lambda: shutil.disk_usage(root).free)
        self._clock = clock
        self._topics = tuple(topics)
        self._hidden = hidden
        self._storage_config = storage_config
        self._label = label
        self._process: Process | None = None
        self._said_at = float("-inf")
        self.active: Path | None = None
        self.last: Pruned | None = None

    @property
    def recording(self) -> bool:
        """A recorder is running."""
        return self._process is not None and self._process.poll() is None

    def step(self) -> None:
        """One pass: enforce the cap, then start a recorder if none runs and there is room, or
        stop the one that runs when even an emptied directory leaves the card under the floor."""
        label = self._label
        if self._process is not None and self._process.poll() is not None:
            logger.warning("%s: the recorder exited with %s", label, self._process.returncode)
            self._process, self.active = None, None
        self._root.mkdir(parents=True, exist_ok=True)
        now = self._clock()
        pruned = prune(
            self._root,
            self.cap_bytes,
            self.floor_bytes,
            self._free,
            self.active,
            keep_s=self.keep_s,
            now=now,
        )
        self.last = pruned
        if pruned.deleted:
            logger.info(
                "%s: deleted %d oldest minute files (%.0f MB); %.2f GB held, %.1f GB free",
                label,
                pruned.deleted,
                pruned.deleted_bytes / 1e6,
                pruned.held_bytes / GB,
                pruned.free_bytes / GB,
            )
        if now - self._said_at >= STATUS_EVERY_S:
            self._said_at = now
            logger.info(
                "%s: %s; %.2f GB held, %.1f GB free",
                label,
                f"recording to {self.active}" if self.recording else "not recording",
                pruned.held_bytes / GB,
                pruned.free_bytes / GB,
            )
        room = pruned.free_bytes >= self.floor_bytes
        if not room and self.recording:
            logger.error(
                "%s: %.1f GB free, under the %.1f GB floor with nothing left to delete:"
                " recording stopped until there is room",
                label,
                pruned.free_bytes / GB,
                self.floor_bytes / GB,
            )
            self.stop()
        elif room and not self.recording:
            self._begin()

    def _begin(self) -> None:
        """Start a recorder on a new directory."""
        output = self._root / bag_name(self._clock())
        command = record_command(
            output,
            self._qos,
            self._topics,
            hidden=self._hidden,
            storage_config=self._storage_config,
        )
        self._process = self._start(command)
        self.active = output
        logger.info(
            "%s: recording %d topics to %s, a file every %d s, cap %.1f GB, floor %.1f GB%s",
            self._label,
            len(self._topics),
            output,
            SPLIT_S,
            self.cap_bytes / GB,
            self.floor_bytes / GB,
            "" if self.keep_s is None else f", kept {self.keep_s / 3600:.1f} h",
        )

    def stop(self) -> None:
        """SIGINT, so the open file gets its summary; TERM and KILL only past the timeout."""
        process, self._process, self.active = self._process, None, None
        if process is None or process.poll() is not None:
            return
        process.send_signal(signal.SIGINT)
        try:
            process.wait(timeout=STOP_TIMEOUT_S)
            return
        except subprocess.TimeoutExpired:
            logger.warning("%s: the recorder did not close on SIGINT; terminating", self._label)
        process.terminate()
        try:
            process.wait(timeout=3.0)
        except subprocess.TimeoutExpired:
            process.kill()


def run(supervisor: Supervisor, every_s: float, stop: threading.Event) -> None:
    """Step every ``every_s`` until ``stop``; the recorder is closed on the way out."""
    try:
        while not stop.is_set():
            supervisor.step()
            stop.wait(every_s)
    finally:
        supervisor.stop()


def main(argv: Sequence[str] | None = None) -> None:
    """Parse the arguments, stop on SIGINT/SIGTERM (the launch's stop), supervise."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dir", type=Path, default=Path("/maps/board_rec"))
    parser.add_argument("--cap-gb", type=float, default=CAP_GB)
    parser.add_argument("--floor-gb", type=float, default=FLOOR_GB)
    parser.add_argument("--every-s", type=float, default=EVERY_S)
    parser.add_argument("--qos-overrides", type=Path, default=QOS_OVERRIDES)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname).1s %(name)s: %(message)s")
    stop = threading.Event()

    def on_signal(*_: Any) -> None:
        stop.set()

    signal.signal(signal.SIGINT, on_signal)
    signal.signal(signal.SIGTERM, on_signal)
    qos = args.qos_overrides if args.qos_overrides.is_file() else None
    supervisor = Supervisor(
        args.dir,
        cap_bytes=int(args.cap_gb * GB),
        floor_bytes=int(args.floor_gb * GB),
        qos_overrides=qos,
    )
    run(supervisor, args.every_s, stop)


if __name__ == "__main__":
    main()
