"""The robot as one object on the laptop: configuration, feeds and the wheel link, owned together.

Every entry point used to assemble the robot by hand — resolve the board,
load three JSON files, start the lidar reader, the ToF reader, the base link,
the camera — and each copy drifted. :meth:`Robot.connect` does it once from
``config/robot.json`` (which feeds are enabled, which ports), and the control
loop asks :meth:`Robot.observe` for one :class:`Observation` per tick: the
board's odometry state plus a :class:`pepin.navigator.Sense` built from every
feed's newest reading. Nothing in here waits on the network. A feed switched
off in the config simply never appears; a required feed that goes quiet shows
up as a large age and the navigator holds.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from types import TracebackType

from pepin.base_link import BASE_PORT, BaseClient, BaseState
from pepin.feeds import Feed
from pepin.geometry import BaseConfig
from pepin.kinematics import Twist
from pepin.lidar import LaserScan, LidarClient, LidarMount
from pepin.navigator import Sense
from pepin.tof import TOF_PORT, TofClient, TofMount, load_mounts
from pepin.transport import LIDAR_PORT, board_address
from pepin.video import CameraRecorder

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
"""The repo's config directory, found from this file so scripts work from any cwd."""

BASE_TIMEOUT_S = 1.0  # no word from the base server for this long: the board has stopped anyway


@dataclass(frozen=True)
class FeedConfig:
    """One entry of ``config/robot.json`` "feeds": run it or not, and whether driving needs it."""

    enabled: bool = True
    required: bool = True


@dataclass(frozen=True)
class RobotConfig:
    """Everything static about this robot, loaded from the ``config/`` directory."""

    base: BaseConfig
    lidar_mount: LidarMount
    tof_mounts: Mapping[str, TofMount]
    feeds: Mapping[str, FeedConfig]
    ports: Mapping[str, int]

    @classmethod
    def load(cls, config_dir: Path = CONFIG_DIR) -> RobotConfig:
        """Read robot.json, base.json, lidar.json and tof.json from ``config_dir``."""
        robot = json.loads((config_dir / "robot.json").read_text())
        feeds = {name: FeedConfig(**entry) for name, entry in robot.get("feeds", {}).items()}
        return cls(
            base=BaseConfig.from_json(str(config_dir / "base.json")),
            lidar_mount=LidarMount.from_json(str(config_dir / "lidar.json")),
            tof_mounts=load_mounts(config_dir / "tof.json"),
            feeds=feeds,
            ports={str(k): int(v) for k, v in robot.get("ports", {}).items()},
        )

    def enabled(self, feed: str) -> bool:
        """Whether ``feed`` ("lidar", "tof", "camera") is switched on."""
        entry = self.feeds.get(feed)
        return entry.enabled if entry is not None else False

    def without(self, *feeds: str) -> RobotConfig:
        """A copy with the named feeds switched off (a CLI ``--no-tof``, a broken sensor)."""
        off = {name: FeedConfig(enabled=False, required=False) for name in feeds}
        return replace(self, feeds={**self.feeds, **off})

    def port(self, name: str, default: int) -> int:
        """TCP port of a board service by name, falling back to the package default."""
        return self.ports.get(name, default)


@dataclass(frozen=True)
class Observation:
    """One tick of the world as the laptop knows it: the board's state and the navigator's input."""

    state: BaseState
    sense: Sense
    scans: Sequence[LaserScan]  # the revolutions that arrived since the previous tick


class Robot:
    """Feeds and the wheel link behind one object; use as a context manager."""

    def __init__(
        self,
        config: RobotConfig,
        host: str,
        link: BaseClient,
        lidar: LidarClient | None = None,
        tof: TofClient | None = None,
        camera: CameraRecorder | None = None,
    ) -> None:
        """Wrap already-built parts; :meth:`connect` is the factory that builds and starts them."""
        self.config = config
        self.host = host
        self.link = link
        self.lidar = lidar
        self.tof = tof
        self.camera = camera

    @classmethod
    def connect(
        cls, config: RobotConfig, *, host: str | None = None, video_name: str | None = None
    ) -> Robot:
        """Resolve the board; start the base link, every enabled feed and, if asked, the camera."""
        host = host or board_address()
        link = BaseClient(host, config.port("base", BASE_PORT)).start()
        lidar = (
            LidarClient(host, config.lidar_mount, port=config.port("lidar", LIDAR_PORT)).start()
            if config.enabled("lidar")
            else None
        )
        tof = (
            TofClient(host, config.port("tof", TOF_PORT)).start() if config.enabled("tof") else None
        )
        camera = CameraRecorder(host, video_name) if video_name else None
        if camera is not None:
            camera.start()
        logger.info(
            "robot at %s: lidar %s, tof %s, camera %s",
            host,
            "on" if lidar else "off",
            "on" if tof else "off",
            "recording" if camera else "off",
        )
        return cls(config, host, link, lidar, tof, camera)

    @property
    def mount(self) -> LidarMount:
        """Where the lidar sits on the robot (for turning scans into robot-frame points)."""
        return self.config.lidar_mount

    @property
    def feeds(self) -> dict[str, Feed]:
        """The running feeds by name, for health displays and shutdown."""
        running: dict[str, Feed] = {"base": self.link}
        if self.lidar is not None:
            running["lidar"] = self.lidar
        if self.tof is not None:
            running["tof"] = self.tof
        return running

    def wait_ready(self, timeout_s: float = 5.0) -> BaseState:
        """First word from the base server, or ``RuntimeError`` with the fix if it never comes."""
        state = self.link.wait_for_state(timeout_s)
        if state is None:
            raise RuntimeError(
                "no telemetry from the base server — on the board: systemctl status pepin-base"
            )
        return state

    def observe(self, now: float) -> Observation | None:
        """Everything new since the last tick, or None when the board has gone quiet."""
        state = self.link.state(now)
        if state is None or state.age_s > BASE_TIMEOUT_S:
            return None
        scans = self.lidar.drain() if self.lidar is not None else []
        sense = Sense(
            now=now,
            odom_pose=state.pose,
            scans=[scan.points_xy(self.mount) for scan in scans],
            scan_age_s=self.lidar.age_s(now) if self.lidar is not None else float("inf"),
            tof=self.tof.ranges(now) if self.tof is not None else None,
        )
        return Observation(state, sense, scans)

    def drive(self, twist: Twist) -> None:
        """Send the body velocity to the board (also its deadman heartbeat)."""
        self.link.set_twist(twist)

    def stop(self) -> None:
        """Stop the wheels now."""
        self.link.stop()

    def close(self) -> None:
        """Stop the wheels, then release every feed and the camera; safe to call twice."""
        self.link.stop()
        for feed in self.feeds.values():
            feed.close()
        if self.camera is not None:
            self.camera.stop()

    def __enter__(self) -> Robot:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()
