"""Everything static about this robot, loaded once from ``config/``: :class:`RobotConfig`.

``config/robot.json`` says which feeds are enabled and which ports the board serves them on;
base.json, lidar.json and tof.json say what the base is and where the sensors sit. The laptop's
dashboard reads it. The pre-ROS drive loop that owned the feeds and the wheel link (``Robot``) is
in git history (its callers went on 2026-09-27).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path

from pepin.footprint import Footprint
from pepin.geometry import BaseConfig
from pepin.lidar import LidarMount
from pepin.tof import TofMount, load_mounts

logger = logging.getLogger(__name__)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "config"
"""The repo's config directory, found from this file so scripts work from any cwd."""

KNOWN_FEEDS = ("lidar", "tof", "camera")


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
    footprint: Footprint = field(default_factory=Footprint)

    @classmethod
    def load(cls, config_dir: Path = CONFIG_DIR) -> RobotConfig:
        """Read robot.json, base.json, lidar.json and tof.json from ``config_dir``.

        A feed missing from robot.json is on (the safe default: sensors run
        unless switched off); a misspelt feed name is an error, not a silently
        ignored sensor.
        """
        robot = json.loads((config_dir / "robot.json").read_text())
        base_raw = json.loads((config_dir / "base.json").read_text())
        entries = robot.get("feeds", {})
        unknown = sorted(set(entries) - set(KNOWN_FEEDS))
        if unknown:
            raise ValueError(f"robot.json: unknown feeds {unknown}; known: {list(KNOWN_FEEDS)}")
        feeds = {name: FeedConfig(**entries.get(name, {})) for name in KNOWN_FEEDS}
        return cls(
            base=BaseConfig.from_dict(base_raw),
            lidar_mount=LidarMount.from_json(str(config_dir / "lidar.json")),
            tof_mounts=load_mounts(config_dir / "tof.json"),
            feeds=feeds,
            ports={str(k): int(v) for k, v in robot.get("ports", {}).items()},
            footprint=(
                Footprint.from_config(base_raw["footprint"])
                if "footprint" in base_raw
                else Footprint()
            ),
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
