"""How the SO-101 arm may move: config/arm.json's ``motion`` block, read by the board's arm
driver (which enforces it) and by the laptop's planner (which plans within it). Standard
library only.

Angles are degrees in config/arm.json's convention, lerobot's: the middle of each calibrated
range is 0, ``angle = (ticks - (min + max) / 2) * 360 / 4096``; the gripper is linear from its
closed end, ``(ticks - min) * 360 / 4096 - 10``. The calibration lives in the servos' own EEPROM
(Min_Position_Limit and Max_Position_Limit = lerobot's ``range_min``/``range_max``, read
2026-10-11), so the board converts with what the servos say and needs no calibration file;
``limits_deg`` here may only narrow those ranges.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pepin.expression.trajectory import JOINTS

TICKS_PER_TURN = 4096
DEG_PER_TICK = 360.0 / TICKS_PER_TURN
GRIPPER = "gripper"
GRIPPER_ZERO_DEG = -10.0  # the gripper's closed end, lerobot's 0 of its 0..100 joint
# What the live and the file's numbers may be (the servos' own reach; a torque limit share)
RANGES = {
    "max_speed_deg_s": (5.0, 300.0),
    "max_acc_deg_s2": (10.0, 2232.0),
    "torque_limit": (0.05, 1.0),
    "max_duration_s": (1.0, 60.0),
    "hold_max_s": (0.0, 10.0),
}


@dataclass(frozen=True)
class ArmMotion:
    """config/arm.json's ``motion``: the bus ids, the rest pose, the soft limits, the caps, the
    torque limit while energised, and how the arm rides the wheels' encoder read."""

    ids: Mapping[str, int]
    rest_deg: Mapping[str, float]
    limits_deg: Mapping[str, tuple[float, float]]
    max_speed_deg_s: float = 90.0
    max_acc_deg_s2: float = 360.0
    max_duration_s: float = 20.0
    hold_max_s: float = 3.0
    release_after_s: float = 1.0
    torque_limit: float = 0.5
    read_window_ms: float = 3.0
    silent_ticks: int = 3
    retry_s: float = 5.0
    idle_read_hz: float = 5.0

    def rest(self) -> tuple[float, ...]:
        """The rest pose, one angle per joint in :data:`JOINTS` order."""
        return tuple(self.rest_deg[j] for j in JOINTS)

    def limits(self) -> tuple[tuple[float, float], ...]:
        """The soft limits, one ``(low, high)`` per joint in :data:`JOINTS` order."""
        return tuple(self.limits_deg[j] for j in JOINTS)


def arm_motion(data: Mapping[str, Any]) -> ArmMotion:
    """config/arm.json's ``motion`` block from the file's object; ``ValueError`` naming what
    is missing or out of range, and a rest pose outside the limits."""
    block = data.get("motion")
    if not isinstance(block, dict):
        raise ValueError("config/arm.json has no motion block")
    ids = {str(k): int(v) for k, v in block["ids"].items()}
    rest = {str(k): float(v) for k, v in block["rest_deg"].items()}
    limits = {str(k): (float(v[0]), float(v[1])) for k, v in block["limits_deg"].items()}
    for name, table in (("ids", ids), ("rest_deg", rest), ("limits_deg", limits)):
        if set(table) != set(JOINTS):
            raise ValueError(f"motion.{name}: one entry for each of {', '.join(JOINTS)}")
    for joint in JOINTS:
        low, high = limits[joint]
        if not low <= rest[joint] <= high:
            raise ValueError(f"motion.rest_deg.{joint} {rest[joint]} is outside {low}..{high}")
    for key, (low, high) in RANGES.items():
        if key in block and not low <= float(block[key]) <= high:
            raise ValueError(f"motion.{key} {block[key]} is outside {low}..{high}")
    d = ArmMotion(ids, rest, limits)
    return ArmMotion(
        ids,
        rest,
        limits,
        max_speed_deg_s=float(block.get("max_speed_deg_s", d.max_speed_deg_s)),
        max_acc_deg_s2=float(block.get("max_acc_deg_s2", d.max_acc_deg_s2)),
        max_duration_s=float(block.get("max_duration_s", d.max_duration_s)),
        hold_max_s=float(block.get("hold_max_s", d.hold_max_s)),
        release_after_s=float(block.get("release_after_s", d.release_after_s)),
        torque_limit=float(block.get("torque_limit", d.torque_limit)),
        read_window_ms=float(block.get("read_window_ms", d.read_window_ms)),
        silent_ticks=int(block.get("silent_ticks", d.silent_ticks)),
        retry_s=float(block.get("retry_s", d.retry_s)),
        idle_read_hz=float(block.get("idle_read_hz", d.idle_read_hz)),
    )


def load_arm_motion(path: str | Path | None = None) -> ArmMotion:
    """config/arm.json's motion block (wherever this library runs, or ``path``)."""
    if path is None:
        from pepin.deployment import config_file

        path = config_file("arm.json")
    return arm_motion(json.loads(Path(path).read_text()))


@dataclass(frozen=True)
class ServoRange:
    """One joint's calibrated encoder range, as its EEPROM holds it."""

    min_ticks: int
    max_ticks: int
    gripper: bool = False

    def degrees(self, ticks: int) -> float:
        """Encoder ticks as the joint's angle."""
        if self.gripper:
            return (ticks - self.min_ticks) * DEG_PER_TICK + GRIPPER_ZERO_DEG
        return (ticks - (self.min_ticks + self.max_ticks) / 2.0) * DEG_PER_TICK

    def ticks(self, degrees: float) -> int:
        """The joint's angle as encoder ticks (the nearest)."""
        if self.gripper:
            return round(self.min_ticks + (degrees - GRIPPER_ZERO_DEG) / DEG_PER_TICK)
        return round((self.min_ticks + self.max_ticks) / 2.0 + degrees / DEG_PER_TICK)

    def span_deg(self) -> tuple[float, float]:
        """The calibrated range in degrees."""
        return self.degrees(self.min_ticks), self.degrees(self.max_ticks)
