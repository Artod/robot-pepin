"""Near-field safety: the ToF stop rule that trims a command right before it reaches the wheels.

The planner routes around what the map and the lidar know; this is the last
line, fed by the three ToF sensors that look where the lidar cannot (low, in
front). It is pure decision logic over ranges and a twist; the hull sweep
against the lidar lives in :mod:`pepin.footprint`.
"""

from __future__ import annotations

from dataclasses import dataclass

from pepin.kinematics import Twist
from pepin.tof import TofRanges


@dataclass(frozen=True)
class ReflexConfig:
    """Distances (m) below which forward motion is refused, and how stale data may be."""

    front_stop_m: float = 0.22
    # Side sensors sit level at 0.16 m; their 27-degree cone would only reach the floor
    # at ~0.67 m, so anything nearer than this is a real object beside the front wheels.
    side_stop_m: float = 0.30
    max_age_s: float = 0.5
    blocked_when_stale: bool = False
    # Below the sensor's own minimum range a value is a failure code, not an object.
    min_valid_m: float = 0.04
    # True only if the left/right sensors are aimed at the flanks; then a side hit also
    # forbids turning toward it. Ours look forward (config/tof.json yaw 0), so it is off.
    side_sensors_look_sideways: bool = False


@dataclass(frozen=True)
class ReflexDecision:
    """What the reflex allows: the (possibly zeroed) twist and why."""

    twist: Twist
    blocked: bool
    reason: str = ""


def apply_reflex(
    command: Twist, ranges: TofRanges, config: ReflexConfig | None = None
) -> ReflexDecision:
    """Zero the forward speed when something is closer than the stop distance ahead.

    Only forward motion is blocked: backing away from an obstacle must stay
    possible. Turning in place is always allowed.
    """
    config = config or ReflexConfig()
    if command.linear <= 0:
        return ReflexDecision(command, blocked=False)
    if ranges.age_s > config.max_age_s:
        if config.blocked_when_stale:
            return ReflexDecision(Twist(0.0, command.angular), True, "no fresh ToF data")
        return ReflexDecision(command, blocked=False)
    front = ranges.front
    if front is not None and config.min_valid_m <= front < config.front_stop_m:
        return ReflexDecision(Twist(0.0, command.angular), True, f"front {front:.2f} m")
    for name, value in (("left", ranges.left), ("right", ranges.right)):
        if value is not None and config.min_valid_m <= value < config.side_stop_m:
            return ReflexDecision(Twist(0.0, command.angular), True, f"{name} {value:.2f} m")
    return ReflexDecision(command, blocked=False)


class Reflex:
    """ToF stop rule with hysteresis and direction, for autonomous driving.

    A sensor that tripped stays tripped until its range opens by
    ``release_margin_m`` (no chattering around the threshold), and a side hit
    only forbids moving toward it: forward, and turning to that side. Backing
    away and turning away are always allowed, so the planner can steer out.
    """

    def __init__(
        self,
        config: ReflexConfig | None = None,
        release_margin_m: float = 0.08,
        release_after_none: int = 5,
    ) -> None:
        """``config`` holds the stop distances; ``release_margin_m`` the hysteresis band.

        ``release_after_none``: a tripped sensor that reports "no return" is only
        released after that many frames in a row — the VL53L1X reports no return
        both for an empty room and for a failure at very close range.
        """
        self._cfg = config or ReflexConfig()
        self._margin = release_margin_m
        self._release_after_none = release_after_none
        self._tripped: set[str] = set()
        self._none_streak: dict[str, int] = {}

    @property
    def tripped(self) -> frozenset[str]:
        """Sensors currently holding the robot back."""
        return frozenset(self._tripped)

    def step(self, command: Twist, ranges: TofRanges) -> ReflexDecision:
        """Trim ``command`` by what the sensors see; ``blocked`` says whether anything changed."""
        cfg = self._cfg
        if ranges.age_s > cfg.max_age_s:
            if cfg.blocked_when_stale and command.linear > 0.0:
                return ReflexDecision(Twist(0.0, command.angular), True, "no fresh ToF data")
            return ReflexDecision(command, blocked=False)
        limits = {"front": cfg.front_stop_m, "left": cfg.side_stop_m, "right": cfg.side_stop_m}
        values = {"front": ranges.front, "left": ranges.left, "right": ranges.right}
        for name, value in values.items():
            limit = limits[name] + (self._margin if name in self._tripped else 0.0)
            if value is None:
                self._none_streak[name] = self._none_streak.get(name, 0) + 1
                if self._none_streak[name] >= self._release_after_none:
                    self._tripped.discard(name)
                continue
            self._none_streak[name] = 0
            if cfg.min_valid_m <= value < limit:
                self._tripped.add(name)
            else:
                self._tripped.discard(name)
        linear, angular = command.linear, command.angular
        if self._tripped and linear > 0.0:
            linear = 0.0
        if cfg.side_sensors_look_sideways:
            # Only meaningful for sensors aimed at the flanks; ours all look forward.
            if "left" in self._tripped and angular > 0.0:
                angular = 0.0  # turning left sweeps the front-left corner into it
            if "right" in self._tripped and angular < 0.0:
                angular = 0.0
        trimmed = Twist(linear, angular)
        if trimmed == command:
            return ReflexDecision(command, blocked=False)
        reason = ", ".join(
            f"{n} {values[n]:.2f} m" if values[n] is not None else f"{n} no return (held)"
            for n in ("front", "left", "right")
            if n in self._tripped
        )
        return ReflexDecision(trimmed, True, reason)
