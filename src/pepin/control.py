"""Path following for the differential-drive base: waypoints in, twist out.

A carrot-on-a-stick controller: aim at the first waypoint further than the
look-ahead distance, turn toward it proportionally, drive forward only when
roughly facing it, and stop inside the goal tolerance. Simple, predictable,
and easy to tune by watching the robot.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

from pepin.kinematics import Twist
from pepin.odometry import Pose2D, wrap_angle


@dataclass(frozen=True)
class ControllerConfig:
    """Speeds, gains and tolerances in SI units."""

    cruise_speed_m_s: float = 0.15
    max_yaw_rate_rad_s: float = 0.6  # a 0.2 s bus stall at 0.8 rad/s overshot turns by 10 deg
    yaw_gain: float = 1.5  # rad/s per rad of heading error
    # Paths come as straight legs (line-of-sight shortcut), so a 30 cm carrot no longer
    # cuts corners into obstacles; a shorter one made the robot weave about the line.
    lookahead_m: float = 0.30
    goal_tolerance_m: float = 0.12
    face_before_driving_rad: float = math.radians(35.0)
    # Once turning in place, keep turning until this close: flipping between "turn" and
    # "drive" right at the threshold made the robot weave after every turn.
    resume_driving_rad: float = math.radians(15.0)  # > max_yaw_rate x ~0.25 s telemetry lag


@dataclass(frozen=True)
class ControlOutput:
    twist: Twist
    target: tuple[float, float] | None  # the waypoint being chased, None when done
    done: bool


class PathFollower:
    """Drives along a polyline of world waypoints using the current pose estimate."""

    def __init__(
        self, path: Sequence[tuple[float, float]], config: ControllerConfig | None = None
    ) -> None:
        if not path:
            raise ValueError("path must have at least one waypoint")
        self._path = list(path)
        self._cfg = config or ControllerConfig()
        self._index = 0
        self._facing = False  # turning in place toward the next waypoint

    @property
    def goal(self) -> tuple[float, float]:
        """The last waypoint."""
        return self._path[-1]

    @property
    def facing(self) -> bool:
        """True while turning in place; a replacement follower inherits it across a replan."""
        return self._facing

    @facing.setter
    def facing(self, value: bool) -> None:
        self._facing = value

    def _advance(self, pose: Pose2D) -> tuple[float, float]:
        """Skip waypoints already within the look-ahead, never past the final one."""
        while self._index < len(self._path) - 1:
            wx, wy = self._path[self._index]
            if math.hypot(wx - pose.x, wy - pose.y) > self._cfg.lookahead_m:
                break
            self._index += 1
        return self._path[self._index]

    def step(self, pose: Pose2D) -> ControlOutput:
        """Twist for this control tick given the pose estimate in the path's frame."""
        cfg = self._cfg
        gx, gy = self.goal
        if math.hypot(gx - pose.x, gy - pose.y) <= cfg.goal_tolerance_m:
            return ControlOutput(Twist(0.0, 0.0), None, done=True)
        tx, ty = self._advance(pose)
        heading_error = wrap_angle(math.atan2(ty - pose.y, tx - pose.x) - pose.theta)
        yaw = max(
            -cfg.max_yaw_rate_rad_s, min(cfg.max_yaw_rate_rad_s, cfg.yaw_gain * heading_error)
        )
        if self._facing:
            self._facing = abs(heading_error) > cfg.resume_driving_rad
        elif abs(heading_error) > cfg.face_before_driving_rad:
            self._facing = True
        if self._facing:
            return ControlOutput(Twist(0.0, yaw), (tx, ty), done=False)
        # Slow down as the heading error grows and as the goal approaches.
        speed = cfg.cruise_speed_m_s * (
            1.0 - abs(heading_error) / cfg.face_before_driving_rad * 0.5
        )
        remaining = math.hypot(gx - pose.x, gy - pose.y)
        speed = min(speed, max(0.05, remaining))
        return ControlOutput(Twist(speed, yaw), (tx, ty), done=False)
