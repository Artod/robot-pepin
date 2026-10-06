"""Where the head looks while the cart drives: along the path ahead, or back when it reverses.

PATH GAZE (:func:`path_aim`): the point of the plan at an arc length of ``lookahead_s`` seconds of
the current speed, clamped to ``min_m``..``max_m`` from the plan's vertex nearest the cart — the
controllers' own lookahead idea — is the pan; the tilt stays at home while that point is
``near_m`` or farther and dips below it nearer (``atan(lens height / L) - near_offset_deg``), so
the floor under the point stays in the lower third of the picture. The pan is clamped to
``pan_clamp_deg``: a goal behind starts with a turn, the clamp holds the head toward the turn and
the body brings the rest, so the head never swings backwards for nothing. :func:`settle` keeps
the current aim while the new one is within ``deadband_deg``: the head saccades and holds, it
does not creep.

FOLLOWING (:class:`PathFollower`): the held aim stays while the plan's aim is within the zone
(``deadband_deg``) of it; one that leaves the zone moves the head only after it has stayed out
``hyst_s``, ``cooldown_s`` after the previous move at the soonest, and never in the drive's
TAIL: the plan's last ``tail_m`` (:func:`remaining_m`), or its last ``tail_s`` at the current
speed (:func:`time_to_end`). The distance is what binds while parking: the controller slows
toward the goal, so the time at the current speed grows as the end nears (drive 0330: 0.16 m
left at 0.06 m/s read 2.5 s, and a 40 deg saccade went out 0.5 s before the end). Every move
costs the visual odometry samples (drive 306: a >= 45 deg swing at ~300 deg/s lost OpenVINS
4.7-6.5 of them) and writes no frame while it lasts. A path look that does not hold the head
(another look had it) is aimed at once. All five at 0 are :func:`settle` exactly.

REVERSE GAZE (:class:`ReverseWatch`, :func:`reverse_aim`): a reverse leg that has lasted
``min_s``, or any reverse with lethal cells within ``rear_m`` behind the hull
(:func:`tight_rear`), turns the head to ``pan_deg`` on the side the rear swings to (with the cart
turning left, w > 0, the rear swings right), at ``tilt_deg``, and keeps that side until the leg
ends.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from pepin.footprint import HULL, Footprint
from pepin.gaze import Aim, Reach
from pepin.marks_audit import LETHAL
from pepin.stall_look import path_ahead

Array = npt.NDArray[np.float64]


@dataclass(frozen=True)
class PathGazeLaw:
    """Path gaze's numbers (config/knobs.json's ``gaze`` block)."""

    lookahead_s: float = 2.0
    min_m: float = 0.6
    max_m: float = 1.5
    deadband_deg: float = 22.0
    pan_clamp_deg: float = 60.0
    near_m: float = 1.0
    near_offset_deg: float = 15.0
    hyst_s: float = 0.3
    cooldown_s: float = 2.0
    tail_s: float = 0.5
    tail_m: float = 0.35
    hold_s: float = 60.0


def lookahead_m(speed_m_s: float, law: PathGazeLaw) -> float:
    """How far along the plan the head looks at this speed."""
    return min(max(law.lookahead_s * abs(speed_m_s), law.min_m), law.max_m)


def path_aim(
    path_xy: Array,
    pose: tuple[float, float, float],
    speed_m_s: float,
    law: PathGazeLaw,
    *,
    lens_z_m: float,
    home: Aim,
    reach: Reach,
) -> Aim | None:
    """The aim at the plan's point ``lookahead_m`` ahead of the cart (pose: x, y, yaw in the
    plan's frame); ``None`` when the plan has nothing ahead."""
    ahead = lookahead_m(speed_m_s, law)
    poses, arc = path_ahead(path_xy, (pose[0], pose[1]), ahead, step_m=ahead / 8.0)
    if len(poses) == 0 or arc[-1] <= 0.0:
        return None
    x, y = poses[-1, 0], poses[-1, 1]
    bearing = math.atan2(y - pose[1], x - pose[0]) - pose[2]
    bearing = math.atan2(math.sin(bearing), math.cos(bearing))
    clamp = math.radians(law.pan_clamp_deg)
    pan = min(max(bearing, -clamp), clamp)
    distance = float(arc[-1])
    tilt = home.tilt_rad
    if distance < law.near_m:
        tilt = max(
            home.tilt_rad, math.atan2(lens_z_m, distance) - math.radians(law.near_offset_deg)
        )
    return reach.clamp(Aim(pan, tilt))


def settle(current: Aim | None, wanted: Aim, deadband_deg: float) -> Aim:
    """``current`` while ``wanted`` is within the dead-band of it, else ``wanted``."""
    if current is not None and current.off(wanted) <= math.radians(deadband_deg):
        return current
    return wanted


MOVING_M_S = 0.02  # slower than this the cart is taken as stopped: no end in sight


def remaining_m(path_xy: Array, cart_xy: tuple[float, float]) -> float:
    """The plan's arc from its vertex nearest the cart to its end."""
    path = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if len(path) < 2:
        return 0.0
    start = int(np.argmin(np.hypot(path[:, 0] - cart_xy[0], path[:, 1] - cart_xy[1])))
    seg = np.diff(path[start:], axis=0)
    return float(np.hypot(seg[:, 0], seg[:, 1]).sum())


def time_to_end(remaining: float, speed_m_s: float) -> float:
    """Seconds to the plan's end at this speed; infinite for a cart that stands."""
    speed = abs(speed_m_s)
    return remaining / speed if speed >= MOVING_M_S else math.inf


class PathFollower:
    """The path look's aim through one drive (see FOLLOWING above); :meth:`reset` at a drive's
    start."""

    def __init__(self) -> None:
        self.aim: Aim | None = None
        self._out_since: float | None = None
        self._moved_at: float | None = None

    def reset(self) -> None:
        """A new drive: no aim held."""
        self.aim, self._out_since, self._moved_at = None, None, None

    def update(
        self,
        wanted: Aim,
        now: float,
        law: PathGazeLaw,
        *,
        end_in_s: float = math.inf,
        left_m: float = math.inf,
        fresh: bool = False,
    ) -> Aim | None:
        """The aim to hold now for the plan's ``wanted``, ``left_m`` of plan and ``end_in_s``
        seconds at the current speed from the plan's end; ``None`` in the drive's tail when the
        path look holds nothing. ``fresh``: the path look does not hold the head, so a move
        waits for neither the hysteresis nor the cooldown (and in the tail is not made)."""
        current = self.aim
        tail = (law.tail_s > 0.0 and end_in_s <= law.tail_s) or (
            law.tail_m > 0.0 and left_m <= law.tail_m
        )
        if tail and fresh:
            return None  # a head elsewhere is not brought back in the last seconds either
        if current is not None and current.off(wanted) <= math.radians(law.deadband_deg):
            self._out_since = None
            return current
        if tail:
            return current
        if current is not None and not fresh:
            if self._out_since is None:
                self._out_since = now
            if now - self._out_since < law.hyst_s:
                return current
            if self._moved_at is not None and now - self._moved_at < law.cooldown_s:
                return current
        self.aim, self._out_since, self._moved_at = wanted, None, now
        return wanted


@dataclass(frozen=True)
class ReverseLaw:
    """Reverse gaze's numbers (config/knobs.json's ``gaze`` block)."""

    pan_deg: float = 150.0
    tilt_deg: float = 23.8
    min_s: float = 1.0
    rear_m: float = 0.30
    min_speed_m_s: float = 0.02


class ReverseWatch:
    """How long the cart has been reversing without a break, and which side its rear last swung
    to (+1 left, -1 right). Once a reverse look has taken the side (:meth:`hold`) it stays for the
    rest of the leg: a controller that swings its turn through zero while it backs (drive 306:
    w -0.5, +0.6, -0.9, +0.9 rad/s within 4 s at -0.06 m/s) would otherwise send the head 300 deg
    across the back at every swing."""

    def __init__(self) -> None:
        self._since: float | None = None
        self._held = False
        self.side = 1

    def update(self, v: float, w: float, now: float, law: ReverseLaw) -> None:
        """One commanded twist."""
        if v < -law.min_speed_m_s:
            if self._since is None:
                self._since = now
            if self._held:
                return
            if w > 0.0:
                self.side = -1
            elif w < 0.0:
                self.side = 1
        else:
            self._since = None
            self._held = False

    def hold(self) -> int:
        """The side for this leg's reverse look, kept from now until the leg ends."""
        self._held = True
        return self.side

    def reversing_for(self, now: float) -> float:
        """Seconds of this reverse leg so far (0 when not reversing)."""
        return 0.0 if self._since is None else now - self._since

    @property
    def reversing(self) -> bool:
        """Whether the last twist reversed."""
        return self._since is not None


def reverse_aim(side: int, law: ReverseLaw, reach: Reach) -> Aim:
    """The head toward the rear on ``side`` (+1 left), inside the reach."""
    return reach.clamp(Aim(side * math.radians(law.pan_deg), math.radians(law.tilt_deg)))


def tight_rear(
    grid: npt.NDArray[np.integer],
    origin: tuple[float, float],
    resolution: float,
    pose: tuple[float, float, float],
    rear_m: float,
    hull: Footprint = HULL,
) -> bool:
    """Whether a lethal cell stands within ``rear_m`` behind the hull (and as wide as it)."""
    values = np.asarray(grid)
    rows, cols = np.nonzero(values >= LETHAL)
    if len(rows) == 0:
        return False
    dx = origin[0] + (cols + 0.5) * resolution - pose[0]
    dy = origin[1] + (rows + 0.5) * resolution - pose[1]
    cos, sin = math.cos(pose[2]), math.sin(pose[2])
    along, across = cos * dx + sin * dy, -sin * dx + cos * dy
    behind = (along <= -hull.rear_m) & (along >= -hull.rear_m - rear_m)
    return bool(np.any(behind & (np.abs(across) <= hull.half_width_m)))
