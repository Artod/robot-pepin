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

ECONOMY (:class:`StandWatch` feeds the follower): while the cart STANDS (``|v|`` under
``still_m_s``, ``|w|`` under :data:`STILL_RAD_S`) the path look makes no saccade unless the plan's
aim is more than ``still_deg`` off where the head is; one that does not hold the head then takes
it where it is (no write), and the first saccade waits for the first command that moves the cart
(drives 0348-0371: 59 fresh path looks made standing, 21 of them in 0351's recovery loop). For
``stall_guard_s`` after the controller's command dropped to zero the path look makes no saccade at
all: a stall look is coming and takes the head from where it is (29 path looks held still under
0.5 s, 18 frames in all). RE-CENTRE: when the plan's aim has been within ``recentre_deg`` of
straight ahead for ``recentre_s`` while the held aim is more than half the zone from it (the
cart's own turn carried the path ahead, the head still looks aside: drive 0365, 24 deg right for
2.4 s along a straight path), the head moves once to the plan's aim, the cooldown permitting.
``still_m_s``, ``stall_guard_s`` and ``recentre_s`` at 0 are the follower of 2026-10-06.

STRAIGHT AHEAD BY DEFAULT (:func:`path_look`, ``bend_deg``): the wanted aim is straight ahead
(the pan 0 at the plan point's tilt) unless the plan point is more than ``bend_deg`` off the nose:
a BEND look at it, which ends once the point is back within :data:`BEND_EXIT` of the threshold.
A held aim off the nose by more than :data:`AHEAD_TOL_DEG` is out of the zone of a wanted
straight-ahead one, so the head comes back as soon as the bend is passed (the hysteresis and the
cooldown permitting) — the re-centre's job, without its window. Drives 0330-0379 (eleven tapes):
on straight legs the plan point sits a median 5 deg off the nose (p90 13), in bends over 30 deg a
median 37; the head looked aside 54 % of straight driving with the zone alone. ``bend_lead_s``
reads the bend further along the plan; at the cart's <= 0.3 m/s the lookahead is its 0.6 m floor,
and a bend look settles a median 0.63 s before the cart has turned 15 deg (3 or 4 s of lead: the
same, bends are taken at pivot speeds). PARKING (``park_ahead_m``): once the goal server has
handed the drive to the parker, or the plan left is that short, the look is read off the parking
SPOT, :func:`park_spot` of the plan at that moment (``park_beyond_m`` past its end along its last
step: what the nose parks against), at the driving tilt, the same magnet — and the drive's tail
does not hold it (drive 0376: the tail froze the head 53 deg right for the last 3.4 s while the
cart pivoted to face the printer). ``bend_deg`` and ``park_ahead_m`` at 0 are the head of
2026-10-06.

REVERSE GAZE (:class:`ReverseWatch`, :func:`reverse_aim`): a reverse leg that was ANNOUNCED —
the plan leaves the cart backwards for ``min_m`` or more (:func:`reverse_leg_m`), or the tree's
recovery drives it (``recovery``) — turns the head back at its first reversing command; one that
was not, once it has lasted ``min_s``; any reverse with lethal cells within ``rear_m`` behind the
hull (:func:`tight_rear`) from its first. The head goes to ``pan_deg`` on the side the rear swings
to (with the cart turning left, w > 0, the rear swings right), at ``tilt_deg``, and keeps that
side until the leg ends — with ``hold_s`` above 0, through a stand after it of up to ``hold_s``
(a reverse that follows within it is the same leg), and the look is let go at once at the first
forward command or the stand's end, even mid-glance. Drives 0348-0363 before it: requested
1.17 s into the leg by ``min_s`` alone, the head settled after 1-2 s legs had ended (9 of 12) and
still looked back 0.54 s into the forward drive. ``min_m``, ``recovery`` and ``hold_s`` at 0 are
the reverse gaze of 2026-10-05.
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
    still_m_s: float = 0.02
    still_deg: float = 60.0
    stall_guard_s: float = 0.5
    recentre_s: float = 1.0
    recentre_deg: float = 5.0
    bend_deg: float = 30.0
    bend_lead_s: float = 0.0
    park_ahead_m: float = 1.0
    park_beyond_m: float = 0.5


def lookahead_m(speed_m_s: float, law: PathGazeLaw, seconds: float = 0.0) -> float:
    """How far along the plan the head looks at this speed (``seconds`` of it instead of the
    law's ``lookahead_s`` when above 0)."""
    span = seconds if seconds > 0.0 else law.lookahead_s
    return min(max(span * abs(speed_m_s), law.min_m), law.max_m)


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
    bearing = _bearing(pose, float(poses[-1, 0]), float(poses[-1, 1]))
    clamp = math.radians(law.pan_clamp_deg)
    pan = min(max(bearing, -clamp), clamp)
    return reach.clamp(Aim(pan, _tilt(float(arc[-1]), law, lens_z_m, home)))


def _bearing(pose: tuple[float, float, float], x: float, y: float) -> float:
    """The bearing of (x, y) off the cart's nose, radians, left positive."""
    bearing = math.atan2(y - pose[1], x - pose[0]) - pose[2]
    return math.atan2(math.sin(bearing), math.cos(bearing))


def _tilt(distance: float, law: PathGazeLaw, lens_z_m: float, home: Aim) -> float:
    """Home's tilt for a point ``near_m`` or farther, the near dip for a nearer one."""
    if distance >= law.near_m:
        return home.tilt_rad
    dip = math.atan2(lens_z_m, distance) - math.radians(law.near_offset_deg)
    return max(home.tilt_rad, dip)


AHEAD, BEND, PARK, POINT = "ahead", "bend", "park", "point"
BEND_EXIT = 0.6  # a bend or parking look ends once its bearing is back within this share of
# path_bend_deg (it starts beyond the whole of it): an aim hovering at the threshold does not
# send the head out and back
AHEAD_TOL_DEG = 2.0  # a held pan this close to the nose is straight ahead (the encoders' settle
# tolerance is 1 deg)


@dataclass(frozen=True)
class PathLook:
    """Path gaze's wanted aim and its kind: :data:`AHEAD` (the plan does not bend off the nose
    within the lookahead), :data:`BEND` (it does), :data:`PARK` (the parking spot) or
    :data:`POINT` (the plan point itself: the ahead magnet off); ``spot``: read off the parking
    spot, a fixed point past the plan's end, which the drive's tail does not make unstable."""

    aim: Aim
    kind: str
    spot: bool = False


SPOT_CHORD_M = 0.05  # the plan's approach to its end is read over at least this much of its
# arc: its last step (0.07-0.09 m; the drives' final headings within 4-11 deg of it at the
# hand-over), not a millimetre step; a chord of 0.3 m bends with the approach's curve (drives
# 0372/0376: 36-38 deg off the final heading)


def park_spot(path_xy: Array, beyond_m: float) -> tuple[float, float] | None:
    """The point ``beyond_m`` past the plan's end along its approach (the chord over its last
    :data:`SPOT_CHORD_M`): what a cart parked nose-to-furniture stands against; ``None`` for a
    plan without a step."""
    path = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if len(path) < 2:
        return None
    back = np.hypot(*np.diff(path[::-1], axis=0).T).cumsum()
    start = len(path) - 2 - int(np.searchsorted(back, SPOT_CHORD_M))
    dx, dy = path[-1] - path[max(start, 0)]
    length = math.hypot(dx, dy)
    if length <= 1e-6:
        return None
    return (
        float(path[-1, 0] + dx / length * beyond_m),
        float(path[-1, 1] + dy / length * beyond_m),
    )


def path_look(
    path_xy: Array,
    pose: tuple[float, float, float],
    speed_m_s: float,
    law: PathGazeLaw,
    *,
    lens_z_m: float,
    home: Aim,
    reach: Reach,
    spot: tuple[float, float] | None = None,
    bending: bool = False,
) -> PathLook | None:
    """Path gaze's wanted aim (pose: x, y, yaw in the plan's frame); ``None`` when the plan has
    nothing ahead and there is no ``spot``. The tilt is :func:`path_aim`'s, from the plan point
    ``lookahead_m`` ahead. The pan: with ``bend_deg`` above 0 (the AHEAD MAGNET) straight ahead
    while the plan point ``bend_lead_s`` of the speed ahead (``lookahead_s`` while 0) is within
    ``bend_deg`` of the nose (``bending``, a bend look held: within :data:`BEND_EXIT` of it), else
    that point's bearing; while parking the parking ``spot`` (:func:`park_spot`, fixed when the
    parking began) instead of the plan point, at the driving tilt, the magnet the same. Clamped
    to ``pan_clamp_deg``. Everything at 0 and no spot is :func:`path_aim`."""
    ahead = lookahead_m(speed_m_s, law)
    if spot is not None:
        # the driving tilt: no tilt move at the hand-over, none as the spot nears
        reach_m = math.hypot(spot[0] - pose[0], spot[1] - pose[1])
        tilt = _tilt(min(max(reach_m, law.min_m), ahead), law, lens_z_m, home)
        return _magnet(PARK, _bearing(pose, *spot), tilt, law, reach, bending=bending)
    poses, arc = path_ahead(path_xy, (pose[0], pose[1]), ahead, step_m=ahead / 8.0)
    if len(poses) == 0 or arc[-1] <= 0.0:
        return None
    kind = POINT
    target = (float(poses[-1, 0]), float(poses[-1, 1]))
    tilt = _tilt(float(arc[-1]), law, lens_z_m, home)
    if law.bend_deg > 0.0:
        kind = BEND
        if law.bend_lead_s > 0.0:
            lead = lookahead_m(speed_m_s, law, law.bend_lead_s)
            far, far_arc = path_ahead(path_xy, (pose[0], pose[1]), lead, step_m=lead / 8.0)
            if len(far) and far_arc[-1] > 0.0:
                target = (float(far[-1, 0]), float(far[-1, 1]))
    return _magnet(kind, _bearing(pose, *target), tilt, law, reach, bending=bending)


def _magnet(
    kind: str, bearing: float, tilt: float, law: PathGazeLaw, reach: Reach, *, bending: bool
) -> PathLook:
    """The look at ``bearing`` (clamped), or straight ahead while the ahead magnet holds it."""
    spot = kind == PARK
    if law.bend_deg > 0.0:
        threshold = math.radians(law.bend_deg) * (BEND_EXIT if bending else 1.0)
        if abs(bearing) <= threshold:
            return PathLook(reach.clamp(Aim(0.0, tilt)), AHEAD, spot)
    clamp = math.radians(law.pan_clamp_deg)
    pan = min(max(bearing, -clamp), clamp)
    return PathLook(reach.clamp(Aim(pan, tilt)), kind, spot)


def settle(current: Aim | None, wanted: Aim, deadband_deg: float) -> Aim:
    """``current`` while ``wanted`` is within the dead-band of it, else ``wanted``."""
    if current is not None and current.off(wanted) <= math.radians(deadband_deg):
        return current
    return wanted


MOVING_M_S = 0.02  # slower than this the cart is taken as stopped: no end in sight
STILL_RAD_S = 0.1  # a cart turning slower than this stands (with |v| under the law's still_m_s)
ZERO_CMD = 1e-3  # a command this small in both v and w is the controller's zero


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


class StandWatch:
    """Whether the cart stands, and how long ago the command last dropped to zero from a moving
    one (the controller stopped: a stall look may be coming); fed every commanded twist."""

    def __init__(self) -> None:
        self.standing = False
        self._moved = False  # a nonzero command since the drive's start
        self._zero_since: float | None = None

    def reset(self) -> None:
        """A new drive: nothing commanded yet."""
        self.standing, self._moved, self._zero_since = False, False, None

    def update(self, v: float, w: float, now: float, law: PathGazeLaw) -> None:
        """One commanded twist."""
        self.standing = law.still_m_s > 0.0 and abs(v) < law.still_m_s and abs(w) < STILL_RAD_S
        if abs(v) <= ZERO_CMD and abs(w) <= ZERO_CMD:
            if self._moved and self._zero_since is None:
                self._zero_since = now
            return
        self._moved, self._zero_since = True, None

    def stopped_for(self, now: float) -> float:
        """Seconds since the command dropped to zero; infinite while it moves the cart or has
        not since the drive's start."""
        return math.inf if self._zero_since is None else now - self._zero_since


class PathFollower:
    """The path look's aim through one drive (see FOLLOWING and ECONOMY above); :meth:`reset` at
    a drive's start."""

    def __init__(self) -> None:
        self.aim: Aim | None = None
        self._out_since: float | None = None
        self._moved_at: float | None = None
        self._ahead_since: float | None = None  # the re-centre's condition held since
        self._adopted = False  # the aim is where another look left the head, not one chosen
        self.kind = ""  # the held aim's PathLook kind ("": none held, or adopted)

    def reset(self) -> None:
        """A new drive: no aim held."""
        self.aim, self._out_since, self._moved_at = None, None, None
        self._ahead_since, self._adopted, self.kind = None, False, ""

    @property
    def bending(self) -> bool:
        """Whether the held aim is a bend or a parking look (:func:`path_look`'s ``bending``)."""
        return self.kind in (BEND, PARK)

    def update(
        self,
        wanted: Aim,
        now: float,
        law: PathGazeLaw,
        *,
        end_in_s: float = math.inf,
        left_m: float = math.inf,
        fresh: bool = False,
        standing: bool = False,
        stopped_s: float = math.inf,
        head: Aim | None = None,
        home: Aim | None = None,
        kind: str = POINT,
        spot: bool = False,
    ) -> Aim | None:
        """The aim to hold now for the plan's ``wanted``, ``left_m`` of plan and ``end_in_s``
        seconds at the current speed from the plan's end; ``None`` when the path look holds
        nothing (the drive's tail, a stand with the head home). ``fresh``: the path look does
        not hold the head, so a move waits for neither the hysteresis nor the cooldown (and in
        the tail is not made). ``standing`` (:class:`StandWatch`), ``stopped_s`` since the
        command dropped to zero, ``head`` (where a look sent the head; ``None``: home, or found
        there) and ``home`` drive the ECONOMY rules. ``kind`` (:class:`PathLook`'s): an
        :data:`AHEAD` aim is out of the zone of any held pan more than :data:`AHEAD_TOL_DEG` off
        the nose, so the head comes back straight ahead once a bend is passed (the hysteresis
        and the cooldown permitting), however small the bend was. ``spot``: the aim is read off
        the parking spot, so in the tail the head still comes back straight ahead once the cart
        faces the spot (the tail makes no look aside, nor holds one)."""
        tail = not (spot and kind == AHEAD) and (
            (law.tail_s > 0.0 and end_in_s <= law.tail_s)
            or (law.tail_m > 0.0 and left_m <= law.tail_m)
        )
        if tail and fresh:
            return None  # a head elsewhere is not brought back in the last seconds either
        keep, kept = self._economy(
            wanted, law, fresh=fresh, standing=standing, stopped_s=stopped_s, head=head, home=home
        )
        if keep:
            return kept
        # an aim taken where another look left the head is no choice: the first move is made
        current = None if self._adopted else self.aim
        if current is not None and self._inside(current, wanted, kind, law):
            self._out_since = None
            if tail or not self._recentre(current, wanted, now, law):
                return current
            self.aim, self._moved_at, self._ahead_since, self.kind = wanted, now, None, kind
            return wanted
        self._ahead_since = None
        if tail:
            return self.aim
        if current is not None and not fresh:
            if self._out_since is None:
                self._out_since = now
            if now - self._out_since < law.hyst_s:
                return current
            if self._moved_at is not None and now - self._moved_at < law.cooldown_s:
                return current
        self.aim, self._out_since, self._moved_at = wanted, None, now
        self._adopted, self.kind = False, kind
        return wanted

    @staticmethod
    def _inside(current: Aim, wanted: Aim, kind: str, law: PathGazeLaw) -> bool:
        """Whether the held aim stays for the wanted one: within the zone, and straight ahead
        when the wanted aim is (the ahead magnet)."""
        if kind == AHEAD and abs(current.pan_rad - wanted.pan_rad) > math.radians(AHEAD_TOL_DEG):
            return False
        return current.off(wanted) <= math.radians(law.deadband_deg)

    def _economy(
        self,
        wanted: Aim,
        law: PathGazeLaw,
        *,
        fresh: bool,
        standing: bool,
        stopped_s: float,
        head: Aim | None,
        home: Aim | None,
    ) -> tuple[bool, Aim | None]:
        """Whether the ECONOMY rules keep the head now (no saccade), and the aim to hold: just
        after the command dropped to zero, or standing with the plan's aim within ``still_deg``
        of the head. A path look that does not hold the head takes it where a look left it,
        until the first move the rules let through; a head at home stays there without one."""
        guard = law.stall_guard_s > 0.0 and stopped_s < law.stall_guard_s
        if not (guard or (standing and law.still_m_s > 0.0)):
            return False, None
        where = (head if head is not None else home) if fresh else self.aim
        if where is None:
            return False, None
        if not guard and wanted.off(where) > math.radians(law.still_deg):
            return False, None
        if fresh:
            if head is None:
                return True, None  # home: the arbiter keeps it there, no look needed
            self.aim, self._out_since, self._ahead_since = head, None, None
            self._adopted, self.kind = True, ""
        return True, self.aim

    def _recentre(self, current: Aim, wanted: Aim, now: float, law: PathGazeLaw) -> bool:
        """Whether the held aim moves to the plan's now: its aim has been within
        ``recentre_deg`` of straight ahead for ``recentre_s``, the held one more than half the
        zone from it, and the cooldown has passed."""
        aside = abs(current.pan_rad - wanted.pan_rad) > math.radians(law.deadband_deg) / 2.0
        ahead = abs(wanted.pan_rad) <= math.radians(law.recentre_deg)
        if law.recentre_s <= 0.0 or not (aside and ahead):
            self._ahead_since = None
            return False
        if self._ahead_since is None:
            self._ahead_since = now
        if now - self._ahead_since < law.recentre_s:
            return False
        return self._moved_at is None or now - self._moved_at >= law.cooldown_s


@dataclass(frozen=True)
class ReverseLaw:
    """Reverse gaze's numbers (config/knobs.json's ``gaze`` block)."""

    pan_deg: float = 150.0
    tilt_deg: float = 23.8
    min_s: float = 1.0
    rear_m: float = 0.30
    min_speed_m_s: float = 0.02
    min_m: float = 0.15
    recovery: bool = True
    hold_s: float = 1.5


class ReverseWatch:
    """How long the cart has been reversing without a break, whether the leg was announced, and
    which side its rear last swung to (+1 left, -1 right). Once a reverse look has taken the side
    (:meth:`hold`) it stays for the rest of the leg — with the law's ``hold_s``, through a stand
    of up to that long after it, until the first forward twist: a controller that swings its turn
    through zero while it backs (drive 306: w -0.5, +0.6, -0.9, +0.9 rad/s within 4 s at
    -0.06 m/s) would otherwise send the head 300 deg across the back at every swing."""

    def __init__(self) -> None:
        self._since: float | None = None
        self._stood: float | None = None  # when the stand after a held leg began
        self._held = False
        self.side = 1
        self.announced = ""  # why this leg was announced ("" while it was not)

    def update(self, v: float, w: float, now: float, law: ReverseLaw) -> None:
        """One commanded twist."""
        if v < -law.min_speed_m_s:
            if self._since is None:
                self._since = now
            self._stood = None
            if self._held:
                return
            if w > 0.0:
                self.side = -1
            elif w < 0.0:
                self.side = 1
            return
        self._since = None
        self.announced = ""
        if not self._held:
            return
        if law.hold_s <= 0.0 or v > law.min_speed_m_s:
            self._held, self._stood = False, None
            return
        if self._stood is None:
            self._stood = now
        if now - self._stood >= law.hold_s:
            self._held, self._stood = False, None

    def announce(self, why: str) -> None:
        """This leg was announced (the plan's reverse leg, a recovery), for the rest of it."""
        self.announced = why

    def hold(self) -> int:
        """The side for this leg's reverse look, kept from now until the leg ends (with
        ``hold_s``: until the first forward twist or the stand's end)."""
        self._held = True
        return self.side

    def release(self) -> None:
        """No reverse look any more (a drive's start or end): the next leg chooses again."""
        self._held, self._stood = False, None

    def reversing_for(self, now: float) -> float:
        """Seconds of this reverse leg so far (0 when not reversing)."""
        return 0.0 if self._since is None else now - self._since

    @property
    def reversing(self) -> bool:
        """Whether the last twist reversed."""
        return self._since is not None

    @property
    def looking(self) -> bool:
        """Whether a reverse look holds its side (from :meth:`hold` until it is let go)."""
        return self._held


def reverse_aim(side: int, law: ReverseLaw, reach: Reach) -> Aim:
    """The head toward the rear on ``side`` (+1 left), inside the reach."""
    return reach.clamp(Aim(side * math.radians(law.pan_deg), math.radians(law.tilt_deg)))


def reverse_leg_m(path_xy: Array, pose: tuple[float, float, float]) -> float:
    """The plan's reverse leg at the cart (pose: x, y, yaw in the plan's frame): from the plan's
    vertex nearest the cart, the arc of its steps up to the first cusp (two steps that point
    apart, as Nav2's controllers find it), when the first of them points behind the cart; 0 when
    the plan leaves the cart forwards or has nothing left."""
    path = np.asarray(path_xy, dtype=float).reshape(-1, 2)
    if len(path) < 2:
        return 0.0
    start = int(np.argmin(np.hypot(path[:, 0] - pose[0], path[:, 1] - pose[1])))
    steps = np.diff(path[start:], axis=0)
    lengths = np.hypot(steps[:, 0], steps[:, 1])
    steps, lengths = steps[lengths > 1e-6], lengths[lengths > 1e-6]
    if len(steps) == 0:
        return 0.0
    if steps[0, 0] * math.cos(pose[2]) + steps[0, 1] * math.sin(pose[2]) >= 0.0:
        return 0.0
    apart = np.flatnonzero(np.einsum("ij,ij->i", steps[1:], steps[:-1]) < 0.0)
    end = int(apart[0]) + 1 if len(apart) else len(steps)
    return float(lengths[:end].sum())


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
