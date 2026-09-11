"""Time alignment of scans and odometry: every beam is placed where the robot was when it was taken.

A lidar revolution is not a photograph. The LD19 turns in 100 ms and stamps the moment the
revolution ends; in a pivot at 0.6 rad/s the first beam and the last one are taken 3.4 degrees
apart, and a matcher fed the raw revolution settles between them, a degree or two off, on every
scan of the turn. Worse, the tracker used to pair a scan with whatever ``odom -> base_link`` was
newest when the scan arrived: the filter runs 35-70 ms behind the scan, so in the same pivot the
"pose at the scan" was the pose from 1-2 degrees earlier, and that difference went into
``map -> odom`` as a correction. Both errors change sign with the turn; the belief wobbled and
the cart steered by the wobble (runs 0080-0083, 2026-09-09).

This module fixes the time of every measurement once, in pure code:

* :class:`OdomHistory` keeps the odometry poses of the last seconds and interpolates the pose
  at any time in between — never outside.
* :func:`beam_times` and :func:`deskew` place every beam at its own moment and move it into the
  frame of the stamp.
* :class:`ScanGate` holds the newest scan until odometry covers its whole revolution; a scan is
  matched against the pose it was taken at, or not at all.
* :class:`MotionFilter` spares the matcher while the cart stands still.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from pepin.odometry import Pose2D, wrap_angle

__all__ = [
    "MatchPacer",
    "MotionFilter",
    "OdomHistory",
    "PacerStats",
    "ScanGate",
    "TimedScan",
    "beam_times",
    "deskew",
    "standing_still",
]


class OdomHistory:
    """``odom -> base_link`` over the last ``horizon_s`` seconds, readable at any time in between.

    Poses are interpolated linearly in x, y and in the unwrapped heading, which is what the EKF's
    20 Hz output is between two samples (2.5 mm and a quarter of a degree at full speed). Nothing
    is ever extrapolated: a time after the newest sample or before the oldest reads as ``None``.
    """

    def __init__(self, horizon_s: float = 5.0) -> None:
        self._horizon_s = horizon_s
        self._t: deque[float] = deque()
        self._x: deque[float] = deque()
        self._y: deque[float] = deque()
        self._yaw: deque[float] = deque()  # unwrapped: continuous across +-pi

    def add(self, t: float, pose: Pose2D) -> None:
        """Append a sample; one older than the newest is ignored (a late message, a clock step)."""
        if self._t and t <= self._t[-1]:
            return
        yaw = (
            pose.theta if not self._yaw else self._yaw[-1] + wrap_angle(pose.theta - self._yaw[-1])
        )
        self._t.append(t)
        self._x.append(pose.x)
        self._y.append(pose.y)
        self._yaw.append(yaw)
        while self._t and self._t[0] < t - self._horizon_s:
            self._t.popleft()
            self._x.popleft()
            self._y.popleft()
            self._yaw.popleft()

    def __len__(self) -> int:
        return len(self._t)

    @property
    def newest_t(self) -> float | None:
        """Time of the newest sample, or ``None`` when empty."""
        return self._t[-1] if self._t else None

    @property
    def oldest_t(self) -> float | None:
        """Time of the oldest sample kept, or ``None`` when empty."""
        return self._t[0] if self._t else None

    @property
    def newest(self) -> Pose2D | None:
        """The newest pose, or ``None`` when empty."""
        if not self._t:
            return None
        return Pose2D(self._x[-1], self._y[-1], wrap_angle(self._yaw[-1]))

    def covers(self, t0: float, t1: float) -> bool:
        """True when both ``t0`` and ``t1`` lie inside the recorded span."""
        return bool(self._t) and self._t[0] <= t0 and t1 <= self._t[-1]

    def at(self, t: float) -> Pose2D | None:
        """The pose at ``t``, or ``None`` when ``t`` is outside the recorded span."""
        if not self.covers(t, t):
            return None
        x, y, yaw = self.poses_at(np.array([t]))
        return Pose2D(float(x[0]), float(y[0]), wrap_angle(float(yaw[0])))

    def poses_at(
        self, times: NDArray[np.float64]
    ) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """x, y and unwrapped heading at each of ``times`` (all must be inside the span)."""
        t = np.fromiter(self._t, dtype=np.float64, count=len(self._t))
        x = np.fromiter(self._x, dtype=np.float64, count=len(self._x))
        y = np.fromiter(self._y, dtype=np.float64, count=len(self._y))
        yaw = np.fromiter(self._yaw, dtype=np.float64, count=len(self._yaw))
        return np.interp(times, t, x), np.interp(times, t, y), np.interp(times, t, yaw)


def beam_times(stamp: float, n: int, period_s: float) -> NDArray[np.float64]:
    """When each of the ``n`` beams of one LD19 revolution was taken.

    The driver stamps the END of the revolution (the message is 8 ms old on arrival, not 100),
    and with its counter-clockwise verse writes the newest beam at index 0 and the oldest at the
    last index. Verified on runs 0080-0083: deskewing with this order scores best while pivoting
    and the opposite order scores worst (scratch/deskew_sign_check.py).
    """
    if n < 2:
        return np.full(max(n, 0), stamp, dtype=np.float64)
    return stamp - np.arange(n, dtype=np.float64) / (n - 1) * period_s


def deskew(
    points: NDArray[np.float64],
    times: NDArray[np.float64],
    history: OdomHistory,
    t_ref: float,
) -> NDArray[np.float64] | None:
    """Move every point from the base frame at its own time into the base frame at ``t_ref``.

    ``points`` is (N, 2) in the base frame as measured, ``times`` (N,) when each was taken.
    Returns the (N, 2) points as they would have been seen from the pose at ``t_ref`` — a wall
    scanned during a turn becomes straight again — or ``None`` when the history does not cover
    every time asked for (then nothing can be said about where the robot was).
    """
    if len(points) == 0:
        return points
    lo, hi = float(times.min()), float(times.max())
    if not history.covers(min(lo, t_ref), max(hi, t_ref)):
        return None
    x, y, yaw = history.poses_at(times)
    xr, yr, yawr = history.poses_at(np.array([t_ref]))
    # the pose at each beam's time, relative to the pose at t_ref, in the t_ref frame
    c, s = math.cos(yawr[0]), math.sin(yawr[0])
    dx, dy = x - xr[0], y - yr[0]
    ox, oy = c * dx + s * dy, -s * dx + c * dy
    dyaw = yaw - yawr[0]
    cc, ss = np.cos(dyaw), np.sin(dyaw)
    px, py = points[:, 0], points[:, 1]
    return np.column_stack((ox + cc * px - ss * py, oy + ss * px + cc * py))


@dataclass(frozen=True)
class TimedScan:
    """One revolution as received: points in the base frame and the time each one was taken."""

    stamp: float  # end of the revolution (the message stamp)
    points: NDArray[np.float64]  # (N, 2) valid returns in the base frame, as measured
    times: NDArray[np.float64]  # (N,) when each valid return was taken
    ranges: NDArray[np.float64]  # every beam's range, NaN where nothing came back
    scan_id: int  # which scan this is; a second opinion needs a new one

    @property
    def first_t(self) -> float:
        """Time of the oldest beam (the stamp itself when the scan is empty)."""
        return float(self.times.min()) if len(self.times) else self.stamp


@dataclass
class GateStats:
    """What the gate did since the last report."""

    offered: int = 0  # scans received
    released: int = 0  # scans handed to the matcher, odometry covering their whole revolution
    replaced: int = 0  # scans superseded by a newer one before odometry caught up
    expired: int = 0  # scans dropped: odometry never covered them within the wait allowed
    waited_s: list[float] = field(default_factory=list)  # release delay of each released scan

    def summary(self) -> str:
        """One log line: counts and the wait for odometry (mean and worst)."""
        waited = (
            f"{np.mean(self.waited_s) * 1000:.0f}/{max(self.waited_s) * 1000:.0f} ms"
            if self.waited_s
            else "-"
        )
        return (
            f"scans {self.offered}, released {self.released}, replaced {self.replaced}, "
            f"expired {self.expired}, waited for odom mean/max {waited}"
        )


class ScanGate:
    """Holds the newest scan until odometry covers its whole revolution.

    A scan is released exactly once, with a guarantee: the history has a pose at the time of its
    every beam, so the matcher pairs the scan with where the robot was — never with where it
    is now. A newer scan replaces a waiting one (the old picture is stale anyway); a scan the
    odometry has not covered within ``max_wait_s`` is dropped and counted, which the report
    turns into a warning that the odometry runs late.
    """

    def __init__(self, max_wait_s: float = 0.5) -> None:
        self._max_wait_s = max_wait_s
        self._pending: TimedScan | None = None
        self.stats = GateStats()

    @property
    def pending(self) -> TimedScan | None:
        """The scan waiting for odometry, if any."""
        return self._pending

    def offer(self, scan: TimedScan) -> None:
        """A new scan arrived; it takes the place of any scan still waiting."""
        self.stats.offered += 1
        if self._pending is not None:
            self.stats.replaced += 1
        self._pending = scan

    def take(self, history: OdomHistory, now: float) -> TimedScan | None:
        """The waiting scan once ``history`` covers it, else ``None``; expired scans are dropped."""
        scan = self._pending
        if scan is None:
            return None
        if history.covers(scan.first_t, scan.stamp):
            self._pending = None
            self.stats.released += 1
            self.stats.waited_s.append(max(0.0, now - scan.stamp))
            return scan
        self.expire(now)
        return None

    def expire(self, now: float) -> bool:
        """Drop the waiting scan once it has waited longer than ``max_wait_s`` uncovered,
        counted as expired; True when it did. What :meth:`take` does with a scan it cannot
        release: the odometry ran late."""
        scan = self._pending
        if scan is None or now - scan.stamp <= self._max_wait_s:
            return False
        self._pending = None
        self.stats.expired += 1
        return True

    def drop(self) -> TimedScan | None:
        """Forget the waiting scan and return it: another source's update took it along, or it
        aged out beside the anchor (:class:`pepin.sources.SourceFeed`). Not a release and not an
        expiry: the counters are the caller's."""
        scan, self._pending = self._pending, None
        return scan

    def report(self) -> GateStats:
        """The counters since the previous report, which are reset."""
        stats, self.stats = self.stats, GateStats()
        return stats


class MotionFilter:
    """Says when a scan is worth matching: the cart moved, or a while passed standing still.

    Matching every revolution of a standing cart is 60% of an A53 core spent confirming the same
    pose. A match every ``max_gap_s`` keeps the confidence fresh; any step of ``min_m`` or
    ``min_deg`` since the last match brings the matcher back to every scan.
    """

    def __init__(self, min_m: float = 0.005, min_deg: float = 0.3, max_gap_s: float = 1.0) -> None:
        self._min_m = min_m
        self._min_rad = math.radians(min_deg)
        self._max_gap_s = max_gap_s
        self._last: tuple[Pose2D, float] | None = None

    def due(self, odom: Pose2D, t: float) -> bool:
        """True when this scan should be matched; remembers it as the last match if so."""
        if self._last is None:
            self._last = (odom, t)
            return True
        before, t0 = self._last
        moved = math.hypot(odom.x - before.x, odom.y - before.y) >= self._min_m
        turned = abs(wrap_angle(odom.theta - before.theta)) >= self._min_rad
        if moved or turned or t - t0 >= self._max_gap_s:
            self._last = (odom, t)
            return True
        return False

    def reset(self) -> None:
        """Forget the last match (a new map, a re-seed): the next scan is due."""
        self._last = None


class MotionEdge:
    """Did the cart move between two looks at the odometry?

    An edge detector, not a speedometer: each call compares the pose it is given with the one
    of the previous call and keeps it. Calling it twice in the same tick compares a reading
    with itself and answers "standing still" — which once killed the tracker's "never re-seed
    a moving robot" gate (2026-09-09), so the caller samples it exactly once per tick.
    """

    def __init__(self, min_m: float = 0.01, min_deg: float = 1.0) -> None:
        self._min_m = min_m
        self._min_rad = math.radians(min_deg)
        self._last: Pose2D | None = None

    def moved(self, now: Pose2D | None) -> bool:
        """True when ``now`` is more than ``min_m`` or ``min_deg`` from the previous call's
        pose. False without odometry, and false on the first call: there is nothing to compare."""
        if now is None:
            return False
        before, self._last = self._last, now
        if before is None:
            return False
        turned = abs(wrap_angle(now.theta - before.theta)) > self._min_rad
        return math.hypot(now.x - before.x, now.y - before.y) > self._min_m or turned


@dataclass
class PacerStats:
    """What the pacer did since the last report: matches timed, scans skipped and why."""

    matched: int = 0
    took_s: float = 0.0  # seconds spent matching, in total
    worst_s: float = 0.0  # the longest single match
    skipped_gap: int = 0  # released too soon after the previous match
    skipped_busy: int = 0  # the previous match ran long; the board got that time back
    skipped_searching: int = 0  # a whole-map search runs in the worker

    @property
    def skipped(self) -> int:
        """Scans released by the gate and matched by nobody, for any reason."""
        return self.skipped_gap + self.skipped_busy + self.skipped_searching

    def summary(self) -> str:
        """One log line: the match cost and the skips by reason."""
        cost = (
            f"{self.took_s / self.matched * 1000:.0f} ms mean, {self.worst_s * 1000:.0f} ms worst"
            if self.matched
            else "-"
        )
        return (
            f"matched {self.matched} ({cost}), skipped {self.skipped} (gap {self.skipped_gap}, "
            f"busy {self.skipped_busy}, searching {self.skipped_searching})"
        )


class MatchPacer:
    """Spares the board: says whether a released scan may be matched right now, and counts the
    ones that may not, so the gate's ``released`` is accounted for scan by scan.

    A scan is skipped while a whole-map search runs in the worker, for about as long as the
    previous match took when it ran past ``long_match_s`` (the executor gets that time back for
    the odometry and the controller), and within ``min_gap_s`` of the previous match (two
    scans in one burst say nothing new).
    """

    def __init__(self, min_gap_s: float = 0.05, long_match_s: float = 0.12) -> None:
        self._min_gap_s = min_gap_s
        self._long_match_s = long_match_s
        self._last_match_at = -math.inf
        self._busy_until = -math.inf
        self.stats = PacerStats()

    def skip(self, now: float, searching: bool) -> bool:
        """True when this scan is not to be matched at ``now`` (monotonic seconds); counted."""
        if searching:
            self.stats.skipped_searching += 1
        elif now < self._busy_until:
            self.stats.skipped_busy += 1
        elif now - self._last_match_at < self._min_gap_s:
            self.stats.skipped_gap += 1
        else:
            return False
        return True

    def matched(self, now: float, took_s: float) -> None:
        """A match started at ``now`` and took ``took_s``; a long one buys the board a pause."""
        self._last_match_at = now
        if took_s > self._long_match_s:
            self._busy_until = now + took_s
        self.stats.matched += 1
        self.stats.took_s += took_s
        self.stats.worst_s = max(self.stats.worst_s, took_s)

    def report(self) -> PacerStats:
        """The counters since the previous report, which are reset."""
        stats, self.stats = self.stats, PacerStats()
        return stats


REST_WINDOW_S = 0.6  # how far back "the cart has not moved" is asked about
REST_MOVE_M = 0.01  # travel allowed over that window and still called rest
REST_TURN_DEG = 0.6  # turn allowed over that window and still called rest
REST_YAW_RATE_DEG_S = 1.5  # the gyro must be this quiet too; a slow pivot is 20-35 deg/s


def standing_still(
    history: OdomHistory,
    t: float,
    yaw_rate: float,
    window_s: float = REST_WINDOW_S,
    max_move_m: float = REST_MOVE_M,
    max_turn_deg: float = REST_TURN_DEG,
    max_yaw_rate_deg_s: float = REST_YAW_RATE_DEG_S,
) -> bool:
    """True when the cart has not moved for ``window_s`` before ``t`` and the gyro is quiet.

    Two independent witnesses on purpose: the wheels (through the fused odometry in ``history``)
    and ``yaw_rate`` in rad/s, which comes from the gyro and answers before the wheels have turned
    a countable tick. Unknown history — the window reaches before the oldest sample — reads as
    "moving", so the answer is only ever used to hold a pose that is known to be at rest.
    """
    if abs(math.degrees(yaw_rate)) > max_yaw_rate_deg_s:
        return False
    now, before = history.at(t), history.at(t - window_s)
    if now is None or before is None:
        return False
    moved = math.hypot(now.x - before.x, now.y - before.y)
    turned = abs(wrap_angle(now.theta - before.theta))
    return moved <= max_move_m and turned <= math.radians(max_turn_deg)


def timed_scan_from_ros(
    stamp: float,
    ranges: Any,
    angle_min: float,
    angle_increment: float,
    range_max: float,
    scan_time: float,
    laser: tuple[float, float, float, bool],
    scan_id: int,
) -> TimedScan:
    """A :class:`TimedScan` from the fields of a ``sensor_msgs/LaserScan``.

    ``laser`` is the mount (x, y, yaw, mirrored) of the sensor in the base frame. Beams shorter
    than 5 cm or beyond ``range_max`` are dropped from the points but kept as NaN in ``ranges``.
    """
    r = np.asarray(ranges, dtype=np.float64)
    n = len(r)
    ok = np.isfinite(r) & (r > 0.05) & (r < range_max)
    angles = angle_min + np.arange(n) * angle_increment
    px, py = r[ok] * np.cos(angles[ok]), r[ok] * np.sin(angles[ok])
    lx, ly, lyaw, mirrored = laser
    if mirrored:
        py = -py
    c, s = math.cos(lyaw), math.sin(lyaw)
    points = np.column_stack((lx + c * px - s * py, ly + s * px + c * py))
    return TimedScan(
        stamp=stamp,
        points=points,
        times=beam_times(stamp, n, scan_time)[ok],
        ranges=np.where(ok, r, np.nan),
        scan_id=scan_id,
    )
