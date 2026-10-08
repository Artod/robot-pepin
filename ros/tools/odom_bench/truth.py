"""The lidar truth: where the cart really stood at each taped scan, matched against a frozen map.

Every taped scan (robot-frame angles, hull-filtered by bag_to_tape; the cart-post sectors of the
mount masked here too; the mount's x/y offset added) is matched with pepin.scanmatch's
CorrelativeMatcher against a map_server map. The first scan's seed is the tape's ``loc`` row
(RTAB-Map + odometry), optionally through a fixed SE(2) transform into another map's frame,
searched wide (+-0.6 m by 5 cm, +-25 deg by 1 deg); every later seed is the previous MATCHED pose
carried over the wheel odometry (``pose`` rows) between the two scan stamps. Each scan then gets
two passes: +-0.15 m by 1.5 cm, +-6 deg by 0.5 deg, then +-2 cm by 0.4 cm, +-0.8 deg by 0.1 deg
with parabolic sub-lattice interpolation, on all beams (max_points 500). The fit is the matched
pose's inlier fraction; a scan under ``min_fit`` (or under 60 beams) is not written and its seed is
carried on the wheels; after 5 such scans in a row the next is re-seeded from ``loc``, searched
wide.
"""

from __future__ import annotations

import bisect
import dataclasses
import hashlib
import json
import math
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.lidar import LidarMount
from pepin.mapping import grid_from_pgm
from pepin.odometry import Pose2D, wrap_angle
from pepin.scanmatch import CorrelativeMatcher, SearchWindow, apply_motion, relative_motion

Array = npt.NDArray[np.float64]
Row = dict[str, Any]
WIDE = SearchWindow(xy_m=0.6, xy_step_m=0.05, theta_deg=25.0, theta_step_deg=1.0)
COARSE = SearchWindow(xy_m=0.15, xy_step_m=0.015, theta_deg=6.0, theta_step_deg=0.5)
FINE = SearchWindow(xy_m=0.02, xy_step_m=0.004, theta_deg=0.8, theta_step_deg=0.1)
MIN_BEAMS = 60
RESEED_AFTER = 5
MIN_FIT = 0.5
HALF = 0.5  # s, the truth's velocity fit: +-HALF around the stamp


def load_tape(path: Path) -> dict[str, list[Row]]:
    """The tape's rows by topic, each list sorted by stamp."""
    rows: dict[str, list[Row]] = {}
    for line in path.read_text(errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        rows.setdefault(r.get("topic", ""), []).append(r)
    for v in rows.values():
        v.sort(key=lambda r: r["t"])
    return rows


class Track:
    """A pose topic of the tape, interpolated at any stamp inside it (None outside or far)."""

    def __init__(self, rows: list[Row]) -> None:
        self.t = np.array([r["t"] for r in rows])
        self.x = np.array([r["x"] for r in rows])
        self.y = np.array([r["y"] for r in rows])
        self.th = np.unwrap(np.array([r["theta"] for r in rows]))

    def at(self, t: float, max_gap: float = 0.5) -> Pose2D | None:
        """The pose at t, or None more than ``max_gap`` s from every row."""
        if len(self.t) == 0 or t < self.t[0] - max_gap or t > self.t[-1] + max_gap:
            return None
        i = bisect.bisect_left(self.t.tolist(), t)
        lo, hi = max(i - 1, 0), min(i, len(self.t) - 1)
        if min(abs(self.t[lo] - t), abs(self.t[hi] - t)) > max_gap:
            return None
        return Pose2D(
            float(np.interp(t, self.t, self.x)),
            float(np.interp(t, self.t, self.y)),
            wrap_angle(float(np.interp(t, self.t, self.th))),
        )


def points_of(record: Row, mount: LidarMount) -> Array:
    """A tape scan as base-frame points: None and short ranges dropped, the post sectors masked."""
    angles = np.array(record["angles"], dtype=float)
    ranges = np.array([np.nan if r is None else r for r in record["ranges"]], dtype=float)
    sensor_deg = (
        np.degrees(angles) + mount.yaw_offset_deg
    ) % 360.0  # mirror false: robot = s - yaw
    masked = np.array([mount.is_masked(a) for a in sensor_deg], dtype=bool)
    ok = np.isfinite(ranges) & (ranges > max(0.05, mount.min_range_m)) & ~masked
    return np.column_stack(
        (
            mount.x_m + ranges[ok] * np.cos(angles[ok]),
            mount.y_m + ranges[ok] * np.sin(angles[ok]),
        )
    )


def compose(t: tuple[float, float, float], p: Pose2D) -> Pose2D:
    """The fixed transform (x, y, yaw) applied to a pose: T * p."""
    c, s = math.cos(t[2]), math.sin(t[2])
    return Pose2D(t[0] + c * p.x - s * p.y, t[1] + s * p.x + c * p.y, wrap_angle(p.theta + t[2]))


def match(matcher: CorrelativeMatcher, guess: Pose2D, pts: Array, wide: bool) -> Pose2D:
    """The two-pass (three with ``wide``) search around ``guess``."""
    if wide:
        guess = matcher.match(guess, pts, WIDE).pose
    guess = matcher.match(guess, pts, COARSE).pose
    return matcher.match(guess, pts, FINE).pose


def still_stretches(rows: dict[str, list[Row]], wheels: Track, stamps: Array) -> list[Array]:
    """Index arrays of consecutive matched scans while the cart stood: the last /cmd_vel at or
    before the scan is zero (or none yet) and the wheels moved < 2 mm and < 0.3 deg around it."""
    cmd = rows.get("cmd", [])
    cmd_t = [r["t"] for r in cmd]

    def commanded(t: float) -> bool:
        i = bisect.bisect_right(cmd_t, t) - 1
        if i < 0:
            return False
        return bool(abs(cmd[i]["linear"]) > 1e-3 or abs(cmd[i]["angular"]) > 1e-3)

    out: list[Array] = []
    run: list[int] = []
    for k in range(len(stamps)):
        a, b = wheels.at(stamps[k] - 0.1), wheels.at(stamps[k] + 0.1)
        quiet = a is not None and b is not None
        if a is not None and b is not None:
            m = relative_motion(a, b)
            quiet = math.hypot(m.x, m.y) < 0.002 and abs(math.degrees(m.theta)) < 0.3
        if quiet and not commanded(stamps[k]):
            run.append(k)
        else:
            if len(run) >= 5:
                out.append(np.array(run))
            run = []
    if len(run) >= 5:
        out.append(np.array(run))
    return out


def match_tape(
    tape: Path,
    map_yaml: Path,
    mount: LidarMount,
    seed_tf: tuple[float, float, float] = (0.0, 0.0, 0.0),
    min_fit: float = MIN_FIT,
) -> tuple[list[tuple[float, float, float, float, float, int]], str]:
    """The matched scans (stamp, x, y, yaw, fit, beams) of one tape and the truth's own quality
    report (fit distribution, jitter standing still, scan-to-scan white noise moving)."""
    matcher = CorrelativeMatcher(grid_from_pgm(map_yaml), max_points=500, interpolate=True)
    rows = load_tape(tape)
    wheels, loc = Track(rows["pose"]), Track(rows["loc"])
    out: list[tuple[float, float, float, float, float, int]] = []
    fits_all: list[float] = []
    carried: Pose2D | None = None
    carried_t = 0.0
    misses = 0
    skipped = 0
    for record in rows["scan"]:
        t = record["t"]
        pts = points_of(record, mount)
        if len(pts) < MIN_BEAMS:
            skipped += 1
            continue
        if carried is None or misses >= RESEED_AFTER:
            wide = True
            seed = loc.at(t)
            if seed is None:
                skipped += 1
                continue
            guess = compose(seed_tf, seed)
        else:
            wide = False
            a, b = wheels.at(carried_t), wheels.at(t)
            if a is None or b is None:
                skipped += 1
                continue
            guess = apply_motion(carried, relative_motion(a, b))
        pose = match(matcher, guess, pts, wide)
        fit = matcher.inlier_fraction(pose, pts)
        fits_all.append(fit)
        if fit >= min_fit:
            out.append((t, pose.x, pose.y, pose.theta, fit, len(pts)))
            carried, carried_t, misses = pose, t, 0
        else:
            misses += 1
            if carried is not None:  # keep the seed alive on the wheels
                a, b = wheels.at(carried_t), wheels.at(t)
                if a is not None and b is not None:
                    carried, carried_t = apply_motion(carried, relative_motion(a, b)), t
    report = quality(tape, rows, wheels, loc, out, fits_all, skipped, seed_tf, min_fit)
    return out, report


def quality(
    tape: Path,
    rows: dict[str, list[Row]],
    wheels: Track,
    loc: Track,
    out: list[tuple[float, float, float, float, float, int]],
    fits_all: list[float],
    skipped: int,
    seed_tf: tuple[float, float, float],
    min_fit: float,
) -> str:
    """The truth's own quality, as text: scans kept, the fit distribution, the jitter of the
    matched pose standing still (the floor) and the white noise moving (2nd difference / sqrt 6)."""
    fits = np.array(fits_all)
    data = np.array([r[:4] for r in out])
    stamps, xs, ys = data[:, 0], data[:, 1], data[:, 2]
    yaws = np.unwrap(data[:, 3])
    path = float(np.sum(np.hypot(np.diff(xs), np.diff(ys))))
    wsel = (wheels.t >= stamps[0]) & (wheels.t <= stamps[-1])
    wpath = float(np.sum(np.hypot(np.diff(wheels.x[wsel]), np.diff(wheels.y[wsel]))))
    lines = [
        f"{tape.name}: {len(rows['scan'])} scans, {len(fits)} matched-tried, {len(out)} kept (fit"
        f" >= {min_fit}), {skipped} skipped; fit p10/p50/p90"
        f" {np.percentile(fits, 10):.2f}/{np.median(fits):.2f}/{np.percentile(fits, 90):.2f}, min"
        f" {fits.min():.2f}; span {stamps[-1] - stamps[0]:.1f} s; path {path:.2f} m (wheels"
        f" {wpath:.2f} m)"
    ]
    dts = np.diff(stamps)
    lines.append(f"  scan gaps: median {np.median(dts):.3f} s, max {dts.max():.2f} s")
    stretches = still_stretches(rows, wheels, stamps)
    if stretches:
        sx = np.concatenate([xs[s] - xs[s].mean() for s in stretches])
        sy = np.concatenate([ys[s] - ys[s].mean() for s in stretches])
        sth = np.concatenate([yaws[s] - yaws[s].mean() for s in stretches])
        n = sum(len(s) for s in stretches)
        spans = ", ".join(
            f"{stamps[s[0]] - stamps[0]:.1f}-{stamps[s[-1]] - stamps[0]:.1f} s" for s in stretches
        )
        wx = 100 * max(np.ptp(xs[s]) for s in stretches)
        wy = 100 * max(np.ptp(ys[s]) for s in stretches)
        lines.append(
            f"  STILL: {len(stretches)} stretches, {n} scans ({spans}): jitter std x"
            f" {100 * sx.std():.2f} cm, y {100 * sy.std():.2f} cm, 2-D"
            f" {100 * math.hypot(sx.std(), sy.std()):.2f} cm, yaw {math.degrees(sth.std()):.2f}"
            f" deg; worst spread {wx:.1f}/{wy:.1f} cm"
        )
    else:
        lines.append("  STILL: no stretch of >= 5 scans with the cart standing")
    # second differences only over three consecutive scans with no dropped scan between them
    tight = (np.diff(stamps)[:-1] < 0.15) & (np.diff(stamps)[1:] < 0.15)
    d2x, d2y = np.diff(xs, 2)[tight], np.diff(ys, 2)[tight]
    d2th = np.diff(yaws, 2)[tight]
    lines.append(
        f"  MOVING white noise (std of 2nd diff / sqrt 6): x {100 * d2x.std() / math.sqrt(6):.2f}"
        f" cm, y {100 * d2y.std() / math.sqrt(6):.2f} cm, yaw"
        f" {math.degrees(d2th.std() / math.sqrt(6)):.2f} deg"
    )
    s0 = compose(seed_tf, loc.at(stamps[0]) or Pose2D(0, 0, 0))
    lines.append(
        f"  first matched pose ({xs[0]:+.3f}, {ys[0]:+.3f}, {math.degrees(yaws[0]):+.1f} deg) vs"
        f" its loc seed ({s0.x:+.3f}, {s0.y:+.3f}, {math.degrees(s0.theta):+.1f} deg)"
    )
    return "\n".join(lines) + "\n"


def write_truth(
    tape: Path,
    map_yaml: Path,
    mount: LidarMount,
    out: Path,
    seed_tf: tuple[float, float, float] = (0.0, 0.0, 0.0),
    min_fit: float = MIN_FIT,
) -> str:
    """``match_tape`` written as CSV (stamp, x, y, yaw, fit, n) to ``out``; returns the report."""
    rows, report = match_tape(tape, map_yaml, mount, seed_tf, min_fit)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        f.write("stamp,x,y,yaw,fit,n\n")
        for t, x, y, th, fit, n in rows:
            f.write(f"{t:.6f},{x:.5f},{y:.5f},{th:.6f},{fit:.3f},{n}\n")
    return report


def truth_key(map_yaml: Path, mount: LidarMount) -> str:
    """What a truth depends on besides its tape: the map (yaml and image) and the mount."""
    import yaml

    image = map_yaml.parent / yaml.safe_load(map_yaml.read_text())["image"]
    h = hashlib.sha256(map_yaml.read_bytes())
    h.update(image.read_bytes())
    return json.dumps({"map": h.hexdigest(), "mount": dataclasses.asdict(mount)}, sort_keys=True)


def _truth_job(job: tuple[Path, Path, LidarMount, Path]) -> str:
    return write_truth(*job)


def cached_truths(items: list[tuple[Path, Path]], map_yaml: Path, mount: LidarMount) -> int:
    """(tape, out CSV) pairs: each truth computed (8 at a time, its report beside it as
    truth.txt) unless the CSV is newer than its tape and was made from this map and mount (the
    truth.key beside it); returns how many were computed."""
    key = truth_key(map_yaml, mount)

    def fresh(tape: Path, out: Path) -> bool:
        k = out.with_name("truth.key")
        return (
            out.exists()
            and out.stat().st_mtime > tape.stat().st_mtime
            and k.exists()
            and k.read_text() == key
        )

    todo = [(tape, out) for tape, out in items if not fresh(tape, out)]
    jobs = [(tape, map_yaml, mount, out) for tape, out in todo]
    with ProcessPoolExecutor(8) as ex:
        for (_tape, out), report in zip(todo, ex.map(_truth_job, jobs), strict=True):
            out.with_name("truth.txt").write_text(report)
            out.with_name("truth.key").write_text(key)
    return len(todo)


def load_truth(path: Path, heading_deg: float = 0.0) -> Array:
    """A truth CSV as (stamp, x, y, unwrapped yaw + heading_deg) rows."""
    a = np.genfromtxt(path, delimiter=",", names=True)
    yaw = np.unwrap(a["yaw"])
    if heading_deg:
        yaw = yaw + math.radians(heading_deg)
    return np.column_stack([a["stamp"], a["x"], a["y"], yaw])


def fit_vw(tr: Array, t: float) -> tuple[float, float] | None:
    """The truth's body forward speed (rotated by its yaw at t) and yaw rate at t: a linear fit
    over the scans within +-0.5 s (at least 4), None with fewer."""
    i0, i1 = np.searchsorted(tr[:, 0], [t - HALF, t + HALF + 1e-12])
    while i0 < i1 and abs(tr[i0, 0] - t) > HALF:
        i0 += 1
    while i1 > i0 and abs(tr[i1 - 1, 0] - t) > HALF:
        i1 -= 1
    if i1 - i0 < 4:
        return None
    tt = tr[i0:i1, 0] - t
    d = tt - tt.mean()
    w = d / float(np.sum(d * d))
    yaw = float(np.interp(t, tr[:, 0], tr[:, 3]))
    return (
        math.cos(yaw) * float(w @ tr[i0:i1, 1]) + math.sin(yaw) * float(w @ tr[i0:i1, 2]),
        float(w @ tr[i0:i1, 3]),
    )
