#!/usr/bin/env python3
"""The offline A/B's scorer: each arm's fused odometry against the lidar truth, pre-registered.

vio.md section 6 fixes these metrics BEFORE any drive is recorded (the arms: A the EKF without
/vo, B today's stereo VO with grab stamps and the SE(2) track, C/D cheap guesses, E/E' OpenVINS):

* PRIMARY: the EKF-level relative pose error, translation per metre, of each arm's
  ``/odometry/filtered`` against the lidar truth, over consecutive 1 m segments of the TRUTH's
  path (non-overlapping), the median over all segments of all drives; and the paired difference of
  each arm against B on the same segments.
* the decision rule for making ``vio`` the default: the paired bootstrap 95 % CI of
  median(E) - median(B), resampling 1 m segments WITHIN drives (the same resample for both arms),
  lies wholly below 0, AND E beats B on at least 5 of the 6 S3 drives by the per-drive median.
* secondary, reported and deciding nothing: yaw RPE per metre, the end-point error, the segment
  count, and the per-drive medians.

Input, one directory per drive: ``truth.csv`` (stamp, x, y, yaw[, fit]: ros/tools/lidar_truth.py's)
and ``<arm>.csv`` per arm (stamp, x, y, yaw of /odometry/filtered, from the arm's replay)::

    uv run python ros/tools/vio_score.py runs/0601 runs/0602 ... --arms A B E --baseline B
    uv run python ros/tools/vio_score.py runs/06* --arms B E --s3 0601 0602 0603 0604 0605 0606

The bootstrap is seeded (--seed, default 1) and its count fixed (--resamples, 2000), so a table is
reproducible from its inputs.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
SEGMENT_M = 1.0


@dataclass(frozen=True)
class Trajectory:
    """Planar poses by stamp (seconds, metres, radians), sorted."""

    t: Array
    x: Array
    y: Array
    yaw: Array

    @classmethod
    def from_rows(cls, rows: Sequence[Sequence[float]]) -> Trajectory:
        """From (stamp, x, y, yaw) rows in any order; duplicate stamps keep the first."""
        data = np.array(sorted({float(r[0]): r for r in rows}.values(), key=lambda r: r[0]))
        return cls(data[:, 0], data[:, 1], data[:, 2], np.unwrap(data[:, 3]))

    @classmethod
    def load(cls, path: Path) -> Trajectory:
        """A csv with a header naming stamp, x, y, yaw (any extra columns ignored)."""
        with path.open() as stream:
            reader = csv.DictReader(stream)
            rows = [
                (float(r["stamp"]), float(r["x"]), float(r["y"]), float(r["yaw"])) for r in reader
            ]
        return cls.from_rows(rows)

    def at(self, stamps: Array) -> tuple[Array, Array, Array]:
        """x, y, yaw interpolated at ``stamps`` (inside the trajectory's span; NaN outside)."""
        inside = (stamps >= self.t[0]) & (stamps <= self.t[-1])
        out = [np.interp(stamps, self.t, v) for v in (self.x, self.y, self.yaw)]
        for values in out:
            values[~inside] = np.nan
        return out[0], out[1], out[2]


def segments(truth: Trajectory, length_m: float = SEGMENT_M) -> list[tuple[int, int, float]]:
    """Consecutive, non-overlapping stretches of the truth's path at least ``length_m`` long, as
    (start index, end index, path length)."""
    step = np.hypot(np.diff(truth.x), np.diff(truth.y))
    travelled = np.concatenate([[0.0], np.cumsum(step)])
    out: list[tuple[int, int, float]] = []
    start = 0
    while True:
        target = travelled[start] + length_m
        end = int(np.searchsorted(travelled, target))
        if end >= len(travelled):
            return out
        out.append((start, end, float(travelled[end] - travelled[start])))
        start = end


def relative(
    x0: float, y0: float, yaw0: float, x1: float, y1: float, yaw1: float
) -> tuple[float, float, float]:
    """The motion from pose 0 to pose 1 in pose 0's frame (dx, dy, dyaw)."""
    c, s = math.cos(yaw0), math.sin(yaw0)
    dx, dy = x1 - x0, y1 - y0
    return c * dx + s * dy, -s * dx + c * dy, math.remainder(yaw1 - yaw0, math.tau)


@dataclass
class DriveScore:
    """One drive, one arm: the per-segment translation and yaw errors per metre, and the end-point
    error (m) against the truth's last pose, both aligned at the first pose."""

    rpe: Array
    yaw_rpe: Array
    end_point_m: float


def score(truth: Trajectory, arm: Trajectory, length_m: float = SEGMENT_M) -> DriveScore:
    """An arm's errors on the truth's segments (a segment the arm does not cover is NaN)."""
    spans = segments(truth, length_m)
    rpe, yaw_rpe = [], []
    for i, j, length in spans:
        stamps = np.array([truth.t[i], truth.t[j]])
        ax, ay, ayaw = arm.at(stamps)
        if not np.all(np.isfinite(ax)):
            rpe.append(math.nan)
            yaw_rpe.append(math.nan)
            continue
        t_rel = relative(truth.x[i], truth.y[i], truth.yaw[i], truth.x[j], truth.y[j], truth.yaw[j])
        a_rel = relative(ax[0], ay[0], ayaw[0], ax[1], ay[1], ayaw[1])
        rpe.append(math.hypot(a_rel[0] - t_rel[0], a_rel[1] - t_rel[1]) / length)
        yaw_rpe.append(abs(math.degrees(math.remainder(a_rel[2] - t_rel[2], math.tau))) / length)
    end = math.nan
    first = np.array([truth.t[0], truth.t[-1]])
    ax, ay, ayaw = arm.at(first)
    if np.all(np.isfinite(ax)):
        t_rel = relative(
            truth.x[0], truth.y[0], truth.yaw[0], truth.x[-1], truth.y[-1], truth.yaw[-1]
        )
        a_rel = relative(ax[0], ay[0], ayaw[0], ax[1], ay[1], ayaw[1])
        end = math.hypot(a_rel[0] - t_rel[0], a_rel[1] - t_rel[1])
    return DriveScore(np.array(rpe), np.array(yaw_rpe), end)


@dataclass
class Comparison:
    """An arm against the baseline over all drives: the medians, the paired difference with its
    bootstrap CI, and the per-drive wins."""

    arm: str
    baseline: str
    median_arm: float
    median_baseline: float
    ci: tuple[float, float]
    wins: int
    drives: int
    segments: int
    per_drive: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def difference(self) -> float:
        """median(arm) - median(baseline): negative is the arm's way."""
        return self.median_arm - self.median_baseline

    def passes(self, min_wins: int) -> bool:
        """The decision rule: the CI wholly below 0 and at least ``min_wins`` drive wins."""
        return self.ci[1] < 0.0 and self.wins >= min_wins


def paired_bootstrap(
    pairs: dict[str, tuple[Array, Array]], resamples: int, seed: int
) -> tuple[float, float]:
    """95 % CI of median(arm) - median(baseline) over segments, resampling segments within each
    drive with replacement, the same draw for both arms."""
    rng = np.random.default_rng(seed)
    stats = np.empty(resamples)
    for k in range(resamples):
        arm_all, base_all = [], []
        for arm, base in pairs.values():
            draw = rng.integers(0, len(arm), len(arm))
            arm_all.append(arm[draw])
            base_all.append(base[draw])
        stats[k] = np.median(np.concatenate(arm_all)) - np.median(np.concatenate(base_all))
    low, high = np.percentile(stats, [2.5, 97.5])
    return float(low), float(high)


def compare(
    drives: dict[str, dict[str, DriveScore]],
    arm: str,
    baseline: str,
    resamples: int = 2000,
    seed: int = 1,
    counted: Sequence[str] | None = None,
) -> Comparison:
    """``arm`` against ``baseline`` on the segments both cover; wins over ``counted`` drives (all
    by default; the S3 drives for the decision rule)."""
    pairs: dict[str, tuple[Array, Array]] = {}
    per_drive: dict[str, tuple[float, float]] = {}
    for name, arms in drives.items():
        a, b = arms[arm].rpe, arms[baseline].rpe
        both = np.isfinite(a) & np.isfinite(b)
        if not np.any(both):
            continue
        pairs[name] = (a[both], b[both])
        per_drive[name] = (float(np.median(a[both])), float(np.median(b[both])))
    if not pairs:
        raise ValueError(f"no segment that both {arm} and {baseline} cover")
    all_a = np.concatenate([p[0] for p in pairs.values()])
    all_b = np.concatenate([p[1] for p in pairs.values()])
    names = list(counted) if counted is not None else list(per_drive)
    wins = sum(1 for n in names if n in per_drive and per_drive[n][0] < per_drive[n][1])
    return Comparison(
        arm=arm,
        baseline=baseline,
        median_arm=float(np.median(all_a)),
        median_baseline=float(np.median(all_b)),
        ci=paired_bootstrap(pairs, resamples, seed),
        wins=wins,
        drives=len([n for n in names if n in per_drive]),
        segments=len(all_a),
        per_drive=per_drive,
    )


def main(argv: list[str] | None = None) -> int:
    """The table: per arm the primary and secondaries, then each arm against the baseline."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("drives", nargs="+", type=Path, help="one directory per drive")
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--baseline", default="B")
    parser.add_argument("--s3", nargs="*", default=None, help="the drives the wins count over")
    parser.add_argument("--min-wins", type=int, default=5)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args(argv)
    drives: dict[str, dict[str, DriveScore]] = {}
    for directory in args.drives:
        truth = Trajectory.load(directory / "truth.csv")
        drives[directory.name] = {
            arm: score(truth, Trajectory.load(directory / f"{arm}.csv")) for arm in args.arms
        }
    print(f"{len(drives)} drives, segments of {SEGMENT_M:g} m of the lidar truth's path")
    for arm in args.arms:
        rpe = np.concatenate([d[arm].rpe for d in drives.values()])
        yaw = np.concatenate([d[arm].yaw_rpe for d in drives.values()])
        ends = [d[arm].end_point_m for d in drives.values()]
        print(
            f"{arm}: RPE {100 * np.nanmedian(rpe):.2f} %/m median"
            f" ({np.sum(np.isfinite(rpe))} segments), yaw {np.nanmedian(yaw):.2f} deg/m,"
            f" end point {100 * np.nanmedian(ends):.1f} cm median"
        )
    for arm in args.arms:
        if arm == args.baseline:
            continue
        result = compare(drives, arm, args.baseline, args.resamples, args.seed, args.s3)
        verdict = "PASSES" if result.passes(args.min_wins) else "does not pass"
        print(
            f"{arm} - {args.baseline}: {100 * result.difference:+.2f} %/m, 95 % CI"
            f" [{100 * result.ci[0]:+.2f}, {100 * result.ci[1]:+.2f}] %/m over {result.segments}"
            f" segments; wins {result.wins}/{result.drives}; the decision rule {verdict}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
