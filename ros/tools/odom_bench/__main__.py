"""The odometry bench's command line (ros/README.md "Odometry bench").

uv run --with rosbags==0.11.5 python -m ros.tools.odom_bench run [--stage analyse]
uv run --with rosbags==0.11.5 python -m ros.tools.odom_bench freeze 0412 0413 --group 20fps
uv run --with rosbags==0.11.5 python -m ros.tools.odom_bench crab 0398 [0399 ...]
uv run python -m ros.tools.odom_bench truth TAPE --out truth.csv
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from pathlib import Path

from pepin.lidar import LidarMount

from .bench import bench_for, run
from .crab import report, report_many
from .drives import REC, REPO, SETS, DriveSet, entry, find, sha256
from .truth import cached_truths, write_truth

DEFAULT_SET = SETS / "0355-0411.json"
CACHE = Path(__file__).resolve().parent / ".cache"


def _run(a: argparse.Namespace) -> int:
    b = bench_for(a.set, a.rec, a.work, a.map, a.image, a.cpus, a.parallel)
    run(b, a.stage)
    return 0


def _freeze(a: argparse.Namespace) -> int:
    """Print set entries for new drives, their truth hashed (computed into the set's work dir)."""
    b = bench_for(a.set, a.rec, a.work, a.map, "", "", 1)
    if a.group not in b.drives.groups:
        raise SystemExit(f"--group {a.group}: not one of {list(b.drives.groups)}")
    entries = {n.zfill(4): entry(a.rec, n.zfill(4), a.group) for n in a.drives}
    outs = {n: b.work.runs / n / "truth.csv" for n in entries}
    cached_truths([(find(a.rec, n)[1], out) for n, out in outs.items()], b.truth_map, b.mount)
    for n, e in entries.items():
        e["truth_sha256"] = sha256(outs[n])
        print(f'    "{n}": {json.dumps(e)},')
    return 0


def _truth(a: argparse.Namespace) -> int:
    mount = LidarMount.from_json(REPO / "config" / "lidar.json")
    if a.lidar_yaw is not None:
        mount = dataclasses.replace(mount, yaw_offset_deg=a.lidar_yaw)
    seed = (a.seed_transform[0], a.seed_transform[1], math.radians(a.seed_transform[2]))
    print(write_truth(a.tape, a.map, mount, a.out, seed, a.min_fit), end="")
    return 0


def _crab(a: argparse.Namespace) -> int:
    ns = [n.zfill(4) for n in a.drives]
    if a.span and len(ns) > 1:
        raise SystemExit("--span is for one drive")
    mount = LidarMount.from_json(REPO / "config" / "lidar.json")
    truths = {n: a.work / "runs" / n / "truth.csv" for n in ns}
    cached_truths([(find(a.rec, n)[1], out) for n, out in truths.items()], a.map, mount)
    if len(ns) == 1:
        report(ns[0], a.rec, truths[ns[0]], tuple(a.span) if a.span else None)
    else:
        report_many(ns, a.rec, truths)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Parse and dispatch."""
    p = argparse.ArgumentParser(
        prog="python -m ros.tools.odom_bench", description=(__doc__ or "").split("\n")[0]
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    default_map = DriveSet.load(DEFAULT_SET).truth_map

    r = sub.add_parser("run", help="the bench on a frozen set: tables, replays, verdicts")
    r.add_argument(
        "--set", type=Path, default=DEFAULT_SET, help="the frozen drive set (sets/*.json)"
    )
    r.add_argument("--rec", type=Path, default=REC, help="where the drives are (ros/maps/rec)")
    r.add_argument("--work", type=Path, help="the work directory (default .cache/<set name>)")
    r.add_argument("--map", type=Path, help="the truth's map instead of the set's")
    r.add_argument("--stage", choices=["prepare", "analyse", "all"], default="all")
    r.add_argument("--image", default="pepin-laptop:vio", help="the replays' docker image")
    r.add_argument("--cpus", default="1.5", help="each replay container's CPU cap")
    r.add_argument("--parallel", type=int, default=5, help="replays at a time")
    r.set_defaults(fn=_run)

    f = sub.add_parser("freeze", help="set entries for new drives (bag, tape and truth hashed)")
    f.add_argument("drives", nargs="+")
    f.add_argument("--group", required=True, help="the drives' camera-rate group, e.g. 20fps")
    f.add_argument("--set", type=Path, default=DEFAULT_SET, help="the set they will join")
    f.add_argument("--rec", type=Path, default=REC)
    f.add_argument("--work", type=Path)
    f.add_argument("--map", type=Path)
    f.set_defaults(fn=_freeze)

    c = sub.add_parser("crab", help="the lidar yaw from straight driving: the truth's crab angle")
    c.add_argument("drives", nargs="+")
    c.add_argument("--rec", type=Path, default=REC)
    c.add_argument("--work", type=Path, default=CACHE / "crab", help="where the truths are kept")
    c.add_argument("--map", type=Path, default=default_map)
    c.add_argument("--span", type=float, nargs=2, metavar=("T0", "T1"), help="instead of the goal")
    c.set_defaults(fn=_crab)

    t = sub.add_parser("truth", help="one tape's lidar truth as CSV (stamp, x, y, yaw, fit, n)")
    t.add_argument("tape", type=Path)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--map", type=Path, default=default_map)
    t.add_argument("--lidar-yaw", type=float, help="the mount's yaw instead of config/lidar.json's")
    t.add_argument(
        "--seed-transform",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.0),
        metavar=("X", "Y", "YAW_DEG"),
        help="the tape's map frame into --map's",
    )
    t.add_argument("--min-fit", type=float, default=0.5, help="the inlier fraction a scan needs")
    t.set_defaults(fn=_truth)

    a = p.parse_args(argv)
    code: int = a.fn(a)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
