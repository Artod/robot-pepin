"""The replay stand: every recorded drive through Nav2's own costmaps with candidate parameters,
faster than real time, one fixed score per drive.

QUESTION: would these costmap parameters (or flags) have blocked the recorded drives more or less
than the ones that drove them — answered on the bags, before the robot drives?

METHOD (inside a throwaway container of the board's image, started by ros/replay.sh):
    1. the stand — the static grid and the named places — from RTAB-Map's saved database, or the
       one a baseline was scored on (``--against``, so a diff is always on the same map);
    2. each drive's bag prepared once (ros/replay/prepare.py; cached by bag, stand and version);
    3. costmap_replay (ros/replay/engine) steps the stock nav2_costmap_2d local and global
       costmaps through it at their live update rates in bag time, parameters from
       ros/params/nav2_params.yaml + nav2_map_from_laptop.yaml, then ``--params`` files, then
       ``--set`` values, then the stand's own override (update_frequency 0: this program is the
       loop);
    4. ros/replay/score.py turns the snapshots into the drive's row.
ANSWER: the table on stdout (and ``--save``: a JSON baseline with the stand, the parameters' hash
and the rows, plus the same table as .txt); ``--against BASELINE.json`` prints the delta per drive.

    ros/replay.sh 483-498                                  # the current parameters
    ros/replay.sh 483-498 --set both.camera_layer.enabled=false \
        --against ros/replay/baselines/0483-0498.json      # the delta per drive
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
# This checkout's pepin and pepin_bringup before the image's own copies (the entrypoint's
# workspace puts the image's first): the replay rebuilds inputs with the code under review.
for _path in (REPO / "ros" / "pepin_bringup", REPO / "src", HERE):
    sys.path.insert(0, str(_path))
import prepare as prep  # noqa: E402
import score  # noqa: E402

PARAMS = REPO / "ros" / "params"
REC = Path(os.environ.get("PEPIN_REPLAY_REC", "/rec"))
MAPS = Path(os.environ.get("PEPIN_REPLAY_MAPS", "/maps"))
CACHE = Path(os.environ.get("PEPIN_REPLAY_CACHE", "/cache"))
# The checkout and the working directory as the host sees them (ros/replay.sh): a path given on
# the command line is the host's, and the checkout is mounted at REPO.
HOST_REPO = os.environ.get("PEPIN_REPLAY_HOST_REPO")
HOST_CWD = os.environ.get("PEPIN_REPLAY_HOST_CWD")


def in_container(arg: str) -> Path:
    """A path from the host's command line as this process sees it (inside the checkout)."""
    if HOST_REPO is None or HOST_CWD is None:
        return Path(arg)
    host = Path(arg) if Path(arg).is_absolute() else Path(HOST_CWD) / arg
    host = Path(os.path.normpath(host))
    try:
        return REPO / host.relative_to(HOST_REPO)
    except ValueError:
        raise SystemExit(
            f"{arg}: only paths inside the checkout {HOST_REPO} reach the replay"
        ) from None


BASE_FILES = ("nav2_params.yaml", "nav2_map_from_laptop.yaml")
TREE = "pepin_nav_to_pose.xml"
COSTMAPS = {"local": "local_costmap", "global": "global_costmap"}
ENGINE = CACHE / "install" / "pepin_replay" / "lib" / "pepin_replay" / "costmap_replay"


def build_engine() -> None:
    """Build costmap_replay into the cache when a source is newer than the binary."""
    sources = [p for p in (HERE / "engine").rglob("*") if p.is_file()]
    if ENGINE.exists() and all(p.stat().st_mtime <= ENGINE.stat().st_mtime for p in sources):
        return
    print("building costmap_replay ...", flush=True)
    subprocess.run(
        [
            "colcon",
            "build",
            "--base-paths",
            str(HERE / "engine"),
            "--build-base",
            str(CACHE / "build"),
            "--install-base",
            str(CACHE / "install"),
            "--cmake-args",
            "-DCMAKE_BUILD_TYPE=Release",
        ],
        cwd=CACHE,
        check=True,
        stdout=subprocess.DEVNULL,
    )


def parse_value(text: str) -> Any:
    """A ``--set`` value as YAML reads it (true, 0.45, [a, b], /topic)."""
    import yaml

    return yaml.safe_load(text)


def candidate_params(sets: list[str]) -> dict[str, Any]:
    """``--set local.inflation_layer.inflation_radius=0.4`` (``local``, ``global`` or ``both``)
    as one ROS parameters document."""
    doc: dict[str, Any] = {}
    for item in sets:
        key, eq, value = item.partition("=")
        which, dot, name = key.partition(".")
        if not eq or not dot or not name or which not in ("local", "global", "both"):
            raise SystemExit(f"--set wants local|global|both.<parameter>=<value>, got {item}")
        for costmap in COSTMAPS.values() if which == "both" else [COSTMAPS[which]]:
            params = (
                doc.setdefault(costmap, {})
                .setdefault(costmap, {})
                .setdefault("ros__parameters", {})
            )
            params[name] = parse_value(value)
    return doc


def merged(files: list[Path], costmap: str, name: str) -> Any:
    """The value a parameter ends with over ``files`` in order (the last file that sets it)."""
    import yaml

    value = None
    for f in files:
        doc = yaml.safe_load(f.read_text()) or {}
        params = doc.get(costmap, {}).get(costmap, {}).get("ros__parameters", {})
        if name in params:
            value = params[name]
    return value


def params_hash(files: list[Path]) -> str:
    """A short hash of the parameter files' contents, in order."""
    h = hashlib.sha1()
    for f in files:
        h.update(f.read_bytes())
    return h.hexdigest()[:10]


def static_floor(stand: prep.Stand) -> dict[str, int]:
    """Per place, the static map's own occupied cells within the goal radius: what goal_max
    counts whatever the sensors do."""
    return {
        name: score.cells_near_point(
            np.where(stand.grid >= 100, score.LETHAL, 0).astype(np.uint8),
            stand.origin,
            stand.resolution,
            (goal[0], goal[1]),
            score.GOAL_RADIUS_M,
        )
        for name, goal in sorted(stand.goals.items())
    }


def prepare_key(bag: Path, stand: prep.Stand) -> str:
    """What a prepared drive depends on: the bag, the stand, the tree's clears and the tree's
    tick, the hull box and this checkout's prepare code — never the costmap parameters, so a
    candidate is replayed on the drives prepared once."""
    from pepin.footprint import hull_box

    source = next(bag.glob("*.mcap"))
    h = hashlib.sha1()
    for f in (
        PARAMS / TREE,
        HERE / "prepare.py",
        REPO / "ros/pepin_bringup/pepin_bringup/tof_bridge.py",
    ):
        h.update(f.read_bytes())
    h.update(repr(prep.bt_loop_s(PARAMS / BASE_FILES[0])).encode())
    h.update(json.dumps(hull_box(), sort_keys=True).encode())
    stat = source.stat()
    version = f"{prep.PREPARE_VERSION} {stand.fingerprint()} {h.hexdigest()[:12]}"
    return f"{version} {stat.st_size} {stat.st_mtime}"


def run_drive(job: dict[str, Any]) -> dict[str, Any]:
    """Prepare (cached), replay and score one drive; returns its row plus the engine's counts."""
    bag = Path(job["bag"])
    stand = prep.Stand.from_json(job["stand"])
    out = CACHE / "prepared" / f"{prep.run_number(bag):04d}"
    drive_json = out / "drive.json"
    key = prepare_key(bag, stand)
    stamp = out / "prepared.key"
    t0 = time.monotonic()
    if not (drive_json.exists() and stamp.exists() and stamp.read_text() == key):
        prep.prepare(bag, out, stand, PARAMS / TREE, PARAMS / BASE_FILES[0])
        stamp.write_text(key)
    t1 = time.monotonic()
    drive = json.loads(drive_json.read_text())
    snap = Path(job["workdir"]) / f"{drive['run']:04d}.snap"
    command = [
        str(ENGINE),
        "--bag",
        str(out / "bag"),
        "--out",
        str(snap),
        "--local-hz",
        str(job["local_hz"]),
        "--global-hz",
        str(job["global_hz"]),
        "--clears",
        job["clears"],
        "--ros-args",
        "--log-level",
        "warn",
    ]
    for f in job["params"]:
        command += ["--params-file", f]
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise RuntimeError(f"run {drive['run']}: costmap_replay failed:\n{done.stderr[-3000:]}")
    counts = json.loads(done.stdout.strip().splitlines()[-1])
    t2 = time.monotonic()
    recorded = dict(np.load(out / "recorded_local.npz"))
    row = score.score_drive(
        drive,
        score.read_snapshots(snap),
        recorded,
        1.0 / job["local_hz"],
        clears_applied=job["clears"] == "on",
    )
    t3 = time.monotonic()
    if not job["keep"]:
        snap.unlink()
    bag_s = float(drive["bag_end"]) - float(drive["bag_start"])
    row["wall_s"] = round(t3 - t1, 2)
    row["x_rt"] = round(bag_s / max(t3 - t1, 1e-6), 1)
    row["_engine"] = counts
    row["_times"] = {
        "prepare_s": round(t1 - t0, 2),
        "replay_s": round(t2 - t1, 2),
        "score_s": round(t3 - t2, 2),
    }
    row["_goal_xy"] = drive.get("goal")
    return row


def main(argv: list[str] | None = None) -> int:
    """Entry point: parse, run every drive, print the table (and the diff, and save)."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("runs", nargs="+", help="run numbers or ranges: 483-498 500")
    ap.add_argument(
        "--params", action="append", default=[], help="extra ROS params file (repo path)"
    )
    ap.add_argument(
        "--set", action="append", default=[], dest="sets", help="local|global|both.<param>=<value>"
    )
    ap.add_argument("--against", help="baseline JSON to diff against (its stand is used)")
    ap.add_argument(
        "--save", help="write the rows as a baseline: PATH.json and PATH.txt (ros/replay.sh copies)"
    )
    ap.add_argument(
        "--clears",
        choices=("bt", "none"),
        default="bt",
        help="emulate the tree's ClearEntireCostmap",
    )
    ap.add_argument("--db", help="database file in ros/maps for the stand (default rtabmap.db)")
    ap.add_argument("--jobs", type=int, default=6, help="drives replayed at once")
    ap.add_argument("--keep", action="store_true", help="keep the snapshot files in the cache")
    args = ap.parse_args(argv)

    baseline = json.loads(in_container(args.against).read_text()) if args.against else None
    if baseline is not None and not args.db:
        stand = prep.Stand.from_json(baseline["stand"])
    else:
        db = MAPS / (args.db or "rtabmap.db")
        stand = prep.stand_from_db(db, MAPS / "rtabmap.places.json")
    tag = time.strftime("%Y%m%d_%H%M%S")
    workdir = CACHE / "runs" / tag
    workdir.mkdir(parents=True, exist_ok=True)
    files = [PARAMS / f for f in BASE_FILES] + [in_container(p) for p in args.params]
    candidate = candidate_params(args.sets)
    if candidate:
        import yaml

        (workdir / "candidate.yaml").write_text(yaml.safe_dump(candidate))
        files.append(workdir / "candidate.yaml")
    hz = {
        name: float(merged(files, costmap, "update_frequency") or 0.0)
        for name, costmap in COSTMAPS.items()
    }
    if min(hz.values()) <= 0.0:
        raise SystemExit(f"both costmaps need an update_frequency above 0 to replay, got {hz}")
    override = {
        costmap: {costmap: {"ros__parameters": {"update_frequency": 0.0}}}
        for costmap in COSTMAPS.values()
    }
    (workdir / "stand_override.yaml").write_text(json.dumps(override))
    build_engine()
    jobs = [
        {
            "bag": str(bag),
            "stand": stand.to_json(),
            "workdir": str(workdir),
            "local_hz": hz["local"],
            "global_hz": hz["global"],
            "clears": "on" if args.clears == "bt" else "off",
            "params": [str(f) for f in [*files, workdir / "stand_override.yaml"]],
            "keep": args.keep,
        }
        for bag in prep.bag_dirs(REC, prep.parse_runs(args.runs))
    ]
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        rows = sorted(pool.map(run_drive, jobs), key=lambda r: r["run"])
    total_s = time.monotonic() - started
    goals_s = sum(float(r["active_s"]) for r in rows)
    floors = ", ".join(f"{name} {n}" for name, n in static_floor(stand).items())
    layered = ", ".join(f.name for f in files) + ("; " + " ".join(args.sets) if args.sets else "")
    summary = (
        f"{len(rows)} drives, {goals_s:.0f} s of goals replayed in {total_s:.1f} s wall"
        f" ({args.jobs} jobs); local {hz['local']:g} Hz, global {hz['global']:g} Hz;"
        f" clears {args.clears}; params {params_hash(files)} ({layered})\n"
        f"stand {stand.fingerprint()} ({stand.source}); goal_max floor, the map's own: {floors}"
    )
    print(score.table(rows))
    print(summary)
    if baseline is not None:
        if (
            baseline["stand"]
            and prep.Stand.from_json(baseline["stand"]).fingerprint() != stand.fingerprint()
        ):
            print(
                "WARNING: the stand differs from the baseline's: the delta mixes map and parameters"
            )
        print(f"\nagainst {args.against} (candidate minus baseline):")
        print(score.diff_table(rows, baseline["rows"]))
    if args.save:
        target = CACHE / "last.json"
        record = {
            "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "git": os.environ.get("PEPIN_REPLAY_GIT", "unknown"),
            "params": [
                str(f.relative_to(REPO)) if f.is_relative_to(REPO) else f.name for f in files
            ],
            "params_hash": params_hash(files),
            "sets": args.sets,
            "clears": args.clears,
            "local_hz": hz["local"],
            "global_hz": hz["global"],
            "total_wall_s": round(total_s, 1),
            "jobs": args.jobs,
            "stand": stand.to_json(),
            "rows": rows,
        }
        target.write_text(json.dumps(record, indent=1) + "\n")
        command = f"ros/replay.sh {' '.join(argv if argv is not None else sys.argv[1:])}"
        header = f"{command}  ({record['git']}, {record['created']})"
        target.with_suffix(".txt").write_text(f"{header}\n{score.table(rows)}\n{summary}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
