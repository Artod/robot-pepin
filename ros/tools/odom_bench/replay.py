"""The board's EKF replayed on a drive's own recorded inputs: ros/vio_replay.sh's arm-A command in
a throwaway container with no network, one arm's play exclusions, its /odometry/filtered out."""

from __future__ import annotations

import math
import os
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from vio_score import Trajectory

from .drives import REPO

EKF_YAML = REPO / "ros" / "params" / "ekf.yaml"
# vio_replay.sh's outputs minus /vo and /vo_twist: a drive bag since 0330 carries them, and the
# replay must make its own
OV_OUT = (
    "/odometry/filtered",
    "/vo/raw",
    "/ov_msckf/poseimu",
    "/ov_msckf/odomimu",
    "/ov_msckf/health",
    "/ov_msckf/points_msckf",
    "/ov_msckf/points_slam",
    "/vio/seed_twist",
)
VIO_IN = ("/vo", "/vo_twist")
INNER = r"""
set -eo pipefail
set -m
source /opt/ros/jazzy/setup.bash; source /ws/install/setup.bash
SIM=(--ros-args -p use_sim_time:=true)
pids=()
python3 /repo/ros/tools/head_static_tf.py "${SIM[@]}" & pids+=($!)
ros2 run robot_localization ekf_node "${SIM[@]}" --params-file /params/ekf.yaml \
    -p publish_tf:=false -r __node:=ekf_filter_node & pids+=($!)
ros2 bag record --storage mcap -o "$OUT.bag" /odometry/filtered --use-sim-time & REC_PID=$!
sleep 6
ros2 bag play -i "$DRIVE" --clock 100 --exclude-topics $EXCLUDE
sleep 3
kill -INT "$REC_PID"; wait "$REC_PID" || true
for p in "${pids[@]}"; do kill -INT -- "-$p" 2>/dev/null || true; done
for _ in $(seq 1 20); do jobs -r | grep -q . || break; sleep 0.5; done
for p in "${pids[@]}"; do kill -TERM -- "-$p" 2>/dev/null || true; done; wait || true
python3 /repo/ros/tools/bag_poses.py "$OUT.bag" /odometry/filtered > "$OUT.csv"
echo "wrote $OUT.bag and $OUT.csv"
"""


@dataclass(frozen=True)
class Arm:
    """A replay arm: whose bag copies it plays ("base" or its own), what it leaves out."""

    bags: str
    exclude: tuple[str, ...]
    about: str


def base_arms() -> dict[str, Arm]:
    """The arms on the base copies: A (all sources), A2 (A again: the replay's own noise), and the
    source combinations W, WG, WGR, WGV."""
    return {
        "A": Arm("base", (), "today's live configuration (base claims), all sources"),
        "A2": Arm("base", (), "A replayed again (the replay's own noise)"),
        "W": Arm("base", (*VIO_IN, "/imu/data_raw", "/odom_laser"), "wheels only"),
        "WG": Arm("base", (*VIO_IN, "/odom_laser"), "wheels + gyro"),
        "WGR": Arm("base", VIO_IN, "wheels + gyro + rf2o (= ros/vio_replay.sh --arm A)"),
        "WGV": Arm("base", ("/odom_laser",), "wheels + gyro + VIO"),
    }


@dataclass(frozen=True)
class ReplayJob:
    """One arm on one drive."""

    arm: str
    n: str
    bag: str
    bag_dir: Path
    out_dir: Path
    log: Path
    exclude: tuple[str, ...]
    image: str
    cpus: str


def done(out_dir: Path, n: str) -> bool:
    """The replay's CSV is there (and not an empty run's header)."""
    out = out_dir / f"{n}.csv"
    return out.exists() and out.stat().st_size > 1000


def replay_one(job: ReplayJob) -> str:
    """Run one replay (cached by its CSV)."""
    if done(job.out_dir, job.n):
        return f"{job.arm} {job.n} cached"
    job.out_dir.mkdir(parents=True, exist_ok=True)
    if (job.out_dir / f"{job.n}.bag").exists():
        shutil.rmtree(job.out_dir / f"{job.n}.bag")
    name = f"odombench-{job.arm.lower().replace('_', '-')}-{job.n}"
    subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    cmd = [
        "docker", "run", "--rm", "--network", "none", "--cpus", job.cpus, "--name", name,
        "-v", f"{REPO}:/repo:ro",
        "-v", f"{job.bag_dir}:/rec_in:ro",
        "-v", f"{job.out_dir}:/out",
        "-v", f"{EKF_YAML}:/params/ekf.yaml:ro",
        "-v", f"{REPO}/config:/ws/config:ro",
        "-v", f"{REPO}/src/pepin:/ws/pepin_src/pepin:ro",
        "-v", f"{REPO}/ros/pepin_bringup/pepin_bringup:"
              "/ws/install/pepin_bringup/lib/python3.12/site-packages/pepin_bringup:ro",
        "-e", "PYTHONDONTWRITEBYTECODE=1", "-e", "ROS_DOMAIN_ID=78",
        "-e", "ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST",
        "-e", "RMW_IMPLEMENTATION=rmw_cyclonedds_cpp",
        "-e", f"DRIVE=/rec_in/{job.bag}", "-e", f"OUT=/out/{job.n}",
        "-e", f"EXCLUDE={' '.join(OV_OUT + job.exclude)}",
        "--entrypoint", "/bin/bash", job.image, "-c", INNER,
    ]  # fmt: skip
    job.log.parent.mkdir(parents=True, exist_ok=True)
    t = time.time()
    with job.log.open("w") as fh:
        r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
    ok = done(job.out_dir, job.n)
    return (
        f"{job.arm} {job.n} exit {r.returncode} {'ok' if ok else 'NO CSV'} {time.time() - t:.0f} s"
    )


def replays(jobs: list[ReplayJob], parallel: int) -> list[str]:
    """Every job not yet done, ``parallel`` at a time; their log lines."""
    todo = [j for j in jobs if not done(j.out_dir, j.n)]
    lines = [f"{len(jobs)} jobs, {len(todo)} to run, {parallel} at a time"]
    with ThreadPoolExecutor(parallel) as ex:
        lines += list(ex.map(replay_one, todo))
    return lines


def check_vio_replay(n: str, bag: Path, rec_dir: Path, logs: Path, mine: Path) -> str:
    """ros/vio_replay.sh --arm A itself on drive n's original bag (copied to ``rec_dir``) against
    this bench's WGR arm (the same exclusions): the largest pose difference over the drive."""
    rec_dir.mkdir(parents=True, exist_ok=True)
    if not (rec_dir / bag.name).exists():
        shutil.copytree(bag, rec_dir / bag.name)
    out = rec_dir / f"{bag.name}_arm_A.csv"
    if not out.exists():
        logs.mkdir(parents=True, exist_ok=True)
        with (logs / f"vio_replay_{n}.log").open("w") as fh:
            subprocess.run(
                ["bash", str(REPO / "ros/vio_replay.sh"), n, "--arm", "A"],
                cwd=REPO,
                stdout=fh,
                stderr=subprocess.STDOUT,
                env={**os.environ, "PEPIN_REPLAY_REC": str(rec_dir)},
            )
    if not (out.exists() and mine.exists()):
        return f"vio_replay.sh check on {n}: missing output"
    a, b = Trajectory.load(out), Trajectory.load(mine)
    t = np.linspace(max(a.t[0], b.t[0]) + 1, min(a.t[-1], b.t[-1]) - 1, 500)
    dx = [
        np.nanmax(np.abs(np.interp(t, a.t, u) - np.interp(t, b.t, v)))
        for u, v in ((a.x, b.x), (a.y, b.y))
    ]
    dy = np.nanmax(np.abs(np.interp(t, a.t, a.yaw) - np.interp(t, b.t, b.yaw)))
    return (
        f"ros/vio_replay.sh {n} --arm A (main, unchanged) vs this bench's WGR on the same drive:"
        f" largest difference x {100 * dx[0]:.2f} cm, y {100 * dx[1]:.2f} cm,"
        f" yaw {math.degrees(dy):.3f} deg over the drive"
    )
