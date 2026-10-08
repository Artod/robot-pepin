"""The lidar mount's yaw read from straight driving: the crab angle of the lidar truth.

A truth whose base_link is turned by a constant angle against the cart's real direction of travel
moves sideways when the cart drives straight: the crab angle atan2(vy, vx) of its own body
velocity. Straight samples every 0.1 s of the goal span: cart gyro |w| < 0.05 rad/s and wheels
vx > 0.08 m/s; the truth's body velocity is a +-0.25 s linear fit rotated by its own yaw.
Reversing samples (vx < -0.08) are printed apart and never enter the suggestion. rf2o is
/odom_laser's twist (|vx| >= 0.05), the VIO /vo_twist (cov[0] < 1e5), each gated on the gyro and
wheels at its own stamp. One drive's interval: the median's 95 % bootstrap over 1 s blocks of
consecutive samples; several drives pool by resampling drives (seed 1).

Sign: + = the truth moves to the LEFT of its own x axis. The tape's scans are placed at robot =
-a - yaw (pepin.recording.scan_record_from_ros); a mount really at yaw Y read with yaw Y0 puts the
matched base_link (Y - Y0) clockwise of the real one, and straight travel reads a crab of
+(Y - Y0). So Y = Y0 + crab, Y0 the yaw the tape was converted with (read back from the bag's raw
scan, printed beside config/lidar.json's).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from .bags import stamp, typestore
from .drives import REPO, find, goal_span
from .truth import load_truth

Array = npt.NDArray[np.float64]
Sample = tuple[float, float, float]  # stamp, crab deg, wheel speed
CONFIG = REPO / "config" / "lidar.json"
W_MAX, V_MIN = 0.05, 0.08
STREAMS = ("/imu/data_raw", "/odom", "/odom_laser", "/vo_twist")


def truth_vel(tr: Array, t: float, half: float = 0.25) -> tuple[float, float, float] | None:
    """Body (vx, vy, wz) of the truth at t: a linear fit of x, y, yaw within +-half s, rotated by
    the yaw at t; None with under 4 scans or a one-sided window."""
    k = np.abs(tr[:, 0] - t) <= half
    if k.sum() < 4:
        return None
    tt = tr[k, 0] - t
    if tt.min() > -0.6 * half or tt.max() < 0.6 * half:
        return None
    sl = [float(np.polyfit(tt, tr[k, c], 1)[0]) for c in (1, 2, 3)]
    yaw = float(np.interp(t, tr[:, 0], tr[:, 3]))
    c, s = math.cos(yaw), math.sin(yaw)
    return c * sl[0] + s * sl[1], -s * sl[0] + c * sl[1], sl[2]


def interp(a: Array, col: int, t: float) -> float:
    """Column ``col`` of rows ``a`` (stamp first) at t."""
    a = a[np.argsort(a[:, 0])]
    return float(np.interp(t, a[:, 0], a[:, col]))


def read_bag(bag: Path) -> tuple[dict[str, Array], float | None]:
    """The four streams (header stamp first) and the first raw scan's angle_min."""
    from rosbags.highlevel import AnyReader

    out: dict[str, list[list[float]]] = {k: [] for k in STREAMS}
    raw_a0: float | None = None
    with AnyReader([bag], default_typestore=typestore()) as r:
        conns = [c for c in r.connections if c.topic in out or c.topic == "/ldlidar_node/scan"]
        for c, t, raw in r.messages(connections=conns):
            m = r.deserialize(raw, c.msgtype)
            s = stamp(m.header)
            if c.topic == "/imu/data_raw":
                w = m.angular_velocity
                out[c.topic].append([s, t * 1e-9, w.x, w.y, w.z])
            elif c.topic in ("/odom", "/odom_laser"):
                tw = m.twist.twist
                out[c.topic].append(
                    [s, t * 1e-9, math.nan, math.nan, math.nan, tw.linear.x, tw.linear.y]
                )
            elif c.topic == "/vo_twist":
                tw = m.twist.twist
                out[c.topic].append(
                    [s, t * 1e-9, tw.linear.x, tw.linear.y, tw.angular.z, m.twist.covariance[0]]
                )
            elif raw_a0 is None:
                raw_a0 = float(m.angle_min)
    width = {"/vo_twist": 6, "/imu/data_raw": 5}
    arr: dict[str, Array] = {
        k: np.array(v, dtype=float).reshape(-1, width.get(k, 7)) for k, v in out.items()
    }
    return arr, raw_a0


def tape_yaw(tape: Path, raw_angle_min: float | None) -> float | None:
    """The yaw_offset_deg the tape was converted with: its first scan's first angle =
    -angle_min - yaw."""
    if raw_angle_min is None:
        return None
    with tape.open() as f:
        for line in f:
            if '"scan"' in line[:80]:
                r = json.loads(line)
                if r.get("topic") == "scan":
                    return math.degrees((-raw_angle_min - r["angles"][0]) % (2 * math.pi))
    return None


def samples(t0: float, t1: float, tr: Array, x: dict[str, Array]) -> dict[str, list[Sample]]:
    """The straight samples of one drive, by source: truth (forward), truth rev, rf2o, vio."""
    src: dict[str, list[Sample]] = {k: [] for k in ("truth", "truth rev", "rf2o", "vio")}
    imu, odom = x["/imu/data_raw"], x["/odom"]
    for t in np.arange(t0, t1, 0.1).tolist():
        w, v = interp(imu, 4, t), interp(odom, 5, t)
        if abs(w) > W_MAX or abs(v) < V_MIN:
            continue
        tv = truth_vel(tr, t)
        if tv is None:
            continue
        sgn = 1.0 if v > 0 else -1.0
        src["truth" if v > 0 else "truth rev"].append(
            (float(t), math.degrees(math.atan2(sgn * tv[1], sgn * tv[0])), abs(v))
        )
    vt = x["/vo_twist"]
    if len(vt):
        vt = vt[(vt[:, 0] >= t0) & (vt[:, 0] <= t1) & (vt[:, 5] < 1e5)]
    for row in vt:
        w, v = interp(imu, 4, row[0] - 0.05), interp(odom, 5, row[0] - 0.05)
        if abs(w) > W_MAX or v < V_MIN:
            continue
        src["vio"].append((float(row[0]), math.degrees(math.atan2(row[3], row[2])), v))
    lz = x["/odom_laser"]
    lz = lz[(lz[:, 0] >= t0) & (lz[:, 0] <= t1)]
    for row in lz:
        w, v = interp(imu, 4, row[0]), interp(odom, 5, row[0])
        if abs(w) > W_MAX or v < V_MIN or abs(row[5]) < 0.05:
            continue
        src["rf2o"].append((float(row[0]), math.degrees(math.atan2(row[6], row[5])), v))
    return src


def block_ci(rows: list[Sample], block_s: float = 1.0) -> tuple[float, float, float]:
    """Median and its 95 % bootstrap interval over blocks of block_s of consecutive samples."""
    t = np.array([r[0] for r in rows])
    v = np.array([r[1] for r in rows])
    keys = np.floor((t - t[0]) / block_s).astype(int)
    blocks = [v[keys == k] for k in np.unique(keys)]
    rng = np.random.default_rng(1)
    bs = [
        np.median(np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))]))
        for _ in range(1000)
    ]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(np.median(v)), float(lo), float(hi)


def pooled(by: dict[str, list[float]]) -> str:
    """The pooled median over drives with its 95 % interval by resampling drives."""
    keys = [k for k, v in by.items() if v]
    allv = np.concatenate([by[k] for k in keys])
    rng = np.random.default_rng(1)
    bs = [
        np.median(np.concatenate([by[k] for k in rng.choice(keys, len(keys))])) for _ in range(1000)
    ]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    per = ", ".join(f"{k} {np.median(by[k]):+.1f}" for k in sorted(keys))
    return (
        f"{np.median(allv):+.2f} deg [{lo:+.2f}, {hi:+.2f}] n {len(allv)} / {len(keys)} drives;"
        f" per drive {per}"
    )


def fmt(rows: list[Sample]) -> str:
    """A source's median and interval, or 'no samples'."""
    if not rows:
        return "no samples"
    med, lo, hi = block_ci(rows)
    return f"{med:+.2f} deg [{lo:+.2f}, {hi:+.2f}]  n {len(rows)}"


def one(
    n: str, rec: Path, truth_csv: Path, given: tuple[float, float] | None
) -> tuple[dict[str, list[Sample]], dict[str, Any]]:
    """Drive n's straight samples and what the report needs (bag, tape, span, the tape's yaw)."""
    bag, tape = find(rec, n)
    t0, t1 = given or goal_span(tape)
    tr = load_truth(truth_csv)
    x, a0 = read_bag(bag)
    meta = {"bag": bag, "t0": t0, "t1": t1, "tape_yaw": tape_yaw(tape, a0), "tr": tr}
    return samples(t0, t1, tr, x), meta


def report(n: str, rec: Path, truth_csv: Path, given: tuple[float, float] | None) -> None:
    """One drive: each source's crab, the truth's displacement over the span, the suggestion."""
    src, m = one(n, rec, truth_csv, given)
    cfg = float(json.loads(CONFIG.read_text())["yaw_offset_deg"])
    y0 = m["tape_yaw"] if m["tape_yaw"] is not None else cfg
    fw, rv = src["truth"], src["truth rev"]
    metres = 0.1 * sum(r[2] for r in fw)
    print(
        f"drive {n}  {m['bag'].name}  goal span {m['t1'] - m['t0']:.1f} s; truth {len(m['tr'])}"
        f" matched scans ({truth_csv.parent / 'truth.txt'})"
    )
    print(
        f"straight forward: {len(fw)} samples (0.1 s), {metres:.2f} m by the wheels   (gyro |w| <"
        f" {W_MAX} rad/s, wheels vx > {V_MIN} m/s)"
    )
    print(
        "crab atan2(vy, vx), + = moving left of its own x axis; median [95 % interval, 1 s block"
        " bootstrap]"
    )
    print(f"  truth     : {fmt(fw)}")
    print(f"  rf2o      : {fmt(src['rf2o'])}")
    print(f"  vio       : {fmt(src['vio']) if src['vio'] else 'no /vo_twist in the bag'}")
    print(f"  (reversing, apart: truth {fmt(rv)})")
    k = (m["tr"][:, 0] >= m["t0"]) & (m["tr"][:, 0] <= m["t1"])
    if k.sum() >= 2:
        a, b = m["tr"][k][0], m["tr"][k][-1]
        dx, dy, dh = b[1] - a[1], b[2] - a[2], b[3] - a[3]
        mid = a[3] + dh / 2
        along = math.cos(a[3]) * dx + math.sin(a[3]) * dy
        across = -math.sin(a[3]) * dx + math.cos(a[3]) * dy
        chord = math.atan2(dy, dx)
        print(
            f"truth over the span ({b[0] - a[0]:.1f} s): heading {math.degrees(a[3]):+.1f} ->"
            f" {math.degrees(b[3]):+.1f} deg (change {math.degrees(dh):+.2f}); displacement"
            f" {math.hypot(dx, dy):.2f} m, along the start heading {along:+.2f} m, sideways"
            f" {across:+.3f} m (+ left); chord angle vs start heading"
            f" {math.degrees(chord - a[3]):+.2f} deg, vs mean heading"
            f" {math.degrees(chord - mid):+.2f} deg (an arc puts the chord at half the heading"
            " change from the start heading and on the mean one; a frame error puts it off the"
            " mean heading by the crab)"
        )
    if not fw:
        print("no straight forward samples: no suggestion")
        return
    crab = block_ci(fw)
    note = (
        f"tape converted at {y0:.2f}"
        if m["tape_yaw"] is not None
        else "tape yaw unread, config assumed"
    )
    if abs(y0 - cfg) > 0.05:
        note += f" (NOT config's {cfg:g}: the suggestion is from the tape's)"
    print(
        f"config/lidar.json yaw_offset_deg {cfg:g} -> suggested {y0 + crab[0]:.1f} "
        f" [{y0 + crab[1]:.1f}, {y0 + crab[2]:.1f}]  (yaw = tape yaw + crab; {note})"
    )


def report_many(ns: list[str], rec: Path, truths: dict[str, Path]) -> None:
    """Several drives: one line each (tape spans), then each source pooled over the drives."""
    by: dict[str, dict[str, list[float]]] = {k: {} for k in ("truth", "truth rev", "rf2o", "vio")}
    for n in ns:
        src, m = one(n, rec, truths[n], None)
        for k, store in by.items():
            if src[k]:
                store[n] = [r[1] for r in src[k]]
        ty = "unread" if m["tape_yaw"] is None else f"{m['tape_yaw']:.2f}"
        print(
            f"  {n}: truth {fmt(src['truth'])}; rf2o {fmt(src['rf2o'])}; vio {fmt(src['vio'])};"
            f" tape yaw {ty}"
        )
    for k, store in by.items():
        if store:
            print(f"  pooled {k:9s}: {pooled(store)}")
