"""The speed sources of one drive on its goal span, and the windows the hat and the scores use.

V = /vo_twist's full twists (cov[0] < 1e5), each step from the previous /ov_msckf/poseimu to its
stamp; W = /odom (step = the previous message, < 0.2 s); R = /odom_laser (rf2o, < 0.5 s); G =
/imu/data_raw's z (< 0.05 s); T = the lidar truth's +-0.5 s fit at W's moving steps. Moving: the
wheels' mean |vx| >= 3 cm/s or the rest-bias-corrected gyro mean >= 3 deg/s over the step.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from vio_score import Trajectory, relative, segments

from .rewrite import ODOM_REST
from .truth import fit_vw, load_truth

Array = npt.NDArray[np.float64]
Pose = tuple[float, float, float]
GYRO_MOVE = math.radians(3.0)
TRIPLE = {"vx": ("V", "W", "R"), "wz": ("V", "W", "G")}  # the hat's three per quantity
LENS = (1.0, 2.0, 5.0)  # s, the NEES windows
LAGS = np.arange(-300, 301, 10)  # ms, the stamp offsets searched
PAIRS = (("V", "vx"), ("R", "vx"), ("T", "vx"), ("G", "wz"), ("Vw", "wz"))  # against the wheels


class Src:
    """One source's samples: steps [a, b], values and claimed sigmas per key."""

    def __init__(self, a: Array, b: Array, vals: dict[str, Array], sds: dict[str, Array]) -> None:
        o = np.argsort(a)
        self.a, self.b = a[o], b[o]
        self.v = {k: x[o] for k, x in vals.items()}
        self.s = {k: x[o] for k, x in sds.items()}

    def _clip(self, a: float, b: float) -> tuple[npt.NDArray[np.int64], Array]:
        i0 = max(int(np.searchsorted(self.b, a, "right")) - 1, 0)
        i1 = int(np.searchsorted(self.a, b, "left"))
        idx = np.arange(i0, i1)
        if not len(idx):
            return idx, idx.astype(float)
        idx = idx[(self.b[idx] > a) & (self.a[idx] < b)]
        ov = np.minimum(self.b[idx], b) - np.maximum(self.a[idx], a)
        ok = ov > 0
        return idx[ok], ov[ok]

    def over(self, key: str, a: float, b: float, need: float) -> tuple[float, float] | None:
        """The overlap-weighted mean over [a, b] and its claimed sigma as if the samples were
        independent; None if less than ``need`` s is covered."""
        idx, ov = self._clip(a, b)
        c = float(ov.sum())
        if not len(idx) or c < need:
            return None
        v, s = self.v[key][idx], self.s[key][idx] if key in self.s else np.zeros(len(idx))
        return float(v @ ov) / c, math.sqrt(float(np.sum((s * ov) ** 2))) / c

    def over_set(
        self, key: str, pieces: list[tuple[float, float]], need: float
    ) -> tuple[float, float] | None:
        """``over`` on a set of disjoint intervals (a sample spanning two counts once, with its
        total overlap); None if less than ``need`` of the set's length is covered."""
        acc: dict[int, float] = defaultdict(float)
        for a, b in pieces:
            idx, ov = self._clip(a, b)
            for i, o in zip(idx.tolist(), ov.tolist(), strict=True):
                acc[i] += o
        total = sum(b - a for a, b in pieces)
        if not acc:
            return None
        idx = np.fromiter(acc.keys(), int)
        ov = np.fromiter(acc.values(), float)
        c = float(ov.sum())
        if c < need * total:
            return None
        s = self.s[key][idx] if key in self.s else np.zeros(len(idx))
        return float(self.v[key][idx] @ ov) / c, math.sqrt(float(np.sum((s * ov) ** 2))) / c

    def pieces(self, a: float, b: float) -> list[tuple[float, float]]:
        """This source's steps clipped to [a, b] (its covered time set)."""
        idx, _ = self._clip(a, b)
        return [(max(float(self.a[i]), a), min(float(self.b[i]), b)) for i in idx.tolist()]

    def integ(self, a: float, b: float, need: float) -> Pose | None:
        """The source's own vx, vy, wz integrated over [a, b] into a relative pose, the covered
        part scaled to the window; None if less than ``need`` of it is covered."""
        idx, ov = self._clip(a, b)
        c = float(ov.sum())
        if not len(idx) or c < need * (b - a):
            return None
        wz = self.v["wz"][idx] * ov
        th = np.cumsum(wz) - 0.5 * wz
        vx = self.v.get("vx", np.zeros(len(self.a)))[idx] * ov
        vy = self.v.get("vy", np.zeros(len(self.a)))[idx] * ov
        k = (b - a) / c
        return (
            float(np.sum(vx * np.cos(th) - vy * np.sin(th))) * k,
            float(np.sum(vx * np.sin(th) + vy * np.cos(th))) * k,
            float(wz.sum()) * k,
        )


@dataclass
class Sources:
    """One drive's sources on the goal span: moving samples (V W R G, Rs, and T) and every
    sample (the displacement domain)."""

    n: str
    span: tuple[float, float]
    f: dict[str, Array]
    tr: Array
    kept: int
    scans: int
    mov: dict[str, Src]
    allv: dict[str, Src]


@dataclass(frozen=True)
class DriveJob:
    """What a worker needs to read one drive."""

    n: str
    run_dir: Path
    tape: Path
    span: tuple[float, float]
    heading_deg: float
    shifts: dict[str, float]


def load_sources(job: DriveJob) -> Sources:
    """One drive's sources; ``shifts`` (s) are subtracted from a source's stamps (R aligned to
    the wheels' clock, T the truth's), as measured by the offsets stage."""
    shifts = job.shifts
    f = dict(np.load(job.run_dir / "facts.npz"))
    meta = json.loads((job.run_dir / "meta.json").read_text())
    t0, t1 = job.span
    tr = load_truth(job.run_dir / "truth.csv", job.heading_deg)
    kept = int(((tr[:, 0] >= t0) & (tr[:, 0] <= t1)).sum())
    tr[:, 0] -= shifts["T"]
    scans = 0
    with job.tape.open() as fh:
        for line in fh:
            if '"topic":"scan"' in line[:60]:
                scans += t0 <= json.loads(line)["t"] <= t1
    imu = f["/imu/data_raw"][np.argsort(f["/imu/data_raw"][:, 0])]
    odom = f["/odom"][np.argsort(f["/odom"][:, 0])]
    rest = np.zeros(len(imu), bool)
    for z in f.get("/zupt", np.zeros((0, 2)))[:, 0]:
        rest |= np.abs(imu[:, 0] - z) < 0.1
    bias = float(np.mean(imu[rest, 4])) if rest.sum() > 20 else 0.0
    gsd = math.sqrt(meta["gyro_var"])
    ci = np.concatenate([[0.0], np.cumsum(imu[:, 4])])
    co = np.concatenate([[0.0], np.cumsum(odom[:, 5])])

    def moving(a: Array, b: Array) -> npt.NDArray[np.bool_]:
        """Wheels' mean |vx| >= 3 cm/s or rest-bias-corrected gyro mean >= 3 deg/s over (a, b];
        a step shorter than 30 ms is judged over the 30 ms around its middle."""
        mid, half = 0.5 * (a + b), np.maximum(0.5 * (b - a), 0.015)
        a, b = mid - half, mid + half
        ia0, ia1 = np.searchsorted(imu[:, 0], a, "right"), np.searchsorted(imu[:, 0], b, "right")
        io0, io1 = np.searchsorted(odom[:, 0], a, "right"), np.searchsorted(odom[:, 0], b, "right")
        ki, ko = ia1 - ia0, io1 - io0
        ok = (ki >= 1) & (ko >= 1)
        g = np.where(ok, (ci[ia1] - ci[ia0]) / np.maximum(ki, 1), 0.0) - bias
        vw = np.where(ok, (co[io1] - co[io0]) / np.maximum(ko, 1), 0.0)
        out: npt.NDArray[np.bool_] = ok & ((np.abs(vw) >= 0.03) | (np.abs(g) >= GYRO_MOVE))
        return out

    src: dict[str, tuple[Array, Array, dict[str, Array], dict[str, Array]]] = {}
    vt = f.get("/vo_twist", np.zeros((0, 8)))
    vt = vt[(vt[:, 0] >= t0) & (vt[:, 0] <= t1) & (vt[:, 5] < 1e5)] if len(vt) else vt
    if len(vt) and "/ov_msckf/poseimu" in f:
        poses = np.sort(f["/ov_msckf/poseimu"][:, 0])
        i = np.searchsorted(poses, vt[:, 0] - 1e-6) - 1
        ok = i >= 0
        start = np.where(ok, poses[np.maximum(i, 0)], np.nan)
        dt = vt[:, 0] - start
        k = ok & (dt > 0) & (dt < 0.5)
        r = vt[k]
        src["V"] = (
            start[k],
            r[:, 0],
            {"vx": r[:, 2], "vy": r[:, 3], "wz": r[:, 4]},
            {"vx": np.sqrt(r[:, 5]), "wz": np.sqrt(r[:, 7])},
        )
    lz = f["/odom_laser"][np.argsort(f["/odom_laser"][:, 0])]
    for name, arr, cap in (("R", lz, 0.5), ("W", odom, 0.2)):
        a, b = arr[:-1], arr[1:]
        k = (b[:, 0] >= t0) & (b[:, 0] <= t1) & (b[:, 0] - a[:, 0] > 0) & (b[:, 0] - a[:, 0] < cap)
        src[name] = (
            a[k, 0],
            b[k, 0],
            {"vx": b[k, 5], "vy": b[k, 6], "wz": b[k, 7]},
            {"vx": np.sqrt(b[k, 8]), "wz": np.sqrt(b[k, 10])},
        )
    a, b = imu[:-1], imu[1:]
    k = (b[:, 0] >= t0) & (b[:, 0] <= t1) & (b[:, 0] - a[:, 0] > 0) & (b[:, 0] - a[:, 0] < 0.05)
    src["G"] = (a[k, 0], b[k, 0], {"wz": b[k, 4]}, {"wz": np.full(int(k.sum()), gsd)})
    mov, allv = {}, {}
    if "R" in src and shifts.get("R"):
        src["Rs"] = src["R"]  # rf2o as stamped (what the EKF fuses), beside the aligned R
    for name, (a, b, vals, sds) in src.items():
        a, b = a - shifts.get(name, 0.0), b - shifts.get(name, 0.0)
        allv[name] = Src(a, b, vals, sds)
        m = moving(a, b)
        mov[name] = Src(
            a[m], b[m], {kk: x[m] for kk, x in vals.items()}, {kk: x[m] for kk, x in sds.items()}
        )
    w = mov["W"]
    fits = [fit_vw(tr, 0.5 * (x + y)) for x, y in zip(w.a, w.b, strict=True)]
    okf = np.array([x is not None for x in fits], bool)
    fv = np.array([x if x is not None else (np.nan, np.nan) for x in fits]).reshape(-1, 2)
    mov["T"] = Src(w.a[okf], w.b[okf], {"vx": fv[okf, 0], "wz": fv[okf, 1]}, {})
    return Sources(job.n, job.span, f, tr, kept, scans, mov, allv)


def windows(d: Sources, kind: str) -> dict[str, Array]:
    """Per window length L: rows of (V, sV, W, sW, X3, sX3, T, Rs, sRs) on the L-s grid.

    A window's time set is the VIO's moving steps clipped to it (>= 0.5 L); the other two and the
    truth are averaged over exactly that set (>= 90 % covered), so the three estimate the same
    quantity and a velocity change inside the window is nobody's error; T NaN without a fit.
    'ps_<S>': per-sample rows on S's own steps (the others over the step, >= 80 % covered; T the
    fit at the step's middle).
    """
    s1, s2, s3 = TRIPLE[kind]
    out: dict[str, Array] = {}
    if s1 not in d.mov or not len(d.mov[s1].a):
        return out
    t0, t1 = d.span
    for length in LENS:
        rows = []
        for k in np.arange(math.floor(t0 / length) * length, t1, length).tolist():
            pcs = d.mov[s1].pieces(k, k + length)
            if sum(b - a for a, b in pcs) < 0.5 * length:
                continue
            m = [d.mov[s].over_set(kind, pcs, 0.999 if s == s1 else 0.9) for s in (s1, s2, s3)]
            m0, m1, m2 = m
            if m0 is None or m1 is None or m2 is None:
                continue
            t = d.mov["T"].over_set(kind, pcs, 0.9)
            rs = d.mov["Rs"].over_set(kind, pcs, 0.9) if kind == "vx" and "Rs" in d.mov else None
            tv = t[0] if t else np.nan
            rv, rsd = (rs[0], rs[1]) if rs else (np.nan, np.nan)
            rows.append([m0[0], m0[1], m1[0], m1[1], m2[0], m2[1], tv, rv, rsd])
        out[f"w{length:g}"] = np.array(rows).reshape(-1, 9)
    for own in (s1, s2, s3):
        x = d.mov[own]
        others = [s for s in (s1, s2, s3) if s != own]
        rows = []
        steps = zip(x.a.tolist(), x.b.tolist(), x.v[kind].tolist(), x.s[kind].tolist(), strict=True)
        for a, b, v, s in steps:
            o0 = d.mov[others[0]].over(kind, a, b, 0.8 * (b - a))
            o1 = d.mov[others[1]].over(kind, a, b, 0.8 * (b - a))
            if o0 is None or o1 is None:
                continue
            ft = fit_vw(d.tr, 0.5 * (a + b))
            vals = {own: (v, s), others[0]: o0, others[1]: o1}
            tv = (ft[0] if kind == "vx" else ft[1]) if ft else np.nan
            v1, v2, v3 = vals[s1], vals[s2], vals[s3]
            rows.append([v1[0], v1[1], v2[0], v2[1], v3[0], v3[1], tv, np.nan, np.nan])
        out[f"ps_{own}"] = np.array(rows).reshape(-1, 9)
    return out


def acf_bins(d: Sources, kind: str, dt: float = 0.1) -> Array:
    """0.1 s bins over the span: each triple source's mean over the bin from its moving samples
    (NaN if under 80 % covered)."""
    t0, t1 = d.span
    grid = np.arange(t0, t1 - dt, dt)
    out = np.full((len(grid), 3), np.nan)
    for j, s in enumerate(TRIPLE[kind]):
        if s not in d.mov:
            continue
        for i, g in enumerate(grid.tolist()):
            m = d.mov[s].over(kind, g, g + dt, 0.8 * dt)
            if m is not None:
                out[i, j] = m[0]
    return out


def smooth(v: Array, k: int = 5) -> Array:
    """Centred moving average over k samples (edges: shorter windows)."""
    h = k // 2
    return np.array([v[max(0, i - h) : i + h + 1].mean() for i in range(len(v))])


def disp_windows(d: Sources) -> dict[str, list[dict[str, Any]]]:
    """Displacement-domain windows, each with the truth's relative pose (start frame), its path
    length and every source's own integrated relative pose (None if under 80 % covered).

    's1': moving whole seconds [k, k + 1) (the wheels' moving samples cover >= 0.5 s; truth scans
    within 0.15 s of both ends); 'm1': 1 m pieces of the 5-scan-smoothed truth path.
    """
    t0, t1 = d.span
    tr = d.tr[(d.tr[:, 0] >= t0) & (d.tr[:, 0] <= t1)]
    tru = Trajectory(tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3])
    out: dict[str, list[dict[str, Any]]] = {"s1": [], "m1": []}

    def srcs(a: float, b: float) -> dict[str, Pose | None]:
        return {
            s: (d.allv[s].integ(a, b, 0.8) if s in d.allv else None) for s in ("V", "W", "R", "G")
        }

    for k in range(math.ceil(t0), math.floor(t1)):
        a, b = float(k), float(k + 1)
        j0, j1 = np.searchsorted(tru.t, [a, b])
        if (
            j0 == 0
            or j1 >= len(tru.t)
            or min(a - tru.t[j0 - 1], tru.t[j0] - a) > 0.15
            or min(b - tru.t[j1 - 1], tru.t[j1] - b) > 0.15
        ):
            continue
        if d.mov["W"].over("vx", a, b, 0.5) is None:
            continue
        x, y, w = tru.at(np.array([a, b]))
        rel = relative(x[0], y[0], w[0], x[1], y[1], w[1])
        out["s1"].append(
            {"a": a, "b": b, "T": rel, "len": math.hypot(rel[0], rel[1]), **srcs(a, b)}
        )
    path = Trajectory(tru.t, smooth(tru.x), smooth(tru.y), tru.yaw)
    for i, j, length in segments(path):
        rel = relative(tru.x[i], tru.y[i], tru.yaw[i], tru.x[j], tru.y[j], tru.yaw[j])
        out["m1"].append(
            {
                "a": float(tru.t[i]),
                "b": float(tru.t[j]),
                "T": rel,
                "len": length,
                **srcs(tru.t[i], tru.t[j]),
            }
        )
    return out


def information(d: Sources, vt: Array, lz: Array, odom: Array, imu: Array, gvar: float) -> Array:
    """Per moving second (wheels' mean |vx| > 3 cm/s or |wz| > 3 deg/s): the sum of 1 / variance
    each source sent in it: VIO vx, wheels vx under the law, wheels vx at rest, rf2o vx, VIO vyaw,
    wheels vyaw law / rest, and the gyro's samples / its variance."""
    t0, t1 = d.span

    def inv(v: Array) -> float:
        return float(np.sum(np.where(v < 1e3, 1.0 / np.maximum(v, 1e-12), 0.0)))

    info = []
    for s0 in np.arange(t0, t1 - 1, 1.0):
        k = (odom[:, 0] >= s0) & (odom[:, 0] < s0 + 1)
        if not k.any() or not (
            np.mean(np.abs(odom[k, 5])) > 0.03 or np.mean(np.abs(odom[k, 7])) > math.radians(3)
        ):
            continue
        law = k & (np.abs(odom[:, 8] - ODOM_REST) > 1e-9)
        rst = k & ~law
        kv = (vt[:, 0] >= s0) & (vt[:, 0] < s0 + 1)
        kl = (lz[:, 0] >= s0) & (lz[:, 0] < s0 + 1)
        ki = (imu[:, 0] >= s0) & (imu[:, 0] < s0 + 1)
        info.append(
            (
                inv(vt[kv, 5]),
                inv(odom[law, 8]),
                inv(odom[rst, 8]),
                inv(lz[kl, 8]),
                inv(vt[kv, 7]),
                inv(odom[law, 10]),
                inv(odom[rst, 10]),
                ki.sum() / gvar,
            )
        )
    return np.array(info).reshape(-1, 8)


def drive_facts(job: DriveJob) -> dict[str, Any]:
    """Everything the analysis needs from one drive, computed once (the work dir's drives.pkl)."""
    d = load_sources(job)
    f = d.f
    odom = f["/odom"][np.argsort(f["/odom"][:, 0])]
    gvar = json.loads((job.run_dir / "meta.json").read_text())["gyro_var"]
    vt = f.get("/vo_twist", np.zeros((0, 8)))
    info = information(d, vt, f["/odom_laser"], odom, f["/imu/data_raw"], gvar)
    return {
        "n": job.n,
        "span": job.span,
        "kept": d.kept,
        "scans": d.scans,
        "tr": d.tr,
        "win": {k: windows(d, k) for k in ("vx", "wz")},
        "bins": {k: acf_bins(d, k) for k in ("vx", "wz")},
        "disp": disp_windows(d),
        "cnt": {s: len(d.mov[s].a) for s in d.mov},
        "movsec": float(np.sum(d.mov["W"].b - d.mov["W"].a)),
        "info": info,
        "live": (f["/odometry/filtered"], f["/odometry/filtered:pcov"]),
    }


def series(src: Src, key: str, grid: npt.NDArray[np.floating[Any]], box: int = 10) -> Array:
    """The source's value on a 10 ms grid (the step containing each grid time; NaN outside its
    steps), then its 0.1 s box mean (>= 80 % of the box covered)."""
    i = np.searchsorted(src.b, grid)
    ok = i < len(src.a)
    i = np.minimum(i, len(src.a) - 1)
    ok &= src.a[i] <= grid
    v = np.where(ok, src.v[key][i], np.nan)
    k = np.ones(box)
    sm = np.convolve(np.nan_to_num(v), k, "same")
    ct = np.convolve(np.isfinite(v).astype(float), k, "same")
    return np.where(ct >= 0.8 * box, sm / np.maximum(ct, 1), np.nan)


def offsets_one(job: DriveJob) -> dict[str, Array]:
    """Per pair (source, the wheels): the sum and count of (X(t + s) - W(t))^2 over the moving
    0.1 s means for every offset s in LAGS (X late by s when the minimum sits at s > 0). T = the
    truth's fit (stamps as recorded) on a 50 ms grid; Vw = the VIO's yaw rate against the
    wheels'."""
    d = load_sources(job)
    t0, t1 = job.span
    grid = np.arange(t0, t1, 0.01)
    w = {"vx": series(d.mov["W"], "vx", grid), "wz": series(d.mov["W"], "wz", grid)}
    tg = np.arange(t0, t1, 0.05)
    fv = [fit_vw(d.tr, t) for t in tg.tolist()]
    tv = np.array([x[0] if x else np.nan for x in fv])
    out: dict[str, Array] = {}
    for name, key in PAIRS:
        src = "V" if name == "Vw" else name
        if src == "T":
            x = np.interp(grid, tg, tv, left=np.nan, right=np.nan)
        elif src not in d.mov or not len(d.mov[src].a):
            continue
        else:
            x = series(d.mov[src], key, grid)
        res = np.zeros((len(LAGS), 2))
        for j, lag in enumerate(LAGS):
            k = int(lag) // 10  # the lags are whole 10 ms steps of the grid
            dd = x[k:] - w[key][: len(grid) - k] if k >= 0 else x[: len(grid) + k] - w[key][-k:]
            ok = np.isfinite(dd)
            res[j] = (float(np.sum(dd[ok] ** 2)), float(ok.sum()))
        out[name] = res
    return out
