"""The scores of one fused odometry against the truth: RPE@1s, RPE@1m, the end error, NEES.

RPE@1s is the translation error per moving second, RPE@1m per 1 m piece of the smoothed truth
path, both "net" of the truth's own along-track and yaw error (the hat's floor; across-track is
as read, so net is an upper bound). The end error is the goal's whole relative motion against
the truth's. vNEES: the filter's forward error per second net of the floor over its own twist
variance; pose NEES: its pose-covariance growth P(t1) - P(t0) against the pose error.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import numpy.typing as npt
from vio_score import Trajectory, relative

from .hat import NBOOT, RNG_SEED, floors, sq

Array = npt.NDArray[np.float64]
Scores = dict[str, dict[str, Any]]  # drive -> one arm's per-drive scores
Floors = dict[str, dict[str, Array]]  # window kind -> drive -> floor_parts rows
PAIRED = ("end_mean", "head_abs", "m1_net", "s1_net", "m1_yaw_net", "s1_vnees")


def end_parts(tr: Trajectory, arm: Trajectory) -> tuple[float, float, float, float, float]:
    """The whole span's (error cm, along cm, across cm (+ left), signed end heading deg, net
    displacement m)."""
    ax, ay, aw = arm.at(np.array([tr.t[0], tr.t[-1]]))
    t_r = relative(tr.x[0], tr.y[0], tr.yaw[0], tr.x[-1], tr.y[-1], tr.yaw[-1])
    a_r = relative(ax[0], ay[0], aw[0], ax[1], ay[1], aw[1])
    disp = math.hypot(t_r[0], t_r[1])
    ux, uy = t_r[0] / disp, t_r[1] / disp
    ex, ey = a_r[0] - t_r[0], a_r[1] - t_r[1]
    dyaw = math.degrees(math.remainder(a_r[2] - t_r[2], math.tau))
    return (
        100 * math.hypot(ex, ey),
        100 * (ex * ux + ey * uy),
        100 * (-ex * uy + ey * ux),
        dyaw,
        disp,
    )


def pmat(c: Array) -> Array:
    """The planar 3x3 pose covariance from a pcov row (stamp, then 9 entries)."""
    return np.array([[c[1], c[2], c[3]], [c[4], c[5], c[6]], [c[7], c[8], c[9]]])


def pose_nees(e: Array, ca: Array, cb: Array, yaw0: float) -> float:
    """e^T dP^-1 e / 3 with dP = P(b) - P(a) rotated into the start frame; NaN unless dP is
    positive definite."""
    dp = pmat(cb) - pmat(ca)
    cs, sn = math.cos(yaw0), math.sin(yaw0)
    rot = np.array([[cs, sn, 0.0], [-sn, cs, 0.0], [0.0, 0.0, 1.0]])
    dp = rot @ dp @ rot.T
    try:
        if np.min(np.linalg.eigvalsh(0.5 * (dp + dp.T))) <= 0:
            return math.nan
        return float(e @ np.linalg.solve(dp, e)) / 3.0
    except np.linalg.LinAlgError:
        return math.nan


def score_arm(d: dict[str, Any], od: tuple[Array, Array]) -> dict[str, Any]:
    """One arm on one drive: the end, the 1 s and 1 m window errors with the filter's NEES terms
    (rows: ex, ey, eth, length, twist variance x window^2, pose NEES)."""
    o, c = od
    arm = Trajectory.from_rows(o[:, [0, 2, 3, 4]].tolist())
    t0, t1 = d["span"]
    tr = d["tr"][(d["tr"][:, 0] >= t0) & (d["tr"][:, 0] <= t1)]
    tru = Trajectory(tr[:, 0], tr[:, 1], tr[:, 2], tr[:, 3])
    out: dict[str, Any] = {"end": end_parts(tru, arm)}
    ostamps = o[:, 0]

    def near(t: float) -> int:
        return int(np.clip(np.searchsorted(ostamps, t), 0, len(ostamps) - 1))

    ax, ay, aw = arm.at(np.array([tru.t[0], tru.t[-1]]))
    if np.all(np.isfinite(ax)):
        tr_r = relative(tru.x[0], tru.y[0], tru.yaw[0], tru.x[-1], tru.y[-1], tru.yaw[-1])
        ar_r = relative(ax[0], ay[0], aw[0], ax[1], ay[1], aw[1])
        e = np.array(
            [ar_r[0] - tr_r[0], ar_r[1] - tr_r[1], math.remainder(ar_r[2] - tr_r[2], math.tau)]
        )
        out["end_nees"] = pose_nees(e, c[near(tru.t[0])], c[near(tru.t[-1])], aw[0])
    for kind in ("s1", "m1"):
        rows = []
        for w in d["disp"][kind]:
            a, b = w["a"], w["b"]
            x, y, yw = arm.at(np.array([a, b]))
            if not np.all(np.isfinite(x)):
                continue
            r = relative(x[0], y[0], yw[0], x[1], y[1], yw[1])
            ex, ey = r[0] - w["T"][0], r[1] - w["T"][1]
            eth = math.remainder(r[2] - w["T"][2], math.tau)
            k = (ostamps >= a) & (ostamps <= b)
            pv = float(np.mean(o[k, 8])) if k.any() else math.nan
            nees = pose_nees(np.array([ex, ey, eth]), c[near(a)], c[near(b)], yw[0])
            rows.append([ex, ey, eth, w["len"], pv * (b - a) ** 2, nees])
        out[kind] = np.array(rows).reshape(-1, 6)
    return out


def metrics(
    scores: Scores,
    parts: dict[str, Array],
    ds: list[str],
    draw: npt.NDArray[np.int64] | None = None,
) -> dict[str, float]:
    """Pooled metrics of one arm over drives ds (or a bootstrap draw of them)."""
    sel = [ds[i] for i in draw] if draw is not None else ds
    sel = [n for n in sel if n in scores]
    fl = floors(parts, ds, draw)
    out: dict[str, float] = {}
    for kind in ("s1", "m1"):
        if not sel or kind not in scores[sel[0]]:
            continue
        rows = np.concatenate([scores[n][kind] for n in sel])
        if not len(rows):
            continue
        ms = float(np.mean(rows[:, 0] ** 2 + rows[:, 1] ** 2))
        msy = float(np.mean(rows[:, 2] ** 2))
        length = float(np.mean(rows[:, 3]))
        out[f"{kind}_read"] = math.sqrt(ms)
        out[f"{kind}_net"] = sq(ms - fl[0] - fl[1])
        out[f"{kind}_yaw_read"] = math.degrees(math.sqrt(msy))
        out[f"{kind}_yaw_net"] = math.degrees(sq(msy - fl[2]))
        out[f"{kind}_pct_net"] = 100 * out[f"{kind}_net"] / length
        out[f"{kind}_pct_read"] = 100 * out[f"{kind}_read"] / length
        out[f"{kind}_len"] = length
        pv = rows[:, 4]
        ok = np.isfinite(pv) & (pv > 0)
        out[f"{kind}_vnees"] = (
            float(np.mean(rows[ok, 0] ** 2 / pv[ok]) - fl[0] * np.mean(1 / pv[ok]))
            if ok.any()
            else math.nan
        )
        pn = rows[:, 5]
        out[f"{kind}_pnees"] = float(np.nanmean(pn)) if np.isfinite(pn).any() else math.nan
        out[f"{kind}_floor"] = sq(fl[0] + fl[1])
    ends = np.array([scores[n]["end"] for n in sel])
    out["end_mean"] = float(np.mean(ends[:, 0]))
    out["end_median"] = float(np.median(ends[:, 0]))
    out["head_abs"] = float(np.mean(np.abs(ends[:, 3])))
    en = [scores[n].get("end_nees", math.nan) for n in sel]
    out["end_nees"] = float(np.nanmean(en)) if np.isfinite(en).any() else math.nan
    return out


def metrics_k(
    scores: Scores, floors_k: Floors, ds: list[str], draw: npt.NDArray[np.int64] | None = None
) -> dict[str, float]:
    """``metrics`` per window kind. The 1 m pieces' own hat has too few windows with the VIO
    covered, so both kinds take the per-second floor: a lower bound of the truth's error over a
    longer piece, which makes the net RPE@1m conservative."""
    out = {}
    for kind in ("s1", "m1"):
        m = metrics(
            {
                n: {
                    kind: scores[n][kind],
                    "end": scores[n]["end"],
                    "end_nees": scores[n].get("end_nees", math.nan),
                }
                for n in scores
            },
            floors_k["s1"],
            ds,
            draw,
        )
        out.update({k: v for k, v in m.items() if k.startswith(kind) or kind == "s1"})
    return out


def draws_for(ds: list[str]) -> npt.NDArray[np.int64]:
    """The drive-bootstrap draws every interval of a drive list shares (seed 1)."""
    return np.random.default_rng(RNG_SEED).integers(0, len(ds), (NBOOT, len(ds)))


def boot_metrics_k(
    scores: Scores, floors_k: Floors, ds: list[str]
) -> tuple[dict[str, float], list[dict[str, float]]]:
    """``metrics_k`` and its bootstrap draws."""
    return metrics_k(scores, floors_k, ds), [
        metrics_k(scores, floors_k, ds, dr) for dr in draws_for(ds)
    ]


def ci(bs: list[dict[str, float]], key: str) -> tuple[float, float]:
    """The 95 % interval of one metric over the bootstrap draws (NaN draws skipped)."""
    v = np.array([b.get(key, np.nan) for b in bs], float)
    v = v[np.isfinite(v)]
    return (
        (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
        if len(v)
        else (math.nan, math.nan)
    )


def paired_k(sa: Scores, sb: Scores, floors_k: Floors, ds: list[str]) -> dict[str, Any]:
    """Arm B minus arm A on the same drives and the same bootstrap draws: each PAIRED metric's
    difference with its interval, the drives B ends closer on, and the per-drive end errors."""
    pa, pb = metrics_k(sa, floors_k, ds), metrics_k(sb, floors_k, ds)
    bs = []
    for dr in draws_for(ds):
        ma, mb = metrics_k(sa, floors_k, ds, dr), metrics_k(sb, floors_k, ds, dr)
        bs.append({k: mb.get(k, np.nan) - ma.get(k, np.nan) for k in PAIRED})
    out: dict[str, Any] = {k: (pb.get(k, np.nan) - pa.get(k, np.nan), *ci(bs, k)) for k in PAIRED}
    ea = np.array([sa[n]["end"][0] for n in ds])
    eb = np.array([sb[n]["end"][0] for n in ds])
    out["wins"] = (int(np.sum(eb < ea)), len(ds))
    out["per_drive"] = {n: (sa[n]["end"][0], sb[n]["end"][0]) for n in ds}
    return out


def shares(
    facts: dict[str, dict[str, Any]], ds: list[str], var: dict[str, float]
) -> dict[str, Array]:
    """Information per moving second (the sum of 1 / variance each source sent), the base claims
    scaled by the arm's variance factors: p50 of the per-second shares and the share of the
    pooled sums, %. vx: VIO, wheels, rf2o; vyaw: VIO, wheels, gyro."""
    info = np.concatenate([facts[n]["info"] for n in ds if len(facts[n]["info"])]).copy()
    info[:, 0] /= var.get("V", 1.0)
    info[:, 4] /= var.get("Y", 1.0)
    info[:, 1] /= var.get("Wv", 1.0)
    info[:, 5] /= var.get("Ww", 1.0)
    info[:, 3] /= var.get("R", 1.0)
    vx = np.column_stack([info[:, 0], info[:, 1] + info[:, 2], info[:, 3]])
    wz = np.column_stack([info[:, 4], info[:, 5] + info[:, 6], info[:, 7]])
    return {
        k: np.concatenate(
            [100 * np.median(M / M.sum(1, keepdims=True), 0), 100 * M.sum(0) / M.sum()]
        )
        for k, M in (("vx", vx), ("wz", wz))
    }
