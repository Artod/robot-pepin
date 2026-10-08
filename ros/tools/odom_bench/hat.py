"""The three-cornered hat: each of three independent sources' own error, with no truth in it.

Per window the mean squares M_XY of the pairwise differences give each source's own squared error
e_X^2 = (M_XY + M_XZ - M_YZ) / 2. The truth enters as a fourth source, e_T^2 = M_VT - e_V^2 (only
the VIO assumed independent of it); its covariance with the wheels (it seeds each scan match on
them) and with rf2o (the same scans) follows, c_WT = (e_W^2 + e_T^2 - M_WT) / 2. Intervals are 95 %
by resampling drives (1000 draws, seed 1).
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from .sources import LENS, TRIPLE

Array = npt.NDArray[np.float64]
Interval = tuple[float, float, float]
Boot = tuple[float, float, float, Array]
RNG_SEED, NBOOT = 1, 1000


def sq(v: float) -> float:
    """Signed square root: a negative mean square reads as a negative rms."""
    return math.copysign(math.sqrt(abs(v)), v)


def own2(r: Array, p: int) -> Array:
    """The hat: own squared error of triple member p (0, 1, 2) per row (values in columns 0, 2,
    4)."""
    x = [r[:, 0], r[:, 2], r[:, 4]]
    i, j = [q for q in range(3) if q != p]
    out: Array = ((x[p] - x[i]) ** 2 + (x[p] - x[j]) ** 2 - (x[i] - x[j]) ** 2) / 2
    return out


def boot_mean(q: dict[str, Array], drives: list[str], seed: int = RNG_SEED) -> Boot:
    """Pooled mean of the per-row values over drives (NaN rows skipped), its 95 % interval by
    resampling drives, and the bootstrap draws."""
    ds = [n for n in drives if n in q and np.isfinite(q[n]).any()]
    if not ds:
        return math.nan, math.nan, math.nan, np.full(NBOOT, np.nan)
    s = np.array([np.nansum(q[n]) for n in ds])
    c = np.array([np.isfinite(q[n]).sum() for n in ds], float)
    draws = np.random.default_rng(seed).integers(0, len(ds), (NBOOT, len(ds)))
    bs = s[draws].sum(1) / c[draws].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(s.sum() / c.sum()), float(lo), float(hi), bs


def boot_ratio(
    qn: dict[str, Array], qd: dict[str, Array], drives: list[str], seed: int = RNG_SEED
) -> Boot:
    """sum(qn) / sum(qd) over drives (rows finite in both), its 95 % drive-bootstrap interval and
    the draws."""
    ds = [n for n in drives if n in qn and len(qn[n])]
    if not ds:
        return math.nan, math.nan, math.nan, np.full(NBOOT, np.nan)
    ok = {n: np.isfinite(qn[n]) & np.isfinite(qd[n]) for n in ds}
    a = np.array([qn[n][ok[n]].sum() for n in ds])
    b = np.array([qd[n][ok[n]].sum() for n in ds])
    draws = np.random.default_rng(seed).integers(0, len(ds), (NBOOT, len(ds)))
    bs = a[draws].sum(1) / b[draws].sum(1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(a.sum() / b.sum()), float(lo), float(hi), bs


Windows = dict[str, dict[str, dict[str, Array]]]  # drive -> kind -> window name -> rows


def source_stats(win: Windows, kind: str, drives: list[str]) -> dict[Any, Any]:
    """NEES of each triple member: own (the hat), as read against the truth, and the truth's floor
    in the member's units; per sample and per window length; the 4-source decomposition (the
    truth's own error, c_WT, c_3T); rf2o as stamped (Rs); the VIO's upper bound (the wheels'
    error counted as its own); the (V, W, truth) hat as a cross-check."""
    res: dict[Any, Any] = {}
    tri = TRIPLE[kind]
    for p, s in enumerate(tri):
        for key in [f"w{length:g}" for length in LENS] + [f"ps_{s}"]:
            rows = {n: win[n][kind][key] for n in drives if kind in win[n] and key in win[n][kind]}

            def sd(r: Array, p: int = p) -> Array:
                out: Array = r[:, 2 * p + 1] ** 2
                return out

            own = {n: own2(r, p) / sd(r) for n, r in rows.items()}
            ok_t = {n: np.isfinite(r[:, 6]) for n, r in rows.items()}
            read = {
                n: np.where(ok_t[n], (r[:, 2 * p] - r[:, 6]) ** 2 / sd(r), np.nan)
                for n, r in rows.items()
            }
            floor = {
                n: np.where(ok_t[n], ((r[:, 0] - r[:, 6]) ** 2 - own2(r, 0)) / sd(r), np.nan)
                for n, r in rows.items()
            }
            nrow = sum(len(r) for r in rows.values())
            csd = (
                float(np.median(np.concatenate([np.sqrt(sd(r)) for r in rows.values()])))
                if nrow
                else math.nan
            )
            res[(s, key)] = {
                "own": boot_mean(own, drives),
                "read": boot_mean(read, drives),
                "floor": boot_mean(floor, drives),
                "n": nrow,
                "sd": csd,
                "own_rm": boot_ratio(
                    {n: own2(r, p) for n, r in rows.items()},
                    {n: sd(r) for n, r in rows.items()},
                    drives,
                ),
                "ndr": len([n for n, r in rows.items() if len(r)]),
            }
    rows1 = {n: win[n][kind]["w1"] for n in drives if kind in win[n] and "w1" in win[n][kind]}
    rows_t = {n: r[np.isfinite(r[:, 6])] for n, r in rows1.items()}
    parts: dict[str, Callable[[Array], Array]] = {
        "e_V": lambda r: own2(r, 0),
        "e_W": lambda r: own2(r, 1),
        "e_3": lambda r: own2(r, 2),
        "e_T": lambda r: (r[:, 0] - r[:, 6]) ** 2 - own2(r, 0),
        "c_WT": lambda r: (
            (own2(r, 1) + (r[:, 0] - r[:, 6]) ** 2 - own2(r, 0) - (r[:, 2] - r[:, 6]) ** 2) / 2
        ),
        "c_3T": lambda r: (
            (own2(r, 2) + (r[:, 0] - r[:, 6]) ** 2 - own2(r, 0) - (r[:, 4] - r[:, 6]) ** 2) / 2
        ),
    }
    res["M"] = {
        lab: boot_mean({n: fn(r) for n, r in rows_t.items()}, drives) for lab, fn in parts.items()
    }
    if kind == "vx":
        for length in LENS:
            rr = {
                n: win[n][kind][f"w{length:g}"]
                for n in drives
                if kind in win[n] and f"w{length:g}" in win[n][kind]
            }
            rr = {
                n: np.column_stack([r[:, :4], r[:, 7], r[:, 8]])[np.isfinite(r[:, 7])]
                for n, r in rr.items()
            }
            res[("Rs", f"w{length:g}")] = boot_mean(
                {n: own2(r, 2) / r[:, 5] ** 2 for n, r in rr.items()}, drives
            )
            res[("Rs", f"w{length:g}", "rm")] = boot_ratio(
                {n: own2(r, 2) for n, r in rr.items()},
                {n: r[:, 5] ** 2 for n, r in rr.items()},
                drives,
            )
    res[("V", "upper")] = boot_mean(
        {n: (r[:, 0] - r[:, 2]) ** 2 / r[:, 1] ** 2 for n, r in rows1.items()}, drives
    )
    if kind == "vx":  # the (V, W, truth) hat
        vwt = {
            n: np.column_stack([r[:, 0], r[:, 1], r[:, 2], r[:, 3], r[:, 6], np.zeros(len(r))])
            for n, r in rows_t.items()
        }
        for p, s in ((0, "V"), (1, "W")):
            res[(s, "vwt")] = boot_mean(
                {n: own2(r, p) / r[:, 2 * p + 1] ** 2 for n, r in vwt.items()}, drives
            )
    return res


def sokal(sums: Array, counts: Array, maxlag: int) -> Array:
    """Each member's error autocorrelation from the summed lagged products of the three pairwise
    differences (and their counts): per member rho(0..maxlag), tau_int (s, 0.1 s bins, Sokal's
    window: the smallest M >= 5 tau(M)), the window, the lag-0 own variance."""
    mp = sums.sum(0) / np.maximum(counts.sum(0), 1)  # (3 pairs, lags)
    own = np.array(
        [
            (mp[0] + mp[1] - mp[2]) / 2,
            (mp[0] + mp[2] - mp[1]) / 2,
            (mp[1] + mp[2] - mp[0]) / 2,
        ]
    )
    out = np.full((3, maxlag + 4), np.nan)
    for p in range(3):
        rho = own[p] / own[p][0] if own[p][0] > 0 else np.full(maxlag + 1, np.nan)
        t_m, chosen = math.nan, maxlag
        for m in range(1, maxlag + 1):
            t_m = 1 + 2 * float(np.sum(rho[1 : m + 1]))
            if m >= 5 * t_m:
                chosen = m
                break
        out[p, : maxlag + 1] = rho
        out[p, maxlag + 1] = t_m * 0.1
        out[p, maxlag + 2] = chosen
        out[p, maxlag + 3] = own[p][0]
    return out


def acf_stats(
    bins: dict[str, dict[str, Array]], kind: str, drives: list[str], maxlag: int = 30
) -> dict[str, Any]:
    """The hat on the lagged second moments of the pairwise differences of the 0.1 s bins: each
    member's error autocorrelation and tau_int, with drive-bootstrap intervals."""
    ds = [
        n
        for n in drives
        if n in bins and kind in bins[n] and np.isfinite(bins[n][kind][:, 0]).any()
    ]
    pairs = ((0, 1), (0, 2), (1, 2))
    sums = np.zeros((len(ds), 3, maxlag + 1))
    counts = np.zeros((len(ds), 3, maxlag + 1))
    for i, n in enumerate(ds):
        b = bins[n][kind]
        full = np.isfinite(b).all(1)  # the three pairs on the same bins, or nothing cancels
        for j, (x, y) in enumerate(pairs):
            dd = np.where(full, b[:, x] - b[:, y], np.nan)
            for lag in range(maxlag + 1):
                pr = dd[: len(dd) - lag] * dd[lag:]
                ok = np.isfinite(pr)
                sums[i, j, lag], counts[i, j, lag] = pr[ok].sum(), ok.sum()
    point = sokal(sums, counts, maxlag)
    draws = np.random.default_rng(RNG_SEED).integers(0, len(ds), (NBOOT, len(ds)))
    bs = np.array([sokal(sums[dr], counts[dr], maxlag) for dr in draws])
    return {
        "point": point,
        "lo": np.nanpercentile(bs, 2.5, axis=0),
        "hi": np.nanpercentile(bs, 97.5, axis=0),
        "ndr": len(ds),
    }


def floor_parts(d: dict[str, Any], kind: str) -> Array:
    """Per displacement window with V, W, R (and G) integrated, the squared pairwise differences
    for the truth's floor: dx VW VR WR VT, dy the same, dth GV GW VW GT (NaN where missing)."""
    rows = []
    for w in d["disp"][kind]:
        tru = w["T"]
        vio, whl, lsr, gyr = w["V"], w["W"], w["R"], w["G"]
        row = [np.nan] * 12
        if vio and whl and lsr:
            for j, c_ in enumerate((0, 1)):
                row[4 * j : 4 * j + 4] = [
                    (vio[c_] - whl[c_]) ** 2,
                    (vio[c_] - lsr[c_]) ** 2,
                    (whl[c_] - lsr[c_]) ** 2,
                    (vio[c_] - tru[c_]) ** 2,
                ]
        if gyr and vio and whl:
            th = math.remainder
            row[8:12] = [
                th(gyr[2] - vio[2], math.tau) ** 2,
                th(gyr[2] - whl[2], math.tau) ** 2,
                th(vio[2] - whl[2], math.tau) ** 2,
                th(gyr[2] - tru[2], math.tau) ** 2,
            ]
        rows.append(row)
    return np.array(rows).reshape(-1, 12)


def floors(
    parts: dict[str, Array],
    ds: list[str],
    draw: npt.NDArray[np.int64] | None = None,
    across: bool = False,
) -> Array:
    """The truth's own mean squared error of (dx, dy, dth) over the windows: e_T^2 = M_VT - e_V^2
    (VWR hat; yaw: the GVW hat, M_GT - e_G^2).

    Across (dy) is not separable: the VIO and the wheels share a lateral model error (the hat
    gives the VIO a negative own error there), so its floor is 0 (nothing subtracted: the net RPE
    is an upper bound) unless ``across`` asks for the hat's value.
    """
    sel = [ds[i] for i in draw] if draw is not None else ds
    rows = np.concatenate([parts[n] for n in sel]) if sel else np.zeros((0, 12))
    m = np.nanmean(rows, 0) if len(rows) else np.full(12, np.nan)
    out = []
    for j in (0, 1):
        vw, vr, wr, vt = m[4 * j : 4 * j + 4]
        out.append(vt - (vw + vr - wr) / 2 if (j == 0 or across) else 0.0)
    gv, gw, vw, gt = m[8:12]
    out.append(gt - (gv + gw - vw) / 2)
    return np.array(out)
