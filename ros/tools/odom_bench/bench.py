"""The bench's stages and its report: is each speed source's claimed covariance honest, what
multiplier makes it so, does applying it improve held-out drives, how much does each source add,
and does the VIO's per-second error grow with the camera rate (the sqrt(rate) rule)?

Method (pre-registered 2026-10-07, before the first run):

- DRIVES: a frozen set (``sets/*.json``: bag slice, tape and truth hashes). Dropped: truth kept
  under 75 % of the goal span's taped scans. Halves by drive number: TRAIN odd, TEST even. Two
  camera-rate groups; VIO numbers are never pooled across them, wheels, rf2o and the gyro (the
  same sensors in every drive) may pool. The VERDICT is for the live group's configuration:
  VIO multipliers from its TRAIN, wheels/rf2o from all TRAIN, judged on its TEST.
- BASE COPIES: every bag copied with its claims normalised to the base configuration (the
  reference drives' VIO scales, no sqrt(rate) factor, the wheels on the shipped law, rf2o as
  recorded). Each drive's recorded scale is measured (the relay's /vo_twist covariance over
  OpenVINS's own); a drive off every known configuration stops the bench.
- TRUTH: the lidar truth (truth.py) against the set's map, heading corrected by the set's angle.
  Its stamps and rf2o's run late against the wheels: each source's offset is measured (STAMP
  OFFSETS) and rf2o and the truth are aligned to the wheels' clock before the hat.
- THE TRUTH'S OWN ERROR IS REMOVED by the three-cornered hat on the three independent speed
  sources (hat.py): VIO, wheels, rf2o for the forward speed; VIO, wheels, gyro for the turn rate.
- NEES (1-D, mean e_own^2 / sigma_claimed^2; 1.0 = honest) per sample and per 1 / 2 / 5 s window
  (each source's mean over the window, the claimed sigma of that mean as if its samples were
  independent: what the EKF believes). The multiplier is sqrt(per-1-s NEES on TRAIN).
- AUTOCORRELATION TIME of each source's error: 0.1 s bins, the hat on the lagged second moments,
  tau_int with Sokal's window; 1 / tau_int independent samples per second against those counted.
- REPLAYS (TEST): the board's EKF (ros/params/ekf.yaml) on the drive's recorded inputs (replay.py).
  A = base copy; A2 = A again (the null); B_* = base x one tuned multiplier each, B = all of
  them; C arms with sources left out: W, WG, WGR, WGV.
- METRICS per arm (metrics.py): RPE@1s, RPE@1m, the end error, the filter's NEES.
- DECISION RULE (fixed before the replays): an item is APPLIED if B_item - A on the live group's
  TEST drives lowers the end position error with the 95 % drive-bootstrap interval of the mean
  paired difference wholly below 0 and no secondary metric (RPE@1m, RPE@1s, |end heading|) worse
  with its interval wholly above 0; otherwise KEEP AS IS. An item whose TRAIN multiplier interval
  reaches <= 0 (unresolved) is KEEP without a replay.
- SQRT(RATE): the VIO's per-1-s NEES on the base copies, live-rate TRAIN over old-rate TRAIN; the
  rule predicts the ratio of the rates. Supported if the ratio's interval excludes 1 and contains
  the prediction; contradicted if it excludes the prediction.
"""

from __future__ import annotations

import dataclasses
import json
import math
import pickle
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.lidar import LidarMount
from pepin.wheel_noise import WheelNoiseLaw

from .bags import detect_claims, extract_facts, read_filtered
from .drives import REPO, DriveSet, goal_span, sha256
from .hat import NBOOT, RNG_SEED, acf_stats, boot_mean, boot_ratio, floor_parts, floors, own2, sq
from .hat import source_stats as hat_stats
from .metrics import Floors, Scores, boot_metrics_k, ci, paired_k, score_arm, shares
from .replay import EKF_YAML, Arm, ReplayJob, base_arms, check_vio_replay, done, replays
from .rewrite import Claims, classify, rewrite_bag
from .sources import LAGS as LAG_MS
from .sources import PAIRS as LAG_PAIRS
from .sources import TRIPLE, DriveJob, drive_facts, offsets_one
from .truth import cached_truths

Array = npt.NDArray[np.float64]
Interval = tuple[Any, ...]  # a point, its interval (and maybe the bootstrap draws)
COVER_MIN = 0.75  # the truth must keep this share of the goal span's taped scans
NAMES = {"V": "VIO", "W": "wheels", "R": "rf2o", "G": "gyro", "T": "truth"}
ITEMS = {  # tuned item: label, quantity, source, fitted on ("live": the live group's TRAIN)
    "V": ("VIO vx/vy sigma (vio_sigma_scale)", "vx", "V", "live"),
    "Y": ("VIO yaw sigma (vio_yaw_sigma_scale)", "wz", "V", "live"),
    "Wv": ("wheels vx (odometry_noise v_*)", "vx", "W", "all"),
    "Ww": ("wheels vyaw (odometry_noise yaw_*)", "wz", "W", "all"),
    "R": ("rf2o vx covariance", "vx", "R", "all"),
}
VIO_ITEMS = {"V", "Y"}
RULE = "=" * 118


def f3(t: Interval, k: float = 1.0, fmt: str = "5.2f") -> str:
    """A point with its interval: 'p [lo, hi]'."""
    return f"{t[0] * k:{fmt}} [{t[1] * k:{fmt}}, {t[2] * k:{fmt}}]"


def sqt(t: Interval) -> tuple[float, float, float]:
    """``sq`` of a point and its interval (NEES -> multiplier, mean square -> rms)."""
    return sq(t[0]), sq(t[1]), sq(t[2])


class Report:
    """The report's lines: printed as they come and kept for bench_out.txt."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, *a: Any) -> None:
        s = " ".join(str(x) for x in a)
        print(s, flush=True)
        self.lines.append(s)


def log(*a: Any) -> None:
    """Progress, not part of the report."""
    print(*a, flush=True)


@dataclass(frozen=True)
class Work:
    """The bench's work directory: everything it computes, cached by stage."""

    root: Path

    @property
    def runs(self) -> Path:
        """Per drive: truth.csv (+ its quality report), facts.npz, meta.json."""
        return self.root / "runs"

    @property
    def cache(self) -> Path:
        """The analysis caches."""
        return self.root / "cache"

    @property
    def out(self) -> Path:
        """The replays' outputs, per arm."""
        return self.root / "out"

    @property
    def logs(self) -> Path:
        """The replays' logs."""
        return self.root / "logs"

    def bags(self, name: str) -> Path:
        """A bag-copy set: "base" or a B arm's own."""
        return self.root / "rec" / name


@dataclass
class Bench:
    """One run's inputs and settings."""

    drives: DriveSet
    rec: Path
    work: Work
    truth_map: Path
    image: str
    cpus: str
    parallel: int
    law: WheelNoiseLaw
    mount: LidarMount
    say: Report
    arms: dict[str, Arm]

    @property
    def live(self) -> str:
        """The live group's name."""
        return self.drives.live_group

    @property
    def old(self) -> str:
        """The other group's name."""
        return self.drives.old_group

    def fps(self, group: str) -> int:
        """A group's camera rate."""
        return self.drives.groups[group]


@dataclass
class Split:
    """The drives kept (truth coverage) and dropped, selectable by group and half."""

    drives: DriveSet
    kept: list[str]
    dropped: list[tuple[str, float]]

    def sel(self, grp: str = "all", half: str = "all") -> list[str]:
        """Kept drives of a group ("all": every group) and half ("TRAIN", "TEST", "all")."""
        return [
            n
            for n in self.kept
            if (grp == "all" or self.drives.group(n) == grp)
            and (half == "all" or (int(n) % 2 == 1) == (half == "TRAIN"))
        ]


# ------------------------------------------------------------------------- preparation
def verify(b: Bench) -> dict[str, dict[str, Any]]:
    """Each drive's bag slice and tape hashed (cached by their mtimes) against the frozen hashes;
    the goal span from the tape."""
    p = b.work.cache / "manifest.json"
    old = json.loads(p.read_text()) if p.exists() else {}
    man: dict[str, dict[str, Any]] = {}
    for n in b.drives.numbers:
        d = b.drives.drives[n]
        bag, tape = d.bag_dir(b.rec), d.tape(b.rec)
        mcap = sorted(bag.glob("*.mcap"))
        if len(mcap) != 1:
            raise SystemExit(f"{n}: {bag} holds {len(mcap)} .mcap files, expected one")
        o = old.get(n, {})
        mtimes = [mcap[0].stat().st_mtime, tape.stat().st_mtime]
        if o.get("bag") == bag.name and o.get("mtimes") == mtimes:
            man[n] = o
        else:
            man[n] = {
                "bag": bag.name,
                "mcap_sha256": sha256(mcap[0]),
                "tape_sha256": sha256(tape),
                "mtimes": mtimes,
                "span": list(goal_span(tape)),
            }
        bad = [k for k in ("mcap_sha256", "tape_sha256") if man[n][k] != getattr(d, k)]
        if bad:
            raise SystemExit(f"{n}: {', '.join(bad)} not the frozen set's: the recording changed")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(man, indent=1))
    return man


def truths(b: Bench) -> str:
    """Every drive's truth (cached by tape, map and mount), checked against the frozen sha256;
    returns the check's summary."""
    items = [
        (b.drives.drives[n].tape(b.rec), b.work.runs / n / "truth.csv") for n in b.drives.numbers
    ]
    t = time.time()
    k = cached_truths(items, b.truth_map, b.mount)
    log(f"[truth] {k} computed in {time.time() - t:.0f} s")
    same: list[str] = []
    differ: list[str] = []
    new: list[str] = []
    for n in b.drives.numbers:
        frozen = b.drives.drives[n].truth_sha256
        got = sha256(b.work.runs / n / "truth.csv")
        (new if frozen is None else same if got == frozen else differ).append(n)
        if frozen is not None and got != frozen:
            log(f"[truth] {n}: sha256 {got} DIFFERS from the frozen {frozen}")
    summary = f"{len(same)} of {len(b.drives.drives)} truths as frozen"
    if differ:
        summary += f", {len(differ)} DIFFER ({' '.join(differ)})"
    if new:
        summary += f", {len(new)} not frozen yet ({' '.join(new)})"
    return summary


def detect_job(job: tuple[str, Path, WheelNoiseLaw]) -> dict[str, Any]:
    """``detect_claims`` for the process pool."""
    return detect_claims(*job)


def base_job(job: tuple[str, Path, Path, WheelNoiseLaw, Claims]) -> str:
    """A drive's base copy: its claims normalised (none if it exists)."""
    n, src, dst, law, cfg = job
    if dst.exists():
        return f"{n} base exists"
    c = rewrite_bag(src, dst, law, (cfg.vio_lin_var, cfg.vio_yaw_var), restamp_law=not cfg.law_live)
    return f"{n} base {c}"


def extract_job(job: tuple[str, Path, Path]) -> str:
    """``extract_facts`` for the process pool."""
    n, bag, run_dir = job
    return f"{n} {extract_facts(bag, run_dir)}"


def prepare(b: Bench) -> tuple[dict[str, Any], dict[str, Any], dict[str, Claims], str]:
    """Hashes, the truth, the recorded claims, the base copies, the extracted facts."""
    man = verify(b)
    log(f"[verify] {len(man)} drives as frozen in {b.drives.path.name}")
    truth_line = truths(b)
    log(f"[truth] {truth_line}")
    dp = b.work.cache / "detect.json"
    det = json.loads(dp.read_text()) if dp.exists() else {}
    todo = [n for n in b.drives.numbers if n not in det]
    if todo:
        jobs = [(n, b.drives.drives[n].bag_dir(b.rec), b.law) for n in todo]
        with ProcessPoolExecutor(8) as ex:
            for d in ex.map(detect_job, jobs):
                det[d["n"]] = d
        dp.write_text(json.dumps(det, indent=1))
    cfg = classify(det, b.drives)
    base = [
        (n, d.bag_dir(b.rec), b.work.bags("base") / d.bag, b.law, cfg[n])
        for n, d in sorted(b.drives.drives.items())
    ]
    ext = [
        (n, b.work.bags("base") / d.bag, b.work.runs / n)
        for n, d in sorted(b.drives.drives.items())
    ]
    with ProcessPoolExecutor(8) as ex:
        for line in ex.map(base_job, base):
            log("[base]", line)
        for line in ex.map(extract_job, ext):
            log("[extract]", line)
    return man, det, cfg, truth_line


# ------------------------------------------------------------------------- offsets, facts
def job_for(b: Bench, man: dict[str, Any], n: str, shifts: dict[str, float]) -> DriveJob:
    """The worker's view of drive n."""
    return DriveJob(
        n=n,
        run_dir=b.work.runs / n,
        tape=b.drives.drives[n].tape(b.rec),
        span=(man[n]["span"][0], man[n]["span"][1]),
        heading_deg=b.drives.heading_deg,
        shifts=shifts,
    )


def offsets(b: Bench, man: dict[str, Any]) -> dict[str, dict[str, float]]:
    """The stamp offset of each source against the wheels per group, pooled over the group's
    TRAIN drives (the minimum of the mean squared difference of the 0.1 s means); rf2o's and the
    truth's are returned for alignment."""
    say = b.say
    p = b.work.cache / "offsets.pkl"
    if p.exists():
        lagged = pickle.loads(p.read_bytes())
    else:
        jobs = [job_for(b, man, n, {"T": 0.0}) for n in b.drives.numbers]
        with ProcessPoolExecutor(8) as ex:
            lagged = dict(zip(b.drives.numbers, ex.map(offsets_one, jobs), strict=True))
        p.write_bytes(pickle.dumps(lagged))
    say("")
    say(RULE)
    say(
        "STAMP OFFSETS against the wheels' clock (ms; + = the source's stamps are late), the"
        " minimum of the mean squared"
    )
    say(
        "difference of the moving 0.1 s means, pooled over the group's TRAIN drives [95 % by"
        " resampling drives]. A"
    )
    say(
        "shared clock error between two sources correlates their errors through the acceleration"
        " and breaks the hat"
    )
    say(
        "(own_V gains a^2 (tau_V - tau_R)(tau_V - tau_W)), so rf2o and the truth are aligned to"
        " the wheels before it."
    )
    sh: dict[str, dict[str, float]] = {}
    labels = {"V": "VIO vx", "R": "rf2o vx", "T": "truth vx", "G": "gyro wz", "Vw": "VIO wz"}
    for g in (b.old, b.live):
        ds = [n for n in b.drives.numbers if b.drives.group(n) == g and int(n) % 2 == 1]
        sh[g] = {}
        line = []
        for name, key in LAG_PAIRS:
            have = [n for n in ds if name in lagged[n]]
            if not have:
                continue
            arr = np.array([lagged[n][name] for n in have])  # drives, lags, 2

            def best(idx: npt.NDArray[np.int64], arr: Array = arr) -> float:
                tot = arr[idx].sum(0)
                return float(LAG_MS[int(np.argmin(tot[:, 0] / np.maximum(tot[:, 1], 1)))])

            pt = best(np.arange(len(have)))
            draws = np.random.default_rng(RNG_SEED).integers(0, len(have), (NBOOT, len(have)))
            bs = np.array([best(dr) for dr in draws])
            lo, hi = np.percentile(bs, [2.5, 97.5])
            tot = arr.sum(0)
            msd = np.sqrt(tot[:, 0] / np.maximum(tot[:, 1], 1))
            at0 = msd[int(np.flatnonzero(LAG_MS == 0)[0])]
            unit = 100 if key == "vx" else 57.2958
            line.append(
                f"{labels[name]} {pt:+.0f} [{lo:+.0f}, {hi:+.0f}] (rms diff {at0 * unit:.2f} ->"
                f" {msd.min() * unit:.2f})"
            )
            sh[g][name] = pt / 1000.0
        say(f"  {g} ({len(ds)} TRAIN drives): " + "; ".join(line))
    say(
        "  applied: rf2o (the hat's third member) and the truth aligned to the wheels by their"
        " group's offset (the truth's"
    )
    say(
        "  in place of the fixed 0.10 s); VIO and gyro kept as stamped: within 10-30 ms of the"
        " wheels, and the EKF"
    )
    say(
        "  fuses them so. rf2o's NEES is shown both aligned (its noise) and as stamped (what the"
        " EKF fuses)."
    )
    return {g: {"R": v["R"], "T": v["T"]} for g, v in sh.items()}


def load_all(
    b: Bench, man: dict[str, Any], shifts: dict[str, dict[str, float]]
) -> dict[str, dict[str, Any]]:
    """Every drive's analysis facts (sources.drive_facts), cached by the drive's hash, its facts'
    mtime, the truth's heading and the shifts."""
    p = b.work.cache / "drives.pkl"
    sh = {n: dict(shifts[b.drives.group(n)]) for n in b.drives.numbers}
    stamp = {
        n: (
            man[n]["mcap_sha256"],
            (b.work.runs / n / "facts.npz").stat().st_mtime,
            b.drives.heading_deg,
            tuple(sorted(sh[n].items())),
        )
        for n in b.drives.numbers
    }
    if p.exists():
        old = pickle.loads(p.read_bytes())
        if old.get("stamp") == stamp:
            facts: dict[str, dict[str, Any]] = old["D"]
            return facts
    with ProcessPoolExecutor(8) as ex:
        res = list(ex.map(drive_facts, [job_for(b, man, n, sh[n]) for n in b.drives.numbers]))
    facts = {r["n"]: r for r in res}
    p.write_bytes(pickle.dumps({"stamp": stamp, "D": facts}))
    return facts


# ------------------------------------------------------------------------- the sources' tables
def analyse(
    b: Bench, facts: dict[str, dict[str, Any]], det: dict[str, Any]
) -> tuple[Split, dict[str, Any]]:
    """Coverage filter, halves, the source tables, the TRAIN fits, the sqrt(rate) question."""
    say, old, live = b.say, b.old, b.live
    fo, fl = b.fps(old), b.fps(live)
    numbers = b.drives.numbers
    cov = {n: facts[n]["kept"] / max(facts[n]["scans"], 1) for n in numbers}
    split = Split(
        b.drives,
        [n for n in numbers if cov[n] >= COVER_MIN],
        [(n, cov[n]) for n in numbers if cov[n] < COVER_MIN],
    )
    say(RULE)
    say("DRIVE SET (truth kept / the tape's scans in the goal span; < 75 % dropped)")
    say("  dropped: " + ", ".join(f"{n} {100 * c:.0f} %" for n, c in split.dropped))
    for g in (old, live):
        say(
            f"  {g}: {len(split.sel(g))} kept; TRAIN (odd) {len(split.sel(g, 'TRAIN'))}:"
            f" {' '.join(split.sel(g, 'TRAIN'))}"
        )
        say(f"  {'':5s}  TEST (even) {len(split.sel(g, 'TEST'))}: {' '.join(split.sel(g, 'TEST'))}")
    say("  coverage %: " + " ".join(f"{n}:{100 * cov[n]:.0f}" for n in numbers))
    no_vio = [n for n in split.kept if det[n]["matched"] == 0]
    say("  no VIO in the bag (hat impossible, wheels/rf2o replays only): " + " ".join(no_vio))
    win = {n: facts[n]["win"] for n in split.kept}
    sets = {
        f"{old} TRAIN": split.sel(old, "TRAIN"),
        f"{live} TRAIN": split.sel(live, "TRAIN"),
        "all TRAIN": split.sel("all", "TRAIN"),
        f"{old} TEST": split.sel(old, "TEST"),
        f"{live} TEST": split.sel(live, "TEST"),
        "all TEST": split.sel("all", "TEST"),
    }
    stats = {(k, kind): hat_stats(win, kind, ds) for k, ds in sets.items() for kind in ("vx", "wz")}
    bins = {n: facts[n]["bins"] for n in split.kept}
    acf = {(k, kind): acf_stats(bins, kind, ds) for k, ds in sets.items() for kind in ("vx", "wz")}
    unit = {"vx": (100.0, "cm/s"), "wz": (57.2958, "deg/s")}
    say("")
    say(RULE)
    say(
        "PER-SOURCE NEES (1-D, mean e^2 / sigma_claimed^2; 1.00 = honest; the multiplier is its"
        " sqrt) on the base claims."
    )
    say(
        "own = the three-cornered hat (forward speed: VIO, wheels, rf2o; turn rate: VIO, wheels,"
        " gyro) - no truth in it;"
    )
    say(
        "read = against the lidar truth as read; floor = the truth's own error in the source's"
        " units (read - floor ~ own"
    )
    say("when the source is independent of the truth). Intervals: 95 % by resampling drives.")
    for kind in ("vx", "wz"):
        source_tables(b, facts, win, stats, acf, sets, kind, unit[kind])
    fit = fits(b, stats)
    sqr = sqrt_rate(b, win, stats, sets, fo, fl)
    return split, {"ST": stats, "AC": acf, "FIT": fit, "SQ": sqr, "sets": sets}


def source_tables(
    b: Bench,
    facts: dict[str, dict[str, Any]],
    win: dict[str, Any],
    stats: dict[tuple[str, str], dict[Any, Any]],
    acf: dict[tuple[str, str], dict[str, Any]],
    sets: dict[str, list[str]],
    kind: str,
    unit: tuple[float, str],
) -> None:
    """The per-source NEES table of one quantity, every set."""
    say = b.say
    k_u, u = unit
    say("")
    say(f"--- {'FORWARD SPEED vx' if kind == 'vx' else 'TURN RATE vyaw'} ---")
    names = (f"{b.old} TRAIN", f"{b.live} TRAIN", "all TRAIN", f"{b.old} TEST", f"{b.live} TEST")
    for setname in names:
        st = stats[(setname, kind)]
        ac = acf[(setname, kind)]
        say(f"  [{setname}]")
        for p, s in enumerate(TRIPLE[kind]):
            if s == "V" and setname.startswith("all"):
                continue  # VIO never pooled across rates
            r1, rps = st[(s, "w1")], st[(s, f"ps_{s}")]
            if not r1["n"]:
                continue
            pt, lo, hi = ac["point"][p], ac["lo"][p], ac["hi"][p]
            ml = 30
            rate = sum(facts[n]["cnt"].get(s, 0) for n in sets[setname]) / sum(
                facts[n]["movsec"] for n in sets[setname]
            )
            rms = boot_mean(
                {
                    n: own2(win[n][kind]["w1"], p)
                    for n in sets[setname]
                    if kind in win[n] and "w1" in win[n][kind]
                },
                sets[setname],
            )
            say(
                f"    {NAMES[s]:6s} per 1 s: n {r1['n']:4d} ({r1['ndr']} drives), claimed sigma of"
                f" the mean p50 {r1['sd'] * k_u:5.2f} {u}; NEES own {f3(r1['own'])} -> multiplier"
                f" {f3(sqt(r1['own']))}; read {f3(r1['read'])}, floor {f3(r1['floor'])}"
            )
            say(
                f"    {'':6s} per 1 s ratio of means (mean own e^2 / mean claimed sigma^2): NEES"
                f" {f3(r1['own_rm'])} -> multiplier {f3(sqt(r1['own_rm']))}; own rms"
                f" {f3(sqt(rms), k_u)} {u}"
            )
            say(
                f"    {'':6s} per sample: n {rps['n']:5d}, claimed p50 {rps['sd'] * k_u:5.2f} {u};"
                f" NEES own {f3(rps['own'])}; read {f3(rps['read'])}, floor {f3(rps['floor'])}"
            )
            say(
                f"    {'':6s} per 2 s own {f3(st[(s, 'w2')]['own'])} (n {st[(s, 'w2')]['n']}), per"
                f" 5 s own {f3(st[(s, 'w5')]['own'])} (n {st[(s, 'w5')]['n']})"
            )
            say(
                f"    {'':6s} error autocorrelation (0.1 s bins, hat): rho 0.1/0.2/0.5/1/2 s"
                f" {pt[1]:+.2f}/{pt[2]:+.2f}/{pt[5]:+.2f}/{pt[10]:+.2f}/{pt[20]:+.2f}; tau_int"
                f" {pt[ml + 1]:.2f} s [{lo[ml + 1]:.2f}, {hi[ml + 1]:.2f}] (window"
                f" {pt[ml + 2]:.0f} bins) -> {1 / pt[ml + 1]:.1f} independent samples/s vs"
                f" {rate:.1f} counted; NEES 1 s / sample {r1['own'][0] / rps['own'][0]:.1f}"
            )
            if s == "V":
                up = st[("V", "upper")]
                say(
                    f"    {'':6s} upper bound per 1 s (the wheels' error counted as the VIO's):"
                    f" NEES {f3(up)} -> multiplier <= {f3(sqt(up))}"
                )
            if s == "R" and ("Rs", "w1") in st:
                say(
                    f"    {'':6s} AS STAMPED (what the EKF fuses, 140 ms late): NEES own per 1 s"
                    f" {f3(st[('Rs', 'w1')])} -> multiplier {f3(sqt(st[('Rs', 'w1')]))} (ratio of"
                    f" means {f3(sqt(st[('Rs', 'w1', 'rm')]))}); per 2 s {f3(st[('Rs', 'w2')])},"
                    f" per 5 s {f3(st[('Rs', 'w5')])}"
                )
            if (s, "vwt") in st:
                say(
                    f"    {'':6s} cross-check, hat.py's (V, W, truth) hat per 1 s: NEES own"
                    f" {f3(st[(s, 'vwt')])}"
                )
        mom = st["M"]
        third = "rf2o" if kind == "vx" else "gyro"
        say(
            f"    4-source per 1 s ({u}^2 -> rms): truth own {f3(sqt(mom['e_T']), k_u)} {u}; VIO"
            f" {f3(sqt(mom['e_V']), k_u)}, wheels {f3(sqt(mom['e_W']), k_u)}, {third}"
            f" {f3(sqt(mom['e_3']), k_u)}; truth-wheels covariance"
            f" {f3(mom['c_WT'], k_u * k_u, '6.3f')} {u}^2, truth-{third}"
            f" {f3(mom['c_3T'], k_u * k_u, '6.3f')}"
        )


def fits(b: Bench, stats: dict[tuple[str, str], dict[Any, Any]]) -> dict[str, dict[str, Any]]:
    """The tuned multipliers on TRAIN (VIO on the live group's, the rest on all)."""
    say, fl = b.say, b.fps(b.live)
    say("")
    say(RULE)
    say(
        f"TUNED MULTIPLIERS (sigma x m; m = sqrt(per-1-s NEES own) on TRAIN). VIO from {fl} fps"
        " TRAIN only; wheels and rf2o"
    )
    say("from all TRAIN (same sensors). Resolved = the NEES interval wholly above 0.")
    say(
        "rf2o's from its claims AS STAMPED (the EKF fuses it 140 ms late; the replay cannot"
        " restamp it). Beside it the"
    )
    say(
        "ratio-of-means estimate (mean own e^2 / mean claimed sigma^2: the same NEES weighted by"
        " the claim), added after"
    )
    say(
        "the first run because the pre-registered mean of ratios is dominated by the windows with"
        " the smallest claims;"
    )
    say(
        "an item resolved only by it is replayed as EXPLORATORY and does not enter the"
        " pre-registered verdict."
    )
    fit = {}
    for it, (label, kind, s, grp) in ITEMS.items():
        setname = f"{b.live} TRAIN" if grp == "live" else "all TRAIN"
        st = stats[(setname, kind)]
        r = st[("Rs", "w1")] if s == "R" else st[(s, "w1")]["own"]
        rm = st[("Rs", "w1", "rm")] if s == "R" else st[(s, "w1")]["own_rm"]
        m, mrm = sqt(r), sqt(rm)
        ok, okrm = r[1] > 0, rm[1] > 0
        fit[it] = {"m": m, "resolved": ok, "label": label, "m_rm": mrm, "resolved_rm": okrm}
        tst = stats[(f"{b.live} TEST", kind)]
        tests = tst[("Rs", "w1")] if s == "R" else tst[(s, "w1")]["own"]
        say(
            f"  {it:3s} {label:40s}: m {f3(m)} {'resolved' if ok else 'UNRESOLVED'}; ratio of"
            f" means {f3(mrm)} {'resolved' if okrm else 'UNRESOLVED'}; ({fl} fps TEST, not used:"
            f" {f3(sqt(tests))})"
        )
    gy = stats[("all TRAIN", "wz")][("G", "w1")]["own"]
    say(f"  (gyro, not tuned: per-1-s NEES own on all TRAIN {f3(gy)} -> m {f3(sqt(gy))})")
    return fit


def sqrt_rate(
    b: Bench,
    win: dict[str, Any],
    stats: dict[tuple[str, str], dict[Any, Any]],
    sets: dict[str, list[str]],
    fo: int,
    fl: int,
) -> dict[str, Any]:
    """The VIO's per-1-s NEES, live-rate TRAIN over old-rate TRAIN, against the rule's
    prediction (the ratio of the rates)."""
    say = b.say
    pred = fl / fo
    to, tl = f"{b.old} TRAIN", f"{b.live} TRAIN"
    say("")
    say(RULE)
    say(
        f"SQRT(RATE) RULE: the VIO's per-1-s NEES (base claims, no rule) at {fl} fps TRAIN over"
        f" {fo} fps TRAIN; the rule predicts"
    )
    say(f"{pred:.1f} (sigma x{math.sqrt(pred):.2f}). Independent drive resampling per group.")

    def verdict(lo: float, hi: float) -> str:
        if lo > 1 and hi >= pred:
            return "SUPPORTED"
        if hi < pred or lo > pred:
            return f"CONTRADICTED (excludes {pred:g})"
        return "UNDETERMINED"

    sqr: dict[str, Any] = {}
    for kind in ("vx", "wz"):
        name = "vx" if kind == "vx" else "vyaw"
        a = stats[(to, kind)][("V", "w1")]["own"]
        bb = stats[(tl, kind)][("V", "w1")]["own"]
        a2 = boot_mean(
            {
                n: own2(win[n][kind]["w1"], 0) / win[n][kind]["w1"][:, 1] ** 2
                for n in sets[to]
                if "w1" in win[n][kind]
            },
            sets[to],
            seed=2,
        )
        lo, hi = np.nanpercentile(bb[3] / a2[3], [2.5, 97.5])
        pa = stats[(to, kind)][("V", "ps_V")]["own"]
        pb = stats[(tl, kind)][("V", "ps_V")]["own"]
        pa2 = boot_mean(
            {
                n: own2(win[n][kind]["ps_V"], 0) / win[n][kind]["ps_V"][:, 1] ** 2
                for n in sets[to]
                if "ps_V" in win[n][kind]
            },
            sets[to],
            seed=2,
        )
        plo, phi = np.nanpercentile(pb[3] / pa2[3], [2.5, 97.5])
        sqr[kind] = (bb[0] / a[0], lo, hi, verdict(lo, hi))
        ra = stats[(to, kind)][("V", "w1")]["own_rm"]
        rb = stats[(tl, kind)][("V", "w1")]["own_rm"]
        ra2 = boot_ratio(
            {n: own2(win[n][kind]["w1"], 0) for n in sets[to] if "w1" in win[n][kind]},
            {n: win[n][kind]["w1"][:, 1] ** 2 for n in sets[to] if "w1" in win[n][kind]},
            sets[to],
            seed=2,
        )
        rlo, rhi = np.nanpercentile(rb[3] / ra2[3], [2.5, 97.5])
        sqr[kind + "_rm"] = (rb[0] / ra[0], rlo, rhi, verdict(rlo, rhi))
        say(
            f"  {name} ratio of means per 1 s: {fo} fps {f3(ra)}, {fl} fps {f3(rb)}: ratio"
            f" {rb[0] / ra[0]:.2f} [{rlo:.2f}, {rhi:.2f}] -> {verdict(rlo, rhi)}"
        )
        say(
            f"  {name}: per 1 s {fo} fps {f3(a)}, {fl} fps {f3(bb)}: ratio {bb[0] / a[0]:.2f}"
            f" [{lo:.2f}, {hi:.2f}] -> {verdict(lo, hi)}; per sample {fo} fps {f3(pa)}, {fl} fps"
            f" {f3(pb)} (ratio {pb[0] / pa[0]:.2f} [{plo:.2f}, {phi:.2f}])"
        )
        sqr[kind + "_ps"] = (pb[0] / pa[0], plo, phi)
    return sqr


# ------------------------------------------------------------------------- replays and scores
def b_arms(b: Bench, fit: dict[str, dict[str, Any]]) -> dict[str, dict[str, float]]:
    """The B arms from the resolved TRAIN multipliers: one per item and B with all of them;
    variance factors on the base claims."""
    var = {it: fit[it]["m"][0] ** 2 for it in fit if fit[it]["resolved"]}
    arms = {f"B_{it}": {it: v} for it, v in var.items()}
    if len(var) > 1:
        arms["B"] = dict(var)
    for name, items in arms.items():
        about = "base x " + ", ".join(f"{it} var x{v:.3f}" for it, v in items.items())
        b.arms[name] = Arm(name, (), about)
    return arms


def rewrite_job(job: tuple[str, Path, Path, WheelNoiseLaw, dict[str, float]]) -> str:
    """A B arm's copy of one drive: the base copy x the arm's variance factors."""
    arm, src, dst, law, items = job
    if dst.exists():
        return f"{arm} {dst.name} exists"
    vio = (items.get("V", 1.0), items.get("Y", 1.0))
    wheels = (
        (items.get("Wv", 1.0), items.get("Ww", 1.0)) if ("Wv" in items or "Ww" in items) else None
    )
    c = rewrite_bag(src, dst, law, vio, wheels, False, items.get("R", 1.0))
    return f"{arm} {dst.name} {c}"


def replay_job(b: Bench, arm: str, n: str) -> ReplayJob:
    """One arm on one drive, as replay.py runs it."""
    a = b.arms[arm]
    return ReplayJob(
        arm=arm,
        n=n,
        bag=b.drives.drives[n].bag,
        bag_dir=b.work.bags(a.bags),
        out_dir=b.work.out / arm,
        log=b.work.logs / f"{arm}_{n}.log",
        exclude=a.exclude,
        image=b.image,
        cpus=b.cpus,
    )


def arm_odom(
    b: Bench, arm: str, n: str, facts: dict[str, dict[str, Any]]
) -> tuple[Array, Array] | None:
    """(odometry rows, pose-covariance rows) of /odometry/filtered: 'live' from the drive bag,
    else the replay's (cached as npz)."""
    if arm == "live":
        live: tuple[Array, Array] = facts[n]["live"]
        return live
    p = b.work.cache / "arms" / f"{arm}_{n}.npz"
    bag = b.work.out / arm / f"{n}.bag"
    if not bag.exists():
        return None
    if p.exists() and p.stat().st_mtime > max(f.stat().st_mtime for f in bag.iterdir()):
        z = np.load(p)
        return z["o"], z["c"]
    o, c = read_filtered(bag)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, o=o, c=c)
    return o, c


def report_arms(b: Bench, facts: dict[str, dict[str, Any]], split: Split) -> dict[str, Any]:
    """Score every arm; the truth's floor, the drift-vs-noise table, the filter's consistency and
    the paired differences."""
    say, live, old = b.say, b.live, b.old
    test = {
        f"{live} TEST": split.sel(live, "TEST"),
        f"{old} TEST": split.sel(old, "TEST"),
        "all TEST": split.sel("all", "TEST"),
    }
    parts = {n: {k: floor_parts(facts[n], k) for k in ("s1", "m1")} for n in split.kept}
    scores: dict[str, Scores] = defaultdict(dict)
    for arm in ["live", *b.arms]:
        for n in split.kept:
            od = arm_odom(b, arm, n, facts)
            if od is not None and len(od[0]) > 10:
                scores[arm][n] = score_arm(facts[n], od)
    floors_k: Floors = {k: {n: parts[n][k] for n in parts} for k in ("s1", "m1")}

    def met(arm: str, ds: list[str]) -> tuple[dict[str, float], list[dict[str, float]]]:
        ds = [n for n in ds if n in scores[arm]]
        return boot_metrics_k(scores[arm], floors_k, ds)

    say("")
    say(RULE)
    say(
        "THE TRUTH'S OWN ERROR in the relative-motion domain (the 4-source hat on each window:"
        " VIO, wheels, rf2o (aligned)"
    )
    say(
        "integrate their own vx, vy, wz; yaw: gyro, VIO, wheels), rms per window [95 % by"
        " drives]; n = windows with all"
    )
    say(
        "sources >= 80 % covered. Across is printed but NOT subtracted (not separable: see"
        " floors())."
    )
    for name, ds in test.items():
        for kind, lab in (("s1", "per moving second"), ("m1", "per 1 m piece")):
            fl = floors(floors_k[kind], ds, across=True)
            draws = np.random.default_rng(RNG_SEED).integers(0, len(ds), (NBOOT, len(ds)))
            fbs = np.array([floors(floors_k[kind], ds, dr, across=True) for dr in draws])
            nw = sum(int(np.isfinite(floors_k[kind][n][:, 0]).sum()) for n in ds)
            ny = sum(int(np.isfinite(floors_k[kind][n][:, 8]).sum()) for n in ds)
            lo = np.nanpercentile(fbs, 2.5, 0)
            hi = np.nanpercentile(fbs, 97.5, 0)
            say(
                f"  {name:10s} {lab:17s} (n {nw}/{ny}): along {100 * sq(fl[0]):5.2f}"
                f" [{100 * sq(lo[0]):5.2f}, {100 * sq(hi[0]):5.2f}] cm, across"
                f" {100 * sq(fl[1]):5.2f} [{100 * sq(lo[1]):5.2f}, {100 * sq(hi[1]):5.2f}] cm, yaw"
                f" {math.degrees(sq(fl[2])):5.2f} [{math.degrees(sq(lo[2])):5.2f},"
                f" {math.degrees(sq(hi[2])):5.2f}] deg"
            )
    say("")
    say(RULE)
    say(
        "DRIFT VS NOISE per source combination (replays of the board's EKF, rewind on, base"
        " claims; live = the recorded"
    )
    say(
        "/odometry/filtered, each drive's own configuration). RPE@1s: translation error per"
        " moving second, net of the"
    )
    say(
        "truth's along floor (across as read: an upper bound), as % of the mean distance per"
        " second; RPE@1m: per 1 m"
    )
    say(
        "piece, the per-second floor subtracted (a lower bound of the piece's: conservative), %"
        " of the piece; yaw net deg per"
    )
    say(
        "second / per metre; end: the goal's end position error cm mean (median) and |end"
        " heading| deg; vNEES: the filter's"
    )
    say(
        "forward error per second net of the floor over its own twist variance (1.0 ="
        " consistent); [95 % by drives]."
    )
    for name, ds in test.items():
        say(f"  [{name}: {len(ds)} drives {' '.join(ds)}]")
        for arm in ("live", "W", "WG", "WGR", "WGV", "A", "A2"):
            if not all(n in scores[arm] for n in ds):
                miss = [n for n in ds if n not in scores[arm]]
                if len(miss) == len(ds):
                    continue
                say(f"    ({arm} missing on {miss})")
            m, bs = met(arm, ds)
            desc = "recorded /odometry/filtered" if arm == "live" else b.arms[arm].about
            s1, m1, end = ci(bs, "s1_pct_net"), ci(bs, "m1_pct_net"), ci(bs, "end_mean")
            say(
                f"    {arm:4s} RPE@1s {m['s1_pct_net']:5.1f} % [{s1[0]:5.1f}, {s1[1]:5.1f}] (read"
                f" {m['s1_pct_read']:4.1f}), yaw {m['s1_yaw_net']:4.2f} deg; RPE@1m"
                f" {m['m1_pct_net']:4.1f} % [{m1[0]:4.1f}, {m1[1]:4.1f}] (read"
                f" {m['m1_pct_read']:4.1f}), yaw {m['m1_yaw_net']:4.2f} deg; end"
                f" {m['end_mean']:4.1f} ({m['end_median']:4.1f}) cm [{end[0]:4.1f},"
                f" {end[1]:4.1f}], |head| {m['head_abs']:4.2f}; vNEES {m['s1_vnees']:6.3f}  --"
                f" {desc}"
            )
        m0, _ = met("A", ds)
        say(
            f"    (mean distance per moving second {100 * m0['s1_len']:.1f} cm, per piece"
            f" {100 * m0['m1_len']:.1f} cm; the floor per second {100 * m0['s1_floor']:.2f} cm,"
            f" per piece {100 * m0['m1_floor']:.2f} cm)"
        )
    say("")
    say(RULE)
    say(
        "FILTER CONSISTENCY of /odometry/filtered (A and live, all TEST): pose-covariance growth"
        " P(t1) - P(t0) against the"
    )
    say(
        "pose error, 3 dof, per dof (1.0 = consistent), per second / per 1 m piece / over the"
        " goal; vNEES as above."
    )
    for arm in ("live", "A", "WGR", "W"):
        if not any(n in scores[arm] for n in test["all TEST"]):
            continue
        m, bs = met(arm, test["all TEST"])
        v = ci(bs, "s1_vnees")
        say(
            f"  {arm:4s} pose NEES per s {m['s1_pnees']:.4f}, per piece {m['m1_pnees']:.4f}, goal"
            f" {m['end_nees']:.4f}; vNEES per s {m['s1_vnees']:.3f} [{v[0]:.3f}, {v[1]:.3f}], per"
            f" piece {m['m1_vnees']:.3f}"
        )
    say("")
    say(RULE)
    say(
        "PAIRED DIFFERENCES, arm - A, same drives and bootstrap draws (negative = better): end"
        " position cm mean, |end"
    )
    say(
        "heading| deg, RPE@1m net cm, RPE@1s net cm, yaw per 1 m deg, vNEES; [95 % by drives];"
        " drives closer at the end."
    )
    paired: dict[tuple[str, str], dict[str, Any]] = {}
    for name in (f"{live} TEST", f"{old} TEST"):
        ds = test[name]
        say(f"  [{name}]")
        for arm in [a for a in b.arms if a != "A" and a not in EXPLORE]:
            dsa = [n for n in ds if n in scores[arm] and n in scores["A"]]
            if not dsa:
                continue
            pdif = paired_k(scores["A"], scores[arm], floors_k, dsa)
            paired[(name, arm)] = pdif
            say(
                f"    {arm:5s} end {pm(pdif, 'end_mean')} cm ({pdif['wins'][0]}/{pdif['wins'][1]}"
                f" closer); |head| {pm(pdif, 'head_abs')}; RPE@1m {pm(pdif, 'm1_net', 100)} cm;"
                f" RPE@1s {pm(pdif, 's1_net', 100)} cm; yaw/m {pm(pdif, 'm1_yaw_net')}; vNEES"
                f" {pm(pdif, 's1_vnees')}"
            )
            if name == f"{live} TEST":
                say("          per drive end A -> arm cm: " + per_drive(pdif))
    return {"SC": scores, "PD": paired, "test": test, "Fk": floors_k}


def pm(pdif: dict[str, Any], k: str, s: float = 1.0) -> str:
    """A paired difference with its interval, signed."""
    return f"{pdif[k][0] * s:+6.2f} [{pdif[k][1] * s:+6.2f}, {pdif[k][2] * s:+6.2f}]"


def per_drive(pdif: dict[str, Any]) -> str:
    """Each drive's end error, A -> arm."""
    return ", ".join(f"{n} {a:.1f}->{b:.1f}" for n, (a, b) in pdif["per_drive"].items())


EXPLORE = ("Ar",)  # the sqrt(rate) rule at the live rate: both VIO variances x the rates' ratio


def verdicts(
    b: Bench,
    facts: dict[str, dict[str, Any]],
    split: Split,
    res: dict[str, Any],
    scored: dict[str, Any],
    var_b: dict[str, dict[str, float]],
) -> None:
    """The pre-registered decision per item on the live group's TEST drives, and the information
    shares claimed and under B."""
    say, fl = b.say, b.fps(b.live)
    say("")
    say(RULE)
    say(
        f"VERDICT for today's live configuration ({fl} fps, rewind on, vio_sigma_scale 1.75,"
        " vio_yaw_sigma_scale 2.5, sqrt"
    )
    say(f"rule off), decided on {fl} fps TEST by the rule fixed in the docstring:")
    for it, f in res["FIT"].items():
        pd = scored["PD"].get((f"{b.live} TEST", f"B_{it}"))
        if not f["resolved"]:
            v = "KEEP AS IS (TRAIN multiplier unresolved: its interval reaches <= 0; no replay)"
        elif pd is None:
            v = "KEEP AS IS (no replay)"
        else:
            e = pd["end_mean"]
            worse = [k for k in ("m1_net", "s1_net", "head_abs") if pd[k][1] > 0]
            if e[2] < 0 and not worse:
                v = f"APPLY: end {e[0]:+.2f} cm [{e[1]:+.2f}, {e[2]:+.2f}]"
            else:
                v = (
                    f"KEEP AS IS: end {e[0]:+.2f} cm [{e[1]:+.2f}, {e[2]:+.2f}]"
                    + (" (interval not below 0)" if e[2] >= 0 else "")
                    + (f"; worse with the interval above 0: {worse}" if worse else "")
                )
        say(f"  {it:3s} {f['label']:40s} m {f3(f['m'])} -> {v}")
    say("")
    say(
        f"INFORMATION SHARES on {fl} fps TEST (p50 of the per-second shares / share of the pooled"
        " sums), %:"
    )
    ds = split.sel(b.live, "TEST")
    rows: list[tuple[str, dict[str, float]]] = [("claimed (A, base)", {})]
    rows += [(k, v) for k, v in var_b.items() if k == "B"]
    for name, var in rows:
        sh = shares(facts, ds, var)
        x, w = sh["vx"], sh["wz"]
        say(
            f"  {name:18s} vx VIO {x[0]:4.0f}/{x[3]:4.0f}, wheels {x[1]:4.0f}/{x[4]:4.0f}, rf2o"
            f" {x[2]:4.0f}/{x[5]:4.0f}; vyaw VIO {w[0]:5.1f}/{w[3]:5.1f}, wheels"
            f" {w[1]:4.1f}/{w[4]:4.1f}, gyro {w[2]:5.1f}/{w[5]:5.1f}"
        )


def report_explore(
    b: Bench, split: Split, scored: dict[str, Any], vio: list[str], ruled: list[str]
) -> None:
    """EXPLORATORY, added after the first look (not pre-registered, nothing fitted in it): does
    the VIO's effect on the end error (WGR - A) replicate on the live group's TRAIN drives; the
    sqrt(rate) rule as the relay applies it, Ar - A, on every live-rate drive with a VIO
    (``ruled``: the drives recorded with it)."""
    say, scores, floors_k = b.say, scored["SC"], scored["Fk"]
    pred = b.fps(b.live) / b.fps(b.old)
    say("")
    say(RULE)
    say(
        "EXPLORATORY (added after the first look; not part of the pre-registered verdict; no"
        " number in it is fitted):"
    )
    say(
        "arm - A, paired as above. WGR = no VIO; Ar = the sqrt(rate) rule on (VIO vx/vy and yaw"
        f" variances x{pred:g}, i.e. the"
    )
    say(
        f"relay's x{math.sqrt(pred):.2f} on both sigma scales at {b.fps(b.live)} fps;"
        f" {ruled[0]}-{ruled[-1]} were recorded so)."
    )
    groups = (
        (f"{b.live} TRAIN", [n for n in vio if int(n) % 2]),
        (f"{b.live} TEST", [n for n in vio if not int(n) % 2]),
        (f"{b.live} all", vio),
        (f"{b.old} TEST", split.sel(b.old, "TEST")),
    )
    for name, ds in groups:
        for arm in ("WGR", "Ar"):
            dsa = [n for n in ds if n in scores[arm] and n in scores["A"]]
            if not dsa:
                continue
            pdif = paired_k(scores["A"], scores[arm], floors_k, dsa)
            say(
                f"  [{name}, {len(dsa)} drives] {arm:4s} - A: end {pm(pdif, 'end_mean')} cm"
                f" ({pdif['wins'][0]}/{pdif['wins'][1]} closer); |head| {pm(pdif, 'head_abs')};"
                f" RPE@1m {pm(pdif, 'm1_net', 100)} cm; RPE@1s {pm(pdif, 's1_net', 100)} cm"
            )
            say("      per drive end A -> arm cm: " + per_drive(pdif))


# ------------------------------------------------------------------------- the run
def header(b: Bench, truth_line: str) -> None:
    """Where the numbers come from: the checkout, the EKF, the image, the truth."""
    sha_ = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True, text=True
    ).stdout.strip()
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True
    ).stdout.split("\n")
    b.say(
        f"odom_bench {time.strftime('%Y-%m-%d %H:%M %Z')}; git {sha_}; uncommitted tracked: "
        + ", ".join(x[3:] for x in dirty if x[:2].strip() == "M")
        + f"; set {b.drives.path.relative_to(REPO)} sha256 {sha256(b.drives.path)[:16]}"
    )
    b.say(
        f"EKF {EKF_YAML.relative_to(REPO)} sha256 {sha256(EKF_YAML)[:16]}; image {b.image}; truth"
        f" map {display(b.truth_map)} ({truth_line}); truth heading +{b.drives.heading_deg} deg;"
        f" seed {RNG_SEED}, {NBOOT} resamples"
    )


def display(p: Path) -> str:
    """A path relative to the repository when inside it."""
    try:
        return str(p.relative_to(REPO))
    except ValueError:
        return str(p)


def run(b: Bench, stage: str) -> None:
    """The stages up to ``stage`` (prepare, analyse, all); bench_out.txt in the work dir."""
    say = b.say
    man, det, cfg, truth_line = prepare(b)
    header(b, truth_line)
    say(
        "RECORDED CLAIMS (relay /vo_twist over OpenVINS's own covariance; the yaw ratio is"
        " exactly 2.5^2 on every VIO drive):"
    )
    for n in b.drives.numbers:
        say(
            f"  {n} {b.drives.group(n)} {cfg[n].status}; wheels law live {cfg[n].law_live}; rf2o"
            f" vx var {det[n]['rf2o_cov0']}"
        )
    coasting = [
        n
        for n in b.drives.numbers
        if det[n]["matched"] and det[n]["lin_p50"] > 3 * det[n]["lin_p10"]
    ]
    say(
        "  coasting (the relay's sigma x up to 2.25 while OpenVINS has no update): p50 of the"
        " linear"
        " ratio at the ceiling on " + ", ".join(coasting)
    )
    if stage == "prepare":
        return
    sh = offsets(b, man)
    facts = load_all(b, man, sh)
    split, res = analyse(b, facts, det)
    if stage == "analyse":
        write_out(b)
        return
    var_b = b_arms(b, res["FIT"])
    t_all, t_live = split.sel("all", "TEST"), split.sel(b.live, "TEST")
    jobs = [(arm, n) for arm in ("A", "A2", "W", "WG", "WGR", "WGV") for n in t_all]
    rw: list[tuple[str, str, dict[str, float]]] = []
    for arm, items in var_b.items():
        ds = t_live if set(items) & VIO_ITEMS else t_all
        rw += [(arm, n, items) for n in ds]
        jobs += [(arm, n) for n in ds]
    vio = [n for n in split.sel(b.live) if det[n]["matched"]]
    pred = b.fps(b.live) / b.fps(b.old)
    b.arms["Ar"] = Arm(
        "Ar", (), f"EXPLORATORY: base with the sqrt(rate) rule (VIO variances x{pred:g})"
    )
    rw += [("Ar", n, {"V": pred, "Y": pred}) for n in vio]
    jobs += [("Ar", n) for n in vio]
    jobs += [(arm, n) for arm in ("A", "WGR") for n in vio if int(n) % 2]
    rjobs = [
        (
            arm,
            b.work.bags("base") / b.drives.drives[n].bag,
            b.work.bags(arm) / b.drives.drives[n].bag,
            b.law,
            items,
        )
        for arm, n, items in rw
        if not done(b.work.out / arm, n)  # a done replay needs no copy
    ]
    with ProcessPoolExecutor(8) as ex:
        for line in ex.map(rewrite_job, rjobs):
            log("[rewrite]", line)
    for line in replays([replay_job(b, arm, n) for arm, n in jobs], b.parallel):
        log("[replay]", line)
    check = b.drives.replay_check
    say("")
    say(
        "REPLAY CHECK: "
        + check_vio_replay(
            check,
            b.drives.drives[check].bag_dir(b.rec),
            b.work.bags("vr"),
            b.work.logs,
            b.work.out / "WGR" / f"{check}.csv",
        )
    )
    scored = report_arms(b, facts, split)
    verdicts(b, facts, split, res, scored, var_b)
    ruled = [n for n in b.drives.numbers if b.drives.drives[n].sqrt_rule and det[n]["matched"]]
    report_explore(b, split, scored, vio, ruled)
    write_out(b)


def write_out(b: Bench) -> None:
    """The report as <work>/bench_out.txt."""
    out = b.work.root / "bench_out.txt"
    out.write_text("\n".join(b.say.lines) + "\n")
    log(f"[out] {out}")


def bench_for(
    set_path: Path,
    rec: Path,
    work: Path | None,
    truth_map: Path | None,
    image: str,
    cpus: str,
    parallel: int,
) -> Bench:
    """A run's settings: the set, where its drives are, the work directory (default: the
    package's .cache/<set name>)."""
    drives = DriveSet.load(set_path)
    mount = dataclasses.replace(
        LidarMount.from_json(REPO / "config" / "lidar.json"),
        yaw_offset_deg=drives.lidar_yaw_offset_deg,
    )
    root = work or Path(__file__).resolve().parent / ".cache" / drives.name
    return Bench(
        drives=drives,
        rec=rec,
        work=Work(root),
        truth_map=truth_map or drives.truth_map,
        image=image,
        cpus=cpus,
        parallel=parallel,
        law=WheelNoiseLaw.from_json(REPO / "config" / "base.json"),
        mount=mount,
        say=Report(),
        arms=base_arms(),
    )
