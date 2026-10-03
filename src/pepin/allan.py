"""Allan deviation of a parked IMU recording, and the noise block the VIO and Kalibr are given.

The head IMU's white noise and bias random walk are not on any datasheet worth trusting (a clone,
WHO_AM_I 0x72, read through an ESP32 beside a backlit display): they are measured, parked, from
the samples themselves (vio.md section 2b). Record 2-3 h with the cart parked and the head at
home (``python -m pepin.head_imu record ...`` keeps head_server's raw lines), then::

    uv run python -m pepin.allan parked.jsonl.gz                 # the table and the fit
    uv run python -m pepin.allan parked.jsonl.gz --plot allan.png --inflate 10

The overlapping Allan deviation of each axis (Riley's estimator on the integrated signal, on
octave-spaced cluster times), and from it the three numbers that matter: the white noise density
N (the -1/2 slope read at tau = 1 s; rad/s/sqrt(Hz) or m/s^2/sqrt(Hz)), the random walk K (the
+1/2 slope read at tau = 3 s; rad/s^2/sqrt(Hz) or m/s^3/sqrt(Hz)) and the bias instability B (the
curve's floor / 0.664). The printed block is config/head_imu.json's ``noise`` (Kalibr's names),
INFLATED by ``--inflate`` for the VIO (OpenVINS advises 10-20x for unmodelled errors; the raw
values are printed beside it). A recording with gaps is refused past ``--max-gap-share``: Allan
assumes one sample per period.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
# The -1/2 and +1/2 slopes are read where they dominate: white noise below a few seconds, the
# random walk above tens of seconds; the windows are where each fit looks for its slope.
WHITE_TAU_S = (0.02, 2.0)
WALK_TAU_S = (20.0, 600.0)
BIAS_INSTABILITY_FACTOR = 0.664  # sqrt(2 ln 2 / pi)


def cluster_times(n: int, rate_hz: float, per_octave: int = 4) -> Array:
    """Cluster times from one sample period to a ninth of the record, ``per_octave`` a doubling,
    each a whole number of samples (deduplicated)."""
    max_m = max(1, n // 9)
    steps = np.unique(
        np.round(np.logspace(0, math.log2(max_m), int(math.log2(max_m) * per_octave) + 1, base=2.0))
    )
    return steps[steps >= 1] / rate_hz


def overlapping_adev(signal: Sequence[float] | Array, rate_hz: float, taus: Array) -> Array:
    """The overlapping Allan deviation of a rate signal (one sample per period) at ``taus``:
    sigma^2(tau) = <(theta[k+2m] - 2 theta[k+m] + theta[k])^2> / (2 tau^2), theta the integral."""
    data = np.asarray(signal, dtype=np.float64)
    theta = np.concatenate([[0.0], np.cumsum(data)]) / rate_hz
    out = np.full(len(taus), np.nan)
    for i, tau in enumerate(taus):
        m = round(tau * rate_hz)
        if m < 1 or 2 * m >= len(theta):
            continue
        d = theta[2 * m :] - 2.0 * theta[m:-m] + theta[: -2 * m]
        out[i] = math.sqrt(float(np.mean(d * d)) / (2.0 * tau * tau))
    return out


@dataclass(frozen=True)
class AllanFit:
    """One axis: the white noise density, the random walk and the bias instability (NaN where
    the record does not reach the slope's window)."""

    white: float
    walk: float
    instability: float
    floor_tau_s: float


def fit(taus: Array, adev: Array) -> AllanFit:
    """The three numbers from one curve: N = median(adev sqrt(tau)) over WHITE_TAU_S, K =
    median(adev sqrt(3 / tau)) over WALK_TAU_S, B = min(adev) / 0.664."""
    ok = np.isfinite(adev)
    t, a = taus[ok], adev[ok]
    white = t[(t >= WHITE_TAU_S[0]) & (t <= WHITE_TAU_S[1])]
    walk = t[(t >= WALK_TAU_S[0]) & (t <= WALK_TAU_S[1])]
    n = float(np.median(a[np.isin(t, white)] * np.sqrt(white))) if len(white) else math.nan
    k = float(np.median(a[np.isin(t, walk)] * np.sqrt(3.0 / walk))) if len(walk) else math.nan
    floor = int(np.argmin(a)) if len(a) else 0
    b = float(a[floor] / BIAS_INSTABILITY_FACTOR) if len(a) else math.nan
    return AllanFit(n, k, b, float(t[floor]) if len(t) else math.nan)


def gap_share(stamps: Sequence[float] | Array, rate_hz: float) -> float:
    """The share of sample intervals longer than 1.5 periods."""
    t = np.asarray(stamps, dtype=np.float64)
    if len(t) < 2:
        return 1.0
    return float(np.mean(np.diff(t) > 1.5 / rate_hz))


def noise_block(
    gyro: Sequence[AllanFit], accel: Sequence[AllanFit], inflate: float
) -> dict[str, float]:
    """config/head_imu.json's ``noise`` from the axes' fits: the worst axis of each, times
    ``inflate``."""

    def worst(values: list[float]) -> float:
        finite = [v for v in values if math.isfinite(v)]
        return max(finite) * inflate if finite else math.nan

    return {
        "gyro_noise_density": worst([f.white for f in gyro]),
        "gyro_random_walk": worst([f.walk for f in gyro]),
        "accel_noise_density": worst([f.white for f in accel]),
        "accel_random_walk": worst([f.walk for f in accel]),
    }


def load(path: Path) -> tuple[Array, Array, Array, float]:
    """A recording as (stamps, gyro Nx3, accel Nx3, rate): head_server lines (``pepin.head_imu
    record``, .jsonl or .jsonl.gz) or an .npz with t, gyro, accel, rate_hz."""
    if path.suffix == ".npz":
        data = np.load(path)
        return data["t"], data["gyro"], data["accel"], float(data["rate_hz"])
    from pepin.head_imu import parse_config, read_recording

    rate = math.nan
    opener = __import__("gzip").open if str(path).endswith(".gz") else open
    with opener(path, "rt") as stream:
        for line in stream:
            try:
                config = parse_config(json.loads(line))
            except ValueError:
                continue
            if config is not None:
                rate = config.rate_hz
                break
    samples = list(read_recording(path))
    t = np.array([s.t_mono_s for s in samples])
    gyro = np.array([s.gyro for s in samples])
    accel = np.array([s.accel for s in samples])
    if not math.isfinite(rate) and len(t) > 1:
        rate = float(1.0 / np.median(np.diff(t)))
    return t, gyro, accel, rate


def main(argv: list[str] | None = None) -> int:
    """The table, the fit and the noise block of one recording."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("recording", type=Path)
    parser.add_argument("--inflate", type=float, default=10.0)
    parser.add_argument("--max-gap-share", type=float, default=0.001)
    parser.add_argument("--plot", type=Path, default=None)
    args = parser.parse_args(argv)
    t, gyro, accel, rate = load(args.recording)
    share = gap_share(t, rate)
    hours = (t[-1] - t[0]) / 3600.0 if len(t) > 1 else 0.0
    print(f"{len(t)} samples at {rate:.1f} Hz, {hours:.2f} h, {100 * share:.3f} % gaps")
    if share > args.max_gap_share:
        print(f"refused: gaps over {100 * args.max_gap_share:.2f} % (Allan wants one per period)")
        return 3
    taus = cluster_times(len(t), rate)
    fits: dict[str, list[AllanFit]] = {"gyro": [], "accel": []}
    curves: dict[str, Array] = {}
    for kind, data in (("gyro", gyro), ("accel", accel)):
        for axis, name in enumerate("xyz"):
            curve = overlapping_adev(data[:, axis] - np.mean(data[:, axis]), rate, taus)
            curves[f"{kind}_{name}"] = curve
            f = fit(taus, curve)
            fits[kind].append(f)
            print(
                f"{kind} {name}: white {f.white:.3g}, random walk {f.walk:.3g}, bias instability"
                f" {f.instability:.3g} (floor at {f.floor_tau_s:.0f} s)"
            )
    raw = noise_block(fits["gyro"], fits["accel"], 1.0)
    block = noise_block(fits["gyro"], fits["accel"], args.inflate)
    print("measured (worst axis):", json.dumps({k: float(f"{v:.3g}") for k, v in raw.items()}))
    print(f"config/head_imu.json noise (x{args.inflate:g}):")
    print(json.dumps({k: float(f"{v:.3g}") for k, v in block.items()}, indent=2))
    if args.plot is not None:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for ax, kind, unit in ((axes[0], "gyro", "rad/s"), (axes[1], "accel", "m/s^2")):
            for name in "xyz":
                ax.loglog(taus, curves[f"{kind}_{name}"], label=name)
            ax.set_xlabel("tau (s)")
            ax.set_ylabel(f"Allan deviation ({unit})")
            ax.set_title(f"head IMU {kind}, {hours:.1f} h parked")
            ax.grid(True, which="both", alpha=0.3)
            ax.legend()
        figure.tight_layout()
        figure.savefig(args.plot, dpi=120)
        print(f"wrote {args.plot}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
