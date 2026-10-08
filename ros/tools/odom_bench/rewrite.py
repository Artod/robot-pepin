"""The drive bags' claimed covariances rewritten: normalised to the base configuration, or scaled
by a tuned multiplier for a replay arm."""

from __future__ import annotations

import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.wheel_noise import WheelNoiseLaw

from .drives import DriveSet

ODOM_REST = 0.001  # /odom's at-rest cov[0] (the base bridge's constant)
WHEEL_RATE_HZ = 50  # /odom's sample rate: each sample carries 50 sigma^2 of the law
WITHHELD = 1e5  # a /vo_twist whose cov[0] is at least this carries no linear velocity


def law_variances(law: WheelNoiseLaw, wz: float) -> tuple[float, float]:
    """The shipped law's per-sample variances (vx, vyaw) at the measured yaw rate."""
    s_v, s_w = law.sigmas(wz)
    return WHEEL_RATE_HZ * s_v**2, WHEEL_RATE_HZ * s_w**2


def scale_twist_cov(cov: npt.NDArray[np.float64], kl: float, ky: float) -> None:
    """In place: the vx/vy block x kl, vyaw x ky, the vx/vy - vyaw cross terms x sqrt(kl ky)
    (the relay's scaling, on variances); a yaw-only or withheld twist only its vyaw."""
    full = cov[0] < WITHHELD
    for i in range(6):
        for j in range(6):
            a, b = i in (0, 1), j in (0, 1)
            if a and b and full:
                cov[i * 6 + j] *= kl
            elif (a and j == 5) or (b and i == 5):
                if full:
                    cov[i * 6 + j] *= math.sqrt(kl * ky)
            elif i == 5 and j == 5:
                cov[35] *= ky


@dataclass(frozen=True)
class Claims:
    """The factors that bring one drive's recorded claims to the base configuration."""

    vio_lin_var: float
    vio_yaw_var: float
    status: str
    law_live: bool


def classify(det: dict[str, dict[str, Any]], drives: DriveSet) -> dict[str, Claims]:
    """Per drive, the VIO variance factors and whether the wheels already carry the law.

    The reference drives of each group run the base scales; their median ratio (relay /vo_twist
    over OpenVINS's own) is the group's yardstick. A drive recorded with the sqrt(rate) rule has
    both variances x fps / 10, removed here; any drive more than 25 % off its group's yardstick
    (after the rule) stops the bench. The linear ratio is read at its 10th percentile: the relay's
    coasting multiplier (up to x2.25 sigma while OpenVINS has no update) is live behaviour of
    every configuration and stays in the claims.
    """
    ref = {}
    for g in drives.groups:
        ns = [n for n in drives.numbers if drives.group(n) == g and drives.drives[n].reference]
        ref[g] = (
            float(np.median([det[n]["lin_p10"] for n in ns])),
            float(np.median([det[n]["yaw_p50"] for n in ns])),
        )
    out = {}
    for n, d in det.items():
        rule = drives.fps(n) / 10 if drives.drives[n].sqrt_rule else 1.0
        rl, ry = ref[drives.group(n)]
        if d["matched"] == 0:
            kl = ky = 1.0
            status = "no VIO in the bag"
        else:
            el, ey = d["lin_p10"] / (rl * rule), d["yaw_p50"] / (ry * rule)
            if not (0.75 < el < 1.25 and 0.75 < ey < 1.25):
                raise SystemExit(
                    f"{n}: recorded VIO claim off every known configuration: lin x{el:.2f}"
                    f" yaw x{ey:.2f} of the {drives.group(n)} reference with rule {rule}"
                )
            kl = ky = 1.0 / rule
            status = f"lin {el:.2f} yaw {ey:.2f} of the reference" + (
                f", rule x{math.sqrt(rule):.2f} removed" if rule > 1 else ""
            )
        law_ok = (
            d["odom_moving"] > 0
            and abs(d["law_ratio_p50"] - 1) < 1e-3
            and d["law_ratio_p90"] < 1e-3
        )
        out[n] = Claims(kl, ky, status, law_ok)
    return out


def rewrite_bag(
    src: Path,
    dst: Path,
    law: WheelNoiseLaw,
    vio: tuple[float, float] = (1.0, 1.0),
    wheels: tuple[float, float] | None = None,
    restamp_law: bool = False,
    rf2o: float = 1.0,
) -> dict[str, int]:
    """Copy a drive bag message by message (rosbags, sqlite3) with its claims changed:
    /vo_twist by the VIO variance factors (lin, yaw); /odom's moving samples (cov[0] not the rest
    constant) first re-stamped to the law when ``restamp_law``, then x the (vx, vyaw) variance
    factors; /odom_laser's whole twist covariance x ``rf2o``."""
    from rosbags.highlevel import AnyReader
    from rosbags.rosbag2 import Writer

    from .bags import typestore

    ts = typestore()
    n = {"vo_twist": 0, "odom": 0, "odom_laser": 0}
    tmp = dst.with_name(dst.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with AnyReader([src], default_typestore=ts) as r, Writer(tmp, version=8) as w:
        conns = {
            c.id: w.add_connection(
                c.topic, c.msgtype, typestore=ts, offered_qos_profiles=c.ext.offered_qos_profiles
            )
            for c in r.connections
        }
        for c, t, raw in r.messages():
            tp = c.topic
            if (
                (tp == "/vo_twist" and vio != (1.0, 1.0))
                or (tp == "/odom" and (wheels or restamp_law))
                or (tp == "/odom_laser" and rf2o != 1.0)
            ):
                m = r.deserialize(raw, c.msgtype)
                cov = m.twist.covariance.copy()
                if tp == "/vo_twist":
                    scale_twist_cov(cov, *vio)
                    n["vo_twist"] += 1
                elif tp == "/odom":
                    if abs(cov[0] - ODOM_REST) > 1e-9:
                        if restamp_law:
                            cov[0], cov[35] = law_variances(law, m.twist.twist.angular.z)
                        if wheels:
                            cov[0] *= wheels[0]
                            cov[35] *= wheels[1]
                            cov[5] *= math.sqrt(wheels[0] * wheels[1])
                            cov[30] *= math.sqrt(wheels[0] * wheels[1])
                        n["odom"] += 1
                else:
                    cov *= rf2o
                    n["odom_laser"] += 1
                m.twist.covariance = cov
                raw = ts.serialize_cdr(m, c.msgtype)
            w.write(conns[c.id], t, raw)
    tmp.rename(dst)
    return n
