"""The drive bags read with rosbags: the recorded claims, the bench's facts, a replay's output."""

from __future__ import annotations

import functools
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.wheel_noise import WheelNoiseLaw

from .rewrite import ODOM_REST, WITHHELD, law_variances

Array = npt.NDArray[np.float64]
ODOMS = ("/odometry/filtered", "/odom", "/zupt", "/odom_laser")
PCOV = (0, 1, 5, 6, 7, 11, 30, 31, 35)  # the planar entries of a pose covariance


@functools.cache
def typestore() -> Any:
    """ROS 2 Jazzy's message types (rosbags); fails with the way to run the bench without it."""
    try:
        from rosbags.typesys import Stores, get_typestore
    except ImportError as e:
        raise SystemExit(
            "the bench reads bags with rosbags: uv run --with rosbags==0.11.5 python -m"
            " ros.tools.odom_bench ..."
        ) from e
    return get_typestore(Stores.ROS2_JAZZY)


def stamp(h: Any) -> float:
    """A std_msgs/Header's stamp in seconds."""
    return float(h.stamp.sec + 1e-9 * h.stamp.nanosec)


def yaw_q(q: Any) -> float:
    """Yaw of a geometry_msgs/Quaternion."""
    return math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))


def twist_cov(cv: Any) -> list[float]:
    """A twist covariance's vx, vy and vyaw variances."""
    return [cv[0], cv[7], cv[35]]


def detect_claims(n: str, bag: Path, law: WheelNoiseLaw) -> dict[str, Any]:
    """The recorded claims of one drive: the relay's scales (its /vo_twist covariance over
    OpenVINS's odomimu velocity covariance at the same stamp: the xy trace over the IMU frame's
    3-D trace, yaw over z), the wheels' cov[0] against the law at each moving sample's |w|, and
    rf2o's vx variance."""
    from rosbags.highlevel import AnyReader

    om: dict[float, tuple[float, float]] = {}
    vt: list[tuple[float, float, float, float]] = []
    od: list[tuple[float, float, float]] = []
    lz: list[float] = []
    with AnyReader([bag], default_typestore=typestore()) as r:
        keep = {"/vo_twist", "/ov_msckf/odomimu", "/odom", "/odom_laser"}
        for c, _t, raw in r.messages(connections=[c for c in r.connections if c.topic in keep]):
            m = r.deserialize(raw, c.msgtype)
            cv = m.twist.covariance
            if c.topic == "/ov_msckf/odomimu":
                om[round(stamp(m.header), 4)] = (cv[0] + cv[7] + cv[14], cv[35])
            elif c.topic == "/vo_twist":
                vt.append((stamp(m.header), cv[0], cv[7], cv[35]))
            elif c.topic == "/odom":
                od.append((cv[0], cv[35], m.twist.twist.angular.z))
            else:
                lz.append(cv[0])
    lin, yaw = [], []
    keys = np.array(sorted(om))
    for s, c0, c7, c35 in vt:
        if c0 >= WITHHELD or not len(keys):
            continue
        i = int(np.argmin(np.abs(keys - s)))
        if abs(keys[i] - s) > 0.002:
            continue
        o = om[float(keys[i])]
        lin.append((c0 + c7) / o[0])
        yaw.append(c35 / o[1])
    odm = np.array(od) if od else np.zeros((0, 3))
    mv = odm[np.abs(odm[:, 0] - ODOM_REST) > 1e-9] if len(odm) else odm
    lawr = [c0 / law_variances(law, w)[0] for c0, _c35, w in mv]
    lq = np.percentile(lin, [10, 50]) if lin else [math.nan] * 2
    return {
        "n": n,
        "matched": len(lin),
        "lin_p10": float(lq[0]),
        "lin_p50": float(lq[1]),
        "yaw_p50": float(np.median(yaw)) if yaw else math.nan,
        "odom_moving": len(mv),
        "law_ratio_p50": float(np.median(lawr)) if lawr else math.nan,
        "law_ratio_p90": float(np.percentile(np.abs(np.array(lawr) - 1), 90)) if lawr else math.nan,
        "rf2o_cov0": sorted({round(float(x), 9) for x in lz})[:3],
    }


def extract_facts(bag: Path, run_dir: Path) -> str:
    """<run_dir>/facts.npz from a (base) bag: the odometries (stamp, receipt, x, y, yaw, vx, vy,
    wz, twist cov 0/7/35), /odometry/filtered's planar pose covariance, /vo_twist (stamp, receipt,
    vx, vy, wz, cov 0/7/35), /imu/data_raw (stamp, receipt, w), /ov_msckf/poseimu's stamps;
    meta.json the gyro's claimed variance. Cached while newer than the bag."""
    from rosbags.highlevel import AnyReader

    out_p = run_dir / "facts.npz"
    if out_p.exists() and out_p.stat().st_mtime > max(f.stat().st_mtime for f in bag.iterdir()):
        return "cached"
    out: dict[str, list[list[float]]] = defaultdict(list)
    meta: dict[str, Any] = {"bag": bag.name, "gyro_var": None}
    with AnyReader([bag], default_typestore=typestore()) as r:
        keep = set(ODOMS) | {"/vo_twist", "/imu/data_raw", "/ov_msckf/poseimu"}
        for c, t, raw in r.messages(connections=[c for c in r.connections if c.topic in keep]):
            m = r.deserialize(raw, c.msgtype)
            rt, tp = t * 1e-9, c.topic
            if tp in ODOMS:
                p, tw, cv = m.pose.pose, m.twist.twist, m.twist.covariance
                pose = [stamp(m.header), rt, p.position.x, p.position.y, yaw_q(p.orientation)]
                out[tp].append([*pose, tw.linear.x, tw.linear.y, tw.angular.z, *twist_cov(cv)])
                if tp == "/odometry/filtered":
                    pc = m.pose.covariance
                    out[tp + ":pcov"].append([stamp(m.header), *(pc[i] for i in PCOV)])
            elif tp == "/vo_twist":
                tw, cv = m.twist.twist, m.twist.covariance
                twist = [tw.linear.x, tw.linear.y, tw.angular.z]
                out[tp].append([stamp(m.header), rt, *twist, *twist_cov(cv)])
            elif tp == "/imu/data_raw":
                w = m.angular_velocity
                out[tp].append([stamp(m.header), rt, w.x, w.y, w.z])
                if meta["gyro_var"] is None:
                    meta["gyro_var"] = float(m.angular_velocity_covariance[8])
            else:
                out[tp].append([stamp(m.header), rt])
    run_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, Any] = {k: np.array(v, dtype=float) for k, v in out.items()}
    np.savez(out_p, **arrays)
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=1))
    return " ".join(f"{k.split('/')[-1]}:{len(v)}" for k, v in out.items())


def read_filtered(bag: Path) -> tuple[Array, Array]:
    """A replay's /odometry/filtered (rosbag2 directory ``<n>.bag``): the odometry rows of
    ``extract_facts`` and the pose-covariance rows, each sorted by stamp."""
    from rosbags.rosbag2 import Reader

    ts = typestore()
    rows: list[list[float]] = []
    pcov: list[list[float]] = []
    with Reader(bag) as r:
        conns = [c for c in r.connections if c.topic == "/odometry/filtered"]
        for c, t, raw in r.messages(connections=conns):
            m = ts.deserialize_cdr(raw, c.msgtype)
            p, tw, cv, pc = m.pose.pose, m.twist.twist, m.twist.covariance, m.pose.covariance
            pose = [stamp(m.header), t * 1e-9, p.position.x, p.position.y, yaw_q(p.orientation)]
            rows.append([*pose, tw.linear.x, tw.linear.y, tw.angular.z, *twist_cov(cv)])
            pcov.append([stamp(m.header), *(pc[i] for i in PCOV)])
    o, c = np.array(rows, dtype=float), np.array(pcov, dtype=float)
    k = np.argsort(o[:, 0])
    return o[k], c[k]
