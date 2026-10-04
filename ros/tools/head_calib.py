#!/usr/bin/env python3
"""The head's Kalibr camera-IMU calibration around the recording (docs/head_imu_calibration.md):
the exposure check before it, the bag's counts after it, and Kalibr's result read, checked,
compared between runs and written into config/camera.json's ``stereo.head_imu``.

    ros/exposure.sh show | uv run python ros/tools/head_calib.py exposure   # capped? (exit 1: no)
    uv run --with rosbags==0.11.5 python ros/tools/head_calib.py summary BAG   # frames, IMU, gaps
    uv run python ros/tools/head_calib.py meta BAG --camera-stamp grab --stamp-lag 0.09 ...
    uv run python ros/tools/head_calib.py report BAG [BAG2]       # the verdict (exit 1: rejected)
    uv run python ros/tools/head_calib.py apply BAG [BAG2] [--force]   # into config/camera.json

BAG is a recording of ros/calib_record.sh (ros/maps/rec/calib_<UTC>Z); ros/calib_run.sh puts
Kalibr's files into BAG/kalibr/ (``calib-camchain-imucam.yaml``, ``calib-results-imucam.txt``)
and ros/calib_record.sh the recording's facts into BAG/calib_meta.json.

THE DIRECTION: Kalibr's ``T_cam_imu`` (its "T_ci: imu0 to cam0") carries a point in the IMU's
axes into cam0's, the rectified left eye's optical frame: exactly what ``stereo.head_imu``
stores, so it is written as Kalibr gives it; ros/tools/vio_config.py inverts it for OpenVINS.

THE TIME SHIFT: Kalibr's ``timeshift_cam_imu`` (t_imu = t_cam + shift) is measured against the
bag's stamps, which camera_stream dated earlier by its live ``camera_stamp_lag_s``;
``time_offset_s`` is defined at the knob's default (config/knobs.json, vio_config.py), so the
stored value is ``shift - (live lag - default)``.

ACCEPTANCE (vio.md S8): mean reprojection error under 0.5 px in each eye; two runs within
0.5 deg of rotation, 5 mm of translation and 2 ms of time shift. ``apply`` writes the mean of
the runs given and refuses a set that fails, unless ``--force``.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import re
import struct
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"
KALIBR_DIR = "kalibr"
BAG_STEM = "calib"  # ros/calib_run.sh converts into BAG/kalibr/calib.bag; Kalibr names after it
META = "calib_meta.json"
TOPICS = ("/camera/image", "/camera/right/image", "/head/imu")
IMAGE_TOPICS = TOPICS[:2]
IMU_TOPIC = TOPICS[2]
MAX_REPROJECTION_PX = 0.5
MAX_ROTATION_DEG = 0.5
MAX_TRANSLATION_MM = 5.0
MAX_TIME_MS = 2.0
CAPPED_MS = 10.0  # an exposure at or under this counts as capped (vio.md: capped 8)
# The bag's health: what a calibration recording must not have.
MAX_IMU_GAP_S = 0.05
MAX_FRAME_GAP_S = 0.35
MIN_IMU_HZ = 150.0
MIN_FRAME_HZ = 8.0
KALIBR_IMAGE = (
    "prehensile/kalibr:arm64"
    "@sha256:68d089b2c5514d2f143ce95dd015981bce22e73cfd97edf1ee3fd7ebe41840ef"
)

Array = npt.NDArray[np.float64]


# ---- the exposure ---------------------------------------------------------------------------
_CONTROL = re.compile(r"^(?P<name>\w+): value (?P<value>-?\d+)")
EXPOSURE_MANUAL, EXPOSURE_SHUTTER_PRIORITY = 1, 2  # V4L2_EXPOSURE_* (pepin.camera_controls)


def exposure_verdict(show: str, capped_ms: float = CAPPED_MS) -> tuple[bool, str]:
    """Whether ``ros/exposure.sh show``'s lines describe an exposure held at or under
    ``capped_ms`` (manual or shutter priority with the time there), and why in words."""
    values: dict[str, int] = {}
    for line in show.splitlines():
        match = _CONTROL.match(line.strip())
        if match:
            values[match["name"]] = int(match["value"])
    mode = values.get("auto_exposure", values.get("exposure_auto"))
    units = values.get("exposure_time_absolute", values.get("exposure_absolute"))
    if mode is None:
        return False, "the exposure controls were not read (ros/exposure.sh show said nothing)"
    if mode not in (EXPOSURE_MANUAL, EXPOSURE_SHUTTER_PRIORITY):
        return False, (
            f"auto exposure (mode {mode}): the camera picks the time, up to a frame period;"
            " ros/exposure.sh capped 8 (or manual 8) first"
        )
    if units is None:
        return False, "no exposure time control read"
    ms = units * 0.1
    kind = "manual" if mode == EXPOSURE_MANUAL else "shutter priority"
    if ms > capped_ms:
        return False, f"{kind} at {ms:g} ms, over {capped_ms:g} ms: ros/exposure.sh capped 8"
    return True, f"{kind} at {ms:g} ms"


# ---- the bag ----------------------------------------------------------------------------------
def cdr_stamp(raw: bytes) -> float:
    """A message's ``header.stamp`` from its serialised CDR bytes (Image and Imu both begin with
    the header): the 4-byte encapsulation, then sec (int32) and nanosec (uint32)."""
    little = raw[1] == 1
    sec, nanosec = struct.unpack_from("<iI" if little else ">iI", raw, 4)
    return float(sec) + float(nanosec) * 1e-9


@dataclass(frozen=True)
class Stream:
    """One topic of the bag: how many messages, over how long, and its worst gap."""

    topic: str
    count: int
    span_s: float
    max_gap_s: float
    note: str = ""

    @property
    def rate_hz(self) -> float:
        """Messages per second over the span."""
        return (self.count - 1) / self.span_s if self.span_s > 0 else 0.0


def stream_of(topic: str, stamps: Sequence[float], note: str = "") -> Stream:
    """A topic's numbers from its header stamps (in bag order)."""
    ordered = np.sort(np.asarray(stamps, dtype=float))
    gaps = np.diff(ordered)
    return Stream(
        topic,
        len(ordered),
        float(ordered[-1] - ordered[0]) if len(ordered) > 1 else 0.0,
        float(gaps.max()) if len(gaps) else 0.0,
        note,
    )


def problems(streams: dict[str, Stream]) -> list[str]:
    """What makes the recording unfit for Kalibr, in words."""
    out = []
    for topic in TOPICS:
        if topic not in streams or streams[topic].count < 2:
            out.append(f"{topic}: no messages")
    imu = streams.get(IMU_TOPIC)
    if imu is not None and imu.count >= 2:
        if imu.rate_hz < MIN_IMU_HZ:
            out.append(f"{IMU_TOPIC}: {imu.rate_hz:.0f} Hz, under {MIN_IMU_HZ:.0f}")
        if imu.max_gap_s > MAX_IMU_GAP_S:
            out.append(f"{IMU_TOPIC}: a {imu.max_gap_s * 1e3:.0f} ms gap")
    for topic in IMAGE_TOPICS:
        eye = streams.get(topic)
        if eye is None or eye.count < 2:
            continue
        if eye.rate_hz < MIN_FRAME_HZ:
            out.append(f"{topic}: {eye.rate_hz:.1f} Hz, under {MIN_FRAME_HZ:g}")
        if eye.max_gap_s > MAX_FRAME_GAP_S:
            out.append(f"{topic}: a {eye.max_gap_s * 1e3:.0f} ms gap")
    return out


def summary(bag: Path) -> dict[str, Stream]:
    """Every calibration topic's numbers, read from the ROS 2 bag with rosbags (the stamps
    straight from the CDR bytes; the first image fully, for its encoding and size)."""
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_JAZZY)
    stamps: dict[str, list[float]] = {topic: [] for topic in TOPICS}
    notes: dict[str, str] = {}
    with Reader(bag) as reader:
        connections = [c for c in reader.connections if c.topic in stamps]
        for connection, _t, raw in reader.messages(connections=connections):
            stamps[connection.topic].append(cdr_stamp(raw))
            if connection.topic in IMAGE_TOPICS and connection.topic not in notes:
                msg = typestore.deserialize_cdr(raw, connection.msgtype)
                notes[connection.topic] = f"{msg.encoding} {msg.width}x{msg.height}"
    return {
        topic: stream_of(topic, values, notes.get(topic, ""))
        for topic, values in stamps.items()
        if values
    }


# ---- the recording's facts ------------------------------------------------------------------
def stamp_lag_default(config_dir: Path = CONFIG) -> float:
    """camera_stream's ``camera_stamp_lag_s`` default (config/knobs.json); 0 without it."""
    knobs = json.loads((config_dir / "knobs.json").read_text())
    knob = knobs.get("camera_stream", {}).get("camera_stamp_lag_s")
    return float(knob["default"]) if knob else 0.0


def git_state() -> str:
    """The checkout's commit, with ``+dirty`` when tracked files differ from it."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "diff", "--quiet", "HEAD"], check=False
        ).returncode
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return sha + ("+dirty" if dirty else "")


# ---- Kalibr's result --------------------------------------------------------------------------
_REPROJECTION = re.compile(r"Reprojection error \(cam(\d+)\) \[px\]:\s+mean ([-+\d.eE]+|nan)")
_GYRO = re.compile(r"Gyroscope error \(imu0\) \[rad/s\]:\s+mean ([-+\d.eE]+|nan)")
_ACCEL = re.compile(r"Accelerometer error \(imu0\) \[m/s\^2\]:\s+mean ([-+\d.eE]+|nan)")


@dataclass(frozen=True)
class KalibrResult:
    """One run's result: cam0's T_cam_imu and time shift as Kalibr gives them, the mean
    residuals, and the camera_stamp_lag_s the bag was recorded at (None: not known)."""

    name: str
    t_cam_imu: Array
    timeshift_s: float
    reprojection_px: tuple[float, ...]
    gyro_error: float | None
    accel_error: float | None
    stamp_lag_s: float | None
    recorded: str

    def time_offset_s(self, default_lag_s: float) -> float:
        """``stereo.head_imu.time_offset_s``: the shift at the knob's default lag."""
        lag = default_lag_s if self.stamp_lag_s is None else self.stamp_lag_s
        return self.timeshift_s - (lag - default_lag_s)


def read_result(bag: Path) -> KalibrResult:
    """Kalibr's files under ``bag/kalibr`` and the recording's calib_meta.json."""
    import yaml

    folder = bag / KALIBR_DIR
    chain_path = folder / f"{BAG_STEM}-camchain-imucam.yaml"
    text_path = folder / f"{BAG_STEM}-results-imucam.txt"
    if not chain_path.is_file():
        raise SystemExit(f"no Kalibr result in {folder} (ros/calib_run.sh first)")
    chain = yaml.safe_load(chain_path.read_text())
    cam0 = chain["cam0"]
    text = text_path.read_text() if text_path.is_file() else ""
    reprojection = tuple(float(m[2]) for m in _REPROJECTION.finditer(text))
    gyro, accel = _GYRO.search(text), _ACCEL.search(text)
    meta_path = bag / META
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    lag = meta.get("camera_stamp_lag_s")
    return KalibrResult(
        name=bag.name,
        t_cam_imu=np.asarray(cam0["T_cam_imu"], dtype=float),
        timeshift_s=float(cam0.get("timeshift_cam_imu", 0.0)),
        reprojection_px=reprojection,
        gyro_error=float(gyro[1]) if gyro else None,
        accel_error=float(accel[1]) if accel else None,
        stamp_lag_s=None if lag is None else float(lag),
        recorded=str(meta.get("recorded_local", ""))[:10]
        or dt.date.fromtimestamp(chain_path.stat().st_mtime).isoformat(),
    )


def rotation_deg(a: Array, b: Array) -> float:
    """The angle between two transforms' rotations, degrees."""
    relative = a[:3, :3].T @ b[:3, :3]
    return math.degrees(math.acos(max(-1.0, min(1.0, (float(np.trace(relative)) - 1.0) / 2.0))))


def translation_mm(a: Array, b: Array) -> float:
    """The distance between two transforms' translations, millimetres."""
    return float(np.linalg.norm(a[:3, 3] - b[:3, 3])) * 1000.0


@dataclass(frozen=True)
class Agreement:
    """Two runs side by side and whether they agree within vio.md's bounds."""

    rotation_deg: float
    translation_mm: float
    time_ms: float

    @property
    def ok(self) -> bool:
        """Within 0.5 deg, 5 mm and 2 ms."""
        return (
            self.rotation_deg <= MAX_ROTATION_DEG
            and self.translation_mm <= MAX_TRANSLATION_MM
            and abs(self.time_ms) <= MAX_TIME_MS
        )


def agreement(a: KalibrResult, b: KalibrResult, default_lag_s: float) -> Agreement:
    """How far apart two runs are (the time shifts each at the knob's default lag)."""
    return Agreement(
        rotation_deg(a.t_cam_imu, b.t_cam_imu),
        translation_mm(a.t_cam_imu, b.t_cam_imu),
        (a.time_offset_s(default_lag_s) - b.time_offset_s(default_lag_s)) * 1000.0,
    )


def mean_transform(transforms: Sequence[Array]) -> Array:
    """The mean of rigid transforms: the rotations' chordal mean (their sum projected back
    onto SO(3)) and the translations' mean."""
    total = np.sum([t[:3, :3] for t in transforms], axis=0)
    u, _s, vt = np.linalg.svd(total)
    rotation = u @ np.diag([1.0, 1.0, float(np.linalg.det(u @ vt))]) @ vt
    out = np.eye(4)
    out[:3, :3] = rotation
    out[:3, 3] = np.mean([t[:3, 3] for t in transforms], axis=0)
    return out


def failures(results: Sequence[KalibrResult], default_lag_s: float) -> list[str]:
    """Every acceptance check the runs fail, in words; empty when they pass."""
    out = []
    for result in results:
        if not result.reprojection_px:
            out.append(f"{result.name}: no reprojection error in Kalibr's results")
        for cam, error in enumerate(result.reprojection_px):
            if not error <= MAX_REPROJECTION_PX:
                out.append(f"{result.name}: cam{cam} reprojection {error:.3f} px")
    for a, b in pairwise(results):
        agree = agreement(a, b, default_lag_s)
        if not agree.ok:
            out.append(
                f"{a.name} vs {b.name}: {agree.rotation_deg:.2f} deg, {agree.translation_mm:.1f}"
                f" mm, {agree.time_ms:+.1f} ms"
            )
    if len(results) < 2:
        out.append("one run only: procedure D wants two that agree")
    return out


def head_imu_block(
    results: Sequence[KalibrResult], default_lag_s: float, image: str = KALIBR_IMAGE
) -> dict[str, Any]:
    """config/camera.json's ``stereo.head_imu`` from the runs: their mean T_cam_imu as Kalibr
    gives it, their mean time offset at the knob's default lag, the date, how, and the numbers."""
    t = mean_transform([r.t_cam_imu for r in results])
    offset = float(np.mean([r.time_offset_s(default_lag_s) for r in results]))
    runs = ", ".join(r.name for r in results)
    numbers = "; ".join(
        f"{r.name}: reprojection {'/'.join(f'{e:.3f}' for e in r.reprojection_px)} px,"
        f" shift {r.timeshift_s * 1e3:+.2f} ms at camera_stamp_lag_s"
        f" {r.stamp_lag_s if r.stamp_lag_s is not None else default_lag_s:g}"
        for r in results
    )
    pairs = "; ".join(
        f"{a.name} vs {b.name}: {g.rotation_deg:.2f} deg, {g.translation_mm:.1f} mm,"
        f" {g.time_ms:+.2f} ms"
        for a, b in pairwise(results)
        for g in [agreement(a, b, default_lag_s)]
    )
    return {
        "T_cam_imu": [[round(float(v), 7) for v in row] for row in t],
        "time_offset_s": round(offset, 5),
        "date": max(r.recorded for r in results),
        "method": (
            f"kalibr_calibrate_imu_camera ({image}), procedure D (ros/calib_record.sh: the"
            f" neck dance through the gaze arbiter, cart parked) x{len(results)}: {runs}"
            + (" (mean)" if len(results) > 1 else "")
            + "; T_cam_imu as Kalibr gives it (imu0 into the rectified left eye's optical"
            " frame), time_offset_s (t_imu = t_cam + shift) against camera_stream's stamps at"
            f" camera_stamp grab with camera_stamp_lag_s at its default {default_lag_s:g}"
        ),
        "note": numbers + (f"; {pairs}" if pairs else ""),
    }


def _matrix_lines(t: Array) -> list[str]:
    return ["    [" + ", ".join(f"{v:+.6f}" for v in row) + "]" for row in t]


def report(
    results: Sequence[KalibrResult], current: Array | None, default_lag_s: float
) -> list[str]:
    """The verdict in lines: each run's residuals, T_cam_imu and time shift, its distance from
    the block in config/camera.json, the runs' agreement and the acceptance."""
    lines = []
    for r in results:
        errors = ", ".join(f"cam{i} {e:.3f} px" for i, e in enumerate(r.reprojection_px))
        lines.append(
            f"{r.name}: reprojection (mean) {errors or 'not found'}; gyro"
            f" {r.gyro_error if r.gyro_error is not None else float('nan'):.4f} rad/s, accel"
            f" {r.accel_error if r.accel_error is not None else float('nan'):.4f} m/s^2"
        )
        lines.append("  T_cam_imu (imu0 into cam0, the rectified left eye's optical frame):")
        lines += _matrix_lines(r.t_cam_imu)
        lag = r.stamp_lag_s if r.stamp_lag_s is not None else default_lag_s
        lines.append(
            f"  timeshift_cam_imu {r.timeshift_s * 1e3:+.2f} ms (t_imu = t_cam + shift) at"
            f" camera_stamp_lag_s {lag:g}"
            + ("" if r.stamp_lag_s is not None else " (not recorded: the default assumed)")
            + f" -> time_offset_s {r.time_offset_s(default_lag_s) * 1e3:+.2f} ms at the"
            f" default {default_lag_s:g}"
        )
        if current is not None:
            lines.append(
                f"  against config/camera.json's block: {rotation_deg(current, r.t_cam_imu):.2f}"
                f" deg, {translation_mm(current, r.t_cam_imu):.1f} mm"
            )
    for a, b in pairwise(results):
        g = agreement(a, b, default_lag_s)
        lines.append(
            f"{a.name} vs {b.name}: rotation {g.rotation_deg:.2f} deg (<= {MAX_ROTATION_DEG}),"
            f" translation {g.translation_mm:.1f} mm (<= {MAX_TRANSLATION_MM:g}), time"
            f" {g.time_ms:+.2f} ms (<= {MAX_TIME_MS:g}): {'AGREE' if g.ok else 'DISAGREE'}"
        )
    failed = failures(results, default_lag_s)
    lines.append("ACCEPTED" if not failed else "NOT ACCEPTED: " + "; ".join(failed))
    return lines


def current_t_cam_imu(config_dir: Path = CONFIG) -> Array | None:
    """The T_cam_imu config/camera.json's stereo block holds now, if any."""
    data = json.loads((config_dir / "camera.json").read_text())
    block = data.get("stereo", {}).get("head_imu")
    return np.asarray(block["T_cam_imu"], dtype=float) if block else None


# ---- the commands -----------------------------------------------------------------------------
def _cmd_exposure(args: argparse.Namespace) -> int:
    capped, why = exposure_verdict(sys.stdin.read(), args.capped_ms)
    print(f"exposure: {'capped' if capped else 'NOT CAPPED'} ({why})")
    return 0 if capped else 1


def _cmd_summary(args: argparse.Namespace) -> int:
    streams = summary(args.bag)
    for stream in streams.values():
        print(
            f"{stream.topic}: {stream.count} messages over {stream.span_s:.1f} s,"
            f" {stream.rate_hz:.1f} Hz, worst gap {stream.max_gap_s * 1e3:.0f} ms"
            + (f", {stream.note}" if stream.note else "")
        )
    found = problems(streams)
    for problem in found:
        print(f"!! {problem}")
    return 1 if found else 0


def _cmd_meta(args: argparse.Namespace) -> int:
    exposure = args.exposure_file.read_text() if args.exposure_file else ""
    capped, why = exposure_verdict(exposure)
    meta = {
        "bag": args.bag.name,
        "recorded_local": dt.datetime.now().astimezone().isoformat(timespec="seconds"),
        "git": git_state(),
        "camera_stamp": args.camera_stamp,
        "camera_stamp_lag_s": args.stamp_lag,
        "camera_stamp_lag_default_s": stamp_lag_default(),
        "exposure": exposure.strip().splitlines(),
        "exposure_capped": capped,
        "exposure_verdict": why,
        "gaze_slow_deg_s": args.slow_deg_s,
        "gaze_move_timeout_s": args.move_timeout_s,
        "grid": {
            "centre_deg": args.centre,
            "distance_m": args.distance,
            "height_m": args.height,
            "tag_m": args.tag,
        },
        "dance_exit": args.dance_exit,
    }
    path = args.bag / META
    path.write_text(json.dumps(meta, indent=2) + "\n")
    print(f"wrote {path}")
    return 0


def _cmd_report(args: argparse.Namespace) -> int:
    results = [read_result(bag) for bag in args.bags]
    default = stamp_lag_default()
    print("\n".join(report(results, current_t_cam_imu(), default)))
    return 0 if not failures(results, default) else 1


def _cmd_apply(args: argparse.Namespace) -> int:
    sys.path.insert(0, str(REPO / "src"))
    from pepin.camera import write_head_imu

    results = [read_result(bag) for bag in args.bags]
    default = stamp_lag_default(args.config)
    failed = failures(results, default)
    if failed and not args.force:
        print("refused (--force writes it anyway): " + "; ".join(failed))
        return 1
    block = head_imu_block(results, default)
    write_head_imu(args.config / "camera.json", block)
    print(json.dumps(block, indent=2))
    print(f"wrote {args.config / 'camera.json'} stereo.head_imu")
    print(
        "next: uv run python ros/tools/vio_config.py (or ros/laptop.sh vio, which writes it in"
        " its image); ros/laptop.sh kick camera_stream (publishes camera_optical -> head_imu)"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """The subcommands."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("exposure", help="ros/exposure.sh show on stdin: capped or not")
    p.add_argument("--capped-ms", type=float, default=CAPPED_MS)
    p.set_defaults(run=_cmd_exposure)
    p = sub.add_parser("summary", help="the bag's frames, IMU samples and gaps (needs rosbags)")
    p.add_argument("bag", type=Path)
    p.set_defaults(run=_cmd_summary)
    p = sub.add_parser("meta", help="the recording's facts into BAG/calib_meta.json")
    p.add_argument("bag", type=Path)
    p.add_argument("--camera-stamp", required=True)
    p.add_argument("--stamp-lag", type=float, required=True)
    p.add_argument("--exposure-file", type=Path, default=None)
    p.add_argument("--slow-deg-s", type=float, default=None)
    p.add_argument("--move-timeout-s", type=float, default=None)
    p.add_argument("--distance", type=float, default=None)
    p.add_argument("--height", type=float, default=None)
    p.add_argument("--tag", type=float, default=None)
    p.add_argument("--centre", type=float, nargs=2, default=None, metavar=("PAN", "TILT"))
    p.add_argument("--dance-exit", type=int, default=None)
    p.set_defaults(run=_cmd_meta)
    p = sub.add_parser("report", help="Kalibr's result of one or more runs, and the verdict")
    p.add_argument("bags", type=Path, nargs="+")
    p.set_defaults(run=_cmd_report)
    p = sub.add_parser("apply", help="the runs' mean into config/camera.json's stereo.head_imu")
    p.add_argument("bags", type=Path, nargs="+")
    p.add_argument("--force", action="store_true", help="write it although a check fails")
    p.add_argument("--config", type=Path, default=CONFIG)
    p.set_defaults(run=_cmd_apply)
    args = parser.parse_args(argv)
    return int(args.run(args))


if __name__ == "__main__":
    sys.exit(main())
