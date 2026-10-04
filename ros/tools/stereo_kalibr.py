#!/usr/bin/env python3
"""The stereo head's camera calibration with Kalibr, on the RAW eyes: record, bag, report, apply.

``ros/calib_stereo.sh`` is the wrapper (docs/stereo_calibration.md). The steps:

    record --out DIR --seconds 90   ustreamer's side-by-side MJPEG copied as it arrives
                                    (DIR/stereo.mjpeg, every part's headers kept, the format
                                    of the drives' camera clips), with a tag count per eye
                                    every two seconds while the board is waved
    bag DIR --hz 4                  the sharpest frame of every 1/hz window, cut into the two
                                    upright eyes (pepin.stereo.SideBySide, the module's own
                                    turn undone), mono8, into a ROS 1 bag for Kalibr
                                    (DIR/kalibr/stereo.bag, /cam0/image_raw + /cam1/image_raw,
                                    one stamp per pair); needs rosbags (uv run --with)
    report DIR                      Kalibr's camchain against config/stereo_calibration.json:
                                    per-eye reprojection, intrinsics, distortion, the bar, the
                                    rectified eye's turn, the range bias the current file has
                                    if Kalibr is right, and the acceptance (exit 1 if refused)
    apply DIR [--force]             the camchain into config/stereo_calibration.json (the file
                                    before kept once as stereo_calibration.pre-kalibr-<day>.json)
                                    and camera.json's eye + head_imu carried through the turn

Stamps do not matter here: both eyes of a pair are halves of ONE transport frame (two global
shutters, one capture), so a pair is synchronous whatever the stamp, and nothing in a camera
calibration reads time across pairs. The capture stamp is used when the stream carries it.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import IO, Any

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
CONFIG = REPO / "config"
CLIP = "stereo.mjpeg"
BAG = "stereo.bag"
STATUS_S = 2.0
BOTH_EYES_TAGS = 4  # OpenCV's count above which an eye is said to see the board
OPEN_TRIES = 6
GUIDANCE = """\
Wave the AprilGrid in front of the head, by hand, for {seconds:.0f} s:
  - distance {near:.2f}-{far:.2f} m from the lens ({tag_mm:.0f} mm tags: under ~20 px, beyond
    {far:.2f} m, Kalibr finds no tags); both eyes must see the whole grid most of the time;
  - tilt it +-30 deg about both axes (top towards / away, left towards / away), not only flat;
  - visit all parts of the picture: centre, the four corners, the edges (the distortion lives
    in the corners);
  - move SLOWLY and pause a beat at each pose (blur is the enemy; the sharpest frame of every
    quarter second is kept);
  - the tag counts below are a rough check: OpenCV's detector finds fewer tags than Kalibr's
    (6 where Kalibr reads the whole grid); 0 in an eye means that eye does not see the board."""


# ---- record --------------------------------------------------------------------------------
class Tee:
    """A stream whose every read is also written to a file, byte for byte."""

    def __init__(self, source: IO[bytes], sink: IO[bytes]) -> None:
        self._source, self._sink = source, sink

    def read(self, size: int = -1) -> bytes:
        """Read from the source and copy what came into the sink."""
        data = self._source.read(size)
        if data:
            self._sink.write(data)
        return data


def tag_counter() -> Any:
    """A function image -> number of AprilTag 36h11 markers OpenCV finds (Kalibr's grid has a
    two-bit border), or ``None`` when this OpenCV has no aruco."""
    import cv2

    try:
        dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
        params = cv2.aruco.DetectorParameters()
        params.markerBorderBits = 2
        detector = cv2.aruco.ArucoDetector(dictionary, params)
    except AttributeError:
        return None

    def count(image: Any) -> int:
        _corners, ids, _rejected = detector.detectMarkers(image)
        return 0 if ids is None else len(ids)

    return count


def git_commit() -> str:
    """The checkout's commit and whether it is dirty."""
    try:
        sha = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()  # fmt: skip
        dirty = subprocess.run(
            ["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def _cmd_record(args: argparse.Namespace) -> int:
    import cv2

    from pepin.camera import CameraConfig
    from pepin.mjpeg import capture_time, has_grab, parts
    from pepin.stereo import SideBySide

    cfg = CameraConfig.load(CONFIG / "camera.json", name=args.camera, board=args.host)
    if cfg.rig is None:
        print(f"refused: camera {cfg.name} is not a stereo rig")
        return 2
    rig = cfg.rig
    fx = 530.0
    try:
        from pepin.stereo import StereoCalibration

        fx = StereoCalibration.load(rig.calibration_path(CONFIG)).k_left[0][0]
    except (OSError, KeyError, ValueError):
        pass
    tag_px_min = 20.0
    far = fx * args.tag / tag_px_min
    print(GUIDANCE.format(seconds=args.seconds, near=0.25, far=far, tag_mm=args.tag * 1e3))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    split = SideBySide(rig.upside_down)
    count = tag_counter()
    print(f"stream {cfg.stream}; recording into {out / CLIP}")
    response = None
    for attempt in range(1, OPEN_TRIES + 1):
        try:
            response = urllib.request.urlopen(cfg.stream, timeout=5.0)
            break
        except (urllib.error.URLError, OSError) as exc:
            print(f"stream open {attempt}/{OPEN_TRIES} failed: {exc}")
            if attempt == OPEN_TRIES:
                return 1
            time.sleep(2.0)
    assert response is not None
    frames = grabbed = 0
    sampled = both = 0
    size: tuple[int, int] | None = None
    first: float | None = None
    last: float | None = None
    start = time.monotonic()
    next_status = start + STATUS_S
    try:
        with response, (out / CLIP).open("wb") as sink:
            for headers, body in parts(Tee(response, sink)):  # type: ignore[arg-type]
                frames += 1
                grabbed += has_grab(headers)
                stamp = capture_time(headers, "grab")
                if stamp is not None:
                    first = stamp if first is None else first
                    last = stamp
                now = time.monotonic()
                if now >= next_status or size is None:
                    frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_GRAYSCALE)
                    if frame is not None:
                        size = (frame.shape[1], frame.shape[0])
                        if size != (rig.frame_width, rig.frame_height):
                            print(
                                f"refused: the stream's frame is {size[0]}x{size[1]}, the rig"
                                f" says {rig.frame_width}x{rig.frame_height}"
                            )
                            return 2
                        tags = ""
                        if count is not None:
                            left, right = split.eyes(frame)
                            n_l, n_r = count(left), count(right)
                            sampled += 1
                            both += n_l >= BOTH_EYES_TAGS and n_r >= BOTH_EYES_TAGS
                            tags = f"  tags L {n_l:2d} R {n_r:2d}"
                        elapsed = now - start
                        print(
                            f"  {elapsed:5.1f} s  {frames:5d} frames"
                            f" ({frames / max(elapsed, 1e-3):4.1f}/s){tags}",
                            flush=True,
                        )
                    next_status = now + STATUS_S
                if now - start >= args.seconds:
                    break
    except KeyboardInterrupt:
        print("stopped by ^C: the clip so far is kept")
    elapsed = time.monotonic() - start
    meta = {
        "stream": cfg.stream,
        "camera": cfg.name,
        "frame": [rig.frame_width, rig.frame_height],
        "upside_down": rig.upside_down,
        "seconds": round(elapsed, 1),
        "frames": frames,
        "grab_stamped": grabbed,
        "tag_m": args.tag,
        "status_checks_both_eyes_seeing_tags": [both, sampled],
        "date_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_commit(),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    span = f", stamps span {last - first:.1f} s" if first is not None and last is not None else ""
    print(
        f"recorded {frames} frames in {elapsed:.1f} s ({frames / max(elapsed, 1e-3):.1f}/s),"
        f" {grabbed} with the capture stamp{span}; status checks with tags in both eyes:"
        f" {both} of {sampled}"
    )
    if frames == 0:
        return 1
    return 0


# ---- bag -----------------------------------------------------------------------------------
def read_clip(path: Path) -> tuple[list[float], list[bytes]]:
    """Every part of a raw MJPEG copy: its stamp (the capture, else the send, else the part's
    index at 15 frames a second) and its JPEG bytes. Stamps are made strictly increasing."""
    from pepin.mjpeg import capture_time, parts

    stamps: list[float] = []
    bodies: list[bytes] = []
    with path.open("rb") as stream:
        for i, (headers, body) in enumerate(parts(stream)):
            stamp = capture_time(headers, "grab")
            if stamp is None:
                stamp = 1_700_000_000.0 + i / 15.0
            if stamps and stamp <= stamps[-1]:
                stamp = stamps[-1] + 1e-6
            stamps.append(stamp)
            bodies.append(body)
    return stamps, bodies


def sharpness(gray: Any) -> float:
    """The variance of the Laplacian of a half-size picture: high when sharp."""
    import cv2

    small = cv2.resize(gray, (gray.shape[1] // 2, gray.shape[0] // 2), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(small, cv2.CV_64F).var())


def _cmd_bag(args: argparse.Namespace) -> int:
    import cv2
    from rosbags.rosbag1 import Writer
    from rosbags.typesys import Stores, get_typestore

    from pepin.kalibr_stereo import TOPICS, sharpest_per_window
    from pepin.stereo import SideBySide

    rec = Path(args.rec)
    meta = json.loads((rec / "meta.json").read_text())
    stamps, bodies = read_clip(rec / CLIP)
    if not bodies:
        print(f"no frames in {rec / CLIP}")
        return 1
    split = SideBySide(bool(meta["upside_down"]))
    width, height = (int(v) for v in meta["frame"])
    sharp: list[float] = []
    for body in bodies:
        frame = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_GRAYSCALE)
        if frame is None or (frame.shape[1], frame.shape[0]) != (width, height):
            sharp.append(-1.0)
            continue
        left, right = split.eyes(frame)
        sharp.append(min(sharpness(left), sharpness(right)))
    chosen = [i for i in sharpest_per_window(stamps, sharp, 1.0 / args.hz) if sharp[i] >= 0.0]
    work = rec / "kalibr"
    work.mkdir(exist_ok=True)
    target = work / BAG
    scratch = work / (BAG + ".tmp")
    scratch.unlink(missing_ok=True)
    typestore = get_typestore(Stores.ROS1_NOETIC)
    image_t = typestore.types["sensor_msgs/msg/Image"]
    header_t = typestore.types["std_msgs/msg/Header"]
    time_t = typestore.types["builtin_interfaces/msg/Time"]
    with Writer(scratch) as writer:
        connections = [
            writer.add_connection(topic, image_t.__msgtype__, typestore=typestore)
            for topic in TOPICS
        ]
        for seq, i in enumerate(chosen):
            frame = np.asarray(
                cv2.imdecode(np.frombuffer(bodies[i], np.uint8), cv2.IMREAD_GRAYSCALE)
            )
            ns = round(stamps[i] * 1e9)
            stamp = time_t(sec=ns // 1_000_000_000, nanosec=ns % 1_000_000_000)
            for eye, conn, frame_id in zip(
                split.eyes(frame), connections, ("cam0", "cam1"), strict=True
            ):
                eye = np.ascontiguousarray(eye)
                msg = image_t(
                    header=header_t(seq=seq, stamp=stamp, frame_id=frame_id),
                    height=eye.shape[0],
                    width=eye.shape[1],
                    encoding="mono8",
                    is_bigendian=0,
                    step=eye.shape[1],
                    data=eye.reshape(-1),
                )
                writer.write(conn, ns, typestore.serialize_ros1(msg, image_t.__msgtype__))
    scratch.replace(target)
    span = stamps[-1] - stamps[0]
    kept = [sharp[i] for i in chosen]
    print(
        f"{len(bodies)} frames over {span:.1f} s -> {len(chosen)} pairs at {args.hz:g} Hz"
        f" (sharpness of the kept: median {np.median(kept):.0f}, min {min(kept):.0f};"
        f" of all: median {np.median([s for s in sharp if s >= 0]):.0f}) -> {target}"
    )
    return 0


# ---- report / apply ------------------------------------------------------------------------
def _show(path: Path) -> str:
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def _calibration_path() -> Path:
    from pepin.camera import CameraConfig

    cfg = CameraConfig.load(CONFIG / "camera.json")
    if cfg.rig is None:
        raise SystemExit("the active camera is not a stereo rig")
    return cfg.rig.calibration_path(CONFIG)


def _result(rec: Path) -> tuple[Any, list[float], int, str]:
    """Kalibr's result for a recording: the new calibration, the reprojection errors, the views
    used, the tag size."""
    from pepin.kalibr_stereo import (
        calibration_from_camchain,
        load_camchain,
        reprojection_rms,
        views_used,
    )

    work = rec / "kalibr"
    camchain = work / "stereo-camchain.yaml"
    results = work / "stereo-results-cam.txt"
    if not camchain.exists():
        raise SystemExit(f"no Kalibr result: {camchain} (see {work / 'kalibr.log'})")
    text = results.read_text() if results.exists() else ""
    log = (
        (work / "kalibr.log").read_text(errors="replace") if (work / "kalibr.log").exists() else ""
    )
    errors = reprojection_rms(text) or reprojection_rms(log)
    used = views_used(log)
    tag = ""
    april = work / "april.yaml"
    if april.exists():
        for line in april.read_text().splitlines():
            if line.startswith("tagSize:"):
                tag = line.split(":", 1)[1].strip()
    calibration = calibration_from_camchain(
        load_camchain(camchain),
        rms_px=max(errors) if errors else float("nan"),
        date=datetime.now(UTC).strftime("%Y-%m-%d"),
        method="kalibr",
        views=used,
    )
    return calibration, errors, used, tag


def _eye_line(name: str, k_old: Any, k_new: Any) -> str:
    o = np.asarray(k_old, np.float64)
    n = np.asarray(k_new, np.float64)
    cells = []
    for label, (r, c) in (("fx", (0, 0)), ("fy", (1, 1)), ("cx", (0, 2)), ("cy", (1, 2))):
        cells.append(f"{label} {o[r, c]:7.2f} -> {n[r, c]:7.2f} ({n[r, c] - o[r, c]:+6.2f})")
    return f"  {name:5s} " + "  ".join(cells)


def _camera_blocks() -> tuple[dict[str, Any], dict[str, Any]]:
    data = json.loads((CONFIG / "camera.json").read_text())
    return dict(data["stereo"].get("eye", {})), dict(data["stereo"].get("head_imu", {}))


def report(rec: Path) -> tuple[Any, Any, Any, int]:
    """Print the comparison; answer (new calibration, frame shift, acceptance, exit code)."""
    from pepin.camera import CameraConfig
    from pepin.kalibr_stereo import (
        accept,
        carry_eye,
        carry_head_imu,
        depth_ratio,
        rectification,
        rectified_frame_shift,
        rotation_vector_deg,
    )
    from pepin.stereo import StereoCalibration

    new, errors, used, tag = _result(rec)
    path = _calibration_path()
    old = StereoCalibration.load(path)
    cfg = CameraConfig.load(CONFIG / "camera.json")
    nominal = cfg.rig.baseline_m_nominal if cfg.rig is not None else 0.063
    print(
        f"== Kalibr on {rec.name} (tag {tag or '?'} m, {used} views used) against {path.name}"
        f" ({old.method} {old.date}, rms {old.rms_px:.2f} px)"
    )
    for cam, px in enumerate(errors):
        print(f"  cam{cam} reprojection {px:.3f} px (RMS)")
    print(_eye_line("left", old.k_left, new.k_left))
    print(_eye_line("right", old.k_right, new.k_right))
    print(
        "  left  distortion "
        + " ".join(f"{v:+.4f}" for v in new.d_left[:4])
        + "   (was "
        + " ".join(f"{v:+.4f}" for v in old.d_left[:5])
        + f", {len(old.d_left)} terms)"
    )
    print(
        "  right distortion "
        + " ".join(f"{v:+.4f}" for v in new.d_right[:4])
        + "   (was "
        + " ".join(f"{v:+.4f}" for v in old.d_right[:5])
        + f", {len(old.d_right)} terms)"
    )
    t_old = np.asarray(old.translation_m) * 1e3
    t_new = np.asarray(new.translation_m) * 1e3
    print(
        f"  baseline {old.baseline_m * 1e3:.2f} -> {new.baseline_m * 1e3:.2f} mm (nominal"
        f" {nominal * 1e3:.1f}); t {np.array2string(t_old, precision=2)} ->"
        f" {np.array2string(t_new, precision=2)} mm (right eye <- left)"
    )
    r_old = rotation_vector_deg(old.rotation)
    r_new = rotation_vector_deg(new.rotation)
    r_diff = rotation_vector_deg(np.asarray(new.rotation) @ np.asarray(old.rotation).T)
    print(
        f"  cam1 rotation (deg about x, y, z) {np.array2string(r_old, precision=3)} ->"
        f" {np.array2string(r_new, precision=3)}; difference {np.linalg.norm(r_diff):.3f} deg"
    )
    _r1o, _r2o, p1o, p2o = rectification(old)
    _r1n, _r2n, p1n, p2n = rectification(new)
    print(
        f"  rectified pinhole fx {p1o[0, 0]:.2f} -> {p1n[0, 0]:.2f}, baseline"
        f" {abs(p2o[0, 3]) / p2o[0, 0] * 1e3:.2f} -> {abs(p2n[0, 3]) / p2n[0, 0] * 1e3:.2f} mm"
        " (this OpenCV; camera_stream's 4.6 differs by under 1 px)"
    )
    shift = rectified_frame_shift(old, new)
    print(
        f"  the rectified left eye turns {shift.angle_deg:.3f} deg (about x, y, z:"
        f" {np.array2string(rotation_vector_deg(shift.rotation), precision=3)}); rays left"
        f" {shift.residual_deg * 60:.2f} arcmin apart after it ({shift.rays} rays)"
    )
    print("  the current file read against Kalibr's truth (old depth / true depth):")
    for d in depth_ratio(old, new):
        print(
            f"    at {d.range_m:4.2f} m: {d.median:.3f} (p10 {d.p10:.3f}, p90 {d.p90:.3f}),"
            f" disparity offset {d.offset_px:+.2f} px -> {d.range_m * d.median:.3f} m"
        )
    eye, head_imu = _camera_blocks()
    if eye:
        moved = carry_eye(eye, shift.rotation)
        rpy = ("roll_deg", "pitch_deg", "yaw_deg")
        before = "/".join(f"{float(eye.get(k, 0.0)):+.2f}" for k in rpy)
        after = "/".join(f"{moved[k]:+.2f}" for k in rpy)
        print(f"  camera.json eye (carried): roll/pitch/yaw {before} -> {after} deg")
    if head_imu.get("T_cam_imu"):
        t = carry_head_imu(head_imu["T_cam_imu"], shift.rotation)
        t_before = np.asarray(head_imu["T_cam_imu"], np.float64)[:3, 3] * 1e3
        print(
            "  camera.json head_imu.T_cam_imu (carried): t"
            f" {np.array2string(t_before, precision=1)} ->"
            f" {np.array2string(t[:3, 3] * 1e3, precision=1)} mm,"
            f" rotation {shift.angle_deg:.3f} deg"
        )
    verdict = accept(errors, new.baseline_m, nominal)
    if verdict.accepted:
        print(
            f"ACCEPTED: reprojection {'/'.join(f'{e:.3f}' for e in errors)} px < 0.3,"
            f" baseline {new.baseline_m * 1e3:.2f} mm within 2 mm of {nominal * 1e3:.0f}"
        )
    else:
        print("REFUSED: " + "; ".join(verdict.failures))
    return new, shift, verdict, 0 if verdict.accepted else 1


def _cmd_report(args: argparse.Namespace) -> int:
    return report(Path(args.rec))[3]


def _quiet_matmul() -> Any:
    """numpy's matmul on this Mac's Accelerate raises divide/overflow/invalid warnings on
    finite inputs; the report's arrays are checked for finiteness where it matters."""
    return np.errstate(divide="ignore", over="ignore", invalid="ignore")


KALIBR_IMAGE = (
    "prehensile/kalibr:arm64@sha256:"
    "68d089b2c5514d2f143ce95dd015981bce22e73cfd97edf1ee3fd7ebe41840ef"
)


def _cmd_apply(args: argparse.Namespace) -> int:
    from dataclasses import replace

    from pepin.kalibr_stereo import (
        carry_eye,
        carry_head_imu,
        rotation_vector_deg,
        write_camera_blocks,
    )
    from pepin.stereo import StereoCalibration

    rec = Path(args.rec)
    new, shift, verdict, _code = report(rec)
    if not verdict.accepted and not args.force:
        print("not written (REFUSED above; --force writes it anyway)")
        return 1
    path = _calibration_path()
    old = StereoCalibration.load(path)
    day = datetime.now().strftime("%Y%m%d")
    backup = path.with_name(f"{path.stem}.pre-kalibr-{day}.json")
    if not backup.exists():
        backup.write_text(path.read_text())
        print(f"kept the previous file as {_show(backup)}")
    _new, errors, used, tag = _result(rec)
    turn = np.array2string(rotation_vector_deg(shift.rotation), precision=3)
    reprojection = "/".join(f"{e:.3f}" for e in errors)
    new = replace(
        new,
        method=f"kalibr_calibrate_cameras pinhole-radtan x2 ({KALIBR_IMAGE})",
        board=(
            f"6x6 AprilGrid, tag {tag} m, spacing 0.3 | raw eyes of {rec.name}"
            f" (ros/calib_stereo.sh, hand-held), {used} views | reprojection {reprojection} px"
            f" (RMS) | radtan [k1 k2 p1 p2] as plumb_bob with k3 0 | replaced {old.method}"
            f" {old.date} (rms {old.rms_px:.2f}, baseline {old.baseline_m * 1e3:.2f} mm), kept as"
            f" {backup.name}; the rectified left eye turned {shift.angle_deg:.3f} deg ({turn}),"
            " camera.json's eye and head_imu carried through it"
        ),
    )
    new.write(path)
    print(f"wrote {_show(path)}")
    if not args.no_carry:
        eye, head_imu = _camera_blocks()
        note = (
            f"CARRIED {datetime.now().strftime('%Y-%m-%d')} through the Kalibr stereo recalibration"
            f" ({rec.name}): the rectified left eye turned {shift.angle_deg:.3f} deg ({turn} about"
            " x, y, z), the link-to-optical rotation became R_old shift^T"
        )
        new_eye = None
        if eye:
            moved = carry_eye(eye, shift.rotation)
            before = {k: eye.get(k) for k in ("roll_deg", "pitch_deg", "yaw_deg")}
            new_eye = {
                **eye,
                **moved,
                "note": f"{note}; before: {json.dumps(before)}. {eye.get('note', '')}",
            }
        new_imu = None
        if head_imu.get("T_cam_imu"):
            t = carry_head_imu(head_imu["T_cam_imu"], shift.rotation)
            new_imu = {
                **head_imu,
                "T_cam_imu": [[round(float(v), 7) for v in row] for row in t],
                "note": f"{note} ([shift 0; 0 1] T_cam_imu). {head_imu.get('note', '')}",
            }
        write_camera_blocks(CONFIG / "camera.json", new_eye, new_imu)
        print("carried config/camera.json's stereo.eye and stereo.head_imu through the turn")
    print(
        "next (Artem's call, the robot's side):\n"
        "  camera_stream rebuilds the rectifier from the new file by itself (mtime poll), but the\n"
        "  eye and head_imu frames are read at start:   ros/laptop.sh kick camera_stream\n"
        "  the stereo law was fitted under the old rectification:\n"
        "    mv ros/maps/depth_law_stereo.json ros/maps/depth_law_stereo.pre-kalibr.json\n"
        "    ros/laptop.sh kick depth_stream\n"
        "  VIO (if up; at rest): ros/laptop.sh vio down && ros/laptop.sh vio   (vio_config\n"
        "    rewrites the camchain from the new rectified pinhole and the carried T_cam_imu)\n"
        "  RTAB-Map takes the new P from /camera/camera_info; a map started before is in the old\n"
        "    rectification. The board: nothing (the hand-eye eye x T_cam_imu is unchanged)."
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """The command line."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    rec = sub.add_parser("record")
    rec.add_argument("--out", required=True)
    rec.add_argument("--seconds", type=float, default=90.0)
    rec.add_argument("--host", default="10.0.0.187")
    rec.add_argument("--camera", default=None)
    rec.add_argument("--tag", type=float, default=0.025)
    bag = sub.add_parser("bag")
    bag.add_argument("rec")
    bag.add_argument("--hz", type=float, default=4.0)
    rep = sub.add_parser("report")
    rep.add_argument("rec")
    rep.add_argument("--config", default=None, help="the config directory (default: the repo's)")
    app = sub.add_parser("apply")
    app.add_argument("rec")
    app.add_argument("--config", default=None, help="the config directory (default: the repo's)")
    app.add_argument("--force", action="store_true")
    app.add_argument("--no-carry", action="store_true")
    args = parser.parse_args(argv)
    global CONFIG
    if getattr(args, "config", None):
        CONFIG = Path(args.config).resolve()
    commands = {"record": _cmd_record, "bag": _cmd_bag, "report": _cmd_report, "apply": _cmd_apply}
    with _quiet_matmul():
        return commands[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
