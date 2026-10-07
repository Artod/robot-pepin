#!/usr/bin/env python3
"""The VIO's configuration for this rig, generated from the repo's one source of each number.

OpenVINS (ros/laptop.sh vio) and Kalibr (the camera-IMU calibration, docs/head_imu_calibration.md)
read YAML files of their own; every number in them already lives in this repo, so they are written
from there and never edited by hand:

* config/stereo_calibration.json -> the RECTIFIED pinhole both eyes share (the pictures every
  consumer of /camera/image sees: fx, fy, cx, cy of the rectified P, no distortion, 800x600) and
  the baseline (cam1 sits +baseline along cam0's optical x). That P is OpenCV's stereoRectify at
  alpha 0, whose focal moves with OpenCV's version: 494.22 px under the laptop image's 4.6 (where
  camera_stream rectifies the pictures) and 495.08 under uv's 4.13 (2026-10-04), so
  ``ros/laptop.sh vio`` writes the files inside its container, with camera_stream's OpenCV, at
  every start; the version is printed into the files;
* config/camera.json's stereo.head_imu -> Kalibr's T_cam_imu (a point in the IMU's axes into the
  rectified LEFT eye's optical frame) and the time shift; OpenVINS wants T_imu_cam, its INVERSE;
* config/head_imu.json -> the IMU's rate on /head/imu and its noise densities.

Written into --out (default ros/maps/vio, mounted in the containers as /maps/vio):

    kalibr_imucam_chain.yaml, kalibr_imu_chain.yaml, estimator_config.yaml   OpenVINS
    camchain.yaml, imu.yaml, april.yaml                                         Kalibr's inputs

    uv run python ros/tools/vio_config.py                          # needs camera.json's head_imu
    uv run python ros/tools/vio_config.py --nominal "x,-z,-y" 0.0 0.03 0.04
                                                                   # before Kalibr: the axis photo
    uv run python ros/tools/vio_config.py --kalibr-only --tag-size 0.0341   # Kalibr's three files

``--nominal`` stands in for the missing head_imu block with a 90 deg permutation (which optical
axis each chip axis points along, as the axis photo shows) and the tape's offset of the chip in
the optical frame; it is printed in every file it shapes. ``--calib-extrinsics`` lets OpenVINS
refine the camera-IMU transform online (a dedicated well-excited session only, vio.md section 3).

REST AND START. OpenVINS's own zero-velocity update is on (:data:`ZUPT`): a camera interval
whose IMU reads rest under the current biases and OpenVINS's own speed is not propagated,
and the biases and the tilt are updated from gravity instead, so hours at rest neither move the
pose nor walk the biases (2026-10-04: with it off, the accel bias walked 0.3-1 m/s^2 over 25 min
to 2.6 h at rest and diverged at the first metre). With the ZUPT on, its static initialiser does
not wait for a jerk either: it starts from ~1 s of stillness. ``--no-zupt`` writes the first
design's (ZUPT off, the init waits for a jerk). ``--dyn-init`` lets OpenVINS initialise IN MOTION
too (a restart while the head turns or the cart drives); still windows keep the static one.

THE CAMERA'S RATE is config/camera.json's active rig's ``rate.fps``, the one number the board's
stream, camera_stream's stamps and the frame gates follow too: ``track_frequency`` is
:data:`TRACK_PER_FPS` times it (OpenVINS skips a frame that arrives sooner than 1 /
track_frequency after the last one it took, so a margin over the rate passes every frame and the
grab stamps' 1-2 ms jitter).

THE TIME SHIFT is head_imu's ``time_offset_s``, measured against camera_stream's stamps dated
``head_imu.stamp_lag_s`` before the grab while the camera ran ``head_imu.camera_fps``. The stamps
now are dated ``L`` before the grab (``--camera-stamp-lag``: ros/laptop.sh vio passes the
live knob of camera_stream; 0 or none is its follow mode, the rate's rule ``lag(period)``), and
the grab itself sits ``lag(period)`` after the picture at the rate now, so the shift written is
``time_offset_s + (L - stamp_lag_s) - (lag(period now) - lag(period then))``: a rate change with
the knob following moves it only by what the rule says the calibration's fixed lag was off by.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt

Array = npt.NDArray[np.float64]
HERE = Path(__file__).resolve()
# The same file runs in a laptop container (ros/laptop.sh vio does), where ros/tools is /tools and
# the repo's pieces sit at ros/laptop.sh's mounts: config/ at /ws/config, src/pepin at
# /ws/pepin_src/pepin, ros/maps at /maps.
IN_CONTAINER = HERE.parent == Path("/tools")
REPO = Path("/ws") if IN_CONTAINER else HERE.parents[2]
SRC = REPO / "pepin_src" if IN_CONTAINER else REPO / "src"
CONFIG = REPO / "config"
DEFAULT_OUT = Path("/maps/vio") if IN_CONTAINER else REPO / "ros/maps/vio"
LEFT_TOPIC = "/camera/image"
RIGHT_TOPIC = "/camera/right/image"
IMU_TOPIC = "/head/imu"
# OpenVINS skips a frame that arrives sooner than 1 / track_frequency after the last it took:
# at 10 fps the camera's 9.5-11.6 Hz passed whole under 15, and 15 under a 30 fps camera took one
# frame in three (2026-10-07). track_frequency is this many times config/camera.json's rate.
TRACK_PER_FPS = 1.5
# OpenVINS's zero-velocity update (UpdaterZeroVelocity::try_update), tried at every frame: it is
# ACCEPTED when the IMU reads rest (the camera interval's gyro and accel residuals under the
# current biases pass chi2 at 95 % times zupt_chi2_multipler, with the densities times
# zupt_noise_multiplier) and OpenVINS's own speed is under zupt_max_velocity, OR when the picture
# stands still (mean track motion between two frames under zupt_max_disparity px, more than 20
# tracks) whatever the filter says. With the x10 densities of config/head_imu.json and multiplier
# 1 the IMU test passes a still head in 99.6-100 % of 0.1 s windows and refuses any neck turning
# faster than ~0.8 deg/s (the fastest window accepted while the encoders moved; the arbiter's
# slowest sweep is 20 deg/s), measured on the two parked recordings of 2026-10-04 (3.8k + 2.9k
# still windows). The picture's path is OFF (0): its update holds the pose but has no velocity row,
# so it froze a wrong speed for good (live 2026-10-04: a respawn's start left 0.28 m/s, accepted by
# the still picture at chi2 256 and kept at rest); through the IMU test a wrong speed or bias is
# refused and the visual updates repair it first. A still picture at rest reads 0.2-0.3 px.
ZUPT = {
    "try_zupt": "true",
    "zupt_chi2_multipler": "1",
    "zupt_max_velocity": "0.05",
    "zupt_noise_multiplier": "1",
    "zupt_max_disparity": "0.0",
    "zupt_only_at_beginning": "false",
}
ZUPT_NOTES = {
    "try_zupt": "rest held by OpenVINS itself; the static init needs no jerk",
    "zupt_max_velocity": "m/s, OpenVINS's own speed: a slow drive is not rest",
    "zupt_noise_multiplier": "a still head passes, a neck over ~0.8 deg/s never (2026-10-04)",
    "zupt_max_disparity": "off, the IMU test alone (a still picture froze a wrong speed)",
    "zupt_only_at_beginning": "all day, a home robot stands for hours",
}
# The first design's (vio.md M4, the board's /zupt owning rest): no ZUPT, the init waits for a jerk
NO_ZUPT = {
    "try_zupt": "false",
    "zupt_chi2_multipler": "1",
    "zupt_max_velocity": "0.1",
    "zupt_noise_multiplier": "10",
    "zupt_max_disparity": "0.5",
    "zupt_only_at_beginning": "false",
}


@dataclass(frozen=True)
class Rig:
    """What the generated files need: the rectified pinhole, the baseline, T_cam_imu, t_d, and
    where those came from (printed into the files)."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    baseline_m: float
    t_cam_imu: Array
    time_offset_s: float
    imu_source: str
    calibration_source: str


AXES = {"x": 0, "y": 1, "z": 2}


def nominal_t_cam_imu(axes: str, offset_m: Sequence[float]) -> Array:
    """T_cam_imu from the axis photo: ``axes`` names, for the chip's x, y and z in turn, the
    optical axis it points along (``"x,-z,-y"``: chip x along optical x, chip y along optical -z,
    chip z along optical -y); ``offset_m`` the chip's origin in the optical frame (m). Refuses
    anything that is not a proper rotation."""
    words = [w.strip().lower() for w in axes.split(",")]
    if len(words) != 3:
        raise ValueError(f"three axes, not {axes!r}")
    rotation = np.zeros((3, 3))
    for chip_axis, word in enumerate(words):
        sign = -1.0 if word.startswith("-") else 1.0
        name = word.lstrip("+-")
        if name not in AXES:
            raise ValueError(f"an axis is x, y or z with a sign, not {word!r}")
        rotation[AXES[name], chip_axis] = sign  # column = the chip axis in optical coordinates
    if not np.isclose(np.linalg.det(rotation), 1.0):
        raise ValueError(f"{axes!r} is not a right-handed rotation (det {np.linalg.det(rotation)})")
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(offset_m, dtype=float)
    return matrix


def nominal_block(axes: str, offset_m: Sequence[float]) -> dict[str, object]:
    """config/camera.json's ``stereo.head_imu`` block for a nominal guess: what camera_stream
    publishes as camera_optical -> head_imu until Kalibr's numbers replace it."""
    return {
        "T_cam_imu": [
            [round(float(v), 6) for v in row] for row in nominal_t_cam_imu(axes, offset_m)
        ],
        "time_offset_s": 0.0,
        "method": f"nominal: axis photo {axes}, tape {list(offset_m)} m (Kalibr pending)",
        "note": "chip axes against the rectified left eye's optical frame, from the photo",
    }


def track_frequency_hz(fps: float) -> float:
    """OpenVINS's track_frequency for a camera running ``fps``."""
    return TRACK_PER_FPS * fps


def with_stamp_lag(
    rig: Rig, lag_s: float | None, config_dir: Path = CONFIG, camera: str | None = None
) -> Rig:
    """The rig with its time shift moved from the stamps ``time_offset_s`` was measured against
    (head_imu's ``stamp_lag_s`` at ``camera_fps``) to the stamps camera_stream publishes now:
    dated ``lag_s`` before the grab, or by the rate's rule when ``lag_s`` is 0 or ``None`` (the
    knob's follow mode), at config/camera.json's rate. A rig without a measured head_imu block is
    left as it is; a block without its reference stamps is refused."""
    sys.path.insert(0, str(SRC))
    from pepin.camera import active_camera, camera_rate

    data = json.loads((config_dir / "camera.json").read_text())
    block = data[active_camera(data, camera)].get("head_imu")
    if not block:
        return rig
    if "stamp_lag_s" not in block or "camera_fps" not in block:
        raise SystemExit(
            "config/camera.json's head_imu has no stamp_lag_s / camera_fps: the stamps its"
            " time_offset_s was measured against are unknown (ros/tools/head_calib.py writes them)"
        )
    rate = camera_rate(config_dir / "camera.json", camera)
    following = lag_s is None or lag_s <= 0.0
    now = rate.grab_lag_s() if following else float(lag_s)  # type: ignore[arg-type]
    then = float(block["stamp_lag_s"])
    truth = rate.grab_lag_s() - rate.grab_lag_s(1.0 / float(block["camera_fps"]))
    shift = (now - then) - truth
    if abs(shift) < 1e-9:
        return rig
    how = "following the rate" if following else "the live camera_stamp_lag_s"
    return replace(
        rig,
        time_offset_s=rig.time_offset_s + shift,
        imu_source=f"{rig.imu_source}; time shift moved {shift * 1e3:+.1f} ms: stamps dated"
        f" {now * 1e3:.1f} ms before the grab ({how}) at {rate.fps:g} fps against"
        f" {then * 1e3:.1f} ms at {float(block['camera_fps']):g} fps when measured",
    )


def camera_fps(config_dir: Path = CONFIG, camera: str | None = None) -> float:
    """config/camera.json's rate for the rig (the active one by default)."""
    sys.path.insert(0, str(SRC))
    from pepin.camera import camera_rate

    return camera_rate(config_dir / "camera.json", camera).fps


def invert(transform: Array) -> Array:
    """The inverse of a rigid 4x4 transform, exactly (R^T, -R^T t)."""
    rotation, translation = transform[:3, :3], transform[:3, 3]
    out = np.eye(4)
    out[:3, :3] = rotation.T
    out[:3, 3] = -rotation.T @ translation
    return out


def load_rig(
    config_dir: Path = CONFIG,
    camera: str | None = None,
    nominal: tuple[str, Sequence[float]] | None = None,
    need_imu: bool = True,
) -> Rig:
    """The rig from the repo's config files; ``nominal`` replaces a missing head_imu block, and
    without ``need_imu`` (Kalibr's inputs, which are what measures it) none is needed."""
    sys.path.insert(0, str(SRC))
    from pepin.camera import CameraConfig
    from pepin.stereo import Rectifier, StereoCalibration

    cfg = CameraConfig.load(config_dir / "camera.json", name=camera)
    if cfg.rig is None:
        raise SystemExit(f"{cfg.name} is not a stereo rig")
    calibration = StereoCalibration.load(cfg.rig.calibration_path(config_dir))
    rectifier = Rectifier.from_calibration(calibration)
    if cfg.head_imu:
        t_cam_imu = np.asarray(cfg.head_imu, dtype=float)
        offset = cfg.head_imu_time_offset_s
        source = "config/camera.json stereo.head_imu (Kalibr's T_cam_imu)"
    elif nominal is not None:
        t_cam_imu = nominal_t_cam_imu(*nominal)
        offset = 0.0
        source = f"NOMINAL: axes {nominal[0]}, offset {list(nominal[1])} m (Kalibr pending)"
    elif not need_imu:
        t_cam_imu, offset, source = np.eye(4), 0.0, "not needed (Kalibr measures it)"
    else:
        raise SystemExit(
            "config/camera.json has no stereo.head_imu block: pass --nominal AXES X Y Z from the"
            " axis photo, or run Kalibr first (docs/head_imu_calibration.md)"
        )
    return Rig(
        width=rectifier.width,
        height=rectifier.height,
        fx=rectifier.fx,
        fy=rectifier.fy,
        cx=rectifier.cx,
        cy=rectifier.cy,
        baseline_m=rectifier.baseline_m,
        t_cam_imu=t_cam_imu,
        time_offset_s=offset,
        imu_source=source,
        calibration_source=(
            f"config/stereo_calibration.json: {calibration.method} {calibration.date}, rms"
            f" {calibration.rms_px:.3f} px, rectified by OpenCV {_opencv_version()}"
        ),
    )


def _opencv_version() -> str:
    import cv2

    return str(cv2.__version__)


def _matrix(rows: Array, indent: str = "    ") -> str:
    return "\n".join(f"{indent}- [{', '.join(f'{v:.10g}' for v in row)}]" for row in rows)


def _list(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{v:.10g}" for v in values) + "]"


def cam1_from_cam0(baseline_m: float) -> Array:
    """Kalibr's T_cn_cnm1 of the rectified pair: a point in cam0's optical frame into cam1's;
    cam1 sits +baseline along cam0's x, so the translation is (-baseline, 0, 0)."""
    out = np.eye(4)
    out[0, 3] = -baseline_m
    return out


def imucam_chain(rig: Rig) -> str:
    """OpenVINS's kalibr_imucam_chain.yaml: T_imu_cam of both eyes = inv(T_cam_imu) (and through
    the baseline for cam1), the rectified pinhole, no distortion, the time shift."""
    t_imu_cam0 = invert(rig.t_cam_imu)
    t_imu_cam1 = t_imu_cam0 @ invert(cam1_from_cam0(rig.baseline_m))
    intrinsics = _list([rig.fx, rig.fy, rig.cx, rig.cy])
    blocks = []
    for name, transform, overlap, topic in (
        ("cam0", t_imu_cam0, 1, LEFT_TOPIC),
        ("cam1", t_imu_cam1, 0, RIGHT_TOPIC),
    ):
        blocks.append(
            f"{name}:\n"
            f"  T_imu_cam: # rotation from camera to IMU R_CtoI, position of camera in IMU p_CinI\n"
            f"{_matrix(transform)}\n"
            f"  cam_overlaps: [{overlap}]\n"
            f"  camera_model: pinhole\n"
            f"  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
            f"  distortion_model: radtan\n"
            f"  intrinsics: {intrinsics} # fu, fv, cu, cv of the rectified P\n"
            f"  resolution: [{rig.width}, {rig.height}]\n"
            f"  rostopic: {topic}\n"
            f"  timeshift_cam_imu: {rig.time_offset_s:.10g}\n"
        )
    return (
        f"%YAML:1.0\n# Generated by ros/tools/vio_config.py: do not edit.\n"
        f"# cameras: {rig.calibration_source}; baseline {rig.baseline_m * 1000:.2f} mm\n"
        f"# IMU: {rig.imu_source}\n\n" + "".join(blocks)
    )


def imu_chain(noise: dict[str, float], rate_hz: float) -> str:
    """OpenVINS's kalibr_imu_chain.yaml: the four densities, /head/imu, the rate, no intrinsics."""
    identity3 = "\n".join(
        f"    - [{', '.join(str(float(i == j)) for j in range(3))}]" for i in range(3)
    )
    zero3 = "\n".join("    - [0.0, 0.0, 0.0]" for _ in range(3))
    return (
        "%YAML:1.0\n"
        "# Generated by ros/tools/vio_config.py from config/head_imu.json: do not edit.\n\n"
        "imu0:\n"
        "  T_i_b:\n" + _matrix(np.eye(4)) + "\n"
        f"  accelerometer_noise_density: {noise['accel_noise_density']:.6g}\n"
        f"  accelerometer_random_walk: {noise['accel_random_walk']:.6g}\n"
        f"  gyroscope_noise_density: {noise['gyro_noise_density']:.6g}\n"
        f"  gyroscope_random_walk: {noise['gyro_random_walk']:.6g}\n"
        f"  rostopic: {IMU_TOPIC}\n"
        "  time_offset: 0.0\n"
        f"  update_rate: {rate_hz:.6g}\n"
        '  model: "kalibr"\n'
        f"  Tw:\n{identity3}\n  R_IMUtoGYRO:\n{identity3}\n  Ta:\n{identity3}\n"
        f"  R_IMUtoACC:\n{identity3}\n  Tg:\n{zero3}\n"
    )


def zupt_block(zupt: bool) -> str:
    """The estimator's ZUPT keys: :data:`ZUPT` with its notes, or the first design's
    :data:`NO_ZUPT`."""
    if not zupt:
        values = dict(NO_ZUPT)
        notes = {"try_zupt": "--no-zupt, the first design's; the init waits for a jerk"}
    else:
        values, notes = ZUPT, ZUPT_NOTES
    return "".join(
        f"{key}: {value}" + (f" # {notes[key]}" if key in notes else "") + "\n"
        for key, value in values.items()
    )


def estimator_config(
    calib_extrinsics: bool, zupt: bool = True, dyn_init: bool = False, fps: float = 10.0
) -> str:
    """OpenVINS's estimator_config.yaml for this rig (vio.md section 4): stereo, the time offset
    estimated online, extrinsics fixed, OpenVINS's own ZUPT (:data:`ZUPT`, off with ``zupt``
    false), the static init at a ground robot's threshold (and the dynamic one in motion with
    ``dyn_init``), 15 Hz tracking of a 10 Hz camera."""
    dyn_note = (
        "in motion too (--dyn-init); a still window keeps the static init"
        if dyn_init
        else "static init only, from stillness (with the ZUPT) or a jerk; --dyn-init in motion"
    )
    return f"""%YAML:1.0 # Generated by ros/tools/vio_config.py: do not edit.

verbosity: "INFO"

use_fej: true
integration: "rk4"
use_stereo: true
max_cameras: 2

calib_cam_extrinsics: {str(calib_extrinsics).lower()} # Kalibr's fixed; true in a check session
calib_cam_intrinsics: false # the rectified pinhole is fixed by construction
calib_cam_timeoffset: true # t_d online, always
calib_imu_intrinsics: false
calib_imu_g_sensitivity: false

max_clones: 11
max_slam: 50
max_slam_in_update: 25
max_msckf_in_update: 40
dt_slam_delay: 1

gravity_mag: 9.806

feat_rep_msckf: "GLOBAL_3D"
feat_rep_slam: "ANCHORED_MSCKF_INVERSE_DEPTH"
feat_rep_aruco: "ANCHORED_MSCKF_INVERSE_DEPTH"

{zupt_block(zupt)}
init_window_time: 1.0
init_imu_thresh: 0.4 # a cart rolls off at ~0.25 m/s^2: the drone default 1.5 never triggers
init_max_disparity: 10.0
init_max_features: 50

init_dyn_use: {str(dyn_init).lower()} # {dyn_note}
init_dyn_mle_opt_calib: false
init_dyn_mle_max_iter: 50
init_dyn_mle_max_time: 0.05
init_dyn_mle_max_threads: 2
init_dyn_num_pose: 6
init_dyn_min_deg: 10.0

init_dyn_inflation_ori: 10
init_dyn_inflation_vel: 100
init_dyn_inflation_bg: 10
init_dyn_inflation_ba: 100
init_dyn_min_rec_cond: 1e-12

init_dyn_bias_g: [0.0, 0.0, 0.0]
init_dyn_bias_a: [0.0, 0.0, 0.0]

record_timing_information: false
record_timing_filepath: "/tmp/traj_timing.txt"

save_total_state: false
filepath_est: "/tmp/ov_estimate.txt"
filepath_std: "/tmp/ov_estimate_std.txt"
filepath_gt: "/tmp/ov_groundtruth.txt"

use_klt: true
num_pts: 200
fast_threshold: 20
grid_x: 5
grid_y: 5
min_px_dist: 10
knn_ratio: 0.70
track_frequency: {track_frequency_hz(fps):.1f} # {TRACK_PER_FPS:g} x the camera's {fps:g} fps
downsample_cameras: false # true = 400x300 if the CPU bites
num_opencv_threads: 2
histogram_method: "HISTOGRAM"

use_aruco: false
num_aruco: 1024
downsize_aruco: true

up_msckf_sigma_px: 1.5 # rectified rms 0.36 px, MJPEG adds
up_msckf_chi2_multipler: 1
up_slam_sigma_px: 1.5
up_slam_chi2_multipler: 1
up_aruco_sigma_px: 1
up_aruco_chi2_multipler: 1

use_mask: false

relative_config_imu: "kalibr_imu_chain.yaml"
relative_config_imucam: "kalibr_imucam_chain.yaml"
"""


def kalibr_camchain(rig: Rig) -> str:
    """Kalibr's camchain.yaml: the rectified upright eyes as two undistorted pinholes."""
    intrinsics = _list([rig.fx, rig.fy, rig.cx, rig.cy])
    common = (
        "  camera_model: pinhole\n"
        f"  intrinsics: {intrinsics}\n"
        "  distortion_model: radtan\n"
        "  distortion_coeffs: [0.0, 0.0, 0.0, 0.0]\n"
        f"  resolution: [{rig.width}, {rig.height}]\n"
    )
    return (
        f"# Generated by ros/tools/vio_config.py ({rig.calibration_source}): do not edit.\n"
        f"cam0:\n{common}  rostopic: {LEFT_TOPIC}\n"
        f"cam1:\n  T_cn_cnm1:\n{_matrix(cam1_from_cam0(rig.baseline_m), '  ')}\n"
        f"{common}  rostopic: {RIGHT_TOPIC}\n"
    )


def kalibr_imu(noise: dict[str, float], rate_hz: float) -> str:
    """Kalibr's imu.yaml (the same densities OpenVINS is given)."""
    return (
        "# Generated by ros/tools/vio_config.py from config/head_imu.json: do not edit.\n"
        f"rostopic: {IMU_TOPIC}\n"
        f"update_rate: {rate_hz:.6g}\n"
        f"accelerometer_noise_density: {noise['accel_noise_density']:.6g}\n"
        f"accelerometer_random_walk: {noise['accel_random_walk']:.6g}\n"
        f"gyroscope_noise_density: {noise['gyro_noise_density']:.6g}\n"
        f"gyroscope_random_walk: {noise['gyro_random_walk']:.6g}\n"
    )


def kalibr_april(tag_size_m: float) -> str:
    """Kalibr's target yaml: the 6x6 AprilGrid of docs/aprilgrid, the RULER's tag size."""
    return (
        "# Generated by ros/tools/vio_config.py: tagSize is the printed black square, measured.\n"
        "target_type: aprilgrid\n"
        "tagCols: 6\n"
        "tagRows: 6\n"
        f"tagSize: {tag_size_m:.5g}\n"
        "tagSpacing: 0.3\n"
    )


def write_all(out: Path, files: dict[str, str]) -> None:
    """Each file into ``out``, created if needed."""
    out.mkdir(parents=True, exist_ok=True)
    for name, text in files.items():
        (out / name).write_text(text)
        print(f"wrote {out / name}")


def main(argv: list[str] | None = None) -> int:
    """Generate the files."""
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--config", type=Path, default=CONFIG, help="the config directory")
    parser.add_argument("--camera", default=None, help="the rig by name (default: active)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--nominal", nargs=4, metavar=("AXES", "X", "Y", "Z"), help="T_cam_imu from the photo"
    )
    parser.add_argument("--calib-extrinsics", action="store_true")
    parser.add_argument(
        "--no-zupt",
        action="store_true",
        help="the first design's: ZUPT off, the init waits for a jerk",
    )
    parser.add_argument(
        "--dyn-init", action="store_true", help="OpenVINS may also initialise in motion"
    )
    parser.add_argument("--tag-size", type=float, default=None, help="metres, measured")
    parser.add_argument("--kalibr-only", action="store_true")
    parser.add_argument(
        "--camera-stamp-lag",
        type=float,
        default=None,
        metavar="S",
        help="camera_stream's live camera_stamp_lag_s (default: the knob's default)",
    )
    parser.add_argument(
        "--print-block",
        action="store_true",
        help="print --nominal as config/camera.json's stereo.head_imu block (for the live TF)",
    )
    args = parser.parse_args(argv)
    nominal = None
    if args.nominal is not None:
        nominal = (args.nominal[0], [float(v) for v in args.nominal[1:]])
    if args.print_block:
        if nominal is None:
            raise SystemExit("--print-block prints the --nominal guess")
        print(json.dumps(nominal_block(*nominal), indent=2))
        return 0
    rig = load_rig(args.config, args.camera, nominal, need_imu=not args.kalibr_only)
    rig = with_stamp_lag(rig, args.camera_stamp_lag, args.config, args.camera)
    fps = camera_fps(args.config, args.camera)
    sys.path.insert(0, str(SRC))
    from pepin.head_imu import HeadImuConfig

    head = HeadImuConfig.load(args.config / "head_imu.json")
    files: dict[str, str] = {
        "camchain.yaml": kalibr_camchain(rig),
        "imu.yaml": kalibr_imu(head.noise, head.rate_hz),
    }
    if args.tag_size is not None:
        files["april.yaml"] = kalibr_april(args.tag_size)
    if not args.kalibr_only:
        files.update(
            {
                "kalibr_imucam_chain.yaml": imucam_chain(rig),
                "kalibr_imu_chain.yaml": imu_chain(head.noise, head.rate_hz),
                "estimator_config.yaml": estimator_config(
                    args.calib_extrinsics, zupt=not args.no_zupt, dyn_init=args.dyn_init, fps=fps
                ),
            }
        )
    write_all(args.out, files)
    if not args.kalibr_only:
        print(
            f"OpenVINS: {fps:g} fps (track_frequency {track_frequency_hz(fps):g}),"
            f" ZUPT {'off' if args.no_zupt else 'on'}, init"
            f" {'static + dynamic' if args.dyn_init else 'static'}"
        )
    print(f"IMU extrinsics: {rig.imu_source}; time shift {rig.time_offset_s * 1e3:+.1f} ms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
