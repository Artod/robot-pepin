"""Kalibr's stereo camera calibration as ``config/stereo_calibration.json``.

``kalibr_calibrate_cameras`` on the two RAW eyes (cut and turned upright as
:class:`pepin.stereo.SideBySide` does, before any rectification) measures what the file holds:
each eye's pinhole and distortion and the right eye's pose in the left eye's frame. Its camchain
writes them in Kalibr's words, and this module is the translation, with the checks a person
reads before the file is replaced:

* the distortion: Kalibr's ``radtan`` is ``[k1, k2, r1, r2]`` with the tangential terms in
  OpenCV's convention (``x += 2 p1 x y + p2 (r^2 + 2 x^2)``), so OpenCV's plumb_bob vector is the
  same four numbers and a zero ``k3``: ``[k1, k2, p1, p2, 0]``;
* the bar: Kalibr's ``T_cn_cnm1`` of cam1 carries a point from cam0's frame into cam1's
  (``x_cam1 = T x_cam0``), which is OpenCV's ``x_right = R x_left + T`` with cam0 the left eye:
  the rotation and translation are copied, not inverted (the left eye sits at -x of the right);
* the rectified left eye moves: ``stereoRectify`` turns both eyes to a common orientation that
  depends on the bar, so a new bar turns the frame every depth point, ``config/camera.json``'s
  ``eye`` block and ``head_imu.T_cam_imu`` are expressed in. :func:`rectified_frame_shift`
  measures that turn from the two models' rays over the picture, and :func:`carry_eye` /
  :func:`carry_head_imu` carry both blocks through it, so ``base_link`` geometry (the arm fit,
  the hand-eye the board runs) stays where it was measured;
* :func:`depth_ratio`: what the OLD rectification reads at a range if the new model is the
  truth, the prediction of the range bias the recalibration should remove.

OpenCV and yaml are imported inside the functions that need them.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from pepin.camera import OPTICAL_RPY
from pepin.mounts import rotation_from_rpy, rpy_from_rotation
from pepin.stereo import StereoCalibration

Array = npt.NDArray[np.float64]
Mask = npt.NDArray[np.bool_]

TOPICS = ("/cam0/image_raw", "/cam1/image_raw")
# Acceptance of a run: the RMS reprojection error of each eye (OpenCV's convention, the root of
# the mean squared pixel error, comparable with the chessboard's 0.36) and the bar against the
# vendor's nominal baseline (config/camera.json's rig.baseline_m_nominal, measured on the
# module with a tape at 63 mm).
REPROJECTION_MAX_PX = 0.3
BASELINE_TOLERANCE_M = 0.002
# The rays compared between two models: within this angle of the old left eye's axis, where both
# lens models are fitted (the rational model folds beyond its data at the picture's corners).
COMPARE_ANGLE_DEG = 35.0
_ROUND_TRIP_PX = 0.05


@dataclass(frozen=True)
class KalibrEye:
    """One camera of a Kalibr camchain: pinhole intrinsics, distortion and resolution."""

    intrinsics: tuple[float, float, float, float]  # fu, fv, pu, pv
    distortion_model: str
    distortion: tuple[float, ...]
    resolution: tuple[int, int]

    def k(self) -> tuple[tuple[float, ...], ...]:
        """The 3x3 camera matrix."""
        fu, fv, pu, pv = self.intrinsics
        return ((fu, 0.0, pu), (0.0, fv, pv), (0.0, 0.0, 1.0))

    def opencv_distortion(self) -> tuple[float, ...]:
        """OpenCV's distortion vector for this eye (plumb_bob, five numbers)."""
        return opencv_distortion(self.distortion_model, self.distortion)


def opencv_distortion(model: str, coeffs: Sequence[float]) -> tuple[float, ...]:
    """Kalibr's distortion as OpenCV's plumb_bob ``[k1, k2, p1, p2, k3]``: radtan's four in the
    same order and ``k3 = 0``. Any other model is refused: equidistant and the omni models need
    ``cv2.fisheye`` or another camera model, and :class:`pepin.stereo.Rectifier` runs neither."""
    values = tuple(float(v) for v in coeffs)
    if model == "radtan":
        if len(values) != 4:
            raise ValueError(f"radtan has four coefficients, not {len(values)}")
        return (*values, 0.0)
    if model == "none":
        return (0.0, 0.0, 0.0, 0.0, 0.0)
    raise ValueError(f"Kalibr's {model!r} distortion has no plumb_bob equivalent")


def _eye(block: Mapping[str, Any]) -> KalibrEye:
    if str(block.get("camera_model", "pinhole")) != "pinhole":
        raise ValueError(f"camera_model {block.get('camera_model')!r}: only pinhole is mapped")
    fu, fv, pu, pv = (float(v) for v in block["intrinsics"])
    width, height = (int(v) for v in block["resolution"])
    return KalibrEye(
        intrinsics=(fu, fv, pu, pv),
        distortion_model=str(block["distortion_model"]),
        distortion=tuple(float(v) for v in block["distortion_coeffs"]),
        resolution=(width, height),
    )


def parse_camchain(data: Mapping[str, Any]) -> tuple[KalibrEye, KalibrEye, Array]:
    """``(cam0, cam1, T_cn_cnm1)`` from a camchain's dictionary; ``T_cn_cnm1`` is cam1's 4x4
    (cam1 <- cam0), checked to be rigid."""
    left, right = _eye(data["cam0"]), _eye(data["cam1"])
    if left.resolution != right.resolution:
        raise ValueError(f"the eyes differ in size: {left.resolution} vs {right.resolution}")
    t = np.asarray(data["cam1"]["T_cn_cnm1"], dtype=np.float64)
    if t.shape != (4, 4) or not np.allclose(t[3], [0.0, 0.0, 0.0, 1.0]):
        raise ValueError("cam1.T_cn_cnm1 is not a 4x4 rigid transform")
    r = t[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=1e-6) or np.linalg.det(r) < 0.0:
        raise ValueError("cam1.T_cn_cnm1's rotation is not a rotation")
    return left, right, t


def load_camchain(path: str | Path) -> dict[str, Any]:
    """A Kalibr camchain yaml as a dictionary."""
    import yaml

    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError(f"{path} is not a camchain")
    return data


def calibration_from_camchain(
    data: Mapping[str, Any],
    rms_px: float,
    date: str,
    method: str,
    views: int = 0,
    board: str = "",
) -> StereoCalibration:
    """The camchain as the file :class:`pepin.stereo.StereoCalibration` reads: cam0 the left
    eye, cam1 the right, ``rotation``/``translation_m`` = ``T_cn_cnm1`` as it stands."""
    left, right, t = parse_camchain(data)
    return StereoCalibration(
        width=left.resolution[0],
        height=left.resolution[1],
        k_left=left.k(),
        d_left=left.opencv_distortion(),
        k_right=right.k(),
        d_right=right.opencv_distortion(),
        rotation=tuple(tuple(float(v) for v in row) for row in t[:3, :3]),
        translation_m=(float(t[0, 3]), float(t[1, 3]), float(t[2, 3])),
        rms_px=rms_px,
        date=date,
        method=method,
        views=views,
        board=board,
    )


_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_REPROJECTION = re.compile(
    rf"reprojection error:\s*\[\s*({_NUMBER}),\s*({_NUMBER})\s*\]\s*\+-\s*"
    rf"\[\s*({_NUMBER}),\s*({_NUMBER})\s*\]"
)
_USED = re.compile(r"Processed (\d+) images with (\d+) images used")


def reprojection_rms(text: str) -> list[float]:
    """Each camera's RMS reprojection error from Kalibr's ``-results-cam.txt`` (or its log), in
    camera order: ``sqrt(mean_x^2 + mean_y^2 + std_x^2 + std_y^2)`` of its ``reprojection
    error: [mean] +- [std]`` line, the root of the mean squared pixel error. The first line per
    camera is taken (the results file prints each camera once)."""
    values = [
        math.sqrt(sum(float(v) ** 2 for v in m.groups())) for m in _REPROJECTION.finditer(text)
    ]
    return values


def views_used(text: str) -> int:
    """The images Kalibr's optimisation kept (``Processed N images with M images used``)."""
    found = _USED.findall(text)
    return int(found[-1][1]) if found else 0


def rotation_angle_deg(rotation: Any) -> float:
    """The angle of a rotation matrix, degrees."""
    r = np.asarray(rotation, dtype=np.float64)
    cos = (float(np.trace(r)) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, cos))))


def rotation_vector_deg(rotation: Any) -> Array:
    """A rotation matrix as its rotation vector in degrees (about x, y, z of its frame)."""
    import cv2

    rvec = cv2.Rodrigues(np.asarray(rotation, dtype=np.float64))[0].reshape(3)
    return np.asarray(np.degrees(rvec), dtype=np.float64)


def _lens(cal: StereoCalibration, eye: str) -> tuple[Array, Array]:
    if eye == "left":
        return np.asarray(cal.k_left, np.float64), np.asarray(cal.d_left, np.float64)
    return np.asarray(cal.k_right, np.float64), np.asarray(cal.d_right, np.float64)


def rectification(cal: StereoCalibration) -> tuple[Array, Array, Array, Array]:
    """``(R1, R2, P1, P2)`` exactly as :class:`pepin.stereo.Rectifier` builds them (zero
    disparity at infinity, alpha 0): R1 turns the raw left eye's rays into the rectified one."""
    import cv2

    k_l, d_l = _lens(cal, "left")
    k_r, d_r = _lens(cal, "right")
    r1, r2, p1, p2, _q, _a, _b = cv2.stereoRectify(
        k_l, d_l, k_r, d_r, (cal.width, cal.height),
        np.asarray(cal.rotation, np.float64),
        np.asarray(cal.translation_m, np.float64).reshape(3, 1),
        flags=cv2.CALIB_ZERO_DISPARITY, alpha=0.0,
    )  # fmt: skip
    return (
        np.asarray(r1, np.float64),
        np.asarray(r2, np.float64),
        np.asarray(p1, np.float64),
        np.asarray(p2, np.float64),
    )


def _rays(pixels: Array, k: Array, d: Array, r: Array) -> tuple[Array, Mask]:
    """Unit rays of raw pixels in the frame ``r`` turns the raw eye into, and which of them
    survive the round trip back through the lens model (a fold or an extrapolation does not)."""
    import cv2

    pts = pixels.reshape(-1, 1, 2).astype(np.float64)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    normalised = np.asarray(
        cv2.undistortPointsIter(pts, k, d, r, np.eye(3), criteria), np.float64
    ).reshape(-1, 2)
    rays: Array = np.column_stack([normalised, np.ones(len(normalised))])
    finite = np.all(np.isfinite(rays), axis=1)
    rays[~finite] = (0.0, 0.0, 1.0)  # a model that gave up on a pixel: dropped below
    rays /= np.linalg.norm(rays, axis=1, keepdims=True)
    raw_rays = rays @ r  # r^T applied to each row: back into the raw eye's frame
    back, _ = cv2.projectPoints(raw_rays.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k, d)
    error = np.asarray(back, np.float64).reshape(-1, 2) - pixels.reshape(-1, 2)
    ok: Mask = (np.linalg.norm(error, axis=1) < _ROUND_TRIP_PX) & (raw_rays[:, 2] > 0.0) & finite
    return rays, ok


def _grid(cal: StereoCalibration, step_px: int) -> Array:
    us, vs = np.meshgrid(
        np.arange(step_px / 2, cal.width, step_px), np.arange(step_px / 2, cal.height, step_px)
    )
    return np.column_stack([us.ravel(), vs.ravel()]).astype(np.float64)


@dataclass(frozen=True)
class FrameShift:
    """How the rectified left eye's frame turned between two calibrations: ``rotation`` takes a
    vector in the old rectified frame into the new one, ``residual_deg`` is what is left between
    the two models' rays once that rotation is taken out (focal and distortion differences)."""

    rotation: Array
    residual_deg: float
    rays: int

    @property
    def angle_deg(self) -> float:
        """The size of the turn, degrees."""
        return rotation_angle_deg(self.rotation)


def wahba(source: Array, target: Array) -> Array:
    """The rotation Q minimising ``sum |target_i - Q source_i|^2`` (rows are vectors)."""
    b = target.T @ source
    u, _s, vt = np.linalg.svd(b)
    sign = np.sign(np.linalg.det(u @ vt))
    q: Array = u @ np.diag([1.0, 1.0, sign]) @ vt
    return q


def rectified_frame_shift(
    old: StereoCalibration,
    new: StereoCalibration,
    step_px: int = 16,
    max_angle_deg: float = COMPARE_ANGLE_DEG,
) -> FrameShift:
    """The turn of the rectified left eye from ``old`` to ``new``: the same raw left pixels
    become rays under each model and its rectification, and the best rotation between the two
    ray sets (Wahba) is the turn. The raw eye is the same glass under both models, so what a
    principal point or a bar moved is measured, not assumed."""
    if (old.width, old.height) != (new.width, new.height):
        raise ValueError("the two calibrations are for different image sizes")
    pixels = _grid(old, step_px)
    r1_old = rectification(old)[0]
    r1_new = rectification(new)[0]
    k_o, d_o = _lens(old, "left")
    k_n, d_n = _lens(new, "left")
    rays_old, ok_old = _rays(pixels, k_o, d_o, r1_old)
    rays_new, ok_new = _rays(pixels, k_n, d_n, r1_new)
    axis = r1_old @ np.array([0.0, 0.0, 1.0])
    central = rays_old @ axis > math.cos(math.radians(max_angle_deg))
    keep = ok_old & ok_new & central
    if keep.sum() < 10:
        raise ValueError("fewer than ten rays survive both models: nothing to compare")
    q = wahba(rays_old[keep], rays_new[keep])
    moved = rays_old[keep] @ q.T
    angles = np.degrees(np.arccos(np.clip(np.sum(moved * rays_new[keep], axis=1), -1.0, 1.0)))
    return FrameShift(q, float(np.sqrt(np.mean(angles**2))), int(keep.sum()))


def carry_eye(eye: Mapping[str, Any], shift: Array) -> dict[str, float]:
    """``config/camera.json``'s ``eye`` block (camera_link <- the rectified left eye, composed
    with the optical axes) after the rectified frame turned by ``shift`` (old -> new): the
    link-to-optical rotation becomes ``R_link_optical_old shift^T``; the translation (the left
    eye's optical centre) does not move. Only the six numbers are returned."""
    optical = rotation_from_rpy(*OPTICAL_RPY)
    r_eye = rotation_from_rpy(
        math.radians(float(eye.get("roll_deg", 0.0))),
        math.radians(float(eye.get("pitch_deg", 0.0))),
        math.radians(float(eye.get("yaw_deg", 0.0))),
    )
    r_new = r_eye @ optical @ np.asarray(shift, np.float64).T @ optical.T
    roll, pitch, yaw = rpy_from_rotation(r_new)
    return {
        "x_m": float(eye.get("x_m", 0.0)),
        "y_m": float(eye.get("y_m", 0.0)),
        "z_m": float(eye.get("z_m", 0.0)),
        "roll_deg": round(math.degrees(roll), 3),
        "pitch_deg": round(math.degrees(pitch), 3),
        "yaw_deg": round(math.degrees(yaw), 3),
    }


def carry_head_imu(t_cam_imu: Any, shift: Array) -> Array:
    """Kalibr's ``T_cam_imu`` (imu -> the rectified left eye) after that eye's frame turned by
    ``shift`` (old -> new): ``[shift 0; 0 1] T_cam_imu``."""
    lift = np.eye(4)
    lift[:3, :3] = np.asarray(shift, np.float64)
    out: Array = lift @ np.asarray(t_cam_imu, np.float64)
    return out


@dataclass(frozen=True)
class DepthRatio:
    """What one rectification reads against the truth of another at one range: the ratio of
    measured to true depth over the central picture (median, 10th and 90th percentile), and the
    disparity offset in pixels behind it."""

    range_m: float
    median: float
    p10: float
    p90: float
    offset_px: float


def depth_ratio(
    old: StereoCalibration,
    truth: StereoCalibration,
    ranges_m: Sequence[float] = (0.6, 1.25, 2.5),
    step_px: int = 24,
    max_angle_deg: float = 30.0,
) -> list[DepthRatio]:
    """Points at each range in front of the left eye, seen by both eyes under ``truth``, then
    rectified and triangulated under ``old`` (``z = fx B / d`` with the old rectified pinhole):
    the ratio of the old depth to the true one. A ratio under 1 growing with range is a positive
    disparity offset, the bias the floor showed."""
    import cv2

    pixels = _grid(truth, step_px)
    k_tl, d_tl = _lens(truth, "left")
    k_tr, d_tr = _lens(truth, "right")
    rays, ok = _rays(pixels, k_tl, d_tl, np.eye(3))
    rays = rays[ok & (rays[:, 2] > math.cos(math.radians(max_angle_deg)))]
    r1, r2, p1, p2 = rectification(old)
    fx, baseline = float(p1[0, 0]), float(abs(p2[0, 3]) / p2[0, 0])
    k_ol, d_ol = _lens(old, "left")
    k_or, d_or = _lens(old, "right")
    r_true = np.asarray(truth.rotation, np.float64)
    t_true = np.asarray(truth.translation_m, np.float64)
    criteria = (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12)
    out = []
    for z in ranges_m:
        points = rays * (z / rays[:, 2:3])  # the true left eye's frame, depth z
        left_px, _ = cv2.projectPoints(
            points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k_tl, d_tl
        )
        right_pts = points @ r_true.T + t_true
        right_px, _ = cv2.projectPoints(
            right_pts.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), k_tr, d_tr
        )
        nl = np.asarray(
            cv2.undistortPointsIter(np.asarray(left_px), k_ol, d_ol, r1, np.eye(3), criteria),
            np.float64,
        ).reshape(-1, 2)
        nr = np.asarray(
            cv2.undistortPointsIter(np.asarray(right_px), k_or, d_or, r2, np.eye(3), criteria),
            np.float64,
        ).reshape(-1, 2)
        disparity = fx * (nl[:, 0] - nr[:, 0])
        measured = fx * baseline / disparity
        true_depth = (points @ r1.T)[:, 2]  # depth along the old rectified axis
        ratio = measured / true_depth
        expected = fx * baseline / true_depth
        out.append(
            DepthRatio(
                range_m=float(z),
                median=float(np.median(ratio)),
                p10=float(np.percentile(ratio, 10)),
                p90=float(np.percentile(ratio, 90)),
                offset_px=float(np.median(disparity - expected)),
            )
        )
    return out


@dataclass(frozen=True)
class Acceptance:
    """A run's verdict: the failures (empty: accepted) and the numbers they were judged on."""

    reprojection_px: tuple[float, ...]
    baseline_m: float
    nominal_m: float
    failures: tuple[str, ...]

    @property
    def accepted(self) -> bool:
        """Whether every check passed."""
        return not self.failures


def accept(
    reprojection_px: Sequence[float],
    baseline_m: float,
    nominal_m: float,
    max_px: float = REPROJECTION_MAX_PX,
    tolerance_m: float = BASELINE_TOLERANCE_M,
) -> Acceptance:
    """Reprojection under ``max_px`` in each of the two eyes and the baseline within
    ``tolerance_m`` of the tape's nominal."""
    failures = []
    if len(reprojection_px) < 2:
        failures.append(f"Kalibr reported {len(reprojection_px)} reprojection errors, not 2")
    for cam, px in enumerate(reprojection_px[:2]):
        if not (px < max_px):
            failures.append(f"cam{cam} reprojection {px:.3f} px (bound {max_px:.2f})")
    if not (abs(baseline_m - nominal_m) <= tolerance_m):
        failures.append(
            f"baseline {baseline_m * 1e3:.2f} mm, {abs(baseline_m - nominal_m) * 1e3:.2f} mm from"
            f" the nominal {nominal_m * 1e3:.1f} (bound {tolerance_m * 1e3:.1f}): the tag size?"
        )
    return Acceptance(tuple(reprojection_px), baseline_m, nominal_m, tuple(failures))


def sharpest_per_window(
    stamps: Sequence[float], sharpness: Sequence[float], period_s: float
) -> list[int]:
    """One frame per ``period_s`` window from the first stamp on: the index of the sharpest
    frame of each window that has any, in time order. A hand-held board blurs in the moves, and
    the sharpest frame of a quarter second is the one Kalibr's corners should come from."""
    if len(stamps) != len(sharpness):
        raise ValueError("one sharpness per stamp")
    if not stamps:
        return []
    t0 = min(stamps)
    best: dict[int, int] = {}
    for i, (t, s) in enumerate(zip(stamps, sharpness, strict=True)):
        slot = int((t - t0) // period_s)
        if slot not in best or s > sharpness[best[slot]]:
            best[slot] = i
    return [best[slot] for slot in sorted(best)]


def write_camera_blocks(
    path: str | Path,
    eye: Mapping[str, Any] | None,
    head_imu: Mapping[str, Any] | None,
    camera: str = "stereo",
) -> None:
    """Replace ``<camera>.eye`` and/or ``<camera>.head_imu`` in ``config/camera.json`` (keys not
    given stay); the file is written as it is kept (two-space indent, ``\\u`` escapes), through
    a temporary file next to it."""
    from pepin.camera import head_imu_transform

    file = Path(path)
    data = json.loads(file.read_text())
    if camera not in data or "rig" not in data[camera]:
        raise ValueError(f"{camera} is not a stereo block of {file}")
    if eye is not None:
        data[camera]["eye"] = dict(eye)
    if head_imu is not None:
        head_imu_transform(head_imu)
        data[camera]["head_imu"] = dict(head_imu)
    tmp = file.with_suffix(file.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=True) + "\n")
    tmp.replace(file)
