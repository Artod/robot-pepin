"""Kalibr's stereo camchain as config/stereo_calibration.json (pepin.kalibr_stereo) and the
runner's pure parts (ros/tools/stereo_kalibr.py)."""

from __future__ import annotations

import importlib.util
import itertools
import json
import math
import shutil
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
import yaml

from pepin import kalibr_stereo as ks
from pepin.camera import OPTICAL_RPY
from pepin.mounts import rotation_from_rpy
from pepin.stereo import Rectifier, StereoCalibration

REPO = Path(__file__).resolve().parents[2]
# numpy's matmul on macOS Accelerate warns divide/overflow/invalid on finite arrays
pytestmark = pytest.mark.filterwarnings("ignore:.*encountered in matmul:RuntimeWarning")

K0 = (530.0, 531.5, 401.5, 268.0)
D0 = (-0.29, 0.085, 0.0009, -0.0004)
K1 = (528.6, 530.2, 397.2, 271.4)
D1 = (-0.285, 0.080, -0.0006, 0.0005)
RVEC_DEG = (0.35, -0.6, 0.15)
T_10 = (-0.0627, 0.0006, -0.0004)  # cam1 <- cam0: the left eye sits at -x of the right


def _rotation(rvec_deg: Any) -> np.ndarray:
    r, _ = cv2.Rodrigues(np.radians(np.asarray(rvec_deg, dtype=float)).reshape(3, 1))
    return np.asarray(r)


def _transform(rvec_deg: Any = RVEC_DEG, t: Any = T_10) -> np.ndarray:
    m = np.eye(4)
    m[:3, :3] = _rotation(rvec_deg)
    m[:3, 3] = t
    return m


def _camchain(t_10: np.ndarray | None = None) -> dict[str, Any]:
    t = _transform() if t_10 is None else t_10
    return {
        "cam0": {
            "cam_overlaps": [1],
            "camera_model": "pinhole",
            "distortion_coeffs": list(D0),
            "distortion_model": "radtan",
            "intrinsics": list(K0),
            "resolution": [800, 600],
            "rostopic": "/cam0/image_raw",
        },
        "cam1": {
            "T_cn_cnm1": t.tolist(),
            "cam_overlaps": [0],
            "camera_model": "pinhole",
            "distortion_coeffs": list(D1),
            "distortion_model": "radtan",
            "intrinsics": list(K1),
            "resolution": [800, 600],
            "rostopic": "/cam1/image_raw",
        },
    }


def _calibration(t_10: np.ndarray | None = None) -> StereoCalibration:
    return ks.calibration_from_camchain(_camchain(t_10), 0.1, "2026-10-04", "kalibr")


def _kalibr_radtan(points: np.ndarray, intrinsics: Any, coeffs: Any) -> np.ndarray:
    """aslam_cv's RadialTangentialDistortion::distort + PinholeProjection, transcribed."""
    fu, fv, cu, cv_ = intrinsics
    k1, k2, p1, p2 = coeffs
    y0 = points[:, 0] / points[:, 2]
    y1 = points[:, 1] / points[:, 2]
    mx2, my2, mxy = y0 * y0, y1 * y1, y0 * y1
    rho2 = mx2 + my2
    rad = k1 * rho2 + k2 * rho2 * rho2
    d0 = y0 + y0 * rad + 2.0 * p1 * mxy + p2 * (rho2 + 2.0 * mx2)
    d1 = y1 + y1 * rad + 2.0 * p2 * mxy + p1 * (rho2 + 2.0 * my2)
    return np.column_stack([fu * d0 + cu, fv * d1 + cv_])


def test_radtan_is_plumb_bob_with_a_zero_k3() -> None:
    assert ks.opencv_distortion("radtan", [0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4, 0.0)
    assert ks.opencv_distortion("none", []) == (0.0,) * 5
    with pytest.raises(ValueError):
        ks.opencv_distortion("equidistant", [0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError):
        ks.opencv_distortion("radtan", [0.1, 0.2, 0.3, 0.4, 0.5])


def test_mapped_distortion_projects_as_kalibr_does() -> None:
    """The order claim proven by projection: OpenCV with the mapped vector lands every point
    where Kalibr's own radtan puts it, tangential terms included (strong p1/p2 on purpose)."""
    rng = np.random.default_rng(3)
    points = np.column_stack([rng.uniform(-0.8, 0.8, (200, 2)), np.ones(200)]) * 0.7
    coeffs = (-0.29, 0.085, 0.004, -0.003)
    eye = ks.KalibrEye(K0, "radtan", coeffs, (800, 600))
    projected, _ = cv2.projectPoints(
        points.reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        np.asarray(eye.k()),
        np.asarray(eye.opencv_distortion()),
    )
    assert np.allclose(projected.reshape(-1, 2), _kalibr_radtan(points, K0, coeffs), atol=1e-9)
    # swapped tangential terms (the mistake this guards) would miss by pixels
    swapped = _kalibr_radtan(points, K0, (coeffs[0], coeffs[1], coeffs[3], coeffs[2]))
    assert np.abs(projected.reshape(-1, 2) - swapped).max() > 1.0


def test_camchain_yaml_maps_to_the_file() -> None:
    text = yaml.safe_dump(_camchain(), default_flow_style=None)
    cal = ks.calibration_from_camchain(yaml.safe_load(text), 0.12, "2026-10-04", "kalibr", 150)
    assert (cal.width, cal.height) == (800, 600)
    assert cal.k_left == ((530.0, 0.0, 401.5), (0.0, 531.5, 268.0), (0.0, 0.0, 1.0))
    assert cal.k_right[0][2] == 397.2 and cal.k_right[1][2] == 271.4
    assert cal.d_left == (*D0, 0.0) and cal.d_right == (*D1, 0.0)
    assert np.allclose(cal.rotation, _rotation(RVEC_DEG))
    assert cal.translation_m == pytest.approx(T_10)
    assert cal.baseline_m == pytest.approx(math.dist(T_10, (0, 0, 0)))
    assert StereoCalibration.from_json(json.loads(json.dumps(cal.to_json()))) == cal


def _rectified_depth(cal: StereoCalibration, points: np.ndarray) -> np.ndarray:
    """Depth by the file's rectification of points projected under the TRUE model (Kalibr's
    convention: x_cam1 = T_cn_cnm1 x_cam0)."""
    truth = _transform()
    left = _kalibr_radtan(points, K0, D0)
    right = _kalibr_radtan(points @ truth[:3, :3].T + truth[:3, 3], K1, D1)
    r1, r2, p1, p2 = ks.rectification(cal)
    nl = cv2.undistortPointsIter(
        left.reshape(-1, 1, 2),
        np.asarray(cal.k_left),
        np.asarray(cal.d_left),
        r1,
        p1[:, :3],
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12),
    ).reshape(-1, 2)
    nr = cv2.undistortPointsIter(
        right.reshape(-1, 1, 2),
        np.asarray(cal.k_right),
        np.asarray(cal.d_right),
        r2,
        p2[:, :3],
        (cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 100, 1e-12),
    ).reshape(-1, 2)
    baseline = abs(p2[0, 3]) / p2[0, 0]
    return np.asarray(p1[0, 0] * baseline / (nl[:, 0] - nr[:, 0]))


def test_t_cn_cnm1_is_copied_not_inverted() -> None:
    """cam1 <- cam0 is OpenCV's x_right = R x_left + T: the mapped file triangulates points the
    true head saw to their depth; the inverse transform (the slip) does not."""
    rng = np.random.default_rng(5)
    points = np.column_stack([rng.uniform(-0.4, 0.4, (50, 2)), np.ones(50)]) * 1.2
    cal = _calibration()
    r1 = ks.rectification(cal)[0]
    true_depth = (points @ r1.T)[:, 2]
    assert np.allclose(_rectified_depth(cal, points), true_depth, rtol=1e-5)
    assert Rectifier.from_calibration(cal).baseline_m == pytest.approx(cal.baseline_m, rel=1e-6)
    inverted = _calibration(np.linalg.inv(_transform()))
    wrong = _rectified_depth(inverted, points)
    assert not np.all(np.isfinite(wrong)) or np.abs(wrong / true_depth - 1).max() > 0.01


def test_parse_camchain_refuses_a_non_rigid_bar() -> None:
    bad = _transform()
    bad[:3, :3] *= 1.1
    with pytest.raises(ValueError):
        ks.parse_camchain(_camchain(bad))
    chain = _camchain()
    chain["cam1"]["resolution"] = [640, 480]
    with pytest.raises(ValueError):
        ks.parse_camchain(chain)


RESULTS = """Calibration results
====================
Camera-system parameters:
cam0 (/cam0/image_raw):
    type: <class 'aslam_cv.libaslam_cv_python.DistortedPinholeCameraGeometry'>
    distortion: [-0.29  0.085  0.0009 -0.0004] +- [0.001 0.002 0.0001 0.0001]
    projection: [530.  531.5 401.5 268. ] +- [0.3 0.3 0.2 0.2]
    reprojection error: [0.000012, -0.000003] +- [0.120000, 0.090000]

cam1 (/cam1/image_raw):
    type: <class 'aslam_cv.libaslam_cv_python.DistortedPinholeCameraGeometry'>
    reprojection error: [-0.000004, 0.000007] +- [0.200000, 0.150000]

baseline T_1_0:
    q: [ 0.003 -0.005  0.001  0.99998] +- [0.0001 0.0001 0.0001]
    t: [-0.0627  0.0006 -0.0004] +- [0.0001 0.0001 0.0001]
"""


def test_reprojection_rms_reads_each_camera() -> None:
    errors = ks.reprojection_rms(RESULTS)
    assert errors == pytest.approx([0.15, 0.25], abs=1e-4)
    assert ks.views_used("Processed 180 images with 97 images used\n") == 97
    assert ks.views_used("") == 0


def test_accept() -> None:
    assert ks.accept([0.15, 0.25], 0.0625, 0.063).accepted
    refused = ks.accept([0.15, 0.31], 0.0625, 0.063)
    assert not refused.accepted and "cam1" in refused.failures[0]
    assert not ks.accept([0.1, 0.1], 0.0605, 0.063).accepted
    assert not ks.accept([0.1], 0.063, 0.063).accepted
    assert not ks.accept([float("nan"), 0.1], 0.063, 0.063).accepted


def test_frame_shift_of_a_new_bar_is_the_rectifications_turn() -> None:
    """Same lenses, another bar: the rectified left eye turns by exactly R1_new R1_old^T."""
    old = _calibration()
    new = _calibration(_transform((0.1, 0.4, -0.3), (-0.0612, -0.0021, 0.0015)))
    shift = ks.rectified_frame_shift(old, new)
    expected = ks.rectification(new)[0] @ ks.rectification(old)[0].T
    assert ks.rotation_angle_deg(shift.rotation @ expected.T) < 1e-4
    assert shift.angle_deg > 0.5
    assert shift.residual_deg < 1e-4
    same = ks.rectified_frame_shift(old, old)
    assert same.angle_deg < 1e-6


def test_frame_shift_sees_a_principal_point_move() -> None:
    """A left principal point 3 px to the right turns the left eye's axis by ~3/f rad."""
    old = _calibration()
    data = _camchain()
    data["cam0"]["intrinsics"] = [K0[0], K0[1], K0[2] + 3.0, K0[3]]
    new = ks.calibration_from_camchain(data, 0.1, "d", "m")
    shift = ks.rectified_frame_shift(old, new)
    assert shift.angle_deg == pytest.approx(math.degrees(3.0 / K0[0]), rel=0.35)


def test_carrying_eye_and_imu_keeps_the_hand_eye() -> None:
    """camera_link <- imu (the eye block, the optical axes, T_cam_imu) is the same before and
    after both blocks are carried through any turn of the rectified frame."""
    eye = {"x_m": 0.0, "y_m": 0.0305, "z_m": 0.0, "roll_deg": -0.18, "pitch_deg": 0.49,
           "yaw_deg": 0.04, "note": "kept by the caller"}  # fmt: skip
    t_cam_imu = np.eye(4)
    t_cam_imu[:3, :3] = rotation_from_rpy(-1.55, 1.52, 0.03)
    t_cam_imu[:3, 3] = (0.0286, -0.0193, -0.0293)
    shift = _rotation((0.4, -0.7, 0.2))
    optical = rotation_from_rpy(*OPTICAL_RPY)

    def link_imu(e: dict[str, Any], t: np.ndarray) -> np.ndarray:
        r_eye = rotation_from_rpy(
            *(math.radians(e[k]) for k in ("roll_deg", "pitch_deg", "yaw_deg"))
        )
        out = np.eye(4)
        out[:3, :3] = r_eye @ optical @ t[:3, :3]
        out[:3, 3] = np.array([e["x_m"], e["y_m"], e["z_m"]]) + r_eye @ optical @ t[:3, 3]
        return out

    moved = ks.carry_eye(eye, shift)
    carried = ks.carry_head_imu(t_cam_imu, shift)
    assert set(moved) == {"x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"}
    before, after = link_imu(eye, t_cam_imu), link_imu(moved, carried)
    assert ks.rotation_angle_deg(before[:3, :3] @ after[:3, :3].T) < 2e-3  # rounded to 1e-3 deg
    assert np.allclose(before[:3, 3], after[:3, 3], atol=1e-6)
    still = ks.carry_eye(eye, np.eye(3))
    assert all(still[k] == pytest.approx(eye[k], abs=1e-9) for k in still)


def test_depth_ratio_is_one_for_the_truth_and_short_for_a_right_cx_slip() -> None:
    truth = _calibration()
    exact = ks.depth_ratio(truth, truth)
    assert all(abs(d.median - 1.0) < 1e-6 and abs(d.offset_px) < 1e-4 for d in exact)
    data = _camchain()
    data["cam1"]["intrinsics"] = [K1[0], K1[1], K1[2] + 2.0, K1[3]]
    slipped = ks.calibration_from_camchain(data, 0.1, "d", "m")
    ratios = ks.depth_ratio(slipped, truth, ranges_m=(0.6, 1.25, 2.5))
    assert 1.0 > ratios[0].median > ratios[1].median > ratios[2].median
    assert all(d.offset_px == pytest.approx(2.0, abs=0.3) for d in ratios)


def test_sharpest_per_window() -> None:
    stamps = [0.0, 0.1, 0.2, 0.26, 0.3, 0.6, 0.7]
    sharp = [1.0, 5.0, 2.0, 1.0, 3.0, 9.0, 8.0]
    assert ks.sharpest_per_window(stamps, sharp, 0.25) == [1, 4, 5]
    assert ks.sharpest_per_window([], [], 0.25) == []
    with pytest.raises(ValueError):
        ks.sharpest_per_window([0.0], [], 0.25)


def test_write_camera_blocks_touches_only_those(tmp_path: Path) -> None:
    file = tmp_path / "camera.json"
    shutil.copy(REPO / "config" / "camera.json", file)
    before = json.loads(file.read_text())
    eye = {**before["stereo"]["eye"], "roll_deg": 1.0}
    ks.write_camera_blocks(file, eye, None)
    after = json.loads(file.read_text())
    assert after["stereo"]["eye"]["roll_deg"] == 1.0
    after["stereo"]["eye"] = before["stereo"]["eye"]
    assert after == before
    with pytest.raises(ValueError):
        ks.write_camera_blocks(file, None, {"T_cam_imu": [[1, 0], [0, 1]]})


# ---- the runner ----------------------------------------------------------------------------
def _tool() -> Any:
    spec = importlib.util.spec_from_file_location(
        "stereo_kalibr", REPO / "ros/tools/stereo_kalibr.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("stereo_kalibr", module)
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


def _part(body: bytes, stamp: float | None) -> bytes:
    stamp_line = f"X-Timestamp: {stamp:.6f}\r\n" if stamp is not None else ""
    return (
        (
            f"--boundarydonotcross\r\nContent-Type: image/jpeg\r\nContent-Length: {len(body)}\r\n"
            f"{stamp_line}\r\n"
        ).encode()
        + body
        + b"\r\n"
    )


def test_read_clip_and_tee(tmp_path: Path) -> None:
    import io

    clip = _part(b"abc", 10.0) + _part(b"defg", 10.0) + _part(b"h", 9.0) + _part(b"ij", None)
    sink = io.BytesIO()
    tee = TOOL.Tee(io.BytesIO(clip), sink)
    while tee.read(5):
        pass
    assert sink.getvalue() == clip
    (tmp_path / "stereo.mjpeg").write_bytes(clip)
    stamps, bodies = TOOL.read_clip(tmp_path / "stereo.mjpeg")
    assert bodies == [b"abc", b"defg", b"h", b"ij"]
    assert stamps[0] == 10.0 and all(b > a for a, b in itertools.pairwise(stamps))


def test_report_and_apply_on_a_config_copy(tmp_path: Path, capsys: Any) -> None:
    """The runner end to end on a copy of config/: report prints, apply writes the file, keeps
    the previous one, carries camera.json's blocks, and a second apply turns nothing."""
    config = tmp_path / "config"
    config.mkdir()
    for name in ("camera.json", "stereo_calibration.json"):
        shutil.copy(REPO / "config" / name, config / name)
    rec = tmp_path / "stereo_test"
    work = rec / "kalibr"
    work.mkdir(parents=True)
    chain = _camchain(_transform((0.4, -0.5, 0.1), (-0.0625, -0.0003, 0.0002)))
    (work / "stereo-camchain.yaml").write_text(yaml.safe_dump(chain))
    (work / "stereo-results-cam.txt").write_text(RESULTS)
    (work / "april.yaml").write_text("target_type: aprilgrid\ntagSize: 0.025\n")
    TOOL.CONFIG = config
    try:
        assert TOOL.main(["report", str(rec), "--config", str(config)]) == 0
        out = capsys.readouterr().out
        assert "ACCEPTED" in out and "the rectified left eye turns" in out
        old_eye = json.loads((config / "camera.json").read_text())["stereo"]["eye"]
        assert TOOL.main(["apply", str(rec), "--config", str(config)]) == 0
        written = StereoCalibration.load(config / "stereo_calibration.json")
        t_10 = np.asarray(chain["cam1"]["T_cn_cnm1"])
        assert written.translation_m == pytest.approx(tuple(t_10[:3, 3]))
        assert written.method.startswith("kalibr_calibrate_cameras")
        assert list(config.glob("stereo_calibration.pre-kalibr-*.json"))
        new_eye = json.loads((config / "camera.json").read_text())["stereo"]["eye"]
        assert new_eye["note"].startswith("CARRIED") and new_eye != old_eye
        capsys.readouterr()
        assert TOOL.main(["apply", str(rec), "--config", str(config)]) == 0
        assert "turns 0.000 deg" in capsys.readouterr().out
    finally:
        TOOL.CONFIG = REPO / "config"
