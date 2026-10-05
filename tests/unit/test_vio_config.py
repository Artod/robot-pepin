"""ros/tools/vio_config.py: OpenVINS's and Kalibr's files written from the repo's own numbers,
the direction of every transform held by tests that would catch a flip (vio.md S7)."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _tool() -> Any:
    spec = importlib.util.spec_from_file_location("vio_config", REPO / "ros/tools/vio_config.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("vio_config", module)
    spec.loader.exec_module(module)
    return module


TOOL = _tool()


def _asymmetric() -> np.ndarray:
    """A 17 deg rotation about a skew axis and an offset: nothing about it is symmetric, so an
    inverse taken the wrong way round cannot pass for the right one."""
    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    rotation, _ = cv2.Rodrigues(axis * math.radians(17.0))
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = (0.031, -0.012, 0.047)
    return matrix


def test_the_inverse_is_exact_and_round_trips_an_asymmetric_transform() -> None:
    t = _asymmetric()
    inverse = TOOL.invert(t)
    assert np.allclose(inverse @ t, np.eye(4), atol=1e-12)
    assert np.allclose(TOOL.invert(inverse), t, atol=1e-12)
    assert not np.allclose(inverse, t), "an asymmetric transform is not its own inverse"
    point_in_imu = np.array([0.1, 0.2, 0.3, 1.0])
    in_camera = t @ point_in_imu  # T_cam_imu: IMU coordinates into the camera's
    assert np.allclose(inverse @ in_camera, point_in_imu)


def test_the_nominal_extrinsics_are_a_proper_rotation_from_the_axis_photo() -> None:
    t = TOOL.nominal_t_cam_imu("x,-z,y", (0.0, 0.03, 0.04))
    assert np.allclose(t[:3, 0], (1, 0, 0)) and np.allclose(t[:3, 1], (0, 0, -1))
    assert np.allclose(t[:3, 3], (0.0, 0.03, 0.04))
    with pytest.raises(ValueError, match="right-handed"):
        TOOL.nominal_t_cam_imu("x,y,-z", (0, 0, 0))
    with pytest.raises(ValueError, match="axis"):
        TOOL.nominal_t_cam_imu("x,y,w", (0, 0, 0))


def _config_with_imu(tmp_path: Path, t_cam_imu: np.ndarray | None) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    for name in ("stereo_calibration.json", "head_imu.json"):
        (tmp_path / name).write_text((REPO / "config" / name).read_text())
    data = json.loads((REPO / "config/camera.json").read_text())
    if t_cam_imu is not None:
        data["stereo"]["head_imu"] = {
            "T_cam_imu": t_cam_imu.tolist(),
            "time_offset_s": 0.0042,
        }
    else:
        data["stereo"].pop("head_imu", None)  # the shipped config may carry one
    (tmp_path / "camera.json").write_text(json.dumps(data))
    return tmp_path


def _read(path: Path, *keys: str) -> Any:
    """A value the way OpenVINS reads it: cv::FileStorage."""
    storage = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
    node = storage.root()
    for key in keys:
        node = node.getNode(key)
    if node.isSeq():
        values = [node.at(i) for i in range(node.size())]
        if values and values[0].isSeq():
            return np.array([[v.at(j).real() for j in range(v.size())] for v in values])
        return [v.real() if v.isReal() or v.isInt() else v.string() for v in values]
    return node.real() if node.isReal() or node.isInt() else node.string()


def _bool(path: Path, key: str) -> bool:
    """A bool the way OpenVINS's YamlParser reads it: the scalar's first word, before any '#'
    or space (cv::FileStorage keeps a trailing comment in the string)."""
    value = str(_read(path, key))
    word = value.split("#")[0].split(" ")[0]
    assert word in ("true", "false"), f"{key}: {value!r} is no bool OpenVINS reads"
    return word == "true"


def test_the_openvins_files_carry_the_rectified_pinhole_the_inverse_and_the_baseline(
    tmp_path: Path,
) -> None:
    """The files parse with cv::FileStorage as OpenVINS reads them; the intrinsics are the
    rectified P camera_stream publishes; cam0's T_imu_cam is inv(T_cam_imu); cam1 sits +baseline
    along cam0's optical x; the time shift and the IMU's densities, topic and rate ride along."""
    from pepin.stereo import Rectifier, StereoCalibration

    t_cam_imu = _asymmetric()
    config = _config_with_imu(tmp_path / "config", t_cam_imu)
    out = tmp_path / "vio"
    assert TOOL.main(["--config", str(config), "--out", str(out), "--tag-size", "0.0341"]) == 0
    rectifier = Rectifier.from_calibration(
        StereoCalibration.load(config / "stereo_calibration.json")
    )
    chain = out / "kalibr_imucam_chain.yaml"
    intrinsics = _read(chain, "cam0", "intrinsics")
    assert intrinsics == pytest.approx([rectifier.fx, rectifier.fy, rectifier.cx, rectifier.cy])
    assert _read(chain, "cam0", "resolution") == [800, 600]
    assert _read(chain, "cam0", "distortion_coeffs") == [0.0, 0.0, 0.0, 0.0]
    t_imu_cam0 = _read(chain, "cam0", "T_imu_cam")
    assert np.allclose(t_imu_cam0, np.linalg.inv(t_cam_imu), atol=1e-9)
    t_imu_cam1 = _read(chain, "cam1", "T_imu_cam")
    cam1_in_cam0 = np.linalg.inv(t_imu_cam0) @ t_imu_cam1
    assert cam1_in_cam0[0, 3] == pytest.approx(rectifier.baseline_m, abs=1e-9), "right of cam0"
    assert np.allclose(cam1_in_cam0[:3, :3], np.eye(3), atol=1e-9), "rectified: no rotation"
    assert _read(chain, "cam0", "timeshift_cam_imu") == pytest.approx(0.0042)
    assert _read(chain, "cam0", "rostopic") == "/camera/image"
    assert _read(chain, "cam1", "rostopic") == "/camera/right/image"
    imu = out / "kalibr_imu_chain.yaml"
    assert _read(imu, "imu0", "rostopic") == "/head/imu"
    assert _read(imu, "imu0", "update_rate") == 200.0
    noise = json.loads((REPO / "config/head_imu.json").read_text())["noise"]
    assert _read(imu, "imu0", "gyroscope_noise_density") == pytest.approx(
        noise["gyro_noise_density"]
    ), "the IMU yaml carries config/head_imu.json's noise block (the Allan block x10)"
    estimator = out / "estimator_config.yaml"
    assert _bool(estimator, "try_zupt") is True, "OpenVINS's own ZUPT holds rest"
    assert _read(estimator, "track_frequency") == 15.0
    assert _read(estimator, "relative_config_imucam") == "kalibr_imucam_chain.yaml"
    text = estimator.read_text()
    assert "calib_cam_timeoffset: true" in text and "calib_cam_extrinsics: false" in text
    camchain = (out / "camchain.yaml").read_text()
    assert f"- [1, 0, 0, {-rectifier.baseline_m:.10g}]" in camchain, "Kalibr's T_cn_cnm1"
    assert "tagSize: 0.0341" in (out / "april.yaml").read_text()


def test_without_the_head_imu_block_only_kalibr_s_files_or_a_nominal_guess(tmp_path: Path) -> None:
    config = _config_with_imu(tmp_path / "config", None)
    with pytest.raises(SystemExit, match="head_imu"):
        TOOL.main(["--config", str(config), "--out", str(tmp_path / "a")])
    assert TOOL.main(["--config", str(config), "--out", str(tmp_path / "b"), "--kalibr-only"]) == 0
    assert sorted(p.name for p in (tmp_path / "b").iterdir()) == ["camchain.yaml", "imu.yaml"]
    args = [
        "--config",
        str(config),
        "--out",
        str(tmp_path / "c"),
        "--nominal",
        "x,-z,y",
        "0",
        "0.03",
        "0.04",
    ]
    assert TOOL.main(args) == 0
    chain = (tmp_path / "c" / "kalibr_imucam_chain.yaml").read_text()
    assert "NOMINAL: axes x,-z,y" in chain


def test_the_nominal_block_is_one_camera_json_accepts(tmp_path: Path) -> None:
    """--print-block writes the guess as config/camera.json's head_imu block, which the camera's
    reader accepts as a rigid transform and the generator then uses as T_cam_imu."""
    from pepin.camera import head_imu_transform

    block = TOOL.nominal_block("x,-z,y", (0.0, 0.03, 0.04))
    assert head_imu_transform(block)[0][:3] == (1.0, 0.0, 0.0)
    config = _config_with_imu(tmp_path / "config", np.asarray(block["T_cam_imu"], dtype=float))
    rig = TOOL.load_rig(config)
    assert np.allclose(rig.t_cam_imu, TOOL.nominal_t_cam_imu("x,-z,y", (0.0, 0.03, 0.04)))


def test_a_live_stamp_lag_moves_the_time_shift_by_its_distance_from_the_knob_default(
    tmp_path: Path,
) -> None:
    """time_offset_s is measured against camera_stream's stamps at camera_stamp_lag_s's default;
    a live lag dates every frame earlier by the difference, and the time shift follows it."""
    config = _config_with_imu(tmp_path / "config", _asymmetric())
    knobs = json.loads((REPO / "config/knobs.json").read_text())
    knobs["camera_stream"]["camera_stamp_lag_s"]["default"] = 0.02
    (config / "knobs.json").write_text(json.dumps(knobs))
    assert TOOL.stamp_lag_default(config) == pytest.approx(0.02)
    out = tmp_path / "vio"
    assert TOOL.main(["--config", str(config), "--out", str(out)]) == 0
    chain = out / "kalibr_imucam_chain.yaml"
    assert _read(chain, "cam0", "timeshift_cam_imu") == pytest.approx(0.0042), "the default"
    args = ["--config", str(config), "--out", str(out), "--camera-stamp-lag", "0.093"]
    assert TOOL.main(args) == 0
    assert _read(chain, "cam0", "timeshift_cam_imu") == pytest.approx(0.0042 + 0.073)
    assert _read(chain, "cam1", "timeshift_cam_imu") == pytest.approx(0.0042 + 0.073)
    assert "camera_stamp_lag_s 0.0930" in chain.read_text()
    assert TOOL.stamp_lag_default(tmp_path) == 0.0, "no knobs.json: no lag"


def test_the_zupt_holds_rest_and_the_dynamic_init_is_a_switch(tmp_path: Path) -> None:
    """By default OpenVINS's ZUPT is on (all day, the IMU test at multiplier 1, a slow drive's
    speed and a moving picture's disparity out) and the init is static only; --dyn-init adds the
    dynamic one, --no-zupt writes the first design's (ZUPT off), each alone."""
    config = _config_with_imu(tmp_path / "config", _asymmetric())

    def estimator(*flags: str) -> Path:
        out = tmp_path / ("vio" + "".join(flags))
        assert TOOL.main(["--config", str(config), "--out", str(out), *flags]) == 0
        return out / "estimator_config.yaml"

    default = estimator()
    for path in (default, estimator("--dyn-init"), estimator("--no-zupt")):
        for line in path.read_text().splitlines():
            key, _, value = line.partition(":")
            if value.split("#")[0].strip() in ("true", "false"):
                _bool(path, key)  # cv::FileStorage reads 'false # a: b' as '': a colon breaks it
    assert _bool(default, "try_zupt") is True
    assert _bool(default, "zupt_only_at_beginning") is False, "all day, not only at the start"
    assert _read(default, "zupt_chi2_multipler") == 1.0, "0 would leave the picture alone"
    assert _read(default, "zupt_noise_multiplier") == 1.0
    assert _read(default, "zupt_max_velocity") == pytest.approx(0.05)
    assert _read(default, "zupt_max_disparity") == pytest.approx(0.5), "a still picture passes"
    assert _bool(default, "init_dyn_use") is False
    assert _read(default, "init_imu_thresh") == pytest.approx(0.4)

    dynamic = estimator("--dyn-init")
    assert _bool(dynamic, "init_dyn_use") is True
    assert _bool(dynamic, "try_zupt") is True, "the switch is the init's alone"
    assert _read(dynamic, "init_dyn_num_pose") == 6.0

    first = estimator("--no-zupt")
    assert _bool(first, "try_zupt") is False
    assert _bool(first, "init_dyn_use") is False
    without = default.read_text().splitlines()
    differ = [a for a, b in zip(without, first.read_text().splitlines(), strict=True) if a != b]
    assert all(line.startswith("zupt_") or line.startswith("try_zupt") for line in differ), differ
