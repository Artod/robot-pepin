"""pepin.stereo: the side-by-side cut, the calibration file and the rectified pinhole."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pepin.stereo import MIN_DISPARITY_PX, Rectifier, SideBySide, StereoCalibration

IDENTITY = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))


def _calibration(fx: float = 400.0, baseline: float = 0.063) -> StereoCalibration:
    """Two identical ideal pinholes, the right one ``baseline`` metres to the right."""
    k = ((fx, 0.0, 400.0), (0.0, fx, 300.0), (0.0, 0.0, 1.0))
    return StereoCalibration(
        width=800, height=600, k_left=k, d_left=(0.0,) * 5, k_right=k, d_right=(0.0,) * 5,
        rotation=IDENTITY, translation_m=(-baseline, 0.0, 0.0), rms_px=0.2, date="2026-09-21",
        method="chessboard", views=30, board="9x6 24.5 mm",
    )  # fmt: skip


def test_an_upright_frame_is_cut_down_the_middle() -> None:
    frame = np.arange(2 * 6 * 3, dtype=np.uint8).reshape(2, 6, 3)
    left, right = SideBySide().eyes(frame)
    assert np.array_equal(left, frame[:, :3]) and np.array_equal(right, frame[:, 3:])


def test_an_upside_down_module_is_turned_back_and_its_eyes_trade_places() -> None:
    """Rotated 180 degrees about the optical axis, the module's left lens is on the robot's right:
    each half is rotated back and the halves swap, so a near object keeps a POSITIVE disparity."""
    frame = np.zeros((4, 8), dtype=np.uint8)
    frame[1, 1] = 200  # in the transport frame's first half ...
    frame[1, 6] = 100  # ... and in its second
    left, right = SideBySide(upside_down=True).eyes(frame)
    assert left[2, 1] == 100, "the robot's left eye is the second half, rotated"
    assert right[2, 2] == 200, "the robot's right eye is the first half, rotated"


def test_the_calibration_file_round_trips_and_a_half_written_one_is_refused(tmp_path: Path) -> None:
    calibration = _calibration()
    path = tmp_path / "stereo_calibration.json"
    calibration.write(path)
    assert StereoCalibration.load(path) == calibration
    assert calibration.baseline_m == pytest.approx(0.063)
    broken = calibration.to_json()
    del broken["k_right"]
    path.write_text(json.dumps(broken))
    with pytest.raises(KeyError):
        StereoCalibration.load(path)


def test_a_disparity_becomes_metres_and_the_far_end_is_unknown() -> None:
    """z = fx * baseline / disparity: 400 px * 0.063 m over 12.6 px is 2 m; under half a pixel the
    answer would be tens of metres with an error as large, so it is NaN."""
    rectifier = Rectifier.from_calibration(_calibration())
    assert rectifier.fx == pytest.approx(400.0, rel=1e-3)
    assert rectifier.baseline_m == pytest.approx(0.063, rel=1e-6)
    disparity = np.array([[12.6, 25.2, MIN_DISPARITY_PX / 2, -1.0, np.nan]], dtype=np.float32)
    depth = rectifier.depth_m(disparity)
    assert depth[0, 0] == pytest.approx(2.0, rel=1e-3) and depth[0, 1] == pytest.approx(
        1.0, rel=1e-3
    )
    assert np.isnan(depth[0, 2:]).all()
    assert rectifier.right_projection_tx() == pytest.approx(-400.0 * 0.063, rel=1e-3)


def test_ideal_eyes_come_out_of_the_rectifier_as_they_went_in() -> None:
    rectifier = Rectifier.from_calibration(_calibration())
    left = np.random.default_rng(1).integers(0, 255, (600, 800), dtype=np.uint8)
    out_left, out_right = rectifier.rectify(left, left)
    assert out_left.shape == (600, 800)
    assert (
        np.abs(out_left[50:550, 50:750].astype(int) - left[50:550, 50:750].astype(int)).max() <= 1
    )
    assert np.array_equal(out_left, out_right)
