"""Copies of ``config/camera.json`` with the optics pinned, for tests that need one provenance.

The committed config's ``calibrated`` flag flips the day the robot's camera is actually
calibrated (``ros/calibrate.sh`` writes it), so a test that needs measured optics — or needs the
nominal pinhole — builds its own file here instead of asserting against the repo's. The one test
that still reads the committed file pins its *shape*, not its state (``test_camera.py``).

The same for the RIG: which camera is active moves in the committed file the day the head is
changed, so a copy pins that too — :func:`camera_config` is the mono webcam's fixture and
:func:`stereo_config` the stereo module's, with :func:`ideal_stereo_calibration` as a calibration
whose two eyes are already perfect pinholes (its rectification is the identity, so a test can
assert on the pixels that came out of the splitter).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pepin.stereo import StereoCalibration

REPO = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO / "config"

#: A checkerboard's block as ``write_calibration`` writes one: 1280x720, a wide lens with real
#: barrel distortion, the principal point a few pixels off centre.
CALIBRATION: dict[str, Any] = {
    "width": 1280,
    "height": 720,
    "hfov_deg": 70.83,
    "fx": 900.0,
    "fy": 896.0,
    "cx": 646.0,
    "cy": 354.0,
    "dist": [-0.31, 0.1, 0.0005, -0.0004, 0.0],
    "model": "plumb_bob",
    "rms": 0.28,
    "views": 26,
    "board": "9x6 inner corners, 24.0 mm squares",
    "date": "2026-09-12",
}


def camera_config(
    tmp_path: Path,
    calibration: dict[str, Any] | None = None,
    calibrated: bool | None = None,
) -> str:
    """A copy of ``config/camera.json`` in ``tmp_path`` with its optics pinned: ``calibration``
    as the ``intrinsics`` block (no block at all when it is ``None``) and ``calibrated`` set to
    whether there is one unless it is given — ``calibrated=False`` beside a block is the
    switched-off calibration. ``config/lidar.json`` is copied beside it, which the camera node
    reads for the laser's static edge. Returns the path of the copy.

    The copy's active camera is the MONO one: these are the fixtures of the overview webcam and
    its checkerboard, and they must keep saying what they say on the day the robot's head is a
    stereo module (``stereo_config`` below is the fixture for that one)."""
    data = json.loads((CONFIG_DIR / "camera.json").read_text())
    data["active"] = "overview"
    block = data["overview"]
    if calibration is None:
        block.pop("intrinsics", None)
    else:
        block["intrinsics"] = calibration
    block["calibrated"] = calibration is not None if calibrated is None else calibrated
    (tmp_path / "camera.json").write_text(json.dumps(data))
    (tmp_path / "lidar.json").write_text((CONFIG_DIR / "lidar.json").read_text())
    return str(tmp_path / "camera.json")


def ideal_stereo_calibration(
    width: int = 800, height: int = 600, fx: float = 700.0, baseline_m: float = 0.063
) -> StereoCalibration:
    """A stereo head of two IDENTICAL, undistorted pinholes ``baseline_m`` apart, the right one
    to the robot's right and no rotation between them (OpenCV's ``x_right = R x_left + T``, so
    the translation is -baseline along x). Rectifying such a pair is the identity, which is what
    lets a test see the splitter's own pixels on the wire."""
    k = ((fx, 0.0, width / 2.0), (0.0, fx, height / 2.0), (0.0, 0.0, 1.0))
    eye = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    return StereoCalibration(
        width=width,
        height=height,
        k_left=k,
        d_left=(0.0,) * 5,
        k_right=k,
        d_right=(0.0,) * 5,
        rotation=eye,
        translation_m=(-baseline_m, 0.0, 0.0),
        rms_px=0.21,
        date="2026-09-20",
        method="opencv stereo",
        views=24,
        board="9x6 inner corners, 24.5 mm squares",
    )


def stereo_config(tmp_path: Path, calibration: StereoCalibration | None = None) -> str:
    """A copy of ``config/camera.json`` in ``tmp_path`` with the STEREO rig active (and
    ``config/lidar.json`` beside it, which the camera node reads for the laser's static edge).
    With a ``calibration`` its file is written beside them, which is exactly what makes the rig
    calibrated; without one the head is a head with no calibration yet. Returns the config's
    path."""
    data = json.loads((CONFIG_DIR / "camera.json").read_text())
    data["active"] = "stereo"
    (tmp_path / "camera.json").write_text(json.dumps(data))
    (tmp_path / "lidar.json").write_text((CONFIG_DIR / "lidar.json").read_text())
    if calibration is not None:
        calibration.write(tmp_path / data["stereo"]["rig"]["calibration"])
    return str(tmp_path / "camera.json")
