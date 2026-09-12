"""Copies of ``config/camera.json`` with the optics pinned, for tests that need one provenance.

The committed config's ``calibrated`` flag flips the day the robot's camera is actually
calibrated (``ros/calibrate.sh`` writes it), so a test that needs measured optics — or needs the
nominal pinhole — builds its own file here instead of asserting against the repo's. The one test
that still reads the committed file pins its *shape*, not its state (``test_camera.py``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

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
    reads for the laser's static edge. Returns the path of the copy."""
    data = json.loads((CONFIG_DIR / "camera.json").read_text())
    block = data["overview"]
    if calibration is None:
        block.pop("intrinsics", None)
    else:
        block["intrinsics"] = calibration
    block["calibrated"] = calibration is not None if calibrated is None else calibrated
    (tmp_path / "camera.json").write_text(json.dumps(data))
    (tmp_path / "lidar.json").write_text((CONFIG_DIR / "lidar.json").read_text())
    return str(tmp_path / "camera.json")
