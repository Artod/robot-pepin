"""The camera's numbers: optics from a field of view, the mount as transforms."""

import math
from pathlib import Path

import numpy as np

from pepin.camera import (
    CameraConfig,
    camera_info_arrays,
    intrinsics,
    mount_transform,
    optical_rotation,
    quaternion_from_rpy,
)

REPO = Path(__file__).resolve().parents[2]


def test_the_config_loads_and_names_the_board() -> None:
    cfg = CameraConfig.load(REPO / "config/camera.json", board="10.0.0.187")
    assert cfg.stream == "http://10.0.0.187:8080/stream"
    assert (cfg.width, cfg.height) == (1280, 720)
    assert cfg.z_m == 1.23 and cfg.x_m == 0.0 and cfg.pitch_deg == 0.0
    assert not cfg.calibrated  # nominal optics until a checkerboard says otherwise


def test_a_nominal_pinhole_puts_the_field_of_view_across_the_image() -> None:
    fx, fy, cx, cy = intrinsics(1280, 720, 90.0)
    assert fx == fy and abs(fx - 640.0) < 1e-9  # 90 degrees: f equals half the width
    assert (cx, cy) == (640.0, 360.0)
    k, d, r, p = camera_info_arrays(1280, 720, 70.0)
    assert len(k) == 9 and len(d) == 5 and len(r) == 9 and len(p) == 12
    assert k[0] == p[0] and k[2] == p[2] == 640.0 and all(v == 0.0 for v in d)
    assert abs(k[0] - 640.0 / math.tan(math.radians(35.0))) < 1e-9


def test_the_mount_and_the_optical_frame_follow_rep_103() -> None:
    cfg = CameraConfig.load(REPO / "config/camera.json")
    x, y, z, roll, pitch, yaw = mount_transform(cfg)
    assert (x, y, z) == (0.0, 0.0, 1.23) and (roll, pitch, yaw) == (0.0, 0.0, 0.0)
    q = quaternion_from_rpy(*optical_rotation())
    # rotate the optical z axis (0, 0, 1) back into the link frame: it must point along +x
    x_, y_, z_, w = q
    rot = np.array(
        [
            [1 - 2 * (y_**2 + z_**2), 2 * (x_ * y_ - z_ * w), 2 * (x_ * z_ + y_ * w)],
            [2 * (x_ * y_ + z_ * w), 1 - 2 * (x_**2 + z_**2), 2 * (y_ * z_ - x_ * w)],
            [2 * (x_ * z_ - y_ * w), 2 * (y_ * z_ + x_ * w), 1 - 2 * (x_**2 + y_**2)],
        ]
    )
    assert np.allclose(rot @ np.array([0.0, 0.0, 1.0]), [1.0, 0.0, 0.0], atol=1e-9)  # z -> forward
    assert np.allclose(rot @ np.array([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0], atol=1e-9)  # x -> right
    assert np.allclose(rot @ np.array([0.0, 1.0, 0.0]), [0.0, 0.0, -1.0], atol=1e-9)  # y -> down
