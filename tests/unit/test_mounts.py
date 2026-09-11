"""One loader for every sensor's mount: the config files, the frames, the rotations."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from pepin.camera import OPTICAL_RPY, CameraConfig, mount_transform, optical_rotation
from pepin.deployment import config_file
from pepin.lidar import LidarMount
from pepin.mounts import OPTICAL_MOUNT, Mount, Mounts, lidar_mount, rotation_from_rpy

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "config"


def test_a_mount_block_reads_its_six_fields_and_nothing_else() -> None:
    """A note is for people; height_m (tof.json's word) is z_m; the rest defaults to zero."""
    mount = Mount.from_json({"x_m": 0.1, "y_m": -0.2, "height_m": 0.3, "yaw_deg": 90, "note": "x"})
    assert mount == Mount(x_m=0.1, y_m=-0.2, z_m=0.3, yaw_deg=90.0)
    x, y, z, roll, pitch, yaw = mount.transform()
    assert (x, y, z, roll, pitch) == (0.1, -0.2, 0.3, 0.0, 0.0) and yaw == pytest.approx(
        math.pi / 2
    )
    assert Mount.from_json({"z_m": 1.0, "height_m": 2.0}).z_m == 1.0  # z_m wins when both are there


def test_the_rotation_turns_the_sensor_s_axes_into_base_link() -> None:
    """The IMU's roll of +90 deg maps the chip's Y onto base_link's Z (what the C++ bridge
    applies); a yaw of +90 maps the sensor's x onto base_link's y."""
    up = Mount(roll_deg=90.0).rotation() @ np.array([0.0, 1.0, 0.0])
    assert np.allclose(up, [0.0, 0.0, 1.0], atol=1e-12)
    left = Mount(yaw_deg=90.0).rotation() @ np.array([1.0, 0.0, 0.0])
    assert np.allclose(left, [0.0, 1.0, 0.0], atol=1e-12)
    assert np.allclose(Mount().rotation(), np.eye(3))
    assert np.array_equal(Mount(x_m=1.0, z_m=2.0).translation(), [1.0, 0.0, 2.0])
    # the composition is Rz Ry Rx: a pitch after a roll, seen from the parent
    r = rotation_from_rpy(0.3, -0.2, 1.1)
    assert np.allclose(r @ r.T, np.eye(3)) and np.linalg.det(r) == pytest.approx(1.0)


def test_the_files_load_and_each_sensor_is_where_the_stack_has_always_put_it() -> None:
    """config/lidar.json, imu.json, camera.json and tof.json through one loader, from a
    directory or from wherever pepin.deployment.config_file finds them."""
    mounts = Mounts.load(CONFIG)
    assert mounts == Mounts.load()  # config_file resolves the same checkout
    x, y, z, roll, pitch, yaw = mounts.lidar.transform()
    assert (x, y, z) == (0.005, 0.0, 0.20) and pitch == 0.0
    assert roll == math.pi and yaw == pytest.approx(-1.5272, abs=1e-4)  # the launch's old defaults
    assert mounts.lidar_sensor == LidarMount.from_json(CONFIG / "lidar.json")
    assert mounts.lidar_sensor.masked_sectors_deg == ((192, 218), (317, 343))
    ix, iy, iz, iroll, ipitch, iyaw = mounts.imu.transform()
    assert (ix, iy, iz) == (0.0, 0.0, 0.10) and (ipitch, iyaw) == (0.0, 0.0)
    assert math.isclose(iroll, math.pi / 2), "the chip's Y up: roll +90 deg"
    assert set(mounts.tof) == {"front", "left", "right"}
    assert mounts.tof["left"] == Mount(x_m=0.027, y_m=0.148, z_m=0.16)
    assert mounts.tof["right"].z_m == 0.165 and mounts.tof["front"].z_m == 0.27


def test_the_lidar_mount_is_the_lidar_s_own_transform() -> None:
    """pepin.lidar.LidarMount.transform() and the Mount built from it agree exactly: the yaw
    is the negative of the calibrated offset, the roll is the hanging sensor's."""
    sensor = LidarMount.from_json(CONFIG / "lidar.json")
    assert lidar_mount(sensor).transform() == sensor.transform()
    assert lidar_mount(LidarMount(yaw_offset_deg=30.0, z_m=0.1)) == Mount(z_m=0.1, yaw_deg=-30.0)


def test_the_camera_s_two_frames_are_exactly_what_the_camera_node_publishes() -> None:
    """base_link -> camera_link is pepin.camera.mount_transform (the neck's pitch, x forward);
    camera_link -> camera_optical is the REP 103 rotation (-90, 0, -90 deg) and no offset."""
    cfg = CameraConfig.load(CONFIG / "camera.json")
    camera = Mounts.load(CONFIG).camera
    assert camera.link.transform() == mount_transform(cfg)
    assert camera.link.pitch_deg == 26.0 and camera.link.z_m == 1.23
    assert camera.optical.transform()[:3] == (0.0, 0.0, 0.0)
    assert camera.optical.transform()[3:] == pytest.approx(optical_rotation(), abs=1e-15)
    assert OPTICAL_MOUNT.transform()[3:] == pytest.approx(OPTICAL_RPY, abs=1e-15)
    assert (camera.link_frame, camera.optical_frame) == ("camera_link", "camera_optical")
    # the optical z looks along the link's x, its x to the right, its y down
    r = camera.optical.rotation()
    assert np.allclose(r @ [0.0, 0.0, 1.0], [1.0, 0.0, 0.0], atol=1e-12)
    assert np.allclose(r @ [1.0, 0.0, 0.0], [0.0, -1.0, 0.0], atol=1e-12)
    assert np.allclose(r @ [0.0, 1.0, 0.0], [0.0, 0.0, -1.0], atol=1e-12)


def test_every_static_frame_of_the_cart_is_listed_once_under_its_name() -> None:
    frames = Mounts.load(CONFIG).static_frames()
    names = [(parent, child) for parent, child, _mount in frames]
    assert names == [
        ("base_link", "laser"),
        ("base_link", "imu_link"),
        ("base_link", "camera_link"),
        ("camera_link", "camera_optical"),
        ("base_link", "tof_front"),
        ("base_link", "tof_left"),
        ("base_link", "tof_right"),
    ]
    assert len(set(names)) == len(names)


def test_a_sensor_without_a_measured_mount_is_left_out(tmp_path: Path) -> None:
    """tof.json says null for a sensor nobody measured: it feeds the reflex, not a frame."""
    for name in ("lidar.json", "imu.json", "camera.json"):
        (tmp_path / name).write_text((CONFIG / name).read_text())
    tof = json.loads((CONFIG / "tof.json").read_text())
    tof["sensors"]["left"]["mount"] = None
    (tmp_path / "tof.json").write_text(json.dumps(tof))
    assert set(Mounts.load(tmp_path).tof) == {"front", "right"}
    assert config_file("tof.json") == CONFIG / "tof.json"
