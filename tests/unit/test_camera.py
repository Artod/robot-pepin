"""The camera's numbers: which rig the head is, optics from a field of view, the mount."""

import json
import math
from pathlib import Path

import numpy as np
import pytest
from camera_configs import ideal_stereo_calibration

from pepin.camera import (
    CameraConfig,
    active_camera,
    camera_info_arrays,
    camera_names,
    intrinsics,
    mount_transform,
    optical_rotation,
    optics,
    quaternion_from_rpy,
)

REPO = Path(__file__).resolve().parents[2]
CAMERA_JSON = REPO / "config/camera.json"


def test_the_config_loads_and_names_the_board() -> None:
    """The committed config's shape, not its state: this is the one test that reads the real
    file, and a checkerboard run (ros/calibrate.sh) must not break it — it adds an intrinsics
    block and flips calibrated, and rewrites nothing asserted here. The mono rig by NAME, so
    what it pins stays pinned while the robot's head is whatever it is."""
    cfg = CameraConfig.load(REPO / "config/camera.json", name="overview", board="10.0.0.187")
    assert cfg.stream == "http://10.0.0.187:8080/stream"
    assert (cfg.width, cfg.height) == (1280, 720)
    # the mount measured 2026-09-12: the pitch off the encoder and the level frames
    # (scratch/neck_tilt_scale.txt), the height off a tape
    assert cfg.z_m == 1.203 and cfg.x_m == 0.0 and cfg.pitch_deg == 23.8
    # The nominal pinhole the one reader falls back to with calibrated off. A calibration never
    # rewrites it, but it is the best field of view known: 82.94, what the checkerboard measured.
    assert cfg.hfov_deg == 82.94
    # calibrated and the block go together: on means there are numbers, off means the fallback
    assert cfg.calibrated == (cfg.calibration is not None)


def test_a_nominal_pinhole_puts_the_field_of_view_across_the_image() -> None:
    fx, fy, cx, cy = intrinsics(1280, 720, 90.0)
    assert fx == fy and abs(fx - 640.0) < 1e-9  # 90 degrees: f equals half the width
    assert (cx, cy) == (640.0, 360.0)
    k, d, r, p = camera_info_arrays(1280, 720, 70.0)
    assert len(k) == 9 and len(d) == 5 and len(r) == 9 and len(p) == 12
    assert k[0] == p[0] and k[2] == p[2] == 640.0 and all(v == 0.0 for v in d)
    assert abs(k[0] - 640.0 / math.tan(math.radians(35.0))) < 1e-9


def test_the_mount_and_the_optical_frame_follow_rep_103() -> None:
    cfg = CameraConfig.load(REPO / "config/camera.json", name="overview")
    x, y, z, roll, pitch, yaw = mount_transform(cfg)
    assert (x, y, z) == (0.0, 0.0, 1.203) and (roll, yaw) == (0.0, 0.0)
    assert pitch == pytest.approx(math.radians(23.8))  # down is positive (REP 103)
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


# ---- which rig the head is -------------------------------------------------------------------
def test_the_file_names_its_cameras_and_says_which_one_is_the_head() -> None:
    """The committed config's shape again: two rigs by name, an ``active`` that is one of them,
    and the keys that are not cameras (``active``, ``notes``) left out of the list."""
    data = json.loads(CAMERA_JSON.read_text())
    assert camera_names(data) == ["overview", "stereo"]
    assert data["active"] in camera_names(data)
    assert active_camera(data, environ={}) == data["active"]


def test_the_active_camera_is_decided_in_one_place_and_in_one_order() -> None:
    """An explicit name beats PEPIN_CAMERA, which beats the file's ``active``, which beats the
    mono rig that was here first. An empty string anywhere is "nobody said", so a launch passes
    its argument through without having to know whether it was given."""
    data = json.loads(CAMERA_JSON.read_text())
    assert active_camera(data, "overview", {"PEPIN_CAMERA": "stereo"}) == "overview"
    assert active_camera(data, "", {"PEPIN_CAMERA": "overview"}) == "overview"
    assert active_camera(data, None, {"PEPIN_CAMERA": ""}) == data["active"]
    bare = {name: block for name, block in data.items() if name != "active"}
    assert active_camera(bare, None, {}) == "overview"


def test_an_unknown_camera_stops_the_node_and_names_the_ones_there_are() -> None:
    """A typo in PEPIN_CAMERA or in ``camera:=`` must not leave a node publishing another rig's
    optics: it raises at start, saying what it could have been asked for."""
    data = json.loads(CAMERA_JSON.read_text())
    with pytest.raises(ValueError, match=r"no camera named 'sterio'.*overview, stereo"):
        active_camera(data, "sterio", {})
    with pytest.raises(ValueError, match="no camera named 'stereo'"):
        active_camera({"overview": data["overview"]}, None, {"PEPIN_CAMERA": "stereo"})


def test_the_environment_overrides_the_file_for_one_process(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ros/laptop.sh forwards PEPIN_CAMERA into the container, and that is the whole of running
    the other rig for one container: the loader with no name reads it."""
    monkeypatch.setenv("PEPIN_CAMERA", "overview")
    assert CameraConfig.load(CAMERA_JSON).name == "overview"
    monkeypatch.delenv("PEPIN_CAMERA")
    assert CameraConfig.load(CAMERA_JSON).name == json.loads(CAMERA_JSON.read_text())["active"]


# ---- the stereo rig --------------------------------------------------------------------------
def test_the_stereo_block_is_one_eye_beside_the_frame_that_carries_two() -> None:
    """width/height are ONE EYE — the picture every consumer of /camera/image sees — while the
    rig block carries what is on the wire: one 1600x600 side-by-side frame from a module that is
    mounted upside down, and the vendor's baseline as a number to compare a calibration with."""
    cfg = CameraConfig.load(CAMERA_JSON, name="stereo", board="10.0.0.187")
    assert cfg.stereo and cfg.rig is not None
    assert (cfg.width, cfg.height) == (800, 600) and cfg.hfov_deg == 94.0
    assert (cfg.rig.frame_width, cfg.rig.frame_height) == (1600, 600)
    assert cfg.rig.layout == "side_by_side" and cfg.rig.upside_down
    assert cfg.rig.baseline_m_nominal == 0.063
    assert cfg.stream == "http://10.0.0.187:8080/stream"  # one camera on the board, one stream
    assert cfg.rig.calibration_path("/ws/config") == Path("/ws/config/stereo_calibration.json")


def test_the_stereo_mount_is_the_neck_s_link_and_the_eye_is_a_block_of_its_own() -> None:
    """The module is taped onto the webcam: its link IS the webcam's measured one, on the centre
    line (the board publishes it from the neck). What the left eye adds — measured against the
    floor, a door and the lidar — is the ``eye`` block, and its one unmeasured number says so."""
    mono = CameraConfig.load(CAMERA_JSON, name="overview")
    stereo = CameraConfig.load(CAMERA_JSON, name="stereo")
    assert stereo.rig is not None
    assert mount_transform(stereo) == mount_transform(mono)
    assert mono.eye == ()
    eye = dict(stereo.eye)
    assert set(eye) == {"x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg"}
    assert eye["pitch_deg"] < 0.0 < eye["yaw_deg"], "the module looks up and left of the webcam"
    note = json.loads(CAMERA_JSON.read_text())["stereo"]["eye"]["note"]
    assert "MEASURED" in note and "NOT measured" in note


def test_a_stereo_rig_is_calibrated_exactly_when_its_calibration_file_loads(
    tmp_path: Path,
) -> None:
    """There is no intrinsics block for a stereo head: what makes it able to measure is
    config/stereo_calibration.json, and half a file is not a calibration. The optics answered
    here stay the nominal ONE-EYE pinhole either way — the measured one is the rectified pinhole
    the camera node puts on /camera/camera_info — but the source says which the reader holds."""
    data = json.loads(CAMERA_JSON.read_text())
    config = tmp_path / "camera.json"
    config.write_text(json.dumps(data))
    blind = CameraConfig.load(config, name="stereo")
    assert not blind.calibrated and blind.calibration is None
    assert "uncalibrated stereo head" in optics(blind, 800, 600).source
    (tmp_path / "stereo_calibration.json").write_text('{"width": 800, "height"')  # a torn write
    assert not CameraConfig.load(config, name="stereo").calibrated
    ideal_stereo_calibration().write(tmp_path / "stereo_calibration.json")
    seeing = CameraConfig.load(config, name="stereo")
    assert seeing.calibrated and seeing.calibration is None
    lens = optics(seeing, 800, 600)
    assert not lens.calibrated and "on /camera/camera_info" in lens.source
    assert lens.hfov_deg == pytest.approx(94.0), "the nominal pinhole of one eye"
