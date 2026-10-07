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
    assert cfg.stream == "http://10.0.0.187:8080/stream?extra_headers=1"  # the grab stamps
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
    assert (cfg.width, cfg.height) == (800, 600)
    assert (cfg.rig.frame_width, cfg.rig.frame_height) == (1600, 600)
    assert cfg.rig.layout == "side_by_side" and cfg.rig.upside_down
    assert cfg.rig.baseline_m_nominal == 0.063
    assert cfg.stream == "http://10.0.0.187:8080/stream?extra_headers=1"  # one camera, one stream
    assert cfg.rig.calibration_path("/ws/config") == Path("/ws/config/stereo_calibration.json")


def test_the_stereo_mount_is_the_neck_s_link_and_the_eye_is_a_block_of_its_own() -> None:
    """The module sits on the webcam's bracket: its link IS the webcam's measured one, on the
    centre line (the board publishes it from the neck). What the left eye adds is the ``eye``
    block — since the rigid mount of 2026-10-02 within a degree of square (the taped module's
    1.7/-2.3/3.6 of 09-30 went stale with the tape; re-derived 2026-10-04 from the head IMU's
    hand-eye and Kalibr)."""
    mono = CameraConfig.load(CAMERA_JSON, name="overview")
    stereo = CameraConfig.load(CAMERA_JSON, name="stereo")
    assert stereo.rig is not None
    assert mount_transform(stereo) == mount_transform(mono)
    assert mono.eye == ()
    eye = dict(stereo.eye)
    assert all(abs(eye[k]) < 2.0 for k in ("roll_deg", "pitch_deg", "yaw_deg")), (
        "the rigid module is square to its bracket within two degrees (the rectified eye's"
        " turn of the stereo calibration rides on the block)"
    )


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


def test_the_rate_is_one_number_per_rig_and_the_grab_lag_is_one_period_at_any_rate(
    tmp_path: Path,
) -> None:
    """config/camera.json's rate block: the stereo head's 10 fps, and the grab stamp's lag
    behind the picture as the 2026-10-07 bench fitted it (92 / 48 / 33 ms at 10 / 20 / 30 fps
    measured; 0.885 x period + 3.5 ms). A rig without the block is refused, never guessed."""
    from pepin.camera import CameraRate, camera_rate, follow_period

    config = REPO / "config/camera.json"
    rate = camera_rate(config, "stereo", environ={})
    assert rate.fps == 10 and rate.period_s == pytest.approx(0.1)
    for fps, measured in ((10, 0.092), (20, 0.048), (30, 0.033)):
        assert rate.grab_lag_s(1.0 / fps) == pytest.approx(measured, abs=0.0015)
    assert rate.grab_lag_s() == pytest.approx(0.092)
    assert camera_rate(config, "overview", environ={}).fps == 15
    data = json.loads(config.read_text())
    del data["stereo"]["rate"]
    (tmp_path / "camera.json").write_text(json.dumps(data))
    with pytest.raises(KeyError, match="no rate block"):
        camera_rate(tmp_path / "camera.json", "stereo", environ={})
    with pytest.raises(ValueError, match="not a positive rate"):
        CameraRate.from_json({"fps": 0, "lag_per_period": 1.0, "lag_offset_s": 0.0})
    assert follow_period(0.0, 0.35, 0.05) == pytest.approx(0.0175)
    assert follow_period(0.02, 0.35, 0.05) == 0.02
