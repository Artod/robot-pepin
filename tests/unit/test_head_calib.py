"""The head's Kalibr calibration around the recording: ros/tools/neck_dance.py's dance (inside the
neck's reach, the grid in the picture, every step inside the arbiter's move timeout) and
ros/tools/head_calib.py (the exposure check, the bag's health, Kalibr's result read in the
direction config/camera.json stores it, the runs compared and written without touching anything
else in the file)."""

from __future__ import annotations

import importlib.util
import json
import math
import struct
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[2]


def _tool(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, REPO / f"ros/tools/{name}.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, module)
    spec.loader.exec_module(module)
    return module


DANCE = _tool("neck_dance")
CALIB = _tool("head_calib")
VIO = _tool("vio_config")


@pytest.fixture(scope="module")
def neck() -> Any:
    return DANCE.neck_config(REPO / "config")


@pytest.fixture(scope="module")
def eye() -> Any:
    return DANCE.rectified_eye(REPO / "config")


# ---- the dance --------------------------------------------------------------------------------
@pytest.mark.parametrize("distance", [0.25, 0.28, 0.30])
@pytest.mark.parametrize("centre", [None, (0.0, 0.0), (3.0, -6.0)])
def test_the_dance_stays_in_reach_and_keeps_the_whole_grid_in_view(
    neck: Any, eye: Any, distance: float, centre: tuple[float, float] | None
) -> None:
    """The wall board 0.25-0.30 m from the lens, 25 mm tags: every pose reachable, accepted by
    the arbiter as a slow held look, the grid whole in the picture, within +-20 / +-15 deg."""
    from pepin.gaze import OPERATOR, Aim, GazeSettings, Look, Reach, look_from_json

    if centre is None:
        grid = DANCE.grid_ahead(neck, distance, 1.2, 0.025)
    else:
        grid = DANCE.grid_seen(neck, eye, centre[0], centre[1], distance, 0.025)
    poses = DANCE.grid_dance(neck, eye, grid)
    reach = Reach.of(neck)
    pan0, tilt0 = poses[0].pan_deg, poses[0].tilt_deg
    for pose in poses:
        assert reach.refusal(Aim(math.radians(pose.pan_deg), math.radians(pose.tilt_deg))) is None
        assert pose.visible == 1.0, pose
        assert DANCE.visible_fraction(neck, eye, grid, pose.pan_deg, pose.tilt_deg) == 1.0
        assert abs(pose.pan_deg - pan0) <= 20.0 + 1e-6 and abs(pose.tilt_deg - tilt0) <= 15.0 + 1e-6
        look = look_from_json(
            DANCE.look_request(pose, 7.0), GazeSettings(), neck, lambda frame, xyz: "unused"
        )
        assert isinstance(look, Look), look
        assert (look.band, look.speed, look.frames, look.dwell_s) == (OPERATOR, "slow", 0, 7.0)
    pans = [p.pan_deg - pan0 for p in poses]
    tilts = [p.tilt_deg - tilt0 for p in poses]
    # both rotation axes excited, both ways, and the diagonals
    assert max(pans) >= 12.0 and min(pans) <= -12.0
    assert max(tilts) >= DANCE.MIN_AMPLITUDE_DEG and min(tilts) <= -DANCE.MIN_AMPLITUDE_DEG
    assert any(p != 0 and t != 0 for p, t in zip(pans, tilts, strict=True))
    assert poses[0].hold_s == poses[-1].hold_s == 3.0
    assert {p.hold_s for p in poses[1:-1]} == {1.0}


def test_the_centre_pose_puts_the_grid_in_the_middle_of_the_picture(neck: Any, eye: Any) -> None:
    grid = DANCE.grid_ahead(neck, 0.28, 1.2, 0.025)
    pan, tilt = DANCE.centre_aim(neck, eye, grid.centre)
    u, v, ahead = DANCE.project(neck, eye, pan, tilt, np.asarray([grid.centre]))
    assert (float(u[0]), float(v[0])) == pytest.approx((eye.width / 2, eye.height / 2), abs=0.5)
    assert float(ahead[0]) > 0.2
    # --centre's grid is centred at the pose it was given, at the distance it was given
    seen = DANCE.grid_seen(neck, eye, 2.0, -5.0, 0.3, 0.025)
    assert DANCE.centre_aim(neck, eye, seen.centre) == pytest.approx((2.0, -5.0), abs=0.05)
    position, _ = DANCE.lens(neck, 2.0, -5.0)
    assert float(np.linalg.norm(np.asarray(seen.centre) - position)) == pytest.approx(0.3)


def test_a_grid_out_of_the_picture_is_counted_out(neck: Any, eye: Any) -> None:
    grid = DANCE.grid_ahead(neck, 0.28, 1.2, 0.025)
    pan, tilt = DANCE.centre_aim(neck, eye, grid.centre)
    assert DANCE.visible_fraction(neck, eye, grid, pan, tilt) == 1.0
    assert DANCE.visible_fraction(neck, eye, grid, pan + 45.0, tilt) < 0.25  # vio.md's 45
    assert 0.25 < DANCE.visible_fraction(neck, eye, grid, pan + 30.0, tilt) < 1.0
    assert DANCE.visible_fraction(neck, eye, grid, pan, tilt + 40.0) < 0.25
    assert DANCE.visible_fraction(neck, eye, grid, 180.0, tilt) == 0.0  # behind the lens


def test_every_step_ends_inside_the_move_timeout_and_the_dance_lasts_60_to_90_s(
    neck: Any, eye: Any
) -> None:
    poses = DANCE.grid_dance(neck, eye, DANCE.grid_ahead(neck, 0.28, 1.2, 0.025))
    assert DANCE.too_long(poses, 20.0, 3.0) == []
    assert 60.0 <= DANCE.duration_s(poses, 20.0) <= 90.0
    slow = DANCE.too_long(poses, 5.0, 3.0)  # the knob turned down: the plan is refused
    assert slow and all(seconds + DANCE.STEP_MARGIN_S > 3.0 for _, seconds in slow)


def test_a_grid_too_close_leaves_no_dance(neck: Any, eye: Any) -> None:
    with pytest.raises(ValueError, match=r"move it further away|too close"):
        DANCE.grid_dance(neck, eye, DANCE.grid_ahead(neck, 0.15, 1.2, 0.025))


def test_the_dance_ends_with_a_home_look_the_arbiter_accepts(neck: Any) -> None:
    from pepin.gaze import GazeSettings, Look, look_from_json

    look = look_from_json(DANCE.home_request(), GazeSettings(), neck, lambda f, x: "unused")
    assert isinstance(look, Look) and look.kind == "home" and look.views == ()


def _sighting_of(neck: Any, eye: Any, grid: Any, head: tuple[float, float]) -> dict[str, Any]:
    """What kalibr_detect.py would print for this grid seen with the head at ``head``."""
    position, rotation = DANCE.lens(neck, *head)
    link = rotation.T @ (np.asarray(grid.centre) - position)
    u, v, _ = DANCE.project(neck, eye, *head, grid.tag_corners().reshape(-1, 3))
    return {
        "success": True,
        "corners": 144,
        "of": 144,
        "tags": 36,
        "tag_px": 41.0,
        "corners_px": np.stack([u, v], axis=1).tolist(),
        "centre_cam": [float(-link[1]), float(-link[2]), float(link[0])],  # optical: right, down
    }


def test_check_gives_back_the_placement_the_grid_was_seen_at(neck: Any, eye: Any) -> None:
    """--check: the grid's centre in the optical frame (Kalibr's target pose), seen with the head
    anywhere, turned into the --centre/--distance that put the grid mid-picture."""
    grid = DANCE.grid_ahead(neck, 0.28, 1.2, 0.025)
    want = DANCE.centre_aim(neck, eye, grid.centre)
    for head in [(0.0, -4.0), (5.0, 0.0), (-3.0, -8.0)]:
        seen = DANCE.sighting(_sighting_of(neck, eye, grid, head), eye)
        assert seen.tags == 36 and seen.room_px is not None and seen.room_px > 10
        where = DANCE.placement(neck, eye, seen, head)
        assert where is not None
        assert where[:2] == pytest.approx(want, abs=0.05)
        position, _ = DANCE.lens(neck, *head)
        assert where[2] == pytest.approx(float(np.linalg.norm(np.asarray(grid.centre) - position)))
    assert DANCE.verdict(seen) == []


def test_check_says_what_stands_between_the_picture_and_a_recording() -> None:
    def seen(**kw: Any) -> Any:
        base = {"success": True, "corners": 144, "of": 144, "tags": 36, "tag_px": 41.0,
                "room_px": 80.0, "centre_cam": (0.0, 0.0, 0.3)}  # fmt: skip
        return DANCE.Sighting(**{**base, **kw})

    assert DANCE.verdict(seen()) == []
    assert "Kalibr does not accept" in DANCE.verdict(seen(success=False, tags=3))[0]
    assert "29 of 36 tags" in DANCE.verdict(seen(tags=29))[0]
    assert "under 20" in DANCE.verdict(seen(tag_px=12.0))[0]
    assert "edge" in DANCE.verdict(seen(room_px=4.0))[0]


# ---- the checks around the recording ------------------------------------------------------------
SHOW_AUTO = """\
auto_exposure: value 3 default 3 range 0..3 [1 Manual Mode] [3 Aperture Priority Mode]
exposure_time_absolute: value 156 default 156 range 1..5000 flags inactive
exposure_dynamic_framerate: value 0 default 0 range 0..1
gain: value 0 default 0 range 0..100"""


def test_the_exposure_verdict_reads_ros_exposure_show() -> None:
    capped, why = CALIB.exposure_verdict(SHOW_AUTO)
    assert not capped and "auto" in why
    shutter = SHOW_AUTO.replace("auto_exposure: value 3", "auto_exposure: value 2").replace(
        "value 156", "value 80"
    )
    assert CALIB.exposure_verdict(shutter) == (True, "shutter priority at 8 ms")
    manual = SHOW_AUTO.replace("auto_exposure: value 3", "auto_exposure: value 1")
    capped, why = CALIB.exposure_verdict(manual)  # 15.6 ms: manual but long
    assert not capped and "15.6 ms" in why
    old_names = "exposure_auto: value 1 default 3\nexposure_absolute: value 50 default 156"
    assert CALIB.exposure_verdict(old_names)[0]
    assert not CALIB.exposure_verdict("")[0]


def test_a_stamp_is_read_straight_from_the_cdr_bytes() -> None:
    raw = b"\x00\x01\x00\x00" + struct.pack("<iI", 1_791_100_000, 123_456_789) + b"\x00" * 8
    assert CALIB.cdr_stamp(raw) == pytest.approx(1_791_100_000.123456789, abs=1e-6)
    big = b"\x00\x00\x00\x00" + struct.pack(">iI", 7, 500_000_000)
    assert CALIB.cdr_stamp(big) == 7.5


def test_the_bag_s_gaps_and_rates_are_what_makes_it_unfit() -> None:
    imu = [i / 200.0 for i in range(2000)]
    eyes = [i / 10.0 for i in range(100)]
    streams = {
        "/head/imu": CALIB.stream_of("/head/imu", imu),
        "/camera/image": CALIB.stream_of("/camera/image", eyes),
        "/camera/right/image": CALIB.stream_of("/camera/right/image", eyes),
    }
    assert streams["/head/imu"].rate_hz == pytest.approx(200.0)
    assert CALIB.problems(streams) == []
    holed = imu[:500] + imu[530:]  # a 155 ms hole
    streams["/head/imu"] = CALIB.stream_of("/head/imu", holed)
    del streams["/camera/right/image"]
    found = CALIB.problems(streams)
    assert any("ms gap" in p for p in found) and any("right/image: no messages" in p for p in found)


# ---- Kalibr's result into config/camera.json -----------------------------------------------------
def _asymmetric(
    angle_deg: float = 17.0, t: tuple[float, float, float] = (0.031, -0.012, 0.047)
) -> np.ndarray:
    axis = np.array([0.3, -0.5, 0.81])
    axis /= np.linalg.norm(axis)
    rotation, _ = cv2.Rodrigues(axis * math.radians(angle_deg))
    matrix = np.eye(4)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = t
    return matrix


def _kalibr_run(
    root: Path,
    name: str,
    t_cam_imu: np.ndarray,
    shift_s: float,
    lag_s: float | None = 0.09,
    reprojection: tuple[float, float] = (0.21, 0.23),
) -> Path:
    """A recording directory as ros/calib_run.sh leaves it: Kalibr's two files (in Kalibr's own
    format) and the recording's calib_meta.json."""
    bag = root / name
    (bag / "kalibr").mkdir(parents=True)
    t_cam1 = VIO.cam1_from_cam0(0.061) @ t_cam_imu
    blocks = []
    for cam, t in (("cam0", t_cam_imu), ("cam1", t_cam1)):
        rows = "\n".join(f"  - [{', '.join(repr(float(v)) for v in row)}]" for row in t)
        blocks.append(f"{cam}:\n  T_cam_imu:\n{rows}\n  timeshift_cam_imu: {shift_s!r}\n")
    (bag / "kalibr/calib-camchain-imucam.yaml").write_text("".join(blocks))
    (bag / "kalibr/calib-results-imucam.txt").write_text(
        "Calibration results\n===================\nNormalized Residuals\n"
        f"Reprojection error (cam0):     mean 0.4, median 0.3, std: 0.2\n"
        "Residuals\n----------------------------\n"
        f"Reprojection error (cam0) [px]:     mean {reprojection[0]}, median 0.18, std: 0.1\n"
        f"Reprojection error (cam1) [px]:     mean {reprojection[1]}, median 0.19, std: 0.1\n"
        "Gyroscope error (imu0) [rad/s]:     mean 0.0031, median 0.0027, std: 0.002\n"
        "Accelerometer error (imu0) [m/s^2]: mean 0.041, median 0.035, std: 0.02\n"
    )
    if lag_s is not None:
        meta = {"camera_stamp_lag_s": lag_s, "recorded_local": "2026-10-04T15:00:00-04:00"}
        (bag / "calib_meta.json").write_text(json.dumps(meta))
    return bag


def _config(tmp_path: Path) -> Path:
    config = tmp_path / "config"
    config.mkdir()
    for name in ("camera.json", "stereo_calibration.json", "head_imu.json", "knobs.json"):
        (config / name).write_text((REPO / "config" / name).read_text())
    return config


def test_kalibr_s_t_cam_imu_is_stored_as_kalibr_gives_it_and_inverted_only_for_openvins(
    tmp_path: Path,
) -> None:
    truth = _asymmetric()
    a = CALIB.read_result(_kalibr_run(tmp_path, "calib_a", truth, 0.0042))
    assert np.allclose(a.t_cam_imu, truth)
    assert a.reprojection_px == (0.21, 0.23) and a.gyro_error == 0.0031 and a.accel_error == 0.041
    config = _config(tmp_path)
    # Kalibr and the tapes ran with the camera at 10 fps; the shipped rate is 20 since 2026-10-07,
    # so pin the temp config's rate to 10 before the lag default is read and the block written.
    cam = json.loads((config / "camera.json").read_text())
    cam["stereo"]["rate"]["fps"] = 10
    (config / "camera.json").write_text(json.dumps(cam, indent=2) + "\n")
    block = CALIB.head_imu_block([a], CALIB.stamp_lag_default(config), 10)
    sys.path.insert(0, str(REPO / "src"))
    from pepin.camera import write_head_imu

    write_head_imu(config / "camera.json", block)
    stored = np.asarray(
        json.loads((config / "camera.json").read_text())["stereo"]["head_imu"]["T_cam_imu"]
    )
    assert np.allclose(stored, truth, atol=1e-6)
    assert not np.allclose(stored, np.linalg.inv(truth), atol=1e-3)
    rig = VIO.load_rig(config)
    assert np.allclose(rig.t_cam_imu, truth, atol=1e-6)
    # recorded at a fixed 0.09, stored against the knob's default: following the rate at 10 fps,
    # 0.885 x 100 ms + 3.5 ms = 92 ms, so 2 ms more
    assert block["stamp_lag_s"] == pytest.approx(0.092) and block["camera_fps"] == 10
    assert rig.time_offset_s == pytest.approx(0.0042 + 0.002)
    chain = VIO.imucam_chain(rig)
    first = chain.split("T_imu_cam:")[1].splitlines()[1:5]
    t_imu_cam = np.array(
        [[float(v) for v in row.split("[")[1].rstrip("]").split(",")] for row in first]
    )
    assert np.allclose(t_imu_cam, np.linalg.inv(truth), atol=1e-6)


def test_the_block_is_written_and_nothing_else_in_the_file_moves(tmp_path: Path) -> None:
    config = _config(tmp_path)
    before = (config / "camera.json").read_text()
    assert "\\u2014" in before  # the file keeps its non-ASCII as escapes
    a = CALIB.read_result(_kalibr_run(tmp_path, "calib_a", _asymmetric(), 0.0042))
    sys.path.insert(0, str(REPO / "src"))
    from pepin.camera import write_head_imu

    write_head_imu(config / "camera.json", CALIB.head_imu_block([a], 0.09, 10))
    after = (config / "camera.json").read_text()
    old, new = json.loads(before), json.loads(after)
    old["stereo"].pop("head_imu"), new["stereo"].pop("head_imu")
    assert old == new
    assert after.count("\\u2014") == before.count("\\u2014") and "—" not in after
    head = before[: before.index('"head_imu"')]
    assert after.startswith(head)  # every byte before the block is the same
    tail = before[before.index('"notes": "The lens is measured') :]
    assert after.endswith(tail)  # and every byte after the stereo block
    with pytest.raises(ValueError, match=r"orthonormal|4x4"):
        write_head_imu(config / "camera.json", {"T_cam_imu": [[2, 0, 0, 0]] * 3 + [[0, 0, 0, 1]]})
    assert (config / "camera.json").read_text() == after  # a refused block leaves the file


def test_the_time_offset_is_taken_back_to_the_knob_s_default_lag(tmp_path: Path) -> None:
    run = CALIB.read_result(_kalibr_run(tmp_path, "calib_a", _asymmetric(), 0.0142, lag_s=0.10))
    assert run.time_offset_s(0.09) == pytest.approx(0.0042)  # stamps dated 10 ms more
    unknown = CALIB.read_result(_kalibr_run(tmp_path, "calib_b", _asymmetric(), 0.0042, lag_s=None))
    assert unknown.time_offset_s(0.09) == pytest.approx(0.0042)  # no meta: the default assumed


def test_two_runs_must_agree_within_half_a_degree_5_mm_and_2_ms(tmp_path: Path) -> None:
    a = CALIB.read_result(_kalibr_run(tmp_path, "a", _asymmetric(17.0), 0.0042))
    b = CALIB.read_result(
        _kalibr_run(tmp_path, "b", _asymmetric(17.3, (0.033, -0.012, 0.047)), 0.0050)
    )
    agree = CALIB.agreement(a, b, 0.09)
    assert agree.rotation_deg == pytest.approx(0.3, abs=1e-6)
    assert agree.translation_mm == pytest.approx(2.0, abs=1e-6)
    assert agree.time_ms == pytest.approx(-0.8, abs=1e-6)
    assert agree.ok and CALIB.failures([a, b], 0.09) == []
    far = CALIB.read_result(_kalibr_run(tmp_path, "c", _asymmetric(17.7), 0.0042))
    assert not CALIB.agreement(a, far, 0.09).ok
    assert CALIB.failures([a], 0.09) == ["one run only: procedure D wants two that agree"]
    blurry = CALIB.read_result(
        _kalibr_run(tmp_path, "d", _asymmetric(), 0.0042, reprojection=(0.71, 0.3))
    )
    assert any("cam0 reprojection 0.710" in f for f in CALIB.failures([a, blurry], 0.09))
    mean = CALIB.mean_transform([a.t_cam_imu, b.t_cam_imu])
    assert CALIB.rotation_deg(mean, a.t_cam_imu) == pytest.approx(0.15, abs=1e-3)
    assert np.allclose(mean[:3, 3], (0.032, -0.012, 0.047))
    block = CALIB.head_imu_block([a, b], 0.09, 10)
    assert block["time_offset_s"] == pytest.approx(0.0046)
    assert block["date"] == "2026-10-04" and "a, b (mean)" in block["method"]


def test_apply_refuses_a_set_that_fails_unless_forced(tmp_path: Path) -> None:
    config = _config(tmp_path)
    before = (config / "camera.json").read_text()
    one = _kalibr_run(tmp_path, "calib_a", _asymmetric(), 0.0042)
    assert CALIB.main(["apply", str(one), "--config", str(config)]) == 1
    assert (config / "camera.json").read_text() == before
    assert CALIB.main(["apply", str(one), "--config", str(config), "--force"]) == 0
    stored = json.loads((config / "camera.json").read_text())["stereo"]["head_imu"]
    assert np.allclose(stored["T_cam_imu"], _asymmetric(), atol=1e-6)


def test_a_run_recorded_with_the_lag_following_the_rate_is_read_at_the_rules_lag(
    tmp_path: Path,
) -> None:
    """calib_meta.json's camera_stamp_lag_s 0 is camera_stream's follow mode: the run's stamps
    were dated by the rate's rule at the rate it recorded (its camera_fps)."""
    bag = _kalibr_run(tmp_path, "calib_f", _asymmetric(), 0.0042, lag_s=0.0)
    meta = json.loads((bag / "calib_meta.json").read_text())
    meta["camera_fps"] = 20
    (bag / "calib_meta.json").write_text(json.dumps(meta))
    run = CALIB.read_result(bag)
    assert run.stamp_lag_s == pytest.approx(0.885 * 0.05 + 0.0035)
