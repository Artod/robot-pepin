"""The neck's numbers: ticks to angles, angles to the camera's pose, the base server's reply."""

import json
import math
from pathlib import Path

import pytest

from pepin.camera import CameraConfig, mount_transform
from pepin.neck import (
    JOINT_NAMES,
    RAD_PER_TICK,
    NeckAngles,
    NeckConfig,
    NeckPivot,
    NeckReference,
    camera_pose,
    joint_angles,
    neck_servo_ids,
    pan_pivot,
    parse_neck,
    ticks_from,
)

REPO = Path(__file__).resolve().parents[2]
NECK = REPO / "config/neck.json"
CAMERA = REPO / "config/camera.json"


def _with(cfg: NeckConfig, **reference: object) -> NeckConfig:
    """``cfg`` with fields of its reference replaced."""
    return NeckConfig(
        cfg.pan, cfg.tilt, NeckReference(**{**cfg.reference.__dict__, **reference}), cfg.pivot
    )


def _read(cfg: NeckConfig, pan: int = 2048, tilt: int = 2300) -> NeckConfig:
    """``cfg`` with the reference ticks filled in: the reading that is still to be taken off
    the robot, so the model can be exercised before the hardware answers."""
    return _with(cfg, pan_ticks=pan, tilt_ticks=tilt)


def test_the_config_loads_the_servos_and_the_reference() -> None:
    cfg = NeckConfig.from_json(NECK)
    assert (cfg.pan.name, cfg.pan.motor_id) == ("neck", 9)
    assert (cfg.tilt.name, cfg.tilt.motor_id) == ("head", 10)
    assert cfg.motor_ids() == {"neck": 9, "head": 10} == neck_servo_ids(NECK)
    assert cfg.pan.within_limits(cfg.pan.center) and cfg.tilt.within_limits(cfg.tilt.center)
    assert not cfg.pan.within_limits(cfg.pan.max_ticks + 1)
    assert not cfg.tilt.within_limits(cfg.tilt.min_ticks - 1)
    assert cfg.reference.pan_sign in (1, -1) and cfg.reference.tilt_sign in (1, -1)
    assert JOINT_NAMES == ("neck_pan", "head_tilt")
    with pytest.raises(KeyError):
        NeckConfig.from_json(CAMERA)  # not a neck file: says what is missing, never guesses


def test_the_reference_pose_is_the_camera_mount_of_the_same_day() -> None:
    """Two files carry the camera's measured pose (the static mount and the neck's reference):
    they must agree, or the dynamic transform jumps the moment the flag is flipped. What the
    file does not yet know it says so: the ticks are null until read, and the note says with
    which command; the signs are UNVERIFIED until someone watched the picture move."""
    cfg = NeckConfig.from_json(NECK)
    camera = CameraConfig.load(CAMERA)
    ref = cfg.reference
    assert (ref.x_m, ref.y_m, ref.z_m, ref.pitch_deg) == (
        camera.x_m,
        camera.y_m,
        camera.z_m,
        camera.pitch_deg,
    )
    raw = json.loads(NECK.read_text())["reference"]
    assert "mount_measured" in raw, "the mount says when it was measured"
    if not ref.known:
        assert '{"cmd":"neck"}' in raw["ticks_note"], "say how the ticks are read, exactly"
    if not ref.signs_verified:
        assert "UNVERIFIED" in raw["signs_note"], "unverified signs say so where a reader looks"


def test_an_unread_reference_ignores_the_encoders_and_answers_the_static_mount() -> None:
    """The ticks at the measured pose are a hardware reading nobody has taken: null in the file.
    The model then has no anchor, so every reading answers the mount config/camera.json carries
    — the transform the laptop would have published, with the head free to move under it."""
    blind = _with(NeckConfig.from_json(NECK), pan_ticks=None, tilt_ticks=None)
    assert not blind.reference.known
    at_rest = NeckAngles(0.0, math.radians(blind.reference.pitch_deg))
    for pan, tilt in ((2048, 2048), (1000, 3000), (4095, 0)):
        assert joint_angles(blind, pan, tilt) == at_rest
        assert camera_pose(blind, joint_angles(blind, pan, tilt)) == pytest.approx(
            mount_transform(CameraConfig.load(CAMERA)), abs=1e-12
        )
    half = _with(blind, pan_ticks=2048)  # one of the two is no reference either
    assert not half.reference.known and joint_angles(half, 2148, 2300) == at_rest
    assert _read(blind).reference.known


def test_at_the_reference_ticks_the_dynamic_transform_equals_the_static_one() -> None:
    cfg = _read(NeckConfig.from_json(NECK))
    angles = joint_angles(cfg, cfg.reference.pan_ticks or 0, cfg.reference.tilt_ticks or 0)
    assert angles == NeckAngles(0.0, math.radians(cfg.reference.pitch_deg))
    assert camera_pose(cfg, angles) == pytest.approx(
        mount_transform(CameraConfig.load(CAMERA)), abs=1e-12
    )
    for sign in (1, -1):  # the signs decide the direction of motion, never the rest pose
        flipped = _with(cfg, pan_sign=sign, tilt_sign=sign)
        assert camera_pose(flipped, joint_angles(flipped, 2048, 2300)) == pytest.approx(
            camera_pose(cfg, angles), abs=1e-12
        )


def test_ticks_turn_into_radians_with_the_configured_signs() -> None:
    cfg = _with(NeckConfig.from_json(NECK), pan_ticks=2048, tilt_ticks=2300, pitch_deg=26.0)
    plus = joint_angles(cfg, 2048 + 100, 2300 + 100)
    assert plus.pan_rad == pytest.approx(cfg.reference.pan_sign * 100 * RAD_PER_TICK)
    assert plus.pitch_rad == pytest.approx(
        math.radians(26.0) + cfg.reference.tilt_sign * 100 * RAD_PER_TICK
    )
    assert pytest.approx(math.radians(8.789), abs=1e-4) == 100 * RAD_PER_TICK
    mirrored = _with(cfg, pan_sign=-cfg.reference.pan_sign, tilt_sign=-cfg.reference.tilt_sign)
    minus = joint_angles(mirrored, 2048 + 100, 2300 + 100)
    assert minus.pan_rad == pytest.approx(-plus.pan_rad)
    assert minus.pitch_rad - math.radians(26.0) == pytest.approx(
        -(plus.pitch_rad - math.radians(26.0))
    )
    # a full turn is 4096 ticks and a reading is never more than half a turn away
    assert ticks_from(0, 4095) == -1 and ticks_from(4095, 0) == 1
    assert ticks_from(2048, 2048 + 2047) == 2047 and ticks_from(2048, 0) == -2048


def test_a_bad_sign_in_the_file_is_refused(tmp_path: Path) -> None:
    data = json.loads(NECK.read_text())
    data["reference"]["tilt_sign"] = 0
    bad = tmp_path / "neck.json"
    bad.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        NeckConfig.from_json(bad)


def test_a_pan_turns_the_camera_about_the_vertical_and_a_tilt_dips_it() -> None:
    cfg = NeckConfig.from_json(NECK)
    z = cfg.reference.z_m
    left = camera_pose(cfg, NeckAngles(math.radians(30.0), 0.0))
    assert left[:3] == pytest.approx((cfg.reference.x_m, cfg.reference.y_m, z))  # no lever arm
    assert left[3:] == pytest.approx((0.0, 0.0, math.radians(30.0)))  # yaw only
    down = camera_pose(cfg, NeckAngles(0.0, math.radians(45.0)))
    assert down[3:] == pytest.approx((0.0, math.radians(45.0), 0.0))


def test_lever_arms_move_the_lens_and_the_reference_still_lands_on_the_mount() -> None:
    """A lens 5 cm ahead of the tilt axis: at the reference the pose is the measured mount
    (the pivot is derived from it); pitched straight down the lens sits 5 cm lower and no
    longer ahead; panned a quarter turn left it sits 5 cm to the left of the pivot."""
    base = NeckConfig.from_json(NECK)
    cfg = NeckConfig(
        base.pan,
        base.tilt,
        NeckReference(2048, 2048, 0.0, 0.0, 1.23, 0.0),
        NeckPivot(camera_from_tilt_x_m=0.05),
    )
    assert pan_pivot(cfg) == pytest.approx((-0.05, 0.0, 1.23))
    rest = camera_pose(cfg, joint_angles(cfg, 2048, 2048))
    assert rest[:3] == pytest.approx((0.0, 0.0, 1.23))
    down = camera_pose(cfg, NeckAngles(0.0, math.pi / 2))
    assert down[:3] == pytest.approx((-0.05, 0.0, 1.23 - 0.05))
    left = camera_pose(cfg, NeckAngles(math.pi / 2, 0.0))
    assert left[:3] == pytest.approx((-0.05, 0.05, 1.23))
    # a tilt axis above the pan pivot: the same rules, one level up the chain
    tall = NeckConfig(
        base.pan,
        base.tilt,
        NeckReference(2048, 2048, 0.0, 0.0, 1.23, 0.0),
        NeckPivot(tilt_from_pan_z_m=0.10, camera_from_tilt_x_m=0.05),
    )
    assert pan_pivot(tall) == pytest.approx((-0.05, 0.0, 1.13))
    assert camera_pose(tall, NeckAngles(0.0, 0.0))[:3] == pytest.approx((0.0, 0.0, 1.23))


def test_the_base_server_s_reply_is_parsed_and_other_lines_are_not() -> None:
    reading = parse_neck(
        {"type": "neck", "pan_ticks": 2048, "tilt_ticks": 2360, "age_s": 0.01, "read_ms": 1.4}
    )
    assert reading is not None and reading.ticks == (2048, 2360)
    assert reading.age_s == 0.01 and reading.read_ms == 1.4 and reading.error is None
    silent = parse_neck({"type": "neck", "error": "no reply from ids [10]"})
    assert silent is not None and silent.ticks is None and silent.error is not None
    assert math.isinf(silent.age_s)
    stale = parse_neck(
        {"type": "neck", "pan_ticks": 1, "tilt_ticks": 2, "age_s": 7.0, "error": "x"}
    )
    assert stale is not None and stale.ticks == (1, 2) and stale.error == "x"
    assert parse_neck({"type": "state", "x": 0.0}) is None
    assert parse_neck({"type": "pong"}) is None
