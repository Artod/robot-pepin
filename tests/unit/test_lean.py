"""The cart's lean: the estimator's gates, the gyro's speed, the history in between two
samples, and the poser that puts the lean under a planar odometry."""

import math
from pathlib import Path

import numpy as np
import pytest

from pepin.lean import (
    GRAVITY,
    LEAN_TAU_S,
    Lean,
    LeanEstimator,
    LeanHistory,
    imu_mount_rotation,
)

MOUNT = imu_mount_rotation(90.0, 0.0, 0.0)  # the GY-521 as it is bolted: its Y axis up


def _chip_reading(pitch_deg: float, roll_deg: float = 0.0) -> np.ndarray:
    """The MPU6050's accelerometer (Y up, roll +90 mount) when the cart leans by
    ``roll_deg``/``pitch_deg``: gravity leans onto the chip's own axes."""
    up = Lean(math.radians(roll_deg), math.radians(pitch_deg), 0.0).up_vector()
    reading: np.ndarray = MOUNT.T @ (up * GRAVITY)
    return reading


def _chip_rates(roll_rate_deg: float, pitch_rate_deg: float) -> np.ndarray:
    """The gyro reading (the chip's own axes) of a body turning at these rates about
    base_link's x and y."""
    rates: np.ndarray = MOUNT.T @ np.radians(np.array([roll_rate_deg, pitch_rate_deg, 0.0]))
    return rates


def _level(estimator: LeanEstimator, t: float, seconds: float = 1.0, dt: float = 0.02) -> float:
    """Feed level readings for ``seconds`` and return the time reached."""
    while seconds > 0:
        estimator.observe(_chip_reading(0.0), t, _chip_rates(0.0, 0.0))
        t += dt
        seconds -= dt
    return t


# ---- the record ------------------------------------------------------------------------------
def test_a_lean_and_its_up_vector_are_the_same_thing_read_two_ways() -> None:
    """``up_vector`` and ``from_up`` are exact inverses, and the rotation the poser composes
    turns a point on the leaning body into the level frame by the same angles."""
    for roll, pitch in ((0.0, 0.0), (0.05, -0.09), (0.3, 0.2)):
        lean = Lean(roll, pitch, 7.0)
        back = Lean.from_up(lean.up_vector(), 7.0)
        assert (back.roll, back.pitch) == pytest.approx((roll, pitch), abs=1e-12)
        assert float(np.linalg.norm(lean.up_vector())) == pytest.approx(1.0)
    nose_down = Lean(0.0, math.radians(5.0), 0.0)
    ahead = nose_down.rotation() @ np.array([3.0, 0.0, 0.0])
    assert ahead[2] == pytest.approx(-3.0 * math.sin(math.radians(5.0)))  # the point dips
    assert nose_down.size_deg == pytest.approx(5.0)


# ---- the estimator ---------------------------------------------------------------------------
def test_a_static_three_degree_lean_is_recovered_within_a_fifth_of_a_degree() -> None:
    """The cart parked on a slope: the first reading seeds the up vector outright, and after a
    time constant of slope the lean is the slope to well under a fifth of a degree."""
    estimator = LeanEstimator(MOUNT)
    t = 0.0
    while t < LEAN_TAU_S + 1.0:
        estimator.observe(_chip_reading(3.0), t, _chip_rates(0.0, 0.0))
        t += 0.02
    roll, pitch = estimator.roll_pitch_deg
    assert (roll, pitch) == pytest.approx((0.0, 3.0), abs=0.2)
    assert estimator.quality == pytest.approx(1.0, abs=1e-6)


def test_a_bump_is_followed_at_the_gyro_s_speed_and_not_at_the_accelerometer_s() -> None:
    """A wheel climbing a threshold: 5 degrees nose down in 0.2 s. The accelerometer alone
    calls that a push and holds level for the whole bump (the 1 degree gate: what the floor
    anchor has always done); with the gyro the lean is followed sample by sample."""
    profile = [(0.5 * (i + 1), 25.0) for i in range(10)]  # 5 deg in 0.2 s at 25 deg/s
    for use_gyro, expected in ((False, 0.0), (True, 5.0)):
        estimator = LeanEstimator(MOUNT, use_gyro=use_gyro)
        t = _level(estimator, 0.0)
        for pitch_deg, rate in profile:
            estimator.observe(_chip_reading(pitch_deg), t, _chip_rates(0.0, rate))
            t += 0.02
        assert estimator.roll_pitch_deg[1] == pytest.approx(expected, abs=0.05)
    # and the gyro's lean is a reading a consumer can ask for at the bump's own stamp
    assert estimator.lean_at(t - 0.01) is not None


def test_braking_at_two_metres_per_second_squared_is_not_a_lean() -> None:
    """A hard stop leans the apparent gravity 11.5 degrees while the body stands level. The
    norm gate throws that reading away, the gyro says the body did not turn, and the estimator
    stays level — with a quality that says the lean is no longer gravity's word."""
    estimator = LeanEstimator(MOUNT)
    t = _level(estimator, 0.0, seconds=2.0)
    braking = MOUNT.T @ np.array([2.0, 0.0, 0.0]) + _chip_reading(0.0)  # 2 m/s^2 backwards
    assert abs(float(np.linalg.norm(braking)) - GRAVITY) > 0.1  # the norm gate can see it
    for _ in range(50):  # a full second of braking
        estimator.observe(braking, t, _chip_rates(0.0, 0.0))
        t += 0.02
    assert estimator.roll_pitch_deg == pytest.approx((0.0, 0.0), abs=0.05)
    assert estimator.quality < 0.5  # measured no more: the report line says so


def test_a_gentle_push_is_gated_but_a_lean_that_outlasts_the_time_constant_is_the_floor() -> None:
    """Level at the first sample; a bump (not 1 g) is ignored; a push (the cart accelerating
    leans the apparent gravity 3 degrees at the same 1 g) is ignored for as long as any push
    lasts; the same lean outlasting the time constant is a slope and the up vector follows."""
    estimator = LeanEstimator(MOUNT, use_gyro=False)
    estimator.observe(np.array([0.0, GRAVITY, 0.0]), 0.0)  # the chip's Y up: level
    assert np.allclose(estimator.up, [0.0, 0.0, 1.0])
    assert estimator.roll_pitch_deg == pytest.approx((0.0, 0.0), abs=1e-9)
    estimator.observe(np.array([0.0, GRAVITY, 6.0]), 0.5)  # a bump: not 1 g, ignored
    assert np.allclose(estimator.up, [0.0, 0.0, 1.0])
    t = 0.5
    while t < 3.0:  # a push: 3 degrees for 2.5 s
        t += 0.1
        estimator.observe(_chip_reading(3.0), t)
    assert estimator.roll_pitch_deg[1] == pytest.approx(0.0, abs=1e-9)
    while t < 3.0 + LEAN_TAU_S + 0.5:  # the lean persists past the time constant: a slope
        t += 0.1
        estimator.observe(_chip_reading(3.0), t)
    assert estimator.roll_pitch_deg[1] == pytest.approx(3.0, abs=1e-6)


def test_a_small_lean_is_followed_slowly_and_a_nan_sample_is_ignored() -> None:
    """Half a degree is inside the gate and is low-passed with the 10 s time constant: after
    30 s the filter has come 95 % of the way. A NaN reading passes no gate and leaves the up
    vector finite and unchanged."""
    estimator = LeanEstimator(MOUNT, use_gyro=False)
    estimator.observe(np.array([0.0, GRAVITY, 0.0]), 0.0)
    for i in range(1, 301):
        estimator.observe(_chip_reading(0.5), 0.1 * i)
    assert estimator.roll_pitch_deg[1] == pytest.approx(
        0.5 * (1 - math.exp(-30.0 / LEAN_TAU_S)), abs=0.01
    )
    before = estimator.up.copy()
    estimator.observe(np.array([np.nan, GRAVITY, 0.0]), 31.0)
    assert np.array_equal(estimator.up, before) and np.isfinite(estimator.up).all()


def test_a_dead_accelerometer_leaves_the_lean_alone_instead_of_dividing_by_its_zero() -> None:
    """A bridge that publishes zeros (the chip unplugged, a read that failed) must not turn the
    up vector into NaN — every pixel of the floor would then stop being the floor."""
    estimator = LeanEstimator(MOUNT)
    estimator.observe(np.array([0.0, GRAVITY, 0.0]), 0.0)
    for i in range(10):
        estimator.observe(np.zeros(3), 0.1 * (i + 1), np.zeros(3))
    assert np.allclose(estimator.up, [0.0, 0.0, 1.0])
    assert np.isfinite(estimator.up).all()


def test_the_imu_mount_of_the_config_is_the_rotation_the_cpp_bridge_applies() -> None:
    """config/imu.json says roll +90 deg; base_bridge.cpp's to_base_axes for a Y-up chip maps
    base (x, y, z) <- chip (x, -z, y). The two must agree, or a reading published in base_link
    would be rotated twice."""
    import json

    mount = json.loads((Path(__file__).parents[2] / "config/imu.json").read_text())["mount"]
    rot = imu_mount_rotation(mount["roll_deg"], mount["pitch_deg"], mount["yaw_deg"])
    chip = np.array([1.0, 2.0, 3.0])
    assert rot @ chip == pytest.approx([1.0, -3.0, 2.0])


def test_a_sideways_imu_mount_reads_as_a_roll_of_ninety_degrees_and_no_floor() -> None:
    """The chip's Y up through an identity mount (as if config/imu.json said the chip were
    not turned): the up vector lands on base_link's y, a 90 degree roll, and a floor
    perpendicular to that passes through the wheels edge-on — no ray meets it, every pixel's
    floor depth is NaN. The mount's roll +90 is what turns this into a level floor."""
    from pepin.depth import CameraPose, Intrinsics, floor_depth

    estimator = LeanEstimator(np.eye(3))
    estimator.observe(np.array([0.0, GRAVITY, 0.0]), 0.0)
    assert estimator.up == pytest.approx([0.0, 1.0, 0.0])
    assert estimator.roll_pitch_deg == pytest.approx((90.0, 0.0))
    intr = Intrinsics(fx=457.0, fy=457.0, cx=320.0, cy=180.0, width=640, height=360)
    cam = CameraPose(0.0, 0.0, 1.23, math.radians(26.0))
    assert np.isnan(floor_depth(intr, cam, up=estimator.up)).all()


def test_the_yaw_rate_of_a_pivot_does_not_move_the_up_vector() -> None:
    """The big rate this robot ever sees is the yaw of a pivot (0.6 rad/s). It turns about the
    up vector itself, so it must not lean anything: a whole second of pivoting on level ground
    leaves the lean where it was."""
    estimator = LeanEstimator(MOUNT)
    t = _level(estimator, 0.0)
    spin = MOUNT.T @ np.array([0.0, 0.0, 0.6])
    for _ in range(50):
        estimator.observe(_chip_reading(0.0), t, spin)
        t += 0.02
    assert estimator.roll_pitch_deg == pytest.approx((0.0, 0.0), abs=1e-6)


# ---- the history -----------------------------------------------------------------------------
def test_the_history_interpolates_between_two_samples_and_refuses_what_it_never_saw() -> None:
    """A lean is asked for at a frame's stamp, which falls between two IMU samples: roll,
    pitch and quality come out linearly interpolated. Before the first sample there is no
    answer; past the last one the newest lean is held for the slack and then refused."""
    history = LeanHistory(horizon_s=1.0, slack_s=0.1)
    history.add(Lean(0.0, 0.0, 10.0, 1.0))
    history.add(Lean(0.02, 0.04, 10.02, 0.5))
    history.add(Lean(0.01, 0.02, 10.01, 1.0))  # older than the newest: ignored
    assert len(history) == 2
    middle = history.at(10.01)
    assert middle is not None
    assert (middle.roll, middle.pitch, middle.quality) == pytest.approx((0.01, 0.02, 0.75))
    assert middle.stamp == 10.01
    assert history.at(10.02) == history.newest
    assert history.at(10.11) is not None and history.at(10.2) is None  # the slack, then nothing
    assert history.at(9.99) is None
    history.add(Lean(0.0, 0.0, 11.5, 1.0))  # past the horizon: the old samples are dropped
    assert len(history) == 1 and history.at(10.01) is None


def test_the_history_is_bounded_by_its_length_as_well_as_by_its_horizon() -> None:
    """A clock that jumps (or an IMU that floods) must not make the history grow without end."""
    history = LeanHistory(horizon_s=1e9, max_len=50)
    for i in range(500):
        history.add(Lean(0.0, 0.0, float(i), 1.0))
    assert len(history) == 50
    newest = history.newest
    assert newest is not None and newest.stamp == 499.0
