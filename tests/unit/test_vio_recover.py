"""OpenVINS's recovery policy without ROS: the velocity seed, the head's stillness, the rules."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pepin.vio_recover import (
    HeadStill,
    Health,
    Recoveries,
    VioWatch,
    imu_velocity,
    twist_block,
)
from pepin.visual_odometry import VioHealth


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def base_in_imu(r_b_i: np.ndarray, r_bi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """TF head_imu <- base_link (R_I_B, p_I_B) from the IMU's mount in base_link."""
    r_i_b = r_b_i.T
    return r_i_b, -(r_i_b @ r_bi)


# ---- the velocity seed -------------------------------------------------------------------------


def test_an_imu_on_the_base_origin_moves_with_the_base() -> None:
    v, cov = imu_velocity((0.2, -0.05, 0.3), np.diag([1e-4, 4e-4, 1e-3]), np.eye(3), np.zeros(3))
    assert v == pytest.approx([0.2, -0.05, 0.0])
    assert np.diag(cov) == pytest.approx([1e-4, 4e-4, 0.02**2])


def test_a_turn_moves_an_imu_off_the_axis_sideways() -> None:
    # the IMU 0.1 m ahead of base_link and 0.8 m up, axes aligned; the cart spins at 1 rad/s
    r_i_b, p_i_b = base_in_imu(np.eye(3), np.array([0.1, 0.0, 0.8]))
    v, _ = imu_velocity((0.0, 0.0, 1.0), np.zeros((3, 3)), r_i_b, p_i_b)
    assert v == pytest.approx([0.0, 0.1, 0.0])  # w x r: forward lever arm -> left velocity


def test_the_velocity_is_in_the_imus_own_axes() -> None:
    # the IMU's x points left of the cart (yawed 90 deg): forward motion is the IMU's -y
    r_i_b, p_i_b = base_in_imu(rot_z(math.pi / 2), np.zeros(3))
    v, _ = imu_velocity((0.2, 0.0, 0.0), np.zeros((3, 3)), r_i_b, p_i_b)
    assert v == pytest.approx([0.0, -0.2, 0.0], abs=1e-12)


def test_the_seed_is_the_imus_true_velocity_on_a_pitched_panned_head() -> None:
    """Frame conventions end to end: the IMU's world velocity by finite differences of
    T_W_B(t) T_B_I, turned into the IMU's axes, against the carry."""
    r_b_i = rot_z(0.6) @ rot_y(math.radians(23.8))  # panned 34 deg, the mount's 23.8 deg pitch
    r_bi = np.array([0.04, -0.02, 0.81])
    r_i_b, p_i_b = base_in_imu(r_b_i, r_bi)
    vx, vy, wz = 0.25, 0.03, -0.4
    dt = 1e-6

    def imu_world(t: float) -> tuple[np.ndarray, np.ndarray]:
        yaw = 0.3 + wz * t
        r_w_b = rot_z(yaw)
        p_w_b = np.array([1.0, 2.0, 0.0]) + rot_z(0.3) @ np.array([vx, vy, 0.0]) * t
        return r_w_b @ r_b_i, p_w_b + r_w_b @ r_bi

    r0, p0 = imu_world(0.0)
    _, p1 = imu_world(dt)
    truth = r0.T @ (p1 - p0) / dt
    v, _ = imu_velocity((vx, vy, wz), np.zeros((3, 3)), r_i_b, p_i_b)
    assert v == pytest.approx(truth, abs=1e-5)


def test_the_seed_covariance_is_the_twists_through_the_same_map() -> None:
    rng = np.random.default_rng(1)
    r_b_i = rot_z(-0.4) @ rot_y(0.42)
    r_i_b, p_i_b = base_in_imu(r_b_i, np.array([0.05, 0.02, 0.8]))
    twist = (0.2, 0.0, 0.3)
    cov = np.array([[4e-4, 1e-5, 2e-5], [1e-5, 1e-4, 0.0], [2e-5, 0.0, 9e-4]])
    _, analytic = imu_velocity(twist, cov, r_i_b, p_i_b, vz_sigma_m_s=0.0)
    with np.errstate(all="ignore"):  # Accelerate's spurious matmul warnings on macOS
        draws = np.asarray(twist) + rng.standard_normal((40000, 3)) @ np.linalg.cholesky(cov).T
    seeds = np.array([imu_velocity(tuple(d), cov, r_i_b, p_i_b)[0] for d in draws])
    sampled = np.cov(seeds.T)
    # 40000 draws: a variance's sampling error is ~0.7 %, its cross terms' a little more
    assert np.allclose(sampled, analytic, rtol=0.04, atol=1e-6), (sampled, analytic)
    assert np.allclose(analytic, analytic.T)
    assert np.all(np.linalg.eigvalsh(analytic) > -1e-12)


def test_the_twist_block_is_vx_vy_and_the_yaw_rate() -> None:
    full = np.zeros((6, 6))
    full[0, 0], full[1, 1], full[5, 5], full[0, 5], full[5, 0], full[2, 2] = 1, 2, 3, 0.5, 0.5, 99
    assert twist_block(full.flatten()) == pytest.approx(
        np.array([[1, 0, 0.5], [0, 2, 0], [0.5, 0, 3]])
    )


# ---- the head's stillness ----------------------------------------------------------------------


def test_the_head_is_still_only_when_it_has_not_turned_and_the_tf_is_fresh() -> None:
    head = HeadStill(still_s=0.3, still_deg=1.0)
    assert not head.still(0.0)  # nothing heard
    for i in range(30):  # encoder jitter of 0.1 deg
        head.reading(i * 0.02, rot_z(0.5 + math.radians(0.1 if i % 2 else 0.0)) @ rot_y(0.4))
    assert head.still(0.58)
    head.reading(0.60, rot_z(0.5 + math.radians(2.0)) @ rot_y(0.4))  # a saccade starts
    assert not head.still(0.60)
    for i in range(1, 20):
        head.reading(0.60 + i * 0.02, rot_z(0.5 + math.radians(2.0)) @ rot_y(0.4))
    assert head.still(0.98)  # settled 0.3 s later
    assert not head.still(1.6)  # the transform went quiet
    fresh = HeadStill(still_s=0.3)
    fresh.reading(0.0, np.eye(3))
    assert not fresh.still(0.1), "watched for less than the window"


# ---- the failure rules -------------------------------------------------------------------------


def health(
    t: float,
    *,
    tracks: int = 80,
    zupt: bool = False,
    gyro: float = 0.05,
    speed: float = 0.2,
    v_i: tuple[float, float, float] = (0.0, -0.2, 0.0),
    epoch: int = 0,
    initialized: bool = True,
) -> Health:
    return Health(
        t=t,
        epoch=epoch,
        initialized=initialized,
        tracked=100,
        persistent=tracks,
        msckf=3,
        slam=0,
        zupt=zupt,
        gyro=gyro,
        speed=speed if initialized else None,
        v_i=v_i if initialized else None,
        resets=epoch,
        warm_seeds=0,
        standard_inits=0,
        seed="",
        frame="global" if epoch == 0 else f"global_{epoch}",
    )


def run(
    watch: VioWatch, frames: list[Health], reference: tuple[float, float, float] | None = None
) -> list[tuple[float, str]]:
    """Each frame judged with ``reference`` (the EKF's velocity at the IMU) at its own time."""
    verdicts = []
    for h in frames:
        reason = watch.observe(h, None if reference is None else (h.t, np.array(reference)))
        if reason is not None:
            verdicts.append((h.t, reason))
    return verdicts


def test_a_dark_stretch_is_lost_after_dark_s_of_still_frames_once_the_grace_is_over() -> None:
    watch = VioWatch(dark_tracks=10, dark_s=1.0, grace_s=2.0)
    light = [health(i / 10) for i in range(30)]  # 3 s of a good picture
    dark = [health(3.0 + i / 10, tracks=0) for i in range(40)]
    verdicts = run(watch, light + dark)
    assert len(verdicts) == 1, "one verdict per episode"
    t, reason = verdicts[0]
    assert t == pytest.approx(3.9) and reason.startswith("dark: 0 persistent tracks")
    assert watch.counts == {"disagree": 0, "dark": 1, "runaway": 0}


def test_a_head_swing_in_the_light_is_not_lost() -> None:
    watch = VioWatch(dark_tracks=10, dark_s=1.0, grace_s=0.0)
    frames = [health(i / 10) for i in range(10)]
    # a saccade: 0.6 s of blurred frames with no persistent track, then the picture again
    frames += [health(1.0 + i / 10, tracks=0, gyro=4.0) for i in range(6)]
    frames += [health(1.6 + i / 10, tracks=2 + 10 * i) for i in range(4)]  # the tracks rebuild
    frames += [health(2.0 + i / 10) for i in range(20)]
    assert run(watch, frames) == []


def test_a_dark_stretch_with_swings_through_it_still_adds_up_between_them() -> None:
    watch = VioWatch(dark_tracks=10, dark_s=1.0, grace_s=0.0)
    frames = [health(0.0)]
    t = 0.0
    for k in range(30):  # alternating 0.1 s still, 0.1 s swinging, all dark
        t += 0.1
        frames.append(health(t, tracks=0, gyro=0.1 if k % 2 == 0 else 5.0))
    verdicts = run(watch, frames)
    assert len(verdicts) == 1
    assert verdicts[0][0] == pytest.approx(1.9, abs=0.05)  # 10 still frames of 0.1 s


def test_the_zupt_holds_a_cart_at_rest_in_the_dark_unless_dark_at_rest() -> None:
    frames = [health(i / 10, tracks=0, zupt=True, speed=0.0) for i in range(40)]
    assert run(VioWatch(grace_s=0.0), frames) == []
    verdicts = run(VioWatch(grace_s=0.0, dark_at_rest=True), frames)
    assert len(verdicts) == 1 and verdicts[0][0] == pytest.approx(1.0)


def test_a_velocity_that_leaves_the_ekfs_is_lost_after_disagree_s_and_not_without_one() -> None:
    watch = VioWatch(disagree_m_s=0.2, disagree_s=1.0, grace_s=0.0)
    ekf = (0.0, -0.25, 0.0)
    frames = [health(i / 10, v_i=(0.02, -0.3, 0.01)) for i in range(10)]  # 0.05 off: agrees
    # the 0330 replay's epoch 9: 0.32 -> 1.17 m/s while the wheels held 0.29
    frames += [health(1.0 + i / 10, v_i=(0.0, -0.5 - 0.05 * i, 0.0)) for i in range(15)]
    verdicts = run(watch, frames, ekf)
    assert len(verdicts) == 1 and verdicts[0][0] == pytest.approx(1.9)  # its 10th frame
    assert verdicts[0][1].startswith("disagree: 0.70 m/s from the EKF's velocity at the IMU")
    # no reference (the head moving: no seed), or a stale one: nothing is judged
    assert run(VioWatch(grace_s=0.0), frames) == []
    stale = VioWatch(grace_s=0.0)
    assert [stale.observe(h, (h.t - 0.5, np.array(ekf))) for h in frames] == [None] * len(frames)


def test_a_runaway_speed_is_lost_after_speed_s() -> None:
    watch = VioWatch(max_speed_m_s=1.0, speed_s=0.3, grace_s=0.0)
    frames = [health(i / 10, speed=0.25) for i in range(5)]
    frames += [health(0.5 + i / 10, speed=3.0 + i) for i in range(5)]
    verdicts = run(watch, frames)
    assert len(verdicts) == 1 and verdicts[0][0] == pytest.approx(0.7)  # the third fast frame
    assert verdicts[0][1].startswith("runaway: 5.00 m/s")


def test_nothing_is_judged_while_initialising_nor_in_the_grace_after_a_reset() -> None:
    watch = VioWatch(dark_s=1.0, grace_s=2.0)
    frames = [health(i / 10, tracks=0, initialized=False, epoch=1) for i in range(30)]
    assert run(watch, frames) == []
    # the new filter publishes at t 3.0: dark again, judged from t 5.0 on, lost on its 10th frame
    frames = [health(3.0 + i / 10, tracks=0, epoch=1) for i in range(40)]
    verdicts = run(watch, frames)
    assert len(verdicts) == 1 and verdicts[0][0] == pytest.approx(5.9)  # frames 5.0 .. 5.9
    # a further reset (epoch 2) starts the grace again
    assert run(watch, [health(7.0 + i / 10, tracks=0, epoch=2) for i in range(25)]) == []


def test_health_parses_openvins_key_values() -> None:
    values = {
        "t": "1791255800.123456",
        "epoch": "2",
        "initialized": "1",
        "tracked": "88",
        "persistent": "61",
        "msckf": "23",
        "slam": "17",
        "zupt": "0",
        "gyro": "0.0412",
        "speed": "0.2100",
        "v_I": "0.0100,-0.2000,0.0030",
        "pos_sigma": "0.0500",
        "bias_age": "3.0",
        "resets": "2",
        "warm_seeds": "2",
        "standard_inits": "0",
        "seed": "warm 0.10 s after the reset's first frame",
    }
    h = Health.parse(values, "global_2")
    assert h is not None
    assert (h.epoch, h.initialized, h.used, h.persistent, h.zupt, h.frame) == (
        2,
        True,
        40,
        61,
        False,
        "global_2",
    )
    assert h.speed == pytest.approx(0.21) and h.v_i == (0.01, -0.2, 0.003)
    del values["speed"], values["v_I"]
    values["initialized"] = "0"
    initialising = Health.parse(values)
    assert initialising is not None and initialising.speed is None
    del values["msckf"]
    assert Health.parse(values) is None


def test_a_recovery_is_timed_from_the_verdict_to_the_new_filters_first_published_frame() -> None:
    recoveries = Recoveries()
    recoveries.asked(100.0, "dark", epoch=0)
    assert recoveries.health(100.2, health(1.0, epoch=0)) is None  # the old filter, still
    assert recoveries.health(100.4, health(1.1, epoch=1, initialized=False)) is None
    done = recoveries.health(100.9, health(1.6, epoch=1))
    assert done is not None and done.seconds == pytest.approx(0.9)
    assert recoveries.open is None
    assert recoveries.report().startswith("1 recovered in p50 0.9 s, max 0.9 s")


# ---- the relay across a reset -----------------------------------------------------------------


def test_the_relay_sees_a_reset_by_the_new_frame_name_even_without_a_covariance_step() -> None:
    health_ = VioHealth()
    cov = [0.0] * 36
    for i in range(6):
        cov[i * 6 + i] = 0.01
    assert not health_.observe(cov, "global")
    assert not health_.observe(cov, "global")
    assert health_.observe(cov, "global_1")  # same covariance, a new gravity frame
    assert not health_.observe(cov, "global_1")
    big = [c * 100 for c in cov]
    assert health_.observe(big, "global_1")  # the trace step still counts
    assert health_.reinits == 2
    assert not VioHealth().observe(cov)  # an OpenVINS without names: the step rule alone
