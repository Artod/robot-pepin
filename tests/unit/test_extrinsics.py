"""A known rotation between two synthetic fans must come back out of the estimator."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pepin.extrinsics import (
    Fan,
    bearing_grid,
    corrected_pan_reference,
    estimate_yaw_offset,
    fit_mount,
    median_fan,
    mount_yaw_shift,
    range_bias,
    sample_fan,
    slope_artefact,
    synthetic_camera_fan,
)

DEG_PER_TICK = 360.0 / 4096


PILLAR = (1.2, 0.5, 0.2)  # a post ahead and to the left: the corner that constrains a yaw


def room_range(bearing_rad: float) -> float:
    """A 3 m by 4 m box with a pillar in it, seen from its middle: the range at a bearing, in
    metres. The pillar gives the camera's cone one real corner — two flat walls constrain a yaw
    only through their own corner, which a 80 deg cone may not hold."""
    half_x, half_y = 2.0, 1.5
    c, s = math.cos(bearing_rad), math.sin(bearing_rad)
    hits = []
    if abs(c) > 1e-9:
        hits.append(half_x / abs(c))
    if abs(s) > 1e-9:
        hits.append(half_y / abs(s))
    px, py, radius = PILLAR
    along = c * px + s * py
    if along > 0.0:
        perp2 = px * px + py * py - along * along
        if perp2 < radius * radius:
            hits.append(along - math.sqrt(radius * radius - perp2))
    return min(hits)


def lidar_fan(step_deg: float = 0.5) -> Fan:
    """The box seen all round, the way the lidar sees it."""
    bearings = np.radians(np.arange(-180.0, 180.0, step_deg))
    return Fan(bearings, np.array([room_range(float(b)) for b in bearings]))


def camera_fan(turned_deg: float, half_width_deg: float = 40.0, step_deg: float = 0.5) -> Fan:
    """The same box seen through the camera's cone, but really looking ``turned_deg`` CCW of
    where its bearings claim — so the fan must be turned CCW by ``turned_deg`` to be right, and
    that is what the estimator must answer."""
    bearings = np.radians(np.arange(-half_width_deg, half_width_deg + 1e-9, step_deg))
    truth = bearings + math.radians(turned_deg)
    return Fan(bearings, np.array([room_range(float(b)) for b in truth]))


@pytest.mark.parametrize("turned", [0.0, 2.5, -2.5, 1.3, -4.0])
def test_known_rotation_is_recovered(turned: float) -> None:
    out = estimate_yaw_offset(lidar_fan(), camera_fan(turned))
    assert out.shift_deg == pytest.approx(turned, abs=0.2)
    assert out.bearings > 50
    assert out.sharp


def test_a_sharp_minimum_beats_its_neighbourhood() -> None:
    out = estimate_yaw_offset(lidar_fan(), camera_fan(2.5))
    assert out.score_m < 0.01  # the walls coincide at the truth
    assert out.depth_m > 0.01  # and part company a degree away


def test_a_fan_that_sees_only_one_flat_wall_is_not_sharp() -> None:
    """Straight at one wall the range barely changes with bearing: nothing constrains the yaw."""
    bearings = np.radians(np.arange(-5.0, 5.01, 0.5))
    flat = Fan(bearings, np.full(len(bearings), 2.0))
    wide = Fan(np.radians(np.arange(-20.0, 20.01, 0.5)), np.full(81, 2.0))
    out = estimate_yaw_offset(wide, flat)
    assert not out.sharp
    assert out.depth_m < 0.01


def test_no_overlap_is_an_error_not_a_number() -> None:
    lidar = Fan(np.radians(np.arange(90.0, 130.0, 0.5)), np.full(80, 2.0))
    camera = Fan(np.radians(np.arange(-40.0, 0.0, 0.5)), np.full(80, 2.0))
    with pytest.raises(ValueError, match="common bearing"):
        estimate_yaw_offset(lidar, camera)


def test_sample_fan_interpolates_between_returns_but_not_across_an_edge() -> None:
    fan = Fan(np.radians(np.array([0.0, 1.0, 2.0, 3.0])), np.array([2.0, 2.1, 3.0, 3.1]))
    out = sample_fan(fan, np.radians(np.array([0.5, 1.5, 5.0])))
    assert out[0] == pytest.approx(2.05)  # inside one surface
    assert math.isnan(out[1])  # across a 0.9 m step: no surface to interpolate
    assert math.isnan(out[2])  # outside the fan's span


def test_sample_fan_refuses_a_gap_wider_than_the_limit() -> None:
    fan = Fan(np.radians(np.array([0.0, 10.0])), np.array([2.0, 2.05]))
    assert math.isnan(float(sample_fan(fan, np.radians(np.array([5.0])))[0]))
    close = sample_fan(fan, np.radians(np.array([5.0])), max_gap_deg=20.0)
    assert float(close[0]) == pytest.approx(2.025)


def test_median_fan_takes_the_nearest_return_per_cell_and_the_median_over_sweeps() -> None:
    grid = bearing_grid(1.0)
    b = np.radians(np.array([0.2, 0.4]))  # both land in the same 1 deg cell
    sweeps = [(b, np.array([2.0, 1.5])), (b, np.array([2.0, 1.7])), (b, np.array([2.0, 9.0]))]
    fan = median_fan(sweeps, grid)
    cell = int(np.argmin(np.abs(fan.bearings_rad)))
    assert float(fan.ranges_m[cell]) == pytest.approx(1.7)  # nearest per sweep, then median
    assert fan.finite == 1


def test_median_fan_leaves_unlit_cells_nan_and_survives_an_empty_sweep_list() -> None:
    grid = bearing_grid(1.0)
    assert median_fan([], grid).finite == 0
    fan = median_fan([(np.radians(np.array([0.0])), np.array([np.nan]))], grid)
    assert fan.finite == 0


def test_range_bias_reads_a_pure_scale_as_a_flat_ratio() -> None:
    lidar = lidar_fan()
    cam = camera_fan(0.0)
    scaled = Fan(cam.bearings_rad, cam.ranges_m * 1.1)
    bias = range_bias(lidar, scaled)
    assert bias.ratio == pytest.approx(1.1, abs=0.01)
    assert abs(bias.slope_per_deg) < 1e-3
    assert bias.bearings > 50


def test_range_bias_reads_a_tilted_fan_as_a_sloping_ratio() -> None:
    lidar = lidar_fan()
    cam = camera_fan(0.0)
    tilt = 1.0 + 0.004 * np.degrees(cam.bearings_rad)  # one side long, the other short
    bias = range_bias(lidar, Fan(cam.bearings_rad, cam.ranges_m * tilt))
    assert bias.slope_per_deg == pytest.approx(0.004, abs=5e-4)


def test_range_bias_with_nothing_in_common_is_nan() -> None:
    lidar = Fan(np.radians(np.arange(90.0, 130.0, 0.5)), np.full(80, 2.0))
    camera = Fan(np.radians(np.arange(-40.0, 0.0, 0.5)), np.full(80, 2.0))
    bias = range_bias(lidar, camera)
    assert math.isnan(bias.ratio)
    assert bias.bearings == 0


def test_corrected_pan_reference_reproduces_the_move_of_2026_09_13() -> None:
    """2021 -> 1993: depth_fusion's align turned every frame by a signed median of -2.5 deg, so
    the camera's projected view sat 2.5 deg LEFT of the truth — the truth minus the belief is
    -2.5, and the reference had to fall by 28 ticks (journal 2026-09-13 20:30)."""
    assert corrected_pan_reference(2021, -2.5, pan_sign=-1, deg_per_tick=DEG_PER_TICK) == 1993
    assert corrected_pan_reference(1993, 0.0, pan_sign=-1, deg_per_tick=DEG_PER_TICK) == 1993
    assert corrected_pan_reference(1993, 1.0, pan_sign=-1, deg_per_tick=DEG_PER_TICK) == 2004
    # The other sign moves the reference the other way for the same correction.
    assert corrected_pan_reference(2021, -2.5, pan_sign=1, deg_per_tick=DEG_PER_TICK) == 2049


def test_corrected_pan_reference_takes_the_pan_out_of_a_panless_projection() -> None:
    """/depth_scan drops the pan, so the shift is the true heading: the error is shift - belief.
    Measured 2026-09-15: shift +6.00 deg while /neck/state believed -2.46."""
    error = 6.00 - (-2.46)
    assert corrected_pan_reference(1993, error, pan_sign=-1, deg_per_tick=DEG_PER_TICK) == 2089


def test_corrected_pan_reference_rejects_nonsense() -> None:
    with pytest.raises(ValueError, match="pan_sign"):
        corrected_pan_reference(2021, 1.0, pan_sign=0, deg_per_tick=DEG_PER_TICK)
    with pytest.raises(ValueError, match="deg_per_tick"):
        corrected_pan_reference(2021, 1.0, pan_sign=-1, deg_per_tick=0.0)


def test_a_fan_wants_matching_arrays() -> None:
    with pytest.raises(ValueError, match="same length"):
        Fan(np.zeros(3), np.zeros(4))


def test_bearing_grid_wants_more_than_one_cell() -> None:
    with pytest.raises(ValueError, match="at least two cells"):
        median_fan([], np.zeros(1))


def test_a_pure_range_scale_is_not_read_as_a_yaw_while_scale_free_is_on() -> None:
    """The flag's reason for existing: ranges 1.1x and 0.573x the truth, no rotation at all.

    With it on the scale is divided out and the answer is 0.0; with it off (the old absolute
    score) the search buys a fake -4.8 / +8.0 deg to pay for the scale
    (scratch/fan_yaw_confounds.py, 2026-09-15).
    """
    lidar, cam = lidar_fan(), camera_fan(0.0)
    for k in (1.1, 0.573):
        scaled = Fan(cam.bearings_rad, cam.ranges_m * k)
        assert estimate_yaw_offset(lidar, scaled).shift_deg == pytest.approx(0.0, abs=0.2)
        loose = estimate_yaw_offset(lidar, scaled, scale_free=False)
        assert abs(loose.shift_deg) > 4.0  # the scale is paid for in degrees
        assert not loose.sharp


def test_a_bearing_dependent_range_bias_fakes_a_sharp_yaw_that_is_not_there() -> None:
    """What `sharp` does NOT catch, and why 2026-09-15's live +5.8 deg is not the neck's.

    A camera/lidar ratio that slides by 0.002 per degree across the fan — half of the -0.004/deg
    measured live — with the camera NOT turned by one degree, moves the minimum by about 6 deg
    and the minimum stays sharp. A yaw and a bearing-dependent range bias are not separable by
    this instrument, whatever the scale flag does about the constant part.
    """
    lidar, cam = lidar_fan(), camera_fan(0.0)
    tilted = Fan(
        cam.bearings_rad, cam.ranges_m * 0.573 * (1.0 + 0.002 * np.degrees(cam.bearings_rad))
    )
    out = estimate_yaw_offset(lidar, tilted)
    assert abs(out.shift_deg) > 4.0  # degrees of yaw bought by a range slope alone
    assert out.sharp  # and it looks exactly like a good measurement
    assert abs(range_bias(lidar, tilted).slope_per_deg) > 1e-3  # the only tell there is


def test_the_proposed_reference_makes_the_neck_believe_the_measured_heading() -> None:
    """The tick chain end to end, through pepin.neck itself rather than its arithmetic.

    Head at 2021 ticks with the reference at 1993 (the live state of 2026-09-15): the model
    believes -2.46 deg; a measured true heading of +5.40 deg is an error of +7.86, and under the
    reference that closes it joint_angles must answer the measured heading at those same ticks.
    """
    from pepin.neck import NeckConfig, NeckJoint, NeckPivot, NeckReference, joint_angles

    def config(pan_reference: int) -> NeckConfig:
        return NeckConfig(
            NeckJoint("neck", 9, 2048, 257, 3812),
            NeckJoint("head", 10, 2048, 1814, 3090),
            NeckReference(pan_reference, 2311, 0.0, 0.0, 1.203, 23.8, pan_sign=-1, tilt_sign=1),
            NeckPivot(),
        )

    believed = math.degrees(joint_angles(config(1993), 2021, 2311).pan_rad)
    assert believed == pytest.approx(-2.46, abs=0.01)
    proposed = corrected_pan_reference(
        1993, 5.40 - believed, pan_sign=-1, deg_per_tick=DEG_PER_TICK
    )
    assert proposed == 2082
    after = math.degrees(joint_angles(config(proposed), 2021, 2311).pan_rad)
    assert after == pytest.approx(5.40, abs=0.05)


# ---- the mount behind the shifts: a pan, a roll, or neither -----------------------------------
MEASURED_2026_09_15 = ((14.1, 3.1), (23.8, 5.4), (33.6, 3.5))  # (head pitch, median yaw shift)


def test_a_roll_shows_as_yaw_only_when_the_head_looks_down() -> None:
    """The geometry the joint fit rests on: a pan is the same degree at every pitch, a roll is
    ``roll * sin(pitch)`` — invisible at a level head, 0.55 of itself at 33.6 deg down."""
    for pitch in (0.0, 14.1, 23.8, 33.6):
        assert mount_yaw_shift(2.0, 0.0, pitch) == pytest.approx(2.0)
    assert mount_yaw_shift(0.0, 3.0, 0.0) == pytest.approx(0.0)
    assert mount_yaw_shift(0.0, 3.0, 33.6) == pytest.approx(3.0 * math.sin(math.radians(33.6)))


def test_fit_mount_recovers_a_pan_and_a_roll_it_was_given() -> None:
    """Exact samples of a known mount come back as that mount, with nothing left over."""
    samples = [(p, mount_yaw_shift(1.7, -2.4, p)) for p in (14.1, 23.8, 33.6)]
    fit = fit_mount(samples)
    assert fit.pan_deg == pytest.approx(1.7, abs=1e-6)
    assert fit.roll_deg == pytest.approx(-2.4, abs=1e-6)
    assert fit.rms_deg < 1e-9
    assert fit.explains(0.1)


def test_one_pitch_alone_cannot_tell_a_pan_from_a_roll() -> None:
    """Two samples at the same pitch are one equation: the fit refuses instead of inventing."""
    with pytest.raises(ValueError):
        fit_mount([(23.8, 5.4), (23.8, 5.5)])
    with pytest.raises(ValueError):
        fit_mount([(23.8, 5.4)])


def test_no_rigid_mount_explains_the_three_shifts_of_2026_09_15() -> None:
    """The measurement's own verdict: +3.1 / +5.4 / +3.5 deg at 14.1 / 23.8 / 33.6 deg down.

    A pan and a roll together can only make a shift MONOTONIC in pitch (``pan + roll *
    sin pitch``), and these rise then fall. The best joint fit leaves 0.8 deg rms — eight times
    the 0.1 deg the three windows at the working pitch repeat to — so the shifts are not a
    mount's, and the pan reference must not be set from them.
    """
    fit = fit_mount(list(MEASURED_2026_09_15))
    assert fit.rms_deg > 0.7
    assert not fit.explains(0.3)  # the worst within-pitch spread of the three pitches
    assert abs(fit.residuals_deg[1]) > 1.0  # the working pitch is the one that does not fit


def test_the_estimator_returns_a_synthetic_fan_s_own_yaw() -> None:
    """synthetic_camera_fan is the truth the artefact is measured against: with no range bias
    the estimator gives back exactly the yaw the fan was built with."""
    lidar = lidar_fan()
    for turned in (-3.0, 0.0, 2.5):
        fan = synthetic_camera_fan(lidar, shift_deg=turned)
        assert estimate_yaw_offset(lidar, fan).shift_deg == pytest.approx(turned, abs=0.2)
        assert range_bias(lidar, fan, shift_deg=turned).ratio == pytest.approx(1.0, abs=0.02)


def test_the_live_range_slope_invents_more_yaw_than_the_live_shift_measured() -> None:
    """The artefact reproduced with the estimator itself, at the slopes measured on 2026-09-15.

    A camera not turned by one degree, whose ranges carry the live scale (0.58) and the live
    slope (-0.0040/deg at the working pitch, -0.0082/deg at 14.1 deg), reads back as several
    degrees of yaw — the size of the shifts the live probe reported. The artefact grows with the
    slope, and it is signed, so it cannot be subtracted off a single measurement either.
    """
    lidar = lidar_fan()
    for slope in (-0.0040, -0.0082):
        faked = slope_artefact(lidar, slope, scale=0.58)
        assert abs(faked.shift_deg) > 2.0
    assert abs(slope_artefact(lidar, 0.0, scale=0.58).shift_deg) == pytest.approx(0.0, abs=0.2)
