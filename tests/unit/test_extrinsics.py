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
    median_fan,
    range_bias,
    sample_fan,
)

DEG_PER_TICK = 360.0 / 4096


def room_range(bearing_rad: float) -> float:
    """A 3 m by 4 m box seen from its middle: the range to the wall at a bearing, in metres."""
    half_x, half_y = 2.0, 1.5
    c, s = math.cos(bearing_rad), math.sin(bearing_rad)
    hits = []
    if abs(c) > 1e-9:
        hits.append(half_x / abs(c))
    if abs(s) > 1e-9:
        hits.append(half_y / abs(s))
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
    """2021 -> 1993: the camera looked 2.5 deg left, so the fan had to turn 2.5 deg clockwise."""
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
