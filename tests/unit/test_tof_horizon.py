"""A ToF may only be believed as far as its cone stays off the floor."""

import math

from pepin.tof_horizon import floor_horizon, trusted_max_range

FOV = 0.47  # the VL53L1X cone, radians


def test_the_low_sensors_are_believed_to_about_half_a_metre() -> None:
    """0.16 m up with a 27 deg cone reaches the floor at 0.67 m; belief stops short of that."""
    assert 0.55 < floor_horizon(0.16, FOV) < 0.60
    assert 0.55 < floor_horizon(0.165, FOV) < 0.62


def test_a_higher_sensor_may_be_believed_further() -> None:
    assert floor_horizon(0.27, FOV) > floor_horizon(0.16, FOV)
    assert 0.9 < floor_horizon(0.27, FOV) < 1.0


def test_the_ceiling_never_rises_above_the_sensor_s_own_range() -> None:
    assert trusted_max_range(0.27, FOV, 1.3) == floor_horizon(0.27, FOV)
    assert trusted_max_range(2.0, FOV, 1.3) == 1.3


def test_an_unmeasured_mount_is_not_clamped() -> None:
    """A sensor whose height nobody wrote down keeps its own ceiling instead of a made-up one."""
    assert math.isinf(floor_horizon(0.0, FOV))
    assert trusted_max_range(0.0, FOV, 1.3) == 1.3
