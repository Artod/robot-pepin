"""A ToF may only be believed as far as its cone stays off the floor."""

import math

from pepin.tof_horizon import RangeHold, floor_horizon, trusted_max_range

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


def test_a_dropped_frame_does_not_erase_a_fresh_mark() -> None:
    """Half the frames with a hand in front were invalid; the mark must survive the gaps."""
    hold = RangeHold(hold_s=1.2)
    assert hold.publish("front", 0.30, 0.96, now=0.0) == 0.30
    assert hold.publish("front", None, 0.96, now=0.3) == 0.30, "held, not cleared"
    assert hold.publish("front", None, 0.96, now=1.0) == 0.30
    assert hold.publish("front", None, 0.96, now=1.5) == 0.96, "old evidence expires: clear"


def test_a_return_past_the_ceiling_is_the_floor_and_counts_as_nothing() -> None:
    hold = RangeHold(hold_s=1.0)
    assert hold.publish("left", 0.65, 0.57, now=0.0) == 0.57
    assert hold.publish("left", 0.40, 0.57, now=0.5) == 0.40


def test_sensors_are_held_independently() -> None:
    hold = RangeHold(hold_s=1.0)
    hold.publish("left", 0.3, 0.57, now=0.0)
    assert hold.publish("right", None, 0.59, now=0.1) == 0.59
    assert hold.publish("left", None, 0.57, now=0.5) == 0.3
