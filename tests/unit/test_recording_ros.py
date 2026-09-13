"""A ROS ``LaserScan`` turned into a session record: angles, masking, round trip; and the
``imu`` record, whose accelerometer is what lets a replay know which way was up."""

import json
import math
from typing import Any

import pytest

from pepin.lean import LEAN_TAU_S
from pepin.recording import (
    imu_record,
    imu_sample,
    lean_history,
    scan_from_record,
    scan_record_from_ros,
)

RANGE_MIN, RANGE_MAX = 0.05, 12.0
HULL = (-0.30, 0.0625, 0.275)  # x_back, x_front, half_width of the cart, around the sensor


def record(ranges: list[float], **overrides: Any) -> dict[str, Any]:
    """A scan record from ``ranges``, with a full revolution and a clean sensor by default."""
    fields: dict[str, Any] = {
        "stamp": 1234.5,
        "angle_min": 0.0,
        "angle_increment": math.tau / len(ranges),
        "ranges": ranges,
        "intensities": [100.0] * len(ranges),
        "range_min": RANGE_MIN,
        "range_max": RANGE_MAX,
        "scan_time": 0.1,
        "mount_yaw_rad": 0.0,
        "mount_x_m": 0.0,
    }
    return scan_record_from_ros(**{**fields, **overrides})


def test_angles_run_clockwise_from_the_mount_yaw() -> None:
    """The sensor hangs upside down and turned: a ROS angle a becomes -a - yaw."""
    out = record([2.0] * 4, angle_min=-1.0, angle_increment=1.0, mount_yaw_rad=0.5)
    assert out["angles"] == pytest.approx([0.5, 5.7832, 4.7832, 3.7832])


def test_returns_outside_the_sensor_limits_are_dropped() -> None:
    ranges = [RANGE_MIN, 0.04, 1.0, RANGE_MAX, 13.0, math.nan, math.inf]
    assert record(ranges)["ranges"] == [None, None, 1.0, None, None, None, None]


def test_a_return_inside_the_carts_own_hull_is_dropped() -> None:
    """Forward, right, back, left at 0.05 m of mount offset: only the beams that clear
    the cart survive."""
    beams = {"angle_increment": math.pi / 2, "mount_x_m": 0.05, "hull": HULL}
    assert record([0.30, 0.20, 0.30, 0.30], **beams)["ranges"] == [0.3, None, None, 0.3]
    assert record([0.30, 0.30, 0.40, 0.30], **beams)["ranges"] == [0.3, 0.3, 0.4, 0.3]


def test_an_intensity_that_cannot_be_read_becomes_zero() -> None:
    out = record([1.0, 1.0, 1.0], intensities=[math.nan, 200.0])
    assert out["intensities"] == [0, 200, 0]


def test_the_turn_rate_comes_from_the_scan_time() -> None:
    assert record([1.0], scan_time=0.1)["speed_rps"] == 10.0
    assert record([1.0], scan_time=0.25)["speed_rps"] == 4.0
    assert record([1.0], scan_time=0.0)["speed_rps"] == 10.0  # no timing: assume the LD19's rate


def test_the_record_round_trips_into_a_laser_scan() -> None:
    out = record([1.0, math.nan, 2.5, 3.0])
    scan = scan_from_record(out)
    assert out["topic"] == "scan"
    assert scan.stamp == 1234.5 and scan.speed_rps == 10.0
    assert len(scan.angles) == len(scan.ranges) == 4
    assert scan.ranges[0] == 1.0 and math.isnan(scan.ranges[1])
    assert scan.angles.tolist() == pytest.approx(out["angles"])
    assert scan.intensities.tolist() == [100, 100, 100, 100]


def test_an_imu_record_keeps_its_old_key_and_carries_the_lean_for_a_replay() -> None:
    """``wz`` stays exactly where and as it was — every reader of an old tape reads a new one —
    and the accelerometer and the other two rates ride along, so an offline replay can run the
    same estimator the nodes run."""
    record = imu_record(1789070956.371132, (0.001, -0.002, 0.4567891), (0.1, -0.2, 9.8123456))
    assert record["topic"] == "imu" and record["t"] == 1789070956.371132
    assert record["wz"] == 0.4568 and record["wx"] == 0.001 and record["wy"] == -0.002
    assert (record["ax"], record["ay"], record["az"]) == (0.1, -0.2, 9.812)
    assert json.loads(json.dumps(record)) == record  # it survives the tape as it is written
    sample = imu_sample(record)
    assert sample is not None
    accel, gyro = sample
    assert accel == pytest.approx([0.1, -0.2, 9.812]) and gyro == pytest.approx(
        [0.001, -0.002, 0.4568]
    )
    assert imu_sample({"t": 0.0, "topic": "imu", "wz": 0.1}) is None  # a tape taped before this


def test_a_tape_s_leans_are_replayed_through_the_estimator_the_nodes_run() -> None:
    """Six seconds of level standing and then a 4 degree nose-down slope held: the history
    answers for any moment in between, and a tape with only ``wz`` in it answers for none."""
    g, records = 9.81, []
    t = 0.0
    while t < 6.0:
        records.append(imu_record(t, (0.0, 0.0, 0.0), (0.0, 0.0, g)))
        t += 0.02
    pitch = math.radians(4.0)
    tipped = (-g * math.sin(pitch), 0.0, g * math.cos(pitch))
    while t < 6.0 + 3 * LEAN_TAU_S:
        records.append(imu_record(t, (0.0, 0.0, 0.0), tipped))
        t += 0.02
    history = lean_history(records)
    assert len(history) == len(records)
    early = history.at(3.0)
    late = history.at(t - 0.1)
    assert early is not None and late is not None
    assert early.pitch_deg == pytest.approx(0.0, abs=1e-6)
    assert late.pitch_deg == pytest.approx(4.0, abs=0.2)
    assert history.at(-1.0) is None
    old = [{"t": 0.0, "topic": "imu", "wz": 0.1}, {"t": 0.02, "topic": "scan"}]
    assert len(lean_history(old)) == 0
