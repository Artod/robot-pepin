"""When the board's IMU and ToF measurements happened: the numbers in config/imu.json and
config/tof.json, the C++ bridge and the launch that carry the IMU's, and the line-age arithmetic
the ToF bridge dates its lines with (pepin.sensor_timing, 2026-10-02)."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
import source_facts as sf

from pepin.sensor_timing import (
    LINE_MAX_AGE_S,
    MPU6050_GYRO_DELAY_S,
    ImuTiming,
    imu_timing,
    measurement_lag_s,
    tof_timing_offset_s,
)
from pepin.tof_server import INTER_MEASUREMENT_MS, TIMING_BUDGET_MS

REPO = Path(__file__).resolve().parents[2]
DRIVER = REPO / "ros/pepin_base_cpp/include/pepin_base_cpp/mpu6050.hpp"
BRIDGE = REPO / "ros/pepin_base_cpp/src/base_bridge.cpp"


def test_the_imu_filter_delay_is_the_register_maps_for_the_dlpf_the_driver_writes() -> None:
    """One filter, three places: mpu6050.hpp writes kDlpfConfig to CONFIG, its comment and
    MPU6050_GYRO_DELAY_S carry the register map's table, and config/imu.json's filter_delay_s is
    that table's entry for it. A DLPF change that leaves the delay behind fails here."""
    driver = DRIVER.read_text()
    dlpf = int(re.search(r"constexpr std::uint8_t kDlpfConfig = (0x[0-9a-fA-F]+);", driver)[1], 16)
    assert "write_register(mpu6050_register::kConfig, kDlpfConfig, error)" in driver
    timing = imu_timing(REPO / "config/imu.json")
    assert timing.filter_delay_s == MPU6050_GYRO_DELAY_S[dlpf] == 0.0048
    row = next(line for line in driver.splitlines() if "gyro delay" in line)
    assert [float(v) for v in re.findall(r"([\d.]+) ms", row)] == pytest.approx(
        [1000 * MPU6050_GYRO_DELAY_S[cfg] for cfg in range(7)]
    )


def test_the_imu_output_rate_is_the_chips_whole_khz_and_the_bridge_reads_slower() -> None:
    """The chip refreshes its registers at 1 kHz (SMPLRT_DIV 0) so the sample read is at most
    1 ms old; the bridge's read rate is its own, pepin.deployment.IMU_RATE_HZ."""
    from pepin.deployment import IMU_RATE_HZ

    timing = imu_timing(REPO / "config/imu.json")
    assert timing.output_rate_hz == 1000.0 > IMU_RATE_HZ
    assert timing.bridge_parameters() == {
        "imu_output_rate_hz": 1000.0,
        "imu_filter_delay_s": 0.0048,
    }


def test_imu_timing_refuses_a_missing_block_and_impossible_numbers(tmp_path: Path) -> None:
    """No block is a KeyError (the launch says so and the bridge keeps its defaults); a rate that
    is not positive or a negative delay is a ValueError, never a stamp."""
    path = tmp_path / "imu.json"
    path.write_text(json.dumps({"mount": {}}))
    with pytest.raises(KeyError):
        imu_timing(path)
    for rate, delay in ((0.0, 0.0048), (1000.0, -0.001)):
        path.write_text(json.dumps({"timing": {"output_rate_hz": rate, "filter_delay_s": delay}}))
        with pytest.raises(ValueError):
            imu_timing(path)
    path.write_text(json.dumps({"timing": {"output_rate_hz": 100, "filter_delay_s": 0}}))
    assert imu_timing(path) == ImuTiming(100.0, 0.0), "the old divider, the read's own stamp"


def test_the_imu_bridge_opens_at_the_output_rate_and_stamps_the_read_less_the_delay() -> None:
    """base_bridge.cpp declares both numbers, opens the chip at the OUTPUT rate (the read loop
    keeps imu_rate_hz), takes the ROS clock right after the burst and publishes that moment less
    the filter's delay -- no second now() in publish_imu."""
    cpp = BRIDGE.read_text()
    assert 'declare_parameter<double>("imu_output_rate_hz", 1000.0)' in cpp
    assert 'declare_parameter<double>("imu_filter_delay_s", 0.0)' in cpp
    assert "imu_.open_device(imu_device_, imu_address_, imu_output_rate_hz_, error)" in cpp
    assert "const auto tick = period(1.0 / imu_rate_hz_);" in cpp, "the read loop's own rate"
    read = cpp[cpp.index("const auto sample = imu_.read_sample(error);") :]
    assert read.index("const rclcpp::Time read_at = now();") < read.index("if (!sample")
    publish = cpp[cpp.index("void publish_imu(") :]
    publish = publish[: publish.index("\n  }\n")]
    assert (
        "message.header.stamp = read_at - rclcpp::Duration::from_seconds(imu_filter_delay_s_);"
        in publish
    )
    assert "now()" not in publish


def test_the_launch_hands_the_imu_timing_to_the_bridge() -> None:
    """robot.launch.py reads config/imu.json's timing block at every (re)spawn and passes it
    beside the read rate."""
    robot = sf.tree("ros/pepin_bringup/launch/robot.launch.py")
    assert robot is not None
    imu = next(
        ast.unparse(f)
        for f in ast.walk(robot)
        if isinstance(f, ast.FunctionDef) and f.name == "imu_parameters"
    )
    assert "imu_timing().bridge_parameters()" in imu
    assert "**imu_parameters()" in ast.unparse(robot)


def test_the_tof_offset_lies_inside_one_ranging_window_before_t() -> None:
    """config/tof.json's offset: the mean middle of the three ranging windows against the line's
    t, -19 ms (each result ready -1/+6/+14 ms around t, each window the 50 ms budget ending
    there). Whatever it is re-measured to, it sits between one inter-measurement period before
    t and half a budget after it."""
    offset = tof_timing_offset_s(REPO / "config/tof.json")
    assert offset == -0.019
    assert -INTER_MEASUREMENT_MS / 1000 < offset < TIMING_BUDGET_MS / 2000


def test_a_tof_line_lags_by_its_board_age_less_the_offset() -> None:
    """The ROS clock read with the board's monotonic clock carries t across: a line 25 ms old
    whose windows' middle is 19 ms before t happened 44 ms before now."""
    assert measurement_lag_s(100.025, 100.0, -0.019) == pytest.approx(0.044)
    assert measurement_lag_s(100.025, 100, 0.0) == pytest.approx(0.025), "an int t counts"
    assert measurement_lag_s(100.0, 100.0, 0.010) == pytest.approx(-0.010), "a later window"


@pytest.mark.parametrize(
    "t",
    [None, "100.0", True, float("nan"), float("inf"), 100.0 - LINE_MAX_AGE_S - 0.001, 100.001],
)
def test_a_tof_line_without_a_believable_t_is_dated_on_arrival(t: object) -> None:
    """No t, a t that is not a number, one older than 0.5 s or from the future (another
    machine's monotonic clock): None, and the caller stamps the arrival."""
    assert measurement_lag_s(100.0, t, -0.019) is None
