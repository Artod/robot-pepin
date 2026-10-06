"""The board's EKF is a config file, so its fused indices are asserted like code.

Which quantity comes from which source is an architecture decision (one source must never be a
single point of failure), and a yaml is one careless edit from silently dropping one. These
tests fail loudly when a fusion appears or disappears without the comment that explains it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from pepin.base_link import STATE_HZ
from pepin.deployment import IMU_RATE_HZ

EKF = Path(__file__).resolve().parents[2] / "ros" / "params" / "ekf.yaml"
# The 15 states robot_localization fuses, in message order.
STATES = (
    "x",
    "y",
    "z",
    "roll",
    "pitch",
    "yaw",
    "vx",
    "vy",
    "vz",
    "vroll",
    "vpitch",
    "vyaw",
    "ax",
    "ay",
    "az",
)


@pytest.fixture(scope="module")
def params() -> dict[str, Any]:
    """The ekf_filter_node's parameters as a dict."""
    loaded = yaml.safe_load(EKF.read_text())
    return dict(loaded["ekf_filter_node"]["ros__parameters"])


def fused(params: dict[str, Any], key: str) -> set[str]:
    """The names of the states a source's ``*_config`` vector switches on."""
    return {name for name, on in zip(STATES, params[key], strict=True) if on}


def test_wheels_give_speed_and_not_the_commanded_yaw_rate(params: dict[str, Any]) -> None:
    """odom0 is /odom: forward and sideways speed, and the wheels' yaw rate as the gyro's backup.

    vyaw stays off while the bridge on the board publishes the COMMANDED twist rather than a
    measured one (base_server.py:466) -- fusing it would feed the controller's output back in
    as a heading sensor. The yaml says so in full; this is the assertion.
    """
    # vyaw since 2026-09-15: the twist is measured, ~4 % of the gyro's weight
    assert fused(params, "odom0_config") == {"vx", "vy", "vyaw"}
    assert params["odom0_differential"] is False


def test_camera_gives_position_and_yaw_differentially(params: dict[str, Any]) -> None:
    """odom1 is /vo: x, y and yaw, differenced into velocities so its origin cannot move odom."""
    assert fused(params, "odom1_config") == {"x", "y", "yaw"}
    assert params["odom1_differential"] is True


def test_the_vio_twist_is_its_own_input_with_no_gate(params: dict[str, Any]) -> None:
    """twist0 is /vo_twist: the VIO's body velocity (vx, vy, vyaw) under vo_output twist, its own
    topic so that no /vo with an empty twist block can clamp the velocity (a zero variance is
    lifted to 1e-9), queued like odom1 and pinned to the bridged QoS; no rejection threshold (the
    relay's guard is the gate)."""
    from pepin.deployment import VO_TWIST_TOPIC, bridged_qos

    assert params["twist0"] == VO_TWIST_TOPIC
    assert fused(params, "twist0_config") == {"vx", "vy", "vyaw"}
    assert fused(params, "odom1_config") == {"x", "y", "yaw"}, "odom1 reads no twist"
    assert bridged_qos(VO_TWIST_TOPIC) == ("reliable", params["twist0_queue_size"])
    assert "twist0_rejection_threshold" not in params


def test_imu_gives_the_yaw_rate_and_nothing_else(params: dict[str, Any]) -> None:
    """imu0 is /imu/data_raw: the gyro's yaw rate alone.

    ax/ay stay off because what the tapes measure on them is a BIAS (-0.229 to +0.066 m/s^2 with
    the cart standing still), and a filter with no bias state cannot be told about one with a
    covariance: fusing them costs up to 4.6 % of the forward speed while the wheels are alive
    and runs to 0.98 m/s of invented speed in 5 s when they are not (scratch/accel_bias_cost.py).
    The yaml carries the table; this is the assertion.
    """
    assert fused(params, "imu0_config") == {"vyaw"}
    # Declared and true for the day a bias state or a real orientation makes ax/ay fusable; with
    # no acceleration index on, robot_localization never prepares an acceleration measurement.
    assert params["imu0_remove_gravitational_acceleration"] is True


def test_every_heading_source_is_named_in_the_file(params: dict[str, Any]) -> None:
    """Heading must not rest on one sensor: the gyro fuses it, the camera seconds it, and the
    wheels' path back in is written down rather than forgotten."""
    assert fused(params, "imu0_config") & {"vyaw"}
    assert fused(params, "odom1_config") & {"yaw"}


def test_the_filter_runs_at_the_wheels_rate_and_queues_a_fifth_of_a_second(
    params: dict[str, Any],
) -> None:
    """One wheel line per filter cycle (a slower filter publishes odom -> base_link staler than
    the wheels measured it), and the input queues of the fast sources hold 0.2 s of them, so a
    cycle the board delays drops nothing."""
    assert params["frequency"] == STATE_HZ
    assert params["odom0_queue_size"] / STATE_HZ >= 0.2
    assert params["imu0_queue_size"] / IMU_RATE_HZ >= 0.2
