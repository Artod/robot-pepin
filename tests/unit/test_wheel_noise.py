"""The wheels' noise law (pepin.wheel_noise): /odom's twist covariance from the measured motion,
its numbers, its config and the bridge's switch that falls back to the constant."""

from __future__ import annotations

import json
import math
import re
from dataclasses import fields
from pathlib import Path

import pytest

from pepin.flags import load_table
from pepin.geometry import BaseConfig
from pepin.wheel_noise import WheelNoiseLaw

REPO = Path(__file__).resolve().parents[2]
CPP = REPO / "ros/pepin_base_cpp/src/base_bridge.cpp"
HPP = REPO / "ros/pepin_base_cpp/include/pepin_base_cpp/wheel_noise.hpp"
LAW = WheelNoiseLaw()


def test_at_rest_the_law_says_nothing_and_the_constant_stays() -> None:
    assert LAW.sigmas(0.0, 0.0) is None
    assert LAW.sample_variances(0.0, 0.0, 50.0) is None
    # A wheel's one-tick jitter on a parked cart (2.4 mm/s, 0.5 deg/s) and a slow creep are rest.
    assert LAW.sample_variances(0.0024, 0.0088, 50.0) is None
    assert LAW.sample_variances(-0.029, math.radians(2.9), 50.0) is None


def test_a_straight_at_0_3_m_s_gets_the_floors() -> None:
    assert LAW.sigmas(0.3, 0.0) == pytest.approx((0.026, 0.038))
    var_v, var_w = LAW.sample_variances(0.3, 0.0, 50.0) or (0.0, 0.0)
    assert var_v == pytest.approx(50 * 0.026**2) and var_v == pytest.approx(0.0338)
    assert var_w == pytest.approx(50 * 0.038**2) and var_w == pytest.approx(0.0722)
    # 34x and 7x the constant 0.001 / 0.01: what a moving second of these wheels is worth.
    assert var_v / 0.001 == pytest.approx(33.8) and var_w / 0.01 == pytest.approx(7.22)


def test_a_pivot_at_1_rad_s_adds_the_turns_share_in_either_direction() -> None:
    for w in (1.0, -1.0):
        assert LAW.sigmas(0.0, w) == pytest.approx((0.060, 0.238))
        var_v, var_w = LAW.sample_variances(0.0, w, 50.0) or (0.0, 0.0)
        assert var_v == pytest.approx(0.18)
        assert var_w == pytest.approx(2.8322)


def test_motion_starts_at_either_threshold() -> None:
    assert LAW.moving(0.03, 0.0) and LAW.moving(-0.03, 0.0)
    assert LAW.moving(0.0, 0.052) and LAW.moving(0.0, -0.052)
    assert not LAW.moving(0.0299, 0.0519)


def test_the_variance_scales_with_the_stream_rate() -> None:
    """One sample's share of a second: half the rate, half the variance per sample."""
    at_50 = LAW.sample_variances(0.2, 0.5, 50.0)
    at_25 = LAW.sample_variances(0.2, 0.5, 25.0)
    assert at_50 is not None and at_25 is not None
    assert at_25 == pytest.approx((at_50[0] / 2, at_50[1] / 2))


def test_the_repo_config_carries_the_fit_and_the_effective_track() -> None:
    raw = json.loads((REPO / "config/base.json").read_text())
    law = WheelNoiseLaw.from_json(REPO / "config/base.json")
    assert law == LAW, "config/base.json's block and the code's defaults are one fit"
    assert "note" in raw["odometry_noise"]
    cfg = BaseConfig.from_json(REPO / "config/base.json")
    assert cfg.geometry.track_width_m == 0.544
    assert cfg.geometry.wheel_diameter_m == 0.125


def test_a_file_without_the_block_gives_the_fit_and_a_bad_number_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "base.json"
    path.write_text("{}")
    assert WheelNoiseLaw.from_json(path) == LAW
    with pytest.raises(ValueError):
        WheelNoiseLaw.from_dict({"v_floor_m_s": -0.01})
    with pytest.raises(ValueError):
        WheelNoiseLaw.from_dict({"yaw_per_yaw_rate": float("nan")})


def test_the_cpp_header_defaults_are_the_same_fit() -> None:
    text = HPP.read_text()
    for f in fields(WheelNoiseLaw):
        m = re.search(rf"double {f.name} = ([0-9.]+);", text)
        assert m, f"{f.name} missing from wheel_noise.hpp"
        assert float(m.group(1)) == getattr(LAW, f.name), f.name
    assert re.search(r"double rate_hz = 50\.0;", text)


def test_the_launch_hands_every_coefficient_to_the_bridge_read_only() -> None:
    params = LAW.bridge_parameters(50)
    assert params["odom_law_rate_hz"] == 50.0
    cpp = CPP.read_text()
    for name in params:
        assert f'law_number("{name}", ' in cpp, f"{name} not declared by the bridge"
    assert "fixed.read_only = true;" in cpp
    launch = (REPO / "ros/pepin_bringup/launch/robot.launch.py").read_text()
    assert "**wheel_noise_parameters()," in launch
    assert 'WheelNoiseLaw.from_json(config_file("base.json")).bridge_parameters(STATE_HZ)' in launch


def test_the_switch_falls_back_to_the_constant_live() -> None:
    """odom_covariance: `law` by default, `constant` one live set away, read per state line, named
    in the report line; the law always reads the MEASURED wheel twist, whatever the twist source."""
    flag = load_table(REPO / "ros/pepin_bringup/pepin_bringup/base_bridge.py").flag(
        "odom_covariance"
    )
    assert flag.default == "law" and set(flag.choices) == {"law", "constant"} and flag.live
    cpp = CPP.read_text()
    assert 'declare_parameter<std::string>("odom_covariance", "law")' in cpp
    state = cpp[cpp.index("void publish_state(const BaseState & state, const LineTime & when)") :]
    state = state[: state.index("\n  }\n")]
    assert 'get_parameter("odom_covariance").as_string() != "constant"' in state, "read per line"
    assert "wheel_twist_covariance(wheel_law_, wheel_twist_, twist_covariance_)" in state
    assert "law ?" in state and ": twist_covariance_;" in state, "the constant path kept"
    twist = cpp[cpp.index("BodyTwist odom_twist(const BaseState & state)") :]
    twist = twist[: twist.index("\n  }\n")]
    assert "wheel_twist_ = wheels;" in twist, "the measured twist, never the command"
    assert '" odom_covariance="' in cpp, "the report line names the switch"
