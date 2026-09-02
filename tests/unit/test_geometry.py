import math
from pathlib import Path

import pytest

from pepin.geometry import BaseConfig, BaseGeometry, WheelMotor

REPO = Path(__file__).resolve().parents[2]


def test_repo_config_loads_with_known_motor_layout() -> None:
    cfg = BaseConfig.from_json(REPO / "config" / "base.json")
    assert cfg.left.motor_id == 7 and cfg.left.direction == -1
    assert cfg.right.motor_id == 8 and cfg.right.direction == 1
    assert cfg.geometry.ticks_per_rev == 4096


def test_meters_per_tick_is_circumference_over_ticks() -> None:
    g = BaseGeometry(wheel_diameter_m=0.125, ticks_per_rev=4096)
    assert g.m_per_tick == pytest.approx(math.pi * 0.125 / 4096)


def test_direction_must_be_a_sign() -> None:
    with pytest.raises(ValueError):
        WheelMotor(motor_id=7, direction=2)
