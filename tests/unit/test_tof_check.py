"""The ToF preflight check's verdict: silent, unknown and healthy sensors over one window."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

REPO = Path(__file__).resolve().parents[2]


def _tool() -> ModuleType:
    """Import ros/tools/tof_check.py by path (ros/tools is not a package)."""
    spec = importlib.util.spec_from_file_location("tof_check", REPO / "ros/tools/tof_check.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses looks the module up while the class is built
    spec.loader.exec_module(module)
    return module


tof = _tool()


def test_three_live_sensors_pass_and_a_partly_unknown_one_says_its_share() -> None:
    tallies = {"front": tof.Tally(30, 0), "left": tof.Tally(30, 12), "right": tof.Tally(29, 0)}
    assert tof.verdict(tallies, 2.0) == (
        True,
        "front 15.0 Hz, left 15.0 Hz (40% unknown), right 14.5 Hz",
    )


def test_a_silent_sensor_and_an_all_unknown_one_fail_by_name() -> None:
    tallies = {"front": tof.Tally(30, 0), "left": tof.Tally(30, 30), "right": tof.Tally(3, 0)}
    passed, line = tof.verdict(tallies, 2.0)
    assert not passed
    assert line == "left unknown 30 of 30 (not answering, or < 12 cm); right silent (3 in 2 s)"


def test_nothing_at_all_fails_all_three() -> None:
    passed, line = tof.verdict({}, 2.0)
    assert not passed and line.count("silent (0 in 2 s)") == 3
