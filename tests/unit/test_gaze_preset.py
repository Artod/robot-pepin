"""ros/tools/gaze_preset.py and config/gaze_presets.json: the baseline set (the head of drives
306/307) and the follow set (the knobs' defaults), each a valid live value of a gaze knob."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pepin.flags import load_knobs

REPO = Path(__file__).resolve().parents[2]


def _tool() -> ModuleType:
    spec = importlib.util.spec_from_file_location("gaze_preset", REPO / "ros/tools/gaze_preset.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("gaze_preset", module)
    spec.loader.exec_module(module)
    return module


TOOL = _tool()
KNOBS = load_knobs("gaze", REPO / "config/knobs.json")


def test_both_sets_name_the_same_gaze_knobs_with_values_they_take() -> None:
    sets = TOOL.presets()
    assert set(sets) == {"baseline", "follow"}
    assert list(sets["baseline"]) == list(sets["follow"])
    for values in sets.values():
        for name, value in values.items():
            assert KNOBS.flag(name).parse(str(value)) == value, name


def test_follow_is_the_knobs_defaults_and_baseline_turns_each_one_off() -> None:
    sets = TOOL.presets()
    assert all(KNOBS[name] == value for name, value in sets["follow"].items())
    baseline = sets["baseline"]
    assert baseline.pop("path_deadband_deg") == 8.0  # the dead-band before 2026-10-05
    assert set(baseline.values()) == {0}


def test_show_prints_both_sets_and_a_set_goes_through_flags_sh(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert TOOL.main(["show"]) == 0
    assert capsys.readouterr().out.splitlines()[1].split() == [
        "path_deadband_deg",
        "8.0",
        "22.0",
        "22.0",
    ]
    calls: list[list[str]] = []

    class Done:
        returncode = 0

    def run(command: list[str], **_kwargs: Any) -> Done:
        calls.append(command)
        return Done()

    monkeypatch.setattr(TOOL.subprocess, "run", run)
    assert TOOL.main(["baseline"]) == 0
    assert calls[0] == [str(REPO / "ros/flags.sh"), "set", "gaze", "path_deadband_deg", "8.0"]
    assert len(calls) == len(TOOL.presets()["baseline"])
    assert TOOL.main(["nope"]) == 2
