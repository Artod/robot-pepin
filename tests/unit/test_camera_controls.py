"""The head camera's exposure as config (pepin.camera_controls): a v4l2-ctl listing in, the writes
out, every value checked against what the camera itself offers.

The listings are in v4l2-ctl's own format for a UVC camera of this kind (the module's real one is
read on the robot with ros/exposure.sh show); v4l2-ctl itself is faked by a recorder that answers
the listing, accepts the writes and reads back what was written — or what a stubborn camera
keeps.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import pytest

from pepin import camera_controls as cc
from pepin.camera_controls import Control, Exposure, Step, apply, parse_controls, plan, show

REPO = Path(__file__).resolve().parents[2]
APERTURE_ONLY = """
User Controls

                     brightness 0x00980900 (int)    : min=-64 max=64 step=1 default=0 value=0
                           gain 0x00980913 (int)    : min=0 max=100 step=1 default=32 value=32

Camera Controls

                  auto_exposure 0x009a0901 (menu)   : min=0 max=3 default=3 value=3 (Aperture Priority Mode)
				1: Manual Mode
				3: Aperture Priority Mode
         exposure_time_absolute 0x009a0902 (int)    : min=1 max=5000 step=1 default=157 value=157 flags=inactive
     exposure_dynamic_framerate 0x009a0903 (bool)   : default=0 value=1
"""  # noqa: E501 — v4l2-ctl's own lines
SHUTTER = APERTURE_ONLY.replace(
    "\t\t\t\t3: Aperture Priority Mode",
    "\t\t\t\t2: Shutter Priority Mode\n\t\t\t\t3: Aperture Priority Mode",
)
OLD_KERNEL = """
                  exposure_auto 0x009a0901 (menu)   : min=0 max=3 default=3 value=3
				1: Manual Mode
				3: Aperture Priority Mode
              exposure_absolute 0x009a0902 (int)    : min=3 max=2047 step=1 default=250 value=250 flags=inactive
         exposure_auto_priority 0x009a0903 (bool)   : default=0 value=0
"""  # noqa: E501


class FakeV4l2:
    """v4l2-ctl as the module calls it: the listing, then writes it remembers and reads back."""

    def __init__(self, listing: str, ignores: Sequence[str] = (), fails: str = "") -> None:
        self.listing = listing
        self.values: dict[str, int] = {}
        self.ignores = set(ignores)  # controls a stubborn camera does not take
        self.fails = fails  # an argument whose call fails
        self.calls: list[list[str]] = []

    def __call__(self, argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
        argv = list(argv)
        self.calls.append(argv)
        if self.fails and any(self.fails in a for a in argv):
            return subprocess.CompletedProcess(argv, 1, "", "VIDIOC_S_EXT_CTRLS: failed")
        arg = argv[-1]
        if arg == "--list-ctrls-menus":
            return subprocess.CompletedProcess(argv, 0, self.listing, "")
        if arg.startswith("--set-ctrl="):
            name, value = arg.removeprefix("--set-ctrl=").split("=")
            if name not in self.ignores:
                self.values[name] = int(value)
            return subprocess.CompletedProcess(argv, 0, "", "")
        name = arg.removeprefix("--get-ctrl=")
        return subprocess.CompletedProcess(argv, 0, f"{name}: {self.values.get(name, 0)}\n", "")


def sets(fake: FakeV4l2) -> list[str]:
    return [c[-1].removeprefix("--set-ctrl=") for c in fake.calls if c[-1].startswith("--set")]


# ---- the listing ----------------------------------------------------------------------------
def test_a_listing_is_read_into_controls_with_their_ranges_menus_and_flags() -> None:
    controls = parse_controls(APERTURE_ONLY)
    assert controls["auto_exposure"] == Control(
        "auto_exposure", "menu", 0, 3, 3, 3, (), {1: "Manual Mode", 3: "Aperture Priority Mode"}
    )
    time = controls["exposure_time_absolute"]
    assert (time.minimum, time.maximum, time.default, time.flags) == (1, 5000, 157, ("inactive",))
    assert controls["exposure_dynamic_framerate"].default == 0
    assert controls["gain"].maximum == 100 and "brightness" in controls
    assert controls["auto_exposure"].refuse(2) is not None, "not offered: refused, never sent"
    assert controls["exposure_time_absolute"].refuse(6000) is not None


# ---- the modes ------------------------------------------------------------------------------
def test_auto_puts_back_the_camera_s_own_defaults() -> None:
    steps, notes = plan(Exposure(), parse_controls(APERTURE_ONLY))
    assert [(s.control, s.value) for s in steps] == [
        ("auto_exposure", 3),
        ("exposure_dynamic_framerate", 0),
        ("gain", 32),
    ]
    assert notes == [] and all(s.why == "the camera's default" for s in steps)


def test_manual_sets_the_mode_before_the_time_it_unlocks_and_the_gain_after() -> None:
    steps, _ = plan(Exposure("manual", 8.0, 40), parse_controls(APERTURE_ONLY))
    assert steps == [
        Step("auto_exposure", 1, "manual exposure"),
        Step("exposure_time_absolute", 80, "8 ms"),
        Step("gain", 40, "the configured gain"),
    ]
    without_gain, _ = plan(Exposure("manual", 5.0), parse_controls(APERTURE_ONLY))
    assert [s.control for s in without_gain] == ["auto_exposure", "exposure_time_absolute"]


def test_a_value_the_camera_does_not_offer_is_refused_with_its_range() -> None:
    controls = parse_controls(APERTURE_ONLY)
    with pytest.raises(ValueError, match="over the camera's maximum 5000"):
        plan(Exposure("manual", 600.0), controls)
    with pytest.raises(ValueError, match="over the camera's maximum 100"):
        plan(Exposure("manual", 5.0, 400), controls)
    no_gain = {k: v for k, v in controls.items() if k != "gain"}
    with pytest.raises(ValueError, match="no gain control"):
        plan(Exposure("manual", 5.0, 10), no_gain)
    with pytest.raises(ValueError, match="needs exposure_ms"):
        Exposure("capped")
    with pytest.raises(ValueError, match="not one of"):
        Exposure("sport")


def test_capped_holds_the_time_under_shutter_priority_where_the_camera_has_it() -> None:
    steps, notes = plan(Exposure("capped", 10.0), parse_controls(SHUTTER))
    assert [(s.control, s.value) for s in steps] == [
        ("auto_exposure", 2),
        ("exposure_time_absolute", 100),
    ]
    assert notes == []


def test_capped_without_shutter_priority_says_its_cap_is_one_frame_period() -> None:
    steps, notes = plan(Exposure("capped", 10.0), parse_controls(APERTURE_ONLY))
    assert [(s.control, s.value) for s in steps] == [
        ("auto_exposure", 3),
        ("exposure_dynamic_framerate", 0),
    ]
    assert len(notes) == 1 and "ONE FRAME PERIOD" in notes[0] and "10 ms" in notes[0]
    bare = {
        k: v for k, v in parse_controls(APERTURE_ONLY).items() if k != "exposure_dynamic_framerate"
    }
    with pytest.raises(ValueError, match="cannot be capped here"):
        plan(Exposure("capped", 10.0), bare)


def test_an_older_kernel_s_control_names_are_the_same_controls() -> None:
    controls = parse_controls(OLD_KERNEL)
    steps, _ = plan(Exposure("manual", 2.5), controls)
    assert [(s.control, s.value) for s in steps] == [
        ("exposure_auto", 1),
        ("exposure_absolute", 25),
    ]
    steps, _ = plan(Exposure(), controls)
    assert [s.control for s in steps] == ["exposure_auto", "exposure_auto_priority"]
    with pytest.raises(ValueError, match="no auto_exposure"):
        plan(Exposure(), {})


# ---- on the device --------------------------------------------------------------------------
def test_apply_writes_in_order_reads_every_value_back_and_names_one_ignored() -> None:
    fake = FakeV4l2(APERTURE_ONLY, ignores=("gain",))
    lines = apply("/dev/cam", Exposure("manual", 8.0, 40), run=fake)
    assert sets(fake) == ["auto_exposure=1", "exposure_time_absolute=80", "gain=40"]
    assert all(c[:3] == ["v4l2-ctl", "-d", "/dev/cam"] for c in fake.calls)
    assert lines[0] == "exposure manual 8 ms, gain 40 on /dev/cam"
    assert lines[1] == "set auto_exposure=1 (manual exposure)"
    assert "BUT the camera reads 0: it ignored it" in lines[3]


def test_a_dry_run_sets_nothing_and_a_failing_v4l2_ctl_is_an_error() -> None:
    fake = FakeV4l2(APERTURE_ONLY)
    lines = apply("/dev/cam", Exposure("manual", 8.0), run=fake, dry_run=True)
    assert sets(fake) == [] and lines[-1].startswith("would set exposure_time_absolute=80")
    with pytest.raises(RuntimeError, match="could not set"):
        apply("/dev/cam", Exposure("manual", 8.0), run=FakeV4l2(APERTURE_ONLY, fails="--set"))
    with pytest.raises(RuntimeError, match="could not list"):
        show("/dev/cam", run=FakeV4l2(APERTURE_ONLY, fails="--list"))
    assert show("/dev/cam", run=FakeV4l2(APERTURE_ONLY))[0].startswith(
        "auto_exposure: value 3 default 3 range 0..3 [1 Manual Mode] [3 Aperture Priority Mode]"
    )


def test_the_command_line_applies_the_config_or_an_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    fake = FakeV4l2(APERTURE_ONLY)
    monkeypatch.setattr(cc, "_run", fake)
    monkeypatch.delenv("PEPIN_CAMERA", raising=False)
    config = tmp_path / "camera.json"
    config.write_text(json.dumps({"active": "stereo", "stereo": {"exposure": {"mode": "auto"}}}))
    assert cc.main(["apply", "--device", "/dev/cam", "--config", str(config)]) == 0
    assert sets(fake)[0] == "auto_exposure=3"
    argv = ["apply", "--device", "/dev/cam", "--mode", "manual", "--exposure-ms", "4"]
    assert cc.main(argv) == 0
    assert sets(fake)[-1] == "exposure_time_absolute=40"
    assert cc.main(["apply", "--device", "/dev/cam", "--mode", "manual"]) == 1
    assert "needs exposure_ms" in capsys.readouterr().err
    monkeypatch.delenv("PEPIN_CAMERA_DEVICE", raising=False)
    assert cc.main(["show"]) == 2


# ---- the config and the unit ----------------------------------------------------------------
def test_the_shipped_config_is_auto_and_a_rig_without_a_block_is_auto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PEPIN_CAMERA", raising=False)
    data = json.loads((REPO / "config/camera.json").read_text())
    assert Exposure.from_camera_json(data) == Exposure("auto", 8.0, None), (
        "auto as shipped: the camera as it ran before the block existed"
    )
    assert Exposure.from_camera_json(data, "overview") == Exposure(), "no block: auto"
    monkeypatch.setenv("PEPIN_CAMERA", "overview")
    assert Exposure.from_camera_json(data) == Exposure()
    with pytest.raises(ValueError, match="no rig"):
        Exposure.from_camera_json(data, "fisheye")


def test_the_camera_service_applies_it_first_and_streams_whatever_happens() -> None:
    unit = (REPO / "board/pepin-camera.service").read_text()
    pre = [line for line in unit.splitlines() if line.startswith("ExecStartPre=")]
    assert pre == [
        "ExecStartPre=-/opt/pepin/bin/python -m pepin.camera_controls apply --device"
        " ${PEPIN_CAMERA_DEVICE} --config /opt/pepin/config/camera.json"
    ], "the leading '-': a failure there never keeps the stream down"
    assert "Environment=PYTHONPATH=/opt/pepin" in unit
    assert unit.index("ExecStartPre=") < unit.index("ExecStart=/usr/bin/ustreamer")


def test_the_module_needs_nothing_but_the_standard_library() -> None:
    """It runs on the board, where src/pepin is rsynced without its dependencies."""
    tree = ast.parse((REPO / "src/pepin/camera_controls.py").read_text())
    imported = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.level == 0
    }
    assert imported <= set(sys.stdlib_module_names) | {"__future__"}, imported
