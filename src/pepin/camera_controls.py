"""The head camera's exposure as config: auto, manual exposure and gain, or auto with a cap.

How long the sensor integrates decides how far a turning head smears a frame (blur = rate x
exposure x 8.6 px/deg on the stereo eye) and how long a frame's exposure window is for the gaze
gate (:mod:`pepin.gaze_gate`); left to the module's own auto exposure it is unknown and changes
with the room. ustreamer sets no exposure ("no change" is its default for every control it does
know), so the board's camera service applies the ``exposure`` block of config/camera.json's active
rig with ``v4l2-ctl`` before ustreamer starts (board/pepin-camera.service, ExecStartPre). UVC
controls live in the camera, so what is set holds while ustreamer streams, and a change made
while it streams takes the next frame — which is how a mode is tried live
(``ros/exposure.sh``).

The modes, in V4L2's own control names (older kernels' aliases accepted):

* ``auto`` — the camera's power-on defaults for ``auto_exposure``, ``exposure_dynamic_framerate``
  and ``gain``: exactly what the module did before this file existed, and what undoes a manual
  setting without a power cycle.
* ``manual`` — ``auto_exposure`` = 1 (manual), ``exposure_time_absolute`` = ``exposure_ms`` (the
  control counts 100 us), and ``gain`` when the block names one (the camera's raw units).
* ``capped`` — auto exposure that never integrates longer than ``exposure_ms``. UVC has no such
  control as such: where the camera offers shutter priority (``auto_exposure`` = 2) the exposure is
  held at ``exposure_ms`` and the camera adapts the rest itself; where it does not, aperture
  priority with ``exposure_dynamic_framerate`` off is the closest the standard has — the auto
  exposure may not stretch a frame, so the cap is one frame period (66 ms at 15 fps), not
  ``exposure_ms``, and the output says so.

Every value is checked against the range the camera itself reports (``--list-ctrls-menus``), and
one it cannot take is refused with the reason, never clamped. Each value is read back after it is
written, and a camera that ignored one is reported. Standard library only: this runs on the board.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

MODES = ("auto", "manual", "capped")
# V4L2's names first, the pre-4.x uvcvideo names after them.
AUTO = ("auto_exposure", "exposure_auto")
TIME = ("exposure_time_absolute", "exposure_absolute")
DYNAMIC = ("exposure_dynamic_framerate", "exposure_auto_priority")
GAIN = ("gain",)
# V4L2_EXPOSURE_* (videodev2.h): the values of the auto-exposure menu.
EXPOSURE_MANUAL = 1
EXPOSURE_SHUTTER_PRIORITY = 2
EXPOSURE_APERTURE_PRIORITY = 3
TIME_UNIT_MS = 0.1  # exposure_time_absolute counts 100 us
CONFIG = Path(__file__).resolve().parents[1] / "config" / "camera.json"  # /opt/pepin/config

_LINE = re.compile(r"^\s*(?P<name>\w+)\s+0x[0-9a-f]+\s+\((?P<kind>\w+)\)\s*:\s*(?P<rest>.*)$")
_MENU = re.compile(r"^\s+(?P<index>\d+):\s*(?P<label>.+?)\s*$")
_FIELD = re.compile(r"(\w+)=(-?\d+|\w+(?:,\w+)*)")

Run = Callable[[Sequence[str]], "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class Control:
    """One V4L2 control as ``v4l2-ctl --list-ctrls-menus`` prints it."""

    name: str
    kind: str
    minimum: int | None = None
    maximum: int | None = None
    default: int | None = None
    value: int | None = None
    flags: tuple[str, ...] = ()
    menu: dict[int, str] = field(default_factory=dict)

    def refuse(self, value: int) -> str | None:
        """Why this control cannot take ``value``, or ``None`` when it can."""
        if self.menu and value not in self.menu:
            offered = ", ".join(f"{i} {label}" for i, label in sorted(self.menu.items()))
            return f"{self.name}: {value} is not offered (the camera offers {offered})"
        if self.minimum is not None and value < self.minimum:
            return f"{self.name}: {value} is under the camera's minimum {self.minimum}"
        if self.maximum is not None and value > self.maximum:
            return f"{self.name}: {value} is over the camera's maximum {self.maximum}"
        return None


def parse_controls(text: str) -> dict[str, Control]:
    """The controls of a ``v4l2-ctl --list-ctrls-menus`` listing, by name, menus included."""
    controls: dict[str, Control] = {}
    current: Control | None = None
    for line in text.splitlines():
        match = _LINE.match(line)
        if match is not None:
            fields = dict(_FIELD.findall(match["rest"]))
            current = Control(
                name=match["name"],
                kind=match["kind"],
                minimum=_int(fields.get("min")),
                maximum=_int(fields.get("max")),
                default=_int(fields.get("default")),
                value=_int(fields.get("value")),
                flags=tuple(fields["flags"].split(",")) if "flags" in fields else (),
            )
            controls[current.name] = current
            continue
        item = _MENU.match(line)
        if item is not None and current is not None and current.kind == "menu":
            current.menu[int(item["index"])] = item["label"]
    return controls


def _int(text: str | None) -> int | None:
    try:
        return None if text is None else int(text)
    except ValueError:
        return None


def find(controls: Mapping[str, Control], names: Sequence[str]) -> Control | None:
    """The first of ``names`` (a control and its aliases) the camera has."""
    for name in names:
        if name in controls:
            return controls[name]
    return None


@dataclass(frozen=True)
class Exposure:
    """The ``exposure`` block of a rig in config/camera.json."""

    mode: str = "auto"
    exposure_ms: float | None = None
    gain: int | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"exposure mode {self.mode!r} is not one of {', '.join(MODES)}")
        if self.mode != "auto" and (self.exposure_ms is None or self.exposure_ms <= 0.0):
            raise ValueError(f"exposure mode {self.mode} needs exposure_ms above 0")

    @classmethod
    def from_camera_json(cls, data: Mapping[str, Any], rig: str | None = None) -> Exposure:
        """The active rig's block (``PEPIN_CAMERA`` or ``rig`` over the file's ``active``);
        ``auto`` when the rig has none."""
        name = rig or os.environ.get("PEPIN_CAMERA") or str(data.get("active", ""))
        block = data.get(name)
        if not isinstance(block, Mapping):
            raise ValueError(f"camera.json has no rig {name!r}")
        raw = block.get("exposure")
        if raw is None:
            return cls()
        if not isinstance(raw, Mapping):
            raise ValueError(f"camera.json: {name}.exposure is not an object")
        ms, gain = raw.get("exposure_ms"), raw.get("gain")
        return cls(
            mode=str(raw.get("mode", "auto")),
            exposure_ms=None if ms is None else float(ms),
            gain=None if gain is None else int(gain),
        )

    def text(self) -> str:
        """The mode in a few words for the output."""
        if self.mode == "auto":
            return "auto (the camera's own defaults)"
        gain = f", gain {self.gain}" if self.gain is not None else ""
        return f"{self.mode} {self.exposure_ms:g} ms{gain}"


@dataclass(frozen=True)
class Step:
    """One control to write, in order, and why."""

    control: str
    value: int
    why: str


def plan(exposure: Exposure, controls: Mapping[str, Control]) -> tuple[list[Step], list[str]]:
    """The writes that put the camera in ``exposure`` (in the order they must happen: the mode
    before the time it unlocks) and the notes for the output; ``ValueError`` with the reason
    when the camera cannot do it."""
    auto = find(controls, AUTO)
    if auto is None:
        raise ValueError("the camera has no auto_exposure control: nothing to set")
    steps: list[Step] = []
    notes: list[str] = []
    if exposure.mode == "auto":
        for names in (AUTO, DYNAMIC, GAIN):
            control = find(controls, names)
            if control is not None and control.default is not None:
                steps.append(Step(control.name, control.default, "the camera's default"))
        return _checked(steps, controls), notes
    time = find(controls, TIME)
    if time is None:
        raise ValueError("the camera has no exposure_time_absolute control")
    assert exposure.exposure_ms is not None
    units = max(1, round(exposure.exposure_ms / TIME_UNIT_MS))
    if exposure.mode == "manual":
        steps.append(Step(auto.name, EXPOSURE_MANUAL, "manual exposure"))
        steps.append(Step(time.name, units, f"{units * TIME_UNIT_MS:g} ms"))
        if exposure.gain is not None:
            gain = find(controls, GAIN)
            if gain is None:
                raise ValueError("a gain is configured but the camera has no gain control")
            steps.append(Step(gain.name, exposure.gain, "the configured gain"))
        return _checked(steps, controls), notes
    if EXPOSURE_SHUTTER_PRIORITY in auto.menu:
        steps.append(Step(auto.name, EXPOSURE_SHUTTER_PRIORITY, "shutter priority"))
        steps.append(Step(time.name, units, f"held at {units * TIME_UNIT_MS:g} ms"))
        return _checked(steps, controls), notes
    dynamic = find(controls, DYNAMIC)
    if dynamic is None:
        raise ValueError(
            "the camera offers neither shutter priority nor exposure_dynamic_framerate: an auto"
            " exposure cannot be capped here; use manual"
        )
    aperture = EXPOSURE_APERTURE_PRIORITY if EXPOSURE_APERTURE_PRIORITY in auto.menu else None
    if aperture is None:
        raise ValueError(f"{auto.name} offers no aperture priority: use manual")
    steps.append(Step(auto.name, aperture, "aperture priority (the camera's auto)"))
    steps.append(Step(dynamic.name, 0, "the auto exposure may not stretch a frame"))
    notes.append(
        f"no shutter priority on this camera: the cap is ONE FRAME PERIOD (1 / the fps ustreamer"
        f" asks for), not the {exposure.exposure_ms:g} ms configured; manual holds an exact one"
    )
    return _checked(steps, controls), notes


def _checked(steps: list[Step], controls: Mapping[str, Control]) -> list[Step]:
    for step in steps:
        reason = controls[step.control].refuse(step.value)
        if reason is not None:
            raise ValueError(reason)
    return steps


def _run(argv: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(list(argv), capture_output=True, text=True, timeout=10.0, check=False)


def apply(device: str, exposure: Exposure, run: Run = _run, dry_run: bool = False) -> list[str]:
    """Put the camera at ``device`` in ``exposure``: list its controls, write the plan, read each
    value back. Returns the lines to print; ``RuntimeError`` when v4l2-ctl fails, ``ValueError``
    when the camera cannot do the mode."""
    listing = run(["v4l2-ctl", "-d", device, "--list-ctrls-menus"])
    if listing.returncode != 0:
        raise RuntimeError(f"v4l2-ctl could not list {device}: {listing.stderr.strip()}")
    controls = parse_controls(listing.stdout)
    steps, notes = plan(exposure, controls)
    lines = [f"exposure {exposure.text()} on {device}"]
    lines += [f"note: {note}" for note in notes]
    for step in steps:
        line = f"{step.control}={step.value} ({step.why})"
        if dry_run:
            lines.append(f"would set {line}")
            continue
        wrote = run(["v4l2-ctl", "-d", device, f"--set-ctrl={step.control}={step.value}"])
        if wrote.returncode != 0:
            raise RuntimeError(f"v4l2-ctl could not set {line}: {wrote.stderr.strip()}")
        read = run(["v4l2-ctl", "-d", device, f"--get-ctrl={step.control}"])
        now = _int(read.stdout.split(":")[-1].strip()) if read.returncode == 0 else None
        verdict = "" if now == step.value else f" BUT the camera reads {now}: it ignored it"
        lines.append(f"set {line}{verdict}")
    return lines


def show(device: str, run: Run = _run) -> list[str]:
    """The exposure controls as the camera reports them (ranges, defaults, current values)."""
    listing = run(["v4l2-ctl", "-d", device, "--list-ctrls-menus"])
    if listing.returncode != 0:
        raise RuntimeError(f"v4l2-ctl could not list {device}: {listing.stderr.strip()}")
    controls = parse_controls(listing.stdout)
    lines = []
    for names in (AUTO, TIME, DYNAMIC, GAIN):
        control = find(controls, names)
        if control is None:
            lines.append(f"{names[0]}: absent")
            continue
        menu = "".join(f" [{i} {label}]" for i, label in sorted(control.menu.items()))
        lines.append(
            f"{control.name}: value {control.value} default {control.default} range"
            f" {control.minimum}..{control.maximum}{menu}"
            + (f" flags {','.join(control.flags)}" if control.flags else "")
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    """``show`` or ``apply`` (the config's mode, or ``--mode`` and its numbers to try one live)."""
    parser = argparse.ArgumentParser(prog="python -m pepin.camera_controls", description=__doc__)
    parser.add_argument("verb", choices=("show", "apply"))
    parser.add_argument("--device", default=os.environ.get("PEPIN_CAMERA_DEVICE", ""))
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--rig", default=None, help="the rig block (default: the file's active)")
    parser.add_argument("--mode", choices=MODES, default=None, help="instead of the config's")
    parser.add_argument("--exposure-ms", type=float, default=None)
    parser.add_argument("--gain", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true", help="say what would be set")
    args = parser.parse_args(argv)
    if not args.device:
        print("no --device and no PEPIN_CAMERA_DEVICE", file=sys.stderr)
        return 2
    try:
        if args.verb == "show":
            lines = show(args.device, run=_run)
        else:
            if args.mode is not None:
                exposure = Exposure(args.mode, args.exposure_ms, args.gain)
            else:
                exposure = Exposure.from_camera_json(json.loads(args.config.read_text()), args.rig)
            lines = apply(args.device, exposure, run=_run, dry_run=args.dry_run)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f"camera exposure: {exc}", file=sys.stderr)
        return 1
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
