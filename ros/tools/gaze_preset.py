#!/usr/bin/env python3
"""The gaze arbiter's follow-in-zone and glance knobs set live in one go (config/gaze_presets.json).

    uv run python ros/tools/gaze_preset.py baseline   the head of drives 306/307 (2026-10-05)
                                                      exactly: a path re-aim at every 8 deg
                                                      shift, a path look that lapses home when
                                                      its source goes quiet, the reverse look
                                                      answered on arrival, no atomic glance, a
                                                      saccade home after the drive
    uv run python ros/tools/gaze_preset.py follow     the knobs' defaults: a 22 deg zone, 0.3 s
                                                      hysteresis, 2 s cooldown, no saccade in
                                                      the plan's last 0.5 s or 0.35 m, the aim
                                                      held through a mode change, glances of 3
                                                      frames or 0.6 s, home at 45 deg/s
    uv run python ros/tools/gaze_preset.py show       both sets beside the knobs' defaults; no
                                                      node touched

Each value goes through ``ros/flags.sh set gaze KNOB VALUE`` (checked against the knob's range
first). A set lives until the gaze node restarts (``ros/laptop.sh kick gaze``, ``nav`` down and
up), which brings back the defaults, i.e. ``follow``. The flags path_gaze, stall_look and
reverse_gaze are not part of a preset.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from pepin.flags import load_knobs

REPO = Path(__file__).resolve().parents[2]
PRESETS = REPO / "config/gaze_presets.json"
NAMES = ("baseline", "follow")


def presets(path: Path = PRESETS) -> dict[str, dict[str, float]]:
    """The named sets: {preset: {knob: value}}."""
    data = json.loads(path.read_text())
    return {name: dict(data[name]) for name in NAMES}


def table() -> str:
    """Both sets beside the knobs' defaults, one knob a line."""
    sets, knobs = presets(), load_knobs("gaze")
    rows = [f"{'knob':20} {'baseline':>9} {'follow':>9} {'default':>9}"]
    for name, value in sets["baseline"].items():
        rows.append(f"{name:20} {value:>9} {sets['follow'][name]:>9} {knobs[name]:>9}")
    return "\n".join(rows)


def main(argv: list[str]) -> int:
    """Set a preset live, or show both."""
    which = argv[0] if argv else ""
    if which == "show":
        print(table())
        return 0
    if which not in NAMES:
        print("usage: ros/tools/gaze_preset.py baseline|follow|show", file=sys.stderr)
        return 2
    for knob, value in presets()[which].items():
        print(f"gaze {knob} = {value}: ", end="", flush=True)
        done = subprocess.run([str(REPO / "ros/flags.sh"), "set", "gaze", knob, str(value)])
        if done.returncode != 0:
            return done.returncode
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
