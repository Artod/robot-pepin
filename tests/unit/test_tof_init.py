"""board/tof_init.sh against a simulated I2C bus: the three VL53L1X get 0x30, 0x31 and 0x32, the
result is VERIFIED, and a sequence that left one silent is run again until it converges.

On 2026-09-23 all three sensors read status 255 (no I2C answer) for a whole day — the bus showed
one sensor at the factory 0x29 and none at 0x30-0x32 — until a manual restart of tof-init. The
script had run its sequence once and exited 0 whatever it found. Here ``gpioset``, ``i2cdetect``,
``i2ctransfer`` and ``sleep`` are fakes on PATH sharing one bus model: the PC9 sensor is always
awake, the PC5 and PC6 sensors follow their XSHUT lines (low resets one to 0x29), an address write
moves EVERY sensor awake at the old address, and a sensor can be told to acknowledge a write and
ignore it (``refuse``) or to be gone (``dead``).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "board/tof_init.sh"

# gpioset is bash, so its lines are written within milliseconds of the `&`: the levels are a file
# the bus model reads, and the process lingers as the real one does until the script kills it.
FAKE_GPIOSET = r"""#!/bin/bash
shift 2  # -c CHIP
printf '%s\n' "$*" > "$TOF_SIM/levels.tmp" && mv "$TOF_SIM/levels.tmp" "$TOF_SIM/levels"
exec /bin/sleep 30
"""
# The script's waits are real but short: the hold gives the backgrounded gpioset time to land.
FAKE_SLEEP = """#!/bin/bash
exec /bin/sleep 0.1
"""
FAKE_I2C = """#!{python}
import json, os, sys
from pathlib import Path

SIM = Path(os.environ["TOF_SIM"])
state = json.loads((SIM / "state.json").read_text())
levels = {{}}
if (SIM / "levels").exists():
    for pair in (SIM / "levels").read_text().split():
        line, value = pair.split("=")
        levels[line] = int(value)
awake = []
for sensor in state["sensors"]:
    line = sensor["line"]
    on = line is None or levels.get(str(line), 1) == 1
    if not on:
        sensor["addr"] = 0x29  # XSHUT low: the address is volatile and gone
    if on and not sensor["dead"]:
        awake.append(sensor)
state.setdefault("calls", []).append(" ".join([Path(sys.argv[0]).name, *sys.argv[1:]]))
code = 0
if Path(sys.argv[0]).name == "i2cdetect":
    here = {{s["addr"] for s in awake}}
    print("     " + " ".join(f" {{c:x}}" for c in range(16)))
    for row in range(0, 0x80, 0x10):
        cells = [f"{{a:02x}}" if a in here else "--" for a in range(row, row + 16)]
        print(f"{{row:02x}}: " + " ".join(cells))
else:
    first = sys.argv[3]
    address = int(first.split("@")[1], 16)
    there = [s for s in awake if s["addr"] == address]
    if not there:
        code = 1
        print("Error: Sending messages failed: Remote I/O error", file=sys.stderr)
    elif first.startswith("w3@"):
        target = int(sys.argv[6], 16)
        refuse = state.setdefault("refuse", {{}})
        if refuse.get(f"{{target:x}}", 0) > 0:
            refuse[f"{{target:x}}"] -= 1  # acknowledged, not taken
        else:
            for sensor in there:
                sensor["addr"] = target
    else:
        print("0xea 0xcc")
(SIM / "state.json").write_text(json.dumps(state))
sys.exit(code)
"""


def _run(tmp_path: Path, refuse: dict[str, int] | None = None, dead: str = "") -> dict[str, Any]:
    """tof_init.sh on a fresh boot (all three awake at 0x29, both XSHUT lines high; ``dead`` names
    the sensors that are gone, space-separated); returns the bus afterwards, the script's output,
    its lines and its log file."""
    sim = tmp_path / "sim"
    bin_dir = tmp_path / "bin"
    sim.mkdir()
    bin_dir.mkdir()
    sensors = [
        {"name": name, "line": line, "addr": 0x29, "dead": name in dead.split()}
        for name, line in (("pc9", None), ("pc5", 69), ("pc6", 70))
    ]
    (sim / "state.json").write_text(json.dumps({"sensors": sensors, "refuse": refuse or {}}))
    fakes = {
        "gpioset": FAKE_GPIOSET,
        "sleep": FAKE_SLEEP,
        "i2cdetect": FAKE_I2C.format(python=sys.executable),
        "i2ctransfer": FAKE_I2C.format(python=sys.executable),
    }
    for name, text in fakes.items():
        (bin_dir / name).write_text(text)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "tof_init.log"
    run = subprocess.run(
        ["bash", str(SCRIPT)],
        capture_output=True,
        text=True,
        timeout=120,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "TOF_SIM": str(sim),
            "TOF_INIT_LOG": str(log),
        },
    )
    state = json.loads((sim / "state.json").read_text())
    return {
        "code": run.returncode,
        "out": run.stdout + run.stderr,
        "lines": (run.stdout + run.stderr).splitlines(),
        "addr": {s["name"]: s["addr"] for s in state["sensors"]},
        "log": log.read_text() if log.exists() else "",
        "readdr": [c for c in state["calls"] if " w3@" in c],
    }


def test_the_script_parses() -> None:
    result = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr


@pytest.mark.slow
def test_a_clean_boot_is_addressed_once_verified_and_logged(tmp_path: Path) -> None:
    got = _run(tmp_path)
    assert got["code"] == 0, got["out"]
    assert got["addr"] == {"pc9": 0x30, "pc5": 0x31, "pc6": 0x32}
    assert "tof_init: attempt 1/3: all three answer" in got["lines"]
    assert "0x30 OK" in got["out"] and "MISSING" not in got["out"]
    assert (
        got["log"]
        .rstrip()
        .endswith("tof_init: 0x30 0x31 0x32 answer (VL53L1X model id) after 1 attempt(s)")
    )


@pytest.mark.slow
def test_a_sensor_that_missed_its_address_is_retried_until_all_three_answer(
    tmp_path: Path,
) -> None:
    """The day of status 255 in one sensor: PC5's sensor acknowledges 0x31 and keeps 0x29, so
    PC6's wakes beside it and one write sends both to 0x32 — 0x31 is silent. The second attempt
    holds both XSHUT lines low (both back to 0x29) and addresses them afresh."""
    got = _run(tmp_path, refuse={"31": 1})
    assert got["code"] == 0, got["out"]
    assert "tof_init: attempt 1/3: silent at 0x31" in got["lines"]
    assert "tof_init: attempt 2/3: all three answer" in got["lines"]
    assert got["addr"] == {"pc9": 0x30, "pc5": 0x31, "pc6": 0x32}
    assert "after 2 attempt(s)" in got["log"]


@pytest.mark.slow
def test_the_always_on_sensor_is_taken_back_from_where_a_shared_write_left_it(
    tmp_path: Path,
) -> None:
    """PC9's sensor cannot be reset, so if it misses 0x30, PC5's wakes beside it at 0x29 and one
    write moves BOTH to 0x31. A retry that only looked at 0x29 would never find it again; phase 1
    takes it from 0x31 while the other two are held in reset, since 0x30 is empty."""
    got = _run(tmp_path, refuse={"30": 1})
    assert got["code"] == 0, got["out"]
    assert "tof_init: attempt 1/3: silent at 0x30" in got["lines"]
    assert "i2ctransfer -y 2 w3@0x31 0x00 0x01 0x30" in got["readdr"]
    assert got["addr"] == {"pc9": 0x30, "pc5": 0x31, "pc6": 0x32}
    assert "after 2 attempt(s)" in got["log"]


@pytest.mark.slow
def test_a_sensor_that_is_gone_ends_in_a_logged_failure_and_leaves_the_others_up(
    tmp_path: Path,
) -> None:
    """Three attempts, then a FAILED line naming the silent address — in the journal and in the
    log file that outlives it — and exit 0: pepin-tof Requires= this unit and streams the two
    sensors that answer."""
    got = _run(tmp_path, dead="pc6")
    assert got["code"] == 0, got["out"]
    assert "tof_init: attempt 3/3: silent at 0x32" in got["lines"]
    assert "0x32 MISSING" in got["out"] and "0x31 OK" in got["out"]
    assert "FAILED after 3 attempts: silent at 0x32" in got["log"]
    assert got["addr"]["pc9"] == 0x30 and got["addr"]["pc5"] == 0x31


@pytest.mark.slow
def test_two_silent_sensors_are_named_one_address_each(tmp_path: Path) -> None:
    """The attempt line lists every silent address once, "0x30 0x32" — it read "0x30 0x3230 32"
    until the message was built before the echo — and so does the line the log file keeps."""
    got = _run(tmp_path, dead="pc9 pc6")
    assert got["code"] == 0, got["out"]
    assert "tof_init: attempt 1/3: silent at 0x30 0x32" in got["lines"]
    assert "tof_init: attempt 3/3: silent at 0x30 0x32" in got["lines"]
    assert "FAILED after 3 attempts: silent at 0x30 0x32 (0x29" in got["log"]
    assert got["addr"]["pc5"] == 0x31, "the one that answers is addressed all the same"
