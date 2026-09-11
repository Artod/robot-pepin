"""Every entry point answers --help: its imports resolve and its parser builds.

The servo bench tools (jog, calibrate_neck, scan_bus, setup_motor_id) import
lerobot, which pulls torch; they are left out to keep the unit tier fast.
"""

import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
BENCH = {"jog.py", "calibrate_neck.py", "scan_bus.py", "setup_motor_id.py"}
SCRIPTS = sorted(p.name for p in (REPO / "scripts").glob("*.py") if p.name not in BENCH)


@pytest.mark.parametrize("script", SCRIPTS)
def test_script_answers_help(script: str) -> None:
    result = subprocess.run(
        [sys.executable, str(REPO / "scripts" / script), "--help"],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=REPO,
    )
    assert result.returncode == 0, result.stderr[-600:]


def test_the_depth_host_launcher_parses_and_is_on_wherever_the_gpu_is() -> None:
    """ros/depth_host.sh runs the depth network on the laptop's GPU beside the containers;
    ros/laptop.sh vslam starts it and points the node at it when torch's Metal backend answers
    True — PEPIN_DEPTH_HOST=0 keeps the CPU model in the container, =1 insists without asking.
    The decision is one shell function, run here with a fake ``uv`` in torch's place."""
    import re

    result = subprocess.run(
        ["bash", "-n", str(REPO / "ros/depth_host.sh")], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    launcher = (REPO / "ros/depth_host.sh").read_text()
    assert "pepin.depth_service" in launcher and "--group depth" in launcher
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert "if depth_host_wanted; then" in laptop
    assert "PEPIN_DEPTH_BACKEND=auto" in laptop and "host.docker.internal" in laptop
    assert laptop.count("depth_host.sh") >= 2  # started with vslam, stopped with stop
    function = re.search(r"^depth_host_wanted\(\) \{.*?^\}", laptop, re.M | re.S)
    assert function is not None
    assert "torch.backends.mps.is_available()" in function.group(0)

    def wanted(env: dict[str, str], torch_says: str) -> bool:
        script = (
            f'HERE="{REPO}/ros"; uv() {{ [ -n "{torch_says}" ] && echo "{torch_says}"; }}; '
            f"{function.group(0)}; depth_host_wanted && echo yes || echo no"
        )
        out = subprocess.run(
            ["bash", "-c", script], env=env, capture_output=True, text=True, timeout=20
        )
        assert out.returncode == 0, out.stderr
        return out.stdout.strip() == "yes"

    base = {"PATH": "/usr/bin:/bin"}
    assert wanted(base, "True") and not wanted(base, "False") and not wanted(base, "")
    assert not wanted({**base, "PEPIN_DEPTH_HOST": "0"}, "True"), "0 keeps the CPU"
    assert wanted({**base, "PEPIN_DEPTH_HOST": "1"}, ""), "1 insists, even without torch"


def test_go_sh_survives_a_camera_that_died_before_the_drive_ended(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """go.sh once captured the camera itself and stopped ffmpeg by writing "q" into a fifo; when
    ffmpeg was already dead that write raised SIGPIPE in the shell's builtin printf and killed the
    script before its verdict, so `trip` stopped after the printer (runs 0086, 0088). The clip is
    captured on the board now and go.sh only converts a local file, but the guard stays, and the
    two shell variants below show why: the guarded pattern lives, the old one dies silently."""
    from pathlib import Path

    src = (Path(__file__).resolve().parents[2] / "ros/go.sh").read_text()
    assert "trap '' PIPE" in src
    assert "ffmpeg" in src and "stream" not in src.split("ffmpeg")[1].split("\n")[0], (
        "the laptop must not capture the camera stream itself: the board does (goal_server)"
    )
    setup = (
        'set -uo pipefail; mkfifo "$1/f"; sleep 30 < "$1/f" & R=$!; exec 4>"$1/f"; '
        "kill $R; wait $R 2>/dev/null; "
    )
    variants = {
        "guarded": (
            "trap '' PIPE; " + setup + "/usr/bin/printf q >&4 2>/dev/null; echo alive",
            "alive",
        ),
        "old": (setup + "printf q >&4 2>/dev/null; echo alive", ""),
    }
    for name, (script, expect) in variants.items():
        d = tmp_path / name
        d.mkdir()
        out = subprocess.run(
            ["bash", "-c", script, "_", str(d)], capture_output=True, text=True, timeout=20
        ).stdout.strip()
        assert out == expect, (name, out)
