"""Every entry point answers --help: its imports resolve and its parser builds.

The servo bench tools (jog, calibrate_neck, scan_bus, setup_motor_id) import
lerobot, which pulls torch; they are left out to keep the unit tier fast.
"""

import subprocess
import sys
from pathlib import Path
from typing import Any

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


def test_neck_sh_asks_the_base_server_and_prints_ticks_and_degrees() -> None:
    """ros/neck.sh is the neck's hand: one JSON line to the base server's port, the answer in
    ticks and degrees. A fake server stands in for the board here — the script must pick its
    reply out of the state lines the real port also broadcasts, and fail on a refusal."""
    import json
    import os
    import socket
    import threading

    assert subprocess.run(["bash", "-n", str(REPO / "ros/neck.sh")], timeout=20).returncode == 0
    usage = (REPO / "ros/neck.sh").read_text()
    for line in ("neck.sh read", "neck.sh home", "neck.sh goto PAN TILT", "neck.sh hold PAN TILT"):
        assert line in usage
    assert '"cmd": "neck_home"' in usage and '"cmd": "neck_goto"' in usage

    def board(answers: list[dict[str, Any]]) -> tuple[int, list[dict[str, Any]], threading.Thread]:
        """A one-client fake of the base server: collects what it is asked, then answers."""
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        asked: list[dict[str, Any]] = []

        def run() -> None:
            conn, _ = listener.accept()
            with conn:
                listener.close()
                asked.append(json.loads(conn.recv(4096).split(b"\n")[0]))
                conn.sendall(b'{"type":"state","x":0.0}\n')  # 20 Hz of noise around the answer
                for answer in answers:
                    conn.sendall((json.dumps(answer) + "\n").encode())

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return listener.getsockname()[1], asked, thread

    def run_script(port: int, *args: str) -> subprocess.CompletedProcess[str]:
        env = os.environ | {"PEPIN_HOST": "127.0.0.1", "PEPIN_BASE_PORT": str(port)}
        return subprocess.run(
            ["bash", str(REPO / "ros/neck.sh"), *args],
            capture_output=True,
            text=True,
            timeout=60,
            env=env,
        )

    port, asked, thread = board(
        [{"type": "neck", "pan_ticks": 2021, "tilt_ticks": 2311, "age_s": 0.01}]
    )
    read = run_script(port, "read")
    thread.join(timeout=5)
    assert read.returncode == 0, read.stderr[-400:]
    assert asked == [{"cmd": "neck"}]
    assert "pan 2021 ticks (+0.0 deg left)" in read.stdout, read.stdout
    assert "tilt 2311 ticks (+26.0 deg down)" in read.stdout, read.stdout

    port, asked, thread = board(
        [{"type": "neck_goto", "pan_ticks": 2100, "tilt_ticks": 2311, "reached": True, "ms": 840.0}]
    )
    moved = run_script(port, "goto", "2100", "2311")
    thread.join(timeout=5)
    assert moved.returncode == 0, moved.stderr[-400:]
    assert asked == [{"cmd": "neck_goto", "pan_ticks": 2100, "tilt_ticks": 2311, "hold": False}]
    assert "reached in 840 ms" in moved.stdout, moved.stdout

    port, asked, thread = board(
        [{"type": "neck_goto", "reached": False, "error": "neck target 4000 is outside its limits"}]
    )
    refused = run_script(port, "hold", "4000", "2311")
    thread.join(timeout=5)
    assert refused.returncode == 1, refused.stdout
    assert asked[0]["hold"] is True
    assert "outside its limits" in refused.stderr
