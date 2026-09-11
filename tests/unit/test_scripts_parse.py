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


# ros/sensor.sh talks to exactly three things: ``ssh`` and ``docker`` (both functions, from
# ros/lib.sh) and ros/flags.sh beside it. A copy of the script in a directory whose lib.sh and
# flags.sh are fakes therefore runs whole, with no robot and no ros2, and every command it would
# have sent is a line in a log — which is what the tests below read.
FAKE_LIB = r"""#!/bin/bash
BOARD="${BOARD:-${PEPIN_HOST:-10.0.0.187}}"
log() { printf '%s\n' "$*" >> "$FAKE_LOG"; }
answer() {  # the canned reply of the board or the laptop, by what was asked of it
    case "$1" in
        *PEPIN_SIDE*) printf '%s\n' "${FAKE_SIDE:-}" ;;
        *logs*pepin-ros*) printf '%s\n' "${FAKE_TRACKER_LINE:-}" ;;
        *logs*pepin-vslam*) printf '%s\n' "${FAKE_DEPTH_LINE:-}" "${FAKE_CAMERA_LINE:-}" ;;
        *"lifecycle set"*)
            case "$1" in
                *deactivate*) printf 'inactive\n' > "$FAKE_DRIVER_STATE" ;;
                *activate*) printf 'active\n' > "$FAKE_DRIVER_STATE" ;;
            esac ;;
        *"lifecycle get"*)
            [ "${FAKE_DRIVER:-active}" = none ] && return 1
            printf '%s [3]\n' \
                "$(cat "$FAKE_DRIVER_STATE" 2>/dev/null || printf '%s' "${FAKE_DRIVER:-active}")" ;;
        *_action/status*) printf '%s\n' "${FAKE_GOAL:-}"; [ -n "${FAKE_GOAL:-}" ] || return 124 ;;
        *"param dump"*)
            [ "${FAKE_LAYERS:-false}" = none ] && return 1
            printf '%s\n' "${1##* }:" "  ros__parameters:"
            for layer in lidar_layer camera_layer contact_layer inflation_layer; do
                printf '    %s:\n      enabled: %s\n' "$layer" "${FAKE_LAYERS:-false}"
            done ;;
    esac
    return 0
}
ssh() { log "ssh $*"; answer "$*"; }
docker() { log "docker $*"; answer "$*"; }
"""

FAKE_FLAGS = r"""#!/bin/bash
printf 'flags %s\n' "$*" >> "$FAKE_LOG"
[ "${FAKE_SOURCES:-lidar}" = none ] && exit 2
[ "$1" = get ] && printf 'String value is: %s\n' "${FAKE_SOURCES:-lidar}"
exit 0
"""


def _sensor(tmp_path, *args, **env):  # type: ignore[no-untyped-def]
    """Run ros/sensor.sh against the fakes; returns (exit status, stdout, the commands sent)."""
    import os

    here = tmp_path / "ros"
    here.mkdir(exist_ok=True)
    for name, text in (("lib.sh", FAKE_LIB), ("flags.sh", FAKE_FLAGS)):
        (here / name).write_text(text)
        (here / name).chmod(0o755)
    (here / "sensor.sh").write_text((REPO / "ros/sensor.sh").read_text())
    log = tmp_path / "log"
    log.write_text("")
    run = subprocess.run(
        ["bash", str(here / "sensor.sh"), *args],
        capture_output=True,
        text=True,
        timeout=30,
        env={
            **os.environ,
            "FAKE_LOG": str(log),
            "FAKE_DRIVER_STATE": str(tmp_path / "driver"),
            **env,
        },
    )
    sent = [line.strip() for line in log.read_text().splitlines() if line.strip()]
    return run.returncode, run.stdout + run.stderr, sent


BOARD = "ssh root@10.0.0.187 docker exec pepin-ros /pepin_entrypoint.sh"
LAPTOP = "docker exec pepin-laptop /pepin_entrypoint.sh"
LOCAL, GLOBAL = "/local_costmap/local_costmap", "/global_costmap/global_costmap"


def test_sensor_sh_parses() -> None:
    result = subprocess.run(
        ["bash", "-n", str(REPO / "ros/sensor.sh")], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    code = "\n".join(
        line
        for line in (REPO / "ros/sensor.sh").read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "cmd_vel" not in code, "a sensor switch never commands a velocity"
    assert "systemctl" not in code and "docker restart" not in code, "nor restarts anything"


def test_camera_on_sets_the_tracker_s_sources_and_both_costmaps_layers(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """One command, the two ends of the same switch: the camera's two scans join the tracker's
    sources and its two layers come up on the local AND the global costmap. The global costmap
    lives with the planner, which a split stack (PEPIN_SIDE=board) puts on the laptop."""
    code, out, sent = _sensor(
        tmp_path, "camera", "on", FAKE_SIDE="board", FAKE_SOURCES="lidar", FAKE_LAYERS="false"
    )
    assert code == 0, out
    assert "flags set relocalizer sources lidar,depth,contact" in sent
    for layer in ("camera_layer", "contact_layer"):
        assert f"{BOARD} ros2 param set {LOCAL} {layer}.enabled true" in sent
        assert f"{LAPTOP} ros2 param set {GLOBAL} {layer}.enabled true" in sent
    assert not [c for c in sent if "lifecycle" in c], "the camera owns no lifecycle node"
    assert "relocalizer sources lidar -> lidar,depth,contact" in out


def test_a_whole_stack_keeps_both_costmaps_on_the_board(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Without the split (PEPIN_SIDE unset) the planner runs on the board too, so the global
    costmap is reached there and never through a laptop container that does not exist."""
    _, out, sent = _sensor(tmp_path, "camera", "on", FAKE_SIDE="", FAKE_LAYERS="false")
    assert f"{BOARD} ros2 param set {GLOBAL} camera_layer.enabled true" in sent
    assert not [c for c in sent if "pepin-laptop" in c], out


def test_a_switch_that_changes_nothing_sends_no_set_and_says_so(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Idempotent: the layers are read first and only what differs is written."""
    code, out, sent = _sensor(
        tmp_path, "camera", "off", FAKE_SIDE="board", FAKE_SOURCES="lidar", FAKE_LAYERS="false"
    )
    assert code == 0, out
    assert not [c for c in sent if "param set" in c], sent
    assert not [c for c in sent if c.startswith("flags set")], sent
    assert "already so" in out
    # One whole dump per costmap, never one `ros2 param get` per layer: a ros2 CLI call is a
    # Python node that costs about 10 s on the board (measured 2026-09-11 at load 9.5).
    assert len([c for c in sent if "param dump" in c]) == 2
    assert not [c for c in sent if "param get" in c], sent


def test_lidar_off_leaves_the_driver_alone_and_hard_off_deactivates_it(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The soft off is an ignored scan; --hard is the absence of one. Only --hard touches the
    lifecycle node, and `lidar on` brings it back so a demo cannot end with /scan stopped."""
    _, out, soft = _sensor(
        tmp_path, "lidar", "off", FAKE_SOURCES="lidar,depth,contact", FAKE_LAYERS="true"
    )
    assert not [c for c in soft if "lifecycle" in c], out
    assert "flags set relocalizer sources depth,contact" in soft
    assert f"{BOARD} ros2 param set {LOCAL} lidar_layer.enabled false" in soft

    _, out, hard = _sensor(
        tmp_path, "lidar", "off", "--hard", FAKE_SOURCES="lidar,depth,contact", FAKE_LAYERS="true"
    )
    assert f"{BOARD} ros2 lifecycle set /ldlidar_node deactivate" in hard
    assert "/ldlidar_node active -> inactive" in out

    _, out, back = _sensor(
        tmp_path,
        "lidar",
        "on",
        FAKE_SOURCES="depth,contact",
        FAKE_LAYERS="false",
        FAKE_DRIVER="inactive",
    )
    assert f"{BOARD} ros2 lifecycle set /ldlidar_node activate" in back
    assert "flags set relocalizer sources lidar,depth,contact" in back
    assert "/ldlidar_node inactive -> active" in out


def test_the_lifecycle_half_is_refused_under_a_running_goal_and_under_a_blind_guard(  # type: ignore[no-untyped-def]
    tmp_path,
) -> None:
    """An absent /scan under a moving robot is an experiment nobody chose, so --hard reads the
    navigation action's latched status first (ros/tools/turn_full.py's check) and refuses before
    anything at all is applied — a half-applied switch is worse than a refusal. A status the
    guard could not read refuses too: a guard that cannot see does not wave through."""
    code, out, sent = _sensor(
        tmp_path, "lidar", "off", "--hard", FAKE_GOAL="  status: 2", FAKE_LAYERS="true"
    )
    assert code == 1
    assert "refused: a navigation goal is running (navigate_to_pose" in out
    assert not [c for c in sent if "param set" in c or c.startswith("flags set")], sent

    code, out, sent = _sensor(tmp_path, "lidar", "off", "--hard", FAKE_DRIVER="none")
    assert code == 1 and "did not answer a lifecycle get" in out
    assert not [c for c in sent if "param set" in c], sent

    # A finished goal (status 4) is not a running one.
    code, out, sent = _sensor(
        tmp_path, "lidar", "off", "--hard", FAKE_GOAL="  status: 4", FAKE_LAYERS="true"
    )
    assert code == 0, out
    assert f"{BOARD} ros2 lifecycle set /ldlidar_node deactivate" in sent


def test_a_node_that_does_not_answer_is_reported_not_guessed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A read that failed must never become a write: the tracker's source list is read-modify-
    write, and writing it from an empty read would silently drop the other sensor."""
    code, out, sent = _sensor(tmp_path, "camera", "on", FAKE_SOURCES="none", FAKE_LAYERS="false")
    assert code == 1, out
    assert "relocalizer did not answer" in out
    assert not [c for c in sent if c.startswith("flags set")], sent
    assert [c for c in sent if "param set" in c], "the costmaps still take their half"

    code, out, sent = _sensor(tmp_path, "camera", "on", FAKE_LAYERS="none")
    assert code == 1 and "did not answer a parameter dump: its layers are unchanged" in out
    assert not [c for c in sent if "param set" in c], sent


def test_status_reads_every_end_of_the_switch(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """One picture, and the cheap half of it is free: the tracker's own sources come out of the
    report line it already prints every 30 s (pepin.flags renders `sources=...` into it), so
    status asks ROS for nothing but the two costmap dumps and the driver's lifecycle state."""
    tracker = "[relocalizer-5] [INFO] [1.0] [relocalizer]: tracker: released 42 of 42 scans; "
    tracker += "watch fit 0.81; flags: rest_lock=on sources=lidar,contact fusion=on"
    camera = "[contact_scan-3] [INFO] [2.0] [contact_scan]: contact: 2.9 scans/s published"
    depth = "[depth_stream-2] [INFO] [2.0] [depth_stream]: depth: 3.1 frames/s published"
    code, out, sent = _sensor(
        tmp_path,
        "status",
        FAKE_SIDE="board",
        FAKE_LAYERS="true",
        FAKE_TRACKER_LINE=tracker,
        FAKE_CAMERA_LINE=camera,
        FAKE_DEPTH_LINE=depth,
    )
    assert code == 0, out
    assert "tracker sources: lidar,contact" in out
    assert f"costmap {LOCAL}:  lidar_layer=on  camera_layer=on  contact_layer=on" in out
    assert "lidar driver /ldlidar_node: active" in out
    for line in (tracker, depth, camera):
        assert line in out, "each node's own word, printed whole"
    assert not [c for c in sent if "param get" in c or "topic echo" in c], sent
    assert [c for c in sent if "param dump" in c and GLOBAL in c and c.startswith(LAPTOP)]
    assert [c for c in sent if "docker logs --since 90s pepin-vslam" in c], sent


def test_status_says_so_when_a_node_printed_no_report(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A missing report line is reported as missing, never as an empty source list."""
    code, out, _ = _sensor(tmp_path, "status", FAKE_LAYERS="true")
    assert code == 0
    assert "tracker sources: ? (relocalizer printed no report in 90 s" in out
    assert "(nothing)" in out
