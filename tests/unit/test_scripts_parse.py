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
        [{"type": "neck", "pan_ticks": 1993, "tilt_ticks": 2311, "age_s": 0.01}]
    )
    read = run_script(port, "read")
    thread.join(timeout=5)
    assert read.returncode == 0, read.stderr[-400:]
    assert asked == [{"cmd": "neck"}]
    # the reference ticks of config/neck.json read as straight ahead (1993 since 2026-09-13)
    assert "pan 1993 ticks (+0.0 deg left)" in read.stdout, read.stdout
    assert "tilt 2311 ticks (+23.8 deg down)" in read.stdout, read.stdout

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
        *PEPIN_SLAM*) printf '%s\n' "${FAKE_SLAM:-}" ;;
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
        *python3*)  # the navigation guard's rclpy pass (ros/tools/nav_goal_running.py)
            printf '%s\n' "${FAKE_GOALS-navigate_to_pose=no navigate_through_poses=no}"
            [ -n "${FAKE_GOALS-x}" ] || return 124 ;;
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
if [ "$1" = get ]; then
    case "$3" in
        sources) printf 'String value is: %s\n' "${FAKE_SOURCES:-lidar}" ;;
        camera_sources) printf 'String value is: %s\n' "${FAKE_CAMERA_SOURCES-depth,contact}" ;;
        *) [ "${FAKE_PUBLISH:-True}" = none ] && exit 2
           printf 'Boolean value is: %s\n' "${FAKE_PUBLISH:-True}" ;;
    esac
fi
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
    # The navigation guard pipes this file into the board's python, so it must be where the
    # script looks for it — the real one, unfaked: what answers is the fake ssh.
    (here / "tools").mkdir(exist_ok=True)
    (here / "tools/nav_goal_running.py").write_text(
        (REPO / "ros/tools/nav_goal_running.py").read_text()
    )
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
    """One command, the two ends of the same switch: the camera joins the tracker's sources —
    as one name, `camera`, since the laptop matches both its scans and sends the pose they
    measured (pepin.measurements) — and its two layers come up on the local AND the global
    costmap. The global costmap lives with the planner, which a split stack (PEPIN_SIDE=board)
    puts on the laptop."""
    code, out, sent = _sensor(
        tmp_path, "camera", "on", FAKE_SIDE="board", FAKE_SOURCES="lidar", FAKE_LAYERS="false"
    )
    assert code == 0, out
    assert "flags set relocalizer sources lidar,camera" in sent
    for layer in ("camera_layer", "contact_layer"):
        assert f"{BOARD} ros2 param set {LOCAL} {layer}.enabled true" in sent
        assert f"{LAPTOP} ros2 param set {GLOBAL} {layer}.enabled true" in sent
    assert not [c for c in sent if "lifecycle" in c], "the camera owns no lifecycle node"
    assert "relocalizer sources lidar -> lidar,camera" in out


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


def test_hard_is_refused_on_an_on_instead_of_stopping_the_driver_it_just_switched_on(  # type: ignore[no-untyped-def]
    tmp_path,
) -> None:
    """`lidar on --hard` used to put the lidar back into the tracker's sources and into both
    costmaps and then deactivate the driver: everything told to use a lidar that no longer
    publishes. --hard is the deep half of an off; on an `on` the line is refused, and nothing
    at all is applied."""
    code, out, sent = _sensor(
        tmp_path, "lidar", "on", "--hard", FAKE_SOURCES="depth,contact", FAKE_LAYERS="false"
    )
    assert code == 2, out
    assert "--hard belongs to 'lidar off'" in out
    assert sent == [], sent
    # and the usage line no longer advertises the form it refuses
    code, out, _ = _sensor(tmp_path, "lidar")
    assert code == 2 and "lidar on|off | lidar off --hard" in out, out


def test_the_lifecycle_half_is_refused_under_a_running_goal_and_under_a_blind_guard(  # type: ignore[no-untyped-def]
    tmp_path,
) -> None:
    """An absent /scan under a moving robot is an experiment nobody chose, so --hard asks the
    navigation actions first (ros/tools/nav_goal_running.py) and refuses before anything at all
    is applied — a half-applied switch is worse than a refusal. An answer the guard could not
    get refuses too: a guard that cannot see does not wave through."""
    goal = "navigate_to_pose=yes navigate_through_poses=no"
    code, out, sent = _sensor(
        tmp_path, "lidar", "off", "--hard", FAKE_GOALS=goal, FAKE_LAYERS="true"
    )
    assert code == 1
    assert "refused: a navigation goal is running (navigate_to_pose" in out
    assert not [c for c in sent if "param set" in c or c.startswith("flags set")], sent

    code, out, sent = _sensor(tmp_path, "lidar", "off", "--hard", FAKE_DRIVER="none")
    assert code == 1 and "did not answer a lifecycle get" in out
    assert not [c for c in sent if "param set" in c], sent

    # A pass that answered "no" for both actions is the only way through.
    code, out, sent = _sensor(tmp_path, "lidar", "off", "--hard", FAKE_LAYERS="true")
    assert code == 0, out
    assert f"{BOARD} ros2 lifecycle set /ldlidar_node deactivate" in sent


def test_a_guard_that_could_not_see_refuses_instead_of_reading_it_as_no_goal(  # type: ignore[no-untyped-def]
    tmp_path,
) -> None:
    """The guard used to run `ros2 topic echo --once` under a 12 s `timeout` and read the 124 as
    "nobody published a status, so no goal is running" — but a loaded board (the only state in
    which a goal IS running) is exactly what makes that query time out: one successful echo
    measured 8.1 s on an idle board. Both shapes of "I could not tell" now refuse: the pass's
    own `?`, and a pass that printed nothing at all."""
    for goals in ("navigate_to_pose=? navigate_through_poses=no", "", "bogus output"):
        code, out, sent = _sensor(
            tmp_path, "lidar", "off", "--hard", FAKE_GOALS=goals, FAKE_LAYERS="true"
        )
        assert code == 1, (goals, out)
        assert "the guard could not read /navigate_to_pose/_action/status" in out, goals
        assert not [c for c in sent if "param set" in c or "lifecycle set" in c], (goals, sent)
    # the pass is one rclpy node piped in from the laptop, not a ros2 CLI call per action
    _, _, sent = _sensor(tmp_path, "lidar", "off", "--hard", FAKE_LAYERS="true")
    assert not [c for c in sent if "topic echo" in c], sent
    piped = [c for c in sent if "python3 -" in c]
    assert len(piped) == 1 and "navigate_to_pose navigate_through_poses" in piped[0], sent
    assert "docker exec -i pepin-ros" in piped[0], piped


def test_a_node_that_does_not_answer_is_reported_not_guessed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A read that failed must never become a write: the tracker's source list is read-modify-
    write, and writing it from an empty read would silently drop the other sensor."""
    code, out, sent = _sensor(tmp_path, "camera", "on", FAKE_SOURCES="none", FAKE_LAYERS="false")
    assert code == 1, out
    assert "relocalizer did not answer about its sources" in out
    # the same refusal covers a tracker that carries no `sources` flag at all, so it names both
    assert "no sources flag in this build, or the node is down" in out
    assert not [c for c in sent if c.startswith("flags set")], sent
    assert [c for c in sent if "param set" in c], "the costmaps still take their half"

    code, out, sent = _sensor(tmp_path, "camera", "on", FAKE_LAYERS="none")
    assert code == 1 and "did not answer a parameter dump: its layers are unchanged" in out
    assert not [c for c in sent if "param set" in c], sent


def test_in_slam_mode_the_costmap_half_switches_and_the_tracker_half_says_it_is_absent(  # type: ignore[no-untyped-def]
    tmp_path,
) -> None:
    """Online SLAM runs no relocalizer at all — RTAB-Map owns the pose — so there is nobody to
    ask about `sources`. The switch used to fall into the "did not answer" path and exit 1 with
    the tracker half unexplained; now the costmap half applies, the tracker half is named as
    absent, and the status is 0: the mode is not a failure."""
    code, out, sent = _sensor(
        tmp_path,
        "camera",
        "off",
        FAKE_SLAM="true",
        FAKE_SOURCES="none",  # the fake flags.sh refuses: nothing may ask it in this mode
        FAKE_LAYERS="true",
    )
    assert code == 0, out
    assert "tracker: none in slam mode" in out
    assert "did not answer about its sources" not in out
    assert not [c for c in sent if c.startswith("flags ")], "no tracker is asked anything"
    for layer in ("camera_layer", "contact_layer"):
        assert f"{BOARD} ros2 param set {LOCAL} {layer}.enabled false" in sent

    code, out, sent = _sensor(tmp_path, "status", FAKE_SLAM="true", FAKE_LAYERS="true")
    assert code == 0, out
    assert "tracker: none in slam mode" in out
    assert f"costmap {LOCAL}:  lidar_layer=on  camera_layer=on  contact_layer=on" in out


def test_switching_the_camera_on_says_what_it_did_to_the_last_two_drives(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The one command that stalled two drives on 2026-09-13 (near-hull marks, 2 cm of
    clearance) says so as it is run, not only in the README."""
    _, out, _ = _sensor(tmp_path, "camera", "on", FAKE_SOURCES="lidar", FAKE_LAYERS="false")
    assert "stalled two drives on 2026-09-13" in out
    _, off, _ = _sensor(
        tmp_path, "camera", "off", FAKE_SOURCES="lidar,depth,contact", FAKE_LAYERS="true"
    )
    assert "stalled two drives" not in off, "an off is the safe direction: no sermon"


def test_a_source_this_script_has_not_heard_of_survives_the_other_sensor_s_switch(  # type: ignore[no-untyped-def]
    tmp_path,
) -> None:
    """The source list is read-modify-write, and the script's SOURCE_ORDER is only an order: a
    name outside it (a source added to pepin.sources before this list hears of it) is carried
    through, never quietly deleted by a switch of a different sensor."""
    code, out, sent = _sensor(
        tmp_path, "camera", "on", FAKE_SOURCES="lidar,sonar,depth", FAKE_LAYERS="false"
    )
    assert code == 0, out
    assert "flags set relocalizer sources lidar,depth,camera,sonar" in sent, sent
    assert "relocalizer sources lidar,depth,sonar -> lidar,depth,camera,sonar" in out
    # and the same on the way out: switching the camera off keeps it too
    _, out, sent = _sensor(
        tmp_path, "camera", "off", FAKE_SOURCES="lidar,sonar,depth,camera", FAKE_LAYERS="true"
    )
    assert "flags set relocalizer sources lidar,depth,sonar" in sent, sent


def test_the_navigation_guard_separates_no_goal_from_could_not_see() -> None:
    """ros/tools/nav_goal_running.py exists for one distinction a `timeout` cannot make: an
    action server that has simply never had a goal (nothing latched to receive) looks exactly
    like a query too slow to discover anything. The subscription's own match, and the graph,
    are what tell them apart — and every case it cannot settle answers `?`, which the caller
    refuses on."""
    import importlib.util

    path = REPO / "ros/tools/nav_goal_running.py"
    spec = importlib.util.spec_from_file_location("nav_goal_running", path)
    assert spec is not None and spec.loader is not None
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)  # imports no rclpy: that lives inside main()
    grace, blind = tool.MATCH_GRACE_S, tool.GRAPH_GRACE_S

    # a status arrived and carries a running goal / carries none
    assert tool.verdict(True, True, 0.1, 0.1) == "yes"
    assert tool.verdict(False, True, 0.1, 0.1) == "no"
    # the server is up and stayed silent: nothing is latched, so no goal has run since it started
    assert tool.verdict(False, False, grace, 0.0) == "no"
    assert tool.verdict(False, False, grace - 0.5, 0.0) == "?", "still in flight is not an answer"
    # no publisher at all: only a graph this node has really seen makes that an answer
    assert tool.verdict(False, False, None, blind) == "no"
    assert tool.verdict(False, False, None, blind - 0.5) == "?"
    assert tool.verdict(False, False, None, 0.0) == "?", "a blind pass never says no"


def test_the_scripts_source_order_is_the_rosters_own() -> None:
    """ros/sensor.sh cannot ask Python for pepin.sources' roster on every run (it is a shell
    script that must also run against a bare checkout), so it carries the order as a literal.
    This is the check that keeps the copy honest — the order it prints is the roster's."""
    from pepin.sources import DEFAULT_SOURCES

    line = next(
        line
        for line in (REPO / "ros/sensor.sh").read_text().splitlines()
        if line.startswith("SOURCE_ORDER=")
    )
    order = line.split('"')[1].split()
    assert order == [s.name for s in DEFAULT_SOURCES], "src/pepin/sources.py owns this order"


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


# ---- ros/restart.sh -----------------------------------------------------------------------------
# One command brings a half back and then checks, in one place, every failure a restart has hidden
# from us. The fakes below are the two hosts: a board answering over ssh and a laptop answering
# `docker`, both scripted by environment variables so one test can break one thing at a time.
TRACKER_LINE = (
    "[python3-6] [INFO] [1789428031.2] [relocalizer]: tracker: scans 227, matched 133, "
    "silenced 4155 returns over 133 scans, fit 0.66/0.47/0.83 at the match; "
    "sources: anchor lidar; watch fit 0.80, last source 0.2 s ago; "
    "map /map (id 239x215@-18.53,-4.38, 0 republications ignored); "
    "flags: rest_lock=on sources=lidar,graph map_topic=map"
)
VSLAM_LOG = "\n".join(
    (
        "[bridge_watch-9] [INFO] [1.0] [bridge_watch]: bridge watch: 10 topics Hz [scan 9.7]; "
        "flow_watch=on; dead routes 0",
        "[depth_stream-3] [INFO] [2.0] [depth_stream]: depth: 9.4 frames/s published (282 through "
        "the net, 0 dropped); lidar_anchor on [a 1.71 b +0.004 on 600 pairs], backend remote",
        "[depth_fusion-5] [INFO] [3.0] [depth_fusion]: fusion: 281 frames (9.3/s, 0 dropped, 0 "
        "unpaired), integrate 7 ms; skipped: low fit 0, at bound 0, self-heals 0",
        "[visual_odometry-7] [INFO] [4.0] [visual_odometry]: vo: 9.1 poses/s from rtabmap, 9.1 "
        "published, 0 dropped",
        "[laptop_localizer-6] [INFO] [5.0] [laptop_localizer]: laptop localizer: 4 candidates "
        "from 27 scans; tracker fit 0.66; skipped: off 0",
        "[rtabmap_frame-8] [INFO] [6.0] [rtabmap_frame]: rtabmap frame: 27 graphs, anchor "
        "(-0.11, +0.02, +1.3 deg) from file, last word (-11.32, +0.71, +132 deg), 6 cm from the "
        "tracker; graph trusted 1.00 over 27 infos; flags: graph_trust=on",
    )
)
FAKE_RESTART_LIB = r"""#!/bin/bash
BOARD="${BOARD:-10.0.0.187}"
log() { printf '%s\n' "$*" >> "$FAKE_LOG"; }
ssh() {
    log "ssh $*"
    case "$*" in
        *"PEPIN_MAP"*) printf '%s\n' "PEPIN_MAP=${FAKE_MAP-/maps/flat3.yaml}" ;;
        *"relocalizer"*) printf '%s\n' "${FAKE_TRACKER-$FAKE_TRACKER_DEFAULT}" ;;
        *Failed*update*rate*) printf '%s\n' "${FAKE_LATE-0}" ;;
        *Extrapolation*) printf '%s\n' "${FAKE_TF_ERRORS-0}" ;;
        *topic_rate.py*) printf '%s\n' "${FAKE_RATE-${*##*topic_rate.py }: 9.1 Hz over 5 s}" ;;
        *"is-active pepin-base"*)
            printf '%s\n%s\n' "${FAKE_BASE-active}" "${FAKE_TORQUE-idle: parked, torque off}" ;;
        *"restart pepin-ros"*) printf 'active\n' ;;
    esac
    return 0
}
docker() {
    log "docker $*"
    case "$*" in
        *inspect*) [ -n "${FAKE_VSLAM-x}" ] && printf '2026-09-14T19:00:00Z\n' || return 1 ;;
        *logs*vslam*) printf '%s\n' "${FAKE_VSLAM-$FAKE_VSLAM_DEFAULT}" ;;
        *"ps -eo"*) printf '%s\n' "${FAKE_PROCS-rtabmap
rgbd_odometry
python3}" ;;
    esac
    return 0
}
"""
FAKE_SUB = """#!/bin/bash
printf '%s %s\\n' "$(basename "$0")" "$*" >> "$FAKE_LOG"
case "$(basename "$0")$*" in
    board.shcensus) printf '%s\\n' "${FAKE_CENSUS-VERDICT: green — every process accounted for}"
                    [ -z "${FAKE_CENSUS_RED-}" ] || exit 1 ;;
    goto.shwhere) printf '%s\\n' "${FAKE_WHERE-at (-11.32, 0.71) facing 132 deg, fit 0.66}"
                  [ -z "${FAKE_WHERE_DOWN-}" ] || exit 1 ;;
    flags.shdrift*) printf '%s' "${FAKE_DRIFT-}" ;;
    foxglove.shcheck) printf '%s\n' "${FAKE_FOXGLOVE-foxglove: 17 checks, none failed}"
                      [ -z "${FAKE_FOXGLOVE_RED-}" ] || exit 1 ;;
    foxglove.shreopen) printf 'foxglove: the app was told to reconnect\n' ;;
esac
exit 0
"""


def _restart(tmp_path, *args, **env):  # type: ignore[no-untyped-def]
    """Run ros/restart.sh against a faked board and laptop; (exit status, output, commands)."""
    import os

    here = tmp_path / "ros"
    here.mkdir(exist_ok=True)
    (here / "lib.sh").write_text(FAKE_RESTART_LIB)
    for name in ("sync.sh", "board.sh", "goto.sh", "laptop.sh", "flags.sh", "foxglove.sh"):
        (here / name).write_text(FAKE_SUB)
        (here / name).chmod(0o755)
    (here / "restart.sh").write_text((REPO / "ros/restart.sh").read_text())
    (here / "maps").mkdir(exist_ok=True)
    log = tmp_path / "log"
    log.write_text("")
    run = subprocess.run(
        ["bash", str(here / "restart.sh"), *args],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "PATH": f"{tmp_path / 'bin'}:{os.environ['PATH']}",
            "FAKE_LOG": str(log),
            # One poll per wait: the fakes answer at once, and a wait that never sees its line
            # must not sit here for the 90 s a real board is given.
            "PEPIN_RESTART_WAIT_S": "0",
            "PEPIN_RESTART_POLL_S": "0",
            "FAKE_TRACKER_DEFAULT": TRACKER_LINE,
            "FAKE_VSLAM_DEFAULT": VSLAM_LOG,
            **env,
        },
    )
    sent = [line.strip() for line in log.read_text().splitlines() if line.strip()]
    return run.returncode, run.stdout + run.stderr, sent


def test_mute_imu_sets_the_publisher_s_own_flag_and_says_what_a_consumer_will_see(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The publisher end of the switch: one live flag on the node that publishes the sensor, no
    restart, no other flag touched — and the line the operator needs, which is what the EKF does
    once the yaw rate stops arriving."""
    code, out, sent = _sensor(tmp_path, "mute", "imu", FAKE_PUBLISH="True")
    assert code == 0, out
    assert "flags set base_bridge imu_publish false" in sent
    assert not [c for c in sent if "param set" in c or "lifecycle" in c], sent
    assert "yaw-rate source" in out and "the wheels" in out


def test_muting_what_is_already_muted_writes_nothing(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Idempotent like the rest of the script: the live value is read first."""
    code, out, sent = _sensor(tmp_path, "mute", "odom", FAKE_PUBLISH="False")
    assert code == 0, out
    assert not [c for c in sent if c.startswith("flags set")], sent
    assert "already so" in out


def test_unmuting_the_camera_restores_the_two_scans_the_table_ships(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The camera's mute is a list flag, not a bool: empty is silence, and an unmute must put
    back exactly what laptop_localizer's FLAGS table defaults to — the test below holds the two
    in step."""
    code, out, sent = _sensor(tmp_path, "unmute", "camera", FAKE_CAMERA_SOURCES="")
    assert code == 0, out
    assert "flags set laptop_localizer camera_sources depth,contact" in sent
    assert "/localization/measurement" in out


def test_the_scripts_camera_scans_are_the_localizer_s_own_default() -> None:
    """ros/sensor.sh cannot ask Python what an unmuted camera is (it is a shell script on the
    laptop), so the value is written down — and this test fails the day the table moves."""
    from pepin.flags import load_table

    flags = load_table(REPO / "ros/pepin_bringup/pepin_bringup/laptop_localizer.py")
    default = ",".join(flags["camera_sources"])
    assert [
        line
        for line in (REPO / "ros/sensor.sh").read_text().splitlines()
        if line.startswith(f'CAMERA_SCANS="{default}"')
    ], f"ros/sensor.sh must restore {default}"


def test_muting_the_lidar_is_the_consumer_set_no_node_of_ours_publishes_it(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Our own node in the lidar's chain is laser_filters' scan_filter and it has no flag of
    ours; a relay on the board that could drop /scan is refused by CLAUDE.md rule 20. So the
    mute is the documented pair — the tracker's sources without `lidar`, lidar_layer off on both
    costmaps — and the driver is left running (`lidar off --hard` is the real absence)."""
    code, out, sent = _sensor(
        tmp_path, "mute", "lidar", FAKE_SOURCES="lidar,camera", FAKE_LAYERS="true"
    )
    assert code == 0, out
    assert "flags set relocalizer sources camera" in sent
    assert f"{BOARD} ros2 param set {LOCAL} lidar_layer.enabled false" in sent
    assert not [c for c in sent if "lifecycle set" in c], "a mute never stops the driver"
    assert "scan_filter" in out and "--hard" in out


def test_a_publisher_that_does_not_answer_is_reported_and_the_run_goes_red(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A node that is down must not read as an unmuted sensor."""
    code, out, sent = _sensor(tmp_path, "mute", "vo", FAKE_PUBLISH="none")
    assert code == 1, out
    assert not [c for c in sent if c.startswith("flags set")], sent
    assert "did not answer" in out


def test_status_lists_every_sensor_s_mute_state(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """One reading of the whole publisher end: each sensor's own live flag, and the lidar
    pointed at the two consumer settings that are its mute."""
    code, out, _ = _sensor(
        tmp_path, "status", FAKE_PUBLISH="False", FAKE_CAMERA_SOURCES="", FAKE_LAYERS="true"
    )
    assert code == 0, out
    for sensor in ("imu", "odom", "vo", "graph", "camera"):
        assert f"{sensor}: MUTED" in out, out
    assert "lidar: see the tracker sources" in out


def test_an_unmuted_stack_says_so_sensor_by_sensor(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The other half of the reading: the shipping state is named, not left blank."""
    _, out, _ = _sensor(tmp_path, "status", FAKE_PUBLISH="True", FAKE_LAYERS="true")
    assert "imu: on (base_bridge imu_publish=True)" in out
    assert "camera: on (laptop_localizer camera_sources=depth,contact)" in out


def test_restart_sh_parses_and_never_drives() -> None:
    result = subprocess.run(
        ["bash", "-n", str(REPO / "ros/restart.sh")], capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    code = "\n".join(
        line
        for line in (REPO / "ros/restart.sh").read_text().splitlines()
        if not line.lstrip().startswith("#")
    )
    assert "cmd_vel" not in code and "goto_ros" not in code, "a restart is not a drive"
    # One rate probe per topic and no `ros2` CLI on the board: the CLI is seconds of A53 per call.
    assert code.count("topic_rate.py") == 1 and "ros2 topic" not in code
    for half in ("board", "laptop", "both"):
        assert f"{half} " in code or f"{half})" in code


def test_the_laptop_half_is_seeded_with_the_map_the_board_serves(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """The camera half's fused volume is snapped to the lattice of the map it is seeded with, so
    the seed is read from the board's own /etc/default/pepin-ros — never guessed, never a default.
    The neck owns base_link -> camera_link, so --neck always goes with it."""
    code, out, sent = _restart(
        tmp_path, "laptop", "--no-check", FAKE_MAP="/maps/flat3_straight.yaml"
    )
    assert code == 0, out
    assert "laptop.sh start" in sent
    assert "laptop.sh vslam --neck --seed-map=/maps/flat3_straight.yaml" in sent
    assert not [c for c in sent if "--fresh" in c], "no --fresh without --fresh-graph"
    assert not [c for c in sent if c.startswith("ssh") and "restart pepin-ros" in c], sent
    assert "checks skipped" in out


def test_a_board_without_a_map_refuses_instead_of_seeding_the_wrong_one(tmp_path) -> None:  # type: ignore[no-untyped-def]
    code, out, sent = _restart(tmp_path, "laptop", "--no-check", FAKE_MAP="")
    assert code == 0, out  # --no-check: the restart failed, its reason printed, nothing checked
    assert "the board does not say which map it serves" in out
    assert not [c for c in sent if c.startswith("laptop.sh")], "nothing started on a guess"


def test_fresh_graph_empties_the_database_and_takes_the_anchor_of_that_map_with_it(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    """The anchor ties one map to one graph database (pepin.anchors). A fresh database beside a
    kept anchor speaks in the previous database's frame, so --fresh-graph removes both, and says
    which file it removed."""
    anchor = tmp_path / "ros/maps/239x215_-18.53_-4.38.graph_anchor.json"
    anchor.parent.mkdir(parents=True, exist_ok=True)
    anchor.write_text("{}")
    (tmp_path / "bin").mkdir(exist_ok=True)
    (tmp_path / "bin/uv").write_text(f'#!/bin/bash\nprintf "%s\\n" "{anchor}"\n')  # pepin.anchors
    (tmp_path / "bin/uv").chmod(0o755)
    code, out, sent = _restart(tmp_path, "laptop", "--no-check", "--fresh-graph")
    assert code == 0, out
    assert "laptop.sh vslam --neck --seed-map=/maps/flat3.yaml --fresh" in sent
    assert not anchor.exists(), out
    assert str(anchor) in out and "map <-> database" in out


def test_fresh_graph_is_refused_on_the_board_half_that_owns_neither(tmp_path) -> None:  # type: ignore[no-untyped-def]
    code, out, sent = _restart(tmp_path, "board", "--fresh-graph")
    assert code == 2, out
    assert sent == [], "refused before a host is touched"


def test_both_brings_the_board_back_first_and_checks_only_once_the_laptop_feeds_it(
    tmp_path,
) -> None:  # type: ignore[no-untyped-def]
    """Order matters twice: the board restarts first, and NOTHING is checked until the laptop is
    up — /depth_scan and /vo are fed by the laptop, so checking the board first would fail two
    checks by construction."""
    code, out, sent = _restart(tmp_path, "both")
    assert code == 0, out
    order = [i for i, c in enumerate(sent) if "restart pepin-ros" in c or c.startswith("laptop.sh")]
    assert sent[order[0]].endswith("restart pepin-ros && sleep 8 && systemctl is-active pepin-ros")
    assert sent[order[1]] == "laptop.sh start"
    first_rate = next(i for i, c in enumerate(sent) if "topic_rate.py" in c)
    assert first_rate > order[-1], "the board is asked about the laptop's topics after it is up"
    for number in (
        "1.1",
        "1.2",
        "1.3",
        "1.4",
        "1.5",
        "1.6",
        "1.7",
        "1.8",
        "2.1",
        "2.8",
        "2.9",
        "3.1",
    ):
        assert f"PASS {number}" in out, out
    assert "green: " in out and "none failed" in out


def test_the_operators_window_is_checked_and_the_app_is_reconnected_last(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A restart kills the bridge and with it the desktop app's socket: the panels stay on screen,
    empty, until a client re-attaches. So `ros/foxglove.sh check` is check 2.9 and `reopen` is the
    last thing the restart does — and a bridge that does not answer turns the run red like any
    other check."""
    code, out, sent = _restart(tmp_path, "both")
    assert code == 0, out
    assert "PASS 2.9" in out and "none failed" in out
    calls = [c for c in sent if c.startswith("foxglove.sh")]
    assert calls == ["foxglove.sh check", "foxglove.sh reopen"]
    assert out.index("== foxglove ==") < out.index("== verdict ==")

    code, out, sent = _restart(
        tmp_path,
        "both",
        FAKE_FOXGLOVE_RED="1",
        FAKE_FOXGLOVE="FAIL fg.3   channels: 0 advertised\nfoxglove: 1 of 17 checks failed",
    )
    assert code == 1, out
    assert "FAIL 2.9" in out and "1 of 17 checks failed" in out
    assert "FAIL fg.3" in out, "the failing lines of the check are shown under it"
    assert "PASS 3.1" in out, "the checks after it still ran"

    # The board alone never touches the laptop's app.
    code, out, sent = _restart(tmp_path, "board")
    assert code == 0, out
    assert not [c for c in sent if c.startswith("foxglove.sh")]


def test_every_check_runs_even_when_the_first_ones_fail_and_the_run_goes_red(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A check that fails is one line and never the end of the run: the point of the script is
    the whole picture. The three below are the three that cost us a session each."""
    code, out, _ = _restart(
        tmp_path,
        "both",
        FAKE_TRACKER="",  # the tracker never reported
        FAKE_TORQUE="arming: torque on",  # the wheels left armed
        FAKE_VSLAM=VSLAM_LOG.replace("dead routes 0", "DEAD ROUTES 2 [/scan /odom]"),
    )
    assert code == 1, out
    assert "FAIL 1.2" in out and "no report line" in out
    assert "FAIL 1.8" in out and "still armed" in out
    assert "FAIL 2.1" in out and "DEAD ROUTES 2" in out
    assert "PASS 1.4" in out and "PASS 2.8" in out, "the checks after a failure still ran"
    assert "red: 3 of " in out


@pytest.mark.parametrize(
    ("broken", "number", "why"),
    [
        ({"FAKE_CENSUS_RED": "1"}, "1.1", "census"),
        ({"FAKE_LATE": "7"}, "1.4", "Failed to meet update rate"),
        ({"FAKE_TF_ERRORS": "3"}, "1.5", "error(s)"),
        # a log that cannot be read is never counted as zero errors
        ({"FAKE_LATE": "ssh: connect to host: No route"}, "1.4", "could not be read"),
        ({"FAKE_RATE": "/vo: not advertised"}, "1.6", "does not reach the board"),
        ({"FAKE_WHERE_DOWN": "1"}, "1.3", "did not answer"),
        ({"FAKE_PROCS": "python3"}, "2.6", "no rtabmap process"),
    ],
)
def test_each_known_failure_is_named_on_its_own_line(tmp_path, broken, number, why) -> None:  # type: ignore[no-untyped-def]
    code, out, _ = _restart(tmp_path, "both", **broken)
    assert code == 1, out
    assert f"FAIL {number}" in out and why in out, out


def test_a_thin_report_line_fails_the_node_it_belongs_to_not_the_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """Rates and counters are read from the nodes' own report lines; a number below the floor is
    that node's failure, with the number in the line so it can be read at a glance."""
    code, out, _ = _restart(
        tmp_path,
        "laptop",
        FAKE_VSLAM=VSLAM_LOG.replace("9.4 frames/s", "1.2 frames/s")
        .replace("at bound 0", "at bound 14")
        .replace("over 27 infos", "over 0 infos"),
    )
    assert code == 1, out
    assert "FAIL 2.2" in out and "1.2 frames/s" in out
    assert "FAIL 2.3" in out and "14 frames refused at bound" in out
    assert "FAIL 2.7" in out and "trust is deaf" in out


def test_a_flag_off_its_default_is_seen_but_never_fails_the_run(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """A restart puts every flag back to its default, so a flag that is NOT its default was set
    on purpose — legitimate, and the one thing a restart silently throws away. It is shown, and
    the run stays green."""
    code, out, sent = _restart(
        tmp_path, "laptop", FAKE_DRIFT="relocalizer/sources lidar,camera (default lidar)\n"
    )
    assert code == 0, out
    assert "WARN 3.1" in out and "relocalizer/sources lidar,camera (default lidar)" in out
    assert "flags.sh drift laptop" in sent
    assert "green: " in out


def _uncommented(rel: str) -> str:
    """A script's lines with the whole-line comments dropped: what it DOES, not what it says."""
    return "\n".join(
        line for line in (REPO / rel).read_text().splitlines() if not line.lstrip().startswith("#")
    )


def test_there_is_one_way_to_stop_a_container_and_it_is_gentle() -> None:
    """Every stop of a container on this laptop goes through ros/lib.sh's helper, which sends the
    container's stop signal (SIGINT, what ros2 launch answers by shutting its nodes down) and
    waits the repo's one window. A bare `docker stop`, `docker kill` or `docker rm -f` next to it
    would be the SIGKILL mid-write that made ros/maps/rtabmap.db malformed on 2026-09-13."""
    from pepin.deployment import CONTAINER_STOP_TIMEOUT_S

    lib = _uncommented("ros/lib.sh")
    assert 'docker stop -t "$PEPIN_STOP_TIMEOUT_S"' in lib
    assert f'PEPIN_STOP_TIMEOUT_S="${{PEPIN_STOP_TIMEOUT_S:-{CONTAINER_STOP_TIMEOUT_S}}}"' in lib, (
        "ros/lib.sh and pepin.deployment.CONTAINER_STOP_TIMEOUT_S must be the same number"
    )
    # pepin_remove_container stops before it removes, so the rm is never what ends the process.
    remove = lib[lib.index("pepin_remove_container()") :]
    assert remove.index("pepin_stop_container") < remove.index("docker rm -f")

    for script in sorted(p.name for p in (REPO / "ros").glob("*.sh")):
        code = _uncommented(f"ros/{script}")
        for verb in ("docker stop", "docker kill", "docker rm -f"):
            if verb not in code:
                continue
            assert script in {"lib.sh", "thin.sh"}, (
                f"ros/{script} runs `{verb}` itself; use pepin_stop_container /"
                " pepin_remove_container from ros/lib.sh"
            )
    # thin.sh's one `docker rm -f` is on the BOARD over ssh, after its unit's own gentle ExecStop,
    # and the container it names is the bridge sidecar — nothing of ours writes a file in it.
    thin = _uncommented("ros/thin.sh")
    assert "systemctl disable --now pepin-bridge" in thin and "docker rm -f zenoh-bridge" in thin


def test_the_board_s_containers_answer_sigint_and_the_unit_waits_for_them() -> None:
    """`systemctl restart pepin-ros` must not be a SIGKILL. The container carries SIGINT as its
    stop signal (the image's STOPSIGNAL and the run line's flag), the unit's ExecStop gives it the
    repo's window, and TimeoutStopSec stays above that window — when it expires systemd stops
    waiting for ExecStop and kills the cgroup the `docker run` client sits in."""
    import re

    from pepin.deployment import CONTAINER_STOP_TIMEOUT_S

    assert "STOPSIGNAL SIGINT" in (REPO / "ros/Dockerfile").read_text()
    run = _uncommented("ros/run.sh")
    assert "--stop-signal SIGINT" in run

    unit = (REPO / "board/pepin-ros.service").read_text()
    stop = re.search(r"^ExecStop=.*docker stop -t (\d+) pepin-ros", unit, re.M)
    assert stop is not None and int(stop.group(1)) == CONTAINER_STOP_TIMEOUT_S
    timeout = re.search(r"^TimeoutStopSec=(\d+)", unit, re.M)
    assert timeout is not None and int(timeout.group(1)) > int(stop.group(1)), (
        "systemd must outwait the docker stop it runs, or it kills the client mid-stop"
    )
    assert re.search(r"^KillMode=mixed", unit, re.M), (
        "the nodes are docker's children, not this cgroup's"
    )
