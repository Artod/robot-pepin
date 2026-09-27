"""The kick's wait (ros/kick_ready.awk, ros/thin.sh kick, ros/laptop.sh kick): the ready line it
reports is the NEW process's — after the launch's exit line of the signalled pid and the start of
its successor under the same tag — never an older line that happens to match."""
# ruff: noqa: E501 — the log below is written as the launch writes it, one line per record.

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
AWK = REPO / "ros/kick_ready.awk"
KICKED = "2026-09-27T10:00:05.100000000Z"

# The launch's own lines around one kick of the recorder (the shape of the board's log of
# 2026-09-23, scratch/lidar/data/board_pepin_ros_20260923_234052_baseline.log), docker logs -t.
KICK_LOG = """\
2026-09-27T10:00:04.900000000Z [python3-7] [INFO] [1790206544.1] [run_recorder]: run recorder ready: the OLD one
2026-09-27T10:00:05.300000000Z [python3-7] rclpy._rclpy_pybind11.RCLError: failed to shutdown
2026-09-27T10:00:05.400000000Z [ERROR] [python3-9]: process has died [pid 51, exit code 1, cmd 'python3 -m x']
2026-09-27T10:00:05.450000000Z [ERROR] [python3-7]: process has died [pid 47, exit code 1, cmd 'python3 -m pepin_bringup.run_recorder'].
2026-09-27T10:00:05.500000000Z [python3-7] [INFO] [1790206544.2] [run_recorder]: run recorder ready: a late flush of the OLD one
2026-09-27T10:00:07.450000000Z [INFO] [python3-7]: process started with pid [836]
2026-09-27T10:00:08.000000000Z [python3-8] [INFO] [1790206548.0] [goal_server]: run recorder ready: another process
2026-09-27T10:00:09.000000000Z [python3-7] [INFO] [1790206549.0] [run_recorder]: run recorder waiting for /scan
2026-09-27T10:00:11.300000000Z [python3-7] [INFO] [1790206551.3] [run_recorder]: run recorder ready: tapes in /maps/rec
"""


def ready(log: str, old: str = "47", line: str = "run recorder ready", kicked: str = KICKED) -> str:
    """The matcher's one output line for this log."""
    run = subprocess.run(
        [
            "awk",
            "-v",
            "name=run_recorder",
            "-v",
            f"old={old}",
            "-v",
            f"line={line}",
            "-v",
            f"kicked={kicked}",
            "-f",
            str(AWK),
        ],
        input=log,
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    return run.stdout.strip()


def test_the_ready_line_is_the_successor_s_under_the_signalled_pid_s_tag() -> None:
    assert ready(KICK_LOG) == (
        "ready\trun_recorder kicked at 10:00:05.1, ready at 10:00:11.3 UTC, 6.2 s,"
        " pid 47 -> 836: run recorder ready: tapes in /maps/rec"
    )


def test_each_stage_says_what_it_still_waits_for() -> None:
    lines = KICK_LOG.splitlines(keepends=True)
    assert ready("".join(lines[:3])).startswith("wait\tpid 47 has not exited since the SIGINT")
    assert ready("".join(lines[:5])).startswith("wait\tpython3-7 (pid 47) exited; no successor")
    assert ready("".join(lines[:8])) == (
        "wait\tpid 836 (python3-7) started but has not printed 'run recorder ready' yet"
    )


def test_a_successor_that_dies_before_its_ready_line_is_waited_through() -> None:
    lines = KICK_LOG.splitlines(keepends=True)
    died = (
        "2026-09-27T10:00:08.500000000Z [ERROR] [python3-7]: process has died [pid 836, exit"
        " code 1, cmd 'x'].\n"
    )
    assert "its successor died 1 time(s)" in ready("".join(lines[:7]) + died)
    again = (
        "2026-09-27T10:00:10.600000000Z [INFO] [python3-7]: process started with pid [900]\n"
        "2026-09-27T10:00:12.100000000Z [python3-7] [INFO] [1.0] [run_recorder]: run recorder"
        " ready: second try\n"
    )
    out = ready("".join(lines[:7]) + died + again)
    assert out.startswith("ready\t") and "7.0 s, pid 47 -> 900: run recorder ready: second" in out


def test_any_of_several_signalled_pids_names_the_tag_and_a_log_without_times_still_answers() -> (
    None
):
    assert "pid 47 -> 836" in ready(KICK_LOG, old="12 47")
    bare = "\n".join(line.split(" ", 1)[1] for line in KICK_LOG.splitlines()) + "\n"
    assert ready(bare, kicked="") == (
        "ready\trun_recorder kicked at ?, ready at ? UTC, ? s,"
        " pid 47 -> 836: run recorder ready: tapes in /maps/rec"
    )


def test_a_ready_line_across_midnight_counts_seconds_not_minus_a_day() -> None:
    log = (
        "2026-09-28T00:00:00.100000000Z [ERROR] [python3-7]: process has died [pid 47, x].\n"
        "2026-09-28T00:00:02.200000000Z [INFO] [python3-7]: process started with pid [836]\n"
        "2026-09-28T00:00:03.000000000Z [python3-7] [INFO] [1.0] [run_recorder]: run recorder"
        " ready: after midnight\n"
    )
    assert ", 3.1 s," in ready(log, kicked="2026-09-27T23:59:59.900000000Z")


# ---- the two kick scripts against fakes (slow: whole scripts, fakes on PATH) --------------------

FAKE_SSH = """#!/bin/bash
# ssh as the remote shell sees it: the arguments after the host joined into ONE command line.
printf 'ssh %s\\n' "$*" >> "$FAKE_LOG"
while [ $# -gt 0 ]; do case "$1" in -o) shift 2 ;; root@*) shift; break ;; *) shift ;; esac; done
exec bash -c "$*"
"""
FAKE_DOCKER = """#!/bin/bash
printf 'docker %s\\n' "$*" >> "$FAKE_LOG"
case "$1" in
    exec)
        case "$*" in
            *date*) echo "$FAKE_KICKED"; [ -z "$FAKE_PIDS" ] || printf '%s\\n' $FAKE_PIDS ;;
            *kill*) ;;
        esac ;;
    logs) cat "$FAKE_DOCKER_LOG" ;;
esac
"""


def _kick(
    tmp_path: Path, script: str, node: str, line: str = "run recorder ready", pids: str = "47"
) -> tuple[int, str, str]:
    """Run ``ros/<script> kick <node>`` with fake ssh/docker/curl; (exit, output, commands)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("ssh", FAKE_SSH), ("docker", FAKE_DOCKER), ("curl", "#!/bin/bash\n")):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "log"
    log.write_text("")
    docker_log = tmp_path / "docker.log"
    # The node's own name and its ready line (the scripts' lines may end in ": ").
    ready_text = line.rstrip(": ")
    docker_log.write_text(
        KICK_LOG.replace("run_recorder", node).replace("run recorder ready", ready_text)
    )
    run = subprocess.run(
        ["bash", str(REPO / "ros" / script), "kick", node],
        capture_output=True,
        text=True,
        timeout=20,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "FAKE_LOG": str(log),
            "FAKE_KICKED": KICKED,
            "FAKE_PIDS": pids,
            "FAKE_DOCKER_LOG": str(docker_log),
        },
    )
    return run.returncode, run.stdout + run.stderr, log.read_text()


@pytest.mark.slow  # ~0.3-0.45 s: two dozen processes of bash, fakes and awk
def test_the_board_kick_reaches_the_board_with_its_whole_ready_line(tmp_path: Path) -> None:
    """ssh joins its arguments into one command line: unquoted, "run recorder ready" arrived as
    "run" and the first line holding that word — the old process's own — passed for the new."""
    code, out, sent = _kick(tmp_path, "thin.sh", "run_recorder")
    assert code == 0, out
    assert out.splitlines()[0] == (
        "run_recorder kicked at 10:00:05.1, ready at 10:00:11.3 UTC, 6.2 s,"
        " pid 47 -> 836: run recorder ready: tapes in /maps/rec"
    )
    assert 'sh -c kill -INT "$@" sh 47' in sent
    assert f"logs -t --since {KICKED} pepin-ros" in sent


@pytest.mark.slow  # ~0.3-0.45 s: two dozen processes of bash, fakes and awk
def test_a_board_node_that_is_not_running_is_said_and_nothing_is_signalled(tmp_path: Path) -> None:
    code, out, sent = _kick(tmp_path, "thin.sh", "run_recorder", pids="")
    assert code == 3 and "no run_recorder process in pepin-ros" in out
    assert "kill" not in sent


@pytest.mark.slow  # ~0.3-0.45 s: two dozen processes of bash, fakes and awk
def test_the_laptop_kick_waits_for_the_new_pid_the_same_way(tmp_path: Path) -> None:
    code, out, sent = _kick(tmp_path, "laptop.sh", "depth_fusion", line="fusion up: ")
    assert code == 0, out
    assert out.splitlines()[0].startswith(
        "depth_fusion kicked at 10:00:05.1, ready at 10:00:11.3 UTC, 6.2 s,"
        " pid 47 -> 836: fusion up: tapes in /maps/rec"
    ), out
    assert "docker exec pepin-vslam sh -c kill -INT" in sent
    assert "ssh" not in sent


def test_both_kicks_read_the_one_matcher() -> None:
    for script in ("ros/thin.sh", "ros/laptop.sh"):
        text = (REPO / script).read_text()
        assert "kick_ready.awk" in text, script
        assert 'grep -F "$LINE"' not in text, f"{script}: the bare --since grep is back"
