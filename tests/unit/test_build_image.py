"""ros/build-image.sh --ship-only against fake ssh, scp and docker: what reaches the board, in
which order, and what refuses before anything is loaded."""

import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]

# One fake for all three tools: every call is a line in the log; the board's answers come from
# the FAKE_* variables, matched on the remote command the script sends.
FAKE = r"""#!/bin/bash
tool="$(basename "$0")"
printf '%s %s\n' "$tool" "$*" >> "$FAKE_LOG"
if [ "$tool" = ssh ]; then  # the remote command alone, for a syntax check
    printf '%s' "${@: -1}" > "$FAKE_LOG.$(wc -l < "$FAKE_LOG" | tr -d ' ')"
fi
case "$tool:$*" in
    "docker:image inspect"*) echo 1000000000 ;;
    "docker:save"*) echo tarball ;;
    "ssh:"*"docker ps"*) exit "${FAKE_RUNNING:-1}" ;;
    "ssh:"*PEPIN_*) printf '%s\n' "${FAKE_NAV_LINES:-}" ;;
    "ssh:"*"df -Pk"*) echo "${FAKE_FREE_KB-8000000}" ;;
    "ssh:"*"docker tag"*pre-sensors*) exit "${FAKE_TAG_EXIT:-0}" ;;
    "ssh:"*"docker load"*) cat > /dev/null ;;
esac
exit 0
"""


def _ship(tmp_path: Path, **env: str) -> tuple[int, str, list[str]]:
    """Run the real script with --ship-only; returns (status, output, calls)."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    for name in ("ssh", "scp", "docker"):
        (bin_dir / name).write_text(FAKE)
        (bin_dir / name).chmod(0o755)
    log = tmp_path / "log"
    log.write_text("")
    run = subprocess.run(
        ["bash", str(REPO / "ros/build-image.sh"), "--ship-only"],
        capture_output=True,
        text=True,
        timeout=60,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "PEPIN_HOST": "127.0.0.1",
            "FAKE_LOG": str(log),
            **env,
        },
    )
    return run.returncode, run.stdout + run.stderr, log.read_text().splitlines()


def _index(calls: list[str], needle: str) -> int:
    return next(i for i, c in enumerate(calls) if needle in c)


@pytest.mark.slow
def test_a_ship_tags_the_rollback_loads_then_installs_the_unit(tmp_path: Path) -> None:
    code, out, calls = _ship(tmp_path, FAKE_NAV_LINES="PEPIN_NAV=false")
    assert code == 0, out
    assert _index(calls, "pre-sensors-2026-10-01") < _index(calls, "docker load")
    assert _index(calls, "docker load") < _index(calls, "scp ")
    assert "/etc/systemd/system/pepin-ros.service" in calls[_index(calls, "scp ")]
    assert _index(calls, "scp ") < _index(calls, "daemon-reload")
    assert "PEPIN_NAV=false" in out
    assert not [c for c in calls if "systemctl restart" in c or "systemctl start" in c]
    remote = sorted(tmp_path.glob("log.*"))
    assert len(remote) >= 6
    for path in remote:  # every command the board would run parses in its shell
        subprocess.run(["sh", "-n", str(path)], check=True)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("env", "said"),
    [
        ({"FAKE_NAV_LINES": "PEPIN_NAV=true"}, "set to run navigation"),
        ({"FAKE_NAV_LINES": "PEPIN_NAV=false\nPEPIN_SLAM_TOOLBOX=true"}, "set to run navigation"),
        ({"FAKE_FREE_KB": "1000000"}, "976 MB free"),
        ({"FAKE_FREE_KB": ""}, "could not read"),
        ({"FAKE_TAG_EXIT": "1"}, ""),
        ({"FAKE_RUNNING": "0"}, "pepin-ros is running"),
    ],
)
def test_a_ship_refuses_before_anything_is_loaded(
    tmp_path: Path, env: dict[str, str], said: str
) -> None:
    code, out, calls = _ship(tmp_path, **env)
    assert code != 0, out
    assert said in out
    assert not [c for c in calls if "docker load" in c or c.startswith(("scp ", "docker save"))]
