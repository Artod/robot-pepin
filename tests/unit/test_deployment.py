"""The split between the board and the laptop, and the link watch that makes it safe."""

from pathlib import Path

import pytest

from pepin.deployment import (
    BOARD_NAV_NODES,
    LAPTOP_NAV_NODES,
    SIDES,
    LinkWatch,
    nav_nodes,
    runs_here,
)

REPO = Path(__file__).resolve().parents[2]


def test_all_is_exactly_the_union_of_the_two_sides_with_nothing_shared() -> None:
    assert set(nav_nodes("all")) == set(BOARD_NAV_NODES) | set(LAPTOP_NAV_NODES)
    assert not set(BOARD_NAV_NODES) & set(LAPTOP_NAV_NODES), "a node on both sides collides"


def test_the_reflexes_stay_on_the_board_and_the_planner_leaves() -> None:
    for reflex in ("controller_server", "behavior_server", "bt_navigator", "velocity_smoother"):
        assert runs_here("board", reflex) and not runs_here("laptop", reflex)
    assert runs_here("laptop", "planner_server") and not runs_here("board", "planner_server")
    assert runs_here("board", "map_server")
    assert runs_here("laptop", "goal_server") and not runs_here("board", "goal_server")
    assert runs_here("board", "link_watch") and not runs_here("all", "link_watch")
    assert not runs_here("board", "goal_server"), "the heartbeat rides with the goal server"


def test_an_unknown_side_is_refused_loudly() -> None:
    with pytest.raises(ValueError):
        nav_nodes("cloud")
    assert SIDES == ("all", "board", "laptop")


def test_a_lost_link_cuts_a_running_drive_once_and_only_after_the_patience() -> None:
    watch = LinkWatch(patience_s=2.5)
    watch.beat(now=0.0)
    assert not watch.should_cut(navigating=True, now=2.0)
    assert watch.should_cut(navigating=True, now=3.0), "silent past the patience: cut"
    assert watch.should_cut(navigating=True, now=3.5), "still asking until the cancel was sent"
    watch.cut_sent()
    assert not watch.should_cut(navigating=True, now=4.0), "once per outage, once it was sent"
    watch.beat(now=5.0)
    assert not watch.should_cut(navigating=True, now=6.0)
    assert watch.should_cut(navigating=True, now=9.0), "a second outage cuts again"


def test_no_drive_no_cut_and_no_laptop_no_cut() -> None:
    watch = LinkWatch()
    assert not watch.should_cut(navigating=True, now=100.0), (
        "never heard a laptop: not a split stack"
    )
    watch.beat(now=0.0)
    assert not watch.should_cut(navigating=False, now=100.0), "standing still needs no plan"


def test_the_board_half_is_brought_up_by_the_laptop_one_step_at_a_time() -> None:
    from pepin.deployment import (
        BOARD_NAV_NODES,
        TRANSITION_ACTIVATE,
        TRANSITION_CONFIGURE,
        autostart_for,
        next_transition,
    )

    assert autostart_for("all") and autostart_for("laptop") and not autostart_for("board")
    assert BOARD_NAV_NODES[-1] == "bt_navigator", "the tree loads last: it needs the planner side"
    fresh = dict.fromkeys(BOARD_NAV_NODES, "unconfigured")
    assert next_transition(fresh) == ("controller_server", TRANSITION_CONFIGURE)
    half = {**fresh, "controller_server": "active", "behavior_server": "inactive"}
    assert next_transition(half) == (
        "behavior_server",
        TRANSITION_ACTIVATE,
    )  # a failed try continues
    almost = dict.fromkeys(BOARD_NAV_NODES, "active")
    almost["bt_navigator"] = "inactive"
    assert next_transition(almost) == ("bt_navigator", TRANSITION_ACTIVATE)
    assert next_transition(dict.fromkeys(BOARD_NAV_NODES, "active")) is None
    assert next_transition({**almost, "bt_navigator": "activating"}) is None  # in transit: wait
    assert next_transition({}) is None  # nobody answered: wait, do not guess


def test_the_tape_is_written_where_the_sensors_are() -> None:
    from pepin.deployment import runs_here

    assert runs_here("all", "run_recorder") and runs_here("board", "run_recorder")
    assert not runs_here("laptop", "run_recorder")
    assert runs_here("laptop", "goal_server") and not runs_here("board", "goal_server")


def test_the_container_s_names_are_the_nodes_each_side_composes() -> None:
    from pepin.deployment import laptop_launch_nodes, nav_container_nodes

    board = nav_container_nodes("board")
    assert board[0] == "/nav2_container_board" and "/bt_navigator" in board
    assert "/local_costmap/local_costmap" in board and "/global_costmap/global_costmap" not in board
    assert "/map_server" in board and "/lifecycle_manager_navigation_board" in board
    laptop = nav_container_nodes("laptop")
    assert "/planner_server" in laptop and "/global_costmap/global_costmap" in laptop
    assert "/map_server" not in laptop and "/controller_server" not in laptop
    assert set(laptop_launch_nodes("nav")) == set(laptop) | {"/goal_server"}
    whole = nav_container_nodes("all")
    assert whole[0] == "/nav2_container" and set(board[1:]) | set(laptop[1:]) <= set(whole) | {
        "/lifecycle_manager_navigation_board",
        "/lifecycle_manager_navigation_laptop",
    }


def test_the_mounts_are_read_from_config_wherever_the_library_runs(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """config/imu.json and config/lidar.json are found from a checkout, from a directory named
    by PEPIN_CONFIG_DIR, and a missing file names every place looked instead of guessing. What
    the files then say is pepin.mounts' business (test_mounts.py); this is the search."""
    from pathlib import Path

    from pepin.deployment import config_file
    from pepin.mounts import Mounts

    repo = Path(__file__).resolve().parents[2]
    monkeypatch.delenv("PEPIN_CONFIG_DIR", raising=False)
    assert config_file("imu.json") == repo / "config/imu.json"
    assert config_file("lidar.json") == repo / "config/lidar.json"
    with pytest.raises(FileNotFoundError, match=r"config/no_such\.json is in none of"):
        config_file("no_such.json")
    assert Mounts.load() == Mounts.load(repo / "config"), "config_file finds the checkout's own"
    elsewhere = repo / "tests"
    monkeypatch.setenv("PEPIN_CONFIG_DIR", str(elsewhere))
    assert config_file("coverage_floor.txt") == elsewhere / "coverage_floor.txt"
    assert config_file("imu.json") == repo / "config/imu.json", "the override is searched first"


def test_a_node_s_flags_are_reached_where_its_process_lives() -> None:
    """ros/flags.sh execs into the container a node runs in: the laptop's SLAM container for
    the camera nodes, its navigation container for the planner and the goal server when the
    stack is split, the board's for them on a whole board and for the tracker, the sensors and
    anything it does not know — pepin-laptop exists only in the split (ros/laptop.sh)."""
    from pepin.deployment import node_host

    for split in (False, True):
        assert node_host("depth_stream", split=split) == ("laptop", "pepin-vslam")
        assert node_host("/depth_fusion", split=split) == ("laptop", "pepin-vslam")
        assert node_host("neck_state", split=split) == ("board", "pepin-ros")
    assert node_host("goal_server", split=True) == ("laptop", "pepin-laptop")
    assert node_host("planner_server", split=True) == ("laptop", "pepin-laptop")
    assert node_host("goal_server") == ("board", "pepin-ros"), "a whole board, the default"
    assert node_host("planner_server") == ("board", "pepin-ros")


def test_a_cross_machine_topic_carries_one_qos_on_both_sides() -> None:
    """A reader and a writer that disagree on reliability do not match: /imu/data_raw is written
    RELIABLE ten deep by the board, and every reader of it asks for the same."""
    from pepin.deployment import BRIDGED_QOS, bridged_qos

    assert bridged_qos("/imu/data_raw") == ("reliable", 10) == bridged_qos("imu/data_raw")
    assert bridged_qos("/scan") is None
    assert set(BRIDGED_QOS) == {"/imu/data_raw", "/vo", "/odom"}
    for topic, (reliability, depth) in BRIDGED_QOS.items():
        assert reliability in ("reliable", "best_effort") and depth > 0, topic


def _unit_command(unit: str, prefix: str, marker: str) -> str:
    """The shell text of the one ``prefix`` line of a board unit that contains ``marker``, with
    systemd's own escapes undone (``$$`` and ``%%``) — what /bin/sh is actually handed."""
    text = (REPO / "board" / unit).read_text()
    line = next(ln for ln in text.splitlines() if ln.startswith(prefix) and marker in ln)
    return line.split("-c '", 1)[1].rsplit("'", 1)[0].replace("$$", "$").replace("%%", "%")


def test_both_routers_log_the_transport_lifecycle_and_nothing_per_message() -> None:
    """2026-09-23: after a Mac wake the router-to-router session stayed dead for 28 minutes and
    neither router wrote a line, because zenoh logs re-dials, refused handshakes and expired links
    only at debug. Both routers get the same RUST_LOG, and it names no per-message module."""
    import re

    lib = (REPO / "ros/lib.sh").read_text()
    shell = re.search(r'PEPIN_ZROUTER_LOG="\$\{PEPIN_ZROUTER_LOG:-([^}]+)\}"', lib)
    assert shell, "ros/lib.sh defines the laptop router's filter"
    unit = (REPO / "board/pepin-zrouter.service").read_text()
    board = re.search(r"^Environment=PEPIN_ZROUTER_LOG=(\S+)$", unit, re.M)
    assert board and board.group(1) == shell.group(1), "one filter for both routers"
    directives = shell.group(1).split(",")
    assert directives[0] == "info", "everything else at the level rmw_zenohd always had"
    assert "zenoh::net::runtime::orchestrator=debug" in directives, "the laptop's re-dials"
    assert "zenoh_transport::unicast::establishment=debug" in directives, "refused handshakes"
    for noisy in ("universal::rx", "universal::tx", "routing", "pipeline", "=trace"):
        assert not any(noisy in d for d in directives), f"{noisy}: a line per message"
    runs = [ln for ln in unit.splitlines() if ln.startswith("ExecStart=")]
    assert runs[0].count("-e RUST_LOG=$PEPIN_ZROUTER_LOG") == 2, "both docker run bodies"
    laptop = (REPO / "ros/laptop.sh").read_text()
    assert '-e "RUST_LOG=$PEPIN_ZROUTER_LOG"' in laptop.split("zrouter_up() {")[1].split("\n}")[0]


# Transports each router served on 2026-09-24 04:00Z (established TCP sessions on 7447): the
# board's 10 nodes on the loopback and the laptop's router; the laptop router's 14 sessions of
# pepin-vslam alone and its link to the board — pepin-laptop's nodes, the goal and flag tools and
# `docker exec` probes come on top of that one, about ten more.
BOARD_ROUTER_TRANSPORTS = 11
LAPTOP_ROUTER_TRANSPORTS = 15
LAPTOP_ROUTER_UNCOUNTED = 10


def test_both_routers_have_more_rx_workers_than_the_sessions_they_serve() -> None:
    """2026-09-23: with zenoh's two RX workers blocked in 20-s pushes to the sleeping laptop, the
    board router's close of that link never ran and its own nodes' sessions timed out on it. Each
    session pushing toward a frozen peer holds one worker and the close needs one more (4
    publishers wedged 4 workers and not 5, scratch/link_autopsy/wedge_threshold.py), so each
    router gets more workers than its census; 2 is the old behaviour, one variable away."""
    import re

    lib = (REPO / "ros/lib.sh").read_text()
    shell = re.search(r'PEPIN_ZROUTER_RX_WORKERS="\$\{PEPIN_ZROUTER_RX_WORKERS:-(\d+)\}"', lib)
    unit = (REPO / "board/pepin-zrouter.service").read_text()
    board = re.search(r"^Environment=PEPIN_ZROUTER_RX_WORKERS=(\d+)$", unit, re.M)
    assert shell and board
    assert int(board.group(1)) > BOARD_ROUTER_TRANSPORTS + 1, "the board's census, with room"
    laptop_peak = LAPTOP_ROUTER_TRANSPORTS + LAPTOP_ROUTER_UNCOUNTED
    assert int(shell.group(1)) > laptop_peak + 1, "the laptop's census and what it missed"
    runtime = '-e "ZENOH_RUNTIME=(rx: (worker_threads: $PEPIN_ZROUTER_RX_WORKERS))"'
    runs = [ln for ln in unit.splitlines() if ln.startswith("ExecStart=")]
    assert runs[0].count(runtime) == 2, "both docker run bodies"
    assert runtime in (REPO / "ros/laptop.sh").read_text().split("zrouter_up() {")[1]


def test_the_board_router_keeps_its_log_across_a_restart_like_the_stack_does() -> None:
    """Under `docker run --rm` the board router's log of the 2026-09-23 wake (the 20-s closures
    to the sleeping laptop) went with its first restart. The container now outlives its process,
    and the next start copies the log before it removes the name."""
    unit = (REPO / "board/pepin-zrouter.service").read_text()
    runs = [ln for ln in unit.splitlines() if ln.startswith("ExecStart=")]
    assert runs and all("docker run --rm" not in ln for ln in runs), "the log must outlive it"
    pre = [ln for ln in unit.splitlines() if ln.startswith("ExecStartPre=")]
    archive = next(i for i, ln in enumerate(pre) if "docker logs -t pepin-zrouter" in ln)
    remove = next(i for i, ln in enumerate(pre) if "docker rm -f pepin-zrouter" in ln)
    assert archive < remove, "copied before the name is removed"
    assert pre[archive].startswith("ExecStartPre=-"), "a failed copy never blocks the router"
    stack = (REPO / "board/pepin-ros.service").read_text()
    assert "docker logs pepin-ros > /root/pepin-ros/logs/" in stack, "the same directory"


@pytest.mark.slow
def test_the_board_router_archive_line_does_what_it_says(tmp_path: Path) -> None:
    """The unit's archive command run by /bin/sh against a fake docker: a log file for a
    container that exists, nothing for one that does not, nothing when switched off."""
    import os
    import subprocess

    fake = tmp_path / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        'case "$1 $2" in\n'
        '  "container inspect") [ -f "$FAKE_EXISTS" ] ;;\n'
        '  "logs -t") echo "2026-09-23T21:26:52Z Unable to push"; echo "err line" >&2 ;;\n'
        "  *) exit 3 ;;\n"
        "esac\n"
    )
    fake.chmod(0o755)
    logs = tmp_path / "logs"
    cmd = (
        _unit_command("pepin-zrouter.service", "ExecStartPre=", "docker logs")
        .replace("/usr/bin/docker", str(fake))
        .replace("/root/pepin-ros/logs", str(logs))
    )
    exists = tmp_path / "exists"

    def run(archive: str | None) -> None:
        env = {"PATH": os.environ["PATH"], "FAKE_EXISTS": str(exists)}
        if archive is not None:
            env["PEPIN_LOG_ARCHIVE"] = archive
        subprocess.run(["/bin/sh", "-c", cmd], env=env, check=True, timeout=10)

    run(None)
    assert not logs.exists(), "no container, no file (a first boot)"
    exists.touch()
    run("off")
    assert not logs.exists(), "switched off: nothing kept, as before"
    run(None)
    (saved,) = logs.glob("zrouter_*.log")
    assert saved.read_text() == "2026-09-23T21:26:52Z Unable to push\nerr line\n"


@pytest.mark.slow
def test_every_laptop_container_removal_keeps_the_log_first(tmp_path: Path) -> None:
    """ros/lib.sh's pepin_remove_container is the one way a container leaves this laptop
    (test_scripts_parse holds that); it must stop, copy the log, and only then remove."""
    import os
    import subprocess

    calls = tmp_path / "calls"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "docker"
    fake.write_text(
        "#!/bin/sh\n"
        f'echo "$*" >> "{calls}"\n'
        'case "$1 $2" in\n'
        '  "container inspect") [ "$3" = alive ] ;;\n'
        '  "logs -t") echo "vslam line for $3" ;;\n'
        "esac\n"
    )
    fake.chmod(0o755)
    out = tmp_path / "archive"
    script = "source ros/lib.sh && pepin_remove_container alive gone"
    env = {
        "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}",
        "HOME": str(tmp_path),
        "PEPIN_LOG_DIR": str(out),
        "PEPIN_STOP_TIMEOUT_S": "1",
    }
    result = subprocess.run(
        ["bash", "-c", script], cwd=REPO, env=env, capture_output=True, text=True, timeout=20
    )
    assert result.returncode == 0, result.stderr
    (saved,) = out.glob("*_alive.log")
    assert saved.read_text() == "vslam line for alive\n"
    assert not list(out.glob("*_gone.log")), "a container that is not there leaves no file"
    order = calls.read_text().splitlines()
    stop = next(i for i, c in enumerate(order) if c.startswith("stop"))
    copy = next(i for i, c in enumerate(order) if c.startswith("logs -t alive"))
    remove = next(i for i, c in enumerate(order) if c.startswith("rm -f"))
    assert stop < copy < remove, order
    env["PEPIN_LOG_ARCHIVE"] = "off"
    env["PEPIN_LOG_DIR"] = str(tmp_path / "archive_off")
    subprocess.run(["bash", "-c", script], cwd=REPO, env=env, check=True, timeout=20)
    assert not (tmp_path / "archive_off").exists(), "PEPIN_LOG_ARCHIVE=off keeps nothing"
