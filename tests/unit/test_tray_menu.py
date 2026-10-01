"""The menu-bar app's menu as data (apps/macos/tray_menu.py): built from a health report with
no rumps, no AppKit and no robot — what ``tray.py --check`` renders."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from pepin.health import FAST_POLL_S, SLOW_POLL_S, BoardVitals, HealthReport, Probe

REPO = Path(__file__).resolve().parents[2]
APP = REPO / "apps/macos"


def _module() -> ModuleType:
    spec = importlib.util.spec_from_file_location("tray_menu", APP / "tray_menu.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("tray_menu", module)
    spec.loader.exec_module(module)
    return module


tm = _module()
NOW = datetime(2026, 10, 1, 21, 30, 0)


def _texts(items: list[Any]) -> list[str]:
    return ["---" if item.separator else item.text for item in items]


def test_a_green_report_is_all_go_with_the_laptop_lines_under_their_own_rule() -> None:
    report = tm.fake_report()
    report.probes = [Probe(p.system, True, p.detail) for p in report.probes]
    poll = tm.Poll(NOW - timedelta(seconds=5), report=report)
    items = tm.menu(poll, FAST_POLL_S, polling=False, now=NOW)
    texts = _texts(items)
    assert texts[0] == "■ STOP THE ROBOT" and items[0].action == "on_stop"
    assert texts[2] == "Pepin · ALL GO (4.2s) · updated 21:29:55 (5 s ago)"
    board = texts.index("✓ board — up 2 days, cpu 51C, 812 MB free")
    goal = texts.index("✓ goal server — 127.0.0.1:3337 (Nav2 on this Mac)")
    assert texts[goal - 1] == "---" and board < goal, "this Mac's half is its own section"
    assert "CPU 51 °C · 812 MB free · disk 41% · up 2 days" in texts
    assert "wifi power save off" in texts
    assert tm.title_for(poll, FAST_POLL_S, NOW) is None, "nothing next to the icon when all go"
    probes = [item for item in items if item.ok is not None]
    assert len(probes) == len(report.probes) and all(item.ok for item in probes)


def test_a_red_probe_makes_the_header_no_go_and_the_bar_show_a_warning() -> None:
    poll = tm.Poll(NOW, report=tm.fake_report())  # the fake report's lidar is red
    items = tm.menu(poll, FAST_POLL_S, polling=True, now=NOW)
    texts = _texts(items)
    assert texts[2] == "Pepin · NO GO: lidar · updated 21:30:00 (0 s ago) · refreshing…"
    lidar = next(item for item in items if item.text.startswith("✗ lidar"))
    assert lidar.ok is False
    assert tm.title_for(poll, FAST_POLL_S, NOW) == tm.TITLE_WARN


def test_an_unreachable_board_is_a_cross_in_the_bar_and_no_vitals() -> None:
    report = HealthReport(
        probes=[
            Probe("board", False, "unreachable"),
            Probe("goal server", False, "nobody listens on 127.0.0.1:3337 or 10.0.0.187:3337"),
            Probe("laptop stack", False, "docker daemon not running"),
        ]
    )
    poll = tm.Poll(NOW, report=report)
    texts = _texts(tm.menu(poll, FAST_POLL_S, polling=False, now=NOW))
    assert texts[2].startswith("Pepin · NO GO: board, goal server, laptop stack")
    assert not any(text.startswith("CPU ") for text in texts), "no vitals from a silent board"
    assert tm.title_for(poll, FAST_POLL_S, NOW) == tm.TITLE_DEAD
    failed = tm.Poll(NOW, error="pepin.local: name not resolved")
    texts = _texts(tm.menu(failed, FAST_POLL_S, polling=False, now=NOW))
    assert texts[2].startswith("Pepin · UNREACHABLE")
    assert "✗ board — pepin.local: name not resolved" in texts


def test_before_the_first_poll_the_menu_says_so_and_still_has_every_action() -> None:
    items = tm.menu(None, FAST_POLL_S, polling=True, now=NOW)
    texts = _texts(items)
    assert texts[2] == "Pepin · first poll running…"
    assert tm.title_for(None, FAST_POLL_S, NOW) is None
    actions = [item.action for item in items if item.action is not None]
    assert actions == [
        "on_stop",
        "on_refresh",
        "on_where",
        "on_neck_home",
        "on_dashboard",
        "on_logs",
        "on_teleop_game",
        "on_teleop",
        "on_turn",
        "on_quit",
    ]


def test_a_result_older_than_twice_its_cadence_is_stale_in_bar_and_header() -> None:
    poll = tm.Poll(NOW - timedelta(minutes=11), report=tm.fake_report())
    assert tm.title_for(poll, SLOW_POLL_S, NOW) == tm.TITLE_STALE
    header = tm.header(poll, SLOW_POLL_S, polling=False, now=NOW)
    assert "updated 21:19:00 (11 min ago) · STALE" in header
    assert tm.cadence_text(SLOW_POLL_S) == "5 min" and tm.cadence_text(FAST_POLL_S) == "30 s"
    assert "Polling every 5 min" in _texts(tm.menu(poll, SLOW_POLL_S, False, NOW))[-11]


def test_vitals_show_question_marks_for_what_the_board_did_not_say() -> None:
    assert tm.vitals_lines(BoardVitals()) == ["CPU ? °C · ? MB free · disk ? · up ?"]
    assert tm.vitals_lines(BoardVitals("3 h", 48.6, 700, "40%", False)) == [
        "CPU 49 °C · 700 MB free · disk 40% · up 3 h",
        "wifi power save ON",
    ]


def test_the_terminal_items_run_their_exact_commands_from_the_repo_root() -> None:
    """The game teleop is launched exactly as ordered; the others keep their scripts."""
    repo = "/Users/artem/robots/pepin"
    assert (
        tm.terminal_script(repo, tm.TELEOP_GAME)
        == "cd /Users/artem/robots/pepin && uv run python -m pepin.teleop --game"
    )
    assert tm.terminal_script(Path(repo), tm.TELEOP_TERMINAL) == f"cd {repo} && ros/teleop.sh"
    assert tm.terminal_script(repo, tm.TURN_ONCE) == f"cd {repo} && ros/go.sh round tray"
    assert (REPO / "ros/teleop.sh").exists() and (REPO / "ros/go.sh").exists()


def test_where_is_one_line_with_the_pose_the_planner_and_the_lidar() -> None:
    answer = {
        "event": "where",
        "planner": "hybrid",
        "pose": "tf",
        "lidar": "10.2 Hz",
        "x": -0.3612,
        "y": 2.79,
        "yaw_deg": 87.4,
        "age_s": 0.042,
    }
    assert tm.where_line(answer) == (
        "x -0.36 m, y +2.79 m, yaw +87° (map -> base_link 42 ms old) · planner hybrid"
        " · lidar 10.2 Hz"
    )
    assert tm.where_line({"event": "where", "planner": "navfn", "pose": "none"}) == (
        "no pose: map -> base_link is not published · planner navfn"
    )
    assert tm.where_line({"event": "error", "detail": "busy"}).startswith(
        "where: the goal server answered error: busy"
    )


def test_neck_home_is_one_line_reached_refused_or_silent() -> None:
    reached = {"type": "neck_goto", "pan_ticks": 2029, "tilt_ticks": 2360, "reached": True}
    assert tm.neck_home_line({**reached, "ms": 1180.4}) == (
        "neck home reached in 1180 ms (pan 2029, tilt 2360 ticks)"
    )
    late = {**reached, "pan_ticks": 2100, "reached": False, "ms": 3000.0}
    assert tm.neck_home_line(late) == "neck home NOT reached in 3000 ms (pan 2100, tilt 2360 ticks)"
    refused = {"type": "neck_goto", "reached": False, "error": "the wheels are moving"}
    assert tm.neck_home_line(refused) == "neck home refused: the wheels are moving"
    assert tm.neck_home_line(None) == "neck home: no answer from the base server"


def test_the_stop_waits_the_red_button_s_three_seconds_for_the_goal_server() -> None:
    """ros/stop.sh gives its own polite step 3 s; the tray's socket cancel gets the same."""
    assert tm.STOP_CONFIRM_S == 3.0


# ---- the launcher -----------------------------------------------------------------------------


def test_the_installer_parses_and_names_uv_in_full_for_a_finder_launch() -> None:
    """Finder-launched apps have no PATH: the bundle's executable must call uv by its absolute
    path, from the repo, with the macos group, and keep a log where Console finds it."""
    script = APP / "install_app.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    text = script.read_text()
    assert "/opt/homebrew/bin/uv" in text
    assert "--group macos python apps/macos/tray.py" in text
    assert "Library/Logs/Pepin.log" in text
    assert "<key>LSUIElement</key>" in text and "com.artem.pepin.tray" in text


@pytest.mark.slow
def test_the_installer_builds_a_bundle_whose_launcher_runs_the_tray(tmp_path: Path) -> None:
    """Installed into a scratch folder: the bundle's layout, the plist keys Spotlight and
    LaunchServices read, and the launcher's exact command line with the repo path baked in."""
    repo = tmp_path / "repo"
    (repo / "apps/macos").mkdir(parents=True)
    (repo / "apps/macos/tray.py").write_text("")  # the installer checks the tray is there
    done = subprocess.run(
        ["bash", str(APP / "install_app.sh"), str(tmp_path / "Applications")],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), "PEPIN_REPO": str(repo)},
    )
    assert done.returncode == 0, done.stdout + done.stderr
    bundle = tmp_path / "Applications/Pepin.app"
    plist = (bundle / "Contents/Info.plist").read_text()
    for key, value in (
        ("CFBundleName", "Pepin"),
        ("CFBundleDisplayName", "Pepin"),
        ("CFBundleIdentifier", "com.artem.pepin.tray"),
        ("CFBundleExecutable", "Pepin"),
    ):
        assert f"<key>{key}</key>\n\t<string>{value}</string>" in plist, key
    assert "<key>LSUIElement</key>\n\t<true/>" in plist
    launcher = bundle / "Contents/MacOS/Pepin"
    assert launcher.stat().st_mode & 0o111, "the executable bit"
    text = launcher.read_text()
    assert (
        f'exec /opt/homebrew/bin/uv run --directory "{repo}" --group macos python'
        ' apps/macos/tray.py "$@"'
    ) in text
    assert '>> "$HOME/Library/Logs/Pepin.log" 2>&1' in text
