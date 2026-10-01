"""The menu-bar app's menu as data: what one health poll turns into, line by line.

Nothing here imports rumps or AppKit, so the menu can be built and tested with the
robot off (``tray.py --check`` does exactly that); :mod:`tray` renders these lines
into NSMenuItems and wires the actions to its methods by name.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from pepin.health import (
    FAST_POLL_S,
    SLOW_POLL_S,
    BoardVitals,
    HealthReport,
    Probe,
    is_stale,
)

# The status item's title next to the icon: nothing when all is well.
TITLE_OK, TITLE_WARN, TITLE_DEAD, TITLE_STALE = None, "⚠", "✕", "?"
# Probes that describe this Mac's half of the stack, drawn under their own rule.
LAPTOP_SYSTEMS = frozenset({"goal server", "laptop stack"})
# The commands behind the menu's Terminal items, run from the repo root (``terminal_script``).
TELEOP_GAME = "uv run python -m pepin.teleop --game"
TELEOP_TERMINAL = "ros/teleop.sh"
TURN_ONCE = "ros/go.sh round tray"
# The red button waits this long for the goal server's cancel before the hard stop (ros/stop.sh,
# the same 3 s it gives its own polite step): a longer wait here is wheels turning.
STOP_CONFIRM_S = 3.0


@dataclass(frozen=True)
class Poll:
    """One background poll: when it ran, and either a report or the error that stopped it."""

    at: datetime
    report: HealthReport | None = None
    error: str | None = None

    @property
    def reachable(self) -> bool:
        """True when the board itself answered — a failed board probe means ssh is dead."""
        if self.report is None:
            return False
        return all(p.ok for p in self.report.probes if p.system == "board")


@dataclass(frozen=True)
class Item:
    """One menu line: an action (a method name on the app), a probe (a coloured glyph) or text."""

    text: str
    action: str | None = None
    ok: bool | None = None
    separator: bool = False


SEPARATOR = Item("", separator=True)


def age_s(poll: Poll, now: datetime) -> float:
    """Seconds since the poll ran."""
    return (now - poll.at).total_seconds()


def title_for(poll: Poll | None, cadence_s: float, now: datetime) -> str | None:
    """A glyph next to the icon: stale, unreachable, degraded — or nothing when all go."""
    if poll is None:
        return None
    if is_stale(age_s(poll, now), cadence_s):
        return TITLE_STALE
    if not poll.reachable:
        return TITLE_DEAD
    return TITLE_OK if poll.report is not None and poll.report.all_go else TITLE_WARN


def header(poll: Poll | None, cadence_s: float, polling: bool, now: datetime) -> str:
    """``Pepin · ALL GO (12.3s) · updated 12:34:56 (5 s ago)`` and its NO GO / unreachable
    variants; ``refreshing…`` while a poll is in flight."""
    if poll is None:
        return "Pepin · first poll running…"
    age = age_s(poll, now)
    ago = f"{age:.0f} s ago" if age < 90 else f"{age / 60:.0f} min ago"
    stale = " · STALE" if is_stale(age, cadence_s) else ""
    stamp = f"updated {poll.at:%H:%M:%S} ({ago}){stale}" + (" · refreshing…" if polling else "")
    if poll.report is None:
        return f"Pepin · UNREACHABLE · {stamp}"
    if poll.report.all_go:
        return f"Pepin · ALL GO ({poll.report.duration_s:.1f}s) · {stamp}"
    return f"Pepin · NO GO: {', '.join(poll.report.failed)} · {stamp}"


def probe_text(probe: Probe) -> str:
    """One probe as a menu line: a check or a cross, then ``system — detail``."""
    return f"{'✓' if probe.ok else '✗'} {probe.system} — {probe.detail}"


def vitals_lines(vitals: BoardVitals) -> list[str]:
    """Board vitals as one or two human lines (CPU, memory, disk, uptime, wifi power save)."""
    temp = f"{vitals.cpu_temp_c:.0f} °C" if vitals.cpu_temp_c is not None else "? °C"
    mem = f"{vitals.mem_free_mb} MB free" if vitals.mem_free_mb is not None else "? MB free"
    lines = [f"CPU {temp} · {mem} · disk {vitals.disk_used_pct} · up {vitals.uptime}"]
    if vitals.wifi_power_save_off is not None:
        lines.append("wifi power save off" if vitals.wifi_power_save_off else "wifi power save ON")
    return lines


def cadence_text(cadence_s: float) -> str:
    """``30 s`` or ``5 min``."""
    return f"{cadence_s:.0f} s" if cadence_s < 120 else f"{cadence_s / 60:.0f} min"


def actions(cadence_s: float) -> list[Item]:
    """The bottom of the menu: refresh, the two read-only asks, tools, the moves, quit."""
    return [
        Item("Refresh now", action="on_refresh"),
        Item(
            f"Polling every {cadence_text(cadence_s)} ({FAST_POLL_S:.0f} s after a refresh or any"
            f" NO GO, {SLOW_POLL_S / 60:.0f} min when all go)"
        ),
        Item("Where (the goal server's pose)", action="on_where"),
        Item("Neck home", action="on_neck_home"),
        Item("Open dashboard", action="on_dashboard"),
        Item("Open logs folder", action="on_logs"),
        SEPARATOR,
        Item("Teleop (game)", action="on_teleop_game"),
        Item("Teleop in a terminal (arrows, Shift slow, space stops)", action="on_teleop"),
        Item("Turn once in place (recorded)", action="on_turn"),
        SEPARATOR,
        Item("Quit", action="on_quit"),
    ]


def menu(poll: Poll | None, cadence_s: float, polling: bool, now: datetime) -> list[Item]:
    """The whole menu: the red button, header, the board's probes, this Mac's, vitals, actions."""
    items: list[Item] = [
        Item("■ STOP THE ROBOT", action="on_stop"),
        SEPARATOR,
        Item(header(poll, cadence_s, polling, now)),
        SEPARATOR,
    ]
    if poll is not None and poll.report is not None:
        board = [p for p in poll.report.probes if p.system not in LAPTOP_SYSTEMS]
        laptop = [p for p in poll.report.probes if p.system in LAPTOP_SYSTEMS]
        items += [Item(probe_text(p), ok=p.ok) for p in board]
        if laptop:
            items += [SEPARATOR, *(Item(probe_text(p), ok=p.ok) for p in laptop)]
        items.append(SEPARATOR)
        if poll.reachable:
            items += [Item(text) for text in vitals_lines(poll.report.vitals)]
    elif poll is not None:
        items += [Item(f"✗ board — {poll.error}"), SEPARATOR]
    return [*items, Item("Battery: no sensor"), SEPARATOR, *actions(cadence_s)]


def terminal_script(repo_root: Path | str, command: str) -> str:
    """What a Terminal window runs for one menu action: the command, from the repo root."""
    return f"cd {repo_root} && {command}"


def where_line(answer: dict[str, Any]) -> str:
    """The goal server's ``where`` answer as one notification line."""
    if answer.get("event") != "where":
        return f"where: the goal server answered {answer.get('event', '?')}: " + str(
            answer.get("detail", "")
        )
    parts: list[str]
    if answer.get("pose") == "tf" and all(k in answer for k in ("x", "y", "yaw_deg")):
        parts = [
            f"x {float(answer['x']):+.2f} m, y {float(answer['y']):+.2f} m,"
            f" yaw {float(answer['yaw_deg']):+.0f}°"
        ]
        if "age_s" in answer:
            parts[0] += f" (map -> base_link {float(answer['age_s']) * 1000:.0f} ms old)"
    else:
        parts = ["no pose: map -> base_link is not published"]
    if answer.get("planner"):
        parts.append(f"planner {answer['planner']}")
    if answer.get("lidar"):
        parts.append(f"lidar {answer['lidar']}")
    return " · ".join(parts)


def neck_home_line(reply: dict[str, Any] | None) -> str:
    """The base server's answer to ``neck_home`` as one notification line."""
    if reply is None:
        return "neck home: no answer from the base server"
    if reply.get("error") and not reply.get("reached"):
        return f"neck home refused: {reply['error']}"
    arrival = "reached" if reply.get("reached") else "NOT reached"
    ms = f" in {float(reply.get('ms', 0.0)):.0f} ms" if "ms" in reply else ""
    where = ""
    if "pan_ticks" in reply and "tilt_ticks" in reply:
        where = f" (pan {reply['pan_ticks']}, tilt {reply['tilt_ticks']} ticks)"
    return f"neck home {arrival}{ms}{where}"


def fake_report() -> HealthReport:
    """A report like a real evening's: the board up with one red probe, Nav2 on this Mac."""
    report = HealthReport(
        probes=[
            Probe("board", True, "up 2 days, cpu 51C, 812 MB free"),
            Probe("board budget", True, "load 1.9 on 4 cores, every budget kept"),
            Probe("bridges", True, "ser2net 3333; lidar owned by ROS (up 2 hours)"),
            Probe("servo bus", True, "all 10 answer (via base server)"),
            Probe("lidar", False, "ROS container up, driver not holding the tty"),
            Probe("tof", True, "front 1.20m, left 0.45m, right none"),
            Probe("imu", True, "MPU6050 at 0x68 on i2c-2"),
            Probe("camera overview", True, "present"),
            Probe("camera wrist", True, "present"),
            Probe("goal server", True, "127.0.0.1:3337 (Nav2 on this Mac)"),
            Probe("laptop stack", True, "pepin-vslam up 2 hours, pepin-macnav up 35 minutes"),
        ],
        vitals=BoardVitals("2 days", 51.0, 812, "41%", True),
        duration_s=4.2,
    )
    return report
