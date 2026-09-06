"""macOS menu-bar app: the robot's health at a glance, one click away.

A daemon thread polls :func:`pepin.health.run_health` (ssh + TCP, read-only) on
an interval; a rumps timer drains the results on the main thread and rebuilds
the menu, so no AppKit call ever happens off the main thread.
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import rumps
from AppKit import (
    NSColor,
    NSFont,
    NSFontAttributeName,
    NSForegroundColorAttributeName,
    NSMutableAttributedString,
)

from pepin.health import HealthReport, Probe, run_health
from pepin.log import setup_logging
from pepin.transport import board_address

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "logs"
UV = shutil.which("uv") or "/opt/homebrew/bin/uv"
# Adaptive polling: every 30 s for the first minutes after start or after a manual refresh
# (the robot is being brought up, things change), then every 5 minutes. The menu keeps the
# last report with its time stamp; "Refresh now" polls at once and goes back to the fast cadence.
FAST_S, SLOW_S, FAST_FOR_S = 30.0, 300.0, 300.0
# The status item is a monochrome template icon (a robot head drawn black on transparent;
# macOS renders it white or black to match the other items); the title next to it is
# empty when all is well.
ICON = Path(__file__).with_name("icon_template.png")
TITLE_OK, TITLE_WARN, TITLE_DEAD = None, "⚠", "✕"  # polling shows inside the menu, not in the bar

log = logging.getLogger("tray")


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


def _noop(_sender: Any) -> None:
    """Callback for informational lines: with none, macOS greys the line out as disabled."""


def _line(text: str) -> Any:
    """An informational menu line in full-strength text (clicking it just closes the menu)."""
    return rumps.MenuItem(text, callback=_noop)


def _probe_line(probe: Probe) -> Any:
    """One probe as a menu line: a green check or a red cross, then ``system — detail``."""
    glyph = "✓" if probe.ok else "✗"
    item = _line(f"{glyph} {probe.system} — {probe.detail}")
    color = NSColor.systemGreenColor() if probe.ok else NSColor.systemRedColor()
    title = NSMutableAttributedString.alloc().initWithString_(item.title)
    title.addAttribute_value_range_(
        NSFontAttributeName, NSFont.menuFontOfSize_(0), (0, len(item.title))
    )
    title.addAttribute_value_range_(
        NSForegroundColorAttributeName, NSColor.labelColor(), (0, len(item.title))
    )
    title.addAttribute_value_range_(NSForegroundColorAttributeName, color, (0, len(glyph)))
    item._menuitem.setAttributedTitle_(title)
    return item


def _vitals_lines(report: HealthReport) -> list[str]:
    """Board vitals as one or two human lines (CPU, memory, disk, uptime, wifi power save)."""
    v = report.vitals
    temp = f"{v.cpu_temp_c:.0f} °C" if v.cpu_temp_c is not None else "? °C"
    mem = f"{v.mem_free_mb} MB free" if v.mem_free_mb is not None else "? MB free"
    lines = [f"CPU {temp} · {mem} · disk {v.disk_used_pct} · up {v.uptime}"]
    if v.wifi_power_save_off is not None:
        lines.append("wifi power save off" if v.wifi_power_save_off else "wifi power save ON")
    return lines


class TrayApp(rumps.App):
    """Menu-bar app showing the last health report and offering a manual refresh."""

    def __init__(self) -> None:
        super().__init__("Pepin", title=None, icon=str(ICON), template=True, quit_button=None)
        self._results: queue.Queue[Poll] = queue.Queue()
        self._wake = threading.Event()
        self._fast_until = time.monotonic() + FAST_FOR_S
        self._forced = False
        self._polling = True
        self._host: str | None = None
        self._last: Poll | None = None
        self._was_all_go: bool | None = None
        self._show(None)
        threading.Thread(target=self._worker, name="health-poll", daemon=True).start()
        self._timer = rumps.Timer(self._drain, 1)
        self._timer.start()

    def _worker(self) -> None:
        """Poll loop: run on the interval or on demand, hand results to the queue."""
        while True:
            self._wake.clear()
            self._forced = False
            self._polling = True
            try:
                result = self._poll_once()
            finally:
                self._polling = False  # cleared before the result lands, so the menu updates
            self._results.put(result)
            self._wake.wait(FAST_S if time.monotonic() < self._fast_until else SLOW_S)

    def _poll_once(self) -> Poll:
        """Resolve the board address if it is not known yet, then run the quick health tier."""
        try:
            if self._host is None:
                self._host = board_address()
            report = run_health(self._host, full=False)
            log.info(
                "poll %s: %s (%.1fs)%s",
                self._host,
                "ALL GO" if report.all_go else "NO GO",
                report.duration_s,
                ""
                if report.all_go
                else " down: "
                + "; ".join(f"{p.system} ({p.detail})" for p in report.probes if not p.ok),
            )
            return Poll(datetime.now(), report=report)
        except Exception as exc:
            self._host = None  # re-resolve on the next poll
            log.warning("poll failed: %s", exc)
            return Poll(datetime.now(), error=str(exc))

    def _drain(self, _timer: Any) -> None:
        """Main-thread tick: apply the newest result, or show that a poll is in flight."""
        try:
            latest: Poll | None = None
            while not self._results.empty():
                latest = self._results.get_nowait()
            if latest is not None:
                self._show(latest)
                self._notify(latest)
            elif self._polling and self._last is not None:
                self._show(self._last)  # same report, header says a refresh is in flight
        except Exception:
            # The rumps timer keeps ticking; a menu-building bug must be visible in the log.
            log.exception("menu update failed")

    def _show(self, poll: Poll | None) -> None:
        """Rebuild title and menu from one poll result (``None`` before the first poll)."""
        self._last = poll
        self.title = self._title_for(poll)
        self.menu.clear()
        self.menu.update(self._items(poll))

    @staticmethod
    def _title_for(poll: Poll | None) -> str | None:
        """A glyph next to the icon: polling, unreachable, degraded — or nothing when all go."""
        if poll is None:
            return None
        if not poll.reachable:
            return TITLE_DEAD
        return TITLE_OK if poll.report is not None and poll.report.all_go else TITLE_WARN

    def _items(self, poll: Poll | None) -> list[Any]:
        """The whole menu: header, probes, vitals, battery note, actions."""
        items: list[Any] = [_line(self._header(poll)), rumps.separator]
        if poll is not None and poll.report is not None:
            items += [_probe_line(p) for p in poll.report.probes]
            items += [rumps.separator, *(_line(text) for text in _vitals_lines(poll.report))]
        elif poll is not None:
            items += [_line(f"✗ board — {poll.error}"), rumps.separator]
        return [*items, _line("Battery: no sensor"), rumps.separator, *self._actions()]

    def _header(self, poll: Poll | None) -> str:
        """``Pepin · ALL GO (12.3s) · updated 12:34:56`` and its NO GO / unreachable variants."""
        if poll is None:
            return "Pepin · first poll running…"
        stamp = f"updated {poll.at:%H:%M:%S}" + (" · refreshing…" if self._polling else "")
        if poll.report is None:
            return f"Pepin · UNREACHABLE · {stamp}"
        if poll.report.all_go:
            return f"Pepin · ALL GO ({poll.report.duration_s:.1f}s) · {stamp}"
        return f"Pepin · NO GO: {', '.join(poll.report.failed)} · {stamp}"

    def _actions(self) -> list[Any]:
        """Refresh, poll interval, dashboard, logs, quit."""
        cadence = "30 s" if time.monotonic() < self._fast_until else "5 min"
        return [
            rumps.MenuItem("Refresh now", callback=self._on_refresh),
            _line(f"Polling every {cadence} (30 s for 5 min after start or a refresh)"),
            rumps.MenuItem("Open dashboard", callback=self._on_dashboard),
            rumps.MenuItem("Open logs folder", callback=self._on_logs),
            rumps.separator,
            rumps.MenuItem("Quit", callback=rumps.quit_application),
        ]

    def _notify(self, poll: Poll) -> None:
        """Notify on a GO <-> NO GO transition only, never on every poll."""
        all_go = poll.report is not None and poll.report.all_go
        if self._was_all_go is not None and all_go != self._was_all_go:
            if all_go:
                body = "all systems go"
            elif poll.report is not None:
                body = "down: " + ", ".join(poll.report.failed)
            else:
                body = poll.error or "board unreachable"
            try:
                rumps.notification("Pepin", "ALL GO" if all_go else "NO GO", body)
            except Exception as exc:
                log.warning("notification failed: %s", exc)
        self._was_all_go = all_go

    def _on_refresh(self, _sender: Any) -> None:
        """Poll right now and go back to the fast cadence for a while."""
        self._forced = True
        self._fast_until = time.monotonic() + FAST_FOR_S
        self._wake.set()

    def _on_dashboard(self, _sender: Any) -> None:
        """Launch the local dashboard script from the repo root."""
        self._spawn([UV, "run", "python", "scripts/dashboard.py"])

    def _on_logs(self, _sender: Any) -> None:
        """Reveal the log folder in Finder."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._spawn(["open", str(LOGS_DIR)])

    @staticmethod
    def _spawn(command: list[str]) -> None:
        """Fire and forget a helper process in the repo root; failures only reach the log."""
        try:
            subprocess.Popen(command, cwd=REPO_ROOT, start_new_session=True)
        except OSError as exc:
            log.warning("cannot run %s: %s", command, exc)


def main() -> None:
    """Set up file logging and hand control to the macOS run loop."""
    setup_logging("tray", log_dir=LOGS_DIR, console=False)
    log.info("tray starting, repo=%s", REPO_ROOT)
    TrayApp().run()


if __name__ == "__main__":
    main()
