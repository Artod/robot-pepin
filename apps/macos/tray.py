"""macOS menu-bar app: the robot's health at a glance, one click away.

A daemon thread polls :func:`pepin.health.run_health` (ssh + TCP, read-only) for the
board and :func:`pepin.health.probe_laptop` for this Mac's half (the goal server, the
containers) on an interval; a rumps timer drains the results on the main thread and
rebuilds the menu, so no AppKit call ever happens off the main thread. The menu itself
is :mod:`tray_menu`'s data, rendered here.

    uv run --group macos python apps/macos/tray.py            the app
    uv run --group macos python apps/macos/tray.py --check    build the menu from a fake
                                                              report, touch nothing, exit 0
"""

from __future__ import annotations

import logging
import queue
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
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
from tray_menu import (
    STOP_CONFIRM_S,
    TELEOP_GAME,
    TELEOP_TERMINAL,
    TURN_ONCE,
    Item,
    Poll,
    fake_report,
    menu,
    neck_home_line,
    terminal_script,
    title_for,
    where_line,
)

from pepin import goal_link
from pepin.base_link import NECK_MOVE_WAIT_S
from pepin.base_link import ask as ask_base
from pepin.health import (
    FAST_FOR_S,
    FAST_POLL_S,
    HealthReport,
    Probe,
    next_poll_s,
    probe_laptop,
    run_health,
)
from pepin.log import setup_logging
from pepin.transport import board_address

REPO_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = REPO_ROOT / "logs"
UV = shutil.which("uv") or "/opt/homebrew/bin/uv"
# Adaptive polling (pepin.health.next_poll_s): every 30 s after start, after a manual refresh
# and after any result that is not ALL GO; every 5 minutes once the robot has been all go for a
# while. The menu keeps the last report with its time stamp and says how old it is; a result
# older than twice its cadence is marked stale in the bar.
# The status item is a monochrome template icon (a robot head drawn black on transparent;
# macOS renders it white or black to match the other items); the title next to it is
# empty when all is well.
ICON = Path(__file__).with_name("icon_template.png")

log = logging.getLogger("tray")


def _noop(_sender: Any) -> None:
    """Callback for informational lines: with none, macOS greys the line out as disabled."""


def _line(text: str) -> Any:
    """An informational menu line in full-strength text (clicking it just closes the menu)."""
    return rumps.MenuItem(text, callback=_noop)


def _probe_line(text: str, ok: bool) -> Any:
    """A probe line: its first character (the check or the cross) green or red."""
    item = _line(text)
    color = NSColor.systemGreenColor() if ok else NSColor.systemRedColor()
    title = NSMutableAttributedString.alloc().initWithString_(item.title)
    title.addAttribute_value_range_(
        NSFontAttributeName, NSFont.menuFontOfSize_(0), (0, len(item.title))
    )
    title.addAttribute_value_range_(
        NSForegroundColorAttributeName, NSColor.labelColor(), (0, len(item.title))
    )
    title.addAttribute_value_range_(NSForegroundColorAttributeName, color, (0, 1))
    item._menuitem.setAttributedTitle_(title)
    return item


def render(items: list[Item], callback: Callable[[str], Callable[[Any], None]]) -> list[Any]:
    """The menu's data as rumps items; ``callback(name)`` is the method an action line calls."""
    rendered: list[Any] = []
    for item in items:
        if item.separator:
            rendered.append(rumps.separator)
        elif item.action is not None:
            rendered.append(rumps.MenuItem(item.text, callback=callback(item.action)))
        elif item.ok is not None:
            rendered.append(_probe_line(item.text, item.ok))
        else:
            rendered.append(_line(item.text))
    return rendered


class TrayApp(rumps.App):
    """Menu-bar app showing the last health report and offering a manual refresh."""

    def __init__(self) -> None:
        super().__init__("Pepin", title=None, icon=str(ICON), template=True, quit_button=None)
        self._results: queue.Queue[Poll] = queue.Queue()
        self._notes: queue.Queue[tuple[str, str]] = queue.Queue()  # (title, body) from helpers
        self._wake = threading.Event()
        self._fast_until = time.monotonic() + FAST_FOR_S
        self._polling = True
        self._host: str | None = None
        self._last: Poll | None = None
        self._was_all_go: bool | None = None
        self._cadence_s = FAST_POLL_S
        self._show(None)
        threading.Thread(target=self._worker, name="health-poll", daemon=True).start()
        self._timer = rumps.Timer(self._drain, 1)
        self._timer.start()

    # -- polling ----------------------------------------------------------------------------

    def _worker(self) -> None:
        """Poll loop: run on the interval or on demand, hand results to the queue."""
        while True:
            self._wake.clear()
            self._polling = True
            try:
                result = self._poll_once()
            finally:
                self._polling = False  # cleared before the result lands, so the menu updates
            self._results.put(result)
            all_go = result.report.all_go if result.report is not None else None
            self._cadence_s = next_poll_s(all_go, self._fast_until - time.monotonic())
            self._wake.wait(self._cadence_s)

    def _poll_once(self) -> Poll:
        """Resolve the board if it is not known yet, run the quick tier, then this Mac's half.

        A board that cannot be resolved is one red probe, not a blank menu: the laptop's
        lines are still worth seeing with the robot off.
        """
        try:
            try:
                if self._host is None:
                    self._host = board_address()
                report = run_health(self._host, full=False)
            except ConnectionError as exc:
                self._host = None  # re-resolve on the next poll
                report = HealthReport(probes=[Probe("board", False, str(exc)[:80])])
            report.probes.extend(probe_laptop())
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
            self._host = None
            log.warning("poll failed: %s", exc)
            return Poll(datetime.now(), error=str(exc))

    def _drain(self, _timer: Any) -> None:
        """Main-thread tick: apply the newest result, or show that a poll is in flight."""
        try:
            while not self._notes.empty():
                title, body = self._notes.get_nowait()
                rumps.notification("Pepin", title, body)
            latest: Poll | None = None
            while not self._results.empty():
                latest = self._results.get_nowait()
            if latest is not None:
                self._show(latest)
                self._notify(latest)
            elif self._polling and self._last is not None:
                self._show(self._last)  # same report, header says a refresh is in flight
            elif self._last is not None and self._age_s(self._last) % 60 < 1.0:
                self._show(self._last)  # once a minute: the age in the header, stale in the bar
        except Exception:
            # The rumps timer keeps ticking; a menu-building bug must be visible in the log.
            log.exception("menu update failed")

    def _show(self, poll: Poll | None) -> None:
        """Rebuild title and menu from one poll result (``None`` before the first poll)."""
        self._last = poll
        now = datetime.now()
        self.title = title_for(poll, self._cadence_s, now)
        self.menu.clear()
        items = menu(poll, self._cadence_s, self._polling, now)
        self.menu.update(render(items, lambda name: getattr(self, name)))

    @staticmethod
    def _age_s(poll: Poll) -> float:
        return (datetime.now() - poll.at).total_seconds()

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

    # -- actions ----------------------------------------------------------------------------

    def on_refresh(self, _sender: Any) -> None:
        """Poll right now and go back to the fast cadence for a while."""
        self._fast_until = time.monotonic() + FAST_FOR_S
        self._wake.set()

    def on_dashboard(self, _sender: Any) -> None:
        """Launch the local dashboard script from the repo root."""
        self._spawn([UV, "run", "python", "scripts/dashboard.py"])

    def on_logs(self, _sender: Any) -> None:
        """Reveal the log folder in Finder."""
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self._spawn(["open", str(LOGS_DIR)])

    def on_quit(self, _sender: Any) -> None:
        """Leave the menu bar."""
        rumps.quit_application()

    def on_stop(self, _sender: Any) -> None:
        """The red button, in the background (no Terminal window to wait for).

        First the goal server's cancel over its socket — this Mac's 3337, where Nav2 runs
        (ros/laptop.sh nav) — given STOP_CONFIRM_S to confirm; a cancel the navigators confirmed
        is the whole stop, the board untouched.
        Unreachable or unconfirmed, ros/stop.sh takes over: it kills the board's ROS
        processes so the base's deadman cuts the wheels, then restarts that stack (~45 s, the
        odometry starts from zero). The outcome comes back as a notification.
        """
        log.info("STOP requested from the tray")

        def run() -> None:
            body = self._cancel_through_goal_server()
            if body is None:
                body = self._hard_stop()
            log.info("STOP: %s", body)
            self._notes.put(("STOP", body))

        threading.Thread(target=run, name="stop-robot", daemon=True).start()

    def _cancel_through_goal_server(self) -> str | None:
        """The cancel line when a navigator confirmed the goal server's cancel; None otherwise."""
        host = goal_link.find_server()
        if host is None:
            log.info("STOP: no goal server on %s:%d", goal_link.HOST, goal_link.PORT)
            return None
        try:
            answer = goal_link.ask({"cmd": "cancel"}, host, timeout_s=STOP_CONFIRM_S)
        except goal_link.GoalServerUnreachableError as exc:
            log.warning("STOP: the goal server at %s did not confirm: %s", host, exc)
            return None
        line = goal_link.cancel_line(answer)
        if not goal_link.cancel_confirmed(answer) or line is None:
            log.warning("STOP: no navigator confirmed the cancel: %s", answer)
            return None
        return f"{line} (goal server {host})"

    @staticmethod
    def _hard_stop() -> str:
        """ros/stop.sh: its last line."""
        try:
            done = subprocess.run(
                ["bash", "ros/stop.sh"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=90
            )
            lines = (done.stdout + done.stderr).strip().splitlines()
            return "hard stop: " + (lines[-1] if lines else f"stop.sh exited {done.returncode}")
        except (OSError, subprocess.TimeoutExpired) as exc:
            return f"hard stop: stop.sh failed: {exc}"

    def on_where(self, _sender: Any) -> None:
        """The cart's pose from the goal server (``where``), notified."""

        def run() -> None:
            host = goal_link.find_server()
            if host is None:
                body = f"no goal server on {goal_link.HOST}:{goal_link.PORT}"
            else:
                try:
                    body = where_line(goal_link.ask({"cmd": "where"}, host))
                except goal_link.GoalServerUnreachableError as exc:
                    body = f"the goal server at {host} did not answer: {exc}"
            log.info("where: %s", body)
            self._notes.put(("Where", body))

        threading.Thread(target=run, name="where", daemon=True).start()

    def on_neck_home(self, _sender: Any) -> None:
        """``neck_home`` to the base server on the board (the head's reference pose, then
        torque off; refused while the wheels turn), its answer as a notification."""

        def run() -> None:
            try:
                host = self._host or board_address()
                reply = ask_base(host, {"cmd": "neck_home"}, "neck_goto", NECK_MOVE_WAIT_S)
                body = neck_home_line(reply)
            except (OSError, ConnectionError) as exc:
                body = f"neck home: the base server did not answer: {exc}"
            log.info("%s", body)
            self._notes.put(("Neck", body))

        threading.Thread(target=run, name="neck-home", daemon=True).start()

    def on_turn(self, _sender: Any) -> None:
        """One full turn in place (ros/goto.sh round: 372 deg by the gyro, recorded), in a
        Terminal window so the sweep and the verdict are visible; the cart moves."""
        self._in_terminal(TURN_ONCE)

    def on_teleop(self, _sender: Any) -> None:
        """Keyboard driving in a Terminal window (ros/teleop.sh: arrows drive, Shift+arrows
        slow, space stops, Ctrl-C ends; needs the laptop's pepin-vslam container)."""
        self._in_terminal(TELEOP_TERMINAL)

    def on_teleop_game(self, _sender: Any) -> None:
        """Game-mode teleop (pepin.teleop --game: a focused window, keys act while held,
        WASD moves the neck) in a Terminal window, from the repo root."""
        self._in_terminal(TELEOP_GAME)

    def _in_terminal(self, command: str) -> None:
        """Open Terminal.app on the repo root running one command; the window stays for
        its output. Failures only reach the log."""
        script = terminal_script(REPO_ROOT, command)
        self._spawn(
            [
                "osascript",
                "-e",
                'tell application "Terminal" to activate',
                "-e",
                f'tell application "Terminal" to do script "{script}"',
            ]
        )

    @staticmethod
    def _spawn(command: list[str]) -> None:
        """Fire and forget a helper process in the repo root; failures only reach the log."""
        try:
            subprocess.Popen(command, cwd=REPO_ROOT, start_new_session=True)
        except OSError as exc:
            log.warning("cannot run %s: %s", command, exc)


def check() -> int:
    """Smoke test with the robot off: every import done, the menu built from a fake report and
    rendered into real NSMenuItems, nothing polled, no status item made. Exit 0 when it all held."""
    now = datetime.now()
    poll = Poll(now, report=fake_report())
    items = menu(poll, FAST_POLL_S, polling=False, now=now)
    rendered = render(items, lambda _name: _noop)
    for item in items:
        print("-" * 40 if item.separator else item.text)
    actions = sorted(item.action for item in items if item.action is not None)
    missing = [name for name in actions if not callable(getattr(TrayApp, name, None))]
    if missing:
        print(f"tray --check: menu actions without a method: {missing}", file=sys.stderr)
        return 1
    print(
        f"tray --check OK: {len(rendered)} menu items, {len(actions)} actions, title"
        f" {title_for(poll, FAST_POLL_S, now)!r}, rumps {rumps.__version__}, repo {REPO_ROOT}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    """``--check`` builds the menu and exits; otherwise set up file logging and hand control to
    the macOS run loop."""
    args = sys.argv[1:] if argv is None else argv
    if "--check" in args:
        return check()
    setup_logging("tray", log_dir=LOGS_DIR, console=False)
    log.info("tray starting, repo=%s", REPO_ROOT)
    TrayApp().run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
