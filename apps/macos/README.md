# Pepin menu-bar app

A macOS status-bar item that keeps an eye on the robot: a monochrome icon like
the system's own, with a glyph next to it only when something needs a look
(`⚠` something is down, `…` polling, `✕` board unreachable) and one click drops
down the last health report — every subsystem probe with a green check or a red
cross, board vitals, and actions to refresh, open the dashboard or open the log
folder. Choosing "Refresh now" closes the menu (macOS closes a menu on any
choice; only the system's own menu extras stay open) — the poll takes about
15 s, then reopen it.

Run it: `uv run --group macos python apps/macos/tray.py`

Start it at login: add a Login Item pointing at a one-line launcher such as
`cd /path/to/pepin && nohup uv run --group macos python apps/macos/tray.py >/dev/null 2>&1 &`
(a `launchd` plist in `~/Library/LaunchAgents` works too, and restarts it on crash).

It is read-only: the app runs the same quick probes as `scripts/health_check.py`
(ssh vitals, servo ping, lidar and ToF stream rates) and never commands the robot.
