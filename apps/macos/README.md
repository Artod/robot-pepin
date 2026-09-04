# Pepin menu-bar app

A macOS status-bar item that keeps an eye on the robot: the icon shows the
overall verdict (`🤖` all go, `🤖⚠` something is down, `🤖…` polling, `🤖✕`
board unreachable) and one click drops down the last health report — every
subsystem probe, board vitals, and actions to refresh, open the dashboard or
open the log folder.

Run it: `uv run --group macos python apps/macos/tray.py`

Start it at login: add a Login Item pointing at a one-line launcher such as
`cd /path/to/pepin && nohup uv run --group macos python apps/macos/tray.py >/dev/null 2>&1 &`
(a `launchd` plist in `~/Library/LaunchAgents` works too, and restarts it on crash).

It is read-only: the app runs the same quick probes as `scripts/health_check.py`
(ssh vitals, servo ping, lidar and ToF stream rates) and never commands the robot.
