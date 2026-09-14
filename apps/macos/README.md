# Pepin menu-bar app

A macOS status-bar item that keeps an eye on the robot: a monochrome icon like
the system's own, with a glyph next to it only when something needs a look
(`⚠` something is down, `✕` board unreachable) and one click drops
down the last health report — every subsystem probe with a green check or a red
cross, board vitals, and actions to refresh, open the dashboard or open the log
folder. Choosing "Refresh now" closes the menu (macOS closes a menu on any
choice; only the system's own menu extras stay open) — the poll takes about
15 s, then reopen it.

Run it: `uv run --group macos python apps/macos/tray.py`

Start it at login: add a Login Item pointing at a one-line launcher such as
`cd /path/to/pepin && nohup uv run --group macos python apps/macos/tray.py >/dev/null 2>&1 &`
(a `launchd` plist in `~/Library/LaunchAgents` works too, and restarts it on crash).

The first item, **■ STOP THE ROBOT**, is the red button: it runs `ros/stop.sh` at once in the background (Nav2 is asked to cancel; if that is not confirmed within 3 s the ROS processes are killed and the base's deadman stops the wheels, then the stack restarts in ~45 s) and reports the outcome as a notification.

Two menu actions move the cart, each in its own Terminal window so the output is visible: *Turn once in place* (`ros/go.sh round`, one recorded 372-degree turn judged by the gyro) and *Teleop* (`ros/teleop.sh`, keys i / , / j / l, k or space stops, Ctrl-C ends). Everything else is read-only: the app runs the same quick probes as `scripts/health_check.py`
(ssh vitals, servo ping, lidar and ToF stream rates) and never commands the robot.
