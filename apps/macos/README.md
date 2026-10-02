# Pepin menu-bar app

A macOS status-bar item that keeps an eye on the robot: a monochrome icon like
the system's own, with a glyph next to it only when something needs a look
(`⚠` something is down, `✕` board unreachable, `?` the last report is stale) and one
click drops down the last health report — every subsystem probe with a green check or a
red cross, board vitals, and actions.

Run it: `uv run --group macos python apps/macos/tray.py`

Smoke-test it with the robot off: `uv run --group macos python apps/macos/tray.py --check`
imports everything, builds the menu from a fake report, renders it into real menu items and
exits 0 without touching the network or the menu bar.

## Pepin.app

`apps/macos/install_app.sh` builds `~/Applications/Pepin.app`, so Spotlight's "Pepin" starts the
tray (no Dock icon). The bundle runs `/opt/homebrew/bin/uv run --directory <this checkout> --group
macos python apps/macos/tray.py` (uv by its full path: a Finder-launched app has no PATH) and
appends its output to `~/Library/Logs/Pepin.log`. `PEPIN_REPO=/path/to/pepin` points it at another
checkout, a directory argument installs elsewhere, `--desktop` adds a Desktop alias. Check:
`~/Applications/Pepin.app/Contents/MacOS/Pepin --check` (exit 0). Start it at login: System
Settings → General → Login Items, add `Pepin.app`.

## What the menu does

Two halves of the stack are probed, both read-only. The **board** (the quick tier of
`pepin.health`, the same probes as `scripts/health_check.py --quick`: ssh vitals, the
process census against its budget, bridges, servo ping, lidar, ToF, IMU, camera presence)
and **this Mac** (`pepin.health.probe_laptop`, no ssh): whether the goal server answers on
`127.0.0.1:3337`, where Nav2 runs (`ros/laptop.sh nav`), and which of `pepin-vslam` /
`pepin-macnav` are up (`docker ps`; red without the mapper). ALL GO means both halves.

The first item, **■ STOP THE ROBOT**, is the red button, run in the background and reported
as a notification: it is `ros/stop.sh`, and the notification is its last line. The base's own
stop, every goal cancelled through the goal server on this Mac (3 s to confirm), and the base's
stop again, believed only when its state stream says the wheels are still; when either is not
confirmed, Nav2's container and a measured motion on the board are killed before that stop, and
Nav2 is stopped (`ros/laptop.sh nav` brings it back; the board is never restarted, so the
odometry and the map stay). See ros/README.md, "Driving".

Read-only asks, each answered as a notification: **Where** (the goal server's `where`: the
cart's pose from `map -> base_link`, the planner, the lidar rate) and **Neck home**
(`{"cmd": "neck_home"}` to the base server on the board, port 3336: the head returns to the
reference pose of `config/neck.json`, then torque off; refused while the wheels turn —
`ros/neck.sh home` from the terminal).

Three actions move the cart or the head, each in its own Terminal window so the output is
visible: **Teleop (game)** runs `uv run python -m pepin.teleop --game` (a focused window,
keys act while held, WASD moves the neck); **Teleop in a terminal** runs `ros/teleop.sh`
(arrows drive, Shift+arrows slow, space stops, Ctrl-C ends; needs the laptop's `pepin-vslam`
container); **Turn once in place** runs `ros/goto.sh round` (one recorded 372-degree turn
judged by the gyro, a ROS process on the board).

The rest: refresh, the polling cadence (30 s after a refresh or any NO GO, 5 min once all go
for a while), `scripts/dashboard.py`, the log folder, quit.
