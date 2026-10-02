# Pepin menu-bar app

A macOS status-bar item that keeps an eye on the robot: a monochrome icon like
the system's own, with a glyph next to it only when something needs a look
(`⚠` something is down, `✕` board unreachable, `?` the last report is stale) and one
click drops down the last health report — every subsystem probe with a green check or a
red cross, board vitals, and actions. Choosing "Refresh now" closes the menu (macOS closes
a menu on any choice; only the system's own menu extras stay open) — the poll takes a few
seconds, then reopen it.

Run it: `uv run --group macos python apps/macos/tray.py`

Smoke-test it with the robot off: `uv run --group macos python apps/macos/tray.py --check`
imports everything, builds the menu from a fake report, renders it into real menu items and
exits 0 without touching the network or the menu bar.

## Pepin.app: Spotlight and Finder

`apps/macos/install_app.sh` builds `~/Applications/Pepin.app`, so command-space, "Pepin"
starts the tray (no Dock icon; it lives in the menu bar). The bundle's executable is a
shell script that runs `/opt/homebrew/bin/uv run --directory <this checkout> --group macos
python apps/macos/tray.py` — uv by its full path, because a Finder-launched app has no
PATH — with everything it prints appended to `~/Library/Logs/Pepin.log` (the tray's own
log stays in `logs/`). The icon is `icon_template.png` scaled with `sips` + `iconutil`.

- `apps/macos/install_app.sh` — build (or rebuild) the bundle for the checkout the script is
  in; `PEPIN_REPO=/path/to/pepin` points it at another one; a directory argument installs
  elsewhere than `~/Applications`.
- `apps/macos/install_app.sh --desktop` — also an alias on the Desktop (Finder makes it;
  macOS may ask once to let the shell control Finder).
- Check: `~/Applications/Pepin.app/Contents/MacOS/Pepin --check` (exit 0), and
  `mdfind "kMDItemDisplayName == 'Pepin*'"` lists the bundle once Spotlight has indexed it
  (the installer runs `mdimport` on it; a fresh index can lag a minute).

Start it at login: System Settings → General → Login Items, add `Pepin.app` (a `launchd`
plist in `~/Library/LaunchAgents` works too, and restarts it on crash).

## What the menu does

Two halves of the stack are probed, both read-only. The **board** (the quick tier of
`pepin.health`, the same probes as `scripts/health_check.py --quick`: ssh vitals, the
process census against its budget, bridges, servo ping, lidar, ToF, IMU, camera presence)
and **this Mac** (`pepin.health.probe_laptop`, no ssh): where the goal server answers on
port 3337 — `127.0.0.1` when Nav2 runs here (the macnav container or `ros/laptop.sh`), the
board otherwise — and which of `pepin-vslam` / `pepin-macnav` / `pepin-laptop` are up
(`docker ps`; red without the mapper). ALL GO means both halves.

The first item, **■ STOP THE ROBOT**, is the red button, run in the background and reported
as a notification. It cancels every goal through the goal server's socket on this Mac,
where Nav2 runs (`pepin.goal_link`, the same cancel as `ros/goto.sh cancel`), giving it 3 s to
confirm; a confirmed cancel is the whole stop and the board is untouched. If no goal server
answers or no navigator confirms, `ros/stop.sh` takes over: the base server's own stop on the
board, Nav2's container stopped on this Mac, and the base's stop once more (`ros/laptop.sh nav`
brings Nav2 back; the board is never restarted, so the odometry and the map stay).

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
