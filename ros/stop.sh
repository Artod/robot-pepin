#!/bin/bash
# The red button. Stops the wheels within a second no matter what Nav2 is doing:
#   1. cancel every goal through the goal server beside Nav2 on this Mac (pepin.goal_link,
#      127.0.0.1:3337), confirmed by a navigator within 3 s: that is the whole stop;
#   2. if it is not: the base server's own stop on the board (TCP 3336), Nav2's container stopped
#      here (nothing commands the wheels until ros/laptop.sh nav brings it back), and the base's
#      stop once more. Never a board restart: it zeroes the odometry the map is tied to.
# The tray's red button takes the same first step (apps/macos/tray.py) and runs this when it fails.
# Usage: ros/stop.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"
T0=$(date +%s)
if PYTHONPATH="$HERE/../src" python3 -m pepin.goal_link --timeout 3 cancel; then
    echo "every goal cancelled, confirmed ($(( $(date +%s) - T0 )) s)"
    exit 0
fi
# The base server zeroes the wheels on a stop from any client; Nav2's controller could command
# them again until its container is down, so the stop is sent on both sides of that.
base_stop() {  # one line, then hang up: the server streams state and would hold an nc open
    python3 -c 'import socket, sys
with socket.create_connection((sys.argv[1], 3336), timeout=2) as s:
    s.sendall(b"{\"cmd\": \"stop\"}\n")' "$BOARD" 2>/dev/null
}
echo "cancel not confirmed — the base's own stop, then Nav2 stopped on this Mac"
if base_stop; then echo "base: stop sent (+$(( $(date +%s) - T0 )) s)"; else echo "!! the base server on $BOARD:3336 did not take the stop"; fi
pepin_stop_container pepin-macnav
base_stop
echo "hard stop: wheels stopped, Nav2 down on this Mac (ros/laptop.sh nav brings it back)"
exit 1
