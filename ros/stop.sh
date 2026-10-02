#!/bin/bash
# The red button. Stops the wheels whatever Nav2 is doing, and calls it done only when the wheels
# say so:
#   1. the base server's own stop on the board (TCP 3336), at once;
#   2. every goal cancelled through the goal server beside Nav2 on this Mac (pepin.goal_link,
#      confirmed by a navigator within 3 s);
#   3. the base's stop again, believed only from its state stream (pepin.red_button: the last
#      state lines of a 1 s window say the wheels are commanded still). Cancelled AND still is
#      the whole stop: Nav2 stays up.
#   4. Otherwise what commands the wheels is killed first — Nav2's composed container
#      (controller, behaviours, velocity smoother: pkill -9; its launch respawns it idle) and a
#      measured motion of ros/goto.sh round/move on the board — then the base's stop and its
#      confirmation once more, then Nav2's container stopped (ros/laptop.sh nav brings it back).
# The base's stop is not latched: the bridge forwards /cmd_vel as it comes and a live controller
# overwrites the stop within 50 ms, so the producers go before the stop that is believed. Never a
# board restart: it zeroes the odometry the map is tied to. Every docker and ssh call is bounded
# here (PEPIN_STOP_BOUND_S, 8 s): a hung one is left to finish without the button.
# Exit 0 when the wheels read still at the end, 1 otherwise. The tray's red button runs this.
# Usage: ros/stop.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"
GOAL_PORT="${PEPIN_GOAL_PORT:-3337}"  # = pepin_bringup.goal_server.PORT
BASE_PORT="${PEPIN_BASE_PORT:-3336}"  # = pepin.base_link.BASE_PORT
BOUND_S="${PEPIN_STOP_BOUND_S:-8}"
T0=$(date +%s)
since() { echo "+$(($(date +%s) - T0)) s"; }
py() { PYTHONPATH="$HERE/../src" python3 -m "$@"; }
base() { py pepin.red_button --host "$BOARD" --port "$BASE_PORT" "$@"; }
bounded() {  # SECONDS CMD...: run CMD and stop waiting for it after SECONDS (its status, or 124)
    local limit="$1" pid timer rc
    shift
    "$@" &
    pid=$!
    (sleep "$limit"; kill -KILL "$pid" 2>/dev/null) &
    timer=$!
    wait "$pid"
    rc=$?
    kill "$timer" 2>/dev/null
    wait "$timer" 2>/dev/null
    return "$rc"
}

base --confirm-s 0 >/dev/null
CANCELLED=0
py pepin.goal_link --port "$GOAL_PORT" --timeout 3 cancel && CANCELLED=1
STILL=0
base && STILL=1
if [ "$CANCELLED" = 1 ] && [ "$STILL" = 1 ]; then
    echo "stopped: every goal cancelled and the wheels still ($(since))"
    exit 0
fi
echo "!! $([ "$CANCELLED" = 1 ] && echo "cancel confirmed" || echo "cancel NOT confirmed")," \
    "$([ "$STILL" = 1 ] && echo "wheels still" || echo "wheels NOT confirmed still")" \
    "— killing what commands the wheels ($(since))"
if bounded "$BOUND_S" docker exec pepin-macnav pkill -9 -f '__node:=nav2_container'; then
    echo "Nav2's composed container killed (controller, behaviours, smoother)"
else
    echo "no Nav2 container process killed in pepin-macnav"
fi
# shellcheck disable=SC2016  # expanded on the board, inside its container
if bounded "$BOUND_S" ssh -o ConnectTimeout=3 "root@$BOARD" \
    'docker exec pepin-ros sh -c '"'"'p=$(cat /tmp/pepin_motion.pid 2>/dev/null) && kill -KILL "$p"'"'" \
    2>/dev/null; then
    echo "the board's measured motion (ros/goto.sh round/move) killed"
fi
STILL=0
base && STILL=1
bounded "$BOUND_S" pepin_stop_container pepin-macnav \
    && echo "Nav2 down on this Mac (ros/laptop.sh nav brings it back)" \
    || echo "!! pepin-macnav did not stop within ${BOUND_S} s (docker finishes it)"
if [ "$STILL" = 1 ]; then
    echo "hard stop: the wheels read still ($(since))"
    exit 0
fi
echo "!! hard stop NOT confirmed: the base did not read still ($(since)) — power the base off"
exit 1
