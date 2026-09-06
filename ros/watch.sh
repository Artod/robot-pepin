#!/bin/bash
# What is the robot thinking? Live view of the navigation stack's own words, in the terminal:
# goals, planner/controller verdicts, recoveries (spin/backup/wait), AMCL and relocalizer lines,
# with local wall-clock time. Runs until Ctrl-C; never sends anything to the robot.
#   ros/watch.sh          live
#   ros/watch.sh FILE     the same view over a saved board log (ros/maps/rec/*_board.log)
set -uo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
if [ -n "${1:-}" ]; then
    grep -E "$PEPIN_WATCH_KEEP" "$1" | grep -vE "$PEPIN_WATCH_DROP" | pepin_render
else
    echo "watching the stack on $BOARD (Ctrl-C to stop)..."
    ssh "root@$BOARD" "docker logs -f --since 2m pepin-ros 2>&1" | grep --line-buffered -E "$PEPIN_WATCH_KEEP" | grep --line-buffered -vE "$PEPIN_WATCH_DROP" | pepin_render
fi
