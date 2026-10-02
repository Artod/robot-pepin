#!/bin/bash
# What is the robot thinking? Live view of the navigation stack's own words, in the terminal:
# goals, planner/controller verdicts, recoveries (spin/backup/wait), with local wall-clock time.
# Runs until Ctrl-C; never sends anything to the robot.
#   ros/watch.sh          live, from the Mac's Nav2 container (ros/laptop.sh nav)
#   ros/watch.sh FILE     the same view over a saved log
set -uo pipefail
. "$(dirname "$0")/lib.sh"
if [ -n "${1:-}" ]; then
    grep -E "$PEPIN_WATCH_KEEP" "$1" | grep -vE "$PEPIN_WATCH_DROP" | pepin_render
else
    echo "watching Nav2 in pepin-macnav (Ctrl-C to stop)..."
    docker logs -f --since 2m pepin-macnav 2>&1 | grep --line-buffered -E "$PEPIN_WATCH_KEEP" | grep --line-buffered -vE "$PEPIN_WATCH_DROP" | pepin_render
fi
