#!/bin/bash
# The whole checkout's code to the robot without rebuilding the image: rsync ros/, src/pepin and
# config/ to the board. Usage:
#   ros/sync.sh            the files only: the running stack keeps running, and each node picks
#                          the new code up at its next start (ros/push.sh FILE... kicks exactly
#                          the nodes a change reaches; ros/thin.sh kick NODE one of them)
#   ros/sync.sh --restart  the files, then the board's stack (systemd unit pepin-ros): what a
#                          launch, params or config change needs; ros/restart.sh board --deploy
# A restart by default was a surprise more than once (journal 2026-09-22): a laptop-only change
# restarted the board. --no-restart, the old way to say the default, is still accepted.
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
HERE="$(cd "$(dirname "$0")" && pwd)"
RESTART=false
for arg in "$@"; do
    case "$arg" in
        --restart) RESTART=true ;;
        --no-restart) ;;
        *) echo "usage: ros/sync.sh [--restart]"; exit 2 ;;
    esac
done
# The board is a sensor box: of params/ it reads ekf.yaml alone, and of maps/ nothing — maps,
# databases and places books are the laptop's (hundreds of MB that once held a deploy for
# minutes). maps/rec and logs are written BY the board and fetched to the laptop. An excluded
# path is also out of --delete's reach, so what the board wrote there stays.
rsync -a --delete --exclude '__pycache__' --exclude 'pepin_src' --exclude 'logs' --exclude 'maps/*' --include 'params/ekf.yaml' --exclude 'params/*' "$HERE/" "root@$BOARD:/root/pepin-ros/"
ssh "root@$BOARD" "mkdir -p /root/pepin-ros/pepin_src/pepin /root/pepin-ros/pepin_src/config"
rsync -a --delete --exclude '__pycache__' "$HERE/../src/pepin/" "root@$BOARD:/root/pepin-ros/pepin_src/pepin/"
# The mounts (config/lidar.json, config/imu.json) beside the library: the container mounts
# /root/pepin-ros/pepin_src as /ws/pepin_src and the launch reads them from there at start
# (pepin.deployment.config_file) — the same files the laptop reads from its checkout.
rsync -a --delete "$HERE/../config/" "root@$BOARD:/root/pepin-ros/pepin_src/config/"
if [ "$RESTART" = true ]; then
    pepin_board_restart  # ros/lib.sh: the stack and its router, the router first
    # A census taken now would call the last nodes MISSING and read every CPU number as the
    # start-up burst it is.
    echo "letting the stack settle before the census"; sleep 25
fi
# What the board now runs, against config/board_manifest.json (ros/README.md, "What runs on the
# board"). A red census is information, never a failed deploy: the code is already on the robot
# by this line, and a sync that exits 1 would read as "the sync broke".
"$HERE/board.sh" census || true
