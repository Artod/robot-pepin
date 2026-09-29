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
# maps/rec and logs are written BY the board and fetched to the laptop; pushing them back would ship
# hundreds of MB of camera video over WiFi onto the SD card (it did, 2026-09-06: a 4-minute sync).
# A database set aside in a SUBDIRECTORY of maps/ is not caught by 'maps/rtabmap*': on 2026-09-19 a
# 1.1 GB rtabmap.worldv.db under maps/_aside_*/ held a deploy for ten minutes before it was seen.
# The volume's snapshot (*.world.npz, ~11 MB) stays on the laptop that paints it; the pgm/yaml
# pair exported from it DOES go, because that pair is the map the board serves.
# maps/map_cache.json is the BOARD's own: the tracker writes down the grid it adopted so that it
# has a map with the laptop gone. It exists only on the board, so --delete removed it on every
# deploy — unseen while a tied RTAB-Map sent a fresh grid seconds later, and on 2026-09-20 a
# restarted tracker beside an untied RTAB-Map came up with "NO MAP AND NO CACHE" and published
# nothing for an hour while the bridge watch restarted the bridges over the silence.
rsync -a --delete --exclude '__pycache__' --exclude 'pepin_src' --exclude 'maps/rec' --exclude 'maps/rtabmap*' --exclude 'maps/**/*.db' --exclude 'maps/_aside*' --exclude 'maps/*.world.npz' --exclude 'maps/world_*.npz*' --exclude 'logs' --exclude 'maps/*.places.yaml' --exclude 'maps/last_pose.json' --exclude 'maps/map_cache.json' "$HERE/" "root@$BOARD:/root/pepin-ros/"
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
