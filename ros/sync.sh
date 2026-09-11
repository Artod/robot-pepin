#!/bin/bash
# Push code changes to the robot without rebuilding the image: rsync ros/, src/pepin and
# config/, then restart the sensors container (systemd unit). Nav2 must be started again
# afterwards: ros/nav.sh
# Usage: ros/sync.sh [--no-restart]
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
HERE="$(cd "$(dirname "$0")" && pwd)"
# maps/rec and logs are written BY the board and fetched to the laptop; pushing them back would ship
# hundreds of MB of camera video over WiFi onto the SD card (it did, 2026-09-06: a 4-minute sync).
rsync -a --delete --exclude '__pycache__' --exclude 'pepin_src' --exclude 'maps/rec' --exclude 'maps/rtabmap*' --exclude 'logs' --exclude 'maps/*.places.yaml' --exclude 'maps/last_pose.json' "$HERE/" "root@$BOARD:/root/pepin-ros/"
ssh "root@$BOARD" "mkdir -p /root/pepin-ros/pepin_src/pepin /root/pepin-ros/pepin_src/config"
rsync -a --delete --exclude '__pycache__' "$HERE/../src/pepin/" "root@$BOARD:/root/pepin-ros/pepin_src/pepin/"
# The mounts (config/lidar.json, config/imu.json) beside the library: the container mounts
# /root/pepin-ros/pepin_src as /ws/pepin_src and the launch reads them from there at start
# (pepin.deployment.config_file) — the same files the laptop reads from its checkout.
rsync -a --delete "$HERE/../config/" "root@$BOARD:/root/pepin-ros/pepin_src/config/"
if [ "${1:-}" != "--no-restart" ]; then
    ssh "root@$BOARD" "systemctl restart pepin-ros && sleep 8 && systemctl is-active pepin-ros"
fi
