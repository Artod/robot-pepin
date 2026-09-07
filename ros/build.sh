#!/bin/bash
# Sync ros/ and our Python library to the board and build the image there. Usage: ros/build.sh
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
HERE="$(cd "$(dirname "$0")" && pwd)"
rsync -a --delete --exclude '__pycache__' --exclude 'pepin_src' --exclude 'maps/rec' --exclude 'logs' --exclude 'maps/*.places.yaml' --exclude 'maps/last_pose.json' "$HERE/" "root@$BOARD:/root/pepin-ros/"
ssh "root@$BOARD" "mkdir -p /root/pepin-ros/pepin_src/pepin"
rsync -a --delete --exclude "__pycache__" "$HERE/../src/pepin/" "root@$BOARD:/root/pepin-ros/pepin_src/pepin/"
# Stop the robot's container first: a build next to Nav2 drove the board into swap (load 100).
# The full build log stays on the board (logs/build_<time>.log); the terminal gets the tail.
STAMP=$(date +%Y%m%d_%H%M%S); T0=$(date +%s)
ssh "root@$BOARD" "systemctl stop pepin-ros; mkdir -p /root/pepin-ros/logs; cd /root/pepin-ros && DOCKER_BUILDKIT=1 docker build -t pepin-ros -f Dockerfile . > logs/build_$STAMP.log 2>&1; echo BUILD_EXIT=\$? >> logs/build_$STAMP.log; tail -25 logs/build_$STAMP.log; systemctl start pepin-ros"
echo "build took $(( $(date +%s) - T0 )) s; full log on the board: /root/pepin-ros/logs/build_$STAMP.log"
