#!/bin/bash
# Sync ros/ and our Python library to the board and build the image there. Usage: ros/build.sh
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
HERE="$(cd "$(dirname "$0")" && pwd)"
rsync -a --delete --exclude '__pycache__' "$HERE/" "root@$BOARD:/root/pepin-ros/"
ssh "root@$BOARD" "mkdir -p /root/pepin-ros/pepin_src/pepin"
rsync -a --delete --exclude "__pycache__" "$HERE/../src/pepin/" "root@$BOARD:/root/pepin-ros/pepin_src/pepin/"
# Stop the robot's container first: a build next to Nav2 drove the board into swap (load 100).
ssh "root@$BOARD" 'systemctl stop pepin-ros; cd /root/pepin-ros && DOCKER_BUILDKIT=1 docker build -t pepin-ros -f Dockerfile . 2>&1 | tail -1; systemctl start pepin-ros'
