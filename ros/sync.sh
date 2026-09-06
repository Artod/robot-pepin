#!/bin/bash
# Push code changes to the robot without rebuilding the image: rsync ros/ and src/pepin, then
# restart the sensors container (systemd unit). Nav2 must be started again afterwards: ros/nav.sh
# Usage: ros/sync.sh [--no-restart]
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
HERE="$(cd "$(dirname "$0")" && pwd)"
rsync -a --delete --exclude '__pycache__' --exclude 'pepin_src' "$HERE/" "root@$BOARD:/root/pepin-ros/"
ssh "root@$BOARD" "mkdir -p /root/pepin-ros/pepin_src/pepin"
rsync -a --delete --exclude '__pycache__' "$HERE/../src/pepin/" "root@$BOARD:/root/pepin-ros/pepin_src/pepin/"
if [ "${1:-}" != "--no-restart" ]; then
    ssh "root@$BOARD" "systemctl restart pepin-ros && sleep 8 && systemctl is-active pepin-ros"
fi
