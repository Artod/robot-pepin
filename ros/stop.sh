#!/bin/bash
# The red button. Stops the wheels within a second no matter what Nav2 is doing:
#   1. ask Nav2 to cancel the task (clean, 3 s at most);
#   2. if that is not confirmed, kill the ROS processes in the container: the base bridge dies with
#      them and the base's own 0.5 s deadman cuts the wheels; then restart the stack (~45 s).
# A polite cancel needs a responsive action server; under load it was not, and a plain restart
# waited up to 15 s for a graceful shutdown while the wheels kept turning (2026-09-07 12:18).
# Usage: ros/stop.sh          (also what goto.sh and tour.sh run on Ctrl-C)
set -uo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
T0=$(date +%s)
OUT=$(ssh -o ConnectTimeout=3 "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh timeout 3 python3 /tools/goto_ros.py cancel" 2>&1)
if echo "$OUT" | grep -q "cancel requested"; then
    echo "navigation task cancelled ($(( $(date +%s) - T0 )) s)"
    exit 0
fi
echo "cancel not confirmed in 3 s — killing the ROS processes (wheels stop on the base's deadman)..."
ssh -o ConnectTimeout=3 "root@$BOARD" "docker exec pepin-ros pkill -9 -f 'component_container_isolated|relocalizer|tof_bridge' >/dev/null 2>&1; echo killed at +$(( $(date +%s) - T0 )) s; systemctl restart pepin-ros" 2>&1 | tail -1
echo "stack restarting, back in ~45 s"
