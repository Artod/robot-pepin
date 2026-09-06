#!/bin/bash
# The red button. Stops the robot no matter what Nav2 is doing, in two layers:
#   1. cancel the navigation task on the board (clean: the controller sends zero and stays quiet);
#   2. if that cannot be confirmed within 8 s, restart the ROS unit: the container's base bridge
#      dies, the base's own 0.5 s deadman cuts the wheels, the stack is back in ~40 s.
# Usage: ros/stop.sh          (also what tour.sh runs on Ctrl-C)
set -uo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
OUT=$(ssh -o ConnectTimeout=5 "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh timeout 8 python3 /tools/goto_ros.py cancel" 2>&1)
echo "$OUT" | tail -1
if ! echo "$OUT" | grep -q "cancel requested"; then
    echo "cancel not confirmed — restarting the ROS unit (wheels stop on the base's deadman)"
    ssh -o ConnectTimeout=5 "root@$BOARD" "systemctl restart pepin-ros" && echo "unit restarted; stack back in ~40 s"
fi
