#!/bin/bash
# Save the map slam_toolbox is building as ros/maps/NAME.{pgm,yaml} on the board and fetch it here.
# Usage: ros/savemap.sh NAME
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
NAME="${1:?map name}"
HERE="$(cd "$(dirname "$0")" && pwd)"
if ! ssh "root@$BOARD" "docker logs pepin-ros 2>&1 | grep -q 'slam_toolbox\]: Activating'"; then
    echo "slam_toolbox is not running in the container: nothing to save (ros/mode.sh slam first)"; exit 1
fi
# 1) the occupancy grid for map_server/AMCL, 2) slam_toolbox's pose graph for its localization mode later
ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh ros2 run nav2_map_server map_saver_cli -f /maps/$NAME --ros-args -p save_map_timeout:=10.0 2>&1 | tail -2"
ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh ros2 service call /slam_toolbox/serialize_map slam_toolbox/srv/SerializePoseGraph \"{filename: /maps/$NAME}\" 2>&1 | tail -1"
scp "root@$BOARD:/root/pepin-ros/maps/$NAME.yaml" "root@$BOARD:/root/pepin-ros/maps/$NAME.pgm" "$HERE/maps/"
echo "saved: $HERE/maps/$NAME.yaml — navigate with: ros/run.sh ros2 launch pepin_bringup nav.launch.py map:=/maps/$NAME.yaml"
