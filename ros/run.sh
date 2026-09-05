#!/bin/bash
# Run the ROS 2 container on the board. Usage:
#   ros/run.sh                      -> interactive shell inside the container
#   ros/run.sh ros2 launch pepin_bringup robot.launch.py
#   ros/run.sh ros2 launch pepin_bringup nav.launch.py map:=/maps/lap3.yaml
# Host networking (DDS + foxglove_bridge on 8765 reach the wifi directly), the lidar's serial
# device, and the repo's ros/maps and ros/params mounted read-only.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LIDAR="$(readlink -f /dev/lidar 2>/dev/null || echo /dev/ttyUSB0)"
exec docker run --rm -it \
    --network host --ipc host \
    --device "$LIDAR:/dev/lidar" \
    -v "$HERE/maps:/maps:ro" \
    -v "$HERE/params:/params:ro" \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}" \
    --name pepin-ros \
    pepin-ros "$@"
