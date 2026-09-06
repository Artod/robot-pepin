#!/bin/bash
# Run the ROS 2 container on the board (used by board/pepin-ros.service and by hand). Usage:
#   ros/run.sh                      -> interactive shell inside the container
#   ros/run.sh ros2 launch pepin_bringup robot.launch.py
# Development mounts: the Python package, launch files, tools, params, maps and our pepin
# library come from /root/pepin-ros on the host, over the paths baked into the image — so an
# edit needs `ros/sync.sh` (rsync + container restart, ~20 s), not an image rebuild. The image
# is rebuilt (ros/build.sh, container stopped) only when the Dockerfile changes.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
LIDAR="$(readlink -f /dev/lidar 2>/dev/null || echo /dev/ttyUSB0)"
SITE=/ws/install/pepin_bringup/lib/python3.12/site-packages/pepin_bringup
TTY=""
[ -t 0 ] && TTY="-it"  # a terminal when run by hand; none under systemd
I2C=""
# The IMU (0x68) and the ToF sensors (0x30-0x32) share this bus; docker run refuses to start
# when a --device is missing, so a board without it simply gets no bus.
[ -e /dev/i2c-2 ] && I2C="--device /dev/i2c-2"
# shellcheck disable=SC2086
exec docker run --rm $TTY \
    --network host --ipc host --cap-add SYS_NICE \
    --device "$LIDAR:/dev/lidar" $I2C \
    -v "$HERE/pepin_bringup/pepin_bringup:$SITE:ro" \
    -v "$HERE/pepin_bringup/launch:/ws/install/pepin_bringup/share/pepin_bringup/launch:ro" \
    -v "$HERE/tools:/tools:ro" \
    -v "$HERE/entrypoint.sh:/pepin_entrypoint.sh:ro" \
    -v "$HERE/pepin_src:/ws/pepin_src:ro" \
    -v "$HERE/params:/params:ro" \
    -v "$HERE/maps:/maps" \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}" \
    --name pepin-ros \
    pepin-ros "$@"
