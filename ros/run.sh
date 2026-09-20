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
# The one directory this container writes for the board's own systemd: the laptop's request to
# restart the zenoh bridge lands here as a file (pepin_bringup.bridge_kick) and
# board/pepin-bridge-kick.path turns it into `systemctl restart pepin-bridge`. Same path inside
# and outside, and on tmpfs: nothing survives a reboot and nothing touches the SD card. NOT
# under /root/pepin-ros — ros/sync.sh rsyncs that tree with --delete.
mkdir -p /run/pepin
# Which middleware this container speaks (ros/lib.sh has the whole story; the value comes from
# /etc/default/pepin-ros through board/pepin-ros.service). Unset or "cyclone" leaves everything
# exactly as it was: the image's own ENV is rmw_cyclonedds_cpp and nothing below is added.
# "zenoh" swaps the image for one with rmw_zenoh_cpp in it and adds three variables:
#   RMW_IMPLEMENTATION    the middleware itself
#   PEPIN_RMW             read by the launches, which then start no bridge watch
#   ZENOH_ROUTER_CHECK_ATTEMPTS=0  do not block on the router at start-up. The session's own
#       connect retry is infinite for a peer (connect/timeout_ms -1, exit_on_failure false in
#       the shipped session config), so a node started before pepin-zrouter comes up stays
#       alive and joins when the router appears, instead of dying on a start-order race.
# No session config is passed: the shipped default (peer, connect tcp/localhost:7447, listen
# tcp/localhost:0) is already the shape the board wants — nodes talk to each other directly
# over the host loopback, and only what leaves the board goes through the router.
IMAGE="${PEPIN_IMAGE:-pepin-ros}"
RMWENV=""
if [ "${PEPIN_RMW:-cyclone}" = zenoh ]; then
    IMAGE="${PEPIN_IMAGE:-pepin-ros:zenoh}"
    RMWENV="-e RMW_IMPLEMENTATION=rmw_zenoh_cpp -e PEPIN_RMW=zenoh -e ZENOH_ROUTER_CHECK_ATTEMPTS=0"
fi
# shellcheck disable=SC2086
# Not auto-removed: a stopped container keeps its log until the unit's ExecStartPre has saved it.
# --stop-signal SIGINT beside the image's own STOPSIGNAL (ros/Dockerfile): SIGINT is what ros2
# launch answers by shutting its nodes down, SIGTERM it answers by cancelling itself and the
# nodes are SIGKILLed mid-write — every stop of this container used to end in the 137 the unit
# still tolerates. The window is board/pepin-ros.service's ExecStop, ros/lib.sh's number.
exec docker run $TTY \
    --network host --ipc host --cap-add SYS_NICE \
    --stop-signal SIGINT \
    --device "$LIDAR:/dev/lidar" $I2C \
    -v "$HERE/pepin_bringup/pepin_bringup:$SITE:ro" \
    -v "$HERE/pepin_bringup/launch:/ws/install/pepin_bringup/share/pepin_bringup/launch:ro" \
    -v "$HERE/tools:/tools:ro" \
    -v "$HERE/entrypoint.sh:/pepin_entrypoint.sh:ro" \
    -v "$HERE/pepin_src:/ws/pepin_src:ro" \
    -v "$HERE/params:/params:ro" \
    -v "$HERE/maps:/maps" \
    -v /run/pepin:/run/pepin \
    -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}" \
    $RMWENV \
    --name pepin-ros \
    "$IMAGE" "$@"
