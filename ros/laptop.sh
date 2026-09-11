#!/bin/bash
# The laptop half of the thin-client split: the planner (with the global costmap) and the goal
# server run here, the board keeps the reflexes. Usage:
#   ros/laptop.sh            start (or restart) the bridge and the laptop-side Nav2 launch
#   ros/laptop.sh stop       stop both
#   ros/laptop.sh logs       follow the launch's output
# Prerequisites: the image built here (ros/laptop-build.sh) and the board on side=board
# (ros/thin.sh on). A Docker container on macOS lives behind the VM's NAT, so DDS discovery
# cannot cross to the LAN; zenoh-bridge-ros2dds does the crossing over one TCP connection to
# the board's bridge sidecar (started by ros/thin.sh), and the graph appears here whole.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
NET=pepin-net
# The map here chooses the places book, so it must be the board's map, not merely a valid one.
MAP="${PEPIN_MAP:-$(ssh "root@$BOARD" "grep -oE 'PEPIN_MAP=.*' /etc/default/pepin-ros" 2>/dev/null | cut -d= -f2 || true)}"
[ -n "$MAP" ] || { echo "the board does not say which map it runs (ros/mode.sh nav MAP first)"; exit 1; }
# The library is mounted live, like the ROS package: a copy went stale whenever a container was
# restarted by the bridge watch rather than by this script (the depth node died on an import of a
# function that existed in src/ but not in the copy, 2026-09-11).
# The board's bridge is restarted once this side's bridge is up and BEFORE this side's containers
# start: a subscription made against one bridge does not follow it through a restart (a costmap
# kept a deaf transform listener for 139 s, run 0148), and a bridge restarted after the
# containers breaks exactly those subscriptions. Later restarts of the board's bridge are handled
# by pepin_bringup.bridge_watch inside each container.
settle_bridge() {
    ssh "root@$BOARD" "systemctl restart pepin-bridge" 2>/dev/null
    for _ in $(seq 1 30); do
        curl -s -m 3 "http://$BOARD:8000/@/local/router" | grep -q '"ros2dds"' && return 0
        sleep 3
    done
    echo "the board's bridge did not come back after its restart"
}
# A container's nodes get the time to leave DDS properly (RTAB-Map closes its database): a
# killed node lingers in the bridge for its ten-second lease, and its successor's launch waits
# that ghost out (pepin_bringup.ghost_wait) before it starts.
stop_gently() { docker stop -t 15 "$@" >/dev/null 2>&1 || true; docker rm -f "$@" >/dev/null 2>&1 || true; }
# The laptop image (ros/laptop-build.sh) carries RTAB-Map on top of the board's image.
IMG=pepin-ros; docker image inspect pepin-laptop:latest >/dev/null 2>&1 && IMG=pepin-laptop
MOUNTS=(-v "$HERE/pepin_bringup/pepin_bringup:/ws/install/pepin_bringup/lib/python3.12/site-packages/pepin_bringup:ro"
        -v "$HERE/pepin_bringup/launch:/ws/install/pepin_bringup/share/pepin_bringup/launch:ro"
        -v "$HERE/tools:/tools:ro" -v "$HERE/entrypoint.sh:/pepin_entrypoint.sh:ro"
        -v "$HERE/../src/pepin:/ws/pepin_src/pepin:ro" -v "$HERE/params:/params:ro" -v "$HERE/maps:/maps"
        -v "$HERE/../config:/ws/config:ro")
case "${1:-start}" in
    stop)
        stop_gently pepin-laptop pepin-vslam; docker rm -f pepin-zenoh >/dev/null 2>&1 || true
        echo "laptop side stopped"; exit 0 ;;
    logs)
        exec docker logs -f "pepin-${2:-laptop}" ;;
    vslam)
        # Camera + lidar SLAM beside the navigation half (ros/pepin_bringup/launch/vslam.launch.py).
        stop_gently pepin-vslam
        docker run -d --name pepin-vslam --network "$NET" -p 8765:8765 --restart unless-stopped "${MOUNTS[@]}" \
            -e ROS_DOMAIN_ID=7 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
            "$IMG" ros2 launch pepin_bringup vslam.launch.py "board:=$BOARD" >/dev/null
        echo "vslam up (RTAB-Map + camera + depth): Foxglove at ws://localhost:8765, ros/laptop.sh logs vslam"; exit 0 ;;
esac
# Which half the board expects: on side=all (ros/thin.sh vision) the board drives by itself and
# this side starts only the bridge — RTAB-Map and the camera come with "ros/laptop.sh vslam".
# "|| true": with pipefail a slow ssh (the board's stack just restarted) would end this script
# here, silently, before the bridge is touched (2026-09-10 20:02).
SIDE="$(ssh -o ConnectTimeout=15 "root@$BOARD" "grep -oE 'PEPIN_SIDE=.*' /etc/default/pepin-ros" 2>/dev/null | cut -d= -f2 || true)"
SIDE="${SIDE:-all}"
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
stop_gently pepin-laptop; docker rm -f pepin-zenoh >/dev/null 2>&1 || true
# The board's bridge must be alive before this side connects: its REST admin answers when its
# zenoh runtime does (a wedged bridge stays "Up" and answers nothing — 2026-09-09).
for _ in $(seq 1 30); do
    curl -s -m 3 "http://$BOARD:8000/@/local/router" | grep -q '"ros2dds"' && break
    sleep 2
done
curl -s -m 3 "http://$BOARD:8000/@/local/router" | grep -q '"ros2dds"' || { echo "the board's bridge does not answer on :8000 (ros/thin.sh on, then wait for it)"; exit 1; }
# ROS_DISTRO matters: without it the bridge assumes Iron. Router mode on both sides, this one
# connecting to the board's: the pairing measured to pass samples (peer and client here did not).
echo "laptop bridge: restarting with $(basename "$HERE/zenoh-bridge-laptop.json")"
docker run -d --name pepin-zenoh --network "$NET" -p 8001:8000 -v "$HERE/zenoh-bridge-laptop.json:/config.json:ro" \
    -e ROS_DISTRO=jazzy eclipse/zenoh-bridge-ros2dds:1.7.0 -c /config.json \
    -e "tcp/$BOARD:7447" -d 7 --rest-http-port 8000 >/dev/null
settle_bridge  # BEFORE the containers: their subscriptions must be made against the bridge they will live with
SITE=/ws/install/pepin_bringup/lib/python3.12/site-packages/pepin_bringup
if [ "$SIDE" != board ]; then
    echo "board on side=$SIDE: it drives by itself; bridge up for the laptop's SLAM (ros/laptop.sh vslam)"; exit 0
fi
docker run -d --name pepin-laptop --network "$NET" -p 3337:3337 --restart unless-stopped "${MOUNTS[@]}" \
    -e ROS_DOMAIN_ID=7 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    "$IMG" ros2 launch pepin_bringup nav.launch.py side:=laptop "map:=$MAP" "board:=$BOARD" >/dev/null
echo "laptop side up: planner + goal server (port 3337 here), bridged to $BOARD; ros/laptop.sh logs"
