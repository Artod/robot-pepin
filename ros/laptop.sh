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
case "${1:-start}" in
    stop)
        docker rm -f pepin-laptop pepin-zenoh >/dev/null 2>&1 || true; echo "laptop side stopped"; exit 0 ;;
    logs)
        exec docker logs -f pepin-laptop ;;
esac
docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
docker rm -f pepin-laptop pepin-zenoh >/dev/null 2>&1 || true
# The board's bridge must be alive before this side connects: its REST admin answers when its
# zenoh runtime does (a wedged bridge stays "Up" and answers nothing — 2026-09-09).
for _ in $(seq 1 30); do
    curl -s -m 3 "http://$BOARD:8000/@/local/router" | grep -q '"ros2dds"' && break
    sleep 2
done
curl -s -m 3 "http://$BOARD:8000/@/local/router" | grep -q '"ros2dds"' || { echo "the board's bridge does not answer on :8000 (ros/thin.sh on, then wait for it)"; exit 1; }
# ROS_DISTRO matters: without it the bridge assumes Iron. Router mode on both sides, this one
# connecting to the board's: the pairing measured to pass samples (peer mode here did not).
docker run -d --name pepin-zenoh --network "$NET" -e ROS_DISTRO=jazzy eclipse/zenoh-bridge-ros2dds:1.5.1 \
    -e "tcp/$BOARD:7447" -d 7 --rest-http-port 8000 >/dev/null
# The library is copied into the build context the same way sync.sh does for the board.
mkdir -p "$HERE/pepin_src" && rsync -a --delete --exclude __pycache__ "$HERE/../src/pepin/" "$HERE/pepin_src/pepin/"
SITE=/ws/install/pepin_bringup/lib/python3.12/site-packages/pepin_bringup
docker run -d --name pepin-laptop --network "$NET" \
    -p 3337:3337 \
    -v "$HERE/pepin_bringup/pepin_bringup:$SITE:ro" \
    -v "$HERE/pepin_bringup/launch:/ws/install/pepin_bringup/share/pepin_bringup/launch:ro" \
    -v "$HERE/tools:/tools:ro" \
    -v "$HERE/entrypoint.sh:/pepin_entrypoint.sh:ro" \
    -v "$HERE/pepin_src:/ws/pepin_src:ro" \
    -v "$HERE/params:/params:ro" \
    -v "$HERE/maps:/maps" \
    -e ROS_DOMAIN_ID=7 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    pepin-ros ros2 launch pepin_bringup nav.launch.py side:=laptop "map:=$MAP" >/dev/null
echo "laptop side up: planner + goal server (port 3337 here), bridged to $BOARD; ros/laptop.sh logs"
