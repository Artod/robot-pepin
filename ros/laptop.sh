#!/bin/bash
# The laptop half of the thin-client split: the planner (with the global costmap) and the goal
# server run here, the board keeps the reflexes. Usage:
#   ros/laptop.sh            start (or restart) the bridge and the laptop-side Nav2 launch
#   ros/laptop.sh stop       stop both
#   ros/laptop.sh logs       follow the launch's output
#   ros/laptop.sh vslam      start (or restart) the camera mapping container beside them, in the
#                            mode `start` last read from the board (ros/.mode): beside the known
#                            map the RTAB-Map database (ros/maps/rtabmap.db) is kept and the map
#                            survives; in SLAM mode the session starts from an empty one
#   ros/laptop.sh vslam --slam | --known-map   force the mode instead of taking the recorded one
#   ros/laptop.sh vslam --camera-only   SLAM without the lidar: the grid is the camera's depth
#   ros/laptop.sh vslam --resume  SLAM from the session's existing database instead of empty
#   ros/laptop.sh vslam --fresh   delete this mode's database before the run
#   ros/laptop.sh vslam --neck    the board's neck node owns base_link -> camera_link (ros/feature.sh
#                            neck on): the camera node here keeps its static edge off
#   ros/laptop.sh kick NODE  restart one node from the mounted sources (seconds, no container restart)
#   ros/laptop.sh vslam      also starts the depth network on the laptop's GPU (ros/depth_host.sh)
#                            and tells the node to use it, when torch's Metal backend is there;
#                            PEPIN_DEPTH_HOST=0 keeps the network on the CPU in the container,
#                            PEPIN_DEPTH_HOST=1 insists on the service (it falls back to the CPU)
# Only `start` talks to the board (its side and its map); stop, logs, vslam and kick never do.
# Prerequisites: the image built here (ros/laptop-build.sh) and the board on side=board
# (ros/thin.sh on). A Docker container on macOS lives behind the VM's NAT, so DDS discovery
# cannot cross to the LAN; zenoh-bridge-ros2dds does the crossing over one TCP connection to
# the board's bridge sidecar (started by ros/thin.sh), and the graph appears here whole.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"  # one multiplexed ssh with a connect timeout: a frozen board fails fast, not silently
NET=pepin-net
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
    echo "the board's bridge did not come back after its restart (ssh root@$BOARD journalctl -u pepin-bridge)"
    return 1
}
# A container's nodes get the time to leave DDS properly (RTAB-Map closes its database): the
# containers run with --stop-signal SIGINT, the signal the launch answers by shutting its nodes
# down (SIGTERM it answers by cancelling itself, and the nodes were SIGKILLed without a dispose:
# their names lingered in the bridge for the DDS lease on every stop). The launch's ghost wait
# (pepin_bringup.ghost_wait) still covers whatever a crash left behind.
stop_gently() { docker stop -t 15 "$@" >/dev/null 2>&1 || true; docker rm -f "$@" >/dev/null 2>&1 || true; }
# The laptop image (ros/laptop-build.sh) carries RTAB-Map on top of the board's image.
image() { docker image inspect pepin-laptop:latest >/dev/null 2>&1 && echo pepin-laptop || echo pepin-ros; }
# The nodes a kick can reach here, the container each lives in and the line it prints once up
# (the kick waits for that line): the Python modules of vslam.launch.py, and the goal server of
# the navigation half (it runs here on side=board only; on side=all: ros/thin.sh kick goal_server).
KICKABLE="camera_stream depth_stream contact_scan depth_fusion rtabmap_frame goal_server"
kick_target() {  # node name -> "container|start-up line"
    case "$1" in
        camera_stream) echo "pepin-vslam|camera stream from " ;;
        depth_stream) echo "pepin-vslam|depth stream up" ;;
        contact_scan) echo "pepin-vslam|contact scan up" ;;
        depth_fusion) echo "pepin-vslam|fusion up: " ;;
        rtabmap_frame) echo "pepin-vslam|rtabmap frame up: " ;;
        goal_server) echo "pepin-laptop|goal server ready on port" ;;
        *) return 1 ;;
    esac
}
now_ms() { perl -MTime::HiRes=time -e 'printf "%.0f", time*1000'; }
# Whether the depth network runs on this laptop's GPU (ros/depth_host.sh) beside the container:
# PEPIN_DEPTH_HOST=1 or 0 decides outright; otherwise torch's Metal backend is asked (~2 s: the
# import), the same question the service itself answers before it falls back to the CPU.
depth_host_wanted() {
    case "${PEPIN_DEPTH_HOST:-}" in 1) return 0 ;; 0) return 1 ;; esac
    (cd "$HERE/.." && uv run --group depth python -c 'import torch; print(torch.backends.mps.is_available())' 2>/dev/null) | grep -qx True
}
# How many participants the bridge admin at $1 lists under node name $2 (its keys read
# @/<zid>/ros2/node/<participant>/<name>): 1 is the live node alone, 2 is the node beside its ghost.
bridge_count() { curl -s -m 3 "$1/@/local/ros2/node/**" | grep -o "/ros2/node/[^/\"]*/$2\"" | wc -l | tr -d ' '; }
MOUNTS=(-v "$HERE/pepin_bringup/pepin_bringup:/ws/install/pepin_bringup/lib/python3.12/site-packages/pepin_bringup:ro"
        -v "$HERE/pepin_bringup/launch:/ws/install/pepin_bringup/share/pepin_bringup/launch:ro"
        -v "$HERE/tools:/tools:ro" -v "$HERE/entrypoint.sh:/pepin_entrypoint.sh:ro"
        -v "$HERE/../src/pepin:/ws/pepin_src/pepin:ro" -v "$HERE/params:/params:ro" -v "$HERE/maps:/maps"
        -v "$HERE/../config:/ws/config:ro")
case "${1:-start}" in
    stop)
        stop_gently pepin-laptop pepin-vslam; docker rm -f pepin-zenoh >/dev/null 2>&1 || true
        [ "${PEPIN_DEPTH_HOST:-}" = 0 ] || "$HERE/depth_host.sh" stop
        echo "laptop side stopped"; exit 0 ;;
    logs)
        exec docker logs -f "pepin-${2:-laptop}" ;;
    kick)
        # One node, not its container. SIGINT is what the launch itself sends at shutdown: the
        # node's main destroys the node and the context, its DDS participant is disposed and the
        # bridge forgets the name at once; the launch respawns the module from the mounted
        # sources two seconds after the EXIT (RESPAWN in the launch files), so the successor
        # never overlaps a ghost of its name. A replaced container is the slow case for exactly
        # that reason: `docker stop` sends SIGTERM, the launch answers it by cancelling itself,
        # the namespace SIGKILLs the nodes without a dispose, and the bridge keeps their names
        # for the DDS lease — the 7-9 s the ghost wait sits through on every container start,
        # before the depth network loads and RTAB-Map starts. Measured start-ups after the
        # respawn pause: camera/fusion/frame 0.3 s, depth 2.1 s (the network), so a kick is
        # 3-5 s here. A crashed node (no dispose) is the one case a respawn meets a ghost: the
        # count below says so, and a second kick after the lease clears it.
        NAME="${2:-}"; TARGET="$(kick_target "$NAME")" || { echo "usage: ros/laptop.sh kick <node>; nodes: $KICKABLE"; exit 2; }
        C="${TARGET%%|*}"; LINE="${TARGET#*|}"
        T0="$(date -u +%FT%TZ)"; MS0="$(now_ms)"
        docker exec "$C" pkill -INT -f "pepin_bringup[./]$NAME" || { echo "no $NAME process in $C (ros/laptop.sh logs ${C#pepin-})"; exit 3; }
        for _ in $(seq 1 240); do
            SEEN="$(docker logs --since "$T0" "$C" 2>&1 | grep -F "$LINE" || true)"
            if [ -n "$SEEN" ]; then
                SEEN="${SEEN%%$'\n'*}"; DT=$(( $(now_ms) - MS0 ))
                printf '%s back in %d.%d s: %s\n' "$NAME" $((DT / 1000)) $((DT % 1000 / 100)) "${SEEN#*]: }"
                case "$(bridge_count http://localhost:8001 "$NAME")" in
                    1) echo "the bridge lists $NAME once: clean" ;;
                    0) ;;  # no bridge admin to ask, or a name it does not list
                    *) echo "WARNING: the bridge lists $NAME beside a ghost of itself: its routes over the bridge drop when the ghost expires; kick again in 10 s" ;;
                esac
                exit 0
            fi
            sleep 0.5
        done
        echo "$NAME did not print '$LINE' within 120 s: ros/laptop.sh logs ${C#pepin-}"; exit 4 ;;
    vslam)
        # Camera + lidar mapping beside the navigation half (ros/pepin_bringup/launch/vslam.launch.py),
        # in one of two modes. Beside a KNOWN map the database is the map: it is kept across
        # restarts (the launch never wipes it) and deleted only here, on request. In SLAM mode
        # the map is what this session builds, in a database of its own, empty unless --resume.
        #
        # Which mode: the flags win, otherwise the one ros/laptop.sh start recorded when it last
        # read the board (ros/.mode) — one source of truth, the board's own /etc/default/pepin-ros,
        # read once on the only path that talks to it. This subcommand asks the board nothing.
        MODE="$(cat "$HERE/.mode" 2>/dev/null || echo vision)"
        if [ "$MODE" = slam ]; then SLAM=true; else SLAM=false; fi
        CAMERA_ONLY=false; RESUME=false; FRESH=false
        # --neck: the board's neck node publishes base_link -> camera_link live (ros/feature.sh
        # neck on), so the camera node's static edge goes off. Explicit on purpose: a wrong guess
        # would be two publishers of one edge; the camera node's report warns of a mismatch.
        STATIC_CAMERA_TF=true
        for arg in ${*:2}; do
            case "$arg" in
                --slam) SLAM=true ;;
                --known-map) SLAM=false ;;
                --camera-only) CAMERA_ONLY=true ;;
                --resume) RESUME=true ;;
                --fresh) FRESH=true ;;
                --neck) STATIC_CAMERA_TF=false ;;
                *) echo "usage: ros/laptop.sh vslam [--slam|--known-map] [--fresh|--resume] [--camera-only] [--neck]"; exit 2 ;;
            esac
        done
        if [ "$FRESH" = true ] && [ "$SLAM" = true ]; then
            rm -f "$HERE"/maps/rtabmap_slam.db "$HERE"/maps/rtabmap_slam.db-*
            echo "vslam: ros/maps/rtabmap_slam.db deleted (a SLAM session starts empty anyway)"
        elif [ "$FRESH" = true ]; then
            rm -f "$HERE"/maps/rtabmap.db "$HERE"/maps/rtabmap.db-*
            echo "vslam: ros/maps/rtabmap.db deleted; RTAB-Map starts an empty map"
        fi
        stop_gently pepin-vslam
        # The depth network on the laptop's GPU (ros/depth_host.sh): 20 ms a frame on Metal
        # against 170 ms on the CPU in the container (2026-09-11), so it is on wherever it can
        # run (depth_host_wanted). The node reads PEPIN_DEPTH_BACKEND (auto: the service, the
        # CPU model while it does not answer; its depth_backend flag switches live) and
        # PEPIN_DEPTH_URL (the host as the container sees it); without them the node runs on
        # the CPU as before.
        DEPTH_ENV=()
        if depth_host_wanted; then
            "$HERE/depth_host.sh" start
            DEPTH_ENV=(-e PEPIN_DEPTH_BACKEND=auto -e "PEPIN_DEPTH_URL=http://host.docker.internal:${PEPIN_DEPTH_PORT:-8790}")
        else
            echo "depth network on the CPU in the container (PEPIN_DEPTH_HOST=1 for the GPU service)"
        fi
        docker run -d --name pepin-vslam --network "$NET" -p 8765:8765 --restart unless-stopped --stop-signal SIGINT "${MOUNTS[@]}" \
            -e ROS_DOMAIN_ID=7 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp ${DEPTH_ENV[@]+"${DEPTH_ENV[@]}"} \
            "$(image)" ros2 launch pepin_bringup vslam.launch.py "board:=$BOARD" "static_camera_tf:=$STATIC_CAMERA_TF" \
            "slam:=$SLAM" "camera_only:=$CAMERA_ONLY" "resume:=$RESUME" >/dev/null
        [ "$SLAM" = true ] \
            && echo "vslam up in SLAM mode (camera_only $CAMERA_ONLY, resume $RESUME, static camera tf $STATIC_CAMERA_TF): the map grows on /map; board must be on ros/thin.sh slam. Foxglove ws://localhost:8765, save with ros/map.sh save NAME" \
            || echo "vslam up beside the known map (static camera tf $STATIC_CAMERA_TF): Foxglove at ws://localhost:8765, ros/laptop.sh logs vslam"
        exit 0 ;;
    start) ;;
    *) echo "usage: ros/laptop.sh [start | stop | logs [vslam] | vslam [--slam|--known-map] [--fresh|--resume] [--camera-only] [--neck] | kick NODE]"; exit 2 ;;
esac
# Which half the board expects: on side=all (ros/thin.sh vision) the board drives by itself and
# this side starts only the bridge — RTAB-Map and the camera come with "ros/laptop.sh vslam".
# A board that does not answer is fatal here, loudly: with pipefail a failed ssh in a command
# substitution once ended this script silently, before the bridge was touched (2026-09-10 20:02).
# No PEPIN_SIDE line in the board's file is side=all (ros/thin.sh vision and off delete it).
SIDE="$(ssh "root@$BOARD" "grep -oE '^PEPIN_SIDE=.*' /etc/default/pepin-ros || echo PEPIN_SIDE=all" 2>/dev/null | cut -d= -f2 || true)"
[ -n "$SIDE" ] || { echo "cannot read the board's side over ssh (root@$BOARD, /etc/default/pepin-ros): is it up?"; exit 1; }
# ...and whether it is mapping from scratch: in SLAM mode the board serves no map, so /map and
# the correction travel the other way and the bridge needs its own allow-list. No line at all is
# false (ros/thin.sh on, vision and off delete it).
SLAM_ON="$(ssh "root@$BOARD" "grep -oE '^PEPIN_SLAM=.*' /etc/default/pepin-ros || echo PEPIN_SLAM=false" 2>/dev/null | cut -d= -f2 || true)"
if [ "$SIDE" = board ]; then
    # The map here chooses the places book, so it must be the board's map, not merely a valid one.
    MAP="${PEPIN_MAP:-$(ssh "root@$BOARD" "grep -oE '^PEPIN_MAP=.*' /etc/default/pepin-ros" 2>/dev/null | cut -d= -f2 || true)}"
    [ -n "$MAP" ] || { echo "the board does not say which map it runs (ros/mode.sh nav MAP first)"; exit 1; }
    CONFIG=zenoh-bridge-laptop.json  # the split: this side publishes the plan and takes goals
    MODE=split
elif [ "$SLAM_ON" = true ]; then
    CONFIG=zenoh-bridge-laptop-slam.json  # this side publishes the only map there is
    MODE=slam
else
    CONFIG=zenoh-bridge-laptop-vision.json  # the board publishes the plan too; this side maps only
    MODE=vision
fi
# The mode the board is in, recorded for `ros/laptop.sh vslam`, which never asks the board itself.
printf '%s\n' "$MODE" > "$HERE/.mode"
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
# The allow-lists (pepin.deployment.bridge_config) are one-way by side AND by mode: a topic
# allowed as a publisher on both sides loops.
echo "laptop bridge: restarting with $CONFIG (board on side=$SIDE, mode $MODE)"
docker run -d --name pepin-zenoh --network "$NET" -p 8001:8000 -v "$HERE/$CONFIG:/config.json:ro" \
    -e ROS_DISTRO=jazzy eclipse/zenoh-bridge-ros2dds:1.7.0 -c /config.json \
    -e "tcp/$BOARD:7447" -d 7 --rest-http-port 8000 >/dev/null
settle_bridge  # BEFORE the containers: their subscriptions must be made against the bridge they will live with
if [ "$SIDE" != board ]; then
    echo "board on side=$SIDE, mode $MODE: it drives by itself; bridge up for the laptop's mapping (ros/laptop.sh vslam)"; exit 0
fi
docker run -d --name pepin-laptop --network "$NET" -p 3337:3337 --restart unless-stopped --stop-signal SIGINT "${MOUNTS[@]}" \
    -e ROS_DOMAIN_ID=7 -e RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
    "$(image)" ros2 launch pepin_bringup nav.launch.py side:=laptop "map:=$MAP" "board:=$BOARD" >/dev/null
echo "laptop side up: planner + goal server (port 3337 here), bridged to $BOARD; ros/laptop.sh logs"
