#!/bin/bash
# The camera grid A/B (2026-09-24), live and reversible from the laptop: the costmaps DRAW the
# volume's current occupied columns (camera_grid_layer, a StaticLayer) instead of accumulating
# /depth_marks in camera_layer (an ObstacleLayer) — ros/README.md, "Camera grid A/B".
#   ros/camera_grid.sh status   depth_fusion's grid_out, both costmaps' two camera layers, and
#                               the fusion's last "grid:" report
#   ros/camera_grid.sh on       grid_out true; once each grid is seen on its topic, that
#                               costmap's camera_grid_layer on and camera_layer off
#   ros/camera_grid.sh off      camera_layer on and camera_grid_layer off on both costmaps, then
#                               grid_out false (the node sends one empty grid on its way out)
# ORDER IS THE GUARD. A StaticLayer enabled before its first grid is never current, and the
# controller and the planner then answer every goal "Costmap timed out waiting for update"; so a
# layer is enabled only after its latched grid has been read back from the topic, and a costmap
# whose grid did not come keeps camera_layer and says so (exit 1). /camera_grid_map needs the
# map (grid_map_topic) and this laptop's map -> odom.
# Idempotent: each costmap is read with one parameter dump and only what differs is set. Nothing
# here commands a velocity or restarts anything. Exit 1 when something was not applied.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command

NODE=depth_fusion
VSLAM=pepin-vslam  # the container depth_fusion runs in (ros/tools/flags_doc.py where depth_fusion)
LOCAL_COSTMAP=/local_costmap/local_costmap
GLOBAL_COSTMAP=/global_costmap/global_costmap
WAIT_S="${PEPIN_GRID_WAIT_S:-15}"  # a latched grid is read back at once; this covers discovery
REPORT_WINDOW_S=90  # the fusion reports every 30 s
FAILED=0

usage() { echo "usage: ros/camera_grid.sh [status | on | off]"; exit 2; }

flags() { "$HERE/flags.sh" "$@"; }  # the one place a feature flag is read or written

ros2_in() {  # SIDE CONTAINER ros2 ...: the ros2 CLI inside the container the node runs in
    local side="$1" container="$2"
    shift 2
    if [ "$side" = laptop ]; then
        docker exec "$container" /pepin_entrypoint.sh "$@"
    else
        ssh "root@$BOARD" "docker exec $container /pepin_entrypoint.sh $(printf '%q ' "$@")"
    fi
}

board_side() {  # the board's PEPIN_SIDE ("board" when the stack is split, empty when it is whole)
    ssh "root@$BOARD" "grep -oE '^PEPIN_SIDE=[a-z]*' /etc/default/pepin-ros 2>/dev/null | cut -d= -f2" \
        2>/dev/null || true
}

costmap_in() {  # COSTMAP ros2 ...: the local costmap is always the board's; the global one follows
    # the planner, which the split (PEPIN_SIDE=board) puts in the laptop's container
    local costmap="$1"
    shift
    if [ "$costmap" = "$GLOBAL_COSTMAP" ] && [ "${SIDE:-}" = board ]; then
        ros2_in laptop pepin-laptop "$@"
    else
        ros2_in board pepin-ros "$@"
    fi
}

layer_in_dump() {  # DUMP LAYER -> the layer's enabled value (true/false), empty when not in it
    printf '%s\n' "$1" | awk -v key="$2:" '
        !inside && $1 == key { inside = 1; depth = match($0, /[^ ]/); next }
        inside {
            here = match($0, /[^ ]/)
            if (here > 0 && here <= depth) exit
            if ($1 == "enabled:") { print $2; exit }
        }'
}

grid_topic() {  # COSTMAP -> the topic its camera_grid_layer draws (ros/params/nav2_params.yaml)
    [ "$1" = "$LOCAL_COSTMAP" ] && echo /camera_grid || echo /camera_grid_map
}

grid_seen() {  # TOPIC -> 0 when a latched grid is read back from it within WAIT_S
    local out
    out="$(ros2_in laptop "$VSLAM" timeout "$WAIT_S" ros2 topic echo --once \
        --qos-durability transient_local --qos-reliability reliable "$1" \
        nav_msgs/msg/OccupancyGrid --field info 2>/dev/null || true)"
    case "$out" in *width*) return 0 ;; *) return 1 ;; esac
}

set_layers() {  # COSTMAP on|off: the two switches of one costmap, only where they differ. The
    # camera never leaves both layers at once: on, the grid layer comes up before camera_layer
    # goes (and camera_layer stays when the grid layer could not); off, camera_layer comes back
    # before the grid layer goes.
    local costmap="$1" state="$2" dump
    dump="$(costmap_in "$costmap" ros2 param dump "$costmap" 2>/dev/null || true)"
    if [ -z "$dump" ]; then
        echo "  $costmap did not answer a parameter dump: unchanged"
        FAILED=1
        return 0
    fi
    if [ "$state" = on ]; then
        apply_layer "$costmap" camera_grid_layer "$dump" true || return 0
        apply_layer "$costmap" camera_layer "$dump" false || true
    else
        apply_layer "$costmap" camera_layer "$dump" true || true
        apply_layer "$costmap" camera_grid_layer "$dump" false || true
    fi
}

apply_layer() {  # COSTMAP LAYER DUMP WANT -> 1 when the costmap has no such layer
    local costmap="$1" layer="$2" want="$4" now
    now="$(layer_in_dump "$3" "$layer")"
    if [ -z "$now" ]; then
        echo "  $costmap has no $layer: Nav2 runs an older nav2_params.yaml (restart it)"
        FAILED=1
        return 1
    fi
    if [ "$now" = "$want" ]; then
        echo "  $costmap $layer.enabled already $want"
        return 0
    fi
    costmap_in "$costmap" ros2 param set "$costmap" "$layer.enabled" "$want" >/dev/null
    echo "  $costmap $layer.enabled $now -> $want"
}

on() {
    SIDE="$(board_side)"
    flags set "$NODE" grid_out true >/dev/null
    echo "  $NODE grid_out -> true"
    for costmap in "$LOCAL_COSTMAP" "$GLOBAL_COSTMAP"; do
        topic="$(grid_topic "$costmap")"
        if ! grid_seen "$topic"; then
            echo "  no grid on $topic within $WAIT_S s: $costmap keeps camera_layer"
            [ "$topic" = /camera_grid_map ] && echo "  (it needs the map and map -> odom: ros/camera_grid.sh status)"
            FAILED=1
            continue
        fi
        set_layers "$costmap" on
    done
}

off() {
    SIDE="$(board_side)"
    for costmap in "$LOCAL_COSTMAP" "$GLOBAL_COSTMAP"; do
        set_layers "$costmap" off
    done
    flags set "$NODE" grid_out false >/dev/null
    echo "  $NODE grid_out -> false"
}

status() {
    local costmap dump line layer value report
    SIDE="$(board_side)"
    value="$(flags get "$NODE" grid_out 2>/dev/null | sed -n 's/^.*value is: *//p' || true)"
    echo "$NODE grid_out: ${value:-? (the node did not answer)}"
    for costmap in "$LOCAL_COSTMAP" "$GLOBAL_COSTMAP"; do
        dump="$(costmap_in "$costmap" ros2 param dump "$costmap" 2>/dev/null || true)"
        line=""
        for layer in camera_layer camera_grid_layer; do
            value="$(layer_in_dump "$dump" "$layer")"
            case "$value" in true) value=on ;; false) value=off ;; *) value="?" ;; esac
            line="$line  $layer=$value"
        done
        echo "costmap $costmap:$line  (draws $(grid_topic "$costmap"))"
    done
    report="$(docker logs --since "${REPORT_WINDOW_S}s" "$VSLAM" 2>&1 | grep -aE ']: fusion: ' | tail -1 \
        | grep -oE 'grid: [^;]*;[^;]*' || true)"
    echo "last report: ${report:-(no fusion report in ${REPORT_WINDOW_S} s)}"
}

case "${1:-status}" in
    status) [ $# -le 1 ] || usage; status ;;
    on) [ $# -eq 1 ] || usage; on ;;
    off) [ $# -eq 1 ] || usage; off ;;
    *) usage ;;
esac
exit "$FAILED"
