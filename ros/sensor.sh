#!/bin/bash
# One command per sensor, for the redundancy demo: the lidar and the camera go on and off while
# the robot runs, in the two places that matter, together — what the tracker matches the map
# against (the relocalizer's `sources` flag) and what writes into the costmaps (the per-sensor
# layers of ros/params/nav2_params.yaml, on the local and the global costmap alike).
#   ros/sensor.sh status            what the tracker matches on, which layers are on, what is fresh
#   ros/sensor.sh lidar on|off      the lidar as a tracker source and as lidar_layer
#   ros/sensor.sh lidar off --hard  ... and the driver deactivated: /scan stops, a real absence
#                                   instead of an ignored scan; `lidar on` activates it again
#   ros/sensor.sh camera on|off     the camera as the depth and contact sources and as
#                                   camera_layer and contact_layer — one camera, two readings of
#                                   the same frames: the band 8 cm-1.3 m, and where the floor ends
# Idempotent: only what differs is set, and only what changed is printed. This script writes no
# velocity and restarts nothing. The lifecycle half is refused while a navigation goal runs, and
# refused when that check itself cannot be made: taking /scan away from a moving robot is an
# experiment nobody chose. Exit 1 when something did not answer and was therefore not applied.
#
# Slow on purpose, and here is the bill (board at load 9.5, 2026-09-11): every ros2 CLI call is a
# Python node that must start and discover, and that costs about 10 s there — so this script
# reads a whole `ros2 param dump` per costmap instead of one `ros2 param get` per layer (the
# lesson ros/flags.sh already learned), writes only what differs, and takes "what is fresh" from
# the nodes' own report lines in `docker logs`, which costs nothing at all.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command

TRACKER=relocalizer
LIDAR_DRIVER=/ldlidar_node
LOCAL_COSTMAP=/local_costmap/local_costmap
GLOBAL_COSTMAP=/global_costmap/global_costmap
COSTMAPS="$LOCAL_COSTMAP $GLOBAL_COSTMAP"
SOURCE_ORDER="lidar depth contact"  # pepin.sources' own order, so two lists compare as text.
# Only an ORDER: a name outside it is carried through, never dropped (normalize_sources), and
# tests/unit/test_scripts_parse.py fails when it drifts from src/pepin/sources.py.
LAYER_ORDER="lidar_layer camera_layer contact_layer"
NAV_ACTIONS="navigate_to_pose navigate_through_poses"
REPORT_WINDOW_S=90  # the nodes report every 30 s: three windows, so one missed line is not a verdict
GUARD_TIMEOUT_S=30  # the navigation guard's whole rclpy pass (ros/tools/nav_goal_running.py):
                    # python and rclpy start, the node discovers, and it spins its own window —
                    # about 13 s on the board, and this cap is what a hung DDS runs into

CHANGED=0  # settings this run actually moved (each one is printed)
FAILED=0   # something did not answer and was not applied: the exit status

usage() {
    echo "usage: ros/sensor.sh [status | lidar on|off | lidar off --hard | camera on|off]"
    exit 2
}

ros2_in() {  # SIDE CONTAINER ros2 ...: the ros2 CLI inside the container the node runs in
    local side="$1" container="$2"
    shift 2
    if [ "$side" = laptop ]; then
        docker exec "$container" /pepin_entrypoint.sh "$@"
    else
        ssh "root@$BOARD" "docker exec $container /pepin_entrypoint.sh $(printf '%q ' "$@")"
    fi
}

on_board() { ros2_in board pepin-ros "$@"; }

flags() { "$HERE/flags.sh" "$@"; }  # the one place a feature flag is read or written

board_side() {  # the board's PEPIN_SIDE ("board" when the stack is split, empty when it is whole)
    ssh "root@$BOARD" "grep -oE '^PEPIN_SIDE=[a-z]*' /etc/default/pepin-ros 2>/dev/null | cut -d= -f2" \
        2>/dev/null || true
}

costmap_in() {  # COSTMAP ros2 ...: the local costmap is the controller's and always on the board;
    # the global one follows the planner, which side=board puts in the laptop's container
    # (pepin.deployment.nav_nodes). SIDE is read once per run, by the caller.
    local costmap="$1"
    shift
    if [ "$costmap" = "$GLOBAL_COSTMAP" ] && [ "${SIDE:-}" = board ]; then
        ros2_in laptop pepin-laptop "$@"
    else
        ros2_in board pepin-ros "$@"
    fi
}

param_dump() {  # COSTMAP -> its whole parameter dump as YAML, empty when it did not answer
    costmap_in "$1" ros2 param dump "$1" 2>/dev/null || true
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

board_report() {  # PATTERN -> the board container's last matching log line in the window.
    # Free: a node's own report line is already in the log, so nothing is asked of ROS. Grepped
    # on the board, because 90 s of that log is hundreds of kilobytes over the wifi.
    ssh "root@$BOARD" "docker logs --since ${REPORT_WINDOW_S}s pepin-ros 2>&1 | grep -aE $(printf '%q' "$1") | tail -1" \
        2>/dev/null || true
}

vslam_log() {  # the laptop SLAM container's window of log, empty when it is not running
    docker logs --since "${REPORT_WINDOW_S}s" pepin-vslam 2>&1 || true
}

last_line() {  # LOG PATTERN -> the last matching line of LOG
    printf '%s\n' "$1" | grep -aE "$2" | tail -1 || true
}

sensor_sources() {  # lidar|camera -> the tracker sources this sensor owns
    case "$1" in
        lidar) echo "lidar" ;;
        camera) echo "depth contact" ;;  # one camera read twice: the band, and the floor's end
        *) return 1 ;;
    esac
}

sensor_layers() {  # lidar|camera -> the costmap layers this sensor owns
    case "$1" in
        lidar) echo "lidar_layer" ;;
        camera) echo "camera_layer contact_layer" ;;
        *) return 1 ;;
    esac
}

normalize_sources() {  # CSV -> the same sources, SOURCE_ORDER's first and in its order, then any
    # name this script does not know, in the order it arrived. Nothing is ever dropped: the
    # roster lives in src/pepin/sources.py, SOURCE_ORDER is only an order, and a source added
    # there before this list hears of it must survive a switch of the other sensor untouched.
    local csv="$1" seen="," out="" item
    for item in $SOURCE_ORDER; do
        case ",$csv," in *",$item,"*) out="$out,$item"; seen="$seen$item," ;; esac
    done
    for item in ${csv//,/ }; do
        case "$seen" in *",$item,"*) ;; *) out="$out,$item"; seen="$seen$item," ;; esac
    done
    echo "${out#,}"
}

sources_after() {  # CSV SENSOR on|off -> the tracker's source list once SENSOR is that way: this
    # sensor's own sources put in or taken out, every other name carried through as it was
    local csv="$1" sensor="$2" state="$3" mine kept="" item
    mine=" $(sensor_sources "$sensor") "
    for item in ${csv//,/ }; do
        case "$mine" in *" $item "*) ;; *) kept="$kept,$item" ;; esac
    done
    if [ "$state" = on ]; then
        for item in $(sensor_sources "$sensor"); do kept="$kept,$item"; done
    fi
    normalize_sources "${kept#,}"
}

tracker_sources() {  # the tracker's sources flag as a comma list; exit 1 when it did not answer
    local reply
    reply="$(flags get "$TRACKER" sources 2>/dev/null)" || return 1
    case "$reply" in
        *"value is:"*) printf '%s' "$reply" | sed -n 's/^.*value is: *//p' | tr -d ' \n' ;;
        *) return 1 ;;
    esac
}

driver_state() {  # the lidar driver's lifecycle state as one word; "?" when it did not answer
    local state
    state="$(on_board ros2 lifecycle get "$LIDAR_DRIVER" 2>/dev/null | sed 's/ \[.*//' || true)"
    echo "${state:-?}"
}

nav_goals() {  # ACTION=yes|no|? for every navigation action, in one rclpy pass on the board.
    # The tool is piped in from the laptop's copy, so this guard needs no deploy and works on a
    # board whose /tools are older than this script. Empty when the pass did not run at all.
    local remote="docker exec -i pepin-ros /pepin_entrypoint.sh"  # -i: the tool arrives on stdin
    ssh "root@$BOARD" "$remote timeout $GUARD_TIMEOUT_S python3 - $NAV_ACTIONS" \
        < "$HERE/tools/nav_goal_running.py" 2>/dev/null || true
}

goal_running() {  # LINE ACTION -> yes, no, or ? (the pass could not tell, or never answered)
    local word
    for word in $1; do
        case "$word" in
            "$2=yes") echo yes; return 0 ;;
            "$2=no") echo no; return 0 ;;
            "$2="*) echo "?"; return 0 ;;  # the pass's own "?": it could not see
        esac
    done
    echo "?"  # the line carries no word for this action: the pass did not run
}

refuse_if_navigating() {  # the lifecycle half never runs under a goal, nor under a blind guard
    local action verdict line
    echo "  checking that no navigation goal is running"
    echo "  (up to $GUARD_TIMEOUT_S s: an rclpy pass that must start and discover on the board)"
    line="$(nav_goals)"
    for action in $NAV_ACTIONS; do
        verdict="$(goal_running "$line" "$action")"
        case "$verdict" in
            no) ;;
            yes) echo "  refused: a navigation goal is running ($action); ros/go.sh cancel first"
                 exit 1 ;;
            *) echo "  refused: the guard could not read /$action/_action/status, so it cannot"
               echo "  see whether the robot is driving; ros/watch.sh, or leave --hard off"
               exit 1 ;;
        esac
    done
}

apply_sources() {  # SENSOR on|off: the tracker's sources flag; prints what changed
    local sensor="$1" state="$2" have want
    have="$(tracker_sources)" || {
        echo "  $TRACKER did not answer: its sources are unchanged"
        FAILED=1
        return 0
    }
    have="$(normalize_sources "$have")"
    want="$(sources_after "$have" "$sensor" "$state")"
    [ "$have" = "$want" ] && return 0
    flags set "$TRACKER" sources "$want" >/dev/null
    echo "  $TRACKER sources ${have:-(none)} -> ${want:-(none)}"
    CHANGED=$((CHANGED + 1))
    case ",$want," in
        *,lidar,*) ;;
        ,,) echo "  no source left: the tracker dead reckons until one comes back" ;;
        *) echo "  no lidar among the sources: against the lidar's own map the camera alone loses"
           echo "  it within seconds (scratch/camera_only_localization.py) — the costmap half of"
           echo "  the demo stands, the tracker half does not" ;;
    esac
}

apply_layers() {  # SENSOR true|false: the sensor's layers on both costmaps; prints what changed
    local sensor="$1" want="$2" costmap dump layer now
    for costmap in $COSTMAPS; do
        dump="$(param_dump "$costmap")"
        if [ -z "$dump" ]; then
            echo "  $costmap did not answer a parameter dump: its layers are unchanged"
            FAILED=1
            continue
        fi
        for layer in $(sensor_layers "$sensor"); do
            now="$(layer_in_dump "$dump" "$layer")"
            if [ -z "$now" ]; then
                echo "  $costmap has no $layer: unchanged (ros/params/nav2_params.yaml)"
                FAILED=1
                continue
            fi
            [ "$now" = "$want" ] && continue
            costmap_in "$costmap" ros2 param set "$costmap" "$layer.enabled" "$want" >/dev/null
            echo "  $costmap $layer.enabled $now -> $want"
            CHANGED=$((CHANGED + 1))
        done
    done
}

apply_driver() {  # activate|deactivate WAS: the lidar driver's lifecycle, already known to differ
    local want="$1" was="$2"
    on_board ros2 lifecycle set "$LIDAR_DRIVER" "$want" >/dev/null
    echo "  $LIDAR_DRIVER $was -> $([ "$want" = activate ] && echo active || echo inactive)"
    CHANGED=$((CHANGED + 1))
}

driver_wanted() {  # SENSOR on|off HARD -> activate, deactivate or nothing for the lidar driver.
    # --hard is the deep half of an OFF and of nothing else: an `on` always ends with /scan
    # running, whatever else is on the command line (switch() refuses that line anyway).
    local sensor="$1" state="$2" hard="$3"
    [ "$sensor" = lidar ] || return 0
    if [ "$state" = on ]; then
        echo activate  # a hard off is undone by the plain on: never leave /scan stopped by accident
    elif [ "$hard" = --hard ]; then
        echo deactivate
    fi
}

switch() {  # SENSOR on|off [--hard]: the flag, the layers and, for the lidar, the driver
    local sensor="$1" state="$2" hard="${3:-}" want was
    [ "$state" = on ] || [ "$state" = off ] || usage
    case "$hard" in
        "") ;;
        --hard)
            [ "$sensor" = lidar ] || { echo "--hard belongs to the lidar only"; exit 2; }
            # "on --hard" reads as "on, and mean it"; it used to mean "on everywhere, then stop
            # the driver" — the tracker and both costmaps told to use a lidar that no longer
            # publishes. There is no deep half of an on, so the line is refused, not guessed.
            [ "$state" = off ] || {
                echo "--hard belongs to 'lidar off': it stops the driver, and 'lidar on' is what"
                echo "starts it again"
                exit 2
            } ;;
        *) usage ;;
    esac
    echo "$sensor $state${hard:+ (hard)}:"
    SIDE="$(board_side)"
    want="$(driver_wanted "$sensor" "$state" "$hard")"
    was=""
    if [ -n "$want" ]; then
        was="$(driver_state)"
        case "$want:$was" in
            activate:active | deactivate:inactive) want="" ;;  # already there
            *:'?')
                [ "$hard" != --hard ] || {
                    echo "  refused: $LIDAR_DRIVER did not answer a lifecycle get, so --hard"
                    echo "  cannot know what it is turning off; is the stack up? ros/thin.sh"
                    exit 1
                }
                echo "  $LIDAR_DRIVER did not answer a lifecycle get: the driver is left alone"
                FAILED=1
                want="" ;;
        esac
    fi
    # The guard runs before anything is applied: a half-applied switch is worse than a refusal.
    [ -z "$want" ] || refuse_if_navigating
    apply_sources "$sensor" "$state"
    apply_layers "$sensor" "$([ "$state" = on ] && echo true || echo false)"
    [ -z "$want" ] || apply_driver "$want" "$was"
    [ "$CHANGED" -gt 0 ] || echo "  already so"
}

status() {  # what the tracker matches on, what the costmaps take, and what each node last said
    local costmap dump layer line value tracker vslam depth contact sensor
    SIDE="$(board_side)"
    tracker="$(board_report "$TRACKER\\]: tracker:")"
    value="$(printf '%s' "$tracker" | sed -n 's/.*[ ,]sources=\([a-z,]*\).*/\1/p')"
    if [ -z "$tracker" ]; then
        echo "tracker sources: ? ($TRACKER printed no report in ${REPORT_WINDOW_S} s; ros/watch.sh)"
    elif [ -z "$value" ]; then
        echo "tracker sources: ? (its report line carries no sources= flag; ros/flags.sh list $TRACKER)"
    else
        echo "tracker sources: $value"
    fi
    for costmap in $COSTMAPS; do
        dump="$(param_dump "$costmap")"
        line=""
        for layer in $LAYER_ORDER; do
            value="$(layer_in_dump "$dump" "$layer")"
            case "$value" in true) value=on ;; false) value=off ;; *) value="?" ;; esac
            line="$line  $layer=$value"
        done
        echo "costmap $costmap:$line"
    done
    echo "lidar driver $LIDAR_DRIVER: $(driver_state)"
    vslam="$(vslam_log)"  # one fetch, two greps: the camera's two nodes live in one container
    depth="$(last_line "$vslam" ']: depth: ')"
    contact="$(last_line "$vslam" ']: contact: ')"
    echo "what each node last said (within ${REPORT_WINDOW_S} s; a silent one is a dead sensor):"
    for line in "$tracker" "$depth" "$contact"; do
        printf '  %s\n' "${line:-(nothing)}"
    done
    for sensor in lidar camera; do
        echo "$sensor owns: sources $(sensor_sources "$sensor" | tr ' ' ','), layers $(sensor_layers "$sensor" | tr ' ' ',')"
    done
}

case "${1:-status}" in
    status) [ $# -le 1 ] || usage; status ;;
    lidar | camera) { [ $# -ge 2 ] && [ $# -le 3 ]; } || usage; switch "$1" "$2" "${3:-}" ;;
    *) usage ;;
esac
exit "$FAILED"
