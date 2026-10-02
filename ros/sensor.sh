#!/bin/bash
# One command per sensor, for the redundancy demo: the lidar and the camera go on and off while
# the robot runs, in what writes into the costmaps (the per-sensor layers of
# ros/params/nav2_params.yaml, on the local and the global costmap alike). The board tracker's
# `sources` half of this switch is on the tag alt/tracker-2026-09-22.
#   ros/sensor.sh status            which layers are on, what is fresh
#   ros/sensor.sh lidar on|off      the lidar as lidar_layer
#   ros/sensor.sh lidar off --hard  ... and the driver deactivated: /scan stops, a real absence
#                                   instead of an ignored scan; `lidar on` activates it again
#   ros/sensor.sh camera on|off     the camera as camera_layer and contact_layer — one camera,
#                                   two readings of the same frames: the band 8 cm-1.3 m, and
#                                   where the floor ends
#
# MUTING, the other half, added 2026-09-15: `on|off` above is the CONSUMER end — the costmaps
# stop listening while the sensor keeps publishing. `mute` is the PUBLISHER end: the
# node that owns the sensor stops sending, and every consumer meets what a dead sensor really
# looks like — silence, a sensor_timeout, a TF that stops moving — without a restart, without
# losing the other live flags, and with the sensor itself still read.
#   ros/sensor.sh mute|unmute SENSOR    imu odom vo lidar
#   ros/sensor.sh status                ... also prints each sensor's mute state
# Each sensor is one live flag of the node that publishes it (ros/flags.sh), except the lidar:
# our own node in its chain is `scan_filter` (laser_filters, external) and it has no flag of
# ours, so `mute lidar` is the documented consumer set instead — lidar_layer off on both costmaps,
# which is exactly `lidar off` above. A relay node
# on the board that could drop /scan is not the answer (CLAUDE.md rule 20: the board carries only
# what is real-time critical); `lidar off --hard` is the real absence when one is wanted.
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

LIDAR_DRIVER=/ldlidar_node
LOCAL_COSTMAP=/local_costmap/local_costmap
GLOBAL_COSTMAP=/global_costmap/global_costmap
COSTMAPS="$LOCAL_COSTMAP $GLOBAL_COSTMAP"
LAYER_ORDER="lidar_layer camera_layer contact_layer"
MUTE_ORDER="imu odom vo lidar"  # what `mute` knows, in the order status prints
NAV_ACTIONS="navigate_to_pose navigate_through_poses"
REPORT_WINDOW_S=90  # the nodes report every 30 s: three windows, so one missed line is not a verdict
GUARD_TIMEOUT_S=30  # the navigation guard's whole rclpy pass (ros/tools/nav_goal_running.py):
                    # python and rclpy start, the node discovers, and it spins its own window —
                    # about 13 s on the board, and this cap is what a hung DDS runs into

CHANGED=0  # settings this run actually moved (each one is printed)
FAILED=0   # something did not answer and was not applied: the exit status

usage() {
    echo "usage: ros/sensor.sh [status | lidar on|off | lidar off --hard | camera on|off"
    echo "                      | mute|unmute imu|odom|vo|lidar]"
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

costmap_in() {  # COSTMAP ros2 ...: both costmaps live in the Mac's Nav2 container (ros/laptop.sh nav)
    shift
    ros2_in laptop pepin-macnav "$@"
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

vslam_log() {  # the laptop SLAM container's window of log, empty when it is not running
    docker logs --since "${REPORT_WINDOW_S}s" pepin-vslam 2>&1 || true
}

last_line() {  # LOG PATTERN -> the last matching line of LOG
    printf '%s\n' "$1" | grep -aE "$2" | tail -1 || true
}

sensor_layers() {  # lidar|camera -> the costmap layers this sensor owns
    case "$1" in
        lidar) echo "lidar_layer" ;;
        camera) echo "camera_layer contact_layer" ;;
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

switch() {  # SENSOR on|off [--hard]: the layers and, for the lidar, the driver
    local sensor="$1" state="$2" hard="${3:-}" want was
    [ "$state" = on ] || [ "$state" = off ] || usage
    case "$hard" in
        "") ;;
        --hard)
            [ "$sensor" = lidar ] || { echo "--hard belongs to the lidar only"; exit 2; }
            # "on --hard" reads as "on, and mean it"; it used to mean "on everywhere, then stop
            # the driver" — both costmaps told to use a lidar that no longer publishes. There is no deep half of an on, so the line is refused, not guessed.
            [ "$state" = off ] || {
                echo "--hard belongs to 'lidar off': it stops the driver, and 'lidar on' is what"
                echo "starts it again"
                exit 2
            }
            # REFUSED since 2026-09-24: deactivating the LD19 driver ABORTS its process ("*** bit
            # out of range 0 - FD_SETSIZE on fd_set ***", exit -6, a select() on the descriptor its
            # own close invalidated). In the shared sensors_container of before that took the wheels
            # and the IMU down for hours; in the split one (sensor_split) the lidar process simply
            # respawns 2 s later and the driver is active again, so a hard off can hold neither way.
            # A camera-only cart is 'lidar off' (the layers) plus sensor_pack sources=camera; a dead
            # lidar is simulated by killing the lidar_container process (it respawns).
            # PEPIN_LIDAR_HARD=allow keeps the old path reachable (CLAUDE.md rule 19) for a driver
            # that survives its own deactivate.
            if [ "${PEPIN_LIDAR_HARD:-refuse}" != allow ]; then
                echo "refused: --hard deactivates the LD19 driver, which aborts its whole process (2026-09-24)."
                echo "  Camera-only: ros/sensor.sh lidar off, then sensor_pack sources=camera on the laptop."
                echo "  A dead lidar: kill the lidar_container process on the board (it respawns in 2 s)."
                echo "  The old path, knowingly: PEPIN_LIDAR_HARD=allow ros/sensor.sh lidar off --hard"
                exit 2
            fi
            ;;
        *) usage ;;
    esac
    echo "$sensor $state${hard:+ (hard)}:"
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
    apply_layers "$sensor" "$([ "$state" = on ] && echo true || echo false)"
    [ -z "$want" ] || apply_driver "$want" "$was"
    if [ "$sensor" = camera ] && [ "$state" = on ]; then
        # 2026-09-13: with these two layers on, the near-field marks (the contact ring at
        # 1.2-1.5 m, the depth band beside the hull) left 2 cm of clearance ahead and stalled
        # two drives in a row. Switching them on is a demo of the costmaps, not a drive-safe
        # state — said here, where the operator is, and not only in the README.
        echo "  note: the camera's costmap layers stalled two drives on 2026-09-13 (2 cm"
        echo "  clearance from near-hull marks); switch them off before driving to a goal"
    fi
    [ "$CHANGED" -gt 0 ] || echo "  already so"
}

# ---------------------------------------------------------------------------------------------
# Muting: the publisher end. One live flag per sensor, set through ros/flags.sh so the value is
# checked by the flag itself before any host is touched, and nothing is restarted.

sensor_node() {  # SENSOR -> the node that publishes it; exit 1 for a sensor with no flag of ours
    case "$1" in
        imu | odom) echo base_bridge ;;
        vo) echo visual_odometry ;;
        *) return 1 ;;
    esac
}

sensor_flag() {  # SENSOR -> the flag of that node whose off value is this sensor's silence
    case "$1" in
        imu) echo imu_publish ;;
        odom) echo odom_publish ;;
        vo) echo vo_publish ;;
        *) return 1 ;;
    esac
}

sensor_value() {  # SENSOR mute|unmute -> the value its flag takes
    case "$1:$2" in
        *:mute) echo false ;;
        *) echo true ;;
    esac
}

muted_by() {  # SENSOR VALUE -> yes, no or ? : whether that live value is this sensor's silence
    local value
    value="$(printf '%s' "$2" | tr '[:upper:]' '[:lower:]')"
    case "$value" in
        false) echo yes ;;
        true) echo no ;;
        *) echo "?" ;;
    esac
}

flag_value() {  # NODE FLAG -> the value the running node holds; exit 1 when it did not answer
    local reply
    reply="$(flags get "$1" "$2" 2>/dev/null)" || return 1
    case "$reply" in
        *"value is:"*) printf '%s' "$reply" | sed -n 's/^.*value is: *//p' | tr -d " \n" ;;
        *) return 1 ;;
    esac
}

consequence() {  # SENSOR mute|unmute: what a consumer should now see, in one or two lines
    if [ "$2" = unmute ]; then
        case "$1" in
            imu) echo "  /imu/data_raw is back within one IMU period (50 Hz)" ;;
            odom) echo "  /odom is back on the next state line (20 Hz), and the transform with it" ;;
            vo) echo "  /vo is back at about 9.4 poses/s and the EKF fuses it again" ;;
            lidar) echo "  lidar_layer feeds both costmaps again" ;;
        esac
        return 0
    fi
    case "$1" in
        imu) echo "  /imu/data_raw goes silent: the EKF loses its yaw-rate source, and with odom0's"
             echo "  vyaw on (ros/params/ekf.yaml, since 2026-09-15) the heading follows the wheels —"
             echo "  which over-report a turn in place by 10-25 % on carpet" ;;
        odom) echo "  /odom goes silent, and odom -> base_link with it: past the EKF's sensor_timeout"
              echo "  of 0.5 s the filter has no velocity measurement left (ax/ay are off), so the"
              echo "  pose stops advancing while the gyro still turns it. The wheels still obey" ;;
        vo) echo "  /vo stops: the board's EKF is the wheels and the gyro, exactly as it was before"
            echo "  2026-09-14; the node still measures and still reports" ;;
    esac
}

mute_lidar() {  # mute|unmute: the documented consumer set, because the lidar has no flag of ours
    echo "$1 lidar: our own node in the lidar's chain is scan_filter (laser_filters, external)"
    echo "and it has no flag of ours; the mute is therefore the consumer set — lidar_layer off on"
    echo "both costmaps (ros/sensor.sh lidar off)."
    echo "For a real absence of /scan: ros/sensor.sh lidar off --hard"
    switch lidar "$([ "$1" = mute ] && echo off || echo on)"
}

mute() {  # mute|unmute SENSOR: the publisher's flag, and what a consumer should now see
    local action="$1" sensor="$2" node flag want have
    case " $MUTE_ORDER " in *" $sensor "*) ;; *) usage ;; esac
    if [ "$sensor" = lidar ]; then
        mute_lidar "$action"
        consequence lidar "$action"
        return 0
    fi
    node="$(sensor_node "$sensor")"
    flag="$(sensor_flag "$sensor")"
    want="$(sensor_value "$sensor" "$action")"
    echo "$action $sensor: $node $flag"
    have="$(flag_value "$node" "$flag")" || {
        echo "  $node did not answer about $flag: it is unchanged"
        echo "  (is the node up? ros/flags.sh list $node)"
        FAILED=1
        return 0
    }
    case "$action:$(muted_by "$sensor" "$have")" in
        mute:yes | unmute:no) echo "  already so (${have:-(none)})"; return 0 ;;
    esac
    flags set "$node" "$flag" "$want" >/dev/null
    echo "  $node $flag ${have:-(none)} -> ${want:-(none)}"
    CHANGED=$((CHANGED + 1))
    consequence "$sensor" "$action"
}

mute_status() {  # every sensor's mute state, read from the live flags one by one
    local sensor node flag have state
    echo "muted (the publisher's own flag; a muted sensor is read and not sent):"
    for sensor in $MUTE_ORDER; do
        if [ "$sensor" = lidar ]; then
            echo "  lidar: see lidar_layer above (no publisher flag of ours)"
            continue
        fi
        node="$(sensor_node "$sensor")"
        flag="$(sensor_flag "$sensor")"
        if have="$(flag_value "$node" "$flag")"; then
            state="$(muted_by "$sensor" "$have")"
        else
            have=""
            state="?"
        fi
        case "$state" in
            yes) echo "  $sensor: MUTED ($node $flag=${have:-(none)})" ;;
            no) echo "  $sensor: on ($node $flag=${have:-(none)})" ;;
            *) echo "  $sensor: ? ($node did not answer about $flag; ros/flags.sh list $node)" ;;
        esac
    done
}

status() {  # what the costmaps take, and what each node last said
    local costmap dump layer line value vslam depth contact sensor
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
    for line in "$depth" "$contact"; do
        printf '  %s\n' "${line:-(nothing)}"
    done
    for sensor in lidar camera; do
        echo "$sensor owns: layers $(sensor_layers "$sensor" | tr ' ' ',')"
    done
    mute_status
}

case "${1:-status}" in
    status) [ $# -le 1 ] || usage; status ;;
    lidar | camera) { [ $# -ge 2 ] && [ $# -le 3 ]; } || usage; switch "$1" "$2" "${3:-}" ;;
    mute | unmute) [ $# -eq 2 ] || usage; mute "$1" "$2" ;;
    *) usage ;;
esac
exit "$FAILED"
