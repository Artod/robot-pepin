#!/bin/bash
# Drive the robot through Nav2 on this Mac (ros/laptop.sh nav), with feedback. Usage:
#   ros/goto.sh NAME               drive to a remembered place (the graph's book first, the map
#                                  file's second, with a warning)
#   ros/goto.sh X Y [YAW_DEG]      drive to map coordinates (meters, degrees)
#   ros/goto.sh cancel             cancel every goal on both navigators and say what came of it
#                                  (ros/stop.sh is the hard stop that also brakes)
#   ros/goto.sh where              the pose right now (map -> base_link, from the goal server)
#   ros/goto.sh planner NAME       the planner for the next goals (navfn|lattice|theta|smac|hybrid)
#   ros/goto.sh mark NAME          remember where the robot stands as NAME: a labelled RTAB-Map
#                                  node plus the cart's offset from it, so the place rides the node
#                                  when a loop closure bends the map
#   ros/goto.sh places             list both books, each entry saying which one it came from
#   ros/goto.sh seed X Y [YAW]     after placing the robot by hand: tell RTAB-Map where it is
#   ros/goto.sh round [NAME]       one full turn in place, judged by the gyro, recorded as NAME
#   ros/goto.sh move NAME SEG...   measured legs without the planner: f0.40 = 0.40 m straight
#                                  (negative = back), t90 = 90 deg left; recorded as NAME
# A goal goes to the goal server on 127.0.0.1:3337 (pepin.goal_link); Ctrl-C, a closed terminal or
# a kill cancels it. The exit status is the drive's verdict (0: reached), so
# `ros/goto.sh printer && ros/goto.sh home` chains. round and move go only while the goal server
# says nothing navigates, and Ctrl-C stops them on the board.
# Every goal leaves, under ros/maps/rec/:
#   <stamp>_goto.log        the goal's own lines (accepted, progress, result, arrival)
#   <stamp>_goto.nav2.log   Nav2's reasons as they happen ("nav2|": planner refusals, controller
#                           failures, recoveries and why) and the behaviour tree's transitions
#                           ("bt|", ros/tools/bt_watch.py), also printed here
#   <stamp>_goto_cam.mkv    the head camera (ros/clip.sh), loud when there is no picture
#   NNNN_<utc>Z_<place>.jsonl  the numbered tape the run recorder beside Nav2 wrote, named in the
#                           log (a bag under PEPIN_RECORDER=bag, turned into the same tape here)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"
NAV=pepin-macnav  # = pepin.deployment.NAV_CONTAINER
GOAL_PORT="${PEPIN_GOAL_PORT:-3337}"  # = pepin_bringup.goal_server.PORT
CANCEL_S=5  # a cancel's answer is waited for this long: the first loud line comes within it
goal_link() { PYTHONPATH="$HERE/../src" python3 -m pepin.goal_link --port "$GOAL_PORT" "$@"; }
base_stop() {  # the base server's own stop, believed from its state stream (pepin.red_button)
    PYTHONPATH="$HERE/../src" python3 -m pepin.red_button --host "$BOARD" --port "${PEPIN_BASE_PORT:-3336}"
}
goto_ros() { docker exec "$NAV" /pepin_entrypoint.sh python3 /tools/goto_ros.py "$@"; }
places_book() {  # the map file's book: <the map pepin-macnav was started with>.places.yaml
    local map
    map="$(docker inspect -f '{{range .Args}}{{println .}}{{end}}' "$NAV" 2>/dev/null | sed -n 's/^map:=//p')"
    if [ -n "$map" ]; then echo "${map%.yaml}.places.yaml"; else echo /maps/places.yaml; fi
}
cancel() {  # the goal server's cancel, asked twice; then goto_ros.py's, in Nav2's container
    goal_link --timeout "$CANCEL_S" cancel && return 0
    goal_link --timeout "$CANCEL_S" cancel && return 0
    echo "!! the goal server did not confirm the cancel (above): goto_ros.py cancel in $NAV"
    goto_ros cancel
}
# THE MEASURED MOTIONS run on the board, beside the wheels and the gyro, and write /cmd_vel past
# every guard of Nav2's. So they go only when the goal server says nothing navigates — a goal of
# anyone's, the pivot included — and refuse when it does not answer at all. Their own check of
# the action status on the board fails open now that the status comes over the WiFi. The tool
# writes its pid in the container, where the signal traps below and ros/stop.sh find it.
MOTION=""
motion_stop() {
    [ -n "$MOTION" ] || return 0
    echo
    echo "stopping the measured motion on the board"
    # shellcheck disable=SC2016  # expanded on the board, inside its container
    ssh -o ConnectTimeout=3 "root@$BOARD" \
        'docker exec pepin-ros sh -c '"'"'p=$(cat /tmp/pepin_motion.pid 2>/dev/null) && kill -KILL "$p"'"'"
    base_stop
    kill "$MOTION" 2>/dev/null
    MOTION=""
}
motion() {  # TOOL ARGS...
    local tool="$1" answer remote
    shift
    answer="$(goal_link where 2>&1)"
    case "$answer" in
        *'"navigating": false'*) ;;
        *'"navigating": true'*)
            echo "refused: a navigation goal is running; ros/goto.sh cancel first"; exit 2 ;;
        *) echo "refused: the goal server did not say Nav2 is idle: ${answer:0:200}"; exit 2 ;;
    esac
    trap 'motion_stop; exit 130' INT
    trap 'motion_stop; exit 129' HUP
    trap 'motion_stop; exit 143' TERM
    # shellcheck disable=SC2016  # $$ and "$@" are the container shell's
    remote="docker exec -i pepin-ros /pepin_entrypoint.sh sh -c $(printf %q 'echo $$ > /tmp/pepin_motion.pid; exec python3 - "$@"') sh $(printf '%q ' "$@")"
    # shellcheck disable=SC2086  # the ssh options are words (ros/lib.sh)
    command ssh $PEPIN_SSH_OPTS "root@$BOARD" "$remote" <"$HERE/tools/$tool" &
    MOTION=$!
    wait "$MOTION"
    local rc=$?
    MOTION=""
    exit "$rc"
}
USAGE="usage: ros/goto.sh NAME | X Y [YAW] | cancel | where | planner NAME | mark NAME | places | seed X Y [YAW] | round [NAME] | move NAME SEG..."
case "${1:-}" in
    "") echo "$USAGE"; exit 2 ;;
    cancel) cancel; exit ;;
    where) goal_link where; exit ;;
    planner) goal_link planner "${2:?$USAGE}"; exit ;;
    # A PLACE LIVES IN RTAB-MAP'S GRAPH: goto_ros.py asks the laptop's places node (/places/mark,
    # answered on /places/marked), which labels the graph node the cart is at and stores the
    # cart's offset from it. --places names the map file's book, the fallback it falls back TO.
    mark | places | seed) goto_ros --places "$(places_book)" "$@"; exit ;;
    round) motion turn_full.py "${2:-round}" ;;
    move) motion move.py "${@:2}" ;;
esac

REC="${PEPIN_REC_DIR:-$HERE/maps/rec}"  # the containers' /maps/rec
mkdir -p "$REC"
STAMP=$(date +%Y%m%d_%H%M%S)
LOG="$REC/${STAMP}_goto.log"
NAVLOG="$REC/${STAMP}_goto.nav2.log"
CAM="$REC/${STAMP}_goto_cam.mkv"
GOAL="" CLIP="" STREAMS=""
INTERRUPTED=0
FINISHED=0
finish() {  # everything of this drive closed and named, always, once
    if [ "$FINISHED" = 1 ]; then return 0; fi  # a signal runs this, then EXIT runs it again
    FINISHED=1
    set +e
    if [ "$INTERRUPTED" = 1 ]; then
        # A cancel and nothing else, asked again until confirmed: never ros/stop.sh from here,
        # whose hard branch takes Nav2 down (and until 2026-10-01 restarted the board, zeroing
        # the odometry: 2026-09-28 21:48, 2026-09-29 00:17). The red button is typed.
        echo
        echo "interrupted: cancelling every goal"
        cancel || echo "!! the cancel was NOT confirmed: if the cart still moves, ros/stop.sh is the hard stop"
    fi
    if [ -n "$GOAL" ]; then  # the goal's last lines (its result after a cancel), 5 s at most
        for _ in 1 2 3 4 5 6 7 8 9 10; do kill -0 "$GOAL" 2>/dev/null || break; sleep 0.5; done
        kill "$GOAL" 2>/dev/null
        wait "$GOAL" 2>/dev/null
    fi
    sleep 1  # Nav2's last words about this drive reach the log
    [ -z "$STREAMS" ] || kill -TERM -- "-$STREAMS" 2>/dev/null
    if [ -n "$CLIP" ]; then
        kill -TERM "$CLIP" 2>/dev/null
        wait "$CLIP" 2>/dev/null  # its closing line: the clip's size, or why there is none
    fi
    # The numbered tape the recorder opened for this goal, named in the goal's log. ros/maps is
    # the container's /maps, so the tape is already here; a bag (a DIRECTORY, PEPIN_RECORDER=bag)
    # is turned into the very same tape.
    local taped
    taped=$(grep -o 'taped /maps/rec/[^ ]*' "$LOG" 2>/dev/null | tail -1 | cut -d' ' -f2)
    case "$taped" in
        "") if grep -q " accepted$" "$LOG" 2>/dev/null; then
                echo "!! NO TAPE: the run recorder opened none for this goal (ros/laptop.sh nav logs)"
            fi ;;
        *.jsonl) echo "numbered tape: ros$taped" ;;
        *)
            echo "numbered bag: ros$taped; converting it to a tape"
            if pepin_bag_to_tape "$taped" >>"$LOG" 2>&1; then
                echo "numbered tape: ros$taped.jsonl"
            else
                echo "!! the bag is here but not converted: see the end of $LOG"
            fi ;;
    esac
    echo "recorded: ${REC#"$(dirname "$HERE")/"}/${STAMP}_goto.log, ${STAMP}_goto.nav2.log$([ -s "$CAM" ] && echo ", ${STAMP}_goto_cam.mkv")"
}
# Every way this script can end cancels the goal: Ctrl-C, a closed terminal (HUP), a kill (TERM).
# Set before anything is started, so no helper outlives a signal that came early.
on_signal() { INTERRUPTED=1; finish; exit "$1"; }
trap 'on_signal 130' INT
trap 'on_signal 129' HUP
trap 'on_signal 143' TERM
trap finish EXIT
# Each helper in a process group of its own: one signal ends every member, and the operator's
# Ctrl-C reaches this script alone (python's setpgrp: job control needs a terminal). A command,
# not a function: a function run with & is a subshell, and $! would be that subshell, not the
# group. Both helpers end by themselves when this script is gone (SIGKILL included).
OWN_GROUP=(python3 -c 'import os, sys; os.setpgrp(); os.execvp(sys.argv[1], sys.argv[1:])')
"${OWN_GROUP[@]}" bash "$HERE/clip.sh" "$CAM" "${PEPIN_CAMERA_STREAM:-http://$BOARD:8080/stream}" &
CLIP=$!
# Nav2's own reasons beside the goal's events, as they happen: every planner refusal, controller
# failure and recovery the behaviour tree runs (and why), prefixed "nav2|"; and the tree's own
# transitions, prefixed "bt|", from the one watcher that runs as long as the container does.
NAV2_REASONS="planner_server.*(failed|Start occupied|Goal occupied|exceeded|no valid|timed out|Aborting)|controller_server.*(Failed to make progress|Path is empty|Optimizer fail|[Cc]ollision|Aborting|Reached the goal|not.*valid|fail|error|Error)|behavior_server.*(Running|failed|collision|Exceeded|completed)|bt_navigator.*(Goal failed|Goal succeeded|aborted|canceled)|IsPathValid|is_path_valid"
pepin_bt_watch || echo "!! no behaviour-tree watcher in $NAV: no bt| lines for this goal"
touch "$PEPIN_BT_LOG"
export NAV NAV2_REASONS PEPIN_BT_LOG
"${OWN_GROUP[@]}" bash -c '
    { docker logs -f --since 1s "$NAV" 2>&1 | grep --line-buffered -E "$NAV2_REASONS" \
        | sed -u -E "s/^\[[^]]*\] \[[A-Z]+\] \[[0-9.]+\] /nav2| /" &
      tail -n0 -F "$PEPIN_BT_LOG" 2>/dev/null | grep --line-buffered "^bt|" &
      while kill -0 "$2" 2>/dev/null; do sleep 1; done; kill -TERM 0; } | tee -a "$1"' \
    streams "$NAVLOG" "$$" &
STREAMS=$!
# The goal in the background and waited for: bash runs a trap only between commands, and a
# foreground goal_link would hold a HUP or a TERM for as long as the drive lasts (three goto.sh
# outlived their TERM that way on 2026-09-13). A background job ignores SIGINT, so the cancel
# is the trap's alone.
goal_link --log "$LOG" go "$@" &
GOAL=$!
wait "$GOAL"
VERDICT=$?
GOAL=""
finish
exit "$VERDICT"
