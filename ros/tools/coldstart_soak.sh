#!/bin/bash
# The cold-start acceptance test this stack did not have. Usage:
#
#   ros/tools/coldstart_soak.sh [N]        N cold starts of the board half, default 10
#
# THE ROBOT DOES NOT MOVE. This restarts processes and reads logs: no velocity is commanded, no
# goal is sent, and the run refuses to begin while a navigation goal is running (the same guard
# ros/sensor.sh uses, ros/tools/nav_goal_running.py) — a stack restarted under a goal leaves the
# cart driving on a dead controller.
#
# What it measures, and why this and nothing else. Nav2's planner_server activates by starting
# its global costmap, and Costmap2DROS::start() waits for the first full update with no timeout.
# A RangeSensorLayer whose Range messages cannot be transformed costs the WHOLE
# transform_tolerance per message inside that update (tf2's canTransform never short-circuits),
# and the ToF layers receive 15 Hz each: the backlog grows faster than it drains, every message
# ages past the 10 s TF cache, and the update never ends. planner_server then sits in
# "Activating" for ever — goals accepted, nothing planned, services timing out — on some starts
# and not others, which is why this is a SOAK and not a single run (4 of 7 starts on 2026-09-21;
# the arithmetic is scratch/nav2_hang/wedge_gain.py, the fix is tof_bridge's dynamic_mounts and
# tf_gate flags plus pepin-zrouter.service's wait for the router's port).
#
# Per start it records:
#   activation -> bond   seconds from "Activating planner_server" to "connected with bond",
#                        or TIMEOUT when the bond never came within ACTIVATION_TIMEOUT_S
#   can't transform      "Range sensor layer can't transform" lines: the wedge running. Any is a
#                        failure — one of them is already a blocked costmap update. Since
#                        2026-09-21 no costmap lists a RangeSensorLayer at all (the whiskers
#                        arrive as scan fans for an ObstacleLayer, tof_bridge's range_as), so a
#                        line here means the board is running an older ros/params/nav2_params.yaml
#   invalid frame        `Invalid frame ID` lines: a consumer that was handed a frame it does not
#                        have. Recorded, not judged: the lidar's own frame is late on healthy
#                        starts too, and only the ToF ones wedge a costmap
#
# A start passes when the bond arrived and there was no range-layer complaint. The exit status is
# 0 only when every start passed; anything else prints the failing rows and exits 1.
set -euo pipefail
HERE="$(cd "$(dirname "$0")/.." && pwd)"   # ros/
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"  # the multiplexed ssh; N restarts over one handshake

RUNS="${1:-10}"
ACTIVATION_TIMEOUT_S="${PEPIN_SOAK_TIMEOUT_S:-120}"  # a healthy board bonds in well under a minute
POLL_S=5
SETTLE_S=8          # what restart.sh gives systemd before it believes the unit is up
NAV_ACTIONS="navigate_to_pose navigate_through_poses"
GUARD_TIMEOUT_S=30  # the rclpy pass on the board: python, discovery, its own window

case "$RUNS" in
    ''|*[!0-9]*|0) echo "usage: ros/tools/coldstart_soak.sh [N]  (N cold starts, default 10)"; exit 2 ;;
esac

# restart.sh's two log readers, kept here rather than sourced: that script RUNS a restart when it
# is sourced, and this tool must own its own. Grepping happens on the board — a minute of that
# log is hundreds of kilobytes over the WiFi.
board_count() {  # WINDOW_S PATTERN -> how many lines match, or ? when the log could not be read
    local answer
    answer="$(ssh "root@$BOARD" "docker logs --since ${1}s pepin-ros 2>&1 | grep -acE $(printf '%q' "$2")" 2>/dev/null || true)"
    [[ "$answer" =~ ^[0-9]+$ ]] && printf '%s\n' "$answer" || printf '?\n'
}
board_stamp() {  # WINDOW_S PATTERN -> the epoch of the LAST matching line, empty when there is none
    ssh "root@$BOARD" "docker logs --since ${1}s pepin-ros 2>&1 | grep -aE $(printf '%q' "$2") | tail -1" 2>/dev/null \
        | sed -n 's/.*\[\([0-9]\{10\}\.[0-9]*\)\].*/\1/p'
}

refuse_if_navigating() {  # a stack restarted under a goal leaves the cart driving on nothing
    local line word verdict action
    echo "checking that no navigation goal is running (up to ${GUARD_TIMEOUT_S} s: an rclpy pass on the board)"
    line="$(ssh "root@$BOARD" \
        "docker exec -i pepin-ros /pepin_entrypoint.sh timeout $GUARD_TIMEOUT_S python3 - $NAV_ACTIONS" \
        < "$HERE/tools/nav_goal_running.py" 2>/dev/null || true)"
    for action in $NAV_ACTIONS; do
        verdict="?"
        for word in $line; do
            case "$word" in
                "$action=yes") verdict=yes ;;
                "$action=no") verdict=no ;;
            esac
        done
        case "$verdict" in
            no) ;;
            yes) echo "refused: a navigation goal is running ($action); ros/go.sh cancel first"; exit 1 ;;
            *) echo "refused: the guard could not read /$action/_action/status, so it cannot say the cart is standing still"
               exit 1 ;;
        esac
    done
}

one_start() {  # -> "SECONDS|CANT|INVALID|VERDICT"; SECONDS is TIMEOUT when the bond never came
    local t0 elapsed window bond activation seconds cant invalid verdict
    ssh "root@$BOARD" "systemctl restart pepin-ros && sleep $SETTLE_S && systemctl is-active pepin-ros" >/dev/null
    t0=$(date +%s)
    while :; do
        elapsed=$(($(date +%s) - t0))
        window=$((elapsed + SETTLE_S + 5))   # the window covers the restart itself
        bond="$(board_stamp "$window" 'planner_server connected with bond')"
        [ -z "$bond" ] || break
        [ "$elapsed" -lt "$ACTIVATION_TIMEOUT_S" ] || break
        sleep "$POLL_S"
    done
    window=$(($(date +%s) - t0 + SETTLE_S + 5))
    activation="$(board_stamp "$window" 'Activating planner_server')"
    cant="$(board_count "$window" "Range sensor layer can't transform")"
    invalid="$(board_count "$window" 'Invalid frame ID')"
    if [ -n "$bond" ] && [ -n "$activation" ]; then
        seconds="$(awk -v a="$activation" -v b="$bond" 'BEGIN { printf "%.1f", b - a }')"
    elif [ -n "$bond" ]; then
        seconds="bonded"   # bonded, but the activation line fell outside the window
    else
        seconds="TIMEOUT"
    fi
    verdict=FAIL
    { [ "$seconds" = TIMEOUT ] || [ "$cant" != 0 ]; } || verdict=PASS
    printf '%s|%s|%s|%s\n' "$seconds" "$cant" "$invalid" "$verdict"
}

echo "cold-start soak: $RUNS restarts of pepin-ros on $BOARD, up to ${ACTIVATION_TIMEOUT_S} s each."
echo "the robot does not move: this restarts processes and reads logs, it sends no goal and no velocity."
refuse_if_navigating

PASSED=0
printf '\n%-4s %16s %16s %14s  %s\n' run "activation->bond" "can't transform" "invalid frame" verdict
for i in $(seq 1 "$RUNS"); do
    # The start runs BEFORE the read's IFS exists: inside the here-string it would inherit
    # IFS='|' and lib.sh's ssh wrapper would stop splitting its options.
    outcome="$(one_start)"
    IFS='|' read -r seconds cant invalid verdict <<<"$outcome"
    [ "$verdict" != PASS ] || PASSED=$((PASSED + 1))
    printf '%-4s %16s %16s %14s  %s\n' "$i" "$seconds" "$cant" "$invalid" "$verdict"
done

printf '\n%s of %s starts passed (a pass is: planner_server bonded, and no range-layer transform failure)\n' \
    "$PASSED" "$RUNS"
if [ "$PASSED" -ne "$RUNS" ]; then
    echo "the failing starts are the ones to read: ssh root@$BOARD 'docker logs pepin-ros' (the previous"
    echo "container's log is saved by board/pepin-ros.service under /root/pepin-ros/logs/)"
    exit 1
fi
