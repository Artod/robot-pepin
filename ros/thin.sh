#!/bin/bash
# Flip the board between the whole stack and the reflex half. Usage:
#   ros/thin.sh on      board runs side=board + the zenoh bridge; the laptop plans and takes goals
#   ros/thin.sh vision  board runs the whole stack AND the bridge: the laptop only maps and watches
#   ros/thin.sh off     board runs the whole stack, bridge stopped
#   ros/thin.sh kick NODE  restart one node of the board's stack from the synced sources (seconds)
#   ros/thin.sh         show the current side and bridge
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
HERE="$(cd "$(dirname "$0")" && pwd)"
. "$HERE/lib.sh"
# The nodes a kick can reach on the board and the line each prints once up (the kick waits for
# it): our own processes of nav.launch.py, and the neck node of robot.launch.py (ros/feature.sh
# neck on). Only one of the two recorders runs (PEPIN_RECORDER, ros/feature.sh recorder
# jsonl|bag); a kick of the other one finds nothing and says so. The goal server is here on
# side=all only (on side=board it lives on the laptop:
# ros/laptop.sh kick goal_server). slam_frame is the retired owner of map -> odom and only runs
# with PEPIN_SLAM=true (CLAUDE.md rule 19), but a kick still reaches it where it does.
KICKABLE="relocalizer run_recorder bag_recorder goal_server neck_state slam_frame tof_bridge"
# ...of which the relocalizer is launched only under PEPIN_LOCALIZER=tracker (ros/lib.sh): under
# rtabmap the laptop's RTAB-Map owns map -> odom and no tracker process exists here, so a kick of
# it is refused with that reason instead of the bare "no relocalizer process in pepin-ros".
not_launched() {  # node name -> "" when it can be kicked, else why it cannot be
    case "$1" in
        relocalizer) pepin_localizer_is_tracker ||
            echo "relocalizer is not launched under PEPIN_LOCALIZER=$PEPIN_LOCALIZER: RTAB-Map on the laptop owns map -> odom" ;;
    esac
}
kick_line() {  # node name -> start-up line
    case "$1" in
        relocalizer) echo "relocalizer up: " ;;
        run_recorder) echo "run recorder ready" ;;
        bag_recorder) echo "bag recorder ready" ;;
        goal_server) echo "goal server ready on port" ;;
        neck_state) echo "neck state up: " ;;
        slam_frame) echo "slam frame up: " ;;
        tof_bridge) echo "tof ceilings: " ;;
        *) return 1 ;;
    esac
}
case "${1:-}" in
    on)
        # The bridge is a systemd unit tied to the stack (board/pepin-bridge.service): it starts
        # after the stack's last node is up and restarts with it. Started by hand before the stack
        # it wedged silently (2026-09-09).
        # The split's allow-list is the unit's default (zenoh-bridge-board.json): no
        # PEPIN_BRIDGE_CONFIG line. The states are printed, never judged: `systemctl is-active`
        # exits 3 while the bridge is still in its ExecStartPre (it waits for the tracker), and
        # under set -e that ended this script with an error for a stack that was fine.
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d; /^PEPIN_BRIDGE=/d; /^PEPIN_BRIDGE_CONFIG=/d; /^PEPIN_SLAM=/d' /etc/default/pepin-ros; echo PEPIN_SIDE=board >> /etc/default/pepin-ros;
            systemctl enable pepin-bridge >/dev/null 2>&1; systemctl daemon-reload; systemctl restart pepin-ros && sleep 8; systemctl is-active pepin-ros pepin-bridge | tr '\\n' ' '; echo"
        echo "board on side=board; the bridge follows the stack; now: ros/laptop.sh" ;;
    vision)
        # Every drive stays on the board (the proven stack); the bridge carries topics only, for
        # RTAB-Map and the camera on the laptop. Actions over the bridge aborted the navigation
        # container ("Failed to accept new goal", 2026-09-10 16:06); topics never failed.
        # The bridge reads the vision allow-list (zenoh-bridge-board-vision.json, synced with
        # ros/: the board publishes the plan and the costmaps too, the laptop the ONE map — its
        # RTAB-Map grid on /map — and the camera's scans and words) through PEPIN_BRIDGE_CONFIG,
        # written here together with the mode. PEPIN_SLAM is deleted in the same breath: there is
        # no mode in which the board takes map -> odom from a message any more, and the tracker
        # stands down only if somebody sets that line by hand (CLAUDE.md rule 19).
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d; /^PEPIN_BRIDGE=/d; /^PEPIN_BRIDGE_CONFIG=/d; /^PEPIN_SLAM=/d' /etc/default/pepin-ros; printf 'PEPIN_BRIDGE=on\\nPEPIN_BRIDGE_CONFIG=zenoh-bridge-board-vision.json\\n' >> /etc/default/pepin-ros;
            test -f /root/pepin-ros/zenoh-bridge-board-vision.json || echo 'WARNING: no zenoh-bridge-board-vision.json on the board: ros/sync.sh first';
            systemctl enable pepin-bridge >/dev/null 2>&1; systemctl daemon-reload; systemctl restart pepin-ros && sleep 8; systemctl is-active pepin-ros pepin-bridge | tr '\\n' ' '; echo"
        echo "board on side=all with the bridge; now: ros/laptop.sh (bridge) and ros/laptop.sh vslam" ;;
    off)
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d; /^PEPIN_BRIDGE=/d; /^PEPIN_BRIDGE_CONFIG=/d; /^PEPIN_SLAM=/d' /etc/default/pepin-ros; systemctl disable --now pepin-bridge >/dev/null 2>&1; docker rm -f zenoh-bridge >/dev/null 2>&1; systemctl restart pepin-ros && sleep 8; systemctl is-active pepin-ros; true"
        echo "board on side=all (whole stack on the robot)" ;;
    kick)
        # One node of the stack, not the stack: `ros/sync.sh` (or ros/push.sh) puts the sources
        # on the board, this ends the node with SIGINT (what the launch sends at shutdown, so it
        # leaves the middleware properly and a bridge forgets its name at once) and the launch
        # respawns it from those sources two seconds after its exit (RESPAWN in nav.launch.py).
        # No ghost is possible: the successor starts only after the exit. A stack restart is the
        # slow case because everything hangs on it. Measured at a cold boot: the relocalizer
        # prints its line 7 s after its start, the goal server 9 s, the recorder 5 s; a kick is
        # that plus the two-second pause. The tracker is gone for those seconds (no map -> odom):
        # kick it at rest, never mid-drive.
        #   The wait proves the ready line is the NEW process's (ros/kick_ready.awk): the launch's
        # exit line for the signalled pid names the process's tag, the successor's start line
        # under that tag gives the new pid, and only a ready line under that tag after it counts
        # — never a `--since` grep, which returned the old process's line (2026-09-20). The times
        # are the container's clock: the kick's own `date` and the log's timestamps. The
        # arguments reach the board quoted (ssh joins its arguments into one command line:
        # unquoted, the ready line was cut at its first space) and the matcher travels with them,
        # so a board that has not been synced since it changed still waits the same way.
        NAME="${2:-}"; LINE="$(kick_line "$NAME")" || { echo "usage: ros/thin.sh kick <node>; nodes: $KICKABLE"; exit 2; }
        WHY="$(not_launched "$NAME")"; [ -z "$WHY" ] || { echo "$WHY"; exit 2; }
        { printf 'AWK=%q\n' "$(cat "$HERE/kick_ready.awk")"; cat <<'EOF'; } | ssh "root@$BOARD" "bash -s -- $(printf '%q ' "$NAME" "$LINE")"
set -u
NAME=$1; LINE=$2; TAB=$(printf '\t')
# The console scripts run as pepin_bringup/<name>, the modules as pepin_bringup.<name>. One exec
# reads the container's clock and the pids to signal.
OUT=$(docker exec pepin-ros sh -c 'date -u +%FT%T.%NZ; pgrep -f "pepin_bringup[./]$1"' sh "$NAME" 2>/dev/null) || true
KICKED=${OUT%%$'\n'*}; OLD=$(printf '%s\n' "$OUT" | sed 1d | tr '\n' ' ')
[ -n "${OLD// /}" ] || { echo "no $NAME process in pepin-ros ($(grep -oE 'PEPIN_SIDE=.*' /etc/default/pepin-ros || echo side=all))"; exit 3; }
# shellcheck disable=SC2086
docker exec pepin-ros sh -c 'kill -INT "$@"' sh $OLD
R="wait${TAB}nothing read from the log yet"
for _ in $(seq 1 240); do
    R=$(docker logs -t --since "$KICKED" pepin-ros 2>&1 | awk -v name="$NAME" -v old="$OLD" -v line="$LINE" -v kicked="$KICKED" "$AWK")
    if [ "${R%%"$TAB"*}" = ready ]; then
        echo "${R#*"$TAB"}"
        # The board's bridge (side=board or bridge=on) keys routes by node name: a ghost of the
        # kicked node beside the new one means its routes drop when the ghost expires.
        N=$(curl -s -m 3 'http://localhost:8000/@/local/ros2/node/**' | grep -o "/ros2/node/[^/\"]*/$NAME\"" | wc -l)
        case "$N" in
            1) echo "the bridge lists $NAME once: clean" ;;
            0) ;;
            *) echo "WARNING: the bridge lists $NAME beside a ghost of itself: its routes drop when the ghost expires; kick again in 10 s" ;;
        esac
        exit 0
    fi
    sleep 0.5
done
echo "$NAME not ready within 120 s: ${R#*"$TAB"} (ros/watch.sh)"; exit 4
EOF
        ;;
    *)
        # Printed, not judged: is-active exits non-zero for anything but "active" (3 while
        # activating), and this is a report.
        ssh "root@$BOARD" "grep -oE 'PEPIN_(SIDE|BRIDGE|BRIDGE_CONFIG|NAV|SLAM|SLAM_TOOLBOX|LOCALIZER)=.*' /etc/default/pepin-ros | tr '\\n' ' '; echo; systemctl is-active pepin-ros pepin-bridge | tr '\\n' ' '; echo" ;;
esac
