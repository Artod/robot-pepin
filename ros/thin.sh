#!/bin/bash
# Flip the board between the whole stack and the reflex half. Usage:
#   ros/thin.sh on      board runs side=board + the zenoh bridge; the laptop plans and takes goals
#   ros/thin.sh vision  board runs the whole stack AND the bridge: the laptop only maps and watches
#   ros/thin.sh off     board runs the whole stack, bridge stopped
#   ros/thin.sh kick NODE  restart one node of the board's stack from the synced sources (seconds)
#   ros/thin.sh         show the current side and bridge
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"
# The nodes a kick can reach on the board and the line each prints once up (the kick waits for
# it): our own processes of nav.launch.py. The goal server is here on side=all only (on
# side=board it lives on the laptop: ros/laptop.sh kick goal_server).
KICKABLE="relocalizer run_recorder goal_server"
kick_line() {  # node name -> start-up line
    case "$1" in
        relocalizer) echo "relocalizer up: " ;;
        run_recorder) echo "run recorder ready" ;;
        goal_server) echo "goal server ready on port" ;;
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
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d; /^PEPIN_BRIDGE=/d; /^PEPIN_BRIDGE_CONFIG=/d' /etc/default/pepin-ros; echo PEPIN_SIDE=board >> /etc/default/pepin-ros;
            systemctl enable pepin-bridge >/dev/null 2>&1; systemctl daemon-reload; systemctl restart pepin-ros && sleep 8; systemctl is-active pepin-ros pepin-bridge | tr '\\n' ' '; echo"
        echo "board on side=board; the bridge follows the stack; now: ros/laptop.sh" ;;
    vision)
        # Every drive stays on the board (the proven stack); the bridge carries topics only, for
        # RTAB-Map and the camera on the laptop. Actions over the bridge aborted the navigation
        # container ("Failed to accept new goal", 2026-09-10 16:06); topics never failed.
        # The bridge reads the vision allow-list (zenoh-bridge-board-vision.json, synced with
        # ros/: the board publishes the plan and the costmaps too, the laptop only its map and
        # depth) through PEPIN_BRIDGE_CONFIG, written here together with the mode.
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d; /^PEPIN_BRIDGE=/d; /^PEPIN_BRIDGE_CONFIG=/d' /etc/default/pepin-ros; printf 'PEPIN_BRIDGE=on\\nPEPIN_BRIDGE_CONFIG=zenoh-bridge-board-vision.json\\n' >> /etc/default/pepin-ros;
            test -f /root/pepin-ros/zenoh-bridge-board-vision.json || echo 'WARNING: no zenoh-bridge-board-vision.json on the board: ros/sync.sh first';
            systemctl enable pepin-bridge >/dev/null 2>&1; systemctl daemon-reload; systemctl restart pepin-ros && sleep 8; systemctl is-active pepin-ros pepin-bridge | tr '\\n' ' '; echo"
        echo "board on side=all with the bridge; now: ros/laptop.sh (bridge) and ros/laptop.sh vslam" ;;
    off)
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d; /^PEPIN_BRIDGE=/d; /^PEPIN_BRIDGE_CONFIG=/d' /etc/default/pepin-ros; systemctl disable --now pepin-bridge >/dev/null 2>&1; docker rm -f zenoh-bridge >/dev/null 2>&1; systemctl restart pepin-ros && sleep 8; systemctl is-active pepin-ros; true"
        echo "board on side=all (whole stack on the robot)" ;;
    kick)
        # One node of the stack, not the stack: `ros/sync.sh --no-restart` puts the sources on
        # the board, this ends the node with SIGINT (what the launch sends at shutdown, so it
        # leaves DDS properly and the bridge forgets its name at once) and the launch respawns
        # it from those sources two seconds after its exit (RESPAWN in nav.launch.py). No ghost
        # is possible: the successor starts only after the exit. A stack restart is the slow
        # case because everything hangs on it — the bridge unit follows the stack, the laptop
        # containers are restarted by their bridge watch, and their SIGKILLed nodes linger in
        # the laptop's bridge for the DDS lease. Measured at a cold boot: the relocalizer prints
        # its line 7 s after its start, the goal server 9 s, the recorder 5 s; a kick is that
        # plus the two-second pause. The tracker is gone for those seconds (no map -> odom):
        # kick it at rest, never mid-drive.
        NAME="${2:-}"; LINE="$(kick_line "$NAME")" || { echo "usage: ros/thin.sh kick <node>; nodes: $KICKABLE"; exit 2; }
        ssh "root@$BOARD" bash -s -- "$NAME" "$LINE" <<'EOF'
set -u
NAME=$1; LINE=$2; T0=$(date -u +%FT%TZ); MS0=$(date +%s%3N)
# The console scripts run as pepin_bringup/<name>, the modules as pepin_bringup.<name>.
docker exec pepin-ros pkill -INT -f "pepin_bringup[./]$NAME" \
    || { echo "no $NAME process in pepin-ros ($(grep -oE 'PEPIN_SIDE=.*' /etc/default/pepin-ros || echo side=all))"; exit 3; }
for _ in $(seq 1 240); do
    SEEN=$(docker logs --since "$T0" pepin-ros 2>&1 | grep -F "$LINE" || true)
    if [ -n "$SEEN" ]; then
        SEEN="${SEEN%%$'\n'*}"; DT=$(( $(date +%s%3N) - MS0 ))
        printf '%s back in %d.%d s: %s\n' "$NAME" $((DT / 1000)) $((DT % 1000 / 100)) "${SEEN#*]: }"
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
echo "$NAME did not print '$LINE' within 120 s: ros/watch.sh"; exit 4
EOF
        ;;
    *)
        # Printed, not judged: is-active exits non-zero for anything but "active" (3 while
        # activating), and this is a report.
        ssh "root@$BOARD" "grep -oE 'PEPIN_(SIDE|BRIDGE|BRIDGE_CONFIG)=.*' /etc/default/pepin-ros | tr '\\n' ' '; echo; systemctl is-active pepin-ros pepin-bridge | tr '\\n' ' '; echo" ;;
esac
