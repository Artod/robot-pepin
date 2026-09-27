#!/bin/bash
# Flip the board between the whole stack and the reflex half. Usage:
#   ros/thin.sh on      board runs side=board; the laptop plans and takes goals
#   ros/thin.sh vision  board runs the whole stack: the laptop only maps and watches
#   ros/thin.sh off     the same side as vision (the whole stack on the robot)
#   ros/thin.sh kick NODE  restart one node of the board's stack from the synced sources (seconds)
#   ros/thin.sh         show the current side
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"
# The nodes a kick can reach on the board and the line each prints once up (the kick waits for
# it): our own processes of nav.launch.py, and the neck node of robot.launch.py (ros/feature.sh
# neck on). Only one of the two recorders runs (PEPIN_RECORDER, ros/feature.sh recorder
# jsonl|bag); a kick of the other one finds nothing and says so. The goal server is here on
# side=all only (on side=board it lives on the laptop:
# ros/laptop.sh kick goal_server).
KICKABLE="run_recorder bag_recorder goal_server neck_state tof_bridge"
kick_line() {  # node name -> start-up line
    case "$1" in
        run_recorder) echo "run recorder ready" ;;
        bag_recorder) echo "bag recorder ready" ;;
        goal_server) echo "goal server ready on port" ;;
        neck_state) echo "neck state up: " ;;
        tof_bridge) echo "tof ceilings: " ;;
        *) return 1 ;;
    esac
}
# Both routers are always up, so a side change is one line in /etc/default/pepin-ros and a stack
# restart. The lines of the retired bridge (PEPIN_BRIDGE, PEPIN_BRIDGE_CONFIG) and of the retired
# frame owner (PEPIN_SLAM) are deleted in the same breath, and a board that still has the bridge
# units gets them disabled (tag alt/cyclone-bridges-2026-09-20). The state is printed, never
# judged: `systemctl is-active` exits non-zero while a unit is activating.
CLEAN="sed -i '/^PEPIN_SIDE=/d; /^PEPIN_BRIDGE=/d; /^PEPIN_BRIDGE_CONFIG=/d; /^PEPIN_SLAM=/d' /etc/default/pepin-ros; systemctl disable --now pepin-bridge pepin-bridge-kick.path >/dev/null 2>&1 || true"
case "${1:-}" in
    on)
        ssh "root@$BOARD" "$CLEAN; echo PEPIN_SIDE=board >> /etc/default/pepin-ros;
            systemctl restart pepin-ros && sleep 8; systemctl is-active pepin-ros | tr '\\n' ' '; echo"
        echo "board on side=board; now: ros/laptop.sh" ;;
    vision|off)
        # Every drive stays on the board; the laptop maps and watches over the routers.
        ssh "root@$BOARD" "$CLEAN; systemctl restart pepin-ros && sleep 8; systemctl is-active pepin-ros; true"
        echo "board on side=all (whole stack on the robot); the laptop maps with ros/laptop.sh vslam" ;;
    kick)
        # One node of the stack, not the stack: `ros/sync.sh --no-restart` puts the sources on
        # the board, this ends the node with SIGINT (what the launch sends at shutdown) and the
        # launch respawns it from those sources two seconds after its exit (RESPAWN in
        # nav.launch.py). Measured at a cold boot: the goal server prints its line 9 s after its
        # start, the recorder 5 s; a kick is that plus the two-second pause. Kick at rest, never
        # mid-drive.
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
        ssh "root@$BOARD" "grep -oE 'PEPIN_(SIDE|NAV|SLAM_TOOLBOX)=.*' /etc/default/pepin-ros | tr '\\n' ' '; echo; systemctl is-active pepin-ros pepin-zrouter | tr '\\n' ' '; echo" ;;
esac
