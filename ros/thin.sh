#!/bin/bash
# Flip the board between the whole stack and the reflex half. Usage:
#   ros/thin.sh on      board runs side=board + the zenoh bridge; the laptop plans and takes goals
#   ros/thin.sh vision  board runs the whole stack AND the bridge: the laptop only maps and watches
#   ros/thin.sh off     board runs the whole stack, bridge stopped
#   ros/thin.sh         show the current side and bridge
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"
case "${1:-}" in
    on)
        # The bridge is a systemd unit tied to the stack (board/pepin-bridge.service): it starts
        # after the stack's last node is up and restarts with it. Started by hand before the stack
        # it wedged silently (2026-09-09).
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d' /etc/default/pepin-ros; sed -i '/^PEPIN_BRIDGE=/d' /etc/default/pepin-ros; echo PEPIN_SIDE=board >> /etc/default/pepin-ros;
            systemctl enable pepin-bridge >/dev/null 2>&1; systemctl daemon-reload; systemctl restart pepin-ros && sleep 8 && systemctl is-active pepin-ros pepin-bridge | tr '\\n' ' '"
        echo; echo "board on side=board; the bridge follows the stack; now: ros/laptop.sh" ;;
    vision)
        # Every drive stays on the board (the proven stack); the bridge carries topics only, for
        # RTAB-Map and the camera on the laptop. Actions over the bridge aborted the navigation
        # container ("Failed to accept new goal", 2026-09-10 16:06); topics never failed.
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d; /^PEPIN_BRIDGE=/d' /etc/default/pepin-ros; echo PEPIN_BRIDGE=on >> /etc/default/pepin-ros;
            systemctl enable pepin-bridge >/dev/null 2>&1; systemctl daemon-reload; systemctl restart pepin-ros && sleep 8 && systemctl is-active pepin-ros pepin-bridge | tr '\\n' ' '"
        echo; echo "board on side=all with the bridge; now: ros/laptop.sh (bridge) and ros/laptop.sh vslam" ;;
    off)
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d; /^PEPIN_BRIDGE=/d' /etc/default/pepin-ros; systemctl disable --now pepin-bridge >/dev/null 2>&1; docker rm -f zenoh-bridge >/dev/null 2>&1; systemctl restart pepin-ros && sleep 8 && systemctl is-active pepin-ros"
        echo "board on side=all (whole stack on the robot)" ;;
    *)
        ssh "root@$BOARD" "grep -oE 'PEPIN_(SIDE|BRIDGE)=.*' /etc/default/pepin-ros | tr '\\n' ' '; echo; systemctl is-active pepin-bridge" ;;
esac
