#!/bin/bash
# Flip the board between the whole stack and the reflex half. Usage:
#   ros/thin.sh on      board runs side=board + the zenoh bridge sidecar; run ros/laptop.sh next
#   ros/thin.sh off     board runs the whole stack again (side=all), bridge stopped
#   ros/thin.sh         show the current side
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"
case "${1:-}" in
    on)
        # The bridge is a systemd unit tied to the stack (board/pepin-bridge.service): it starts
        # after the stack's last node is up and restarts with it. Started by hand before the stack
        # it wedged silently (2026-09-09).
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d' /etc/default/pepin-ros; echo PEPIN_SIDE=board >> /etc/default/pepin-ros;
            systemctl enable pepin-bridge >/dev/null 2>&1; systemctl daemon-reload; systemctl restart pepin-ros && sleep 8 && systemctl is-active pepin-ros pepin-bridge | tr '\\n' ' '"
        echo; echo "board on side=board; the bridge follows the stack; now: ros/laptop.sh" ;;
    off)
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d' /etc/default/pepin-ros; systemctl disable --now pepin-bridge >/dev/null 2>&1; docker rm -f zenoh-bridge >/dev/null 2>&1; systemctl restart pepin-ros && sleep 8 && systemctl is-active pepin-ros"
        echo "board on side=all (whole stack on the robot)" ;;
    *)
        ssh "root@$BOARD" "grep -oE 'PEPIN_SIDE=.*' /etc/default/pepin-ros || echo PEPIN_SIDE=all" ;;
esac
