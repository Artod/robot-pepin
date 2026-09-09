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
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d' /etc/default/pepin-ros; echo PEPIN_SIDE=board >> /etc/default/pepin-ros;
            docker rm -f zenoh-bridge >/dev/null 2>&1; docker run -d --name zenoh-bridge --network host --restart unless-stopped -e ROS_DISTRO=jazzy eclipse/zenoh-bridge-ros2dds:1.5.1 -d 7 -l tcp/0.0.0.0:7447 >/dev/null;
            systemctl restart pepin-ros && sleep 8 && systemctl is-active pepin-ros"
        echo "board on side=board; now: ros/laptop.sh" ;;
    off)
        ssh "root@$BOARD" "sed -i '/^PEPIN_SIDE=/d' /etc/default/pepin-ros; docker rm -f zenoh-bridge >/dev/null 2>&1; systemctl restart pepin-ros && sleep 8 && systemctl is-active pepin-ros"
        echo "board on side=all (whole stack on the robot)" ;;
    *)
        ssh "root@$BOARD" "grep -oE 'PEPIN_SIDE=.*' /etc/default/pepin-ros || echo PEPIN_SIDE=all" ;;
esac
