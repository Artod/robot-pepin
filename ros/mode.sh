#!/bin/bash
# Switch what the robot's container runs, then restart it (one launch process). Usage:
#   ros/mode.sh sensors            lidar, base bridge, Foxglove — nothing that localises
#   ros/mode.sh slam               + slam_toolbox: build a map while driving (ros/teleop.sh)
#   ros/mode.sh nav [MAP.yaml]     + Nav2 with AMCL and the relocalizer on a saved map
# Leaving slam mode saves the map being built as /maps/autosave_<time> first: slam_toolbox holds it
# in memory only, and a restart would throw away the drive that produced it.
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:?sensors | slam | nav}"
MAP="${2:-/maps/20260903_182653_lap3_loop.yaml}"
case "$MODE" in
  sensors) NAV=false; SLAM=false ;;
  slam)    NAV=false; SLAM=true ;;
  nav)     NAV=true;  SLAM=false ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
current_slam=$(ssh "root@$BOARD" "grep -c 'PEPIN_SLAM=true' /etc/default/pepin-ros 2>/dev/null || true")
if [ "$MODE" != slam ] && [ "$current_slam" = "1" ]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    echo "slam mode is running: saving its map as /maps/autosave_$stamp before switching"
    "$HERE/savemap.sh" "autosave_$stamp" || { echo "could not save the map — not switching"; exit 1; }
fi
if [ "$MODE" = nav ]; then
    ssh "root@$BOARD" "test -f /root/pepin-ros$MAP" || { echo "no such map on the board: $MAP (save one with ros/savemap.sh first)"; exit 1; }
fi
ssh "root@$BOARD" "printf 'PEPIN_NAV=%s\\nPEPIN_SLAM=%s\\nPEPIN_MAP=%s\\n' $NAV $SLAM '$MAP' > /etc/default/pepin-ros; systemctl restart pepin-ros"
echo "mode $MODE requested; waiting for the container..."
WAIT=$([ "$MODE" = nav ] && echo 100 || echo 40)
sleep "$WAIT"
ssh "root@$BOARD" "docker logs pepin-ros 2>&1 | grep -cE 'Managed nodes are active|slam_toolbox\\]: Activating' || true; free -m | sed -n 2p"
