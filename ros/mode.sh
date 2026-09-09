#!/bin/bash
# Switch what the robot's container runs. A map change while Nav2 already runs is a live
# map swap (seconds); everything else restarts the one launch process. Usage:
#   ros/mode.sh sensors            lidar, base bridge, Foxglove — nothing that localises
#   ros/mode.sh slam               + slam_toolbox: build a map while driving (ros/teleop.sh)
#   ros/mode.sh nav [MAP.yaml]     + Nav2 with AMCL and the relocalizer on a saved map
# Leaving slam mode saves the map being built as /maps/autosave_<time> first: slam_toolbox holds it
# in memory only, and a restart would throw away the drive that produced it.
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
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
T0=$(date +%s)
already_nav=$(ssh "root@$BOARD" "grep -c 'PEPIN_NAV=true' /etc/default/pepin-ros 2>/dev/null; docker ps --format '{{.Names}}' | grep -c '^pepin-ros\$'" | tr '\n' ' ')
if [ "$MODE" = nav ] && [ "$already_nav" = "1 1 " ]; then
    # Nav2 is already up: swap the map under it (about 5 s) instead of restarting the stack (about 60 s).
    ssh "root@$BOARD" "{ grep -E '^PEPIN_(CPP_BRIDGE|IMU|TOF|SIDE)=' /etc/default/pepin-ros 2>/dev/null; printf 'PEPIN_NAV=%s\\nPEPIN_SLAM=%s\\nPEPIN_MAP=%s\\n' $NAV $SLAM '$MAP'; } > /etc/default/pepin-ros.new && mv /etc/default/pepin-ros.new /etc/default/pepin-ros"
    ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh timeout 60 python3 /tools/load_map.py '$MAP'" || exit 1
    echo "map swapped in $(( $(date +%s) - T0 )) s, no restart"
    exit 0
fi
ssh "root@$BOARD" "{ grep -E '^PEPIN_(CPP_BRIDGE|IMU|TOF|SIDE)=' /etc/default/pepin-ros 2>/dev/null; printf 'PEPIN_NAV=%s\\nPEPIN_SLAM=%s\\nPEPIN_MAP=%s\\n' $NAV $SLAM '$MAP'; } > /etc/default/pepin-ros.new && mv /etc/default/pepin-ros.new /etc/default/pepin-ros; systemctl restart pepin-ros"
echo -n "mode $MODE requested; restarting the stack..."
READY=$([ "$MODE" = slam ] && echo 'slam_toolbox\\]: Activating' || { [ "$MODE" = nav ] && echo 'lifecycle_manager_navigation.*Managed nodes are active' || echo 'lifecycle_manager_sensors.*Managed nodes are active'; })
for i in $(seq 1 60); do
    ssh "root@$BOARD" "docker logs pepin-ros 2>&1 | grep -qE '$READY'" 2>/dev/null && break
    sleep 2
done
echo " up in $(( $(date +%s) - T0 )) s"
ssh "root@$BOARD" "free -m | sed -n 2p"
