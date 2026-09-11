#!/bin/bash
# Switch what the robot's container runs. A map change while Nav2 already runs is a live
# map swap (seconds); everything else restarts the one launch process. Usage:
#   ros/mode.sh sensors            lidar, base bridge, Foxglove — nothing that localises
#   ros/mode.sh slam_toolbox       + slam_toolbox: build a map while driving (ros/teleop.sh),
#                                  to SAVE and navigate on later — it cannot drive on it
#   ros/mode.sh nav [MAP.yaml]     + Nav2 with the relocalizer on a saved map
# Online SLAM — one map built while Nav2 drives it — is not a mode of this script: it lives on
# both machines at once (ros/thin.sh slam on the board, ros/laptop.sh vslam --slam here).
# Leaving slam_toolbox saves the map being built as /maps/autosave_<time> first: slam_toolbox holds
# it in memory only, and a restart would throw away the drive that produced it.
# Every mode here leaves online SLAM (PEPIN_SLAM=false): the board serves a map again.
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
HERE="$(cd "$(dirname "$0")" && pwd)"
MODE="${1:?sensors | slam_toolbox | nav}"
MAP="${2:-/maps/20260903_182653_lap3_loop.yaml}"
case "$MODE" in
  sensors)      NAV=false; TOOLBOX=false ;;
  slam_toolbox) NAV=false; TOOLBOX=true ;;
  nav)          NAV=true;  TOOLBOX=false ;;
  slam) echo "online SLAM is ros/thin.sh slam (the laptop's RTAB-Map builds the map, Nav2 drives it);
             the old lidar-only mapper is ros/mode.sh slam_toolbox"; exit 2 ;;
  *) echo "unknown mode $MODE"; exit 2 ;;
esac
current_toolbox=$(ssh "root@$BOARD" "grep -c 'PEPIN_SLAM_TOOLBOX=true' /etc/default/pepin-ros 2>/dev/null || true")
if [ "$MODE" != slam_toolbox ] && [ "$current_toolbox" = "1" ]; then
    stamp=$(date +%Y%m%d_%H%M%S)
    echo "slam_toolbox is running: saving its map as /maps/autosave_$stamp before switching"
    "$HERE/savemap.sh" "autosave_$stamp" || { echo "could not save the map — not switching"; exit 1; }
fi
if [ "$MODE" = nav ]; then
    ssh "root@$BOARD" "test -f /root/pepin-ros$MAP" || { echo "no such map on the board: $MAP (save one with ros/savemap.sh first)"; exit 1; }
fi
T0=$(date +%s)
already_nav=$(ssh "root@$BOARD" "grep -c 'PEPIN_NAV=true' /etc/default/pepin-ros 2>/dev/null; grep -c 'PEPIN_SLAM=true' /etc/default/pepin-ros 2>/dev/null; docker ps --format '{{.Names}}' | grep -c '^pepin-ros\$'" | tr '\n' ' ')
# "1 0 1": Nav2 is up on a saved map and nothing is mapping — the only state a live swap fits.
if [ "$MODE" = nav ] && [ "$already_nav" = "1 0 1 " ]; then
    # Nav2 is already up: swap the map under it (about 5 s) instead of restarting the stack (about 60 s).
    ssh "root@$BOARD" "{ grep -E '^PEPIN_(CPP_BRIDGE|IMU|TOF|NECK|SIDE|BRIDGE|BRIDGE_CONFIG)=' /etc/default/pepin-ros 2>/dev/null; printf 'PEPIN_NAV=%s\\nPEPIN_SLAM=false\\nPEPIN_SLAM_TOOLBOX=%s\\nPEPIN_MAP=%s\\n' $NAV $TOOLBOX '$MAP'; } > /etc/default/pepin-ros.new && mv /etc/default/pepin-ros.new /etc/default/pepin-ros"
    ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh timeout 60 python3 /tools/load_map.py '$MAP'" || exit 1
    echo "map swapped in $(( $(date +%s) - T0 )) s, no restart"
    exit 0
fi
ssh "root@$BOARD" "{ grep -E '^PEPIN_(CPP_BRIDGE|IMU|TOF|NECK|SIDE|BRIDGE|BRIDGE_CONFIG)=' /etc/default/pepin-ros 2>/dev/null; printf 'PEPIN_NAV=%s\\nPEPIN_SLAM=false\\nPEPIN_SLAM_TOOLBOX=%s\\nPEPIN_MAP=%s\\n' $NAV $TOOLBOX '$MAP'; } > /etc/default/pepin-ros.new && mv /etc/default/pepin-ros.new /etc/default/pepin-ros; systemctl restart pepin-ros"
echo -n "mode $MODE requested; restarting the stack..."
READY=$([ "$MODE" = slam_toolbox ] && echo 'slam_toolbox\\]: Activating' || { [ "$MODE" = nav ] && echo 'lifecycle_manager_navigation.*Managed nodes are active' || echo 'lifecycle_manager_sensors.*Managed nodes are active'; })
for i in $(seq 1 60); do
    ssh "root@$BOARD" "docker logs pepin-ros 2>&1 | grep -qE '$READY'" 2>/dev/null && break
    sleep 2
done
echo " up in $(( $(date +%s) - T0 )) s"
ssh "root@$BOARD" "free -m | sed -n 2p"
