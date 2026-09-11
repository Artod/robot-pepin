#!/bin/bash
# Freeze the map the laptop is building into a file the known-map stack can drive on.
#   ros/map.sh save NAME    write ros/maps/NAME.{pgm,yaml} from /map as it stands right now
# RTAB-Map keeps its map as a pose graph in its database and publishes the current occupancy grid
# on /map (transient local, one message per update). nav2's map_saver_cli subscribes to exactly
# that and writes the pair map_server reads later, so a SLAM session ends with a map the robot
# can be sent back into: ros/mode.sh nav /maps/NAME.yaml. The container mounts ros/maps, so the
# files land in the checkout with no copying; the places book of the new map starts empty
# (ros/go.sh mark NAME while standing somewhere).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
case "${1:-}" in
    save) ;;
    *) echo "usage: ros/map.sh save NAME"; exit 2 ;;
esac
NAME="${2:?a map name, e.g. flat3_slam}"
docker ps --format '{{.Names}}' | grep -qx pepin-vslam \
    || { echo "no pepin-vslam container: ros/laptop.sh vslam first"; exit 1; }
MODE="$(cat "$HERE/.mode" 2>/dev/null || echo unknown)"
[ "$MODE" = slam ] \
    || echo "note: the laptop last read the board in mode '$MODE', not slam — /map is then the board's own saved map, and this would only copy it back"
# save_map_timeout: the grid arrives on the next publish (map_always_update: one a second), so
# ten seconds is a dead publisher, not a slow one. The thresholds are map_saver's defaults, the
# ones ros/savemap.sh has always written: a cell is free below 0.25 and occupied above 0.65.
docker exec pepin-vslam /pepin_entrypoint.sh ros2 run nav2_map_server map_saver_cli \
    -t /map -f "/maps/$NAME" --ros-args -p save_map_timeout:=10.0 2>&1 | tail -2
[ -s "$HERE/maps/$NAME.pgm" ] || { echo "nothing was written to ros/maps/$NAME.pgm: is /map being published? (ros/laptop.sh logs vslam)"; exit 1; }
echo "saved: ros/maps/$NAME.yaml + .pgm ($(wc -c < "$HERE/maps/$NAME.pgm" | tr -d ' ') bytes)"
echo "    drive on it later: ros/thin.sh vision (or on), then ros/mode.sh nav /maps/$NAME.yaml"
