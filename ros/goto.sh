#!/bin/bash
# Drive the robot through Nav2 on the board, with feedback. Usage:
#   ros/goto.sh X Y [YAW_DEG]    drive to map coordinates (meters, degrees)
#   ros/goto.sh home             back to the marked start spot (map origin, facing +x)
#   ros/goto.sh seed X Y [YAW]   after placing the robot by hand: tell AMCL where it is
#   ros/goto.sh cancel           stop the current task (the base's deadman stops the wheels)
#   ros/goto.sh where            pose and scan-to-map fit right now
#   ros/goto.sh relocalize       whole-map search now (after a carry or a push)
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
case "${1:-}" in
  where) exec ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh ros2 service call /where_am_i std_srvs/srv/Trigger" ;;
  relocalize) exec ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh ros2 service call /relocalize std_srvs/srv/Trigger" ;;
esac
exec ssh -t "root@$BOARD" "docker exec -e PYTHONUNBUFFERED=1 pepin-ros /pepin_entrypoint.sh python3 /tools/goto_ros.py $*"
