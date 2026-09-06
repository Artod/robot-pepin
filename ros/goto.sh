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
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
case "${1:-}" in
  where) ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh python3 /tools/call.py /where_am_i"; exit ;;
  relocalize) ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh python3 /tools/call.py /relocalize 90"; exit ;;
esac
# -t + -it: Ctrl-C travels through both ptys to goto_ros.py, which cancels the task on the board
STAMP=$(date +%Y%m%d_%H%M%S)
REC="/maps/rec/${STAMP}_goto.jsonl"
finish() {  # everything recorded, always: the scans, odometry and AMCL poses of the goal, and the board log
    watch_stop
    ssh "root@$BOARD" "docker exec pepin-ros pkill -INT -f session_logger.py" 2>/dev/null || true
    mkdir -p "$(dirname "$0")/maps/rec"
    ssh "root@$BOARD" "docker logs --since 10m pepin-ros 2>&1" > "$(dirname "$0")/maps/rec/${STAMP}_goto_board.log" 2>/dev/null || true
    rsync -aq "root@$BOARD:/root/pepin-ros/maps/rec/${STAMP}_goto.jsonl" "$(dirname "$0")/maps/rec/" 2>/dev/null || true
    echo "recorded: ros/maps/rec/${STAMP}_goto.jsonl + _board.log"
}
trap finish EXIT
ssh "root@$BOARD" "docker exec -d pepin-ros /pepin_entrypoint.sh python3 /tools/session_logger.py $REC"
watch_start
ssh -t "root@$BOARD" "docker exec -it -e PYTHONUNBUFFERED=1 pepin-ros /pepin_entrypoint.sh python3 /tools/goto_ros.py $*"
