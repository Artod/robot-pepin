#!/bin/bash
# Drive the robot through Nav2 on the board, with feedback. Usage:
#   ros/goto.sh X Y [YAW_DEG]    drive to map coordinates (meters, degrees)
#   ros/goto.sh home             back to the start spot (map origin, facing as at start)
#   ros/goto.sh mark NAME        stand the robot somewhere: remember that spot as NAME (per map)
#   ros/goto.sh NAME             drive to a remembered place      ros/goto.sh places   list them
#   ros/goto.sh seed X Y [YAW]   after placing the robot by hand: tell AMCL where it is
#   ros/goto.sh cancel           stop the current task (the base's deadman stops the wheels)
#   ros/goto.sh where            pose and scan-to-map fit right now
#   ros/goto.sh relocalize       whole-map search now (after a carry or a push)
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
MAP=$(ssh "root@$BOARD" "grep -oE 'PEPIN_MAP=.*' /etc/default/pepin-ros" | cut -d= -f2)
PLACES="/maps/$(basename "${MAP:-places}" .yaml).places.yaml"  # one book of places per map
case "${1:-}" in
  where) ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh python3 /tools/call.py /where_am_i"; exit ;;
  relocalize) ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh python3 /tools/call.py /relocalize 90"; exit ;;
  mark|places) ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh python3 /tools/goto_ros.py --places $PLACES $*"
               rsync -aq "root@$BOARD:/root/pepin-ros$PLACES" "$(dirname "$0")/maps/" 2>/dev/null; exit ;;  # the book is backed up on the laptop too
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
# With a terminal: ptys on both hops so Ctrl-C reaches goto_ros; without one (a script, Claude): plain pipes.
if [ -t 0 ]; then TTY_SSH="-t"; TTY_DOCKER="-it"; else TTY_SSH=""; TTY_DOCKER="-i"; fi
ssh $TTY_SSH "root@$BOARD" "docker exec $TTY_DOCKER -e PYTHONUNBUFFERED=1 pepin-ros /pepin_entrypoint.sh python3 /tools/goto_ros.py --places $PLACES $*"
