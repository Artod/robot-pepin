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
CAM="$(dirname "$0")/maps/rec/${STAMP}_goto_cam.mkv"
mkdir -p "$(dirname "$0")/maps/rec"
# The head camera, always: any goal may be the clip for the tweet (mjpeg copied as-is, no CPU).
ffmpeg -loglevel error -y -f mjpeg -use_wallclock_as_timestamps 1 -i "http://$BOARD:8080/stream" -c copy "$CAM" &
FFPID=$!
LOG="/maps/rec/${STAMP}_goto.log"   # goto_ros' own words, kept on the board next to the recording
INTERRUPTED=0
trap 'INTERRUPTED=1' INT
finish() {  # everything recorded, always: scans, odometry, tracked pose, the goal's own log, the board log
    kill -INT $FFPID 2>/dev/null; wait $FFPID 2>/dev/null
    watch_stop
    if [ "$INTERRUPTED" = 1 ]; then
        echo; echo "Ctrl-C: cancelling the navigation task on the board..."
        "$(dirname "$0")/stop.sh"
    elif ssh "root@$BOARD" "docker exec pepin-ros pgrep -f '^python3 /tools/goto_ros.py' >/dev/null" 2>/dev/null; then
        echo; echo "!! the link to the board dropped but the drive goes on there; ros/stop.sh stops it, ros/watch.sh shows it"
    fi
    # Every cleanup step leaves its exit code in a trace file: a silent failure here once cost a run's
    # files (2026-09-07 13:23: the logger kept running, nothing was fetched, no error shown).
    local rec trace
    rec="$(dirname "$0")/maps/rec"; trace="$rec/${STAMP}_goto_finish.log"
    mkdir -p "$rec"
    ssh "root@$BOARD" "docker exec pepin-ros pkill -INT -f 'session_logger.py $REC'" >> "$trace" 2>&1; echo "logger stopped: $?" >> "$trace"
    ssh "root@$BOARD" "docker logs --since 10m pepin-ros 2>&1" > "$rec/${STAMP}_goto_board.log" 2>> "$trace"; echo "board log: $? $(wc -c < "$rec/${STAMP}_goto_board.log") bytes" >> "$trace"
    sleep 1
    rsync -aq "root@$BOARD:/root/pepin-ros/maps/rec/${STAMP}_goto.*" "$rec/" >> "$trace" 2>&1 || { sleep 2; rsync -aq "root@$BOARD:/root/pepin-ros/maps/rec/${STAMP}_goto.*" "$rec/" >> "$trace" 2>&1; }
    echo "fetched: $?" >> "$trace"
    echo "recorded: $(ls "$rec" | grep -c "^${STAMP}_goto") files ros/maps/rec/${STAMP}_goto* (jsonl, log, board log, camera; cleanup trace in _goto_finish.log)"
}
trap finish EXIT
ssh "root@$BOARD" "docker exec -d pepin-ros /pepin_entrypoint.sh python3 /tools/session_logger.py $REC"
# The goal lives on the board (a WiFi hiccup must not become a cancel); this terminal only watches.
ssh "root@$BOARD" "touch /root/pepin-ros$LOG; docker exec -d -e PYTHONUNBUFFERED=1 pepin-ros /pepin_entrypoint.sh sh -c 'python3 /tools/goto_ros.py --places $PLACES $* > $LOG 2>&1; echo GOTO_EXIT=\$? >> $LOG'"
watch_start
ssh "root@$BOARD" "tail -n +1 -F /root/pepin-ros$LOG 2>/dev/null | sed -u '/^GOTO_EXIT=/q'" 2>/dev/null || true
