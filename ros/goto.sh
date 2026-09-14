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
# Every run is taped ONCE, by the board's run recorder: the numbered tape
# 0249_<utc>Z_<place>.jsonl, opened on the goal's word — scans, odometry, the tracked pose, the
# commands, the costmap, the EKF and IMU, the ToF, the camera's measurements (meas) and the
# tracker's account of each update (srcs), with the camera clip beside it; goto names it in its
# log and fetches it here. Until 2026-09-14 this script also started a second recorder in the
# container (ros/tools/session_logger.py) which re-deserialised the same 10 Hz lidar stream for
# 15 % of a core; the numbered tape now carries everything it carried.
# PEPIN_SESSION_LOGGER=1 starts it anyway, next to the numbered tape (its file is named by the
# LAPTOP's local clock: ros/maps/rec/<stamp>_goto.jsonl). With PEPIN_GOTO_TAPE=off it is started
# on its own, so a drive without a numbered tape is still recorded. The two clocks are explained
# in ros/maps/README.md.
# PEPIN_REC_CAMERA_SCANS=1 adds /depth_scan and /contact_scan to the session logger's tape.
# PEPIN_GOTO_TAPE=off drives without asking the recorder for a numbered one.
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
MAP=$(ssh "root@$BOARD" "grep -oE 'PEPIN_MAP=.*' /etc/default/pepin-ros" | cut -d= -f2)
PLACES="/maps/$(basename "${MAP:-places}" .yaml).places.yaml"  # one book of places per map
case "${1:-}" in
  where) ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh python3 /tools/call.py /where_am_i"; exit ;;
  # The header promised this for weeks while the case fell through to "drive to a place called
  # cancel" (2026-09-14 11:20: the cart went on butting a table for a minute after the "cancel").
  cancel) ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh timeout 5 python3 /tools/goto_ros.py cancel"; exit ;;
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
MAX_REC_S=900          # the logger's own stop, in seconds: a recorder nobody stops is a bug
LOG="/maps/rec/${STAMP}_goto.log"   # goto_ros' own words, kept on the board next to the recording
INTERRUPTED=0
FINISHED=0
TAILPID=""   # the ssh streaming the goal's log: a signal must not wait for it (see the tail below)
trap 'INTERRUPTED=1' INT
finish() {  # everything recorded, always: scans, odometry, tracked pose, the goal's own log, the board log
    set +e  # nothing here may end the cleanup: a dead ffmpeg (the board rebooted mid-run, 2026-09-14)
           # made `kill` fail and set -e dropped every step below it, tapes included
    if [ "$FINISHED" = 1 ]; then return 0; fi   # TERM runs this, then EXIT runs it again
    FINISHED=1
    if [ -n "$TAILPID" ]; then kill "$TAILPID" 2>/dev/null || true; fi
    kill -INT $FFPID 2>/dev/null; wait $FFPID 2>/dev/null || true  # ffmpeg exits 255 on INT: not an error here
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
    ssh "root@$BOARD" "docker exec pepin-ros pkill -INT -f 'session_logger.py $REC'" >> "$trace" 2>&1 || true; echo "logger stopped: $?" >> "$trace"  # no logger to stop is the normal case now: pkill's 1 must not end finish under set -e
    ssh "root@$BOARD" "docker logs --since 10m pepin-ros 2>&1" > "$rec/${STAMP}_goto_board.log" 2>> "$trace"; echo "board log: $? $(wc -c < "$rec/${STAMP}_goto_board.log") bytes" >> "$trace"
    sleep 1
    rsync -aq "root@$BOARD:/root/pepin-ros/maps/rec/${STAMP}_goto.*" "$rec/" >> "$trace" 2>&1 || { sleep 2; rsync -aq "root@$BOARD:/root/pepin-ros/maps/rec/${STAMP}_goto.*" "$rec/" >> "$trace" 2>&1; }
    echo "fetched: $?" >> "$trace"
    # The numbered tape the recorder opened for this goal, named in the log we just fetched: its
    # jsonl comes home too (the clip stays on the board — this script films the drive itself).
    local taped
    taped=$(grep -o 'taped /maps/rec/[^ ]*\.jsonl' "$rec/${STAMP}_goto.log" 2>/dev/null | tail -1 | cut -d' ' -f2 || true)
    if [ -n "$taped" ]; then
        rsync -aq "root@$BOARD:/root/pepin-ros$taped" "$rec/" >> "$trace" 2>&1 || true
        echo "numbered tape: ros/maps/rec/$(basename "$taped")"
    elif [ -z "${PEPIN_SESSION_LOGGER:-}" ] && [ "${PEPIN_GOTO_TAPE:-on}" != off ]; then
        # The numbered tape is the only tape now: if the recorder never opened one, say so loudly
        # instead of leaving a drive with no record at all.
        echo "!! NO TAPE: the run recorder opened none for this goal (is it up? docker logs pepin-ros)."
        echo "!! Re-run with PEPIN_SESSION_LOGGER=1 to record the drive from this script instead."
    fi
    echo "recorded: $(ls "$rec" | grep -c "^${STAMP}_goto") files ros/maps/rec/${STAMP}_goto* (jsonl, log, board log, camera; cleanup trace in _goto_finish.log)"
}
trap finish EXIT
# ...and a signal ENDS the run, it does not merely mark it. bash defers a trap until the running
# foreground command returns, and the last thing this script did was a foreground `ssh | sed`
# that blocks until the board writes GOTO_EXIT: a TERM to this pid never reached that ssh, so the
# pipeline never ended, the trap never ran and the script sat there holding its ssh for hours
# (three of them, 1-2 h old, 2026-09-13). The watcher is a background job waited on instead —
# `wait` is interrupted by a trapped signal, the foreground pipeline was not — and `finish` kills
# it. `exit` here so the run really ends; `finish` runs once, whichever trap gets there first.
trap 'finish; exit 143' HUP TERM
# Any recorder left over from a run whose cleanup never ran would keep a core busy: clear it first.
ssh "root@$BOARD" "docker exec pepin-ros pkill -INT -f session_logger.py >/dev/null 2>&1; true"
# PEPIN_REC_CAMERA_SCANS=1 also tapes /depth_scan and /contact_scan on the board. Off by
# default: /contact_scan has no other consumer there, so recording it opens a bridge route from
# the laptop (the board carries what is real-time critical and nothing else).
REC_FLAGS=""
if [ -n "${PEPIN_REC_CAMERA_SCANS:-}" ]; then REC_FLAGS="--camera-scans"; fi
# The goal lives on the board (a WiFi hiccup must not become a cancel); this terminal only watches.
TAPE_FLAG=""
if [ "${PEPIN_GOTO_TAPE:-on}" = off ]; then TAPE_FLAG="--no-tape"; fi
# One recorder per drive. The run recorder is already subscribed to every one of these topics
# (it writes the numbered tape), so the session logger only runs when there is no numbered tape
# to write — or when it is asked for by name.
if [ -n "$TAPE_FLAG" ] || [ -n "${PEPIN_SESSION_LOGGER:-}" ]; then
    ssh "root@$BOARD" "docker exec -d pepin-ros /pepin_entrypoint.sh python3 /tools/session_logger.py $REC $MAX_REC_S $REC_FLAGS"
fi
ssh "root@$BOARD" "touch /root/pepin-ros$LOG; docker exec -d -e PYTHONUNBUFFERED=1 pepin-ros /pepin_entrypoint.sh sh -c 'python3 /tools/goto_ros.py --places $PLACES $TAPE_FLAG $* > $LOG 2>&1; echo GOTO_EXIT=\$? >> $LOG'"
echo "laptop: the goal was sent at $(date +%H:%M:%S.%2N)"
watch_start
# Backgrounded on purpose: see the trap above. `sed -u /q` ends it when the board writes
# GOTO_EXIT, and `finish` kills it when a signal ends the run first.
ssh "root@$BOARD" "tail -n +1 -F /root/pepin-ros$LOG 2>/dev/null | sed -u '/^GOTO_EXIT=/q'" 2>/dev/null &
TAILPID=$!
wait "$TAILPID" 2>/dev/null || true
