#!/bin/bash
# Drive the robot through Nav2 on the board, with feedback. Usage:
#   ros/goto.sh X Y [YAW_DEG]    drive to map coordinates (meters, degrees)
#   ros/goto.sh home             back to the start spot (map origin, facing as at start)
#   ros/goto.sh mark NAME        stand the robot somewhere: remember that spot as NAME — a labelled
#                                RTAB-Map node plus the cart's offset from it, so the place rides
#                                the node when a loop closure bends the map
#   ros/goto.sh NAME             drive to a remembered place: the graph's book (/places, latched)
#                                first, the map file's coordinates second and with a warning
#   ros/goto.sh places           list both books, each entry saying which one it came from
#   ros/goto.sh seed X Y [YAW]   after placing the robot by hand: tell RTAB-Map where it is
#   ros/goto.sh cancel           cancel every goal on the board's navigators; it prints what came
#                                of it within 30 s (ros/stop.sh is the hard stop that also brakes)
#   ros/goto.sh where            the pose right now (map -> base_link, from the goal server)
# Before a goal is sent, goto prints its preflight, one line per check, and any REFUSED stops the
# drive with the reading behind it:
#   preflight frame      ok       map -> base_link 42 ms old
#   preflight placement  ok       ...RTAB-Map's start recognised the loaded map (or was seeded)
# Every run is taped ONCE, by the board's run recorder: the numbered tape
# 0249_<utc>Z_<place>.jsonl, opened on the goal's word — scans, odometry, the tracked pose, the
# commands, the costmap, the EKF and IMU, the ToF, with the camera clip beside it; goto names it in its
# log and fetches it here. Until 2026-09-14 this script also started a second recorder in the
# container (ros/tools/session_logger.py) which re-deserialised the same 10 Hz lidar stream for
# 15 % of a core; the numbered tape now carries everything it carried.
# PEPIN_SESSION_LOGGER=1 starts it anyway, next to the numbered tape (its file is named by the
# LAPTOP's local clock: ros/maps/rec/<stamp>_goto.jsonl). With PEPIN_GOTO_TAPE=off it is started
# on its own, so a drive without a numbered tape is still recorded. The two clocks are explained
# in ros/maps/README.md.
# PEPIN_REC_CAMERA_SCANS=1 adds /depth_scan, /depth_marks and /contact_scan to the session
# logger's tape.
# PEPIN_GOTO_TAPE=off drives without asking the recorder for a numbered one.
# With PEPIN_RECORDER=bag on the board (ros/feature.sh recorder bag) the same word opens an MCAP
# BAG instead of a tape — `ros2 bag record` under pepin_bringup.bag_recorder, which costs the
# board a copy of serialised bytes instead of 34-43 % of a core. This script fetches the bag and
# turns it into the very same numbered tape here (ros/tools/bag_to_tape.py in pepin-vslam), so
# nothing downstream changes; ros/README.md, "Two recorders".
set -euo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
# PEPIN_GOAL_TCP: what talks to the board's goal server over its socket (port 3337, from this
# laptop, src/pepin/goal_link.py) instead of starting a ROS process on the board. Every such
# process is a new zenoh session, and each stalled ALL laptop -> board delivery for 2.6-3.1 s
# about 1.5 s after it started, then again when it exited (journal 2026-09-25) — a cancel did
# that in the middle of the drive it was stopping.
#   1 (default)  cancel and where; the goal itself still starts goto_ros.py on the board
#   goal         cancel, where AND the goal (no preflight lines, the goal server's own gate)
#   0            the old path for everything
# A goal server that does not answer, whose cancel reaches only its own goal (a build before its
# flag cancel_every_goal) or whose cancel no navigator confirmed sends the command down the old
# path, with a warning: a dead goal server never leaves the operator without a cancel.
GOAL_TCP="${PEPIN_GOAL_TCP:-1}"
GOAL_PORT="${PEPIN_GOAL_PORT:-3337}"  # = pepin_bringup.goal_server.PORT
goal_link() { PYTHONPATH="$(dirname "$0")/../src" python3 -m pepin.goal_link --host "$BOARD" --port "$GOAL_PORT" "$@"; }
if [ "$GOAL_TCP" != 0 ]; then
    case "${1:-}" in
        cancel | where)
            if goal_link "$1"; then exit 0; fi
            echo "!! the goal server did not do it (above): the old path, a ROS process on the board"
            ;;
    esac
fi
MAP=$(ssh "root@$BOARD" "grep -oE 'PEPIN_MAP=.*' /etc/default/pepin-ros" | cut -d= -f2)
PLACES="/maps/$(basename "${MAP:-places}" .yaml).places.yaml"  # one book of places per map
case "${1:-}" in
  # The pose lives in TF (map -> base_link), which the goal server's socket composes and answers on.
  where) "$(dirname "$0")/go.sh" where; exit ;;
  # The header promised this for weeks while the case fell through to "drive to a place called
  # cancel" (2026-09-14 11:20: the cart went on butting a table for a minute after the "cancel").
  cancel) ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh timeout 25 python3 /tools/goto_ros.py cancel"; exit ;;
  # A whole-map search was the board tracker's recovery (tag alt/tracker-2026-09-22): RTAB-Map
  # recognises the room by itself, so the refusal says so rather than hang on a missing service.
  relocalize)
    echo "no whole-map search in this stack: RTAB-Map recognises the room by itself; drive the cart where it can see more of it, or seed it (ros/goto.sh seed X Y YAW)"; exit 2 ;;
  # A place lives in RTAB-Map's GRAPH now: the client asks the laptop's places node over the bridge
  # (/places/mark, answered on /places/marked) and the book is written on the LAPTOP, beside the
  # graph database it hangs on (ros/maps/rtabmap.places.json) — a node id means nothing without the
  # database it is an id in, so the two stay together and nothing is rsynced back from the board.
  # --places still names the old per-map file: it is the fallback the client falls back TO, and
  # `places` prints both books with each entry saying which it came from.
  mark|places) ssh "root@$BOARD" "docker exec pepin-ros /pepin_entrypoint.sh python3 /tools/goto_ros.py --places $PLACES $*"; exit ;;
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
# The goal through the goal server's socket (PEPIN_GOAL_TCP=goal): its log is written HERE, and
# nothing of this drive runs on the board but the server itself. A seed is goto_ros.py's alone.
TCP_GOAL=0
if [ "$GOAL_TCP" = goal ] && [ "${1:-}" != seed ]; then TCP_GOAL=1; fi
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
        # A cancel and nothing else, asked twice: never ros/stop.sh from here. Its hard branch
        # restarts the board's stack, which zeroes the odometry — the pose and the volume are
        # lost and both halves need a restart (2026-09-28 21:48 and 2026-09-29 00:17, each an
        # interrupted drive). The red button stays a command the operator types.
        if goal_link cancel || goal_link cancel; then :; else
            echo "!! the cancel was NOT confirmed: if the cart still moves, ros/stop.sh is the hard stop (it restarts the board)"
        fi
    elif [ "$TCP_GOAL" = 1 ]; then
        :  # no process of this drive on the board to look for; pepin.goal_link said how it ended
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
    if [ "$TCP_GOAL" = 1 ]; then
        echo "fetched: nothing, the goal's log was written here" >> "$trace"
    else
        sleep 1
        rsync -aq "root@$BOARD:/root/pepin-ros/maps/rec/${STAMP}_goto.*" "$rec/" >> "$trace" 2>&1 || { sleep 2; rsync -aq "root@$BOARD:/root/pepin-ros/maps/rec/${STAMP}_goto.*" "$rec/" >> "$trace" 2>&1; }
        echo "fetched: $?" >> "$trace"
    fi
    # The numbered recording the recorder opened for this goal, named in the log we just fetched:
    # it comes home too (the clip stays on the board — this script films the drive itself). Which
    # recorder wrote it is read off the name and never asked of the board: the JSONL recorder
    # opens a FILE ending in .jsonl, the bag recorder a DIRECTORY (`ros2 bag record -o`), and a
    # bag is turned into the very same tape here, so the operator always ends with one
    # ros/maps/rec/NNNN_*.jsonl whichever half wrote the drive.
    local taped
    taped=$(grep -o 'taped /maps/rec/[^ ]*' "$rec/${STAMP}_goto.log" 2>/dev/null | tail -1 | cut -d' ' -f2 || true)
    if [ -n "$taped" ]; then
        rsync -aq "root@$BOARD:/root/pepin-ros$taped" "$rec/" >> "$trace" 2>&1 || true
        echo "fetched recording: $?" >> "$trace"
        case "$taped" in
            *.jsonl) echo "numbered tape: ros/maps/rec/$(basename "$taped")" ;;
            *)
                echo "numbered bag: ros/maps/rec/$(basename "$taped"); converting to a tape"
                if pepin_bag_to_tape "$taped" >> "$trace" 2>&1; then
                    echo "numbered tape: ros/maps/rec/$(basename "$taped").jsonl"
                else
                    echo "!! the bag is here but not converted: see $trace"
                fi
                ;;
        esac
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
# PEPIN_REC_CAMERA_SCANS=1 also tapes /depth_scan, /depth_marks and /contact_scan on the board.
# Off by default: /contact_scan has no other consumer there, so recording it opens a route from
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
if [ "$TCP_GOAL" = 1 ]; then
    # The goal server's socket: goto_ros.py's lines (taped ..., t+ ..., result: ...) rendered by
    # pepin.goal_link into this run's log HERE, where finish reads the tape's name. The server
    # tapes every goal itself, so PEPIN_GOTO_TAPE=off has no counterpart on this path. Started
    # as a plain command, not through the function, so TAILPID is the client and not a subshell.
    if [ -n "$TAPE_FLAG" ]; then echo "!! PEPIN_GOTO_TAPE=off: the goal server tapes every goal anyway"; fi
    PYTHONPATH="$(dirname "$0")/../src" python3 -m pepin.goal_link --host "$BOARD" --port "$GOAL_PORT" --log "$(dirname "$0")$LOG" go "$@" &
    TAILPID=$!
    echo "laptop: the goal was sent at $(date +%H:%M:%S) through the goal server's socket"
    watch_start
    wait "$TAILPID" 2>/dev/null || true
    exit 0
fi
ssh "root@$BOARD" "touch /root/pepin-ros$LOG; docker exec -d -e PYTHONUNBUFFERED=1 pepin-ros /pepin_entrypoint.sh sh -c 'python3 /tools/goto_ros.py --places $PLACES $TAPE_FLAG $* > $LOG 2>&1; echo GOTO_EXIT=\$? >> $LOG'"
echo "laptop: the goal was sent at $(date +%H:%M:%S.%2N)"
watch_start
# Backgrounded on purpose: see the trap above. `sed -u /q` ends it when the board writes
# GOTO_EXIT, and `finish` kills it when a signal ends the run first.
ssh "root@$BOARD" "tail -n +1 -F /root/pepin-ros$LOG 2>/dev/null | sed -u '/^GOTO_EXIT=/q'" 2>/dev/null &
TAILPID=$!
wait "$TAILPID" 2>/dev/null || true
