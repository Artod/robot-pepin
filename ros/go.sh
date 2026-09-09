#!/bin/bash
# Send the robot somewhere, instantly: the goal server on the board is already connected to Nav2,
# so a command costs one ssh hop and a socket write instead of booting a client (8-15 s before).
#   ros/go.sh printer | home | NAME     drive to a named place
#   ros/go.sh -1.0 0.3 [YAW_DEG]        drive to map coordinates
#   ros/go.sh mark NAME                 remember this spot under NAME
#   ros/go.sh where | places | cancel
#   ros/go.sh planner navfn|lattice|theta   swap the planner (and the controller that follows it)
# Ctrl-C closes the connection and cancels the goal.
# A drive brings its own tape home: the board's jsonl (scans, odometry, tracked pose), the
# container log, and the head camera — any goal may turn out to be the clip worth posting.
set -uo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"
PORT=3337
case "${1:-}" in
    "") echo "usage: ros/go.sh printer | home | X Y [YAW] | mark NAME | where | places | cancel | planner navfn|lattice|theta"; exit 2 ;;
    mark)   REQUEST="{\"cmd\":\"mark\",\"name\":\"${2:?a name}\"}" ;;
    where)  REQUEST='{"cmd":"where"}' ;;
    places) REQUEST='{"cmd":"places"}' ;;
    cancel) REQUEST='{"cmd":"cancel"}' ;;
    planner) REQUEST="{\"cmd\":\"planner\",\"name\":\"${2:?navfn, lattice or theta}\"}" ;;
    -*|[0-9]*) REQUEST="{\"cmd\":\"go\",\"x\":$1,\"y\":${2:?y},\"yaw_deg\":${3:-0}}" ;;
    *)      REQUEST="{\"cmd\":\"go\",\"place\":\"$1\"}" ;;
esac
trap 'ssh "root@$BOARD" "printf %s\\\\n {\\\"cmd\\\":\\\"cancel\\\"} | timeout 3 bash -c \"exec 3<>/dev/tcp/127.0.0.1/$PORT; cat >&3\"" 2>/dev/null' INT
CAM=""; FFPID=""
case "$REQUEST" in '{"cmd":"go"'*)
    CAM="$(mktemp -u)_cam.mkv"  # mjpeg copied as-is: no re-encoding, no CPU taken from the drive
    # Stopped by writing "q" to its stdin, not by a signal: SIGINT makes ffmpeg abandon the file
    # and print "Error during demuxing: Immediate exit requested", which looks like a failed run
    # and is not one. The fifo's write end is held open so ffmpeg never sees an early EOF.
    CAM_FIFO="$(mktemp -u)"; mkfifo "$CAM_FIFO"
    ffmpeg -loglevel error -y -f mjpeg -use_wallclock_as_timestamps 1 \
           -i "http://$BOARD:8080/stream" -c copy "$CAM" < "$CAM_FIFO" & FFPID=$!
    exec 4>"$CAM_FIFO"
    trap 'printf q >&4 2>/dev/null; sleep 1; kill -INT $FFPID 2>/dev/null' EXIT
    ;;
esac
REPLY_FILE=$(mktemp); REPLY_COPY=$(mktemp)
ssh "root@$BOARD" "exec 3<>/dev/tcp/127.0.0.1/$PORT; printf '%s\n' '$REQUEST' >&3; cat <&3" | tee "$REPLY_FILE"
if [ -n "$FFPID" ]; then
    printf q >&4 2>/dev/null; exec 4>&-          # ffmpeg finishes the file and exits quietly
    for _ in 1 2 3 4 5 6; do kill -0 "$FFPID" 2>/dev/null || break; sleep 0.5; done
    kill -INT "$FFPID" 2>/dev/null; wait "$FFPID" 2>/dev/null; rm -f "$CAM_FIFO"; trap - EXIT
fi
# The run recorded itself on the board; bring it home so every drive is on the laptop too.
RECORDING=$(grep -o '"recording": *"[^"]*"' "$REPLY_FILE" | tail -1 | cut -d'"' -f4)
cp "$REPLY_FILE" "$REPLY_COPY"; rm -f "$REPLY_FILE"
if [ -n "$RECORDING" ]; then
    HERE="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$HERE/maps/rec"
    STAMP=$(basename "$RECORDING" .jsonl)
    ssh "root@$BOARD" "docker logs --since 15m pepin-ros 2>&1" > "$HERE/maps/rec/${STAMP}_board.log" 2>/dev/null
    rsync -aq "root@$BOARD:/root/pepin-ros${RECORDING}" "$HERE/maps/rec/" 2>/dev/null
    # An empty file means the camera server was down; a missing clip must not look like a recorded one.
    if [ -n "$CAM" ] && [ -s "$CAM" ]; then mv "$CAM" "$HERE/maps/rec/${STAMP}_cam.mkv"; fi
    # Which planner actually drove. The tree tries the selected one and falls back to NavFn when
    # it refuses, and that substitution is invisible from the outside — so say it out loud, every
    # run, rather than letting a good drive be credited to the wrong planner.
    LOG="$HERE/maps/rec/${STAMP}_board.log"
    CHOSEN=$(grep -o '"planner": *"[^"]*"' "$REPLY_COPY" 2>/dev/null | tail -1 | cut -d'"' -f4)
    FELL_BACK=$(grep -c "plugin failed to plan" "$LOG" 2>/dev/null || echo 0)
    PLANS=$(grep -c "Passing new path" "$LOG" 2>/dev/null || echo 0)
    if [ "$FELL_BACK" -gt 0 ]; then
        WHO="${CHOSEN:-?} refused $FELL_BACK time(s), NavFn took over ($PLANS plans)"
    else
        WHO="planned by ${CHOSEN:-?} throughout ($PLANS plans)"
    fi
    # The run's number is how Artem names a drive out loud ("look at run 37"), so print it loud.
    echo "=== run #${STAMP%%_*} === $WHO"
    echo "    ros/maps/rec/${STAMP}.{jsonl,_board.log$([ -f "$HERE/maps/rec/${STAMP}_cam.mkv" ] && echo ,_cam.mkv)}"
fi
rm -f "$CAM" "$REPLY_COPY" 2>/dev/null
