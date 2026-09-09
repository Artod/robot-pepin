#!/bin/bash
# Send the robot somewhere, instantly: the goal server on the board is already connected to Nav2,
# so a command costs one ssh hop and a socket write instead of booting a client (8-15 s before).
#   ros/go.sh printer | home | NAME     drive to a named place
#   ros/go.sh -1.0 0.3 [YAW_DEG]        drive to map coordinates
#   ros/go.sh mark NAME                 remember this spot under NAME
#   ros/go.sh where | places | cancel
#   ros/go.sh planner navfn|lattice|theta|smac|hybrid   swap the planner (and its controller)
#   ros/go.sh trip                      printer, then home — the round trip, one command
# Ctrl-C closes the connection and cancels the goal.
# A drive brings its own tape home: the board's jsonl (scans, odometry, tracked pose), the
# container log, and the head camera — any goal may turn out to be the clip worth posting.
set -uo pipefail
# A dead ffmpeg must not kill this script: writing "q" into its fifo raises SIGPIPE in the shell's
# own printf, and the trip silently stopped after the printer whenever the camera was down
# (runs 0086, 0088). With the signal ignored the write just fails and the verdict still prints.
trap '' PIPE
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"
PORT=3337
case "${1:-}" in
    trip) "$0" printer && "$0" home; exit $? ;;
    "") echo "usage: ros/go.sh printer | home | X Y [YAW] | mark NAME | where | places | cancel | planner navfn|lattice|theta|smac|hybrid | trip"; exit 2 ;;
    mark)   REQUEST="{\"cmd\":\"mark\",\"name\":\"${2:?a name}\"}" ;;
    where)  REQUEST='{"cmd":"where"}' ;;
    places) REQUEST='{"cmd":"places"}' ;;
    cancel) REQUEST='{"cmd":"cancel"}' ;;
    planner) REQUEST="{\"cmd\":\"planner\",\"name\":\"${2:?navfn, lattice, theta or smac}\"}" ;;
    -*|[0-9]*) REQUEST="{\"cmd\":\"go\",\"x\":$1,\"y\":${2:?y},\"yaw_deg\":${3:-0}}" ;;
    *)      REQUEST="{\"cmd\":\"go\",\"place\":\"$1\"}" ;;
esac
# One transport decision serves both the command and the Ctrl-C cancel: a goal server on this
# laptop (ros/laptop.sh) is spoken to directly, otherwise the board's over ssh.
if (exec 3<>/dev/tcp/127.0.0.1/$PORT) 2>/dev/null; then
    talk() { bash -c "exec 3<>/dev/tcp/127.0.0.1/$PORT; printf '%s\\n' \"\$1\" >&3; cat <&3" -- "$1"; }
else
    talk() { ssh "root@$BOARD" "exec 3<>/dev/tcp/127.0.0.1/$PORT; printf '%s\\n' '$1' >&3; cat <&3"; }
fi
CANCELLED=0
trap 'CANCELLED=1; talk "{\"cmd\":\"cancel\"}" >/dev/null 2>&1' INT
REPLY_FILE=$(mktemp); REPLY_COPY=$(mktemp)
# The first line of a drive names the run and the planner in plain words, before the JSON:
# a drive whose planner has to be guessed afterwards is a drive that cannot be compared.
talk "$REQUEST" \
  | tee "$REPLY_FILE" \
  | awk '{print; fflush()} /"event": "accepted"/ { match($0, /"run": [0-9]+/); r=substr($0, RSTART+7, RLENGTH-7); match($0, /"planner": "[^"]+"/); p=substr($0, RSTART+12, RLENGTH-13); match($0, /"place": "[^"]+"/); pl=substr($0, RSTART+10, RLENGTH-11); print "=== run #" r " · planner " p " -> " pl " ==="; fflush() }' 
# The run recorded itself on the board; bring it home so every drive is on the laptop too.
RECORDING=$(grep -o '"recording": *"[^"]*"' "$REPLY_FILE" | tail -1 | cut -d'"' -f4)
cp "$REPLY_FILE" "$REPLY_COPY"; rm -f "$REPLY_FILE"
if [ -n "$RECORDING" ]; then
    HERE="$(cd "$(dirname "$0")" && pwd)"; mkdir -p "$HERE/maps/rec"
    STAMP=$(basename "$RECORDING" .jsonl)
    ssh "root@$BOARD" "docker logs --since 15m pepin-ros 2>&1" > "$HERE/maps/rec/${STAMP}_board.log" 2>/dev/null
    cp "$REPLY_COPY" "$HERE/maps/rec/${STAMP}_reply.jsonl"  # every event the server sent, kept with the run
    rsync -aq "root@$BOARD:/root/pepin-ros${RECORDING}" "$HERE/maps/rec/" 2>/dev/null
    # The clip was captured on the board (goal_server, curl on the MJPEG stream): fetched with the
    # tape and wrapped into mkv here, from a local file — no network permission can break it.
    CLIP="$HERE/maps/rec/${STAMP}_cam.mjpeg"
    rsync -aq "root@$BOARD:/root/pepin-ros${RECORDING%.jsonl}_cam.mjpeg" "$HERE/maps/rec/" 2>/dev/null
    if [ -s "$CLIP" ]; then
        ffmpeg -loglevel error -y -framerate 15 -f mjpeg -i "$CLIP" -c copy "$HERE/maps/rec/${STAMP}_cam.mkv" \
            && rm -f "$CLIP"
    else
        echo "no camera clip for this run (the stream gave nothing on the board)"
    fi
    # Which planner drove and how often it refused: the tree has no fallback planner any more (a
    # refusal is a recovery, never a smaller robot), so a refusal count is the drive's own story.
    LOG="$HERE/maps/rec/${STAMP}_board.log"
    CHOSEN=$(grep -o '"planner": *"[^"]*"' "$REPLY_COPY" 2>/dev/null | tail -1 | cut -d'"' -f4)
    # Only this run's lines: the fetched log spans fifteen minutes, and counting the previous
    # runs' refusals credited NavFn with a drive Theta* had done entirely (runs 0049-0051).
    if THIS_RUN=$(awk '/Begin navigating/{n=NR} {l[NR]=$0} END{if (!n) exit 1; for(i=n;i<=NR;i++) print l[i]}' "$LOG" 2>/dev/null); then
        FELL_BACK=$(printf '%s\n' "$THIS_RUN" | grep -c "plugin failed to plan" || true)
        PLANS=$(printf '%s\n' "$THIS_RUN" | grep -c "Passing new path" || true)
        if [ "$FELL_BACK" -gt 0 ]; then
            WHO="${CHOSEN:-?} refused $FELL_BACK time(s), recovered ($PLANS plans)"
        else
            WHO="planned by ${CHOSEN:-?} throughout ($PLANS plans)"
        fi
    else
        WHO="planner unknown: this run's start is not in the fetched log"  # never a made-up count
    fi
    # The run's number is how Artem names a drive out loud ("look at run 37"), so print it loud.
    echo "=== run #${STAMP%%_*} === $WHO"
    echo "    ros/maps/rec/${STAMP}.{jsonl,_board.log$([ -f "$HERE/maps/rec/${STAMP}_cam.mkv" ] && echo ,_cam.mkv)}"
fi
# The exit status is the drive's verdict, not the last cleanup command's: a chained
# "ros/go.sh printer && ros/go.sh home" must run home after a reached goal and stop after a
# failed or refused one. Nav2 status 4 is SUCCEEDED; anything else, or no "done" at all, is not.
VERDICT=0; WHY="ok"
[ -s "$REPLY_COPY" ] || { VERDICT=1; WHY="nobody answered"; }
if ERR=$(grep '"event": "error"' "$REPLY_COPY" 2>/dev/null | tail -1); then
    VERDICT=1; WHY="refused: $(printf '%s' "$ERR" | grep -o '"detail": *"[^"]*"' | cut -d'"' -f4)"
fi
case "$REQUEST" in '{"cmd":"go"'*)
    # keyed on the two fields, not on their order: a reordered event once made every reached
    # goal exit 1 and "trip" stop after the printer (runs 0049-0051)
    DONE=$(grep '"event": "done"' "$REPLY_COPY" 2>/dev/null | tail -1)
    STATUS=$(printf '%s' "$DONE" | grep -o '"status": *[0-9]*' | grep -o '[0-9]*$')
    [ "$CANCELLED" -eq 1 ] && STATUS=ctrlc
    case "${STATUS:-none}" in
        ctrlc) VERDICT=1; WHY="cancelled with Ctrl-C" ;;
        4) [ "$VERDICT" -eq 0 ] && WHY="reached" ;;
        5) VERDICT=1; WHY="cancelled" ;;
        6) VERDICT=1; WHY="aborted by Nav2" ;;
        none) VERDICT=1; WHY="no done event (connection lost?)" ;;
        *) VERDICT=1; WHY="Nav2 status $STATUS" ;;
    esac
    echo "=== verdict: $WHY (exit $VERDICT) ===" ;;
esac
rm -f "$REPLY_COPY" 2>/dev/null
exit $VERDICT
