#!/bin/bash
# Send the robot somewhere, instantly: the goal server on the board is already connected to Nav2,
# so a command costs one ssh hop and a socket write instead of booting a client (8-15 s before).
#   ros/go.sh printer | home | NAME     drive to a named place
#   ros/go.sh -1.0 0.3 [YAW_DEG]        drive to map coordinates
#   ros/go.sh mark NAME                 remember this spot under NAME
#   ros/go.sh where | places | cancel
# Ctrl-C closes the connection and cancels the goal.
set -uo pipefail
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$(dirname "$0")/lib.sh"
PORT=3337
case "${1:-}" in
    "") echo "usage: ros/go.sh printer | home | X Y [YAW] | mark NAME | where | places | cancel"; exit 2 ;;
    mark)   REQUEST="{\"cmd\":\"mark\",\"name\":\"${2:?a name}\"}" ;;
    where)  REQUEST='{"cmd":"where"}' ;;
    places) REQUEST='{"cmd":"places"}' ;;
    cancel) REQUEST='{"cmd":"cancel"}' ;;
    -*|[0-9]*) REQUEST="{\"cmd\":\"go\",\"x\":$1,\"y\":${2:?y},\"yaw_deg\":${3:-0}}" ;;
    *)      REQUEST="{\"cmd\":\"go\",\"place\":\"$1\"}" ;;
esac
trap 'ssh "root@$BOARD" "printf %s\\\\n {\\\"cmd\\\":\\\"cancel\\\"} | timeout 3 bash -c \"exec 3<>/dev/tcp/127.0.0.1/$PORT; cat >&3\"" 2>/dev/null' INT
ssh "root@$BOARD" "exec 3<>/dev/tcp/127.0.0.1/$PORT; printf '%s\n' '$REQUEST' >&3; cat <&3"
