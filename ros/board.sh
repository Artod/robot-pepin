#!/bin/bash
# What the board is allowed to run, and what it actually runs (CLAUDE.md rule 20):
#   ros/board.sh census          one ps over ssh against config/board_manifest.json:
#                                per process measured CPU/RSS vs budget (OK / OVER / MISSING),
#                                unlisted processes above 1 % CPU, zombies, load vs the 4 cores.
#                                Exit 1 on a red verdict.
#   ros/board.sh census --json   the same census as data for a tool
#   ros/board.sh manifest        the registry alone: what runs there, why it is on the board and
#                                what it may cost. Reads the file, touches no host.
#   ros/board.sh kick NODE       restart one node of the board's stack from the synced sources
#                                (seconds, the stack untouched)
# The census is read-only and costs the board one ps: no ros2 CLI (a `ros2 node list` costs
# seconds of CPU on four A53 cores, 2026-09-13), no docker exec, nothing restarted. Run it after
# every deploy — ros/sync.sh ends with it — and whenever the board feels slow.
# The budgets and the reasons live in config/board_manifest.json; the parsing and the verdict in
# src/pepin/census.py.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
usage() { echo "usage: ros/board.sh [census [--json] | manifest | kick NODE]"; exit 2; }
census_py() { (cd "$HERE/.." && uv run -q python -m pepin.census "$@"); }
# The nodes a kick can reach on the board and the line each prints once up (the kick waits for
# it): our own respawned processes of robot.launch.py. The navigation nodes are the Mac's
# (ros/laptop.sh kick).
KICKABLE="neck_state tof_bridge"
kick_line() {  # node name -> start-up line
    case "$1" in
        neck_state) echo "neck state up: " ;;
        tof_bridge) echo "tof ceilings: " ;;
        *) return 1 ;;
    esac
}
case "${1:-census}" in
    census)
        [ $# -le 2 ] || usage
        [ -z "${2:-}" ] || [ "${2:-}" = --json ] || usage
        # The board-side command comes from pepin.census itself: the dump the parser reads is
        # the dump the parser was written for, and only one file has to change to alter it.
        ssh "root@$BOARD" "$(census_py --command)" | census_py ${2:+"$2"} ;;
    manifest)
        [ $# -eq 1 ] || usage
        census_py --manifest ;;
    kick)
        # One node of the stack, not the stack: `ros/sync.sh` (or ros/push.sh) puts the sources
        # on the board, this ends the node with SIGINT (what the launch sends at shutdown) and the
        # launch respawns it from those sources two seconds after its exit (RESPAWN in
        # robot.launch.py). A kick is the node's own start-up plus the two-second pause. Kick at
        # rest, never mid-drive.
        #   The wait proves the ready line is the NEW process's (ros/kick_ready.awk): the launch's
        # exit line for the signalled pid names the process's tag, the successor's start line
        # under that tag gives the new pid, and only a ready line under that tag after it counts
        # — never a `--since` grep, which returned the old process's line (2026-09-20). The times
        # are the container's clock: the kick's own `date` and the log's timestamps. The
        # arguments reach the board quoted (ssh joins its arguments into one command line:
        # unquoted, the ready line was cut at its first space) and the matcher travels with them,
        # so a board that has not been synced since it changed still waits the same way.
        NAME="${2:-}"; LINE="$(kick_line "$NAME")" || { echo "usage: ros/board.sh kick <node>; nodes: $KICKABLE"; exit 2; }
        { printf 'AWK=%q\n' "$(cat "$HERE/kick_ready.awk")"; cat <<'EOF'; } | ssh "root@$BOARD" "bash -s -- $(printf '%q ' "$NAME" "$LINE")"
set -u
NAME=$1; LINE=$2; TAB=$(printf '\t')
# The console scripts run as pepin_bringup/<name>, the modules as pepin_bringup.<name>. One exec
# reads the container's clock and the pids to signal.
OUT=$(docker exec pepin-ros sh -c 'date -u +%FT%T.%NZ; pgrep -f "pepin_bringup[./]$1"' sh "$NAME" 2>/dev/null) || true
KICKED=${OUT%%$'\n'*}; OLD=$(printf '%s\n' "$OUT" | sed 1d | tr '\n' ' ')
[ -n "${OLD// /}" ] || { echo "no $NAME process in pepin-ros"; exit 3; }
# shellcheck disable=SC2086
docker exec pepin-ros sh -c 'kill -INT "$@"' sh $OLD
R="wait${TAB}nothing read from the log yet"
for _ in $(seq 1 240); do
    R=$(docker logs -t --since "$KICKED" pepin-ros 2>&1 | awk -v name="$NAME" -v old="$OLD" -v line="$LINE" -v kicked="$KICKED" "$AWK")
    if [ "${R%%"$TAB"*}" = ready ]; then
        echo "${R#*"$TAB"}"
        exit 0
    fi
    sleep 0.5
done
echo "$NAME not ready within 120 s: ${R#*"$TAB"} (docker logs pepin-ros)"; exit 4
EOF
        ;;
    *) usage ;;
esac
