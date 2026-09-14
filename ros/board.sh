#!/bin/bash
# What the board is allowed to run, and what it actually runs (CLAUDE.md rule 20):
#   ros/board.sh census          one ps over ssh against config/board_manifest.json:
#                                per process measured CPU/RSS vs budget (OK / OVER / MISSING),
#                                unlisted processes above 1 % CPU, zombies, load vs the 4 cores.
#                                Exit 1 on a red verdict.
#   ros/board.sh census --json   the same census as data for a tool
#   ros/board.sh manifest        the registry alone: what runs there, why it is on the board and
#                                what it may cost. Reads the file, touches no host.
# The census is read-only and costs the board one ps: no ros2 CLI (a `ros2 node list` costs
# seconds of CPU on four A53 cores, 2026-09-13), no docker exec, nothing restarted. Run it after
# every deploy — ros/sync.sh ends with it — and whenever the board feels slow.
# The budgets and the reasons live in config/board_manifest.json; the parsing and the verdict in
# src/pepin/census.py.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
usage() { echo "usage: ros/board.sh [census [--json] | manifest]"; exit 2; }
census_py() { (cd "$HERE/.." && uv run -q python -m pepin.census "$@"); }
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
    *) usage ;;
esac
