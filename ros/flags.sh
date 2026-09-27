#!/bin/bash
# The nodes' feature flags (CLAUDE.md rule 19) from the laptop, wherever the node runs:
#   ros/flags.sh list [NODE]          every flag of every node (or of NODE): kind, current value, description
#   ros/flags.sh flag NODE FLAG       one flag in full: what it does, why its default is what it is,
#                                     when to turn it on, when to turn it off (no host touched)
#   ros/flags.sh get NODE FLAG        the current value, as the node holds it
#   ros/flags.sh set NODE FLAG VALUE  change it live; a value the flag refuses is refused here, with
#                                     the reason, before any host is touched
#   ros/flags.sh drift [NODE|board|laptop]   only what has been moved: every flag whose live value
#                                     differs from its table default, one per line, nothing at all
#                                     when every node is as it shipped (ros/restart.sh reads this)
# Each node declares its flags once, in the FLAGS table of ros/pepin_bringup/pepin_bringup/NODE.py,
# and its numbers in config/knobs.json; FLAG below is either (pepin.flags; ros/README.md lists
# them). ros/tools/flags_doc.py reads both, and
# pepin.deployment says where a node runs: the laptop's SLAM or navigation container (docker
# exec) or the board's pepin-ros (ssh, then docker exec). The ros2 CLI runs inside that
# container, on the node's own DDS domain. A change lives until the node restarts; a default
# changes in the table.
# WHERE THE PLANNER AND THE GOAL SERVER RUN is the board's PEPIN_SIDE, read once per call: split
# (PEPIN_SIDE=board) puts them in the laptop's pepin-laptop, which ros/laptop.sh starts in that
# mode only; a whole board runs them in pepin-ros. Until 2026-09-23 this script always exec'd
# into pepin-laptop, a container that does not exist on a whole board. PEPIN_BOARD_SIDE set by
# the caller skips the read. A board that does not answer falls back to the mode ros/laptop.sh
# last recorded in ros/.mode (split or vision; PEPIN_MODE_FILE names another file), and with none
# recorded the call is refused. Silence used to be read as a whole board (to 2026-09-23): a split
# stack with the board down lost the laptop's goal server and planner from every sweep.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
MODE_FILE="${PEPIN_MODE_FILE:-$HERE/.mode}"
. "$HERE/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command
usage() { echo "usage: ros/flags.sh [list [NODE] | drift [NODE|board|laptop] | flag NODE FLAG | get NODE FLAG | set NODE FLAG VALUE]"; exit 2; }
doc() { (cd "$HERE/.." && uv run -q python ros/tools/flags_doc.py "$@"); }
board() { ssh "root@$BOARD" "$@"; }  # the one path to the board
know_side() {  # PEPIN_BOARD_SIDE for flags_doc: the board's PEPIN_SIDE, empty on a whole board.
    # A board that does not answer is not a whole board: in the split the goal server and the
    # planner are the laptop's and reachable without it. The mode ros/laptop.sh last recorded
    # (ros/.mode: split or vision) answers instead, and with none recorded this refuses.
    local mode
    if [ -z "${PEPIN_BOARD_SIDE+set}" ]; then
        if ! PEPIN_BOARD_SIDE="$(board "grep -oE '^PEPIN_SIDE=[a-z]*' /etc/default/pepin-ros | cut -d= -f2" \
            2>/dev/null)"; then
            mode="$(cat "$MODE_FILE" 2>/dev/null || true)"
            case "$mode" in
                split) PEPIN_BOARD_SIDE=board ;;
                vision) PEPIN_BOARD_SIDE="" ;;
                *) echo "flags.sh: cannot read the board's side (root@$BOARD:/etc/default/pepin-ros)" \
                       "and $MODE_FILE records no mode: is the board up? (ros/laptop.sh start writes it)" >&2
                   exit 1 ;;
            esac
            echo "flags.sh: the board does not answer; its side taken from $MODE_FILE ($mode)" >&2
        fi
    fi
    export PEPIN_BOARD_SIDE
}
ros2_in() {  # NODE ros2 ...: the ros2 CLI inside the container NODE runs in (know_side first)
    local node="$1" where side container; shift
    where="$(doc where "$node")" || exit 2
    read -r side container <<<"$where"
    if [ "$side" = laptop ]; then
        docker exec "$container" /pepin_entrypoint.sh "$@"
    else
        board "docker exec $container /pepin_entrypoint.sh $(printf '%q ' "$@")"
    fi
}
case "${1:-list}" in
    list)
        [ $# -le 2 ] || usage
        if [ -n "${2:-}" ]; then NODES="$(doc where "$2" >/dev/null && echo "$2")" || exit 2; else NODES="$(doc nodes)"; fi
        know_side
        for node in $NODES; do
            # one dump per node (a parameter get is a second of discovery each): the values;
            # the kinds and the descriptions come from the table
            dump="$(ros2_in "$node" ros2 param dump "/$node" 2>/dev/null || true)"
            doc list "$node" <<<"$dump"
        done ;;
    drift)
        # What is NOT as the table declares it. A restart puts every flag back to its default, so
        # a line here is a switch someone moved — the one thing a restart silently throws away
        # (2026-09-14: a live flag lost at a node restart went unnoticed for an hour).
        [ $# -le 2 ] || usage
        case "${2:-}" in board | laptop | "") ;; *) doc where "$2" >/dev/null || exit 2 ;; esac
        know_side
        case "${2:-}" in
            "") NODES="$(doc nodes)" ;;
            board | laptop) NODES="$(doc nodes "$2")" ;;
            *) NODES="$(doc where "$2" >/dev/null && echo "$2")" || exit 2 ;;
        esac
        for node in $NODES; do
            dump="$(ros2_in "$node" ros2 param dump "/$node" 2>/dev/null || true)"
            doc drift "$node" <<<"$dump" || true  # a node that is down says so and does not end the sweep
        done ;;
    flag)  # the table alone: the reading matter, no node asked, no container entered
        [ $# -eq 3 ] || usage
        doc flag "$2" "$3" ;;
    get)
        [ $# -eq 3 ] || usage
        doc flag "$2" "$3" >/dev/null || exit 2
        know_side
        ros2_in "$2" ros2 param get "/$2" "$3" ;;
    set)
        [ $# -eq 4 ] || usage
        LITERAL="$(doc value "$2" "$3" "$4")" || exit 2  # the flag's own check, with the reason
        know_side
        ros2_in "$2" ros2 param set "/$2" "$3" "$LITERAL" ;;
    *) usage ;;
esac
