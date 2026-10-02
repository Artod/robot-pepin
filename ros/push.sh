#!/bin/bash
# Put changed files on the running robot and restart only the nodes that import them. Usage:
#   ros/push.sh [--dry-run] FILE...     (PEPIN_PUSH_DRY=1 is the same as --dry-run)
#
# The plan is pepin.push's (`uv run python -m pepin.push plan FILE...`): every running node whose
# Python imports a changed module, through any chain of imports, on each half — or why a file needs
# a restart instead: a launch file, a params file, a unit, config, a module a launch file imports
# (the launch process keeps it), an image layer. Then, in this order:
#   1. a refusal touches nothing (exit 2); --dry-run prints the plan and what would run, and stops
#   2. the laptop's containers must mount THIS checkout: they read the sources live, no copy
#   3. a process of ours the launch does not respawn (RTAB-Map's XFeat adapters) that imports a
#      change refuses the push while it runs
#   4. rsync of exactly these files to the board, never --delete: the board writes files of its
#      own beside ours (maps/rec)
#   5. the kicks, all at once: ros/board.sh kick on the board, ros/laptop.sh kick here, each
#      waiting for its node's NEW pid and that pid's own ready line (ros/kick_ready.awk); a
#      node that is not running on its half (the other recorder, the goal server's other side,
#      the tracker under RTAB-Map) is skipped
# One line per node: kicked at, ready at, seconds. Exit 0 every kicked node is back, 1 a kick
# failed, 2 refused. A kick takes its node away for its seconds: push at rest, never mid-drive.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/.." && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"  # one multiplexed ssh: the preflight opens it, rsync and the kicks reuse it
START=$SECONDS
TAB="$(printf '\t')"
usage() { echo "usage: ros/push.sh [--dry-run] FILE...   (PEPIN_PUSH_DRY=1: the same as --dry-run)"; exit 2; }

DRY="${PEPIN_PUSH_DRY:-0}"
FILES=()
for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY=1 ;;
        -*) usage ;;
        /*) FILES+=("$arg") ;;
        *) FILES+=("$PWD/$arg") ;;  # the planner runs from the checkout's root
    esac
done
[ ${#FILES[@]} -gt 0 ] || usage
planner() { (cd "$ROOT" && uv run -q python -m pepin.push plan --repo "$ROOT" "$@"); }

STATUS=0
planner "${FILES[@]}" || STATUS=$?
[ "$STATUS" -eq 0 ] || exit "$STATUS"
PLAN="$(planner --shell "${FILES[@]}")"
ROS_FILES=(); LIB_FILES=(); KICKS=(); HELD=()
while IFS="$TAB" read -r kind a b c d e f; do
    case "$kind" in
        file) case "$a" in src/*) LIB_FILES+=("${a#src/}") ;; ros/*) ROS_FILES+=("${a#ros/}") ;; esac ;;
        kick) KICKS+=("$a $b") ;;
        held) HELD+=("$a$TAB$b$TAB$c$TAB$d$TAB$e$TAB$f") ;;
    esac
done <<<"$PLAN"
LAPTOP=0
for k in ${KICKS[@]+"${KICKS[@]}"} ${HELD[@]+"${HELD[@]}"}; do
    case "$k" in laptop*) LAPTOP=1 ;; esac
done

say() { printf 'would run: %s\n' "$*"; }
if [ "$DRY" = 1 ]; then
    [ "$LAPTOP" = 0 ] || echo "would check: pepin-vslam and pepin-macnav mount $ROOT/src/pepin"
    for h in ${HELD[@]+"${HELD[@]}"}; do
        IFS="$TAB" read -r half c flag pattern name fix <<<"$h"
        echo "would check: $name in $c on the $half (runs -> refused, $fix)"
    done
    [ ${#ROS_FILES[@]} -eq 0 ] || say "(cd ros && rsync -a --relative ${ROS_FILES[*]} root@$BOARD:/root/pepin-ros/)"
    [ ${#LIB_FILES[@]} -eq 0 ] || say "(cd src && rsync -a --relative ${LIB_FILES[*]} root@$BOARD:/root/pepin-ros/pepin_src/)"
    for k in ${KICKS[@]+"${KICKS[@]}"}; do
        case "$k" in board\ *) say "ros/board.sh kick ${k#* }" ;; *) say "ros/laptop.sh kick ${k#* }" ;; esac
    done
    echo "dry run: nothing was touched"
    exit 0
fi

# ---- 2. the laptop reads this checkout ------------------------------------------------------
same_dir() { [ "$(cd "$1" 2>/dev/null && pwd -P)" = "$(cd "$2" && pwd -P)" ]; }
if [ "$LAPTOP" = 1 ]; then
    for c in pepin-vslam pepin-macnav; do
        SRC="$(docker inspect -f '{{range .Mounts}}{{if eq .Destination "/ws/pepin_src/pepin"}}{{.Source}}{{end}}{{end}}' "$c" 2>/dev/null)" || continue
        [ -n "$SRC" ] || continue
        if ! same_dir "${SRC#/host_mnt}" "$ROOT/src/pepin"; then
            echo "push REFUSED, nothing is touched: $c mounts $SRC, not this checkout's src/pepin"
            echo "  push from that checkout, or ros/restart.sh laptop from this one"
            exit 2
        fi
    done
fi

# ---- 3. what no kick restarts --------------------------------------------------------------
for h in ${HELD[@]+"${HELD[@]}"}; do
    IFS="$TAB" read -r half c flag pattern name fix <<<"$h"
    RC=0
    if [ "$half" = board ]; then
        ssh "root@$BOARD" "docker exec $c pgrep $flag $(printf '%q' "$pattern")" >/dev/null 2>&1 || RC=$?
    else
        docker exec "$c" pgrep "$flag" "$pattern" >/dev/null 2>&1 || RC=$?
    fi
    case "$RC" in
        0) echo "push REFUSED, nothing is touched: $name runs in $c on the $half and no kick restarts it; this needs $fix"; exit 2 ;;
        1) ;;  # not running there (or no such container)
        *) echo "push REFUSED, nothing is touched: could not ask $c on the $half whether $name runs (exit $RC)"; exit 2 ;;
    esac
done

# ---- 4. the files, and nothing else ---------------------------------------------------------
if [ ${#ROS_FILES[@]} -gt 0 ]; then
    (cd "$HERE" && rsync -a --relative "${ROS_FILES[@]}" "root@$BOARD:/root/pepin-ros/")
fi
if [ ${#LIB_FILES[@]} -gt 0 ]; then
    (cd "$ROOT/src" && rsync -a --relative "${LIB_FILES[@]}" "root@$BOARD:/root/pepin-ros/pepin_src/")
fi
echo "rsync: $((${#ROS_FILES[@]} + ${#LIB_FILES[@]})) file(s) on the board"

# ---- 5. the kicks, all at once --------------------------------------------------------------
[ ${#KICKS[@]} -gt 0 ] || { echo "push: nothing to kick, $((SECONDS - START)) s"; exit 0; }
OUTS="$(mktemp -d)"
trap 'rm -rf "$OUTS"' EXIT
i=0
for k in "${KICKS[@]}"; do
    i=$((i + 1)); script=laptop.sh
    case "$k" in board\ *) script=board.sh ;; esac
    (RC=0; "$HERE/$script" kick "${k#* }" >"$OUTS/$i" 2>&1 || RC=$?; echo "$RC" >"$OUTS/$i.rc") &
done
wait
i=0; BACK=0; SKIPPED=0; FAILED=0
for k in "${KICKS[@]}"; do
    i=$((i + 1)); RC="$(cat "$OUTS/$i.rc")"; OUT="$(cat "$OUTS/$i")"
    case "$RC" in
        0) BACK=$((BACK + 1)); printf '%-6s %s\n' "${k%% *}" "${OUT//$'\n'/$'\n'       }" ;;
        2 | 3) SKIPPED=$((SKIPPED + 1)); printf '%-6s %s skipped: %s\n' "${k%% *}" "${k#* }" "${OUT//$'\n'/; }" ;;
        *) FAILED=$((FAILED + 1)); printf '%-6s %s FAILED (exit %s): %s\n' "${k%% *}" "${k#* }" "$RC" "${OUT//$'\n'/; }" ;;
    esac
done
echo "push: $BACK node(s) back, $SKIPPED not running, $FAILED failed, $((SECONDS - START)) s in all"
[ "$FAILED" -eq 0 ]
