#!/bin/bash
# One command for a restart of this robot, and one place where everything that has bitten us
# after a restart is checked. Usage:
#
#   ros/restart.sh board|laptop|both [--deploy] [--fresh-graph] [--no-check]
#
#   board          `systemctl restart pepin-ros` over the multiplexed ssh, then wait for the
#                  tracker's first report line (up to 90 s)
#   --deploy       ros/sync.sh instead of the bare restart: code + the library + config to the
#                  board, the restart, and its census tail
#   laptop         ros/laptop.sh start, then ros/laptop.sh vslam --neck --seed-map=<the map the
#                  board serves>, read from the board's /etc/default/pepin-ros as ros/goto.sh
#                  reads it — a laptop half seeded with another map than the board serves is a
#                  fusion snapped to the wrong lattice
#   --fresh-graph  the camera half starts on an empty RTAB-Map database (laptop.sh vslam --fresh)
#                  AND the graph anchor of the served map is deleted: the anchor is a property of
#                  the map <-> database PAIR (pepin.anchors), so an empty database beside a kept
#                  anchor is a graph speaking in the previous database's frame
#   both           board first, then laptop; the checks run after both are up, so the two topics
#                  the laptop feeds the board are asked for when there is a laptop to feed them
#   --no-check     restart only
#
# The checks (one PASS/FAIL line each, numbered; WARN never fails the run) are listed under
# "Restarting" in ros/README.md. Exit status: 1 if any check failed, 0 otherwise. Everything
# waits with an explicit timeout, and a check that blows up never stops the ones after it —
# the point of the script is the whole picture, not the first bad line.
#
# What this script deliberately does NOT do: drive. It commands no velocity and sends no goal.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"  # multiplexed ssh: one handshake per 10 min, not per command

# How long a half is given to come back, and how often it is looked at. PEPIN_RESTART_WAIT_S
# shortens both for a board known to be slow today (or lengthens them); PEPIN_RESTART_POLL_S the
# cadence. Neither changes what is checked, only how long a missing report line is waited for.
WAIT_BOARD_S="${PEPIN_RESTART_WAIT_S:-90}"    # the board's stack to the tracker's first report line
WAIT_LAPTOP_S="${PEPIN_RESTART_WAIT_S:-120}"  # the camera half to the depth stream's first line
                                              # (a 30 s report timer, the ghost wait, the network)
POLL_S="${PEPIN_RESTART_POLL_S:-5}"
REPORT_WINDOW_S=90  # the nodes report every 30 s: three windows, so one missed line is not a verdict
ERROR_WINDOW_S=60   # how far back the error counts look

usage() {
    echo "usage: ros/restart.sh board|laptop|both [--deploy] [--fresh-graph] [--no-check]"
    exit 2
}
HALF=""; DEPLOY=false; FRESH_GRAPH=false; CHECK=true
case "${1:-}" in board | laptop | both) HALF="$1"; shift ;; *) usage ;; esac
for arg in "$@"; do
    case "$arg" in
        --deploy) DEPLOY=true ;;
        --fresh-graph) FRESH_GRAPH=true ;;
        --no-check) CHECK=false ;;
        *) usage ;;
    esac
done
if [ "$HALF" = board ] && [ "$FRESH_GRAPH" = true ]; then
    echo "--fresh-graph is the laptop's database and its anchor: ros/restart.sh laptop|both"
    exit 2
fi

FAILED=0; TOTAL=0
pass() { TOTAL=$((TOTAL + 1)); printf 'PASS %-4s %s\n' "$1" "$2"; }
fail() { TOTAL=$((TOTAL + 1)); FAILED=$((FAILED + 1)); printf 'FAIL %-4s %s\n' "$1" "$2"; }
warn() { printf 'WARN %-4s %s\n' "$1" "$2"; }   # seen, never fatal
step() { printf '\n== %s ==\n' "$1"; }

# ---- reading the two halves -------------------------------------------------------------------
board_last() {  # WINDOW_S PATTERN -> the last matching line of the board container's log.
    # Grepped on the board: a minute of that log is hundreds of kilobytes over the wifi, and a
    # node's report line is already there, so nothing is asked of ROS.
    ssh "root@$BOARD" "docker logs --since ${1}s pepin-ros 2>&1 | grep -aE $(printf '%q' "$2") | tail -1" \
        2>/dev/null || true
}
board_count() {  # WINDOW_S PATTERN -> how many lines of the board container's log match, or ?
    # A log that could not be read prints ?, never 0: "no errors" and "no answer" are not the
    # same sentence, and only one of them is good news.
    local answer
    answer="$(ssh "root@$BOARD" "docker logs --since ${1}s pepin-ros 2>&1 | grep -acE $(printf '%q' "$2")" 2>/dev/null || true)"
    [[ "$answer" =~ ^[0-9]+$ ]] && printf '%s\n' "$answer" || printf '?\n'
}
board_rate() {  # TOPIC -> one line: is it reaching the board, and how fast.
    # ros/tools/topic_rate.py, not `ros2 topic hz`: the CLI costs ~4.5 s of start-up on four A53
    # cores before it measures anything, the tool is one rclpy node and answers in one line.
    ssh "root@$BOARD" \
        "docker exec pepin-ros /pepin_entrypoint.sh timeout -s KILL 15 python3 /tools/topic_rate.py $1 5" \
        2>&1 || true
}
served_map() {  # the map file the board serves, as the container spells it (/maps/NAME.yaml)
    ssh "root@$BOARD" "grep -oE '^PEPIN_MAP=.*' /etc/default/pepin-ros" 2>/dev/null | cut -d= -f2 || true
}
map_id_now() {  # the identity of the map the board's tracker is matching on, as it spells it
    sed -n 's/.*map \/[^ ]* (id \(.*\), [0-9][0-9]* republications.*/\1/p' \
        <<<"$(board_last "$REPORT_WINDOW_S" "relocalizer\]: tracker:")"
}
over() { awk -v v="${1:-0}" -v t="$2" 'BEGIN { exit !(v + 0 > t) }'; }  # VALUE > THRESHOLD, floats

# ---- the restarts -----------------------------------------------------------------------------
wait_for() {  # WHAT TIMEOUT_S CONTAINER PATTERN: poll a container's log from now until it appears.
    # The window is the seconds since this function started, so nothing printed BEFORE the restart
    # can be read as the stack coming back.
    local what="$1" limit="$2" container="$3" pattern="$4" t0 elapsed window seen
    t0=$(date +%s)
    while :; do
        elapsed=$(($(date +%s) - t0))
        window=$((elapsed > 0 ? elapsed : 1))
        if [ "$container" = pepin-ros ]; then
            seen="$(board_last "$window" "$pattern")"
        else
            seen="$(docker logs --since "${window}s" "$container" 2>&1 | grep -aE "$pattern" | tail -1 || true)"
        fi
        [ -z "$seen" ] || { echo "$what: up, first report line after ${elapsed} s"; return 0; }
        [ "$elapsed" -lt "$limit" ] || {
            echo "$what: no report line within ${limit} s — the checks below say what is missing"
            return 1
        }
        sleep "$POLL_S"
    done
}

restart_board() {
    step "restarting the board"
    if [ "$DEPLOY" = true ]; then
        "$HERE/sync.sh"   # code + params + restart + the census tail; a red census is information
    else
        ssh "root@$BOARD" "systemctl restart pepin-ros && sleep 8 && systemctl is-active pepin-ros"
    fi
    wait_for "board" "$WAIT_BOARD_S" pepin-ros "relocalizer\]: tracker:" || true
}

drop_anchor() {  # --fresh-graph: the anchor of the map the board serves, for the new database
    local id path
    id="$(map_id_now)"
    if [ -z "$id" ]; then
        echo "--fresh-graph: the board's tracker did not say which map id it serves, so no anchor"
        echo "  was removed; delete ros/maps/<map id>.graph_anchor.json by hand if one is stale"
        return 0
    fi
    path="$(cd "$HERE/.." && uv run -q python -c \
        'import sys; from pepin.anchors import anchor_path; print(anchor_path(sys.argv[1], sys.argv[2]))' \
        "$HERE/maps" "$id")"
    if [ -f "$path" ]; then
        rm -f "$path"
        echo "--fresh-graph: removed $path"
    else
        echo "--fresh-graph: no anchor on file for map $id"
    fi
    echo "  (the anchor is map <-> database, not per session: an empty database beside a kept"
    echo "  anchor would put every word the graph says in the old database's frame)"
}

restart_laptop() {
    step "restarting the laptop"
    local map args
    map="$(served_map)"
    if [ -z "$map" ]; then
        echo "the board does not say which map it serves (/etc/default/pepin-ros PEPIN_MAP):"
        echo "  ros/mode.sh nav MAP on the board first — the camera half must be seeded with the"
        echo "  map the board serves, not with a valid one"
        return 1
    fi
    "$HERE/laptop.sh" start
    args=(vslam --neck "--seed-map=$map")
    if [ "$FRESH_GRAPH" = true ]; then
        args+=(--fresh)
        drop_anchor
    fi
    "$HERE/laptop.sh" "${args[@]}"
    wait_for "laptop" "$WAIT_LAPTOP_S" pepin-vslam "\]: depth: " || true
}

# ---- the checks -------------------------------------------------------------------------------
check_board() {
    step "checks: board"
    local out line value n

    if out="$("$HERE/board.sh" census 2>&1)"; then
        pass 1.1 "census: $(grep -a '^VERDICT' <<<"$out" | head -1)"
    else
        fail 1.1 "census: $(sed -n '/^VERDICT/,$p' <<<"$out" | tr '\n' ' ' | cut -c1-160)"
    fi

    line="$(board_last "$REPORT_WINDOW_S" "relocalizer\]: tracker:")"
    if [ -z "$line" ]; then
        fail 1.2 "tracker: no report line in ${REPORT_WINDOW_S} s (is the relocalizer up? ros/watch.sh)"
    else
        value="$(sed -n 's/.* sources=\([a-z,]*\).*/\1/p' <<<"$line")"
        n="$(sed -n 's/.* map_topic=\([a-z_]*\).*/\1/p' <<<"$line")"
        out="$(sed -n 's/.*, fit \([0-9.]*\)[^0-9].*/\1/p' <<<"$line")"
        if [ -n "$value" ] && [ -n "$n" ]; then
            pass 1.2 "tracker: sources=$value, map_topic=$n, fit ${out:-?}, map $(map_id_now)"
        else
            fail 1.2 "tracker: its line carries no sources= or map_topic= (ros/flags.sh list relocalizer)"
        fi
    fi

    if out="$("$HERE/goto.sh" where 2>&1)"; then
        pass 1.3 "pose: $(tr '\n' ' ' <<<"$out" | cut -c1-140)"
    else
        fail 1.3 "pose: /where_am_i did not answer: $(tail -1 <<<"$out")"
    fi

    n="$(board_count "$ERROR_WINDOW_S" 'Failed to meet update rate')"
    if [ "$n" = "?" ]; then
        fail 1.4 "loop rate: the board's log could not be read (ssh root@$BOARD docker logs pepin-ros)"
    elif [ "$n" = 0 ]; then
        pass 1.4 "loop rate: no 'Failed to meet update rate' in the last ${ERROR_WINDOW_S} s"
    else
        fail 1.4 "loop rate: $n x 'Failed to meet update rate' in the last ${ERROR_WINDOW_S} s (a starved loop: ros/board.sh census)"
    fi

    n="$(board_count "$ERROR_WINDOW_S" 'Extrapolation|out of map bounds|Off Grid')"
    if [ "$n" = "?" ]; then
        fail 1.5 "tf and costmaps: the board's log could not be read (ssh root@$BOARD docker logs pepin-ros)"
    elif [ "$n" = 0 ]; then
        pass 1.5 "tf and costmaps: no extrapolation / off-grid error in the last ${ERROR_WINDOW_S} s"
    else
        fail 1.5 "tf and costmaps: $n error(s) in the last ${ERROR_WINDOW_S} s: $(board_last "$ERROR_WINDOW_S" 'Extrapolation|out of map bounds|Off Grid' | cut -c1-140)"
    fi

    n=6
    for value in /depth_scan /vo; do
        out="$(board_rate "$value")"
        if [[ "$out" == *" Hz over "* ]]; then
            pass "1.$n" "$value reaches the board: ${out#*: }"
        else
            fail "1.$n" "$value does not reach the board: $(tail -1 <<<"$out" | cut -c1-140) (the laptop half and the bridge)"
        fi
        n=$((n + 1))
    done

    out="$(ssh "root@$BOARD" "systemctl is-active pepin-base; journalctl -u pepin-base -n 200 --no-pager 2>/dev/null | grep -aE 'torque (on|off)' | tail -1" 2>/dev/null || true)"
    value="$(head -1 <<<"$out")"; line="$(sed -n '2p' <<<"$out")"
    if [ "$value" != active ]; then
        fail 1.8 "base: pepin-base is ${value:-unreachable} (the wheels' own server; systemctl status pepin-base)"
    elif [[ "$line" == *"torque on"* ]]; then
        fail 1.8 "base: the wheels are still armed — its last word is '${line#*: }' (a torque left standing holds the wheels and heats the servos)"
    else
        pass 1.8 "base: active, no torque left standing${line:+ (last: ${line##*: })}"
    fi
}

check_laptop() {
    step "checks: laptop"
    local started LOG line value n
    last() { grep -aE "$1" <<<"$LOG" | tail -1 || true; }

    if ! started="$(docker inspect -f '{{.State.StartedAt}}' pepin-vslam 2>/dev/null)"; then
        fail 2.1 "pepin-vslam is not running (ros/laptop.sh vslam --neck --seed-map=...); no laptop check could run"
        return 0
    fi
    LOG="$(docker logs --since "$started" pepin-vslam 2>&1 || true)"   # one fetch, every grep below

    line="$(last 'bridge watch: [0-9]+ topics')"
    n="$(sed -n 's/.*bridge watch: \([0-9]*\) topics.*/\1/p' <<<"$line")"
    if [ -z "$line" ]; then
        fail 2.1 "bridge watch: no line yet (it reports once the routes settle; ros/laptop.sh logs vslam)"
    elif [[ "$line" == *"DEAD ROUTES"* ]]; then
        fail 2.1 "bridge watch: ${line#*bridge watch: }"
    elif [ "${n:-0}" -lt 10 ]; then
        fail 2.1 "bridge watch: only $n topics carried, 10 expected: ${line#*bridge watch: }"
    else
        pass 2.1 "bridge watch: $n topics, dead routes 0"
    fi

    line="$(last '\]: depth: ')"
    value="$(sed -n 's/.*: depth: \([0-9.]*\) frames\/s.*/\1/p' <<<"$line")"
    n="$(grep -o 'a [0-9.-]* b [+-][0-9.]* on [0-9]* pairs' <<<"$line" | head -1 || true)"
    if [ -z "$line" ]; then
        fail 2.2 "depth stream: no report line (is the node up? ros/laptop.sh logs vslam)"
    elif ! over "$value" 5; then
        fail 2.2 "depth stream: ${value:-no} frames/s (over 5 expected): ${line#*: depth: }"
    elif [ -z "$n" ]; then
        fail 2.2 "depth stream: $value frames/s but no fitted law in its line — the raw network is 1.5-2x too far and its frames are withheld"
    else
        pass 2.2 "depth stream: $value frames/s, law $n"
    fi

    line="$(last '\]: fusion: ')"
    value="$(sed -n 's/.*: fusion: [0-9]* frames (\([0-9.]*\)\/s.*/\1/p' <<<"$line")"
    n="$(sed -n 's/.*at bound \([0-9]*\)[^0-9].*/\1/p' <<<"$line")"
    if [ -z "$line" ]; then
        fail 2.3 "depth fusion: no report line (ros/laptop.sh logs vslam)"
    elif ! over "$value" 5; then
        fail 2.3 "depth fusion: ${value:-no} frames/s (over 5 expected)"
    elif [ "${n:-0}" -ne 0 ] 2>/dev/null; then
        fail 2.3 "depth fusion: $value frames/s but $n frames refused at bound (the volume's edge: the seed map and the lattice)"
    else
        pass 2.3 "depth fusion: $value frames/s, at bound 0"
    fi

    line="$(last '\]: vo: ')"
    value="$(sed -n 's/.*: vo: \([0-9.]*\) poses\/s.*/\1/p' <<<"$line")"
    if [ -z "$line" ]; then
        fail 2.4 "visual odometry: no report line (--no-vo, or rgbd_odometry is down)"
    elif ! over "$value" 5; then
        fail 2.4 "visual odometry: ${value:-no} poses/s from rtabmap (over 5 expected)"
    else
        pass 2.4 "visual odometry: $value poses/s"
    fi

    line="$(last '\]: laptop localizer: ')"
    value="$(sed -n 's/.*tracker fit \([0-9.]*\).*/\1/p' <<<"$line")"
    if [ -z "$line" ]; then
        fail 2.5 "laptop localizer: no report line (ros/laptop.sh logs vslam)"
    elif ! over "$value" 0; then
        fail 2.5 "laptop localizer: it hears no belief from the board (tracker fit ${value:-none}) — /tracker_pose is not crossing the bridge"
    else
        pass 2.5 "laptop localizer: the board's belief heard, tracker fit $value"
    fi

    value="$( { docker exec pepin-vslam ps -eo comm= 2>/dev/null | sort -u | tr '\n' ' '; } || true)"
    if [[ " $value " == *" rtabmap "* ]]; then
        pass 2.6 "rtabmap: alive in pepin-vslam"
    else
        fail 2.6 "rtabmap: no rtabmap process in pepin-vslam (it dies on a database it cannot open): ${value:-nothing answered}"
    fi

    line="$(last '\]: rtabmap frame: ')"
    n="$(sed -n 's/.*over \([0-9]*\) infos.*/\1/p' <<<"$line")"
    value="$(grep -oE 'from (file|learned)' <<<"$line" | head -1 | cut -d' ' -f2 || true)"
    if [ -z "$line" ]; then
        fail 2.7 "rtabmap frame: no report line (ros/laptop.sh logs vslam)"
    elif [ -z "$value" ]; then
        fail 2.7 "rtabmap frame: no anchor yet — the graph has said nothing the tracker could be tied to"
    elif [ "${n:-0}" -eq 0 ] 2>/dev/null; then
        fail 2.7 "rtabmap frame: anchor from $value but over 0 infos — the graph's trust is deaf (rtabmap's /info is not arriving)"
    else
        pass 2.7 "rtabmap frame: anchor from $value, graph trust over $n infos"
    fi

    n="$(grep -ac 'process has died' <<<"$LOG" || true)"
    if [ "${n:-0}" -eq 0 ] 2>/dev/null; then
        pass 2.8 "no node died since the container started"
    else
        fail 2.8 "$n node(s) died since the container started: $(grep -a 'process has died' <<<"$LOG" | tail -1 | cut -c1-140)"
    fi
}

check_flags() {
    step "checks: flags"
    local side n out line
    n=1
    for side in $SIDES; do
        out="$("$HERE/flags.sh" drift "$side" 2>&1 || true)"
        if [ -z "$out" ]; then
            pass "3.$n" "flags ($side): every flag is its table default"
        else
            warn "3.$n" "flags ($side): $(grep -ac . <<<"$out") not at their default — deliberate is fine, a restart never sets one:"
            while IFS= read -r line; do [ -z "$line" ] || printf '          %s\n' "$line"; done <<<"$out"
        fi
        n=$((n + 1))
    done
}

# ---- the run ----------------------------------------------------------------------------------
SIDES=""
case "$HALF" in
    board) SIDES=board ;;
    laptop) SIDES=laptop ;;
    both) SIDES="board laptop" ;;
esac
# Both halves come back before anything is checked: the two topics the board is asked about
# (/depth_scan, /vo) are fed BY the laptop, so checking the board first would fail them on purpose.
[ "$HALF" = laptop ] || restart_board || fail 0.1 "the board's restart did not finish (above); the checks say what is missing"
[ "$HALF" = board ] || restart_laptop || fail 0.2 "the laptop's restart did not finish (above); the checks say what is missing"
if [ "$CHECK" != true ]; then
    printf '\nchecks skipped (--no-check)\n'
    exit 0
fi
[ "$HALF" = laptop ] || check_board
[ "$HALF" = board ] || check_laptop
check_flags   # last: one `ros2 param dump` per node, the slowest thing here
step "verdict"
if [ "$FAILED" -eq 0 ]; then
    echo "green: $TOTAL checks, none failed"
else
    echo "red: $FAILED of $TOTAL checks failed"
    exit 1
fi
