#!/bin/bash
# One command that brings this robot up WORKING, repairs what it can and says what it could not:
# the restart, the proof that the planner plans, and every check that has bitten us. Usage:
#
#   ros/restart.sh board|laptop|both [--deploy] [--fresh-graph] [--no-check] [--fast] [--dry-run]
#
#   board          the board's stack WITH its zenoh router (ros/lib.sh pepin_board_restart: stack
#                  stopped, router restarted, stack started), then its first report line (90 s).
#                  If the laptop half is up, its vslam is restarted after it: RTAB-Map follows the
#                  board's new odometry only from a fresh start (journal 2026-09-22)
#   --deploy       ros/sync.sh --restart instead: code + the library + config to the board, the
#                  same router-and-stack restart, and its census tail
#   laptop         ros/laptop.sh start, then ros/laptop.sh vslam --neck. Nothing about the map is
#                  passed: the database IS the map, the launch reads whether it exists
#   --fresh-graph  the camera half starts on an empty RTAB-Map database (laptop.sh vslam --fresh)
#                  AND the volume of the old frame is moved aside: the volume is painted in the
#                  graph's frame, so a kept snapshot beside an empty database is a room painted
#                  somewhere else
#   both           board first, then laptop. Never the other way: a board restart re-zeroes the
#                  odometry under a running RTAB-Map. The board's Nav2 waits for the laptop's map
#                  (initial_transform_timeout, ros/params/nav2_params.yaml) and the proof below
#                  catches whatever still races
#   --no-check     restart only: no proof, no repair, no checks
#   --fast         --no-check, and no waiting for either half's first report line: the commands
#                  are sent and the script returns; the stack still needs its own time to come
#                  up (Nav2 on the board about a minute), and nothing here says whether it did
#   --dry-run      print the order of everything above and touch nothing
#
# THE PROOF (4.1), whenever the laptop half is up after the restart: from pepin-vslam, the pose
# corrected (ros/tools/map_odom.py) and the planner planning (ros/tools/planner_check.py: a new
# global costmap within 10 s, one ComputePathToPose 0.5 m ahead — a plan, never motion), asked
# again for up to PEPIN_PLANNER_WAIT_S (180) while Nav2 comes up. Broken, it REPAIRS itself: first
# the Nav2 container alone (SIGINT; the launch respawns it in 2 s with fresh nodes, odometry and
# RTAB-Map untouched), then the board half once more with its router and the laptop's vslam after
# it, and fails loudly if the planner still does not plan. A planner that answers "no path" in
# every direction is not repaired: no restart moves furniture. With the laptop half down there is
# no map to plan on, and 4.1 fails saying so.
#
# After a laptop restart the desktop Foxglove is told to reconnect (ros/foxglove.sh reopen): its
# old websocket died with the container and its panels stay empty until a client re-attaches.
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
# How long the proof keeps asking the planner: the board's Nav2 bring-up is ~54 s on the A53 once
# the map is there (journal 2026-09-25), RTAB-Map's first correction comes on top.
PLANNER_WAIT_S="${PEPIN_PLANNER_WAIT_S:-${PEPIN_RESTART_WAIT_S:-180}}"
REPORT_WINDOW_S=90  # the nodes report every 30 s: three windows, so one missed line is not a verdict
ERROR_WINDOW_S=60   # how far back the error counts look

usage() {
    echo "usage: ros/restart.sh board|laptop|both [--deploy] [--fresh-graph] [--no-check] [--fast] [--dry-run]"
    exit 2
}
HALF=""; DEPLOY=false; FRESH_GRAPH=false; CHECK=true; DRY=false; WAIT=true
case "${1:-}" in board | laptop | both) HALF="$1"; shift ;; *) usage ;; esac
for arg in "$@"; do
    case "$arg" in
        --deploy) DEPLOY=true ;;
        --fresh-graph) FRESH_GRAPH=true ;;
        --no-check) CHECK=false ;;
        --fast) CHECK=false; WAIT=false ;;
        --dry-run) DRY=true ;;
        *) usage ;;
    esac
done
if [ "$HALF" = board ] && [ "$FRESH_GRAPH" = true ]; then
    echo "--fresh-graph is the laptop's database and its volume: ros/restart.sh laptop|both"
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
board_hot_thread() {  # -> "TID SHARE COMM" for the busiest thread of the board's Nav2 component
    # container, or nothing at all when the board could not be asked. SHARE is that thread's
    # cumulative CPU time divided by the process's own lifetime, so 1.00 means "has done nothing
    # but run since it started" — the signature of a costmap loop spinning under its mutex, which
    # writes no log line of any kind. One `ps` over the multiplexed ssh and nothing else: the ROS
    # CLI costs seconds of A53 on this board, and `ps -L` reads /proc, which the host sees for
    # every process in every container.
    ssh "root@$BOARD" '
        pid=$(pgrep -f component_container_isolated | head -1)
        [ -n "$pid" ] || exit 0
        ps -L -p "$pid" -o tid=,times=,etimes=,comm= 2>/dev/null |
            awk "\$3 > 0 { printf \"%s %.2f %s\n\", \$1, \$2 / \$3, \$4 }" |
            sort -k2 -rn | head -1' 2>/dev/null || true
}
board_rate() {  # TOPIC [latched] -> one line: is it reaching the board, and how fast.
    # ros/tools/topic_rate.py, not `ros2 topic hz`: the CLI costs ~4.5 s of start-up on four A53
    # cores before it measures anything, the tool is one rclpy node and answers in one line.
    # `latched` asks for the held copy of a topic published once per change instead of a rate.
    ssh "root@$BOARD" \
        "docker exec pepin-ros /pepin_entrypoint.sh timeout -s KILL 15 python3 /tools/topic_rate.py $1 5 ${2:-}" \
        2>&1 || true
}
# Both read the tracker's own "map ... (id <size>@<origin>, N adopted, ..." from its report line,
# anchored on "(id " rather than on the topic: a cold-boot line carries the cache's own phrase in
# between, and that phrase names a topic too.
map_id_now() {  # the identity of the map the board's tracker is matching on, as it spells it
    sed -n 's/.*(id \(.*\), [0-9][0-9]* adopted.*/\1/p' \
        <<<"$(board_last "$REPORT_WINDOW_S" "relocalizer\]: tracker:")"
}
adoptions_now() {  # how many maps that tracker has taken since it started (0 = it has none)
    sed -n 's/.*(id .*, \([0-9][0-9]*\) adopted.*/\1/p' \
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
    step "restarting the board (its zenoh router with it)"
    if [ "$DEPLOY" = true ]; then
        "$HERE/sync.sh" --restart  # code + params + router and stack + the census tail; a red census is information
    else
        pepin_board_restart
    fi
    # Which line says this half is back. Under PEPIN_LOCALIZER=tracker it is the tracker's first
    # report; under rtabmap no tracker is launched at all, so the recorder — the one node of ours
    # that runs on the board in every arrangement — is what is waited for instead.
    if [ "$WAIT" != true ]; then
        echo "--fast: not waiting for the board's first report line"
    elif pepin_localizer_is_tracker; then
        wait_for "board" "$WAIT_BOARD_S" pepin-ros "relocalizer\]: tracker:" || true
    else
        wait_for "board" "$WAIT_BOARD_S" pepin-ros "run recorder ready" || true
    fi
    # A route's DDS endpoint is built when the route is created and only if the far bridge is
    # already announcing, so OF TWO BRIDGES THE ONE THAT STARTS LAST gets working routes. The
    # board's restart takes its bridge with it, which leaves the laptop's publications (/vo,
    # /depth_scan, the localisation words) with no reader on the board — measured after every
    # `restart.sh board --deploy` on 2026-09-16. Restarting the laptop's bridge here makes it
    # the newer one again. Nothing to do when this half is not up.
    # Under zenoh there is no second bridge to be the newer one: both halves are peers of their
    # own router, and a node that lost its router reconnects to it by itself.
    if pepin_rmw_is_zenoh || [ "$SIDES" != board ] || ! docker ps --format '{{.Names}}' | grep -qx pepin-zenoh; then
        return 0
    fi
    step "the laptop's bridge, after the board's: the newer bridge is the one with live routes"
    docker restart pepin-zenoh >/dev/null && echo "laptop bridge restarted"
}

drop_volume() {  # --fresh-graph: the volume shares the database's frame, so it goes with it
    # One frame (World R): the graph's map frame IS `map`, and the fused volume is painted in it.
    # An empty database starts a new frame, and a volume kept from the old one would be a room
    # painted somewhere else — so the snapshot beside the database is moved aside, never deleted.
    local world="$HERE/maps/rtabmap.world.npz"
    if [ -f "$world" ]; then
        mv "$world" "$world.before-fresh-$(date +%Y%m%d_%H%M%S)"
        echo "--fresh-graph: the volume of the old frame moved aside ($world.before-fresh-*)"
    fi
}

restart_laptop() {  # [vslam]: the camera half only (after a board restart), else the whole half
    local what="restarting the laptop"
    [ "${1:-}" != vslam ] || what="restarting the laptop's vslam (RTAB-Map onto the new odometry)"
    step "$what"
    # Nothing about the map is passed: one database, one grid, and the launch reads for itself
    # whether that database exists (an empty room is the file being absent).
    local args
    args=(vslam --neck)
    if [ "$FRESH_GRAPH" = true ]; then
        args+=(--fresh)
        drop_volume
    fi
    [ "${1:-}" = vslam ] || "$HERE/laptop.sh" start
    "$HERE/laptop.sh" "${args[@]}"
    if [ "$WAIT" != true ]; then
        echo "--fast: not waiting for the laptop's first depth line"
        return 0
    fi
    wait_for "laptop" "$WAIT_LAPTOP_S" pepin-vslam "\]: depth: " || true
}

laptop_up() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx pepin-vslam; }

# ---- the proof: the planner plans -------------------------------------------------------------
renav() {  # every Nav2 component container, SIGINT; its launch respawns it in 2 s, fresh nodes
    # The container is launch_kit's respawned_container on either half (nav2_container,
    # nav2_container_board, nav2_container_laptop); odometry, the sensors and RTAB-Map are other
    # processes and never notice. A board whose bring-up gave up before the map came (the
    # start-order race) comes back active in seconds — the sim: 5.7 s on the Mac.
    if ssh "root@$BOARD" "docker exec pepin-ros pkill -INT -f '__node:=nav2_container'" 2>/dev/null; then
        echo "  board: Nav2 container respawning"
    else
        echo "  board: no Nav2 container to respawn (PEPIN_NAV=false, or the board is unreachable)"
    fi
    if docker ps --format '{{.Names}}' 2>/dev/null | grep -qx pepin-laptop &&
        docker exec pepin-laptop pkill -INT -f '__node:=nav2_container' 2>/dev/null; then
        echo "  laptop: Nav2 container respawning"
    fi
}

PROOF=""
prove_planner() {  # -> 0 plans, 1 broken, 3 boxed; PROOF is the planner check's own line
    local t0 elapsed status localised out
    t0=$(date +%s)
    while :; do
        localised=0
        docker exec pepin-vslam /pepin_entrypoint.sh timeout -s KILL 20 python3 /tools/map_odom.py 5 \
            >/dev/null 2>&1 || localised=$?
        status=0
        out="$(docker exec pepin-vslam /pepin_entrypoint.sh timeout -s KILL 60 python3 /tools/planner_check.py 2>&1)" || status=$?
        PROOF="$(grep -a '^planner: ' <<<"$out" | tail -1 || true)"
        [ -n "$PROOF" ] || { PROOF="planner: BROKEN — the check printed no verdict: $(tail -1 <<<"$out" | cut -c1-140)"; status=1; }
        [ "$status" -ne 0 ] || [[ "$PROOF" == "planner: OK"* ]] || status=1
        elapsed=$(($(date +%s) - t0))
        [ "$status" -ne 0 ] || return 0
        # "No path around" is believed only of a corrected pose: before RTAB-Map recognises the
        # room the pose is the odometry's, and the wall in the way may be somewhere else.
        if [ "$status" -eq 3 ] && [ "$localised" -eq 0 ]; then return 3; fi
        if [ "$elapsed" -ge "$PLANNER_WAIT_S" ]; then
            [ "$status" -eq 3 ] && return 3
            return 1
        fi
        echo "  ${PROOF#planner: } — asking again (${elapsed} of ${PLANNER_WAIT_S} s)"
        sleep "$POLL_S"
    done
}

ensure_planner() {  # 4.1: prove it, repair it (Nav2 alone, then the board once more), or fail
    step "the planner"
    if ! laptop_up; then
        fail 4.1 "planner: not proven — the laptop half is down, so there is no map and no map -> odom to plan on (the board's Nav2 waits for them); ros/restart.sh laptop brings them and proves the planner"
        return 0
    fi
    local status=0 repaired=""
    prove_planner || status=$?
    if [ "$status" -eq 1 ]; then
        echo "  $PROOF"
        echo "repair 1: the Nav2 container alone (odometry and RTAB-Map untouched)"
        renav
        repaired="a Nav2 respawn"
        status=0; prove_planner || status=$?
    fi
    if [ "$status" -eq 1 ]; then
        echo "  $PROOF"
        echo "repair 2: the board half once more (router and stack), then the laptop's vslam on its new odometry"
        DEPLOY=false; FRESH_GRAPH=false  # the code went out, and the new database stays
        restart_board || true
        restart_laptop vslam || true
        repaired="a Nav2 respawn and a board restart"
        status=0; prove_planner || status=$?
    fi
    case "$status" in
        0) pass 4.1 "${PROOF#planner: }${repaired:+ — after $repaired}" ;;
        3) fail 4.1 "${PROOF#planner: } — the planner itself works, so no restart can help: the cart is boxed in, or the global costmap is full of marks (Foxglove's planner panel)" ;;
        *) fail 4.1 "${PROOF#planner: } — STILL BROKEN after $repaired: nothing will plan; 1.11 and 1.12 below say where" ;;
    esac
}

# ---- --dry-run: the order, and nothing touched ------------------------------------------------
describe() {
    local n=0 fresh=""
    [ "$FRESH_GRAPH" = false ] || fresh=" --fresh (the old volume moved aside)"
    say() { n=$((n + 1)); printf '%2d. %s\n' "$n" "$1"; }
    echo "dry run, nothing is touched — ros/restart.sh $HALF would:"
    if [ "$HALF" != laptop ]; then
        if [ "$DEPLOY" = true ]; then
            say "ros/sync.sh --restart: the checkout to root@$BOARD, then its zenoh router and its stack (stop pepin-ros, restart pepin-zrouter, start pepin-ros), then the census"
        else
            say "board: stop pepin-ros, restart pepin-zrouter, start pepin-ros (ssh root@$BOARD)"
        fi
        [ "$WAIT" != true ] || say "wait up to ${WAIT_BOARD_S} s for the board's first report line"
    fi
    local waits=""
    [ "$WAIT" != true ] || waits="; wait up to ${WAIT_LAPTOP_S} s for the first depth line"
    case "$HALF" in
        board) say "ask docker ps whether the laptop half is up (pepin-vslam): if it is, ros/laptop.sh vslam --neck — RTAB-Map onto the board's new odometry — and wait up to ${WAIT_LAPTOP_S} s for its first depth line; if it is down, say so" ;;
        *) say "laptop: ros/laptop.sh start, then ros/laptop.sh vslam --neck$fresh$waits" ;;
    esac
    if [ "$CHECK" != true ]; then
        if [ "$WAIT" = true ]; then
            say "stop there: no proof, no repair, no checks (--no-check)"
        else
            say "stop there: nothing waited for, no proof, no repair, no checks (--fast)"
        fi
        return 0
    fi
    say "the proof (4.1), if the laptop half is up (else 4.1 fails: no map to plan on): from pepin-vslam, ros/tools/map_odom.py and ros/tools/planner_check.py — a new global costmap within 10 s, one path 0.5 m ahead — asked again for up to ${PLANNER_WAIT_S} s"
    say "broken -> repair 1: SIGINT to the Nav2 container (board, and pepin-laptop if it runs); its launch respawns it; the proof again"
    say "still broken -> repair 2: the board once more (stop pepin-ros, restart pepin-zrouter, start pepin-ros; no sync), ros/laptop.sh vslam --neck, the proof again; still broken -> FAIL 4.1, loudly. \"No path around\" is never repaired"
    case "$HALF" in
        board) say "the checks: board (1.x), flags (3.x)" ;;
        laptop) say "the checks: laptop (2.x), flags (3.x)" ;;
        both) say "the checks: board (1.x), laptop (2.x), flags (3.x)" ;;
    esac
    [ "$HALF" = board ] || say "tell Foxglove to reconnect (ros/foxglove.sh reopen)"
    say "the verdict: green only if no check failed"
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
    if ! pepin_localizer_is_tracker; then
        # One localiser (PEPIN_LOCALIZER=rtabmap): no tracker is launched here, so there is no
        # report line, no sources and no adoption to judge. What replaces this check is 1.13
        # below — whether the transform the laptop owns is arriving and has corrected anything.
        warn 1.2 "tracker: none on this board (PEPIN_LOCALIZER=$PEPIN_LOCALIZER); the pose is RTAB-Map's, see 1.13"
    elif [ -z "$line" ]; then
        fail 1.2 "tracker: no report line in ${REPORT_WINDOW_S} s (is the relocalizer up? ros/watch.sh)"
    else
        value="$(sed -n 's/.* sources=\([a-z,]*\).*/\1/p' <<<"$line")"
        n="$(adoptions_now)"
        out="$(sed -n 's/.*, fit \([0-9.]*\)[^0-9].*/\1/p' <<<"$line")"
        if [ -z "$value" ]; then
            fail 1.2 "tracker: its line carries no sources= (ros/flags.sh list relocalizer)"
        elif [ "${n:-0}" -eq 0 ] 2>/dev/null; then
            # One map (World R): the grid is RTAB-Map's, over the bridge. A tracker that has adopted
            # none has no map at all — the laptop half is down and the cache was refused or absent,
            # or /map is not crossing.
            fail 1.2 "tracker: sources=$value but 0 maps adopted — it has no map (is the laptop half up? does /map cross? ros/laptop.sh logs vslam)"
        else
            pass 1.2 "tracker: sources=$value, $n map(s) adopted, fit ${out:-?}, map $(map_id_now)"
        fi
    fi

    # WHERE THE CART THINKS IT IS. Two different questions by localiser, and the script asks the
    # one that exists: the tracker's own /where_am_i service where a tracker runs, and the goal
    # server's socket where none does — that answer is composed from TF (map -> base_link), which
    # is the only way this pose can be read at all under rtabmap.
    if pepin_localizer_is_tracker; then
        if out="$("$HERE/goto.sh" where 2>&1)"; then
            pass 1.3 "pose: $(tr '\n' ' ' <<<"$out" | cut -c1-140)"
        else
            fail 1.3 "pose: /where_am_i did not answer: $(tail -1 <<<"$out")"
        fi
    else
        out="$("$HERE/go.sh" where 2>&1 || true)"
        if [[ "$out" != *'"event": "where"'* ]]; then
            fail 1.3 "pose: the goal server did not answer 'where' on its socket: $(tail -1 <<<"$out" | cut -c1-140)"
        elif [[ "$out" == *'"pose": "tf"'* ]]; then
            pass 1.3 "pose: from TF, as this stack has it — $(tr '\n' ' ' <<<"$out" | cut -c1-160)"
        else
            fail 1.3 "pose: the goal server answers '$(sed -n 's/.*"pose": "\([a-z]*\)".*/\1/p' <<<"$out" | tail -1)', not 'tf' — nothing publishes map -> base_link (is ros/laptop.sh vslam up?)"
        fi
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
    # /map first: it is THE map (World R), RTAB-Map's grid, LATCHED — republished when it changes,
    # and while RTAB-Map localises that is once a start (2026-09-24: "1 grids relayed" in a whole
    # start, while the board's static layer held it), so it is asked for its held copy, not a
    # rate. Then the camera's other two words, which do have one.
    for value in /map /depth_scan /vo; do
        mode=""
        [ "$value" = /map ] && mode=latched
        out="$(board_rate "$value" $mode)"
        if [[ "$out" == *" Hz over "* || "$out" == *"latched copy received"* ]]; then
            pass "1.$n" "$value reaches the board: ${out#*: }"
        else
            fail "1.$n" "$value does not reach the board: $(tail -1 <<<"$out" | cut -c1-140) (the laptop half and the bridge)"
        fi
        n=$((n + 1))
    done

    # The costmaps' one source: the grid the tracker ADOPTED, republished latched by it
    # (pepin_bringup.relocalizer.TRACKED_MAP_TOPIC) and read by both static layers
    # (ros/params/nav2_params.yaml). It is published once per adoption, so a RATE says nothing here
    # and the question is whether it is advertised at all: a tracker that has republished no map
    # leaves both costmaps without a static map and the planner with nothing to plan on.
    out="$(board_rate /map_tracked)"
    if ! pepin_localizer_is_tracker; then
        # /map_tracked is the TRACKER's republication of the grid it adopted. With no tracker
        # both costmaps read /map directly instead (ros/params/nav2_map_from_laptop.yaml, loaded
        # by nav.launch.py under this switch), and /map is checked by the rate kit above.
        warn "1.$n" "/map_tracked: n/a under PEPIN_LOCALIZER=$PEPIN_LOCALIZER (no tracker republishes a map; both static layers read /map, checked above)"
    elif [[ "$out" == *"not advertised"* ]]; then
        fail "1.$n" "/map_tracked is not advertised on the board: the tracker republished no map, so neither costmap's static layer has one"
    else
        pass "1.$n" "/map_tracked advertised by the tracker: both static layers have their source (${out#*: })"
    fi

    out="$(ssh "root@$BOARD" "systemctl is-active pepin-base; journalctl -u pepin-base -n 200 --no-pager 2>/dev/null | grep -aE 'torque (on|off)' | tail -1" 2>/dev/null || true)"
    value="$(head -1 <<<"$out")"; line="$(sed -n '2p' <<<"$out")"
    if [ "$value" != active ]; then
        fail 1.10 "base: pepin-base is ${value:-unreachable} (the wheels' own server; systemctl status pepin-base)"
    elif [[ "$line" == *"torque on"* ]]; then
        fail 1.10 "base: the wheels are still armed — its last word is '${line#*: }' (a torque left standing holds the wheels and heats the servos)"
    else
        pass 1.10 "base: active, no torque left standing${line:+ (last: ${line##*: })}"
    fi

    # Nav2 is not "up" until it is ACTIVE. planner_server's activation blocks inside its global
    # costmap's first update, and a costmap whose range layers cannot transform a Range never
    # finishes one: tf2's canTransform costs its whole transform_tolerance per message, the ToF
    # layers receive 15 Hz each, and the backlog outgrows the drain until the update never
    # returns (4 of 7 board starts on 2026-09-21; scratch/nav2_hang/wedge_gain.py, and the
    # tof_bridge module docstring for the fix: the whiskers feed ObstacleLayers, which drop
    # what they cannot place instead of blocking on it). Two questions, one line: did the lifecycle
    # manager get planner_server's bond, and is the log free of the complaint that says the
    # wedge is running. The bond is printed once at activation, so it is looked for across the
    # whole restart, not the report window; the container is recreated on every restart
    # (board/pepin-ros.service), so no older start can match. A board that runs no Nav2
    # (PEPIN_NAV=false) has nothing to judge here and is not failed for it.
    # Since 2026-09-21 the whiskers feed ObstacleLayers instead and no RangeSensorLayer is
    # listed in either costmap, so the count below is zero by construction — which is exactly
    # why it is KEPT: one of these lines means a range layer is running again (an old
    # nav2_params.yaml on the board, a plugins list edited back by hand), and that is worth a
    # failed check on its own.
    value="$(board_last "$((WAIT_BOARD_S + REPORT_WINDOW_S))" 'Activating planner_server')"
    line="$(board_last "$((WAIT_BOARD_S + REPORT_WINDOW_S))" 'planner_server connected with bond')"
    n="$(board_count "$REPORT_WINDOW_S" "Range sensor layer can't transform")"
    if [ -z "$value" ]; then
        warn 1.11 "nav2: no 'Activating planner_server' in the board's log — Nav2 does not run on this half (PEPIN_NAV), nothing to judge"
    elif [ -z "$line" ]; then
        fail 1.11 "nav2: planner_server was activated and never bonded — it is WEDGED in a costmap's first update ($n x 'Range sensor layer can't transform' in the last ${REPORT_WINDOW_S} s). Goals are accepted and nothing is planned; restart the board half (ros/restart.sh board) and, if it comes back, ros/tools/coldstart_soak.sh"
    elif [ "$n" = "?" ]; then
        fail 1.11 "nav2: planner_server bonded, but the board's log could not be read for the range-layer complaint (ssh root@$BOARD docker logs pepin-ros)"
    elif [ "$n" != 0 ]; then
        fail 1.11 "nav2: $n x 'Range sensor layer can't transform' in the last ${REPORT_WINDOW_S} s — a RangeSensorLayer is running on this board although no costmap lists one any more: the board's ros/params/nav2_params.yaml is older than this checkout (ros/restart.sh board --deploy), and that plugin is the Nav2 wedge of 2026-09-21"
    else
        pass 1.11 "nav2: planner_server connected with bond, no RangeSensorLayer running (0 range-layer transform failures in the last ${REPORT_WINDOW_S} s)"
    fi

    # ...and the failure that leaves no line in any log: a costmap thread spinning on its own.
    # range_sensor_layer.cpp:362-369 clamps bx0/by0 at zero, clamps bx1/by1 at the grid's size
    # and then walks `for (unsigned int x = bx0; x <= (unsigned int)bx1; x++)`, so a cone that
    # falls entirely off the grid's left or bottom edge leaves bx1 NEGATIVE and the cast turns
    # it into about 4e9 iterations under the costmap mutex. Nothing is printed; the symptoms are
    # "Pose Goes Off Grid", services timing out, zero plans — and one thread at 100 %. That is
    # the measurement, and it is taken the only way this board is ever measured: /proc through
    # `ps` over ssh, never the ROS CLI (a `ros2 node list` costs seconds of A53 here). The share
    # is a thread's cumulative CPU time over the process's own lifetime — 1.00 is a thread that
    # has done nothing else since it started, and the wedged one measured 415 s of 700 s (0.59)
    # on 2026-09-21 because it only began to spin partway through. Nobody has yet measured what
    # the busiest thread of a HEALTHY container costs, so only the unmistakable case fails and
    # the middle is a WARN carrying the number; tighten it once a few healthy restarts have
    # printed theirs. A board that could not be read says so and is not failed for it.
    line="$(board_hot_thread)"
    value="${line#* }"; value="${value%% *}"   # the share
    if [[ ! "$value" =~ ^[0-9]+\.[0-9]+$ ]]; then
        warn 1.12 "nav2 threads: could not read the container's threads over ssh (ps -L on the board); the pegged-thread check did not run"
    elif over "$value" 0.90; then
        fail 1.12 "nav2 threads: thread ${line%% *} has used $value of the Nav2 container's whole lifetime (${line##* }) — that is a costmap update spinning under the mutex, not work: goals will be accepted and nothing planned. Restart the board half and check that no RangeSensorLayer is listed (ros/params/nav2_params.yaml)"
    elif over "$value" 0.50; then
        warn 1.12 "nav2 threads: busiest thread ${line%% *} at $value of the container's lifetime (${line##* }) — high, but this board has never been measured at rest; watch it (ros/board.sh census)"
    else
        pass 1.12 "nav2 threads: busiest thread ${line%% *} at $value of the container's lifetime (${line##* }), none pegged"
    fi

    check_map_odom
    check_clock
}

# 1.15: ONE CLOCK — the board's clock minus the one every laptop ROS node stamps with (the Docker
# VM's), measured over NTP from the board to the laptop's time server (ros/time.sh offset, which
# pipes src/pepin/timesync.py into the board's python3: one short process and eight 48-byte
# exchanges, no ROS). INFORMATION ONLY: a PASS or a WARN, never a FAIL and never a drive gate —
# the number is there so that a transform that "would require extrapolation" can be read against
# it. Under PEPIN_TIME_SOURCE=pool (the default until the chrony deploy) with no server here there
# is nothing to measure and that IS the configuration: a PASS that says so, not a WARN on every
# restart that teaches the eye to skip WARN lines. A WARN is left for what needs a look — over
# the threshold, a server that should run and does not, a board that cannot reach it (not
# measured), and a switch value that is neither laptop nor pool.
check_clock() {
    local out status=0 source="${PEPIN_TIME_SOURCE:-pool}"
    case "$source" in
        laptop | pool) ;;
        *) warn 1.15 "clock: PEPIN_TIME_SOURCE=$source is neither laptop nor pool (ros/lib.sh)"; return 0 ;;
    esac
    out="$("$HERE/time.sh" offset 2>&1)" || status=$?
    case "$status" in
        0) pass 1.15 "clock: $(tail -1 <<<"$out")" ;;
        1) warn 1.15 "clock: $(tail -1 <<<"$out") — stamps from the two machines disagree by that much (ros/time.sh status)" ;;
        3) if [ "$source" = pool ]; then
               pass 1.15 "clock: not measured, as configured — PEPIN_TIME_SOURCE=pool and no time server here, the board on the internet pool alone"
           else
               warn 1.15 "clock: not measured — PEPIN_TIME_SOURCE=laptop and no time server runs here (ros/time.sh server)"
           fi ;;
        *) warn 1.15 "clock: not measured — $(tail -1 <<<"$out" | cut -c1-160) (PEPIN_TIME_SOURCE=$source; ros/time.sh server, ros/time.sh status)" ;;
    esac
}

# 1.13: WHO IS CORRECTING THE POSE, under PEPIN_LOCALIZER=rtabmap. The board cannot answer this
# by itself: map -> odom is a TF edge and not a topic, and reading TF there means a /tf
# subscription at ~100 messages a second on four A53 cores, which CLAUDE.md rule 20 keeps off it
# (the board is measured through /proc and its nodes' own report lines, never the ROS CLI). So it
# is read WHERE THE PUBLISHER IS — one rclpy node in the laptop's own container
# (ros/tools/map_odom.py) — and the board's half of the same question is check 1.3, the goal
# server's socket `where`, whose pose is composed from this very edge.
#   Two readings, and both matter: FRESH (the transform is re-broadcast at 20 Hz, so seconds of
# silence is a publisher that is gone) and NOT THE IDENTITY (a localiser that has recognised
# nothing publishes map == odom, and every pose composed from it is simply the odometry's). The
# identity is correct for the first seconds of a start and a fault after them, which is why it is
# a WARN while the laptop half is young and a FAIL once it has had a minute to recognise the room.
MAP_ODOM_GRACE_S=60
check_map_odom() {
    local out status=0 started age
    if pepin_localizer_is_tracker; then
        warn 1.13 "map -> odom: owned by the board's tracker in this stack (PEPIN_LOCALIZER=$PEPIN_LOCALIZER); 1.2 is its report line"
        return 0
    fi
    if ! docker ps --format '{{.Names}}' | grep -qx pepin-vslam; then
        fail 1.13 "map -> odom: pepin-vslam is not running, so nothing publishes it at all (ros/laptop.sh vslam)"
        return 0
    fi
    out="$(docker exec pepin-vslam /pepin_entrypoint.sh timeout -s KILL 20 python3 /tools/map_odom.py 5 2>&1)" || status=$?
    started="$(docker inspect -f '{{.State.StartedAt}}' pepin-vslam 2>/dev/null || true)"
    age=$(( $(date +%s) - $(date -j -f '%Y-%m-%dT%H:%M:%S' "${started%%.*}" +%s 2>/dev/null || echo 0) ))
    if [ "$status" -ge 2 ]; then
        fail 1.13 "map -> odom: $(tail -1 <<<"$out" | cut -c1-160)"
    elif [ "$status" -eq 0 ]; then
        pass 1.13 "map -> odom: $(tail -1 <<<"$out" | cut -c1-160)"
    elif [ "$age" -lt "$MAP_ODOM_GRACE_S" ] 2>/dev/null; then
        warn 1.13 "map -> odom: $(tail -1 <<<"$out" | cut -c1-160) — the laptop half is ${age} s old, under the ${MAP_ODOM_GRACE_S} s it is given to recognise the room"
    else
        fail 1.13 "map -> odom: $(tail -1 <<<"$out" | cut -c1-160) — past the ${MAP_ODOM_GRACE_S} s grace, so the pose is the odometry's and nothing has recognised this room (ros/laptop.sh logs vslam)"
    fi
    # Laser odometry: asked for only when the board is configured to run it, and asked of the
    # DATA rather than of a log line — rf2o prints nothing periodic, and a node that is alive but
    # never publishing is exactly the failure this check exists for (it waits for its first scan
    # pair, and with its own `init_pose_from_topic` left at upstream's default it waits for ever).
    # Same measurement as /map and /vo above: one rclpy node inside the container for five
    # seconds, never the ros2 CLI.
    if ssh "root@$BOARD" "grep -q '^PEPIN_LASER_ODOM=false' /etc/default/pepin-ros" 2>/dev/null; then
        warn 1.14 "laser odometry: off on this board (ros/feature.sh laser_odom on); the EKF runs without odom3"
    else
        out="$(board_rate /odom_laser)"
        if [[ "$out" == *" Hz over "* ]]; then
            pass 1.14 "laser odometry: /odom_laser ${out#*: } (the EKF's odom3)"
        else
            fail 1.14 "laser odometry: /odom_laser is not flowing: $(tail -1 <<<"$out" | cut -c1-140) (is rf2o in the image? ros/board.sh census; docker logs pepin-ros | grep -ai rf2o)"
        fi
    fi
}

check_laptop() {
    step "checks: laptop"
    local started LOG line value n
    last() { grep -aE "$1" <<<"$LOG" | tail -1 || true; }

    if ! started="$(docker inspect -f '{{.State.StartedAt}}' pepin-vslam 2>/dev/null)"; then
        fail 2.1 "pepin-vslam is not running (ros/laptop.sh vslam --neck); no laptop check could run"
        return 0
    fi
    LOG="$(docker logs --since "$started" pepin-vslam 2>&1 || true)"   # one fetch, every grep below

    line="$(last 'bridge watch: [0-9]+ topics')"
    n="$(sed -n 's/.*bridge watch: \([0-9]*\) topics.*/\1/p' <<<"$line")"
    if pepin_rmw_is_zenoh; then
        # There is no bridge and no watch of one under zenoh: what this check is really asking —
        # do the board's topics reach this half — is asked again by 2.5 (the localizer hears the
        # board's belief) and by the rate kit, both of which read the data itself.
        pass 2.1 "bridge watch: n/a under PEPIN_RMW=zenoh (no bridge; 2.5 and the rate kit read the flows)"
    elif [ -z "$line" ]; then
        fail 2.1 "bridge watch: no line yet (it reports once the routes settle; ros/laptop.sh logs vslam)"
    elif [[ "$line" == *"DEAD ROUTES"* || "$line" == *"WITHOUT A READER"* ]]; then
        # Either side's routes: ours with no DDS endpoint, or the board's own pub routes with no
        # reader — the fault of 2026-09-15, which read as "dead routes 0" until the watch started
        # judging the far side too.
        fail 2.1 "bridge watch: ${line#*bridge watch: }"
    elif [ "${n:-0}" -lt 10 ]; then
        fail 2.1 "bridge watch: only $n topics carried, 10 expected: ${line#*bridge watch: }"
    else
        pass 2.1 "bridge watch: $n topics, dead routes 0, board routes without a reader 0"
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

    # JUDGED ON WHAT IT RECEIVES, NOT ON WHAT IT PAINTS. A parked cart integrates a view once and
    # then recognises it as the same view for ever ("291/300 revolutions (97 %) were the same view
    # again"), so a rate of PAINTED frames is near zero on a healthy node and this check read that
    # as a dead one (2026-09-19). The frames it was handed is the liveness; "at bound" stays a
    # failure because it means the volume's edge is refusing real observations.
    line="$(last '\]: fusion: ')"
    value="$(sed -n 's/.*: fusion: \([0-9]*\) frames.*/\1/p' <<<"$line")"
    n="$(sed -n 's/.*at bound \([0-9]*\)[^0-9].*/\1/p' <<<"$line")"
    if [ -z "$line" ]; then
        fail 2.3 "depth fusion: no report line (ros/laptop.sh logs vslam)"
    elif ! over "$value" 0; then
        fail 2.3 "depth fusion: ${value:-no} frames received in the window — the camera is not reaching it"
    elif [ "${n:-0}" -ne 0 ] 2>/dev/null; then
        fail 2.3 "depth fusion: $value frames but $n refused at bound (the volume's edge)"
    else
        pass 2.3 "depth fusion: $value frames received, at bound 0"
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
    if ! pepin_localizer_is_tracker; then
        # The belief it listens for is the board tracker's /tracker_pose, and under this switch no
        # tracker runs: silence is the design, not a broken bridge (1.2 says the same of the board).
        warn 2.5 "laptop localizer: n/a under PEPIN_LOCALIZER=$PEPIN_LOCALIZER (no tracker publishes the belief it listens for)"
    elif [ -z "$line" ]; then
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

    # The one input RTAB-Map reads (pepin_bringup.sensor_pack): a snapshot per moment out of
    # whatever sensor is alive. No snapshots is a mapper that is fed nothing, and then neither the
    # graph nor the grid can move, whatever else looks healthy.
    line="$(last '\]: sensor pack: ')"
    value="$(sed -n 's/.*sensor pack: \([0-9.]*\) snapshots\/s.*/\1/p' <<<"$line")"
    if [ -z "$line" ]; then
        fail 2.7 "sensor pack: no report line (is the node up? ros/laptop.sh logs vslam)"
    elif ! over "$value" 0; then
        fail 2.7 "sensor pack: ${value:-no} snapshots/s — RTAB-Map is fed nothing, so its graph and its grid cannot move (does /scan cross? is the camera up?)"
    else
        pass 2.7 "sensor pack: $value snapshots/s into RTAB-Map"
    fi

    # ...and the graph's own word to the board's tracker (pepin_bringup.rtabmap_frame): it is one
    # measurement among the tracker's sources now, never a correction, so what this asks is whether
    # the node hears RTAB-Map at all. The fragment is a LITERAL of that node's own line —
    # "rtabmap frame: 118 updates, 0 recognised a node, 118 localisations heard, 0 words (...)" —
    # because the previous wording ("over N infos") had been gone for a day and this check failed a
    # healthy node on it (2026-09-19).
    line="$(last '\]: rtabmap frame: ')"
    n="$(sed -n 's/.*rtabmap frame: \([0-9]*\) updates.*/\1/p' <<<"$line")"
    value="$(sed -n 's/.*, \([0-9]*\) localisations heard.*/\1/p' <<<"$line")"
    if [ -z "$line" ]; then
        fail 2.10 "rtabmap frame: no report line (ros/laptop.sh logs vslam)"
    elif [ "${n:-0}" -eq 0 ] 2>/dev/null; then
        fail 2.10 "rtabmap frame: 0 updates — it hears nothing from RTAB-Map (/rtabmap/info is not arriving)"
    else
        pass 2.10 "rtabmap frame: $n updates from RTAB-Map, ${value:-0} localisations heard"
    fi

    # 2.12: whether the camera half can run the visual registration's default features.
    # rtabmap_frame's visual_features defaults to xfeat, which needs the Python adapters only
    # pepin-laptop:xfeat carries (ros/laptop.sh picks that image when it exists); in any other
    # image RTAB-Map quietly registers with ORB, which accepted 0 of 630 camera-only updates on
    # 2026-09-23. The adapter itself is asked for, as rtabmap_frame asks, so a rollback image that
    # carries it passes too. PEPIN_XFEAT=0 says ORB is meant.
    value="$(docker inspect -f '{{.Config.Image}}' pepin-vslam 2>/dev/null || true)"
    if docker exec pepin-vslam test -f /opt/xfeat/rtabmap_xfeat.py >/dev/null 2>&1; then
        pass 2.12 "vslam image: ${value:-unknown} carries the xfeat adapters"
    elif [ "${PEPIN_XFEAT:-}" = 0 ]; then
        pass 2.12 "vslam image: ${value:-unknown}, no xfeat adapters — PEPIN_XFEAT=0, ORB is meant"
    else
        fail 2.12 "vslam image: ${value:-unknown} has no xfeat adapters — visual_features falls back to orb (ros/laptop-build.sh xfeat; PEPIN_XFEAT=0 if ORB is meant)"
    fi

    # 2.13: the localisation service on this laptop's GPU (ros/models.sh, launchd): RTAB-Map's
    # XFeat / LighterGlue adapters and sensor_pack's place descriptors call it. It must answer
    # /health with its three models, none of them failed to build. Down, the adapters compute in
    # RTAB-Map's own process (registration_backend auto, 0.67 s a registration on the VM's CPU)
    # and every snapshot carries the null place descriptor, so place_recognition descriptor falls
    # back to the words (descriptor_null_share) — slower, never broken — so that is a WARN, said
    # loudly under place_recognition descriptor; under registration_backend service it is a
    # FAIL, because then a registration the service does not answer finds no features at all.
    line="$(last '\]: rtabmap frame: ')"
    value="$(sed -n 's/.*registration_backend=\([a-z]*\).*/\1/p' <<<"$line")"
    recognition="$(sed -n 's/.*place_recognition=\([a-z]*\).*/\1/p' <<<"$line")"
    local health models_line
    health="$(curl -s -m 3 "http://127.0.0.1:${PEPIN_MODELS_PORT:-8791}/health" || true)"
    models_line="$(python3 -c '
import json, sys
h = json.loads(sys.stdin.read())
names = ("xfeat", "match", "place")
bad = [n for n in names if n not in h["models"] or h["models"][n]["tag"].startswith("failed")]
print(("BAD " if bad else "OK ") + "; ".join(
    "%s %s on %s, %d served, %d refused" % (n, m["tag"], m["device"], m["requests"], m["errors"])
    for n, m in h["models"].items()))' <<<"$health" 2>/dev/null || true)"
    if [[ "$models_line" == OK* ]]; then
        pass 2.13 "localization service: ${models_line#OK }"
    elif [ "${value:-auto}" = service ]; then
        fail 2.13 "localization service ${models_line:-not answering on :${PEPIN_MODELS_PORT:-8791}} under registration_backend service: RTAB-Map's registrations find no features (ros/models.sh status; ros/flags.sh set rtabmap_frame registration_backend auto)"
    elif [ "$recognition" = descriptor ]; then
        warn 2.13 "localization service ${models_line:-not answering on :${PEPIN_MODELS_PORT:-8791}} UNDER place_recognition descriptor: the snapshots carry null place descriptors, so RTAB-Map recognises places by the WORDS until it answers again, and the adapters compute in RTAB-Map's process (${value:-auto}) (ros/models.sh start localization)"
    else
        warn 2.13 "localization service ${models_line:-not answering on :${PEPIN_MODELS_PORT:-8791}}: the adapters compute in RTAB-Map's process (${value:-auto}) and the snapshots carry null place descriptors (ros/models.sh start localization)"
    fi

    # 2.11 is INFORMATIONAL and never fails a restart: who painted the lethal cells of the local
    # costmap right now (pepin_bringup.marks_audit). There is no healthy value — a room with a
    # table in it SHOULD show camera-only cells — so this prints the split and leaves the verdict
    # to the person in front of the robot. Absent when the node is out (marks_audit:=false) or
    # when no grid has crossed yet, and that absence is worth seeing too.
    line="$(last '\]: marks audit: ')"
    if [ -z "$line" ]; then
        warn 2.11 "marks audit: no report line (marks_audit:=false, or no local costmap has reached this laptop yet)"
    else
        warn 2.11 "marks audit: ${line#*: marks audit: }"
    fi

    check_foxglove

    n="$(grep -ac 'process has died' <<<"$LOG" || true)"
    if [ "${n:-0}" -eq 0 ] 2>/dev/null; then
        pass 2.8 "no node died since the container started"
    else
        fail 2.8 "$n node(s) died since the container started: $(grep -a 'process has died' <<<"$LOG" | tail -1 | cut -c1-140)"
    fi
}

check_foxglove() {  # 2.9: the operator's window — the bridge answers, and it advertises what
    # the layout draws. One line here, the failing details indented under it; the whole list is
    # ros/foxglove.sh check.
    local out line status=0
    out="$(PEPIN_FOXGLOVE_PREFIX=2.9 "$HERE/foxglove.sh" check 2>&1)" || status=$?
    if [ "$status" -eq 0 ]; then
        pass 2.9 "$(tail -1 <<<"$out")"
    else
        fail 2.9 "$(tail -1 <<<"$out")"
        while IFS= read -r line; do
            [ -z "$line" ] || printf '          %s\n' "$line"
        done < <(grep '^FAIL' <<<"$out")
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
if [ "$DRY" = true ]; then
    describe
    exit 0
fi
# Both halves come back before anything is checked: the two topics the board is asked about
# (/depth_scan, /vo) are fed BY the laptop, so checking the board first would fail them on purpose.
# Board first, always: its restart re-zeroes the odometry RTAB-Map runs on.
[ "$HALF" = laptop ] || restart_board || fail 0.1 "the board's restart did not finish (above); the checks say what is missing"
if [ "$HALF" != board ]; then
    restart_laptop || fail 0.2 "the laptop's restart did not finish (above); the checks say what is missing"
elif laptop_up; then
    restart_laptop vslam || fail 0.2 "the laptop's vslam restart did not finish (above); the checks say what is missing"
else
    step "the laptop half is down"
    echo "nothing publishes the map or map -> odom: the board's Nav2 waits for them, and ros/restart.sh laptop brings them and proves the planner"
fi
if [ "$CHECK" != true ]; then
    printf '\nno proof and no checks (--no-check)\n'
    [ "$WAIT" = true ] || echo "--fast: nothing was waited for; ros/restart.sh $HALF proves the stack when it matters"
    exit 0
fi
ensure_planner  # before the checks: they then read the stack as it will be driven
[ "$HALF" = laptop ] || check_board
[ "$HALF" = board ] || check_laptop
check_flags   # last: one `ros2 param dump` per node, the slowest thing here
# The restart took the bridge with it, and a Foxglove client bound to the dead socket shows empty
# panels until it reconnects. So the app is told to, as the last act of the restart (it is only
# told when it is running; PEPIN_FOXGLOVE_REOPEN=0 leaves it alone).
if [ "$HALF" != board ] && [ "${PEPIN_FOXGLOVE_REOPEN:-1}" = 1 ]; then
    step "foxglove"
    "$HERE/foxglove.sh" reopen || true
fi
step "verdict"
if [ "$FAILED" -eq 0 ]; then
    echo "green: $TOTAL checks, none failed"
else
    echo "red: $FAILED of $TOTAL checks failed"
    exit 1
fi
