#!/bin/bash
# One command that brings this robot up WORKING, repairs what it can and says what it could not:
# the restart, the proof that the planner plans, and every check that has bitten us. Usage:
#
#   ros/restart.sh board|laptop|both [--deploy] [--fresh-graph] [--no-check] [--fast] [--dry-run]
#
#   board          the board's sensor stack WITH its zenoh router (ros/lib.sh pepin_board_restart:
#                  stack stopped, router restarted, stack started), then the sensors' ready line
#                  (90 s). If the laptop half is up, its vslam is restarted after it: RTAB-Map
#                  follows the board's new odometry only from a fresh start (journal 2026-09-22)
#   --deploy       ros/sync.sh --restart instead: code + the library + config to the board, the
#                  same router-and-stack restart, and its census tail
#   laptop         ros/laptop.sh vslam --neck, then ros/laptop.sh nav (Nav2 on this Mac). Nothing
#                  about the map is passed: the database IS the map, the launch reads whether it
#                  exists
#   --fresh-graph  the camera half starts on an empty RTAB-Map database (laptop.sh vslam --fresh)
#                  AND the volume of the old frame is moved aside: the volume is painted in the
#                  graph's frame, so a kept snapshot beside an empty database is a room painted
#                  somewhere else
#   both           board first, then laptop. Never the other way: a board restart re-zeroes the
#                  odometry under a running RTAB-Map. Nav2 waits for the map and map -> odom
#                  (initial_transform_timeout, ros/params/nav2_params.yaml) and the proof below
#                  catches whatever still races
#   --no-check     restart only: no proof, no repair, no checks
#   --fast         --no-check, and no waiting for either half's first report line: the commands
#                  are sent and the script returns; the stack still needs its own time to come
#                  up, and nothing here says whether it did
#   --dry-run      print the order of everything above and touch nothing
#
# THE PROOF (4.1), whenever the laptop half is up after the restart: from pepin-vslam, the pose
# corrected (ros/tools/map_odom.py) and the planner planning (ros/tools/planner_check.py: a new
# global costmap within 10 s, one ComputePathToPose 0.5 m ahead — a plan, never motion), asked
# again for up to PEPIN_PLANNER_WAIT_S (180) while Nav2 comes up. Broken, it REPAIRS itself: first
# the Nav2 container alone (SIGINT; the launch respawns it in 2 s with fresh nodes, odometry and
# RTAB-Map untouched), then the board half once more with its router and the laptop half after
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
WAIT_BOARD_S="${PEPIN_RESTART_WAIT_S:-90}"    # the board's stack to the recorder's ready line
WAIT_LAPTOP_S="${PEPIN_RESTART_WAIT_S:-120}"  # the camera half to the depth stream's first line
                                              # (a 30 s report timer, the ghost wait, the network)
POLL_S="${PEPIN_RESTART_POLL_S:-5}"
# How long the proof keeps asking the planner: Nav2 comes up once the map and RTAB-Map's first
# correction are there, and that correction can take a minute after a restart.
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
board_rate() {  # TOPIC [latched] -> one line: is it reaching the board, and how fast.
    # ros/tools/topic_rate.py, not `ros2 topic hz`: the CLI costs ~4.5 s of start-up on four A53
    # cores before it measures anything, the tool is one rclpy node and answers in one line.
    # `latched` asks for the held copy of a topic published once per change instead of a rate.
    ssh "root@$BOARD" \
        "docker exec pepin-ros /pepin_entrypoint.sh timeout -s KILL 15 python3 /tools/topic_rate.py $1 5 ${2:-}" \
        2>&1 || true
}
over() { awk -v v="${1:-0}" -v t="$2" 'BEGIN { exit !(v + 0 > t) }'; }  # VALUE > THRESHOLD, floats
nav_log() {  # WINDOW_S -> that much of the Mac's Nav2 container log (ros/laptop.sh nav)
    docker logs --since "${1}s" pepin-macnav 2>&1
}

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
    # Never under a drive: the restart zeroes the odometry a running goal is steered by. Every
    # goal is cancelled first; a goal server that does not answer has none to cancel.
    "$HERE/goto.sh" cancel || echo "no goal confirmed cancelled (is Nav2 up?): the board restarts all the same"
    if [ "$DEPLOY" = true ]; then
        "$HERE/sync.sh" --restart  # code + params + router and stack + the census tail; a red census is information
    else
        pepin_board_restart
    fi
    # Which line says this half is back: the sensors' lifecycle manager activating the lidar,
    # which the board's stack starts in every arrangement.
    if [ "$WAIT" != true ]; then
        echo "--fast: not waiting for the board's first report line"
    else
        wait_for "board" "$WAIT_BOARD_S" pepin-ros "lifecycle_manager_sensors.*Managed nodes are active" || true
    fi
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

restart_laptop() {  # [vslam]: the camera half only (after a board restart), else vslam then Nav2
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
    "$HERE/laptop.sh" "${args[@]}"
    [ "${1:-}" = vslam ] || "$HERE/laptop.sh" nav
    if [ "$WAIT" != true ]; then
        echo "--fast: not waiting for the laptop's first depth line"
        return 0
    fi
    wait_for "laptop" "$WAIT_LAPTOP_S" pepin-vslam "\]: depth: " || true
}

laptop_up() { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx pepin-vslam; }

# ---- the proof: the planner plans -------------------------------------------------------------
renav() {  # the Nav2 component container, SIGINT; its launch respawns it in 2 s, fresh nodes
    # launch_kit's respawned_container in pepin-macnav; odometry, the sensors and RTAB-Map are
    # other processes and never notice. A bring-up that gave up before the map came (the
    # start-order race) comes back active in seconds — the sim: 5.7 s on the Mac.
    if docker exec pepin-macnav pkill -INT -f '__node:=nav2_container' 2>/dev/null; then
        echo "  Nav2 container respawning"
    else
        echo "  no Nav2 container to respawn (is pepin-macnav up? ros/laptop.sh nav)"
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

ensure_planner() {  # 4.1: prove it, repair it (Nav2 alone, then both halves once more), or fail
    step "the planner"
    if ! laptop_up; then
        fail 4.1 "planner: not proven — the laptop half is down, so there is no map and no map -> odom to plan on (Nav2 waits for them); ros/restart.sh laptop brings them and proves the planner"
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
        echo "repair 2: the board half once more (router and stack), then the laptop half (vslam on its new odometry, Nav2)"
        DEPLOY=false; FRESH_GRAPH=false  # the code went out, and the new database stays
        restart_board || true
        restart_laptop || true
        repaired="a Nav2 respawn and a board restart"
        status=0; prove_planner || status=$?
    fi
    case "$status" in
        0) pass 4.1 "${PROOF#planner: }${repaired:+ — after $repaired}" ;;
        3) fail 4.1 "${PROOF#planner: } — the planner itself works, so no restart can help: the cart is boxed in, or the global costmap is full of marks (Foxglove's planner panel)" ;;
        *) fail 4.1 "${PROOF#planner: } — STILL BROKEN after $repaired: nothing will plan (ros/laptop.sh nav logs)" ;;
    esac
}

# ---- --dry-run: the order, and nothing touched ------------------------------------------------
describe() {
    local n=0 fresh=""
    [ "$FRESH_GRAPH" = false ] || fresh=" --fresh (the old volume moved aside)"
    say() { n=$((n + 1)); printf '%2d. %s\n' "$n" "$1"; }
    echo "dry run, nothing is touched — ros/restart.sh $HALF would:"
    if [ "$HALF" != laptop ]; then
        say "cancel every goal (ros/goto.sh cancel): the board's restart zeroes the odometry a drive is steered by"
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
        *) say "laptop: ros/laptop.sh vslam --neck$fresh$waits, then ros/laptop.sh nav (Nav2 on this Mac)" ;;
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
    say "broken -> repair 1: SIGINT to the Nav2 container in pepin-macnav; its launch respawns it; the proof again"
    say "still broken -> repair 2: every goal cancelled, the board once more (stop pepin-ros, restart pepin-zrouter, start pepin-ros; no sync), ros/laptop.sh vslam --neck and ros/laptop.sh nav, the proof again; still broken -> FAIL 4.1, loudly. \"No path around\" is never repaired"
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

    # WHERE THE CART THINKS IT IS: the goal server's socket `where`, composed from TF
    # (map -> base_link) — RTAB-Map's correction with the board's own odometry.
    out="$("$HERE/goto.sh" where 2>&1 || true)"
    if [[ "$out" != *'"event": "where"'* ]]; then
        fail 1.3 "pose: the goal server did not answer 'where' on its socket: $(tail -1 <<<"$out" | cut -c1-140)"
    elif [[ "$out" == *'"pose": "tf"'* ]]; then
        pass 1.3 "pose: from TF — $(tr '\n' ' ' <<<"$out" | cut -c1-160)"
    else
        fail 1.3 "pose: the goal server answers '$(sed -n 's/.*"pose": "\([a-z]*\)".*/\1/p' <<<"$out" | tail -1)', not 'tf' — nothing publishes map -> base_link (is ros/laptop.sh vslam up?)"
    fi

    # Nav2's own complaints, from the Mac's container (ros/laptop.sh nav): a starved costmap or
    # controller loop, and transforms the costmaps could not place.
    local navlog
    if navlog="$(nav_log "$ERROR_WINDOW_S")"; then
        n="$(grep -ac 'Failed to meet update rate' <<<"$navlog" || true)"
        if [ "$n" = 0 ]; then
            pass 1.4 "loop rate: no 'Failed to meet update rate' in the last ${ERROR_WINDOW_S} s"
        else
            fail 1.4 "loop rate: $n x 'Failed to meet update rate' in the last ${ERROR_WINDOW_S} s (ros/laptop.sh nav logs)"
        fi
        line="$(grep -aE 'Extrapolation|out of map bounds|Off Grid' <<<"$navlog" | tail -1 || true)"
        n="$(grep -acE 'Extrapolation|out of map bounds|Off Grid' <<<"$navlog" || true)"
        if [ "$n" = 0 ]; then
            pass 1.5 "tf and costmaps: no extrapolation / off-grid error in the last ${ERROR_WINDOW_S} s"
        else
            fail 1.5 "tf and costmaps: $n error(s) in the last ${ERROR_WINDOW_S} s: $(cut -c1-140 <<<"$line")"
        fi
    else
        fail 1.4 "loop rate: Nav2's log could not be read (is pepin-macnav up? ros/laptop.sh nav)"
        fail 1.5 "tf and costmaps: Nav2's log could not be read (ros/laptop.sh nav)"
    fi

    # The camera's odometry the board's EKF fuses (odom1): the one laptop topic the board needs.
    out="$(board_rate /vo)"
    if [[ "$out" == *" Hz over "* ]]; then
        pass 1.8 "/vo reaches the board: ${out#*: }"
    else
        fail 1.8 "/vo does not reach the board: $(tail -1 <<<"$out" | cut -c1-140) (the laptop half and the routers)"
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

    check_map_odom
    check_clock
}

# 1.15: ONE CLOCK — the board's clock minus the one every laptop ROS node stamps with (the Docker
# VM's), measured over NTP from the board to the laptop's time server (ros/time.sh offset, which
# pipes scripts/timesync.py into the board's python3: one short process and eight 48-byte
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

# 1.13: WHO IS CORRECTING THE POSE. The board cannot answer this
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

    # ...and RTAB-Map's side of the one map (pepin_bringup.rtabmap_frame): whether the node hears
    # RTAB-Map at all. The fragment is a LITERAL of that node's own line —
    # "rtabmap frame: 118 updates, 0 recognised a node, 118 localisations heard; ..." —
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
# Both halves come back before anything is checked: the topic the board is asked about (/vo) is
# fed BY the laptop, so checking the board first would fail it on purpose.
# Board first, always: its restart re-zeroes the odometry RTAB-Map runs on.
[ "$HALF" = laptop ] || restart_board || fail 0.1 "the board's restart did not finish (above); the checks say what is missing"
if [ "$HALF" != board ]; then
    restart_laptop || fail 0.2 "the laptop's restart did not finish (above); the checks say what is missing"
elif laptop_up; then
    restart_laptop vslam || fail 0.2 "the laptop's vslam restart did not finish (above); the checks say what is missing"
else
    step "the laptop half is down"
    echo "nothing publishes the map or map -> odom: Nav2 waits for them, and ros/restart.sh laptop brings them and proves the planner"
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
