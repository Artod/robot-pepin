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
#   laptop         ros/laptop.sh start, then ros/laptop.sh vslam --neck. Nothing about the map is
#                  passed any more: the database IS the map (World R), the launch reads whether it
#                  exists and the board's tracker adopts whatever grid it publishes
#   --fresh-graph  the camera half starts on an empty RTAB-Map database (laptop.sh vslam --fresh)
#                  AND the volume of the old frame is moved aside: the volume is painted in the
#                  graph's frame, so a kept snapshot beside an empty database is a room painted
#                  somewhere else
#   both           board first, then laptop; the checks run after both are up, so the two topics
#                  the laptop feeds the board are asked for when there is a laptop to feed them
#   --no-check     restart only
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
board_rate() {  # TOPIC -> one line: is it reaching the board, and how fast.
    # ros/tools/topic_rate.py, not `ros2 topic hz`: the CLI costs ~4.5 s of start-up on four A53
    # cores before it measures anything, the tool is one rclpy node and answers in one line.
    ssh "root@$BOARD" \
        "docker exec pepin-ros /pepin_entrypoint.sh timeout -s KILL 15 python3 /tools/topic_rate.py $1 5" \
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
    step "restarting the board"
    if [ "$DEPLOY" = true ]; then
        "$HERE/sync.sh"   # code + params + restart + the census tail; a red census is information
    else
        ssh "root@$BOARD" "systemctl restart pepin-ros && sleep 8 && systemctl is-active pepin-ros"
    fi
    wait_for "board" "$WAIT_BOARD_S" pepin-ros "relocalizer\]: tracker:" || true
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

restart_laptop() {
    step "restarting the laptop"
    # Nothing about the map is passed: one database, one grid, and the launch reads for itself
    # whether that database exists (an empty room is the file being absent).
    local args
    args=(vslam --neck)
    if [ "$FRESH_GRAPH" = true ]; then
        args+=(--fresh)
        drop_volume
    fi
    "$HERE/laptop.sh" start
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
    # /map first: it is THE map (World R), RTAB-Map's grid republished at its detection rate (1 Hz,
    # map_always_update) and routed here for the tracker. Then the camera's other two words.
    for value in /map /depth_scan /vo; do
        out="$(board_rate "$value")"
        if [[ "$out" == *" Hz over "* ]]; then
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
    if [[ "$out" == *"not advertised"* ]]; then
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
    # tof_bridge module docstring for the fix). Two questions, one line: did the lifecycle
    # manager get planner_server's bond, and is the log free of the complaint that says the
    # wedge is running. The bond is printed once at activation, so it is looked for across the
    # whole restart, not the report window; the container is recreated on every restart
    # (board/pepin-ros.service), so no older start can match. A board that runs no Nav2
    # (PEPIN_NAV=false) has nothing to judge here and is not failed for it.
    value="$(board_last "$((WAIT_BOARD_S + REPORT_WINDOW_S))" 'Activating planner_server')"
    line="$(board_last "$((WAIT_BOARD_S + REPORT_WINDOW_S))" 'planner_server connected with bond')"
    n="$(board_count "$REPORT_WINDOW_S" "Range sensor layer can't transform")"
    if [ -z "$value" ]; then
        warn 1.11 "nav2: no 'Activating planner_server' in the board's log — Nav2 does not run on this half (PEPIN_NAV), nothing to judge"
    elif [ -z "$line" ]; then
        fail 1.11 "nav2: planner_server was activated and never bonded — it is WEDGED in its global costmap's first update ($n x 'Range sensor layer can't transform' in the last ${REPORT_WINDOW_S} s). Goals are accepted and nothing is planned; restart the board half (ros/restart.sh board) and, if it comes back, ros/tools/coldstart_soak.sh"
    elif [ "$n" = "?" ]; then
        fail 1.11 "nav2: planner_server bonded, but the board's log could not be read for the range-layer complaint (ssh root@$BOARD docker logs pepin-ros)"
    elif [ "$n" != 0 ]; then
        fail 1.11 "nav2: $n x 'Range sensor layer can't transform' in the last ${REPORT_WINDOW_S} s — a costmap update is blocking a whole transform_tolerance per ToF message and is on its way to the wedge: read tof_bridge's gate in its report line (ros/watch.sh, ros/flags.sh list tof_bridge) and restart the board half"
    else
        pass 1.11 "nav2: planner_server connected with bond, 0 range-layer transform failures in the last ${REPORT_WINDOW_S} s"
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
