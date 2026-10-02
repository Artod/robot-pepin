#!/bin/bash
# The kinematic simulator for Nav2, on this Mac (ros/sim/, ros/README.md "Simulation"): the
# stack's own Nav2 launch (nav.launch.py, ros/params/nav2_params.yaml, the goal server,
# the run recorder) driving a simulated cart in the room RTAB-Map saved. Usage:
#   ros/sim.sh up [--rate N] [--world NAME] [--start PLACE] [--nav-first S]   router, world and
#                                Nav2 in throwaway containers; --rate N runs the world's clock at N x
#                                the wall clock (every node on use_sim_time), without it everything
#                                is wall time; --nav-first S starts Nav2 S s before the world
#   ros/sim.sh down              remove the three containers (the run directory stays)
#   ros/sim.sh status            containers, Nav2's bring-up, the world's pose and odometer
#   ros/sim.sh goal NAME | X Y [YAW_DEG]    drive through the goal server's socket, scored
#   ros/sim.sh place NAME | X Y [YAW_DEG]   put the cart there, standing still; costmaps emptied
#   ros/sim.sh boxes FILE | none            furniture in the map frame (a yaml list of boxes)
#   ros/sim.sh scenario FILE [--repeat N]   place, furnish, drive every leg, score each
#   ros/sim.sh stuck [--repeat N]           the shelf pocket: ros/sim/scenarios/shelf_pocket.yaml
#   ros/sim.sh cancel            cancel every goal (pepin.goal_link, as ros/goto.sh cancel does)
#   ros/sim.sh logs [world|nav|router]      follow a container's output
#   ros/sim.sh map               re-export ros/sim/worlds/flat from ros/maps/rtabmap.db (a copy)
# Scores go to the terminal and to ros/sim/run/maps/rec/sim_scores.jsonl; the goal server's
# numbered tapes to ros/sim/run/maps/rec. The planner is the one the stack's goal server picked
# last (ros/maps/rec/.planner), copied into the run directory at the first `up`.
# ISOLATION: the three containers share ONE network namespace whose only interface is loopback
# (the router's --network none; the other two join it), so nothing here reaches the board, the
# LAN or the live pepin-* containers, and no port is published. ROS_DOMAIN_ID 42 besides. This
# script never starts Docker Desktop: that would also restart the live pepin-* containers
# (--restart unless-stopped).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$HERE/.." && pwd)"
SIM="$HERE/sim"
RUN="${PEPIN_SIM_RUN:-$SIM/run}"
ROUTER=pepin-sim-router
WORLD=pepin-sim-world
NAV=pepin-sim-nav
DOMAIN=42
ZPORT=7447
SITE=/ws/install/pepin_bringup/lib/python3.12/site-packages/pepin_bringup
. "$HERE/lib.sh"  # pepin_remove_container: the one gentle stop, the log kept in logs/containers

need_docker() {
    docker info >/dev/null 2>&1 && return 0
    echo "Docker is not running. Starting Docker Desktop also restarts the live pepin-* containers" \
         "(--restart unless-stopped): start it yourself when that is fine, then run this again." >&2
    exit 1
}
# The board's own image (the Nav2 the controller, the behaviours and the tree run in: 1.3.13 on
# 2026-09-27, the laptop image's planner 1.3.12), then the laptop's; PEPIN_SIM_IMAGE wins.
image() {
    local candidate
    for candidate in ${PEPIN_SIM_IMAGE:-} pepin-ros:zenoh pepin-laptop:zenoh; do
        if docker image inspect "$candidate" >/dev/null 2>&1; then echo "$candidate"; return 0; fi
    done
    echo "no image: build pepin-ros (ros/build-image.sh) or set PEPIN_SIM_IMAGE" >&2
    return 1
}
MOUNTS=(-v "$HERE/pepin_bringup/pepin_bringup:$SITE:ro"
        -v "$HERE/pepin_bringup/launch:/ws/install/pepin_bringup/share/pepin_bringup/launch:ro"
        -v "$HERE/tools:/tools:ro" -v "$HERE/entrypoint.sh:/pepin_entrypoint.sh:ro"
        -v "$REPO/src/pepin:/ws/pepin_src/pepin:ro" -v "$HERE/params:/params:ro"
        -v "$REPO/config:/ws/config:ro" -v "$SIM:/sim:ro" -v "$RUN/maps:/maps")
# Every node a peer of the sim's own router, over the shared loopback; the stack's other
# settings as ros/laptop.sh gives them (no tracker: RTAB-Map's role is the world's).
ENVS=(-e RMW_IMPLEMENTATION=rmw_zenoh_cpp -e PEPIN_RMW=zenoh -e ZENOH_ROUTER_CHECK_ATTEMPTS=0
      -e "ZENOH_CONFIG_OVERRIDE=connect/endpoints=[\"tcp/127.0.0.1:$ZPORT\"];listen/endpoints=[\"tcp/127.0.0.1:0\"]"
      -e ROS_DOMAIN_ID=$DOMAIN -e PYTHONUNBUFFERED=1)

down() { pepin_remove_container "$NAV" "$WORLD" "$ROUTER"; }  # Nav2 leaves before its router

in_world() { docker exec "$WORLD" /pepin_entrypoint.sh "$@"; }

inside() {  # FILE: its path in the containers (/sim/...); a file outside ros/sim is refused
    local real
    real="$(cd "$(dirname "$1")" 2>/dev/null && pwd)/$(basename "$1")"
    case "$real" in
        "$SIM"/*) echo "/sim/${real#"$SIM"/}" ;;
        *) [ -f "$SIM/scenarios/$1" ] && { echo "/sim/scenarios/$1"; return 0; }
           echo "$1: the containers see ros/sim only (put it in ros/sim/scenarios)" >&2; return 1 ;;
    esac
}

wait_for() {  # CONTAINER TEXT SECONDS: seconds until TEXT shows in the log, or fail
    local t0; t0=$(date +%s)
    while [ $(( $(date +%s) - t0 )) -lt "$3" ]; do
        # grep reads the whole log: -q would quit at the first match, docker logs would die of
        # SIGPIPE and pipefail would call a found line missing
        if docker logs "$1" 2>&1 | grep -F "$2" >/dev/null; then echo $(( $(date +%s) - t0 )); return 0; fi
        if [ "$(docker inspect -f '{{.State.Running}}' "$1" 2>/dev/null)" != true ]; then
            echo "$1 is not running: ros/sim.sh logs ${1#pepin-sim-}" >&2; return 1
        fi
        sleep 1
    done
    echo "no '$2' from $1 in $3 s: ros/sim.sh logs ${1#pepin-sim-}" >&2
    return 1
}

up() {
    local rate=0 world=flat start=home sim_time=false img nav_first=""
    while [ $# -gt 0 ]; do
        case "$1" in
            --rate) rate="$2"; shift 2 ;;
            --world) world="$2"; shift 2 ;;
            --start) start="$2"; shift 2 ;;
            --nav-first) nav_first="$2"; shift 2 ;;
            *) echo "up: unknown option $1" >&2; exit 2 ;;
        esac
    done
    [ -f "$SIM/worlds/$world.yaml" ] || { echo "no world $SIM/worlds/$world.yaml (ros/sim.sh map)" >&2; exit 2; }
    need_docker
    img="$(image)"
    case "$rate" in 0 | 0.0) ;; *) sim_time=true ;; esac
    down
    mkdir -p "$RUN/maps/rec"
    if [ ! -f "$RUN/maps/rec/.planner" ]; then
        if [ -f "$HERE/maps/rec/.planner" ]; then cp "$HERE/maps/rec/.planner" "$RUN/maps/rec/.planner"
        else echo hybrid > "$RUN/maps/rec/.planner"; fi
    fi
    echo "sim: image $img, world $world, start $start, clock $([ "$sim_time" = true ] && echo "sim x$rate" || echo wall), planner $(cat "$RUN/maps/rec/.planner")"
    docker run -d --name "$ROUTER" --network none \
        -e RMW_IMPLEMENTATION=rmw_zenoh_cpp -e ROS_DOMAIN_ID=$DOMAIN \
        -e "ZENOH_CONFIG_OVERRIDE=listen/endpoints=[\"tcp/127.0.0.1:$ZPORT\"]" \
        -v "$HERE/entrypoint.sh:/pepin_entrypoint.sh:ro" \
        "$img" /opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd >/dev/null
    start_world() {
        docker run -d --name "$WORLD" --network "container:$ROUTER" "${MOUNTS[@]}" "${ENVS[@]}" "$img" \
            python3 /sim/sim_world.py --world "/sim/worlds/$world.yaml" \
            --places "/sim/worlds/$world.places.json" --start "$start" --rate "$rate" >/dev/null
        wait_for "$WORLD" "sim world up" 60 >/dev/null
    }
    start_nav() {
        docker run -d --name "$NAV" --network "container:$ROUTER" "${MOUNTS[@]}" "${ENVS[@]}" "$img" \
            ros2 launch /sim/sim_nav.launch.py "use_sim_time:=$sim_time" "map:=/sim/worlds/$world.yaml" >/dev/null
    }
    # --nav-first S: Nav2 up S seconds BEFORE the world, i.e. before anything publishes /map, TF
    # or the placement word — the order of a `restart.sh both` whose board beats the laptop's
    # RTAB-Map (journal 2026-09-25, "planner_server never activated"). Default: the world first.
    if [ -n "$nav_first" ]; then
        start_nav
        echo "sim: Nav2 started with no world; the world follows in $nav_first s"
        sleep "$nav_first"
        start_world
    else
        start_world
        start_nav
    fi
    local active ready
    if ! active="$(wait_for "$NAV" "Managed nodes are active" 180)"; then
        echo "sim up: Nav2 did NOT come up; its lifecycle manager said:"
        docker logs "$NAV" 2>&1 | grep -E "lifecycle_manager|Timed out|Failed" | tail -8
        return 1
    fi
    ready="$(wait_for "$NAV" "goal server ready on port" 60)"
    echo "sim up: Nav2 active after ${active} s, goal server ready after ${ready} s more"
    status
}

status() {
    docker ps -a --filter name=pepin-sim- --format '{{.Names}}\t{{.Status}}'
    docker logs "$NAV" 2>&1 | grep -E "Managed nodes are active|goal server ready|planner .* with controller" | tail -3 || true
    in_world python3 /sim/sim_score.py state 2>/dev/null || echo "world: not answering"
}

case "${1:-}" in
    up) shift; up "$@" ;;
    down) need_docker; down; echo "sim down (run directory kept: $RUN)" ;;
    status) need_docker; status ;;
    goal | place) need_docker; cmd="$1"; shift; in_world python3 /sim/sim_score.py "$cmd" "$@" ;;
    boxes) need_docker
        if [ "${2:-none}" = none ]; then in_world python3 /sim/sim_score.py boxes none
        else in_world python3 /sim/sim_score.py boxes "$(inside "$2")"; fi ;;
    scenario) need_docker; file="$(inside "${2:?scenario FILE}")"; shift 2
        in_world python3 /sim/sim_score.py scenario "$file" "$@" ;;
    stuck) need_docker; shift; in_world python3 /sim/sim_score.py scenario /sim/scenarios/shelf_pocket.yaml "$@" ;;
    cancel) need_docker; in_world python3 -m pepin.goal_link --host 127.0.0.1 cancel ;;
    logs) need_docker; exec docker logs -f "pepin-sim-${2:-nav}" ;;
    map) cd "$REPO" && uv run python ros/sim/export_world.py ;;
    *) sed -n '2,26p' "$0"; exit 2 ;;
esac
