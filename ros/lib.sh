#!/bin/bash
# Shared by the ros/*.sh scripts: one multiplexed ssh connection to the board.
# The first handshake costs 2.5-3.5 s here (sshd + PAM on a loaded A53, measured 2026-09-06);
# every later command over the same master costs ~0.1 s. Source after BOARD is set;
# `ssh`, `scp` and rsync then reuse the master for 10 minutes of idle.
BOARD="${BOARD:-${PEPIN_HOST:-10.0.0.187}}"
PEPIN_SSH_OPTS="-o ControlMaster=auto -o ControlPath=/tmp/pepin-ssh-%C -o ControlPersist=10m -o ConnectTimeout=6"
ssh() { command ssh $PEPIN_SSH_OPTS "$@"; }
scp() { command scp $PEPIN_SSH_OPTS "$@"; }
export RSYNC_RSH="ssh $PEPIN_SSH_OPTS"

# The one way to stop a container here; no ros/*.sh calls `docker stop`, `docker kill` or
# `docker rm -f` on its own. `docker stop` sends the container's STOPSIGNAL, which is SIGINT for
# everything built from ros/Dockerfile (and passed again as --stop-signal by the docker run
# lines, so a container started from another image is gentle too): SIGINT is the signal ros2
# launch answers by shutting its nodes down, while SIGTERM it answers by cancelling itself and
# the nodes are SIGKILLed mid-write. That difference is how ros/maps/rtabmap.db was made
# malformed (2026-09-13, eight kills of a crash loop; a torn page read on 2026-09-15).
# The window is pepin.deployment.CONTAINER_STOP_TIMEOUT_S seconds — RTAB-Map closing 20-28 GB of
# visual memory is the slowest thing in it — and docker SIGKILLs at its end, which is why the
# database is also configured to survive a kill (DbSqlite3/JournalMode in vslam.launch.py).
PEPIN_STOP_TIMEOUT_S="${PEPIN_STOP_TIMEOUT_S:-30}"  # = pepin.deployment.CONTAINER_STOP_TIMEOUT_S

# Which middleware the stack speaks. zenoh (the default since 2026-09-20) is rmw_zenoh_cpp with
# one rmw_zenohd router per machine and no bridge at all; cyclone is CycloneDDS with the two
# zenoh-bridge-ros2dds sidecars that carry the graph across the Mac's NAT — the stack that ran
# until then, kept whole and reachable with PEPIN_RMW=cyclone. Nothing but this variable changes
# between them. The board reads it from /etc/default/pepin-ros (board/pepin-ros.service's
# EnvironmentFile), so it survives a reboot; the laptop reads it from the environment.
#   Why the default moved, measured on the robot on 2026-09-20: start order stopped mattering and
# a restart of either half, of a single node or of a router heals by itself (under the bridges a
# restarted tracker left Nav2 deaf to map -> odom and a restarted bridge took /tf from the laptop);
# map -> odom reaches the laptop at 20 Hz instead of the bridge's capped 7.7 Hz; the board's CPU is
# the same parked (idle 41.6 % against 42.2 %) and under a drive-like load (13.5 % against 13.9 %);
# ten legs in three sensor modes reached with no transport error. The price is about 200 MB of
# board memory (roughly 40 MB a process).
#
# The topology under zenoh, and why it is this one: rmw_zenoh's SHIPPED session default is
# mode "peer", connect tcp/localhost:7447, listen tcp/localhost:0 — "accept incoming
# connections only from localhost, all communications with other hosts are routed by the Zenoh
# router". On the board (one host network namespace) that makes every node a peer of every
# other over the loopback, so scan, tf, odometry and the costmaps never touch the router
# process: measured 0.0 % router CPU for a 10 Hz x 100 kB flow, against 24 % when the same two
# nodes are clients. So the board needs NO session config at all. The Mac's containers do not
# share a loopback, so its nodes are told to gossip-connect to routers only and reach each
# other through the Mac's own router — which never sends that traffic over the WiFi.
PEPIN_RMW="${PEPIN_RMW:-zenoh}"

# WHO OWNS map -> odom, and it travels exactly as PEPIN_RMW does: this default for the shell,
# `-e PEPIN_LOCALIZER=` into both containers (ros/run.sh, ros/laptop.sh), and
# /etc/default/pepin-ros on the board (board/pepin-ros.service's EnvironmentFile) so it survives
# a reboot. src/pepin/deployment.py's `localizer` is the same question asked from Python, and
# the launches ask it there.
#   rtabmap (the default since 2026-09-22): RTAB-Map on the laptop publishes the transform
# itself and the board's lidar tracker does not start. tracker: the stack that ran until then,
# byte for byte — the relocalizer owns the edge, fuses the laptop's words into it and
# republishes the grid it adopted as /map_tracked for the costmaps.
#   WHY the default moved: two owners of the truth is a race, not a redundancy. The tracker
# trusted its own whole-map search on a fragment grid (fit 0.96 on the wrong place), collapsed
# its sigma and gated RTAB-Map's correct words out; the pose jumped 3.4 m (journal 2026-09-21/22).
# What stays on the board is ODOMETRY — wheels, gyro, visual odometry, laser odometry — which is
# the part that must survive a WiFi loss and close a loop in milliseconds; map -> odom is a slow
# correction every consumer composes with odom -> base_link.
PEPIN_LOCALIZER="${PEPIN_LOCALIZER:-rtabmap}"
pepin_localizer_is_tracker() { [ "$PEPIN_LOCALIZER" = tracker ]; }
# The one pairing that cannot work, said out loud before a half is started rather than debugged
# on the robot: under cyclone the two zenoh-bridge-ros2dds sidecars carry /tf one way only
# (board -> laptop), because a topic allowed as a publisher on BOTH sides is looped back by each
# bridge until nothing crosses at all (scan and tf died that way on 2026-09-09). RTAB-Map's
# map -> odom would have to come back the other way. Under zenoh every topic crosses and the
# pairing is free. Returns 1 so a caller can refuse; the way to give the graph the frame under
# cyclone is the retired message path (ros/nav.launch.py slam:=true, pepin_bringup.slam_frame).
pepin_localizer_check() {
    case "$PEPIN_LOCALIZER" in
        rtabmap | tracker) ;;
        *) echo "PEPIN_LOCALIZER=$PEPIN_LOCALIZER: it is rtabmap or tracker"; return 1 ;;
    esac
    if [ "$PEPIN_LOCALIZER" = rtabmap ] && ! pepin_rmw_is_zenoh; then
        echo "PEPIN_LOCALIZER=rtabmap needs PEPIN_RMW=zenoh: the cyclone bridges carry /tf one way"
        echo "  only, so RTAB-Map's map -> odom cannot reach the board. Use PEPIN_LOCALIZER=tracker,"
        echo "  or the retired message path (ros/nav.launch.py slam:=true)."
        return 1
    fi
    return 0
}
PEPIN_ZROUTER_PORT="${PEPIN_ZROUTER_PORT:-7447}"
PEPIN_ZROUTER_BOARD=pepin-zrouter          # the board's router container (host network)
PEPIN_ZROUTER_LAPTOP=pepin-zrouter-laptop  # the laptop's router container (on pepin-net)
# What the two routers log, as their RUST_LOG (rmw_zenohd reads it; unset, it logs at info). On
# 2026-09-23 the router-to-router session never came back after a Mac wake and BOTH routers said
# nothing for 28 minutes: zenoh logs the laptop's re-dials ("Try to connect", "Unable to connect
# to configured peer ... Retry in"), the handshakes the board refuses, a transport opened or closed
# and a link that expired only at DEBUG. So the transport lifecycle and the connector are at
# debug, and everything else stays at info — the per-message modules (rx, tx, the routing) are
# not in the list, so the volume is a few lines per session event, measured in
# scratch/link_autopsy/wedge_repro.py. PEPIN_ZROUTER_LOG=info is exactly the old level. The
# board's unit carries the same string (a test keeps the two equal); /etc/default/pepin-ros
# overrides it there.
PEPIN_ZROUTER_LOG="${PEPIN_ZROUTER_LOG:-info,zenoh::net::runtime::orchestrator=debug,zenoh_transport::unicast::manager=debug,zenoh_transport::unicast::establishment=debug,zenoh_transport::unicast::universal::link=debug,zenoh_transport::unicast::universal::transport=debug}"
# How many threads each router's zenoh RX runtime has (ZENOH_RUNTIME; zenoh's default is 2, and
# PEPIN_ZROUTER_RX_WORKERS=2 is exactly the old behaviour). WHY, measured in statics on 2026-09-23
# (scratch/link_autopsy/wedge_repro.py, two routers and four RELIABLE publishers, the far router
# frozen as the Mac's sleep froze the laptop): every session's RX task routes its messages on this
# runtime, and a push to a peer that stopped reading waits wait_before_close (20 s here) ON the
# worker; the "Closing transport!" it then schedules runs on the SAME runtime. With 2 workers
# both sat in 20-s pushes for 10.7 min — 33 closures, the close never started, the publishers'
# own sessions dropped 24 times — which is the board router's 2026-09-23 log line for line
# (x23-27 closures a sleep, its nodes' sessions timing out on it). With 8 the close started 9 ms
# after the first closure, the dead transport was gone 35 s later and no local session dropped.
# 16 because a router here serves about ten sessions (the board: 9 on its loopback, the laptop:
# the vslam container's ~15 nodes) and each one can hold a worker while the far side is frozen.
# Idle workers park: no CPU, a thread stack each (rule 20; measured in the same script).
PEPIN_ZROUTER_RX_WORKERS="${PEPIN_ZROUTER_RX_WORKERS:-16}"
pepin_rmw_is_zenoh() { [ "$PEPIN_RMW" = zenoh ]; }
# This checkout's ros/, wherever the sourcing script lives (the container configs are under it).
PEPIN_ROS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ONE CLOCK FOR THE ROBOT (src/pepin/timesync.py has the whole story): where the board takes its
# time from. laptop: this laptop serves its Docker VM's clock — the one every ROS node here stamps
# with — from the pepin-chrony container on udp/123, and the board's chrony prefers it over the
# internet pool (board/chrony.sh). pool: no server here, the board on the pool alone — what it has
# under systemd-timesyncd, so it is the DEFAULT until the chrony deploy has been measured on the
# robot (CLAUDE.md rule 19; ros/README.md "One clock" flips it). Here the variable only decides
# whether ros/laptop.sh (re)starts the server: once `ros/time.sh install` has started it, it
# stays up (--restart unless-stopped) whatever a later shell says. The board keeps its own copy
# in /etc/default/pepin-ros, written by `ros/time.sh source laptop|pool`; `ros/time.sh uninstall`
# puts the board back on systemd-timesyncd exactly as it ran before chrony.
PEPIN_TIME_SOURCE="${PEPIN_TIME_SOURCE:-pool}"
PEPIN_TIMESERVER=pepin-chrony
PEPIN_TIMESERVER_IMAGE=pepin-chrony:latest
pepin_laptop_ip() {  # <board ip> -> this Mac's address on the interface that reaches the board
    # What the board must dial for this laptop's time server: the route to the board names the
    # interface, the interface its address. Nothing when the board is not routable from here.
    local iface
    iface="$(route -n get "$1" 2>/dev/null | awk '/interface:/ { print $2 }')"
    [ -n "$iface" ] && ipconfig getifaddr "$iface"
}
# The laptop's time server, started idempotently beside the zenoh router (ros/laptop.sh) and left
# alone afterwards, like the router: containers come and go, the board's clock should not change
# source every time they do. Its image (ros/chrony/Dockerfile, alpine + chrony) is built on first
# use. It never gates anything: a server that cannot start leaves the board on the pool, and the
# restart check (1.15) says so.
pepin_timeserver_up() {
    [ "$PEPIN_TIME_SOURCE" = laptop ] || return 0
    if docker ps --format '{{.Names}}' | grep -qx "$PEPIN_TIMESERVER"; then
        echo "laptop time server already up ($PEPIN_TIMESERVER); left alone"; return 0
    fi
    if ! docker image inspect "$PEPIN_TIMESERVER_IMAGE" >/dev/null 2>&1; then
        docker build -q -t "$PEPIN_TIMESERVER_IMAGE" "$PEPIN_ROS_DIR/chrony" >/dev/null || {
            echo "laptop time server: the image did not build (ros/chrony); the board stays on the pool"
            return 0
        }
    fi
    pepin_remove_container "$PEPIN_TIMESERVER"
    # No --cap-add SYS_TIME and chronyd -x: it serves the VM's clock and never sets it
    # (ros/chrony/laptop.conf says why). udp/123 on every interface of the Mac: the board dials it.
    docker run -d --name "$PEPIN_TIMESERVER" --restart unless-stopped -p 123:123/udp \
        -v "$PEPIN_ROS_DIR/chrony/laptop.conf:/etc/chrony/chrony.conf:ro" \
        "$PEPIN_TIMESERVER_IMAGE" >/dev/null || {
        echo "laptop time server: did not start (is udp/123 taken on this Mac?); the board stays on the pool"
        return 0
    }
    echo "laptop time server up: $PEPIN_TIMESERVER serves this Docker VM's clock on udp/123"
}
# What a node on the laptop is told: reach the laptop's router by container name, and listen on
# the container's own address rather than on its loopback.
#
# The listen line is not cosmetic, it is the whole reason this side needs a config at all. The
# shipped default is listen tcp/localhost:0, which assumes every peer shares one loopback — true
# on the board (one host network namespace), false here, where pepin-laptop and pepin-vslam are
# separate namespaces on pepin-net. With the default the two learn each other through gossip,
# each dials the other's advertised tcp/127.0.0.1:<port> into its OWN loopback, and NOTHING is
# delivered: measured 0 of ~220 messages at both 100 kB and 1 MB. And a router does not rescue
# them — zenoh peers route peer-to-peer, so two peers that cannot reach each other cannot talk
# even while both are attached to the same router (0 delivered with gossip autoconnect forced to
# routers only). Listening on 0.0.0.0:0 makes the advertised locator the container's address on
# pepin-net, which its neighbour can actually reach: 212 of ~220 delivered. The board never
# learns these locators, because gossip multihop is off by default and the two routers are one
# hop apart.
#
# The fallback, if a future Docker network ever makes container-to-container direct links
# impossible: mode="client" with the same connect endpoint (214 of ~220 delivered, measured),
# which routes everything on this machine through this machine's router. Never a client of the
# BOARD's router — that would send this laptop's camera and depth traffic over the WiFi twice.
pepin_zenoh_session_override() {
    printf 'connect/endpoints=["tcp/%s:%s"];listen/endpoints=["tcp/0.0.0.0:0"]' \
        "$PEPIN_ZROUTER_LAPTOP" "$PEPIN_ZROUTER_PORT"
}
# What the laptop's router is told: listen where the shipped router config already listens
# (tcp/[::]:7447) and dial OUT to the board's router. Router-to-router is the only link that
# crosses the WiFi, and it is made from this side because the board cannot reach into the
# Docker VM's NAT.
pepin_zenoh_router_override() {  # <board ip>
    printf 'connect/endpoints=["tcp/%s:%s"]' "$1" "$PEPIN_ZROUTER_PORT"
}
# WHO WRITES A DRIVE DOWN on the board (board/pepin-ros.service's PEPIN_RECORDER, flipped with
# `ros/feature.sh recorder jsonl|bag`): jsonl is pepin_bringup.run_recorder, the numbered JSONL
# tape written on the board itself (34-43 % of a core: rclpy deserialisation, the TF buffer and
# json.dumps of 450 floats ten times a second); bag is pepin_bringup.bag_recorder, which only
# starts and stops `ros2 bag record` (MCAP, no compression) and subscribes to nothing, so the
# board copies serialised bytes and the laptop makes the tape afterwards. The scripts here do not
# have to be told which one runs — the board names the file it opened, and a bag is a DIRECTORY
# without the .jsonl suffix — so this is only the default for anything that must ask beforehand.
PEPIN_RECORDER="${PEPIN_RECORDER:-jsonl}"
PEPIN_VSLAM_CONTAINER="${PEPIN_VSLAM_CONTAINER:-pepin-vslam}"

# One run's bag turned into the tape every analysis script reads (ros/tools/bag_to_tape.py). It
# runs in the laptop's ROS container, because rosbag2_py and rclpy's deserialisation live there
# and never on the Mac itself; with that container down, the command to run later is printed
# instead of a drive's cleanup failing.
pepin_bag_to_tape() {  # <bag directory under ros/maps/rec>
    local name convert
    name="$(basename "$1")"
    convert="/pepin_entrypoint.sh python3 /tools/bag_to_tape.py /maps/rec/$name --force"
    if ! docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$PEPIN_VSLAM_CONTAINER"; then
        echo "!! $PEPIN_VSLAM_CONTAINER is down, so the bag is not converted yet. Later:"
        echo "   docker exec $PEPIN_VSLAM_CONTAINER $convert"
        return 1
    fi
    docker exec "$PEPIN_VSLAM_CONTAINER" $convert
}

pepin_stop_container() {  # NAME...: stop gently, leave the stopped container (a unit keeps its log)
    [ "$#" -gt 0 ] || return 0
    docker stop -t "$PEPIN_STOP_TIMEOUT_S" "$@" >/dev/null 2>&1 || true
}
# A container's log dies with the container: `docker rm` took the laptop router's and
# pepin-vslam's with every `ros/laptop.sh stop` and every vslam restart, and the 2026-09-23
# wake autopsy had only what someone happened to save before the restart. So every removal here
# first copies the whole log, with docker's own receive stamps (-t: the VM clock, whose steps
# after a Mac wake are themselves evidence), to PEPIN_LOG_DIR as <UTC stamp>_<name>.log — the
# board keeps pepin-ros's and pepin-zrouter's the same way under /root/pepin-ros/logs.
# PEPIN_LOG_ARCHIVE=off is the old behaviour (nothing kept). logs/ is gitignored.
PEPIN_LOG_ARCHIVE="${PEPIN_LOG_ARCHIVE:-on}"
PEPIN_LOG_DIR="${PEPIN_LOG_DIR:-$(cd "$PEPIN_ROS_DIR/.." && pwd)/logs/containers}"
pepin_archive_log() {  # NAME...: each existing container's whole log to PEPIN_LOG_DIR; never fails
    [ "$PEPIN_LOG_ARCHIVE" != off ] || return 0
    local name file
    for name in "$@"; do
        docker container inspect "$name" >/dev/null 2>&1 || continue
        mkdir -p "$PEPIN_LOG_DIR" 2>/dev/null || return 0
        file="$PEPIN_LOG_DIR/$(date -u +%Y%m%dT%H%M%SZ)_$name.log"
        docker logs -t "$name" >"$file" 2>&1 || echo "log of $name not archived ($file)" >&2
    done
    return 0
}
pepin_remove_container() {  # NAME...: stop gently, keep the log, then remove — about to be replaced
    pepin_stop_container "$@"
    pepin_archive_log "$@"
    docker rm -f "$@" >/dev/null 2>&1 || true
}

# What the robot is thinking, printed inline while a script drives it: goals, planner and
# controller verdicts, recoveries (spin/backup/wait), AMCL and relocalizer lines, local time.
# watch_start once before driving, watch_stop at the end (ros/watch.sh is the same view alone).
PEPIN_WATCH_KEEP='bt_navigator|behavior_server|controller_server|planner_server|amcl\]|relocalizer\]|velocity_smoother'
PEPIN_WATCH_DROP='Passing new path|foxglove|Message Filter|bond|Load Library|Found class|Instantiate class|launch_ros|lifecycle node launched|: Creating|: Configuring|: Activating|: Cleaning|Original Node|Setting|\]: $'
pepin_render() {  # "[proc] [LEVEL] [epoch] [node]: text" -> "HH:MM:SS LEVEL node: text"; repeats folded; \r\n so a pty next door cannot stair-step it
    perl -MPOSIX=strftime -ne '
        BEGIN { $| = 1; $last = ""; $n = 0 }
        chomp;
        next if /\[ActionServer\]|begin computing control effort|Received a goal|Client requested to cancel|Cancellation was successful/;
        s/RegulatedPurePursuitController/RPP/; s/GridBased plugin //; s/\[nav2_behaviors\] //;
        if (s/^\[[^\]]*\] \[(INFO|WARN|ERROR|DEBUG)\] \[(\d{10})\.\d+\] \[([^\]]+)\]: /sprintf("%s %-5s %s: ", strftime("%H:%M:%S", localtime($2)), $1, $3)/e) {
            ($msg = $_) =~ s/^\S+ //;
            if ($msg eq $last) { $n++; next }
            print "\r         ... x$n more\r\n" if $n;
            $n = 0; $last = $msg;
        }
        print "\r$_\r\n";
    '
}
watch_start() {
    # A viewer is for a person at a terminal. Run with its output piped (an agent, a script),
    # it has no reader, and every way of stopping it has left some member alive holding the
    # pipe open, so the caller never returned (2026-09-13, three times): no terminal, no viewer.
    [ -t 1 ] || return 0
    # The viewer runs in a process group of its own, so the stop reaches every member — the
    # ssh streaming the log included — with one signal. Killing the subshell alone, or pkill
    # by command line, left the ssh alive; it held the caller's stdout open, and a script whose
    # goto output was piped never returned (2026-09-13: the cart stood at the printer for nine
    # minutes while goto.sh waited on its own viewer). python's setpgrp, not bash's `set -m`:
    # job control needs a terminal, and a script run with its stdin on /dev/null got no group.
    export -f pepin_render
    python3 -c 'import os, sys; os.setpgrp(); os.execvp(sys.argv[1], sys.argv[1:])' bash -c \
        'ssh "root@$0" "docker logs -f --since 3s pepin-ros 2>&1" | grep --line-buffered -E "$1" | grep --line-buffered -vE "$2" | pepin_render' \
        "$BOARD" "$PEPIN_WATCH_KEEP" "$PEPIN_WATCH_DROP" 2>/dev/null &
    PEPIN_WATCH_PID=$!
    disown "$PEPIN_WATCH_PID" 2>/dev/null || true
}
watch_stop() {  # never blocks: a stuck viewer must not delay the stop that follows it
    [ -n "${PEPIN_WATCH_PID:-}" ] || return 0
    kill -TERM -- "-$PEPIN_WATCH_PID" 2>/dev/null || kill "$PEPIN_WATCH_PID" 2>/dev/null
    PEPIN_WATCH_PID=""
}
