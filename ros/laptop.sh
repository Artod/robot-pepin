#!/bin/bash
# The robot's brain on this Mac; the board is a sensor box. Usage:
#   ros/laptop.sh nav        start (or restart) Nav2 in pepin-macnav: planner, controller, costmaps,
#                            behaviour tree, the goal server on 127.0.0.1:3337 (controller mppi),
#                            the run recorder, the gaze arbiter (its door on 127.0.0.1:3339).
#                            PEPIN_MAP names the map whose places book the goal server reads
#                            (map_server stays off: both costmaps read RTAB-Map's /map). Runs on
#                            pepin-laptop:gaze when it is built on the laptop image
#                            (ros/laptop-build.sh gaze: the stall look's BT node), else on that
#                            image, with the tree's stall look cut out
#   ros/laptop.sh nav down   stop it; ros/laptop.sh nav logs follows its output
#   ros/laptop.sh stop       stop every container here, the router with them
#   ros/laptop.sh logs [vslam|macnav]   follow a container's output (macnav by default)
#   ros/laptop.sh vslam      start (or restart) the camera mapping container beside them: RTAB-Map
#                            on its one database (ros/maps/rtabmap.db), whose loop-closed grid is
#                            THE map — published on /map for Nav2's costmaps. The database is
#                            kept across restarts and is only ever deleted by --fresh
#   ros/laptop.sh vslam --camera-only   no lidar in the snapshots: the grid is the camera's depth
#   ros/laptop.sh vslam --fresh   build the room from nothing: the database is deleted and no volume
#                            snapshot is resumed, whatever is on disk. A fresh database is a NEW
#                            frame, so ros/restart.sh --fresh-graph moves the old frame's volume
#                            aside with it
#   ros/laptop.sh vslam --no-vo   no visual odometry: rgbd_odometry and pepin_bringup.visual_odometry
#                            do not start, and the board's EKF is the wheels and the gyro alone
#   ros/laptop.sh vslam --vo-depth   the visual odometry reads the picture and the depth (rgbd_odometry)
#                            at the depth's rate; the default reads the two eyes (stereo_odometry) at
#                            the camera's: 8 poses/s, 0.15-0.2 s behind, measured at rest 2026-10-02
#   ros/laptop.sh vslam --vo-vio   the visual odometry relay reads OpenVINS (ros/laptop.sh vio, its own
#                            container) instead of rtabmap's: no odometry node runs in pepin-vslam
#   ros/laptop.sh vio        start (or restart) OpenVINS in pepin-vio on pepin-laptop:vio
#                            (ros/laptop-build.sh vio): the head IMU (/head/imu) and the two eyes,
#                            config written into ros/maps/vio by ros/tools/vio_config.py inside
#                            the image at every start (PEPIN_VIO_CONFIG_ARGS: its options);
#                            vio down | logs | kick (kick only at rest: OpenVINS inits from stillness)
#   ros/laptop.sh vslam --fixed-head   the camera node here broadcasts base_link -> camera_link from
#                            config/camera.json: for a rig without neck servos. By default the board's
#                            base bridge owns that edge (from the neck's encoders); --neck, the old way
#                            to say the default, is still accepted
#   ros/laptop.sh kick NODE  restart one node from the mounted sources (seconds, no container restart)
#   ros/laptop.sh vslam      runs the camera mapping container on pepin-laptop:xfeat whenever that
#                            image exists and was built on the laptop image below
#                            (ros/laptop-build.sh xfeat): RTAB-Map built with Python, so
#                            rtabmap_frame's visual_features default (xfeat) can run XFeat +
#                            LighterGlue in the visual registration. Without it the usual image
#                            runs, that flag falls back to ORB, and this says so on stderr.
#                            PEPIN_XFEAT=0 is the rollback (the usual image, ORB only);
#                            PEPIN_XFEAT=1 refuses to start without an up-to-date xfeat image
#   PEPIN_CAMERA=overview ros/laptop.sh vslam   run the OTHER camera rig for one container:
#                            config/camera.json's "active" is the standing answer (see the
#                            "Camera rigs" section of ros/README.md), this overrides it
#   ros/laptop.sh vslam      also starts the depth network on the laptop's GPU (ros/depth_host.sh)
#                            and tells the node to use it, when torch's Metal backend is there;
#                            PEPIN_DEPTH_HOST=0 keeps the network on the CPU in the container,
#                            PEPIN_DEPTH_HOST=1 insists on the service (it falls back to the CPU)
#   ros/laptop.sh vslam      starts the localisation models when they are installed
#                            (ros/models.sh: XFeat, LighterGlue, place descriptors, a launchd job
#                            loaded on demand), tells the container where they are
#                            (PEPIN_MODELS_URL), mounts the checkout's two RTAB-Map adapters over
#                            the xfeat image's (PEPIN_ADAPTERS_MOUNT=0: the image's) and passes
#                            PEPIN_GLOBAL_DESCRIPTOR / PEPIN_REGISTRATION_BACKEND through when set;
#                            stop stops the models again
# Nothing here talks to the board. Prerequisites: the image built here (ros/laptop-build.sh) and
# the board's sensors up with no Nav2 of its own (two Nav2 in one graph share every node name).
# A Docker container on macOS lives behind the VM's NAT; this machine's zenoh router holds the one
# TCP link to the board's router, and the graph appears here whole.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BOARD="${PEPIN_HOST:-10.0.0.187}"
. "$HERE/lib.sh"  # the router, container and clock helpers shared with the other scripts
NET=pepin-net
NAV=pepin-macnav  # = pepin.deployment.NAV_CONTAINER
# The library is mounted live, like the ROS package: a copy went stale whenever a container was
# restarted by anything but this script (the depth node died on an import of a function that
# existed in src/ but not in the copy, 2026-09-11).
# A container's nodes get the time to leave DDS properly (RTAB-Map closes its database): the
# containers run with --stop-signal SIGINT, the signal the launch answers by shutting its nodes
# down (SIGTERM it answers by cancelling itself, and the nodes were SIGKILLed mid-write). The
# window and the signal are ros/lib.sh's, the same ones board/pepin-ros.service and the launches use; the 15 s
# this used to spend was under RTAB-Map's own close of a 20 GB database.
# The laptop image (ros/laptop-build.sh) carries RTAB-Map and rmw_zenoh_cpp on top of the
# board's image: PEPIN_IMAGE overrides it.
image() {
    if [ -n "${PEPIN_IMAGE:-}" ]; then echo "$PEPIN_IMAGE"; return; fi
    echo pepin-laptop:zenoh
}
# The camera mapping container's image. pepin-laptop:xfeat BY DEFAULT — rtabmap_frame's
# visual_features defaults to xfeat, which only that image can run, and a restart that forgot an
# environment variable used to bring RTAB-Map up on ORB with nothing but a clause in a report line
# to say so (ORB accepted 0 of 630 camera-only updates on 2026-09-23). Only an xfeat image BUILT ON
# the image above counts: an image built from another carries its layers first, so a laptop image
# rebuilt since is not silently traded for an older one. PEPIN_XFEAT=0 is the rollback (the image
# above, ORB only); PEPIN_XFEAT=1 refuses to start without an up-to-date xfeat image; PEPIN_IMAGE
# (an explicit image) wins over both. ros/restart.sh's check 2.12 reads which image came up.
vslam_image() {
    local base why=""
    base="$(image)"
    if [ -n "${PEPIN_IMAGE:-}" ] || [ "${PEPIN_XFEAT:-}" = 0 ]; then echo "$base"; return; fi
    if [ -n "${PEPIN_XFEAT:-}" ] && [ "$PEPIN_XFEAT" != 1 ]; then
        echo "PEPIN_XFEAT=$PEPIN_XFEAT: 0 (the usual image), 1 (xfeat or nothing) or unset" >&2
        return 1
    fi
    if ! docker image inspect pepin-laptop:xfeat >/dev/null 2>&1; then
        why="there is no pepin-laptop:xfeat image"
    elif ! xfeat_built_on "$base"; then
        why="pepin-laptop:xfeat was not built on $base (rebuilt since?)"
    fi
    if [ -z "$why" ]; then echo pepin-laptop:xfeat; return; fi
    if [ "${PEPIN_XFEAT:-}" = 1 ]; then
        echo "PEPIN_XFEAT=1 but $why: ros/laptop-build.sh xfeat" >&2
        return 1
    fi
    echo "laptop: $why — vslam runs $base and visual_features falls back to orb" \
        "(ros/laptop-build.sh xfeat, about an hour; PEPIN_XFEAT=0 says this is meant)" >&2
    echo "$base"
}
xfeat_built_on() {  # BASE: whether pepin-laptop:xfeat's layers begin with BASE's (built FROM it)
    built_on pepin-laptop:xfeat "$1"
}
built_on() {  # IMAGE BASE: whether IMAGE's layers begin with BASE's (it was built FROM it)
    local base_layers layers
    base_layers="$(docker image inspect -f '{{join .RootFS.Layers " "}}' "$2" 2>/dev/null)" || return 1
    layers="$(docker image inspect -f '{{join .RootFS.Layers " "}}' "$1" 2>/dev/null)" || return 1
    [ -n "$base_layers" ] && [ "${layers#"$base_layers"}" != "$layers" ]
}
# The navigation container's image: pepin-laptop:gaze — the image above plus the stall look's BT
# node (ros/pepin_gaze_bt, ros/laptop-build.sh gaze, under a minute) — whenever it was built on
# the image above, so a laptop image rebuilt since is never traded for an older one; otherwise
# the image above, whose tree then runs without the stall look (nav.launch.py says so in the
# log). PEPIN_IMAGE wins.
nav_image() {
    local base
    base="$(image)"
    if [ -z "${PEPIN_IMAGE:-}" ] && built_on pepin-laptop:gaze "$base"; then
        echo pepin-laptop:gaze; return
    fi
    echo "$base"
}
# The middleware flags every node container here is given: the session (a peer of THIS
# machine's router) plus ZENOH_ROUTER_CHECK_ATTEMPTS=0, so a container started before the router
# survives and joins when it appears instead of dying on the start order.
# The session's tx queues raised to 16 batches each (rmw_zenoh's default is smaller): with the
# default the Foxglove bridge in pepin-vslam hit "Unable to push non droppable network message"
# within minutes of every start and hung silently (2026-09-30, 2026-10-01), as Nav2 did before
# the Mac's Nav2 got the same override. 32 panics zenoh (`*num <= RBLEN`): 16 is the ceiling.
ZENOH_QUEUE="${PEPIN_ZENOH_QUEUE:-16}"
ZENOH_QUEUES=""
for prio in control real_time interactive_high interactive_low data_high data data_low background; do
    ZENOH_QUEUES="$ZENOH_QUEUES;transport/link/tx/queue/size/$prio=$ZENOH_QUEUE"
done
RMW_ENV=(-e RMW_IMPLEMENTATION=rmw_zenoh_cpp -e PEPIN_RMW=zenoh -e ZENOH_ROUTER_CHECK_ATTEMPTS=0
         -e "ZENOH_CONFIG_OVERRIDE=$(pepin_zenoh_session_override)$ZENOH_QUEUES")
# A clock the two halves disagree on is refused BEFORE a container starts, not debugged on the
# robot — and only where one is started, so `stop` and `logs` still work on a misconfigured shell.
start_check() { pepin_time_source_check || exit 1; }
# One zenoh router per machine, and this is the laptop's. It is started before any node here and
# left alone afterwards: a node's connect retry is infinite, so containers may come and go under
# it, and it is the only process on this side that talks to the board. Started idempotently —
# `ros/laptop.sh nav` and `ros/laptop.sh vslam` both need it and either may come first.
zrouter_up() {
    docker network inspect "$NET" >/dev/null 2>&1 || docker network create "$NET" >/dev/null
    if docker ps --format '{{.Names}}' | grep -qx "$PEPIN_ZROUTER_LAPTOP"; then
        echo "laptop zenoh router already up ($PEPIN_ZROUTER_LAPTOP); left alone"; return 0
    fi
    pepin_remove_container "$PEPIN_ZROUTER_LAPTOP"  # the one gentle way to stop a container here
    # The same config file the board's router gets (ros/zenoh/router.json5, one file for both:
    # the only thing that differs between the two routers is who dials whom, and that is the
    # ZENOH_CONFIG_OVERRIDE below, which rmw_zenoh applies ON TOP of the file). It buys the tx
    # queue enough patience to sit out a 1-2 s WiFi stall instead of closing the session.
    # PEPIN_ZROUTER_CONFIG=  (empty) starts this router on the shipped rmw_zenoh default.
    ZCONFIG=()
    if [ -n "${PEPIN_ZROUTER_CONFIG-$HERE/zenoh/router.json5}" ]; then
        ZCONFIG=(-v "${PEPIN_ZROUTER_CONFIG:-$HERE/zenoh/router.json5}:/zenoh/router.json5:ro"
                 -e ZENOH_ROUTER_CONFIG_URI=/zenoh/router.json5)
    fi
    # RUST_LOG: the transport lifecycle and the re-dials at debug (ros/lib.sh's PEPIN_ZROUTER_LOG).
    # ZENOH_RUNTIME: enough RX workers that a frozen board cannot starve this router's own
    # sessions and its close of the dead link (ros/lib.sh's PEPIN_ZROUTER_RX_WORKERS says why).
    docker run -d --name "$PEPIN_ZROUTER_LAPTOP" --network "$NET" --restart unless-stopped \
        -e RMW_IMPLEMENTATION=rmw_zenoh_cpp -e ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-7}" \
        -e "RUST_LOG=$PEPIN_ZROUTER_LOG" \
        -e "ZENOH_RUNTIME=(rx: (worker_threads: $PEPIN_ZROUTER_RX_WORKERS))" \
        -e "ZENOH_CONFIG_OVERRIDE=$(pepin_zenoh_router_override "$BOARD")" \
        ${ZCONFIG[@]+"${ZCONFIG[@]}"} \
        "$(image)" /opt/ros/jazzy/lib/rmw_zenoh_cpp/rmw_zenohd >/dev/null
    echo "laptop zenoh router up: $PEPIN_ZROUTER_LAPTOP, dialling tcp/$BOARD:$PEPIN_ZROUTER_PORT"
}
# The nodes a kick can reach here, the container each lives in and the line it prints once up
# (the kick waits for that line): the Python modules of vslam.launch.py and of nav.launch.py. Only
# one of the two recorders runs (nav.launch.py's recorder argument); a kick of the other one finds
# nothing and says so.
KICKABLE="camera_stream depth_stream contact_scan depth_fusion rtabmap_frame sensor_pack places marks_audit visual_odometry goal_server run_recorder bag_recorder gaze"
kick_target() {  # node name -> "container|start-up line"
    case "$1" in
        camera_stream) echo "pepin-vslam|camera stream from " ;;
        depth_stream) echo "pepin-vslam|depth stream up" ;;
        contact_scan) echo "pepin-vslam|contact scan up" ;;
        depth_fusion) echo "pepin-vslam|fusion up: " ;;
        rtabmap_frame) echo "pepin-vslam|rtabmap frame up: " ;;
        sensor_pack) echo "pepin-vslam|sensor pack up" ;;
        places) echo "pepin-vslam|places up: " ;;
        marks_audit) echo "pepin-vslam|marks audit up: " ;;
        visual_odometry) echo "pepin-vslam|visual odometry up: " ;;
        goal_server) echo "$NAV|goal server ready on port" ;;
        run_recorder) echo "$NAV|run recorder ready" ;;
        bag_recorder) echo "$NAV|bag recorder ready" ;;
        gaze) echo "$NAV|gaze up: " ;;
        *) return 1 ;;
    esac
}
# Whether the depth network runs on this laptop's GPU (ros/depth_host.sh) beside the container:
# PEPIN_DEPTH_HOST=1 or 0 decides outright; otherwise torch's Metal backend is asked (~2 s: the
# import), the same question the service itself answers before it falls back to the CPU.
depth_host_wanted() {
    case "${PEPIN_DEPTH_HOST:-}" in 1) return 0 ;; 0) return 1 ;; esac
    (cd "$HERE/.." && uv run --group depth python -c 'import torch; print(torch.backends.mps.is_available())' 2>/dev/null) | grep -qx True
}
MOUNTS=(-v "$HERE/pepin_bringup/pepin_bringup:/ws/install/pepin_bringup/lib/python3.12/site-packages/pepin_bringup:ro"
        -v "$HERE/pepin_bringup/launch:/ws/install/pepin_bringup/share/pepin_bringup/launch:ro"
        -v "$HERE/tools:/tools:ro" -v "$HERE/entrypoint.sh:/pepin_entrypoint.sh:ro"
        -v "$HERE/../src/pepin:/ws/pepin_src/pepin:ro" -v "$HERE/params:/params:ro" -v "$HERE/maps:/maps"
        -v "$HERE/../config:/ws/config:ro")
case "${1:-}" in
    nav)
        # NAV2 ON THIS MAC (decision of 2026-10-01): the launch the board ran, whole, beside this
        # machine's router; scans, ToF and odometry come from the board over WiFi and the velocity
        # goes back on /cmd_vel. The goal server's port is published on this Mac's loopback only
        # (a goal is a command to the wheels): ros/goto.sh, ros/preflight.sh, ros/stop.sh and the
        # tray speak to 127.0.0.1:3337.
        case "${2:-up}" in
            down) pepin_remove_container "$NAV"; echo "$NAV stopped"; exit 0 ;;
            logs) exec docker logs -f "$NAV" ;;
            up) ;;
            *) echo "usage: ros/laptop.sh nav [up|down|logs]"; exit 2 ;;
        esac
        start_check
        MAP="${PEPIN_MAP:-/maps/flat3_straight.yaml}"
        pepin_remove_container "$NAV"
        zrouter_up
        pepin_timeserver_up  # the clock the board follows (ros/lib.sh)
        # The gaze arbiter's door for the tools (pepin_bringup.gaze, 3339) goes on the loopback
        # beside the goal server's, and it reaches the board's base server at PEPIN_HOST.
        NAV_IMAGE="$(nav_image)"
        docker run -d --name "$NAV" --network "$NET" -p 127.0.0.1:3337:3337 -p 127.0.0.1:3339:3339 --restart unless-stopped --stop-signal SIGINT "${MOUNTS[@]}" \
            -e ROS_DOMAIN_ID=7 -e "PEPIN_HOST=$BOARD" "${RMW_ENV[@]}" \
            "$NAV_IMAGE" ros2 launch pepin_bringup nav.launch.py "map:=$MAP" "recorder:=$PEPIN_RECORDER" >/dev/null
        # The tree's transitions, one watcher per container, read by ros/goto.sh (which starts
        # it itself when it is missing).
        pepin_bt_watch || echo "bt watcher not started: ros/goto.sh starts it with the first goal"
        echo "nav up: Nav2 in $NAV on $NAV_IMAGE, map $MAP, goal server on 127.0.0.1:3337, gaze on 127.0.0.1:3339; ros/laptop.sh nav logs"
        exit 0 ;;
    stop)
        # The router is this side's own, so a stop takes it too. The time server (pepin-chrony) is left running: it is the board's clock, not a part of this
        # half, and a board that lost it would change source at every stop (ros/time.sh server
        # stop takes it down on purpose).
        pepin_remove_container "$NAV" pepin-vslam pepin-vio "$PEPIN_ZROUTER_LAPTOP"
        [ "${PEPIN_DEPTH_HOST:-}" = 0 ] || "$HERE/depth_host.sh" stop
        # The localisation models leave with the session, as the depth host does: ~1 GB of this
        # laptop's memory that nothing else uses (ros/models.sh start localization brings them).
        if "$HERE/models.sh" installed localization; then "$HERE/models.sh" stop localization; fi
        echo "laptop side stopped"; exit 0 ;;
    logs)
        exec docker logs -f "pepin-${2:-macnav}" ;;
    vio)
        # THE VISUAL-INERTIAL ODOMETRY (vio.md): OpenVINS in its own container, pepin-vio on
        # pepin-laptop:vio (ros/laptop-build.sh vio), behind this side's router like vslam; the
        # relay reads it under ros/laptop.sh vslam --vo-vio. Its config is generated into
        # ros/maps/vio by ros/tools/vio_config.py (below, at every start) and never edited.
        #   ros/laptop.sh vio [up]   start (or restart) it
        #   ros/laptop.sh vio down   stop it; vio logs follows it
        #   ros/laptop.sh vio kick   restart the node inside (the launch respawns it): AT REST
        #                            only, OpenVINS initialises from stillness and then motion
        case "${2:-up}" in
            down) pepin_remove_container pepin-vio; echo "pepin-vio stopped"; exit 0 ;;
            logs) exec docker logs -f pepin-vio ;;
            kick)
                docker exec pepin-vio sh -c 'pkill -INT -f run_subscribe_msckf' \
                    || { echo "no OpenVINS process in pepin-vio (ros/laptop.sh vio logs)"; exit 3; }
                echo "OpenVINS signalled; the launch respawns it in 2 s. It initialises at rest, then on the first motion (a head pan will do)"
                exit 0 ;;
            up) ;;
            *) echo "usage: ros/laptop.sh vio [up|down|logs|kick]"; exit 2 ;;
        esac
        start_check
        docker image inspect pepin-laptop:vio >/dev/null 2>&1 \
            || { echo "no pepin-laptop:vio here: ros/laptop-build.sh vio first (10-20 min)"; exit 2; }
        # The config is written from the repo's numbers at every start, INSIDE the image: the
        # rectified focal is OpenCV's stereoRectify's and moves with its version (494.22 px under
        # the image's 4.6, camera_stream's, against 495.08 under uv's 4.13, 2026-10-04).
        # PEPIN_VIO_CONFIG_ARGS passes vio_config.py's options (--calib-extrinsics for a check
        # session that lets OpenVINS refine the camera-IMU transform).
        # shellcheck disable=SC2086
        docker run --rm --network none "${MOUNTS[@]}" --entrypoint python3 pepin-laptop:vio \
            /tools/vio_config.py ${PEPIN_VIO_CONFIG_ARGS:-} | tail -1 \
            || { echo "ros/tools/vio_config.py failed in pepin-laptop:vio"; exit 2; }
        pepin_remove_container pepin-vio
        zrouter_up
        # The image's OpenVINS overlay (/ws_vio) is sourced after the entrypoint's workspace, so
        # the mounted pepin_bringup launch file finds ov_msckf.
        docker run -d --name pepin-vio --network "$NET" --restart unless-stopped --stop-signal SIGINT "${MOUNTS[@]}" \
            -e ROS_DOMAIN_ID=7 "${RMW_ENV[@]}" \
            pepin-laptop:vio bash -c 'source /ws_vio/install/setup.bash && exec ros2 launch pepin_bringup vio.launch.py' >/dev/null
        echo "vio up: OpenVINS in pepin-vio on $(docker run --rm --network none --entrypoint cat pepin-laptop:vio /opt/openvins/SHAS | head -2 | tr '\n' ' ')"
        echo "the relay reads it under ros/laptop.sh vslam --vo-vio; ros/laptop.sh vio logs"
        exit 0 ;;
    kick)
        # One node, not its container. SIGINT is what the launch itself sends at shutdown: the
        # node's main destroys the node and the context; the launch respawns the module from the
        # mounted sources two seconds after the EXIT (RESPAWN in the launch files). Measured
        # start-ups after the respawn pause: camera/fusion/frame 0.3 s, depth 2.1 s (the
        # network), so a kick is 3-5 s here.
        #   The wait proves the ready line is the NEW process's (ros/kick_ready.awk, the same
        # matcher as ros/board.sh kick): the exit line of the signalled pid, the successor's start
        # under the same launch tag, then its ready line. The times are the container's clock
        # (the VM's on this Mac, whose timestamps are the log's): its `date` before the SIGINT.
        NAME="${2:-}"; TARGET="$(kick_target "$NAME")" || { echo "usage: ros/laptop.sh kick <node>; nodes: $KICKABLE"; exit 2; }
        C="${TARGET%%|*}"; LINE="${TARGET#*|}"; TAB="$(printf '\t')"
        OUT="$(docker exec "$C" sh -c 'date -u +%FT%T.%NZ; pgrep -f "pepin_bringup[./]$1"' sh "$NAME" 2>/dev/null)" || true
        KICKED="${OUT%%$'\n'*}"; OLD="$(printf '%s\n' "$OUT" | sed 1d | tr '\n' ' ')"
        [ -n "${OLD// /}" ] || { echo "no $NAME process in $C (ros/laptop.sh logs ${C#pepin-})"; exit 3; }
        # shellcheck disable=SC2086
        docker exec "$C" sh -c 'kill -INT "$@"' sh $OLD
        R="wait${TAB}nothing read from the log yet"
        for _ in $(seq 1 240); do
            R="$(docker logs -t --since "$KICKED" "$C" 2>&1 | awk -v name="$NAME" -v old="$OLD" -v line="$LINE" -v kicked="$KICKED" -f "$HERE/kick_ready.awk")"
            if [ "${R%%"$TAB"*}" = ready ]; then
                echo "${R#*"$TAB"}"
                exit 0
            fi
            sleep 0.5
        done
        echo "$NAME not ready within 120 s: ${R#*"$TAB"} (ros/laptop.sh logs ${C#pepin-})"; exit 4 ;;
    vslam)
        # Camera + lidar mapping beside the navigation half (ros/pepin_bringup/launch/vslam.launch.py),
        # and there is one arrangement of it (World R): the database is the map, it is kept across
        # restarts (the launch never wipes it) and it is deleted only here, on request. Mapping a new
        # room and driving a known one are the same launch with a different file on disk — the launch
        # reads which it is and picks the memory mode itself — so this subcommand needs no mode, and
        # asks the board nothing.
        start_check
        CAMERA_ONLY=false; FRESH=false
        # The camera as a third odometry (rtabmap_odom's rgbd_odometry + pepin_bringup.visual_odometry):
        # on unless --no-vo. It costs this laptop a quarter of a core and the robot nothing at
        # all until the node's vo_publish flag is turned on (ros/flags.sh set visual_odometry
        # vo_publish true).
        VO=true
        VO_INPUT=stereo
        # The board's base bridge publishes base_link -> camera_link live from the neck's
        # encoders, so the camera node's static edge is off; --fixed-head puts it back for a rig
        # without a neck. A wrong choice is two publishers of one edge, or none; the camera node's
        # report warns of a mismatch.
        STATIC_CAMERA_TF=false
        for arg in ${*:2}; do
            case "$arg" in
                --camera-only) CAMERA_ONLY=true ;;
                --fresh) FRESH=true ;;
                --neck) STATIC_CAMERA_TF=false ;;
                --fixed-head) STATIC_CAMERA_TF=true ;;
                --no-vo) VO=false ;;
                --vo-depth) VO_INPUT=depth ;;
                --vo-vio) VO_INPUT=vio ;;
                *) echo "usage: ros/laptop.sh vslam [--fresh] [--camera-only] [--fixed-head] [--no-vo] [--vo-depth] [--vo-vio]"; exit 2 ;;
            esac
        done
        # --fresh: an empty room, which is one fact on disk — the database gone. The launch reads
        # that and starts RTAB-Map in mapping mode. The fused volume is the odometry's rolling
        # window and is born empty at every start whatever is on disk.
        if [ "$FRESH" = true ]; then
            rm -f "$HERE"/maps/rtabmap.db "$HERE"/maps/rtabmap.db-*
            echo "vslam --fresh: ros/maps/rtabmap.db deleted, so RTAB-Map starts an empty graph"
        fi
        VSLAM_IMAGE="$(vslam_image)" || exit 2
        pepin_remove_container pepin-vslam
        zrouter_up  # this half's own router
        pepin_timeserver_up  # the clock the board follows (ros/lib.sh; PEPIN_TIME_SOURCE=pool: none)
        # The depth network on the laptop's GPU (ros/depth_host.sh): 20 ms a frame on Metal
        # against 170 ms on the CPU in the container (2026-09-11), so it is on wherever it can
        # run (depth_host_wanted). The node reads PEPIN_DEPTH_BACKEND (auto: the service, the
        # CPU model while it does not answer; its depth_backend flag switches live) and
        # PEPIN_DEPTH_URL (the host as the container sees it); without them the node runs on
        # the CPU as before.
        # WHICH CAMERA the head is, when this shell says so: config/camera.json's "active" is the
        # standing answer and PEPIN_CAMERA overrides it for one container, so a rig is compared
        # without editing a file (PEPIN_CAMERA=overview ros/laptop.sh vslam). Unset by default:
        # the file decides, on both halves of the robot.
        CAMERA_ENV=()
        if [ -n "${PEPIN_CAMERA:-}" ]; then
            CAMERA_ENV=(-e "PEPIN_CAMERA=$PEPIN_CAMERA")
            echo "camera rig: $PEPIN_CAMERA (PEPIN_CAMERA overrides config/camera.json's active)"
        fi
        DEPTH_ENV=()
        if depth_host_wanted; then
            "$HERE/depth_host.sh" start  # an installed launchd job is started by ros/models.sh
            DEPTH_ENV=(-e PEPIN_DEPTH_BACKEND=auto -e "PEPIN_DEPTH_URL=http://host.docker.internal:${PEPIN_DEPTH_PORT:-8790}")
        else
            echo "depth network on the CPU in the container (PEPIN_DEPTH_HOST=1 for the GPU service)"
        fi
        # THE LOCALISATION MODELS on this laptop's GPU (ros/models.sh): RTAB-Map's XFeat / LighterGlue
        # adapters and sensor_pack's place descriptors call it at PEPIN_MODELS_URL. Started here
        # when installed (a launchd job loaded on demand, restarted by launchd if it dies, stopped
        # by `stop`) and waited for, so RTAB-Map's first registrations find it rather than loading
        # torch into its own process. Down or not installed: the adapters compute in RTAB-Map's
        # process (registration_backend auto) and the snapshots carry null descriptors.
        MODELS_ENV=(-e "PEPIN_MODELS_URL=http://host.docker.internal:${PEPIN_MODELS_PORT:-8791}")
        if "$HERE/models.sh" installed localization; then
            "$HERE/models.sh" start localization \
                || echo "localization service not answering (ros/models.sh logs localization): the adapters compute locally"
        else
            echo "localization service not installed (ros/models.sh install localization): the adapters compute locally"
        fi
        # Two flags read at start from the environment, passed through when this shell sets them:
        # sensor_pack's global_descriptor (auto|on|off; the launch reads it too) and rtabmap_frame's
        # registration_backend (service|local|auto; live afterwards through ros/flags.sh).
        FLAG_ENV=()
        for var in PEPIN_GLOBAL_DESCRIPTOR PEPIN_REGISTRATION_BACKEND; do
            if [ -n "${!var:-}" ]; then FLAG_ENV+=(-e "$var=${!var}"); echo "$var=${!var}"; fi
        done
        # THE ADAPTERS FROM THE CHECKOUT, not the image: the two files RTAB-Map loads by path are
        # mounted over the image's copies, so an adapter change needs a vslam restart and no
        # rebuild (RTAB-Map imports them once, so a restart is needed either way). The files and
        # not ros/xfeat as a whole: the directory would hide the image's XFeat checkout
        # (/opt/xfeat/accelerated_features), whose weights the local fallback loads. Only onto the
        # xfeat image — mounted into another, they would make rtabmap_frame believe RTAB-Map can
        # run Python. PEPIN_ADAPTERS_MOUNT=0 runs the adapters baked into the image.
        ADAPTER_MOUNTS=()
        if [ "$VSLAM_IMAGE" = pepin-laptop:xfeat ] && [ "${PEPIN_ADAPTERS_MOUNT:-1}" != 0 ]; then
            ADAPTER_MOUNTS=(-v "$HERE/xfeat/rtabmap_xfeat.py:/opt/xfeat/rtabmap_xfeat.py:ro"
                            -v "$HERE/xfeat/rtabmap_lighterglue.py:/opt/xfeat/rtabmap_lighterglue.py:ro")
        fi
        # The database's place-descriptor census is taken INSIDE the container by the launch, on
        # every start, of the file RTAB-Map is given (vslam.launch.py census_env): a census taken
        # here once would outlive a restart of the container after the file changed.
        docker run -d --name pepin-vslam --network "$NET" -p 8765:8765 --restart unless-stopped --stop-signal SIGINT "${MOUNTS[@]}" \
            ${ADAPTER_MOUNTS[@]+"${ADAPTER_MOUNTS[@]}"} "${MODELS_ENV[@]}" ${FLAG_ENV[@]+"${FLAG_ENV[@]}"} \
            -e ROS_DOMAIN_ID=7 "${RMW_ENV[@]}" ${DEPTH_ENV[@]+"${DEPTH_ENV[@]}"} ${CAMERA_ENV[@]+"${CAMERA_ENV[@]}"} \
            "$VSLAM_IMAGE" ros2 launch pepin_bringup vslam.launch.py "board:=$BOARD" "static_camera_tf:=$STATIC_CAMERA_TF" \
            "camera_only:=$CAMERA_ONLY" "vo:=$VO" "vo_input:=$VO_INPUT" >/dev/null
        echo "vslam up on $VSLAM_IMAGE (camera_only $CAMERA_ONLY, static camera tf $STATIC_CAMERA_TF): RTAB-Map's grid is /map and RTAB-Map here owns map -> odom; Foxglove ws://localhost:8765, ros/laptop.sh logs vslam"
        echo "foxglove: reconnect the app to $("$HERE/foxglove.sh" url) when you want it (nothing is opened for you)"
        exit 0 ;;
    *) echo "usage: ros/laptop.sh [nav [up|down|logs] | stop | logs [vslam|macnav|vio] | vslam [--fresh] [--camera-only] [--fixed-head] [--no-vo] [--vo-depth] [--vo-vio] | vio [up|down|logs|kick] | kick NODE]"; exit 2 ;;
esac
