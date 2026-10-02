#!/bin/bash
# Build the laptop image (pepin-laptop): the board's sensor image plus Nav2, RTAB-Map and the
# image pipeline.
# Usage: ros/laptop-build.sh        (a few minutes; needs pepin-ros:sensors from ros/build-image.sh)
#        ros/laptop-build.sh xfeat  pepin-laptop:xfeat on top of pepin-laptop:zenoh
#                                   (ros/Dockerfile.xfeat: RTAB-Map rebuilt with Python, XFeat and
#                                   LighterGlue for the visual registration; about an hour)
#        ros/laptop-build.sh gaze   pepin-laptop:gaze: pepin-laptop:zenoh plus the stall look's BT
#                                   node (Dockerfile.laptop's last stage alone; under a minute).
#                                   ros/laptop.sh nav runs it whenever it is built on that image
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
if [ "${1:-}" = gaze ]; then
    BASE="${PEPIN_GAZE_BASE:-pepin-laptop:zenoh}"
    docker build -f "$HERE/Dockerfile.laptop" -t pepin-laptop:gaze --build-arg "BASE=$BASE" "$HERE"
    echo "pepin-laptop:gaze built on $BASE"
    exit 0
fi
if [ "${1:-}" = xfeat ]; then
    BASE="${PEPIN_XFEAT_BASE:-pepin-laptop:zenoh}"
    # The Docker VM also runs the live stack, and a compiler that fills its memory gets the stack
    # killed, not itself (build_rtabmap.sh has the measurement). So the build is watched: once more
    # than MAX_USED_PCT of the VM's memory is in use it is cancelled, whatever it was doing. The
    # VM's /proc/meminfo is read by a short-lived container on the base image every few seconds.
    MAX_USED_PCT="${PEPIN_BUILD_MAX_USED_PCT:-75}"
    vm_used_pct() {
        docker run --rm --memory 64m --network none --entrypoint awk "$BASE" \
            '/MemTotal/ {t = $2} /MemAvailable/ {a = $2} END {print int(100 * (t - a) / t)}' /proc/meminfo
    }
    # The adapters import the library (pepin.xfeat_models), so the build context carries the
    # current one, as ros/build-image.sh's does for the base image.
    mkdir -p "$HERE/pepin_src/pepin"
    rsync -a --delete --exclude '__pycache__' "$HERE/../src/pepin/" "$HERE/pepin_src/pepin/"
    docker build -f "$HERE/Dockerfile.xfeat" -t pepin-laptop:xfeat --build-arg "BASE=$BASE" \
        ${PEPIN_CORE_JOBS:+--build-arg "CORE_JOBS=$PEPIN_CORE_JOBS"} \
        ${PEPIN_ROS_JOBS:+--build-arg "ROS_JOBS=$PEPIN_ROS_JOBS"} "$HERE" &
    BUILD=$!
    PEAK=0
    while kill -0 "$BUILD" 2>/dev/null; do
        # A reading that fails counts as full: a build nobody can watch is not left running.
        USED=$(vm_used_pct) || USED=100
        if [ "$USED" -gt "$PEAK" ]; then PEAK=$USED; fi
        if [ "$USED" -gt "$MAX_USED_PCT" ]; then
            echo "laptop-build: the Docker VM's memory is ${USED} % used (over ${MAX_USED_PCT}): cancelling the build" >&2
            kill "$BUILD"
            wait "$BUILD" || true
            exit 3
        fi
        sleep 3
    done
    wait "$BUILD"
    echo "pepin-laptop:xfeat built on $BASE (the VM's memory peaked at ${PEAK} % used)"
    exit 0
fi
docker build -f "$HERE/Dockerfile.laptop" -t pepin-laptop:latest "$HERE"
# The image carries both middlewares; the default transport starts the :zenoh tag (ros/lib.sh).
docker tag pepin-laptop:latest pepin-laptop:zenoh
echo "pepin-laptop:latest built"
