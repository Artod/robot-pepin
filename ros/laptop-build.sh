#!/bin/bash
# Build the laptop image (pepin-laptop): the board's sensor image plus Nav2, RTAB-Map and the
# image pipeline.
# Usage: ros/laptop-build.sh        (a few minutes; needs pepin-ros:sensors from ros/build-image.sh)
#        ros/laptop-build.sh xfeat  pepin-laptop:xfeat on top of pepin-laptop:zenoh
#                                   (ros/Dockerfile.xfeat: RTAB-Map rebuilt with Python, XFeat and
#                                   LighterGlue for the visual registration; about an hour)
#        ros/laptop-build.sh vio    pepin-laptop:vio on top of pepin-laptop:xfeat (ros/Dockerfile.vio:
#                                   OpenVINS pinned, its simulator as the build's gate; 10-20 min)
#        ros/laptop-build.sh gaze   pepin-laptop:gaze: pepin-laptop:zenoh plus the stall look's BT
#                                   node (Dockerfile.laptop's last stage alone; under a minute).
#                                   ros/laptop.sh nav runs it whenever it is built on that image
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# Every image COPYs the lost-wake-up librmw_zenoh_cpp.so (ros/patches/rmw_zenoh-lost-wakeup.patch):
# built once here, a minute, unless it is already in ros/build/rmw_zenoh_fix/.
[ -f "$HERE/build/rmw_zenoh_fix/librmw_zenoh_cpp.so" ] || "$HERE/tools/build_rmw_zenoh_fix.sh"
if [ "${1:-}" = gaze ]; then
    BASE="${PEPIN_GAZE_BASE:-pepin-laptop:zenoh}"
    docker build -f "$HERE/Dockerfile.laptop" -t pepin-laptop:gaze --build-arg "BASE=$BASE" "$HERE"
    echo "pepin-laptop:gaze built on $BASE"
    exit 0
fi
# The Docker VM also runs the live stack, and a compiler that fills its memory gets the stack
# killed, not itself (build_rtabmap.sh has the measurement). So a long build is watched: once more
# than MAX_USED_PCT of the VM's memory is in use it is cancelled, whatever it was doing. The VM's
# /proc/meminfo is read by a short-lived container on the base image every few seconds.
MAX_USED_PCT="${PEPIN_BUILD_MAX_USED_PCT:-75}"
vm_used_pct() {  # BASE
    docker run --rm --memory 64m --network none --entrypoint awk "$1" \
        '/MemTotal/ {t = $2} /MemAvailable/ {a = $2} END {print int(100 * (t - a) / t)}' /proc/meminfo
}
watched_build() {  # TAG BASE docker-build-args...: the build, cancelled past MAX_USED_PCT
    local tag="$1" base="$2"
    shift 2
    docker build -t "$tag" --build-arg "BASE=$base" "$@" &
    local build=$! peak=0 used
    while kill -0 "$build" 2>/dev/null; do
        # A reading that fails counts as full: a build nobody can watch is not left running.
        used=$(vm_used_pct "$base") || used=100
        if [ "$used" -gt "$peak" ]; then peak=$used; fi
        if [ "$used" -gt "$MAX_USED_PCT" ]; then
            echo "laptop-build: the Docker VM's memory is ${used} % used (over ${MAX_USED_PCT}): cancelling the build" >&2
            kill "$build"
            wait "$build" || true
            exit 3
        fi
        sleep 3
    done
    wait "$build"
    echo "$tag built on $base (the VM's memory peaked at ${peak} % used)"
}
if [ "${1:-}" = xfeat ]; then
    BASE="${PEPIN_XFEAT_BASE:-pepin-laptop:zenoh}"
    # The adapters import the library (pepin.xfeat_models), so the build context carries the
    # current one, as ros/build-image.sh's does for the base image.
    mkdir -p "$HERE/pepin_src/pepin"
    rsync -a --delete --exclude '__pycache__' "$HERE/../src/pepin/" "$HERE/pepin_src/pepin/"
    watched_build pepin-laptop:xfeat "$BASE" -f "$HERE/Dockerfile.xfeat" \
        ${PEPIN_CORE_JOBS:+--build-arg "CORE_JOBS=$PEPIN_CORE_JOBS"} \
        ${PEPIN_ROS_JOBS:+--build-arg "ROS_JOBS=$PEPIN_ROS_JOBS"} "$HERE"
    exit 0
fi
if [ "${1:-}" = vio ]; then
    # OpenVINS pinned (ros/Dockerfile.vio: master 2025-11-30 + PR #500), its simulator as the gate;
    # 10-20 min (estimate). PEPIN_VIO_MASK=1 adds the per-frame mask patch; PEPIN_VIO_EXECUTOR=0
    # builds upstream's executor code (Dockerfile.vio's EXECUTOR).
    BASE="${PEPIN_VIO_BASE:-pepin-laptop:xfeat}"
    watched_build pepin-laptop:vio "$BASE" -f "$HERE/Dockerfile.vio" \
        --build-arg "MASK=${PEPIN_VIO_MASK:-0}" \
        --build-arg "EXECUTOR=${PEPIN_VIO_EXECUTOR:-1}" \
        ${PEPIN_VIO_JOBS:+--build-arg "JOBS=$PEPIN_VIO_JOBS"} "$HERE"
    docker run --rm --network none --entrypoint cat pepin-laptop:vio /opt/openvins/SHAS
    exit 0
fi
docker build -f "$HERE/Dockerfile.laptop" -t pepin-laptop:latest "$HERE"
# The image carries both middlewares; the default transport starts the :zenoh tag (ros/lib.sh).
docker tag pepin-laptop:latest pepin-laptop:zenoh
echo "pepin-laptop:latest built"
