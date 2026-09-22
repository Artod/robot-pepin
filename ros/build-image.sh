#!/bin/bash
# Cross-build the board's image on this laptop (linux/arm64, native on Apple Silicon) and ship
# the result to the board. The board compiles rf2o with one compiler on four A53 cores and a
# rebuild there costs 16-30 min (ros/build.sh, which still works and stays the fallback); the
# same image builds here in minutes and travels over the wire as a tarball.
#
# Usage:
#   ros/build-image.sh                 build only — touches nothing on the board
#   ros/build-image.sh --ship          build, then load the image on the board
#   ros/build-image.sh --ship-only     skip the build, ship the image already built here
#
# Shipping is a separate step on purpose: `docker load` on the board while the robot drives is
# not allowed. Neither step restarts the stack — the restart command is printed at the end.
set -euo pipefail

BOARD="${PEPIN_HOST:-10.0.0.187}"
HERE="$(cd "$(dirname "$0")" && pwd)"
IMAGE=pepin-ros
TAGS=("$IMAGE:latest" "$IMAGE:zenoh")  # both names the board's run.sh may ask for (ros/lib.sh)
# The multiplexed master of ros/lib.sh is deliberately NOT used here: this script runs while the
# robot is being driven from another session, and a shared master is a shared failure.
SSH=(ssh -o ControlPath=none -o ConnectTimeout=6)

DO_BUILD=1
DO_SHIP=0
FORCE=0
for arg in "$@"; do
    case "$arg" in
        --ship) DO_SHIP=1 ;;
        --ship-only) DO_SHIP=1; DO_BUILD=0 ;;
        --force) FORCE=1 ;;
        *) echo "usage: $0 [--ship|--ship-only] [--force]" >&2; exit 2 ;;
    esac
done

if [ "$DO_BUILD" = 1 ]; then
    # The build context is ros/, exactly as on the board, and it needs the same pepin_src tree
    # that ros/build.sh assembles there out of src/pepin.
    mkdir -p "$HERE/pepin_src/pepin"
    rsync -a --delete --exclude '__pycache__' "$HERE/../src/pepin/" "$HERE/pepin_src/pepin/"

    # How many compilers the rf2o layer gets (Dockerfile's RF2O_JOBS, 1 on the board because of
    # its RAM). Here the limit is pointless: the machine has cores and memory.
    JOBS="${PEPIN_BUILD_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}"

    BUILDX=(docker buildx build)
    # By default the daemon's own layer cache is the cache: an unchanged rf2o layer is reused
    # exactly as it is by `docker build`. PEPIN_BUILD_CACHE=<dir> instead exports the cache to a
    # directory, which survives a `docker buildx prune` and needs the container driver.
    if [ -n "${PEPIN_BUILD_CACHE:-}" ]; then
        docker buildx inspect pepin-arm64 >/dev/null 2>&1 \
            || docker buildx create --name pepin-arm64 --driver docker-container >/dev/null
        mkdir -p "$PEPIN_BUILD_CACHE"
        BUILDX+=(--builder pepin-arm64
                 --cache-from "type=local,src=$PEPIN_BUILD_CACHE"
                 --cache-to "type=local,dest=$PEPIN_BUILD_CACHE,mode=max")
    fi

    echo "building $IMAGE for linux/arm64 with RF2O_JOBS=$JOBS"
    T0=$(date +%s)
    "${BUILDX[@]}" --platform linux/arm64 --load \
        --build-arg "RF2O_JOBS=$JOBS" \
        -t "${TAGS[0]}" -t "${TAGS[1]}" \
        -f "$HERE/Dockerfile" "$HERE"
    echo "build took $(( $(date +%s) - T0 )) s"

    # A wrong architecture is silent until the board refuses to start the container.
    ARCH=$(docker image inspect "${TAGS[1]}" --format '{{.Os}}/{{.Architecture}}')
    [ "$ARCH" = linux/arm64 ] || { echo "built $ARCH, not linux/arm64" >&2; exit 1; }
    echo "$ARCH, $(docker image inspect "${TAGS[1]}" --format '{{.Size}}' | awk '{printf "%.2f GB", $1/1e9}') uncompressed"
fi

if [ "$DO_SHIP" = 1 ]; then
    # Loading an image under a driving robot is how a drive is lost; the stack must be down.
    if "${SSH[@]}" "root@$BOARD" 'docker ps --format "{{.Names}}" | grep -qx pepin-ros'; then
        [ "$FORCE" = 1 ] || { echo "pepin-ros is running on the board: stop it first (systemctl stop pepin-ros), or --force" >&2; exit 1; }
    fi
    # The tarball goes over the wire AS IT IS. Measured 2026-09-22: `docker save` of this image
    # is 1.03 GB and zstd -3 takes 0.7 % off it, because Docker Desktop's containerd image store
    # saves layers already compressed — a compressor would only spend an A53 on decompression.
    # A daemon with the classic image store writes uncompressed tars instead, and there
    # PEPIN_SHIP_COMPRESS=zstd (or gzip) pays for itself; the board must have the same tool.
    PACK=(cat); UNPACK='docker load'
    case "${PEPIN_SHIP_COMPRESS:-}" in
        zstd) PACK=(zstd -3 -T0 -c); UNPACK='zstd -d -c | docker load' ;;
        gzip) PACK=(gzip -1 -c);     UNPACK='gzip -d -c | docker load' ;;
    esac
    echo "shipping ${TAGS[*]} to $BOARD ($(docker image inspect "${TAGS[1]}" --format '{{.Size}}' | awk '{printf "%.2f GB", $1/1e9}'), pipe: ${PACK[0]})"
    T0=$(date +%s)
    docker save "${TAGS[0]}" "${TAGS[1]}" | "${PACK[@]}" | "${SSH[@]}" "root@$BOARD" "$UNPACK"
    echo "ship took $(( $(date +%s) - T0 )) s"
    "${SSH[@]}" "root@$BOARD" "docker images $IMAGE"
    echo "the image is loaded, nothing was restarted. To run it:"
    echo "  ros/restart.sh board            # or: ssh root@$BOARD systemctl restart pepin-ros"
    echo "Code, params and maps still travel separately (ros/sync.sh): the container mounts them."
fi
